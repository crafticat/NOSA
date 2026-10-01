"""TREE-VECTOR PIGGYBACK CAPTURE, the GPU driver: a SEPARATE small process (never inside a timing process, never beside one),
spec retroinfer-eval docs/evidence/treecap_scores_L16128/PIGGYBACK_SPEC.md; the pure extraction and its CPU tests: treecap_core.py.

Same scope and inputs as the existing score export (treecap_L16128.npz): 4 PG-19 requests (the first 4 dataset rows with >= L+T
tokens, the trace_selections.py rule), L = 16128, T = 64 teacher-forced decode calls (call 0 = the warm-up), shipped C = 63
(NOSI_POOL_BLOCKS = 0, offload 1), layers TC_LAYERS (default 0 4 8 ... 28), both KV heads. The hooks are WRAPPERS installed in
this process only (treecap_core.install_hooks): nosa_llama.py, cache_engine.py and transfer_trace.py are not edited.

IDENTITY GATE (fail-closed, exit 21): the new capture's per-call block maps, load masks and pooled QK / CIS buffers must equal the
existing export TC_REF_NPZ bit for bit at every call, layer, head and request (the hooks must not perturb the trajectory), the
pooling (pool_offline) and selection (select_offline) of the captured unpooled scores must reproduce the recorded buffers, and
every (call, layer) record must be present. step_ms of this run is VOID for timing (hooks run inside the calls).

    TC_OUT=<fresh dir> python treecap_capture.py
EXIT: 0 ok; 21 a gate / check failed (the export is still written, labelled FAILED); 23 crash.
"""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import treecap_core as TC

L = int(os.environ.get("TC_L", "16128"))
B = int(os.environ.get("TC_B", "4"))
T = int(os.environ.get("TC_STEPS", "64"))
LAYERS = tuple(int(x) for x in os.environ.get("TC_LAYERS", "0 4 8 12 16 20 24 28").split())
RAW_LAYERS = tuple(int(x) for x in os.environ.get("TC_RAW_LAYERS", "0 16").split())
OUT = os.environ.get("TC_OUT", "")
REF_NPZ = os.environ.get("TC_REF_NPZ", "")
HASH_MODEL = os.environ.get("TC_HASH_MODEL", "1") == "1"
EXPECT_SAFETENSORS_SHA = os.environ.get("TC_EXPECT_SAFETENSORS_SHA", "5219c9ec34c4a19650e98b079ccf4911ffb3457e8f67690ca9037c09e89fd477")
RC_GATE, RC_CRASH = 21, 23


def pick_docs(tokenizer, pq):
    """trace_selections.py's rule: the first B dataset rows (datasets parquet 'train' order) with >= L + T tokens."""
    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files=pq)["train"]["text"]
    rows, ids = [], []
    for i in range(len(dataset)):
        t = tokenizer(dataset[i], return_tensors="pt").input_ids
        if t.shape[1] < L + T:
            continue
        rows.append(i)
        ids.append(t[0, :L + T])
        if len(rows) == B:
            break
    assert len(rows) == B, "only %d documents with >= %d tokens" % (len(rows), L + T)
    return rows, torch.stack(ids)


def checkpoint_identity(path):
    out = dict(path=path)
    for n in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        p = os.path.join(path, n)
        if os.path.exists(p):
            out[n + "_sha256"] = TC.sha256_file(p)
    st = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    out["safetensors"] = {f: os.path.getsize(os.path.join(path, f)) for f in st}
    if HASH_MODEL and st:
        t = time.time()
        out["safetensors_sha256"] = {f: TC.sha256_file(os.path.join(path, f)) for f in st}
        out["hash_seconds"] = time.time() - t
        if "model.safetensors" in out["safetensors_sha256"]:
            out["safetensors_sha_matches_export"] = out["safetensors_sha256"]["model.safetensors"] == EXPECT_SAFETENSORS_SHA
    return out


def main():
    if not OUT or os.path.exists(os.path.join(OUT, "treecap_vectors_L16128.npz")):
        print("[treecap] REFUSED: TC_OUT must be a fresh directory (got %r)" % OUT, flush=True)
        return 3
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    from transformers import AutoTokenizer
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as CE
    from nosi import nosa_llama as NL
    from nosi import transfer_trace as _tt
    if CE.POOL_BLOCKS != 0 or CE.VERIFY_ROUND_SLOTS != 0:
        print("[treecap] REFUSED: NOSI_POOL_BLOCKS / NOSI_VERIFY_ROUND_SLOTS must be 0 (the shipped C63 path)", flush=True)
        return 3
    path, pq = os.environ["NOSI_MODEL_PATH"], os.environ["NOSI_PG19_PARQUET"]
    tok = AutoTokenizer.from_pretrained(path)
    rows, ids = pick_docs(tok, pq)
    ids = ids.to("cuda")
    prompt, forced = ids[:, :L], ids[:, L:]
    model = Llama(model_name=path, device="cuda", offload=True)
    rec = TC.Recorder(LAYERS, T)
    Trace = TC.make_treecap_trace(_tt.TransferTrace, rec, model)
    trace = Trace(model.num_layers, "scores", max_steps=T)
    _tt.TRACE = trace
    uninstall = TC.install_hooks(NL, CE.InfLLMv2Cache, rec, lambda: _tt.TRACE, n_heads=model.config.num_attention_heads,
                                 n_kv=model.config.num_key_value_heads)
    t_run = time.time()
    with torch.inference_mode():
        trace.new_document()
        cache = CE.InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
        logits, position_ids = model.batch_prefill(prompt, cache)
        torch.cuda.synchronize()
        position_ids = position_ids[:, -1:] + 1
        cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
        argmax = []
        for it in range(T):
            logits = model.decode_inference(forced[:, it:it + 1], cu, position_ids, cache, warmup=(it == 0))
            if it + 1 < T:
                argmax.append(int((logits[:, -1, :].argmax(-1) == forced[:, it + 1]).sum()))
            position_ids = position_ids + 1
        torch.cuda.synchronize()
    run_s = time.time() - t_run
    uninstall()
    _tt.TRACE = None
    step_rows, _, _, masks, maps = trace.harvest(timed_steps=())        # the one synchronize, after every call
    scores = trace.score_archive[:T].cpu()                                # (T, 32, 2, H, B, M) bf16: [0] pooled QK, [1] pooled CIS
    masks, maps = masks.to(torch.int16), maps.to(torch.int16)
    # ---- host reads (outside every call; state only)
    t_host = time.time()
    sl = list(LAYERS)
    gm, comp, tcis, raw = [], [], [], {}
    n_groups = L // 64
    for l in sl:
        lay = cache.layers[l]
        e = lay.cache_engine
        kh = e._k_cpu[:, :L]
        gm.append(TC.group_means(kh, n_groups))
        comp.append(lay.compress_k_cache_varlen.detach().cpu())
        tcis.append(lay.total_cis[:, :L + T].detach().cpu())
        if l in RAW_LAYERS:
            raw[l] = kh[0].clone()
    host_s = time.time() - t_host
    # ---- assemble per-call arrays
    q = rec.stack("q")
    kn = rec.stack("k")
    pos = rec.stack("pos")
    s1 = rec.stack("stage1")
    cc = rec.stack("ccis")
    t33 = rec.stack("top33")
    t64 = rec.stack("top64")
    ncomp = rec.scalars("ncomp")
    chunk = rec.scalars("chunk_done", fill=0)
    pooled_qk = scores[:, sl, 0]
    pooled_cis = scores[:, sl, 1]
    # ---- checks
    checks = dict(records_missing=rec.complete())
    checks["records_complete"] = all(v == 0 for v in checks["records_missing"].values())
    checks["pooling"] = TC.check_pooling(s1, cc, ncomp, pooled_qk, pooled_cis, t33)
    checks["selection"] = TC.check_selection(pooled_qk, pooled_cis, maps[:, sl].to(torch.int64))
    checks["selection_ids_equal_block_map"] = bool(torch.equal(TC.ids_to_mask(t64, pooled_qk.shape[-1]), TC.ids_to_mask(maps[:, sl].to(torch.int64), pooled_qk.shape[-1])))
    positions = [L + t for t in range(T)]
    checks["positions_equal"] = bool(pos is not None and all(int(pos[t, 0].min()) == int(pos[t, 0].max()) == positions[t] for t in range(T)))
    rule = [TC.n_comp_rule(p) for p in positions]
    checks["n_comp_vs_rule"] = dict(rule=rule, recorded_layer0=ncomp[:, 0].tolist(), equal=bool(all(int(ncomp[t, 0]) == rule[t] for t in range(T))),
                                    chunk_completed_calls=[t for t in range(T) if chunk[t, 0]])
    checks["compressed_prefix"] = {l: TC.check_compressed(cache.layers[l].cache_engine._k_cpu[:, :L], comp[i], n_check=(L - 32) // 16 + 1)
                                   for i, l in enumerate(sl)}
    ident = dict(reference=REF_NPZ or None)
    if REF_NPZ:
        z = np.load(REF_NPZ)
        sc_u16 = scores.view(torch.int16).numpy().view(np.uint16)
        ident.update(block_map=bool(np.array_equal(z["block_map"], maps.numpy())), load_mask=bool(np.array_equal(z["load_mask"], masks.numpy())),
                     qk_pooled=bool(np.array_equal(z["qk_pooled_bf16"], sc_u16[:, :, 0])), cis_pooled=bool(np.array_equal(z["cis_pooled_bf16"], sc_u16[:, :, 1])))
        ident["ok"] = all(ident[k] for k in ("block_map", "load_mask", "qk_pooled", "cis_pooled"))
    else:
        ident["ok"] = None
    checks["identity_vs_existing_export"] = ident
    causal = TC.causal_groups(positions)
    causal["n_comp_per_call"] = ncomp[:, 0].tolist()
    causal["note"] = ("groups 0..%d complete at prefill (L = %d = %d x 64); every decode position lies in tail block %d, so no 64-token group "
                      "completes during the %d calls; compressed chunks complete at the calls listed in checks.n_comp_vs_rule" % (n_groups - 1, L, n_groups, n_groups, T))
    ok = bool(checks["records_complete"] and checks["pooling"]["ok"] and checks["selection"]["ok"] and checks["selection_ids_equal_block_map"]
              and checks["positions_equal"] and (ident["ok"] is not False))
    cache_before = torch.cat([torch.full((1,) + tuple(maps.shape[1:]), -1, dtype=torch.int16), maps[:-1]], dim=0)
    cache_before[0, ..., 63] = n_groups
    arrays = dict(
        q_rope=q.to(torch.bfloat16) if q is not None else None, k_rope_new=kn.to(torch.bfloat16) if kn is not None else None,
        position_ids=(pos[:, 0] if pos is not None else None), n_comp=ncomp.astype(np.int32), chunk_completed=chunk.astype(bool),
        stage1_score=s1, compressed_cis=cc, qk_pooled_bf16=pooled_qk, cis_pooled_bf16=pooled_cis,
        top33_ids=t33.to(torch.int16), top64_ids=t64.to(torch.int16), block_map=maps[:, sl], load_mask=masks[:, sl], cache_before=cache_before[:, sl],
        group_means_f32=torch.stack(gm), compressed_keys_final=torch.stack(comp), total_cis=torch.stack(tcis),
        token_ids=ids.cpu().to(torch.int32), doc_rows=np.asarray(rows, dtype=np.int32), layers=np.asarray(sl, dtype=np.int32),
        q_head_to_kv_head=np.arange(model.config.num_attention_heads, dtype=np.int8) // (model.config.num_attention_heads // model.config.num_key_value_heads),
        query_position=np.asarray(positions, dtype=np.int32), call_ids=np.arange(T, dtype=np.int32), argmax_matches_forced=np.asarray(argmax, dtype=np.int32))
    for l, t in raw.items():
        arrays["raw_keys_req0_layer%d" % l] = t
    meta = dict(
        what="REAL NOSA-8B post-RoPE vectors + unpooled scores for the tree / range analysis (PIGGYBACK_SPEC.md); status %s" % ("OK" if ok else "FAILED"),
        status="OK" if ok else "FAILED", axes=dict(per_call="[call, layer slot (layers), ...]; heads: kv_head 2 / q_head 32; request = doc_rows order",
                                                   q_rope="[call, layer slot, request, q_head 32, 128]", k_rope_new="[call, layer slot, request, kv_head 2, 128]",
                                                   stage1_score="[call, layer slot, kv_head, request, chunk] padded with NaN past n_comp",
                                                   group_means_f32="[layer slot, request, kv_head, group 0..251, 128] fp32",
                                                   compressed_keys_final="[layer slot, request, chunk, kv_head, 128]", total_cis="[layer slot, request, token, kv_head]"),
        config=TC.NOSA_CONFIG, scope=dict(L=L, B=B, calls=T, warmup_call=0, layers=sl, raw_key_layers=list(RAW_LAYERS), pool_blocks=0, offload=1,
                                         teacher_forced="tokens[L:L+T]; call t consumes token L+t"),
        pins=dict(nosi_commit=os.environ.get("NOSI_COMMIT"), repo_commit=os.environ.get("REPO_COMMIT"), torch=torch.__version__, host=os.uname().nodename,
                  parquet=pq, parquet_sha256=TC.sha256_file(pq)),
        checkpoint=checkpoint_identity(path), docs=rows, checks=checks, causal_accounting=causal,
        cost=dict(run_s=run_s, host_read_s=host_s, total_s=time.time() - t_all, peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                  note="capture process: step_ms VOID for timing (hooks inside the calls); not a measurement"),
        hooks="wrappers in this process only (treecap_core.install_hooks): nosa_llama.apply_rope_with_cos_sin_cache_inplace, "
              "nosa_llama.infllmv2_attn_stage1_fast, InfLLMv2Cache.update_compress_k_decode / update_cis; TransferTrace.record_scores "
              "(+ topk_idx_buf_q / topk_idx_buf); record_mask (maps, masks)")
    missing = list(TC.MISSING_ALWAYS) + ["raw per-token keys of requests 1..%d and of layers outside %s (group means cover every layer slot)" % (B - 1, list(RAW_LAYERS))]
    res = TC.assemble(OUT, arrays, meta, missing, optional=[k for k in arrays if k.startswith("raw_keys_req0_")])
    fx = TC.fixture(dict(stage1_score_f32=s1.float().numpy(), compressed_cis_f32=cc.float().numpy(), n_comp=ncomp, qk_pooled_bf16=TC.as_np(pooled_qk),
                         cis_pooled_bf16=TC.as_np(pooled_cis), top33_ids=t33.numpy(), top64_ids=t64.numpy(), block_map=maps[:, sl].numpy()))
    np.savez(os.path.join(OUT, "fixture_small_vectors.npz"), **fx)
    print("[treecap] %s: %s (%.1f MB), checks: records %s, pooling %d/%d + %d/%d, selection %d/%d, identity %s; %.0f s" % (
        meta["status"], res["npz"], res["bytes"] / 1e6, checks["records_complete"], checks["pooling"]["pool_qk_rows_equal"], checks["pooling"]["pool_qk_rows"],
        checks["pooling"]["pool_cis_rows_equal"], checks["pooling"]["pool_cis_rows"], checks["selection"]["rows_equal"], checks["selection"]["rows"],
        ident["ok"], time.time() - t_all), flush=True)
    return 0 if ok else RC_GATE


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(RC_CRASH)
