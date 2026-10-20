"""R2 milestone 2, step 5: the G0 EQUAL-ADVANCE GATE of the per-request decode path (nosi/nosi/r2_decode_path.py)
against the shipped decode, on the GPU, in ONE process.

TWO model instances (A, B; 2 x 16.4 GB) run back to back, interleaved step by step, each with its own fresh cache:
arm A is always the shipped decode; arm B is the phase's decode. After every decode step the two arms are compared
ON THE DEVICE with torch.equal (r2_g0_compare.py): the logits and the greedy tokens; per layer, the selection topk_idx
(the shipped arm's captured at its decode_update_kv call, the R2 arm's from state.topk_log), _block_map, _cache_lens,
the slot-cache rows below cache_lens (_k_gpu, _v_gpu, _kv_bias_gpu), the compressed keys [0, n_comp), the live raw
keys, total_cis [0, pos), the compressed cis [0, M) and the counters (n_comp, live raw keys, cis rows). After the
last step: the host K / V rows [0, S + steps). The prefill logits are compared first.

PHASES (in this order; P FIRST: if P is not bitwise every later phase is NOT_RUN, because nondeterminism would void
the comparison):
  P     shipped vs shipped: the positive control (two model instances, two caches).
  G0T   shipped vs the R2 path with body "shipped" (the lockstep body; the positional tables and the per-request
        kernels are the R2 ones): must be bitwise equal.
  NEGT  G0T with request 0's stage-1 / pooling key length shortened by --neg-short from step --neg-step: the comparator
        must see the first difference exactly at that step and only in request 0 (the per-request isolation of the
        path and the comparator's sensitivity).
  G0F   shipped vs the R2 path with body "r2" (r2_decode_body.py, f4_fix False): must be bitwise equal.
  NEGF  G0F with f4_fix True: the first difference exactly at the block-completing step (S + s + 1 = 0 mod 64) and
        none before.
G0F / NEGF need nosi/nosi/r2_decode_body.py (merged from feature/nosi-r2-m2-engine 4b8a658): without it they are
NOT_RUN (a failed MUST gate).

F2 (docs/R2_ENGINE_FINDINGS.md): the compressed cis is compared on the prefix both arms hold (the shipped width M =
S // 16 - 1, which never grows); the twin's extra columns are reported as the EXPECTED F2 difference (no f2_fix switch).
F4: the R2 body reproduces it (f4_fix False); NEGF is the fix as the negative control.

ENVIRONMENT: NOSI_R2_DECODE=1 (the switch; refused otherwise), NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0,
NOSI_MODEL_PATH or --model. Exit: the number of failed MUST gates; 2 a refusal.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_compare as C  # noqa: E402

PHASES = {
    "P": dict(r2=False, desc="shipped vs shipped: the positive control"),
    "G0T": dict(r2=True, body="shipped", f4_fix=None, neg=False, desc="R2 tables + kernels, shipped (lockstep) body"),
    "NEGT": dict(r2=True, body="shipped", f4_fix=None, neg=True, desc="G0T with request 0's key length shortened"),
    "G0F": dict(r2=True, body="r2", f4_fix=False, neg=False, desc="R2 tables + kernels + per-request body"),
    "NEGF": dict(r2=True, body="r2", f4_fix=True, neg=False, desc="G0F with f4_fix True"),
}
F2_NOTE = ("compressed cis compared on the prefix both arms hold (the shipped width M = S // 16 - 1, which never grows, "
           "findings F2); the twin's extra columns are the EXPECTED F2 difference, reported, not gated; no f2_fix switch")


def log(msg):
    print("[r2_g0 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--phases", default="P,G0T,NEGT,G0F,NEGF")
    ap.add_argument("--neg-step", type=int, default=40)
    ap.add_argument("--neg-short", type=int, default=400)   # 1007 - 400: blocks >= 152 lose keys
    a = ap.parse_args(argv)
    a.batches = [int(x) for x in a.batches.split(",") if x]
    a.phases = [p for p in a.phases.split(",") if p]
    bad = [p for p in a.phases if p not in PHASES]
    if bad or not a.phases or a.phases[0] != "P":
        ap.error("phases must start with P and name only %s (got %s)" % (sorted(PHASES), a.phases))
    if a.S % 64:
        ap.error("S must be 0 mod 64 (the 64-step window of findings F1)")
    if not 1 <= a.steps <= 64:
        ap.error("steps must be in 1 .. 64 (the shipped decode is valid for 64 - S %% 64 steps, findings F1)")
    if not 0 <= a.neg_step < a.steps:
        ap.error("neg-step must be inside the window")
    return a


def block_step(S: int) -> int:
    """The first decode step s whose written position S + s completes a block."""
    return (63 - S % 64) % 64


def load_workload(path, B, L):
    import numpy as np
    import torch
    man = json.load(open(os.path.join(path, "c16k_manifest.json")))
    n, ctx = man["shape"]
    if ctx < L or not 1 <= B <= n:
        raise SystemExit("REFUSED: workload %s x %d cannot give %d documents of %d tokens" % (n, ctx, B, L))
    sums = {}
    for line in open(os.path.join(path, "SHA256SUMS")):
        h, fname = line.split()
        sums[fname] = h
    raw = open(os.path.join(path, "c16k_tokens.bin"), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sums.get("c16k_tokens.bin") != sha:
        raise SystemExit("REFUSED: c16k_tokens.bin sha256 %s is not SHA256SUMS's" % sha)
    arr = np.frombuffer(raw, dtype="<i4").reshape(n, ctx)
    return torch.from_numpy(arr[:B, :L].astype("int64")), sha


def new_cache(model):
    from nosi.cache_engine import InfLLMv2Cache
    return InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)


def capture_topk(cache, store):
    orig = cache.decode_update_kv

    def hook(k, v, kv_bias, layer_idx, topk_idx):
        store[layer_idx] = topk_idx.clone()
        return orig(k, v, kv_bias, layer_idx, topk_idx)

    cache.decode_update_kv = hook


def shipped_view(cache, l, topk_log):
    lay = cache.layers[l]
    e = lay.cache_engine
    cl = int(e._cache_lens.max())
    n = int(lay.cached_compressed_max_seqlen)
    ncl = int(lay.no_compress_k_len)
    return dict(topk=(topk_log[l], 1), block_map=(e._block_map, 1), cache_lens=(e._cache_lens, 0),
                kwin=(e._k_gpu[:, :cl], 0), vwin=(e._v_gpu[:, :cl], 0), bias=(e._kv_bias_gpu[:, :cl], 0),
                comp_k=(lay.compress_k_cache_varlen[:, :n], 0), raw_live=(lay.no_compress_k_cache[:, :ncl], 0),
                total_cis=(lay.total_cis[:, :int(lay.cis_len)], 0), ccis=(lay.compressed_cis, 1),
                counters=((n, ncl, int(lay.cis_len)), None))


def r2_view(cache, st, l, M):
    import torch
    t = st.twins[l]
    e = cache.layers[l].cache_engine
    cl = int(e._cache_lens.max())
    uniform = bool((t.n_comp == t.n_comp[0]).all()) and bool((t.pos == t.pos[0]).all())
    n, p = int(t.n_comp.max()), int(t.pos.max())
    live = p - 16 * n
    raw = torch.stack([t.live_keys(b) for b in range(t.B)]) if uniform else torch.empty(0)
    return dict(topk=(st.topk_log[l], 1), block_map=(e._block_map, 1), cache_lens=(e._cache_lens, 0),
                kwin=(e._k_gpu[:, :cl], 0), vwin=(e._v_gpu[:, :cl], 0), bias=(e._kv_bias_gpu[:, :cl], 0),
                comp_k=(t.comp_k[:, :n], 0), raw_live=(raw, 0), total_cis=(t.total_cis[:, :p], 0),
                ccis=(t.ccis[:, :, :M], 1), counters=((n, live, p) if uniform else ("ragged",), None))


def run_phase(name, a, mA, mB, ids, B):
    import torch
    from nosi.r2_decode_path import R2State
    cfg = PHASES[name]
    S, steps = a.S, a.steps
    x = ids[:B].to("cuda")
    L = mA.num_layers
    with torch.inference_mode():
        cA = new_cache(mA)
        lgA, posA = mA.batch_prefill(x, cA)
        cB = new_cache(mB)
        lgB, posB = mB.batch_prefill(x, cB)
    mA.has_buffers = False                                   # the warm-up step rebinds its graphs to the new cache
    mB.has_buffers = False
    tr = C.Tracker()
    tr.update(-1, None, C.compare_views(dict(prefill_logits=(lgA[:, -1], 0)), dict(prefill_logits=(lgB[:, -1], 0))))
    logA, logB = [None] * L, [None] * L
    capture_topk(cA, logA)
    st = None
    if cfg["r2"]:
        neg = dict(step=a.neg_step, request=0, short=a.neg_short) if cfg["neg"] else None
        with torch.inference_mode():
            st = R2State.attach(mB, cB, S, body=cfg["body"], f4_fix=cfg["f4_fix"], neg=neg)
    else:
        capture_topk(cB, logB)
    M = S // 16 - 1
    tA = tB = 0.0
    with torch.inference_mode():
        tokA = lgA[:, -1].argmax(-1, keepdim=True)
        tokB = lgB[:, -1].argmax(-1, keepdim=True)
        pA = posA[:, -1:] + 1
        pB = posB[:, -1:] + 1
        cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
        for s in range(steps):
            t0 = time.time()
            oA = mA.decode_inference(tokA, cu, pA, cA)
            torch.cuda.synchronize()
            t1 = time.time()
            oB = mB.decode_inference(tokB, cu, pB, cB)
            torch.cuda.synchronize()
            tA += t1 - t0
            tB += time.time() - t1
            lA, lB = oA[:, -1], oB[:, -1]
            nA, nB = lA.argmax(-1, keepdim=True), lB.argmax(-1, keepdim=True)
            tr.update(s, None, C.compare_views(dict(logits=(lA, 0), tokens=(nA, 0)), dict(logits=(lB, 0), tokens=(nB, 0))))
            for l in range(L):
                va = shipped_view(cA, l, logA)
                vb = r2_view(cB, st, l, M) if st is not None else shipped_view(cB, l, logB)
                tr.update(s, l, C.compare_views(va, vb))
            tokA, tokB = nA, nB
            pA, pB = pA + 1, pB + 1
        end = S + steps
        for l in range(L):
            eA, eB = cA.layers[l].cache_engine, cB.layers[l].cache_engine
            tr.update(steps - 1, l, C.compare_views(dict(host_k=(eA._k_cpu[:, :end], 0), host_v=(eA._v_cpu[:, :end], 0)),
                                                    dict(host_k=(eB._k_cpu[:, :end], 0), host_v=(eB._v_cpu[:, :end], 0))))
    f2 = None
    if st is not None:
        extra = [int(t.n_ccis.max()) - M for t in st.twins]
        f2 = dict(expected_extra_ccis_columns=max(extra), per_layer_min=min(extra), note=F2_NOTE)
    res = dict(phase=name, B=B, desc=cfg["desc"], tracker=tr.summary(), shipped_s=round(tA, 3), other_s=round(tB, 3),
               f2=f2, state=st.meta if st is not None else None, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    if name == "NEGT":
        ok, why = C.verdict_neg(tr, a.neg_step, [0])
    elif name == "NEGF":
        ok, why = C.verdict_neg(tr, block_step(S), None)
    else:
        ok, why = C.verdict_equal(tr)
    res.update(passed=ok, verdict=why)
    del cA, cB, st
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    try:
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        if not NL.R2_DECODE:
            raise SystemExit("REFUSED: NOSI_R2_DECODE is not 1 (the gate drives the switch)")
        if NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_ATTN_SPLITS must be > 0 and NOSI_POOL_BLOCKS 0 (%d, %d)" % (NL._ATTN_SPLITS, CE.POOL_BLOCKS))
        ids, sha = load_workload(a.workload, max(a.batches), a.S)
        os.makedirs(a.out)
        t0 = time.time()
        mA = Llama(model_name=a.model, device="cuda", offload=True)
        mB = Llama(model_name=a.model, device="cuda", offload=True)
        log("two models loaded in %.0f s, allocated %.1f GB" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9))
        try:
            import nosi.r2_decode_body  # noqa: F401
            body = True
        except ImportError:
            body = False
        results, gates, stop = [], [], None
        for B in a.batches:
            for name in a.phases:
                key = "%s_b%d" % (name, B)
                if stop:
                    gates.append(dict(name=key, must=True, passed=False, detail="NOT_RUN: " + stop))
                    continue
                if PHASES[name].get("body") == "r2" and not body:
                    gates.append(dict(name=key, must=True, passed=False, detail="NOT_RUN: nosi/nosi/r2_decode_body.py is not in this tree"))
                    continue
                log("phase %s B %d" % (name, B))
                try:
                    r = run_phase(name, a, mA, mB, ids, B)
                except Exception as e:
                    traceback.print_exc()
                    r = dict(phase=name, B=B, passed=False, verdict="EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400]))
                results.append(r)
                gates.append(dict(name=key, must=True, passed=bool(r["passed"]), detail=r["verdict"]))
                log("phase %s B %d: %s %s" % (name, B, "PASS" if r["passed"] else "FAIL", r["verdict"][:300]))
                if name == "P" and not r["passed"]:
                    stop = "the positive control P at B %d is not bitwise (nondeterminism voids G0)" % B
        meta = dict(S=a.S, steps=a.steps, batches=a.batches, phases=a.phases, neg_step=a.neg_step, neg_short=a.neg_short,
                    block_step=block_step(a.S), workload_sha256=sha, body_available=body, f2=F2_NOTE,
                    f4="the R2 body reproduces F4 (f4_fix False in G0F); NEGF sets f4_fix True",
                    attn_splits=NL._ATTN_SPLITS, nosi_commit=os.environ.get("G0_COMMIT", "unknown"),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        json.dump(dict(meta=meta, gates=gates, results=results), open(os.path.join(a.out, "summary.json"), "w"),
                  indent=1, sort_keys=True, default=str)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        for g in gates:
            print("GATE %s %s: %s" % (g["name"], "PASS" if g["passed"] else "FAIL", g["detail"][:400]))
        print("r2_g0: %d MUST gate(s) failed" % nfail)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
