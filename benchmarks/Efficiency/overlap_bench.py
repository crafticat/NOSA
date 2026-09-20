"""E3: fetch / compute concurrency microbench -- kappa, the cost model's largest unconstrained
term (docs/evidence/cost_model/DERIVATION.md; spec 2026-09-20-paired-tick.md ladder E3).

From the resident state R of a gated step (verify_alone's arm A recipe: flushed reference
step -> T_l -> restore -> prewarm T_l, so the ordinary step loads 0 blocks), three timings:
  A      the resident decode step alone (main stream)
  S(D)   a side transfer of D blocks per (layer, head, request) alone, two arms:
           triton  = the engine's own flash_h2d_from_mask / _bias kernels (SM programs that
                     tl.load from the pinned host window) into SCRATCH tensors
           memcpy  = torch copy_(non_blocking) of the same bytes from the same pinned window
                     into the scratch (cudaMemcpy2DAsync: the copy engine, no SMs)
  C(D)   S(D) launched on a side stream, then A on the main stream, in the same window
kappa = (T_C - max(T_A, T_S)) / min(T_A, T_S): 0 = independent resources, 1 = serialized.
The scratch is never read by the step (its logits stay torch.equal to the reference and it
loads 0 blocks: both asserted), so C measures contention only. Env: NOSI_OVERLAP_B (128),
NOSI_OVERLAP_D ("4 8 16"), NOSI_OVERLAP_N (8), NOSI_OVERLAP_WARM (4), NOSI_OVERLAP_REPS (3),
NOSI_OVERLAP_L (16128), NOSI_OVERLAP_OUT. Uses verify_alone's helpers (its knobs are mapped
before import; NOSI_POOL_BLOCKS=0, NOSI_VERIFY_ROUND_SLOTS=0, NOSI_ATTN_SPLITS explicit).
"""
import json
import os
import sys
import time

os.environ.setdefault("NOSI_ALONE_MODE", "points")
for src, dst in (("NOSI_OVERLAP_B", "NOSI_ALONE_B"), ("NOSI_OVERLAP_L", "NOSI_ALONE_L"), ("NOSI_OVERLAP_N", "NOSI_ALONE_N"),
                 ("NOSI_OVERLAP_WARM", "NOSI_ALONE_WARM"), ("NOSI_OVERLAP_OUT", "NOSI_ALONE_OUT")):
    if src in os.environ:
        os.environ[dst] = os.environ[src]
os.environ.setdefault("NOSI_ALONE_DOCS", "1")

import torch  # noqa: E402

import verify_alone as VA  # noqa: E402

D_LIST = tuple(int(x) for x in os.environ.get("NOSI_OVERLAP_D", "4 8 16").split())
REPS = int(os.environ.get("NOSI_OVERLAP_REPS", "3"))
OUT = VA.OUT
B_REQ = VA.BATCH
TAG = os.environ.get("NOSI_OVERLAP_TAG", "overlap_b%d" % B_REQ)
PRED = dict(kappa_triton=(0.10, 0.40), kappa_memcpy_max=0.05, memcpy_bw_min_gbps=20.0, triton_bw_min_gbps=20.0,
            falsifiers="kappa_memcpy > 0.10; kappa_triton > 0.5 (serialization: a sync or a single-queue effect must be named); any logits mismatch or non-zero loads in C; memcpy BW < 15 GB/s at D >= 8")


@torch.inference_mode()
def run(model, ids):
    from nosi import state_snapshot as ss
    from nosi.verify import miss_control as mc
    model, cache, logits, position_ids, forced, meta = VA._setup(model, ids)
    B, nl = ids.shape[0], model.num_layers
    layers = list(cache.layers)
    engines = [lay.cache_engine for lay in layers]
    kernels = VA.engine_kernels()
    diff_fn, gather_k, gather_v_bias = kernels
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    snap = ss.CacheSnapshot(cache)
    e0 = engines[0]
    _, S_gpu, H, Dh = e0._k_gpu.shape
    bs = int(e0.block_size)
    Dmax = max(D_LIST)
    k_s = torch.empty((B, Dmax * bs, H, Dh), dtype=e0._k_gpu.dtype, device="cuda")
    v_s = torch.empty_like(k_s)
    b_s = torch.empty((B, Dmax * bs, H), dtype=e0._kv_bias_gpu.dtype, device="cuda")
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    elem = e0._k_gpu.element_size()

    def loaded_now():
        return int(torch.stack([(e._load_mask >= 0).sum() for e in engines]).sum())

    def prewarm_all(T):
        oks = [mc.prewarm(e, l, T[l], lay.total_cis, kernels).ok for l, (lay, e) in enumerate(zip(layers, engines))]
        return bool(torch.stack(oks).all())

    def ids_for(D):
        # host block ids 0..D-1 per (head, request): valid prompt blocks; the DESTINATION is the scratch, never the cache
        return torch.arange(D, dtype=torch.int32, device="cuda").view(1, 1, D).expand(H, B, D).contiguous()

    def side_triton(D, ids_hbd):
        n = D * bs
        for lay, e in zip(layers, engines):
            gather_k(k_s[:, :n], e._k_cpu, ids_hbd, bs)
            gather_v_bias(v_s[:, :n], e._v_cpu, b_s[:, :n], lay.total_cis, ids_hbd, bs)

    def side_memcpy(D):
        n = D * bs
        for e in engines:
            k_s[:, :n].copy_(e._k_cpu[:, :n], non_blocking=True)
            v_s[:, :n].copy_(e._v_cpu[:, :n], non_blocking=True)

    def side_bytes(D):
        return 2 * B * D * bs * H * Dh * elem * nl          # K + V over all layers (the bias, 4 B/row, is not counted)

    def timed(fn_main=None, fn_side=None):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); tm = torch.cuda.Event(enable_timing=True); ts = torch.cuda.Event(enable_timing=True)
        t0.record(main)
        if fn_side is not None:
            side.wait_event(t0)
            with torch.cuda.stream(side):
                fn_side()
                ts.record(side)
        out = fn_main() if fn_main is not None else None
        tm.record(main)
        torch.cuda.synchronize()
        return out, (t0.elapsed_time(tm) if fn_main is not None else 0.0), (t0.elapsed_time(ts) if fn_side is not None else 0.0)

    rows, fails = [], 0
    trans = None
    for it in range(VA.N):
        tok = forced[:, it:it + 1]
        if it < VA.WARM:
            model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
            torch.cuda.synchronize()
            position_ids = position_ids + 1
            continue
        if trans is None:
            trans = ss.transient_ids(model)
            for D in D_LIST:                                    # JIT / first touch of the scratch shapes, untimed
                timed(None, lambda: side_triton(D, ids_for(D))); timed(None, lambda: side_memcpy(D))
        t_wall = time.time()
        snap.take()
        for e in engines:
            mc.flush_map(e)
        lg_ref, _ = VA._timed_call(lambda: model.decode_inference(tok, cu, position_ids, cache))
        T = [mc.record_targets(e, l) for l, e in enumerate(engines)]

        def fresh():
            snap.restore(); ss.assert_transients_intact(model, trans)
            ok = prewarm_all(T)
            if not ok:
                raise RuntimeError("prewarm failed at step %d" % it)

        tA = []
        for rep in range(REPS):
            fresh()
            lg, t, _ = timed(lambda: model.decode_inference(tok, cu, position_ids, cache))
            ok = torch.equal(lg, lg_ref) and loaded_now() == 0
            fails += int(not ok); tA.append(t)
        tA_mean = sum(tA) / len(tA)
        for D in D_LIST:
            ids_hbd = ids_for(D)
            for arm, fn_side in (("triton", lambda: side_triton(D, ids_hbd)), ("memcpy", lambda: side_memcpy(D))):
                tS = []
                for rep in range(REPS):
                    _, _, t = timed(None, fn_side); tS.append(t)
                tS_mean = sum(tS) / len(tS)
                tM_c, tS_c, tC = [], [], []
                for rep in range(REPS):
                    fresh()
                    lg, tm_, ts_ = timed(lambda: model.decode_inference(tok, cu, position_ids, cache), fn_side)
                    ok = torch.equal(lg, lg_ref) and loaded_now() == 0
                    fails += int(not ok)
                    tM_c.append(tm_); tS_c.append(ts_); tC.append(max(tm_, ts_))
                tM_cm, tS_cm, tC_m = (sum(x) / len(x) for x in (tM_c, tS_c, tC))
                lo, hi = min(tA_mean, tS_mean), max(tA_mean, tS_mean)
                by = side_bytes(D)
                rows.append(dict(batch=B, L=VA.L, step=it, D=D, arm=arm, reps=REPS, t_A_ms=tA_mean, t_A_std=VA._std(tA), t_S_alone_ms=tS_mean,
                                 t_main_in_C_ms=tM_cm, t_side_in_C_ms=tS_cm, t_C_ms=tC_m, kappa=(tC_m - hi) / lo if lo > 0 else float("nan"),
                                 main_slowdown=tM_cm / tA_mean - 1, side_slowdown=tS_cm / tS_mean - 1,
                                 bytes=by, bw_alone_gbps=by / (tS_mean * 1e6), bw_in_C_gbps=by / (tS_cm * 1e6), serial_sum_ms=tA_mean + tS_mean))
                print("[overlap] step %d D=%d %s: A %.1f  S %.1f (%.1f GB/s)  C %.1f (main %.1f, side %.1f)  kappa %.3f  main +%.1f%%  side +%.1f%%"
                      % (it, D, arm, tA_mean, tS_mean, by / (tS_mean * 1e6), tC_m, tM_cm, tS_cm, rows[-1]["kappa"], 100 * rows[-1]["main_slowdown"], 100 * rows[-1]["side_slowdown"]), flush=True)
        snap.restore(); ss.assert_transients_intact(model, trans)
        model.decode_inference(tok, cu, position_ids, cache)          # the advance (natural step)
        torch.cuda.synchronize()
        position_ids = position_ids + 1
        print("[overlap] step %d done in %.0fs (fails so far %d)" % (it, time.time() - t_wall, fails), flush=True)
    return dict(meta=meta, rows=rows, fails=fails, D_list=list(D_LIST), reps=REPS, pred=PRED, peak_gb=torch.cuda.max_memory_allocated() / 1e9)


def summarize(payloads):
    from collections import defaultdict
    acc = defaultdict(list)
    for p in payloads:
        for r in p["rows"]:
            acc[(r["batch"], r["D"], r["arm"])].append(r)
    L = ["## E3 overlap: resident step A vs side transfer S vs both C (mean over gated steps x reps)", "",
         "| B | D | arm | t_A ms | t_S alone ms (GB/s) | t_C ms | main in C ms (+%) | side in C ms (+%, GB/s) | serial sum | kappa |", "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts, summ = {}, {}
    for key in sorted(acc):
        rs = acc[key]; m = lambda k: sum(r[k] for r in rs) / len(rs)
        B, D, arm = key
        summ[str(key)] = dict(B=B, D=D, arm=arm, n=len(rs), t_A=m("t_A_ms"), t_S=m("t_S_alone_ms"), t_C=m("t_C_ms"), kappa=m("kappa"), bw_alone=m("bw_alone_gbps"), bw_in_C=m("bw_in_C_gbps"), main_slowdown=m("main_slowdown"), side_slowdown=m("side_slowdown"))
        s = summ[str(key)]
        L.append("| %d | %d | %s | %.1f | %.1f (%.1f) | %.1f | %.1f (%+.0f%%) | %.1f (%+.0f%%, %.1f) | %.1f | %.3f |" % (B, D, arm, s["t_A"], s["t_S"], s["bw_alone"], s["t_C"], m("t_main_in_C_ms"), 100 * s["main_slowdown"], m("t_side_in_C_ms"), 100 * s["side_slowdown"], s["bw_in_C"], m("serial_sum_ms"), s["kappa"]))
    for (B, D, arm), rs in acc.items():
        if D >= 8:
            k = sum(r["kappa"] for r in rs) / len(rs); bw = sum(r["bw_alone_gbps"] for r in rs) / len(rs)
            if arm == "triton":
                verdicts["P-E3-1 kappa_triton in [0.10, 0.40] (B=%d, D=%d)" % (B, D)] = ("PASS" if PRED["kappa_triton"][0] <= k <= PRED["kappa_triton"][1] else ("REFUTED (serialized)" if k > 0.5 else "REFUTED")) + " (%.3f)" % k
            else:
                verdicts["P-E3-2 kappa_memcpy <= 0.05 (B=%d, D=%d)" % (B, D)] = ("PASS" if k <= PRED["kappa_memcpy_max"] else ("REFUTED > 0.10" if k > 0.10 else "outside 0.05, inside 0.10")) + " (%.3f)" % k
                verdicts["P-E3-3 memcpy BW alone >= 20 GB/s (B=%d, D=%d)" % (B, D)] = ("PASS" if bw >= PRED["memcpy_bw_min_gbps"] else "REFUTED") + " (%.1f)" % bw
    fails = sum(p["fails"] for p in payloads)
    verdicts["correctness: logits torch.equal to the reference and 0 loads in every timed A / C step"] = "PASS" if fails == 0 else "FAILED (%d)" % fails
    L += ["", "## Registered predictions (ledger 2026-09-20 'E3 REGISTERED')", "| prediction | verdict |", "|---|---|"] + ["| %s | %s |" % kv for kv in verdicts.items()]
    L += ["", "FALSIFIERS: " + PRED["falsifiers"]]
    return "\n".join(L) + "\n", dict(summary=summ, verdicts=verdicts, fails=fails)


if __name__ == "__main__":
    if os.environ.get("NOSI_OVERLAP_MODE", "run") == "table":
        import glob
        pays = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "overlap_*.json")))]
        text, summ = summarize(pays)
        print(text); open(os.path.join(OUT, "overlap_table.md"), "w").write(text); json.dump(summ, open(os.path.join(OUT, "overlap_table.json"), "w"), indent=1)
        sys.exit(0 if pays and summ["fails"] == 0 else 1)
    VA.check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    corpus = VA.load_corpus(path, B_REQ)
    model = VA.load_model(path)
    ids, docs, distinct = VA.pick_batch(corpus, B_REQ, 0)
    print("[docs] %s%s (%d distinct)  L=%d N=%d D=%s reps=%d" % (docs[:8], "..." if len(docs) > 8 else "", distinct, VA.L, VA.N, D_LIST, REPS), flush=True)
    payload = run(model, ids)
    payload["docs"] = docs
    fn = os.path.join(OUT, "%s.json" % TAG)
    json.dump(payload, open(fn, "w"), indent=1)
    print("[overlap] saved %s (peak %.2f GB, fails %d)" % (fn, payload["peak_gb"], payload["fails"]), flush=True)
    sys.exit(min(payload["fails"], 200))
