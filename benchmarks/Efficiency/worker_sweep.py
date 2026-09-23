"""Corrected GPU-worker (CTA) sweep of the throttled SM gather beside NOSI's resident decode step
(authorized 2026-09-23 via Codex; ledger 'WORKER SWEEP REGISTERED'). Supersedes the overlap_bench timings, which
started the step's clock before the host enqueued the transfer (literature audit, ledger 4fb9513).

What is fixed and checked here:
  * TIMER: the main stream first runs a GPU sleep (torch.cuda._sleep) while the host enqueues the side transfer
    behind an event recorded after the sleep; the step's start event is recorded after the enqueue, so host launch
    time is never inside the step's bracket. host_lag_ms = elapsed(side gate, step start) must be ~0 (flagged if > 0.1).
  * SAME SOURCE, SAME IDS, SAME DESTINATIONS: every arm reads NOSI's own pinned host cache (engine._k_cpu / _v_cpu of
    every layer) at host blocks 0..D-1 of every (request, KV head) and writes the same GPU scratch slots. Arms:
      workersW   the existing throttled persistent Triton gather (flash_h2d_persistent.py) with W CTAs of 128 threads
      dma2d      the matched copy-engine reference: cudaMemcpy2DAsync of exactly those bytes (per layer, K and V:
                 B rows of D x 64 x H x Dh x 2 bytes at the host cache's request pitch) into the same scratch
  * OVERLAP: per concurrent rep the side interval must cover >= 95% of the step's interval (overlap_frac); the side
    is sized (repeat passes) to last >= 1.2 x the step alone.
  * CORRECTNESS: every timed step loads 0 blocks and its logits are torch.equal to the flushed reference step; after
    every transfer rep the scratch equals the source blocks of the last layer (torch.equal).
  * STATE: a LIGHT restore (state_snapshot.CounterSnapshot with the post-reference map), so a near-capacity batch fits.
  * NUMA: /proc/self/numa_maps pages per node for NOSI's host cache, the process CPU affinity and node cpulists.
Env: WS_B (64), WS_D ('8 16'), WS_WORKERS ('1 2 4 8 16'), WS_N (8), WS_WARM (4), WS_REPS (5), WS_L (16128), WS_OUT, WS_TAG,
WS_SLEEP_MS (15). Mode WS_MODE=run | table.
"""
import ctypes
import json
import os
import sys
import time

os.environ.setdefault("NOSI_ALONE_MODE", "points")
for src, dst in (("WS_B", "NOSI_ALONE_B"), ("WS_L", "NOSI_ALONE_L"), ("WS_N", "NOSI_ALONE_N"), ("WS_WARM", "NOSI_ALONE_WARM"), ("WS_OUT", "NOSI_ALONE_OUT")):
    if src in os.environ:
        os.environ[dst] = os.environ[src]
os.environ.setdefault("NOSI_ALONE_DOCS", "1")
os.environ.setdefault("NOSI_ALONE_D", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import verify_alone as VA  # noqa: E402

D_LIST = tuple(int(x) for x in os.environ.get("WS_D", "8 16").split())
WORKERS = tuple(int(x) for x in os.environ.get("WS_WORKERS", "1 2 4 8 16").split())
REPS = int(os.environ.get("WS_REPS", "5"))
SLEEP_MS = float(os.environ.get("WS_SLEEP_MS", "15"))
OUT = VA.OUT
TAG = os.environ.get("WS_TAG", "sweep_b%d" % VA.BATCH)
TARGET = dict(bw_frac_of_dma=0.90, max_slowdown=0.05)
CUDART = "/venv/nosa/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"


def numa_pages(ptr: int):
    """pages per NUMA node of the mapping that contains `ptr`, from /proc/self/numa_maps (None if not found)."""
    best = None
    try:
        for line in open("/proc/self/numa_maps"):
            parts = line.split()
            start = int(parts[0], 16)
            if start <= ptr and (best is None or start > best[0]):
                best = (start, parts)
    except OSError as e:
        return dict(error=str(e))
    if best is None:
        return None
    nodes = {p.split("=")[0]: int(p.split("=")[1]) for p in best[1] if p.startswith("N") and "=" in p}
    return dict(mapping_start=hex(best[0]), policy=best[1][1], pages_per_node=nodes, flags=[p for p in best[1][2:] if not p.startswith("N")][:6])


def cpu_numa():
    aff = sorted(os.sched_getaffinity(0))
    nodes = {}
    base = "/sys/devices/system/node"
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if d.startswith("node"):
            try:
                nodes[d] = open(os.path.join(base, d, "cpulist")).read().strip()
            except OSError:
                pass
    return dict(affinity=[aff[0], aff[-1], len(aff)] if aff else [], node_cpulists=nodes)


@torch.inference_mode()
def run(model, ids):
    from nosi import state_snapshot as ss
    from nosi.verify import miss_control as mc
    from nosi.flash_cache_engine.flash_h2d_persistent import flash_h2d_persistent
    cudart = ctypes.CDLL(CUDART)
    cudart.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    cudart.cudaMemcpy2DAsync.restype = ctypes.c_int
    model, cache, logits, position_ids, forced, meta = VA._setup(model, ids)
    B, nl = ids.shape[0], model.num_layers
    layers = list(cache.layers)
    engines = [lay.cache_engine for lay in layers]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    e0 = engines[0]
    _, S_gpu, H, Dh = e0._k_gpu.shape
    bs = int(e0.block_size)
    Dmax = max(D_LIST)
    elem = e0._k_gpu.element_size()
    k_s = torch.zeros((B, Dmax * bs, H, Dh), dtype=e0._k_gpu.dtype, device="cuda")
    v_s = torch.zeros_like(k_s)
    k_ref = engines[-1]._k_cpu[:, :Dmax * bs].to("cuda")              # the last layer's source blocks: what the scratch holds after a pass
    v_ref = engines[-1]._v_cpu[:, :Dmax * bs].to("cuda")
    for e in engines:
        assert e._k_cpu.is_pinned() and e._k_cpu.is_contiguous() and e._k_cpu.stride(0) == engines[0]._k_cpu.stride(0)
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    sleep_cycles = int(SLEEP_MS * 1e-3 * 1.41e9)
    numa = dict(k_cpu_layer0=numa_pages(engines[0]._k_cpu.data_ptr()), k_cpu_layer31=numa_pages(engines[-1]._k_cpu.data_ptr()),
                v_cpu_layer0=numa_pages(engines[0]._v_cpu.data_ptr()), cpu=cpu_numa(),
                k_cpu_bytes_per_layer=engines[0]._k_cpu.numel() * elem)
    print("[sweep] NUMA %s" % json.dumps(numa), flush=True)

    def loaded_now():
        return int(torch.stack([(e._load_mask >= 0).sum() for e in engines]).sum())

    def ids_for(D):
        return torch.arange(D, dtype=torch.int32, device="cuda").view(1, 1, D).expand(H, B, D).contiguous()

    def pass_bytes(D):
        return 2 * B * H * D * bs * Dh * elem * nl                     # K + V, all layers, one pass

    def side_gather(W, D, ids_hbd, passes):
        n = D * bs
        def f():
            for _ in range(passes):
                for e in engines:
                    flash_h2d_persistent(k_s[:, :n], e._k_cpu, ids_hbd, bs, n_ctas=W)
                    flash_h2d_persistent(v_s[:, :n], e._v_cpu, ids_hbd, bs, n_ctas=W)
        return f

    def side_dma(D, passes):
        n = D * bs
        width = n * H * Dh * elem
        spitch = engines[0]._k_cpu.stride(0) * elem
        dpitch = k_s.stride(0) * elem
        def f():
            st = ctypes.c_void_p(side.cuda_stream)
            for _ in range(passes):
                for e in engines:
                    for src, dst in ((e._k_cpu, k_s), (e._v_cpu, v_s)):
                        rc = cudart.cudaMemcpy2DAsync(ctypes.c_void_p(dst.data_ptr()), dpitch, ctypes.c_void_p(src.data_ptr()), spitch, width, B, 1, st)
                        if rc != 0:
                            raise RuntimeError("cudaMemcpy2DAsync rc=%d" % rc)
        return f

    def bracket(fn_main, fn_side):
        """GPU sleep on main -> gate -> side enqueued behind the gate -> t0 -> main work. Returns ms dict."""
        torch.cuda.synchronize()
        ev = {k: torch.cuda.Event(enable_timing=True) for k in ("pre", "gate", "t0", "tm", "ts")}
        ev["pre"].record(main)
        torch.cuda._sleep(sleep_cycles)
        ev["gate"].record(main)
        h0 = time.perf_counter()
        if fn_side is not None:
            side.wait_event(ev["gate"])
            with torch.cuda.stream(side):
                fn_side()
                ev["ts"].record(side)
        host_enqueue_ms = 1000 * (time.perf_counter() - h0)
        ev["t0"].record(main)
        out = fn_main() if fn_main is not None else None
        ev["tm"].record(main)
        torch.cuda.synchronize()
        r = dict(host_lag_ms=ev["gate"].elapsed_time(ev["t0"]), sleep_ms=ev["pre"].elapsed_time(ev["gate"]), host_enqueue_ms=host_enqueue_ms)
        if fn_main is not None:
            r["main_ms"] = ev["t0"].elapsed_time(ev["tm"])
        if fn_side is not None:
            r["side_ms"] = ev["gate"].elapsed_time(ev["ts"])
        if fn_main is not None and fn_side is not None:
            r["overlap_frac"] = min(r["side_ms"] - r["host_lag_ms"], r["main_ms"]) / r["main_ms"] if r["main_ms"] > 0 else float("nan")
        return out, r

    snap = ss.CounterSnapshot(cache)
    rows, fails, trans = [], 0, None
    step_fn = lambda tok: (lambda: model.decode_inference(tok, cu, position_ids, cache))
    for it in range(VA.N):
        tok = forced[:, it:it + 1]
        if it < VA.WARM:
            model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
            torch.cuda.synchronize()
            position_ids = position_ids + 1
            continue
        if trans is None:
            trans = ss.transient_ids(model)
            for D in D_LIST:                                            # JIT / first touch, untimed
                bracket(None, side_gather(1, D, ids_for(D), 1)); bracket(None, side_dma(D, 1))
        t_wall = time.time()
        snap.take()                                                     # pre-step counters and tables
        for e in engines:
            mc.flush_map(e)
        lg_ref = model.decode_inference(tok, cu, position_ids, cache)   # the flushed reference: loads the whole selection, map -> T
        torch.cuda.synchronize()
        for i, e in enumerate(engines):                                 # the light restore returns to the post-reference map (the window holds T)
            slot = snap.layers[i]["engine"]
            for name in ("_block_map", "_new_block_map_buf"):
                slot[name].copy_(getattr(e, name))

        def restore():
            snap.restore(); ss.assert_transients_intact(model, trans)

        def resident_step():
            restore()
            return bracket(step_fn(tok), None)

        # decode alone (reps)
        alone = []
        for rep in range(REPS):
            lg, r = resident_step()
            ok = torch.equal(lg, lg_ref) and loaded_now() == 0
            fails += int(not ok); alone.append(r)
        t_alone = sorted(x["main_ms"] for x in alone)[len(alone) // 2]
        for D in D_LIST:
            ids_hbd = ids_for(D)
            arms = [("workers%d" % W, (lambda W=W: (lambda p: side_gather(W, D, ids_hbd, p)))()) for W in WORKERS] + [("dma2d", lambda p: side_dma(D, p))]
            for arm, mk in arms:
                # size the side to last >= 1.2 x the step alone
                _, r1 = bracket(None, mk(1))
                passes = max(1, int(np.ceil(1.2 * t_alone / max(r1["side_ms"] - r1["host_lag_ms"], 1e-3))))
                s_alone = []
                for rep in range(REPS):
                    k_s.zero_(); v_s.zero_()
                    _, r = bracket(None, mk(passes)); s_alone.append(r)
                    good = torch.equal(k_s[:, :D * bs], k_ref[:, :D * bs]) and torch.equal(v_s[:, :D * bs], v_ref[:, :D * bs])
                    fails += int(not good)
                conc = []
                for rep in range(REPS):
                    restore()
                    k_s.zero_(); v_s.zero_()
                    lg, r = bracket(step_fn(tok), mk(passes))
                    ok = torch.equal(lg, lg_ref) and loaded_now() == 0
                    good = torch.equal(k_s[:, :D * bs], k_ref[:, :D * bs]) and torch.equal(v_s[:, :D * bs], v_ref[:, :D * bs])
                    fails += int(not ok) + int(not good)
                    r.update(logits_equal=bool(ok), transfer_equal=bool(good))
                    conc.append(r)
                by = pass_bytes(D) * passes
                rows.append(dict(batch=B, step=it, D=D, arm=arm, workers=(int(arm[7:]) if arm.startswith("workers") else None), passes=passes, bytes=by,
                                 decode_alone_ms=[x["main_ms"] for x in alone], decode_conc_ms=[x["main_ms"] for x in conc],
                                 side_alone_ms=[x["side_ms"] - x["host_lag_ms"] for x in s_alone], side_conc_ms=[x["side_ms"] - x["host_lag_ms"] for x in conc],
                                 overlap_frac=[x["overlap_frac"] for x in conc], host_lag_ms=[x["host_lag_ms"] for x in conc + s_alone],
                                 host_enqueue_ms=[x["host_enqueue_ms"] for x in conc]))
                rr = rows[-1]
                med = lambda xs: float(np.median(xs))
                print("[sweep] step %d D=%d %-9s passes=%d  decode %.2f -> %.2f ms (%+.1f%%)  side alone %.1f GB/s, conc %.1f GB/s  overlap %.2f  host_lag max %.3f ms"
                      % (it, D, arm, passes, med(rr["decode_alone_ms"]), med(rr["decode_conc_ms"]), 100 * (med(rr["decode_conc_ms"]) / med(rr["decode_alone_ms"]) - 1),
                         by / (med(rr["side_alone_ms"]) * 1e6), by / (med(rr["side_conc_ms"]) * 1e6), min(rr["overlap_frac"]), max(rr["host_lag_ms"])), flush=True)
        restore()
        model.decode_inference(tok, cu, position_ids, cache)            # the advance: a resident step from the restored state
        torch.cuda.synchronize()
        position_ids = position_ids + 1
        print("[sweep] step %d done in %.0fs (fails %d)" % (it, time.time() - t_wall, fails), flush=True)
    return dict(meta=meta, rows=rows, fails=fails, numa=numa, D_list=list(D_LIST), workers=list(WORKERS), reps=REPS, sleep_ms=SLEEP_MS, target=TARGET,
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                device_total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9)


def table(out_dir):
    import glob
    pays = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, "sweep_*.json")))]
    L = ["# Corrected worker sweep: SM gather (W CTAs x 128 threads) vs matched copy engine, beside NOSI's resident decode step", "",
         "Medians (and p95) over gated steps x reps. GB/s = bytes / side interval. slowdown = median concurrent / median alone - 1. "
         "tok/s = B / decode ms. Valid = min overlap_frac >= 0.95 and max host_lag <= 0.1 ms and all correctness checks pass.", ""]
    L.append("| B | D | arm | passes | transfer alone GB/s | transfer concurrent GB/s | % of dma2d concurrent | decode alone median / p95 ms | decode concurrent median / p95 ms | slowdown | tok/s alone -> concurrent | min overlap | max host lag ms | meets target |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    verdict_rows = []
    for p in pays:
        B = p["meta"]["batch"]
        by_key = {}
        for r in p["rows"]:
            k = (r["D"], r["arm"])
            a = by_key.setdefault(k, dict(D=r["D"], arm=r["arm"], passes=r["passes"], bytes=r["bytes"], da=[], dc=[], sa=[], sc=[], ov=[], hl=[]))
            a["da"] += r["decode_alone_ms"]; a["dc"] += r["decode_conc_ms"]; a["sa"] += r["side_alone_ms"]; a["sc"] += r["side_conc_ms"]; a["ov"] += r["overlap_frac"]; a["hl"] += r["host_lag_ms"]
        dma = {D: v for (D, arm), v in by_key.items() if arm == "dma2d"}
        for (D, arm), a in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] != "dma2d", kv[0][1])):
            med, p95 = (lambda xs: float(np.median(xs))), (lambda xs: float(np.percentile(xs, 95)))
            bw_a, bw_c = a["bytes"] / (med(a["sa"]) * 1e6), a["bytes"] / (med(a["sc"]) * 1e6)
            d = dma.get(D)
            dma_c = d["bytes"] / (med(d["sc"]) * 1e6) if d else float("nan")
            slow = med(a["dc"]) / med(a["da"]) - 1
            valid = min(a["ov"]) >= 0.95 and max(a["hl"]) <= 0.1
            meets = valid and arm != "dma2d" and bw_c >= TARGET["bw_frac_of_dma"] * dma_c and slow <= TARGET["max_slowdown"]
            verdict_rows.append(dict(B=B, D=D, arm=arm, bw_conc=bw_c, frac_dma=bw_c / dma_c if dma_c == dma_c else None, slowdown=slow, valid=valid, meets=meets))
            L.append("| %d | %d | %s | %d | %.1f | %.1f | %.0f%% | %.2f / %.2f | %.2f / %.2f | %+.1f%% | %.0f -> %.0f | %.2f | %.3f | %s |" % (
                B, D, arm, a["passes"], bw_a, bw_c, 100 * bw_c / dma_c, med(a["da"]), p95(a["da"]), med(a["dc"]), p95(a["dc"]), 100 * slow,
                1000 * B / med(a["da"]), 1000 * B / med(a["dc"]), min(a["ov"]), max(a["hl"]), ("YES" if meets else ("n/a" if arm == "dma2d" else "no")) + ("" if valid else " (INVALID overlap/timer)")))
        L.append("")
        L.append("B=%d: correctness fails %d; peak allocated %.2f GB, reserved %.2f GB of %.1f GB; NUMA %s" % (B, p["fails"], p["peak_allocated_gb"], p["peak_reserved_gb"], p["device_total_gb"], json.dumps(p["numa"])))
        L.append("")
    met = [v for v in verdict_rows if v["meets"]]
    L.append("## Registered target (ledger 'WORKER SWEEP REGISTERED'): >= 90% of the matched copy-engine concurrent bandwidth with <= 5% decode slowdown")
    L.append("MET by: " + (", ".join("B=%d D=%d %s" % (v["B"], v["D"], v["arm"]) for v in met) if met else "NONE -- see the tradeoff columns above"))
    text = "\n".join(L) + "\n"
    open(os.path.join(out_dir, "sweep_table.md"), "w").write(text)
    json.dump(dict(verdicts=verdict_rows, fails=sum(p["fails"] for p in pays)), open(os.path.join(out_dir, "sweep_table.json"), "w"), indent=1)
    print(text)
    return 0 if pays and all(p["fails"] == 0 for p in pays) else 1


if __name__ == "__main__":
    if os.environ.get("WS_MODE", "run") == "table":
        sys.exit(table(OUT))
    VA.check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    corpus = VA.load_corpus(path, VA.BATCH)
    model = VA.load_model(path)
    ids, docs, distinct = VA.pick_batch(corpus, VA.BATCH, 0)
    print("[docs] %s%s (%d distinct books for %d requests)  L=%d N=%d D=%s workers=%s reps=%d" % (docs[:8], "..." if len(docs) > 8 else "", distinct, len(docs), VA.L, VA.N, D_LIST, WORKERS, REPS), flush=True)
    payload = run(model, ids)
    payload["docs"] = docs; payload["distinct_books"] = distinct
    fn = os.path.join(OUT, "%s.json" % TAG)
    json.dump(payload, open(fn, "w"), indent=1)
    print("[sweep] saved %s (fails %d, peak reserved %.2f GB)" % (fn, payload["fails"], payload["peak_reserved_gb"]), flush=True)
    sys.exit(min(payload["fails"], 200))
