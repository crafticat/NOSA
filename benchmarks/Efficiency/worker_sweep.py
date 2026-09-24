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
      hisparseW_<item>  SGLang HiSparse's copy-only kernel (copy_cache_planned_kernel, sglang 87db743, vendored
                 verbatim in nosi/flash_cache_engine/hisparse_copy/) with a grid of W blocks x WS_HS_THREADS threads, one
                 launch per layer (K and V), fed a plan built from the SAME ids by hisparse_copy/plan.py:
                 i256 = one token row of one head per item (the exact per-head adapter), i32k = one (request, block) with
                 both heads per item (exact here: both heads load blocks 0..D-1 into slots 0..D-1). A threads value other
                 than 1024 adds '_t<threads>' to the arm name. Plan-build time (host, upload, device) is recorded per
                 (D, item), outside every bracket.
  * OVERLAP: per concurrent rep the side interval must cover >= 95% of the step's interval (overlap_frac). SIZING
    (side_sizing.py): passes so the side lasts >= 1.2 x the step alone, then a TRIAL concurrent bracket, raising
    passes until the side lasts >= 1.15 x the CONCURRENT step (<= WS_SIZE_TRIALS trials; the miniature's 16-CTA
    arm had overlap 0.92 because the slowed step outlasted a side sized from the step alone).
  * CORRECTNESS: every timed step loads 0 blocks and its logits are torch.equal to the flushed reference step; after
    every transfer rep the scratch equals the source blocks of the last layer (torch.equal).
  * STATE: a LIGHT restore (state_snapshot.CounterSnapshot with the post-reference map), so a near-capacity batch fits.
  * NUMA: /proc/self/numa_maps read ONCE (numa_maps.py); pages per node of EVERY layer's _k_cpu and _v_cpu over every
    mapping each buffer touches (review NB1 / NB11: at B = 336 the cache splits across nodes by layer), the process
    CPU affinity and node cpulists. k_nodes / v_nodes = one character per layer ('0', '1', 's' = split).
  * DURING-STEP (review B3): one event per launch unit (layer, K and V) on the side stream; every concurrent rep
    reports the bytes whose unit ended inside the step / step ms (a lower bound), next to the whole-interval GB/s.
  * JSON is flushed after every gated step (review NB10; 'partial': true until the run ends).
Env: WS_B (64), WS_D ('8 16'), WS_WORKERS ('1 2 4 8 16'), WS_N (8), WS_WARM (4), WS_REPS (5), WS_L (16128), WS_OUT, WS_TAG,
WS_SLEEP_MS (15), WS_HS_BLOCKS ('1 2 4 8 16'; empty = no HiSparse arm), WS_HS_THREADS ('1024'; 256 and/or 1024),
WS_HS_ITEMS ('i256 i32k'), WS_SIZE_TRIALS (3). Mode WS_MODE=run | table.

STRATA / HICACHE ARMS (stage S2 of the Strata / timeline experiment; all default OFF, so the old runs are unchanged):
  strata<W>_t1024_<item>      SGLang's AOT kvcacheio transfer_kv_per_layer (sglang 87db743, vendored verbatim in
                              nosi/flash_cache_engine/sglang_hicache/): the CLOSEST RELEASED IMPLEMENTATION of Strata's IO
                              kernel (Strata's first author; the paper releases no code). W blocks x 32 warps.
  hicachejit<W>_t1024_<item>  SGLang's default JIT hicache_transfer_per_layer (87db743; NOT a Strata author, NOT Strata).
  items: i256 = the NOSI ADAPTER (per-head 256-B rows, exact for any descriptor); i512 = SGLang's NATIVE token item (both
  heads), exact here only because the sweep's ids are coupled (both heads load blocks 0..D-1 into slots 0..D-1).
  Index tensors (int64, one per (D, item), the same for every layer: the sweep's ids are the same in every layer) are
  PREBUILT outside every bracket: every row is a 'prebuilt-plan copy microbenchmark, NOT an integrated LRU getter
  baseline'; the per-layer plan cost an integrated getter would pay (miss-mask compaction + count sync + index build) is
  timed separately (plans[*].nosi_plan_cost) and never added to a row. First touch of every new arm checks layers 0,
  L/2 and L-1 one at a time against their own source (NaN-poisoned scratch).
  WS_ST_BLOCKS (''), WS_ST_ITEMS ('i256 i512'), WS_JIT_BLOCKS (''), WS_JIT_ITEMS ('i256'), WS_ARMS ('' = every arm; else only
  the named arms run).
SELF-REPORTING TIMER (every bracket, measurement only): host_submit_ms, gate_done_at_submit_end, submit_end_ms (see
hisparse_repro.make_bracket, the same lines).
STAGE T, LAUNCH-CORRELATED TIMELINE (WS_T_ARMS, '' = off): after the arms of the FIRST D of a gated step, for each named
arm ('alone', 'alone_late' = the negative control: the host waits for the gate then sleeps WS_T_LATE_MS before it submits
the step, or any arm name of this run): WS_T_UNTRACED untraced reps, then cudaProfilerStart, 1 + WS_T_REPS traced reps
(the first is CUPTI warm-up), cudaProfilerStop, then WS_T_UNTRACED untraced reps again. Every rep sits in an NVTX range
'lt|b<B>|<arm>|step<it>|<phase>|rep<k>' and the step's own submission in 'lt_submit|<the same label>'; run the process under
`nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=repeat` and analyse with
launch_timeline.py. Records land in the JSON's 'timeline' list.
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

import numa_maps as NM  # noqa: E402
import side_sizing as SZ  # noqa: E402
import verify_alone as VA  # noqa: E402

D_LIST = tuple(int(x) for x in os.environ.get("WS_D", "8 16").split())
WORKERS = tuple(int(x) for x in os.environ.get("WS_WORKERS", "1 2 4 8 16").split())
HS_BLOCKS = tuple(int(x) for x in os.environ.get("WS_HS_BLOCKS", "1 2 4 8 16").split())
HS_THREADS = tuple(int(x) for x in os.environ.get("WS_HS_THREADS", "1024").split())
HS_ITEMS = tuple(os.environ.get("WS_HS_ITEMS", "i256 i32k").split())
SIZE_TRIALS = int(os.environ.get("WS_SIZE_TRIALS", "3"))
REPS = int(os.environ.get("WS_REPS", "5"))
SLEEP_MS = float(os.environ.get("WS_SLEEP_MS", "15"))
OUT = VA.OUT
TAG = os.environ.get("WS_TAG", "sweep_b%d" % VA.BATCH)
TARGET = dict(bw_frac_of_dma=0.90, max_slowdown=0.05)
ST_BLOCKS = tuple(int(x) for x in os.environ.get("WS_ST_BLOCKS", "").split())
ST_ITEMS = tuple(os.environ.get("WS_ST_ITEMS", "i256 i512").split())
JIT_BLOCKS = tuple(int(x) for x in os.environ.get("WS_JIT_BLOCKS", "").split())
JIT_ITEMS = tuple(os.environ.get("WS_JIT_ITEMS", "i256").split())
ARMS_ONLY = tuple(os.environ.get("WS_ARMS", "").split())
T_ARMS = tuple(os.environ.get("WS_T_ARMS", "").split())
T_REPS = int(os.environ.get("WS_T_REPS", "3"))
T_UNTRACED = int(os.environ.get("WS_T_UNTRACED", "3"))
T_LATE_MS = float(os.environ.get("WS_T_LATE_MS", "2.0"))
CUDART = "/venv/nosa/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"


def hs_arm_name(W: int, item: str, threads: int) -> str:
    """hisparse<W>_<item>, plus _t<threads> when the block is not the upstream default of 1024 threads."""
    return "hisparse%d_%s" % (W, item) + ("" if threads == 1024 else "_t%d" % threads)


def st_arm_name(family: str, W: int, item: str) -> str:
    """strata<W>_t1024_<item> (the AOT kernel) or hicachejit<W>_t1024_<item> (the JIT kernel)."""
    if family not in ("strata", "hicachejit"):
        raise ValueError(family)
    return "%s%d_t1024_%s" % (family, W, item)


def arm_order(arm: str):
    """Table order: dma2d, workers by W, hisparse by (item, threads suffix) then W, strata, hicachejit."""
    import re
    if arm == "dma2d":
        return (0, "", 0)
    m = re.match(r"(workers|hisparse|strata|hicachejit)(\d+)(.*)$", arm)
    if m:
        return ({"workers": 1, "hisparse": 2, "strata": 3, "hicachejit": 4}[m.group(1)], m.group(3), int(m.group(2)))
    return (5, arm, 0)


def arm_selected(arm: str) -> bool:
    return not ARMS_ONLY or arm in ARMS_ONLY


def self_report(r: dict) -> dict:
    """The self-reporting timer fields of one bracket (measurement only; see the module docstring)."""
    return dict(host_submit_ms=r.get("host_submit_ms"), gate_done_at_submit_end=r.get("gate_done_at_submit_end"),
                submit_end_ms=r.get("submit_end_ms"), host_enqueue_ms=r.get("host_enqueue_ms"), main_ms=r.get("main_ms"))


def timeline_main(fn, label: str, late_ms: float = 0.0):
    """fn inside an NVTX range 'lt_submit|<label>' (the step's own submissions). late_ms > 0 = the NEGATIVE CONTROL: wait
    for the gate (the main stream holds only the gate's sleep and events here), sleep late_ms on the host, then submit, so
    the GPU is idle for >= late_ms before the step's first kernel; the timeline metric must count it as starvation."""
    import torch as _t

    def g():
        if late_ms > 0:
            _t.cuda.current_stream().synchronize()
            time.sleep(late_ms / 1000.0)
        _t.cuda.nvtx.range_push("lt_submit|" + label)
        try:
            return fn()
        finally:
            _t.cuda.nvtx.range_pop()
    return g


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


def numa_layers(vmas, engines, elem):
    """Per-layer NUMA split of NOSI's pinned host cache (review NB1): every layer's _k_cpu / _v_cpu over every mapping
    it touches (numa_maps.range_pages), from ONE parse of /proc/self/numa_maps."""
    per = []
    for i, e in enumerate(engines):
        nb = e._k_cpu.numel() * elem
        k = NM.range_pages(vmas, e._k_cpu.data_ptr(), nb)
        v = NM.range_pages(vmas, e._v_cpu.data_ptr(), nb)
        per.append(dict(layer=i, k=(k or {}).get("pages_per_node"), v=(v or {}).get("pages_per_node"),
                        k_vmas=(k or {}).get("n_vmas"), v_vmas=(v or {}).get("n_vmas"),
                        k_covered_bytes=(k or {}).get("covered_bytes"), v_covered_bytes=(v or {}).get("covered_bytes"),
                        k_node=NM.node_of(k), v_node=NM.node_of(v)))
    return dict(per_layer=per, k_nodes="".join(p["k_node"] for p in per), v_nodes="".join(p["v_node"] for p in per), bytes_per_layer=nb if engines else 0)


@torch.inference_mode()
def run(model, ids, flush=None):
    from nosi import state_snapshot as ss
    from nosi.verify import miss_control as mc
    from nosi.flash_cache_engine.flash_h2d_persistent import flash_h2d_persistent
    cudart = ctypes.CDLL(CUDART)
    cudart.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    cudart.cudaMemcpy2DAsync.restype = ctypes.c_int
    model, cache, logits, position_ids, forced, run_meta = VA._setup(model, ids)
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
    aux = torch.cuda.Stream()                                           # idle: its event marks the step's submission end
    main = torch.cuda.current_stream()
    sleep_cycles = int(SLEEP_MS * 1e-3 * 1.41e9)
    try:
        vmas = NM.parse()                                               # ONE read of /proc/self/numa_maps (review NB1)
        layers_numa = numa_layers(vmas, engines, elem)
    except OSError as e:
        vmas, layers_numa = [], dict(error=str(e))
    numa = dict(k_cpu_layer0=NM.pages_at(engines[0]._k_cpu.data_ptr(), vmas), k_cpu_layer31=NM.pages_at(engines[-1]._k_cpu.data_ptr(), vmas),
                v_cpu_layer0=NM.pages_at(engines[0]._v_cpu.data_ptr(), vmas), cpu=cpu_numa(),
                k_cpu_bytes_per_layer=engines[0]._k_cpu.numel() * elem, layers=layers_numa)
    print("[sweep] NUMA k_nodes %s v_nodes %s; %s" % (layers_numa.get("k_nodes"), layers_numa.get("v_nodes"),
                                                     json.dumps({k: v for k, v in numa.items() if k != "layers"})), flush=True)

    def loaded_now():
        return int(torch.stack([(e._load_mask >= 0).sum() for e in engines]).sum())

    def ids_for(D):
        return torch.arange(D, dtype=torch.int32, device="cuda").view(1, 1, D).expand(H, B, D).contiguous()

    def pass_bytes(D):
        return 2 * B * H * D * bs * Dh * elem * nl                     # K + V, all layers, one pass

    def side_gather(W, D, ids_hbd, passes):
        n = D * bs
        unit = pass_bytes(D) // nl
        def f():
            for _ in range(passes):
                for e in engines:
                    flash_h2d_persistent(k_s[:, :n], e._k_cpu, ids_hbd, bs, n_ctas=W)
                    flash_h2d_persistent(v_s[:, :n], e._v_cpu, ids_hbd, bs, n_ctas=W)
                    SZ.LAUNCH_LOG.mark(unit)
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
                    SZ.LAUNCH_LOG.mark(2 * B * width)
        return f

    def bracket(fn_main, fn_side):
        """GPU sleep on main -> gate -> side enqueued behind the gate -> t0 -> main work. Returns ms dict."""
        torch.cuda.synchronize()
        SZ.LAUNCH_LOG.reset()
        ev = {k: torch.cuda.Event(enable_timing=True) for k in ("pre", "gate", "t0", "tm", "ts", "sub")}
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
        h1 = time.perf_counter()
        out = fn_main() if fn_main is not None else None
        h2 = time.perf_counter()
        gate_done = ev["gate"].query()
        ev["sub"].record(aux)
        ev["tm"].record(main)
        torch.cuda.synchronize()
        r = dict(host_lag_ms=ev["gate"].elapsed_time(ev["t0"]), sleep_ms=ev["pre"].elapsed_time(ev["gate"]), host_enqueue_ms=host_enqueue_ms)
        r.update(host_submit_ms=1000 * (h2 - h1), gate_done_at_submit_end=bool(gate_done), submit_end_ms=ev["gate"].elapsed_time(ev["sub"]))
        if fn_main is not None:
            r["main_ms"] = ev["t0"].elapsed_time(ev["tm"])
        if fn_side is not None:
            r["side_ms"] = ev["gate"].elapsed_time(ev["ts"])
        if fn_main is not None and fn_side is not None:
            r["overlap_frac"] = min(r["side_ms"] - r["host_lag_ms"], r["main_ms"]) / r["main_ms"] if r["main_ms"] > 0 else float("nan")
        if fn_main is not None and fn_side is not None and SZ.LAUNCH_LOG.n:
            r.update(SZ.during_step([ev["gate"].elapsed_time(e) for e, _ in SZ.LAUNCH_LOG.marks()], [b for _, b in SZ.LAUNCH_LOG.marks()], r["host_lag_ms"], r["main_ms"]))
        return out, r

    snap = ss.CounterSnapshot(cache)
    rows, fails, trans = [], 0, None
    tl_records, st_verify = [], {}

    # HiSparse arms: the vendored copy_cache_planned kernel fed a plan of the SAME ids (plan.py), built once per
    # (D, item) OUTSIDE every bracket; host / upload / device build times recorded, the two builds must agree.
    hs, plans, plan_log = None, {}, []
    if HS_BLOCKS:
        from nosi.flash_cache_engine import hisparse_copy as hs
        hs.load()
        S_cpu = int(e0._k_cpu.shape[1])
        for D in D_LIST:
            for item in HS_ITEMS:
                try:
                    pl, tm = hs.timed_build(ids_for(D).cpu(), device="cuda", s_cpu=S_cpu, s_dst=int(k_s.shape[1]), n_heads=H, head_dim=Dh,
                                            elem_size=elem, block_rows=bs, item=item)
                except hs.PlanRefused as e:                             # recorded, never approximated; that item's arms are skipped
                    plan_log.append(dict(D=D, item=item, refused=str(e)))
                    print("[sweep] plan D=%d %s REFUSED: %s" % (D, item, e), flush=True)
                    continue
                hs.validate_plan(pl)
                assert pl.bytes_per_launch * nl == pass_bytes(D), (D, item, pl.bytes_per_launch * nl, pass_bytes(D))
                plans[(D, item)] = pl
                tm.update(D=D, item=item)
                plan_log.append(tm)
                fails += int(not tm["device_build_equal"])
                print("[sweep] plan D=%d %s: %d items of %d B (stride %d); build host %.2f ms, upload %.2f ms, device %.2f ms, device==host %s"
                      % (D, item, tm["n_items"], tm["item_size_bytes"], tm["plan_stride"], tm["build_host_ms"], tm["upload_ms"], tm["build_device_ms"],
                         tm["device_build_equal"]), flush=True)

    def side_hisparse(W, T, D, item, passes):
        pl = plans[(D, item)]
        def f():
            for _ in range(passes):
                for e in engines:
                    hs.copy_plan(pl, e._k_cpu, e._v_cpu, k_s, v_s, W, T)
                    SZ.LAUNCH_LOG.mark(pl.bytes_per_launch)
        return f

    # STRATA / HICACHE arms (module docstring): prebuilt int64 index tensors per (D, item), the same for every layer;
    # the per-layer plan cost an integrated getter would pay is timed separately and never enters a bracket.
    hc, st_idx, st_log = None, {}, []
    if ST_BLOCKS or JIT_BLOCKS:
        from nosi.flash_cache_engine import sglang_hicache as hc
        hc.load_aot()
        if JIT_BLOCKS:
            hc.load_jit()
        S_cpu_st, S_dst_st = int(e0._k_cpu.shape[1]), int(k_s.shape[1])
        for D in D_LIST:
            for item in sorted(set((ST_ITEMS if ST_BLOCKS else ()) + (JIT_ITEMS if JIT_BLOCKS else ()))):
                ids_g = ids_for(D)
                try:
                    t0 = time.perf_counter()
                    si, di = hc.nosi_items(ids_g.cpu(), s_cpu=S_cpu_st, s_dst=S_dst_st, block_rows=bs, item=item)
                    t1 = time.perf_counter()
                except hc.PlanRefused as e:
                    st_log.append(dict(D=D, item=item, refused=str(e)))
                    print("[sweep] strata/jit index D=%d %s REFUSED: %s" % (D, item, e), flush=True)
                    continue
                isz = hc.ITEMS[item]
                recs_src, recs_dst = e0._k_cpu.numel() * elem // isz, k_s.numel() * elem // isz
                hc.validate_indices(si, di, recs_src, recs_dst, isz)
                assert 2 * si.numel() * isz * nl == pass_bytes(D), (D, item, 2 * si.numel() * isz * nl, pass_bytes(D))
                t2 = time.perf_counter()
                si_g, di_g = si.to("cuda", non_blocking=True), di.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                st_idx[(D, item)] = (si_g, di_g, isz)
                cost = hc.timed_nosi_build(ids_g, s_cpu=S_cpu_st, s_dst=S_dst_st, block_rows=bs, item=item)
                st_log.append(dict(D=D, item=item, item_size_bytes=isz, n_items=int(si.numel()), build_host_ms=1000 * (t1 - t0),
                                   upload_ms=1000 * (t3 - t2), nosi_plan_cost=cost, plan_cost=hc.PLAN_COST_LABEL, bench_label=hc.BENCH_LABEL,
                                   aot_grid=dict((W, hc.aot_grid(int(si.numel()), W, 32)[1]) for W in ST_BLOCKS),
                                   jit_grid=dict((W, hc.jit_grid(int(si.numel()), W)) for W in JIT_BLOCKS)))
                print("[sweep] strata/jit index D=%d %s: %d items of %d B; host build %.2f ms, upload %.2f ms; integrated per-layer cost "
                      "(EXCLUDED) %.3f ms" % (D, item, si.numel(), isz, 1000 * (t1 - t0), 1000 * (t3 - t2), cost["per_layer_ms"]), flush=True)

    def side_strata(family, W, D, item, passes, layers=None):
        si, di, isz = st_idx[(D, item)]
        es = isz // elem
        eng = engines if layers is None else [engines[i] for i in layers]
        kv2 = [(e._k_cpu.view(-1, es), e._v_cpu.view(-1, es)) for e in eng]
        kd, vd = k_s.view(-1, es), v_s.view(-1, es)
        unit = 2 * si.numel() * isz
        def f():
            for _ in range(passes):
                for e, (ks2, vs2) in zip(eng, kv2):
                    if family == "strata":
                        hc.strata_per_layer(e._k_cpu, k_s, e._v_cpu, v_s, si, di, isz, W, 32)
                    else:
                        hc.jit_per_layer(kd, vd, di, ks2, vs2, si, W)
                    SZ.LAUNCH_LOG.mark(unit)
        return f

    def verify_layers(family, W, D, item):
        """First touch of a strata / hicachejit arm: layers 0, L/2, L-1 one at a time, each against its own source."""
        bad = []
        for l in sorted({0, nl // 2, nl - 1}):
            k_s.fill_(float("nan")); v_s.fill_(float("nan"))
            bracket(None, side_strata(family, W, D, item, 1, layers=[l]))
            src_k = engines[l]._k_cpu[:, :D * bs].to("cuda"); src_v = engines[l]._v_cpu[:, :D * bs].to("cuda")
            if not (torch.equal(k_s[:, :D * bs], src_k) and torch.equal(v_s[:, :D * bs], src_v)):
                bad.append(l)
        k_s.zero_(); v_s.zero_()
        return bad

    def payload(partial):
        return dict(meta=run_meta, rows=rows, fails=fails, partial=partial, numa=numa, D_list=list(D_LIST), workers=list(WORKERS), reps=REPS,
                    sleep_ms=SLEEP_MS, target=TARGET,
                    hisparse=dict(blocks=list(HS_BLOCKS), threads=list(HS_THREADS), items=list(HS_ITEMS), plans=plan_log,
                                  upstream=(hs.UPSTREAM_COMMIT if hs is not None else None)),
                    strata=dict(blocks=list(ST_BLOCKS), items=list(ST_ITEMS), jit_blocks=list(JIT_BLOCKS), jit_items=list(JIT_ITEMS), plans=st_log,
                                upstream=(hc.UPSTREAM_COMMIT if hc is not None else None), strata_label=(hc.STRATA_LABEL if hc is not None else None),
                                jit_label=(hc.JIT_LABEL if hc is not None else None), verify_layers=st_verify),
                    arms_only=list(ARMS_ONLY), timeline=tl_records, timeline_arms=list(T_ARMS), timeline_late_ms=T_LATE_MS,
                    sizing_rule=dict(alone_margin=SZ.ALONE_MARGIN, conc_margin=SZ.CONC_MARGIN, max_trials=SIZE_TRIALS, overlap_min=SZ.OVERLAP_MIN,
                                     host_lag_max_ms=SZ.HOST_LAG_MAX_MS),
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                    device_total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9)

    def timeline_block(it, D, sized, restore, mk_step, lg_ref):
        """STAGE T (module docstring): untraced, traced (cudaProfilerApi capture range), untraced reps of each T arm."""
        nf = 0
        for tarm in T_ARMS:
            if tarm in ("alone", "alone_late"):
                mk, passes = None, 0
            elif tarm in sized:
                mk, passes = sized[tarm]
            else:
                tl_records.append(dict(batch=B, step=it, D=D, arm=tarm, error="not an arm of this run (WS_ARMS / blocks)"))
                continue
            late = T_LATE_MS if tarm == "alone_late" else 0.0
            for phase, n in (("untraced_pre", T_UNTRACED), ("traced", 1 + T_REPS), ("untraced_post", T_UNTRACED)):
                if phase == "traced":
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                for rep in range(n):
                    restore()
                    k_s.zero_(); v_s.zero_()
                    label = "lt|b%d|%s|step%d|%s|rep%d" % (B, tarm, it, phase, rep)
                    fn = timeline_main(mk_step(), label, late)
                    torch.cuda.nvtx.range_push(label)
                    lg, r = bracket(fn, mk(passes) if mk is not None else None)
                    torch.cuda.nvtx.range_pop()
                    ok = torch.equal(lg, lg_ref) and loaded_now() == 0
                    good = True if mk is None else (torch.equal(k_s[:, :D * bs], k_ref[:, :D * bs]) and torch.equal(v_s[:, :D * bs], v_ref[:, :D * bs]))
                    nf += int(not ok) + int(not good)
                    tl_records.append(dict(batch=B, step=it, D=D, arm=tarm, phase=phase, rep=rep, label=label, passes=passes, late_ms=late,
                                           warmup=(phase == "traced" and rep == 0), logits_equal=bool(ok), transfer_equal=bool(good), **r))
                if phase == "traced":
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
            print("[sweep] timeline step %d %s: %s" % (it, tarm, " ".join("%s=%.2f" % (x["phase"][:9], x["main_ms"]) for x in tl_records
                                                                       if x.get("arm") == tarm and x.get("step") == it and "main_ms" in x)), flush=True)
        return nf

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
                for T in (HS_THREADS if HS_BLOCKS else ()):
                    for item in HS_ITEMS:
                        if (D, item) in plans:
                            bracket(None, side_hisparse(1, T, D, item, 1))
                for fam, blocks, items in (("strata", ST_BLOCKS, ST_ITEMS), ("hicachejit", JIT_BLOCKS, JIT_ITEMS)):
                    for item in (items if blocks else ()):
                        if (D, item) in st_idx:
                            for W in blocks:
                                bad = verify_layers(fam, W, D, item)
                                st_verify[st_arm_name(fam, W, item) + "_D%d" % D] = dict(bad_layers=bad, ok=not bad)
                                fails += int(bool(bad))
                                print("[sweep] first touch %s D=%d: per-layer exact %s" % (st_arm_name(fam, W, item), D, not bad), flush=True)
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
            arms = [("workers%d" % W, (lambda W=W: (lambda p: side_gather(W, D, ids_hbd, p)))(), dict(family="workers", workers=W)) for W in WORKERS]
            arms += [(hs_arm_name(W, item, T), (lambda W=W, T=T, item=item: (lambda p: side_hisparse(W, T, D, item, p)))(),
                      dict(family="hisparse", blocks=W, threads=T, item=item, item_bytes=plans[(D, item)].item_size_bytes,
                           plan_items=plans[(D, item)].n_items))
                     for item in (HS_ITEMS if HS_BLOCKS else ()) if (D, item) in plans for T in HS_THREADS for W in HS_BLOCKS]
            arms += [(st_arm_name(fam, W, item), (lambda fam=fam, W=W, item=item: (lambda p: side_strata(fam, W, D, item, p)))(),
                      dict(family=fam, blocks=W, threads=1024, item=item, item_bytes=st_idx[(D, item)][2], plan_items=int(st_idx[(D, item)][0].numel()),
                           plan_cost=hc.PLAN_COST_LABEL, bench_label=hc.BENCH_LABEL,
                           label=(hc.STRATA_LABEL if fam == "strata" else hc.JIT_LABEL)))
                     for fam, blocks, items in (("strata", ST_BLOCKS, ST_ITEMS), ("hicachejit", JIT_BLOCKS, JIT_ITEMS))
                     for item in (items if blocks else ()) if (D, item) in st_idx for W in blocks]
            arms += [("dma2d", lambda p: side_dma(D, p), dict(family="dma2d"))]
            arms = [a for a in arms if arm_selected(a[0])]
            sized = {}
            for arm, mk, ameta in arms:
                # SIZING (side_sizing.py): >= 1.2 x the step alone from a one-pass side, then TRIAL concurrent brackets
                # raising passes until the side lasts >= 1.15 x the CONCURRENT step (trials are checked like reps)
                _, r1 = bracket(None, mk(1))
                trial_fails = []

                def trial(p, mk=mk):
                    restore()
                    k_s.zero_(); v_s.zero_()
                    lg, rt = bracket(step_fn(tok), mk(p))
                    ok = torch.equal(lg, lg_ref) and loaded_now() == 0
                    good = torch.equal(k_s[:, :D * bs], k_ref[:, :D * bs]) and torch.equal(v_s[:, :D * bs], v_ref[:, :D * bs])
                    trial_fails.append(int(not ok) + int(not good))
                    return rt["side_ms"] - rt["host_lag_ms"], rt["main_ms"]
                sizing = SZ.size_side(trial, t_alone, r1["side_ms"] - r1["host_lag_ms"], max_trials=SIZE_TRIALS)
                passes = sizing["passes"]
                fails += sum(trial_fails)
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
                rows.append(dict(batch=B, step=it, D=D, arm=arm, workers=ameta.get("workers"), family=ameta["family"],
                                 hisparse=(ameta if ameta["family"] == "hisparse" else None), sizing=sizing, passes=passes, bytes=by,
                                 decode_alone_ms=[x["main_ms"] for x in alone], decode_conc_ms=[x["main_ms"] for x in conc],
                                 side_alone_ms=[x["side_ms"] - x["host_lag_ms"] for x in s_alone], side_conc_ms=[x["side_ms"] - x["host_lag_ms"] for x in conc],
                                 overlap_frac=[x["overlap_frac"] for x in conc], host_lag_ms=[x["host_lag_ms"] for x in conc + s_alone],
                                 host_enqueue_ms=[x["host_enqueue_ms"] for x in conc],
                                 during_gbps=[x.get("during_gbps") for x in conc], inside_frac=[x.get("inside_frac") for x in conc],
                                 during_bytes=[x.get("during_bytes") for x in conc], last_end_ms=[x.get("last_end_ms") for x in conc],
                                 strata=(ameta if ameta["family"] in ("strata", "hicachejit") else None),
                                 alone_self=[self_report(x) for x in alone], conc_self=[self_report(x) for x in conc]))
                sized[arm] = (mk, passes)
                rr = rows[-1]
                med = lambda xs: float(np.median(xs))
                print("[sweep] step %d D=%d %-9s passes=%d  decode %.2f -> %.2f ms (%+.1f%%)  side alone %.1f GB/s, conc %.1f GB/s  overlap %.2f  host_lag max %.3f ms"
                      % (it, D, arm, passes, med(rr["decode_alone_ms"]), med(rr["decode_conc_ms"]), 100 * (med(rr["decode_conc_ms"]) / med(rr["decode_alone_ms"]) - 1),
                         by / (med(rr["side_alone_ms"]) * 1e6), by / (med(rr["side_conc_ms"]) * 1e6), min(rr["overlap_frac"]), max(rr["host_lag_ms"])), flush=True)
            if T_ARMS and D == D_LIST[0]:
                fails += timeline_block(it, D, sized, restore, lambda: step_fn(tok), lg_ref)
        restore()
        model.decode_inference(tok, cu, position_ids, cache)            # the advance: a resident step from the restored state
        torch.cuda.synchronize()
        position_ids = position_ids + 1
        print("[sweep] step %d done in %.0fs (fails %d)" % (it, time.time() - t_wall, fails), flush=True)
        if flush is not None:
            flush(payload(True))

    return payload(False)


def arm_footprint(arm: str) -> str:
    """What W means for the SMs (review NB2): hisparse at 1024 threads = W SMs (48 regs x 1024 thr: one copy CTA per
    SM); a _t256 arm = W blocks, not W SMs (up to 4 per SM); workersW = W CTAs of 128 threads, not W SMs."""
    import re
    if arm == "dma2d":
        return "copy engine"
    m = re.match(r"(workers|hisparse|strata|hicachejit)(\d+)(.*)$", arm)
    if not m:
        return "-"
    W = int(m.group(2))
    if m.group(1) == "workers":
        return "%d CTAs x 128 thr (not W SMs)" % W
    if m.group(1) == "strata":
        return "%d SMs (48 regs x 1024 thr)" % W
    if m.group(1) == "hicachejit":
        return "%d blocks x 1024 thr (30-32 regs)" % W
    return ("%d blocks (not W SMs)" % W) if "_t" in m.group(3) else ("%d SMs" % W)


def adapter_table(rows) -> list:
    """NATIVE (i512 token rows, SGLang's item; exact here because the ids are coupled) vs ADAPTED (i256 per-head rows, the
    NOSI adapter) at identical bytes and destinations, per (batch, D, family, W): medians over reps and steps."""
    import re
    acc = {}
    for r in rows:
        m = re.match(r"(strata|hicachejit|hisparse)(\d+)(?:_t1024)?_(i256|i512)$", r["arm"])
        if not m:
            continue
        a = acc.setdefault((r["batch"], r["D"], m.group(1), int(m.group(2)), m.group(3)), dict(ra=[], rc=[], da=[], dc=[]))
        a["ra"] += [r["bytes"] / (x * 1e6) for x in r["side_alone_ms"]]
        a["rc"] += [r["bytes"] / (x * 1e6) for x in r["side_conc_ms"]]
        a["da"] += r["decode_alone_ms"]; a["dc"] += r["decode_conc_ms"]
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    L = []
    for (B, D, fam, W, item), a in sorted(acc.items()):
        if item != "i256" or (B, D, fam, W, "i512") not in acc:
            continue
        b = acc[(B, D, fam, W, "i512")]
        sa, sb = med(a["dc"]) / med(a["da"]) - 1, med(b["dc"]) / med(b["da"]) - 1
        L.append("| %d | %d | %s | %d | %.1f | %.1f | %.2f | %.1f | %.1f | %+.1f%% | %+.1f%% | %+.1f |" % (
            B, D, fam, W, med(a["ra"]), med(b["ra"]), med(a["ra"]) / med(b["ra"]), med(a["rc"]), med(b["rc"]), 100 * sa, 100 * sb, 100 * (sa - sb)))
    if not L:
        return []
    return ["", "## Native (i512 token rows) vs adapted (i256 per-head rows) layout, identical bytes and destinations", "",
            "| B | D | family | W | adapted alone GB/s | native alone GB/s | adapted / native | adapted concurrent GB/s | native concurrent GB/s "
            "| adapted slowdown | native slowdown | delta (pp) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"] + L


def self_report_table(pays) -> list:
    """Per (batch, arm): the self-reporting timer over every concurrent rep (and the alone reps of the step)."""
    L = ["", "## Self-reporting gated timer (measurement only; the launch-correlated timeline decides validity)", "",
         "| B | arm | reps | gate done when submission ended | host submit ms median / max | submit end after gate ms median / max |",
         "|---|---|---|---|---|---|"]
    n = 0
    for p in pays:
        acc = {}
        for r in p["rows"]:
            for key, recs in (("alone", r.get("alone_self") or []), (r["arm"], r.get("conc_self") or [])):
                acc.setdefault(key, []).extend(recs)
        for arm, recs in sorted(acc.items()):
            recs = [x for x in recs if x.get("submit_end_ms") is not None]
            if not recs:
                continue
            n += 1
            hs_, se = [x["host_submit_ms"] for x in recs], [x["submit_end_ms"] for x in recs]
            L.append("| %d | %s | %d | %d/%d | %.2f / %.2f | %.2f / %.2f |" % (p["meta"]["batch"], arm, len(recs), sum(1 for x in recs if x["gate_done_at_submit_end"]),
                                                                            len(recs), float(np.median(hs_)), max(hs_), float(np.median(se)), max(se)))
    return L if n else []


def table(out_dir):
    import glob
    pays = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, "sweep_*.json")))]
    L = ["# Corrected worker sweep: SM gather (workersW: W CTAs x 128 threads), HiSparse copy_cache_planned (hisparseW_item: W blocks x "
         "1024 threads unless _tT) vs matched copy engine, beside NOSI's resident decode step", "",
         "Medians (and p95) over gated steps x reps. Transfer GB/s = median over reps of that rep's bytes / side interval (the concurrent value "
         "includes the tail the side runs alone after the step). During-step GB/s = bytes of the per-layer launches that ENDED inside the step / "
         "step ms (a lower bound). slowdown = median concurrent / median alone - 1; normalized decode throughput = alone / concurrent. "
         "tok/s = B / decode ms. Valid = min overlap_frac >= 0.95 and max host_lag <= 0.1 ms and all correctness checks pass. covered = every "
         "sizing reached a side >= 1.15 x the concurrent step.", ""]
    L.append("| B | D | arm | footprint | passes | transfer alone GB/s | transfer concurrent GB/s | % of dma2d concurrent | during-step GB/s (lower bound) "
             "| side bytes inside step | decode alone median / p95 ms | decode concurrent median / p95 ms | slowdown | normalized decode throughput "
             "| tok/s alone -> concurrent | min overlap | max host lag ms | covered | meets target (whole interval) | meets target (during-step) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    verdict_rows = []
    fnum = lambda x, f="%.1f": "-" if x is None or x != x else f % x
    for p in pays:
        B = p["meta"]["batch"]
        by_key = {}
        for r in p["rows"]:
            k = (r["D"], r["arm"])
            a = by_key.setdefault(k, dict(D=r["D"], arm=r["arm"], passes=[], da=[], dc=[], ra=[], rc=[], ov=[], hl=[], dg=[], inf=[], cov=[]))
            a["da"] += r["decode_alone_ms"]; a["dc"] += r["decode_conc_ms"]; a["ov"] += r["overlap_frac"]; a["hl"] += r["host_lag_ms"]
            # GB/s per rep from THAT row's bytes: the sizing may choose different passes at different steps
            a["passes"].append(r["passes"]); a["ra"] += [r["bytes"] / (x * 1e6) for x in r["side_alone_ms"]]; a["rc"] += [r["bytes"] / (x * 1e6) for x in r["side_conc_ms"]]
            a["dg"] += [x for x in (r.get("during_gbps") or []) if x is not None]; a["inf"] += [x for x in (r.get("inside_frac") or []) if x is not None]
            if r.get("sizing") is not None:
                a["cov"].append(bool(r["sizing"].get("covered")))
        dma = {D: v for (D, arm), v in by_key.items() if arm == "dma2d"}
        for (D, arm), a in sorted(by_key.items(), key=lambda kv: (kv[0][0], arm_order(kv[0][1]))):
            med, p95 = (lambda xs: float(np.median(xs)) if xs else float("nan")), (lambda xs: float(np.percentile(xs, 95)))
            bw_a, bw_c, bw_d = med(a["ra"]), med(a["rc"]), med(a["dg"])
            d = dma.get(D)
            dma_c = med(d["rc"]) if d else float("nan")
            dma_d = med(d["dg"]) if d else float("nan")
            slow = med(a["dc"]) / med(a["da"]) - 1
            valid = min(a["ov"]) >= 0.95 and max(a["hl"]) <= 0.1
            meets = valid and arm != "dma2d" and bw_c >= TARGET["bw_frac_of_dma"] * dma_c and slow <= TARGET["max_slowdown"]
            meets_d = valid and arm != "dma2d" and bw_d == bw_d and dma_d == dma_d and bw_d >= TARGET["bw_frac_of_dma"] * dma_d and slow <= TARGET["max_slowdown"]
            covered = (all(a["cov"]) if a["cov"] else None)
            verdict_rows.append(dict(B=B, D=D, arm=arm, bw_conc=bw_c, bw_during=bw_d, frac_dma=bw_c / dma_c if dma_c == dma_c else None,
                                     frac_dma_during=(bw_d / dma_d if dma_d == dma_d and dma_d > 0 else None), slowdown=slow,
                                     norm_tput=med(a["da"]) / med(a["dc"]), valid=valid, covered=covered, meets=meets, meets_during=meets_d))
            ps = sorted(set(a["passes"]))
            L.append("| %d | %d | %s | %s | %s | %.1f | %.1f | %.0f%% | %s | %s | %.2f / %.2f | %.2f / %.2f | %+.1f%% | %.1f%% | %.0f -> %.0f | %.2f | %.3f | %s | %s | %s |" % (
                B, D, arm, arm_footprint(arm), ("%d" % ps[0]) if len(ps) == 1 else "%d-%d" % (ps[0], ps[-1]), bw_a, bw_c, 100 * bw_c / dma_c, fnum(bw_d),
                fnum(100 * med(a["inf"]) if a["inf"] else None, "%.0f%%"), med(a["da"]), p95(a["da"]), med(a["dc"]), p95(a["dc"]), 100 * slow,
                100 * med(a["da"]) / med(a["dc"]), 1000 * B / med(a["da"]), 1000 * B / med(a["dc"]), min(a["ov"]), max(a["hl"]),
                "-" if covered is None else ("yes" if covered else "NO"),
                ("YES" if meets else ("n/a" if arm == "dma2d" else "no")) + ("" if valid else " (INVALID overlap/timer)"),
                "YES" if meets_d else ("n/a" if arm == "dma2d" else "no")))
        L.append("")
        numa = p.get("numa") or {}
        lay = numa.get("layers") or {}
        L.append("B=%d: correctness fails %d%s; peak allocated %.2f GB, reserved %.2f GB of %.1f GB; host cache NUMA node per layer (0 / 1 / s = split): "
                 "K %s, V %s; %s" % (B, p["fails"], " (PARTIAL: the run did not finish)" if p.get("partial") else "", p["peak_allocated_gb"], p["peak_reserved_gb"],
                                    p["device_total_gb"], lay.get("k_nodes", "?"), lay.get("v_nodes", "?"),
                                    json.dumps({k: v for k, v in numa.items() if k != "layers"})))
        for t in (p.get("hisparse") or {}).get("plans", []):
            if "refused" in t:
                L.append("B=%d plan D=%d %s: REFUSED (%s)" % (B, t["D"], t["item"], t["refused"]))
                continue
            L.append("B=%d plan D=%d %s: %d items of %d B, stride %d; build host %.2f ms, upload %.2f ms, device %.2f ms, device==host %s"
                     % (B, t["D"], t["item"], t["n_items"], t["item_size_bytes"], t["plan_stride"], t["build_host_ms"], t["upload_ms"], t["build_device_ms"], t["device_build_equal"]))
        L.append("")
    L += adapter_table([r for p in pays for r in p["rows"]])
    L += self_report_table(pays)
    labels = {}
    for p in pays:
        st = p.get("strata") or {}
        for pl in st.get("plans") or []:
            if "refused" in pl:
                L.append("B=%d strata/jit index D=%d %s: REFUSED (%s)" % (p["meta"]["batch"], pl["D"], pl["item"], pl["refused"]))
            else:
                c = pl.get("nosi_plan_cost") or {}
                L.append("B=%d strata/jit index D=%d %s: %d items of %d B; host build %.2f ms, upload %.2f ms; INTEGRATED per-layer plan cost "
                         "(EXCLUDED from every row): count sync %.3f ms + index build %.3f ms = %.3f ms per layer per step"
                         % (p["meta"]["batch"], pl["D"], pl["item"], pl["n_items"], pl["item_size_bytes"], pl["build_host_ms"], pl["upload_ms"],
                            c.get("count_sync_ms", float("nan")), c.get("index_build_ms", float("nan")), c.get("per_layer_ms", float("nan"))))
        if st.get("verify_layers"):
            L.append("B=%d first-touch per-layer exactness: %s" % (p["meta"]["batch"], json.dumps(st["verify_layers"])))
        for k in ("strata_label", "jit_label"):
            if st.get(k):
                labels[k] = st[k]
    if labels:
        L += ["", "strata* arms = %s. hicachejit* arms = %s. Every strata / hicachejit / hisparse row is a prebuilt-plan copy "
              "microbenchmark, NOT an integrated LRU getter baseline (plan EXCLUDED)." % (labels.get("strata_label"), labels.get("jit_label")), ""]
    met = [v for v in verdict_rows if v["meets"]]
    met_d = [v for v in verdict_rows if v["meets_during"]]
    L.append("## Registered target (ledger 'WORKER SWEEP REGISTERED'): >= 90% of the matched copy-engine concurrent bandwidth with <= 5% decode slowdown")
    L.append("MET (whole-interval GB/s) by: " + (", ".join("B=%d D=%d %s" % (v["B"], v["D"], v["arm"]) for v in met) if met else "NONE -- see the tradeoff columns above"))
    L.append("MET (during-step GB/s, lower bound) by: " + (", ".join("B=%d D=%d %s" % (v["B"], v["D"], v["arm"]) for v in met_d) if met_d else "NONE"))
    text = "\n".join(L) + "\n"
    open(os.path.join(out_dir, "sweep_table.md"), "w").write(text)
    json.dump(dict(verdicts=verdict_rows, fails=sum(p["fails"] for p in pays), partial=[p.get("partial", False) for p in pays]),
              open(os.path.join(out_dir, "sweep_table.json"), "w"), indent=1)
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
    fn = os.path.join(OUT, "%s.json" % TAG)

    def flush(p):
        """Write the JSON atomically (review NB10: after every gated step, so a timeout keeps the finished steps)."""
        p["docs"] = docs; p["distinct_books"] = distinct
        tmp = fn + ".tmp"
        json.dump(p, open(tmp, "w"), indent=1)
        os.replace(tmp, fn)
    payload = run(model, ids, flush=flush)
    flush(payload)
    print("[sweep] saved %s (fails %d, peak reserved %.2f GB)" % (fn, payload["fails"], payload["peak_reserved_gb"]), flush=True)
    sys.exit(min(payload["fails"], 200))
