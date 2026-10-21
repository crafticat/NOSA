"""R2: the GPU cell of the FUSED twin step and the LEAN body (docs/R2_FUSED.md). One process, one A100.

PARTS (--parts, default all):
  GT  nosi/nosi/r2_twin_fused.py == PositionalTwin.step_unchecked / advance_unchecked with mean_fn = fixed_order_mean,
      BITWISE on the GPU, bfloat16 tables at the R2 sizes (B 64, H 8, D 128, Lk 1013, N 16259; ragged prompts
      16000 .. 16141): 40 single steps with ragged activity, then 10 advance ticks of T 4 with ragged n (0 .. 4), every
      table, counter, the overflow flag and ctr compared after every call (MUST GT-EQ; the key / cis events and block
      crossings are counted: MUST GT-COVER). The three planted faults (r2_twin_fused.PLANTS) are each caught on a copy
      of the state (MUST GT-NEG). REPORT: device ms per call (CUDA events, 20 calls, all active) of the torch ops and of
      the kernel.
  GB  nosi/nosi/r2_body_fused.py FusedBody.decode_fixed == DeviceWritebackBody.decode_fixed BITWISE with the real
      kernels (diff_offload, flash_h2d_from_mask(_bias), the Triton write-back) on a synthetic engine (B 64, H 8, D 128,
      prompts 6016 .. 6079, pinned host rows): 96 steps with ragged activity; every engine table and err after every step
      and the host rows at the end (MUST GB-EQ; write-backs and inactive requests counted: MUST GB-COVER); the three
      planted faults each caught in 8 steps on fresh engines (MUST GB-NEG). REPORT: device ms per call.
  MF  the measurement of the registered prediction (commit message): SH, the shipped decode at B 64 on the r2base
      workload (r2_p3_profile.run_sh: 16 timed + 16 profiled steps), and the captured R2 step in TWO graphs on ONE state:
      G = decode_step_fixed as built (twin.step_fixed, DeviceWritebackBody), F = the same step with the fused twin
      attached as step_fixed (a DECLARED change: per-bank refusal through the overflow flag and the fixed-order mean)
      and FusedBody. Order: eager, eager, capture G, eager F (compiles the kernels), capture F, then 16-step blocks
      G F G F timed (CUDA events), then G and F profiled (torch.profiler; r2_p3_profile's parser and corrected classes).
      MUST: SH-RUN, SH-STEADY, R2-RUN (body err 0, twin err 0, no overflow), PROF (GPU ops seen in SH, G and F).
EXIT: the number of failed MUST gates; 20 refused; 30 an exception outside a part.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_base_cell as BC          # noqa: E402
import r2_p3_profile as P          # noqa: E402
import r2_ragged_gate as RG        # noqa: E402

PARTS = ("GT", "GB", "MF")
N_WIN = 16
BLOCKS = (("G", "time"), ("F", "time"), ("G", "time"), ("F", "time"), ("G", "prof"), ("F", "prof"))
SETUP = ("eager_g", "eager_g", "cap_g", "eager_f", "cap_f")
R2_STEPS = len(SETUP) + N_WIN * len(BLOCKS)                     # 101
SH_STEPS = R2_STEPS + 1                                          # the R2 arm reads SH's tokens 0 .. R2_STEPS - 1
PRED = dict(F_ms=(44.0, 46.0), F_minus_SH_ms=(-1.5, 0.5), G_minus_F_ms=(7.0, 9.5), F_ops=(1150, 1260),
            F_period_minus_SH_ms=0.5)


def log(msg):
    print("[r2_fused %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args(argv)
    a.parts = [p for p in a.parts.split(",") if p]
    if not a.parts or any(p not in PARTS for p in a.parts):
        ap.error("--parts: a list of %s" % (PARTS,))
    if "MF" in a.parts and (not a.workload or not 1 <= a.batch <= 219):
        ap.error("MF needs --workload and --batch in 1 .. 219")
    return a


def schedule():
    """[(t, kind)] of the R2 arm: the SETUP kinds, then (arm, mode, block index) per step of each block."""
    out = [(t, k) for t, k in enumerate(SETUP)]
    t = len(SETUP)
    for i, (arm, mode) in enumerate(BLOCKS):
        for _ in range(N_WIN):
            out.append((t, (arm, mode, i)))
            t += 1
    return out


def verdicts(agg, timing):
    """The registered predictions against the profiled aggregates and the timed periods: [(name, verdict, detail)]."""
    sh, g, f = agg.get("SH"), agg.get("G"), agg.get("F")
    if not sh or not g or not f or min(x["total"]["kernels"] for x in (sh, g, f)) <= 0:
        return [(n, "UNDETERMINED", "an arm's profile is missing or empty") for n in sorted(PRED)]

    def v(ok):
        return "CONFIRMED" if ok else "REFUTED"
    fm, sm, gm = f["total"]["ms"], sh["total"]["ms"], g["total"]["ms"]
    out = []
    lo, hi = PRED["F_ms"]
    out.append(("F_ms", v(lo <= fm <= hi), "F device ms per step %.3f (predicted %.1f .. %.1f)" % (fm, lo, hi)))
    lo, hi = PRED["F_minus_SH_ms"]
    out.append(("F_minus_SH_ms", v(lo <= fm - sm <= hi), "F - SH device ms per step %.3f (predicted %.1f .. %.1f)"
                % (fm - sm, lo, hi)))
    lo, hi = PRED["G_minus_F_ms"]
    out.append(("G_minus_F_ms", v(lo <= gm - fm <= hi), "G - F device ms per step %.3f (predicted %.1f .. %.1f)"
                % (gm - fm, lo, hi)))
    lo, hi = PRED["F_ops"]
    fo = f["total"]["kernels"]
    out.append(("F_ops", v(lo <= fo <= hi), "F GPU ops per step %.1f (predicted %d .. %d; G %.1f, SH %.1f)"
                % (fo, lo, hi, g["total"]["kernels"], sh["total"]["kernels"])))
    fp, sp = timing.get("F", {}).get("median_ms"), timing.get("SH", {}).get("median_ms")
    if fp is None or sp is None:
        out.append(("F_period_minus_SH_ms", "UNDETERMINED", "no period"))
    else:
        out.append(("F_period_minus_SH_ms", v(fp - sp <= PRED["F_period_minus_SH_ms"]),
                    "F - SH median period %.3f ms (predicted <= %.1f)" % (fp - sp, PRED["F_period_minus_SH_ms"])))
    return out


# ------------------------------------------------------------------ GT
def _bits(t):
    import torch
    t = t.contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


TWIN_FIELDS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis", "overflow")


def _twin_diff(a, b):
    import torch
    return [k for k in TWIN_FIELDS if not torch.equal(_bits(getattr(a, k)), _bits(getattr(b, k)))]


def _twin_copy(t):
    from nosi.r2_positional_twin import PositionalTwin
    c = PositionalTwin(t.B, t.H, t.D, t.lk_max, t.n_max, dtype=t.ring.dtype, device=t.pos.device)
    for k in TWIN_FIELDS:
        getattr(c, k).copy_(getattr(t, k))
    c.mean_fn = t.mean_fn
    return c


def part_gt(gates, res):
    import torch
    from nosi.r2_bi_decode import fixed_order_mean
    from nosi.r2_positional_twin import PositionalTwin
    from nosi import r2_twin_fused as FU
    dev = torch.device("cuda")
    B, H, D, LK, N = 64, 8, 128, 1013, 16259
    gc = torch.Generator(device=dev).manual_seed(11)
    gh = torch.Generator().manual_seed(12)
    S = [16000 + 13 * (b % 7) + b for b in range(B)]
    ref = PositionalTwin(B, H, D, LK, N, dtype=torch.bfloat16, device=dev)
    fu = PositionalTwin(B, H, D, LK, N, dtype=torch.bfloat16, device=dev)
    for b in range(B):
        k = torch.randn(S[b], H, D, generator=gc, device=dev).to(torch.bfloat16)
        c = torch.randn(S[b], H, generator=gc, device=dev).to(torch.bfloat16)
        ref.prefill(b, k, c)
        fu.prefill(b, k, c)
    ref.mean_fn = fu.mean_fn = fixed_order_mean
    first, ev = None, dict(key=0, cis=0, cross=0, inactive=0, calls=0)

    def note(where, extra=()):
        nonlocal first
        d = _twin_diff(ref, fu) + list(extra)
        if d and first is None:
            first = (where, d)
    for s in range(40):
        act = (torch.rand(B, generator=gh) < 0.7).to(dev)
        if s % 7 == 0:
            act.fill_(True)
        keys = torch.randn(B, H, D, generator=gc, device=dev).to(torch.bfloat16)
        cis = torch.randn(B, H, generator=gc, device=dev).to(torch.bfloat16)
        p0, k0, c0 = ref.pos.clone(), ref.n_comp.clone(), ref.n_ccis.clone()
        ref.step_unchecked(keys, cis, act)
        FU.step_fused(fu, keys, cis, act)
        ev["key"] += int((ref.n_comp != k0).sum())
        ev["cis"] += int((ref.n_ccis != c0).sum())
        ev["cross"] += int(((p0 // 64) != (ref.pos // 64)).sum())
        ev["inactive"] += int((~act).sum())
        ev["calls"] += 1
        note(("step", s))
    for s in range(10):
        n = torch.randint(0, 5, (B,), generator=gh).to(dev)
        keys = torch.randn(B, 4, H, D, generator=gc, device=dev).to(torch.bfloat16)
        cis = torch.randn(B, 4, H, generator=gc, device=dev).to(torch.bfloat16)
        p0, k0, c0 = ref.pos.clone(), ref.n_comp.clone(), ref.n_ccis.clone()
        c1 = ref.advance_unchecked(keys, cis, n)
        c2 = FU.advance_fused(fu, keys, cis, n)
        ev["key"] += int((ref.n_comp != k0).sum())
        ev["cis"] += int((ref.n_ccis != c0).sum())
        ev["cross"] += int(((p0 // 64) != (ref.pos // 64)).sum())
        ev["calls"] += 1
        note(("advance", s), [] if torch.equal(c1, c2) else ["ctr"])
    gates.append(("GT-EQ", True, first is None and not bool(ref.overflow),
                  "fused twin == step_unchecked / advance_unchecked bitwise over %d calls: first difference %s; "
                  "overflow %s" % (ev["calls"], first, bool(ref.overflow))))
    gates.append(("GT-COVER", True, ev["key"] >= 64 and ev["cis"] >= 64 and ev["cross"] >= 16 and ev["inactive"] > 0,
                  "events %s (needs key >= 64, cis >= 64, cross >= 16, inactive > 0)" % ev))
    caught = {}
    for plant in (1, 2, 3):
        a, b2 = _twin_copy(ref), _twin_copy(ref)
        found = None
        for s in range(24):                     # plant 1 in bfloat16 differs only where the float32 sums round apart
            act =(torch.rand(B, generator=gh) < 0.7).to(dev)
            keys = torch.randn(B, H, D, generator=gc, device=dev).to(torch.bfloat16)
            cis = torch.randn(B, H, generator=gc, device=dev).to(torch.bfloat16)
            a.step_unchecked(keys, cis, act)
            FU.step_fused(b2, keys, cis, act, plant=plant)
            d = _twin_diff(a, b2)
            if d:
                found = (s, d)
                break
        caught[plant] = found
        del a, b2
    gates.append(("GT-NEG", True, all(v is not None for v in caught.values()),
                  "planted faults caught (first step, fields): %s" % caught))
    # REPORT: device ms per call, all active, on copies
    t1, t2 = _twin_copy(ref), _twin_copy(ref)
    act = torch.ones(B, dtype=torch.bool, device=dev)
    keys = torch.randn(B, H, D, generator=gc, device=dev).to(torch.bfloat16)
    cis = torch.randn(B, H, generator=gc, device=dev).to(torch.bfloat16)
    ms = {}
    for name, fn in (("torch_step_unchecked", lambda: t1.step_unchecked(keys, cis, act)),
                     ("fused", lambda: FU.step_fused(t2, keys, cis, act))):
        fn()
        per = []
        for _ in range(20):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            per.append(e0.elapsed_time(e1))
        ms[name] = round(statistics.median(per), 4)
    res["GT"] = dict(events=ev, first=first, caught={k: v for k, v in caught.items()}, ms_per_call=ms)
    del ref, fu, t1, t2
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ GB
BODY_TABLES = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_new_block_map_buf", "_load_mask", "_cache_lens")


def _selection(p_host, H, gh):
    """(H, B, 64) int64: per (h, b) the tail id p // 64 and 63 distinct ids from [0, p // 64), shuffled."""
    import torch
    B = len(p_host)
    out = torch.empty((H, B, 64), dtype=torch.int64)
    for b in range(B):
        T = int(p_host[b]) // 64
        for h in range(H):
            row = torch.cat([torch.randperm(T, generator=gh)[:63], torch.tensor([T])])
            out[h, b] = row[torch.randperm(64, generator=gh)]
    return out


def _gb_engines(B, H, D, S, n_host, gc):
    import torch
    from nosi import r2_decode_body as RB
    dev = torch.device("cuda")
    es = []
    for _ in range(2):
        e = SimpleNamespace(head_num=H, head_dim=D, block_size=64, topk=64, dtype=torch.bfloat16, device=dev,
                            pool_blocks=0)
        RB.DecodeBody(False).allocate(e, B, n_host, pin=True)
        es.append(e)
    a, b = es
    for name in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
        getattr(a, name).copy_(torch.randn(getattr(a, name).shape, generator=gc, device=dev).to(torch.bfloat16))
    for name in ("_new_block_map_buf", "_load_mask"):
        getattr(a, name).copy_(torch.randint(-5, 5, getattr(a, name).shape, generator=gc, device=dev))
    for name in BODY_TABLES:
        getattr(b, name).copy_(getattr(a, name))
    kvb = torch.randn((B, n_host, H), generator=gc, device=dev).to(torch.bfloat16)
    for bb in range(B):
        k = torch.randn((S[bb], H, D), generator=gc, device=dev).to(torch.bfloat16)
        v = torch.randn((S[bb], H, D), generator=gc, device=dev).to(torch.bfloat16)
        for e in es:
            RB.DecodeBody(False).prefill_request(e, bb, k, v, kvb[bb, :S[bb]])
    torch.cuda.synchronize()
    return a, b, kvb


def _gb_run(steps, plant, gc, gh, timing=False):
    import torch
    from nosi import r2_capture as RC
    from nosi import r2_decode_body as RB
    from nosi.r2_body_fused import FusedBody
    (diff, gk, gvb), wb = RC.default_kernels()
    dev = torch.device("cuda")
    B, H, D = 64, 8, 128
    S = [6016 + (7 * b) % 64 for b in range(B)]
    n_host = max(S) + steps + 64
    e1, e2, kvb = _gb_engines(B, H, D, S, n_host, gc)
    ref, fu = RB.DeviceWritebackBody(False, wb), FusedBody(False, wb, plant=plant)
    err1 = torch.zeros((), dtype=torch.int32, device=dev)
    err2 = torch.zeros((), dtype=torch.int32, device=dev)
    p_host = torch.tensor(S, dtype=torch.int64)
    first, st, ms = None, dict(full=0, inactive=0), dict(ref=[], fused=[])
    for i in range(steps):
        act_h = torch.ones(B, dtype=torch.bool) if i == 0 else (torch.rand(B, generator=gh) < 0.6)
        st["full"] += int((act_h & ((p_host + 1) % 64 == 0)).sum())
        st["inactive"] += int((~act_h).sum())
        topk = _selection(p_host, H, gh).to(dev)
        p, act = p_host.to(dev), act_h.to(dev)
        k = torch.randn((B, 1, H, D), generator=gc, device=dev).to(torch.bfloat16)
        v = torch.randn((B, 1, H, D), generator=gc, device=dev).to(torch.bfloat16)
        for name, body, e, err in (("ref", ref, e1, err1), ("fused", fu, e2, err2)):
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            body.decode_fixed(e, k, v, kvb, topk, p, act, err, diff, gk, gvb)
            ev1.record()
            torch.cuda.synchronize()
            ms[name].append(ev0.elapsed_time(ev1))
        d = [n for n in BODY_TABLES if not torch.equal(_bits(getattr(e1, n)), _bits(getattr(e2, n)))]
        d += [] if torch.equal(err1, err2) else ["err"]
        if d and first is None:
            first = (i, d)
            if plant:
                break
        p_host += act_h.to(torch.int64)
    torch.cuda.synchronize()
    if first is None:
        for n in ("_k_cpu", "_v_cpu"):
            if not torch.equal(_bits(getattr(e1, n)), _bits(getattr(e2, n))):
                first = ("end", [n])
                break
    out = dict(first=first, stats=st, err=(int(err1), int(err2)),
               ms_per_call={k: round(statistics.median(v), 4) for k, v in ms.items() if v})
    del e1, e2, kvb
    torch.cuda.empty_cache()
    return out


def part_gb(gates, res):
    import torch
    dev = torch.device("cuda")
    gc = torch.Generator(device=dev).manual_seed(21)
    gh = torch.Generator().manual_seed(22)
    r0 = _gb_run(96, 0, gc, gh)
    gates.append(("GB-EQ", True, r0["first"] is None and r0["err"] == (0, 0),
                  "FusedBody == decode_fixed bitwise over 96 steps (+ host rows): first difference %s; err %s"
                  % (r0["first"], r0["err"])))
    gates.append(("GB-COVER", True, r0["stats"]["full"] >= 32 and r0["stats"]["inactive"] > 0,
                  "write-backs and inactive requests %s (needs full >= 32)" % r0["stats"]))
    caught = {plant: _gb_run(8, plant, gc, gh)["first"] for plant in (1, 2, 3)}
    gates.append(("GB-NEG", True, all(v is not None for v in caught.values()),
                  "planted faults caught (first step, tables): %s" % caught))
    res["GB"] = dict(run=r0, caught=caught)


# ------------------------------------------------------------------ MF
def run_r2(model, ids, inputs, out_dir, clock):
    import torch
    from nosi import r2_capture as RC
    from nosi import r2_twin_fused as FU
    from nosi.r2_body_fused import FusedBody
    from nosi.r2_decode_path import build_ragged
    Bn = ids.shape[0]
    prompts = [ids[b].cpu() for b in range(Bn)]
    clock.sync()
    t0 = time.time()
    cache, st, _ = build_ragged(model, prompts, R2_STEPS, f4_fix=False)
    clock.sync()
    prefill_s = time.time() - t0
    kernels, wb = RC.default_kernels()
    cb = RC.CaptureBuffers(model, st, False, kernels, wb)
    st.active_host.fill_(True)
    st.active.fill_(True)
    cb.pos_ids.copy_(st.pos_host.view(-1, 1))
    graphs, starts, prof, traces, cap_ms = {}, {}, {}, {}, {}
    sched = schedule()
    for t, kind in sched:
        if isinstance(kind, tuple) and kind[1] == "prof" and kind not in prof:
            clock.sync()
            prof[kind] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA])
            prof[kind].start()
        if isinstance(kind, tuple) and kind[1] == "time":
            e = clock.event()
            clock.record(e)
            starts.setdefault(kind, []).append(e)
        cb.ids.copy_(inputs[:, t:t + 1])
        if kind == "eager_g":
            RC.decode_step_fixed(model, st, cb, cache)
        elif kind in ("cap_g", "cap_f"):
            arm = "G" if kind == "cap_g" else "F"
            clock.sync()
            c0 = time.perf_counter()
            graphs[arm] = RC.capture_step(model, st, cb, cache)
            clock.sync()
            cap_ms[arm] = round((time.perf_counter() - c0) * 1e3, 1)
            graphs[arm].replay()
        elif kind == "eager_f":
            for tw in st.twins:
                FU.attach(tw, methods=("step_fixed",), require_fixed_order=False)
            cb.body = FusedBody(False, wb)
            RC.decode_step_fixed(model, st, cb, cache)
        else:
            graphs[kind[0]].replay()
        cb.pos_ids.add_(1)
        st.pos_host += 1
        st.steps += 1
        nxt = sched[t + 1][1] if t + 1 < len(sched) else None
        if isinstance(kind, tuple) and kind[1] == "time" and nxt != kind:
            e = clock.event()
            clock.record(e)
            starts[kind].append(e)
        if isinstance(kind, tuple) and kind[1] == "prof" and nxt != kind:
            clock.sync()
            prof[kind].stop()
            path = os.path.join(out_dir, "trace_%s.json" % kind[0])
            prof[kind].export_chrome_trace(path)
            traces[kind[0]] = path
            del prof[kind]
    clock.sync()
    periods = {"G": [], "F": []}
    for kind, evs in starts.items():
        periods[kind[0]] += [clock.elapsed(evs[i], evs[i + 1]) for i in range(len(evs) - 1)]
    body_err, twin_err = RC.errors(st, cb)
    overflow = [int(t.overflow) for t in st.twins]
    res = dict(periods=periods, traces=traces, capture_ms=cap_ms, prefill_s=round(prefill_s, 1), body_err=body_err,
               twin_err=twin_err, overflow=overflow, widths=st.meta.get("widths"))
    del graphs, cb, st, cache
    return res


def part_mf(a, gates, res, out_dir):
    import torch
    import nosi.cache_engine as CEN
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    if NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != 0:
        raise RuntimeError("MF needs NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS=0")
    docs, sha = RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    P.SH_STEPS = SH_STEPS                                      # run_sh reads it: inputs for every R2 step
    arms, agg, timing = {}, {}, {}
    with torch.inference_mode():
        BC.free()
        log("MF: SH (B %d, %d steps; timed %s, profiled %s)" % (a.batch, SH_STEPS, P.SH_TIMED, P.SH_PROF))
        inputs = None
        try:
            sh = P.run_sh(model, ids, out_dir, clock)
            inputs = sh.pop("inputs")
            arms["SH"] = sh
            bad = [x for x in sh["rewarms"] if P.SH_TIMED[0] <= x[0] < P.SH_PROF[1]]
            gates.append(("SH-RUN", True, True, "%d steps, re-warms %s" % (SH_STEPS, sh["rewarms"])))
            gates.append(("SH-STEADY", True, not bad, "re-warms inside the windows: %s" % (bad,)))
        except Exception as e:                                             # noqa: BLE001
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        BC.free(host=True)
        if inputs is None:
            gates.append(("R2-RUN", True, False, "NOT_RUN: SH failed"))
        else:
            log("MF: R2 (setup %s; blocks %s of %d steps)" % (SETUP, BLOCKS, N_WIN))
            try:
                r2 = run_r2(model, ids, inputs, out_dir, clock)
                arms["R2"] = r2
                ok = not r2["body_err"] and not any(r2["twin_err"]) and not any(r2["overflow"])
                gates.append(("R2-RUN", True, ok, "%d steps; body err %s, twin err %s, overflow %s" % (
                    R2_STEPS, r2["body_err"], sorted(set(r2["twin_err"])), sorted(set(r2["overflow"])))))
            except Exception as e:                                         # noqa: BLE001
                traceback.print_exc()
                gates.append(("R2-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        BC.free(host=True)
    paths = dict(SH=arms.get("SH", {}).get("trace"))
    paths.update(arms.get("R2", {}).get("traces", {}))
    for tag, path in paths.items():
        if path and os.path.exists(path):
            log("parsing %s" % path)
            agg[tag] = P.aggregate(P.load_and_pack(path), N_WIN)
    seen = {k: (agg[k]["total"]["kernels"] if k in agg else 0) for k in ("SH", "G", "F")}
    gates.append(("PROF", True, all(v > 0 for v in seen.values()), "GPU ops per profiled step %s" % seen))
    per = dict(SH=arms.get("SH", {}).get("periods"))
    per.update(arms.get("R2", {}).get("periods", {}))
    for tag, ps in per.items():
        if ps:
            s = BC.period_stats(ps, list(range(len(ps))))
            dev_ms = agg.get(tag, {}).get("total", {}).get("ms")
            timing[tag] = dict(s, device_ms=dev_ms, gap_ms=(round(s["median_ms"] - dev_ms, 3)
                                                           if dev_ms is not None else None))
    res["MF"] = dict(arms=arms, profile=agg, timing=timing, workload_sha256=sha, batch=a.batch, doc_rows=rows,
                     attn_splits=NL._ATTN_SPLITS)
    return verdicts(agg, timing)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    if "MF" in a.parts and not a.model:
        print("REFUSED: MF needs --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 20
    import torch
    os.makedirs(a.out)
    gates, res, preds = [], {}, []
    for part, fn in (("GT", lambda: part_gt(gates, res)), ("GB", lambda: part_gb(gates, res)),
                     ("MF", lambda: preds.extend(part_mf(a, gates, res, a.out)))):
        if part not in a.parts:
            continue
        log("part %s" % part)
        try:
            fn()
        except Exception as e:                                             # noqa: BLE001
            traceback.print_exc()
            gates.append(("%s-RAN" % part, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    meta = dict(parts=a.parts, batch=a.batch, r2_steps=R2_STEPS, sh_steps=SH_STEPS, setup=SETUP, blocks=BLOCKS,
                predictions=PRED, class_rules=P.CLASS_RULES, nosi_commit=os.environ.get("FUSED_COMMIT", "unknown"),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    out = dict(meta=meta, results=res,
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:400]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    for part in ("GT", "GB"):
        if part in res:
            ms = res[part]["ms_per_call"] if part == "GT" else res[part]["run"]["ms_per_call"]
            print("REPORT %s ms per call: %s" % (part, ms))
    mf = res.get("MF", {})
    for tag in ("SH", "G", "F"):
        if tag in mf.get("profile", {}):
            x = mf["profile"][tag]
            cls = ", ".join("%s %.1f/%.3f" % (k, v["kernels"], v["ms"]) for k, v in x["classes"].items())
            print("REPORT %s per step: %s ops / %s ms; classes (ops/ms) %s; timing %s" % (
                tag, x["total"]["kernels"], x["total"]["ms"], cls, mf.get("timing", {}).get(tag)))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_fused: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:                                                      # noqa: BLE001
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
