"""r2-poolpf "pf1": Task 1.2 of kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md on the cached-NOSA baseline
ITSELF (mb1: C 128 = 63 + pool 65, B 216, 16K, NOSI_ATTN_SPLITS 4): the pool prefetch (nosi/nosi/pool_prefetch.py)
and the paper 4.4 A/B of CPU packing against the shipped main-stream GPU gather. One process, one model.

THE OPT-IN FLAG. --pool-prefetch (default off) attaches pool_prefetch.PoolPrefetch to the engine; --fetch gpugather
(default: the shipped gather) or cpupack; --pf-bg-mib N and --pf-bg-commit none|pool add a background load. Without
--pool-prefetch the arm "PF" does not exist and every other arm still names its own configuration (ARMS below).

PARTS (in this order; with --stop-on-gate (default) a failed MUST gate of G1 / G2 skips T):
  G1  PF-HIT and PF-HIT-CTRL (MUST): two small engine sets on the GPU with the REAL kernels (2 layers, B 2, S 100
      blocks + 5 rows, bf16, the import-time pool). Step 0, then a prefetch of X (a block in neither the window nor
      the pool) in set 1 only, then step 1 with X in the selection of one (layer, head, request).
      PF-HIT: set 1's pool_update takes X from the pinned pool slot (SWAP / MOVE_IN at X's window slot, target = the
      pinned slot), the window holds X's host rows (K, V) and total_cis rows (bias) bitwise, counters committed 1 and
      pin_hits 1, and the attended windows of both sets are bitwise equal.
      PF-HIT-CTRL: in set 0 (no prefetch) the same slot is a miss (MOVE_OUT) and the gather wrote X's host rows.
  G2  PF-EQ-BGP, PF-EQ-CP, PF-PIN-CONS (MUST): B gate_batch (8), gate_steps (64) greedy steps, three runs from fresh
      caches: SH (the shipped decode), BGP<gate_bg_mib> (the background load committed to the pool with soft pins: the
      pool fills with pins, so commit overflows and pinned evictions happen) and CP (--fetch cpupack). The logits of
      EVERY step of each run equal SH's bitwise (the pool is a cache of exact bytes; cpupack writes the same bytes).
      PF-PIN-CONS: BGP's committed == pin_hits + pin_evicted + pins_live, with committed > 0 and an overflow path that
      ran (commit_overflow + pin_evicted > 0).
  T   the timing schedule at --batch (216): ONE prefill, then segments of --seg (64) steps, one arm per segment (the
      re-warm steps of the shipped decode fall on the segment starts: 64-token blocks). Default schedule
      SH,PF0,BG64,BG192,CP,CP,BG192,BG64,PF0,SH (a palindrome: a linear drift over the run cancels per arm). Every
      arm's steady step = the median period of its non-re-warm steps (r2_base_cell's statistic), and its ratio to SH
      IN THIS JOB (the same-job baseline, Task 0.4).
ARMS: SH (flag off: the shipped decode), PF0 (pool prefetch on, gpugather, no load: the hooks' own cost), BG<N> (+ N MiB
  of K + V per step of background packing; commit "none": pack + copy engine + landing scatter into the staging),
  BGP<N> (the same, committed to the pool with soft pins), CP (--fetch cpupack: the step's own misses through the CPU
  path on the critical path), PF (the arm of the flags).
REPORTED PER ARM: the steady step (median, mean, p10, p90), the ratio to SH, tok/s; the process CPU cores (rusage CPU
  seconds / wall seconds per segment, after a device sync at each segment end) and their delta to SH; BG: the bytes per
  step, the delivery ms of each job (an issue-time event on an idle stream to an arrival-time event recorded after the
  job's completion event; "late" = the job had landed when the probe was set: an upper bound), the backlog (jobs not
  landed) at each step start and its growth per segment, skipped jobs; CP: the fetch time per layer (device events
  around the replaced gathers, the host wait included), the host split (mask sync, issue, packing wait, scatter
  enqueue), items and chunks; the transport's worker statistics; the pool prefetch counters.
REGISTERED PREDICTIONS: PRED_PF below (TBD for the lead; r2_poolpf_step.sh STOPS while PRED_STAMP is None).
EXIT: the number of failed MUST gates (<= 19); 20 refused; 30 an exception outside a part.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_base_cell as BC          # noqa: E402

RG = BC.RG
SCHEDULE = "SH,PF0,BG64,BG192,CP,CP,BG192,BG64,PF0,SH"
# REGISTERED PREDICTIONS of pf1. TBD: the lead writes each band (low, high) and PRED_STAMP (from `date -u`, L33) BEFORE
# any run. A band left None gives the verdict REPORT. r2_poolpf_step.sh STOPS while PRED_STAMP is None.
PRED_STAMP = None
PRED_PF = {
    "PF-HOOK": None,       # PF0 / SH steady median step (the hooks' cost, no load)
    "PF-BG64": None,       # BG64 / SH steady median step (interference of 64 MiB per step of CPU packing)
    "PF-BG192": None,      # BG192 / SH steady median step
    "PF-CP": None,         # CP / SH steady median step (cpupack on the critical path)
    "PF-DLV64": None,      # BG64 delivery, median ms (issue to arrival)
    "PF-DLV192": None,     # BG192 delivery, median ms
    "PF-KEEP192": None,    # BG192 backlog growth per segment (jobs, last sample - first sample)
    "PF-CORES192": None,   # BG192 cores minus SH cores (rusage)
    "PF-CPFETCH": None,    # CP fetch per layer, median device ms
}
RATIO_KEYS = (("PF-HOOK", "PF0"), ("PF-BG64", "BG64"), ("PF-BG192", "BG192"), ("PF-CP", "CP"))


def log(msg):
    print("[r2_poolpf %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ pure parts (CPU-tested)
def arm_config(name, a):
    """The PoolPrefetch mode of an arm (None: SH, the flag off)."""
    if name == "SH":
        return None
    if name == "PF0":
        return dict(fetch="gpugather", bg_mib=0.0, bg_commit="none")
    if name == "CP":
        return dict(fetch="cpupack", bg_mib=0.0, bg_commit="none")
    m = re.fullmatch(r"BG(P?)([1-9][0-9]{0,3})", name)
    if m:
        return dict(fetch="gpugather", bg_mib=float(m.group(2)), bg_commit="pool" if m.group(1) else "none")
    if name == "PF":
        if not a.pool_prefetch:
            raise ValueError("arm PF needs --pool-prefetch")
        return dict(fetch=a.fetch, bg_mib=float(a.pf_bg_mib), bg_commit=a.pf_bg_commit)
    raise ValueError("unknown arm %r (SH, PF0, CP, BG<N>, BGP<N>, PF)" % name)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--pool", type=int, default=65, help="must equal NOSI_POOL_BLOCKS (> 0: the pool prefetch needs it)")
    ap.add_argument("--seg", type=int, default=64, help="steps per segment (64: one re-warm per segment start)")
    ap.add_argument("--schedule", default=SCHEDULE)
    ap.add_argument("--gate-batch", type=int, default=8)
    ap.add_argument("--gate-steps", type=int, default=64)
    ap.add_argument("--gate-bg-mib", type=int, default=64)
    ap.add_argument("--stop-on-gate", type=int, default=1, choices=(0, 1))
    ap.add_argument("--pool-prefetch", action="store_true", help="the OPT-IN flag of the arm PF")
    ap.add_argument("--fetch", default=None, choices=("gpugather", "cpupack"))
    ap.add_argument("--pf-bg-mib", type=float, default=None)
    ap.add_argument("--pf-bg-commit", default=None, choices=("none", "pool"))
    ap.add_argument("--pf-lanes", type=int, default=128)
    ap.add_argument("--pf-threads", type=int, default=8)
    ap.add_argument("--pf-backlog-cap", type=int, default=16)
    a = ap.parse_args(argv)
    if not a.pool_prefetch and (a.fetch is not None or a.pf_bg_mib is not None or a.pf_bg_commit is not None):
        ap.error("--fetch / --pf-bg-mib / --pf-bg-commit go with --pool-prefetch (the opt-in flag)")
    a.fetch = a.fetch or "gpugather"
    a.pf_bg_mib = 0.0 if a.pf_bg_mib is None else a.pf_bg_mib
    a.pf_bg_commit = a.pf_bg_commit or "none"
    a.schedule = [x for x in a.schedule.split(",") if x]
    try:
        cfgs = [arm_config(x, a) for x in a.schedule]
    except ValueError as e:
        ap.error(str(e))
    if a.fetch == "cpupack" and a.pf_bg_mib > 0:
        ap.error("--fetch cpupack and a background load are separate arms")
    if "SH" not in a.schedule:
        ap.error("the schedule needs the same-job baseline SH")
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d (the pool prefetch writes pool slots)" % BC.POOL_MAX)
    steps = a.seg * len(a.schedule)
    if a.batch < 1 or a.seg < 2 or steps > 1000 or not 2 <= a.gate_steps <= 1000 or a.gate_batch < 1:
        ap.error("--batch >= 1, --seg >= 2, seg x arms <= 1000 (total_cis holds S + 1024), --gate-steps in 2 .. 1000")
    if a.pf_lanes < 1 or not 1 <= a.pf_threads <= 256:
        ap.error("--pf-lanes >= 1, --pf-threads in 1 .. 256")
    a.cfgs = cfgs
    return a


def pick_cores(n, allowed):
    """The C++ worker's cores: the last n of the allowed set (None when the set has no more than n)."""
    allowed = sorted(allowed)
    return allowed[-n:] if len(allowed) > n else None


def summarize(B, schedule, periods, rewarms, segs):
    """Per arm: steady statistics over its non-re-warm steps, the ratio to SH, the cores, and the merged pool prefetch
    records of its segments."""
    rw = {int(t) for t, _ in rewarms}
    arms = {}
    for name in dict.fromkeys(schedule):
        mine = [s for s in segs if s["arm"] == name]
        keep = [t for s in mine for t in range(s["t0"], s["t1"]) if t not in rw]
        st = BC.period_stats(periods, keep)
        cpu, wall = sum(s["cpu_s"] for s in mine), sum(s["wall_s"] for s in mine)
        rec = dict(steady=st, segments=len(mine), steps=sum(s["t1"] - s["t0"] for s in mine),
                   steady_tok_s=(round(1e3 * B / st["median_ms"], 1) if st["median_ms"] else None),
                   cores=(round(cpu / wall, 3) if wall > 0 else None), cpu_s=round(cpu, 3), wall_s=round(wall, 3))
        reps = [s["pf"] for s in mine if s.get("pf")]
        if reps:
            dl = [x for r in reps for x in r["delivery_ms"]]
            ms = sorted(x[0] for x in dl)
            cp = sorted(x for r in reps for x in r["cp_fetch_ms_per_layer"])
            growth = [r["backlog"][-1] - r["backlog"][0] for r in reps if len(r["backlog"]) >= 2]
            stats = {}
            for r in reps:
                for k, v in r["stats"].items():
                    stats[k] = stats.get(k, 0) + v
            rec.update(pf_stats=stats, counters=reps[-1]["counters"], transport=reps[-1]["transport"],
                       geometry=reps[-1]["geometry"],
                       delivery=dict(n=len(ms), late=sum(1 for x in dl if x[1]),
                                     median_ms=(round(statistics.median(ms), 4) if ms else None),
                                     p90_ms=(ms[min(len(ms) - 1, int(0.9 * (len(ms) - 1) + 0.5))] if ms else None),
                                     max_ms=(ms[-1] if ms else None)),
                       backlog=dict(max=max((max(r["backlog"]) for r in reps if r["backlog"]), default=None),
                                    growth_per_segment=growth, last=[r["backlog"][-1] for r in reps if r["backlog"]]),
                       bytes_per_step=(round(stats.get("bytes_issued", 0) / max(1, stats.get("steps", 0)))),
                       cp_fetch_ms_per_layer=dict(n=len(cp), median=(round(statistics.median(cp), 4) if cp else None),
                                                  p90=(cp[min(len(cp) - 1, int(0.9 * (len(cp) - 1) + 0.5))]
                                                       if cp else None)))
        arms[name] = rec
    sh = arms.get("SH")
    for name, rec in arms.items():
        if sh and sh["steady"]["median_ms"] and rec["steady"]["median_ms"]:
            rec["ratio_median"] = round(rec["steady"]["median_ms"] / sh["steady"]["median_ms"], 4)
            rec["ratio_mean"] = round(rec["steady"]["mean_ms"] / sh["steady"]["mean_ms"], 4)
            if rec["cores"] is not None and sh["cores"] is not None:
                rec["cores_minus_SH"] = round(rec["cores"] - sh["cores"], 3)
    return arms


def verdicts_pf(arms):
    """The registered predictions against this job's numbers: [(name, verdict, detail)]."""
    def judge(key, val, what):
        band = PRED_PF.get(key)
        if val is None:
            return (key, "NOT_RUN", "%s: no value" % what)
        if band is None:
            return (key, "REPORT", "%s %s (no registered band)" % (what, val))
        lo, hi = band
        return (key, "CONFIRMED" if lo <= val <= hi else "REFUTED", "%s %s (band [%s, %s])" % (what, val, lo, hi))
    out = []
    for key, arm in RATIO_KEYS:
        out.append(judge(key, (arms.get(arm) or {}).get("ratio_median"), "%s / SH steady median" % arm))
    for key, arm in (("PF-DLV64", "BG64"), ("PF-DLV192", "BG192")):
        out.append(judge(key, ((arms.get(arm) or {}).get("delivery") or {}).get("median_ms"),
                         "%s delivery median ms" % arm))
    g = ((arms.get("BG192") or {}).get("backlog") or {}).get("growth_per_segment")
    out.append(judge("PF-KEEP192", max(g) if g else None, "BG192 backlog growth per segment (max, jobs)"))
    out.append(judge("PF-CORES192", (arms.get("BG192") or {}).get("cores_minus_SH"), "BG192 cores - SH cores"))
    out.append(judge("PF-CPFETCH", ((arms.get("CP") or {}).get("cp_fetch_ms_per_layer") or {}).get("median"),
                     "CP fetch per layer, median device ms"))
    return out


# ------------------------------------------------------------------ the GPU parts
def _sel(cache, rnd, dev):
    import torch
    out = []
    for layer in cache.layers:
        e = layer.cache_engine
        tail = int(e.seq_length) // e.block_size
        H, Bn = e._block_map.shape[0], e._block_map.shape[1]
        t = torch.empty((H, Bn, e.topk), dtype=torch.int64)
        for h in range(H):
            for b in range(Bn):
                t[h, b] = torch.tensor(rnd.sample(range(tail), e.topk - 1) + [tail])
        out.append(t.to(dev))
    return out


def _engine_step(cache, sels, gen, dev, H, D):
    import torch
    for l, layer in enumerate(cache.layers):
        Bn = layer.cache_engine._k_gpu.shape[0]
        k = torch.randn((Bn, 1, H, D), generator=gen).to(torch.bfloat16).to(dev)
        v = torch.randn((Bn, 1, H, D), generator=gen).to(torch.bfloat16).to(dev)
        layer.cache_engine.decode_update(k, v, layer.total_cis, sels[l])


def part_hit(CEN, PFM, ext, gates, rec, threads, cores):
    """G1 (module docstring): PF-HIT and PF-HIT-CTRL with the real kernels on two small engine sets."""
    import random
    import torch
    dev = torch.device("cuda")
    Lh, Bh, H, D, BSZ = 2, 2, 2, 128, 64
    Sh = 100 * BSZ + 5

    def world(seed):
        g = torch.Generator().manual_seed(seed)
        layers = []
        for _ in range(Lh):
            e = CEN.CacheEngine(gpu_mem_usage=1, num_layers=Lh, block_size=BSZ, head_num=H, head_dim=D,
                                dtype=torch.bfloat16, device=dev, topk=64, has_kv_bias=True)
            e.max_gen_len = 2 * BSZ
            k = torch.randn((Bh, Sh, H, D), generator=g).to(torch.bfloat16).to(dev)
            v = torch.randn((Bh, Sh, H, D), generator=g).to(torch.bfloat16).to(dev)
            tc = torch.randn((Bh, Sh + 1024, H), generator=g).to(torch.bfloat16).to(dev)
            e.prefill_update(k, v, tc[:, :Sh], 0, Bh)
            layers.append(types.SimpleNamespace(cache_engine=e, total_cis=tc))
        torch.cuda.synchronize()
        return types.SimpleNamespace(layers=layers)
    w1, w0 = world(1), world(1)
    pf1 = PFM.PoolPrefetch(w1, ext, lanes=8, threads=threads, cores=cores)
    pf0 = PFM.PoolPrefetch(w0, ext, lanes=8, threads=threads, cores=cores)
    pf1.attach()
    pf0.attach()
    s0 = _sel(w1, random.Random(7), dev)
    g1, g0 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    pf1.begin_step()
    _engine_step(w1, s0, g1, dev, H, D)
    pf0.begin_step()
    _engine_step(w0, s0, g0, dev, H, D)
    torch.cuda.synchronize()
    l, h, b = 0, 0, 1
    e1, e0 = w1.layers[l].cache_engine, w0.layers[l].cache_engine
    tc = w1.layers[l].total_cis
    tail = int(e1.seq_length) // BSZ
    resident = set(e1._block_map[h, b].tolist()) | set(e1._pool_map[h, b].tolist())
    X = next(x for x in range(tail) if x not in resident)
    n = pf1.prefetch(torch.tensor([[l, h, b, X]]))
    pf1.tp.flush()
    pf1.begin_step()
    torch.cuda.synchronize()
    q = (e1._pool_map[h, b] == X).nonzero().view(-1).tolist()
    s1 = [t.clone() for t in s0]
    s1[l][h, b, 0] = X
    _engine_step(w1, s1, g1, dev, H, D)
    pf0.begin_step()
    _engine_step(w0, s1, g0, dev, H, D)
    torch.cuda.synchronize()

    def rows(t, blk):
        return t[b, blk * BSZ:(blk + 1) * BSZ, h]
    s_1 = (e1._block_map[h, b] == X).nonzero().view(-1).tolist()
    s_0 = (e0._block_map[h, b] == X).nonzero().view(-1).tolist()
    c = pf1.report()["counters"]
    ok_hit = (n == 1 and len(q) == 1 and len(s_1) == 1
              and int(e1._pool_action[h, b, s_1[0]]) in (PFM.POOL_SWAP, PFM.POOL_MOVE_IN)
              and int(e1._pool_target[h, b, s_1[0]]) == q[0]
              and torch.equal(rows(e1._k_gpu, s_1[0]), rows(e1._k_cpu, X))
              and torch.equal(rows(e1._v_gpu, s_1[0]), rows(e1._v_cpu, X))
              and torch.equal(rows(e1._kv_bias_gpu, s_1[0]), rows(tc, X))
              and c["committed"] == 1 and c["pin_hits"] == 1 and c["conserved"])
    win = all(torch.equal(getattr(x.cache_engine, nm)[:, :64 * BSZ], getattr(y.cache_engine, nm)[:, :64 * BSZ])
              for x, y in zip(w0.layers, w1.layers) for nm in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"))
    ok_ctrl = (len(s_0) == 1 and int(e0._pool_action[h, b, s_0[0]]) == PFM.POOL_MOVE_OUT
               and torch.equal(rows(e0._k_gpu, s_0[0]), rows(e0._k_cpu, X)))
    rec.update(X=X, pinned_slot=q, window_slot_1=s_1, window_slot_0=s_0, counters=c, windows_equal=win,
               action_1=(int(e1._pool_action[h, b, s_1[0]]) if s_1 else None),
               action_0=(int(e0._pool_action[h, b, s_0[0]]) if s_0 else None))
    gates.append(("PF-HIT", True, ok_hit and win, "X %d pinned in slot %s; step 1 action %s target %s; bytes %s; "
                  "counters %s; windows equal %s" % (X, q, rec["action_1"],
                                                     int(e1._pool_target[h, b, s_1[0]]) if s_1 else None,
                                                     "equal" if ok_hit else "see record", c, win)))
    gates.append(("PF-HIT-CTRL", True, ok_ctrl, "without the prefetch: action %s (MOVE_OUT = a miss), gather bytes %s"
                  % (rec["action_0"], "equal" if ok_ctrl else "see record")))
    pf1.close()
    pf0.close()


def run_segments(model, ids, segments, clock, a, ext, cores, keep_logits=False):
    """ONE prefill, then the segments [(arm, steps)] (r2_base_cell.run_shipped's loop, re-warms included). The pool
    prefetch is built after the prefill at the first non-SH segment, attached per segment and detached at its end.
    Returns the periods (ms), re-warms, segment records (cores, the pool prefetch report), logits (keep_logits)."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import pool_prefetch as PFM
    steps = sum(n for _, n in segments)
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, pos = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    pos = pos[:, -1:] + 1
    cu = torch.arange(0, Bn + 1, dtype=torch.int, device="cuda")
    model.has_buffers = False
    warm, rewarms, keep, segs = None, [], ([] if keep_logits else None), []
    ev = [clock.event() for _ in range(steps)]
    ev_end = [clock.event() for _ in segments]
    pf, pf_build_s = None, None
    t = 0
    for k, (arm, n) in enumerate(segments):
        mode = arm_config(arm, a)
        if mode is not None:
            if pf is None:
                tb = time.time()
                pf = PFM.PoolPrefetch(cache, ext, lanes=a.pf_lanes, threads=a.pf_threads, cores=cores,
                                      backlog_cap=a.pf_backlog_cap, **mode)
                pf_build_s = round(time.time() - tb, 3)
            else:
                pf.set_mode(**mode)
            pf.attach()
        ru0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        t_start = t
        for _ in range(n):
            l0 = cache.layers[0]
            need, why = BC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen,
                                       warm)
            if need:
                warm = BC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
                rewarms.append((t, why))
                model.has_buffers = False
            clock.record(ev[t])
            if mode is not None:
                pf.begin_step(capture=need)              # a re-warm step captures graphs: the worker stays idle
            lg = model.decode_inference(tok, cu, pos, cache)
            tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            if keep is not None:
                keep.append(lg.detach().clone())
            pos = pos + 1
            t += 1
        clock.record(ev_end[k])
        clock.sync()
        ru1, w1 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        rep = None
        if mode is not None:
            rep = pf.report()
            pf.detach()
        cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
        segs.append(dict(arm=arm, t0=t_start, t1=t, cpu_s=cpu, wall_s=w1 - w0, pf=rep))
        log("segment %d %s steps %d..%d: wall %.2f s, cpu %.2f s" % (k, arm, t_start, t - 1, w1 - w0, cpu))
    periods = []                                         # a segment's last step ends at the segment's end event
    for i in range(steps):
        k = next(j for j, s in enumerate(segs) if s["t0"] <= i < s["t1"])
        end = ev_end[k] if i + 1 == segs[k]["t1"] else ev[i + 1]
        periods.append(clock.elapsed(ev[i], end))
    fr, tot = torch.cuda.mem_get_info()
    geometry = None
    if pf is not None:
        geometry = dict(staging_bytes=pf.staging_bytes, transport_staging_bytes=getattr(pf.tp, "staging_bytes", None),
                        build_s=pf_build_s)
        pf.close()
    del cache
    return dict(periods=periods, rewarms=rewarms, segs=segs, logits=keep, prefill_s=round(prefill_s, 1),
                device_used_end_GB=round((tot - fr) / 1e9, 3), pf=geometry)


def part_eq(model, docs, a, ext, cores, clock, gates, rec):
    """G2 (module docstring): PF-EQ-BGP, PF-EQ-CP, PF-PIN-CONS at B gate_batch."""
    import torch
    ids = docs[BC.doc_rows(a.gate_batch)].contiguous().to("cuda")
    runs = {}
    for arm in ("SH", "BGP%d" % a.gate_bg_mib, "CP"):
        log("G2 run %s (B %d, %d steps)" % (arm, a.gate_batch, a.gate_steps))
        runs[arm] = run_segments(model, ids, [(arm, a.gate_steps)], clock, a, ext, cores, keep_logits=True)
        BC.free()
    ref = runs["SH"]["logits"]
    for arm, name in (("BGP%d" % a.gate_bg_mib, "PF-EQ-BGP"), ("CP", "PF-EQ-CP")):
        got = runs[arm]["logits"]
        bad = [i for i, (x, y) in enumerate(zip(ref, got)) if not torch.equal(x, y)]
        ok = len(got) == len(ref) == a.gate_steps and not bad
        gates.append((name, True, ok, "%s logits vs SH, bitwise, %d steps: %d differing steps (first %s)"
                      % (arm, len(got), len(bad), bad[0] if bad else None)))
    rep = runs["BGP%d" % a.gate_bg_mib]["segs"][0]["pf"]
    c = rep["counters"]
    ok = c["conserved"] and c["committed"] > 0 and c["commit_overflow"] + c["pin_evicted"] > 0
    gates.append(("PF-PIN-CONS", True, ok, "committed %d == hits %d + evicted %d + live %d: %s; commit_overflow %d"
                  % (c["committed"], c["pin_hits"], c["pin_evicted"], c["pins_live"], c["conserved"],
                     c["commit_overflow"])))
    for arm, r in runs.items():
        rec[arm] = dict(prefill_s=r["prefill_s"], rewarms=r["rewarms"], pf=r["segs"][0]["pf"],
                        steady_ms=BC.period_stats(r["periods"], [i for i in range(len(r["periods"]))
                                                                 if i not in {x for x, _ in r["rewarms"]}]))
    del runs, ref


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 20
    import torch
    import nosi.cache_engine as CEN
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    from nosi import pool_prefetch as PFM
    import gilfree_pack as GP
    if NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != a.pool:
        print("REFUSED: needs NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS == --pool (%d; the knob is %d)" % (
            a.pool, CEN.POOL_BLOCKS), file=sys.stderr)
        return 20
    try:
        docs, sha = RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, rec = [], dict(G1={}, G2={}, T=None)
    cores = pick_cores(a.pf_threads, os.sched_getaffinity(0))
    clock = BC.CudaClock()
    ext, build_s = None, None
    try:
        tb = time.time()
        ext = GP.load(GP.build_dir_from_env())
        build_s = round(time.time() - tb, 1)
        gates.append(("PF-BUILD", True, int(ext.VERSION) == 3 and int(ext.DESC) >= 2 and int(ext.OPENMP) == 1,
                      "gilfree_pack v%s DESC %s OPENMP %s, built in %s s" % (ext.VERSION, ext.DESC, ext.OPENMP,
                                                                           build_s)))
    except Exception as e:
        traceback.print_exc()
        gates.append(("PF-BUILD", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    model, arms = None, None
    if ext is not None:
        with torch.inference_mode():
            log("G1 PF-HIT (real kernels, small engines)")
            try:
                part_hit(CEN, PFM, ext, gates, rec["G1"], a.pf_threads, cores)
            except Exception as e:
                traceback.print_exc()
                gates.append(("PF-HIT", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            BC.free()
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s" % (time.time() - t0))
        with torch.inference_mode():
            try:
                part_eq(model, docs, a, ext, cores, clock, gates, rec["G2"])
            except Exception as e:
                traceback.print_exc()
                gates.append(("PF-EQ", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            BC.free(host=True)
            failed = [g[0] for g in gates if g[1] and not g[2]]
            if failed and a.stop_on_gate:
                log("T NOT_RUN: MUST gate(s) failed: %s" % failed)
                rec["T"] = dict(not_run="MUST gate(s) failed: %s" % failed)
            else:
                ids = docs[BC.doc_rows(a.batch)].contiguous().to("cuda")
                segments = [(arm, a.seg) for arm in a.schedule]
                log("T: B %d, %d segments of %d steps: %s" % (a.batch, len(segments), a.seg, ",".join(a.schedule)))
                try:
                    res = run_segments(model, ids, segments, clock, a, ext, cores)
                    pa, pr = BC.gpu_peak()
                    arms = summarize(a.batch, a.schedule, res["periods"], res["rewarms"], res["segs"])
                    rec["T"] = dict(prefill_s=res["prefill_s"], rewarms=res["rewarms"], gpu_peak_alloc_GB=pa,
                                    gpu_peak_reserved_GB=pr, device_used_end_GB=res["device_used_end_GB"],
                                    pf=res["pf"], arms=arms, segments=res["segs"],
                                    periods_ms=[round(x, 4) for x in res["periods"]])
                    gates.append(("T-RUN", True, True, "%d steps, %d re-warms, peak %s GB" % (
                        len(res["periods"]), len(res["rewarms"]), pa)))
                except Exception as e:
                    traceback.print_exc()
                    gates.append(("T-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    preds = verdicts_pf(arms or {})
    meta = dict(cell="pf1", batch=a.batch, pool_blocks=CEN.POOL_BLOCKS, capacity_C=BC.C_BASE + a.pool, seg=a.seg,
                schedule=a.schedule, arm_modes={x: arm_config(x, a) for x in dict.fromkeys(a.schedule)},
                gate_batch=a.gate_batch, gate_steps=a.gate_steps, gate_bg_mib=a.gate_bg_mib,
                pf_lanes=a.pf_lanes, pf_threads=a.pf_threads, pf_backlog_cap=a.pf_backlog_cap, cores=cores,
                affinity=sorted(os.sched_getaffinity(0)), ext_build_s=build_s, S0=BC.S0, workload_sha256=sha,
                attn_splits=NL._ATTN_SPLITS, pred_stamp=PRED_STAMP, predictions=PRED_PF,
                nosi_commit=os.environ.get("BASE_COMMIT", "unknown"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), host_mem=BC.host_mem())
    out = dict(meta=meta, G1=rec["G1"], G2=rec["G2"], T=rec["T"],
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:300]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    for name, r in (arms or {}).items():
        print("REPORT %s: steady %s ms (mean %s), ratio to SH %s / %s, %s tok/s, cores %s (- SH %s)%s%s" % (
            name, r["steady"]["median_ms"], r["steady"]["mean_ms"], r.get("ratio_median"), r.get("ratio_mean"),
            r["steady_tok_s"], r["cores"], r.get("cores_minus_SH"),
            ("; delivery median %s ms p90 %s (n %d, late %d); backlog %s; bytes/step %s" % (
                r["delivery"]["median_ms"], r["delivery"]["p90_ms"], r["delivery"]["n"], r["delivery"]["late"],
                r["backlog"], r["bytes_per_step"]) if r.get("delivery") and r["delivery"]["n"] else ""),
            ("; cp fetch/layer median %s ms p90 %s" % (r["cp_fetch_ms_per_layer"]["median"],
                                                      r["cp_fetch_ms_per_layer"]["p90"])
             if r.get("cp_fetch_ms_per_layer") and r["cp_fetch_ms_per_layer"]["n"] else "")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_poolpf: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
