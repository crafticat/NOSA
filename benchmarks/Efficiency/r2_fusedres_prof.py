"""The KERNEL TIME of the residency resolve per draft tick (review-fusedres-0e28cf6 M3): a parser of torch.profiler's
Chrome / Kineto trace. JSON only: no torch, no GPU (the CPU test FR8 feeds it synthetic traces).

r2_tick_sg5.py --b2-res-profile wraps every TickTimer span of the draft-only engine in
torch.profiler.record_function("t.<cat>") (so the ranges are named "t.tick", "t.res", "t.att", ...) and exports the
draft-only phase's trace. This parser:
  RANGES   the CPU "user_annotation" events named "t.tick" and "t.<cat>" per (pid, tid);
  LAUNCHES the CPU runtime / driver events (cat "cuda_runtime" or "cuda_driver": cudaLaunchKernel, cuLaunchKernel for
           Triton, cudaMemcpyAsync, ...) with args.correlation;
  DEVICE   the GPU events (cat "kernel", "gpu_memcpy", "gpu_memset") with args.correlation and dur (us).
A device event belongs to category <cat> when its launch lies inside a "t.<cat>" range on the launching thread, and to
the tick whose "t.tick" range (same thread) contains that launch. The ranges of one thread do not overlap (TickTimer's
spans are disjoint; t.tick encloses them), so the INNERMOST range is the category. Output per tick: the category's
device ms and its device-event count, and per kernel name (count, ms) over the window; device events without a
launch event (no correlation) are counted, never silently dropped.
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, List

LAUNCH_CATS = ("cuda_runtime", "cuda_driver")
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def _ranges(events: List[Dict[str, Any]], name: str):
    out: Dict[tuple, List[tuple]] = {}
    for e in events:
        if e.get("cat") == "user_annotation" and e.get("name") == name and e.get("ph") == "X":
            out.setdefault((e.get("pid"), e.get("tid")), []).append((float(e["ts"]), float(e["ts"]) + float(e["dur"])))
    for v in out.values():
        v.sort()
    return out


def _find(ranges, key, ts):
    """The index of the range of thread `key` that contains ts (None: none)."""
    rs = ranges.get(key)
    if not rs:
        return None
    i = bisect.bisect_right(rs, (ts, float("inf"))) - 1
    if i >= 0 and rs[i][0] <= ts <= rs[i][1]:
        return i
    return None


def parse_res_trace(trace: Dict[str, Any], cat: str = "res") -> Dict[str, Any]:
    """Per tick (the t.tick ranges of the launching thread, in time order): the device ms and the device events of
    category `cat`, and the per-name totals; plus the counts of the parse's own checks."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    ticks = _ranges(events, "t.tick")
    cats = _ranges(events, "t." + cat)
    if not ticks:
        raise ValueError("the trace has no 't.tick' ranges (was --b2-res-profile's span wrapper active?)")
    launch_of: Dict[Any, tuple] = {}
    for e in events:
        if e.get("cat") in LAUNCH_CATS and e.get("ph") == "X":
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launch_of[c] = ((e.get("pid"), e.get("tid")), float(e["ts"]))
    per_tick: Dict[tuple, Dict[str, float]] = {}
    by_name: Dict[str, List[float]] = {}
    orphans = outside_tick = 0
    for e in events:
        if e.get("cat") not in DEVICE_CATS or e.get("ph") != "X":
            continue
        c = (e.get("args") or {}).get("correlation")
        if c is None or c not in launch_of:
            orphans += 1
            continue
        key, ts = launch_of[c]
        t = _find(ticks, key, ts)
        if t is None:
            outside_tick += 1
            continue
        if _find(cats, key, ts) is None:
            continue
        rec = per_tick.setdefault((key, t), dict(ms=0.0, n=0))
        rec["ms"] += float(e["dur"]) / 1e3
        rec["n"] += 1
        nm = by_name.setdefault(str(e.get("name")), [0, 0.0])
        nm[0] += 1
        nm[1] += float(e["dur"]) / 1e3
    n_ticks = max(len(v) for v in ticks.values())
    main = max(ticks, key=lambda k: len(ticks[k]))                    # the thread that runs the ticks
    ms = [per_tick.get((main, i), dict(ms=0.0))["ms"] for i in range(n_ticks)]
    n = [per_tick.get((main, i), dict(n=0))["n"] for i in range(n_ticks)]
    other = sum(v["n"] for (k, _), v in per_tick.items() if k != main)
    return dict(cat=cat, ticks=n_ticks, ms=ms, n=n, by_name={k: dict(n=v[0], ms=v[1]) for k, v in by_name.items()},
                orphans=orphans, outside_tick=outside_tick, other_thread=other)


def summarize(p: Dict[str, Any], skip: int = 1) -> Dict[str, Any]:
    """The per-tick lists without the first `skip` ticks (compiles and warm-up): mean / median / min / max."""
    def st(xs):
        xs = sorted(xs)
        if not xs:
            return None
        m = len(xs) // 2
        med = xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])
        return dict(mean=sum(xs) / len(xs), median=med, min=xs[0], max=xs[-1], n=len(xs))
    return dict(p, ms_stats=st(p["ms"][skip:]), n_stats=st([float(x) for x in p["n"][skip:]]), skip=skip)
