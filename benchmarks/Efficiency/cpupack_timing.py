"""Stage-timing bookkeeping of the CPU-packing transport. Pure python, CPU-tested (retroinfer-eval
tests/test_cpupack_transport.py).

CLOCKS. Every per-request stage boundary is a stamp on ONE clock: the GPU clock, as milliseconds from the bracket's gate
event (CUDA events on the working streams; host transitions are CUDA events recorded on an idle aux stream right after the
host action = an UPPER bound of the host instant, the hisparse_repro.make_bracket 'sub' technique). Host monotonic
nanoseconds (perf_counter_ns / thread_time_ns) are kept ONLY for host-only durations (descriptor build, pack call, API calls,
backpressure waits) and are never subtracted from a GPU stamp: `seg` raises on a cross-clock difference.

PER REQUEST (layer l, request b with >= 1 missing group; c = the chunk holding b's LAST group):
  CPU arms   points  pr -> list -> hk -> desc -> pk[c] -> h2d0[c] -> h2d1[c] -> sc0[c] -> sc1[c]
             stages  (1) list_d2h = list - pr        (plan ready -> list in pinned memory; includes the plan stream queue)
                         wake     = hk - list        (host received it: sync wake-up, GIL, coordinator busy with layer l-1)
                     (2) desc     = desc - hk        (descriptor preparation)
                         pack     = pk[c] - desc     (packing of chunks up to c, incl. staging backpressure)
                     (3) submit   = h2d0[c] - pk[c]  (API submission + copy-stream queue / landing backpressure)
                         dma      = h2d1[c] - h2d0[c]
                     (4) sc_queue = sc0[c] - h2d1[c]
                         scatter  = sc1[c] - sc0[c]
             e2e = sc1[c] - pr = the sum of the stages EXACTLY (consecutive points on one clock; nothing overlapping is
             stacked). A negative segment (marker lag) is COUNTED, never clipped.
  W8 (GPU gather) points pr -> g0 -> g1 of the layer: queue, gather; e2e = g1 - pr (per LAYER: every request of the layer
             completes at the layer's end event).
PERCENTILES come from the individual samples (never a sum of per-stage medians). STEP SPANS are first-to-last intervals
(and interval unions for busy time), never sums of overlapping stage durations.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

GPU, HOST = "gpu", "host"
CPU_POINTS = ("pr", "list", "hk", "desc", "pk", "h2d0", "h2d1", "sc0", "sc1")
CPU_STAGES = ("list_d2h", "wake", "desc", "pack", "submit", "dma", "sc_queue", "scatter")
W8_POINTS = ("pr", "g0", "g1")
W8_STAGES = ("queue", "gather")
STAGE_GROUP = {"list_d2h": 1, "wake": 1, "desc": 2, "pack": 2, "submit": 3, "dma": 3, "sc_queue": 4, "scatter": 4,
               "queue": 0, "gather": 0}


def stamp(clock: str, value: float) -> Tuple[str, float]:
    if clock not in (GPU, HOST):
        raise ValueError(clock)
    return (clock, float(value))


def seg(a: Tuple[str, float], b: Tuple[str, float]) -> float:
    """b - a for two stamps of the SAME clock; a cross-clock difference raises."""
    if a[0] != b[0]:
        raise ValueError("cross-clock subtraction %s -> %s refused" % (a[0], b[0]))
    return b[1] - a[1]


def segments(points: Sequence[Tuple[str, float]], names: Sequence[str]) -> Dict[str, float]:
    if len(points) != len(names) + 1:
        raise ValueError("%d points for %d stages" % (len(points), len(names)))
    return {n: seg(points[i], points[i + 1]) for i, n in enumerate(names)}


def last_chunk_of_requests(req_groups: Sequence[int], chunk_bounds: Sequence[Tuple[int, int]]) -> List[int]:
    out, cum, ci = [], 0, 0
    for c in (int(x) for x in req_groups):
        if c == 0:
            out.append(-1)
            continue
        cum += c
        while not (chunk_bounds[ci][0] <= cum - 1 < chunk_bounds[ci][1]):
            ci += 1
        out.append(ci)
    return out


def request_samples_cpu(layer: int, lay: Dict, req_groups: Sequence[int]) -> List[Dict]:
    """lay: dict(pr, list, hk, desc (GPU ms from the gate) and chunks = [dict(g0, g1, pk, h2d0, h2d1, sc0, sc1)]).
    One sample per request with >= 1 group."""
    chunks = lay.get("chunks") or []
    if not chunks:
        return []
    last = last_chunk_of_requests(req_groups, [(c["g0"], c["g1"]) for c in chunks])
    out = []
    for b, (ci, n) in enumerate(zip(last, req_groups)):
        if ci < 0:
            continue
        c = chunks[ci]
        vals = [lay["pr"], lay["list"], lay["hk"], lay["desc"], c["pk"], c["h2d0"], c["h2d1"], c["sc0"], c["sc1"]]
        if any(v is None for v in vals):
            raise ValueError("a stamp is missing (lite events?) for layer %d chunk %d" % (layer, ci))
        pts = [stamp(GPU, v) for v in vals]
        s = segments(pts, CPU_STAGES)
        s.update(layer=layer, b=b, groups=int(n), chunk=ci, e2e=seg(pts[0], pts[-1]))
        out.append(s)
    return out


def request_samples_w8(layer: int, pr: float, g0: float, g1: float, req_groups: Sequence[int]) -> List[Dict]:
    pts = [stamp(GPU, pr), stamp(GPU, g0), stamp(GPU, g1)]
    s = segments(pts, W8_STAGES)
    e2e = seg(pts[0], pts[-1])
    return [dict(s, layer=layer, b=b, groups=int(n), e2e=e2e) for b, n in enumerate(req_groups) if int(n) > 0]


def pctl(xs: Iterable[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default) of individual samples; NaN for none."""
    v = sorted(float(x) for x in xs if x is not None and x == x)
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize(samples: Sequence[Dict], keys: Sequence[str]) -> Dict[str, Dict]:
    out = {}
    for k in keys:
        xs = [s[k] for s in samples if k in s]
        out[k] = dict(n=len(xs), p50=pctl(xs, 50), p95=pctl(xs, 95), mean=(sum(xs) / len(xs) if xs else float("nan")),
                      max=(max(xs) if xs else float("nan")), n_negative=sum(1 for x in xs if x < 0))
    return out


def union_len(iv: Iterable[Tuple[float, float]]) -> float:
    tot, cur = 0.0, None
    for s, e in sorted(iv):
        if e <= s:
            continue
        if cur is None or s > cur[1]:
            if cur is not None:
                tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur is not None:
        tot += cur[1] - cur[0]
    return tot


def intersect_len(iv: Iterable[Tuple[float, float]], lo: float, hi: float) -> float:
    return union_len([(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi])


def copy_gaps(flat_chunks: Sequence[Dict]) -> Dict:
    """Copy-stream idle gaps between consecutive chunks of one rep (global order): gap = max(0, h2d0[c] - h2d1[c-1]);
    HOST-STARVED when chunk c was not yet packed when the copy engine became free (pk[c] > h2d1[c-1]), else other
    (landing backpressure / submission)."""
    starved = other = 0.0
    n_st = 0
    for a, b in zip(flat_chunks, flat_chunks[1:]):
        if a.get("h2d1") is None or b.get("h2d0") is None:
            continue
        gap = max(0.0, b["h2d0"] - a["h2d1"])
        if b.get("pk") is not None and b["pk"] > a["h2d1"]:
            starved += gap
            n_st += int(gap > 0)
        else:
            other += gap
    return dict(host_starved_ms=starved, other_gap_ms=other, n_host_starved=n_st)


def rep_aggregate(rec: Dict, useful_per_layer: Sequence[int]) -> Dict:
    """Full-step aggregates of one CPU or W8 rep (explicit denominators): span = first plan-ready -> last completion;
    useful GB/s = useful bytes / span; late = completion after the decode end (tm); during-step bytes = bytes whose
    completion event is <= tm; overlap = |[pr0, last] n [t0, tm]| / |[pr0, last]|; DMA busy = union of H2D intervals."""
    pr = [x for x in rec.get("pr", []) if x is not None]
    comp: List[Tuple[float, int]] = []          # (completion ms, bytes)
    h2d_iv = []
    flat = []
    if rec.get("kind") == "w8":
        for l, g1 in enumerate(rec.get("g1") or []):
            if g1 is not None:
                comp.append((g1, int(useful_per_layer[l])))
    else:
        for l, lay in enumerate(rec.get("layers") or []):
            for c in lay.get("chunks") or []:
                flat.append(c)
                if c.get("sc1") is not None:
                    comp.append((c["sc1"], int(c.get("useful", 0))))
                elif c.get("h2d1") is not None:
                    comp.append((c["h2d1"], int(c.get("useful", 0))))
                if c.get("h2d0") is not None and c.get("h2d1") is not None:
                    h2d_iv.append((c["h2d0"], c["h2d1"]))
    nan = float("nan")
    if not pr or not comp:
        return dict(span_ms=nan, useful_bytes=sum(useful_per_layer), useful_gbps=nan)
    first, last = min(pr), max(t for t, _ in comp)
    span = last - first
    tot = sum(useful_per_layer)
    out = dict(span_ms=span, useful_bytes=tot, useful_gbps=(tot / (span * 1e6) if span > 0 else nan), first_pr_ms=first, last_done_ms=last,
               dma_busy_ms=union_len(h2d_iv))
    t0, tm = rec.get("t0"), rec.get("tm")
    if t0 is not None and tm is not None and rec.get("with_decode"):
        out.update(late_ms=max(0.0, last - tm), during_bytes=sum(b for t, b in comp if t <= tm),
                   overlap_frac=(intersect_len([(first, last)], t0, tm) / span if span > 0 else nan),
                   dma_busy_in_decode_ms=intersect_len(h2d_iv, t0, tm))
        out["during_gbps"] = out["during_bytes"] / ((tm - t0) * 1e6) if tm > t0 else nan
    if flat:
        out.update(copy_gaps(flat))
    return out


def negative_segments(samples: Sequence[Dict], keys: Sequence[str] = CPU_STAGES) -> int:
    return sum(1 for s in samples for k in keys if k in s and s[k] < 0)
