"""FEEDER DIAGNOSTIC metrics, block scoring, the four preregistered diagnostic targets, the failure-diagnosis rule and the
time-budget arithmetic. Pure python (no torch), CPU-tested (retroinfer-eval tests/test_feeder_diag.py).

LABELS: finite-burst transport / placement-ready in SCRATCH only; NOT sustained live drafting or committed throughput.
Occupancy is CUDA-EVENT occupancy of the copy stream, NOT a physical bus counter. Repetitions are HARDWARE repetitions of
the same frozen plans, not independent workloads.

CLOCKS. Device quantities are milliseconds from the bracket's gate event (one common plan-ready origin per repetition:
pr[l] = the plan-ready event of layer l on the plan stream). Host quantities are perf_counter_ns instants. A device and a
host stamp are NEVER subtracted; a ratio of two durations is formed only from the same clock (dev_ratio / host_ratio).

PER REPETITION (rep_metrics; useful = K + V bytes of the plan's groups; wire = index rows + K + V of every chunk H2D):
  fullpath_ms        last scatter end (max sc1) - first plan-ready (min pr)                         [device]
  fullpath_gbps      useful / fullpath_ms  = placement-ready useful GB/s (makespan rate)
  dma_busy_ms        union of the chunk H2D intervals [h2d0, h2d1]                                [device, full only]
  dma_span_ms        last h2d1 - first h2d0
  occupancy          dma_busy_ms / dma_span_ms (first-to-last DMA event occupancy)
  active_useful_gbps useful / dma_busy_ms ; active_wire_gbps = wire / dma_busy_ms
  path_vs_active     fullpath_gbps / active_useful_gbps (target 3)
  gaps (copy-stream idle between consecutive chunks in submission order, max(0, h2d0[k] - h2d1[k-1])):
                     startup_ms = h2d0[first] - pr[first layer]; within_ms (same layer); boundary_ms (a layer change);
                     tail_ms = last sc1 - last h2d1 (scatter drain)
  host (full only):  pack_busy = union [p0, p1]; submit_busy = union [sub0, sc1h]; overlap_ms = |pack n submit|;
                     overlap_frac = overlap_ms / submit_busy; free-slot wait = sum (fw1 - fw0); FIFO-full wait = sum
                     (pub - qf0); starvation = sum over chunks k >= 1 of the submitter's idle wait (pop - iw0); S1-barrier
                     wait = sum (bw1 - bw0); list wait, descriptor time; depth at pop p50; full-chunk pack ms p50 and
                     rawpack GB/s; per-chunk submission ms (sc1h - sub0) p50; copy call (cp1 - cp0) p50 / p95; copy return ->
                     scatter submission (sc0h - cp1) p50; thread CPU ms (producer pack, submitter) separately
  conservation:      chunks, useful and wire sums (checked against the plan by feeder_core.audit_rep)
  contention:        per role (producer, its OpenMP helpers, submitter) the run-queue WAIT ms, on-CPU ms and involuntary context
                     switches between the drain before the gate and the drain after the repetition (/proc schedstat deltas)
  beside the decode: decode_ms = tm - t0 [device]; during_useful = useful of chunks whose sc1 <= tm; during_gbps.
BLOCK (score_blocks): 3 arms x (transfer-alone, resident-alone, concurrent); COMPLETE when all 9 rows exist, every row ok and
its gated step passed the golden gate. Only complete blocks are scored. extra_ms = concurrent decode - the paired
resident-alone decode of the same arm in the same block; slowdown = extra_ms / resident decode.
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ARMS = ("S0", "S1", "S2")
PHASES = ("alone", "resident", "conc")
T1_MIN = 1.10          # (1) S1 B336 transfer-alone delivery >= 1.10 x paired S0
T2_MAX = 0.75          # (2) S2 B64 boundary-gap time <= 0.75 x S1
T3_OCC = 0.90          # (3) S2 B336 first-to-last DMA event occupancy >= 90 %
T3_PATH = 0.85         #     AND fullpath useful rate >= 85 % of its own activeDMA useful rate
T4_DIFF = 0.05         # (4) |heavy / light median fullpath - 1| > 5 % -> light defines performance
DIAG = dict(pack_slow=1.10, starve_frac=0.25, submit_slow=1.25, copy_p95=2.0, drain_share=0.25)


# ------------------------------------------------------------------------------------------------ primitives
def pctl(xs: Iterable[Optional[float]], q: float) -> float:
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


def union_len(iv: Iterable[Tuple[float, float]]) -> float:
    tot, cur = 0.0, None
    for s, e in sorted((a, b) for a, b in iv if a is not None and b is not None):
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


def overlap_len(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    """|union(a) n union(b)| = |A| + |B| - |A u B| (same clock)."""
    return union_len(a) + union_len(b) - union_len(list(a) + list(b))


def _nan():
    return float("nan")


def _div(a, b):
    return a / b if (a is not None and b is not None and b == b and b > 0 and a == a) else float("nan")


def _cols(rec, which):
    cols = rec.get("%s_cols" % which) or []
    return [dict(zip(cols, r)) for r in rec.get(which) or []]


# ------------------------------------------------------------------------------------------------ one repetition
def rep_metrics(r: Dict) -> Dict:
    """Metrics of one bracket row (module docstring). r: dict with pr (ms), t0 / tm (ms, beside the decode), full, and
    rec = dict(lay_cols, lay, ch_cols, ch (host ns), evms_cols, evms (device ms))."""
    rec = r.get("rec") or {}
    chs = _cols(rec, "ch")
    evs = _cols(rec, "evms")
    lays = _cols(rec, "lay")
    pr = [x for x in (r.get("pr") or []) if x is not None]
    out = dict(chunks=len(chs), useful=sum(int(x.get("useful") or 0) for x in chs), wire=sum(int(x.get("wire") or 0) for x in chs))
    sc1 = [e.get("sc1") for e in evs if e.get("sc1") is not None]
    nan = _nan()
    if r.get("with_decode") and r.get("t0") is not None and r.get("tm") is not None:
        t0, tm = r["t0"], r["tm"]
        dur = sum(int(c.get("useful") or 0) for c, e in zip(chs, evs) if e.get("sc1") is not None and e["sc1"] <= tm)
        out.update(decode_ms=tm - t0, during_useful=dur, during_gbps=_div(dur, (tm - t0) * 1e6))
    sched = r.get("sched") or {}
    if sched:                                                            # runnable-thread contention (schedstat deltas, host)
        role = lambda n: "producer" if n == "producer" else ("submitter" if n == "submitter" else ("helper" if n.startswith("helper") else None))
        for k in ("producer", "submitter", "helper"):
            xs = [v for n, v in sched.items() if role(n) == k and v]
            out["rq_wait_ms_" + k] = sum(int(v.get("wait_ns", 0)) for v in xs) / 1e6
            out["run_ms_" + k] = sum(int(v.get("run_ns", 0)) for v in xs) / 1e6
            out["nvcsw_" + k] = sum(int(v.get("nvcsw", 0)) for v in xs)
        out["rq_wait_ms_transport"] = out["rq_wait_ms_producer"] + out["rq_wait_ms_submitter"] + out["rq_wait_ms_helper"]
    if not pr or not sc1:
        out.update(fullpath_ms=nan, fullpath_gbps=nan)
        return out
    first = min(pr)
    last = max(sc1)
    out["fullpath_ms"] = last - first
    out["fullpath_gbps"] = _div(out["useful"], out["fullpath_ms"] * 1e6)
    h2d = [(e.get("h2d0"), e.get("h2d1")) for e in evs]
    full = all(a is not None and b is not None for a, b in h2d) and bool(h2d)
    h1 = [b for _, b in h2d if b is not None]
    out["tail_ms"] = last - max(h1) if h1 else nan
    if full:
        busy = union_len(h2d)
        span = max(b for _, b in h2d) - min(a for a, _ in h2d)
        out.update(dma_busy_ms=busy, dma_span_ms=span, occupancy=_div(busy, span), active_useful_gbps=_div(out["useful"], busy * 1e6),
                   active_wire_gbps=_div(out["wire"], busy * 1e6))
        out["path_vs_active"] = _div(out["fullpath_gbps"], out["active_useful_gbps"])
        within = boundary = 0.0
        n_within = n_boundary = n_neg = 0
        bgaps = []
        for k in range(1, len(chs)):
            g = h2d[k][0] - h2d[k - 1][1]
            if g < 0:
                n_neg += 1
            g = max(0.0, g)
            if chs[k]["l"] == chs[k - 1]["l"]:
                within += g
                n_within += 1
            else:
                boundary += g
                n_boundary += 1
                bgaps.append(g)
        out.update(startup_ms=h2d[0][0] - first, within_ms=within, boundary_ms=boundary, n_within=n_within, n_boundary=n_boundary,
                   boundary_gap_p50=pctl(bgaps, 50), boundary_gap_p95=pctl(bgaps, 95), n_negative_gaps=n_neg,
                   dma_ms_p50=pctl([b - a for a, b in h2d], 50),
                   scatter_ms_p50=pctl([e["sc1"] - e["sc0"] for e in evs if e.get("sc0") is not None and e.get("sc1") is not None], 50),
                   drain_share=_div(out["tail_ms"], out["fullpath_ms"]))
    # host (same clock only)
    if chs and chs[0].get("p0") is not None:
        cap = int(rec.get("cap") or 0)
        ns = 1e6
        pack_iv = [(x["p0"], x["p1"]) for x in chs if x.get("p0") is not None and x.get("p1") is not None]
        sub_iv = [(x["sub0"], x["sc1h"]) for x in chs if x.get("sub0") is not None and x.get("sc1h") is not None]
        pb, sb = union_len(pack_iv), union_len(sub_iv)
        ov = overlap_len(pack_iv, sub_iv)
        full_ch = [x for x in chs if cap and x["n"] == cap and x.get("p0") is not None]
        fpack = [(x["p1"] - x["p0"]) / ns for x in full_ch]
        out.update(
            pack_busy_ms=pb / ns, submit_busy_ms=sb / ns, overlap_ms=ov / ns, overlap_frac=_div(ov, sb),
            free_slot_wait_ms=sum((x["fw1"] - x["fw0"]) for x in chs if x.get("fw0") is not None and x.get("fw1") is not None) / ns,
            fifo_full_wait_ms=sum((x["pub"] - x["qf0"]) for x in chs if x.get("qf0") is not None and x.get("pub") is not None) / ns,
            starve_ms=sum((x["pop"] - x["iw0"]) for x in chs[1:] if x.get("iw0") is not None and x.get("pop") is not None) / ns,
            first_pop_wait_ms=((chs[0]["pop"] - chs[0]["iw0"]) / ns if chs[0].get("iw0") is not None and chs[0].get("pop") is not None else nan),
            barrier_wait_ms=sum((x["bw1"] - x["bw0"]) for x in lays if x.get("bw0") is not None and x.get("bw1") is not None) / ns,
            list_wait_ms=sum((x["lw1"] - x["lw0"]) for x in lays if x.get("lw0") is not None and x.get("lw1") is not None) / ns,
            desc_ms=sum((x["d1"] - x["d0"]) for x in lays if x.get("d0") is not None and x.get("d1") is not None) / ns,
            depth_pop_p50=pctl([x.get("dpop") for x in chs], 50),
            full_pack_ms_p50=pctl(fpack, 50), n_full_chunks=len(full_ch),
            rawpack_gbps=_div(sum(int(x["useful"]) for x in full_ch), sum(fpack) * 1e6) if fpack else nan,
            submit_ms_p50=pctl([(x["sc1h"] - x["sub0"]) / ns for x in chs if x.get("sub0") is not None and x.get("sc1h") is not None], 50),
            copy_call_ms_p50=pctl([(x["cp1"] - x["cp0"]) / ns for x in chs if x.get("cp0") is not None and x.get("cp1") is not None], 50),
            copy_call_ms_p95=pctl([(x["cp1"] - x["cp0"]) / ns for x in chs if x.get("cp0") is not None and x.get("cp1") is not None], 95),
            copy_ret_to_scatter_ms_p50=pctl([(x["sc0h"] - x["cp1"]) / ns for x in chs if x.get("cp1") is not None and x.get("sc0h") is not None], 50),
            producer_cpu_pack_ms=sum(int(x.get("pcpu") or 0) for x in chs) / ns,
            submitter_cpu_ms=sum(int(x.get("scpu") or 0) for x in chs) / ns)
        pops = [x["pop"] for x in chs if x.get("pop") is not None]
        ends = [x["sc1h"] for x in chs if x.get("sc1h") is not None]
        out["submitter_window_ms"] = (max(ends) - min(pops)) / ns if pops and ends else nan
        out["starve_frac"] = _div(out["starve_ms"], out["submitter_window_ms"])
        cyc = [(b["p0"] - a["p0"]) / ns for a, b in zip(chs, chs[1:]) if a["l"] == b["l"] and cap and a["n"] == cap]
        out["pack_cycle_ms_p50"] = pctl(cyc, 50)
    return out


# ------------------------------------------------------------------------------------------------ blocks
def read_raw(paths: Sequence[str]) -> Tuple[List[Dict], Dict]:
    """(bracket rows, {(tag, gated_step): gate verdict}) from the streamed raw JSONL files (a partial last line, cut by a
    hard stop, is counted and skipped)."""
    rows, gates, bad = [], {}, 0
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                try:
                    x = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if x.get("type") == "gate":
                    gates[(x.get("tag"), x.get("gated_step"))] = x
                elif x.get("type") == "bracket":
                    rows.append(x)
    return rows, dict(gates=gates, truncated_lines=bad)


def score_blocks(rows: Sequence[Dict], gates: Dict) -> List[Dict]:
    """One record per (tag, block kind, block): its 9 rows' metrics, completeness and the reasons it is incomplete."""
    by = {}
    for r in rows:
        if r.get("kind") not in ("full", "light"):
            continue
        key = (r.get("tag"), r.get("batch"), r.get("kind"), r.get("block"))
        by.setdefault(key, []).append(r)
    out = []
    for (tag, B, kind, blk), rs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][2]), kv[0][3])):
        g = gates.get((tag, rs[0].get("gated_step")))
        cells = {(r["arm"], r["phase"]): r for r in rs}
        why = []
        for a in ARMS:
            for ph in PHASES:
                x = cells.get((a, ph))
                if x is None:
                    why.append("missing %s/%s" % (a, ph))
                elif x.get("ok") is not True:
                    why.append("%s/%s not ok" % (a, ph))
        if g is None:
            why.append("no gate record")
        elif not g.get("ok"):
            why.append("GATE_FAIL")
        m = {k: rep_metrics(x) for k, x in cells.items()}
        arms = {}
        for a in ARMS:
            al, res, co = m.get((a, "alone"), {}), m.get((a, "resident"), {}), m.get((a, "conc"), {})
            ext = (co.get("decode_ms") - res.get("decode_ms")) if (co.get("decode_ms") is not None and res.get("decode_ms") is not None) else _nan()
            arms[a] = dict(alone=al, conc=co, resident_ms=res.get("decode_ms"), extra_ms=ext, slowdown=_div(ext, res.get("decode_ms")))
        out.append(dict(tag=tag, batch=B, kind=kind, block=blk, order=rs[0].get("order"), plan_step=rs[0].get("plan_step"),
                        gated_step=rs[0].get("gated_step"), complete=not why, why=why, arms=arms))
    return out


def _paired(blocks, key, num, den, phase="alone"):
    xs = []
    for b in blocks:
        a, c = b["arms"][num][phase].get(key), b["arms"][den][phase].get(key)
        r = _div(a, c)
        if r == r:
            xs.append(r)
    return xs


def targets(blocks: Sequence[Dict]) -> Dict:
    """The four preregistered diagnostic targets (registration section 'Targets'); only COMPLETE blocks."""
    res = {}
    full = lambda B: [b for b in blocks if b["complete"] and b["kind"] == "full" and int(b["batch"]) == B]
    b336, b64 = full(336), full(64)
    r1 = _paired(b336, "fullpath_gbps", "S1", "S0")
    m1 = pctl(r1, 50)
    res["T1"] = dict(name="S1 B336 transfer-alone fullpath delivery / paired S0 >= %.2f" % T1_MIN, n_blocks=len(r1), ratios=r1, median=m1,
                     ratio_of_medians=_div(pctl([b["arms"]["S1"]["alone"].get("fullpath_gbps") for b in b336], 50),
                                           pctl([b["arms"]["S0"]["alone"].get("fullpath_gbps") for b in b336], 50)),
                     verdict=("NOT_SCORED" if not r1 else ("SUPPORTED" if m1 >= T1_MIN else "REFUTED")))
    r2 = _paired(b64, "boundary_ms", "S2", "S1")
    m2 = pctl(r2, 50)
    res["T2"] = dict(name="S2 B64 transfer-alone boundary-gap time / S1 <= %.2f" % T2_MAX, n_blocks=len(r2), ratios=r2, median=m2,
                     undefined_blocks=len(b64) - len(r2),
                     delivery_gbps=dict((a, pctl([b["arms"][a]["alone"].get("fullpath_gbps") for b in b64], 50)) for a in ARMS),
                     verdict=("NOT_SCORED" if not r2 else ("SUPPORTED" if m2 <= T2_MAX else "REFUTED")))
    occ = pctl([b["arms"]["S2"]["alone"].get("occupancy") for b in b336], 50)
    pva = pctl([b["arms"]["S2"]["alone"].get("path_vs_active") for b in b336], 50)
    miss = ([] if occ >= T3_OCC else ["occupancy %.3f < %.2f" % (occ, T3_OCC)]) + ([] if pva >= T3_PATH else ["fullpath / activeDMA %.3f < %.2f" % (pva, T3_PATH)])
    res["T3"] = dict(name="S2 B336: DMA event occupancy >= %.2f AND fullpath >= %.2f x own activeDMA useful rate" % (T3_OCC, T3_PATH),
                     n_blocks=len(b336), occupancy_median=occ, path_vs_active_median=pva,
                     verdict=("NOT_SCORED" if not b336 else ("SHOWN" if not miss else "NOT_SHOWN: preparation not shown almost hidden (%s)" % "; ".join(miss))))
    t4 = {}
    for B in sorted({int(b["batch"]) for b in blocks}):
        light = [b for b in blocks if b["complete"] and b["kind"] == "light" and int(b["batch"]) == B]
        lsteps = {b["plan_step"] for b in light}
        heavy = [b for b in blocks if b["complete"] and b["kind"] == "full" and int(b["batch"]) == B and b["plan_step"] in lsteps]
        for a in ARMS:
            h = pctl([b["arms"][a]["alone"].get("fullpath_gbps") for b in heavy], 50)
            l_ = pctl([b["arms"][a]["alone"].get("fullpath_gbps") for b in light], 50)
            d = _div(h, l_)
            t4["b%d_%s" % (B, a)] = dict(heavy_median=h, light_median=l_, heavy_blocks=len(heavy), light_blocks=len(light), plan_steps=sorted(lsteps),
                                         diff=(d - 1 if d == d else d),
                                         verdict=("NOT_SCORED" if d != d else ("LIGHT_DEFINES_PERFORMANCE (full trace = mechanism only)"
                                                                               if abs(d - 1) > T4_DIFF else "heavy within 5% of light")))
    res["T4"] = dict(name="heavy / light median fullpath difference > %.0f%% -> light defines performance" % (100 * T4_DIFF), cells=t4)
    return res


def diagnose(blocks: Sequence[Dict], B: int, arm: str = "S1", ref: str = "S0") -> Dict:
    """The registered failure-diagnosis rule (labels, not gates; complete full blocks, transfer-alone, medians of the
    per-repetition values; every comparison within one clock)."""
    bs = [b for b in blocks if b["complete"] and b["kind"] == "full" and int(b["batch"]) == B]
    med = lambda a, k: pctl([b["arms"][a]["alone"].get(k) for b in bs], 50)
    ev = dict(full_pack_ms=(med(arm, "full_pack_ms_p50"), med(ref, "full_pack_ms_p50")),
              starve_frac=med(arm, "starve_frac"), depth_pop_p50=med(arm, "depth_pop_p50"),
              submit_ms=(med(arm, "submit_ms_p50"), med(ref, "submit_ms_p50")),
              copy_call_p95=(med(arm, "copy_call_ms_p95"), med(ref, "copy_call_ms_p95")),
              drain_share=med(arm, "drain_share"), scatter_vs_dma=(med(arm, "scatter_ms_p50"), med(arm, "dma_ms_p50")))
    labels = []
    if _div(*ev["full_pack_ms"]) > DIAG["pack_slow"]:
        labels.append("PACKING_SLOWS")
    if ev["starve_frac"] >= DIAG["starve_frac"]:
        labels.append("READY_QUEUE_STARVES")
    if _div(*ev["submit_ms"]) > DIAG["submit_slow"] or _div(*ev["copy_call_p95"]) > DIAG["copy_p95"]:
        labels.append("SUBMISSION_STALLS")
    if ev["drain_share"] >= DIAG["drain_share"] or _div(*ev["scatter_vs_dma"]) > 1.0:
        labels.append("SCATTER_DRAIN_DOMINATES")
    return dict(batch=B, arm=arm, ref=ref, n_blocks=len(bs), labels=labels or ["NONE_OF_THE_REGISTERED_CAUSES"], evidence=ev)


SUMMARY_KEYS = ("fullpath_gbps", "fullpath_ms", "active_useful_gbps", "active_wire_gbps", "occupancy", "path_vs_active", "startup_ms",
                "within_ms", "boundary_ms", "tail_ms", "overlap_frac", "starve_ms", "free_slot_wait_ms", "fifo_full_wait_ms",
                "barrier_wait_ms", "depth_pop_p50", "full_pack_ms_p50", "rawpack_gbps", "submit_ms_p50", "copy_call_ms_p50",
                "copy_ret_to_scatter_ms_p50", "producer_cpu_pack_ms", "submitter_cpu_ms", "rq_wait_ms_transport", "rq_wait_ms_producer",
                "rq_wait_ms_submitter", "rq_wait_ms_helper", "nvcsw_submitter", "nvcsw_producer")


def arm_summary(blocks: Sequence[Dict], B: int, kind: str = "full") -> Dict:
    """Linear p50 / p95 over the complete blocks of one batch (each value is one repetition; no stage medians are added)."""
    bs = [b for b in blocks if b["complete"] and b["kind"] == kind and int(b["batch"]) == B]
    out = {}
    for a in ARMS:
        d = {}
        for k in SUMMARY_KEYS:
            xs = [b["arms"][a]["alone"].get(k) for b in bs]
            d[k] = (pctl(xs, 50), pctl(xs, 95))
        for k in ("decode_ms", "during_gbps", "fullpath_gbps", "during_useful"):
            xs = [b["arms"][a]["conc"].get(k) for b in bs]
            d["conc_" + k] = (pctl(xs, 50), pctl(xs, 95))
        d["resident_ms"] = (pctl([b["arms"][a]["resident_ms"] for b in bs], 50), pctl([b["arms"][a]["resident_ms"] for b in bs], 95))
        d["extra_ms"] = (pctl([b["arms"][a]["extra_ms"] for b in bs], 50), pctl([b["arms"][a]["extra_ms"] for b in bs], 95))
        d["slowdown"] = (pctl([b["arms"][a]["slowdown"] for b in bs], 50), pctl([b["arms"][a]["slowdown"] for b in bs], 95))
        out[a] = d
    return dict(batch=B, kind=kind, n_blocks=len(bs), arms=out)


def score_dir(out_dir: str) -> Dict:
    paths = sorted(glob.glob(os.path.join(out_dir, "*.raw.jsonl")))
    rows, meta = read_raw(paths)
    blocks = score_blocks(rows, meta["gates"])
    batches = sorted({int(b["batch"]) for b in blocks})
    res = dict(paths=paths, rows=len(rows), truncated_lines=meta["truncated_lines"], blocks=blocks, targets=targets(blocks),
               summaries=[arm_summary(blocks, B, k) for B in batches for k in ("full", "light")],
               diagnosis=[diagnose(blocks, B) for B in batches],
               incomplete=[dict(tag=b["tag"], kind=b["kind"], block=b["block"], why=b["why"]) for b in blocks if not b["complete"]])
    return res


def _f(x, fmt="%.3f"):
    return "nan" if x is None or x != x else fmt % x


def render(res: Dict) -> str:
    L = ["# Feeder diagnostic (finite-burst transport / placement-ready in SCRATCH only; NOT sustained live drafting or committed throughput)",
         "", "Raw files: %s; bracket rows %d; truncated lines %d. Only COMPLETE paired blocks are scored; every row is kept in the raw files."
         % (", ".join(os.path.basename(p) for p in res["paths"]), res["rows"], res["truncated_lines"]), ""]
    t = res["targets"]
    L += ["## Preregistered diagnostic targets (not promises)", ""]
    for k in ("T1", "T2", "T3"):
        x = t[k]
        L.append("- %s %s: %s (blocks %s; median %s)" % (k, x["name"], x["verdict"], x.get("n_blocks"),
                                                         _f(x.get("median", x.get("occupancy_median")))))
    L.append("- T4 %s:" % t["T4"]["name"])
    for c, x in sorted(t["T4"]["cells"].items()):
        L.append("  - %s: heavy %s vs light %s GB/s (diff %s; %d / %d blocks) -> %s" % (c, _f(x["heavy_median"]), _f(x["light_median"]), _f(x["diff"]),
                                                                              x["heavy_blocks"], x["light_blocks"], x["verdict"]))
    L += ["", "## Per-batch summaries (linear p50 / p95 over complete blocks; hardware repetitions, not independent workloads)", ""]
    for s in res["summaries"]:
        if not s["n_blocks"]:
            continue
        L.append("### B%d %s blocks (n %d)" % (s["batch"], s["kind"], s["n_blocks"]))
        L.append("| metric | " + " | ".join(ARMS) + " |")
        L.append("|---|" + "---|" * len(ARMS))
        for k in list(SUMMARY_KEYS) + ["conc_decode_ms", "conc_during_gbps", "conc_fullpath_gbps", "resident_ms", "extra_ms", "slowdown"]:
            L.append("| %s | %s |" % (k, " | ".join("%s / %s" % (_f(s["arms"][a][k][0]), _f(s["arms"][a][k][1])) for a in ARMS)))
        L.append("")
    L += ["## Failure diagnosis (registered rule; S1 vs S0)", ""]
    for d in res["diagnosis"]:
        L.append("- B%d: %s (n %d) evidence %s" % (d["batch"], ", ".join(d["labels"]), d["n_blocks"], json.dumps(d["evidence"], default=str)))
    L += ["", "## Incomplete blocks", ""] + ["- %s %s block %s: %s" % (x["tag"], x["kind"], x["block"], "; ".join(x["why"])) for x in res["incomplete"]]
    return "\n".join(L) + "\n"


def write_outputs(out_dir: str, res: Dict) -> Dict:
    md = os.path.join(out_dir, "feeder_summary.md")
    with open(md, "w") as f:
        f.write(render(res))
    bc = os.path.join(out_dir, "feeder_blocks.csv")
    with open(bc, "w", newline="") as f:
        w = csv.writer(f)
        keys = ("fullpath_gbps", "active_useful_gbps", "occupancy", "path_vs_active", "startup_ms", "within_ms", "boundary_ms", "tail_ms", "overlap_frac",
                "starve_ms", "useful", "wire", "chunks")
        w.writerow(("tag", "batch", "kind", "block", "order", "plan_step", "gated_step", "complete", "arm") + tuple("alone_" + k for k in keys)
                   + ("conc_decode_ms", "resident_ms", "extra_ms", "slowdown", "conc_during_useful", "why"))
        for b in res["blocks"]:
            for a in ARMS:
                x = b["arms"][a]
                w.writerow((b["tag"], b["batch"], b["kind"], b["block"], b["order"], b["plan_step"], b["gated_step"], b["complete"], a)
                           + tuple(x["alone"].get(k) for k in keys) + (x["conc"].get("decode_ms"), x["resident_ms"], x["extra_ms"], x["slowdown"],
                                                                       x["conc"].get("during_useful"), ";".join(b["why"])))
    js = os.path.join(out_dir, "feeder_targets.json")
    with open(js, "w") as f:
        json.dump(dict(targets=res["targets"], diagnosis=res["diagnosis"], incomplete=res["incomplete"]), f, indent=1, default=str)
    return dict(md=md, blocks_csv=bc, targets_json=js)


# ------------------------------------------------------------------------------------------------ time budget
# RECORDED (job 2179735, nova06, A100; docs/evidence/cpupack_confirm_2179735/job.log and main/cp_b<B>.json):
RECORDED = dict(
    load_s=dict(b336=89.0, b64=53.0),             # stage wall - runner seconds: 1660 - 1571 (B336), 444 - 391 (B64)
    prefill_s=dict(b336=933.1, b64=183.7),
    conversion_s=dict(b336=36.6, b64=7.2),        # in-place orig -> head-major, reorder + bit-exact verification
    restart_s=dict(b336=6.9, b64=1.3),
    gate_pre_s=dict(b336=0.7, b64=0.2),
    gate_post_s=dict(b336=0.45, b64=0.1),
    main_step_s=dict(b336=19.0, b64=7.0),         # one 2179735 MAIN gated step: 62 brackets incl. checks
)
ESTIMATE_NOTE = "new-code parts are ESTIMATES (no feeder run exists yet); recorded parts are 2179735 measurements"


def rec_value(key: str, B: int) -> float:
    """A recorded value at batch B; a batch without a recording (the B320 fallback) scales the B336 value by B / 336 (the
    load time is not scaled)."""
    tab = RECORDED[key]
    k = "b%d" % B
    if k in tab:
        return tab[k]
    return tab["b336"] if key == "load_s" else tab["b336"] * B / 336.0


def bracket_s(B: int) -> float:
    """Wall seconds of one transfer bracket incl. its checks (2179735 MAIN: 62 brackets in 19.0 s at B336 / 7.0 s at B64 ->
    0.31 / 0.11 s; +25 % for the feeder's stamps, the whole-scratch fingerprint and the raw stream)."""
    return 1.25 * rec_value("main_step_s", B) / 62.0


def batch_budget(B: int, n_full: int = 12, n_light: int = 3, warm_per_arm: int = 3, first: bool = True) -> Dict:
    """The per-batch time arithmetic (seconds) of the registered schedule. Recorded parts from RECORDED; estimates marked."""
    gate = rec_value("gate_pre_s", B) + rec_value("gate_post_s", B) + 0.3   # + reference / resident / advance decodes
    n_gated = 1 + n_full + n_light
    br = bracket_s(B)
    parts = dict(
        load=rec_value("load_s", B) if first else 5.0,                      # the model is loaded once per process
        prefill=rec_value("prefill_s", B),
        conversion=rec_value("conversion_s", B),
        snapshot_and_plans=10.0 if B >= 300 else 4.0,                       # estimate: hosted snapshot + plan sha / load
        golden_pass=n_gated * gate + 4 * 0.3,                               # estimate from the recorded gate costs
        restart=rec_value("restart_s", B),
        correct=(3 * 32 + 3 + 12) * (br * 0.6) + 4 * 0.5,                   # estimate: 96 single-layer + 3 full + 12 synthetic
        warmups=3 * warm_per_arm * br,
        blocks=(n_full + n_light) * (9 * br + gate),                        # 9 brackets per block + its gated step
        teardown_and_flush=15.0,
    )
    parts["total"] = sum(parts.values())
    return dict(batch=B, seconds=parts, estimate_note=ESTIMATE_NOTE)


def job_budget(batches=(336, 64), controls_s: float = 60.0, env_s: float = 20.0, score_s: float = 30.0, cap_min: float = 30.0, **kw) -> Dict:
    per = [batch_budget(B, first=(i == 0), **kw) for i, B in enumerate(batches)]
    tot = env_s + controls_s + score_s + sum(p["seconds"]["total"] for p in per)
    fallback = batch_budget(320, first=False, **kw)["seconds"]["total"] if 336 in batches else 0.0
    b336_before_capacity_stop = (RECORDED["load_s"]["b336"] + RECORDED["prefill_s"]["b336"] + RECORDED["conversion_s"]["b336"] + 10.0
                                 if 336 in batches else 0.0)
    return dict(batches=list(batches), per_batch=per, env_s=env_s, controls_s=controls_s, score_s=score_s, total_s=tot, total_min=tot / 60.0,
                cap_min=cap_min, margin_min=cap_min - tot / 60.0, fits=tot / 60.0 <= cap_min,
                b320_fallback_s=fallback,
                fallback_after_stop_min=(env_s + controls_s + b336_before_capacity_stop + fallback + score_s) / 60.0)


def fits(now_s: float, est_s: float, deadline_s: Optional[float], reserve_s: float) -> bool:
    """True when work of est_s started at now_s ends reserve_s before the deadline (no deadline = always)."""
    return deadline_s is None or now_s + est_s + reserve_s <= deadline_s


def block_estimate(observed: Sequence[float], default: float) -> float:
    return max(observed) if observed else float(default)


def predict_batch_s(B: int, per_req_s: float, fixed_s: float) -> float:
    return fixed_s + per_req_s * B
