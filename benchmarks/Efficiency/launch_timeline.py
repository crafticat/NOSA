"""Launch-correlated timeline of GATED decode steps (stage T of the Strata / timeline experiment). Pure python
(sqlite3), CPU-tested (retroinfer-eval tests/test_strata_transfer.py). No GPU.

THE QUESTION (Codex review 2026-09-24): the GPU-sleep-gated timer measures t0 -> tm on the main stream. Its device time
is the step's GPU time ONLY IF every kernel of the step was already submitted when the main stream became free for it.
A NOSI step issues 1061-1220 stream entries and the per-stream launch queue holds ~1022 (observed once), so the host is
still submitting when the gate opens; a total-time rule ('enqueue tail < 0.5 x device time') cannot exclude an idle gap
early in the step that a late heavy kernel hides. This module measures the gaps directly from CUDA API launch records
correlated (correlationId) to GPU activity records.

INPUT: the SQLite export(s) of `nsys profile --trace=cuda,nvtx [--cuda-graph-trace=node] --capture-range=cudaProfilerApi`
of worker_sweep.py with WS_T_ARMS (NVTX 'lt|b<B>|<arm>|step<it>|<phase>|rep<k>' around each bracket and
'lt_submit|<same label>' around the step's own submission) or of hisparse_repro.py with HR_NVTX=1 ('gf|...' ranges, no
inner range: every main-stream op after the gate kernel is a step op).
PER REP (analyse_rep):
  gate      the bracket's GPU sleep kernel (name contains 'sleep' or 'spin') inside the range; main stream = its stream
  step ops  GPU ops on the main stream whose launch API call lies inside 'lt_submit|' (else: after the gate kernel)
  side ops  GPU ops on any other stream whose launch API call lies inside the outer range
  ready_i   END of op i's launch API call (upper bound: the entry exists once the call returns); graph nodes use their
            cudaGraphLaunch's record. starved_lb uses the call's START (lower bound).
  metric    ops sorted by GPU start; prev_end_0 = the gate kernel's END; prev_end_i = max end of the earlier step ops;
            starved_i = max(0, min(start_i, ready_i) - prev_end_i) if ready_i > prev_end_i else 0
            (the main stream was idle because op i had not been submitted). other_idle_i = gap_i - starved_i.
  totals    starved_ms (upper bound), starved_lb_ms, n_starved, max_gap_us, span_ms = first step op start -> last step op end,
            starved_frac = starved / span, main_busy_frac, device_busy_frac, min_backlog_us = min_i (prev_end_i - ready_i)
            (< 0 = starvation), queued_at_gate = step submissions whose call ended before the gate kernel ended, and
            busy_queued_ms = GPU busy time of those ops (the prefix the self-reporting rule needs), submit_end_after_gate_ms =
            last step submission end - gate end (compare with the harness's submit_end_ms).
  grid      side kernels of known families: gridX / blockX against the expected launch (EXPECTED_GRID).
PASS RULE (registered before the run; evaluate_cells):
  a cell (batch, arm) PASSES when EVERY traced non-warm-up rep has starved_ms <= 0.005 x span_ms AND the traced median
  step time (harness main_ms) is within 2% of the untraced median of the same process (else INCONCLUSIVE); the negative
  control (arm 'alone_late') must FAIL the 0.5% rule in every traced rep (DETECTED), otherwise the metric is not trusted
  (NOT_DETECTED). A FAILED cell's gated device times are invalid. No row is rescued by a threshold on totals.
    python launch_timeline.py OUT_PREFIX run1.sqlite [run2.sqlite ...] [--harness sweep_b64.json ...]
"""
import csv
import glob
import json
import os
import sqlite3
import statistics
import sys
from typing import Dict, List, Optional, Tuple

PASS_FRAC = 0.005
REPR_TOL = 0.02
GATE_PATTERNS = ("sleep", "spin")
SIDE_FAMILIES = {                       # kernel-name fragment -> family (for the grid check)
    "transfer_kernel_impl": "strata",
    "hicache_transfer_per_layer": "hicachejit",
    "copy_cache_planned": "hisparse",
    "flash_h2d_persistent": "workers",
}
NEG_CONTROL_ARMS = ("alone_late",)


# --------------------------------------------------------------------------- pure core
def union_len(iv: List[Tuple[float, float]]) -> float:
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


def starvation(ops: List[Dict], gate_end: Optional[float]) -> Tuple[List[Dict], Dict]:
    """ops: dicts with start, end (GPU), api_start, api_end (host API call, the same clock). Returns (per-op rows in GPU
    order, totals). gate_end = end of the gate kernel (None: the first op is not assessed)."""
    ops = sorted(ops, key=lambda o: (o["start"], o["end"]))
    prev = gate_end
    rows, tot = [], dict(starved=0.0, starved_lb=0.0, gap=0.0, n_starved=0, max_starved=0.0, max_gap=0.0, min_backlog=None)
    for i, o in enumerate(ops):
        if prev is None:
            r = dict(o, prev_end=None, gap=None, starved=None, starved_lb=None, other_idle=None, backlog=None)
        else:
            gap = max(0.0, o["start"] - prev)
            sv = max(0.0, min(o["start"], o["api_end"]) - prev) if o["api_end"] > prev else 0.0
            sl = max(0.0, min(o["start"], o["api_start"]) - prev) if o["api_start"] > prev else 0.0
            backlog = prev - o["api_end"]
            r = dict(o, prev_end=prev, gap=gap, starved=sv, starved_lb=sl, other_idle=gap - sv, backlog=backlog)
            tot["starved"] += sv
            tot["starved_lb"] += sl
            tot["gap"] += gap
            tot["n_starved"] += int(sv > 0)
            tot["max_starved"] = max(tot["max_starved"], sv)
            tot["max_gap"] = max(tot["max_gap"], gap)
            tot["min_backlog"] = backlog if tot["min_backlog"] is None else min(tot["min_backlog"], backlog)
        rows.append(r)
        prev = o["end"] if prev is None else max(prev, o["end"])
    return rows, tot


def rep_summary(step_ops: List[Dict], side_ops: List[Dict], gate_end: Optional[float], unit: float = 1e-6) -> Dict:
    """unit = ms per clock tick of the inputs (nsys: ns -> 1e-6 ms)."""
    rows, t = starvation(step_ops, gate_end)
    if not rows:
        return dict(n_step_ops=0, rows=[])
    span0, span1 = rows[0]["start"], max(o["end"] for o in rows)
    span = span1 - span0
    busy = union_len([(o["start"], o["end"]) for o in rows])
    alls = [(max(o["start"], span0), min(o["end"], span1)) for o in list(step_ops) + list(side_ops) if o["end"] > span0 and o["start"] < span1]
    queued = [o for o in rows if gate_end is not None and o["api_end"] <= gate_end]
    last_submit = max(o["api_end"] for o in rows)
    s = dict(n_step_ops=len(rows), n_side_ops=len(side_ops), n_submissions=len({o.get("corr") for o in rows}),
             span_ms=span * unit, starved_ms=t["starved"] * unit, starved_lb_ms=t["starved_lb"] * unit, gap_ms=t["gap"] * unit,
             starved_frac=(t["starved"] / span if span > 0 else float("nan")), n_starved=t["n_starved"],
             max_starved_us=t["max_starved"] * unit * 1e3, max_gap_us=t["max_gap"] * unit * 1e3,
             min_backlog_us=(t["min_backlog"] * unit * 1e3 if t["min_backlog"] is not None else None),
             main_busy_frac=(busy / span if span > 0 else float("nan")), device_busy_frac=(union_len(alls) / span if span > 0 else float("nan")),
             head_ms=((span0 - gate_end) * unit if gate_end is not None else None),
             queued_at_gate=len({o.get("corr") for o in queued}), busy_queued_ms=union_len([(o["start"], o["end"]) for o in queued]) * unit,
             submit_end_after_gate_ms=((last_submit - gate_end) * unit if gate_end is not None else None),
             side_busy_in_span_ms=union_len([(max(o["start"], span0), min(o["end"], span1)) for o in side_ops
                                             if o["end"] > span0 and o["start"] < span1]) * unit,
             passes_rule=bool(span > 0 and t["starved"] <= PASS_FRAC * span))
    s["rows"] = rows
    return s


def parse_label(text: str) -> Dict:
    """'lt|b64|strata8_t1024_i256|step4|traced|rep1' or 'gf|<arm>|<mode>|<regime>|<rep>'."""
    p = text.split("|")
    if p[0] == "lt" and len(p) >= 6:
        return dict(kind="lt", batch=int(p[1][1:]), arm=p[2], step=int(p[3][4:]), phase=p[4], rep=int(p[5][3:]))
    if p[0] == "gf" and len(p) >= 5:
        return dict(kind="gf", batch=None, arm=p[1], mode=p[2], regime=p[3], step=None, phase="traced", rep=int(p[4]))
    raise ValueError("not a timeline label: %r" % text)


def expected_grid(arm: str, harness: Optional[Dict]) -> Optional[Dict]:
    """(family, grid, block) the side kernel of `arm` must launch with, from the harness JSON (strata / jit grids are the
    upstream formulas recorded per plan), or None when unknown."""
    import re
    eg = (harness or {}).get("expected_grids") or {}
    if arm in eg:                                       # stage N0: hisparse_repro's meta.expected_grids (upstream formulas)
        e = eg[arm]
        fam = {"transfer_kernel_impl": "strata", "hicache_transfer_per_layer": "hicachejit"}.get(e.get("kernel"), e.get("family"))
        return dict(family=fam, grid=int(e["grid"]), block=int(e["block"]))
    m = re.match(r"(strata|hicachejit|hisparse)(\d+)(?:_t(\d+))?(?:_(i\d+k?))?", arm)
    if not m:
        return None
    fam, W = m.group(1), int(m.group(2))
    if fam == "hisparse":
        return dict(family=fam, grid=W, block=int(m.group(3) or 1024))
    item = m.group(4)
    for pl in ((harness or {}).get("strata") or {}).get("plans", []):
        if pl.get("item") == item and "refused" not in pl:
            g = (pl.get("aot_grid") if fam == "strata" else pl.get("jit_grid")) or {}
            val = g.get(str(W), g.get(W))
            if val is not None:
                return dict(family=fam, grid=int(val), block=1024)
    return None


def grid_check(side_ops: List[Dict], exp: Optional[Dict]) -> Optional[Dict]:
    if exp is None:
        return None
    ks = [o for o in side_ops if o.get("family") == exp["family"]]
    bad = [(o.get("gridX"), o.get("blockX")) for o in ks if (o.get("gridX"), o.get("blockX")) != (exp["grid"], exp["block"])]
    return dict(expected=exp, n_kernels=len(ks), n_mismatch=len(bad), mismatches=bad[:5], ok=bool(ks) and not bad)


def _median(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.median(xs) if xs else float("nan")


def evaluate_cells(reps: List[Dict], harness_records: List[Dict]) -> List[Dict]:
    """reps: rep summaries with label fields (batch, arm, step, phase, rep). harness_records: worker_sweep 'timeline'
    records (main_ms per label). Returns one verdict per (batch, arm) under the registered PASS RULE."""
    hm = {r["label"]: r for r in harness_records if "label" in r}
    cells: Dict[Tuple, Dict] = {}
    for r in reps:
        if r.get("phase") != "traced" or r.get("warmup"):
            continue
        c = cells.setdefault((r.get("batch"), r["arm"]), dict(batch=r.get("batch"), arm=r["arm"], reps=[]))
        c["reps"].append(r)
    out = []
    for (batch, arm), c in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        rs = c["reps"]
        traced = [hm[r["label"]]["main_ms"] for r in rs if r.get("label") in hm and "main_ms" in hm[r["label"]]]
        untraced = [h["main_ms"] for h in harness_records if h.get("batch") == batch and h.get("arm") == arm
                    and str(h.get("phase", "")).startswith("untraced") and "main_ms" in h]
        all_pass = bool(rs) and all(r["passes_rule"] for r in rs)
        none_pass = bool(rs) and not any(r["passes_rule"] for r in rs)
        repr_ratio = (_median(traced) / _median(untraced) - 1.0) if traced and untraced else float("nan")
        repr_ok = repr_ratio == repr_ratio and abs(repr_ratio) <= REPR_TOL
        if arm in NEG_CONTROL_ARMS:
            verdict = "DETECTED" if none_pass else "NOT_DETECTED"
        elif not all_pass:
            verdict = "FAIL"
        elif not repr_ok:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "PASS"
        out.append(dict(batch=batch, arm=arm, n_traced=len(rs), verdict=verdict,
                        max_starved_frac=max(r["starved_frac"] for r in rs) if rs else None,
                        max_starved_ms=max(r["starved_ms"] for r in rs) if rs else None,
                        max_gap_us=max(r["max_gap_us"] for r in rs) if rs else None,
                        min_backlog_us=min((r["min_backlog_us"] for r in rs if r.get("min_backlog_us") is not None), default=None),
                        traced_median_ms=_median(traced), untraced_median_ms=_median(untraced), traced_vs_untraced=repr_ratio,
                        n_untraced=len(untraced), representative=repr_ok,
                        grid_ok=(all(r["grid"]["ok"] for r in rs if r.get("grid")) if any(r.get("grid") for r in rs) else None)))
    return out


# --------------------------------------------------------------------------- sqlite
def _tables(con) -> set:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(con, t) -> set:
    return {r[1] for r in con.execute("PRAGMA table_info(%s)" % t)}


def load_nsys(path: str) -> Dict:
    """-> dict(api=[...], gpu=[...], ranges=[(start, end, text)]). Times in ns. API records from
    CUPTI_ACTIVITY_KIND_RUNTIME (nsys also stores driver calls such as cuLaunchKernel there) and, if present,
    CUPTI_ACTIVITY_KIND_DRIVER."""
    con = sqlite3.connect(path)
    tabs = _tables(con)
    S = dict(con.execute("SELECT id, value FROM StringIds")) if "StringIds" in tabs else {}
    api = []
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t in tabs:
            for s, e, c, n in con.execute("SELECT start, end, correlationId, nameId FROM %s" % t):
                api.append(dict(start=int(s), end=int(e), corr=int(c) if c is not None else None, name=S.get(n, str(n))))
    gpu = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
        cols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
        nm = "demangledName" if "demangledName" in cols else "shortName"
        gx = "gridX" if "gridX" in cols else "NULL"
        bx = "blockX" if "blockX" in cols else "NULL"
        for s, e, st, c, n, g, b in con.execute("SELECT start, end, streamId, correlationId, %s, %s, %s FROM CUPTI_ACTIVITY_KIND_KERNEL" % (nm, gx, bx)):
            gpu.append(dict(kind="kernel", start=int(s), end=int(e), stream=st, corr=c, name=S.get(n, str(n)), gridX=g, blockX=b))
    for t, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if t in tabs:
            for s, e, st, c in con.execute("SELECT start, end, streamId, correlationId FROM %s" % t):
                gpu.append(dict(kind=kind, start=int(s), end=int(e), stream=st, corr=c, name=kind, gridX=None, blockX=None))
    ranges = []
    if "NVTX_EVENTS" in tabs:
        cols = _cols(con, "NVTX_EVENTS")
        tid = "textId" if "textId" in cols else "NULL"
        for s, e, text, text_id in con.execute("SELECT start, end, text, %s FROM NVTX_EVENTS WHERE end IS NOT NULL" % tid):
            t = text if text is not None else S.get(text_id)
            if t and (t.startswith("lt|") or t.startswith("lt_submit|") or t.startswith("gf|")):
                ranges.append((int(s), int(e), t))
    con.close()
    return dict(api=api, gpu=gpu, ranges=sorted(ranges))


def family_of(name: str) -> Optional[str]:
    n = (name or "")
    for frag, fam in SIDE_FAMILIES.items():
        if frag in n:
            return fam
    return None


def analyse_trace(data: Dict, harness: Optional[Dict] = None) -> List[Dict]:
    api_by_corr = {a["corr"]: a for a in data["api"] if a["corr"] is not None}
    inner = {t[len("lt_submit|"):]: (s, e) for s, e, t in data["ranges"] if t.startswith("lt_submit|")}
    out = []
    for s0, e0, text in data["ranges"]:
        if text.startswith("lt_submit|"):
            continue
        lab = parse_label(text)
        ops = []
        for g in data["gpu"]:
            a = api_by_corr.get(g["corr"])
            if a is None or not (s0 <= a["start"] <= e0):
                continue
            ops.append(dict(g, api_start=a["start"], api_end=a["end"], api=a["name"], family=family_of(g["name"])))
        gates = [o for o in ops if o["kind"] == "kernel" and any(p in (o["name"] or "").lower() for p in GATE_PATTERNS)]
        if not gates:
            out.append(dict(label=text, **lab, error="no gate kernel in the range"))
            continue
        gate = min(gates, key=lambda o: o["start"])
        ms = gate["stream"]
        rest = [o for o in ops if o is not gate and not any(p in (o["name"] or "").lower() for p in GATE_PATTERNS)]
        win = inner.get(text)
        if win is not None:
            step = [o for o in rest if o["stream"] == ms and win[0] <= o["api_start"] <= win[1]]
        else:
            step = [o for o in rest if o["stream"] == ms and o["start"] >= gate["end"]]
        side = [o for o in rest if o["stream"] != ms]
        summ = rep_summary(step, side, gate["end"])
        rows = summ.pop("rows")
        rec = dict(label=text, **lab, main_stream=ms, gate_end_ns=gate["end"], submit_range=win is not None, **summ)
        rec["warmup"] = bool(lab.get("phase") == "traced" and lab.get("rep") == 0 and lab.get("kind") == "lt")
        rec["grid"] = grid_check(side, expected_grid(lab["arm"], harness))
        h = {r.get("label"): r for r in ((harness or {}).get("timeline") or [])}.get(text)
        if h is not None:
            rec["harness_main_ms"] = h.get("main_ms")
            rec["harness_submit_end_ms"] = h.get("submit_end_ms")
            rec["harness_gate_done_at_submit_end"] = h.get("gate_done_at_submit_end")
            rec["harness_host_submit_ms"] = h.get("host_submit_ms")
        rec["_rows"] = rows
        out.append(rec)
    return out


ROW_FIELDS = ["label", "role", "idx", "stream", "kind", "name", "corr", "api", "api_start_us", "api_end_us", "gpu_start_us", "gpu_end_us",
              "gpu_dur_us", "prev_end_us", "gap_us", "starved_us", "starved_lb_us", "other_idle_us", "backlog_us", "gridX", "blockX"]


def write_rows(path: str, rec: Dict) -> None:
    t0 = rec["gate_end_ns"]
    us = lambda x: "" if x is None else "%.3f" % ((x - t0) / 1e3)
    d = lambda x: "" if x is None else "%.3f" % (x / 1e3)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for i, o in enumerate(rec["_rows"]):
            w.writerow(dict(label=rec["label"], role="step", idx=i, stream=o["stream"], kind=o["kind"], name=(o["name"] or "")[:120], corr=o["corr"],
                            api=o["api"], api_start_us=us(o["api_start"]), api_end_us=us(o["api_end"]), gpu_start_us=us(o["start"]),
                            gpu_end_us=us(o["end"]), gpu_dur_us=d(o["end"] - o["start"]), prev_end_us=us(o.get("prev_end")), gap_us=d(o.get("gap")),
                            starved_us=d(o.get("starved")), starved_lb_us=d(o.get("starved_lb")), other_idle_us=d(o.get("other_idle")),
                            backlog_us=d(o.get("backlog")), gridX=o.get("gridX"), blockX=o.get("blockX")))


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out = argv[1]
    args = argv[2:]
    harness_paths = args[args.index("--harness") + 1:] if "--harness" in args else []
    sqls = args[:args.index("--harness")] if "--harness" in args else args
    sqls = [p for pat in sqls for p in sorted(glob.glob(pat))]
    harness = {}
    for hp in harness_paths:
        h = json.load(open(hp))
        harness.setdefault("timeline", []).extend(h.get("timeline") or [])
        for k in ("strata",):
            if h.get(k) and not harness.get(k):
                harness[k] = h[k]
        harness.setdefault("expected_grids", {}).update(((h.get("meta") or {}).get("expected_grids")) or {})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    reps = []
    for sp in sqls:
        for rec in analyse_trace(load_nsys(sp), harness):
            rec["sqlite"] = sp
            if "_rows" in rec:
                safe = rec["label"].replace("|", "_")
                write_rows("%s.%s.csv" % (out, safe), rec)
                del rec["_rows"]
            reps.append(rec)
    cells = evaluate_cells([r for r in reps if "error" not in r], harness.get("timeline") or [])
    json.dump(dict(pass_frac=PASS_FRAC, repr_tol=REPR_TOL, reps=reps, cells=cells), open(out + ".json", "w"), indent=1, default=str)
    keys = ["label", "batch", "arm", "step", "phase", "rep", "warmup", "n_step_ops", "n_submissions", "span_ms", "starved_ms", "starved_lb_ms",
            "starved_frac", "n_starved", "max_starved_us", "max_gap_us", "min_backlog_us", "main_busy_frac", "device_busy_frac", "head_ms",
            "queued_at_gate", "busy_queued_ms", "submit_end_after_gate_ms", "harness_submit_end_ms", "harness_main_ms", "harness_host_submit_ms",
            "side_busy_in_span_ms", "passes_rule", "error", "sqlite"]
    with open(out + ".reps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in reps:
            w.writerow(r)
    L = ["# Launch-correlated timeline of gated steps (stage T)", "",
         "PASS RULE (registered): every traced non-warm-up rep starved <= %.1f%% of the step span AND traced median step within %.0f%% of "
         "the untraced median (else INCONCLUSIVE); the negative control 'alone_late' must fail the rule in every traced rep (DETECTED)."
         % (100 * PASS_FRAC, 100 * REPR_TOL), "",
         "| batch | arm | traced reps | verdict | max starved % of span | max starved ms | largest gap us | min backlog us | traced / untraced median ms | grid ok |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        L.append("| %s | %s | %d | %s | %.3f%% | %.3f | %.1f | %s | %.2f / %.2f (%+.1f%%) | %s |" % (
            c["batch"], c["arm"], c["n_traced"], c["verdict"], 100 * (c["max_starved_frac"] or 0), c["max_starved_ms"] or 0, c["max_gap_us"] or 0,
            "-" if c["min_backlog_us"] is None else "%.0f" % c["min_backlog_us"], c["traced_median_ms"], c["untraced_median_ms"],
            100 * c["traced_vs_untraced"] if c["traced_vs_untraced"] == c["traced_vs_untraced"] else float("nan"), c["grid_ok"]))
    text = "\n".join(L) + "\n"
    open(out + ".md", "w").write(text)
    print(text)
    return 0 if reps else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
