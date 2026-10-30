"""Cell family I4C of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): THE PROFILER
WINDOW of the current best arm L4PTFD (rung I4A, lever 1: L4PTF with the lean one-row pass, benchmarks/Efficiency/
r2_i4a_lean.py). Lever "L3" of KS/analysis/plan/LEDGER_model_vs_measured.md revision 5 (R5.4-R5.6); the levers L5 (the
union lists, the chain, the writes) and L7 (the probe-vs-steady gap) share the window, and lever L1 (the draft phase)
gets its marks. MEASUREMENT ONLY: no arm's computation changes. NO NEW KERNEL: torch.profiler, CUDA events and
record_function ranges only. Docs: docs/NOTES_i4c.md. Step script: benchmarks/Efficiency/r2_i4c_step.sh (cells i4cm,
i4ca, i4cb). A NEW script: no older cell is edited (r2_i4a_cell.py, r2_i3_cell.py, r2_wsa_cell.py and r2_p3_profile.py
are imported unchanged).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in this order, each with its own shipped prefill into the chain arena,
on the inputs of i4aa / i4ab (the same workload pin, S0, document rows, K 3, 512 steps, fused twin and body, graph,
redirect, probe 16, store X 129, --tu-gemm natural):
  SH       the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTFD   I4A's arm as it ran in i4aa / i4ab (run_arm_c = r2_i4a_cell.run_arm_x's steps with lean True): THE
           REFERENCE. Its steady medians are the denominators of every share below.
  L4PTFDP  the same arm with a Probe on the runner (instance attributes only; the class code is not changed):
           MARKED ROUNDS (the probe rounds that are not profiled): the joint pass records CUDA events at every phase
             boundary (L4PTF's own marks "resid" .. "post" plus "embed", "pre", "twin", "select", "head", "end": full
             coverage from the pass start to its end), and each D-slot draft pass (R = 1) records "embed", per layer
             "pre", "twin", "select", "resid" (the lean row's start), "write", "amask" (after the write: the mask and
             the scratch bias), "attnA" (call A), "bmask" (after call A), "attnB" (call B), "merge" (after call B),
             "post" (after the row: wo, MLP), then "head", "end".
           PROFILED ROUNDS (--profile-rounds, probe rounds): the joint pass and each D-slot pass run under
             torch.profiler, ONE window per pass (r2_i1p_cell.profile_pass, copied unchanged), with the marks OFF,
             TentState.checking OFF and the meter's timing OFF (the steady code path). In the LABELLED rounds
             (--label-rounds) the same labels open record_function ranges ("i4cL.<label>@<layer>"), so every GPU op
             of the pass is attributed to its label (L5: "union", "chain", "write"; L7: the lead and the tail).
REPORTED (REPORT; the predictions PRED_I4C_BY_B): device busy / steady joint median (the host exposure share = 1 -
busy / steady), also against the local steady median (+-16 rounds); the kernel classes of a profiled steady joint
pass (r2_p3_profile classes: count, device ms; SMALL + COPY vs large; duration buckets); the kernels per label (L5);
the L7 split (steady joint - probe mark sum = the uncovered labels + the J slot's plan - the probe overhead); the
draft marks per label and the draft phase outside the two passes.
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN (r2_wsa_cell.arm_gates_w), -RESID,
-ATTN1, -FILL1 (r2_i3_cell.fork_gates), -LEAN (r2_i4a_cell.lean_gate); L4PTFDP-PEQ (the profiled arm's committed
tokens, check-round fingerprints, acceptance record and state checksums = the reference arm's, bit for bit:
r2_i4a_cell.eq_gate; MEASURED cross-arm determinism in i4aa / i4ab), L4PTFDP-MARKS-NEUTRAL, -MARKS-SHAPE, -PHASE-SUM.
REPORT: MEM, -PROF, -PROF-NEUTRAL. REGISTERED PREDICTIONS PRED_I4C_BY_B: every value None until the lead registers
them with PRED_STAMP_I4C; the step script STOPS the timing cells (i4ca, i4cb) while PRED_STAMP_I4C is None.
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_i4a_cell as I4A         # noqa: E402
import r2_i4a_lean as LN          # noqa: E402
import r2_p3_profile as P3        # noqa: E402  (parse_trace, aggregate, LAUNCH_CATS: unchanged)

I3C = I4A.I3C
WA, SC, BC = I4A.WA, I4A.SC, I4A.BC
ARMS = ("L4PTFD", "L4PTFDP")
REF, PROBED = ARMS
TOPK = I4A.TOPK
FORK_BLOB = I4A.FORK_BLOB
STRONGEST = I4A.STRONGEST
BUDGET_GB = I4A.BUDGET_GB
USABLE_GB = I4A.USABLE_GB
RANGE = "i4c.joint"                                 # the profiled joint pass's record_function range
RANGE_PREFIX = "i4c."                               # the pass ranges: i4c.joint, i4c.draft0, i4c.draft1
LABEL_PREFIX = "i4cL."                              # the label ranges of a labelled pass (NOT under RANGE_PREFIX)
JOINT = WA.PHASES_W["L4P"].index("joint")
DRAFT = WA.PHASES_W["L4P"].index("draft")
JOINT_LAYER = ("pre", "twin", "select", "resid", "write", "chain", "union", "fetch", "attn", "check", "post")
DRAFT_LAYER = ("pre", "twin", "select", "resid", "write", "amask", "attnA", "bmask", "attnB", "merge", "post")
DRAFT_ROW = ("resid", "write", "amask", "attnA", "bmask", "attnB", "merge")    # the lean row (I1P's _row labels)
L5_LABELS = ("union", "chain", "write")             # LEDGER R5.5 item 5 (the probe labels 14.97 + 3.85 + 11.15 ms)
LEAD = ("embed", "pre", "twin", "select")           # layer 0 (with "embed"): before L4PTF's first mark
SMALL_CLASSES = ("SMALL", "COPY")
SIZE_EDGES_US = (5.0, 20.0, 100.0)
SIZE_NAMES = ("lt5us", "5to20us", "20to100us", "ge100us")
PHASE_SUM_TOL = (2.0, 0.01)                         # I1P's: (ms, fraction of the probe joint)
LOCAL_W = 16                                        # the local steady median: steady rounds within +-16 rounds
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4C_BY_B (per batch) and PRED_STAMP_I4C
# (`date -u`, L33) in a commit BEFORE i4ca / i4cb (the step script stops while PRED_STAMP_I4C is None). Keys, each
# (low, high) unless noted; medians over the profiled / marked passes; REF = the L4PTFD arm of the same job:
#   I4C_busy_frac        profiled joint pass device busy ms / REF steady joint median ms (1 - it = the host share)
#   I4C_draft_busy_frac  (draft0 + draft1 busy ms of a profiled round) / REF steady draft phase median ms
#   I4C_joint_gpu_ops    GPU ops (kernels + copies) in one profiled joint pass
#   I4C_joint_small_ms   device ms of the SMALL + COPY classes (r2_p3_profile) in one profiled joint pass
#   I4C_L5_device_frac   device ms of the kernels in the labels union + chain + write (labelled joint pass) / the same
#                        labels' elapsed ms (the marked probe passes of L4PTFDP)
#   I4C_L7_gap_ms        REF steady joint median - REF probe mark sum (I3S: 5.312 / 5.789)
#   I4C_L7_uncovered_ms  the time that L4PTF's marks do not cover: lead (embed + layer 0 pre / twin / select) + tail
#                        (the last layer's post + head)
#   I4C_L7_residual_ms   I4C_L7_gap - (uncovered + the J slot's plan - the probe overhead)
#   I4C_draft_row_ms     one marked draft pass: resid + write + amask + attnA + bmask + attnB + merge (32 layers)
#   I4C_draft_outside_ms one marked round: the draft phase - the two marked draft passes (embed .. end)
#   I4C_prof_neutral     L4PTFDP / REF steady joint median (the probe off the steady rounds: about 1)
#   I4C_ref_ratio        REF median steady committed tok/s / SH's (I4A re-measured: 0.5207 / 0.5980)
#   I4C_peq              True: L4PTFDP-PEQ passes (a flag; the gate is MUST anyway)
# The builder's INFERENCE (docs/NOTES_i4c.md section 5) is an input for the lead, not a registration.
PRED_STAMP_I4C = "2026-10-29T08:57:45Z"
PRED_KEYS = ("I4C_busy_frac", "I4C_draft_busy_frac", "I4C_joint_gpu_ops", "I4C_joint_small_ms", "I4C_L5_device_frac",
             "I4C_L7_gap_ms", "I4C_L7_uncovered_ms", "I4C_L7_residual_ms", "I4C_draft_row_ms", "I4C_draft_outside_ms",
             "I4C_prof_neutral", "I4C_ref_ratio", "I4C_peq")
# The lead's registration (before any I4C cell ran). DECISION RULE: the CUDA-graph trial over the per-layer union +
# fork section (NOTES_i4c section 7) is built only if 1 - I4C_busy_frac > 0.05 at either batch (MEASURED, not band).
PRED_I4C_BY_B = {
    216: dict(I4C_busy_frac=(0.90, 0.98), I4C_draft_busy_frac=(0.80, 0.98), I4C_joint_gpu_ops=(5000, 11000),
              I4C_joint_small_ms=(20.0, 75.0), I4C_L5_device_frac=(0.30, 0.90), I4C_L7_gap_ms=(4.0, 7.5),
              I4C_L7_uncovered_ms=(5.0, 10.0), I4C_L7_residual_ms=(-2.0, 2.0), I4C_draft_row_ms=(30.0, 60.0),
              I4C_draft_outside_ms=(0.5, 8.0), I4C_prof_neutral=(0.98, 1.02), I4C_ref_ratio=(0.50, 0.54),
              I4C_peq=True),
    318: dict(I4C_busy_frac=(0.90, 0.98), I4C_draft_busy_frac=(0.80, 0.99), I4C_joint_gpu_ops=(5000, 11000),
              I4C_joint_small_ms=(25.0, 90.0), I4C_L5_device_frac=(0.30, 0.90), I4C_L7_gap_ms=(4.0, 8.0),
              I4C_L7_uncovered_ms=(5.0, 12.0), I4C_L7_residual_ms=(-2.0, 2.0), I4C_draft_row_ms=(25.0, 75.0),
              I4C_draft_outside_ms=(0.5, 8.0), I4C_prof_neutral=(0.98, 1.02), I4C_ref_ratio=(0.58, 0.62),
              I4C_peq=True)}


def log(msg):
    print("[r2_i4c %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDP", help="a subset of %s, in this order (SH runs first)" % (
        ",".join(ARMS)))
    ap.add_argument("--fork-store", type=int, default=0, help="X (0: the bound 1 + (K - 1) x 64)")
    ap.add_argument("--prefill-hook", default="nocopy", choices=("nocopy", "copy"))
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"))
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--check-every", type=int, default=0,
                    help="0 = the probe rounds; 1 = every round and the union / fork checks on")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    ap.add_argument("--profile-rounds", default="96,112",
                    help="probe rounds whose joint pass and D-slot passes run under torch.profiler (not marked)")
    ap.add_argument("--label-rounds", default="112", help="profiled rounds whose passes get the label ranges")
    ap.add_argument("--draft-marks", default="on", choices=("on", "off"), help="the D-slot marks on the marked rounds")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if a.arms != [x for x in ARMS if x in a.arms]:
        ap.error("--arms: L4PTFD before L4PTFDP (L4PTFDP-PEQ compares with L4PTFD's record)")
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1:
        ap.error("--probe-every >= 2, --check-every >= 0, journal >= 1")
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(TOPK, a.spec):
        ap.error("--fork-store %d: below the bound 1 + (K - 1) x 64 = %d (0 = the bound)"
                 % (a.fork_store, I3C.KA.store_slots(TOPK, a.spec)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    try:
        a.profile_rounds = sorted({int(x) for x in a.profile_rounds.split(",") if x})
        a.label_rounds = sorted({int(x) for x in a.label_rounds.split(",") if x})
    except ValueError:
        ap.error("--profile-rounds / --label-rounds: integers")
    if any(g <= 0 or g % a.probe_every for g in a.profile_rounds):
        ap.error("--profile-rounds: probe rounds only (g > 0, g %% %d == 0)" % a.probe_every)
    if len(a.profile_rounds) > 4:
        ap.error("--profile-rounds: at most 4")
    if not set(a.label_rounds) <= set(a.profile_rounds):
        ap.error("--label-rounds: a subset of --profile-rounds")
    return a


# ------------------------------------------------------------------ COPIED UNCHANGED from r2_i1p_cell.py at 30578a0
# (nosi-r2-i1p; tests_cpu/test_i4c.py compares each function's AST with that commit's): _med, merge_intervals,
# pass_profile, profile_summary, prof_after, profile_pass.
def _med(xs, nd: int = 4):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), nd) if xs else None


def merge_intervals(iv):
    """[(start, end)] -> the sorted union as disjoint [(start, end)]."""
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def pass_profile(trace, name: str = RANGE, top: int = 10) -> dict:
    """The host exposure of ONE profiled pass from a Kineto chrome trace (module docstring, PROFILER WINDOW): the GPU ops
    launched inside the record_function range `name` (r2_p3_profile.parse_trace) give the device window [first start,
    last end]; every GPU op inside the window counts (a graph's kernels too); busy = the union of their intervals."""
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    X = [e for e in ev if e.get("ph") == "X"]
    ann = [e for e in X if e.get("cat") == "user_annotation" and e.get("name") == name]
    if len(ann) != 1:
        return dict(error="range %r found %d times" % (name, len(ann)), n_gpu_ops=0, busy_ms=None)
    a = ann[0]
    h0 = float(a["ts"])
    h1 = h0 + float(a.get("dur", 0.0))
    key = (a.get("pid"), a.get("tid"))
    calls = [str(e.get("name", "")) for e in X if e.get("cat") in P3.LAUNCH_CATS and (e.get("pid"), e.get("tid")) == key
             and h0 <= float(e.get("ts", 0.0)) <= h1]
    out = dict(error=None, host_ms=round((h1 - h0) / 1e3, 4), runtime_calls=len(calls),
               launch_calls=sum(1 for n in calls if "Launch" in n), graph_launches=sum(1 for n in calls
                                                                                       if "GraphLaunch" in n),
               copy_calls=sum(1 for n in calls if "Memcpy" in n or "Memset" in n),
               event_calls=sum(1 for n in calls if "EventRecord" in n))
    ops = P3.parse_trace(trace, prefixes=(name,))
    inr = [o for o in ops if o["range"] == name]
    if not inr:
        out.update(n_gpu_ops=0, busy_ms=None, device_ms=None, busy_ratio=None)
        return out
    d0 = min(o["ts"] for o in inr)
    d1 = max(o["ts"] + o["dur_us"] for o in inr)
    win = [o for o in ops if o["ts"] >= d0 and o["ts"] + o["dur_us"] <= d1 + 1e-3]
    merged = merge_intervals([(o["ts"], o["ts"] + o["dur_us"]) for o in win])
    busy = sum(b - x for x, b in merged)
    gaps = [(merged[i][1], merged[i + 1][0] - merged[i][1]) for i in range(len(merged) - 1)]
    gaps = [g for g in gaps if g[1] > 0]
    dev = d1 - d0
    agg = P3.aggregate(win, 1)
    out.update(n_gpu_ops=len(win), n_ranged=len(inr), n_kernels=sum(1 for o in win if o["cat"] == "kernel"),
               n_copies=sum(1 for o in win if o["cat"] != "kernel"), device_ms=round(dev / 1e3, 4),
               busy_ms=round(busy / 1e3, 4), idle_ms=round((dev - busy) / 1e3, 4),
               busy_ratio=(round(busy / dev, 4) if dev > 0 else None),
               gaps_top=[dict(at_ms=round((s - d0) / 1e3, 4), ms=round(g / 1e3, 4))
                         for s, g in sorted(gaps, key=lambda x: -x[1])[:top]],
               gaps_count={k: sum(1 for _, g in gaps if g >= t) for k, t in (("ge_10us", 10.0), ("ge_100us", 100.0),
                                                                             ("ge_1ms", 1000.0))},
               gaps_ge_100us_ms=round(sum(g for _, g in gaps if g >= 100.0) / 1e3, 4),
               device_start_after_host_ms=round((d0 - h0) / 1e3, 4), device_end_after_host_ms=round((d1 - h1) / 1e3, 4),
               classes=agg["classes"], top_names=agg["names"][:15])
    return out


def profile_summary(passes, steady_joint):
    """The profiled passes of an arm: per pass the analysis and busy / steady joint median; medians."""
    rows = []
    for p in passes:
        b = p.get("busy_ms")
        rows.append(dict(p, busy_over_steady=(round(b / steady_joint, 4) if (b and steady_joint) else None)))
    ok = [r for r in rows if r.get("busy_ms")]
    return dict(n=len(rows), n_ok=len(ok), passes=rows, busy_ms_median=_med([r["busy_ms"] for r in ok]),
                busy_ratio_median=_med([r["busy_ratio"] for r in ok]),
                busy_over_steady_median=_med([r["busy_over_steady"] for r in ok]),
                wall_ms_median=_med([r.get("wall_ms") for r in ok]), steady_joint_median=steady_joint)


def prof_after(rd, pe: int, prof_rounds):
    """REPORT: the steady joint median before / after the first profiled round (a profiler after-effect would show)."""
    if not prof_rounds:
        return None
    g0 = min(prof_rounds)
    st = SC.steady_rounds(rd["rounds"], pe)
    before = [rd["phases"][g][JOINT] for g in st if g < g0]
    after = [rd["phases"][g][JOINT] for g in st if g > g0]
    return dict(first=g0, before_median=_med(before, 3), after_median=_med(after, 3), n_before=len(before),
                n_after=len(after), after_minus_before=(round(_med(after, 3) - _med(before, 3), 3)
                                                        if before and after else None))


def profile_pass(fn, clock, path, name: str = RANGE):
    """CHECK-ONLY, REPORT: fn() once under torch.profiler (CPU, and CUDA when present) inside the record_function range
    `name`, with clock events around it. The trace goes to path + ".gz". Returns (fn's result, the record: wall_ms and
    pass_profile's analysis, or an error). fn runs without the profiler when it cannot start; an exception of fn itself
    propagates (the arm fails, as without profiling)."""
    import torch
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        prof = profile(activities=acts)
        prof.__enter__()
    except Exception as ex:                                      # the profiler is evidence, not a gate
        return fn(), dict(error="start: %s: %s" % (type(ex).__name__, str(ex)[:200]), n_gpu_ops=0, busy_ms=None)
    e0, e1 = clock.event(), clock.event()
    try:
        with record_function(name):
            clock.record(e0)
            out = fn()
            clock.record(e1)
        clock.sync()
    finally:
        prof.__exit__(None, None, None)
    rec = dict(wall_ms=round(float(clock.elapsed(e0, e1)), 4), trace=None, error=None)
    try:
        prof.export_chrome_trace(path)
        with open(path, "rb") as f:
            raw = f.read()
        with gzip.open(path + ".gz", "wb") as g:
            g.write(raw)
        os.remove(path)
        rec["trace"] = os.path.basename(path) + ".gz"
        rec.update(pass_profile(json.loads(raw.decode("utf-8")), name))
    except Exception as ex:
        rec.update(error="read: %s: %s" % (type(ex).__name__, str(ex)[:200]), n_gpu_ops=0, busy_ms=None)
    return out, rec


# ------------------------------------------------------------------ pure parts (CPU-tested: tests_cpu/test_i4c.py)
def expected_labels(kind: str, L: int):
    """(labels, layers) of one fully marked pass: "embed" @ 0, per layer l the layer labels @ l, "head" and "end" @ L.
    kind "joint" (JOINT_LAYER: L4PTF's marks with the added ones) or "draft" (DRAFT_LAYER: the lean one-row pass)."""
    per = JOINT_LAYER if kind == "joint" else DRAFT_LAYER
    labs, lays = ["embed"], [0]
    for l in range(L):
        labs += list(per)
        lays += [l] * len(per)
    return labs + ["head", "end"], lays + [L, L]


def pass_stats(marks, elapsed, L: int) -> dict:
    """One fully marked pass [(layer, label, event)]: per label the ms summed over the layers (a label runs to the next
    mark), total (first .. last mark), lead (first .. "resid" of layer 0), tail ("post" of layer L - 1 .. last), span
    ("resid" @ 0 .. "post" @ L - 1: what L4PTF's own marks cover), and the label / layer sequence."""
    labs = [m[1] for m in marks]
    lays = [m[0] for m in marks]
    ph = {}
    for (_, lab, e0), (_, _, e1) in zip(marks[:-1], marks[1:]):
        ph[lab] = ph.get(lab, 0.0) + float(elapsed(e0, e1))
    out = dict(labels={k: round(v, 4) for k, v in ph.items()}, seq=labs, layers=lays, total=None, lead=None, tail=None,
               span=None)
    if len(marks) < 2:
        return out
    out["total"] = round(float(elapsed(marks[0][2], marks[-1][2])), 4)
    ir = next((i for i, m in enumerate(marks) if m[0] == 0 and m[1] == "resid"), None)
    ip = next((i for i in range(len(marks) - 1, -1, -1) if marks[i][0] == L - 1 and marks[i][1] == "post"), None)
    if ir is not None and ip is not None and ip > ir:
        out.update(lead=round(float(elapsed(marks[0][2], marks[ir][2])), 4),
                   tail=round(float(elapsed(marks[ip][2], marks[-1][2])), 4),
                   span=round(float(elapsed(marks[ir][2], marks[ip][2])), 4))
    return out


def marks_shape(seqs, lays, kind: str, L: int):
    """(ok, detail): every marked pass of `kind` has expected_labels(kind, L), labels and layers."""
    want, wl = expected_labels(kind, L)
    bad = []
    for i, (s, y) in enumerate(zip(seqs, lays)):
        if list(s) != want or list(y) != wl:
            j = next((k for k, (a, b) in enumerate(zip(s, want)) if a != b), min(len(s), len(want)))
            bad.append((i, "label %d: %s != %s (len %d vs %d)" % (j, s[j] if j < len(s) else None,
                                                                 want[j] if j < len(want) else None, len(s), len(want))))
    ok = bool(seqs) and len(seqs) == len(lays) and not bad
    return ok, "%d marked %s passes of %d marks each (L %d); bad %s" % (len(seqs), kind, len(want), L, bad[:3])


def joint_probe_summary(jst, rd) -> dict:
    """The marked joint passes of L4PTFDP (pass_stats with "round"): medians per label, total, lead, tail, span,
    uncovered (lead + tail), the probe joint phase of the same rounds and the J slot's plan (probe joint - total)."""
    if not jst:
        return dict(n=0)
    pj = [rd["phases"][x["round"]][JOINT] if x["round"] < len(rd["phases"]) else None for x in jst]
    plan = [(round(p - x["total"], 4) if (p is not None and x["total"] is not None) else None) for p, x in zip(pj, jst)]
    unc = [(round(x["lead"] + x["tail"], 4) if x["lead"] is not None else None) for x in jst]
    labels = sorted({k for x in jst for k in x["labels"]})
    return dict(n=len(jst), rounds=[x["round"] for x in jst],
                labels_median={k: _med([x["labels"].get(k, 0.0) for x in jst]) for k in labels},
                total=[x["total"] for x in jst], probe_joint=pj, plan=plan, total_median=_med([x["total"] for x in jst]),
                span_median=_med([x["span"] for x in jst]), lead_median=_med([x["lead"] for x in jst]),
                tail_median=_med([x["tail"] for x in jst]), uncovered_median=_med(unc),
                probe_joint_median=_med(pj), plan_median=_med(plan))


def draft_probe_summary(dst, rd) -> dict:
    """The marked draft passes of L4PTFDP (pass_stats with "round", "slot"): medians per label (all passes and per
    slot), the row labels' sum per pass (DRAFT_ROW), and per round the draft phase - the sum of its passes' totals
    (the work between and around the two D-slot passes)."""
    if not dst:
        return dict(n=0)
    labels = sorted({k for x in dst for k in x["labels"]})
    row = [round(sum(x["labels"].get(k, 0.0) for k in DRAFT_ROW), 4) for x in dst]
    rounds = sorted({x["round"] for x in dst})
    per_round = []
    for g in rounds:
        xs = [x for x in dst if x["round"] == g]
        ph = rd["phases"][g][DRAFT] if g < len(rd["phases"]) else None
        tot = sum(x["total"] for x in xs if x["total"] is not None)
        per_round.append(dict(round=g, passes=len(xs), draft_phase=ph, passes_total=round(tot, 4),
                              outside=(round(ph - tot, 4) if ph is not None else None)))
    slots = sorted({x["slot"] for x in dst})
    return dict(n=len(dst), rounds=rounds, labels_median={k: _med([x["labels"].get(k, 0.0) for x in dst]) for k in labels},
                by_slot={s: dict(n=sum(1 for x in dst if x["slot"] == s),
                                 total_median=_med([x["total"] for x in dst if x["slot"] == s]),
                                 labels_median={k: _med([x["labels"].get(k, 0.0) for x in dst if x["slot"] == s])
                                                for k in labels}) for s in slots},
                row_ms=row, row_ms_median=_med(row), total_median=_med([x["total"] for x in dst]),
                lead_median=_med([x["lead"] for x in dst]), tail_median=_med([x["tail"] for x in dst]),
                per_round=per_round, outside_median=_med([x["outside"] for x in per_round]),
                two_pass_rounds=sum(1 for x in per_round if x["passes"] == 2))


def l7_record(steady_ref, sum_ref, js) -> dict:
    """L7 (LEDGER R5.6 item 1): gap = REF steady joint - REF probe mark sum (L4PTF's labels only). Its split from the
    fully marked passes of L4PTFDP (the same rounds and tokens): gap = uncovered (lead + tail) + plan (the J slot
    before the pass) - overhead (probe joint - REF steady joint: the check-only work and the marks). residual = gap -
    that sum (medians are not additive: a residual of a few tenths of a ms is the expected size)."""
    out = dict(steady_joint_ref=steady_ref, probe_sum_ref=sum_ref, gap_ms=None, uncovered_ms=js.get("uncovered_median"),
               plan_ms=js.get("plan_median"), probe_overhead_ms=None, predicted_gap_ms=None, residual_ms=None)
    if steady_ref is None or sum_ref is None:
        return out
    out["gap_ms"] = round(steady_ref - sum_ref, 4)
    if js.get("probe_joint_median") is not None:
        out["probe_overhead_ms"] = round(js["probe_joint_median"] - steady_ref, 4)
    if None not in (out["uncovered_ms"], out["plan_ms"], out["probe_overhead_ms"]):
        out["predicted_gap_ms"] = round(out["uncovered_ms"] + out["plan_ms"] - out["probe_overhead_ms"], 4)
        out["residual_ms"] = round(out["gap_ms"] - out["predicted_gap_ms"], 4)
    return out


def local_steady(rd, pe: int, g: int, w: int = LOCAL_W, phase: int = JOINT):
    """The median of a phase over the steady rounds within +-w rounds of round g (the drain makes late rounds cheaper:
    I1P's PROF-AFTER was -10 to -19 ms)."""
    st = [x for x in SC.steady_rounds(rd["rounds"], pe) if abs(x - g) <= w and x < len(rd["phases"])]
    return _med([rd["phases"][x][phase] for x in st], 4)


def size_table(ops) -> dict:
    """The GPU ops by device duration: SIZE_NAMES buckets (count, ms)."""
    out = {n: [0, 0.0] for n in SIZE_NAMES}
    for o in ops:
        i = next((j for j, e in enumerate(SIZE_EDGES_US) if o["dur_us"] < e), len(SIZE_EDGES_US))
        out[SIZE_NAMES[i]][0] += 1
        out[SIZE_NAMES[i]][1] += o["dur_us"]
    return {k: dict(ops=v[0], ms=round(v[1] / 1e3, 4)) for k, v in out.items()}


def window_ops(trace, name: str):
    """pass_profile's window: the GPU ops inside [first start, last end] of the ops launched in range `name`."""
    ops = P3.parse_trace(trace, prefixes=(name,))
    inr = [o for o in ops if o["range"] == name]
    if not inr:
        return []
    d0 = min(o["ts"] for o in inr)
    d1 = max(o["ts"] + o["dur_us"] for o in inr)
    return [o for o in ops if o["ts"] >= d0 and o["ts"] + o["dur_us"] <= d1 + 1e-3]


def split_label(rng: str, prefix: str = LABEL_PREFIX):
    """"i4cL.resid@3" -> ("resid", 3)."""
    s = rng[len(prefix):]
    lab, _, lay = s.rpartition("@")
    return lab, int(lay)


def pass_detail(trace, name: str, labelled: bool, L: int, top: int = 6) -> dict:
    """The kernels of one profiled pass beyond pass_profile: SMALL + COPY vs the other classes, the duration buckets;
    labelled: per label (its range "i4cL.<label>@<layer>", the innermost range of the launch) the GPU ops, device ms,
    classes, top names, the ranges' host ms; lead (embed + layer 0's pre / twin / select) and tail (layer L - 1's post +
    head); the GPU ops of the window outside every label range (unlabelled: 0 when the labels cover the pass)."""
    win = window_ops(trace, name)
    cls = {}
    for o in win:
        x = cls.setdefault(o["cls"], [0, 0.0])
        x[0] += 1
        x[1] += o["dur_us"]
    small = [sum(v[i] for k, v in cls.items() if k in SMALL_CLASSES) for i in (0, 1)]
    large = [sum(v[i] for k, v in cls.items() if k not in SMALL_CLASSES) for i in (0, 1)]
    out = dict(n_window=len(win), sizes=size_table(win),
               small=dict(ops=small[0], ms=round(small[1] / 1e3, 4)), large=dict(ops=large[0], ms=round(large[1] / 1e3, 4)),
               sizes_small=size_table([o for o in win if o["cls"] in SMALL_CLASSES]))
    if not labelled:
        return out
    key = lambda o: (o["corr"], o["ts"], o["name"])                    # noqa: E731
    inwin = {key(o) for o in win}
    lops = [o for o in P3.parse_trace(trace, prefixes=(LABEL_PREFIX,)) if key(o) in inwin]
    labs, unl = {}, [0, 0.0]
    lead, tail = [0, 0.0], [0, 0.0]
    for o in lops:
        if o["range"] is None:
            unl[0] += 1
            unl[1] += o["dur_us"]
            continue
        lab, lay = split_label(o["range"])
        d = labs.setdefault(lab, dict(ops=0, us=0.0, classes={}, names={}, host_us=0.0, ranges=0))
        d["ops"] += 1
        d["us"] += o["dur_us"]
        c = d["classes"].setdefault(o["cls"], [0, 0.0])
        c[0] += 1
        c[1] += o["dur_us"]
        nm = d["names"].setdefault(o["name"][:120], [0, 0.0])
        nm[0] += 1
        nm[1] += o["dur_us"]
        for box, hit in ((lead, lab in LEAD and lay == 0), (tail, (lab == "post" and lay == L - 1) or lab == "head")):
            if hit:
                box[0] += 1
                box[1] += o["dur_us"]
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith(LABEL_PREFIX):
            lab, _ = split_label(e["name"])
            d = labs.setdefault(lab, dict(ops=0, us=0.0, classes={}, names={}, host_us=0.0, ranges=0))
            d["host_us"] += float(e.get("dur", 0.0))
            d["ranges"] += 1
    out["labels"] = {k: dict(ops=v["ops"], ms=round(v["us"] / 1e3, 4), host_ms=round(v["host_us"] / 1e3, 4),
                             ranges=v["ranges"],
                             classes={c: dict(ops=x[0], ms=round(x[1] / 1e3, 4)) for c, x in sorted(v["classes"].items())},
                             top_names=[dict(name=n, ops=x[0], ms=round(x[1] / 1e3, 4)) for n, x in
                                        sorted(v["names"].items(), key=lambda kv: -kv[1][1])[:top]])
                     for k, v in sorted(labs.items())}
    out.update(lead=dict(ops=lead[0], ms=round(lead[1] / 1e3, 4)), tail=dict(ops=tail[0], ms=round(tail[1] / 1e3, 4)),
               unlabelled=dict(ops=unl[0], ms=round(unl[1] / 1e3, 4)))
    return out


def prof_record(passes, ref_rd, pe: int, steady_joint_ref, steady_draft_ref) -> dict:
    """The profiled passes of L4PTFDP against the REFERENCE arm (L4PTFD, same job, same tokens): the joint passes
    (r2_i1p's profile_summary against the REF steady joint median, and against the local REF steady median), the
    SMALL + COPY ms and GPU ops per joint pass, the draft busy per round against the REF steady draft median."""
    joint = [p for p in passes if p.get("slot") == "joint"]
    js = profile_summary(joint, steady_joint_ref)
    for p in js["passes"]:
        loc = local_steady(ref_rd, pe, p.get("round", -1)) if ref_rd is not None else None
        p["local_steady_ref"] = loc
        p["busy_over_local"] = round(p["busy_ms"] / loc, 4) if (p.get("busy_ms") and loc) else None
        p["small_ms"] = ((p.get("detail") or {}).get("small") or {}).get("ms")
    rounds = sorted({p.get("round") for p in passes})
    dr = []
    for g in rounds:
        xs = [p for p in passes if p.get("round") == g and str(p.get("slot", "")).startswith("draft")]
        b = [p.get("busy_ms") for p in xs]
        tot = round(sum(b), 4) if (b and all(x is not None for x in b)) else None
        dr.append(dict(round=g, passes=len(xs), busy_ms=tot,
                       busy_over_steady=(round(tot / steady_draft_ref, 4) if (tot and steady_draft_ref) else None)))
    dpass = [p for p in passes if str(p.get("slot", "")).startswith("draft")]
    return dict(joint=js, busy_over_local_median=_med([p["busy_over_local"] for p in js["passes"]]),
                gpu_ops_median=_med([p.get("n_gpu_ops") for p in js["passes"] if p.get("busy_ms")]),
                small_ms_median=_med([p["small_ms"] for p in js["passes"]]), draft=dr,
                draft_busy_over_steady_median=_med([x["busy_over_steady"] for x in dr]),
                steady_draft_ref=steady_draft_ref, draft_passes=dpass)


def l5_record(passes, js_probe) -> dict:
    """L5 and the other labels of the LABELLED joint pass(es): device ms per label against the label's elapsed median
    (the marked joint passes of L4PTFDP). L5 = union + chain + write."""
    lab = [p for p in passes if p.get("slot") == "joint" and (p.get("detail") or {}).get("labels")]
    if not lab:
        return dict(n=0)
    el = (js_probe or {}).get("labels_median") or {}
    names = sorted({k for p in lab for k in p["detail"]["labels"]})
    dev = {k: _med([p["detail"]["labels"].get(k, {}).get("ms") for p in lab]) for k in names}
    l5d = round(sum(dev.get(k) or 0.0 for k in L5_LABELS), 4)
    l5e = sum(el.get(k) or 0.0 for k in L5_LABELS) if all(el.get(k) is not None for k in L5_LABELS) else None
    return dict(n=len(lab), rounds=[p.get("round") for p in lab], device_ms=dev, elapsed_ms={k: el.get(k) for k in names},
                device_over_elapsed={k: (round(dev[k] / el[k], 4) if (dev.get(k) and el.get(k)) else None) for k in names},
                l5_device_ms=l5d, l5_elapsed_ms=(round(l5e, 4) if l5e is not None else None),
                l5_device_frac=(round(l5d / l5e, 4) if l5e else None),
                lead=[p["detail"].get("lead") for p in lab], tail=[p["detail"].get("tail") for p in lab],
                unlabelled=[p["detail"].get("unlabelled") for p in lab],
                labels=lab[-1]["detail"]["labels"])


def marks_neutral_gate(arm, rd, pe: int, mark_rounds, prof_rounds, rec):
    """<arm>-MARKS-NEUTRAL (MUST; r2_i1p_cell's gate with the draft-marked rounds in mark_rounds): the marked and the
    profiled rounds are probe rounds, disjoint, and never steady rounds; the steady joint median recomputed = the
    record's."""
    G = rd["rounds"]
    steady = set(SC.steady_rounds(G, pe))
    probe = {g for g in range(G) if pe > 0 and g > 0 and g % pe == 0}
    mr, pr = sorted(set(mark_rounds)), list(prof_rounds)
    jm = WA._med([rd["phases"][g][JOINT] for g in sorted(steady) if g < len(rd["phases"])
                  and len(rd["phases"][g]) > JOINT])
    want = (rec.get("phases_median") or {}).get("joint")
    ok = (pe > 0 and len(mr) > 0 and set(mr) <= probe and set(pr) <= probe and not (set(mr) & steady)
          and not (set(pr) & steady) and not (set(mr) & set(pr)) and jm == want and len(steady) + len(probe) + 1 == G)
    return ("%s-MARKS-NEUTRAL" % arm, True, ok,
            "%d rounds: %d steady, %d probe; marked rounds %s; profiled rounds %s; overlap with steady %s; steady joint "
            "median recomputed %s = record %s" % (G, len(steady), len(probe), mr, pr,
                                                  sorted((set(mr) | set(pr)) & steady), jm, want))


def phase_sum_gate(arm, js, tol=PHASE_SUM_TOL):
    """<arm>-PHASE-SUM (MUST; I1P's): every fully marked joint pass 0 <= probe joint - total <= max(tol[0], tol[1] x
    probe joint): the marks cover the whole pass; the probe joint adds only the J slot's plan."""
    bad = []
    for i, (t, j) in enumerate(zip(js.get("total") or [], js.get("probe_joint") or [])):
        lim = max(tol[0], tol[1] * j) if j is not None else None
        if j is None or t is None or not (-0.01 <= j - t <= lim):
            bad.append((i, t, j))
    plan = [p for p in (js.get("plan") or []) if p is not None]
    ok = bool(js.get("n")) and not bad
    return ("%s-PHASE-SUM" % arm, True, ok,
            "probe joint - mark total over %s marked passes: median %s, max %s ms (allowed 0 .. max(%.1f ms, %.0f %% of "
            "the probe joint)); out of bound %s" % (js.get("n"), _med(plan), max(plan) if plan else None, tol[0],
                                                    100 * tol[1], bad[:3]))


def peq_gate(arm, cp, cref, steps: int):
    """<arm>-PEQ (MUST): r2_i4a_cell.eq_gate (tokens, fingerprints, acceptance record, state checksums) against the
    reference arm, renamed."""
    n, m, ok, d = I4A.eq_gate(arm, cp, cref, steps, True)
    return ("%s-PEQ" % arm, True, ok, d.replace("L4PTF's", "L4PTFD's").replace("no L4PTF record", "no L4PTFD record"))


def probe_gates(arm, ev, rec, rd, pe: int, L: int, draft_marks: bool):
    """MARKS-NEUTRAL, MARKS-SHAPE (joint and draft), PHASE-SUM (MUST); PROF (REPORT)."""
    jm, dm = ev["joint"], ev["draft"]
    pr = ev["profile"]
    g = [marks_neutral_gate(arm, rd, pe, [x["round"] for x in jm] + [x["round"] for x in dm], pr["done"], rec)]
    okj, dj = marks_shape([x["seq"] for x in jm], [x["layers"] for x in jm], "joint", L)
    okd, dd = (marks_shape([x["seq"] for x in dm], [x["layers"] for x in dm], "draft", L) if draft_marks
               else (True, "draft marks off"))
    g.append(("%s-MARKS-SHAPE" % arm, True, okj and okd, "%s; %s" % (dj, dd)))
    g.append(phase_sum_gate(arm, rec["probe_joint"]))
    p = rec.get("profile") or {}
    js = p.get("joint") or {}
    want = len(pr["rounds"])
    g.append(("%s-PROF" % arm, False, want > 0 and js.get("n_ok", 0) == want and len(pr["done"]) == want,
              "profiled rounds %s (done %s); joint windows %s (busy %s), busy / REF steady joint %s, busy / local %s; "
              "draft windows %s; errors %s" % (pr["rounds"], pr["done"], js.get("n_ok"), js.get("busy_ms_median"),
                                               js.get("busy_over_steady_median"), p.get("busy_over_local_median"),
                                               [(x["round"], x["passes"]) for x in p.get("draft") or []],
                                               [q.get("error") for q in pr["passes"] if q.get("error")][:3])))
    return g


def verdicts(sh, sp, B: int, peq=None):
    """PRED_I4C_BY_B[B] against the records: [(name, verdict, detail)]; a None band gives REPORT; NOT_RUN without
    L4PTFDP (the probe keys) or without L4PTFD (every key)."""
    P = PRED_I4C_BY_B.get(B) or {k: None for k in PRED_KEYS}
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))
    ref, pp = (sp or {}).get(REF), (sp or {}).get(PROBED)
    if ref is None:
        out.append(("L4PTFD-RUN", "NOT_RUN", "the reference arm did not run"))
        return out
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    med = ref.get("rates", {}).get("median_tok_s")
    band("I4C-REF-RATIO", (med / shm) if (med and shm) else None, P.get("I4C_ref_ratio"))
    if pp is None:
        out.append(("L4PTFDP-RUN", "NOT_RUN", "the profiled arm did not run"))
        return out
    pr, jp, dp, l7, l5 = (pp.get("profile") or {}), (pp.get("probe_joint") or {}), (pp.get("probe_draft") or {}), (
        pp.get("l7") or {}), (pp.get("l5") or {})
    band("I4C-BUSY-FRAC", (pr.get("joint") or {}).get("busy_over_steady_median"), P.get("I4C_busy_frac"))
    band("I4C-DRAFT-BUSY-FRAC", pr.get("draft_busy_over_steady_median"), P.get("I4C_draft_busy_frac"))
    band("I4C-JOINT-GPU-OPS", pr.get("gpu_ops_median"), P.get("I4C_joint_gpu_ops"))
    band("I4C-JOINT-SMALL-MS", pr.get("small_ms_median"), P.get("I4C_joint_small_ms"))
    band("I4C-L5-DEVICE-FRAC", l5.get("l5_device_frac"), P.get("I4C_L5_device_frac"))
    band("I4C-L7-GAP-MS", l7.get("gap_ms"), P.get("I4C_L7_gap_ms"))
    band("I4C-L7-UNCOVERED-MS", l7.get("uncovered_ms"), P.get("I4C_L7_uncovered_ms"))
    band("I4C-L7-RESIDUAL-MS", l7.get("residual_ms"), P.get("I4C_L7_residual_ms"))
    band("I4C-DRAFT-ROW-MS", dp.get("row_ms_median"), P.get("I4C_draft_row_ms"))
    band("I4C-DRAFT-OUTSIDE-MS", dp.get("outside_median"), P.get("I4C_draft_outside_ms"))
    jr, jd = (ref.get("phases_median") or {}).get("joint"), (pp.get("phases_median") or {}).get("joint")
    band("I4C-PROF-NEUTRAL", (jd / jr) if (jd and jr) else None, P.get("I4C_prof_neutral"))
    want = P.get("I4C_peq")
    out.append(("I4C-PEQ", "REPORT" if (want is None or peq is None) else ("CONFIRMED" if peq == bool(want) else
                                                                          "REFUTED"),
                "L4PTFDP-PEQ passed %s; registered %s" % (peq, want)))
    return out


# ------------------------------------------------------------------ the probe (GPU; CPU-tested on the stand-ins)
_MISS = object()


def _patch(stack, obj, name, new):
    """obj.name = new as an INSTANCE attribute; the old instance attribute (or its absence) goes on the stack."""
    stack.append((obj, name, obj.__dict__.get(name, _MISS)))
    setattr(obj, name, new)


def _unpatch(stack):
    """Restores the stack in reverse order: an instance attribute that did not exist is removed (the class's
    method is visible again); one that existed (a fused twin's bound step) is put back."""
    while stack:
        obj, name, old = stack.pop()
        if old is _MISS:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


class _NullClock:
    """vs.clock of a labelled profiled joint pass: no event (verify_rows._mark records nothing; the range is the mark)."""

    def event(self):
        return None

    def record(self, e):
        return None


class Marker:
    """One pass's marks. sink not None: (layer, label, event) appended with clock (verify_rows._mark's form: CUDA events
    on the GPU). sink None: a torch.profiler record_function range per label, LABEL_PREFIX + "<label>@<layer>"; each
    mark closes the open range; "end" opens none. cur: the last (layer, label)."""

    def __init__(self, sink=None, clock=None):
        self.sink, self.clock = sink, clock
        self.cur = None
        self.rf = None

    def mark(self, l: int, label: str) -> None:
        self.cur = (l, label)
        if self.sink is not None:
            e = self.clock.event()
            self.clock.record(e)
            self.sink.append((l, label, e))
            return
        self.close()
        if label != "end":
            from torch.profiler import record_function
            self.rf = record_function("%s%s@%d" % (LABEL_PREFIX, label, l))
            self.rf.__enter__()

    def close(self) -> None:
        if self.rf is not None:
            rf, self.rf = self.rf, None
            rf.__exit__(None, None, None)


class _Sink:
    """vs.marks of a labelled profiled joint pass: each verify_rows._mark (L4PTF's own marks) is a Marker boundary."""

    def __init__(self, mk):
        self.mk = mk

    def append(self, x) -> None:
        self.mk.mark(x[0], x[1])


class Probe:
    """MEASUREMENT ONLY (module docstring): wraps runner.T.run_pass (an instance attribute) of an L4PTFD runner
    (r2_i4a_lean's TentRunnerFD, phase_marks True). Rounds g >= 1 only; a one-row J slot is never probed.
      a profiled round (profile_rounds): each joint pass (R > 1) and each D-slot pass (kind "draft") runs once under
        profile_pass, ONE window per pass ("i4c.joint", "i4c.draft0", "i4c.draft1"), with vs.marks None,
        TentState.checking False and the meter's timing False (restored after the pass); a labelled round
        (label_rounds) adds the label ranges (Marker without a sink; L4PTF's own marks reach it through _Sink).
      a marked round (vs.marks is a list: a probe round of phase_marks): the joint pass gets the added marks in
        vs.marks (full coverage); each D-slot pass gets its own event list (draft_marks).
    The added marks are instance-attribute wrappers set for ONE pass and removed after it: T._layer ("pre"; "head"
    after the last layer), each twin's step_unchecked ("twin", the first call of the layer) and comp_k_flat
    ("select"); one-row passes also T._rows ("post" after the row), T._row_lean ("resid"), T._write_lean ("write";
    "amask" after it) and T.attn ("attnA" / "attnB"; "bmask" / "merge" after each call). No wrapper calls a torch op.
    box: profile_rounds, label_rounds, passes (profile_pass records + round, slot, labelled, rows, active, detail),
    done, joint / draft ([(round, slot, marks)]), skipped (a profiled round's one-row J slot)."""

    def __init__(self, runner, clock, out_dir, arm: str, profile_rounds=(), label_rounds=(), draft_marks: bool = True):
        T = runner.T
        if not hasattr(T, "_row_lean"):
            raise ValueError("the probe takes an L4PTFD runner (r2_i4a_lean.TentRunnerFD)")
        if not set(label_rounds) <= set(profile_rounds):
            raise ValueError("label rounds %s: a subset of the profiled rounds %s" % (label_rounds, profile_rounds))
        self.r, self.T, self.clock = runner, T, clock
        self.out_dir, self.arm = out_dir, arm
        self.L = len(T.rs.st.twins)
        self.prof = sorted(int(g) for g in profile_rounds)
        self.lab = sorted(int(g) for g in label_rounds)
        self.draft_marks = bool(draft_marks)
        self.box = dict(profile_rounds=list(self.prof), label_rounds=list(self.lab), passes=[], done=[], joint=[],
                        draft=[], skipped=[])
        self._slots = {}
        self._stack = []
        self._run = T.run_pass
        _patch(self._stack, T, "run_pass", self._run_pass)

    def detach(self) -> None:
        _unpatch(self._stack)

    # -------------------------------------------------------------- the dispatch
    def _run_pass(self, model, plan):
        g = self.r.round
        if g is None or g < 1:
            return self._run(model, plan)
        if plan.R > 1:
            slot = "joint"
        elif plan.kind == "draft":
            n = self._slots.get(g, 0)
            self._slots[g] = n + 1
            slot = "draft%d" % n
        else:
            if g in self.prof:
                self.box["skipped"].append((g, plan.kind))
            return self._run(model, plan)
        if g in self.prof:
            return self._profiled(model, plan, g, slot)
        if self.r.vs.marks is not None and (plan.R > 1 or self.draft_marks):
            return self._marked(model, plan, g, slot)
        return self._run(model, plan)

    # -------------------------------------------------------------- the wrappers of one pass
    def _install(self, stack, mk, one_row: bool) -> None:
        T, L = self.T, self.L
        layer0 = T._layer

        def layer(l, *a, **k):
            mk.mark(l, "pre")
            out = layer0(l, *a, **k)
            if l == L - 1:
                mk.mark(L, "head")
            return out
        _patch(stack, T, "_layer", layer)
        for l, tw in enumerate(T.rs.st.twins):
            _wrap_twin(stack, mk, l, tw)
        if not one_row:
            return
        rows0, row0, write0, attn0 = T._rows, T._row_lean, T._write_lean, T.attn
        st = dict(l=0, a=0)

        def rows(l, *a, **k):
            out = rows0(l, *a, **k)
            mk.mark(l, "post")
            return out

        def row(l, *a, **k):
            st["l"], st["a"] = l, 0
            mk.mark(l, "resid")
            return row0(l, *a, **k)

        def write(l, *a, **k):
            mk.mark(l, "write")
            out = write0(l, *a, **k)
            mk.mark(l, "amask")
            return out

        def attn(*a, **k):
            i = st["a"]
            st["a"] = i + 1
            mk.mark(st["l"], "attnA" if i == 0 else "attnB")
            out = attn0(*a, **k)
            mk.mark(st["l"], "bmask" if i == 0 else "merge")
            return out
        _patch(stack, T, "_rows", rows)
        _patch(stack, T, "_row_lean", row)
        _patch(stack, T, "_write_lean", write)
        _patch(stack, T, "attn", attn)

    # -------------------------------------------------------------- a marked pass (events)
    def _marked(self, model, plan, g: int, slot: str):
        vs, L = self.r.vs, self.L
        joint = plan.R > 1
        sink = vs.marks if joint else []
        mk = Marker(sink, vs.clock if joint else self.clock)
        i0 = len(sink)
        stack = []
        self._install(stack, mk, not joint)
        try:
            mk.mark(0, "embed")
            out = self._run(model, plan)
            mk.mark(L, "end")
        finally:
            _unpatch(stack)
        (self.box["joint"] if joint else self.box["draft"]).append((g, slot, list(sink[i0:])))
        return out

    # -------------------------------------------------------------- a profiled pass (torch.profiler)
    def _profiled(self, model, plan, g: int, slot: str):
        r, T = self.r, self.T
        vs, L = r.vs, self.L
        labelled = g in self.lab
        saved = (vs.marks, vs.clock, T.checking, r.meter.timing)
        stack = []
        mk = Marker() if labelled else None
        vs.marks, T.checking, r.meter.timing = None, False, False
        if mk is not None:
            if plan.R > 1:
                vs.marks, vs.clock = _Sink(mk), _NullClock()
            self._install(stack, mk, plan.R == 1)

        def fn():
            try:
                if mk is not None:
                    mk.mark(0, "embed")
                out_ = self._run(model, plan)
                if mk is not None:
                    mk.mark(L, "end")
                return out_
            finally:
                if mk is not None:
                    mk.close()
        path = os.path.join(self.out_dir, "trace_%s_g%d_%s.json" % (self.arm, g, slot))
        try:
            out, rec = profile_pass(fn, self.clock, path, RANGE_PREFIX + slot)
        finally:
            _unpatch(stack)
            vs.marks, vs.clock, T.checking, r.meter.timing = saved
        rec.update(round=g, slot=slot, labelled=labelled, rows=int(plan.R), active=int(plan.act.sum()))
        if rec.get("trace"):
            try:
                with gzip.open(os.path.join(self.out_dir, rec["trace"]), "rt") as f:
                    rec["detail"] = pass_detail(json.load(f), RANGE_PREFIX + slot, labelled, L)
            except Exception as ex:                                    # evidence, not a gate
                rec["detail"] = dict(error="%s: %s" % (type(ex).__name__, str(ex)[:200]))
        self.box["passes"].append(rec)
        if g not in self.box["done"]:
            self.box["done"].append(g)
        return out

    # -------------------------------------------------------------- after the run (events complete: after a sync)
    def evidence(self, elapsed) -> dict:
        j = [dict(pass_stats(m, elapsed, self.L), round=g) for g, _, m in self.box["joint"]]
        d = [dict(pass_stats(m, elapsed, self.L), round=g, slot=s) for g, s, m in self.box["draft"]]
        p = dict(rounds=list(self.prof), label_rounds=list(self.lab), passes=list(self.box["passes"]),
                 done=list(self.box["done"]), skipped=list(self.box["skipped"]))
        return dict(joint=j, draft=d, profile=p)


def _wrap_twin(stack, mk, l: int, tw) -> None:
    """Layer l's twin: "twin" at its first step of the layer, "select" at the selection's comp_k_flat."""
    step0, comp0 = tw.step_unchecked, tw.comp_k_flat

    def step(*a, **k):
        if mk.cur == (l, "pre"):
            mk.mark(l, "twin")
        return step0(*a, **k)

    def comp(*a, **k):
        if mk.cur in ((l, "pre"), (l, "twin")):
            mk.mark(l, "select")
        return comp0(*a, **k)
    _patch(stack, tw, "step_unchecked", step)
    _patch(stack, tw, "comp_k_flat", comp)


# ------------------------------------------------------------------ one arm (GPU)
def run_arm_c(model, ids, inputs, steps: int, clock, opt, FORK, out_dir=None, probe: bool = False):
    """r2_i4a_cell.run_arm_x's steps with lean True (L4PTFD), line for line; probe True (L4PTFDP): a Probe on the runner
    before runner.run and its evidence after the run (after the sync: the events are complete)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    Runner = LN.lean_classes(STF)[1]
    gb = 1024 ** 3
    K = int(opt["K"])
    X = I3C.fork_store_slots("L4PTF", K, opt["fork_store"])
    cache, arena, prefill_s, first, mem = I3C._prefill_chain(model, ids, inputs, steps, clock, X, opt["prefill_hook"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    I3C.KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()                              # the decode peak (the prefill peak is above)
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=opt["tu_gemm"], extra=X, journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"])
    check = opt["check_every"] == 1
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                    check_every=ce, union_check=check, union_stats=check, phase_marks=True)
    if check:
        runner.vs.fa_check = {}                                        # CHECK-ONLY: the "F" / "RA" probes
    pr = (Probe(runner, clock, out_dir, opt["arm"], opt["profile_rounds"], opt["label_rounds"], opt["draft_marks"])
          if probe else None)
    runner.run(steps)
    if pr is not None:
        pr.detach()
    rd = runner.read()
    rd.pop("trace_all", None)
    clock.sync()
    mem.update(decode_peak_GB=round(torch.cuda.max_memory_allocated() / gb, 3),
               decode_peak_reserved_GB=round(torch.cuda.max_memory_reserved() / gb, 3))
    phases = [VR.phase_ms(m, clock.elapsed) for m in runner.marks if m]
    layers = [VR.layer_ms(m, clock.elapsed) for m in runner.marks if m]
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()),
               fp_ok=runner.fp_ok.clone(), K=K, probe_every=opt["probe_every"], prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, spec=runner.meta), undo=None, mem=mem,
               joint_phases=I3C.fork_phase_summary(phases, layers), fa_check=runner.vs.fa_check,
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4), cb0_GB=round(arena.chain_bytes() / 1e9, 4)),
               compile=I3C._compile_report(FORK), state=I4A.state_checksum(runner),
               probe=(pr.evidence(clock.elapsed) if pr is not None else None), L=len(runner.rs.st.twins))
    runner.detach()
    del runner, rs, cache, arena
    return res


def probe_record(res, ref_rec, ref_rd, pe: int) -> dict:
    """The probe's part of the L4PTFDP record: the marked joint and draft passes, the profiled passes against the
    reference arm, L5 and L7. ref_rec / ref_rd: the L4PTFD record and read() (None: no reference in this job)."""
    ev, rd = res["probe"], res["read"]
    js = joint_probe_summary(ev["joint"], rd)
    ds = draft_probe_summary(ev["draft"], rd)
    pm = (ref_rec or {}).get("phases_median") or {}
    sj, sd = pm.get("joint"), pm.get("draft")
    sref = (((ref_rec or {}).get("joint_phases_fork") or {}).get("groups_median") or {}).get("total")
    passes = ev["profile"]["passes"]
    return dict(probe_joint=js, probe_draft=ds, profile=prof_record(passes, ref_rd, pe, sj, sd),
                l5=l5_record(passes, js), l7=l7_record(sj, sref, js),
                profile_rounds=ev["profile"]["rounds"], label_rounds=ev["profile"]["label_rounds"],
                profile_done=ev["profile"]["done"], profile_skipped=ev["profile"]["skipped"])


def print_probe(name, r):
    js, ds, p, l5, l7 = (r.get("probe_joint") or {}), (r.get("probe_draft") or {}), (r.get("profile") or {}), (
        r.get("l5") or {}), (r.get("l7") or {})
    if js.get("n"):
        print("MARKS %s joint (%d fully marked probe passes, rounds %s; medians, ms; not additive): labels %s; total %s, "
              "span (L4PTF's marks) %s, lead %s, tail %s, uncovered %s; probe joint %s, plan %s" % (
                  name, js["n"], js["rounds"], js["labels_median"], js["total_median"], js["span_median"],
                  js["lead_median"], js["tail_median"], js["uncovered_median"], js["probe_joint_median"],
                  js["plan_median"]))
    if ds.get("n"):
        print("MARKS %s draft (%d marked D-slot passes in rounds %s; %d rounds with two): labels %s; row (resid .. merge) "
              "%s; total %s; by slot %s; outside the passes per round %s (median %s)" % (
                  name, ds["n"], ds["rounds"], ds["two_pass_rounds"], ds["labels_median"], ds["row_ms_median"],
                  ds["total_median"], {s: v["total_median"] for s, v in ds["by_slot"].items()},
                  [(x["round"], x["outside"]) for x in ds["per_round"]], ds["outside_median"]))
    print("L7 %s: %s" % (name, l7))
    for q in (p.get("joint") or {}).get("passes", []):
        print("PROFILE %s round %s %s: wall %s ms; device window %s, busy %s, busy / window %s, busy / REF steady joint "
              "%s, busy / local REF steady %s (%s); GPU ops %s (kernels %s, copies %s); host %s ms, launches %s; idle "
              "gaps %s, sum >= 100 us %s ms; largest %s; small (SMALL + COPY) %s, large %s; sizes %s; trace %s; error "
              "%s" % (name, q.get("round"), q.get("slot"), q.get("wall_ms"), q.get("device_ms"), q.get("busy_ms"),
                      q.get("busy_ratio"), q.get("busy_over_steady"), q.get("busy_over_local"),
                      q.get("local_steady_ref"), q.get("n_gpu_ops"), q.get("n_kernels"), q.get("n_copies"),
                      q.get("host_ms"), q.get("launch_calls"), q.get("gaps_count"), q.get("gaps_ge_100us_ms"),
                      (q.get("gaps_top") or [])[:3], (q.get("detail") or {}).get("small"),
                      (q.get("detail") or {}).get("large"), (q.get("detail") or {}).get("sizes"), q.get("trace"),
                      q.get("error")))
        print("PROFILE %s round %s classes: %s" % (name, q.get("round"), q.get("classes")))
    print("PROFILE %s drafts: %s; draft busy / REF steady draft %s (REF steady draft %s ms)" % (
        name, p.get("draft"), p.get("draft_busy_over_steady_median"), p.get("steady_draft_ref")))
    for q in p.get("draft_passes") or []:
        d = q.get("detail") or {}
        print("PROFILE %s round %s %s: wall %s ms; device window %s, busy %s, busy / window %s; GPU ops %s; host %s ms; "
              "small %s, large %s; classes %s; error %s" % (
                  name, q.get("round"), q.get("slot"), q.get("wall_ms"), q.get("device_ms"), q.get("busy_ms"),
                  q.get("busy_ratio"), q.get("n_gpu_ops"), q.get("host_ms"), d.get("small"), d.get("large"),
                  q.get("classes"), q.get("error")))
        for k, v in sorted((d.get("labels") or {}).items()):
            print("LABEL %s %s %s: ops %s, device %s ms, host %s ms (%s ranges); classes %s" % (
                name, q.get("slot"), k, v["ops"], v["ms"], v["host_ms"], v["ranges"], v["classes"]))
    if l5.get("n"):
        print("L5 %s (labelled joint pass, rounds %s): device ms %s; elapsed %s; device / elapsed %s; L5 device %s / "
              "elapsed %s = %s; lead %s; tail %s; unlabelled %s" % (
                  name, l5["rounds"], l5["device_ms"], l5["elapsed_ms"], l5["device_over_elapsed"], l5["l5_device_ms"],
                  l5["l5_elapsed_ms"], l5["l5_device_frac"], l5["lead"], l5["tail"], l5["unlabelled"]))
        for k, v in sorted(l5["labels"].items()):
            print("LABEL %s %s: ops %s, device %s ms, host %s ms (%s ranges); classes %s; top %s" % (
                name, k, v["ops"], v["ms"], v["host_ms"], v["ranges"], v["classes"], v["top_names"][:3]))


# ------------------------------------------------------------------ main
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
    if NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != a.pool:
        print("REFUSED: needs NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS == --pool (%d; the knob is %d)" % (
            a.pool, CEN.POOL_BLOCKS), file=sys.stderr)
        return 20
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, inputs, keep = [], None, {}, None, {}
    FORK, g0 = I3C.import_fork()
    gates.append(g0)
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    mem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
    L = len(model.layers)
    dims = dict(L=L, H=H, D=D, topk=TOPK, block_bytes=64 * D * 2 * 2 + 64 * 2)
    check_t = a.check_every == 1
    peq, ref_rd = None, None
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps, C %d)" % (a.batch, a.steps, BC.C_BASE + a.pool))
        fp_sh = (torch.zeros((a.batch, a.steps), dtype=torch.float32, device="cuda"),
                 torch.zeros((a.batch, a.steps), dtype=torch.float64, device="cuda"))
        try:
            sh_start = torch.cuda.memory_allocated() / 1024 ** 3
            out = BC.run_shipped(model, ids, a.steps, clock, fp=fp_sh)
            pa, pr = BC.gpu_peak()
            rw = {t for t, _ in out["rewarms"]}
            steady = BC.period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
            sh = dict(steady=steady, rates=BC.rates(a.batch, a.steps, out["periods"], steady),
                      rewarms=out["rewarms"], prefill_s=out["prefill_s"], gpu_peak_alloc_GB=pa,
                      gpu_peak_reserved_GB=pr, device_used_end_GB=out["device_used_end_GB"],
                      memory=dict(start_GB=round(sh_start, 3), peak_GB=pa, budget_GB=BUDGET_GB))
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            gates.append(I4A.mem_record("SH", "prefill + decode", pa))
            del out
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        mem["after_SH"] = BC.host_mem()
        BC.free()
        for arm in a.arms:
            if inputs is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: SH failed (every arm uses SH's tokens)"))
                continue
            if FORK is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: the fork did not import (C0_FORK)"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, journal=a.spec_journal_blocks, tu_gemm=a.tu_gemm, twin=a.spec_twin,
                       body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                       probe_every=a.probe_every, check_every=a.check_every, fork_store=a.fork_store,
                       prefill_hook=a.prefill_hook, profile_rounds=list(a.profile_rounds),
                       label_rounds=list(a.label_rounds), draft_marks=a.draft_marks == "on")
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_c(model, ids, inputs, a.steps, clock, opt, FORK, out_dir=a.out, probe=(arm == PROBED))
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(res["read"], L), fork=I3C.fork_record(res["read"]),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=res["read"].get("lean"))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = I4A.peak_gb(rec)
                if arm == PROBED:
                    rec.update(probe_record(res, sp.get(REF), ref_rd, a.probe_every))
                    rec["prof_after"] = prof_after(res["read"], a.probe_every, res["probe"]["profile"]["done"])
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(I4A.mem_record(arm, res["mem"], pa))
                gates.append(I4A.lean_gate(arm, dict(res["read"], K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=I4A.acct_record(res["read"]), state=res["state"])
                if arm == REF:
                    keep[REF] = cpu
                    ref_rd = res["read"]
                else:
                    g = peq_gate(arm, cpu, keep.get(REF), a.steps)
                    peq = bool(g[2])
                    gates.append(g)
                    gates.extend(probe_gates(arm, res["probe"], rec, res["read"], a.probe_every, L,
                                             a.draft_marks == "on"))
                    ref = sp.get(REF)
                    jr = ((ref or {}).get("phases_median") or {}).get("joint")
                    rr = ((ref or {}).get("rates") or {}).get("round_ms_median")
                    rp = rec["rates"].get("round_ms_median")
                    gates.append(("%s-PROF-NEUTRAL" % arm, False, bool(jr and jm),
                                  "steady joint median %s vs REF %s (ratio %s); round median %s vs REF %s; PROF-AFTER %s"
                                  % (jm, jr, round(jm / jr, 4) if (jm and jr) else None, rp, rr, rec["prof_after"])))
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch, peq)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                profile_rounds=a.profile_rounds, label_rounds=a.label_rounds, draft_marks=a.draft_marks,
                predictions=dict(stamp=PRED_STAMP_I4C, by_batch=PRED_I4C_BY_B), phase_sum_tol=PHASE_SUM_TOL,
                labels=dict(joint=JOINT_LAYER, draft=DRAFT_LAYER, l5=L5_LABELS, lead=LEAD), size_edges_us=SIZE_EDGES_US,
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB, dims=dims, host_mem=mem,
                base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-22s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-20s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; vs %.1f %s; "
              "round median %s ms; phases %s; draft step %s ms = %s SH; joint %s SH steps; calls %s; lean %s; tokens "
              "equal %.6f (first %s); peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"], r["draft_step_sh"],
                  r["joint_sh_steps"], r["calls"], r.get("lean"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        print("REPORT %s joint-pass phases (probe, L4PTF's marks) %s" % (name, r["joint_phases_fork"]))
        if name == PROBED:
            print_probe(name, r)
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4c: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
