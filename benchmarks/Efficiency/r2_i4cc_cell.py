"""Rung I4CC of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of the
COMPILED arms (benchmarks/Efficiency/r2_i4cc_compile.py: what is compiled, what stays eager, how the compile composes
with our capture):
  L4PTFDGUDC    L4PTFDGUD with its joint section's integer / mask / index / bit-copy helpers from torch.compile
  L4PTFDGUDKWC  L4PTFDGUDKW, the same
A NEW script on top of the I4S cell (benchmarks/Efficiency/r2_i4s_cell.py, imported, not edited: its pure parts, its
arms, its imports of the older cells). Every arm runs run_arm_c = the I4S cell's run_arm_s steps with the runner of
runner_choice(arm). Step script: benchmarks/Efficiency/r2_i4cc_step.sh (cells i4cm2, i4cca, i4ccb).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order of ARMS (L4PTFD first: every arm's EQF compares with its
tokens), each with its own shipped prefill, on the I4S cells' inputs (the same workload pin, S0, document rows, K 3, 512
steps, fused twin and body, graph, redirect, probe 16, store X 129, --tu-gemm natural). The I4S arms run as in the I4S
cell; the C arms build r2_i4cc_compile's runners (cc_backend "inductor"); dynamo's process settings once before the
first C arm (r2_i4cc_compile.configure: a recompile past the cache limit raises).
MUST GATES: the I4S cell's for every arm (a C arm takes the gates of its base arm: -RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN,
-RESID, -ATTN1, -FILL1, -LEAN, -MEM, -GRAPH, -COUNTS, -DGRAPH, -HOIST, the keep gates, the check gates in i4cm2, -EQF =
L4PTFD bit for bit) and the C arms' -COMPILE (r2_i4cc_compile.compile_gate: every section compiled, every helper
called, no graph break, no compile at or after the first capture, at most one compile per helper). KEEP REFERENCES:
L4PTFKW, else L4PTFDGKW; for L4PTFDGUDKWC also L4PTFDGUDKW (the same arm without the compile). L4PTFDGUDKW without a
keep reference in the job (i4ccb) reports its KEEP-EQ (REPORT, not MUST: i4sb passed it against L4PTFDGKW) and compares
its COUNTS with L4PTFD's without round_blocks.
REGISTERED PREDICTIONS PRED_I4C2_BY_B: every value None until the lead registers them with PRED_STAMP_I4C2 (a None key
is REPORT); the step script STOPS the timing cells (i4cca, i4ccb) while PRED_STAMP_I4C2 is None. The builder's MODEL:
the comment block above PRED_I4C2_BY_B. COMPILE TIME is reported apart (one-time setup, in round 0: outside the steady
set, as the captures, L61). EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_i4cc_compile as CC      # noqa: E402
import r2_i4s_cell as S4          # noqa: E402

IA, GC, DG, LN, GR, HC, KC, UC, DC, KP, SK = S4.IA, S4.GC, S4.DG, S4.LN, S4.GR, S4.HC, S4.KC, S4.UC, S4.DC, S4.KP, S4.SK
I3C, WA, SC, BC = S4.I3C, S4.WA, S4.SC, S4.BC
BASE = dict(CC.ARMS_C)                               # a C arm -> its base arm (the same arm without the compile)
ARMS = S4.ARMS + tuple(BASE)
GRAPH_ARMS = S4.GRAPH_ARMS + tuple(BASE)
STACK_ARMS = S4.STACK_ARMS + tuple(BASE)
KEEP_ARMS = S4.KEEP_ARMS + ("L4PTFDGUDKWC",)
NAMES = dict(L4PTFDGUDC=("pre", "write", "loc1", "loc2", "lists"),
             L4PTFDGUDKWC=("pre", "write", "loc1", "loc2", "kphys", "lists"))
TOPK = S4.TOPK
FORK_BLOB = S4.FORK_BLOB
STRONGEST = S4.STRONGEST
BUDGET_GB = S4.BUDGET_GB
USABLE_GB = S4.USABLE_GB
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4C2_BY_B (per batch) and PRED_STAMP_I4C2
# (`date -u`, L33) in a commit BEFORE i4cca / i4ccb (the step script stops while PRED_STAMP_I4C2 is None). Keys, each
# (low, high) unless noted; X = the C arm of the cell (L4PTFDGUDC at B 216, L4PTFDGUDKWC at B 318), REF = BASE[X]
# (L4PTFDGUD, L4PTFDGUDKW); every phase / rate on the steady set WITHOUT the capture rounds (L61):
#   I4C2_joint_saving  [ms per round] REF joint phase median - X's (the compiled helpers)
#   I4C2_joint_ratio   [x] X joint median / REF joint median
#   I4C2_draft_ratio   [x] X draft median / REF draft median (the drafts are not compiled: about 1)
#   I4C2_vs_ref        [x] X / REF median steady committed tok/s (same job)
#   I4C2_vs_D          [x] X / L4PTFD median steady committed tok/s (same job)
#   I4C2_ratio         [x SH] X median steady committed tok/s / SH's
#   I4C2_eq            [True] X-EQF passes (= L4PTFD bit for bit)
#   I4C2_above_best    [True / False] X's median above EVERY other arm's median in the job
#   I4C2_peak_gb       [GB] X's peak allocated GB, the prefill included (the MEM gate's value)
#   I4C2_pool_mb       [MB] X's shared graph pool (joint + draft graphs) after the run
#   I4C2_capture_s     [s] X's summed capture time (joint capture_s + draft dcapture_s, host clock)
#   I4C2_compile_s     [s] X's summed host seconds of the calls that compiled (r2_i4cc_compile COMPILE EVENTS)
# THE BUILDER'S MODEL (an input for the lead, not a registration). Labels: MEASURED (a record), DERIVED (arithmetic on
# MEASURED values), MODEL (needs an assumption, stated). Records under /mnt/beegfs/ojerbi/r2-runs/spec/.
#   BASIS. The I4C labelled joint pass of L4PTFD (eager, round 112; i4c_c3a_2195575_i4ca C 128, i4c_c4b_2195576_i4cb
#   C 73), every GPU op with its label and aten op (KS/tmp_sc/i4cc_ana/s1_chains.py, log KS/cpu-logs/i4cc_ana_s1.txt).
#   FUSION MODEL (KS/tmp_sc/i4cc_ana/s2_groups.py, log KS/cpu-logs/i4cc_ana_s2.txt): per label and layer, a run of small
#   ops between two barriers (sort / topk / searchsorted, cumsum, H2D, the fork, GEMM, the shipped kernels) or after a
#   write becomes ONE kernel of cost max(member) + 1 us. Saving per joint pass (C 128 / C 73, ms): write 5.170 / 5.781,
#   union 5.785 / 6.121, fetch 0.445 / 0.475, attn 0.603 / 0.629 (chain 1.765 / 2.298: NOT compiled, arena views);
#   the kept labels give 12.003 (C 128) and 12.525 (C 73, without fetch: the keep's fill stays eager); the replay gaps
#   add up to 3.0 ms (0.8 us per removed kernel). MINUS the hoist (I4U) already in REF: its ops sat in these labels;
#   I4S measured U on G at 1.768 (i4sa) / 1.805 (i4sb) ms per joint pass, and their device time is at most about 3.3 ms
#   (1,312 ops x 2.5 us). MODEL: 12.0 - (1.8 .. 3.3) = 8.7 .. 10.2 (C 128); 12.5 - (1.8 .. 3.3) = 9.2 .. 10.7 (C 73),
#   before Triton realism. RISKS (MODEL, down): Triton does not fuse a read after an indirect write of the same buffer
#   (the four rows of the write: at least one kernel per row and buffer, the CPU backend made one), a reduction and a
#   scatter are kernels of their own, Inductor's searchsorted may fall back to aten, a generated kernel can be slower
#   than the aten one at these sizes. CENTRAL 7 (C 128) / 7.5 (C 73) ms per round, BAND 3 .. 12 / 3 .. 13 ms.
#   C 128 B 216 (X = L4PTFDGUDC, REF = L4PTFDGUD). MEASURED i4sa (i4s_s3a_2195575_i4sa): REF round 342.471 ms, joint
#   196.881, draft 138.941, 1,828.1 tok/s = 0.5429 SH, L4PTFD 1,760.4 (REF / D 1.0385); pool 75.497 MB; capture
#   6.639 + 0.131 s; peak with the prefill 83.891 GB.
#     I4C2_joint_saving (3, 12); I4C2_joint_ratio 1 - (12 .. 3) / 196.9 = (0.939, 0.985)
#     I4C2_draft_ratio (0.99, 1.01) (the drafts are the same code)
#     I4C2_vs_ref 342.5 / (342.5 - 3 .. 12) = (1.0088, 1.0363), central 1.0209 -> (1.008, 1.037)
#     I4C2_vs_D 1.0385 x (1.0088 .. 1.0363) = (1.0476, 1.0762) -> (1.044, 1.080) with the replicate spread (0.3 %)
#     I4C2_ratio 0.5429 x (1.0088 .. 1.0363) = (0.5477, 0.5626) -> (0.545, 0.565) with the cross-job SH spread
#     I4C2_eq True; I4C2_above_best True (central +2.1 % over REF; the lower edge +0.9 % > the 0.3 % spread)
#     I4C2_peak_gb (83.85, 83.95): the prefill peak 83.891 (i4sa) stays above the decode peak
#     I4C2_pool_mb (40, 130): REF 75.497 MB; Inductor's temporaries replace the eager ones inside the joint graphs
#     I4C2_capture_s (5.5, 10): REF 6.77 s; the compiled wrappers add host work per captured call only
#     I4C2_compile_s (5, 180): a cold Inductor compile of 5 helpers (Triton, the worker pool); the CPU tests' C++
#       compile took 2-15 s per helper (KS/cpu-logs/i4cc_*); the earlier Inductor trial's preflight 95 s
#       (KS/analysis/evalfast/SCORING_compile.md, rci)
#   C 73 B 318 (X = L4PTFDGUDKWC, REF = L4PTFDGUDKW). MEASURED i4sb (i4s_s4b_2195576_i4sb): REF round 544.05 ms, joint
#   349.935, draft 185.126, 1,636.0 tok/s = 0.6381 SH, L4PTFD 1,533.8 (REF / D 1.0666); pool 117.441 MB; capture
#   7.251 + 0.133 s; peak 77.685 GB.
#     I4C2_joint_saving (3, 13); I4C2_joint_ratio 1 - (13 .. 3) / 349.9 = (0.963, 0.991)
#     I4C2_draft_ratio (0.99, 1.01)
#     I4C2_vs_ref 544.05 / (544.05 - 3 .. 13) = (1.0055, 1.0245), central 1.0140 -> (1.005, 1.025)
#     I4C2_vs_D 1.0666 x (1.0055 .. 1.0245) = (1.0725, 1.0927) -> (1.066, 1.099) with the replicate spread (0.6 %)
#     I4C2_ratio 0.6381 x (1.0055 .. 1.0245) = (0.6416, 0.6537) -> (0.638, 0.657)
#     I4C2_eq True; I4C2_above_best True (central +1.4 %; the lower edge +0.55 % is inside the 0.6 % spread: weak)
#     I4C2_peak_gb (77.60, 77.80); I4C2_pool_mb (60, 250); I4C2_capture_s (5.5, 10); I4C2_compile_s (5, 180)
PRED_STAMP_I4C2 = None
PRED_KEYS = ("I4C2_joint_saving", "I4C2_joint_ratio", "I4C2_draft_ratio", "I4C2_vs_ref", "I4C2_vs_D", "I4C2_ratio",
             "I4C2_eq", "I4C2_above_best", "I4C2_peak_gb", "I4C2_pool_mb", "I4C2_capture_s", "I4C2_compile_s")
PRED_I4C2_BY_B = {216: {k: None for k in PRED_KEYS}, 318: {k: None for k in PRED_KEYS}}


def log(msg):
    print("[r2_i4cc %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDG,L4PTFDGUD,L4PTFDGUDC",
                    help="a subset of %s, in this order (SH runs first; L4PTFD first)" % ",".join(ARMS))
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
                    help="0 = the probe rounds; 1 = every round, the union / fork checks and the keep's byte checks on "
                         "(the miniature)")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if a.arms != [x for x in ARMS if x in a.arms] or a.arms[0] != "L4PTFD":
        ap.error("--arms: L4PTFD first, then in the order %s (EQF compares with L4PTFD)" % ",".join(ARMS))
    for x, b in BASE.items():
        if x in a.arms and b not in a.arms:
            ap.error("--arms: %s needs its base arm %s (the REF of the predictions)" % (x, b))
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.check_every == 1 and a.batch > 64 and any(x in a.arms for x in KEEP_ARMS):
        ap.error("--check-every 1 with a keep arm: the keep's byte check needs a scratch store: --batch <= 64")
    if a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1:
        ap.error("--probe-every >= 2, --check-every >= 0, journal >= 1")
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(TOPK, a.spec):
        ap.error("--fork-store %d: below the bound 1 + (K - 1) x 64 = %d (0 = the bound)"
                 % (a.fork_store, I3C.KA.store_slots(TOPK, a.spec)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    return a


def runner_choice(arm: str, check: bool):
    """(the runner's kind, its keyword arguments): the I4S cell's for its arms; the C arms with cc_backend inductor."""
    if arm == "L4PTFDGUDC":
        return "SC", dict(cc_backend="inductor")
    if arm == "L4PTFDGUDKWC":
        return "SKC", dict(keep=True, kwin=True, kcheck=bool(check), cc_backend="inductor")
    return S4.runner_choice(arm, check)


def runner_class(kind: str, STF):
    if kind == "SC":
        return CC.compile_classes(STF)["L4PTFDGUDC"][1]
    if kind == "SKC":
        return CC.compile_classes(STF)["L4PTFDGUDKWC"][1]
    return S4.runner_class(kind, STF)


# ------------------------------------------------------------------ pure parts (CPU-tested)
def keep_ref_arm(arm: str, have) -> str:
    """The keep reference of a keep graph arm: L4PTFKW if it ran, else L4PTFDGKW; for L4PTFDGUDKWC also L4PTFDGUDKW
    (the same arm without the compile); None if none ran before `arm`."""
    order = ("L4PTFKW", "L4PTFDGKW") + (("L4PTFDGUDKW",) if arm == "L4PTFDGUDKWC" else ())
    for r in order:
        if r != arm and r in have and ARMS.index(r) < ARMS.index(arm):
            return r
    return None


def arm_gates(arm, rd, rec, cpu, keep, check: bool, B: int, pool: int, steps: int, L: int, pa):
    """The arm-specific gates after the common ones (module docstring): [(name, must, passed, detail)]. A C arm takes
    its base arm's gates (under its own name) and -COMPILE."""
    b = BASE.get(arm, arm)
    g = []
    if b in S4.GRAPH_ARMS:
        g.append(GC.mem_gate(arm, rec))
        g.append(GC.graph_gate(arm, rd, L))
        if b in S4.KEEP_ARMS:
            ref = keep_ref_arm(arm, keep)
            if ref is not None:
                g.append(KC.counts_gate(arm, cpu["counts"], keep[ref]["counts"], ref))
            else:
                d = keep.get("L4PTFD")
                g.append(KC.counts_gate(arm, cpu["counts"], d["counts"] if d else None, "L4PTFD",
                                        skip=("round_blocks",)))
        else:
            d = keep.get("L4PTFD")
            g.append(KC.counts_gate(arm, cpu["counts"], d["counts"] if d else None, "L4PTFD"))
    elif arm != "L4PTFD":
        g.append(IA.mem_record(arm, rec["mem_arena"], pa))
    if b in S4.STACK_ARMS:
        g.append(HC.dgraph_gate(arm, rd, L))
        g.append(UC.hoist_gate(arm, rd, L, check))
    if b in S4.KEEP_ARMS:
        g.append(DC.keep_gate(arm, rd, check))
        if check:
            g.append(DC.cap_gate(arm, rd))
        if b != "L4PTFKW":
            ref = keep_ref_arm(arm, keep)
            if ref is not None:
                g.append(S4.keep_eq_gate(arm, rd.get("keep"), keep[ref]["keep"], ref))
            elif arm == "L4PTFDGUDKW":
                g.append(("%s-KEEP-EQ" % arm, False, False, "REPORT: no keep reference (L4PTFKW or L4PTFDGKW) in this "
                          "job; i4sb passed it against L4PTFDGKW"))
            else:
                g.append(("%s-KEEP-EQ" % arm, True, False, "no keep reference in this job"))
        g.append(KC.keep_ref_gate(arm, rd.get("keep"), B, pool, steps))
    if arm in BASE:
        g.append(CC.compile_gate(arm, rd, L, NAMES[arm]))
    if arm != "L4PTFD":
        g.append(S4.eq_gate(arm, cpu, keep.get("L4PTFD"), steps))
    return g


def c_arm(sp):
    """X: the C arm of the job (L4PTFDGUDKWC before L4PTFDGUDC); None if neither ran."""
    for x in ("L4PTFDGUDKWC", "L4PTFDGUDC"):
        if x in (sp or {}):
            return x
    return None


def effective_wo_setup(rec, rd) -> float:
    """REPORT: every committed token / (wall_s - the arm's capture seconds - its compile seconds)."""
    g, d, c = rd.get("graph") or {}, rd.get("dgraph") or {}, rd.get("compile") or {}
    cs = float(g.get("capture_s") or 0.0) + float(d.get("dcapture_s") or 0.0) + float(c.get("compile_s") or 0.0)
    w = rd.get("wall_s")
    return round(sum(rd["count"]) / (w - cs), 1) if (w and w > cs) else None


def verdicts(sh, sp, B: int, eqf=None):
    """PRED_I4C2_BY_B[B] against X = c_arm(sp) and REF = BASE[X]: [(name, verdict, detail)]; None registered gives
    REPORT; NOT_RUN without a C arm. eqf: X-EQF's passed flag."""
    P = PRED_I4C2_BY_B.get(B) or {k: None for k in PRED_KEYS}
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))

    def flag(name, got, want, detail):
        out.append((name, "REPORT" if (want is None or got is None) else ("CONFIRMED" if got == bool(want) else
                                                                           "REFUTED"), detail))

    def med(r):
        return (r or {}).get("rates", {}).get("median_tok_s")

    def ph(r, name):
        return ((r or {}).get("phases_median") or {}).get(name)

    def ratio(a, b):
        return (a / b) if (a is not None and b) else None

    def diff(a, b):
        return (a - b) if (a is not None and b is not None) else None
    sp = sp or {}
    d = sp.get("L4PTFD")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    for name, arm in (("I4S-VS-D", "L4PTFDGUD"), ("I4SKW-VS-D", "L4PTFDGUDKW"), ("I4G-VS-D", "L4PTFDG")):
        if sp.get(arm) is not None:                                    # the I4S arms, re-measured
            v = ratio(med(sp[arm]), med(d))
            out.append((name, "REPORT", "%s (median tok/s / L4PTFD's, this job)" % (None if v is None else round(v, 4))))
    X = c_arm(sp)
    if X is None:
        out.append(("I4C2-RUN", "NOT_RUN", "no C arm ran"))
        return out
    x, r = sp[X], sp.get(BASE[X])
    gr = x.get("graph") or {}
    band("I4C2-JOINT-SAVING", diff(ph(r, "joint"), ph(x, "joint")), P.get("I4C2_joint_saving"))
    band("I4C2-JOINT-RATIO", ratio(ph(x, "joint"), ph(r, "joint")), P.get("I4C2_joint_ratio"))
    band("I4C2-DRAFT-RATIO", ratio(ph(x, "draft"), ph(r, "draft")), P.get("I4C2_draft_ratio"))
    band("I4C2-VS-REF", ratio(med(x), med(r)), P.get("I4C2_vs_ref"))
    band("I4C2-VS-D", ratio(med(x), med(d)), P.get("I4C2_vs_D"))
    band("I4C2-RATIO", ratio(med(x), shm), P.get("I4C2_ratio"))
    band("I4C2-PEAK-GB", IA.peak_gb(x), P.get("I4C2_peak_gb"))
    band("I4C2-POOL-MB", (gr["pool_bytes"] / 1e6) if gr.get("pool_bytes") is not None else None, P.get("I4C2_pool_mb"))
    band("I4C2-CAPTURE-S", S4.capture_s(x), P.get("I4C2_capture_s"))
    band("I4C2-COMPILE-S", (x.get("cc") or {}).get("compile_s"), P.get("I4C2_compile_s"))
    flag("I4C2-EQ", eqf, P.get("I4C2_eq"), "%s-EQF passed %s; registered %s" % (X, eqf, P.get("I4C2_eq")))
    others = {a: med(v) for a, v in sp.items() if a != X and med(v)}
    above = (all(med(x) > v for v in others.values()) if (med(x) and others) else None)
    flag("I4C2-ABOVE-BEST", above, P.get("I4C2_above_best"), "%s %s vs %s tok/s (not above every other arm of the "
         "job); registered %s" % (X, med(x), others, P.get("I4C2_above_best")))
    out.append(("I4C2-ARM", "REPORT", "X = %s, REF = %s (REF ran: %s)" % (X, BASE[X], r is not None)))
    out.append(("I4C2-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, x.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_c(model, ids, inputs, steps: int, clock, opt, FORK, arm: str):
    """The I4S cell's run_arm_s line for line with the runner of runner_choice(arm, check) (the stacked and the C arms:
    the I4G graph module's backend through the draft-graph module, as L4PTFDGUD); dynamo's settings before a C arm."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, check)
    Runner = runner_class(kind, STF)
    if kind in ("SC", "SKC"):
        CC.configure()
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
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    gk = dict(GR=GR) if kind in ("S", "SK", "SC", "SKC") else {}
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                    check_every=ce, union_check=check, union_stats=check, phase_marks=True, **kw, **gk)
    if check:
        runner.vs.fa_check = {}                                        # CHECK-ONLY: the "F" / "RA" probes
    runner.run(steps)
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
               compile=I3C._compile_report(FORK), state=IA.state_checksum(runner), runner_kind=kind, runner_kw=kw)
    runner.detach()
    del runner, rs, cache, arena
    return res


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
    if any(x in a.arms for x in BASE) and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        print("REFUSED: a C arm needs TORCHINDUCTOR_CACHE_DIR (the step script sets it; never the home cache)",
              file=sys.stderr)
        return 20
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, inputs, keep, rds, caps = [], None, {}, None, {}, {}, {}
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
    bb = KP.block_bytes(D)
    check_t = a.check_every == 1
    eqf = {}
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
            gates.append(IA.mem_record("SH", "prefill + decode", pa))
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
                       prefill_hook=a.prefill_hook)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_c(model, ids, inputs, a.steps, clock, opt, FORK, arm)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), graph=rd.get("graph"), dgraph=rd.get("dgraph"),
                           hoist=rd.get("hoist"), capture_rounds=GC.capture_rounds(rd), runner=res["runner_kind"],
                           runner_kw=res["runner_kw"], cc=rd.get("compile"))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = IA.peak_gb(rec)
                rec["effective_wo_capture_tok_s"] = HC.effective_wo_capture(rec, rd)
                rec["effective_wo_setup_tok_s"] = effective_wo_setup(rec, rd)
                rec["store_record"] = DC.acct_summary(rd.get("keep"), bb)
                sp[arm] = rec
                rds[arm] = rd
                if arm in GRAPH_ARMS:
                    caps[arm] = HC.capture_rounds_x(rd)
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (rd["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(IA.lean_gate(arm, dict(rd, K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=IA.acct_record(rd), state=res["state"], counts=GC.counts_record(rd), keep=rd.get("keep"))
                if arm == "L4PTFD":
                    gates.append(IA.mem_record(arm, res["mem"], pa))
                ag = arm_gates(arm, rd, rec, cpu, keep, check_t, a.batch, a.pool, a.steps, L, pa)
                gates.extend(ag)
                if arm != "L4PTFD":
                    eqf[arm] = next((bool(x[2]) for x in ag if x[0] == "%s-EQF" % arm), None)
                keep[arm] = cpu
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    xr = HC.excluded_rounds(caps)                                      # THE L61 FIX: the same rounds for every arm
    for arm, rec in sp.items():
        HC.restrict(rec, rds[arm], sh, a.batch, a.spec, a.probe_every, xr)
    X = c_arm(sp)
    preds = verdicts(sh, sp, a.batch, eqf.get(X) if X else None)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_I4C2, by_batch=PRED_I4C2_BY_B),
                section_graph=dict(skipped_T=GR.SKIPPED_T, skipped_vs=GR.SKIPPED_VS, eager_ok=GR.EAGER_OK,
                                   hoist_delta=[SK.expected_delta_s(False, H, a.batch, 0)["H"],
                                                SK.expected_delta_s(False, H, a.batch, 1)["H"]]),
                draft_graph=dict(deager_ok=DG.DEAGER_OK, expected_delta=DG.expected_delta(a.spec_graph == "on")),
                keep=dict(block_bytes_head=bb, keep_ref=KC.KEEP_REF.get((a.batch, a.pool, a.steps))),
                compile=dict(helpers=CC.HELPERS, names=NAMES, base=BASE, torch=torch.__version__,
                             inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                             kernel_cache=os.environ.get("TRITON_CACHE_DIR")),
                mem_must_GB=GC.MEM_MUST_GB, capture_rounds=caps, steady_excluded_rounds=xr, c_arm=X,
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-22s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-17s %s: %s" % p)
    print("REPORT L61 capture rounds %s; excluded from every arm's steady set: %s" % (caps, xr))
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s (without capture s %s; "
              "without capture and compile s %s); vs SH %s; vs %.1f %s; round median %s ms; phases %s; draft step %s ms "
              "= %s SH; joint %s SH steps; steady rounds %s (excluded %s; with the capture rounds: median %s mean %s, "
              "phases %s); calls %s; lean %s; hoist %s; tokens equal %.6f (first %s); peak %s GB (with the prefill %s "
              "GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r.get("effective_wo_capture_tok_s"), r.get("effective_wo_setup_tok_s"),
                  r["vs_sh"], STRONGEST, r["vs_strongest"], r["rates"].get("round_ms_median"), r["phases_median"],
                  r["draft_step_ms"], r["draft_step_sh"], r["joint_sh_steps"], r.get("steady_rounds_used"),
                  r.get("steady_excluded_rounds"), (r.get("rates_incl_capture") or {}).get("median_tok_s"),
                  (r.get("rates_incl_capture") or {}).get("mean_tok_s"), r.get("phases_median_incl_capture"),
                  r["calls"], r.get("lean"), r.get("hoist"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        if r.get("graph"):
            gr = r["graph"]
            print("REPORT %s joint graphs: captures %s, warm %s, replays %s, eager %s %s; pool %s MB; static %s MB; "
                  "capture %s s; capture rounds (round, joint ms) %s" % (
                      name, gr.get("captures"), gr.get("warm"), gr.get("replays"), gr.get("eager"),
                      gr.get("eager_why"), round(gr["pool_bytes"] / 1e6, 3) if gr.get("pool_bytes") is not None
                      else None, round(gr.get("static_bytes", 0) / 1e6, 3), gr.get("capture_s"),
                      r.get("capture_rounds")))
        if r.get("dgraph"):
            dr = r["dgraph"]
            print("REPORT %s draft graphs: keys %s, captures %s, capture passes %s, replay passes %s, eager %s %s; "
                  "capture %s s; capture rounds %s" % (
                      name, dr.get("keys"), dr.get("captures"), dr.get("capture_passes"), dr.get("replay_passes"),
                      dr.get("eager"), dr.get("eager_why"), dr.get("dcapture_s"), dr.get("capture_rounds")))
        if r.get("cc"):
            c = r["cc"]
            print("REPORT %s compile: backend %s; compile %s s (one-time, round 0; outside the steady set); events %s; "
                  "calls %s; sections %s; fallback %s; dynamo %s" % (
                      name, c.get("backend"), c.get("compile_s"),
                      [(e["name"], e["joint_pass"], e["layer"], e["s"]) for e in c.get("events") or []], c.get("calls"),
                      c.get("sections"), c.get("fallback"), c.get("dynamo")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4cc: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
