"""Rung I4S of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of the
STACKED arms (benchmarks/Efficiency/r2_i4s_stack.py: the class stack, what is captured where):
  L4PTFDGUD    L4PTFD + I4G joint graphs + I4U hoist (inside layer 0's joint graph) + I4H draft graphs
  L4PTFDGUDKW  the same + I4D's store keep and window loads from the kept store, I4K's way (the keep inside the graphs)
A NEW script: the older cells are not edited (their pure functions are imported: r2_i4h_cell (the base: the L61 fix,
DGRAPH, EQF), r2_i4g_cell (GRAPH, COUNTS, MEM), r2_i4k_cell / r2_i4d_cell (KEEP, CAP, KEEP-EQ, KEEP-REF), r2_i4u_cell
(HOIST)). Every arm runs run_arm_s = r2_i4h_cell.run_arm_h's steps with the runner class of runner_choice(arm).
Step script: benchmarks/Efficiency/r2_i4s_step.sh (cells i4sm, i4sa, i4sb).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order of ARMS (a subset; L4PTFD first: every arm's EQF compares
with its tokens), each with its own shipped prefill into the chain arena, on the inputs of i4ha / i4kb (the same workload
pin, S0, document rows, K 3, 512 steps, fused twin and body, graph, redirect, probe 16, store X 129, --tu-gemm natural):
  SH           the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTFD       I4A (r2_i4a_lean.TentRunnerFD): the base rung, re-measured (the EQF reference)
  L4PTFDG      I4G (r2_i4g_graph.TentRunnerFDG): the reference of L4PTFDGUD
  L4PTFKW      I4D (r2_i4d_keep.TentRunnerFK, keep + kwin; kcheck in the check cell): the eager keep (KEEP-EQ reference)
  L4PTFDGKW    I4K (r2_i4k_combo.TentRunnerFDGK): the reference of L4PTFDGUDKW
  L4PTFDGUD    I4S: r2_i4s_stack's TentRunnerFDGUD
  L4PTFDGUDKW  I4S: r2_i4s_stack's TentRunnerFDGUDK (keep + kwin; kcheck in the check cell)
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN, -RESID, -ATTN1, -FILL1, -LEAN; with
--check-every 1 (i4sm) -PRESENT, -VIS, -VB, -TAIL, -CHAIN, -IDX, -RA; with --require-stress -STRESS, -FSTRESS; every
arm but L4PTFD -EQF (committed tokens, check-round fingerprints, acceptance record, state checksums = L4PTFD's, bit for
bit); the graph arms -GRAPH (r2_i4g_cell.graph_gate: with the hoist the expected skipped counts include the hoist's,
r2_i4s_stack.expected_delta_s), -COUNTS (non-keep against L4PTFD; keep against the keep reference L4PTFKW, else
L4PTFDGKW, else L4PTFD without the meter's round_blocks), -MEM (peak allocated with the prefill <= 84.0 GB); the draft
arms (L4PTFDGUD, L4PTFDGUDKW) -DGRAPH (r2_i4h_cell.dgraph_gate) and -HOIST (r2_i4u_cell.hoist_gate: one hoist per joint
pass, L hoisted layers per joint pass; check cells: is_next recounted 0); the keep arms -KEEP, with --check-every 1
-CAP; L4PTFDGKW and L4PTFDGUDKW -KEEP-EQ (the keep record per pass = the keep reference's). REPORT: SH / L4PTFD /
L4PTFKW MEM; <keep arm>-KEEP-REF; the phases and rates with and without the L61 capture rounds; the graphs; the pools.
L61 FIX (r2_i4h_cell): the capture rounds of every graph arm are excluded from EVERY spec arm's steady set.
REGISTERED PREDICTIONS PRED_I4S_BY_B: every value None until the lead registers them with PRED_STAMP_I4S (a None key is
REPORT); the step script STOPS the timing cells (i4sa, i4sb) while PRED_STAMP_I4S is None. The builder's MODEL: the
comment block above PRED_I4S_BY_B. EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an
arm.
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
import r2_i4h_cell as HC          # noqa: E402
import r2_i4k_cell as KC          # noqa: E402
import r2_i4s_stack as SK         # noqa: E402
import r2_i4u_cell as UC          # noqa: E402

IA, GC, DG, LN, GR = HC.IA, HC.GC, HC.DG, HC.LN, HC.GR
DC, KP, CB = KC.DC, KC.KP, KC.CB
I3C = IA.I3C
WA, SC, BC = IA.WA, IA.SC, IA.BC
ARMS = ("L4PTFD", "L4PTFDG", "L4PTFKW", "L4PTFDGKW", "L4PTFDGUD", "L4PTFDGUDKW")
GRAPH_ARMS = ("L4PTFDG", "L4PTFDGKW", "L4PTFDGUD", "L4PTFDGUDKW")
STACK_ARMS = ("L4PTFDGUD", "L4PTFDGUDKW")          # the draft graphs and the hoist
KEEP_ARMS = ("L4PTFKW", "L4PTFDGKW", "L4PTFDGUDKW")
REF = dict(L4PTFDGUD="L4PTFDG", L4PTFDGUDKW="L4PTFDGKW")   # a stacked arm's reference: the same arm without U and D
TOPK = IA.TOPK
FORK_BLOB = IA.FORK_BLOB
STRONGEST = IA.STRONGEST
BUDGET_GB = IA.BUDGET_GB
USABLE_GB = IA.USABLE_GB
MEM_MUST_GB = GC.MEM_MUST_GB       # <arm>-MEM: peak allocated with the prefill (C 128 B 216: REF 83.891 GB, i4ga)
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4S_BY_B (per batch) and PRED_STAMP_I4S
# (`date -u`, L33) in a commit BEFORE i4sa / i4sb (the step script stops while PRED_STAMP_I4S is None). Keys, each
# (low, high) unless noted; X = the stacked arm of the cell (L4PTFDGUD at B 216, L4PTFDGUDKW at B 318), REF = REF[X]
# (L4PTFDG, L4PTFDGKW); every phase / rate on the steady set WITHOUT the capture rounds (L61):
#   I4S_joint_saving   [ms per round] REF joint phase median - X's (the hoist on top of the graphs)
#   I4S_draft_saving   [ms per round] REF draft phase median - X's (the draft graphs)
#   I4S_joint_ratio    [x] X joint median / REF joint median
#   I4S_draft_ratio    [x] X draft median / REF draft median
#   I4S_vs_ref         [x] X / REF median steady committed tok/s (same job)
#   I4S_vs_D           [x] X / L4PTFD median steady committed tok/s (same job)
#   I4S_ratio          [x SH] X median steady committed tok/s / SH's
#   I4S_eq             [True] X-EQF passes (= L4PTFD bit for bit)
#   I4S_above_best     [True / False] the stack falsifier: X's median above EVERY other arm's median in the job
#   I4S_peak_gb        [GB] X's peak allocated GB, the prefill included (the MEM gate's value)
#   I4S_pool_mb        [MB] X's shared graph pool (joint + draft graphs) after the run
#   I4S_capture_s      [s] X's summed capture time (joint graphs capture_s + draft graphs dcapture_s, host clock)
# THE BUILDER'S MODEL (an input for the lead, not a registration). Labels: MEASURED (a record), DERIVED (arithmetic on
# MEASURED values), INFERENCE. Records in /mnt/beegfs/ojerbi/r2-runs/spec/<run>.step/stdout.txt.
#   ADDITIVITY. U changes only the joint pass, D only the one-row passes: D adds to G + U (disjoint code). U and G
#   OVERLAP: U's removed ops (1,312 per joint pass, r2_i4u_hoist.saved_per_pass(32)) sit inside G's captured section
#   except 3 per layer in the residency (~act, the tent test), which stays eager and un-hoisted under the graphs (93).
#   Under G a removed op no longer costs a host launch; it costs its device time plus the graph's per-node gap. So U on
#   top of G saves about 1,219 x the small-op device time (2.0-3.1 us, i4ua's model: I4C c3a / c4b small-kernel class)
#   = 2.4-3.8 ms per joint pass, NOT U's eager 4.09 ms (i4ua) / 4.46 ms (i4ub). The keep (K) and U touch different ops
#   of the section (K: the store fill; U: the per-layer recomputes): additive (INFERENCE).
#   D at C 128: MEASURED i4ha (i4h_j3a_2195575_i4ha, L61 steady set): L4PTFDG draft 143.342 -> L4PTFDGD 138.989 ms,
#   I4H-DRAFT-SAVING 4.353 ms, L4PTFDGD / L4PTFDG 1.0108, joint ratio 1.0022, pool 75.497 MB (joint + draft), draft
#   capture 0.128 s. D at C 73: not measured (no i4hb record); I4H's model 1.5-6.0 ms, central 3.5 (its C 128 central
#   4.0 met the measured 4.35).
#   C 128 B 216 (X = L4PTFDGUD, REF = L4PTFDG). MEASURED L4PTFDG round 348.601 / 348.453 / 349.068 / 350.27 ms (i4ha,
#   i4ga x3a, i4ka k3a, r4a), joint 198.748 / 198.694 / 198.811 / 199.535, draft 143.342 / 143.364 / 143.396 / 144.374;
#   L4PTFDG / L4PTFD 1.0199 / 1.0209 / 1.0195 / 1.0222; L4PTFDG / SH 0.5322 / 0.5333 / 0.5303 / 0.5331. Saving per
#   round: U 3.0 (1.5-4.5) + D 4.35 (3.0-5.7: the i4ha value +- the job-to-job spread of a phase median) = 7.35
#   (4.5-10.2).
#     I4S_joint_saving (1.5, 4.5); I4S_joint_ratio 1 - (4.5 .. 1.5) / 198.9 = (0.977, 0.993)
#     I4S_draft_saving (3.0, 5.7); I4S_draft_ratio (143.3 - 5.7 .. 3.0) / 143.3 = (0.960, 0.979)
#     I4S_vs_ref 348.6 / (348.6 - 4.5 .. 10.2) = (1.013, 1.030), central 1.0215
#     I4S_vs_D 1.0206 x (1.013 .. 1.030) = (1.034, 1.051) -> (1.030, 1.054) with the replicate spread (0.3 %)
#     I4S_ratio 0.532 x (1.013 .. 1.030) = (0.539, 0.548) -> (0.535, 0.552) with the cross-job SH spread
#     I4S_eq True; I4S_above_best True (central +2.1 % over L4PTFDG, replicate spread 0.3 %)
#     I4S_peak_gb (83.85, 83.95): the prefill peak 83.891 (i4ga, i4ka, i4ha) is above the decode peak 82.567 + the pool
#     I4S_pool_mb (60, 130): i4ha's shared pool 75.497 MB + the hoist's temporaries inside layer 0's graph (< 1 MB); the
#       hoist's buffers are persistent (outside the pool, < 1 MB at B 216)
#     I4S_capture_s (5.5, 10): joint 6.2-6.8 s (i4ga, i4ka, i4ha) + draft 0.128 s (i4ha)
#   C 73 B 318 (X = L4PTFDGUDKW, REF = L4PTFDGKW). MEASURED L4PTFDGKW round 551.898 / 547.822 ms (i4kb k4b, r3b), joint
#   352.62 / 350.189, draft 190.208 / 188.34; L4PTFDGKW / L4PTFD 1.0564 / 1.0531; / SH 0.6307 / 0.6306. Saving per
#   round: U 3.0 (1.5-4.5) + D 3.5 (1.5-6.0) = 6.5 (3.0-10.5).
#     I4S_joint_saving (1.5, 4.5); I4S_joint_ratio 1 - (4.5 .. 1.5) / 351.4 = (0.987, 0.996)
#     I4S_draft_saving (1.5, 6.0); I4S_draft_ratio (189.3 - 6.0 .. 1.5) / 189.3 = (0.968, 0.992)
#     I4S_vs_ref 549.9 / (549.9 - 3.0 .. 10.5) = (1.005, 1.020), central 1.012
#     I4S_vs_D 1.0548 x (1.005 .. 1.020) = (1.060, 1.076) -> (1.055, 1.079)
#     I4S_ratio 0.6306 x (1.005 .. 1.020) = (0.634, 0.643) -> (0.632, 0.645)
#     I4S_eq True; I4S_above_best True (central +1.2 % over L4PTFDGKW; replicate spread 0.6 %: the lower edge is close)
#     I4S_peak_gb (77.60, 77.80): the prefill peak 77.685 (i4kb)
#     I4S_pool_mb (80, 250): joint pool 29.36 MB (i4kb) + i4ha's draft part (75.5 - 14.7 = 60.8 MB) x 318 / 216
#     I4S_capture_s (5.5, 10): joint 7.1-7.2 s (i4kb) + draft (0.13 s at B 216)
#   RISKS. (1) U on top of G may save less than its device time if the graph already overlaps the small kernels (the
#   floor 1.5 ms). (2) D at C 73 is a model (no i4hb record). (3) The stack adds no eager launch (CPU:
#   test_i4s_eager_ops), and the shared pool grows by the draft graphs (i4ha: 75.5 MB); the C 128 margin is 0.36 GB
#   against the usable 84.25 GB (the pool lives in the decode, below the prefill peak). (4) The capture rounds are
#   excluded (L61); the effective rate keeps them (REPORT).
PRED_STAMP_I4S = "2026-10-29T22:50:04Z"
PRED_KEYS = ("I4S_joint_saving", "I4S_draft_saving", "I4S_joint_ratio", "I4S_draft_ratio", "I4S_vs_ref", "I4S_vs_D",
             "I4S_ratio", "I4S_eq", "I4S_above_best", "I4S_peak_gb", "I4S_pool_mb", "I4S_capture_s")
# The lead's registration (before any I4S cell ran): the builder's MODEL bands (cell comment block above).
PRED_I4S_BY_B = {
    216: dict(I4S_joint_saving=(1.5, 4.5), I4S_draft_saving=(3.0, 5.7), I4S_joint_ratio=(0.977, 0.993),
              I4S_draft_ratio=(0.960, 0.979), I4S_vs_ref=(1.013, 1.030), I4S_vs_D=(1.030, 1.054),
              I4S_ratio=(0.535, 0.552), I4S_eq=True, I4S_above_best=True, I4S_peak_gb=(83.85, 83.95),
              I4S_pool_mb=(60, 130), I4S_capture_s=(5.5, 10)),
    318: dict(I4S_joint_saving=(1.5, 4.5), I4S_draft_saving=(1.5, 6.0), I4S_joint_ratio=(0.987, 0.996),
              I4S_draft_ratio=(0.968, 0.992), I4S_vs_ref=(1.005, 1.020), I4S_vs_D=(1.055, 1.079),
              I4S_ratio=(0.632, 0.645), I4S_eq=True, I4S_above_best=True, I4S_peak_gb=(77.60, 77.80),
              I4S_pool_mb=(80, 250), I4S_capture_s=(5.5, 10)),
}


def log(msg):
    print("[r2_i4s %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDG,L4PTFDGUD", help="a subset of %s, in this order (SH runs first; "
                                                                        "L4PTFD first)" % ",".join(ARMS))
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
        ap.error("--arms: L4PTFD first, then in the order %s (EQF compares with L4PTFD; KEEP-EQ with the keep reference)"
                 % ",".join(ARMS))
    if "L4PTFDGUDKW" in a.arms and not ("L4PTFKW" in a.arms or "L4PTFDGKW" in a.arms):
        ap.error("--arms: L4PTFDGUDKW needs a keep reference (L4PTFKW or L4PTFDGKW) for KEEP-EQ and COUNTS")
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
    """(the runner's kind, its keyword arguments): the levers' runners as their cells build them; the keep arms with
    keep, kwin and (check cells) kcheck, no accounting."""
    keep = dict(keep=True, kwin=True, kcheck=bool(check))
    return dict(L4PTFD=("FD", {}), L4PTFDG=("FDG", {}), L4PTFKW=("FK", keep), L4PTFDGKW=("FDGK", keep),
                L4PTFDGUD=("S", {}), L4PTFDGUDKW=("SK", keep))[arm]


def runner_class(kind: str, STF):
    lean = LN.lean_classes(STF)
    return dict(FD=lambda: lean[1], FDG=lambda: GR.graph_classes(STF, lean)[1], FK=lambda: KP.keep_classes(STF)[1],
                FDGK=lambda: CB.combo_classes(STF)[3], S=lambda: SK.stack_classes(STF)["L4PTFDGUD"][1],
                SK=lambda: SK.stack_classes(STF)["L4PTFDGUDKW"][1])[kind]()


# ------------------------------------------------------------------ pure parts (CPU-tested)
def eq_gate(arm, f, fd, steps: int):
    """<arm>-EQF (MUST): r2_i4h_cell.eq_gate (r2_i4a_cell.eq_gate against L4PTFD's record, bit for bit)."""
    return HC.eq_gate(arm, f, fd, steps)


def keep_ref_arm(arm: str, have) -> str:
    """The keep reference of a keep graph arm: L4PTFKW if it ran, else (for L4PTFDGUDKW) L4PTFDGKW; None if none."""
    for r in ("L4PTFKW", "L4PTFDGKW"):
        if r != arm and r in have and ARMS.index(r) < ARMS.index(arm):
            return r
    return None


def keep_eq_gate(arm, kp, kp_ref, ref: str):
    """<arm>-KEEP-EQ (MUST): the keep record per pass (r2_i4k_cell.KEEP_EQ_KEYS present in either record) = ref's."""
    if not kp_ref:
        return ("%s-KEEP-EQ" % arm, True, False, "no %s keep record in this job" % ref)
    if not kp:
        return ("%s-KEEP-EQ" % arm, True, False, "no keep record")
    keys = [k for k in KC.KEEP_EQ_KEYS if k in kp or k in kp_ref]
    bad = [k for k in keys if kp.get(k) != kp_ref.get(k)]
    return ("%s-KEEP-EQ" % arm, True, not bad and (kp.get("listed_total") or 0) > 0,
            "per pass %s compared with %s's; differ %s; totals listed %s / %s, fetched %s / %s, window hits %s / %s"
            % (keys, ref, bad, kp.get("listed_total"), kp_ref.get("listed_total"), kp.get("fetched_total"),
               kp_ref.get("fetched_total"), kp.get("win_hits_total"), kp_ref.get("win_hits_total")))


def arm_gates(arm, rd, rec, cpu, keep, check: bool, B: int, pool: int, steps: int, L: int, pa):
    """The arm-specific gates after the common ones (module docstring): [(name, must, passed, detail)]."""
    g = []
    if arm in GRAPH_ARMS:
        g.append(GC.mem_gate(arm, rec))
        g.append(GC.graph_gate(arm, rd, L))
        if arm in KEEP_ARMS:
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
    if arm in STACK_ARMS:
        g.append(HC.dgraph_gate(arm, rd, L))
        g.append(UC.hoist_gate(arm, rd, L, check))
    if arm in KEEP_ARMS:
        g.append(DC.keep_gate(arm, rd, check))
        if check:
            g.append(DC.cap_gate(arm, rd))
        if arm != "L4PTFKW":
            ref = keep_ref_arm(arm, keep)
            if ref is not None:
                g.append(keep_eq_gate(arm, rd.get("keep"), keep[ref]["keep"], ref))
            elif arm == "L4PTFDGUDKW":
                g.append(("%s-KEEP-EQ" % arm, True, False, "no keep reference (L4PTFKW or L4PTFDGKW) in this job"))
        g.append(KC.keep_ref_gate(arm, rd.get("keep"), B, pool, steps))
    if arm != "L4PTFD":
        g.append(eq_gate(arm, cpu, keep.get("L4PTFD"), steps))
    return g


def stacked_arm(sp):
    """X: the stacked arm of the job (L4PTFDGUDKW before L4PTFDGUD); None if neither ran."""
    for x in ("L4PTFDGUDKW", "L4PTFDGUD"):
        if x in (sp or {}):
            return x
    return None


def capture_s(rec):
    """The summed capture time of an arm's joint and draft graphs (s, host clock); None without graphs."""
    g, d = (rec or {}).get("graph") or {}, (rec or {}).get("dgraph") or {}
    if not g and not d:
        return None
    return float(g.get("capture_s") or 0.0) + float(d.get("dcapture_s") or 0.0)


def verdicts(sh, sp, B: int, eqf=None):
    """PRED_I4S_BY_B[B] against X = stacked_arm(sp) and REF[X]: [(name, verdict, detail)]; None registered gives REPORT;
    NOT_RUN without a stacked arm. eqf: X-EQF's passed flag."""
    P = PRED_I4S_BY_B.get(B) or {k: None for k in PRED_KEYS}
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
    for name, arm in (("I4G-VS-D", "L4PTFDG"), ("I4DW-VS-D", "L4PTFKW"), ("I4K-VS-D", "L4PTFDGKW")):
        if sp.get(arm) is not None:                                    # the single levers, re-measured
            v = ratio(med(sp[arm]), med(d))
            out.append((name, "REPORT", "%s (median tok/s / L4PTFD's, this job)" % (None if v is None else round(v, 4))))
    X = stacked_arm(sp)
    if X is None:
        out.append(("I4S-RUN", "NOT_RUN", "no stacked arm ran"))
        return out
    x, r = sp[X], sp.get(REF[X])
    gr = x.get("graph") or {}
    band("I4S-JOINT-SAVING", diff(ph(r, "joint"), ph(x, "joint")), P.get("I4S_joint_saving"))
    band("I4S-DRAFT-SAVING", diff(ph(r, "draft"), ph(x, "draft")), P.get("I4S_draft_saving"))
    band("I4S-JOINT-RATIO", ratio(ph(x, "joint"), ph(r, "joint")), P.get("I4S_joint_ratio"))
    band("I4S-DRAFT-RATIO", ratio(ph(x, "draft"), ph(r, "draft")), P.get("I4S_draft_ratio"))
    band("I4S-VS-REF", ratio(med(x), med(r)), P.get("I4S_vs_ref"))
    band("I4S-VS-D", ratio(med(x), med(d)), P.get("I4S_vs_D"))
    band("I4S-RATIO", ratio(med(x), shm), P.get("I4S_ratio"))
    band("I4S-PEAK-GB", IA.peak_gb(x), P.get("I4S_peak_gb"))
    band("I4S-POOL-MB", (gr["pool_bytes"] / 1e6) if gr.get("pool_bytes") is not None else None, P.get("I4S_pool_mb"))
    band("I4S-CAPTURE-S", capture_s(x), P.get("I4S_capture_s"))
    flag("I4S-EQ", eqf, P.get("I4S_eq"), "%s-EQF passed %s; registered %s" % (X, eqf, P.get("I4S_eq")))
    others = {a: med(v) for a, v in sp.items() if a != X and med(v)}
    above = (all(med(x) > v for v in others.values()) if (med(x) and others) else None)
    flag("I4S-ABOVE-BEST", above, P.get("I4S_above_best"), "%s %s vs %s tok/s (the stack falsifier: not above every "
         "other arm of the job); registered %s" % (X, med(x), others, P.get("I4S_above_best")))
    out.append(("I4S-ARM", "REPORT", "X = %s, REF = %s (REF ran: %s)" % (X, REF[X], r is not None)))
    out.append(("I4S-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, x.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_s(model, ids, inputs, steps: int, clock, opt, FORK, arm: str):
    """r2_i4h_cell.run_arm_h's steps, line for line, with the runner of runner_choice(arm, check) (the stacked arms:
    r2_i4s_stack; their draft graphs' backend: r2_i4h_graph.make_backend(r2_i4g_graph), as L4PTFDGD), the state
    checksum after the run (EQF) and read()'s graph / dgraph / hoist / keep reports (before detach: the pool exists)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, check)
    Runner = runner_class(kind, STF)
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
    gk = dict(GR=GR) if kind in ("S", "SK") else {}
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
                res = run_arm_s(model, ids, inputs, a.steps, clock, opt, FORK, arm)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), graph=rd.get("graph"), dgraph=rd.get("dgraph"),
                           hoist=rd.get("hoist"), capture_rounds=GC.capture_rounds(rd), runner=res["runner_kind"],
                           runner_kw=res["runner_kw"])
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = IA.peak_gb(rec)
                rec["effective_wo_capture_tok_s"] = HC.effective_wo_capture(rec, rd)
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
    X = stacked_arm(sp)
    preds = verdicts(sh, sp, a.batch, eqf.get(X) if X else None)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_I4S, by_batch=PRED_I4S_BY_B),
                section_graph=dict(skipped_T=GR.SKIPPED_T, skipped_vs=GR.SKIPPED_VS, eager_ok=GR.EAGER_OK,
                                   hoist_delta=[SK.expected_delta_s(False, H, a.batch, 0)["H"],
                                                SK.expected_delta_s(False, H, a.batch, 1)["H"]]),
                draft_graph=dict(deager_ok=DG.DEAGER_OK, expected_delta=DG.expected_delta(a.spec_graph == "on")),
                keep=dict(extra_ops_per_layer=KP.KEEP_EXTRA_OPS, block_bytes_head=bb, key_bits=32,
                          keep_ref=KC.KEEP_REF.get((a.batch, a.pool, a.steps))),
                mem_must_GB=MEM_MUST_GB, capture_rounds=caps, steady_excluded_rounds=xr, stacked_arm=X,
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
        print("PREDICTION %-16s %s: %s" % p)
    print("REPORT L61 capture rounds %s; excluded from every arm's steady set: %s" % (caps, xr))
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s (without capture s %s); vs SH "
              "%s; vs %.1f %s; round median %s ms; phases %s; draft step %s ms = %s SH; joint %s SH steps; steady "
              "rounds %s (excluded %s; with the capture rounds: median %s mean %s, phases %s); calls %s; lean %s; "
              "hoist %s; tokens equal %.6f (first %s); peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r.get("effective_wo_capture_tok_s"), r["vs_sh"], STRONGEST,
                  r["vs_strongest"], r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"],
                  r["draft_step_sh"], r["joint_sh_steps"], r.get("steady_rounds_used"), r.get("steady_excluded_rounds"),
                  (r.get("rates_incl_capture") or {}).get("median_tok_s"),
                  (r.get("rates_incl_capture") or {}).get("mean_tok_s"), r.get("phases_median_incl_capture"),
                  r["calls"], r.get("lean"), r.get("hoist"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        if r.get("store_record"):
            print("REPORT %s store: runner %s %s; NEW host blocks per round %s, host-gather wait median %s ms; record "
                  "%s" % (name, r["runner"], r["runner_kw"], r["kv_at_verify"].get("new_blocks_per_round"),
                          r["kv_at_verify"].get("wait_ms_median"),
                          {k: v for k, v in r["store_record"].items() if k not in ("listed", "fetched", "repeat")}))
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
            print("REPORT %s draft graphs: keys %s, captures %s, warm %s, capture passes %s, replay passes %s, replays "
                  "%s, eager %s %s; capture %s s; capture rounds %s; joint capture rounds %s; static %s B" % (
                      name, dr.get("keys"), dr.get("captures"), dr.get("warm"), dr.get("capture_passes"),
                      dr.get("replay_passes"), dr.get("replays"), dr.get("eager"), dr.get("eager_why"),
                      dr.get("dcapture_s"), dr.get("capture_rounds"), dr.get("joint_capture_rounds"),
                      dr.get("static_bytes")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4s: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
