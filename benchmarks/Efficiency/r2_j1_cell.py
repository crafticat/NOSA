"""Rung J1 (AREA 1 of the RECON7 follow-up; KS/analysis/plan/DESIGN_JOINT_EFF_2026-10-31.md fix F1; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of the arms with the COMPILED JOINT POOLING (benchmarks/Efficiency/
r2_j1_select.py: what is compiled, what stays, the exactness argument):
  L4PTFDGUDCDJ1    L4PTFDGUDCD   (rung I4DC, the best arm at C 128) + the joint pooling compiled
  L4PTFDGUDKWCDJ1  L4PTFDGUDKWCD (rung I4DC, the best arm at C 73), the same
THE I4DC CELL, REUSED AS A PRIVATE MODULE COPY: this file loads benchmarks/Efficiency/r2_i4dc_cell.py a SECOND time
under the name "r2_j1_dcc" (the same bytes; no edit; the imported I4DC module keeps its own tables), extends THAT copy's
arm tables with the J1 arms, wraps its run_arm_d / runner_class (the J1 runner for a J1 arm), keep_ref_arm (the J1 keep
arm's references), arm_gates (+ J1-PEQ, J1-COMPILE) and verdicts (+ the J1 predictions), and runs its main(). Every
gate, record and report line of the I4DC cell stays as it is. Step script: benchmarks/Efficiency/r2_j1_step.sh (cells
j1m, j1a, j1b). Docs: docs/NOTES_j1.md.

A J1 arm needs its I4DC arm EARLIER in the job (BASE_D of the copy: the arm's REF; it compiles the draft helpers, so
the J1 arm compiles the pooling helper only). MUST GATES of a J1 arm: every gate of its I4DC arm under the J1 name (the
I4S root arm's gates, the D arms' COMPILE, EQF = L4PTFD bit for bit), J1-PEQ, J1-COMPILE (r2_j1_select.j1_gates).
REGISTERED PREDICTIONS PRED_J1_BY_B: every value None until the lead registers them with PRED_STAMP_J1 (a None key is
REPORT); the step script STOPS the timing cells (j1a, j1b) while PRED_STAMP_J1 is None.
EXIT: the I4DC cell's (the number of failed MUST gates 0-19; 20 refused; 30 an exception outside an arm).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_j1_select as J1         # noqa: E402

DCC_NAME = "r2_j1_dcc"
DCC_FILE = os.path.join(HERE, "r2_i4dc_cell.py")
ARMS_J1 = tuple(J1.ARMS_J1)
KEEP_J1 = ("L4PTFDGUDKWCDJ1",)
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_J1_BY_B (per batch) and PRED_STAMP_J1 (`date
# -u`, L33) in a commit BEFORE j1a / j1b (the step script stops while PRED_STAMP_J1 is None). Keys, each (low, high)
# unless noted; X = the J1 arm of the cell, REF = its I4DC arm; phases / rates on the steady set without the capture
# rounds (L61):
#   J1_joint_saving  [ms per round] REF joint median - X joint median
#   J1_joint_ratio   [x] X joint median / REF joint median
#   J1_draft_ratio   [x] X draft median / REF draft median (the draft is unchanged: about 1)
#   J1_vs_ref        [x] X / REF median steady committed tok/s (same job)
#   J1_ratio         [x SH] X median steady committed tok/s / SH's
#   J1_eq            [True] X-EQF passes (= L4PTFD bit for bit)
#   J1_peak_gb       [GB] X's peak allocated GB, the prefill included
#   J1_compile_s     [s] the pooling helper's compile seconds (one-time)
# THE BUILDER'S MODEL (an input for the lead, not a registration). MEASURED basis (KS/cpu-logs/recon7_r2_s2.txt, the I3
# joint path = the I4DC arms' selection, one profiled pass): nosa_pool_kernel_per_request 12.28 ms per joint pass at
# C 128 B 216 (line 21: grid (2, 864, 261), 383.88 us per layer) and 18.10 ms at C 73 B 318 (line 615: 565.74 us per
# layer). REF rounds MEASURED (LEDGER_rev7.md 2.1): C 128 324.88 ms (joint 186.01, 1,924.1 tok/s, nova05); C 73 524.66
# ms (joint 338.96, 1,699.0 tok/s, nova07). MODEL: the compiled gather + max reads about 7 MB per layer at C 128 (10 MB
# at C 73) at 0.5-1.0 TB/s with 2 reductions: 15-30 us per layer -> 0.5-1.0 ms per pass (C 73 0.7-1.5).
#   C 128 B 216: J1_joint_saving 11.6 [9.5, 13.5]; J1_joint_ratio (186.01 - 13.5 .. 9.5) / 186.01 = (0.927, 0.949);
#     J1_draft_ratio (0.99, 1.01); J1_vs_ref 324.88 / (324.88 - 9.5 .. 13.5) = (1.030, 1.043); J1_ratio 0.562 x
#     (1.030 .. 1.043) = (0.578, 0.587) with the cross-job SH spread (0.575, 0.590); J1_eq True; J1_peak_gb
#     (83.85, 83.95) (no buffer added; PEQ scratch 1.8 MB, check rounds only); J1_compile_s (0.5, 30)
#   C 73 B 318: J1_joint_saving 17.1 [14.0, 18.5]; J1_joint_ratio (338.96 - 18.5 .. 14.0) / 338.96 = (0.945, 0.959);
#     J1_draft_ratio (0.99, 1.01); J1_vs_ref 524.66 / (524.66 - 14.0 .. 18.5) = (1.027, 1.037); J1_ratio 0.660 x that
#     (REF / SH of i4dcb) = (0.678, 0.684) -> (0.672, 0.690) with the spread; J1_eq True; J1_peak_gb (77.60, 77.80);
#     J1_compile_s (0.5, 30)
PRED_STAMP_J1 = None
# THE CELL GRANTS (default deny; r2_j1_step.sh's granted() reads these lines at <sha>): the lead sets a `date -u` stamp,
# e.g. GRANT_j1m = "2026-10-31T12:00:00Z", in a commit after the protocol review.
GRANT_j1m = None
GRANT_j1a = None
GRANT_j1b = None
PRED_KEYS_J1 = ("J1_joint_saving", "J1_joint_ratio", "J1_draft_ratio", "J1_vs_ref", "J1_ratio", "J1_eq", "J1_peak_gb",
                "J1_compile_s")
PRED_J1_BY_B = {216: {k: None for k in PRED_KEYS_J1}, 318: {k: None for k in PRED_KEYS_J1}}

_CUR = dict(arm=None)
DCC = None


def _private_copy():
    """The I4DC cell loaded a second time as DCC_NAME (the same file; module docstring)."""
    if DCC_NAME in sys.modules:
        return sys.modules[DCC_NAME]
    spec = importlib.util.spec_from_file_location(DCC_NAME, DCC_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[DCC_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def configure():
    """The private copy with the J1 arms (idempotent). Returns it."""
    global DCC
    D = _private_copy()
    if getattr(D, "_J1_CONFIGURED", False):
        DCC = D
        return D
    run0, rclass0, kref0, gates0, verd0 = D.run_arm_d, D.runner_class, D.keep_ref_arm, D.arm_gates, D.verdicts
    D.BASE_D = dict(D.BASE_D, **J1.ARMS_J1)                              # a J1 arm needs its I4DC arm earlier
    D.ROOT_ARM = dict(D.ROOT_ARM, **{x: D.ROOT_ARM[b] for x, b in J1.ARMS_J1.items()})
    D.ARMS = D.ARMS + ARMS_J1
    D.GRAPH_ARMS = D.GRAPH_ARMS + ARMS_J1
    D.STACK_ARMS = D.STACK_ARMS + ARMS_J1
    D.KEEP_ARMS = D.KEEP_ARMS + KEEP_J1
    D.NAMES = dict(D.NAMES, **{x: D.NAMES[b] for x, b in J1.ARMS_J1.items()})

    def run_arm_j1(model, ids, inputs, steps, clock, opt, FORK, arm):
        """The I4DC run_arm_d with the J1 runner for a J1 arm (runner_class reads the arm from _CUR)."""
        _CUR["arm"] = arm
        try:
            return run0(model, ids, inputs, steps, clock, opt, FORK, arm)
        finally:
            _CUR["arm"] = None

    def runner_choice_j1(arm, check):
        if arm in J1.ARMS_J1:
            return rc0(J1.ARMS_J1[arm], check)
        return rc0(arm, check)
    rc0 = D.runner_choice

    def runner_class_j1(kind, STF):
        arm = _CUR["arm"]
        if arm in J1.ARMS_J1:
            return J1.j1_classes(STF)[arm][1]
        return rclass0(kind, STF)

    def keep_ref_arm_j1(arm, have):
        """The J1 keep arm: L4PTFKW, L4PTFDGKW, L4PTFDGUDKW, L4PTFDGUDKWC, L4PTFDGUDKWCD (the first that ran earlier)."""
        if arm not in KEEP_J1:
            return kref0(arm, have)
        for r in ("L4PTFKW", "L4PTFDGKW", "L4PTFDGUDKW", "L4PTFDGUDKWC", "L4PTFDGUDKWCD"):
            if r in have and D.ARMS.index(r) < D.ARMS.index(arm):
                return r
        return None

    def arm_gates_j1(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa):
        g = gates0(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
        if arm in J1.ARMS_J1:
            rec["j1"] = (rd or {}).get("j1")                           # into the summary (the arm's record)
            g = g + J1.j1_gates(arm, rd, L)
        return g

    def verdicts_j1(sh, sp, B, eqf=None):
        return verd0(sh, sp, B, eqf) + verdicts(sh, sp, B, D)

    D.run_arm_d, D.runner_choice, D.runner_class = run_arm_j1, runner_choice_j1, runner_class_j1
    D.keep_ref_arm, D.arm_gates, D.verdicts = keep_ref_arm_j1, arm_gates_j1, verdicts_j1
    D._J1_CONFIGURED = True
    DCC = D
    return D


def j1_arm(sp):
    """X: the J1 arm of the job (the keep arm first); None if none ran."""
    for x in KEEP_J1 + tuple(a for a in ARMS_J1 if a not in KEEP_J1):
        if x in (sp or {}):
            return x
    return None


def verdicts(sh, sp, B: int, D=None):
    """PRED_J1_BY_B[B] against X = j1_arm(sp) and REF = its I4DC arm: [(name, verdict, detail)]; None gives REPORT."""
    D = D or configure()
    P = PRED_J1_BY_B.get(B) or {k: None for k in PRED_KEYS_J1}
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))

    def med(r):
        return (r or {}).get("rates", {}).get("median_tok_s")

    def ph(r, name):
        return ((r or {}).get("phases_median") or {}).get(name)

    def ratio(a, b):
        return (a / b) if (a is not None and b) else None
    sp = sp or {}
    X = j1_arm(sp)
    if X is None:
        out.append(("J1-RUN", "NOT_RUN", "no J1 arm ran"))
        return out
    x, r = sp[X], sp.get(J1.ARMS_J1[X])
    jr, jx = ph(r, "joint"), ph(x, "joint")
    shm = D.SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    band("J1-JOINT-SAVING", (jr - jx) if (jr is not None and jx is not None) else None, P.get("J1_joint_saving"))
    band("J1-JOINT-RATIO", ratio(jx, jr), P.get("J1_joint_ratio"))
    band("J1-DRAFT-RATIO", ratio(ph(x, "draft"), ph(r, "draft")), P.get("J1_draft_ratio"))
    band("J1-VS-REF", ratio(med(x), med(r)), P.get("J1_vs_ref"))
    band("J1-RATIO", ratio(med(x), shm), P.get("J1_ratio"))
    band("J1-PEAK-GB", D.IA.peak_gb(x), P.get("J1_peak_gb"))
    band("J1-COMPILE-S", ((x.get("j1") or {}).get("compile_s")), P.get("J1_compile_s"))
    out.append(("J1-ARM", "REPORT", "X = %s, REF = %s (REF ran: %s); J1 record %s" % (
        X, J1.ARMS_J1[X], r is not None, x.get("j1"))))
    return out


def main(argv=None) -> int:
    D = configure()
    return D.main(argv)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
