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
I4S root arm's gates, the D arms' COMPILE, EQF = L4PTFD bit for bit), J1-PEQ, J1-COMPILE, J1-DRAFT
(r2_j1_select.j1_gates).
F2 ON THE BASELINE (the matched comparison): right after SH, ONE more shipped run SHJ1 with the shipped
nosa_llama.nosa_pooling replaced by r2_j1_select.ShPool (the same compiled arithmetic, dynamic shapes), then
restored (_BCProxy, run_sh_j1). SH's record and tokens stay SH's; SHJ1's record goes into every J1 arm's record
(sh_j1) and the MUST gate J1-SHEQ (emitted once, with the first J1 arm): SHJ1's tokens = SH's bit for bit, the eager
pooling calls' PEQ 0, the pooling inside the captures. Verdicts J1-SH-SAVING, J1-VS-SHJ1 and the MODEL line
J1-STRONGEST-MATCHED (the strongest whole-step graph's rate with SHJ1's per-step saving). --sh-j1 0 skips SHJ1.
NOTE: the SH memory record (SH-MEM) is taken after SHJ1 (the same footprint: one shipped run at a time).
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
#   J1_draft_ratio   [x] X draft median / REF draft median (F2: the draft pooling compiled too)
#   J1_draft_saving  [ms per round] REF draft median - X draft median (2 one-row passes)
#   J1_vs_ref        [x] X / REF median steady committed tok/s (same job)
#   J1_ratio         [x SH] X median steady committed tok/s / SH's
#   J1_eq            [True] X-EQF passes (= L4PTFD bit for bit)
#   J1_peak_gb       [GB] X's peak allocated GB, the prefill included
#   J1_compile_s     [s] the pooling helper's compile seconds (one-time; joint + draft shapes)
#   J1_sh_saving     [ms per step] SH steady step ms - SHJ1 steady step ms (F2 on the matched baseline)
#   J1_vs_shj1       [x] X median steady committed tok/s / SHJ1's steady tok/s (the matched comparison)
# F2 (commit 2 of rung J1): the one-row passes' pooling compiled inside the draft graphs; MEASURED basis lines 87
# (3.14 ms per one-row pass at B 216) and 678 (4.71 ms at B 318). MODEL: 0.2-0.4 ms compiled -> 2.8 / 4.4 ms per pass,
# 5.6 [4.5, 6.2] / 8.6 [7.0, 9.2] ms per round. J1_draft_ratio C 128 (132.29 - 6.2 .. 4.5) / 132.29 = (0.953, 0.966);
# C 73 (176.44 - 9.2 .. 7.0) / 176.44 = (0.948, 0.960). J1_vs_ref then C 128 324.88 / (324.88 - 14.0 .. 19.7) =
# (1.045, 1.065); C 73 524.66 / (524.66 - 21.0 .. 27.7) = (1.042, 1.056). J1_sh_saving: SH's pooling is the scalar
# kernel with the same grid (H, B, M): 2.8 [2.0, 3.2] ms per step at B 216, 4.4 [3.4, 4.7] at B 318 (MODEL: the
# per-request kernel's 3.14 / 4.71 ms as the basis). J1_vs_shj1 = J1_ratio x SH step / (SH step - J1_sh_saving).
# The values in the C 128 / C 73 lines below are F1's; the lead registers F1 + F2 from this block.
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
PRED_KEYS_J1 = ("J1_joint_saving", "J1_joint_ratio", "J1_draft_ratio", "J1_draft_saving", "J1_vs_ref", "J1_ratio",
                "J1_eq", "J1_peak_gb", "J1_compile_s", "J1_sh_saving", "J1_vs_shj1")
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
            rec["sh_j1"] = SH_J1["result"]
            g = g + J1.j1_gates(arm, rd, L)
            if not SH_J1["emitted"]:
                SH_J1["emitted"] = True
                g = g + [sh_gate()]
        return g

    def verdicts_j1(sh, sp, B, eqf=None):
        return verd0(sh, sp, B, eqf) + verdicts(sh, sp, B, D)

    D.run_arm_d, D.runner_choice, D.runner_class = run_arm_j1, runner_choice_j1, runner_class_j1
    D.keep_ref_arm, D.arm_gates, D.verdicts = keep_ref_arm_j1, arm_gates_j1, verdicts_j1
    D.BC = _BCProxy(D.BC)                                                # SHJ1 after SH (the matched baseline)
    D._J1_CONFIGURED = True
    DCC = D
    return D


# ------------------------------------------------------------------ F2 on the baseline: SHJ1 (the matched comparison)
SH_J1 = dict(enabled=True, result=None, emitted=False)


class _BCProxy:
    """The private copy's BC (the base cell module) with run_shipped wrapped: after SH, ONE more shipped run (SHJ1)
    with nosa_llama.nosa_pooling replaced by r2_j1_select.ShPool (F2 on the baseline), then restored. SH's own record,
    inputs and gates are SH's run; SHJ1's record goes to SH_J1 (the J1 arm's record and gate J1-SHEQ)."""

    def __init__(self, bc):
        self._bc = bc

    def __getattr__(self, name):
        return getattr(self._bc, name)

    def run_shipped(self, model, ids, steps, clock, fp=None):
        bc = self._bc
        out = bc.run_shipped(model, ids, steps, clock, fp=fp)
        if SH_J1["enabled"] and SH_J1["result"] is None:
            SH_J1["result"] = run_sh_j1(bc, model, ids, steps, clock, out)
        return out


def run_sh_j1(bc, model, ids, steps, clock, sh_out, NL=None) -> dict:
    """SHJ1: the shipped run with the compiled pooling (the matched baseline). Returns its record: the steady rates (the
    same statistic as SH's), the tokens' bit equality with SH's, the pooling PEQ of the eager calls, the compile record,
    or the exception. NL: the module whose nosa_pooling the shipped layers call (None: nosi.nosa_llama; tests: a
    stand-in)."""
    import torch
    if NL is None:
        import nosi.nosa_llama as NL
    bc.free()
    orig = NL.nosa_pooling
    sp = J1.ShPool(orig)
    rec = dict(ran=False)
    try:
        NL.nosa_pooling = sp
        o2 = bc.run_shipped(model, ids, steps, clock, fp=None)
        rw = {t for t, _ in o2["rewarms"]}
        steady = bc.period_stats(o2["periods"], [t for t in range(steps) if t not in rw])
        rec = dict(ran=True, steady=steady, rates=bc.rates(ids.shape[0], steps, o2["periods"], steady),
                   rewarms=o2["rewarms"], tokens_equal=bool(torch.equal(o2["inputs"], sh_out["inputs"])),
                   pool=sp.report())
        del o2
    except Exception as e:                                             # the gate reports it (the job goes on)
        traceback.print_exc()
        rec = dict(ran=False, error="%s: %s" % (type(e).__name__, str(e)[:300]), pool=sp.report())
    finally:
        NL.nosa_pooling = orig
        bc.free()
    return rec


def sh_gate():
    """J1-SHEQ (MUST, emitted once with the first J1 arm): SHJ1 ran, its tokens = SH's bit for bit, the eager pooling
    PEQ ran with 0 differing values, and its pooling ran inside the captures."""
    r = SH_J1["result"]
    if r is None:
        return ("J1-SHEQ", False, False, "REPORT: SHJ1 did not run (--sh-j1 0 or SH failed)")
    p = r.get("pool") or {}
    ok = (r.get("ran") and r.get("tokens_equal") and p.get("peq_calls", 0) > 0 and p.get("peq_bad") == 0
          and p.get("captured", 0) > 0)
    return ("J1-SHEQ", True, bool(ok), "ran %s; tokens equal to SH %s; pooling calls %s (captured %s), eager PEQ %s "
            "differing of %s in %s calls; compile %s s; error %s" % (
                r.get("ran"), r.get("tokens_equal"), p.get("calls"), p.get("captured"), p.get("peq_bad"),
                p.get("peq_vals"), p.get("peq_calls"), p.get("compile_s"), r.get("error")))


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
    dr, dx = ph(r, "draft"), ph(x, "draft")
    band("J1-DRAFT-SAVING", (dr - dx) if (dr is not None and dx is not None) else None, P.get("J1_draft_saving"))
    s1 = SH_J1["result"] or {}
    shs = ((sh or {}).get("rates") or {}).get("steady_step_ms")
    s1s = (s1.get("rates") or {}).get("steady_step_ms")
    sav = (shs - s1s) if (shs is not None and s1s is not None) else None
    band("J1-SH-SAVING", sav, P.get("J1_sh_saving"))
    s1t = (s1.get("rates") or {}).get("steady_tok_s")
    band("J1-VS-SHJ1", ratio(med(x), s1t), P.get("J1_vs_shj1"))
    strong = getattr(D, "STRONGEST", None)
    matched = (1000.0 * B / (1000.0 * B / strong - sav)) if (strong and sav is not None) else None
    out.append(("J1-STRONGEST-MATCHED", "REPORT", "MODEL: the strongest whole-step graph %s tok/s with the same per-step "
                "pooling saving as SH -> SHJ1 (%s ms): %s tok/s; X / it %s (X / %s: %s)" % (
                    strong, None if sav is None else round(sav, 3), None if matched is None else round(matched, 1),
                    None if (matched is None or med(x) is None) else round(med(x) / matched, 4), strong,
                    None if (strong is None or med(x) is None) else round(med(x) / strong, 4))))
    out.append(("J1-ARM", "REPORT", "X = %s, REF = %s (REF ran: %s); J1 record %s; SHJ1 %s" % (
        X, J1.ARMS_J1[X], r is not None, x.get("j1"), {k: v for k, v in s1.items() if k != "steady"})))
    return out


def main(argv=None) -> int:
    """The I4DC cell's main on the private copy. One extra option of this cell: --sh-j1 {0,1} (default 1: run SHJ1)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--sh-j1" in argv:
        i = argv.index("--sh-j1")
        if i + 1 >= len(argv) or argv[i + 1] not in ("0", "1"):
            print("REFUSED: --sh-j1 0 or 1", file=sys.stderr)
            return 20
        SH_J1["enabled"] = argv[i + 1] == "1"
        del argv[i:i + 2]
    D = configure()
    return D.main(argv)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
