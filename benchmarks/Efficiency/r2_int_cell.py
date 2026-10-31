"""Rung INT (KS/analysis/plan/INTEGRATION_NOTES_2026-10-31.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of
the INTEGRATED arms (benchmarks/Efficiency/r2_int_arms.py: the classes, the two collision fixes):
  L4PTFDGUDCDI    L4PTFDGUDCD   + F1 + F2 + F6 + F3 + ST1 + ST2 + ST4    (C 128 B 216: the best arm there)
  L4PTFDGUDKWCDI  L4PTFDGUDKWCD + F1 + F2 + F6 + ST1 + ST2 + BF1 + ST3 + ST4   (C 73 B 318: the best arm there)
  L4PTFDGUDKWCDIN L4PTFDGUDKWCD + F1 + F2 + F6 + ST1 + ST2 + ST3 + ST4 (C 73 WITHOUT BF1: cell intd, store 129; the
                  BF1 gates and -INT-LOCK do not apply; its bands: PRED_INT_NOBF_BY_B)
  ST3 = the keep fill in two compiled helpers (gate -KFILL), ST4 = the draft residency's integer ops in one compiled
  helper (gate -DRESID): r2_st1_commit.st3_runner / st4_runner over the INT runner, in that order.
THE I4DC CELL, REUSED AS A PRIVATE MODULE COPY (the J1 cell's pattern): this file loads benchmarks/Efficiency/
r2_i4dc_cell.py a second time as "r2_int_dcc" (no edit), extends THAT copy's arm tables and wraps its run_arm_d,
runner_choice, runner_class, keep_ref_arm, arm_gates, verdicts, BC (SHJ1 after SH: r2_j1_cell's _BCProxy) and HC (the
ST1 commit graph's capture round joins the L61 exclusion), then runs its main(). The single-rung arms may run in the
same job as matched controls, with their own cells' gates: the J1 / J3 arms (r2_j1_select + r2_j1_select.j1_gates /
j3_gates), the ST arms L4PTFDGUDCDM / MT / MTR, L4PTFDGUDKWCDM / MT / MTK / MTKR (r2_st1_cell.arm_gates_st1; MTK = MT
+ ST3, the ST3 control; MTR / MTKR = MT / MTK + ST4, intm only), the BF1 arms
L4PTFDGUDKWCDX / B (r2_bf1_cell.arm_gates_bf). Every arm's read() carries ST1's tail checksum (the INT arms' TAIL-EQ
reference). Step script: benchmarks/Efficiency/r2_int_step.sh (cells intm, inta, intb). Docs: docs/NOTES_int.md.

ARM ORDER: the copy's ARMS (L4PTFD first; a D arm after its C arm), then the J1 / J3, ST, BF1 and INT arms. An INT arm
needs its D arm EARLIER (its PEQ, DCOUNTS / COUNTS, LISTED and TAIL-EQ reference; it compiled the draft helpers).
MUST GATES OF AN INT ARM (all components' + EQF = L4PTFD bit for bit):
  the I4DC gates of its D arm under the INT name (MEM, GRAPH, COUNTS, DGRAPH, HOIST, the keep gates, COMPILE with the
    I4CC loc1 helper left out (F6), EQF), the C 73 arm without -COUNTS / -KEEP-EQ / -KEEP-REF (BF1 changes them by
    design) and with BF1's -COUNTS (without round_blocks) / -LISTED / -BF against the D arm;
  J1: -J1-PEQ, -J1-COMPILE, -J1-DRAFT; F3 (C 128): -J1-RESID; J1-SHEQ once per job (SHJ1);
  ST1 / ST2: -PEQ (= the D arm bit for bit), -DCOUNTS (C 128), -TAIL-EQ, -CGRAPH, -TWIN; ST3 (C 73): -KFILL; ST4:
  -DRESID;
  INT: -INT-CLASSES, -INT-LAYER, -INT-LOCK (the BF1 arm) (r2_int_arms.int_gates).
REGISTERED PREDICTIONS PRED_INT_BY_B: every value None until the lead registers them with PRED_STAMP_INT (a None key is
REPORT); the step script STOPS the timing cells (inta, intb) while PRED_STAMP_INT is None. THE CELL GRANTS GRANT_intm /
GRANT_inta / GRANT_intb / GRANT_intc (default deny). intc = the original anchor (C 81 B 287, L4PTFDGUDCDI).
EXIT: the I4DC cell's (the number of failed MUST gates 0-19; 20 refused; 30 an exception outside an arm); 2 an option
or arm-order error of this cell.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_bf1_cell as BFCELL      # noqa: E402
import r2_bf1_fetch as BF         # noqa: E402
import r2_int_arms as INTA        # noqa: E402
import r2_j1_cell as J1C          # noqa: E402
import r2_j1_select as J1         # noqa: E402
import r2_st1_cell as STCELL      # noqa: E402
import r2_st1_commit as STC       # noqa: E402

DCC_NAME = "r2_int_dcc"
DCC_FILE = os.path.join(HERE, "r2_i4dc_cell.py")
ARMS_INT = dict(INTA.ARMS_INT)
ARMS_J = tuple(J1.ARMS_J1) + tuple(J1.ARMS_J3)
ST_ARMS = dict(STCELL.ST13)                       # an ST arm -> its PEQ reference (M: D; MT: M; MTK: MT)
INT_ST3 = ("L4PTFDGUDKWCDI", "L4PTFDGUDKWCDIN")   # the INT arms that also carry ST3 (r2_st1_commit.st3_runner)
INT_ST4 = ("L4PTFDGUDCDI", "L4PTFDGUDKWCDI", "L4PTFDGUDKWCDIN")   # ... ST4 (st4_runner, after st3_runner)
BF_ARMS = dict(BFCELL.BF_ARMS)                    # a BF1 arm -> its fetch mode
D_OF = dict(J1.ARMS_J1, **J1C.BASE_J3, **ARMS_INT)   # the arms that take the I4DC gates of their D arm (own name)
KEEP_NEW = ("L4PTFDGUDKWCDJ1", "L4PTFDGUDKWCDM", "L4PTFDGUDKWCDMT", "L4PTFDGUDKWCDMTK", "L4PTFDGUDKWCDMTKR",
            "L4PTFDGUDKWCDX", "L4PTFDGUDKWCDB", "L4PTFDGUDKWCDI", "L4PTFDGUDKWCDIN")
NOBF_INT = ("L4PTFDGUDKWCDIN",)                   # the C 73 INT arm WITHOUT BF1 (cell intd): PRED_INT_NOBF_BY_B
NEW_ARMS = ARMS_J + tuple(ST_ARMS) + tuple(BF_ARMS) + tuple(ARMS_INT)
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_INT_BY_B (per batch) and PRED_STAMP_INT (`date
# -u`, L33) in a commit BEFORE inta / intb. Keys, each (low, high) unless noted; X = the INT arm of the cell, REF = its D
# arm (same job); phases / rates on the steady set without the capture rounds (L61):
#   INT_vs_ref        [x] X / REF median steady committed tok/s
#   INT_ratio         [x SH] X median steady committed tok/s / SH's
#   INT_vs_shj1       [x] X / SHJ1's steady tok/s (the matched baseline: SH with the compiled pooling)
#   INT_vs_strongest  [x] X / the strongest whole-step graph (3,426.2 tok/s at C 128 B 216; cross-job)
#   INT_joint_saving  [ms per round] REF joint median - X's
#   INT_draft_saving  [ms per round] REF draft median - X's
#   INT_commit_saving [ms per round] REF commit median - X's
#   INT_peak_gb       [GB] X's peak allocated GB with the prefill
#   INT_eq            [True] X-EQF passes
#   INT_st3_joint_saving [ms per round] the ST3 control in this job: L4PTFDGUDKWCDMT joint median - L4PTFDGUDKWCDMTK's
#   INT_st3_vs_mt     [x] L4PTFDGUDKWCDMTK / L4PTFDGUDKWCDMT median steady committed tok/s (the C 73 INT arm carries ST3
#                     too: KS/analysis/plan/INTEGRATION_NOTES_2026-10-31.md section 9; MODEL there: 5.1 .. 8.5 ms)
# THE BUILDER'S MODEL (an input for the lead, not a registration): the components' REGISTERED bands, summed per phase
# (the sum is an assumption: the phases are device-bound and the hooks touch different labelled components):
#   C 128 B 216 (REF round 324.88 ms): joint F1 9.5 .. 13.5 + F6 1.0 .. 1.6 (MODEL) + ST2 1.0 .. 2.8 + F3 0 .. 6.3;
#     draft F2 4.5 .. 6.2 + F3 0 .. 6.0; commit ST1 3.0 .. 5.5 -> 19.0 .. 41.9 ms per round: INT_vs_ref (1.062, 1.148);
#     the F3 part is the least certain (central 3 ms): central about 29 ms, 1.098.
#   C 73 B 318 (REF round 524.66 ms): joint F1 14.0 .. 18.5 + F6 about 1.9 + ST2 1.0 .. 3.0; draft F2 7.0 .. 9.2;
#     commit ST1 4.0 .. 7.0; BF1 (unregistered MODEL) 40 .. 64 minus 1.4 .. 6 -> 60 .. 101 ms: (1.13, 1.24); without
#     BF1 27 .. 39 ms: (1.054, 1.080).
PRED_STAMP_INT = "2026-10-31T18:16:02Z"
PRED_KEYS_INT = ("INT_vs_ref", "INT_ratio", "INT_vs_shj1", "INT_vs_strongest", "INT_joint_saving", "INT_draft_saving",
                 "INT_commit_saving", "INT_peak_gb", "INT_eq", "INT_st3_joint_saving", "INT_st3_vs_mt")
# THE LEAD'S REGISTRATION (2026-10-31T17:03:19Z `date -u`, before any INT GPU run). Basis (MEASURED, same-job ratios):
# C128 B216 D round 327.4 ms (j1a); J1 F1 + F2 + F3 -20.1 ms (j1a: J3 307.3); ST1 + ST2 -4.7 ms (st1a: MT 321.7 vs D
# 326.4); MODEL: F6 -1.3, ST4 -1.9 .. 2.5 -> INT round 296 .. 304 ms. C73 B318 D round 521.3 ms (bf1b same job);
# MODEL: F1 + F2 + F6 -22 .. 30, ST1 -2.5 .. 5.5 (st1a measured 0.6-0.8 of its band), ST2 -1 .. 3, ST3 -3 .. 9, ST4
# -2 .. 2.7, BF1 defer -10 .. 30 (bf1b barrier version: joint -9.0) -> INT round 442 .. 481 ms. SH steps 64.88 /
# 123.42 ms (same jobs); SHJ1 = SH - 2.39 / 4.4 ms; INT_vs_strongest uses 3,407.2 tok/s (the strongest whole-step
# graph at C128 B216, the matched-cap reference).
PRED_INT_BY_B = {
    216: dict(INT_vs_ref=(1.077, 1.106), INT_ratio=(0.617, 0.636), INT_vs_shj1=(0.594, 0.613),
              INT_vs_strongest=(0.603, 0.622), INT_joint_saving=(14.0, 20.0), INT_draft_saving=(7.0, 11.0),
              INT_commit_saving=(2.0, 3.3), INT_peak_gb=(83.85, 83.97), INT_eq=True, INT_st3_joint_saving=None,
              INT_st3_vs_mt=None),
    # 318: RE-REGISTERED 2026-10-31T18:16:02Z (`date -u`) BEFORE intb, for the batched BF1 (64e307f in ea18c76), from
    # MEASURED C73 component effects: J1 -25.4 ms per round (j1b), ST1 + ST2 -6.95 ms (st1b); MODEL: ST3 -3 .. 9, ST4
    # -2 .. 2.7, BF1 batched -8 .. 35 -> INT round 442 .. 475 ms vs D 521.3. The 216 bands stay those of 17:03:19Z.
    318: dict(INT_vs_ref=(1.097, 1.179), INT_ratio=(0.726, 0.781), INT_vs_shj1=(0.700, 0.753),
              INT_vs_strongest=(0.550, 0.591), INT_joint_saving=(31.0, 59.0), INT_draft_saving=(8.5, 12.0),
              INT_commit_saving=(3.8, 4.8), INT_peak_gb=(80.8, 82.0), INT_eq=True, INT_st3_joint_saving=(3.0, 9.0),
              INT_st3_vs_mt=(1.005, 1.018)),
    # 287: THE ORIGINAL ANCHOR (cell intc: C 81 pool 18, B 287; the C 128 INT composition L4PTFDGUDCDI, no BF1 / keep).
    # None until the lead registers them. MEMORY (MEASURED, r7mem): L4PTFD at C 81 B 287 peaks at 76.80 GB; the INT arm
    # adds about 0.1 GB (MODEL: ST1's private pool + the compiled helpers' buffers).
    # REGISTERED 2026-10-31T19:32:23Z (`date -u`) BEFORE intc, MODEL from MEASURED ratios: the INT composition gave
    # x1.090 over its D arm at C128 B216 (inta), the components x1.052 x 1.013 (J1, J3) x 1.015 (ST1 + ST2) at C128 and
    # x1.0515 (J1) x 1.013 (ST) at C73 -> x1.075 .. 1.105 at C81; D / SH at C81 between C128 (0.571, inta) and C73 (0.662,
    # bf1b) -> 0.62 .. 0.66; SH at C81 B287 about 97.9 ms per step (r7mem B64 / B192 line); SHJ1 about 0.96 of SH's step.
    287: dict(INT_vs_ref=(1.075, 1.105), INT_ratio=(0.67, 0.73), INT_vs_shj1=(0.64, 0.70),
              INT_vs_strongest=(0.57, 0.63), INT_joint_saving=(16.0, 26.0), INT_draft_saving=(7.0, 12.0),
              INT_commit_saving=(2.5, 4.5), INT_peak_gb=(76.6, 77.6), INT_eq=True, INT_st3_joint_saving=None,
              INT_st3_vs_mt=None)}
# THE C 73 INT ARM WITHOUT BF1 (L4PTFDGUDKWCDIN, cell intd: C 73 pool 10, B 318, store 129): its own bands (verdicts use
# this dict when X is that arm). None until the lead registers them (with a new PRED_STAMP_INT).
PRED_INT_NOBF_BY_B = {318: {k: None for k in PRED_KEYS_INT}}
# THE CELL GRANTS (default deny; r2_int_step.sh's granted() reads these lines at <sha>): the lead sets a `date -u` stamp.
GRANT_intm = "2026-10-31T17:03:19Z"
GRANT_inta = "2026-10-31T17:03:19Z"
GRANT_intb = "2026-10-31T18:16:02Z"
GRANT_intc = "2026-10-31T19:32:23Z"
GRANT_intd = None
# this cell's options (main strips them before the I4DC main); the BF1 defaults are r2_bf1_cell's
OPT = dict(st1_commit="rows", ineg=None, bf_store=0, bf_ncap=2048, bf_ring=8, bf_worker="thread", bf_acct=0,
           bf_close_wait_ms=0.0, bf_packer="gilfree", bf_threads=14, bf_cap_groups=2048, bf_late="defer",
           bf_late_wait_ms=250.0, bf_batch=8)
_CUR = dict(arm=None)
RECS = {}                                         # arm -> its summary record (the report lines after the main)
EQF = {}                                          # arm -> its EQF gate passed
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


def bf_opt(arm: str) -> dict:
    """BgFetch's keyword arguments of a BF1 arm or of the INT keep arm (mode "ce")."""
    return dict(mode=BF_ARMS.get(arm, "ce"), n_cap=OPT["bf_ncap"], ring=OPT["bf_ring"], worker=OPT["bf_worker"],
                acct=bool(OPT["bf_acct"]), close_wait_ms=OPT["bf_close_wait_ms"], packer=OPT["bf_packer"],
                threads=OPT["bf_threads"], gf=dict(cap_groups=OPT["bf_cap_groups"]), late=OPT["bf_late"],
                late_wait_ms=OPT["bf_late_wait_ms"], batch=OPT["bf_batch"])   # = r2_bf1_cell.bf_opt (64e307f)


class _HCProxy:
    """The copy's HC (the I4H cell module) with capture_rounds_x extended by the ST1 commit graph's capture round (the
    ST1 cell's L61 line)."""

    def __init__(self, hc):
        self._hc = hc

    def __getattr__(self, name):
        return getattr(self._hc, name)

    def capture_rounds_x(self, rd):
        c = self._hc.capture_rounds_x(rd)
        s1 = (rd or {}).get("st1")
        if s1 and s1.get("capture_round") is not None and int(s1["capture_round"]) >= 0:
            c = dict(c, joint=sorted(set(c["joint"]) | {int(s1["capture_round"])}))
        return c


def configure():
    """The private copy with the INT and the single-rung arms (idempotent). Returns it."""
    global DCC
    D = _private_copy()
    if getattr(D, "_INT_CONFIGURED", False):
        DCC = D
        return D
    run0, rc0, rclass0, kref0, gates0, verd0 = (D.run_arm_d, D.runner_choice, D.runner_class, D.keep_ref_arm,
                                                D.arm_gates, D.verdicts)
    D.BASE_D = dict(D.BASE_D, **D_OF)                                    # an INT / J1 arm needs its D arm earlier
    D.ROOT_ARM = dict(D.ROOT_ARM, **{x: D.ROOT_ARM[b] for x, b in D_OF.items()})
    D.ARMS = D.ARMS + NEW_ARMS
    D.GRAPH_ARMS = D.GRAPH_ARMS + NEW_ARMS
    D.STACK_ARMS = D.STACK_ARMS + ARMS_J + tuple(ARMS_INT)
    D.KEEP_ARMS = D.KEEP_ARMS + KEEP_NEW
    D.NAMES = dict(D.NAMES, **{x: tuple(n for n in D.NAMES[b] if n != "loc1") for x, b in D_OF.items()})   # F6

    def run_arm_int(model, ids, inputs, steps, clock, opt, FORK, arm):
        """The I4DC run_arm_d with the arm's runner (runner_class reads the arm from _CUR); a BF1 arm and the INT keep
        arm take the store X of --bf-store (0: --fork-store's)."""
        _CUR["arm"] = arm
        o = dict(opt, fork_store=OPT["bf_store"]) if ((arm in BF_ARMS or arm in INTA.BF_INT) and OPT["bf_store"]) \
            else opt
        try:
            return run0(model, ids, inputs, steps, clock, o, FORK, arm)
        finally:
            _CUR["arm"] = None

    def runner_choice_int(arm, check):
        if arm in D_OF:
            kind, kw = rc0(D_OF[arm], check)
        elif arm in ST_ARMS:
            kind, kw = rc0(STCELL.ROOT[arm], check)
        elif arm in BF_ARMS:
            kind, kw = rc0(BFCELL.REF, check)
        else:
            return rc0(arm, check)
        if arm in BF_ARMS or arm in INTA.BF_INT:
            kw = dict(kw, bf=bf_opt(arm))
        return kind, kw

    def runner_class_int(kind, STF):
        arm = _CUR["arm"]
        if arm in ARMS_J:
            R = J1.j1_classes(STF)[arm][1]
        elif arm in ST_ARMS:
            R = STC.st1_runner(rclass0(kind, STF), OPT["st1_commit"], twin=arm in STCELL.TWIN)
            R = STC.st3_runner(R) if arm in STCELL.KFILL_ARMS else R    # ST3: the MTK / MTKR control arms
            R = STC.st4_runner(R) if arm in STCELL.ST4 else R           # ST4: the MTR / MTKR control arms
        elif arm in BF_ARMS:
            R = BF.bf_classes(STF)[1]
        elif arm in ARMS_INT:
            R = INTA.int_runner(STF, arm, OPT["st1_commit"], OPT["ineg"])
            R = STC.st3_runner(R) if arm in INT_ST3 else R              # ST3 in the C 73 INT arms
            R = STC.st4_runner(R) if arm in INT_ST4 else R              # ST4 in every INT arm (after ST3)
        else:
            R = rclass0(kind, STF)
        return INTA.with_tails(R)

    def keep_ref_arm_int(arm, have):
        """The J1 / INT keep arms: L4PTFKW, L4PTFDGKW, L4PTFDGUDKW, L4PTFDGUDKWC, L4PTFDGUDKWCD (the first that ran
        earlier)."""
        if arm not in ("L4PTFDGUDKWCDJ1", "L4PTFDGUDKWCDI", "L4PTFDGUDKWCDIN"):
            return kref0(arm, have)
        for r in ("L4PTFKW", "L4PTFDGKW", "L4PTFDGUDKW", "L4PTFDGUDKWC", "L4PTFDGUDKWCD"):
            if r in have and D.ARMS.index(r) < D.ARMS.index(arm):
                return r
        return None

    def arm_gates_int(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa):
        rd = rd or {}
        cpu["tails"] = rd.get("tails")                                 # every arm: the TAIL-EQ reference
        for k in ("j1", "j3", "st1", "st2", "st3", "st4", "bf", "int"):
            if rd.get(k) is not None:
                rec[k] = rd[k]
        if arm in BF_ARMS or arm in INTA.BF_INT:
            rec["new"] = BFCELL.new_record(rec, rd)
        RECS[arm] = rec
        if arm in ST_ARMS:
            g = STCELL.arm_gates_st1(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
        elif arm in BF_ARMS:
            g = BFCELL.arm_gates_bf(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
        elif arm in ARMS_INT:
            g = int_arm_gates(gates0, arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
        else:
            g = gates0(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
            if arm in ARMS_J:
                rec["sh_j1"] = J1C.SH_J1["result"]
                g = g + J1.j1_gates(arm, rd, L)
                if arm in J1.ARMS_J3:
                    g = g + J1.j3_gates(arm, rd, L)
                g = g + sheq_once()
        EQF[arm] = next((bool(x[2]) for x in g if x[0] == "%s-EQF" % arm), None)
        return g

    def verdicts_int(sh, sp, B, eqf=None):
        sp = sp or {}
        out = verd0(sh, sp, B, eqf)
        if any(a in sp for a in ARMS_J):
            out = out + J1C.verdicts(sh, sp, B, D)
        if any(a in sp for a in ST_ARMS):
            out = out + STCELL.verdicts(sp, B)
        if any(a in sp for a in BF_ARMS):
            out = out + BFCELL.verdicts(sp, B, EQF.get("L4PTFDGUDKWCDB"))
        return out + verdicts(sh, sp, B, D)

    D.run_arm_d, D.runner_choice, D.runner_class = run_arm_int, runner_choice_int, runner_class_int
    D.keep_ref_arm, D.arm_gates, D.verdicts = keep_ref_arm_int, arm_gates_int, verdicts_int
    D.BC = J1C._BCProxy(D.BC)                                            # SHJ1 after SH (the matched baseline)
    D.HC = _HCProxy(D.HC)                                                # the ST1 commit capture round (L61)
    D._INT_CONFIGURED = True
    DCC = D
    return D


def sheq_once() -> list:
    """J1-SHEQ (r2_j1_cell.sh_gate), emitted once per job with the first J1 / J3 / INT arm."""
    if J1C.SH_J1["emitted"]:
        return []
    J1C.SH_J1["emitted"] = True
    return [J1C.sh_gate()]


def int_arm_gates(gates0, arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa) -> list:
    """The MUST gates of an INT arm (module docstring). gates0: the copy's I4DC arm_gates."""
    d = ARMS_INT[arm]
    ref = keep.get(d)
    KC = STCELL.KC
    g = gates0(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
    if arm in INTA.BF_INT:                                             # BF1: its gate set against the D arm
        g = [x for x in g if not any(x[0] == arm + s for s in BFCELL.DROP)]
        g.append(BFCELL.counts_gate_bf(arm, cpu["counts"], ref["counts"] if ref else None,   # the store X declared
                                       (rd.get("bf") or {}).get("X")))                       # when --bf-store differs
        g.append(BFCELL.listed_gate(arm, rd.get("keep"), (ref or {}).get("keep"), d))
        g.append(BFCELL.bf_gate(arm, rd.get("bf"), "ce"))
    else:                                                              # no BF1 (C 128, and the C 73 arm of intd)
        c = KC.counts_gate(arm, cpu["counts"], ref["counts"] if ref else None, d)
        g.append((arm + "-DCOUNTS",) + tuple(c[1:]))
    g = g + J1.j1_gates(arm, rd, L)
    if arm not in INTA.KEEP_INT:
        g = g + J1.j3_gates(arm, rd, L)
    calls = ((rd.get("tent") or {}).get("calls")) or {}
    g.append(STCELL.peq_gate(arm, cpu, ref, d, steps))
    g.append(STCELL.tail_gate(arm, cpu.get("tails"), (ref or {}).get("tails"), d))
    g.append(STCELL.cgraph_gate(arm, rd.get("st1"), calls.get("commit")))
    g.append(STCELL.twin_gate(arm, rd.get("st2"), calls.get("joint_passes"), L))
    if arm in INT_ST3:                                                 # ST3: every keep fill through its helpers
        g.append(STCELL.kfill_gate(arm, rd.get("st3"), rd.get("compile")))
    if arm in INT_ST4:                                                 # ST4: every draft residency through its helper
        g.append(STCELL.dresid_gate(arm, rd.get("st4"), rd.get("dgraph"), L, rd.get("j3") is not None))
    g = g + INTA.int_gates(arm, rd, L)
    return g + sheq_once()


def hooks(arm: str) -> tuple:
    """The hooks of an INT arm in this cell (r2_int_arms.HOOKS + ST3 for INT_ST3 + ST4 for INT_ST4)."""
    return tuple(INTA.HOOKS[arm]) + (("ST3",) if arm in INT_ST3 else ()) + (("ST4",) if arm in INT_ST4 else ())


def int_arm(sp):
    """X: the INT arm of the job (the keep arm first); None if none ran."""
    for x in INTA.KEEP_INT + tuple(a for a in ARMS_INT if a not in INTA.KEEP_INT):
        if x in (sp or {}):
            return x
    return None


def verdicts(sh, sp, B: int, D=None):
    """PRED_INT_BY_B[B] against X = int_arm(sp) and REF = its D arm: [(name, verdict, detail)]; None gives REPORT."""
    D = D or configure()
    Xn = int_arm(sp)                                                   # the C 73 arm without BF1: its own bands
    PB = PRED_INT_NOBF_BY_B if Xn in NOBF_INT else PRED_INT_BY_B
    P = PB.get(B) or {k: None for k in PRED_KEYS_INT}
    out = []

    def band(name, v, b):
        if v is None:
            out.append((name, "REPORT", "no value (registered %s)" % (b,)))
        elif b is None:
            out.append((name, "REPORT", "%.4f (registered None)" % v))
        else:
            out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))

    def med(r):
        return ((r or {}).get("rates") or {}).get("median_tok_s")

    def ph(r, name):
        return ((r or {}).get("phases_median") or {}).get(name)

    def ratio(a, b):
        return (a / b) if (a is not None and b) else None

    def diff(a, b):
        return (a - b) if (a is not None and b is not None) else None
    sp = sp or {}
    mt, mtk = sp.get("L4PTFDGUDKWCDMT"), sp.get("L4PTFDGUDKWCDMTK")
    if mtk is not None:                                                # the ST3 control of the job
        band("INT-ST3-JOINT-SAVING", diff(ph(mt, "joint"), ph(mtk, "joint")), P.get("INT_st3_joint_saving"))
        band("INT-ST3-VS-MT", ratio(med(mtk), med(mt)), P.get("INT_st3_vs_mt"))
    X = int_arm(sp)
    if X is None:
        return out + [("INT-RUN", "NOT_RUN", "no INT arm ran")]
    x, r = sp[X], sp.get(ARMS_INT[X])
    shm = D.SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    s1 = J1C.SH_J1["result"] or {}
    band("INT-VS-REF", ratio(med(x), med(r)), P.get("INT_vs_ref"))
    band("INT-RATIO", ratio(med(x), shm), P.get("INT_ratio"))
    band("INT-VS-SHJ1", ratio(med(x), (s1.get("rates") or {}).get("steady_tok_s")), P.get("INT_vs_shj1"))
    band("INT-VS-STRONGEST", ratio(med(x), getattr(D, "STRONGEST", None)), P.get("INT_vs_strongest"))
    for nm in ("joint", "draft", "commit"):
        band("INT-%s-SAVING" % nm.upper(), diff(ph(r, nm), ph(x, nm)), P.get("INT_%s_saving" % nm))
    band("INT-PEAK-GB", D.IA.peak_gb(x), P.get("INT_peak_gb"))
    eqf, want = EQF.get(X), P.get("INT_eq")
    out.append(("INT-EQ", "REPORT" if (want is None or eqf is None) else ("CONFIRMED" if eqf == bool(want) else
                                                                          "REFUTED"),
                "%s-EQF passed %s; registered %s" % (X, eqf, want)))
    parts = {a: ratio(med(sp[a]), med(r)) for a in ("L4PTFDGUDCDJ1", "L4PTFDGUDKWCDJ1", "L4PTFDGUDCDJ3",
                                                    "L4PTFDGUDCDMT", "L4PTFDGUDKWCDMT", "L4PTFDGUDKWCDB") if a in sp}
    prod = None
    if parts and all(v is not None for v in parts.values()):
        prod = 1.0
        for v in parts.values():
            prod *= v
    out.append(("INT-ADDITIVITY", "REPORT", "single-rung arms vs %s in this job %s; their product %s; X / REF %s" % (
        ARMS_INT[X], {k: None if v is None else round(v, 4) for k, v in parts.items()},
        None if prod is None else round(prod, 4), None if ratio(med(x), med(r)) is None else round(ratio(med(x), med(r)),
                                                                                                   4))))
    out.append(("INT-ARM", "REPORT", "X = %s, REF = %s (REF ran: %s); hooks %s; INT record %s; ST3 record %s" % (
        X, ARMS_INT[X], r is not None, hooks(X), x.get("int"), x.get("st3"))))
    return out


def parse_own(argv):
    """This cell's own options (stripped before the I4DC main): (options, the rest)."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--sh-j1", type=int, default=1, choices=(0, 1))
    ap.add_argument("--st1-commit", default="rows", choices=STC.MODES)
    ap.add_argument("--bf-store", type=int, default=0, help="X of the BF1 arms and the INT keep arm (0: --fork-store's)")
    ap.add_argument("--bf-ncap", type=int, default=2048)
    ap.add_argument("--bf-ring", type=int, default=8)
    ap.add_argument("--bf-worker", default="thread", choices=BF.WORKERS)
    ap.add_argument("--bf-acct", type=int, default=0, choices=(0, 1))
    ap.add_argument("--bf-close-wait-ms", type=float, default=0.0)
    ap.add_argument("--bf-packer", default="gilfree", choices=BF.PACKERS)
    ap.add_argument("--bf-threads", type=int, default=14)
    ap.add_argument("--bf-cap-groups", type=int, default=2048)
    ap.add_argument("--bf-batch", type=int, default=8)
    ap.add_argument("--bf-late", default="defer", choices=BF.LATES)
    ap.add_argument("--bf-late-wait-ms", type=float, default=250.0)
    return ap.parse_known_args(argv)


def check_arms(arms, own, D) -> str:
    """The arm-order and option rules of this cell beyond the I4DC parse (None: fine; else the error)."""
    for x, ref in ST_ARMS.items():
        if x in arms and (ref not in arms or arms.index(ref) > arms.index(x)):
            return "--arms: %s needs %s earlier (its PEQ reference)" % (x, ref)
    for x in BF_ARMS:
        if x in arms and BFCELL.REF not in arms:
            return "--arms: %s needs %s earlier (its LISTED reference)" % (x, BFCELL.REF)
    bound = D.I3C.KA.store_slots(D.TOPK, 3)
    if own.bf_store != 0 and own.bf_store < bound:
        return "--bf-store %d: below the bound %d (0 = --fork-store's)" % (own.bf_store, bound)
    if own.bf_ncap < 1 or own.bf_ring < 2 or own.bf_close_wait_ms < 0 or not 1 <= own.bf_threads <= 64:
        return "--bf-ncap >= 1, --bf-ring >= 2, --bf-close-wait-ms >= 0, --bf-threads in 1 .. 64"
    if not 16 <= own.bf_cap_groups <= 4096 or own.bf_late_wait_ms < 0 or not 1 <= own.bf_batch <= 64:
        return "--bf-cap-groups in 16 .. 4096, --bf-late-wait-ms >= 0, --bf-batch in 1 .. 64 (r2_bf1_cell's rules)"
    bfa = [x for x in arms if x in BF_ARMS or x in INTA.BF_INT]
    if own.bf_packer == "gilfree" and bfa and not os.environ.get("TORCH_EXTENSIONS_DIR"):
        return "--bf-packer gilfree: TORCH_EXTENSIONS_DIR must name the run's build directory (never HOME)"
    return None


def report_lines() -> list:
    """The BF1 and INT report lines of the arms that ran (after the I4DC main's lines)."""
    out = []
    for name, r in RECS.items():
        if name in BF_ARMS or name in INTA.BF_INT:
            out.extend(BFCELL.bf_lines(name, r))
        if name in ARMS_INT:
            out.append("INT %s: hooks %s; record %s; st1 %s; st2 %s; j1 calls %s f6 calls %s; j3 %s; st3 %s; st4 %s" % (
                name, hooks(name), r.get("int"), r.get("st1"), r.get("st2"), (r.get("j1") or {}).get("calls"),
                (r.get("j1") or {}).get("f6calls"), r.get("j3"), r.get("st3"), r.get("st4")))
    return out


def main(argv=None) -> int:
    """The I4DC cell's main on the private copy, after this cell's own options (parse_own) and arm rules."""
    argv = list(sys.argv[1:] if argv is None else argv)
    own, rest = parse_own(argv)
    D = configure()
    a = D.parse_args(rest)                                             # the I4DC rules (exit 2 on an error)
    err = check_arms(a.arms, own, D)
    if err is not None:
        print("r2_int_cell: error: %s" % err, file=sys.stderr)
        return 2
    J1C.SH_J1["enabled"] = own.sh_j1 == 1
    OPT.update(st1_commit=own.st1_commit, bf_store=own.bf_store, bf_ncap=own.bf_ncap, bf_ring=own.bf_ring,
               bf_worker=own.bf_worker, bf_acct=own.bf_acct, bf_close_wait_ms=own.bf_close_wait_ms,
               bf_packer=own.bf_packer, bf_threads=own.bf_threads, bf_cap_groups=own.bf_cap_groups,
               bf_late=own.bf_late, bf_late_wait_ms=own.bf_late_wait_ms, bf_batch=own.bf_batch)
    rc = D.main(rest)
    for line in report_lines():
        print(line)
    print("r2_int_cell: options %s; INT arms %s" % (OPT, [x for x in a.arms if x in ARMS_INT]))
    return rc


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
