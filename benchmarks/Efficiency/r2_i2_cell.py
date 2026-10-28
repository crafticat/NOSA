"""Rung I2 of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of the arm
L4PTU = the tentative buffers of WS-A (I1, spec_tentative.TentRunner, the schedule of contract 4) with the union residency
of WS-B (ONE residency per layer for the union of the joint pass's rows): nosi/nosi/spec_tentative_union.py, opt-in
--joint-residency union. A NEW script: r2_wsa_cell.py, r2_wsb_cell.py, r2_spec_cell.py and r2_base_cell.py are not edited
(their functions are imported: the I1 arm runs r2_wsa_cell.run_arm, the arm of the wa1 cells, unchanged). Step script:
benchmarks/Efficiency/r2_i2_step.sh (cells i2m, i2a, i2b, i2ca, i2cb). Docs: docs/NOTES_i2.md.

ONE PROCESS, ONE MODEL, the arms in the order given, each with its own shipped prefill; SH runs FIRST. The inputs of
sp3a / sp3b and wa1a / wa1b (the same workload pin, S0, document rows, K 3, 512 steps, fused twin and body, graph,
redirect, probe 16), so extra work is told apart from slower execution and the rung's INCREMENTAL change (I2 against I1)
is measured in ONE job:
  SH      the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
TIMING ARMS (no check-only work in the timing cells; accounting off):
  L4PT    I1: spec_tentative.TentRunner through r2_wsa_cell.run_arm (gemm rowgroup; the serial residency per row)
  L4PTU   I2: spec_tentative_union.TentRunnerU (gemm --tu-gemm, default natural = WS-B's production GEMM; one residency
          per layer and joint pass; X = --joint-extra store slots, the store fills C from the bound)
  L4PTUR  (optional) L4PTU with gemm rowgroup: L4PTU - L4PTUR is the GEMM part of the I2 change
  L4PJ    (optional) the EXACT journal arm of the same schedule (r2_wsa_cell.run_arm): the token proof of the schedule
CHECK ARMS (not timed; the coordinator's binding correction: FAST is checked on itself against SH):
  SHR     r2_wsb_cell.run_shr: the serial ragged step teacher-forced on SH's tokens for 1 + G R steps; its selections, top-16
          logits and fingerprints (MUST: its fingerprints and argmaxes equal SH's)
  TUC     spec_tentative_union.TentRunnerU.run_teacher (gemm natural), teacher-forced on SH's tokens: one plain step, then
          G joint passes of R = K + 1 verify rows, each committed whole, CHECK ON: per pass, layer, row, head and request
          the selection SET against SHR's at the same position (MUST, with the symmetric-difference sizes), the PRESENCE
          recount (dropped 0), the VISIBLE COUNT (vis_bad 0), one residency per layer and pass, VRF-IDX of the window, the
          first store and the buffer call; the logits deltas against SHR's top-16 and fingerprints; one PROFILED pass
  TUCR    TUC with gemm rowgroup (SH's GEMM bits: a change there is the union attention's float order alone)
WS-G CHECK ARMS (cell i2g; the lead, 2026-10-28T13:19Z; r2_wsg_check.py of b1ae579 copied unchanged, with
r2_i2_check.ContractCapture for the contract-4 runners): the arm with the capture, then REF = SH's computation
teacher-forced on the arm's own committed tokens; parts MAP, REF, SEL, GREEDY (tie 0), POS, SLOT, MISS / BITS:
  WJ      L4PJ, the EXACT positive control: WJ-CHK MUST (every part passes)
  WT, WU  L4PT, L4PTU: every gate REPORT
  WUP     L4PTU with drop_spill at X = --chk-plant-extra: the planted control; MUST WUP-PLANT-CHK-PRE and -CAUGHT
MEASURED per timing arm (CUDA events; device counters read once after the arm): r2_wsa_cell.summarize_w's record
(committed tok/s median / mean / effective against SH and 3,407.2; round phases reset, joint, commit, draft; ROLLBACK per
round; the D1 / D3 calls; bytes per round; memory and the fitting batch at 84 GB, MODEL; tokens against SH; the contract-4
schedule record), and for L4PTU: the joint pass in SH steps, the union host counts per joint pass (residency calls,
attention calls, gathers, store fills, GEMM calls), the joint pass's PHASES on the probe rounds (spec_tentative_union's
marks: resid, write, union, fetch, attn, spill, buffer, merge, post; and the complete layer time).
MUST GATES: SH-RUN; per timing arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN (r2_wsa_cell.arm_gates_w; L4PJ -TOK / -FP);
L4PTU -RESID (one residency per layer and joint pass, the union path ran for every joint pass); with check on (i2m)
L4PTU -PRESENT, -VIS, -IDX; with --require-stress (i2m) <arm>-STRESS (WS-A's) and L4PTU-USTRESS (the spill ran);
check arms SHR-RUN / -FP / -TOK, TUC -RUN, -ERR, -SEL, -PRESENT, -VIS, -RESID, -IDX (r2_wsb_cell.check_gates), -IDXT.
REGISTERED PREDICTIONS PRED_I2 / PRED_I2_BY_B: every value None (REPORT) until the lead registers them; the step script
STOPS the timing cells (i2a, i2b) while PRED_STAMP_I2 is None. Inferred bands with their arithmetic: docs/NOTES_i2.md.
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_wsa_cell as WA          # noqa: E402
import r2_wsb_cell as WB          # noqa: E402
import r2_wsg_check as CK         # noqa: E402  (WS-G b1ae579, copied unchanged)

SC = WA.SC
BC = WA.BC
ARMS = ("L4PT", "L4PTU", "L4PTUR", "L4PJ", "SHR", "TUC", "TUCR", "WJ", "WT", "WU", "WUP")
TIMING_ARMS = ("L4PT", "L4PTU", "L4PTUR", "L4PJ")
UNION_TIMING = ("L4PTU", "L4PTUR")
CHECK_ARMS = ("SHR", "TUC", "TUCR")
# WS-G's CHECK arms (r2_wsg_check.py of b1ae579, copied unchanged; r2_i2_check.ContractCapture maps contract-4 rows):
# the arm with the capture, then REF = SH's computation teacher-forced on the arm's own committed tokens
WSG_ARMS = dict(WJ=dict(base="L4PJ", exact=True, tentative=False),       # the EXACT positive control: MUST pass
                WT=dict(base="L4PT", exact=False, tentative=True),       # I1: REPORT
                WU=dict(base="L4PTU", exact=False, tentative=True),      # I2: REPORT
                WUP=dict(base="L4PTU", exact=False, tentative=True, plant="drop_spill"))   # the PLANTED control
UNION_ARMS = ("L4PTU", "L4PTUR", "TUC", "TUCR", "WU", "WUP")
REF_OPS = dict(gemm="natural", twin="fused", body="fused", graph=True)     # r2_wsg_cell.REF_OPS (rg73's RG)
CHK_TIE_LOGIT = 0.0                # r2_wsg_cell.CHK_TIE_LOGIT as the lead registered it (b1ae579): no near-tie exemption
CHECK_GEMM = dict(TUC="natural", TUCR="rowgroup")
STRONGEST = WA.STRONGEST
# the joint pass's phase marks (spec_tentative_union, probe rounds): each label runs to the next mark of the pass
PHASE_GROUPS_U = dict(residency=("resid",), write=("write",), union=("union",), fetch=("fetch",),
                      attention=("attn", "buffer", "merge"), spill=("spill",), dense_next=("post",))
# REGISTERED PREDICTIONS, TBD BY THE LEAD: fill PRED_I2_BY_B (per batch: the two timing cells differ in kind, as
# r2_wsa_cell.PRED_WSA_BY_B) and PRED_STAMP_I2 (`date -u`, L33) in a commit BEFORE i2a / i2b (the step script stops while
# PRED_STAMP_I2 is None). PRED_I2_BY_B[B] overrides PRED_I2 key by key; a batch without an entry uses PRED_I2.
#   I2_ratio         (low, high): L4PTU median steady committed tok/s / SH's steady tok/s
#   I2_vs_I1         (low, high): L4PTU / L4PT median steady committed tok/s (the rung's incremental change, same job)
#   I2_joint_sh      (low, high): L4PTU median joint-phase ms / SH's steady step ms
#   I2_joint_saving  (low, high): L4PT joint ms - L4PTU joint ms (medians)
#   I2_rollback_sh   (low, high): L4PTU (reset + commit) ms / SH step (D1 is kept: about I1's)
#   I1_ratio         (low, high): L4PT / SH in this job (the previous rung, re-measured)
#   I2_agree         a floor: the fraction of L4PTU's committed tokens equal to SH's (REPORT until WS-G's gate)
#   I2_plan          the plan's I2 point (PLAN_L4P_faithful.md: 0.52 at C 128 B 216, 0.63 at C 73 B 318): the verdict
#                    I2-PLAN-15 is CONFIRMED when I2_ratio is within +-15 % of it (the plan's rung falsifier)
#   I2_above_I1      True: the rung falsifier "measured ratio below the previous rung's" (I2-ABOVE-I1)
# The lead's REGISTRATION (2026-10-28T15:18:51Z, `date -u`, after WS-A's I1 cells and BEFORE any I2 timing cell).
# MEASURED I1 inputs (r2-runs/spec/wa_k2a_2195574_wa1a.step and wa_k2b_2195574_wa1b.step, stdout REPORT L4PT): L4PT /
# SH 0.3927 (C 128 B 216) and 0.4833 (C 73 B 318); L4PT phases per round: joint 315.264 / 518.766 ms, draft 150.473 /
# 194.389, commit 6.158 / 8.526, reset 0.11; SH step 64.394 / 123.967 ms; rollback 0.0974 / 0.0696 SH steps; equal-token
# fraction 0.403 / 0.389. MODEL for L4PTU (NOTES_i2.md section 5.1): joint about 212 ms (C 128) / 392 ms (C 73) from WS-B's
# union pass (not yet measured on GPU) plus the GEMM change (natural vs rowgroup, about -27 ms at C 128); band edges joint
# 180-300 / 330-500 ms; the drafts after a union pass may lose window hits (unknown). Round I1 about 472 / 722 ms, so I2 /
# I1 about 1.28 / 1.21 central.
PRED_STAMP_I2 = "2026-10-28T15:18:51Z"
PRED_KEYS = ("I2_ratio", "I2_vs_I1", "I2_joint_sh", "I2_joint_saving", "I2_rollback_sh", "I1_ratio", "I2_agree", "I2_plan",
             "I2_above_I1")
PRED_I2 = dict(stamp=PRED_STAMP_I2, **{k: None for k in PRED_KEYS})
PRED_I2_BY_B = {
    216: dict(I2_ratio=(0.38, 0.57), I2_vs_I1=(0.98, 1.45), I2_joint_sh=(2.6, 4.7), I2_joint_saving=(0.0, 140.0),
              I2_rollback_sh=(0.03, 0.30), I1_ratio=(0.37, 0.42), I2_agree=0.25, I2_plan=0.52, I2_above_I1=True),
    318: dict(I2_ratio=(0.46, 0.68), I2_vs_I1=(0.97, 1.40), I2_joint_sh=(2.5, 4.1), I2_joint_saving=(0.0, 190.0),
              I2_rollback_sh=(0.03, 0.30), I1_ratio=(0.45, 0.51), I2_agree=0.25, I2_plan=0.63, I2_above_I1=True),
}


def log(msg):
    print("[r2_i2 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4: Figure 3 with K = 3)")
    ap.add_argument("--arms", default="L4PT,L4PTU", help="a subset of %s, in run order (SH runs first)" % ",".join(ARMS))
    ap.add_argument("--joint-residency", default="serial", choices=("serial", "union"),
                    help="THE FLAG of I2: the arms %s run only with 'union'" % (UNION_ARMS,))
    ap.add_argument("--joint-extra", type=int, default=48, help="X: store slots per request (slot 0 unused by L4PTU)")
    ap.add_argument("--joint-spill", type=int, default=0,
                    help="store fills per layer (0: the bound ceil((R - 1) topk / (X - 1)): nothing can be dropped)")
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"), help="L4PTU's joint-pass GEMM")
    ap.add_argument("--check-passes", type=int, default=32, help="TUC / TUCR: teacher-forced joint passes G")
    ap.add_argument("--spec-journal-blocks", type=int, default=256, help="L4PJ only")
    ap.add_argument("--spec-extra", type=int, default=16, help="L4PT / L4PJ (r2_wsa_cell's --spec-extra; unused there)")
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--check-every", type=int, default=0,
                    help="timing arms: check rounds (0 = the probe rounds; 1 = every round and the union checks on: "
                         "the miniature)")
    ap.add_argument("--profile", default="on", choices=("on", "off"), help="check arms: one profiled pass / step")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    ap.add_argument("--chk-rows", default="", help="WS-G arms: the sample rows (logits, slots), comma separated; empty: "
                                                   "8 rows spread over the batch")
    ap.add_argument("--chk-slot-every", type=int, default=32, help="WS-G arms: the slot check every N rounds (and at "
                                                                   "the end)")
    ap.add_argument("--chk-plant-extra", type=int, default=4,
                    help="WUP: X of the planted control (drop_spill keeps ONE store fill of X - 1 blocks)")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    un = [x for x in a.arms if x in UNION_ARMS]
    if un and a.joint_residency != "union":
        ap.error("the arms %s need --joint-residency union" % un)
    if a.joint_residency == "union" and not un:
        ap.error("--joint-residency union without a union arm (%s)" % (UNION_ARMS,))
    ck = [x for x in a.arms if x in ("TUC", "TUCR")]
    if ck and ("SHR" not in a.arms or a.arms.index("SHR") > min(a.arms.index(x) for x in ck)):
        ap.error("TUC / TUCR compare against SHR: SHR must run before them")
    if not 3 <= a.spec <= 18:
        ap.error("--spec K in 3 .. 18 (contract 4 needs K >= 3)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if any(x in CHECK_ARMS for x in a.arms) and (a.check_passes < 1 or 1 + a.check_passes * (a.spec + 1) > a.steps):
        ap.error("--check-passes G >= 1 with 1 + G (K + 1) <= --steps (SHR's steps; the check arms)")
    if (a.joint_extra < 2 or a.joint_spill < 0 or a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1
            or a.spec_extra < 2):
        ap.error("--joint-extra >= 2, --joint-spill >= 0, --probe-every >= 2, --check-every >= 0, journal >= 1, "
                 "--spec-extra >= 2")
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    if a.chk_slot_every < 1 or a.chk_plant_extra < 2:
        ap.error("--chk-slot-every >= 1, --chk-plant-extra >= 2")
    if "WUP" in a.arms and "WJ" not in a.arms:
        ap.error("the planted control WUP needs the positive control WJ in the same cell")
    try:
        a.sample = [int(x) for x in a.chk_rows.split(",") if x] if a.chk_rows else None
    except ValueError:
        ap.error("--chk-rows: integers")
    if a.sample is not None and (not a.sample or any(not 0 <= b < a.batch for b in a.sample)):
        ap.error("--chk-rows: rows in 0 .. batch - 1")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def union_groups(ph: dict) -> dict:
    """{label: ms} of one probe pass -> {group: ms} (PHASE_GROUPS_U) plus "total"."""
    out = {g: round(sum(ph.get(x, 0.0) for x in labs), 4) for g, labs in PHASE_GROUPS_U.items()}
    out["total"] = round(sum(ph.values()), 4)
    return out


def union_phase_summary(phases, layers):
    """The probe passes' union phases: medians per label and group, the median complete layer time."""
    if not phases:
        return dict(n_probes=0)
    labels = sorted({k for p in phases for k in p})
    med = {k: round(statistics.median([p.get(k, 0.0) for p in phases]), 4) for k in labels}
    grp = [union_groups(p) for p in phases]
    gmed = {g: round(statistics.median([x[g] for x in grp]), 4) for g in grp[0]}
    lay = [x for ls in layers for x in ls]
    return dict(n_probes=len(phases), labels_median=med, groups_median=gmed,
                layer_ms_median=(round(statistics.median(lay), 4) if lay else None),
                layers_per_pass=(len(layers[0]) if layers else None))


def union_record(rd, L: int) -> dict:
    """L4PTU's union facts from read(): host counts per joint pass, the residency count, the check / accounting
    counters (zero when off)."""
    tr, u = rd["tent"], rd.get("union") or {}
    c = tr["calls"]
    jp = c.get("joint_passes", 0)
    calls = u.get("calls") or {}
    per = {k: (round(v / jp, 3) if jp else None) for k, v in calls.items()}
    keys = ("extras", "extras_pool", "extras_host", "extras_max", "spill", "unserved", "fills_used", "spill_fills_used",
            "dropped", "skipped", "vis_bad", "visible_blocks", "rows_full", "rows_res", "win_sel", "win_host",
            "streamed_rows", "union_mean")
    return dict(joint_passes=jp, joint_resid=c.get("joint_resid"), union_layers=c.get("union_layers"),
                resid_per_joint_pass=(round(c.get("joint_resid", 0) / jp, 3) if jp else None), layers=L,
                calls_per_joint_pass=per, fills=u.get("fills"), cap=u.get("cap"),
                counters={k: u.get(k) for k in keys}, rows=dict(dropped=u.get("row_dropped"),
                                                                skipped=u.get("row_skipped"),
                                                                vis_bad=u.get("row_vis_bad"),
                                                                visible=u.get("row_visible"),
                                                                selected=u.get("row_selected")))


def union_gates(arm, res, L: int, check: bool, require_stress: bool = False):
    """The I2 MUST gates of a union timing arm: [(name, must, passed, detail)] (module docstring)."""
    rd = res["read"]
    c = rd["tent"]["calls"]
    u = rd.get("union") or {}
    jp = c.get("joint_passes", 0)
    g = [("%s-RESID" % arm, True, jp > 0 and c.get("joint_resid") == L * jp and c.get("union_layers") == L * jp
          and u.get("passes") == jp,
          "residency calls in joint passes %s for %d joint passes of %d layers (one per layer and pass); union layers "
          "%s; verify_rows passes %s" % (c.get("joint_resid"), jp, L, c.get("union_layers"), u.get("passes")))]
    if check:
        g.append(("%s-PRESENT" % arm, True, u.get("dropped") == 0 and u.get("unserved") == 0 and
                  (u.get("union_samples") or 0) > 0,
                  "selected blocks of verify rows not present: %s (recount); unserved %s; spill %s of %s extras" % (
                      u.get("dropped"), u.get("unserved"), u.get("spill"), u.get("extras"))))
        g.append(("%s-VIS" % arm, True, u.get("vis_bad") == 0 and (u.get("visible_blocks") or 0) > 0,
                  "active rows whose visible block count is not their selection size (minus a draft row's skips): %s "
                  "(per row %s); visible %s, selected %s, skipped %s per row index" % (
                      u.get("vis_bad"), u.get("row_vis_bad"), u.get("row_visible"), u.get("row_selected"),
                      u.get("row_skipped"))))
        fa = res.get("fa_check") or {}
        ok = all(k in fa and fa[k]["entries"] and all(fa[k]["equal"]) for k in ("A", "T")) and (
            ("B" in fa and all(fa["B"]["equal"])) or not u.get("extras"))
        g.append(("%s-IDX" % arm, True, ok, "window call %s; first store call %s; buffer call %s" % (
            fa.get("A"), fa.get("B"), fa.get("T"))))
    if require_stress:
        g.append(("%s-USTRESS" % arm, True, (u.get("spill") or 0) > 0 and (u.get("spill_fills_used") or 0) > 0
                  and (u.get("extras") or 0) > 0,
                  "extras %s; spill %s blocks; spill fills (c >= 1) that served an entry %s (each > 0; accounting on)" % (
                      u.get("extras"), u.get("spill"), u.get("spill_fills_used"))))
    return g


def summarize_u(arm, res, inputs, fp_sh, sh, B: int, steps: int, dims):
    """A union timing arm's record: r2_wsa_cell.summarize_w's tentative record (as L4PT, renamed) plus the I2 fields."""
    rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, B, steps, dims)
    rec["arm"] = arm
    rec["union"] = union_record(res["read"], dims["L"])
    rec["joint_phases_union"] = res.get("joint_phases")
    jm = rec["phases_median"].get("joint")
    shm = sh["rates"]["steady_step_ms"] if sh else None
    rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
    rec["meta_union"] = (res.get("meta") or {}).get("spec", {}).get("union")
    return rec


def arm_gates_u(arm, res, rec, steps: int, L: int, check: bool, require_stress: bool = False):
    """WS-A's gates of a tentative arm (as L4PT, renamed) and the I2 gates."""
    g = [(n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
         WA.arm_gates_w("L4PT", res, rec, steps, require_stress=require_stress)]
    return g + union_gates(arm, res, L, check, require_stress)


def check_gates_u(arm, res, require_stress: bool = False, L: int = 32):
    """A union check arm (TUC / TUCR): r2_wsb_cell.check_gates (ERR, SEL, PRESENT, VIS, RESID, IDX, STRESS) and IDXT (the
    buffer call's VRF-IDX)."""
    g = list(WB.check_gates(arm, res, require_stress=require_stress, L=L))
    fa = res.get("fa_check") or {}
    t = fa.get("T")
    g.append(("%s-IDXT" % arm, True, bool(t and t["entries"] and all(t["equal"])),
              "buffer call (64-row tentative buffer, cache_batch_idx, per-entry bias) %s" % (t,)))
    return g


def verdicts(sh, sp, B: int):
    """PRED_I2 (PRED_I2_BY_B[B] key by key) against the timing arms: [(name, verdict, detail)]; None registered gives
    REPORT; NOT_RUN without the arm."""
    P = dict(PRED_I2)
    P.update(PRED_I2_BY_B.get(B) or {})
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

    def joint(r):
        return ((r or {}).get("phases_median") or {}).get("joint")
    u, t = (sp or {}).get("L4PTU"), (sp or {}).get("L4PT")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    sh_ms = sh["rates"]["steady_step_ms"] if sh else None
    if t is not None:
        band("I1-RATIO", (med(t) / shm) if (med(t) and shm) else None, P["I1_ratio"])
    if u is None:
        out.append(("L4PTU-RUN", "NOT_RUN", "the arm did not run"))
        return out
    r2 = (med(u) / shm) if (med(u) and shm) else None
    band("I2-RATIO", r2, P["I2_ratio"])
    band("I2-VS-I1", (med(u) / med(t)) if (med(u) and med(t)) else None, P["I2_vs_I1"])
    band("I2-JOINT-SH", (joint(u) / sh_ms) if (joint(u) and sh_ms) else None, P["I2_joint_sh"])
    band("I2-JOINT-SAVING", (joint(t) - joint(u)) if (joint(t) is not None and joint(u) is not None) else None,
         P["I2_joint_saving"])
    band("I2-ROLLBACK-SH", u["rollback_per_round"]["sh_steps"], P["I2_rollback_sh"])
    tc = u["tokens"]
    want = P["I2_agree"]
    out.append(("I2-AGREE", "REPORT" if want is None else ("CONFIRMED" if tc["equal_fraction"] >= want else "REFUTED"),
                "committed tokens equal to SH's: %.6f (first difference %s); registered >= %s" % (
                    tc["equal_fraction"], tc["first"], want)))
    plan = P["I2_plan"]
    if plan is None or r2 is None:
        out.append(("I2-PLAN-15", "REPORT", "ratio %s; the plan's point registered %s" % (r2, plan)))
    else:
        out.append(("I2-PLAN-15", "CONFIRMED" if abs(r2 / plan - 1.0) <= 0.15 else "REFUTED",
                    "ratio %.4f against the plan's %.3f: %+.1f %% (falsifier outside +-15 %%)" % (
                        r2, plan, 100.0 * (r2 / plan - 1.0))))
    above = P["I2_above_I1"]
    if above is None or not (med(u) and med(t)):
        out.append(("I2-ABOVE-I1", "REPORT", "L4PTU %s vs L4PT %s tok/s; registered %s" % (med(u), med(t), above)))
    else:
        out.append(("I2-ABOVE-I1", "CONFIRMED" if (med(u) > med(t)) == bool(above) else "REFUTED",
                    "L4PTU %.1f vs L4PT %.1f tok/s (the rung falsifier: below the previous rung); registered %s" % (
                        med(u), med(t), above)))
    out.append(("I2-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s (the plan's I2: 0.51 / 0.48)" % (
        STRONGEST, u["vs_strongest"])))
    return out


# ------------------------------------------------------------------ the union timing arm (GPU)
def run_arm_u(model, ids, inputs, steps: int, clock, opt):
    """ONE union timing arm (L4PTU / L4PTUR): r2_wsa_cell.run_arm's steps and memory measures with
    spec_tentative_union.TentRunnerU (phase marks on the probe rounds; check-only work and accounting only with
    --check-every 1, the miniature)."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_union as STU
    from nosi import verify_rows as VR
    cfg = model.config
    WB._release_warm(model)
    gb = 1024 ** 3
    start = torch.cuda.memory_allocated() / gb
    torch.cuda.reset_peak_memory_stats()
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    clock.sync()
    t0 = time.time()
    logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    first_agree = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    prefill_peak = torch.cuda.max_memory_allocated() / gb
    K = int(opt["K"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()                              # the decode peak (the prefill peak is above)
    gemm = opt["tu_gemm"] if opt["arm"] == "L4PTU" else "rowgroup"
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=gemm, extra=opt["X"], journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"])
    check = opt["check_every"] == 1
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    runner = STU.TentRunnerU(model, rs, sops, clock, t0=inputs[:, 0], hooks=None, check_every=ce, union=True,
                             union_check=check, union_stats=check, spill=(opt["spill"] or None), phase_marks=True)
    if check:
        runner.vs.fa_check = {}                                        # CHECK-ONLY: VRF-IDX (one host read per tag)
    runner.run(steps)
    rd = runner.read()
    rd.pop("trace_all", None)
    clock.sync()
    mem = dict(start_GB=round(start, 3), prefill_peak_GB=round(prefill_peak, 3),
               decode_peak_GB=round(torch.cuda.max_memory_allocated() / gb, 3),
               decode_peak_reserved_GB=round(torch.cuda.max_memory_reserved() / gb, 3))
    phases = [VR.phase_ms(m, clock.elapsed) for m in runner.marks if m]
    layers = [VR.layer_ms(m, clock.elapsed) for m in runner.marks if m]
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()),
               fp_ok=runner.fp_ok.clone(), K=K, probe_every=opt["probe_every"], prefill_s=round(prefill_s, 1),
               first_token_agreement=first_agree, meta=dict(rs.meta, spec=runner.meta), undo=None, mem=mem,
               joint_phases=union_phase_summary(phases, layers), fa_check=runner.vs.fa_check)
    runner.detach()
    del runner, rs, cache
    return res


# ------------------------------------------------------------------ the union check arm (GPU)
def run_check_tu(model, ids, inputs, steps: int, clock, opt, shr):
    """TUC / TUCR (module docstring): TentRunnerU.run_teacher on SH's tokens, check on, against SHR (whose step i is
    row r of pass g with i = 1 + g R + r)."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_union as STU
    from nosi import verify_rows as VR
    cache, prefill_s, first = WB._prefill(model, ids, inputs, steps, clock)
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=False)
    rs = RE.attach(model, cache, BC.S0, steps + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()
    K = int(opt["K"])
    R = K + 1
    B, L = rs.B, rs.L
    G = int(opt["passes"])
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=CHECK_GEMM[opt["arm"]], extra=opt["X"], journal=8,
                      redirect=True, probe_every=opt["probe_every"])
    runner = STU.TentRunnerU(model, rs, sops, clock, t0=inputs[:, 0], hooks=None, check_every=1, union=True,
                             union_check=True, union_stats=True, spill=(opt["spill"] or None))
    vs = runner.vs
    vs.fa_check = {}
    every = int(opt["probe_every"])
    is_probe = lambda g: every > 0 and g > 0 and g % every == 0         # noqa: E731
    g_prof = next((g for g in range(G // 2, G) if not is_probe(g)), None)   # the first non-probe pass from G // 2
    box = dict(prof=None)
    run0 = runner.T.run_pass

    def run_pass(model_, plan):                                       # CHECK-ONLY: one profiled joint pass
        if opt["profile"] and plan.R > 1 and runner.round == g_prof and box["prof"] is None:
            out, box["prof"] = WB._profile(lambda: run0(model_, plan))
            return out
        return run0(model_, plan)
    runner.T.run_pass = run_pass
    changed_lr = torch.zeros((L, R), dtype=torch.int64, device="cuda")
    hist = torch.zeros(6, dtype=torch.int64, device="cuda")           # symmetric-difference sizes 2, 4, .., 10, 12+
    st = dict(compared=0, first=[], d_top=[], d_max=[], agree=[])

    def on_pass(g, lg):                                               # CHECK-ONLY: against SHR at steps 1 + g R + r
        i0 = 1 + g * R
        ref = shr["sel"][i0:i0 + R].to("cuda").to(torch.int64)       # (R, L, H, B, T)
        for l in range(L):
            cur = vs.chk_sel[l]                                       # (H, B, R, T)
            rf = ref[:, l].permute(1, 2, 0, 3)
            nb = ~(cur.sort(-1).values == rf.sort(-1).values).all(-1)
            changed_lr[l] += nb.sum((0, 1))
            sym = 2 * (~(cur.unsqueeze(-1) == rf.unsqueeze(-2)).any(-1)).sum(-1)
            hist.add_(torch.bincount((sym // 2 - 1).clamp(min=0, max=5)[nb], minlength=6)[:6])
            if len(st["first"]) < 5 and bool(nb.any()):
                for h, b, r in nb.nonzero().tolist()[:5 - len(st["first"])]:
                    st["first"].append((g, l, r, h, b, int(sym[h, b, r])))
        st["compared"] += L * vs.H * B * R
        tv = shr["topv"][i0:i0 + R].to("cuda")                        # (R, B, 16)
        ti = shr["topi"][i0:i0 + R].to("cuda")
        lgr = lg.permute(1, 0, 2)                                     # (R, B, V)
        st["d_top"].append((lgr.gather(-1, ti) - tv).abs().amax(-1).cpu())
        st["d_max"].append((lgr.amax(-1) - shr["fmax"][:, i0:i0 + R].t().to("cuda")).abs().cpu())
        st["agree"].append((lg.argmax(-1) == inputs[:, i0 + 1:i0 + R + 1]).cpu())
    runner.run_teacher(inputs, G, on_pass=on_pass)
    clock.sync()
    stats = VR.read_stats(vs)
    dt = torch.stack(st["d_top"])                                    # (G, R, B)
    dm = torch.stack(st["d_max"])
    ag = torch.stack(st["agree"])                                    # (G, B, R)
    by_row = {r: dict(top16_abs=WB.quantiles(dt[:, r].reshape(-1).tolist()),
                      max_abs=WB.quantiles(dm[:, r].reshape(-1).tolist()),
                      argmax_agree=round(float(ag[:, :, r].float().mean()), 6)) for r in range(R)}
    cl = changed_lr.cpu()
    sel_rec = dict(compared=st["compared"], changed=int(cl.sum()),
                   by_layer_nonzero={l: cl[l].tolist() for l in range(L) if int(cl[l].sum())},
                   by_row=cl.sum(0).tolist(), symdiff_hist=dict(zip(("2", "4", "6", "8", "10", "12+"),
                                                                    hist.cpu().tolist())), first=st["first"])
    tr = runner.T.report()
    res = dict(stats=stats, sel=sel_rec, logits=dict(by_row=by_row, top16_abs=WB.quantiles(dt.reshape(-1).tolist()),
                                                   argmax_agree=round(float(ag.float().mean()), 6)),
               errors=RE.errors(rs), fa_check=vs.fa_check, passes=G, R=R, tent_calls=tr["calls"],
               calls_per_pass=WB.per_pass(stats["calls"], stats["passes"]), profile=box["prof"], prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, spec=runner.meta),
               ready=dict(win_sel=stats["win_sel"], win_host=stats["win_host"], extras_pool=stats["extras_pool"],
                          extras_host=stats["extras_host"]),
               roles=dict(rows_full=stats["rows_full"], rows_res=stats["rows_res"], passes=G))
    runner.detach()
    del runner, vs, rs, cache
    return res


# ------------------------------------------------------------------ WS-G's check arms (GPU)
def wsg_gates(arm, cap_result, ref, tie=CHK_TIE_LOGIT):
    """r2_wsg_check.chk_gates for a WS-G arm of this cell: WJ (the EXACT positive control) <arm>-CHK MUST; WUP (the
    planted control) <arm>-CHK REPORT, <arm>-PLANT-CHK-PRE / -CAUGHT MUST; WT / WU every gate REPORT (the lead,
    2026-10-28T13:19Z). Returns ([(name, must, passed, detail)], record)."""
    w = WSG_ARMS[arm]
    plant = bool(w.get("plant"))
    g, rec = CK.chk_gates(arm, cap_result, ref, tie, plant=plant)
    if not w["exact"] and not plant:
        g = [(n, False, ok, d) for n, m, ok, d in g]
    return g, rec


def run_arm_w(model, ids, inputs, steps: int, clock, opt, sample, slot_every: int):
    """ONE WS-G arm: its own prefill, the contract-4 runner (WJ: PaperJournalRunner, gemm rowgroup; WT: TentRunner,
    rowgroup; WU / WUP: TentRunnerU, --tu-gemm, union checks on; WUP with VerifyOps neg drop_spill and X =
    --chk-plant-extra) with r2_i2_check.ContractCapture installed during runner.run. Returns (res, the capture after
    finish())."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative as ST
    from nosi import spec_tentative_union as STU
    import r2_i2_check as ICK
    w = WSG_ARMS[opt["arm"]]
    cache, prefill_s, first = WB._prefill(model, ids, inputs, steps, clock)
    K = int(opt["K"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()
    base = w["base"]
    union = base == "L4PTU"
    X = opt["plant_X"] if w.get("plant") else opt["X"]
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=(opt["tu_gemm"] if union else "rowgroup"),
                      extra=(X if union else opt["extra"]), journal=opt["journal"], redirect=opt["redirect"],
                      probe_every=opt["probe_every"])
    ce = 1      # every round a check round: a tentative runner fingerprints its committed rows only then (MAP reads them)
    if base == "L4PJ":
        runner = ST.PaperJournalRunner(model, rs, sops, clock, t0=inputs[:, 0])
    elif base == "L4PT":
        runner = ST.TentRunner(model, rs, sops, clock, t0=inputs[:, 0], check_every=ce)
    else:
        runner = STU.TentRunnerU(model, rs, sops, clock, t0=inputs[:, 0], check_every=ce, union=True, union_check=True,
                                 union_stats=True, vneg=w.get("plant"))
    cap = ICK.ContractCapture(runner, steps, BC.S0, sample, exact=w["exact"], RE=RE, tentative=w["tentative"],
                              slot_every=slot_every)
    cap.install()
    try:
        runner.run(steps)
    finally:
        cap.remove()
    rd = runner.read()
    rd.pop("trace_all", None)
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
               probe_every=opt["probe_every"], prefill_s=prefill_s, first_token_agreement=first,
               meta=dict(rs.meta, spec=runner.meta), X=X)
    cap.finish(steps)
    runner.detach()
    del runner, rs, cache
    return res, cap


def run_ref_w(model, ids, inputs, steps: int, clock, cap, inputs_ref, fp_sh):
    """REF (r2_wsg_cell.run_ref_arm's steps): its own shipped prefill, ragged_engine.attach with REF_OPS, then
    r2_wsg_check.run_ref teacher-forced on inputs_ref."""
    import torch
    from nosi import ragged_engine as RE
    cache, prefill_s, _ = WB._prefill(model, ids, inputs, steps, clock)
    ops = RE.RaggedOps(RE.default_kernels(), gemm=REF_OPS["gemm"], twin=REF_OPS["twin"], body=REF_OPS["body"],
                       graph=REF_OPS["graph"])
    rs = RE.attach(model, cache, BC.S0, steps, ops)
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()
    ref = CK.run_ref(model, rs, RE, inputs_ref, BC.S0, steps, cap, fp_sh, inputs)
    ref.update(prefill_s=prefill_s, wall_s=round(time.time() - t0, 1), ops=dict(REF_OPS))
    del rs, cache
    return ref


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
    gates, sh, sp, chk, inputs, shr = [], None, {}, {}, None, None
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
    dims = dict(L=L, H=H, D=D, topk=64, block_bytes=64 * D * 2 * 2 + 64 * 2)
    R = a.spec + 1
    check_t = a.check_every == 1
    sample = a.sample if a.sample is not None else CK.default_rows(a.batch)    # WS-G arms: the sample rows
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
                      memory=dict(start_GB=round(sh_start, 3), peak_GB=pa,
                                  fit_batch_MODEL=WA.fit_batch(a.batch, sh_start, pa), budget_GB=WA.BUDGET_GB))
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
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
            if arm in ("TUC", "TUCR") and shr is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: no SHR record"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, rounds=0, journal=a.spec_journal_blocks, extra=a.spec_extra, X=a.joint_extra,
                       spill=a.joint_spill, tu_gemm=a.tu_gemm, twin=a.spec_twin, body=a.spec_body,
                       graph=a.spec_graph == "on", redirect=a.spec_redirect == "on", probe_every=a.probe_every,
                       check_every=a.check_every, passes=a.check_passes, profile=a.profile == "on",
                       plant_X=a.chk_plant_extra)
            log("arm %s (%s)" % (arm, opt))
            if arm in WSG_ARMS:                                        # WS-G's check: the arm, then REF
                plant = bool(WSG_ARMS[arm].get("plant"))
                stage = "arm"
                try:
                    res, cap = run_arm_w(model, ids, inputs, a.steps, clock, opt, sample, a.chk_slot_every)
                    tc = SC.token_check(res["out"], res["read"]["count"], inputs, a.steps)
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d (CHECK RUN: the capture ran inside "
                                                             "the rounds; no rate here is a timing)" % (
                                                                 res["read"]["rounds"], a.spec)))
                    gates.extend(x for x in SC.arm_gates(arm, res, None) if x[0].endswith("-ERR"))
                    gates.append(WA.commit_n_gate(arm, res, a.steps))
                    out_t = res["out"]
                    union_rd = res["read"].get("union")
                    del res
                    BC.free()
                    stage = "ref"
                    log("REF of %s (teacher-forced on its committed tokens, %s)" % (arm, REF_OPS))
                    inputs_ref = cap.ref_inputs(out_t, inputs[:, 0], inputs)
                    ref = run_ref_w(model, ids, inputs, a.steps, clock, cap, inputs_ref, fp_sh)
                    stage = "compare"
                    g, crec = wsg_gates(arm, cap.result, ref)
                    crec.update(tokens=tc, union=union_rd, ref=dict(prefill_s=ref.get("prefill_s"),
                                                                    wall_s=ref.get("wall_s"), ops=ref.get("ops"),
                                                                    self=ref.get("self"), errors=ref.get("errors")))
                    chk[arm] = crec
                    gates.extend(g)
                    del cap, ref, inputs_ref, out_t
                except Exception as e:
                    traceback.print_exc()
                    why = "EXCEPTION in %s: %s: %s" % (stage, type(e).__name__, str(e)[:300])
                    if stage == "arm":
                        gates.append(("%s-RUN" % arm, True, False, why))
                    gates.append(("%s-CHK" % arm, WSG_ARMS[arm]["exact"], False, why))
                    if plant:
                        gates.append(("%s-PLANT-CHK-PRE" % arm, True, False, why))
                        gates.append(("%s-PLANT-CHK-CAUGHT" % arm, True, False, why))
                mem["after_%s" % arm] = BC.host_mem()
                BC.free()
                continue
            try:
                if arm in ("L4PT", "L4PJ"):                            # r2_wsa_cell's arm, unchanged
                    res = WA.run_arm(model, ids, inputs, a.steps, clock, opt)
                    pa, pr = BC.gpu_peak()
                    rec = WA.summarize_w(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                    rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                    jm = rec["phases_median"].get("joint")
                    shm = sh["rates"]["steady_step_ms"] if sh else None
                    rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                    sp[arm] = rec
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                    gates.extend(WA.arm_gates_w(arm, res, rec, a.steps, require_stress=bool(a.require_stress)))
                elif arm in UNION_TIMING:
                    res = run_arm_u(model, ids, inputs, a.steps, clock, opt)
                    pa, pr = BC.gpu_peak()
                    rec = summarize_u(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                    rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                    sp[arm] = rec
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                    gates.extend(arm_gates_u(arm, res, rec, a.steps, L, check_t, bool(a.require_stress)))
                elif arm == "SHR":
                    n = 1 + a.check_passes * R
                    shr = WB.run_shr(model, ids, inputs, a.steps, n, clock, opt, fp_sh)
                    pa, pr = BC.gpu_peak()
                    fpc = shr["fp"]
                    chk[arm] = dict(n=n, fingerprints=fpc, tok_bad=shr["tok_bad"], errors=shr["errors"],
                                    profile=shr["profile"], prefill_s=shr["prefill_s"], gpu_peak_alloc_GB=pa,
                                    first_token_agreement=shr["first_token_agreement"])
                    gates.append(("SHR-RUN", True, True, "%d teacher-forced steps" % n))
                    gates.append(("SHR-FP", True, fpc["n_unequal"] == 0 if "n_unequal" in fpc else False,
                                  "SHR logits fingerprints against SH's: %s" % (fpc,)))
                    gates.append(("SHR-TOK", True, shr["tok_bad"] == 0,
                                  "SHR argmax against SH's tokens: %d of %d unequal" % (shr["tok_bad"], n * a.batch)))
                else:
                    res = run_check_tu(model, ids, inputs, a.steps, clock, opt, shr)
                    pa, pr = BC.gpu_peak()
                    res.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               ready_record=WB.ready_record(res["ready"], res["passes"], dims["block_bytes"]))
                    chk[arm] = res
                    gates.append(("%s-RUN" % arm, True, True, "%d teacher-forced passes of R %d" % (res["passes"],
                                                                                              res["R"])))
                    gates.extend(check_gates_u(arm, res, bool(a.require_stress), L))
                res = None                                   # frees the arm's record; the SHR branch sets no res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, joint_residency=a.joint_residency,
                joint_extra=a.joint_extra, joint_spill=a.joint_spill, tu_gemm=a.tu_gemm, check_passes=a.check_passes,
                spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin, spec_body=a.spec_body,
                spec_graph=a.spec_graph, spec_redirect=a.spec_redirect, probe_every=a.probe_every,
                check_every=a.check_every, profile=a.profile, require_stress=a.require_stress,
                predictions=dict(PRED_I2, by_batch=PRED_I2_BY_B), phase_groups=PHASE_GROUPS_U, strongest_tok_s=STRONGEST,
                wsg=dict(sample_rows=sample, chk_slot_every=a.chk_slot_every, chk_plant_extra=a.chk_plant_extra,
                         tie_logit=CHK_TIE_LOGIT, ref_ops=REF_OPS, check_only_ops=CK.CHECK_ONLY_OPS),
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"))
    summary = dict(meta=meta, SH=sh, SPEC=sp, CHECK=chk,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-16s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB, memory %s" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"], sh["memory"]))
    for name in a.arms:
        r = sp.get(name)
        if r is not None:
            print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; vs %.1f %s; "
                  "joint pass %s SH steps; phases %s; ROLLBACK per round %s; calls %s; bytes per round %s; tokens equal "
                  "%.6f (first %s); fingerprints %s; memory %s" % (
                      name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                      r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"], r["joint_sh_steps"],
                      r["phases_median"], r["rollback_per_round"], r["calls"], r["bytes_per_round"],
                      r["tokens"]["equal_fraction"], r["tokens"]["first"], r["fingerprints"], r["memory"]))
            if "union" in r:
                print("REPORT %s union: %s; joint-pass phases (probe) %s" % (name, r["union"], r["joint_phases_union"]))
            sr = r.get("schedule_record")
            if sr:
                print("REPORT %s schedule: tokens %s; request 0 trace head %s; tail %s" % (
                    name, sr["tokens_acct"], (sr["trace_head"] or [[]])[0], (sr["trace_tail"] or [[]])[0]))
        c = chk.get(name)
        if c is not None and name in WSG_ARMS:
            print("REPORT %s (WS-G check of %s): %s; parts %s; failed %s; logit %s; tokens equal to SH's %.6f; union %s" % (
                name, WSG_ARMS[name]["base"], "PASS" if c["ok"] else "FAIL", c["parts"], c["failed"],
                c["logit"]["detail"], c["tokens"]["equal_fraction"],
                {k: (c["union"] or {}).get(k) for k in ("dropped", "unserved", "vis_bad", "spill", "extras")}))
        elif c is not None and name != "SHR":
            print("REPORT %s: selections %s; logits %s; stats %s; ready / residual %s; calls per pass %s; tent calls %s; "
                  "profile %s" % (name, c["sel"], c["logits"], c["stats"], c["ready_record"], c["calls_per_pass"],
                                  c["tent_calls"], c["profile"]))
        elif c is not None:
            print("REPORT SHR: fingerprints %s; argmax unequal %d; profile %s" % (c["fingerprints"], c["tok_bad"],
                                                                                  c["profile"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i2: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
