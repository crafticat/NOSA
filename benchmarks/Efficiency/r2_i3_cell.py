"""Rung I3 of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of the arm
L4PTF = the tentative buffers of WS-A (contract 4, I1) with the union residency of WS-B (I2) and ONE fork call per layer
of WS-C (fia6's fork): nosi/nosi/spec_tentative_fork.py, opt-in --joint-attn fork. A NEW script: r2_i2_cell.py,
r2_wsa_cell.py, r2_wsb_cell.py, r2_wsc_cell.py and r2_base_cell.py are not edited (their functions are imported: the
L4PT / L4PJ arms run r2_wsa_cell.run_arm, the L4PTU arm r2_i2_cell.run_arm_u, the WS-G arms WJ / WT / WU
r2_i2_cell.run_arm_w, unchanged). Step script: benchmarks/Efficiency/r2_i3_step.sh (cells i3m, i3a, i3b, i3ca, i3cb,
i3g). Docs: docs/NOTES_i3.md.

ONE PROCESS, ONE MODEL, the arms in the order given, each with its own shipped prefill; SH runs FIRST. The inputs of
sp3a / sp3b, wa1a / wa1b and i2a / i2b (the same workload pin, S0, document rows, K 3, 512 steps, fused twin and body,
graph, redirect, probe 16), so the rung's INCREMENTAL change is measured in ONE job:
  SH      the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
TIMING ARMS (no check-only work in the timing cells; accounting off):
  L4PT    I1 (r2_wsa_cell.run_arm, unchanged; gemm rowgroup)
  L4PTU   I2 at X = --joint-extra (default 193: C = 1, no spill fill; r2_i2_cell.run_arm_u, unchanged; gemm natural)
  L4PTF   I3: spec_tentative_fork.TentRunnerF (gemm --tu-gemm, default natural); the K / V arena (r2_i3_arena.ChainArena)
          made before the prefill, the engines re-homed into it during the prefill (--prefill-hook nocopy: no copy, the
          engine's rows are allocated in the arena; copy: r2_kv_arena.PrefillHook); its store X = 1 + (K - 1) x 64
          (contract 4: at most K verify rows per request and pass) unless --fork-store
  L4PJ    (optional) the EXACT journal arm of the same schedule (r2_wsa_cell.run_arm)
CHECK ARMS (not timed; FAST checked on itself against SH):
  SHR     r2_wsb_cell.run_shr: the serial ragged step teacher-forced on SH's tokens for 1 + phase + G R steps
  TFC     TentRunnerF.run_teacher (gemm natural), teacher-forced on SH's tokens: 1 + --check-phase plain steps, then G
          joint passes of R = K + 1 verify rows (store X = 1 + K x 64), CHECK ON: per pass, layer, row, head and
          request the selection SET against SHR's (MUST), PRESENT, VIS and VB from the tables the KERNEL wrote, TAIL
          (the window's tail slot never read), CHAIN (the chain images' bytes and positions), RESID, ATTN1, FILL1, the
          "F" / "RA" probes; the logits deltas against SHR's top-16; one PROFILED pass
  TFCR    TFC with gemm rowgroup
WS-G CHECK ARMS (cell i3g; r2_wsg_check.py of b1ae579 and r2_i2_check.ContractCapture, both unchanged): the arm with the
capture, then REF = SH's computation teacher-forced on the arm's own committed tokens:
  WJ      L4PJ, the EXACT positive control: WJ-CHK MUST
  WT, WU, WF   L4PT, L4PTU, L4PTF: every gate REPORT
  WFP     L4PTF with the fork's control fork_small_store at X = --chk-plant-extra: the planted control; MUST
          WFP-PLANT-CHK-PRE and -CAUGHT
MUST GATES: C0_FORK (the fork imports); SH-RUN; per timing arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN
(r2_wsa_cell.arm_gates_w); L4PTU -RESID; L4PTF -RESID, -ATTN1, -FILL1; with --check-every 1 (i3m) L4PTF -PRESENT, -VIS,
-VB, -TAIL, -CHAIN, -IDX, -RA; with --require-stress (i3m) <arm>-STRESS (WS-A's) and L4PTF-FSTRESS; TFC / TFCR -RUN,
-ERR, -SEL, -PRESENT, -VIS, -VB, -TAIL, -CHAIN, -RESID, -ATTN1, -FILL1, -IDX, -RA (and -STRESS with --require-stress);
the WS-G gates above. REPORT: MEM (each arm's peak against 84 GB), the joint-pass phases of the probe rounds.
REGISTERED PREDICTIONS PRED_I3_BY_B: every value None until the lead registers them with PRED_STAMP_I3; the step script
STOPS the timing cells (i3a, i3b) while PRED_STAMP_I3 is None. Inferred bands with their arithmetic: docs/NOTES_i3.md.
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
import r2_i2_cell as I2C          # noqa: E402
import r2_kv_arena as KA          # noqa: E402
import r2_i3_arena as IA          # noqa: E402
import r2_wsg_check as CK         # noqa: E402  (WS-G b1ae579, copied unchanged)

SC = WA.SC
BC = WA.BC
ARMS = ("L4PT", "L4PTU", "L4PTF", "L4PJ", "SHR", "TFC", "TFCR", "WJ", "WT", "WU", "WF", "WFP")
TIMING_ARMS = ("L4PT", "L4PTU", "L4PTF", "L4PJ")
CHECK_ARMS = ("SHR", "TFC", "TFCR")
FORK_ARMS = ("L4PTF", "TFC", "TFCR", "WF", "WFP")
CHECK_GEMM = dict(TFC="natural", TFCR="rowgroup")
WSG_I2 = ("WJ", "WT", "WU")                        # r2_i2_cell.run_arm_w / wsg_gates, unchanged
WSG_F = dict(WF=dict(base="L4PTF", exact=False, tentative=True),                     # I3: REPORT
             WFP=dict(base="L4PTF", exact=False, tentative=True, plant="fork_small_store"))   # the PLANTED control
TOPK = 64
FORK_FILE = "fork_vllm_unified_attention.py"
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"   # nosi-r2-fia6 94d5bb0:nosi/nosi/fork_vllm_unified_attention.py
STRONGEST = WA.STRONGEST
BUDGET_GB = WA.BUDGET_GB
STRESS_STORE = 3                   # FSTRESS: a (head, request) listed more store blocks than WS-B's miniature fill
# the joint pass's phase marks (spec_tentative_fork, probe rounds): each label runs to the next mark of the pass
PHASE_GROUPS_F = dict(residency=("resid",), write=("write",), chain=("chain",), union=("union",), fetch=("fetch",),
                      attention=("attn",), check=("check",), dense_next=("post",))
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I3_BY_B (per batch) and PRED_STAMP_I3
# (`date -u`, L33) in a commit BEFORE i3a / i3b (the step script stops while PRED_STAMP_I3 is None). Keys, each
# (low, high) unless noted:
#   I3_ratio          L4PTF median steady committed tok/s / SH's steady tok/s
#   I3_vs_I1          L4PTF / L4PT median steady committed tok/s (same job)
#   I3_vs_I2X         L4PTF / L4PTU (X 193) median steady committed tok/s (same job): the rung's incremental change
#   I3_joint_sh       L4PTF median joint-phase ms / SH's steady step ms
#   I3_joint_saving   L4PTU (X 193) joint ms - L4PTF joint ms (medians, same job)
#   I3_rollback_sh    L4PTF (reset + commit) ms / SH step (D1 kept: about I1's)
#   I1_ratio, I2X_ratio   L4PT / SH and L4PTU / SH in this job (the previous rungs, re-measured)
#   I3_agree          a floor: the fraction of L4PTF's committed tokens equal to SH's (REPORT until WS-G's gate)
#   I3_above_I2X      True: the rung falsifier "L4PTF below I2X in the same job" (I3-ABOVE-I2X)
#   I3_peak_gb        the L4PTF arm's peak allocated GB (gpu_peak_alloc_GB)
# The builder's INFERENCE (docs/NOTES_i3.md section 7, with the arithmetic) is an input for the lead, not a registration.
# The lead's REGISTRATION (2026-10-28T23:36:18Z, `date -u`, before any I3 cell ran). MEASURED inputs: I2X (i2ax / i2bx,
# SCORING_i2x.md): L4PTU-X / SH 0.4310 / 0.5165, joint 273.983 / 466.248 ms, I2X / I1 1.0929 / 1.0680, arm peak 83.461 /
# 78.260 GB; I1 / SH 0.394 / 0.484; WS-C wc1b (C 73, legacy L4 schedule): fork joint saving 131.7 ms vs X 48 union, of which
# about 52 ms is the spill (I2X), so about 80 ms is the fork. MODEL (NOTES_i3.md section 7, the builder's): joint 388 ms
# (376-421) at C 73, 203 ms (185-240) at C 128 (weaker basis). Bands widened slightly from the builder's toward the
# measured I2X and to cover the CB0 / CB1 copies (2-6 ms, MODEL). Memory: L4PTF arm about 83.9 GB at C 128 (0.1 GB margin,
# NOTES risk 1): an OOM there is a RUN failure, not a band miss.
PRED_STAMP_I3 = "2026-10-28T23:36:18Z"
PRED_KEYS = ("I3_ratio", "I3_vs_I1", "I3_vs_I2X", "I3_joint_sh", "I3_joint_saving", "I3_rollback_sh", "I1_ratio",
             "I2X_ratio", "I3_agree", "I3_above_I2X", "I3_peak_gb")
PRED_I3_BY_B = {
    216: dict(I3_ratio=(0.44, 0.54), I3_vs_I1=(1.10, 1.35), I3_vs_I2X=(1.03, 1.22), I3_joint_sh=(2.8, 3.8),
              I3_joint_saving=(20.0, 90.0), I3_rollback_sh=(0.03, 0.30), I1_ratio=(0.37, 0.42), I2X_ratio=(0.41, 0.45),
              I3_agree=0.25, I3_above_I2X=True, I3_peak_gb=(83.0, 84.5)),
    318: dict(I3_ratio=(0.54, 0.61), I3_vs_I1=(1.10, 1.25), I3_vs_I2X=(1.04, 1.16), I3_joint_sh=(2.9, 3.5),
              I3_joint_saving=(40.0, 95.0), I3_rollback_sh=(0.03, 0.30), I1_ratio=(0.45, 0.51), I2X_ratio=(0.49, 0.55),
              I3_agree=0.25, I3_above_I2X=True, I3_peak_gb=(76.5, 79.5)),
}


def log(msg):
    print("[r2_i3 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4, Figure 3: K 3; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PT,L4PTU,L4PTF", help="a subset of %s, in run order (SH runs first)"
                    % ",".join(ARMS))
    ap.add_argument("--joint-attn", default="store", choices=("store", "fork"),
                    help="THE FLAG of I3: the arms %s run only with 'fork'" % (FORK_ARMS,))
    ap.add_argument("--joint-extra", type=int, default=193, help="L4PTU / WU: X store slots per request (I2X: 193)")
    ap.add_argument("--fork-store", type=int, default=0,
                    help="the fork arms' X (0: the bound per arm, 1 + (verify rows - 1) x 64; a smaller X is refused)")
    ap.add_argument("--prefill-hook", default="nocopy", choices=("nocopy", "copy"),
                    help="the fork arms' re-home: nocopy (r2_i3_arena.NoCopyPrefillHook) or copy (r2_kv_arena)")
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"), help="L4PTU / L4PTF joint GEMM")
    ap.add_argument("--check-passes", type=int, default=32, help="TFC / TFCR: teacher-forced joint passes G")
    ap.add_argument("--check-phase", type=int, default=0,
                    help="TFC / TFCR: 1 + phase plain steps before the passes (WS-C2's idea: moves the passes against "
                         "the 64-token block edges)")
    ap.add_argument("--spec-journal-blocks", type=int, default=256, help="L4PJ only")
    ap.add_argument("--spec-extra", type=int, default=16, help="L4PT / L4PJ (r2_wsa_cell's --spec-extra; unused there)")
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--check-every", type=int, default=0,
                    help="timing arms: check rounds (0 = the probe rounds; 1 = every round and the union / fork checks "
                         "on: the miniature)")
    ap.add_argument("--profile", default="on", choices=("on", "off"), help="check arms: one profiled pass / step")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    ap.add_argument("--chk-rows", default="", help="WS-G arms: the sample rows, comma separated; empty: 8 rows")
    ap.add_argument("--chk-slot-every", type=int, default=32, help="WS-G arms: the slot check every N rounds")
    ap.add_argument("--chk-plant-extra", type=int, default=4, help="WFP: X of the planted control (fork_small_store)")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    fk = [x for x in a.arms if x in FORK_ARMS]
    if fk and a.joint_attn != "fork":
        ap.error("the arms %s need --joint-attn fork" % fk)
    if a.joint_attn == "fork" and not fk:
        ap.error("--joint-attn fork without a fork arm (%s)" % (FORK_ARMS,))
    ck = [x for x in a.arms if x in ("TFC", "TFCR")]
    if ck and ("SHR" not in a.arms or a.arms.index("SHR") > min(a.arms.index(x) for x in ck)):
        ap.error("TFC / TFCR compare against SHR: SHR must run before them")
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile of 4 rows per request)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    R = a.spec + 1
    if not 0 <= a.check_phase < 64:
        ap.error("--check-phase in 0 .. 63")
    if ck or "SHR" in a.arms:
        if a.check_passes < 1 or 1 + a.check_phase + a.check_passes * R > a.steps:
            ap.error("--check-passes G >= 1 with 1 + phase + G (K + 1) <= --steps (SHR's steps)")
    if a.joint_extra < 2 or a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1 or a.spec_extra < 2:
        ap.error("--joint-extra >= 2, --probe-every >= 2, --check-every >= 0, journal >= 1, --spec-extra >= 2")
    if a.fork_store != 0 and a.fork_store < KA.store_slots(TOPK, R):
        ap.error("--fork-store %d: below the bound of every fork arm, 1 + (R - 1) x 64 = %d (0 = the bound per arm)"
                 % (a.fork_store, KA.store_slots(TOPK, R)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    if a.chk_slot_every < 1 or a.chk_plant_extra < 2:
        ap.error("--chk-slot-every >= 1, --chk-plant-extra >= 2")
    if "WFP" in a.arms and "WJ" not in a.arms:
        ap.error("the planted control WFP needs the positive control WJ in the same cell")
    try:
        a.sample = [int(x) for x in a.chk_rows.split(",") if x] if a.chk_rows else None
    except ValueError:
        ap.error("--chk-rows: integers")
    if a.sample is not None and (not a.sample or any(not 0 <= b < a.batch for b in a.sample)):
        ap.error("--chk-rows: rows in 0 .. batch - 1")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def fork_store_slots(arm: str, K: int, override: int = 0, plant_x: int = 4) -> int:
    """X of a fork arm: L4PTF / WF at most K verify rows per request and pass (contract 4); TFC / TFCR K + 1 (every
    row a verify row); WFP the plant's X. 0 = the bound; an override must not be below it."""
    if arm == "WFP":
        return int(plant_x)
    full = K + 1 if arm in ("TFC", "TFCR") else K
    x = KA.store_slots(TOPK, full)
    return max(x, int(override)) if override else x


def store_gb(B: int, X: int, H: int = 2, D: int = 128, bytes_per: int = 2) -> float:
    """The arena's store: B x X x 64 tokens x H x (K D + V D + bias 1) x 2 B (bf16), in GB."""
    return round(B * X * 64 * H * (2 * D + 1) * bytes_per / 1e9, 4)


def cb0_gb(B: int, H: int = 2, D: int = 128, bytes_per: int = 2) -> float:
    """CB0 (one block per request): B x 64 x H x (2 D + 1) x 2 B, in GB."""
    return store_gb(B, 1, H, D, bytes_per)


def layer_rows_gb(B: int, pool: int, H: int = 2, D: int = 128, bytes_per: int = 2) -> float:
    """One layer's engine GPU rows (window + pool, NS = 64 + P blocks): the transient of the COPYING re-home."""
    return store_gb(B, TOPK + int(pool), H, D, bytes_per)


def check_facts(S0: int, warm: int, G: int, R: int, bs: int = 64) -> dict:
    """spec_tentative_fork.teacher_facts (the same arithmetic, no torch): per request, the rows of the check passes past
    a 64-token block edge and the F4 rows (in-block rows at p % 64 = 63)."""
    past = f4 = 0
    for g in range(int(G)):
        p0 = int(S0) + int(warm) + g * int(R)
        for r in range(int(R)):
            p = p0 + r
            if p // bs != p0 // bs:
                past += 1
            elif p % bs == bs - 1:
                f4 += 1
    return dict(passes=int(G), past_edge_rows=past, f4_rows=f4, first=int(S0) + int(warm),
                last=int(S0) + int(warm) + int(G) * int(R) - 1)


def fork_groups(ph: dict) -> dict:
    """{label: ms} of one probe pass -> {group: ms} (PHASE_GROUPS_F) plus "total"."""
    out = {g: round(sum(ph.get(x, 0.0) for x in labs), 4) for g, labs in PHASE_GROUPS_F.items()}
    out["total"] = round(sum(ph.values()), 4)
    return out


def fork_phase_summary(phases, layers):
    """The probe passes' fork phases: medians per label and group, the median complete layer time."""
    if not phases:
        return dict(n_probes=0)
    labels = sorted({k for p in phases for k in p})
    med = {k: round(statistics.median([p.get(k, 0.0) for p in phases]), 4) for k in labels}
    grp = [fork_groups(p) for p in phases]
    gmed = {g: round(statistics.median([x[g] for x in grp]), 4) for g in grp[0]}
    lay = [x for ls in layers for x in ls]
    return dict(n_probes=len(phases), labels_median=med, groups_median=gmed,
                layer_ms_median=(round(statistics.median(lay), 4) if lay else None),
                layers_per_pass=(len(layers[0]) if layers else None))


def fork_record(rd) -> dict:
    """L4PTF's fork facts from read(): the store, the bound, the host counts per joint pass, the fork counters."""
    u = rd.get("union") or {}
    keys = ("pool_reads", "store_reads", "extras_host", "extras_pool", "extras_max", "unserved", "dropped", "vis_bad",
            "vb_bad", "vb_rows", "tail_reads", "chain_bad", "chain_rows", "chain_reads", "past_edge", "f4_rows",
            "fork_loaded", "block0_bad")
    return dict(store_slots=u.get("store_slots"), bound=u.get("bound"), chunks=u.get("chunks"),
                counters={k: u.get(k) for k in keys})


def fork_gates(arm, res, L: int, check: bool, require_stress: bool = False):
    """The I3 MUST gates of a fork timing arm: [(name, must, passed, detail)]: r2_i2_cell.union_gates' RESID (one
    residency per layer and joint pass), ATTN1 (ONE fork call per layer and joint pass), FILL1 (ONE store fill); with
    check: PRESENT, VIS, VB, TAIL, CHAIN, IDX ("F"), RA; with require_stress: FSTRESS."""
    rd = res["read"]
    u = rd.get("union") or {}
    calls, passes = (u.get("calls") or {}), u.get("passes", 0)
    g = list(I2C.union_gates(arm, res, L, check=False))
    for key, name in (("attn", "ATTN1"), ("fills", "FILL1")):
        g.append(("%s-%s" % (arm, name), True, passes > 0 and calls.get(key) == L * passes,
                  "%s calls %s for %d joint passes of %d layers (one per layer and pass)" % (key, calls.get(key),
                                                                                           passes, L)))
    if check:
        g.extend(_check_part(arm, u, res.get("fa_check")))
    if require_stress:
        ok = ((u.get("past_edge") or 0) > 0 and (u.get("f4_rows") or 0) > 0 and (u.get("pool_reads") or 0) > 0
              and (u.get("chain_reads") or 0) > 0 and (u.get("store_reads") or 0) > 0)
        g.append(("%s-FSTRESS" % arm, True, ok,
                  "rows past a block edge %s; F4 rows %s; pool blocks read in place %s; chain reads %s; store reads %s "
                  "(each > 0; accounting on)" % (u.get("past_edge"), u.get("f4_rows"), u.get("pool_reads"),
                                                 u.get("chain_reads"), u.get("store_reads"))))
    return g


def _check_part(arm, st, fa):
    """The check-only MUST gates shared by the fork timing arm (check on) and the check arms."""
    fa = fa or {}
    f, ra = fa.get("F") or {}, fa.get("RA") or {}
    return [
        ("%s-PRESENT" % arm, True, st.get("dropped") == 0 and st.get("unserved") == 0 and
         (st.get("union_samples") or 0) > 0,
         "selected blocks of verify rows not present: %s (recount); unserved list entries %s (store X %s, bound %s)" % (
             st.get("dropped"), st.get("unserved"), st.get("store_slots"), st.get("bound"))),
        ("%s-VIS" % arm, True, st.get("vis_bad") == 0 and (st.get("visible_blocks") or 0) > 0,
         "rows whose visible block count (from the kernel's tables) is not their own: %s (per row %s)" % (
             st.get("vis_bad"), st.get("row_vis_bad"))),
        ("%s-VB" % arm, True, st.get("vb_bad") == 0 and (st.get("vb_rows") or 0) > 0,
         "rows whose (block id, tokens) set in the kernel's tables is not their own selection: %s of %s" % (
             st.get("vb_bad"), st.get("vb_rows"))),
        ("%s-TAIL" % arm, True, st.get("tail_reads") == 0,
         "union entries that read the window's committed tail slot (D1: the chain image CB0 replaces it): %s" % (
             st.get("tail_reads"),)),
        ("%s-CHAIN" % arm, True, st.get("chain_bad") == 0 and (st.get("chain_rows") or 0) > 0,
         "chain-image rows with wrong bytes or a wrong position: %s; buffer rows checked %s; chain reads %s" % (
             st.get("chain_bad"), st.get("chain_rows"), st.get("chain_reads"))),
        ("%s-IDX" % arm, True, bool(f.get("entries")) and all(f.get("equal") or [False]),
         "sampled requests alone against the full launch, bitwise: %s" % (f,)),
        ("%s-RA" % arm, True, bool(ra.get("entries")) and all(ra.get("equal") or [False]),
         "a row alone against the full launch, bitwise (rows whose first block is not block 0: %s): %s" % (
             st.get("block0_bad"), ra)),
    ]


def check_gates_tf(arm, res, require_stress: bool = False, L: int = 32):
    """The MUST gates of a fork check arm (TFC / TFCR) from run_check_tf's result: [(name, must, passed, detail)]."""
    st = res["stats"]
    g = []
    body_err, overflow = res["errors"]
    g.append(("%s-ERR" % arm, True, body_err == 0 and not any(overflow),
              "body refusal bits %s; twin overflow flags %s" % (body_err, sorted(set(overflow)))))
    sc = res["sel"]
    g.append(("%s-SEL" % arm, True, sc["compared"] > 0 and sc["changed"] == 0,
              "row selections against SHR's at the same position: %d changed of %d (head, request, row, layer) sets; "
              "by layer %s; symmetric-difference sizes %s; first %s" % (
                  sc["changed"], sc["compared"], sc["by_layer_nonzero"], sc["symdiff_hist"], sc["first"])))
    g.extend(_check_part(arm, st, res.get("fa_check")))
    passes = st["passes"]
    for key, name in (("resid", "RESID"), ("attn", "ATTN1"), ("fills", "FILL1")):
        g.append(("%s-%s" % (arm, name), True, passes > 0 and st["calls"][key] == L * passes,
                  "%s calls %d for %d passes of %d layers" % (key, st["calls"][key], passes, L)))
    if require_stress:
        ok = (st["past_edge"] > 0 and st["f4_rows"] > 0 and st["extras_max"] > STRESS_STORE and st["store_reads"] > 0
              and st["pool_reads"] > 0 and st["chain_reads"] > 0)
        g.append(("%s-STRESS" % arm, True, ok,
                  "rows past a block edge %d; F4 rows %d; the most listed store blocks of one (head, request) %d (> %d); "
                  "store reads %d; pool blocks read in place %d; chain reads %d" % (
                      st["past_edge"], st["f4_rows"], st["extras_max"], STRESS_STORE, st["store_reads"],
                      st["pool_reads"], st["chain_reads"])))
    return g


def wsg_gates_f(arm, cap_result, ref, tie=I2C.CHK_TIE_LOGIT):
    """r2_wsg_check.chk_gates for a fork WS-G arm: WF every gate REPORT; WFP (the planted control) <arm>-CHK REPORT,
    <arm>-PLANT-CHK-PRE / -CAUGHT MUST. Returns ([(name, must, passed, detail)], record)."""
    w = WSG_F[arm]
    plant = bool(w.get("plant"))
    g, rec = CK.chk_gates(arm, cap_result, ref, tie, plant=plant)
    if not plant:
        g = [(n, False, ok, d) for n, m, ok, d in g]
    return g, rec


def mem_record(arm, mem, pa, budget=BUDGET_GB):
    """REPORT: an arm's peak allocated GB against the 84 GB budget: the larger of BC.gpu_peak after the arm (1e9 bytes;
    the arms reset the peak after their prefill, so this is the decode peak) and the arm's prefill peak (its memory
    dict, in GiB as r2_i2_cell.run_arm_u: x 1.073741824)."""
    pre = None
    if isinstance(mem, dict) and mem.get("prefill_peak_GB") is not None:
        pre = round(float(mem["prefill_peak_GB"]) * 1024 ** 3 / 1e9, 3)
    vals = [x for x in (pa, pre) if x is not None]
    peak = max(vals) if vals else None
    return ("%s-MEM" % arm, False, peak is not None and peak <= budget,
            "peak allocated %s GB (after the prefill %s GB, prefill %s GB) against %.1f GB; %s" % (
                peak, pa, pre, budget, mem))


def verdicts(sh, sp, B: int):
    """PRED_I3_BY_B[B] against the timing arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN without
    L4PTF."""
    P = PRED_I3_BY_B.get(B) or {k: None for k in PRED_KEYS}
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
    f, t, u = (sp or {}).get("L4PTF"), (sp or {}).get("L4PT"), (sp or {}).get("L4PTU")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    sh_ms = sh["rates"]["steady_step_ms"] if sh else None
    if t is not None:
        band("I1-RATIO", (med(t) / shm) if (med(t) and shm) else None, P.get("I1_ratio"))
    if u is not None:
        band("I2X-RATIO", (med(u) / shm) if (med(u) and shm) else None, P.get("I2X_ratio"))
    if f is None:
        out.append(("L4PTF-RUN", "NOT_RUN", "the arm did not run"))
        return out
    band("I3-RATIO", (med(f) / shm) if (med(f) and shm) else None, P.get("I3_ratio"))
    band("I3-VS-I1", (med(f) / med(t)) if (med(f) and med(t)) else None, P.get("I3_vs_I1"))
    band("I3-VS-I2X", (med(f) / med(u)) if (med(f) and med(u)) else None, P.get("I3_vs_I2X"))
    band("I3-JOINT-SH", (joint(f) / sh_ms) if (joint(f) and sh_ms) else None, P.get("I3_joint_sh"))
    band("I3-JOINT-SAVING", (joint(u) - joint(f)) if (joint(u) is not None and joint(f) is not None) else None,
         P.get("I3_joint_saving"))
    band("I3-ROLLBACK-SH", f["rollback_per_round"]["sh_steps"], P.get("I3_rollback_sh"))
    band("I3-PEAK-GB", f.get("gpu_peak_alloc_GB"), P.get("I3_peak_gb"))
    tc = f["tokens"]
    want = P.get("I3_agree")
    out.append(("I3-AGREE", "REPORT" if want is None else ("CONFIRMED" if tc["equal_fraction"] >= want else "REFUTED"),
                "committed tokens equal to SH's: %.6f (first difference %s); registered >= %s" % (
                    tc["equal_fraction"], tc["first"], want)))
    above = P.get("I3_above_I2X")
    if above is None or not (med(f) and med(u)):
        out.append(("I3-ABOVE-I2X", "REPORT", "L4PTF %s vs L4PTU %s tok/s; registered %s" % (med(f), med(u), above)))
    else:
        out.append(("I3-ABOVE-I2X", "CONFIRMED" if (med(f) > med(u)) == bool(above) else "REFUTED",
                    "L4PTF %.1f vs L4PTU %.1f tok/s (the rung falsifier: below the previous rung); registered %s" % (
                        med(f), med(u), above)))
    out.append(("I3-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, f.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the fork module and the arena prefill (GPU)
def import_fork():
    """(module, gate): the fork of THIS tree (a byte copy of fia6's; the step script checks its blob); a failed import
    is the MUST gate C0_FORK."""
    try:
        import fork_vllm_unified_attention as FORK
        ok = callable(getattr(FORK, "unified_attention", None)) and hasattr(FORK, "kernel_unified_attention")
        return FORK, ("C0_FORK", True, ok, "%s (blob pinned by the step: %s)" % (FORK.__file__, FORK_BLOB))
    except Exception as e:                                       # the import is evidence: a failure is a MUST fail
        return None, ("C0_FORK", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300]))


def _prefill_chain(model, ids, inputs, steps: int, clock, X: int, hook: str):
    """Its own shipped prefill (SH's cache and max_gen_len) with the engines re-homed into a new ChainArena of X store
    slots per request. Returns (cache, arena, prefill s, first-token agreement, memory dict in GiB as run_arm_u)."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    WB._release_warm(model)
    gb = 1024 ** 3
    start = torch.cuda.memory_allocated() / gb
    torch.cuda.reset_peak_memory_stats()
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    arena = IA.for_engine(cache.layers[0].cache_engine, len(cache.layers), int(ids.shape[0]), X)
    Hook = IA.NoCopyPrefillHook if hook == "nocopy" else KA.PrefillHook
    clock.sync()
    t0 = time.time()
    with Hook(cache, arena):
        logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    first = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    mem = dict(start_GB=round(start, 3), prefill_peak_GB=round(torch.cuda.max_memory_allocated() / gb, 3),
               arena=arena.describe(), hook=hook)
    return cache, arena, round(time.time() - t0, 1), first, mem


# ------------------------------------------------------------------ the fork timing arm (GPU)
def run_arm_f(model, ids, inputs, steps: int, clock, opt, FORK):
    """L4PTF: r2_i2_cell.run_arm_u's steps and memory measures with spec_tentative_fork.TentRunnerF, the arena (made
    before the prefill) and phase marks on the probe rounds; check-only work and accounting only with --check-every 1
    (the miniature)."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    gb = 1024 ** 3
    K = int(opt["K"])
    X = fork_store_slots("L4PTF", K, opt["fork_store"])
    cache, arena, prefill_s, first, mem = _prefill_chain(model, ids, inputs, steps, clock, X, opt["prefill_hook"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()                              # the decode peak (the prefill peak is above)
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=opt["tu_gemm"], extra=X, journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"])
    check = opt["check_every"] == 1
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    runner = STF.TentRunnerF(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                             check_every=ce, union_check=check, union_stats=check, phase_marks=True)
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
               joint_phases=fork_phase_summary(phases, layers), fa_check=runner.vs.fa_check,
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4), cb0_GB=round(arena.chain_bytes() / 1e9, 4)),
               compile=_compile_report(FORK))
    runner.detach()
    del runner, rs, cache, arena
    return res


def _compile_report(FORK):
    ck = (getattr(FORK, "FORK_LAST_LAUNCH", None) or {}).get("kernel") if FORK is not None else None
    if ck is None:
        return None
    md = getattr(ck, "metadata", None)
    return dict(num_warps=getattr(md, "num_warps", None), num_stages=getattr(md, "num_stages", None),
                regs=getattr(ck, "n_regs", None), local_words=getattr(ck, "n_spills", None),
                shared=getattr(md, "shared", None))


# ------------------------------------------------------------------ the fork check arm (GPU)
def run_check_tf(model, ids, inputs, steps: int, clock, opt, shr, FORK):
    """TFC / TFCR (module docstring): TentRunnerF.run_teacher on SH's tokens (warm = 1 + --check-phase), check on,
    against SHR (whose step i is row r of pass g with i = warm + g R + r)."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    K = int(opt["K"])
    R = K + 1
    X = fork_store_slots(opt["arm"], K, opt["fork_store"])
    cache, arena, prefill_s, first, mem = _prefill_chain(model, ids, inputs, steps, clock, X, opt["prefill_hook"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=False)
    rs = RE.attach(model, cache, BC.S0, steps + 8, ops)
    KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    B, L = rs.B, rs.L
    G = int(opt["passes"])
    warm = 1 + int(opt["check_phase"])
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=CHECK_GEMM[opt["arm"]], extra=X, journal=8, redirect=True,
                      probe_every=opt["probe_every"])
    runner = STF.TentRunnerF(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                             check_every=1, union_check=True, union_stats=True, full_max=R)
    vs = runner.vs
    vs.fa_check = {}
    every = int(opt["probe_every"])
    is_probe = lambda g: every > 0 and g > 0 and g % every == 0         # noqa: E731
    g_prof = next((g for g in range(G // 2, G) if not is_probe(g)), None)
    box = dict(prof=None)
    run0 = runner.T.run_pass

    def run_pass(model_, plan):                                       # CHECK-ONLY: one profiled joint pass
        if opt["profile"] and plan.R > 1 and runner.round == g_prof and box["prof"] is None:
            out, box["prof"] = WB._profile(lambda: run0(model_, plan))
            return out
        return run0(model_, plan)
    runner.T.run_pass = run_pass
    changed_lr = torch.zeros((L, R), dtype=torch.int64, device="cuda")
    hist = torch.zeros(6, dtype=torch.int64, device="cuda")
    st = dict(compared=0, first=[], d_top=[], d_max=[], agree=[])

    def on_pass(g, lg):                                               # CHECK-ONLY: against SHR at steps warm + g R + r
        i0 = warm + g * R
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
        tv = shr["topv"][i0:i0 + R].to("cuda")
        ti = shr["topi"][i0:i0 + R].to("cuda")
        lgr = lg.permute(1, 0, 2)
        st["d_top"].append((lgr.gather(-1, ti) - tv).abs().amax(-1).cpu())
        st["d_max"].append((lgr.amax(-1) - shr["fmax"][:, i0:i0 + R].t().to("cuda")).abs().cpu())
        st["agree"].append((lg.argmax(-1) == inputs[:, i0 + 1:i0 + R + 1]).cpu())
    runner.run_teacher(inputs, G, on_pass=on_pass, warm=warm)
    clock.sync()
    stats = VR.read_stats(vs)
    dt = torch.stack(st["d_top"])
    dm = torch.stack(st["d_max"])
    ag = torch.stack(st["agree"])
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
               errors=RE.errors(rs), fa_check=vs.fa_check, passes=G, R=R, warm=warm, tent_calls=tr["calls"],
               calls_per_pass=WB.per_pass(stats["calls"], stats["passes"]), profile=box["prof"], prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, spec=runner.meta), mem=mem,
               facts=check_facts(BC.S0, warm, G, R), store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4)),
               compile=_compile_report(FORK))
    runner.detach()
    del runner, vs, rs, cache, arena
    return res


# ------------------------------------------------------------------ WS-G's fork arms (GPU)
def run_arm_wf(model, ids, inputs, steps: int, clock, opt, sample, slot_every: int, FORK):
    """ONE fork WS-G arm (WF / WFP): its own arena prefill, TentRunnerF (--tu-gemm, union checks on; WFP with the
    VerifyOps control fork_small_store at X = --chk-plant-extra) with r2_i2_check.ContractCapture installed during
    runner.run. Returns (res, the capture after finish())."""
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    import r2_i2_check as ICK
    w = WSG_F[opt["arm"]]
    K = int(opt["K"])
    X = fork_store_slots(opt["arm"], K, opt["fork_store"], opt["plant_X"])
    cache, arena, prefill_s, first, mem = _prefill_chain(model, ids, inputs, steps, clock, X, opt["prefill_hook"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=opt["tu_gemm"], extra=X, journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"])
    runner = STF.TentRunnerF(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], check_every=1,
                             union_check=True, union_stats=True, vneg=w.get("plant"))
    cap = ICK.ContractCapture(runner, steps, BC.S0, sample, exact=False, RE=RE, tentative=True, slot_every=slot_every)
    cap.install()
    try:
        runner.run(steps)
    finally:
        cap.remove()
    rd = runner.read()
    rd.pop("trace_all", None)
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
               probe_every=opt["probe_every"], prefill_s=prefill_s, first_token_agreement=first,
               meta=dict(rs.meta, spec=runner.meta), X=X, mem=mem)
    cap.finish(steps)
    runner.detach()
    del runner, rs, cache, arena
    return res, cap


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
    FORK = None
    if any(x in FORK_ARMS for x in a.arms):
        FORK, g0 = import_fork()
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
    R = a.spec + 1
    warm = 1 + a.check_phase
    check_t = a.check_every == 1
    sample = a.sample if a.sample is not None else CK.default_rows(a.batch)
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
                                  fit_batch_MODEL=WA.fit_batch(a.batch, sh_start, pa), budget_GB=BUDGET_GB))
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            gates.append(mem_record("SH", "prefill + decode", pa))
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
            if arm in ("TFC", "TFCR") and shr is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: no SHR record"))
                continue
            if arm in FORK_ARMS and FORK is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: the fork did not import (C0_FORK)"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, rounds=0, journal=a.spec_journal_blocks, extra=a.spec_extra, X=a.joint_extra,
                       spill=0, tu_gemm=a.tu_gemm, twin=a.spec_twin, body=a.spec_body, graph=a.spec_graph == "on",
                       redirect=a.spec_redirect == "on", probe_every=a.probe_every, check_every=a.check_every,
                       passes=a.check_passes, check_phase=a.check_phase, profile=a.profile == "on",
                       plant_X=a.chk_plant_extra, fork_store=a.fork_store, prefill_hook=a.prefill_hook)
            log("arm %s (%s)" % (arm, opt))
            if arm in WSG_I2 or arm in WSG_F:                          # WS-G's check: the arm, then REF
                plant = bool(WSG_F.get(arm, {}).get("plant"))
                stage = "arm"
                try:
                    if arm in WSG_I2:
                        res, cap = I2C.run_arm_w(model, ids, inputs, a.steps, clock, opt, sample, a.chk_slot_every)
                    else:
                        res, cap = run_arm_wf(model, ids, inputs, a.steps, clock, opt, sample, a.chk_slot_every, FORK)
                    pa, pr = BC.gpu_peak()
                    tc = SC.token_check(res["out"], res["read"]["count"], inputs, a.steps)
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d (CHECK RUN: the capture ran inside "
                                                             "the rounds; no rate here is a timing)" % (
                                                                 res["read"]["rounds"], a.spec)))
                    gates.extend(x for x in SC.arm_gates(arm, res, None) if x[0].endswith("-ERR"))
                    gates.append(WA.commit_n_gate(arm, res, a.steps))
                    gates.append(mem_record(arm, res.get("mem"), pa))
                    out_t = res["out"]
                    union_rd = res["read"].get("union")
                    del res
                    BC.free()
                    stage = "ref"
                    log("REF of %s (teacher-forced on its committed tokens, %s)" % (arm, I2C.REF_OPS))
                    inputs_ref = cap.ref_inputs(out_t, inputs[:, 0], inputs)
                    ref = I2C.run_ref_w(model, ids, inputs, a.steps, clock, cap, inputs_ref, fp_sh)
                    stage = "compare"
                    g, crec = (I2C.wsg_gates(arm, cap.result, ref) if arm in WSG_I2 else
                               wsg_gates_f(arm, cap.result, ref))
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
                    gates.append(("%s-CHK" % arm, arm == "WJ", False, why))
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
                    gates.append(mem_record(arm, rec.get("memory"), pa))
                elif arm == "L4PTU":                                   # r2_i2_cell's arm, unchanged (X --joint-extra)
                    res = I2C.run_arm_u(model, ids, inputs, a.steps, clock, opt)
                    pa, pr = BC.gpu_peak()
                    rec = I2C.summarize_u(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                    rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                    sp[arm] = rec
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                    gates.extend(I2C.arm_gates_u(arm, res, rec, a.steps, L, check_t, False))
                    gates.append(mem_record(arm, rec.get("memory"), pa))
                elif arm == "L4PTF":
                    res = run_arm_f(model, ids, inputs, a.steps, clock, opt, FORK)
                    pa, pr = BC.gpu_peak()
                    rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                    rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               union=I2C.union_record(res["read"], L), fork=fork_record(res["read"]),
                               joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                               mem_arena=res["mem"])
                    jm = rec["phases_median"].get("joint")
                    shm = sh["rates"]["steady_step_ms"] if sh else None
                    rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                    sp[arm] = rec
                    gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                    gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                                 WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                    gates.extend(fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                    gates.append(mem_record(arm, res["mem"], pa))
                elif arm == "SHR":
                    n = warm + a.check_passes * R
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
                else:                                                  # TFC / TFCR
                    res = run_check_tf(model, ids, inputs, a.steps, clock, opt, shr, FORK)
                    pa, pr = BC.gpu_peak()
                    res.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                               ready_record=WB.ready_record(dict(win_sel=res["stats"]["win_sel"],
                                                                 win_host=res["stats"]["win_host"],
                                                                 extras_pool=res["stats"]["extras_pool"],
                                                                 extras_host=res["stats"]["extras_host"]),
                                                            res["passes"], dims["block_bytes"]))
                    chk[arm] = res
                    gates.append(("%s-RUN" % arm, True, True, "%d teacher-forced passes of R %d (warm %d; past-edge "
                                                             "rows per request %d)" % (res["passes"], res["R"], warm,
                                                                                       res["facts"]["past_edge_rows"])))
                    gates.extend(check_gates_tf(arm, res, bool(a.require_stress), L))
                    gates.append(mem_record(arm, res["mem"], pa))
                res = None                                   # frees the arm's record; the SHR branch sets no res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, joint_attn=a.joint_attn,
                joint_extra=a.joint_extra, fork_store=a.fork_store, prefill_hook=a.prefill_hook,
                fork_store_slots={x: fork_store_slots(x, a.spec, a.fork_store, a.chk_plant_extra)
                                  for x in FORK_ARMS if x in a.arms},
                tu_gemm=a.tu_gemm, check_passes=a.check_passes, check_phase=a.check_phase,
                check_facts=(check_facts(BC.S0, warm, a.check_passes, R) if any(x in ("TFC", "TFCR") for x in a.arms)
                             else None),
                spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin, spec_body=a.spec_body,
                spec_graph=a.spec_graph, spec_redirect=a.spec_redirect, probe_every=a.probe_every,
                check_every=a.check_every, profile=a.profile, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_I3, by_batch=PRED_I3_BY_B), phase_groups=PHASE_GROUPS_F,
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB,
                memory_model=dict(store_GB=store_gb(a.batch, fork_store_slots("L4PTF", a.spec, a.fork_store)),
                                  cb0_GB=cb0_gb(a.batch), copy_hook_transient_GB=layer_rows_gb(a.batch, a.pool)),
                wsg=dict(sample_rows=sample, chk_slot_every=a.chk_slot_every, chk_plant_extra=a.chk_plant_extra,
                         tie_logit=I2C.CHK_TIE_LOGIT, ref_ops=I2C.REF_OPS, check_only_ops=CK.CHECK_ONLY_OPS),
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_file=FORK_FILE,
                fork_blob=FORK_BLOB)
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
                  "joint pass %s SH steps; phases %s; ROLLBACK per round %s; calls %s; tokens equal %.6f (first %s); "
                  "fingerprints %s; memory %s; peak %s GB" % (
                      name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                      r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"], r["joint_sh_steps"],
                      r["phases_median"], r["rollback_per_round"], r["calls"], r["tokens"]["equal_fraction"],
                      r["tokens"]["first"], r["fingerprints"], r["memory"], r.get("gpu_peak_alloc_GB")))
            if "union" in r:
                print("REPORT %s union: %s" % (name, r["union"]))
            if "fork" in r:
                print("REPORT %s fork: %s; joint-pass phases (probe) %s; store %s; arena memory %s; compile %s" % (
                    name, r["fork"], r["joint_phases_fork"], r["store"], r["mem_arena"], r["compile"]))
            elif "joint_phases_union" in r:
                print("REPORT %s joint-pass phases (probe) %s" % (name, r["joint_phases_union"]))
        c = chk.get(name)
        if c is not None and (name in WSG_I2 or name in WSG_F):
            print("REPORT %s (WS-G check): %s; parts %s; failed %s; logit %s; tokens equal to SH's %.6f; union %s" % (
                name, "PASS" if c["ok"] else "FAIL", c["parts"], c["failed"], c["logit"]["detail"],
                c["tokens"]["equal_fraction"],
                {k: (c["union"] or {}).get(k) for k in ("dropped", "unserved", "vis_bad", "vb_bad", "chain_bad")}))
        elif c is not None and name != "SHR":
            print("REPORT %s: selections %s; logits %s; stats %s; ready / residual %s; calls per pass %s; tent calls %s; "
                  "facts %s; store %s; memory %s; profile %s" % (
                      name, c["sel"], c["logits"], c["stats"], c["ready_record"], c["calls_per_pass"], c["tent_calls"],
                      c["facts"], c["store"], c["mem"], c["profile"]))
        elif c is not None:
            print("REPORT SHR: fingerprints %s; argmax unequal %d; profile %s" % (c["fingerprints"], c["tok_bad"],
                                                                                  c["profile"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i3: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
