"""Rung I4D of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 4 of
KS/analysis/plan/LEDGER_model_vs_measured.md revision 5 (R5.5 item 4, the store refetch): the GPU cell of the
ACCOUNTING option (--store-acct 1) and of the keep arms L4PTFK / L4PTFKW = L4PTFD (rung I4A) with the store KEPT across
rounds (benchmarks/Efficiency/r2_i4d_keep.py: the block-id map, the eviction rule, the capacity proof). A NEW script:
r2_i3_cell.py and r2_i4a_cell.py are not edited (their functions are imported). Step script:
benchmarks/Efficiency/r2_i4d_step.sh (cells i4dm, i4dc, i4dd, i4da, i4db).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order L4PTFD, L4PTFK, L4PTFKW (the keep arms' tokens are compared
with L4PTFD's), each with its own shipped prefill into the chain arena (NoCopyPrefillHook), on the inputs of i4aa /
i4ab (the same workload pin, S0, document rows, K 3, fused twin and body, graph, redirect, probe 16, store X 129,
--tu-gemm natural):
  SH       the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTFD   I4A: r2_i4a_lean's TentRunnerFD (the default path, unchanged); with --store-acct 1 r2_i4d_keep's TentRunnerFK
           (keep False, acct True): L4PTFD's values plus the accounting counters (parts A and B)
  L4PTFK   I4D: r2_i4d_keep's TentRunnerFK (keep True: the store's listed blocks reuse the kept slots, docstring 2)
  L4PTFKW  I4D: L4PTFK plus kwin (row 0's window loads that the kept store holds are copied from it, docstring 4)
  (both keep arms: acct with --store-acct 1; the fresh-fetch byte checks with --check-every 1; the same arena, store,
  joint pass and memory)
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN, -RESID, -ATTN1, -FILL1, -LEAN; with
--check-every 1 -PRESENT, -VIS, -VB, -TAIL, -CHAIN, -IDX, -RA; with --require-stress -STRESS, -FSTRESS; per keep arm
-KEEP (the keep counted every joint pass and layer: passes = joint passes, listed > 0, 0 <= fetched <= listed, hits >
0, L4PTFKW window hits > 0; with the union statistics on: listed = the union's listed host blocks); with --check-every 1
-CAP (overflow 0, duplicate keys 0, a listed key missing after its step 0, blocks unequal to a fresh fetch 0, store
blocks checked = listed, L4PTFKW window blocks checked = window hits); with --store-acct 1 <arm>-ACCT (passes = joint
passes, listed > 0, repeat <= listed) and ACCT-EQ-<keep arm> (the same lists and window loads as L4PTFD, per pass);
<keep arm>-EQF (its committed tokens, check-round fingerprints, acceptance record and state checksums = L4PTFD's in the
same job, bit for bit: MUST with --eq-must 1, REPORT otherwise). REPORT: MEM, the phases, the accounting (shares,
bytes), the keep (hit shares, bytes not fetched), NEW host blocks and the host-gather wait per round of each arm.
REGISTERED PREDICTIONS PRED_I4D_BY_B: every value None until the lead registers them with PRED_STAMP_I4D (a None key is
REPORT); the step script STOPS the timing cells (i4da, i4db) while PRED_STAMP_I4D is None. EXIT: the number of failed
MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_i4a_cell as A4          # noqa: E402
import r2_i4a_lean as LN          # noqa: E402
import r2_i4d_keep as KP          # noqa: E402

I3C = A4.I3C
WA, SC, BC = A4.WA, A4.SC, A4.BC
ARMS = ("L4PTFD", "L4PTFK", "L4PTFKW")
KEEP_ARMS = ("L4PTFK", "L4PTFKW")
TOPK = A4.TOPK
FORK_BLOB = A4.FORK_BLOB
STRONGEST = A4.STRONGEST
BUDGET_GB = A4.BUDGET_GB
USABLE_GB = A4.USABLE_GB
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4D_BY_B (per batch) and PRED_STAMP_I4D
# (`date -u`, L33) in a commit BEFORE i4da / i4db (the step script stops while PRED_STAMP_I4D is None). Keys, each
# (low, high) unless noted; units in brackets:
#  THE ACCOUNTING (L4PTFD with --store-acct 1: the accounting cells i4dc / i4dd and i4dm):
#   I4D_repeat_share     [fraction] part A: repeat / listed over the joint passes after the first (the store's host
#                        blocks that the previous pass's store list of the same layer held)
#   I4D_listed_pass      [(head, block) per joint pass] listed host blocks per joint pass (all layers, heads, requests),
#                        the mean over the passes after the first
#   I4D_win_prev_share   [fraction] part B: row 0's window host loads that the previous pass's store list of the same
#                        layer held / row 0's window host loads, passes after the first
#   I4D_win_prev_pass    [(head, block) per joint pass] part B's count per joint pass, passes after the first
#  THE ARMS (P = I4D for L4PTFK, the store keep; P = I4DW for L4PTFKW, the store keep + the window from the store):
#   I4D_hit_share        [fraction] L4PTFK: store hits / listed over the passes after the first
#   I4DW_win_hits_pass   [(head, block) per joint pass] L4PTFKW: window loads served by the kept store, passes after
#                        the first
#   P_ratio              [x SH] the arm's median steady committed tok/s / SH's
#   P_vs_I4A             [x] the arm / L4PTFD median steady committed tok/s (same job): the lever's change
#   P_joint_saving       [ms per round] L4PTFD joint phase median - the arm's (same job)
#   P_wait_saving        [ms per round] L4PTFD host-gather wait median - the arm's (kv_at_verify wait_ms_median, the
#                        probe rounds: the ledger's "host-gather spans")
#   P_new_saving         [fraction] 1 - the arm's NEW host blocks per round / L4PTFD's (kv_at_verify, the meter)
#   P_agree              [fraction, a floor] the arm's committed tokens equal to SH's
#   P_eq_I4A             [True] the arm's EQF passed (= L4PTFD bit for bit)
#   P_above_I4A          [True] the rung falsifier "the arm not above L4PTFD in the same job"
#   P_peak_gb            [GB] the arm's peak allocated GB, the prefill included (the MEM gate's value)
#   I4A_ratio            [x SH] L4PTFD / SH in this job (rung I4A, re-measured)
# The builder's MODEL (in the hand-over report, with the arithmetic) is an input for the lead, not a registration.
PRED_STAMP_I4D = "2026-10-29T09:41:32Z"
_ARM_KEYS = ("ratio", "vs_I4A", "joint_saving", "wait_saving", "new_saving", "agree", "eq_I4A", "above_I4A", "peak_gb")
PRED_KEYS = (("I4D_repeat_share", "I4D_listed_pass", "I4D_win_prev_share", "I4D_win_prev_pass", "I4A_ratio",
              "I4D_hit_share", "I4DW_win_hits_pass") + tuple("I4D_" + k for k in _ARM_KEYS)
             + tuple("I4DW_" + k for k in _ARM_KEYS))
# The lead's registration (before any I4D cell ran): the builder's MODEL bands (hand-over report section 4); P_ratio =
# the I4A_ratio band x the P_vs_I4A band; I4D_above_I4A at B 216 is None (REPORT): the model's mid is about 1.000, so a
# flag would be a coin flip.
PRED_I4D_BY_B = {
    216: dict(I4D_repeat_share=(0.05, 0.40), I4D_listed_pass=(3000, 12000), I4D_win_prev_share=(0.2, 0.7),
              I4D_win_prev_pass=(2000, 6400), I4A_ratio=(0.51, 0.55), I4D_hit_share=(0.2, 0.6),
              I4DW_win_hits_pass=(2000, 7000),
              I4D_ratio=(0.507, 0.553), I4D_vs_I4A=(0.995, 1.006), I4D_joint_saving=(-2.0, 2.0),
              I4D_wait_saving=(1.0, 6.0), I4D_new_saving=(0.05, 0.30), I4D_agree=0.25, I4D_eq_I4A=True,
              I4D_above_I4A=None, I4D_peak_gb=(83.5, 84.25),
              I4DW_ratio=(0.508, 0.557), I4DW_vs_I4A=(0.997, 1.012), I4DW_joint_saving=(-1.0, 4.0),
              I4DW_wait_saving=(3.0, 9.0), I4DW_new_saving=(0.15, 0.45), I4DW_agree=0.25, I4DW_eq_I4A=True,
              I4DW_above_I4A=True, I4DW_peak_gb=(83.5, 84.25)),
    318: dict(I4D_repeat_share=(0.05, 0.40), I4D_listed_pass=(6000, 30000), I4D_win_prev_share=(0.05, 0.25),
              I4D_win_prev_pass=(4000, 11000), I4A_ratio=(0.59, 0.62), I4D_hit_share=(0.15, 0.6),
              I4DW_win_hits_pass=(4000, 14000),
              I4D_ratio=(0.590, 0.629), I4D_vs_I4A=(1.000, 1.014), I4D_joint_saving=(0.0, 8.0),
              I4D_wait_saving=(3.0, 18.0), I4D_new_saving=(0.02, 0.12), I4D_agree=0.25, I4D_eq_I4A=True,
              I4D_above_I4A=True, I4D_peak_gb=(77.3, 78.15),
              I4DW_ratio=(0.595, 0.637), I4DW_vs_I4A=(1.009, 1.027), I4DW_joint_saving=(5.0, 15.0),
              I4DW_wait_saving=(9.0, 30.0), I4DW_new_saving=(0.06, 0.20), I4DW_agree=0.25, I4DW_eq_I4A=True,
              I4DW_above_I4A=True, I4DW_peak_gb=(77.3, 78.15)),
}


def log(msg):
    print("[r2_i4d %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFK,L4PTFKW", help="a subset of %s, in this order (SH runs first; "
                                                                    "L4PTFD before the keep arms)" % ",".join(ARMS))
    ap.add_argument("--store-acct", type=int, default=0, choices=(0, 1),
                    help="1: the accounting counters in every arm (L4PTFD then runs r2_i4d_keep with keep off)")
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
                    help="0 = the probe rounds; 1 = every round, the union / fork checks and the keep's byte check on "
                         "(the miniature)")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    ap.add_argument("--eq-must", type=int, default=0, choices=(0, 1), help="the keep arms' EQF a MUST gate")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if a.arms != [x for x in ARMS if x in a.arms]:
        ap.error("--arms: in the order %s (the keep arms' EQF compares with L4PTFD's tokens)" % ",".join(ARMS))
    if a.eq_must and not ("L4PTFD" in a.arms and any(x in a.arms for x in KEEP_ARMS)):
        ap.error("--eq-must 1 needs L4PTFD and a keep arm")
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


def runner_choice(arm: str, acct: bool, check: bool):
    """(the runner's kind, its keyword arguments) of an arm: L4PTFD without the accounting = I4A's TentRunnerFD (the
    default path, unchanged); otherwise r2_i4d_keep's TentRunnerFK."""
    if arm == "L4PTFD" and not acct:
        return "FD", {}
    if arm == "L4PTFD":
        return "FK", dict(keep=False, acct=True)
    kw = dict(keep=True, acct=bool(acct), kcheck=bool(check))
    if arm == "L4PTFKW":
        kw["kwin"] = True
    return "FK", kw


# ------------------------------------------------------------------ pure parts (CPU-tested)
def acct_summary(kp, block_bytes: int) -> dict:
    """The accounting / keep record of read()["keep"]: per pass lists -> the passes after the first (pass 0 has no
    previous list: its repeat is 0 by definition) and all passes; shares and bytes per joint pass."""
    if not kp:
        return None
    n = int(kp.get("passes") or 0)
    lst = list(kp.get("listed") or [])
    fet = list(kp.get("fetched") or [])
    out = dict(passes=n, listed_total=sum(lst), fetched_total=sum(fet), hits_total=sum(lst) - sum(fet),
               block_bytes=block_bytes, keep=kp.get("keep"), acct=kp.get("acct"))
    l1, f1 = lst[1:], fet[1:]
    s1 = sum(l1)
    out["listed_pass_after_first"] = round(statistics.fmean(l1), 2) if l1 else None
    if kp.get("keep"):
        out["hit_share_after_first"] = round((s1 - sum(f1)) / s1, 6) if s1 else None
        out["hit_bytes_pass_after_first"] = round((s1 - sum(f1)) / len(l1) * block_bytes) if l1 else None
    if kp.get("acct"):
        r1 = (kp.get("repeat") or [])[1:]
        out.update(repeat_total=kp.get("repeat_total"),
                   repeat_share_after_first=(round(sum(r1) / s1, 6) if s1 else None),
                   repeat_bytes_pass_after_first=(round(sum(r1) / len(r1) * block_bytes) if r1 else None))
        w1, p1 = (kp.get("win_loads") or [])[1:], (kp.get("win_prev") or [])[1:]
        out.update(win_loads_total=kp.get("win_loads_total"), win_prev_total=kp.get("win_prev_total"),
                   win_loads_pass_after_first=(round(statistics.fmean(w1), 2) if w1 else None),
                   win_prev_pass_after_first=(round(statistics.fmean(p1), 2) if p1 else None),
                   win_prev_share_after_first=(round(sum(p1) / sum(w1), 6) if sum(w1) else None),
                   win_prev_bytes_pass_after_first=(round(sum(p1) / len(p1) * block_bytes) if p1 else None))
        if kp.get("keep"):
            out["victims_alive_total"] = kp.get("victims_alive_total")
    if kp.get("kwin"):
        h1 = (kp.get("win_hits") or [])[1:]
        out.update(win_hits_total=kp.get("win_hits_total"),
                   win_hits_pass_after_first=(round(statistics.fmean(h1), 2) if h1 else None),
                   win_hit_bytes_pass_after_first=(round(sum(h1) / len(h1) * block_bytes) if h1 else None))
    out["check"] = kp.get("check")
    return out


def keep_gate(arm, rd, check: bool):
    """<keep arm>-KEEP (MUST): the keep counted every joint pass (passes = the union passes), listed > 0, 0 <= fetched
    <= listed, hits > 0; L4PTFKW also window hits > 0; with the union statistics on, listed = the union's listed host
    blocks (extras_host)."""
    kp = rd.get("keep") or {}
    u = rd.get("union") or {}
    passes = u.get("passes")
    lt, ft = kp.get("listed_total"), kp.get("fetched_total")
    ok = (bool(kp.get("keep")) and passes is not None and kp.get("passes") == passes and (lt or 0) > 0
          and ft is not None and 0 <= ft <= lt and lt - ft > 0)
    if kp.get("kwin"):
        ok = ok and (kp.get("win_hits_total") or 0) > 0
    eh = u.get("extras_host")
    if check:
        ok = ok and eh == lt
    return ("%s-KEEP" % arm, True, bool(ok),
            "joint passes counted %s (union %s); listed %s, fetched %s, hits %s; window loads from the store %s; union "
            "listed host blocks %s (%s)" % (
                kp.get("passes"), passes, lt, ft, (lt - ft) if (lt is not None and ft is not None) else None,
                kp.get("win_hits_total") if kp.get("kwin") else "(off)", eh,
                "compared" if check else "not compared: the union statistics are off"))


def cap_gate(arm, rd):
    """<keep arm>-CAP (MUST, --check-every 1): the capacity proof's terms counted on every step (r2_i4d_keep docstring
    2: (iii) overflow, (ii) dup, (iv) missing) and the fresh-fetch byte checks (bytes_bad 0, bytes_checked = listed;
    L4PTFKW: window rows checked = window hits)."""
    kp = rd.get("keep") or {}
    c = kp.get("check") or {}
    ok = (c.get("overflow") == 0 and c.get("dup") == 0 and c.get("missing") == 0 and c.get("bytes_bad") == 0
          and (c.get("bytes_checked") or 0) > 0 and c.get("bytes_checked") == kp.get("listed_total"))
    if kp.get("kwin"):
        ok = ok and c.get("win_checked") == kp.get("win_hits_total")
    return ("%s-CAP" % arm, True, bool(ok), "overflow %s, duplicate keys %s, missing %s, blocks unequal to a fresh "
            "fetch %s; store blocks checked %s (listed %s); window blocks checked %s (window hits %s)" % (
                c.get("overflow"), c.get("dup"), c.get("missing"), c.get("bytes_bad"), c.get("bytes_checked"),
                kp.get("listed_total"), c.get("win_checked"), kp.get("win_hits_total")))


def acct_gate(arm, rd):
    """<arm>-ACCT (MUST, --store-acct 1): passes = the union passes, listed > 0, 0 <= repeat <= listed."""
    kp = rd.get("keep") or {}
    u = rd.get("union") or {}
    lt, rt = kp.get("listed_total"), kp.get("repeat_total")
    ok = (bool(kp.get("acct")) and kp.get("passes") == u.get("passes") and (lt or 0) > 0 and rt is not None
          and 0 <= rt <= lt)
    return ("%s-ACCT" % arm, True, bool(ok), "joint passes %s (union %s); listed %s; repeat %s" % (
        kp.get("passes"), u.get("passes"), lt, rt))


def acct_eq_gate(kd, kk, arm="L4PTFK"):
    """ACCT-EQ-<keep arm> (MUST, both arms counting): the keep arm lists the same host blocks and makes the same window
    loads as L4PTFD (per pass listed, repeat, window loads, window loads the previous store held)."""
    name = "ACCT-EQ-%s" % arm
    if not kd or not kk:
        return (name, True, False, "an arm has no accounting record")
    keys = ("listed", "repeat", "win_loads", "win_prev")
    ok = all(kd.get(k) == kk.get(k) for k in keys) and (kd.get("listed_total") or 0) > 0
    return (name, True, bool(ok), "L4PTFD / %s totals: listed %s / %s, repeat %s / %s, window loads %s / %s, from the "
            "previous store %s / %s; per pass equal %s" % (
                arm, kd.get("listed_total"), kk.get("listed_total"), kd.get("repeat_total"), kk.get("repeat_total"),
                kd.get("win_loads_total"), kk.get("win_loads_total"), kd.get("win_prev_total"),
                kk.get("win_prev_total"), ok))


def eq_gate(arm, fk, fd, steps: int, must: bool):
    """<keep arm>-EQF: r2_i4a_cell.eq_gate's comparison (committed tokens, check-round fingerprints, acceptance record,
    state checksums) of a keep arm against L4PTFD in the same job."""
    name, _, ok, det = A4.eq_gate(arm, fk, fd, steps, must)
    if fd is None:
        return ("%s-EQF" % arm, must, False, "no L4PTFD record in this job")
    return ("%s-EQF" % arm, must, ok, det.replace("L4PTF's", "L4PTFD's"))


def verdicts(sh, sp, B: int, eqf=None, acct=None):
    """PRED_I4D_BY_B[B] against the arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN for a
    missing keep arm. eqf: {keep arm: its EQF gate's passed flag}; acct: {arm: acct_summary}."""
    P = PRED_I4D_BY_B.get(B) or {k: None for k in PRED_KEYS}
    out = []
    acct = acct or {}
    eqf = eqf or {}

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

    def kv(r, name):
        return ((r or {}).get("kv_at_verify") or {}).get(name)

    def flag(name, got, want, detail):
        out.append((name, "REPORT" if (want is None or got is None) else ("CONFIRMED" if got == bool(want) else
                                                                           "REFUTED"), detail))
    d = (sp or {}).get("L4PTFD")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    ad = acct.get("L4PTFD") or {}
    if d is not None:
        band("I4A-RATIO", (med(d) / shm) if (med(d) and shm) else None, P.get("I4A_ratio"))
    src = ad if ad.get("acct") else next((acct[x] for x in KEEP_ARMS if (acct.get(x) or {}).get("acct")), {})
    band("I4D-REPEAT-SHARE", src.get("repeat_share_after_first"), P.get("I4D_repeat_share"))
    band("I4D-LISTED-PASS", src.get("listed_pass_after_first"), P.get("I4D_listed_pass"))
    band("I4D-WIN-PREV-SHARE", src.get("win_prev_share_after_first"), P.get("I4D_win_prev_share"))
    band("I4D-WIN-PREV-PASS", src.get("win_prev_pass_after_first"), P.get("I4D_win_prev_pass"))
    for arm, pre in (("L4PTFK", "I4D"), ("L4PTFKW", "I4DW")):
        k = (sp or {}).get(arm)
        ak = acct.get(arm) or {}
        n = pre.replace("_", "-")
        if k is None:
            out.append(("%s-RUN" % arm, "NOT_RUN", "the arm did not run"))
            continue
        if arm == "L4PTFK":
            band("I4D-HIT-SHARE", ak.get("hit_share_after_first"), P.get("I4D_hit_share"))
        else:
            band("I4DW-WIN-HITS-PASS", ak.get("win_hits_pass_after_first"), P.get("I4DW_win_hits_pass"))
        band(n + "-RATIO", (med(k) / shm) if (med(k) and shm) else None, P.get(pre + "_ratio"))
        band(n + "-VS-I4A", (med(k) / med(d)) if (med(k) and med(d)) else None, P.get(pre + "_vs_I4A"))
        band(n + "-JOINT-SAVING", (ph(d, "joint") - ph(k, "joint")) if (ph(d, "joint") is not None and
                                                                        ph(k, "joint") is not None) else None,
             P.get(pre + "_joint_saving"))
        band(n + "-WAIT-SAVING", (kv(d, "wait_ms_median") - kv(k, "wait_ms_median")) if (
            kv(d, "wait_ms_median") is not None and kv(k, "wait_ms_median") is not None) else None,
             P.get(pre + "_wait_saving"))
        band(n + "-NEW-SAVING", (1.0 - kv(k, "new_blocks_per_round") / kv(d, "new_blocks_per_round")) if (
            kv(k, "new_blocks_per_round") is not None and kv(d, "new_blocks_per_round")) else None,
             P.get(pre + "_new_saving"))
        band(n + "-PEAK-GB", A4.peak_gb(k), P.get(pre + "_peak_gb"))
        tc = k["tokens"]
        want = P.get(pre + "_agree")
        out.append((n + "-AGREE", "REPORT" if want is None else ("CONFIRMED" if tc["equal_fraction"] >= want else
                                                                 "REFUTED"),
                    "committed tokens equal to SH's: %.6f (first difference %s); registered >= %s" % (
                        tc["equal_fraction"], tc["first"], want)))
        flag(n + "-EQ-I4A", eqf.get(arm), P.get(pre + "_eq_I4A"), "%s-EQF passed %s; registered %s" % (
            arm, eqf.get(arm), P.get(pre + "_eq_I4A")))
        flag(n + "-ABOVE-I4A", (med(k) > med(d)) if (med(k) and med(d)) else None, P.get(pre + "_above_I4A"),
             "%s %s vs L4PTFD %s tok/s (the rung falsifier: not above the previous rung); registered %s" % (
                 arm, med(k), med(d), P.get(pre + "_above_I4A")))
        out.append((n + "-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST,
                                                                                      k.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_k(model, ids, inputs, steps: int, clock, opt, FORK, arm: str, acct: bool):
    """r2_i4a_cell.run_arm_x's steps, line for line, with the runner of runner_choice(arm, acct, check), plus
    r2_i4a_cell.state_checksum(runner) after the run (EQF)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, acct, check)
    Runner = LN.lean_classes(STF)[1] if kind == "FD" else KP.keep_classes(STF)[1]
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
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                    check_every=ce, union_check=check, union_stats=check, phase_marks=True, **kw)
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
               compile=I3C._compile_report(FORK), state=A4.state_checksum(runner), runner_kind=kind, runner_kw=kw)
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
    gates, sh, sp, inputs, keep, acct = [], None, {}, None, {}, {}
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
            gates.append(A4.mem_record("SH", "prefill + decode", pa))
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
                       prefill_hook=a.prefill_hook, store_acct=a.store_acct)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_k(model, ids, inputs, a.steps, clock, opt, FORK, arm, bool(a.store_acct))
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), runner=res["runner_kind"],
                           runner_kw=res["runner_kw"])
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = A4.peak_gb(rec)
                rec["store_record"] = acct_summary(rd.get("keep"), bb)
                acct[arm] = rec["store_record"]
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (rd["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(A4.mem_record(arm, res["mem"], pa))
                gates.append(A4.lean_gate(arm, dict(rd, K=a.spec), L))
                if a.store_acct:
                    gates.append(acct_gate(arm, rd))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=A4.acct_record(rd), state=res["state"], keep=rd.get("keep"))
                if arm == "L4PTFD":
                    keep["L4PTFD"] = cpu
                else:
                    gates.append(keep_gate(arm, rd, check_t))
                    if check_t:
                        gates.append(cap_gate(arm, rd))
                    g = eq_gate(arm, cpu, keep.get("L4PTFD"), a.steps, bool(a.eq_must))
                    eqf[arm] = bool(g[2])
                    gates.append(g)
                    if a.store_acct and "L4PTFD" in keep:
                        gates.append(acct_eq_gate(keep["L4PTFD"]["keep"], rd.get("keep"), arm))
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch, eqf, acct)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, store_acct=a.store_acct,
                fork_store=a.fork_store, fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store),
                prefill_hook=a.prefill_hook, tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks,
                spec_twin=a.spec_twin, spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                eq_must=a.eq_must, predictions=dict(stamp=PRED_STAMP_I4D, by_batch=PRED_I4D_BY_B),
                keep=dict(extra_ops_per_layer=KP.KEEP_EXTRA_OPS, block_bytes_head=bb, key_bits=32),
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                memory_model=dict(store_GB=I3C.store_gb(a.batch, I3C.fork_store_slots("L4PTF", a.spec, a.fork_store)),
                                  cb0_GB=I3C.cb0_gb(a.batch)),
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-16s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; vs %.1f %s; "
              "round median %s ms; phases %s; draft step %s ms = %s SH; joint %s SH steps; calls %s; lean %s; "
              "tokens equal %.6f (first %s); fingerprints %s; peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"], r["draft_step_sh"],
                  r["joint_sh_steps"], r["calls"], r.get("lean"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r["fingerprints"], r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        print("REPORT %s store: runner %s %s; NEW host blocks per round %s, host-gather wait median %s ms; record %s" % (
            name, r["runner"], r["runner_kw"], r["kv_at_verify"].get("new_blocks_per_round"),
            r["kv_at_verify"].get("wait_ms_median"),
            {k: v for k, v in (r["store_record"] or {}).items() if k not in ("listed", "fetched", "repeat")}))
        print("REPORT %s fork: %s; joint-pass phases (probe) %s; store %s; arena memory %s" % (
            name, r["fork"], r["joint_phases_fork"], r["store"], r["mem_arena"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4d: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
