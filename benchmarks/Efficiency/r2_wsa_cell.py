"""WS-A of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell `wa1` of the
TENTATIVE BUFFERS (nosi/nosi/spec_tentative.py; opt-in --spec-rollback tentative) on THE SCHEDULE OF CONTRACT 4 (the
plan's human decisions (a) refill after a reject, (b) drain at the output limit). A NEW script: r2_spec_cell.py,
r2_spec5_cell.py and r2_base_cell.py are not edited (their functions are imported), so every older cell keeps its bytes.
Step script: benchmarks/Efficiency/r2_wsa_step.sh (cells wa1m, wa1a, wa1b).

ONE PROCESS, ONE MODEL, the arms in this order, each with its own shipped prefill, ON THE INPUTS OF sp3a / sp3b (the
same workload, S0, document rows, K 3, 512 steps, journal 256, fused twin and body, graph, redirect, probe 16):
  SH    the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline, run FIRST
  L4P   the LEGACY paper schedule with the journal (spec_round.SpecRunner, EXACT): the arm of sp3a / sp3b, fixed rounds
  L4PJ  the schedule of contract 4 with the JOURNAL (spec_tentative.PaperJournalRunner, EXACT): the proof of the
        schedule; bitwise SH's tokens (MUST); every request commits exactly --steps tokens
  L4PT  the schedule of contract 4 with the TENTATIVE BUFFERS (spec_tentative.TentRunner, FAST): no residency undo
        (D1), no deferred-row recompute (D3); exactly --steps tokens per request
  L3 / L3T (optional, not in the step script's cells): rung L3 with the journal / with the tentative buffers
L4PT against L4PJ is the like-with-like comparison (the same schedule); L4PJ against L4P shows the schedule's change.
Every speculative arm starts from SH's first token and runs FREE (greedy).

MEASURED per arm (CUDA events; device counters read once after the arm):
  committed tok/s      median / mean / effective over the steady rounds (r2_spec_cell.round_rates); against SH's and
                       against the strongest same-memory baseline 3,407.2 tok/s (SCORING_mbstrong.md, the whole-step
                       graph at C 128 B 216; at C 73 the same number, as the plan's table uses it)
  phases per round     L4P / L4PJ: joint, truncate, draft, commit; L4PT: reset, joint, commit, draft (ms medians, steady)
  ROLLBACK per round   BOUNDARIES. L4P / L4PJ: truncate (the event after the joint pass -> the event after rb.truncate:
                       argmax, accept, the undo of the rejected rows) + commit (the event after the D slots -> the event
                       after rb.commit: the undo of every row but the committed ones, the deferred write-back). L4PT:
                       reset (round start -> the twin counters of group / final / target requests back to the committed
                       ones) + commit (the event after the joint pass -> the event after commit and the reject's counter
                       reset: argmax, accept, publication into the committed tail, the write-back of a filled block, the
                       tail id, the lengths). Both include the acceptance work. A drain round's target commits are in the
                       draft phase. ms, and in SH steps of the same job.
  undo / recompute     ACTUAL CALLS (a flag is no proof). L4P / L4PJ: the journal's _undo_step calls (one per layer and
                       undone step), commit / truncate / reject_all calls, the rows remade (L4PJ: the deferred row of every
                       group, the earlier positions of every fill; L4P: one deferred row per request and pass).
                       L4PT: residency-undo calls (expect 0), resets, commits, the deferred rows written by a joint pass
                       (expect 0: device counter), the joint passes and rows (K + 1 per pass).
  bytes per round      L4PT: the published rows (tentative -> committed, logical), the tail-slot bytes of the masked
                       where() ops (physical, MODEL), the write-back of filled blocks, the counter restore. L4P / L4PJ:
                       the journal's saved slots (copied at every tentative step), the host fallback reloads (upper
                       bound), from its counters.
  memory               the allocated GB at the arm's start (model and leftovers), the prefill peak, the DECODE peak
                       (reset after the prefill), and the FITTING BATCH at 84 GB (MODEL: memory above the start grows in
                       proportion to B): floor(B x (84 - start) / (peak - start)), for the decode and the prefill peak.
  tokens               the committed tokens against SH's (L4P, L4PJ MUST bitwise; L4PT REPORT: the FAST accuracy gate is
                       WS-G's); L4PT: the logits fingerprint deltas on the check rounds' committed rows.
  schedule (L4PJ, L4PT) the token accounting (proposed, verified, accepted, corrected, committed by group / final /
                       target, discarded, discarded unverified) and the POSITION TRACE of every request: the first and the
                       last spec_tentative.TRACE_N passes (fed, verified, deferred, drafted positions; accepted and
                       committed counts; remade positions).
CHECK-ONLY WORK: L4PT checks (fingerprints, extended counters) only on the probe rounds (--check-every 0 = the probe
period), which the steady median excludes; the miniature checks every round (--check-every 1). The journal arms count
their undo calls with a host counter around rb._undo_step (about 450 Python calls per round; declared).
MUST GATES: SH-RUN; per arm <arm>-RUN, -ERR; L4P -COMMIT, -TOK, -FP; L4PJ -COMMIT-N, -TOK, -FP; L4PT -COMMIT-N, -D1 (no
residency-undo call, one reset per round), -D3 (no deferred row written by a joint pass, K + 1 rows per joint pass),
-SPAN (the chain stays within K + 2 positions of the committed one, below MAX_ROLLBACK); with --require-stress (wa1m)
-STRESS (L4PT: commits across a block edge, rows past an edge, F4 fill rows, dropped entries, tentative-block skips, draft
skips, a refill, a final group, target rows; L4P / L4PJ: journal overflow and dead slots). <arm>-COMMIT-N: every request
committed exactly --steps tokens and the per-round sums add up.
REGISTERED PREDICTIONS PRED_WSA: TBD by the lead (None gives REPORT); the step script STOPS wa1a / wa1b while
PRED_STAMP_WSA is None.
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_spec_cell as SC          # noqa: E402

BC = SC.BC
ARMS_W = ("L4P", "L4PJ", "L4PT", "L3", "L3T")
ARM_W = dict(L4P=dict(level="L4P", rollback="journal", schedule="legacy"),
             L4PJ=dict(level="L4P", rollback="journal", schedule="contract4"),
             L4PT=dict(level="L4P", rollback="tentative", schedule="contract4"),
             L3=dict(level="L3", rollback="journal", schedule="legacy"),
             L3T=dict(level="L3", rollback="tentative", schedule="legacy"))
TENT_ARMS = ("L4PT", "L3T")
PAPER_ARMS = ("L4PJ", "L4PT")
PHASES_W = {"L3": ("reset", "draft", "reset2", "verify", "commit"), "L4P": ("reset", "joint", "commit", "draft")}
ROLLBACK_W = {"L3": ("reset", "reset2", "commit"), "L4P": ("reset", "commit")}           # tentative arms
ROLLBACK_J = {"L3": ("commit",), "L4P": ("truncate", "commit")}                           # journal arms
# L3 (journal) is a LOWER BOUND: spec_round's reject_all of the drafts falls in its "draft" phase (no event between).
STRONGEST = 3407.2                 # SCORING_mbstrong.md: the whole-step graph, C 128 B 216 (gs1, nova07)
BUDGET_GB = 84.0
# REGISTERED PREDICTIONS, TBD BY THE LEAD: fill every None and PRED_STAMP_WSA (`date -u`, L33) in a commit BEFORE wa1a /
# wa1b (the step script stops while PRED_STAMP_WSA is None).
#   L4PT_rollback_sh   (low, high): median steady (reset + commit) ms of L4PT / SH's steady step ms (the plan: <= 0.1;
#                      FALSIFIER > 0.3)
#   L4PT_ratio         (low, high): L4PT's median steady committed tok/s / SH's (the plan's I1: about 0.41 at C 128,
#                      0.51 at C 73, a model of the legacy schedule)
#   L4PT_vs_L4PJ       (low, high): L4PT's / L4PJ's median steady committed tok/s (the same schedule: D1 + D3 only)
#   L4PT_joint_saving  (low, high): L4PJ's joint ms - L4PT's joint ms, median (D3: the deferred row, measured +93 / +120
#                      in L4P against L4X)
#   L4PT_agree         the lowest fraction of L4PT's committed tokens equal to SH's (REPORT until WS-G's gate)
#   L4PJ_tok           True / False: L4PJ's committed tokens equal SH's, bitwise
#   L4PJ_vs_L4P        (low, high): L4PJ's / L4P's median steady committed tok/s (the schedule's change, EXACT)
# Inputs (MEASURED, SCORING_specp_fix.md section 7): L4P rollback per round (truncate + commit) 317.3 ms = 4.92 SH steps
# at C 128 B 216, 398.8 ms = 3.22 at C 73 B 318; the fifth (deferred) row +93.5 / +120.5 ms.
PRED_STAMP_WSA = None
PRED_WSA = dict(stamp=PRED_STAMP_WSA, L4PT_rollback_sh=None, L4PT_ratio=None, L4PT_vs_L4PJ=None,
                L4PT_joint_saving=None, L4PT_agree=None, L4PJ_tok=None, L4PJ_vs_L4P=None)


def log(msg):
    print("[r2_wsa %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=0, help="K drafts per round (0 = off: the cell refuses)")
    ap.add_argument("--spec-arms", default="L4P", help="a subset of %s, in run order" % ",".join(ARMS_W))
    ap.add_argument("--spec-rollback", default="journal", choices=("journal", "tentative"),
                    help="THE FLAG of WS-A: the tentative arms (%s) run only with 'tentative'" % ",".join(TENT_ARMS))
    ap.add_argument("--spec-rounds", type=int, default=0, help="legacy arms: rounds (0: ceil((steps - 1) / K))")
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-extra", type=int, default=16)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--check-every", type=int, default=0,
                    help="tentative arms: check rounds (0 = the probe rounds, 1 = every round: the miniature)")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    a = ap.parse_args(argv)
    a.spec_arms = [x for x in a.spec_arms.split(",") if x]
    if not a.spec_arms or any(x not in ARMS_W for x in a.spec_arms) or len(set(a.spec_arms)) != len(a.spec_arms):
        ap.error("--spec-arms: distinct names from %s" % (ARMS_W,))
    tent = [x for x in a.spec_arms if x in TENT_ARMS]
    if tent and a.spec_rollback != "tentative":
        ap.error("the arms %s need --spec-rollback tentative" % tent)
    if a.spec_rollback == "tentative" and not tent:
        ap.error("--spec-rollback tentative without a tentative arm (%s)" % (TENT_ARMS,))
    if a.spec != 0 and not 3 <= a.spec <= 18:
        ap.error("--spec K in 3 .. 18 (L4P needs K >= 3)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.spec_rounds < 0 or a.spec_journal_blocks < 1 or a.spec_extra < 2 or a.probe_every < 2 or a.check_every < 0:
        ap.error("--spec-rounds >= 0, --spec-journal-blocks >= 1, --spec-extra >= 2, --probe-every >= 2, "
                 "--check-every >= 0")
    if not 0 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 0 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def fit_batch(B: int, start_gb, peak_gb, budget: float = BUDGET_GB):
    """MODEL: the largest batch whose memory fits the budget when the memory above the arm's start grows in
    proportion to B. None without numbers."""
    if start_gb is None or peak_gb is None or peak_gb <= start_gb:
        return None
    return int(math.floor(B * (budget - start_gb) / (peak_gb - start_gb)))


def rollback_ms(level: str, rollback: str, phases, steady):
    """Per steady round: the sum of the rollback phases (ROLLBACK_W / ROLLBACK_J) -> (median, mean) ms."""
    names = (PHASES_W if rollback == "tentative" else SC.PHASES)[level]
    pick = [names.index(n) for n in (ROLLBACK_W if rollback == "tentative" else ROLLBACK_J)[level]]
    xs = [sum(phases[g][i] for i in pick) for g in steady if g < len(phases) and len(phases[g]) == len(names)]
    return dict(median_ms=_med(xs), mean_ms=(round(statistics.fmean(xs), 3) if xs else None), rounds=len(xs),
                phases=[names[i] for i in pick], lower_bound=(rollback != "tentative" and level == "L3"))


def fp_delta(fp, fp_ok, count, fp_sh, steps: int):
    """The logits fingerprint deltas of the fingerprinted committed rows (fp_ok: the check rounds) against SH's:
    dict(n, equal (bitwise), max_abs_dmax, max_abs_dsum)."""
    import torch
    fm, fs = fp[0].detach().cpu()[:, :steps], fp[1].detach().cpu()[:, :steps]
    sm, ss = fp_sh[0].detach().cpu()[:, :steps], fp_sh[1].detach().cpu()[:, :steps]
    ok = fp_ok.detach().cpu()[:, :steps]
    B = fm.shape[0]
    n = torch.as_tensor(count).cpu().clamp(max=steps)
    chk = ok & (torch.arange(steps).view(1, steps) < n.view(B, 1))
    nc = int(chk.sum())
    if nc == 0:
        return dict(n=0, equal=None, max_abs_dmax=None, max_abs_dsum=None)
    eq = (fm.view(torch.int32) == sm.view(torch.int32)) & (fs.view(torch.int64) == ss.view(torch.int64))
    return dict(n=nc, equal=round(int((eq & chk).sum()) / nc, 6),
                max_abs_dmax=float((fm.double() - sm.double()).abs()[chk].max()),
                max_abs_dsum=float((fs - ss).abs()[chk].max()))


def tent_bytes(tr, rounds: int, B: int, L: int, H: int, D: int, elt: int = 2):
    """Bytes per round of a tentative arm (module docstring): the published rows (logical), the tail-slot bytes of
    the masked where() ops (MODEL: per layer and commit two phases of three where() ops (K, V, bias), each reads the
    buffer and the tail slot and writes the tail slot: 2 x 3 x B x 64 x the row bytes), the write-back of filled
    blocks, the counter restore."""
    row = (2 * H * D + H) * elt                                       # K, V and the bias of one position and layer
    commits, resets = tr["calls"]["commit"], tr["calls"]["reset"]
    r = max(int(rounds), 1)
    return dict(published=round(tr["appended"] * L * row / r, 1),
                tail_where_physical_MODEL=round(commits * L * 2 * 3 * B * 64 * row / r, 1),
                writeback=round(tr["crossed"] * L * 64 * 2 * H * D * elt / r, 1),
                counter_restore=round((resets + commits) * 3 * L * B * 8 / r, 1),
                residency_restore=0, row_bytes=row)


def journal_bytes(j, rounds: int, block_bytes: int):
    """Bytes per round of a journal arm from its counters: the saved slots (copied at every tentative step) and the
    host fallback reloads (an upper bound: only rejected requests reload)."""
    r = max(int(rounds), 1)
    return dict(journal_saved=round(sum(j["saved"]) * block_bytes / r, 1),
                fallback_reload_upper=round(sum(j["overflow"]) * block_bytes / r, 1),
                destroyed_slots_per_round=round(sum(j["destroyed"]) / r, 1))


def schedule_record(rd):
    """The schedule's record (L4PJ, L4PT): the token accounting and the trace (first / last passes per request)."""
    tr = rd.get("trace") or {}
    return dict(tokens_acct=rd.get("tokens_acct"), trace_head=tr.get("head"), trace_tail=tr.get("tail"))


def summarize_w(arm, res, inputs, fp_sh, sh, B: int, steps: int, dims):
    """The arm's record: r2_spec_cell.summarize's fields (journal arms: that function) and the WS-A contract fields."""
    a = ARM_W[arm]
    level, rb = a["level"], a["rollback"]
    rd = res["read"]
    G = rd["rounds"]
    steady = SC.steady_rounds(G, res["probe_every"])
    if rb == "journal":
        rec = SC.summarize(level, res, inputs, fp_sh, sh, B, steps, dims)
        rec["arm"] = arm
    else:
        rates = SC.round_rates(rd["round_committed"], rd["round_ms"], steady, rd["wall_s"], sum(rd["count"]))
        shr = SC.sh_rates(B, sh) if sh else None
        vs_sh = {k: (round(rates[k] / shr[k], 4) if (shr and shr.get(k) and rates.get(k)) else None)
                 for k in ("median_tok_s", "mean_tok_s", "effective_tok_s")}
        acc = dict(rate=(rd["accepted"] / rd["valid"] if rd["valid"] else None), accepted=rd["accepted"],
                   valid=rd["valid"], hist=rd["acc_hist"], full_fraction=(rd["full"] / (B * G) if G else None),
                   committed_per_request_round=(sum(rd["round_committed"]) / (B * G) if G else None))
        names = PHASES_W[level]
        ph = {nm: _med([rd["phases"][g][i] for g in steady if g < len(rd["phases"])]) for i, nm in enumerate(names)}
        tr = rd["tent"]
        if level == "L4P":
            vh = rd["verify_hist"]
            verify_rows = (sum(i * c for i, c in enumerate(vh)) / sum(vh)) if sum(vh) else res["K"]
        else:
            verify_rows = res["K"]
        streams = tr["draft_rows"] * dims["L"] * dims["H"]
        rec = dict(arm=arm, level=level, K=res["K"], rounds=G, rates=rates, vs_sh=vs_sh, acceptance=acc,
                   phases_median=ph,
                   kv_at_verify=SC.kv_at_verify(rd, steady, verify_rows, dims["L"], dims["H"], B, dims["topk"],
                                                dims["block_bytes"]),
                   tokens=SC.token_check(res["out"], rd["count"], inputs, steps),
                   fingerprints=fp_delta(res["fp"], res["fp_ok"], rd["count"], fp_sh, steps),
                   draft=dict(skip_per_stream_per_call=(sum(tr["skipped"]) / streams if streams else None),
                              rows=tr["draft_rows"], steps=rd["draft_steps"]),
                   tent=tr, prelude_ms=rd["prelude_ms"], wall_s=rd["wall_s"], count_min=min(rd["count"]),
                   count_max=max(rd["count"]), errors=rd["errors"], prefill_s=res["prefill_s"],
                   first_token_agreement=res["first_token_agreement"], meta=res["meta"])
        if level == "L4P":
            ra = [rd["round_accepted"][g] for g in steady if g < len(rd["round_accepted"])]
            rec["paper"] = dict(verify_rows_hist=rd["verify_hist"], accepted_per_round=rd["round_accepted"],
                                accepted_per_round_steady_mean=(round(statistics.fmean(ra), 3) if ra else None),
                                accepted_per_request_round=(rd["accepted"] / (B * G) if G else None))
    rec["rollback"], rec["schedule"] = rb, a["schedule"]
    rbm = rollback_ms(level, rb, rd["phases"], steady)
    sh_ms = sh["rates"]["steady_step_ms"] if sh else None
    rbm["sh_steps"] = round(rbm["median_ms"] / sh_ms, 4) if (rbm["median_ms"] is not None and sh_ms) else None
    rec["rollback_per_round"] = rbm
    rec["vs_strongest"] = (round(rec["rates"]["median_tok_s"] / STRONGEST, 4) if rec["rates"]["median_tok_s"]
                           else None)
    if arm in PAPER_ARMS:
        rec["schedule_record"] = schedule_record(rd)
    acct = (rd.get("tokens_acct") or {})
    if rb == "journal":
        u = res.get("undo") or {}
        remade = acct.get("remade_rows") if arm in PAPER_ARMS else (sum(rd["verify_hist"]) if level == "L4P" else 0)
        rec["calls"] = dict(undo_step_calls=u.get("undo_steps"), commit_calls=u.get("commits"),
                            truncate_calls=u.get("truncates"), reject_all_calls=u.get("rejects"),
                            remade_rows=remade, residency_undo="every undone step (the journal undoes the residency)")
        rec["bytes_per_round"] = journal_bytes(rd["journal"], G, dims["block_bytes"])
    else:
        tr = rd["tent"]
        rec["calls"] = dict(residency_undo_calls=tr["calls"]["residency_undo"], reset_calls=tr["calls"]["reset"],
                            commit_calls=tr["calls"]["commit"], deferred_recompute_rows=tr["deferred_writes"],
                            remade_rows=acct.get("remade_rows", 0), joint_passes=tr["calls"]["joint_passes"],
                            joint_rows=tr["calls"]["joint_rows"], j1_passes=tr["calls"]["j1_passes"],
                            replays=tr["replays"], undo_step_calls=0)
        rec["bytes_per_round"] = tent_bytes(tr, G, B, dims["L"], dims["H"], dims["D"])
    m = res.get("mem") or {}
    rec["memory"] = dict(m, fit_batch_decode_MODEL=fit_batch(B, m.get("start_GB"), m.get("decode_peak_GB")),
                         fit_batch_prefill_MODEL=fit_batch(B, m.get("start_GB"), m.get("prefill_peak_GB")),
                         budget_GB=BUDGET_GB)
    return rec


def commit_n_gate(arm, res, steps: int):
    """<arm>-COMMIT-N (the schedule of contract 4): every request committed exactly `steps` tokens; the warm tokens
    plus the per-round sums equal the total."""
    rd = res["read"]
    cnt = rd["count"]
    warm = rd["warm"] * len(cnt)
    ok = min(cnt) == max(cnt) == steps and sum(rd["round_committed"]) + warm == sum(cnt)
    return ("%s-COMMIT-N" % arm, True, ok, "committed per request min %d max %d (exactly %d wanted); warm %d + per-round "
            "sum %d, total %d; %d rounds" % (min(cnt), max(cnt), steps, warm, sum(rd["round_committed"]), sum(cnt),
                                             rd["rounds"]))


def tent_gates(arm, res, rec, require_stress: bool = False):
    """The WS-A MUST gates of a tentative arm (module docstring)."""
    rd = res["read"]
    tr = rd["tent"]
    K = res["K"]
    c = tr["calls"]
    g = []
    g.append(("%s-D1" % arm, True, c["residency_undo"] == 0 and c["reset"] >= rd["rounds"],
              "residency-undo calls %d (0 expected); resets %d for %d rounds" % (c["residency_undo"], c["reset"],
                                                                               rd["rounds"])))
    if ARM_W[arm]["level"] == "L4P":
        g.append(("%s-D3" % arm, True, tr["deferred_writes"] == 0 and c["joint_rows"] == (K + 1) * c["joint_passes"]
                  and c["joint_passes"] + c["j1_passes"] == rd["rounds"],
                  "deferred rows written by a joint pass %d (0 expected); joint rows %d = (K + 1) x %d passes; one-row "
                  "J slots %d; %d rounds" % (tr["deferred_writes"], c["joint_rows"], c["joint_passes"],
                                             c["j1_passes"], rd["rounds"])))
    lim = K + 2
    g.append(("%s-SPAN" % arm, True, 0 < tr["span_max"] <= lim,
              "the chain's largest span past the committed position %d (at most K + 2 = %d < MAX_ROLLBACK 32)" % (
                  tr["span_max"], lim)))
    if require_stress:
        acct = rd.get("tokens_acct") or {}
        ok = (tr["crossed"] > 0 and tr["edge_rows"] > 0 and tr["fill_rows"] > 0 and tr["dropped"] > 0
              and sum(tr["forced"]) > 0 and sum(tr["skipped"]) > 0)
        if ARM_W[arm]["level"] == "L4P":
            ok = ok and acct.get("corrected", 0) > 0 and acct.get("fills", 0) > 0 and acct.get("finals", 0) > 0
        g.append(("%s-STRESS" % arm, True, ok, "commits across an edge %d, rows past an edge %d, F4 fill rows %d, "
                  "dropped entries %d, tentative-block skips %d, draft skips %d; corrections %s, refills %s, final "
                  "groups %s (target rows %s, reported) (each > 0; counted on check rounds)" % (
                      tr["crossed"], tr["edge_rows"], tr["fill_rows"], tr["dropped"], sum(tr["forced"]),
                      sum(tr["skipped"]), acct.get("corrected"), acct.get("fills"), acct.get("finals"),
                      acct.get("targets"))))
    return g


def arm_gates_w(arm, res, rec, steps: int, require_stress: bool = False):
    """[(name, must, passed, detail)]: r2_spec_cell.arm_gates (ERR, COMMIT; EXACT journal arms TOK and FP; the
    contract-4 arms COMMIT-N in place of COMMIT), then the WS-A gates of a tentative arm or the journal stress gate."""
    a = ARM_W[arm]
    paper = arm in PAPER_ARMS
    if a["rollback"] == "journal":
        g = [(n.replace(a["level"] + "-", arm + "-", 1), m, ok, d)
             for n, m, ok, d in SC.arm_gates(a["level"], res, rec)]
    else:
        g = list(SC.arm_gates(arm, res, None))
    if paper:
        g = [x for x in g if not x[0].endswith("-COMMIT")] + [commit_n_gate(arm, res, steps)]
    if a["rollback"] == "journal":
        if require_stress:
            j = res["read"]["journal"]
            g.append(("%s-STRESS" % arm, True, sum(j["overflow"]) > 0 and sum(j["overflow_dead"]) > 0,
                      "journal overflow %d, overflow_dead %d" % (sum(j["overflow"]), sum(j["overflow_dead"]))))
        return g
    return g + tent_gates(arm, res, rec, require_stress)


def verdicts(sh, sp, B: int):
    """PRED_WSA against the arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN without the arm."""
    P = PRED_WSA
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
    t, j, l4p = (sp or {}).get("L4PT"), (sp or {}).get("L4PJ"), (sp or {}).get("L4P")
    if j is not None:
        tc = j["tokens"]
        want = P["L4PJ_tok"]
        got = tc["n_checked"] > 0 and tc["n_unequal"] == 0
        out.append(("L4PJ-TOK", "REPORT" if want is None else ("CONFIRMED" if got == bool(want) else "REFUTED"),
                    "%d of %d unequal (first %s); registered %s" % (tc["n_unequal"], tc["n_checked"], tc["first"],
                                                                     want)))
        band("L4PJ-VS-L4P", (med(j) / med(l4p)) if (med(j) and med(l4p)) else None, P["L4PJ_vs_L4P"])
    if t is None:
        out.append(("L4PT-RUN", "NOT_RUN", "the arm did not run"))
        return out
    band("L4PT-ROLLBACK-SH", t["rollback_per_round"]["sh_steps"], P["L4PT_rollback_sh"])
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    band("L4PT-RATIO", (med(t) / shm) if (med(t) and shm) else None, P["L4PT_ratio"])
    band("L4PT-VS-L4PJ", (med(t) / med(j)) if (med(t) and med(j)) else None, P["L4PT_vs_L4PJ"])
    jj = (j or {}).get("phases_median", {}).get("joint")
    tj = t["phases_median"].get("joint")
    band("L4PT-JOINT-SAVING", (jj - tj) if (jj is not None and tj is not None) else None, P["L4PT_joint_saving"])
    tc = t["tokens"]
    want = P["L4PT_agree"]
    out.append(("L4PT-AGREE", "REPORT" if want is None else ("CONFIRMED" if tc["equal_fraction"] >= want else
                                                            "REFUTED"),
                "committed tokens equal to SH's: %.6f (first difference %s); registered >= %s" % (
                    tc["equal_fraction"], tc["first"], want)))
    out.append(("L4PT-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s (the plan's I1: 0.41 / 0.39)" % (
        STRONGEST, t["vs_strongest"])))
    return out


# ------------------------------------------------------------------ one arm (GPU)
def _count_undo(rb):
    """Host counters around the journal's undo (the actual calls; declared in the module docstring)."""
    box = dict(undo_steps=0, commits=0, truncates=0, rejects=0)

    def wrap(name, key):
        orig = getattr(rb, name)

        def f(*a, **k):
            box[key] += 1
            return orig(*a, **k)
        setattr(rb, name, f)
    wrap("_undo_step", "undo_steps")
    wrap("commit", "commits")
    wrap("truncate", "truncates")
    wrap("reject_all", "rejects")
    return box


def run_arm(model, ids, inputs, steps: int, clock, opt):
    """ONE speculative arm (r2_spec_cell.run_spec's steps) with the memory measures and, for a journal arm, the undo
    counters. Tentative arms: spec_tentative.TentRunner (check rounds: opt["check_every"], 0 = the probe rounds); the
    contract-4 journal arm: spec_tentative.PaperJournalRunner; the legacy arms: spec_round.SpecRunner (fixed rounds).
    The contract-4 arms run until every request has exactly `steps` tokens."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    cfg = model.config
    model.has_buffers = False
    for name in BC.MODEL_WARM_BUFFERS:
        if hasattr(model, name):
            setattr(model, name, None)
    for layer in model.layers:
        for name in BC.LAYER_WARM_STATE:
            if hasattr(layer, name):
                setattr(layer, name, None)
    gc.collect()
    torch.cuda.empty_cache()
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
    a = ARM_W[opt["arm"]]
    sops = SR.SpecOps(level=a["level"], K=K, mode="exact", gemm="rowgroup", extra=opt["extra"], journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"])
    tent = a["rollback"] == "tentative"
    paper = a["schedule"] == "contract4"
    undo = None
    if tent:
        from nosi import spec_tentative as ST
        ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
        runner = ST.TentRunner(model, rs, sops, clock, t0=inputs[:, 0], hooks=None, check_every=ce)
    elif paper:
        from nosi import spec_tentative as ST
        runner = ST.PaperJournalRunner(model, rs, sops, clock, t0=inputs[:, 0])
        undo = _count_undo(runner.rb)
    else:
        runner = SR.SpecRunner(model, rs, sops, clock, t0=inputs[:, 0])
        undo = _count_undo(runner.rb)
    if paper:
        runner.run(steps)
    else:
        runner.run(steps, rounds=SC.rounds_for(steps, K, opt["rounds"]))
    rd = runner.read()
    rd.pop("trace_all", None)
    clock.sync()
    mem = dict(start_GB=round(start, 3), prefill_peak_GB=round(prefill_peak, 3),
               decode_peak_GB=round(torch.cuda.max_memory_allocated() / gb, 3),
               decode_peak_reserved_GB=round(torch.cuda.max_memory_reserved() / gb, 3))
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()),
               fp_ok=(runner.fp_ok.clone() if tent else None), K=K, probe_every=opt["probe_every"],
               prefill_s=round(prefill_s, 1), first_token_agreement=first_agree,
               meta=dict(rs.meta, spec=runner.meta), undo=(dict(undo) if undo is not None else None), mem=mem)
    runner.detach()
    del runner, rs, cache
    return res


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.spec == 0:
        print("REFUSED: --spec K is the flag of this cell (0 = off)", file=sys.stderr)
        return 20
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
    gates, sh, sp, inputs = [], None, {}, None
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    mem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
    dims = dict(L=len(model.layers), H=H, D=D, topk=64, block_bytes=64 * D * 2 * 2 + 64 * 2)
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
                                  fit_batch_MODEL=fit_batch(a.batch, sh_start, pa), budget_GB=BUDGET_GB,
                                  note="the peak includes SH's prefill (run_shipped resets nothing in between)"))
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            del out
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        mem["after_SH"] = BC.host_mem()
        BC.free()
        for arm in a.spec_arms:
            if inputs is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: SH failed (the arm starts from SH's first token)"))
                continue
            BC.free(host=True)
            opt = dict(arm=arm, K=a.spec, rounds=a.spec_rounds, journal=a.spec_journal_blocks, extra=a.spec_extra,
                       twin=a.spec_twin, body=a.spec_body, graph=a.spec_graph == "on",
                       redirect=a.spec_redirect == "on", probe_every=a.probe_every, check_every=a.check_every)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm(model, ids, inputs, a.steps, clock, opt)
                pa, pr = BC.gpu_peak()
                rec = summarize_w(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend(arm_gates_w(arm, res, rec, a.steps, require_stress=bool(a.require_stress)))
                del res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, spec_arms=a.spec_arms, spec_rollback=a.spec_rollback,
                spec_rounds=a.spec_rounds, spec_journal_blocks=a.spec_journal_blocks, spec_extra=a.spec_extra,
                spec_twin=a.spec_twin, spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=PRED_WSA, strongest_tok_s=STRONGEST, dims=dims, host_mem=mem,
                base_commit=os.environ.get("BASE_COMMIT"))
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-18s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB, memory %s" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"], sh["memory"]))
    for name in a.spec_arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s (%s, %s) K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; vs %.1f "
              "%s; acceptance %s (full %s); phases %s; ROLLBACK per round %s; calls %s; bytes per round %s; tokens "
              "equal %.6f (first %s); fingerprints %s; memory %s" % (
                  name, r["rollback"], r["schedule"], r["K"], r["rounds"], r["rates"]["median_tok_s"],
                  r["rates"]["mean_tok_s"], r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["acceptance"]["rate"], r["acceptance"]["full_fraction"], r["phases_median"],
                  r["rollback_per_round"], r["calls"], r["bytes_per_round"], r["tokens"]["equal_fraction"],
                  r["tokens"]["first"], r["fingerprints"], r["memory"]))
        sr = r.get("schedule_record")
        if sr:
            print("REPORT %s schedule: tokens %s; request 0 trace head %s; tail %s" % (
                name, sr["tokens_acct"], (sr["trace_head"] or [[]])[0], (sr["trace_tail"] or [[]])[0]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_wsa: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
