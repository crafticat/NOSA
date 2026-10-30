"""WS-M (KS/analysis/plan/PLAN_L4P_faithful.md, contract 4; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the MATCHED-MEMORY
cell. It measures ONE speculative arm (default L4P) at ITS OWN feasible configuration under the physical cap of the
reviewer's verdict, 84.0 GB peak allocated on the A100 80GB, and forms the verdict

    MAIN = the arm's median steady committed tok/s at (C, B_fit)
           / the strongest tuned cached NOSA at ITS feasible configuration on the SAME GPU
             (the whole-step graph at C 128 B 216: REG_WSM["strongest_steady"][GPU uuid], registered by the lead)

The same-job SH runs at the arm's (C, B): it is the token reference of the EXACT gates. Its ratio is printed as
DIAGNOSTIC (same config) only. A NEW script: r2_spec_cell.py and r2_base_cell.py are not edited (their functions are
imported). The TIMED path is theirs, byte for byte: BC.run_shipped for SH and SC.run_spec for the arm
(tests_cpu/test_wsm.py test_timed_path_unchanged). Step script: benchmarks/Efficiency/r2_wsm_step.sh. Notes:
docs/NOTES_wsm.md.

TWO MODES (--wsm-mode)
  probe  SH (the token reference), then the arm with probe_spec: a COPY of r2_spec_cell.run_spec plus memory marks only
         (tests_cpu/test_wsm.py test_probe_spec_is_run_spec), for --spec-rounds rounds (default 2 x probe_every + 1 =
         33) at the full run's horizon (--steps 512: the twin, the journal and the joint-pass buffers have the full
         run's shapes). It splits the arm's peak into PREFILL (batch_prefill) and DECODE (attach, every round, the end),
         checks that the probe passed every allocation event of the full run (probe_check, below) and fits the decode
         line of this C through the probe point and the MEASURED full-run anchor (ANCHORS): it prints B_fit.
  match  SH at (C, B), then the arm at (C, B) with SC.run_spec (the timed path): the full run (171 rounds of K 3 for 512
         steps). The arm's peak allocated covers the whole arm: BC.free resets the peak before the arm
         (r2_base_cell.py:1075) and the cell reads it after run_spec returns (as r2_spec_cell.py:508-516).

WHERE THE PEAK IS (the probe's argument, lesson L53; CODE at 40ac348; docs/NOTES_wsm.md section 2 has the details)
  * Preallocated before round 0, never grown: the GPU slots and the victim pool (cache_engine.py:222, :239, at the
    prefill), the twin (ragged_engine.py:217; r2_positional_twin.py:147-151; widths from the horizon,
    r2_decode_path.py:236-240), the pooling graphs (captured once, ragged_engine.py:360-368), the rollback journal and
    the staging blocks (ragged_rollback.py:304-305), the draft scratch (spec_round.py:202-217), the joint-pass buffers
    (spec_round.py:603 -> verify_rows.py:161-193), the output buffers (spec_round.py:997-999).
  * Every round has the same shapes (R = K + 2 rows, a fixed journal N, fixed-shape undo: ragged_rollback.py:547-560).
    The largest transient of a round is the joint pass's logits: R row-group GEMM outputs, their stack and the float
    copy (verify_rows.py:230-231, :578): 6 x (R B) x V bytes; `lg` stays alive to the end of the round (spec_round.py:897
    .. :956), so the K - 1 drafts after it add less than it. So round 0 reaches the decode peak.
  * The only growth per round: three 0-d device tensors (spec_round.py:747, :947, :948): 3 x 512 B = 1,536 B per round.
  * Code paths that start later: the first probe round (round probe_every; CUDA events only, spec_round.py:409-416,
    :1022) and the first block fill (count >= 64 - S0 % 64; the staging copy ragged_rollback.py:829-831, a
    (B, 64, H, D) temporary inside a layer). The probe runs 2 x probe_every + 1 = 33 rounds, so it passes two probe
    rounds and (gate PR-FILL) every request's first fill; then probe_check requires a flat peak over the last
    probe_every rounds and a growth of the allocated bytes of at most GROWTH_TOL_B per round.
  * A growth that starts after the probe cannot be seen by any probe: the matched cell's MUST gate <arm>-CAP reads the
    peak of the FULL run.

MUST GATES (both modes): SH-RUN, <arm>-RUN, <arm>-ERR, <arm>-COMMIT, <arm>-TOK and <arm>-FP for an EXACT arm (r2_spec_cell
arm_gates, unchanged). probe: PR-ROUNDS, PR-FILL, PR-FLAT. match: <arm>-FULL, <arm>-CAP; role mini also PLANT-CAP and
PLANT-FULL (the gates catch planted values on this run's numbers); role matched also REG, FIT-B, GPU and EDGE.
REGISTERED: PRED_STAMP_WSM, FIT_B_73, FIT_B_128 and REG_WSM, TBD by the lead (a None prediction gives REPORT; the step
script STOPS wm73 / wm128 while PRED_STAMP_WSM or their FIT_B is None).
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_spec_cell as SC          # noqa: E402

BC = SC.BC
ARM_OPS = SC.ARM_OPS               # the names that probe_spec shares with r2_spec_cell.run_spec (the AST check)
rounds_for = SC.rounds_for

# ------------------------------------------------------------------ REGISTERED (TBD BY THE LEAD)
# Fill every None and PRED_STAMP_WSM (`date -u` in the same command, L33 / L52) in a commit BEFORE wm73 / wm128: the step
# script STOPS them while PRED_STAMP_WSM or their FIT_B_<C> is None. FIT_B_<C> comes from the probe cell wp<C> (its line
# "FIT ..."). The builder's INFERRED values, with the MEASURED inputs and the arithmetic: docs/NOTES_wsm.md section 3.
# The lead's REGISTRATION (2026-10-28T12:47:21Z, `date -u`, after the probes wp73 / wp128 and BEFORE wm73 / wm128).
# MEASURED inputs (the probes, job 582_exm_wsm_n5 on nova05, records r2-runs/spec/wsm_m173_2195573_wp73.step/stdout.txt:49
# and wsm_m1128_2195573_wp128.step/stdout.txt:49): decode lines 18.367612 + 0.1838063 B GB (C 73) and 18.367833 +
# 0.3000887 B GB (C 128); zero-margin edges B 357 (83.986 GB) and B 218 (83.787 GB). Margin 0.25 GB (NOTES_wsm.md risk 1:
# at C 73 B 357 the device total, peak + 0.693 reserved gap + 0.56 context, is about 85.24 GB > 85.09 GB) gives B 355
# (predicted 83.619 GB) and B 217 (83.487 GB). Strongest baselines (CHECKPOINTS.md header): steady 3,426.2 (nova05) /
# 3,410.1 (nova07); effective under rule E1 nova07 2,778.3, nova05 unknown (no second-in-process baseline). Predictions:
# peak = the line's value -0.8 / +0.4 GB (the cap gate is the MUST); MAIN from I0 (L4P steady 737.1 / 734.4 at C 73 B 318,
# 699.0 / 708.3 at C 128 B 216): B 355 adds 11.6 % requests for a round about 8-11 % longer (MODEL, HBM-bound), so MAIN
# about 0.215-0.222 at C 73 and 0.204-0.208 at C 128; the bands are the builder's INFERRED ones. probe_decode_GB stays
# None: the probes ran before this registration, so no prediction exists for them.
PRED_STAMP_WSM = "2026-10-28T12:47:21Z"
FIT_B_73 = 355
FIT_B_128 = 217
UUID_NOVA05 = "f31fc723-9dfe-407a-9947-28173556232a"      # HOLDERS/2195573.gpu_uuid
UUID_NOVA07 = "33cfb929-315c-8d27-5622-1169affb7e04"      # HOLDERS/2195574.gpu_uuid
REG_WSM = dict(
    margin_GB=0.25,                                   # fit: the predicted peak + margin <= BUDGET_GB
    edge_requests=4,                                  # EDGE: BUDGET_GB - the measured peak < edge_requests x slope
    slope_GB={"73": 0.1838063, "128": 0.3000887},     # GB per request of the decode line (the probe's FIT line)
    strongest_steady={UUID_NOVA05: 3426.2, UUID_NOVA07: 3410.1},   # tok/s: the denominator of MAIN
    strongest_effective={UUID_NOVA05: None, UUID_NOVA07: 2778.3},  # tok/s: the denominator of MAIN-EFF (REPORT)
    probe_decode_GB={"73_64": None, "128_64": None},  # (low, high): the probe's decode peak at C_B (PR-DEC)
    peak_GB={"73": (82.8, 84.0), "128": (82.7, 84.0)},  # (low, high): the matched arm's peak allocated (WSM-PEAK)
    main_ratio={"73": (0.19, 0.25), "128": (0.18, 0.23)},  # (low, high): MAIN (WSM-MAIN)
)

# ------------------------------------------------------------------ constants (MEASURED / CODE; not registrations)
BUDGET_GB = 84.0                   # the reviewer's cap: peak allocated (SCORING_mbstrong.md section 5, reading 1)
INT32 = 2 ** 31
KV_HEADS, HEAD_DIM, BLOCK = 2, 128, 64              # NOSA-8B config.json (as nosi-r2-mbstrong cd3e834 r2_base_cell.py:110)
# The MEASURED full-run anchors: the arm's peak allocated over the whole arm (prefill .. end), above the prefill model at
# both points, so it is the DECODE peak. reserved: the peak reserved of the same arm (the device check).
ANCHORS = {
    ("L4P", 128): dict(B=216, peak_GB=83.187, reserved_GB=83.647,
                       src="I0a sp_y2a_2195574_sp3a (a6c9d10) = I0c sp_a1a_2195573_sp5a (40ac348); "
                           "SCORING_specp_fix.md:324, SCORING_sp5.md:146"),
    ("L4P", 73): dict(B=318, peak_GB=76.818, reserved_GB=77.511,
                      src="I0b sp_y2b_2195574_sp3b (a6c9d10) = I0d sp_a1b_2195573_sp5b (40ac348); "
                          "SCORING_specp_fix.md:324, SCORING_sp5.md:142"),
}
# SH (the in-job reference) at the same C: MEASURED full-run fits (SCORING_mbstrong.md:106-110, :180)
SH_MEASURED = {73: dict(fits=358, oom=369), 128: dict(fits=222, oom=224)}
DEVICE_GB, CONTEXT_GB = 85.09, 0.56                 # SCORING_mbstrong.md:157-158 (79.25 GiB; the C 73 B 369 OOM)
GROWTH_DECLARED_B = 3 * 512        # spec_round.py:747, :947, :948: three 0-d device tensors per round (512 B blocks)
GROWTH_TOL_B = 8192                # probe_check: allocated bytes per round over the flat window, at most
FLAT_TOL_GB = 0.016                # probe_check: a late round may not raise the peak by more
MODES = ("probe", "match")
ROLES = ("probe", "mini", "matched")


def log(msg):
    print("[r2_wsm %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=0, help="K drafts per round (0 = off: the cell refuses)")
    ap.add_argument("--spec-arm", default="L4P", choices=SC.ARMS, help="the ONE speculative arm (r2_spec_cell.ARMS)")
    ap.add_argument("--spec-rounds", type=int, default=0,
                    help="probe: the probe's rounds (0: 2 x probe_every + 1); match: must be 0 (the full run)")
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-extra", type=int, default=16)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--spec-pin", default="off", choices=("off", "on"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--wsm-mode", required=True, choices=MODES)
    ap.add_argument("--wsm-role", required=True, choices=ROLES)
    ap.add_argument("--gpu-uuid", default="", help="the holder's GPU uuid (HOLDERS/<job>.gpu_uuid): the denominator key")
    ap.add_argument("--fit-point", action="append", default=[],
                    help="probe: B:GB, a MEASURED decode peak of the same arm and C (more points for the line)")
    a = ap.parse_args(argv)
    if a.spec != 0 and not 2 <= a.spec <= 18:
        ap.error("--spec K in 2 .. 18")
    if a.spec != 0 and a.spec_arm == "L4P" and a.spec < 3:
        ap.error("--spec-arm L4P needs --spec K >= 3")
    if a.spec_pin == "on" and a.spec_arm != "L4P":
        ap.error("--spec-pin on acts on the L4P arm only")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.spec_rounds < 0 or a.spec_journal_blocks < 1 or a.spec_extra < 2 or a.probe_every < 2:
        ap.error("--spec-rounds >= 0, --spec-journal-blocks >= 1, --spec-extra >= 2, --probe-every >= 2")
    if not 0 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 0 .. %d" % BC.POOL_MAX)
    if (a.wsm_mode == "probe") != (a.wsm_role == "probe"):
        ap.error("--wsm-mode probe goes with --wsm-role probe only (match: mini or matched)")
    if a.wsm_mode == "match" and a.spec_rounds != 0:
        ap.error("--wsm-mode match runs the full run: --spec-rounds must be 0")
    if a.wsm_role == "matched" and a.steps != 512:
        ap.error("--wsm-role matched runs the workload's full decode: --steps 512")
    if a.wsm_mode == "probe" and a.spec_rounds == 0:
        a.spec_rounds = 2 * a.probe_every + 1
    pts = []
    for s in a.fit_point:
        try:
            b, g = s.split(":")
            pts.append((int(b), float(g)))
        except ValueError:
            ap.error("--fit-point B:GB (got %r)" % s)
    a.fit_point = pts
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def spec_opt(a) -> dict:
    """The arm's options, as r2_spec_cell.main builds them (r2_spec_cell.py:509-512; test_wsm checks the keys)."""
    return dict(arm=a.spec_arm, K=a.spec, rounds=a.spec_rounds, journal=a.spec_journal_blocks, extra=a.spec_extra,
                twin=a.spec_twin, body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                probe_every=a.probe_every, pin=(a.spec_arm == "L4P" and a.spec_pin == "on"))


def int32_batch_limit(S: int, steps: int, C: int) -> int:
    """The largest B of the int32 envelope of the pooled engine (cache_engine.py:208-221). A COPY of nosi-r2-mbstrong
    cd3e834 r2_base_cell.py:363-372: host B x (S + steps + 64) x H x D < 2^31; GPU B x (C + 1) x 64 x H x D < 2^31."""
    hd = KV_HEADS * HEAD_DIM
    host = (INT32 - 1) // ((int(S) + int(steps) + 64) * hd)
    gpu = (INT32 - 1) // ((int(C) + 1) * BLOCK * hd)
    return min(host, gpu)


def fit_batch(peak_fn, budget_GB: float, margin_GB: float = 0.0, b_max: int = 4096) -> int:
    """The largest B in 1 .. b_max with peak_fn(B) + margin_GB <= budget_GB (0: none fits): the fit of nosi-r2-mbstrong
    cd3e834 (r2_base_cell.fit_batch, r2_base_cell.py:375-387) for any peak model. peak_fn must not decrease with B
    (checked on the scanned range)."""
    best, prev = 0, None
    for B in range(1, int(b_max) + 1):
        p = float(peak_fn(B))
        if prev is not None and p < prev - 1e-9:
            raise ValueError("fit_batch: the peak model decreases at B %d (%.6f < %.6f)" % (B, p, prev))
        prev = p
        if p + float(margin_GB) <= float(budget_GB):
            best = B
        else:
            break
    return best


def fit_line(points):
    """Least squares through [(B, GB)], at least two distinct B: (a, b) with GB = a + b B (two points: the exact line)."""
    pts = [(float(x), float(y)) for x, y in points]
    if len({x for x, _ in pts}) < 2:
        raise ValueError("fit_line: needs two distinct batch sizes, got %s" % (pts,))
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    b = sxy / sxx
    return my - b * mx, b


def arm_peak(C: int, B: int, line) -> float:
    """The arm's peak model (GB allocated): max(the prefill model BC.mbase_peak_model (MEASURED fit, 0.07 %), the
    decode line a + b B)."""
    a, b = line
    return max(BC.mbase_peak_model(C, B), a + b * B)


def device_check(alloc_GB, gap_GB, capacity_GB: float = DEVICE_GB, context_GB: float = CONTEXT_GB):
    """REPORT, not the budget: the peak allocated + the measured reserved - allocated gap + the CUDA context against the
    device (SCORING_mbstrong.md section 5: the C 73 B 369 OOM). None when a value is missing."""
    if alloc_GB is None or gap_GB is None:
        return None
    need = float(alloc_GB) + float(gap_GB) + float(context_GB)
    return dict(need_GB=round(need, 3), capacity_GB=capacity_GB, gap_GB=round(float(gap_GB), 3),
                fits=need <= capacity_GB)


def fit_arm(C: int, line, margin_GB: float = 0.0, budget_GB: float = BUDGET_GB, steps: int = 512, S: int = None):
    """B_fit at capacity C for the decode line `line` (arm_peak), capped by the int32 envelope."""
    S = BC.S0 if S is None else int(S)
    lim = int32_batch_limit(S, steps, C)
    B = fit_batch(lambda x: arm_peak(C, x, line), budget_GB, margin_GB, b_max=lim)
    return dict(C=int(C), B_fit=B, line=(round(line[0], 6), round(line[1], 7)), margin_GB=float(margin_GB),
                budget_GB=float(budget_GB), int32_limit=lim,
                peak_at_fit=(round(arm_peak(C, B, line), 4) if B else None),
                peak_next=round(arm_peak(C, B + 1, line), 4),
                prefill_at_fit=(round(BC.mbase_peak_model(C, B), 4) if B else None))


def inferred_cross_c(arm: str = "L4P"):
    """INFERENCE, NOT a measurement (docs/NOTES_wsm.md section 3): ONE decode line for both C,
    dec(C, B) = a + B x (s0 + (C - 63) x k), k = MBASE_PEAK slope_per_block_GB (the KV bytes of one slot per request:
    32 layers x 64 rows x (2 x 2 x 128 x 2 B + 2 x 2 B) = 2.105 MB), through the two MEASURED anchors of `arm`."""
    k = BC.MBASE_PEAK["slope_per_block_GB"]
    (C1, A1), (C2, A2) = sorted((C, d) for (r, C), d in ANCHORS.items() if r == arm)
    y1 = A1["peak_GB"] - A1["B"] * (C1 - BC.C_BASE) * k
    y2 = A2["peak_GB"] - A2["B"] * (C2 - BC.C_BASE) * k
    s0 = (y2 - y1) / (A2["B"] - A1["B"])
    a = y1 - s0 * A1["B"]
    return dict(a=a, s0=s0, k=k, lines={C: (a, s0 + (C - BC.C_BASE) * k) for C in (C1, C2)})


def probe_record(marks, trace):
    """CudaMem's marks [(name, peak, current, peak reserved, reserved)] and trace [(round, ...the same)] (bytes) -> GB:
    prefill (start .. prefill), attach (prefill .. attach), rounds (the largest per-round peak; round 0 includes the
    runner's allocations, the warm step and the prelude), end (last round .. end), decode = max(attach, rounds, end),
    arm = max(prefill, decode); the per-round series."""
    m = {x[0]: x for x in marks}
    g = lambda v: round(v / 1e9, 4)                          # noqa: E731
    peaks = [t[1] for t in trace]
    dec = max([m["attach"][1], m["end"][1]] + peaks)
    return dict(prefill_GB=g(m["prefill"][1]), prefill_reserved_GB=g(m["prefill"][3]), attach_GB=g(m["attach"][1]),
                rounds_GB=(g(max(peaks)) if peaks else None), end_GB=g(m["end"][1]), decode_GB=g(dec),
                decode_reserved_GB=g(max([m["attach"][3], m["end"][3]] + [t[3] for t in trace])),
                arm_GB=g(max(dec, m["prefill"][1])), after_prefill_GB=g(m["prefill"][2]),
                after_attach_GB=g(m["attach"][2]), rounds=[t[0] for t in trace], round_peak_B=peaks,
                round_current_B=[t[2] for t in trace],
                decode_from=("rounds" if peaks and max(peaks) >= max(m["attach"][1], m["end"][1]) else
                             ("attach" if m["attach"][1] >= m["end"][1] else "end")))


def probe_check(rec, rounds_req: int, probe_every: int, flat_tol_GB: float = FLAT_TOL_GB,
                growth_tol_B: int = GROWTH_TOL_B):
    """THE L53 RULE of a memory probe (module docstring, WHERE THE PEAK IS):
      rounds  the probe ran rounds_req rounds and rounds_req >= 2 x probe_every + 1 (round 0, the probe rounds
              probe_every and 2 x probe_every, then a window W = probe_every)
      flat    no round of the last W raised the peak above the earlier rounds' peak by more than flat_tol_GB
      growth  the allocated bytes grew by at most growth_tol_B per round over the last W rounds
    Returns dict(ok, rounds_ok, flat_ok, growth_ok, growth_B_per_round, why)."""
    pk, cur = rec["round_peak_B"], rec["round_current_B"]
    G, W = len(pk), int(probe_every)
    need = 2 * W + 1
    rounds_ok = G == int(rounds_req) and G >= need
    why = []
    if not rounds_ok:
        why.append("ran %d of %d rounds; the rule needs >= %d (two probe rounds and a window of %d)" % (
            G, rounds_req, need, W))
    flat_ok = growth_ok = False
    growth = None
    if G >= W + 1 and W >= 1:
        head, tail = pk[:G - W], pk[G - W:]
        flat_ok = max(tail) <= max(head) + flat_tol_GB * 1e9
        if not flat_ok:
            why.append("round %d raised the peak by %.4f GB (> %s GB) after the earlier rounds" % (
                G - W + tail.index(max(tail)), (max(tail) - max(head)) / 1e9, flat_tol_GB))
        growth = (cur[G - 1] - cur[G - 1 - W]) / W
        growth_ok = growth <= growth_tol_B
        if not growth_ok:
            why.append("the allocated bytes grow by %.0f B per round (> %d)" % (growth, growth_tol_B))
    else:
        why.append("fewer than %d rounds: no window" % (W + 1))
    return dict(ok=rounds_ok and flat_ok and growth_ok, rounds_ok=rounds_ok, flat_ok=flat_ok, growth_ok=growth_ok,
                growth_B_per_round=growth, window=W, why=why)


def project_decode(rec, rounds_full: int, growth_B=None) -> float:
    """The full run's decode peak (GB) from a probe: its decode peak + (rounds_full - the probe's rounds) x max(the
    measured growth per round, GROWTH_DECLARED_B)."""
    g = max(float(growth_B or 0.0), float(GROWTH_DECLARED_B))
    left = max(0, int(rounds_full) - len(rec["round_peak_B"]))
    return round(rec["decode_GB"] + left * g / 1e9, 6)


def probe_fit(arm: str, C: int, B_probe: int, decode_GB: float, extra=(), margin_GB: float = 0.0, steps: int = 512):
    """The decode line of (arm, C) through the probe point, the extra MEASURED points and the MEASURED anchor
    (ANCHORS), then fit_arm. None without two distinct B."""
    pts = [(int(B_probe), float(decode_GB))] + [(int(b), float(g)) for b, g in extra]
    anc = ANCHORS.get((arm, int(C)))
    if anc is not None:
        pts.append((anc["B"], anc["peak_GB"]))
    if len({b for b, _ in pts}) < 2:
        return None
    line = fit_line(pts)
    out = fit_arm(C, line, margin_GB, steps=steps)
    out["points"] = pts
    out["anchor"] = anc
    gap = (anc["reserved_GB"] - anc["peak_GB"]) if anc is not None else None
    out["device_at_fit"] = device_check(out["peak_at_fit"], gap)
    return out


def cap_gate(name: str, peak_GB, budget_GB: float = BUDGET_GB):
    ok = peak_GB is not None and float(peak_GB) <= float(budget_GB)
    return (name, True, ok, "peak allocated %s GB over the whole arm (prefill .. end); cap %s GB" % (peak_GB, budget_GB))


def full_gate(name: str, rounds_run, steps: int, K: int):
    want = rounds_for(steps, K, 0)
    return (name, True, rounds_run == want, "%s rounds run; the full run of %d steps is %d rounds of K %d" % (
        rounds_run, steps, want, K))


def edge_gate(name: str, peak_GB, slope_GB, n_req, budget_GB: float = BUDGET_GB):
    """B is the feasible EDGE: the measured headroom is below n_req requests of the decode line's slope."""
    if peak_GB is None or slope_GB is None or n_req is None:
        return (name, True, False, "not registered or no peak (peak %s, slope %s, edge_requests %s)" % (
            peak_GB, slope_GB, n_req))
    head = float(budget_GB) - float(peak_GB)
    return (name, True, head < float(n_req) * float(slope_GB),
            "headroom %.4f GB < %s requests x %s GB = %.4f GB" % (head, n_req, slope_GB, float(n_req) * float(slope_GB)))


def reg_missing(C: int, uuid: str, reg=None, stamp=None, fit_b=None) -> list:
    """The registered values that role matched needs and that are None (empty: complete)."""
    reg = REG_WSM if reg is None else reg
    stamp = PRED_STAMP_WSM if stamp is None else stamp
    fit_b = fit_batch_registered(C) if fit_b is None else fit_b
    out = []
    if stamp is None:
        out.append("PRED_STAMP_WSM")
    if fit_b is None:
        out.append("FIT_B_%d" % C)
    for k in ("margin_GB", "edge_requests"):
        if reg[k] is None:
            out.append(k)
    if reg["slope_GB"].get(str(C)) is None:
        out.append("slope_GB[%d]" % C)
    if reg["strongest_steady"].get(uuid) is None:
        out.append("strongest_steady[%s]" % (uuid or "no uuid"))
    return out


def fit_batch_registered(C: int):
    return {73: FIT_B_73, 128: FIT_B_128}.get(int(C))


def verdict(rec, sh, B: int, uuid: str, reg=None):
    """MAIN (the verdict): the arm's median steady committed tok/s / the strongest steady tok/s of this GPU; MAIN-MEAN,
    MAIN-EFF (REPORT); DIAG: the same-config ratios against the same-job SH (DIAGNOSTIC only)."""
    reg = REG_WSM if reg is None else reg
    r = rec["rates"]
    ss, se = reg["strongest_steady"].get(uuid), reg["strongest_effective"].get(uuid)
    shr = SC.sh_rates(B, sh) if sh else {}
    q = lambda x, y: (round(float(x) / float(y), 4) if (x and y) else None)   # noqa: E731
    return dict(gpu_uuid=uuid, strongest_steady=ss, strongest_effective=se, arm_median_tok_s=r["median_tok_s"],
                arm_mean_tok_s=r["mean_tok_s"], arm_effective_tok_s=r["effective_tok_s"],
                main=q(r["median_tok_s"], ss), main_mean=q(r["mean_tok_s"], ss),
                main_effective=q(r["effective_tok_s"], se),
                diag_same_config=q(r["median_tok_s"], shr.get("median_tok_s")),
                diag_same_config_effective=q(r["effective_tok_s"], shr.get("effective_tok_s")),
                sh_vs_strongest=q(shr.get("median_tok_s"), ss))


def _band(out, name, v, b):
    if v is None:
        out.append((name, "REPORT", "no value (registered %s)" % (b,)))
    elif b is None:
        out.append((name, "REPORT", "%.4f (registered None)" % v))
    else:
        out.append((name, "CONFIRMED" if b[0] <= v <= b[1] else "REFUTED", "%.4f; registered %s" % (v, b)))


def predictions(mode: str, C: int, B: int, rec=None, prec=None, vd=None, reg=None):
    """REG_WSM against the numbers: probe PR-DEC (the decode peak at C_B); match WSM-PEAK and WSM-MAIN. A None band
    gives REPORT."""
    reg = REG_WSM if reg is None else reg
    out = []
    if mode == "probe":
        _band(out, "PR-DEC", (prec or {}).get("decode_GB"), reg["probe_decode_GB"].get("%d_%d" % (C, B)))
    else:
        _band(out, "WSM-PEAK", (rec or {}).get("gpu_peak_alloc_GB"), reg["peak_GB"].get(str(C)))
        _band(out, "WSM-MAIN", (vd or {}).get("main"), reg["main_ratio"].get(str(C)))
    return out


# ------------------------------------------------------------------ the probe (GPU; not a timed path)
class CudaMem:
    """The probe's memory record (allocator statistics, bytes). mark(name): the peak allocated / reserved since the
    previous mark or round, the current allocated / reserved; then the peak is reset. round(runner): the same after
    each round (SpecOps.after_round: after the round's host sync, spec_round.py:1046-1047). stats: an object with
    torch.cuda's five calls (None: torch.cuda; the CPU tests pass a stand-in). Host reads only: no device sync."""

    def __init__(self, stats=None):
        self.stats = stats
        self.marks, self.trace = [], []

    def _take(self):
        s = self.stats
        if s is None:
            import torch
            s = torch.cuda
        v = (int(s.max_memory_allocated()), int(s.memory_allocated()), int(s.max_memory_reserved()),
             int(s.memory_reserved()))
        s.reset_peak_memory_stats()
        return v

    def mark(self, name: str) -> None:
        self.marks.append((name,) + self._take())

    def round(self, runner) -> None:
        self.trace.append((int(runner.round),) + self._take())


def probe_spec(model, ids, inputs, steps: int, clock, opt, mem):
    """r2_spec_cell.run_spec (40ac348) with the probe's memory marks ONLY: every added statement is a call of a `mem`
    method, and SR.SpecOps gets after_round=mem.round. tests_cpu/test_wsm.py test_probe_spec_is_run_spec: without them
    the body is run_spec's AST, so the probe allocates what the timed arm allocates, in the same order. Not timed."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    cfg = model.config
    model.has_buffers = False                            # the shipped warm-up state is not used: release it
    for name in BC.MODEL_WARM_BUFFERS:
        if hasattr(model, name):
            setattr(model, name, None)
    for layer in model.layers:
        for name in BC.LAYER_WARM_STATE:
            if hasattr(layer, name):
                setattr(layer, name, None)
    gc.collect()
    torch.cuda.empty_cache()
    mem.mark("start")
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    mem.mark("prefill")
    prefill_s = time.time() - t0
    first_agree = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    K = int(opt["K"])
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"],
                       graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    gc.collect()
    torch.cuda.empty_cache()                             # the released shipped tables
    mem.mark("attach")
    a = ARM_OPS[opt["arm"]]
    sops = SR.SpecOps(level=a["level"], K=K, mode=a["mode"], gemm=a["gemm"], extra=opt["extra"],
                      journal=opt["journal"], redirect=opt["redirect"], probe_every=opt["probe_every"],
                      pin=(a["level"] == "L4P" and bool(opt.get("pin"))), after_round=mem.round)
    runner = SR.SpecRunner(model, rs, sops, clock, t0=inputs[:, 0])
    G = rounds_for(steps, K, opt["rounds"])
    runner.run(steps, rounds=G)
    rd = runner.read()
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()), K=K,
               probe_every=opt["probe_every"], prefill_s=round(prefill_s, 1), first_token_agreement=first_agree,
               meta=dict(rs.meta, spec=runner.meta))
    mem.mark("end")
    runner.detach()
    del runner, rs, cache
    return res


def device_uuid():
    """REPORT: torch's view of the GPU uuid (None when this torch has no uuid property)."""
    try:
        import torch
        u = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        return None if u is None else str(u)
    except Exception as e:   # the instrument never stops the cell
        return "error %s: %s" % (type(e).__name__, e)


# ------------------------------------------------------------------ the gates of each mode
def probe_gates(arm: str, rd, prec, chk, S0: int, block: int = BLOCK):
    """PR-ROUNDS, PR-FILL, PR-FLAT (MUST): the probe passed every allocation event of the full run."""
    need = block - (int(S0) % block)
    cmin = min(rd["count"]) if rd and rd.get("count") else 0
    return [("PR-ROUNDS", True, chk["rounds_ok"], "; ".join(w for w in chk["why"] if "rounds" in w) or
             "%d rounds (two probe rounds and a window of %d)" % (len(prec["round_peak_B"]), chk["window"])),
            ("PR-FILL", True, cmin >= need,
             "every request committed >= %d tokens (its first block fill; min %d): the fill paths ran" % (need, cmin)),
            ("PR-FLAT", True, chk["flat_ok"] and chk["growth_ok"],
             "flat %s, growth %s B per round (tolerance %d B; declared %d B); %s" % (
                 chk["flat_ok"], chk["growth_B_per_round"], GROWTH_TOL_B, GROWTH_DECLARED_B,
                 "; ".join(w for w in chk["why"] if "rounds" not in w) or "no late rise")),
            ]


def match_gates(role: str, arm: str, C: int, B: int, steps: int, K: int, rec, uuid: str, reg=None, stamp=None,
                fit_b=None):
    """<arm>-FULL and <arm>-CAP (every role); mini: PLANT-CAP, PLANT-FULL; matched: REG, FIT-B, GPU, EDGE."""
    reg = REG_WSM if reg is None else reg
    peak = rec.get("gpu_peak_alloc_GB") if rec else None
    rounds_run = rec.get("rounds") if rec else None
    g = [full_gate("%s-FULL" % arm, rounds_run, steps, K), cap_gate("%s-CAP" % arm, peak)]
    if role == "mini":
        lo = cap_gate("x", peak, peak - 0.001)[2] if peak is not None else True
        hi = cap_gate("x", peak, peak + 0.001)[2] if peak is not None else False
        g.append(("PLANT-CAP", True, (not lo) and hi, "the cap gate on this arm's peak %s GB: planted cap peak - 1 MB "
                  "-> pass %s (must be False), peak + 1 MB -> pass %s (must be True)" % (peak, lo, hi)))
        bad = full_gate("x", (rounds_run - 1) if rounds_run is not None else None, steps, K)[2]
        g.append(("PLANT-FULL", True, rounds_run is not None and not bad,
                  "the full-run gate with one round less -> pass %s (must be False; rounds %s)" % (bad, rounds_run)))
    elif role == "matched":
        miss = reg_missing(C, uuid, reg, stamp, fit_b)
        fb = fit_batch_registered(C) if fit_b is None else fit_b
        g.append(("REG", True, miss == [], "registered values missing: %s" % (miss or "none")))
        g.append(("FIT-B", True, fb is not None and int(B) == int(fb),
                  "the cell's B %d; registered FIT_B_%d %s" % (B, C, fb)))
        ss = reg["strongest_steady"].get(uuid)
        g.append(("GPU", True, bool(uuid) and ss is not None,
                  "GPU %s: registered strongest steady %s tok/s (same-GPU denominator)" % (uuid or "(none)", ss)))
        g.append(edge_gate("EDGE", peak, reg["slope_GB"].get(str(C)), reg["edge_requests"]))
    return g


# ------------------------------------------------------------------ main
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
    C = BC.C_BASE + a.pool
    arm = a.spec_arm
    gates, sh, rec, inputs, res_rd = [], None, None, None, None
    mem, prec, chk, fit, vd = None, None, None, None, None
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    hmem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
    dims = dict(L=len(model.layers), H=H, topk=64, block_bytes=64 * D * 2 * 2 + 64 * 2)
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps, C %d)" % (a.batch, a.steps, C))
        fp_sh = (torch.zeros((a.batch, a.steps), dtype=torch.float32, device="cuda"),
                 torch.zeros((a.batch, a.steps), dtype=torch.float64, device="cuda"))
        try:
            out = BC.run_shipped(model, ids, a.steps, clock, fp=fp_sh)
            pa, pr = BC.gpu_peak()
            rw = {t for t, _ in out["rewarms"]}
            steady = BC.period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
            sh = dict(steady=steady, rates=BC.rates(a.batch, a.steps, out["periods"], steady),
                      rewarms=out["rewarms"], prefill_s=out["prefill_s"], gpu_peak_alloc_GB=pa,
                      gpu_peak_reserved_GB=pr, device_used_end_GB=out["device_used_end_GB"])
            sh["tok_s"] = SC.sh_rates(a.batch, sh)
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            del out
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        hmem["after_SH"] = BC.host_mem()
        BC.free()
        if inputs is None:
            gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: SH failed (the arm starts from SH's first token)"))
        else:
            BC.free(host=True)
            opt = spec_opt(a)
            log("arm %s (%s; mode %s)" % (arm, opt, a.wsm_mode))
            try:
                if a.wsm_mode == "probe":
                    mem = CudaMem()
                    res = probe_spec(model, ids, inputs, a.steps, clock, opt, mem)
                else:
                    res = SC.run_spec(model, ids, inputs, a.steps, clock, opt)
                pa, pr = BC.gpu_peak()
                rec = SC.summarize(arm, res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                if mem is not None:                      # probe: the peaks were reset at the marks
                    prec = probe_record(mem.marks, mem.trace)
                    rec.update(gpu_peak_after_end_GB=pa, gpu_peak_alloc_GB=prec["arm_GB"], probe=prec)
                res_rd = res["read"]
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res_rd["rounds"], a.spec)))
                gates.extend(SC.arm_gates(arm, res, rec))
                del res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            hmem["after_%s" % arm] = BC.host_mem()
            BC.free()
    if a.wsm_mode == "probe":
        if prec is not None:
            chk = probe_check(prec, a.spec_rounds, a.probe_every)
            gates.extend(probe_gates(arm, res_rd, prec, chk, BC.S0))
            dec_full = project_decode(prec, rounds_for(a.steps, a.spec, 0), chk["growth_B_per_round"])
            margin = REG_WSM["margin_GB"] if REG_WSM["margin_GB"] is not None else 0.0
            fit = probe_fit(arm, C, a.batch, dec_full, a.fit_point, margin, a.steps)
        else:
            gates.extend([(n, True, False, "no probe record") for n in ("PR-ROUNDS", "PR-FILL", "PR-FLAT")])
    else:
        gates.extend(match_gates(a.wsm_role, arm, C, a.batch, a.steps, a.spec, rec, a.gpu_uuid))
        if rec is not None:
            vd = verdict(rec, sh, a.batch, a.gpu_uuid)
    preds = predictions(a.wsm_mode, C, a.batch, rec, prec, vd)
    inf = inferred_cross_c(arm) if any(r == arm for r, _ in ANCHORS) else None
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS, capacity_C=C,
                spec_k=a.spec, spec_arm=arm, spec_rounds=a.spec_rounds, opt=spec_opt(a), wsm_mode=a.wsm_mode,
                wsm_role=a.wsm_role, gpu_uuid=a.gpu_uuid, device_uuid=device_uuid(), budget_GB=BUDGET_GB,
                registered=dict(stamp=PRED_STAMP_WSM, FIT_B_73=FIT_B_73, FIT_B_128=FIT_B_128, REG_WSM=REG_WSM),
                anchors={"%s_%d" % k: v for k, v in ANCHORS.items()}, sh_measured=SH_MEASURED, dims=dims,
                host_mem=hmem, base_commit=os.environ.get("BASE_COMMIT"))
    summary = dict(meta=meta, SH=sh, ARM=rec, probe_check=chk, fit=fit, verdict=vd, inferred_cross_c=(
        dict(inf, lines={str(k): v for k, v in inf["lines"].items()}) if inf else None),
        gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
        predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-10s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB (prefill model "
              "%.3f GB)" % (a.batch, C, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
                            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"],
                            BC.mbase_peak_model(C, a.batch)))
    if rec:
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; acceptance %s; tokens equal "
              "%.6f; peak %s GB" % (arm, rec["K"], rec["rounds"], rec["rates"]["median_tok_s"],
                                    rec["rates"]["mean_tok_s"], rec["rates"]["effective_tok_s"],
                                    rec["acceptance"]["rate"], rec["tokens"]["equal_fraction"],
                                    rec["gpu_peak_alloc_GB"]))
    if prec:
        print("REPORT PROBE %s C %d B %d: prefill %s GB (model %.3f); decode %s GB (from %s: attach %s, rounds %s, end "
              "%s); arm %s GB; reserved decode %s GB; check %s" % (
                  arm, C, a.batch, prec["prefill_GB"], BC.mbase_peak_model(C, a.batch), prec["decode_GB"],
                  prec["decode_from"], prec["attach_GB"], prec["rounds_GB"], prec["end_GB"], prec["arm_GB"],
                  prec["decode_reserved_GB"], chk))
    if fit:
        print("FIT %s C %d: decode line a %s GB + b %s GB x B through %s; B_fit %d (margin %s GB, cap %s GB): predicted "
              "peak %s GB, B_fit + 1 %s GB, prefill model at B_fit %s GB; int32 limit %d; device %s" % (
                  arm, C, fit["line"][0], fit["line"][1], fit["points"], fit["B_fit"], fit["margin_GB"],
                  fit["budget_GB"], fit["peak_at_fit"], fit["peak_next"], fit["prefill_at_fit"], fit["int32_limit"],
                  fit["device_at_fit"]))
    if a.wsm_mode == "probe":
        print("REPORT SH fit at C %d: model (mbase_peak_model) B %d; MEASURED %s (SCORING_mbstrong.md:106-110)" % (
            C, fit_batch(lambda x: BC.mbase_peak_model(C, x), BUDGET_GB), SH_MEASURED.get(C)))
    if vd:
        print("VERDICT %s C %d B %d (GPU %s): MAIN %s = median steady %s / strongest steady %s; MAIN-MEAN %s; MAIN-EFF %s "
              "(effective %s / %s)" % (arm, C, a.batch, vd["gpu_uuid"] or "(none)", vd["main"], vd["arm_median_tok_s"],
                                        vd["strongest_steady"], vd["main_mean"], vd["main_effective"],
                                        vd["arm_effective_tok_s"], vd["strongest_effective"]))
        print("DIAGNOSTIC (same config, not the verdict) %s / same-job SH at C %d B %d: median %s, effective %s; SH / "
              "strongest %s" % (arm, C, a.batch, vd["diag_same_config"], vd["diag_same_config_effective"],
                                vd["sh_vs_strongest"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_wsm: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
