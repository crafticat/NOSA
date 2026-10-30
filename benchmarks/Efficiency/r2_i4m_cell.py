"""I4M (KS/analysis/plan/PLAN_L4P_faithful.md, contract 4; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the MATCHED-MEMORY
VERDICT cell of the current best arm L4PTFD (rung I4A, lever 1: r2_i4a_cell.py, r2_i4a_lean.py; tree acbe433). It
applies the WS-M method (nosi-r2-wsm 53fa247; benchmarks/Efficiency/r2_wsm_cell.py is a BYTE COPY of that blob in this
tree, test_i4m.py checks the blob; its pure parts are imported, not rewritten) to L4PTFD and forms the reviewer's verdict

    MAIN = L4PTFD's median steady committed tok/s at (C, FIT_B_<C>)
           / the strongest tuned cached NOSA on the SAME GPU (the whole-step graph at C 128 B 216:
             REG_I4M["strongest_steady"][GPU uuid], registered by the lead)

SH runs first at the arm's (C, B): the same-job baseline and the source of the arm's first token. Its ratio is printed
as DIAGNOSTIC (same config) only. The EFFECTIVE view (MAIN-EFF = effective tok/s / the strongest effective tok/s of the
same GPU) is printed on its own line and is REPORT only: the ledger rule keeps the resource view and the elapsed view
apart. Step script: benchmarks/Efficiency/r2_i4m_step.sh. Notes: docs/NOTES_i4m.md.

WHERE THE PEAK OF L4PTFD IS (MEASURED in I4A, CODE; docs/NOTES_i4m.md section 2)
  L4PTFD has TWO peaks. (1) The PREFILL into the chain arena (r2_i3_cell._prefill_chain): the arena of L x (C + 1) + X
  + 1 blocks per request (65,792 B per block) is allocated BEFORE the shipped prefill, then the prefill's transient comes
  on top. (2) The DECODE (after the attach). I4A MEASURED both: C 128 B 216 prefill 83.891 GB, decode 82.559 GB; C 73
  B 318 prefill 77.685 GB, decode 76.932 GB. The PREFILL binds at both C. So the probe fits TWO lines per C (prefill,
  decode) and the arm's peak model is their max.

TWO MODES (--i4m-mode)
  probe  SH, then L4PTFD through probe_arm_x: a COPY of r2_i4a_cell.run_arm_x plus the instrument only (every added
         statement is a call of a `cmem` method and every added keyword argument has a `cmem` value;
         tests_cpu/test_i4m.py test_probe_arm_is_run_arm_x), for --probe-rounds rounds (default 3 x probe_every + 1 =
         49) at the full run's horizon (--steps 512). The marks split the arm's peak into PREFILL (start .. prefill),
         ATTACH (prefill .. the decode reset), every ROUND and END (WM.probe_record). WM.probe_check applies the L53 rule
         (flat window, bounded growth). probe_fit_x fits a PREFILL line and a DECODE line of this C through the probe
         point and the MEASURED I4A anchor (ANCHORS) and prints B_fit = the largest B with max(lines) + margin <= 84.0
         GB (or the anchor's B when the anchor itself is MEASURED feasible), with the B at the margins 0 / 0.10 / 0.25.
  match  SH at (C, B), then L4PTFD at (C, B) with r2_i4a_cell.run_arm_x (THE TIMED PATH OF I4A, unchanged): the full
         run (contract 4: every request exactly --steps tokens). Role mini: the miniature (plants on the cell's own
         gates). Role matched: the verdict cell at the registered FIT_B_<C>.

MUST GATES: C0_FORK; SH-RUN; L4PTFD-RUN, -ERR, -D1, -D3, -SPAN (r2_wsa_cell.arm_gates_w), -RESID, -ATTN1, -FILL1
(r2_i3_cell.fork_gates), -LEAN (r2_i4a_cell.lean_gate), -MEM (the arm's peak WITH the prefill <= 84.0 GB: the cap); match
also -COMMIT-N (exactly N per request); probe -COMMIT-P (the bounded run: its rounds, no request past N, the sums close)
and PR-ROUNDS, PR-FILL, PR-FLAT (WM.probe_gates), PR-PRE (the instrument's prefill window = the arena prefill's own
peak); role mini PLANT-CAP, PLANT-FULL; role matched REG, FIT-B, GPU, EDGE. REPORT: SH-MEM, the phases, the tokens (FAST:
no token MUST gate; I4A's i4am checked L4PTFD = L4PTF bit for bit).
REGISTERED: PRED_STAMP_I4M, FIT_B_73, FIT_B_128 and REG_I4M, NONE until the lead registers them; the step script STOPS
i4m73 / i4m128 while PRED_STAMP_I4M or their FIT_B is None. EXIT: the number of failed MUST gates (0-19); 20 refused;
30 an exception outside an arm.
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
import r2_i4a_cell as I4A          # noqa: E402  rung I4A: the arm L4PTFD, its gates and its timed path
import r2_wsm_cell as WM           # noqa: E402  WS-M (nosi-r2-wsm 53fa247, byte copy): the fit and the probe instrument

I3C, LN, WA, SC, BC = I4A.I3C, I4A.LN, I4A.WA, I4A.SC, I4A.BC
state_checksum = I4A.state_checksum   # the names that probe_arm_x shares with r2_i4a_cell.run_arm_x (the AST check)
ARM = "L4PTFD"
BUDGET_GB = WM.BUDGET_GB           # 84.0 GB peak allocated: the reviewer's cap
USABLE_GB = I4A.USABLE_GB          # 84.25 GB: the usable allocation ceiling (lesson L59), REPORT
UUID_NOVA05, UUID_NOVA07 = WM.UUID_NOVA05, WM.UUID_NOVA07

# ------------------------------------------------------------------ REGISTERED (NONE UNTIL THE LEAD REGISTERS THEM)
# Fill FIT_B_73, FIT_B_128, every needed value of REG_I4M and PRED_STAMP_I4M (`date -u` in the same command, L33) in ONE
# commit BEFORE i4m73 / i4m128: the step script STOPS them while PRED_STAMP_I4M or their FIT_B_<C> is None. FIT_B_<C>
# comes from the probe cell i4mp<C> (its line "FIT ..."). Keep the form `FIT_B_73 = 351` (the step script reads field
# 3). The builder's MODEL for every key, with the arithmetic: docs/NOTES_i4m.md section 4 (an input, not a
# registration). Keys of REG_I4M:
#   margin_GB            fit: the predicted peak + margin <= 84.0 GB (a MEASURED feasible anchor B is kept)
#   edge_requests        EDGE: 84.0 - the measured peak < edge_requests x slope_GB[C] (FIT_B is the edge)
#   slope_GB[C]          GB per request of the BINDING line at FIT_B (the probe's FIT line)
#   strongest_steady     {GPU uuid: tok/s}: the MAIN denominator (MUST for the GPU gate)
#   strongest_effective  {GPU uuid: tok/s}: the MAIN-EFF denominator (REPORT)
#   probe_prefill_GB     {"C_B": (low, high)}: the probe's prefill peak (PR-PRE prediction; REPORT if None)
#   probe_decode_GB      {"C_B": (low, high)}: the probe's decode peak (PR-DEC prediction; REPORT if None)
#   peak_GB[C]           (low, high): the matched arm's peak WITH the prefill (I4M-PEAK)
#   main_ratio[C]        (low, high): MAIN (I4M-MAIN)
#   diag_ratio[C]        (low, high): L4PTFD / same-job SH, median steady (I4M-DIAG; DIAGNOSTIC)
# The lead's registration, AFTER the probes (m3p73 / m3p128, 586_i4mp_n5, ended 12:35:19Z) and BEFORE i4m73 / i4m128.
# FIT_B from the probes' FIT lines at margin 0.10 GB (prefill line binds: C 73 19.020009 + 0.1844811 B; C 128 19.020232
# + 0.3003276 B). The probe peaks are already MEASURED, so probe_prefill_GB / probe_decode_GB stay None (REPORT).
PRED_STAMP_I4M = "2026-10-29T12:48:07Z"
FIT_B_73 = 351
FIT_B_128 = 216
REG_I4M = dict(
    margin_GB=0.10,
    edge_requests=4,
    slope_GB={"73": 0.1844811, "128": 0.3003276},
    strongest_steady={UUID_NOVA05: 3426.2, UUID_NOVA07: 3410.1},
    strongest_effective={UUID_NOVA05: None, UUID_NOVA07: 2778.3},
    probe_prefill_GB={"73_64": None, "128_64": None},
    probe_decode_GB={"73_64": None, "128_64": None},
    peak_GB={"73": (83.5, 84.0), "128": (83.85, 84.0)},
    main_ratio={"73": (0.44, 0.48), "128": (0.50, 0.53)},
    diag_ratio={"73": (0.58, 0.62), "128": (0.51, 0.53)},
)

# ------------------------------------------------------------------ constants (MEASURED / CODE; not registrations)
GIB = 1024 ** 3
# The MEASURED I4A anchors of L4PTFD (full runs, acbe433's code path, 512 steps): the L4PTFD-MEM line of each record
# (prefill / decode in GiB there; x 1.073741824 here). decode = BC.gpu_peak after the arm (reset after the attach).
ANCHORS = {
    73: dict(B=318, prefill_GB=77.685, decode_GB=76.932, decode_reserved_GB=77.186, rounds=209,
             src="i4a_q4b_2195576_i4ab.step/stdout.txt:59 (prefill 72.35 GiB, decode 71.649 / reserved 71.885 GiB)"),
    128: dict(B=216, prefill_GB=83.891, decode_GB=82.559, decode_reserved_GB=82.751, rounds=194,
              src="i4a_q3a_2195575_i4aa.step/stdout.txt:59 (prefill 78.13 GiB, decode 76.889 / reserved 77.068 GiB)"),
}
# A MEASURED prefill point at another horizon (i4a_q3m_2195575_i4am.step/stdout.txt:77: C 73 B 8, 72 steps, prefill
# 19.088 GiB). The GPU prefill does not depend on the horizon (max_gen_len sizes host buffers only, cache_engine.py:195-196),
# so it checks the prefill line's shape (REPORT "linearity"); it is not a fit point.
EXTRA_PREFILL = {73: [(8, 20.496)]}
SH_MEASURED = WM.SH_MEASURED       # SH's MEASURED fits: C 73 B 358 fits (369 OOM), C 128 B 222 fits (224 OOM)
BLOCK_BYTES = 64 * 2 * (2 * 128 + 1) * 2   # one arena block: 64 tokens x H 2 x (K D + V D + bias) x bf16 = 65,792 B
MARGINS_REPORT = (0.0, 0.10, 0.25)
PR_PRE_TOL_GB = 0.002              # PR-PRE: the instrument's prefill vs the arena prefill's own peak (GiB rounded to 1e-3)
MODES = ("probe", "match")
ROLES = ("probe", "mini", "matched")


def log(msg):
    print("[r2_i4m %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--fork-store", type=int, default=0, help="X (0: the bound 1 + (K - 1) x 64)")
    ap.add_argument("--prefill-hook", default="nocopy", choices=("nocopy", "copy"))
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"))
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--i4m-mode", required=True, choices=MODES)
    ap.add_argument("--i4m-role", required=True, choices=ROLES)
    ap.add_argument("--probe-rounds", type=int, default=0,
                    help="probe: the probe's rounds (0: 3 x probe_every + 1); match: must be 0 (the full run)")
    ap.add_argument("--gpu-uuid", default="", help="the holder's GPU uuid (HOLDERS/<job>.gpu_uuid): the denominator key")
    ap.add_argument("--fit-prefill", action="append", default=[], help="probe: B:GB, one more MEASURED prefill point")
    ap.add_argument("--fit-decode", action="append", default=[], help="probe: B:GB, one more MEASURED decode point")
    a = ap.parse_args(argv)
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.probe_every < 2 or a.spec_journal_blocks < 1 or a.probe_rounds < 0:
        ap.error("--probe-every >= 2, journal >= 1, --probe-rounds >= 0")
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(I4A.TOPK, a.spec):
        ap.error("--fork-store %d: below the bound 1 + (K - 1) x 64 = %d (0 = the bound)"
                 % (a.fork_store, I3C.KA.store_slots(I4A.TOPK, a.spec)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    if (a.i4m_mode == "probe") != (a.i4m_role == "probe"):
        ap.error("--i4m-mode probe goes with --i4m-role probe only (match: mini or matched)")
    if a.i4m_mode == "match" and a.probe_rounds != 0:
        ap.error("--i4m-mode match runs the full run: --probe-rounds must be 0")
    if a.i4m_role == "matched" and a.steps != 512:
        ap.error("--i4m-role matched runs the workload's full decode: --steps 512")
    if a.i4m_mode == "probe" and a.probe_rounds == 0:
        a.probe_rounds = 3 * a.probe_every + 1
    for name in ("fit_prefill", "fit_decode"):
        pts = []
        for s in getattr(a, name):
            try:
                b, g = s.split(":")
                pts.append((int(b), float(g)))
            except ValueError:
                ap.error("--%s B:GB (got %r)" % (name.replace("_", "-"), s))
        setattr(a, name, pts)
    return a


def arm_opt(a) -> dict:
    """The arm's options, as r2_i4a_cell.main builds them (test_i4m checks the keys and the expressions); check_every 0
    (the timing cells' value: no check work)."""
    return dict(arm=ARM, K=a.spec, journal=a.spec_journal_blocks, tu_gemm=a.tu_gemm, twin=a.spec_twin,
                body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                probe_every=a.probe_every, check_every=0, fork_store=a.fork_store, prefill_hook=a.prefill_hook)


# ------------------------------------------------------------------ pure parts (CPU-tested)
def rounds_bound(steps: int) -> int:
    """The runner's own round limit of contract 4 (spec_tentative.py:675: 4 N + 16): the decode projection's horizon
    (an upper bound of the full run's rounds; I4A ran 194 / 209)."""
    return 4 * int(steps) + 16


def arena_gb_per_request(C: int, L: int = 32, X: int = 129) -> float:
    """CODE: the chain arena's bytes of ONE request (r2_i3_arena.ChainArena: L x (C + 1) engine blocks + X store blocks
    + 1 CB0 block, BLOCK_BYTES each), in GB. A lower bound of what one more request adds to the prefill peak."""
    return (int(L) * (int(C) + 1) + int(X) + 1) * BLOCK_BYTES / 1e9


def line_at(line, B) -> float:
    return float(line[0]) + float(line[1]) * float(B)


def arm_peak_x(B, lines) -> float:
    """The arm's peak model (GB allocated): max(the prefill line, the decode line)."""
    return max(line_at(lines["prefill"], B), line_at(lines["decode"], B))


def binding(B, lines) -> str:
    return "prefill" if line_at(lines["prefill"], B) >= line_at(lines["decode"], B) else "decode"


def fit_arm_x(C: int, lines, margin_GB: float = 0.0, budget_GB: float = BUDGET_GB, steps: int = 512, S: int = None,
              floor_B: int = 0):
    """B_fit at capacity C: the largest B with arm_peak_x + margin <= budget (WM.fit_batch), capped by the int32
    envelope (WM.int32_batch_limit); floor_B: a MEASURED feasible B (peak <= budget in a full run) is kept even when the
    margin excludes it (the margin covers an extrapolation, not a measurement)."""
    S = BC.S0 if S is None else int(S)
    lim = WM.int32_batch_limit(S, steps, C)
    B_line = WM.fit_batch(lambda x: arm_peak_x(x, lines), budget_GB, margin_GB, b_max=lim)
    B = max(B_line, int(floor_B)) if 0 < int(floor_B) <= lim else B_line
    pk = round(arm_peak_x(B, lines), 4) if B else None
    sh = SH_MEASURED.get(int(C))
    return dict(C=int(C), B_fit=B, B_line=B_line, floor_B=int(floor_B), margin_GB=float(margin_GB),
                budget_GB=float(budget_GB), int32_limit=lim,
                prefill_line=(round(lines["prefill"][0], 6), round(lines["prefill"][1], 7)),
                decode_line=(round(lines["decode"][0], 6), round(lines["decode"][1], 7)),
                peak_at_fit=pk, peak_next=round(arm_peak_x(B + 1, lines), 4),
                binding=(binding(B, lines) if B else None),
                slope_binding=(round(lines[binding(B, lines)][1], 7) if B else None),
                prefill_at_fit=(round(line_at(lines["prefill"], B), 4) if B else None),
                decode_at_fit=(round(line_at(lines["decode"], B), 4) if B else None),
                headroom_GB=(round(float(budget_GB) - pk, 4) if pk is not None else None),
                arena_next_GB=round(arena_gb_per_request(C), 6),
                sh_fit=(sh or {}).get("fits"), sh_beside=(sh is None or B <= sh["fits"]))


def probe_fit_x(C: int, B_probe: int, prec, growth_B=None, margin_GB: float = 0.0, extra_prefill=(), extra_decode=(),
                steps: int = 512):
    """The two lines of (L4PTFD, C) through the probe point (prefill: its prefill peak; decode: its decode peak
    projected to rounds_bound(steps) rounds by WM.project_decode), the extra MEASURED points and the MEASURED anchor
    (ANCHORS), then fit_arm_x (the anchor's B is the floor when the anchor fits). None without two distinct B in a
    line. REPORT parts: the B_fit at MARGINS_REPORT, the device need, the EXTRA_PREFILL residuals."""
    anc = ANCHORS.get(int(C))
    rf = rounds_bound(steps)
    dec = WM.project_decode(prec, rf, growth_B)
    pp = [(int(B_probe), float(prec["prefill_GB"]))] + [(int(b), float(g)) for b, g in extra_prefill]
    dp = [(int(B_probe), float(dec))] + [(int(b), float(g)) for b, g in extra_decode]
    if anc is not None:
        pp.append((anc["B"], anc["prefill_GB"]))
        dp.append((anc["B"], anc["decode_GB"]))
    if len({b for b, _ in pp}) < 2 or len({b for b, _ in dp}) < 2:
        return None
    lines = dict(prefill=WM.fit_line(pp), decode=WM.fit_line(dp))
    floor = anc["B"] if (anc is not None and max(anc["prefill_GB"], anc["decode_GB"]) <= BUDGET_GB) else 0
    out = fit_arm_x(C, lines, margin_GB, steps=steps, floor_B=floor)
    out.update(points=dict(prefill=pp, decode=dp), anchor=anc, decode_projected_GB=dec, rounds_projected=rf,
               margins={("%.2f" % m): fit_arm_x(C, lines, m, steps=steps, floor_B=floor)["B_fit"]
                        for m in MARGINS_REPORT})
    gaps = [float(prec["prefill_reserved_GB"]) - float(prec["prefill_GB"])]
    if anc is not None:
        gaps.append(anc["decode_reserved_GB"] - anc["decode_GB"])
    out["device_at_fit"] = WM.device_check(out["peak_at_fit"], max(gaps))
    out["linearity"] = [dict(B=b, GB=g, line_GB=round(line_at(lines["prefill"], b), 4),
                             residual_GB=round(g - line_at(lines["prefill"], b), 4)) for b, g in EXTRA_PREFILL.get(int(C), [])]
    return out


def commit_p_gate(arm, rd, rounds_req: int, steps: int):
    """<arm>-COMMIT-P (probe: a bounded run): the run stopped at rounds_req rounds, every request committed more than its
    warm steps and at most `steps` tokens, and the warm tokens plus the per-round sums equal the total."""
    cnt = rd["count"]
    warm = rd["warm"] * len(cnt)
    ok = (rd["rounds"] == int(rounds_req) and min(cnt) > rd["warm"] and max(cnt) <= int(steps)
          and sum(rd["round_committed"]) + warm == sum(cnt))
    return ("%s-COMMIT-P" % arm, True, ok, "%d rounds (%d asked); committed per request min %d max %d (<= %d); warm %d "
            "+ per-round sum %d, total %d" % (rd["rounds"], rounds_req, min(cnt), max(cnt), steps, warm,
                                              sum(rd["round_committed"]), sum(cnt)))


def arm_gates_x(mode: str, res, rec, L: int, steps: int, rounds_req: int = 0):
    """The arm's MUST gates without RUN and MEM: I4A's (ERR, COMMIT-N, D1, D3, SPAN; RESID, ATTN1, FILL1; LEAN) with
    COMMIT-P in place of COMMIT-N in probe mode."""
    g = [(n.replace("L4PT-", ARM + "-", 1), m, ok, d) for n, m, ok, d in WA.arm_gates_w("L4PT", res, rec, steps)]
    if mode == "probe":
        g = [x for x in g if not x[0].endswith("-COMMIT-N")] + [commit_p_gate(ARM, res["read"], rounds_req, steps)]
    g.extend(I3C.fork_gates(ARM, res, L, False, False))
    g.append(I4A.lean_gate(ARM, dict(res["read"], K=res["K"]), L))
    return g


def mem_gate(name: str, peak_GB, budget_GB: float = BUDGET_GB, what: str = ""):
    """THE CAP (MUST): the arm's peak allocated WITH the prefill <= budget_GB."""
    ok = peak_GB is not None and float(peak_GB) <= float(budget_GB)
    return (name, True, ok, "peak allocated %s GB with the prefill against the cap %s GB (usable ceiling %.2f GB, "
            "REPORT); %s" % (peak_GB, budget_GB, USABLE_GB, what))


def pr_pre_gate(prec, mem):
    """PR-PRE (MUST, probe): the instrument's PREFILL window is the arena prefill: its peak = _prefill_chain's own
    prefill peak (GiB, rounded to 1e-3) within PR_PRE_TOL_GB."""
    pre = (mem or {}).get("prefill_peak_GB")
    got = (prec or {}).get("prefill_GB")
    ref = round(float(pre) * GIB / 1e9, 4) if pre is not None else None
    ok = got is not None and ref is not None and abs(float(got) - ref) <= PR_PRE_TOL_GB
    return ("PR-PRE", True, ok, "the probe's prefill mark %s GB; the arena prefill's own peak %s GB (%s GiB); tolerance "
            "%s GB" % (got, ref, pre, PR_PRE_TOL_GB))


def fit_batch_registered(C: int):
    return {73: FIT_B_73, 128: FIT_B_128}.get(int(C))


def reg_missing(C: int, uuid: str, reg=None, stamp=None, fit_b=None) -> list:
    """The registered values that role matched needs and that are None (empty: complete)."""
    reg = REG_I4M if reg is None else reg
    stamp = PRED_STAMP_I4M if stamp is None else stamp
    fit_b = fit_batch_registered(C) if fit_b is None else fit_b
    out = []
    if stamp is None:
        out.append("PRED_STAMP_I4M")
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


def match_gates(role: str, C: int, B: int, steps: int, rd, peak_GB, uuid: str, reg=None, stamp=None, fit_b=None):
    """Role mini: PLANT-CAP (the MEM gate on this arm's peak with a planted cap), PLANT-FULL (COMMIT-N with one request
    one token short); role matched: REG, FIT-B, GPU, EDGE."""
    reg = REG_I4M if reg is None else reg
    g = []
    if role == "mini":
        lo = mem_gate("x", peak_GB, peak_GB - 0.001)[2] if peak_GB is not None else True
        hi = mem_gate("x", peak_GB, peak_GB + 0.001)[2] if peak_GB is not None else False
        g.append(("PLANT-CAP", True, (not lo) and hi, "the cap gate on this arm's peak %s GB: planted cap peak - 1 MB "
                  "-> pass %s (must be False), peak + 1 MB -> pass %s (must be True)" % (peak_GB, lo, hi)))
        real = bad = None
        if rd and rd.get("count"):
            real = WA.commit_n_gate(ARM, dict(read=rd), steps)[2]
            short = dict(rd, count=[rd["count"][0] - 1] + list(rd["count"][1:]))
            bad = WA.commit_n_gate(ARM, dict(read=short), steps)[2]
        g.append(("PLANT-FULL", True, real is True and bad is False,
                  "COMMIT-N on this run -> %s (must be True); with one request one token short -> %s (must be "
                  "False)" % (real, bad)))
    elif role == "matched":
        miss = reg_missing(C, uuid, reg, stamp, fit_b)
        fb = fit_batch_registered(C) if fit_b is None else fit_b
        g.append(("REG", True, miss == [], "registered values missing: %s" % (miss or "none")))
        g.append(("FIT-B", True, fb is not None and int(B) == int(fb),
                  "the cell's B %d; registered FIT_B_%d %s" % (B, C, fb)))
        ss = reg["strongest_steady"].get(uuid)
        g.append(("GPU", True, bool(uuid) and ss is not None,
                  "GPU %s: registered strongest steady %s tok/s (same-GPU denominator)" % (uuid or "(none)", ss)))
        g.append(WM.edge_gate("EDGE", peak_GB, reg["slope_GB"].get(str(C)), reg["edge_requests"]))
    return g


def predictions(mode: str, C: int, B: int, prec=None, peak_GB=None, vd=None, reg=None):
    """REG_I4M against the numbers: probe PR-PRE, PR-DEC (the probe's prefill / decode peak at C_B); match I4M-PEAK,
    I4M-MAIN, I4M-DIAG. A None band gives REPORT."""
    reg = REG_I4M if reg is None else reg
    out = []
    if mode == "probe":
        key = "%d_%d" % (C, B)
        WM._band(out, "PR-PRE", (prec or {}).get("prefill_GB"), reg["probe_prefill_GB"].get(key))
        WM._band(out, "PR-DEC", (prec or {}).get("decode_GB"), reg["probe_decode_GB"].get(key))
    else:
        WM._band(out, "I4M-PEAK", peak_GB, reg["peak_GB"].get(str(C)))
        WM._band(out, "I4M-MAIN", (vd or {}).get("main"), reg["main_ratio"].get(str(C)))
        WM._band(out, "I4M-DIAG", (vd or {}).get("diag_same_config"), reg["diag_ratio"].get(str(C)))
    return out


def verdict_lines(arm: str, C: int, B: int, vd) -> list:
    """The printed verdict: the STEADY view (MAIN, the reviewer's metric), the EFFECTIVE view (REPORT, a line of its
    own: the elapsed view is not mixed into the verdict) and the same-config DIAGNOSTIC."""
    return ["VERDICT %s C %d B %d (GPU %s): MAIN %s = median steady %s / strongest steady %s; MAIN-MEAN %s" % (
                arm, C, B, vd["gpu_uuid"] or "(none)", vd["main"], vd["arm_median_tok_s"], vd["strongest_steady"],
                vd["main_mean"]),
            "EFFECTIVE VIEW (elapsed; REPORT, kept apart from the verdict) %s C %d B %d: MAIN-EFF %s = effective %s / "
            "strongest effective %s; same-job SH effective ratio %s" % (
                arm, C, B, vd["main_effective"], vd["arm_effective_tok_s"], vd["strongest_effective"],
                vd["diag_same_config_effective"]),
            "DIAGNOSTIC (same config, not the verdict) %s / same-job SH at C %d B %d: median %s; SH / strongest %s" % (
                arm, C, B, vd["diag_same_config"], vd["sh_vs_strongest"])]


class ProbeMem(WM.CudaMem):
    """WS-M's CudaMem (the marks and the per-round records) plus the probe's round limit."""

    def __init__(self, rounds: int, stats=None):
        super().__init__(stats)
        self.rounds = int(rounds)


# ------------------------------------------------------------------ the probe arm (GPU; not a timed path)
def probe_arm_x(model, ids, inputs, steps: int, clock, opt, FORK, lean: bool, cmem):
    """r2_i4a_cell.run_arm_x (acbe433) with the probe's instrument ONLY: every added statement is a call of a `cmem`
    method; the added keyword arguments are SpecOps(after_round=cmem.round) and runner.run(max_rounds=cmem.rounds).
    tests_cpu/test_i4m.py test_probe_arm_is_run_arm_x: without them the body is run_arm_x's AST, so the probe
    allocates what the timed arm allocates, in the same order. The marks: start, prefill (after the arena prefill),
    attach (BEFORE run_arm_x's peak reset: the attach window is not lost), end. Not timed."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    Runner = LN.lean_classes(STF)[1] if lean else STF.TentRunnerF
    gb = 1024 ** 3
    K = int(opt["K"])
    X = I3C.fork_store_slots("L4PTF", K, opt["fork_store"])
    cmem.mark("start")
    cache, arena, prefill_s, first, mem = I3C._prefill_chain(model, ids, inputs, steps, clock, X, opt["prefill_hook"])
    cmem.mark("prefill")
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=opt["graph"])
    rs = RE.attach(model, cache, BC.S0, steps + 3 * K + 8, ops)
    I3C.KA.attach(rs, arena)
    gc.collect()
    torch.cuda.empty_cache()
    cmem.mark("attach")
    torch.cuda.reset_peak_memory_stats()                              # the decode peak (the prefill peak is above)
    sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=opt["tu_gemm"], extra=X, journal=opt["journal"],
                      redirect=opt["redirect"], probe_every=opt["probe_every"], after_round=cmem.round)
    check = opt["check_every"] == 1
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                    check_every=ce, union_check=check, union_stats=check, phase_marks=True)
    if check:
        runner.vs.fa_check = {}                                        # CHECK-ONLY: the "F" / "RA" probes
    runner.run(steps, max_rounds=cmem.rounds)
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
               compile=I3C._compile_report(FORK), state=state_checksum(runner))
    cmem.mark("end")
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
    C = BC.C_BASE + a.pool
    gates, sh, rec, inputs, res_rd = [], None, None, None, None
    cmem, prec, chk, fit, vd = None, None, None, None, None
    FORK, g0 = I3C.import_fork()
    gates.append(g0)
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    hmem = dict(after_load=BC.host_mem())
    lay0 = model.layers[0]
    H, D = int(lay0.num_key_value_heads), int(lay0.head_dim)
    L = len(model.layers)
    dims = dict(L=L, H=H, D=D, topk=I4A.TOPK, block_bytes=64 * D * 2 * 2 + 64 * 2)
    opt = arm_opt(a)
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps, C %d)" % (a.batch, a.steps, C))
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
            gates.append(I4A.mem_record("SH", "prefill + decode", pa))
            del out
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        hmem["after_SH"] = BC.host_mem()
        BC.free()
        if inputs is None:
            gates.append(("%s-RUN" % ARM, True, False, "NOT_RUN: SH failed (the arm uses SH's first token)"))
        elif FORK is None:
            gates.append(("%s-RUN" % ARM, True, False, "NOT_RUN: the fork did not import (C0_FORK)"))
        else:
            BC.free(host=True)
            log("arm %s (%s; mode %s, role %s)" % (ARM, opt, a.i4m_mode, a.i4m_role))
            try:
                if a.i4m_mode == "probe":
                    cmem = ProbeMem(a.probe_rounds)
                    res = probe_arm_x(model, ids, inputs, a.steps, clock, opt, FORK, True, cmem)
                else:
                    res = I4A.run_arm_x(model, ids, inputs, a.steps, clock, opt, FORK, lean=True)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(arm=ARM, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(res["read"], L), fork=I3C.fork_record(res["read"]),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=res["read"].get("lean"))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                if cmem is not None:                     # probe: the marks reset the peaks
                    prec = WM.probe_record(cmem.marks, cmem.trace)
                    rec.update(gpu_peak_after_end_GB=pa, gpu_peak_alloc_GB=prec["decode_GB"], probe=prec)
                    rec["peak_with_prefill_GB"] = prec["arm_GB"]
                else:
                    rec["peak_with_prefill_GB"] = I4A.peak_gb(rec)
                res_rd = res["read"]
                gates.append(("%s-RUN" % ARM, True, True, "%d rounds of K %d" % (res_rd["rounds"], a.spec)))
                gates.extend(arm_gates_x(a.i4m_mode, res, rec, L, a.steps, a.probe_rounds))
                gates.append(mem_gate("%s-MEM" % ARM, rec["peak_with_prefill_GB"], what="arena %s" % res["mem"]))
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % ARM, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            hmem["after_%s" % ARM] = BC.host_mem()
            BC.free()
    peak = (rec or {}).get("peak_with_prefill_GB")
    if a.i4m_mode == "probe":
        if prec is not None:
            chk = WM.probe_check(prec, a.probe_rounds, a.probe_every)
            gates.extend(WM.probe_gates(ARM, res_rd, prec, chk, BC.S0))
            gates.append(pr_pre_gate(prec, rec.get("mem_arena")))
            margin = REG_I4M["margin_GB"] if REG_I4M["margin_GB"] is not None else 0.0
            fit = probe_fit_x(C, a.batch, prec, chk["growth_B_per_round"], margin, a.fit_prefill, a.fit_decode,
                              a.steps)
        else:
            gates.extend([(n, True, False, "no probe record") for n in ("PR-ROUNDS", "PR-FILL", "PR-FLAT", "PR-PRE")])
    else:
        gates.extend(match_gates(a.i4m_role, C, a.batch, a.steps, res_rd, peak, a.gpu_uuid))
        if rec is not None:
            vd = WM.verdict(rec, sh, a.batch, a.gpu_uuid, REG_I4M)
    preds = predictions(a.i4m_mode, C, a.batch, prec, peak, vd)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS, capacity_C=C,
                spec_k=a.spec, arm=ARM, opt=opt, i4m_mode=a.i4m_mode, i4m_role=a.i4m_role, probe_rounds=a.probe_rounds,
                gpu_uuid=a.gpu_uuid, device_uuid=WM.device_uuid(), budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                registered=dict(stamp=PRED_STAMP_I4M, FIT_B_73=FIT_B_73, FIT_B_128=FIT_B_128, REG_I4M=REG_I4M),
                anchors={str(k): v for k, v in ANCHORS.items()}, extra_prefill={str(k): v for k, v in EXTRA_PREFILL.items()},
                sh_measured=SH_MEASURED, arena_GB_per_request=round(arena_gb_per_request(C, L), 6), dims=dims,
                host_mem=hmem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=I4A.FORK_BLOB)
    summary = dict(meta=meta, SH=sh, ARM=rec, probe_check=chk, fit=fit, verdict=vd,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-10s %s: %s" % p)
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, C, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"], sh["tok_s"]["mean_tok_s"],
            sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    if rec:
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; round median %s ms; "
              "phases %s; draft step %s ms = %s SH; joint %s SH steps; lean %s; tokens equal %.6f (first %s); peak %s GB "
              "(with the prefill %s GB); arena %s" % (
                  ARM, rec["K"], rec["rounds"], rec["rates"]["median_tok_s"], rec["rates"]["mean_tok_s"],
                  rec["rates"]["effective_tok_s"], rec["vs_sh"], rec["rates"].get("round_ms_median"),
                  rec["phases_median"], rec["draft_step_ms"], rec["draft_step_sh"], rec["joint_sh_steps"],
                  rec.get("lean"), rec["tokens"]["equal_fraction"], rec["tokens"]["first"], rec.get("gpu_peak_alloc_GB"),
                  rec.get("peak_with_prefill_GB"), rec.get("mem_arena")))
    if prec:
        print("REPORT PROBE %s C %d B %d: prefill %s GB (reserved %s); decode %s GB (from %s: attach %s, rounds %s, end "
              "%s); arm %s GB; reserved decode %s GB; check %s" % (
                  ARM, C, a.batch, prec["prefill_GB"], prec["prefill_reserved_GB"], prec["decode_GB"],
                  prec["decode_from"], prec["attach_GB"], prec["rounds_GB"], prec["end_GB"], prec["arm_GB"],
                  prec["decode_reserved_GB"], chk))
    if fit:
        print("FIT %s C %d: prefill line a %s GB + b %s GB x B through %s; decode line a %s + b %s x B through %s "
              "(probe decode projected to %d rounds: %s GB); B_fit %d (line %d, MEASURED floor %d; margin %s GB, cap %s "
              "GB): predicted peak %s GB (%s binds, slope %s GB per request), B_fit + 1 %s GB; prefill %s / decode %s GB "
              "at B_fit; headroom %s GB; one more request adds at least the arena's %s GB; int32 limit %d; SH fits %s "
              "(beside: %s); device %s" % (
                  ARM, C, fit["prefill_line"][0], fit["prefill_line"][1], fit["points"]["prefill"],
                  fit["decode_line"][0], fit["decode_line"][1], fit["points"]["decode"], fit["rounds_projected"],
                  fit["decode_projected_GB"], fit["B_fit"], fit["B_line"], fit["floor_B"], fit["margin_GB"],
                  fit["budget_GB"], fit["peak_at_fit"], fit["binding"], fit["slope_binding"], fit["peak_next"],
                  fit["prefill_at_fit"], fit["decode_at_fit"], fit["headroom_GB"], fit["arena_next_GB"],
                  fit["int32_limit"], fit["sh_fit"], fit["sh_beside"], fit["device_at_fit"]))
        print("FIT-TABLE %s C %d: B_fit by margin (GB) %s; linearity of the prefill line (REPORT) %s" % (
            ARM, C, fit["margins"], fit["linearity"]))
    if vd:
        for line in verdict_lines(ARM, C, a.batch, vd):
            print(line)
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4m: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
