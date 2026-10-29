"""Rung I4K of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU cell of the
COMBINED arm L4PTFDGKW = L4PTFD + the store keep + the window loads from the kept store (rung I4D, arm L4PTFKW) + the
per-layer CUDA graphs of the joint pass's union + fork section (rung I4G, arm L4PTFDG), with the keep's bookkeeping
inside the graphs (benchmarks/Efficiency/r2_i4k_combo.py: THE SPLIT). A NEW script: r2_i4a_cell.py, r2_i4d_cell.py and
r2_i4g_cell.py are not edited (their functions are imported). Step script: benchmarks/Efficiency/r2_i4k_step.sh (cells
i4km, i4ka, i4kb).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order L4PTFD, L4PTFDG, L4PTFKW, L4PTFDGKW (a subset; every arm's
tokens are compared with L4PTFD's), each with its own shipped prefill into the chain arena (NoCopyPrefillHook), on the
inputs of i4aa / i4ab (the same workload pin, S0, document rows, K 3, fused twin and body, graph, redirect, probe 16,
store X 129, --tu-gemm natural):
  SH         the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTFD     I4A: r2_i4a_lean's TentRunnerFD (with --store-acct 1: r2_i4d_keep's TentRunnerFK, keep off, acct on)
  L4PTFDG    I4G: r2_i4g_graph's TentRunnerFDG (the section from per-layer CUDA graphs)
  L4PTFKW    I4D: r2_i4d_keep's TentRunnerFK (keep, kwin; acct with --store-acct 1; kcheck with --check-every 1)
  L4PTFDGKW  I4K: r2_i4k_combo's TentRunnerFDGK (the keep options of L4PTFKW + the graphs)
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN, -RESID, -ATTN1, -FILL1, -LEAN; with
--check-every 1 -PRESENT, -VIS, -VB, -TAIL, -CHAIN, -IDX, -RA; with --require-stress -STRESS, -FSTRESS; every arm but
L4PTFD: <arm>-EQF (committed tokens, check-round fingerprints, acceptance record, state checksums = L4PTFD's, bit for
bit); the graph arms: <arm>-GRAPH, <arm>-COUNTS, <arm>-MEM (r2_i4g_cell's gates; L4PTFDGKW's COUNTS against L4PTFKW when
it ran, else against L4PTFD without the meter's round_blocks, which the keep changes by design); the keep arms: -KEEP,
with --check-every 1 -CAP, with --store-acct 1 -ACCT and ACCT-EQ-<arm> (r2_i4d_cell's gates); L4PTFDGKW-KEEP-EQ (its
keep record per pass = L4PTFKW's, when L4PTFKW ran). REPORT: SH / L4PTFD / L4PTFKW MEM; L4PTFDGKW-KEEP-REF (its keep
totals against the i4da / i4db records at the same C, B and steps); the phases, the store, the graphs.
REGISTERED PREDICTIONS PRED_I4K_BY_B: every value None until the lead registers them with PRED_STAMP_I4K (a None key is
REPORT); the step script STOPS the timing cells (i4ka, i4kb) while PRED_STAMP_I4K is None. EXIT: the number of failed
MUST gates (0-19); 20 refused; 30 an exception outside an arm.
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
import r2_i4a_cell as A4          # noqa: E402
import r2_i4a_lean as LN          # noqa: E402
import r2_i4d_cell as DC          # noqa: E402
import r2_i4d_keep as KP          # noqa: E402
import r2_i4g_cell as GC          # noqa: E402
import r2_i4g_graph as GR         # noqa: E402
import r2_i4k_combo as CB         # noqa: E402

I3C = A4.I3C
WA, SC, BC = A4.WA, A4.SC, A4.BC
ARMS = ("L4PTFD", "L4PTFDG", "L4PTFKW", "L4PTFDGKW")
GRAPH_ARMS = ("L4PTFDG", "L4PTFDGKW")
KEEP_ARMS = ("L4PTFKW", "L4PTFDGKW")
TOPK = A4.TOPK
FORK_BLOB = A4.FORK_BLOB
STRONGEST = A4.STRONGEST
BUDGET_GB = A4.BUDGET_GB
USABLE_GB = A4.USABLE_GB
MEM_MUST_GB = GC.MEM_MUST_GB       # the graph arms' MEM gate: peak allocated with the prefill <= 84.0 GB
COUNT_KEYS = GC.COUNT_KEYS
KEEP_EQ_KEYS = ("passes", "listed", "fetched", "win_hits", "repeat", "win_loads", "win_prev", "victims_alive", "check")
# REPORT only: the keep totals of L4PTFKW in the I4D timing cells (MEASURED, the same inputs, C, B and steps; the run is
# deterministic: L4PTFD's acceptance record is equal across i4aa, i4da and i4ga). (B, pool, steps) -> totals.
KEEP_REF = {
    (216, 65, 512): dict(listed=1869571, fetched=1375827, win_hits=596654,
                         src="/mnt/beegfs/ojerbi/r2-runs/spec/i4d_d3a_2195575_i4da.step/stdout.txt line 76"),
    (318, 10, 512): dict(listed=11538293, fetched=9679310, win_hits=3084760,
                         src="/mnt/beegfs/ojerbi/r2-runs/spec/i4d_d4b_2195576_i4db.step/stdout.txt (L4PTFKW-KEEP)"),
}
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4K_BY_B (per batch) and PRED_STAMP_I4K
# (`date -u`, L33) in a commit BEFORE i4ka / i4kb (the step script stops while PRED_STAMP_I4K is None). Keys, each
# (low, high) unless noted; units in brackets; X = L4PTFDGKW:
#   I4K_joint_saving   [ms per round] L4PTFD joint phase median - X's (same job)
#   I4K_joint_ratio    [x] X joint median / L4PTFD joint median
#   I4K_vs_D           [x] X / L4PTFD median steady committed tok/s (same job)
#   I4K_vs_G           [x] X / L4PTFDG median steady committed tok/s (i4ka: L4PTFDG runs)
#   I4K_vs_KW          [x] X / L4PTFKW median steady committed tok/s (i4kb: L4PTFKW runs)
#   I4K_ratio          [x SH] X median steady committed tok/s / SH's
#   I4K_wait_saving    [ms per round] L4PTFD host-gather wait median - X's (kv_at_verify, the probe rounds)
#   I4K_draft_ratio    [x] X draft phase median / L4PTFD's (the draft passes are not changed: about 1)
#   I4K_eq             [True] X-EQF passes (= L4PTFD bit for bit)
#   I4K_above_best     [True / False] the rung falsifier: X's median above EVERY other arm's median in the job
#   I4K_peak_gb        [GB] X's peak allocated GB, the prefill included (the MEM gate's value)
#   I4K_pool_mb        [MB] X's graph pool segments after the run
#   I4K_capture_s      [s] X's summed capture time (host clock)
# The builder's MODEL (the hand-over report, with the derivation) is an input for the lead, not a registration.
PRED_STAMP_I4K = "2026-10-29T14:57:28Z"
PRED_KEYS = ("I4K_joint_saving", "I4K_joint_ratio", "I4K_vs_D", "I4K_vs_G", "I4K_vs_KW", "I4K_ratio", "I4K_wait_saving",
             "I4K_draft_ratio", "I4K_eq", "I4K_above_best", "I4K_peak_gb", "I4K_pool_mb", "I4K_capture_s")
# The lead's registration (before any I4K cell ran): the builder's MODEL bands (hand-over report section 5).
# I4K_above_best is None at both batches: the model puts the arm level with L4PTFDG (B 216) and L4PTFKW (B 318).
PRED_I4K_BY_B = {
    216: dict(I4K_joint_saving=(2.0, 9.5), I4K_joint_ratio=(0.954, 0.990), I4K_vs_D=(1.006, 1.028),
              I4K_vs_G=(0.986, 1.005), I4K_vs_KW=None, I4K_ratio=(0.522, 0.540), I4K_wait_saving=(4.0, 9.0),
              I4K_draft_ratio=(0.98, 1.02), I4K_eq=True, I4K_above_best=None, I4K_peak_gb=(83.85, 83.95),
              I4K_pool_mb=(5, 150), I4K_capture_s=(0.5, 20)),
    318: dict(I4K_joint_saving=(17.0, 31.0), I4K_joint_ratio=(0.919, 0.956), I4K_vs_D=(1.030, 1.056),
              I4K_vs_G=None, I4K_vs_KW=(0.994, 1.015), I4K_ratio=(0.612, 0.638), I4K_wait_saving=(30.0, 48.0),
              I4K_draft_ratio=(0.98, 1.02), I4K_eq=True, I4K_above_best=None, I4K_peak_gb=(77.60, 77.80),
              I4K_pool_mb=(5, 200), I4K_capture_s=(0.5, 20)),
}


def log(msg):
    print("[r2_i4k %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDG,L4PTFDGKW", help="a subset of %s, in this order (SH runs "
                                                                       "first)" % ",".join(ARMS))
    ap.add_argument("--store-acct", type=int, default=0, choices=(0, 1),
                    help="1: the accounting counters in L4PTFD (then r2_i4d_keep with keep off) and the keep arms")
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
    if a.arms != [x for x in ARMS if x in a.arms]:
        ap.error("--arms: in the order %s (EQF compares with L4PTFD; L4PTFDGKW's COUNTS / KEEP-EQ with L4PTFKW)"
                 % ",".join(ARMS))
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
    """(the runner's kind, its keyword arguments): FD / FK as r2_i4d_cell.runner_choice; FDG = r2_i4g_graph's runner;
    FDGK = r2_i4k_combo's runner with L4PTFKW's keep options."""
    if arm == "L4PTFDG":
        return "FDG", {}
    if arm == "L4PTFDGKW":
        return "FDGK", DC.runner_choice("L4PTFKW", acct, check)[1]
    return DC.runner_choice(arm, acct, check)


def runner_class(kind: str, STF):
    lean = LN.lean_classes(STF)
    return dict(FD=lambda: lean[1], FK=lambda: KP.keep_classes(STF)[1], FDG=lambda: GR.graph_classes(STF, lean)[1],
                FDGK=lambda: CB.combo_classes(STF)[3])[kind]()


# ------------------------------------------------------------------ pure parts (CPU-tested)
def eq_gate(arm, f, fd, steps: int):
    """<arm>-EQF (MUST): r2_i4a_cell.eq_gate against L4PTFD's record (the same four parts, bit for bit)."""
    if fd is None:
        return ("%s-EQF" % arm, True, False, "no L4PTFD record in this job")
    n, m, ok, d = A4.eq_gate(arm, f, fd, steps, True)
    return n, True, ok, d.replace("L4PTF's", "L4PTFD's")


def counts_gate(arm, c, c_ref, ref: str, skip=()):
    """<arm>-COUNTS (MUST): counts_record(arm) = counts_record(ref), every key of r2_i4g_cell.COUNT_KEYS except
    `skip` (declared in the detail)."""
    if c_ref is None:
        return ("%s-COUNTS" % arm, True, False, "no %s record in this job" % ref)
    bad = []
    keys = [k for k in COUNT_KEYS if k not in skip]
    for k in keys:
        a, b = c.get(k), c_ref.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            bad += ["%s.%s: %s vs %s" % (k, s, a.get(s), b.get(s)) for s in sorted(set(a) | set(b), key=str)
                    if a.get(s) != b.get(s)]
        elif a != b:
            bad.append("%s: differs" % k)
    tc, uc = (c.get("tent") or {}).get("calls") or {}, (c.get("union") or {}).get("calls") or {}
    return ("%s-COUNTS" % arm, True, not bad,
            "%d keys compared with %s's (not compared: %s); differences %d %s; host counts chain %s, resid %s, attn %s, "
            "fills %s, gather_host %s" % (len(keys), ref, list(skip) or "none", len(bad), bad[:4], tc.get("chain"),
                                          tc.get("resid"), uc.get("attn"), uc.get("fills"), uc.get("gather_host")))


def keep_eq_gate(arm, kp, kp_ref):
    """<arm>-KEEP-EQ (MUST): the keep record per pass (KEEP_EQ_KEYS present in either record) = L4PTFKW's: the graph
    replays updated the map, the ages and the pass counters as the eager keep did."""
    if not kp_ref:
        return ("%s-KEEP-EQ" % arm, True, False, "no L4PTFKW keep record in this job")
    if not kp:
        return ("%s-KEEP-EQ" % arm, True, False, "no keep record")
    keys = [k for k in KEEP_EQ_KEYS if k in kp or k in kp_ref]
    bad = [k for k in keys if kp.get(k) != kp_ref.get(k)]
    return ("%s-KEEP-EQ" % arm, True, not bad and (kp.get("listed_total") or 0) > 0,
            "per pass %s compared with L4PTFKW's; differ %s; totals listed %s / %s, fetched %s / %s, window hits %s / %s"
            % (keys, bad, kp.get("listed_total"), kp_ref.get("listed_total"), kp.get("fetched_total"),
               kp_ref.get("fetched_total"), kp.get("win_hits_total"), kp_ref.get("win_hits_total")))


def keep_ref_gate(arm, kp, B: int, pool: int, steps: int):
    """<arm>-KEEP-REF (REPORT): the keep totals against the I4D timing cell's L4PTFKW record at the same B, pool,
    steps (KEEP_REF); no reference: REPORT without a verdict."""
    ref = KEEP_REF.get((int(B), int(pool), int(steps)))
    kp = kp or {}
    got = dict(listed=kp.get("listed_total"), fetched=kp.get("fetched_total"), win_hits=kp.get("win_hits_total"))
    if ref is None:
        return ("%s-KEEP-REF" % arm, False, True, "no reference at B %s pool %s steps %s; totals %s" % (B, pool, steps,
                                                                                                     got))
    ok = all(got[k] == ref[k] for k in got)
    return ("%s-KEEP-REF" % arm, False, ok, "totals %s; reference %s (%s)" % (got, {k: ref[k] for k in got},
                                                                              ref["src"]))


def verdicts(sh, sp, B: int, eqf=None):
    """PRED_I4K_BY_B[B] against the arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN without
    L4PTFDGKW. eqf: L4PTFDGKW-EQF's passed flag."""
    P = PRED_I4K_BY_B.get(B) or {k: None for k in PRED_KEYS}
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

    def kv(r, name):
        return ((r or {}).get("kv_at_verify") or {}).get(name)

    def ratio(a, b):
        return (a / b) if (a and b) else None
    sp = sp or {}
    d, g, kw, x = (sp.get(a) for a in ARMS)
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    for name, r in (("I4G-VS-D", g), ("I4DW-VS-D", kw)):                    # the single levers, re-measured
        if r is not None:
            out.append((name, "REPORT", "%s (median tok/s / L4PTFD's, this job)" % (
                None if ratio(med(r), med(d)) is None else round(ratio(med(r), med(d)), 4))))
    if x is None:
        out.append(("L4PTFDGKW-RUN", "NOT_RUN", "the arm did not run"))
        return out
    gr = x.get("graph") or {}
    band("I4K-JOINT-SAVING", (ph(d, "joint") - ph(x, "joint")) if (ph(d, "joint") is not None and
                                                                   ph(x, "joint") is not None) else None,
         P.get("I4K_joint_saving"))
    band("I4K-JOINT-RATIO", ratio(ph(x, "joint"), ph(d, "joint")), P.get("I4K_joint_ratio"))
    band("I4K-VS-D", ratio(med(x), med(d)), P.get("I4K_vs_D"))
    band("I4K-VS-G", ratio(med(x), med(g)), P.get("I4K_vs_G"))
    band("I4K-VS-KW", ratio(med(x), med(kw)), P.get("I4K_vs_KW"))
    band("I4K-RATIO", ratio(med(x), shm), P.get("I4K_ratio"))
    band("I4K-WAIT-SAVING", (kv(d, "wait_ms_median") - kv(x, "wait_ms_median")) if (
        kv(d, "wait_ms_median") is not None and kv(x, "wait_ms_median") is not None) else None, P.get("I4K_wait_saving"))
    band("I4K-DRAFT-RATIO", ratio(ph(x, "draft"), ph(d, "draft")), P.get("I4K_draft_ratio"))
    band("I4K-PEAK-GB", A4.peak_gb(x), P.get("I4K_peak_gb"))
    band("I4K-POOL-MB", (gr["pool_bytes"] / 1e6) if gr.get("pool_bytes") is not None else None, P.get("I4K_pool_mb"))
    band("I4K-CAPTURE-S", gr.get("capture_s"), P.get("I4K_capture_s"))
    flag("I4K-EQ", eqf, P.get("I4K_eq"), "L4PTFDGKW-EQF passed %s; registered %s" % (eqf, P.get("I4K_eq")))
    others = {a: med(r) for a, r in zip(ARMS[:3], (d, g, kw)) if r is not None and med(r)}
    above = (all(med(x) > v for v in others.values()) if (med(x) and others) else None)
    flag("I4K-ABOVE-BEST", above, P.get("I4K_above_best"), "L4PTFDGKW %s vs %s tok/s (the rung falsifier: not above "
         "every other arm of the job); registered %s" % (med(x), others, P.get("I4K_above_best")))
    out.append(("I4K-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, x.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_k4(model, ids, inputs, steps: int, clock, opt, FORK, arm: str, acct: bool):
    """r2_i4a_cell.run_arm_x's steps, line for line (as r2_i4d_cell.run_arm_k / r2_i4g_cell.run_arm_g), with the runner
    of runner_choice(arm, acct, check), the state checksum after the run (EQF) and read()'s graph / keep reports (before
    detach: the graph pool still exists)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, acct, check)
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


def arm_gates(arm, rd, rec, cpu, keep, check: bool, acct: bool, B: int, pool: int, steps: int, L: int, pa):
    """The arm-specific gates after the common ones (module docstring): [(name, must, passed, detail)]."""
    g = []
    if arm in GRAPH_ARMS:
        g.append(GC.mem_gate(arm, rec))
        g.append(GC.graph_gate(arm, rd, L))
        if arm == "L4PTFDG":
            ref = keep.get("L4PTFD")
            g.append(counts_gate(arm, cpu["counts"], ref["counts"] if ref else None, "L4PTFD"))
        elif "L4PTFKW" in keep:
            g.append(counts_gate(arm, cpu["counts"], keep["L4PTFKW"]["counts"], "L4PTFKW"))
        else:
            ref = keep.get("L4PTFD")
            g.append(counts_gate(arm, cpu["counts"], ref["counts"] if ref else None, "L4PTFD", skip=("round_blocks",)))
    else:
        g.append(A4.mem_record(arm, rec["mem_arena"], pa))
    if acct and arm != "L4PTFDG":
        g.append(DC.acct_gate(arm, rd))
    if arm in KEEP_ARMS:
        g.append(DC.keep_gate(arm, rd, check))
        if check:
            g.append(DC.cap_gate(arm, rd))
        if acct and "L4PTFD" in keep:
            g.append(DC.acct_eq_gate(keep["L4PTFD"]["keep"], rd.get("keep"), arm))
    if arm == "L4PTFDGKW":
        if "L4PTFKW" in keep:
            g.append(keep_eq_gate(arm, rd.get("keep"), keep["L4PTFKW"]["keep"]))
        g.append(keep_ref_gate(arm, rd.get("keep"), B, pool, steps))
    if arm != "L4PTFD":
        g.append(eq_gate(arm, cpu, keep.get("L4PTFD"), steps))
    return g


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
    gates, sh, sp, inputs, keep = [], None, {}, None, {}
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
    eqf = None
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
                res = run_arm_k4(model, ids, inputs, a.steps, clock, opt, FORK, arm, bool(a.store_acct))
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), graph=rd.get("graph"),
                           capture_rounds=GC.capture_rounds(rd), runner=res["runner_kind"],
                           runner_kw=res["runner_kw"])
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = A4.peak_gb(rec)
                rec["store_record"] = DC.acct_summary(rd.get("keep"), bb)
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (rd["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(A4.lean_gate(arm, dict(rd, K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=A4.acct_record(rd), state=res["state"], counts=GC.counts_record(rd), keep=rd.get("keep"))
                ag = arm_gates(arm, rd, rec, cpu, keep, check_t, bool(a.store_acct), a.batch, a.pool, a.steps, L, pa)
                gates.extend(ag)
                if arm == "L4PTFDGKW":
                    eqf = next((bool(x[2]) for x in ag if x[0] == "L4PTFDGKW-EQF"), None)
                keep[arm] = cpu
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch, eqf)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, store_acct=a.store_acct,
                fork_store=a.fork_store, fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store),
                prefill_hook=a.prefill_hook, tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks,
                spec_twin=a.spec_twin, spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_I4K, by_batch=PRED_I4K_BY_B),
                combo=dict(captured_keep_ops=CB.CAPTURED_KEEP_OPS, eager_keep_ops=CB.EAGER_KEEP_OPS,
                           eager_pass_ops=CB.EAGER_PASS_OPS, skipped_T=GR.SKIPPED_T, skipped_vs=GR.SKIPPED_VS,
                           eager_ok=GR.EAGER_OK, mem_must_GB=MEM_MUST_GB, keep_ref=KEEP_REF.get(
                               (a.batch, a.pool, a.steps))),
                keep=dict(extra_ops_per_layer=KP.KEEP_EXTRA_OPS, block_bytes_head=bb, key_bits=32),
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-20s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
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
              "tokens equal %.6f (first %s); peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"], r["draft_step_sh"],
                  r["joint_sh_steps"], r["calls"], r.get("lean"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        print("REPORT %s store: runner %s %s; NEW host blocks per round %s, host-gather wait median %s ms; record %s" % (
            name, r["runner"], r["runner_kw"], r["kv_at_verify"].get("new_blocks_per_round"),
            r["kv_at_verify"].get("wait_ms_median"),
            {k: v for k, v in (r["store_record"] or {}).items() if k not in ("listed", "fetched", "repeat")}))
        print("REPORT %s fork: joint-pass phases (probe) %s" % (name, r["joint_phases_fork"]))
        if r.get("graph"):
            gr = r["graph"]
            print("REPORT %s graphs: captures %s, warm %s, replays %s, eager %s %s; pool %s MB; static %s MB; capture "
                  "%s s; capture rounds (round, joint ms) %s" % (
                      name, gr.get("captures"), gr.get("warm"), gr.get("replays"), gr.get("eager"),
                      gr.get("eager_why"), round(gr["pool_bytes"] / 1e6, 3) if gr.get("pool_bytes") is not None
                      else None, round(gr.get("static_bytes", 0) / 1e6, 3), gr.get("capture_s"),
                      r.get("capture_rounds")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4k: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
