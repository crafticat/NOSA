"""Rung I4U of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), lever L5 of the
specKV-on-cached-NOSA ladder (LEDGER_model_vs_measured.md R5.5 item 5; LEDGER_rev6.md R6.6 E3): the GPU cell of the arm
L4PTFDU = L4PTFD (rung I4A) whose joint pass computes its layer-invariant tensors once per pass
(benchmarks/Efficiency/r2_i4u_hoist.py). A NEW script: r2_i4a_cell.py is not edited (its functions are imported; both
arms run run_arm_u = r2_i4a_cell.run_arm_x's steps with the runner class as a parameter). Step script:
benchmarks/Efficiency/r2_i4u_step.sh (cells i4um, i4ua, i4ub).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order given (L4PTFD before L4PTFDU: L4PTFDU's tokens are compared
with L4PTFD's), each with its own shipped prefill into the chain arena (NoCopyPrefillHook), on the inputs of i4aa / i4ab
(the same workload pin, S0, document rows, K 3, 512 steps, fused twin and body, graph, redirect, probe 16, store X 129,
--tu-gemm natural):
  SH       the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTFD   I4A (r2_i4a_lean.TentRunnerFD): the base rung, re-measured
  L4PTFDU  I4U: L4PTFD with the hoisted joint-pass tensors (r2_i4u_hoist.TentRunnerFDU); the same arena, store, passes
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN (r2_wsa_cell.arm_gates_w), -RESID,
-ATTN1, -FILL1 (r2_i3_cell.fork_gates), -LEAN (r2_i4a_cell.lean_gate); with --check-every 1 (i4um) -PRESENT, -VIS, -VB,
-TAIL, -CHAIN, -IDX, -RA; with --require-stress -STRESS, -FSTRESS; L4PTFDU-EQF (its committed tokens, check-round
fingerprints, acceptance record and state checksums = L4PTFD's in the same job, bit for bit: r2_i4a_cell.eq_gate);
L4PTFDU-COUNTS (every host count and device counter of the run = L4PTFD's); L4PTFDU-HOIST (one hoist per joint pass,
L hoisted layers per joint pass; with --check-every 1 the dropped is_next test recounted: 0). REPORT: each arm's MEM
(r2_i4a_cell.mem_record), the phases.
REGISTERED PREDICTIONS PRED_I4U_BY_B: every value None until the lead registers them with PRED_STAMP_I4U; the step script
STOPS the timing cells (i4ua, i4ub) while PRED_STAMP_I4U is None. The builder's MODEL with its derivation: the comment
block above PRED_STAMP_I4U. EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
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
import r2_i4a_cell as IA          # noqa: E402
import r2_i4a_lean as LN          # noqa: E402
import r2_i4u_hoist as HU         # noqa: E402

I3C = IA.I3C
WA, SC, BC = IA.WA, IA.SC, IA.BC
ARMS = ("L4PTFD", "L4PTFDU")
TOPK = IA.TOPK
FORK_BLOB = IA.FORK_BLOB
STRONGEST = IA.STRONGEST
BUDGET_GB = IA.BUDGET_GB
USABLE_GB = IA.USABLE_GB
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4U_BY_B (per batch) and PRED_STAMP_I4U
# (`date -u`, L33) in a commit BEFORE i4ua / i4ub (the step script stops while PRED_STAMP_I4U is None). Keys, each
# (low, high) unless noted:
#   I4U_joint_saving   L4PTFD joint phase median - L4PTFDU joint phase median (ms per round, same job)
#   I4U_joint_ratio    L4PTFDU joint median / L4PTFD joint median
#   I4U_vs_D           L4PTFDU / L4PTFD median steady committed tok/s (same job): the lever's change
#   I4U_ratio          L4PTFDU median steady committed tok/s / SH's
#   I4U_draft_ratio    L4PTFDU draft phase median / L4PTFD's (the one-row passes are L4PTFD's: about 1)
#   I4U_eq             True: L4PTFDU-EQF passes (bit for bit = L4PTFD)
#   I4U_peak_gb        the L4PTFDU arm's peak allocated GB, the PREFILL INCLUDED (r2_i4a_cell.peak_gb)
# THE BUILDER'S MODEL (an input for the lead, not a registration). Steady rounds only (round 0 excluded, as the cells'
# steady statistics; this arm captures nothing, so no capture round, L61, applies).
#   Ops removed per joint pass (CODE + CPU counter, tests_cpu/test_i4u.py COUNT): r2_i4u_hoist.saved_per_pass(32) =
#     42 x 31 + SAVED_LAYER0 = 1,312 aten compute ops; every one is a small elementwise / fill / copy op on (B,), (B, R),
#     (B, 64), (1, B, 1) or (H, B, R x 64) tensors (no GEMM, no sort, no gather, no fork call is removed). By kind per
#     layer >= 1 (the counter): and 8, not 5, add 6, remainder 6, clamp 4, floor_divide 2, full_like 2, eq / ge / gt /
#     lt / or / sub / where / stack / copy_ 1 each.
#   Device cost per removed op (MEASURED, I4C: LEDGER_rev6.md R6.6 / cpu-logs/ana_rev6_s2.txt, L4PTFDP joint pass round
#     112): the SMALL class of the "union" label 7.957 ms / 2,720 kernels = 2.93 us (c3a, C 128 B 216), 8.361 / 2,720 =
#     3.07 us (c4b, C 73 B 318); the joint pass's kernels below 5 us: 17.59 ms / 6,152 = 2.86 us. Band per op 2.0-3.1 us.
#   So the device saving is 1,312 x (2.0 .. 3.1) us = 2.6 .. 4.1 ms per joint pass = per round (one joint pass per
#     round). Host side (eager arm): about 1,312 fewer launches; the union / write / chain labels are device 0.80 of
#     elapsed at C 128 (0.87 at C 73), so a part of the host saving shows: up to device / 0.80 = 5.1 ms (C 128).
#   I4U_joint_saving  C 128: (2.0, 6.0) ms; C 73: (1.5, 5.5) ms (the C 73 joint pass is device bound, i4cb 0.9958).
#   I4U_joint_ratio   L4PTFD joint (MEASURED rev 6 R6.3: d + 3 j1) 205.5 ms (C 128), 384.7 ms (C 73): 1 - (2 .. 6) /
#                     205.5 = (0.970, 0.990); 1 - (1.5 .. 5.5) / 384.7 = (0.985, 0.996).
#   I4U_vs_D          the round 354.3 ms (C 128: 3 x 116.087 + 6.066) and 583.1 ms (C 73: 3 x 191.537 + 8.493):
#                     354.3 / (354.3 - (2 .. 6)) = (1.005, 1.018); 583.1 / (583.1 - (1.5 .. 5.5)) = (1.002, 1.010).
#   I4U_ratio         L4PTFD / SH MEASURED 0.521 (C 128) and 0.598 (C 73) x I4U_vs_D, widened by the cross-job SH
#                     spread: (0.515, 0.535) and (0.590, 0.610).
#   I4U_draft_ratio   (0.98, 1.02): the one-row passes are L4PTFD's code (COUNT: the one-row pass counts are equal).
#   I4U_eq            True: the CPU runs are bit for bit (EQUAL), no arithmetic op changed (only layer-invariant
#                     values computed once, a constant False dropped, two fills replaced by 0-dim constants).
#   I4U_peak_gb       the hoist adds < 1 MB (the (H, B, R x 64) bool fullv: 0.44 MB at B 216): L4PTFD's i4aa 83.891 /
#                     i4ab 77.685: (83.85, 83.95) and (77.60, 77.80).
PRED_STAMP_I4U = "2026-10-29T18:16:01Z"
PRED_KEYS = ("I4U_joint_saving", "I4U_joint_ratio", "I4U_vs_D", "I4U_ratio", "I4U_draft_ratio", "I4U_eq",
             "I4U_peak_gb")
# The lead's registration (before any I4U cell ran): the builder's MODEL bands (hand-over report section 5).
PRED_I4U_BY_B = {
    216: dict(I4U_joint_saving=(2.0, 6.0), I4U_joint_ratio=(0.970, 0.990), I4U_vs_D=(1.005, 1.018),
              I4U_ratio=(0.515, 0.535), I4U_draft_ratio=(0.98, 1.02), I4U_eq=True, I4U_peak_gb=(83.85, 83.95)),
    318: dict(I4U_joint_saving=(1.5, 5.5), I4U_joint_ratio=(0.985, 0.996), I4U_vs_D=(1.002, 1.010),
              I4U_ratio=(0.590, 0.610), I4U_draft_ratio=(0.98, 1.02), I4U_eq=True, I4U_peak_gb=(77.60, 77.80)),
}


def log(msg):
    print("[r2_i4u %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDU", help="a subset of %s, in run order (SH runs first; L4PTFD "
                                                             "before L4PTFDU)" % ",".join(ARMS))
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
                    help="0 = the probe rounds; 1 = every round and the union / fork checks on (the miniature)")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if "L4PTFD" in a.arms and "L4PTFDU" in a.arms and a.arms.index("L4PTFD") > a.arms.index("L4PTFDU"):
        ap.error("--arms: L4PTFD before L4PTFDU (L4PTFDU-EQF compares with L4PTFD's tokens)")
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1:
        ap.error("--probe-every >= 2, --check-every >= 0, journal >= 1")
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(TOPK, a.spec):
        ap.error("--fork-store %d: below the bound 1 + (K - 1) x 64 = %d (0 = the bound)"
                 % (a.fork_store, I3C.KA.store_slots(TOPK, a.spec)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def eq_gate(fdu, fd, steps: int):
    """L4PTFDU-EQF (MUST): r2_i4a_cell.eq_gate against L4PTFD's record (the same four parts, bit for bit)."""
    n, m, ok, d = IA.eq_gate("L4PTFDU", fdu, fd, steps, True)
    return n, m, ok, d.replace("L4PTF's", "L4PTFD's").replace("no L4PTF record", "no L4PTFD record")


COUNT_KEYS = ("tent", "union", "round_blocks", "draft_skipped", "draft_calls", "lean", "rows_calls", "rows_draft",
              "tokens_acct", "round_committed", "round_accepted", "acc_hist", "verify_hist", "count")


def counts_record(rd) -> dict:
    """The counters of a run (read()) that the hoist must leave unchanged: the host counts and the device counters."""
    return {k: rd.get(k) for k in COUNT_KEYS}


def counts_gate(arm, c, c_ref):
    """<arm>-COUNTS (MUST): counts_record(L4PTFDU) = counts_record(L4PTFD), every key and sub-key."""
    if c_ref is None:
        return ("%s-COUNTS" % arm, True, False, "no L4PTFD record in this job")
    bad = []
    for k in COUNT_KEYS:
        a, b = c.get(k), c_ref.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            bad += ["%s.%s: %s vs %s" % (k, s, a.get(s), b.get(s)) for s in sorted(set(a) | set(b), key=str)
                    if a.get(s) != b.get(s)]
        elif a != b:
            bad.append("%s: differs" % k)
    tc, uc = (c.get("tent") or {}).get("calls") or {}, (c.get("union") or {}).get("calls") or {}
    return ("%s-COUNTS" % arm, True, not bad,
            "%d keys compared with L4PTFD's; differences %d %s; host counts chain %s, resid %s, attn %s, fills %s, "
            "gather_host %s" % (len(COUNT_KEYS), len(bad), bad[:4], tc.get("chain"), tc.get("resid"), uc.get("attn"),
                                uc.get("fills"), uc.get("gather_host")))


def hoist_gate(arm, rd, L: int, check: bool):
    """<arm>-HOIST (MUST): one hoist per joint pass (passes = joint passes > 0), every joint-pass layer hoisted
    (layers = L x joint passes); with the checks on (i4um) the dropped is_next test recounted: next_seen 0."""
    h = rd.get("hoist") or {}
    jp = ((rd.get("tent") or {}).get("calls") or {}).get("joint_passes", 0)
    ok = jp > 0 and h.get("passes") == jp and h.get("layers") == L * jp
    if check:
        ok = ok and h.get("next_seen") == 0
    return ("%s-HOIST" % arm, True, bool(ok),
            "hoist passes %s (joint passes %s); hoisted layers %s (L x joint passes %d); is_next recounted %s (%s)" % (
                h.get("passes"), jp, h.get("layers"), L * jp, h.get("next_seen"),
                "checked, must be 0" if check else "not checked: the checks are off"))


def verdicts(sh, sp, B: int, eqf=None):
    """PRED_I4U_BY_B[B] against the arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN without
    L4PTFDU."""
    P = PRED_I4U_BY_B.get(B) or {k: None for k in PRED_KEYS}
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
    d, u = (sp or {}).get("L4PTFD"), (sp or {}).get("L4PTFDU")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    if u is None:
        out.append(("L4PTFDU-RUN", "NOT_RUN", "the arm did not run"))
        return out
    band("I4U-JOINT-SAVING", (ph(d, "joint") - ph(u, "joint")) if (ph(d, "joint") is not None and
                                                                   ph(u, "joint") is not None) else None,
         P.get("I4U_joint_saving"))
    band("I4U-JOINT-RATIO", (ph(u, "joint") / ph(d, "joint")) if (ph(d, "joint") and ph(u, "joint")) else None,
         P.get("I4U_joint_ratio"))
    band("I4U-VS-D", (med(u) / med(d)) if (med(u) and med(d)) else None, P.get("I4U_vs_D"))
    band("I4U-RATIO", (med(u) / shm) if (med(u) and shm) else None, P.get("I4U_ratio"))
    band("I4U-DRAFT-RATIO", (ph(u, "draft") / ph(d, "draft")) if (ph(d, "draft") and ph(u, "draft")) else None,
         P.get("I4U_draft_ratio"))
    band("I4U-PEAK-GB", IA.peak_gb(u), P.get("I4U_peak_gb"))
    want = P.get("I4U_eq")
    out.append(("I4U-EQ", "REPORT" if (want is None or eqf is None) else ("CONFIRMED" if eqf == bool(want) else
                                                                          "REFUTED"),
                "L4PTFDU-EQF passed %s; registered %s" % (eqf, want)))
    out.append(("I4U-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, u.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_u(model, ids, inputs, steps: int, clock, opt, FORK, hoist: bool):
    """r2_i4a_cell.run_arm_x's steps, line for line, with the runner class as the only choice (hoist False: L4PTFD =
    r2_i4a_lean's TentRunnerFD; True: L4PTFDU = r2_i4u_hoist's TentRunnerFDU) and the state checksum after the run
    (EQF)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    lean = LN.lean_classes(STF)
    Runner = HU.hoist_classes(STF, lean)[1] if hoist else lean[1]
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
    check = opt["check_every"] == 1
    ce = opt["check_every"] if opt["check_every"] > 0 else opt["probe_every"]
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
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
               joint_phases=I3C.fork_phase_summary(phases, layers), fa_check=runner.vs.fa_check,
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4), cb0_GB=round(arena.chain_bytes() / 1e9, 4)),
               compile=I3C._compile_report(FORK), state=IA.state_checksum(runner))
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
            gates.append(IA.mem_record("SH", "prefill + decode", pa))
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
                       prefill_hook=a.prefill_hook)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_u(model, ids, inputs, a.steps, clock, opt, FORK, hoist=(arm == "L4PTFDU"))
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(res["read"], L), fork=I3C.fork_record(res["read"]),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=res["read"].get("lean"), hoist=res["read"].get("hoist"))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = IA.peak_gb(rec)
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(IA.lean_gate(arm, dict(res["read"], K=a.spec), L))
                gates.append(IA.mem_record(arm, res["mem"], pa))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=IA.acct_record(res["read"]), state=res["state"], counts=counts_record(res["read"]))
                if arm == "L4PTFD":
                    keep["L4PTFD"] = cpu
                else:
                    gates.append(hoist_gate(arm, res["read"], L, check_t))
                    ref = keep.get("L4PTFD")
                    gates.append(counts_gate(arm, cpu["counts"], ref["counts"] if ref else None))
                    g = eq_gate(cpu, ref, a.steps)
                    eqf = bool(g[2])
                    gates.append(g)
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch, eqf)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_I4U, by_batch=PRED_I4U_BY_B),
                joint_hoist=dict(saved_per_layer=HU.SAVED_PER_LAYER, saved_layer0=HU.SAVED_LAYER0,
                                 saved_per_pass=HU.saved_per_pass(L)),
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-17s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
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
              "round median %s ms; phases %s; draft step %s ms = %s SH; joint %s SH steps; calls %s; lean %s; hoist %s; "
              "tokens equal %.6f (first %s); peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"], r["draft_step_sh"],
                  r["joint_sh_steps"], r["calls"], r.get("lean"), r.get("hoist"), r["tokens"]["equal_fraction"],
                  r["tokens"]["first"], r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        print("REPORT %s fork: %s; joint-pass phases (probe) %s; store %s" % (
            name, r["fork"], r["joint_phases_fork"], r["store"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4u: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
