"""Rung I4A of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 1 of
KS/analysis/plan/LEDGER_model_vs_measured.md revision 5 (the draft phase): the GPU cell of the arm L4PTFD = L4PTF (rung
I3) with the LEAN ONE-ROW PASS (benchmarks/Efficiency/r2_i4a_lean.py). A NEW script: r2_i3_cell.py is not edited (its
functions are imported: the L4PTF arm runs r2_i3_cell.run_arm_f unchanged). Step script:
benchmarks/Efficiency/r2_i4a_step.sh (cells i4am, i4aa, i4ab). Docs: docs/NOTES_i4a.md.

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order given (L4PTF before L4PTFD: L4PTFD's tokens are compared
with L4PTF's), each with its own shipped prefill into the chain arena (NoCopyPrefillHook), on the inputs of i3a / i3b
(the same workload pin, S0, document rows, K 3, 512 steps, fused twin and body, graph, redirect, probe 16, store X 129,
--tu-gemm natural):
  SH      the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTF   I3, r2_i3_cell.run_arm_f unchanged
  L4PTFD  I4A: L4PTF with the lean one-row pass (r2_i4a_lean.lean_classes(nosi.spec_tentative_fork)); the same arena,
          store, joint pass and memory
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN (r2_wsa_cell.arm_gates_w), -RESID,
-ATTN1, -FILL1 (r2_i3_cell.fork_gates); with --check-every 1 (i4am) -PRESENT, -VIS, -VB, -TAIL, -CHAIN, -IDX, -RA; with
--require-stress -STRESS, -FSTRESS; L4PTFD-LEAN (every one-row pass took the lean path: lean layers = L x lean passes,
lean passes = draft steps (the prelude's K included) + mixed steps + one-row J slots); L4PTFD-EQF (L4PTFD's committed tokens and
its check-round fingerprints = L4PTF's in the same job, bit for bit: the lean pass computes L4PTF's values; MUST with
--eq-must 1 (i4am), REPORT otherwise). REPORT: MEM (each arm's peak against 84 GB, the prefill included), the phases.
REGISTERED PREDICTIONS PRED_I4A_BY_B: every value None until the lead registers them with PRED_STAMP_I4A; the step script
STOPS the timing cells (i4aa, i4ab) while PRED_STAMP_I4A is None. Inferred bands with their arithmetic: docs/NOTES_i4a.md
section 5. EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
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
import r2_i3_cell as I3C          # noqa: E402
import r2_i4a_lean as LN          # noqa: E402

WA, WB, SC, BC, IA = I3C.WA, I3C.WB, I3C.SC, I3C.BC, I3C.IA
ARMS = ("L4PTF", "L4PTFD")
TOPK = I3C.TOPK
FORK_BLOB = I3C.FORK_BLOB
STRONGEST = I3C.STRONGEST
BUDGET_GB = I3C.BUDGET_GB
USABLE_GB = 84.25                  # the usable device ceiling (lesson L59; SCORING_wsc.md:52-54)
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4A_BY_B (per batch) and PRED_STAMP_I4A
# (`date -u`, L33) in a commit BEFORE i4aa / i4ab (the step script stops while PRED_STAMP_I4A is None). Keys, each
# (low, high) unless noted:
#   I4A_ratio          L4PTFD median steady committed tok/s / SH's steady tok/s
#   I4A_vs_I3          L4PTFD / L4PTF median steady committed tok/s (same job): the lever's change
#   I4A_draft_step_sh  L4PTFD draft phase median / (K - 1) / SH's steady step ms: one lean draft step in SH steps
#   I4A_draft_saving   L4PTF draft phase median - L4PTFD draft phase median (ms per round, same job)
#   I4A_joint_ratio    L4PTFD joint median / L4PTF joint median (the joint pass is L4PTF's: about 1)
#   I3_ratio           L4PTF / SH in this job (rung I3, re-measured)
#   I4A_agree          a floor: the fraction of L4PTFD's committed tokens equal to SH's
#   I4A_eq_I3          True: L4PTFD's committed tokens = L4PTF's (all of them)
#   I4A_above_I3       True: the rung falsifier "L4PTFD not above L4PTF in the same job"
#   I4A_peak_gb        the L4PTFD arm's peak allocated GB, the PREFILL INCLUDED (the MEM gate's value; SCORING_i3.md
#                      6.1 item 1: I3's key read the decode peak only)
# The builder's INFERENCE (docs/NOTES_i4a.md section 5, with the arithmetic) is an input for the lead, not a registration.
PRED_STAMP_I4A = None
PRED_KEYS = ("I4A_ratio", "I4A_vs_I3", "I4A_draft_step_sh", "I4A_draft_saving", "I4A_joint_ratio", "I3_ratio",
             "I4A_agree", "I4A_eq_I3", "I4A_above_I3", "I4A_peak_gb")
PRED_I4A_BY_B = {216: {k: None for k in PRED_KEYS}, 318: {k: None for k in PRED_KEYS}}


def log(msg):
    print("[r2_i4a %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTF,L4PTFD", help="a subset of %s, in run order (SH runs first; L4PTF before "
                                                           "L4PTFD)" % ",".join(ARMS))
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
    ap.add_argument("--eq-must", type=int, default=0, choices=(0, 1), help="L4PTFD-EQF a MUST gate (i4am)")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if "L4PTF" in a.arms and "L4PTFD" in a.arms and a.arms.index("L4PTF") > a.arms.index("L4PTFD"):
        ap.error("--arms: L4PTF before L4PTFD (L4PTFD-EQF compares with L4PTF's tokens)")
    if a.eq_must and a.arms != ["L4PTF", "L4PTFD"]:
        ap.error("--eq-must 1 needs --arms L4PTF,L4PTFD")
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
def lean_gate(arm, rd, L: int):
    """<arm>-LEAN (MUST): every one-row pass took the lean path. One-row passes = the prelude's K drafts + the D-slot
    passes (draft + mixed steps) + the one-row J slots; lean layers = L x lean passes."""
    ln = rd.get("lean") or {}
    c = (rd.get("tent") or {}).get("calls") or {}
    K = int(rd.get("K") or 3)
    want = c.get("draft_steps", 0) + c.get("mixed_steps", 0) + c.get("j1_passes", 0)
    ok = (ln.get("passes", 0) > 0 and ln.get("layers") == L * ln.get("passes", -1)
          and ln.get("passes") == want and c.get("draft_steps", 0) >= K)
    return ("%s-LEAN" % arm, True, ok,
            "lean passes %s (one-row passes: draft steps %s incl. the prelude's %d + mixed %s + one-row J %s = %d); "
            "lean layers %s (L x passes = %s)" % (ln.get("passes"), c.get("draft_steps"), K, c.get("mixed_steps"),
                                                  c.get("j1_passes"), want, ln.get("layers"),
                                                  L * ln.get("passes", 0)))


def acct_record(rd):
    """The acceptance record of a run (read()): the token accounting and the per-round accepted / committed counts. A
    draft step that computes other values changes the drafts, hence these counts, but NOT the committed tokens (the
    joint pass verifies every draft): so EQF compares them too (docs/NOTES_i4a.md section 3)."""
    return dict(tokens_acct=rd.get("tokens_acct"), round_accepted=[int(x) for x in rd.get("round_accepted") or []],
                round_committed=[int(x) for x in rd.get("round_committed") or []])


def eq_gate(arm, fd, f, steps: int, must: bool):
    """<arm>-EQF: L4PTFD = L4PTF in the same job, bit for bit: the committed tokens (the first `steps` columns), the
    check-round fingerprints (the columns both arms marked) and the acceptance record (acct_record). fd / f: dict(out,
    fp=(fmax, fsum), fp_ok, acct) on the CPU."""
    import torch
    if f is None:
        return ("%s-EQF" % arm, must, False, "no L4PTF record in this job")
    tok_bad = int((fd["out"][:, :steps] != f["out"][:, :steps]).sum())
    both = fd["fp_ok"][:, :steps] & f["fp_ok"][:, :steps]
    fm = torch.equal(fd["fp"][0][:, :steps][both].view(torch.int32), f["fp"][0][:, :steps][both].view(torch.int32))
    fs = torch.equal(fd["fp"][1][:, :steps][both].view(torch.int64), f["fp"][1][:, :steps][both].view(torch.int64))
    n = int(both.sum())
    acc = fd.get("acct") is not None and fd.get("acct") == f.get("acct")
    ok = tok_bad == 0 and fm and fs and n > 0 and acc
    return ("%s-EQF" % arm, must, ok, "committed tokens unequal to L4PTF's: %d of %d; fingerprints compared %d columns, "
            "max equal %s, sum equal %s; acceptance record equal %s (accepted %s vs %s)" % (
                tok_bad, int(fd["out"][:, :steps].numel()), n, fm, fs, acc,
                ((fd.get("acct") or {}).get("tokens_acct") or {}).get("accepted"),
                ((f.get("acct") or {}).get("tokens_acct") or {}).get("accepted")))


def mem_record(arm, mem, pa, budget=BUDGET_GB):
    """r2_i3_cell.mem_record plus the usable ceiling (L59)."""
    n, m, ok, d = I3C.mem_record(arm, mem, pa, budget)
    return n, m, ok, d + "; usable ceiling %.2f GB" % USABLE_GB


def peak_gb(rec):
    """The arm's peak with the prefill (the MEM gate's value): max(decode peak GB, prefill peak GiB x 1.073741824)."""
    pre = (rec.get("mem_arena") or {}).get("prefill_peak_GB")
    vals = [x for x in (rec.get("gpu_peak_alloc_GB"), round(pre * 1024 ** 3 / 1e9, 3) if pre is not None else None)
            if x is not None]
    return max(vals) if vals else None


def verdicts(sh, sp, B: int, K: int = 3, eqf=None):
    """PRED_I4A_BY_B[B] against the arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN without
    L4PTFD. eqf: the L4PTFD-EQF gate's passed flag."""
    P = PRED_I4A_BY_B.get(B) or {k: None for k in PRED_KEYS}
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

    def flag(name, got, want, detail):
        out.append((name, "REPORT" if (want is None or got is None) else ("CONFIRMED" if got == bool(want) else
                                                                           "REFUTED"), detail))
    f, d = (sp or {}).get("L4PTF"), (sp or {}).get("L4PTFD")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    sh_ms = sh["rates"]["steady_step_ms"] if sh else None
    if f is not None:
        band("I3-RATIO", (med(f) / shm) if (med(f) and shm) else None, P.get("I3_ratio"))
    if d is None:
        out.append(("L4PTFD-RUN", "NOT_RUN", "the arm did not run"))
        return out
    band("I4A-RATIO", (med(d) / shm) if (med(d) and shm) else None, P.get("I4A_ratio"))
    band("I4A-VS-I3", (med(d) / med(f)) if (med(d) and med(f)) else None, P.get("I4A_vs_I3"))
    band("I4A-DRAFT-STEP-SH", (ph(d, "draft") / (K - 1) / sh_ms) if (ph(d, "draft") and sh_ms) else None,
         P.get("I4A_draft_step_sh"))
    band("I4A-DRAFT-SAVING", (ph(f, "draft") - ph(d, "draft")) if (ph(f, "draft") is not None and
                                                                   ph(d, "draft") is not None) else None,
         P.get("I4A_draft_saving"))
    band("I4A-JOINT-RATIO", (ph(d, "joint") / ph(f, "joint")) if (ph(d, "joint") and ph(f, "joint")) else None,
         P.get("I4A_joint_ratio"))
    band("I4A-PEAK-GB", peak_gb(d), P.get("I4A_peak_gb"))
    tc = d["tokens"]
    want = P.get("I4A_agree")
    out.append(("I4A-AGREE", "REPORT" if want is None else ("CONFIRMED" if tc["equal_fraction"] >= want else
                                                            "REFUTED"),
                "committed tokens equal to SH's: %.6f (first difference %s); registered >= %s" % (
                    tc["equal_fraction"], tc["first"], want)))
    flag("I4A-EQ-I3", eqf, P.get("I4A_eq_I3"), "L4PTFD-EQF passed %s; registered %s" % (eqf, P.get("I4A_eq_I3")))
    flag("I4A-ABOVE-I3", (med(d) > med(f)) if (med(d) and med(f)) else None, P.get("I4A_above_I3"),
         "L4PTFD %s vs L4PTF %s tok/s (the rung falsifier: not above the previous rung); registered %s" % (
             med(d), med(f), P.get("I4A_above_I3")))
    out.append(("I4A-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, d.get("vs_strongest"))))
    return out


# ------------------------------------------------------------------ the lean arm (GPU)
def run_arm_fd(model, ids, inputs, steps: int, clock, opt, FORK):
    """L4PTFD: r2_i3_cell.run_arm_f's steps with r2_i4a_lean's TentRunnerFD (the same arena prefill, store, ops,
    probe marks and memory measures)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    _, TentRunnerFD = LN.lean_classes(STF)
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
    runner = TentRunnerFD(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
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
               compile=I3C._compile_report(FORK))
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
                if arm == "L4PTF":
                    res = I3C.run_arm_f(model, ids, inputs, a.steps, clock, opt, FORK)
                else:
                    res = run_arm_fd(model, ids, inputs, a.steps, clock, opt, FORK)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(res["read"], L), fork=I3C.fork_record(res["read"]),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=res["read"].get("lean"))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = peak_gb(rec)
                sp[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(mem_record(arm, res["mem"], pa))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=acct_record(res["read"]))
                if arm == "L4PTF":
                    keep["L4PTF"] = cpu
                else:
                    rdl = dict(res["read"], K=a.spec)
                    gates.append(lean_gate(arm, rdl, L))
                    g = eq_gate(arm, cpu, keep.get("L4PTF"), a.steps, bool(a.eq_must))
                    eqf = bool(g[2])
                    gates.append(g)
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts(sh, sp, a.batch, a.spec, eqf)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                eq_must=a.eq_must, predictions=dict(stamp=PRED_STAMP_I4A, by_batch=PRED_I4A_BY_B),
                lean=dict(saved_per_layer=LN.SAVED_PER_LAYER, saved_layer0=LN.SAVED_LAYER0,
                          saved_per_pass=LN.saved_per_pass(L)),
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
              "round median %s ms; phases %s; draft step %s ms = %s SH; joint %s SH steps; ROLLBACK per round %s; "
              "calls %s; lean %s; tokens equal %.6f (first %s); fingerprints %s; peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r["vs_strongest"],
                  r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"], r["draft_step_sh"],
                  r["joint_sh_steps"], r["rollback_per_round"], r["calls"], r.get("lean"),
                  r["tokens"]["equal_fraction"], r["tokens"]["first"], r["fingerprints"], r.get("gpu_peak_alloc_GB"),
                  r.get("peak_with_prefill_GB")))
        print("REPORT %s fork: %s; joint-pass phases (probe) %s; store %s; arena memory %s" % (
            name, r["fork"], r["joint_phases_fork"], r["store"], r["mem_arena"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4a: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
