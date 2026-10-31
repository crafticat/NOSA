"""ST1 cell (AREA 3 of the RECON7 follow-up; KS/analysis/plan/LEDGER_rev7.md rows 5, 8, 9; KS = /mnt/beegfs/ojerbi/
kstep-cpu-runs): the best arms with THE COMMIT CAPTURED AND MOVED BY ROW (benchmarks/Efficiency/r2_st1_commit.py):
  L4PTFDGUDCDM    = L4PTFDGUDCD   (C 128 B 216: the best arm there) + r2_st1_commit (mode --st1-commit, default rows)
  L4PTFDGUDKWCDM  = L4PTFDGUDKWCD (C 73 B 318: the best arm there) + the same
NO NEW KERNEL. The I4DC cell (benchmarks/Efficiency/r2_i4dc_cell.py) is imported, not edited: its arms, its runner
choice, its pure parts, its gates. Step script: benchmarks/Efficiency/r2_st1_step.sh (cells st1m, st1a, st1b).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order of ARMS (L4PTFD first; a D arm after its C arm; an M arm
after its D arm: the D arm is its PEQ reference and its prediction REF). Every arm runs run_arm_st1 = the I4DC cell's
run_arm_d line for line plus the lines marked "# ST1" (the M runner class; the committed-row tail checksum of every arm;
the ST1 record); tests_cpu/test_st1.py checks it.
MUST GATES: the I4DC cell's for every arm (an M arm takes its D arm's gates under its own name: -RUN .. -EQF = L4PTFD bit
for bit, -COMPILE, the keep gates); for an M arm also -PEQ (the committed tokens, check-round fingerprints and their
marks, the acceptance record and the state checksums = its D arm's, bit for bit), -DCOUNTS (every counter = the D arm's),
-TAIL-EQ (the committed rows of every layer's tail slot = the D arm's, bit sums), -CGRAPH (1 warm call, 1 capture,
replays = commits - 1, the overflow counter m > J is 0).
REGISTERED PREDICTIONS PRED_ST1_BY_B: every value None until the lead registers them with PRED_STAMP_ST1 (a None key is
REPORT); the step script STOPS the timing cells (st1a, st1b) while PRED_STAMP_ST1 is None. The builder's MODEL: the
comment block above PRED_ST1_BY_B.
EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
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
import r2_i4dc_cell as DCC        # noqa: E402
import r2_st1_commit as STC       # noqa: E402

CCC, CD = DCC.CCC, DCC.CD
S4, CC, IA, GC, DG, LN, GR, HC, KC, UC, DC, KP, SK = (DCC.S4, DCC.CC, DCC.IA, DCC.GC, DCC.DG, DCC.LN, DCC.GR, DCC.HC,
                                                      DCC.KC, DCC.UC, DCC.DC, DCC.KP, DCC.SK)
I3C, WA, SC, BC = DCC.I3C, DCC.WA, DCC.SC, DCC.BC
ST1 = dict(L4PTFDGUDCDM="L4PTFDGUDCD", L4PTFDGUDKWCDM="L4PTFDGUDKWCD")    # an M arm -> its D arm (the plain arm)
ARMS = DCC.ARMS + tuple(ST1)
GRAPH_ARMS = DCC.GRAPH_ARMS + tuple(ST1)
BASE_D = DCC.BASE_D
TOPK = DCC.TOPK
FORK_BLOB = DCC.FORK_BLOB
STRONGEST = DCC.STRONGEST
BUDGET_GB = DCC.BUDGET_GB
USABLE_GB = DCC.USABLE_GB
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_ST1_BY_B (per batch) and PRED_STAMP_ST1
# (`date -u`) in a commit BEFORE st1a / st1b. Keys, each (low, high); X = the M arm of the cell, REF = its D arm; every
# phase / rate on the steady set WITHOUT the capture rounds (L61):
#   ST1_commit_saving  [ms per round] REF commit phase median - X's
#   ST1_joint_ratio    [x] X joint median / REF's (not changed: about 1)
#   ST1_draft_ratio    [x] X draft median / REF's (not changed: about 1)
#   ST1_vs_ref         [x] X / REF median steady committed tok/s (same job)
#   ST1_peak_delta_gb  [GB] X's peak with the prefill - REF's
# THE BUILDER'S MODEL (an input for the lead, not a registration). MEASURED: the commit phase median 6.058 ms (C 128
# B 216, i4dc_d3a_2195575_i4dca) and 8.416 ms (C 73 B 318, i4dc_d4b_2195576_i4dcb); the r7pm miniature (C 73 B 8): the
# labelled TentState.commit 1.47 ms of 1.73 ms commit-window busy, eager phase median 3.26-3.43 ms.
#   MODEL of the eager TentState.commit at B: 4 tail-slot where()s per layer read 2 and write 1 slot (B x 32,768 B each)
#   at 1.3-1.6 TB/s, + about 5 small kernels per layer at about 3 us: B 216: 32 x (4 x 21.2 MB / 1.45 TB/s + 15 us) =
#   2.35 ms; B 318: 32 x (4 x 31.3 MB / 1.45 TB/s + 15 us) = 3.24 ms; its floor at B 8 = 1.47 ms MEASURED (launch bound).
#   MODEL of the replayed graph, mode rows: per layer 3 gathers + 6 index_put_ + the write-back + 2 copies = 12 nodes of
#   tiny kernels at 1.5-3 us, x 32 = 0.6-1.2 ms, plus about 20 nodes before the layer loop.
#   Saving per round: B 216 (2.35 - 1.2 .. 2.35 - 0.6) = 1.1 .. 1.8 ms; B 318 2.0 .. 2.6 ms. Host launch time of the
#   eager commit (about 300 launches) also leaves the phase; if the phase is host bound the saving is larger (up to the
#   B 8 phase median minus its graph: about 2.5 ms). BAND: B 216 (0.8, 3.0) ms; B 318 (1.5, 3.5) ms.
#   ST1_vs_ref = round / (round - saving): B 216 324.9 / (324.9 - 0.8 .. 3.0) = (1.0025, 1.0093); B 318 524.7 / (524.7 -
#   1.5 .. 3.5) = (1.0029, 1.0067). The run-to-run spread of the best arms is 0.3 % (B 216) and 0.6 % (B 318): the tok/s
#   effect is at or below the spread; the commit phase median is the primary measure.
#   ST1_peak_delta_gb (-0.01, 0.05): the static inputs (< 0.1 MB) and the private pool (MODEL < 20 MB) only.
PRED_STAMP_ST1 = None
PRED_KEYS = ("ST1_commit_saving", "ST1_joint_ratio", "ST1_draft_ratio", "ST1_vs_ref", "ST1_peak_delta_gb")
PRED_ST1_BY_B = {216: {k: None for k in PRED_KEYS}, 318: {k: None for k in PRED_KEYS}}


def log(msg):
    print("[r2_st1 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def runner_choice(arm: str, check: bool):
    """The I4DC cell's runner choice of the arm (an M arm: its D arm's)."""
    return DCC.runner_choice(ST1.get(arm, arm), check)


runner_class = DCC.runner_class


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDM",
                    help="a subset of %s, in this order (SH runs first; L4PTFD first)" % ",".join(ARMS))
    ap.add_argument("--st1-commit", default="rows", choices=STC.MODES, help="the M arms' commit mode")
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
                    help="0 = the probe rounds; 1 = every round (the miniature)")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if a.arms != [x for x in ARMS if x in a.arms] or a.arms[0] != "L4PTFD":
        ap.error("--arms: L4PTFD first, then in the order %s (EQF compares with L4PTFD)" % ",".join(ARMS))
    for x, c in BASE_D.items():
        if x in a.arms and c not in a.arms:
            ap.error("--arms: %s needs its C arm %s (it compiles the joint helpers)" % (x, c))
    for x, d in ST1.items():
        if x in a.arms and d not in a.arms:
            ap.error("--arms: %s needs its D arm %s (the PEQ reference)" % (x, d))
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.check_every == 1 and a.batch > 64 and any(x in a.arms for x in DCC.KEEP_ARMS + ("L4PTFDGUDKWCDM",)):
        ap.error("--check-every 1 with a keep arm: the keep's byte check needs a scratch store: --batch <= 64")
    if a.probe_every < 2 or a.check_every < 0 or a.spec_journal_blocks < 1:
        ap.error("--probe-every >= 2, --check-every >= 0, journal >= 1")
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(TOPK, a.spec):
        ap.error("--fork-store %d: below the bound 1 + (K - 1) x 64 = %d (0 = the bound)"
                 % (a.fork_store, I3C.KA.store_slots(TOPK, a.spec)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_st1(model, ids, inputs, steps: int, clock, opt, FORK, arm: str):
    """The I4DC cell's run_arm_d line for line (the runner of runner_choice(arm, check): an M arm's D arm's) plus the
    lines marked ST1: an M arm's runner class (r2_st1_commit.st1_runner), every arm's committed-row tail checksum (after
    the run, before detach) and the ST1 record."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, check)
    Runner = runner_class(kind, STF)
    Runner = STC.st1_runner(Runner, opt["st1_commit"]) if arm in ST1 else Runner   # ST1
    if kind in ("SC", "SKC", "SCD", "SKCD"):
        CC.configure()
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
    gk = dict(GR=GR) if kind in ("S", "SK", "SC", "SKC", "SCD", "SKCD") else {}
    runner = Runner(model, rs, sops, clock, fork=FORK.unified_attention, t0=inputs[:, 0], hooks=None,
                    check_every=ce, union_check=check, union_stats=check, phase_marks=True, **kw, **gk)
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
               compile=I3C._compile_report(FORK), state=IA.state_checksum(runner), runner_kind=kind, runner_kw=kw)
    res.update(tails=STC.tail_checksum(runner.T), st1=rd.get("st1"))   # ST1
    runner.detach()
    del runner, rs, cache, arena
    return res


# ------------------------------------------------------------------ pure parts (CPU-tested)
def peq_gate(arm, cpu, ref, x: str, steps: int):
    """<arm>-PEQ (MUST): the committed tokens, the check-round fingerprints and their marks, the acceptance record and
    the state checksums = the D arm x's, bit for bit (cpu / ref: dict(out, fp, fp_ok, acct, state) on the CPU)."""
    import torch
    if ref is None:
        return ("%s-PEQ" % arm, True, False, "no %s record in this job" % x)
    tok_bad = int((cpu["out"][:, :steps] != ref["out"][:, :steps]).sum())
    marks = torch.equal(cpu["fp_ok"][:, :steps], ref["fp_ok"][:, :steps])
    both = cpu["fp_ok"][:, :steps] & ref["fp_ok"][:, :steps]
    fm = torch.equal(cpu["fp"][0][:, :steps][both].view(torch.int32), ref["fp"][0][:, :steps][both].view(torch.int32))
    fs = torch.equal(cpu["fp"][1][:, :steps][both].view(torch.int64), ref["fp"][1][:, :steps][both].view(torch.int64))
    acc = cpu.get("acct") is not None and cpu.get("acct") == ref.get("acct")
    st = cpu.get("state") is not None and cpu.get("state") == ref.get("state")
    ok = tok_bad == 0 and marks and fm and fs and acc and st
    return ("%s-PEQ" % arm, True, ok, "vs %s: tokens differing %d; fingerprint marks equal %s, %d fingerprints max %s "
            "sum %s; acceptance record equal %s; state checksums equal %s" % (x, tok_bad, marks, int(both.sum()), fm,
                                                                              fs, acc, st))


def tail_gate(arm, tails, ref_tails, x: str):
    """<arm>-TAIL-EQ (MUST): the committed rows of every layer's tail slot = the D arm x's (bit sums)."""
    if ref_tails is None:
        return ("%s-TAIL-EQ" % arm, True, False, "no %s tail checksum in this job" % x)
    bad = [i for i, (a, b) in enumerate(zip(tails or [], ref_tails)) if a != b]
    ok = tails is not None and len(tails) == len(ref_tails) and not bad
    return ("%s-TAIL-EQ" % arm, True, ok, "%d sums (layers x K, V, bias) vs %s; differing %d (first %s)" % (
        len(ref_tails), x, len(bad), bad[:4]))


def cgraph_gate(arm, s1, calls_commit):
    """<arm>-CGRAPH (MUST): the commit ran 1 warm call, 1 capture, every later call replayed, the overflow counter
    (m > J) is 0, and the record's call count = TentState's host count."""
    if not s1:
        return ("%s-CGRAPH" % arm, True, False, "no ST1 record")
    n = s1.get("commits")
    ok = (s1.get("warm") == 1 and s1.get("captures") == 1 and n is not None and n >= 2
          and s1.get("replays") == n - 1 and s1.get("overflow") == 0 and n == calls_commit)
    return ("%s-CGRAPH" % arm, True, ok, "mode %s J %s: commits %s (TentState count %s), warm %s, captures %s (round %s), "
            "replays %s, overflow %s; capture %s s; pool %s B; static %s B" % (
                s1.get("mode"), s1.get("J"), n, calls_commit, s1.get("warm"), s1.get("captures"),
                s1.get("capture_round"), s1.get("replays"), s1.get("overflow"), s1.get("capture_s"),
                s1.get("pool_bytes"), s1.get("static_bytes")))


def arm_gates_st1(arm, rd, rec, cpu, keep, check: bool, B: int, pool: int, steps: int, L: int, pa):
    """The I4DC cell's arm gates; an M arm: its D arm's gates under its own name, then -PEQ, -DCOUNTS, -TAIL-EQ and
    -CGRAPH against / about its D arm."""
    if arm not in ST1:
        return DCC.arm_gates(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
    x = ST1[arm]
    out = [(arm + n[len(x):] if n.startswith(x + "-") else n, m, ok, d)
           for n, m, ok, d in DCC.arm_gates(x, rd, rec, cpu, keep, check, B, pool, steps, L, pa)]
    ref = keep.get(x)
    out.append(peq_gate(arm, cpu, ref, x, steps))
    g = KC.counts_gate(arm, cpu["counts"], ref["counts"] if ref else None, x)
    out.append((arm + "-DCOUNTS",) + tuple(g[1:]))
    out.append(tail_gate(arm, cpu.get("tails"), ref.get("tails") if ref else None, x))
    calls = (((rd.get("tent") or {}).get("calls")) or {}).get("commit")
    out.append(cgraph_gate(arm, rd.get("st1"), calls))
    return out


def m_arm(sp):
    """X: the M arm of the job (L4PTFDGUDKWCDM before L4PTFDGUDCDM); None if neither ran."""
    for x in ("L4PTFDGUDKWCDM", "L4PTFDGUDCDM"):
        if x in (sp or {}):
            return x
    return None


def verdicts(sp, B: int):
    """PRED_ST1_BY_B[B] against X = m_arm(sp) and REF = its D arm: [(name, verdict, detail)]; None registered gives
    REPORT; NOT_RUN without an M arm."""
    P = PRED_ST1_BY_B.get(B) or {k: None for k in PRED_KEYS}
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

    def ratio(a, b):
        return (a / b) if (a is not None and b) else None

    def diff(a, b):
        return (a - b) if (a is not None and b is not None) else None
    sp = sp or {}
    X = m_arm(sp)
    if X is None:
        return [("ST1-RUN", "NOT_RUN", "no M arm ran")]
    x, r = sp[X], sp.get(ST1[X])
    band("ST1-COMMIT-SAVING", diff(ph(r, "commit"), ph(x, "commit")), P.get("ST1_commit_saving"))
    band("ST1-JOINT-RATIO", ratio(ph(x, "joint"), ph(r, "joint")), P.get("ST1_joint_ratio"))
    band("ST1-DRAFT-RATIO", ratio(ph(x, "draft"), ph(r, "draft")), P.get("ST1_draft_ratio"))
    band("ST1-VS-REF", ratio(med(x), med(r)), P.get("ST1_vs_ref"))
    band("ST1-PEAK-DELTA-GB", diff(IA.peak_gb(x), IA.peak_gb(r)), P.get("ST1_peak_delta_gb"))
    out.append(("ST1-ARM", "REPORT", "X = %s, REF = %s (REF ran: %s); round median %s vs %s ms; commit %s vs %s ms" % (
        X, ST1[X], r is not None, (x.get("rates") or {}).get("round_ms_median"),
        ((r or {}).get("rates") or {}).get("round_ms_median"), ph(x, "commit"), ph(r, "commit"))))
    return out


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
    if any(x in a.arms for x in tuple(CCC.BASE) + tuple(BASE_D) + tuple(ST1)) and \
            not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        print("REFUSED: a C, D or M arm needs TORCHINDUCTOR_CACHE_DIR (the step script sets it; never the home cache)",
              file=sys.stderr)
        return 20
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, inputs, keep, rds, caps = [], None, {}, None, {}, {}, {}
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
                       prefill_hook=a.prefill_hook, st1_commit=a.st1_commit)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_st1(model, ids, inputs, a.steps, clock, opt, FORK, arm)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), graph=rd.get("graph"), dgraph=rd.get("dgraph"),
                           hoist=rd.get("hoist"), capture_rounds=GC.capture_rounds(rd), runner=res["runner_kind"],
                           runner_kw=res["runner_kw"], cc=rd.get("compile"), st1=res["st1"])
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = IA.peak_gb(rec)
                rec["effective_wo_capture_tok_s"] = HC.effective_wo_capture(rec, rd)
                rec["effective_wo_setup_tok_s"] = CCC.effective_wo_setup(rec, rd)
                rec["store_record"] = DC.acct_summary(rd.get("keep"), bb)
                sp[arm] = rec
                rds[arm] = rd
                if arm in GRAPH_ARMS:
                    caps[arm] = HC.capture_rounds_x(rd)
                    s1 = rd.get("st1")
                    if s1 and s1.get("capture_round") is not None and int(s1["capture_round"]) >= 0:   # L61: the
                        caps[arm] = dict(caps[arm], joint=sorted(set(caps[arm]["joint"])            # commit graph's
                                                                 | {int(s1["capture_round"])}))     # capture round
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (rd["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(IA.lean_gate(arm, dict(rd, K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=IA.acct_record(rd), state=res["state"], counts=GC.counts_record(rd), keep=rd.get("keep"),
                           tails=res["tails"])
                if arm == "L4PTFD":
                    gates.append(IA.mem_record(arm, res["mem"], pa))
                gates.extend(arm_gates_st1(arm, rd, rec, cpu, keep, check_t, a.batch, a.pool, a.steps, L, pa))
                keep[arm] = cpu
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    xr = HC.excluded_rounds(caps)                                      # L61: the same rounds for every arm
    for arm, rec in sp.items():
        HC.restrict(rec, rds[arm], sh, a.batch, a.spec, a.probe_every, xr)
    preds = verdicts(sp, a.batch)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, st1_commit=a.st1_commit,
                fork_store=a.fork_store, fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store),
                prefill_hook=a.prefill_hook, tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks,
                spec_twin=a.spec_twin, spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_ST1, by_batch=PRED_ST1_BY_B), st1_arms=ST1,
                compile=dict(torch=torch.__version__, inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                             kernel_cache=os.environ.get("TRITON_CACHE_DIR")),
                mem_must_GB=GC.MEM_MUST_GB, capture_rounds=caps, steady_excluded_rounds=xr, m_arm=m_arm(sp),
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-24s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-22s %s: %s" % p)
    print("REPORT L61 capture rounds %s; excluded from every arm's steady set: %s" % (caps, xr))
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; vs %.1f %s; round "
              "median %s ms; phases %s; steady rounds %s (excluded %s); tokens equal %.6f (first %s); peak %s GB (with "
              "the prefill %s GB); st1 %s" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], STRONGEST, r.get("vs_strongest"),
                  r["rates"].get("round_ms_median"), r["phases_median"], r.get("steady_rounds_used"),
                  r.get("steady_excluded_rounds"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB"), r.get("st1")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_st1: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
