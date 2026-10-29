"""Rung I4H of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), the "draft-step graph" of
the specKV-on-cached-NOSA ladder (KS/analysis/plan/HANDOFF_2026-10-29.md option (b); LEDGER_rev6.md R6.6 E1): the GPU
cell of the arm L4PTFDGD = L4PTFDG (rung I4G) with the one-row (draft) passes replayed from per-layer CUDA graphs
(benchmarks/Efficiency/r2_i4h_graph.py). A NEW script: r2_i4g_cell.py and r2_i4a_cell.py are not edited (their functions
are imported; every arm runs run_arm_h = r2_i4g_cell.run_arm_g's steps with the runner class as a parameter). Step
script: benchmarks/Efficiency/r2_i4h_step.sh (cells i4hm, i4ha, i4hb). Docs: docs/NOTES_i4h.md.

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order given (L4PTFD first: the graph arms' tokens are compared
with its tokens), each with its own shipped prefill into the chain arena, on the inputs of i4ga / i4gb (the same
workload pin, S0, document rows, K 3, 512 steps, fused twin and body, graph, redirect, probe 16, store X 129, --tu-gemm
natural):
  SH        the shipped decode at the cell's C and B (r2_base_cell.run_shipped): the same-job baseline
  L4PTFD    I4A (r2_i4a_lean.TentRunnerFD): the base rung, re-measured
  L4PTFDG   I4G (r2_i4g_graph.TentRunnerFDG): the joint-pass section from CUDA graphs, re-measured
  L4PTFDGD  I4H (r2_i4h_graph.TentRunnerFDGD): L4PTFDG with the one-row passes from per-layer CUDA graphs
MUST GATES: C0_FORK; SH-RUN; per arm <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN (r2_wsa_cell.arm_gates_w), -RESID,
-ATTN1, -FILL1 (r2_i3_cell.fork_gates), -LEAN (r2_i4a_cell.lean_gate); with --check-every 1 (i4hm) -PRESENT, -VIS,
-VB, -TAIL, -CHAIN, -IDX, -RA; with --require-stress -STRESS, -FSTRESS; for each graph arm <arm>-EQF (committed tokens,
check-round fingerprints, acceptance record and state checksums = L4PTFD's in the same job, bit for bit), <arm>-COUNTS
(every host count and device counter of the run = L4PTFD's), <arm>-GRAPH (the joint-pass graphs: r2_i4g_cell.graph_gate),
<arm>-MEM (peak allocated with the prefill <= 84.0 GB); for L4PTFDGD also L4PTFDGD-DGRAPH (dgraph_gate: 2 L graphs per
captured key, every capture's skipped host counts = r2_i4h_graph.expected_delta, no eager one-row pass outside
DEAGER_OK, every one-row pass accounted: warm + eager + capture + replay passes = lean passes).
L61 FIX (KS/analysis/plan/NOTES_graph_mean_gap.md): the rounds in which any graph arm of the job captured graphs (the
joint capture round, the draft capture rounds; r2_i4h_graph records the round, T.round) are REPORTED and EXCLUDED from
the steady set of EVERY spec arm (the same rounds for every arm: the arms run the same schedule, bit for bit). The rates
and phases with the capture rounds kept stay in the record (rates_incl_capture, phases_median_incl_capture).
REGISTERED PREDICTIONS PRED_I4H_BY_B: every value None until the lead registers them with PRED_STAMP_I4H; the step script
STOPS the timing cells (i4ha, i4hb) while PRED_STAMP_I4H is None. The builder's MODEL with its derivation:
docs/NOTES_i4h.md section 5. EXIT: the number of failed MUST gates (0-19); 20 refused; 30 an exception outside an arm.
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
import r2_i4g_cell as GC          # noqa: E402
import r2_i4h_graph as DG         # noqa: E402

IA, LN, GR = GC.IA, GC.LN, GC.GR
I3C = IA.I3C
WA, SC, BC = IA.WA, IA.SC, IA.BC
ARMS = ("L4PTFD", "L4PTFDG", "L4PTFDGD")
GRAPH_ARMS = ("L4PTFDG", "L4PTFDGD")
TOPK = IA.TOPK
FORK_BLOB = IA.FORK_BLOB
STRONGEST = IA.STRONGEST
BUDGET_GB = IA.BUDGET_GB
USABLE_GB = IA.USABLE_GB
MEM_MUST_GB = 84.0                 # <arm>-MEM: peak allocated with the prefill (C 128 B 216: REF 83.891 GB, i4ga)
# REGISTERED PREDICTIONS, NONE UNTIL THE LEAD REGISTERS THEM: fill PRED_I4H_BY_B (per batch) and PRED_STAMP_I4H
# (`date -u`, L33) in a commit BEFORE i4ha / i4hb (the step script stops while PRED_STAMP_I4H is None). Keys, each
# (low, high) unless noted; every phase / rate on the steady set WITHOUT the capture rounds (L61 fix):
#   I4H_draft_saving   L4PTFDG draft phase median - L4PTFDGD draft phase median (ms per round, same job)
#   I4H_draft_ratio    L4PTFDGD draft median / L4PTFDG draft median
#   I4H_joint_ratio    L4PTFDGD joint median / L4PTFDG joint median (the joint pass is the same code: about 1)
#   I4H_vs_G           L4PTFDGD / L4PTFDG median steady committed tok/s (same job): the lever's change
#   I4H_vs_D           L4PTFDGD / L4PTFD median steady committed tok/s (same job)
#   I4H_ratio          L4PTFDGD median steady committed tok/s / SH's
#   I4H_eq             True: L4PTFDGD-EQF passes (bit for bit = L4PTFD)
#   I4H_peak_gb        the L4PTFDGD arm's peak allocated GB, the PREFILL INCLUDED (the MEM gate's value)
#   I4H_pool_mb        the shared graph pool's segments (MB, torch.cuda.memory_snapshot) after the L4PTFDGD run
#   I4H_capture_s      the summed capture time of the draft graphs (s, host clock; begin to end, with the syncs)
# The builder's MODEL (docs/NOTES_i4h.md section 5, with the derivation) is an input for the lead, not a registration.
PRED_STAMP_I4H = None
PRED_KEYS = ("I4H_draft_saving", "I4H_draft_ratio", "I4H_joint_ratio", "I4H_vs_G", "I4H_vs_D", "I4H_ratio", "I4H_eq",
             "I4H_peak_gb", "I4H_pool_mb", "I4H_capture_s")
PRED_I4H_BY_B = {
    216: {k: None for k in PRED_KEYS},
    318: {k: None for k in PRED_KEYS},
}


def log(msg):
    print("[r2_i4h %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default=",".join(ARMS), help="a subset of %s, in run order (SH runs first; L4PTFD first "
                                                          "when a graph arm runs)" % ",".join(ARMS))
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
    if any(x in a.arms for x in GRAPH_ARMS) and a.arms[0] != "L4PTFD":
        ap.error("--arms: L4PTFD first (the graph arms' EQF compares with L4PTFD's tokens)")
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
def eq_gate(arm, f, fd, steps: int):
    """<arm>-EQF (MUST): r2_i4a_cell.eq_gate against L4PTFD's record (the same four parts, bit for bit)."""
    n, m, ok, d = IA.eq_gate(arm, f, fd, steps, True)
    return n, m, ok, d.replace("L4PTF's", "L4PTFD's").replace("no L4PTF record", "no L4PTFD record")


def dgraph_gate(arm, rd, L: int):
    """<arm>-DGRAPH (MUST): per captured key 2 L graphs and one capture pass; a warm pass per key seen; every capture's
    skipped host counts = r2_i4h_graph.expected_delta; every eager one-row pass for a reason in DEAGER_OK; warm + eager
    + capture + replay passes = the lean passes (every one-row pass); 2 L replays per capture or replay pass; at least
    one replay pass."""
    d = rd.get("dgraph") or {}
    passes = (rd.get("lean") or {}).get("passes", 0)
    keys = d.get("keys") or []
    ew = d.get("eager_why") or {}
    cap, cp, rp = d.get("captures", 0), d.get("capture_passes", 0), d.get("replay_passes", 0)
    warm, eag, rep = d.get("warm", 0), d.get("eager", 0), d.get("replays", 0)
    ok = (len(keys) >= 1 and cap == 2 * L * len(keys) and cp == len(keys) and warm >= len(keys)
          and warm == len(d.get("warm_keys") or []) and not d.get("delta_bad") and set(ew) <= set(DG.DEAGER_OK)
          and eag == sum(ew.values()) and warm + eag + cp + rp == passes and rep == 2 * L * (cp + rp) and rp > 0)
    return ("%s-DGRAPH" % arm, True, ok,
            "keys captured %s (warm keys %s); graphs %s (2 L x keys = %d); passes: warm %s + eager %s %s + capture %s + "
            "replay %s = %d (one-row passes %s); replays %s (2 L x (capture + replay passes) = %d); skipped-count "
            "mismatches %s; capture rounds %s" % (
                keys, d.get("warm_keys"), cap, 2 * L * len(keys), warm, eag, ew, cp, rp, warm + eag + cp + rp, passes,
                rep, 2 * L * (cp + rp), d.get("delta_bad"), d.get("capture_rounds")))


def capture_rounds_x(rd) -> dict:
    """The rounds (>= 0) in which the arm captured graphs: "joint" (r2_i4h_graph's T.round record; for L4PTFDG, which
    has no such record, the joint capture pass = the round when no one-row J slot ran) and "draft"."""
    d = rd.get("dgraph")
    if d is not None:
        joint = sorted({int(r) for r in d.get("joint_capture_rounds") or [] if int(r) >= 0})
        draft = sorted({int(r) for r, _ in d.get("capture_rounds") or [] if int(r) >= 0})
        return dict(joint=joint, draft=draft, mapped=True)
    g = rd.get("graph")
    if g is None:
        return dict(joint=[], draft=[], mapped=True)
    j1 = ((rd.get("tent") or {}).get("calls") or {}).get("j1_passes", 0)
    return dict(joint=sorted({int(p) for p in g.get("capture_passes") or []}) if j1 == 0 else [], draft=[],
                mapped=(j1 == 0))


def excluded_rounds(caps) -> list:
    """The L61 exclusion: the union of every graph arm's capture rounds (caps: {arm: capture_rounds_x})."""
    out = set()
    for c in caps.values():
        out |= set(c["joint"]) | set(c["draft"])
    return sorted(out)


def restrict(rec, rd, sh, B: int, K: int, probe_every: int, X) -> dict:
    """The L61 fix on one arm's record: rates, vs_sh, vs_strongest, phases_median, draft_step_ms / _sh and
    joint_sh_steps on the steady set (r2_spec_cell.steady_rounds) WITHOUT the rounds X; the old values kept as
    *_incl_capture. Returns rec (changed in place)."""
    G = rd["rounds"]
    xs = set(int(x) for x in X)
    steady = [g for g in SC.steady_rounds(G, probe_every) if g not in xs]
    rates = SC.round_rates(rd["round_committed"], rd["round_ms"], steady, rd["wall_s"], sum(rd["count"]))
    shr = SC.sh_rates(B, sh) if sh else None
    names = WA.PHASES_W["L4P"]
    ph = {nm: WA._med([rd["phases"][g][i] for g in steady if g < len(rd["phases"])]) for i, nm in enumerate(names)}
    rec["rates_incl_capture"] = rec.get("rates")
    rec["phases_median_incl_capture"] = rec.get("phases_median")
    rec["rates"] = rates
    rec["phases_median"] = ph
    rec["vs_sh"] = {k: (round(rates[k] / shr[k], 4) if (shr and shr.get(k) and rates.get(k)) else None)
                    for k in ("median_tok_s", "mean_tok_s", "effective_tok_s")}
    rec["vs_strongest"] = round(rates["median_tok_s"] / WA.STRONGEST, 4) if rates["median_tok_s"] else None
    shm = sh["rates"]["steady_step_ms"] if sh else None
    jm, dm = ph.get("joint"), ph.get("draft")
    rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
    rec["draft_step_ms"] = round(dm / (K - 1), 3) if dm else None
    rec["draft_step_sh"] = round(dm / (K - 1) / shm, 4) if (dm and shm) else None
    rec["steady_excluded_rounds"] = sorted(xs)
    rec["steady_rounds_used"] = len(steady)
    return rec


def effective_wo_capture(rec, rd) -> float:
    """REPORT (NOTES_graph_mean_gap.md fix 4): every committed token / (wall_s - the arm's capture seconds)."""
    g, d = rd.get("graph") or {}, rd.get("dgraph") or {}
    cs = float(g.get("capture_s") or 0.0) + float(d.get("dcapture_s") or 0.0)
    w = rd.get("wall_s")
    return round(sum(rd["count"]) / (w - cs), 1) if (w and w > cs) else None


def verdicts(sh, sp, B: int, eqf=None):
    """PRED_I4H_BY_B[B] against the arms: [(name, verdict, detail)]; None registered gives REPORT; NOT_RUN without
    L4PTFDGD."""
    P = PRED_I4H_BY_B.get(B) or {k: None for k in PRED_KEYS}
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
    d, g, gd = (sp or {}).get("L4PTFD"), (sp or {}).get("L4PTFDG"), (sp or {}).get("L4PTFDGD")
    shm = SC.sh_rates(B, sh)["median_tok_s"] if sh else None
    if gd is None:
        out.append(("L4PTFDGD-RUN", "NOT_RUN", "the arm did not run"))
        return out
    dr = gd.get("dgraph") or {}
    gr = gd.get("graph") or {}
    band("I4H-DRAFT-SAVING", (ph(g, "draft") - ph(gd, "draft")) if (ph(g, "draft") is not None and
                                                                    ph(gd, "draft") is not None) else None,
         P.get("I4H_draft_saving"))
    band("I4H-DRAFT-RATIO", ratio(ph(gd, "draft"), ph(g, "draft")), P.get("I4H_draft_ratio"))
    band("I4H-JOINT-RATIO", ratio(ph(gd, "joint"), ph(g, "joint")), P.get("I4H_joint_ratio"))
    band("I4H-VS-G", ratio(med(gd), med(g)), P.get("I4H_vs_G"))
    band("I4H-VS-D", ratio(med(gd), med(d)), P.get("I4H_vs_D"))
    band("I4H-RATIO", ratio(med(gd), shm), P.get("I4H_ratio"))
    band("I4H-PEAK-GB", IA.peak_gb(gd), P.get("I4H_peak_gb"))
    band("I4H-POOL-MB", (gr["pool_bytes"] / 1e6) if gr.get("pool_bytes") is not None else None, P.get("I4H_pool_mb"))
    band("I4H-CAPTURE-S", dr.get("dcapture_s"), P.get("I4H_capture_s"))
    want = P.get("I4H_eq")
    out.append(("I4H-EQ", "REPORT" if (want is None or eqf is None) else ("CONFIRMED" if eqf == bool(want) else
                                                                          "REFUTED"),
                "L4PTFDGD-EQF passed %s; registered %s" % (eqf, want)))
    out.append(("I4H-VS-STRONGEST", "REPORT", "median steady tok/s / %.1f: %s" % (STRONGEST, gd.get("vs_strongest"))))
    out.append(("I4G-VS-D", "REPORT", "L4PTFDG / L4PTFD median (the I4G replicate): %s" % (
        round(ratio(med(g), med(d)), 4) if ratio(med(g), med(d)) is not None else None)))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def run_arm_h(model, ids, inputs, steps: int, clock, opt, FORK, kind: str):
    """r2_i4g_cell.run_arm_g's steps, line for line, with the runner class as the only choice (kind "D": L4PTFD =
    r2_i4a_lean's TentRunnerFD; "G": L4PTFDG = r2_i4g_graph's TentRunnerFDG; "GD": L4PTFDGD = r2_i4h_graph's
    TentRunnerFDGD), the state checksum after the run (EQF) and the graph reports (read()["graph"] / ["dgraph"], before
    detach: the pool still exists)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    lean = LN.lean_classes(STF)
    graph = GR.graph_classes(STF, lean)
    Runner = dict(D=lean[1], G=graph[1], GD=DG.draft_classes(STF, graph)[1])[kind]
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
    kw = dict(GR=GR) if kind == "GD" else {}
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
                kind = dict(L4PTFD="D", L4PTFDG="G", L4PTFDGD="GD")[arm]
                res = run_arm_h(model, ids, inputs, a.steps, clock, opt, FORK, kind)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(res["read"], L), fork=I3C.fork_record(res["read"]),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=res["read"].get("lean"), graph=res["read"].get("graph"),
                           dgraph=res["read"].get("dgraph"), capture_rounds=GC.capture_rounds(res["read"]))
                jm = rec["phases_median"].get("joint")
                shm = sh["rates"]["steady_step_ms"] if sh else None
                rec["joint_sh_steps"] = round(jm / shm, 4) if (jm and shm) else None
                dm = rec["phases_median"].get("draft")
                rec["draft_step_ms"] = round(dm / (a.spec - 1), 3) if dm else None
                rec["draft_step_sh"] = round(dm / (a.spec - 1) / shm, 4) if (dm and shm) else None
                rec["peak_with_prefill_GB"] = IA.peak_gb(rec)
                rec["effective_wo_capture_tok_s"] = effective_wo_capture(rec, res["read"])
                sp[arm] = rec
                rds[arm] = res["read"]
                if arm in GRAPH_ARMS:
                    caps[arm] = capture_rounds_x(res["read"])
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (res["read"]["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(IA.lean_gate(arm, dict(res["read"], K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=IA.acct_record(res["read"]), state=res["state"], counts=GC.counts_record(res["read"]))
                if arm == "L4PTFD":
                    gates.append(IA.mem_record(arm, res["mem"], pa))
                    keep["L4PTFD"] = cpu
                else:
                    gates.append(GC.mem_gate(arm, rec))
                    gates.append(GC.graph_gate(arm, res["read"], L))
                    if arm == "L4PTFDGD":
                        gates.append(dgraph_gate(arm, res["read"], L))
                    ref = keep.get("L4PTFD")
                    gates.append(GC.counts_gate(arm, cpu["counts"], ref["counts"] if ref else None))
                    g = eq_gate(arm, cpu, ref, a.steps)
                    if arm == "L4PTFDGD":
                        eqf = bool(g[2])
                    gates.append(g)
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    xr = excluded_rounds(caps)                                         # THE L61 FIX: the same rounds for every arm
    for arm, rec in sp.items():
        restrict(rec, rds[arm], sh, a.batch, a.spec, a.probe_every, xr)
    preds = verdicts(sh, sp, a.batch, eqf)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                predictions=dict(stamp=PRED_STAMP_I4H, by_batch=PRED_I4H_BY_B),
                section_graph=dict(skipped_T=GR.SKIPPED_T, skipped_vs=GR.SKIPPED_VS, eager_ok=GR.EAGER_OK),
                draft_graph=dict(deager_ok=DG.DEAGER_OK, expected_delta=DG.expected_delta(a.spec_graph == "on")),
                mem_must_GB=MEM_MUST_GB, capture_rounds=caps, steady_excluded_rounds=xr,
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB,
                dims=dims, host_mem=mem, base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-18s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-16s %s: %s" % p)
    print("REPORT L61 capture rounds %s; excluded from every arm's steady set: %s" % (caps, xr))
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s (without capture s %s); vs SH "
              "%s; vs %.1f %s; round median %s ms; phases %s; draft step %s ms = %s SH; joint %s SH steps; steady "
              "rounds %s (excluded %s; with the capture rounds: median %s mean %s, phases %s); calls %s; lean %s; "
              "tokens equal %.6f (first %s); peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r.get("effective_wo_capture_tok_s"), r["vs_sh"], STRONGEST,
                  r["vs_strongest"], r["rates"].get("round_ms_median"), r["phases_median"], r["draft_step_ms"],
                  r["draft_step_sh"], r["joint_sh_steps"], r.get("steady_rounds_used"), r.get("steady_excluded_rounds"),
                  (r.get("rates_incl_capture") or {}).get("median_tok_s"),
                  (r.get("rates_incl_capture") or {}).get("mean_tok_s"), r.get("phases_median_incl_capture"),
                  r["calls"], r.get("lean"), r["tokens"]["equal_fraction"], r["tokens"]["first"],
                  r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        if r.get("graph"):
            gr = r["graph"]
            print("REPORT %s joint graphs: captures %s, warm %s, replays %s, eager %s %s; pool %s MB; static %s MB; "
                  "capture %s s; capture rounds (round, joint ms) %s" % (
                      name, gr.get("captures"), gr.get("warm"), gr.get("replays"), gr.get("eager"),
                      gr.get("eager_why"), round(gr["pool_bytes"] / 1e6, 3) if gr.get("pool_bytes") is not None
                      else None, round(gr.get("static_bytes", 0) / 1e6, 3), gr.get("capture_s"),
                      r.get("capture_rounds")))
        if r.get("dgraph"):
            dr = r["dgraph"]
            print("REPORT %s draft graphs: keys %s, captures %s, warm %s, capture passes %s, replay passes %s, replays "
                  "%s, eager %s %s; capture %s s; capture rounds %s; joint capture rounds %s; static %s B" % (
                      name, dr.get("keys"), dr.get("captures"), dr.get("warm"), dr.get("capture_passes"),
                      dr.get("replay_passes"), dr.get("replays"), dr.get("eager"), dr.get("eager_why"),
                      dr.get("dcapture_s"), dr.get("capture_rounds"), dr.get("joint_capture_rounds"),
                      dr.get("static_bytes")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_i4h: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
