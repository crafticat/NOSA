"""RECON7 cell (a) (KS/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): THE
PROFILER-WINDOW CELL OF THE BEST ARMS. Discriminators: R2 D5 (the device split of the best arm under graphs and
compile), R2 D6 (the decision / commit split), R2 D3 (the union pages per (layer, KV head, request) at C 128), R2 D7 (the
NEW host blocks of the profiled passes: the H2D rate), R3 D1 (the draft pass split after graphs + compile), R3 D2 (the
draft-skip flags of the NEW blocks). MEASUREMENT ONLY: no arm's computation changes; NO NEW KERNEL. The probe:
benchmarks/Efficiency/r2_recon7_probe.py (ProbeR7). The protocol: docs/NOTES_recon7.md. Step script:
benchmarks/Efficiency/r2_recon7_step.sh (cells r7pm, r7pa, r7pb).

ONE PROCESS, ONE MODEL, SH FIRST, then the arms in the order of ARMS (the I4DC cell's, benchmarks/Efficiency/
r2_i4dc_cell.py, imported, not edited: L4PTFD first, a D arm after its C arm) and the PROBED arms last:
  L4PTFDGUDCDP    = L4PTFDGUDCD with ProbeR7 (C 128 B 216: the best arm there)
  L4PTFDGUDKWCDP  = L4PTFDGUDKWCD with ProbeR7 (C 73 B 318: the best arm there)
A probed arm needs its plain arm EARLIER in the job (its PEQ reference; --arms refuses otherwise). Every arm runs
run_arm_r7 = the I4DC cell's run_arm_d line for line plus the lines marked "# R7" (the probe on a probed arm, the probe
marks per round); tests_cpu/test_recon7.py checks it.
MUST GATES: the I4DC cell's for every arm (a probed arm takes its plain arm's, under its own name, with COUNTS against the
plain arm, the check-only counters of "tent" declared); for a probed arm also -PEQ (the committed tokens, check-round
fingerprints, acceptance record and state checksums = the plain arm's, bit for bit), -PROF-PATH (every profiled pass
replayed the steady graphs: no warm, capture or eager layer inside a window), -MARKS-NEUTRAL, -MARKS-SHAPE, -PHASE-SUM,
-COUNT (every counted round counted, its recording round recorded). REPORT: -PROF, -PROF-NEUTRAL.
THE STEADY SET (L61 + R7): every arm's steady rounds exclude the capture rounds of every graph arm AND the recording rounds
of the probed arm (counted round - 1: the draft-skip record runs device ops in their D slots).
THE PROBE STATISTIC (R2 section 3.5): every probe median of the record excludes the DRAIN rounds (a request finished
before the round began); the all-rounds value and its bias are printed beside it. The drain rounds come from the probed
arm (the host counts); they hold for every arm whose tokens equal it (the EQF / PEQ gates).
THE CELL GRANT: main() refuses (exit 20) unless r2_recon7_protocol grants --cell (r7pm, r7pa or r7pb) and the arguments
are that cell's granted profile (default deny); the step script stops before the launch as well.
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
import r2_recon7_probe as RP      # noqa: E402
import r2_recon7_protocol as PROT  # noqa: E402

CCC, CD = DCC.CCC, DCC.CD
S4, CC, IA, GC, DG, LN, GR, HC, KC, UC, DC, KP, SK = (DCC.S4, DCC.CC, DCC.IA, DCC.GC, DCC.DG, DCC.LN, DCC.GR, DCC.HC,
                                                      DCC.KC, DCC.UC, DCC.DC, DCC.KP, DCC.SK)
I3C, WA, SC, BC = DCC.I3C, DCC.WA, DCC.SC, DCC.BC
PROBED = dict(L4PTFDGUDCDP="L4PTFDGUDCD", L4PTFDGUDKWCDP="L4PTFDGUDKWCD")    # a probed arm -> its plain arm
ARMS = DCC.ARMS + tuple(PROBED)
GRAPH_ARMS = DCC.GRAPH_ARMS + tuple(PROBED)
TOPK = DCC.TOPK
FORK_BLOB = DCC.FORK_BLOB
STRONGEST = DCC.STRONGEST
BUDGET_GB = DCC.BUDGET_GB
USABLE_GB = DCC.USABLE_GB
BASE_D = DCC.BASE_D


def runner_choice(arm: str, check: bool):
    """The I4DC cell's runner choice of the arm (a probed arm: its plain arm's runner)."""
    return DCC.runner_choice(PROBED.get(arm, arm), check)


runner_class = DCC.runner_class


def log(msg):
    print("[r2_recon7 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _rounds(ap, s, what):
    try:
        return sorted({int(x) for x in s.split(",") if x})
    except ValueError:
        ap.error("%s: integers" % what)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default=None, help="r7pm, r7pa or r7pb: the granted cell (r2_recon7_protocol.py)")
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3, help="K (contract 4; R = 4 = one fork tile)")
    ap.add_argument("--arms", default="L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDP",
                    help="a subset of %s, in this order (SH runs first; L4PTFD first)" % ",".join(ARMS))
    ap.add_argument("--fork-store", type=int, default=0, help="X (0: the bound 1 + (K - 1) x 64)")
    ap.add_argument("--prefill-hook", default="nocopy", choices=("nocopy", "copy"))
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"))
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--check-every", type=int, default=0, choices=(0,),
                    help="0 = the probe rounds (a probed pass needs the steady keys: no every-round checks here)")
    ap.add_argument("--require-stress", type=int, default=0, choices=(0, 1))
    ap.add_argument("--profile-rounds", default="96,112", help="probe rounds profiled (steady graphs, one window per pass)")
    ap.add_argument("--label-rounds", default="112", help="profiled rounds with the label ranges")
    ap.add_argument("--count-rounds", default="48,64,144",
                    help="probe rounds whose joint pass is counted (D2 / D3 / D7); round - 1 records the draft skips")
    ap.add_argument("--draft-marks", default="on", choices=("on", "off"))
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if not a.arms or any(x not in ARMS for x in a.arms) or len(set(a.arms)) != len(a.arms):
        ap.error("--arms: distinct names from %s" % (ARMS,))
    if a.arms != [x for x in ARMS if x in a.arms] or a.arms[0] != "L4PTFD":
        ap.error("--arms: L4PTFD first, then in the order %s (EQF compares with L4PTFD)" % ",".join(ARMS))
    for x, c in BASE_D.items():
        if x in a.arms and c not in a.arms:
            ap.error("--arms: %s needs its C arm %s (it compiles the joint helpers)" % (x, c))
    for p, x in PROBED.items():
        if p in a.arms and x not in a.arms:
            ap.error("--arms: %s needs its plain arm %s (the PEQ reference)" % (p, x))
    if a.spec != 3:
        ap.error("--spec: K 3 only (R = K + 1 = 4 rows = one fork tile)")
    if a.batch < 1 or a.steps > 1000 or a.steps < 8:
        ap.error("--batch >= 1, --steps in 8 .. 1000")
    if a.probe_every < 2 or a.spec_journal_blocks < 1:
        ap.error("--probe-every >= 2, journal >= 1")
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(TOPK, a.spec):
        ap.error("--fork-store %d: below the bound 1 + (K - 1) x 64 = %d (0 = the bound)"
                 % (a.fork_store, I3C.KA.store_slots(TOPK, a.spec)))
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    a.profile_rounds = _rounds(ap, a.profile_rounds, "--profile-rounds")
    a.label_rounds = _rounds(ap, a.label_rounds, "--label-rounds")
    a.count_rounds = _rounds(ap, a.count_rounds, "--count-rounds")
    pe = a.probe_every
    if any(g <= 0 or g % pe for g in a.profile_rounds + a.count_rounds):
        ap.error("--profile-rounds / --count-rounds: probe rounds only (g > 0, g %% %d == 0)" % pe)
    if len(a.profile_rounds) > 4 or len(a.count_rounds) > 6:
        ap.error("--profile-rounds: at most 4; --count-rounds: at most 6")
    if not set(a.label_rounds) <= set(a.profile_rounds):
        ap.error("--label-rounds: a subset of --profile-rounds")
    if set(a.count_rounds) & set(a.profile_rounds):
        ap.error("--count-rounds and --profile-rounds: disjoint")
    return a


# ------------------------------------------------------------------ the arms (GPU)
def probe_for(runner, clock, out_dir, opt):
    """The ProbeR7 of a probed arm (installed before runner.run)."""
    return RP.ProbeR7(runner, clock, out_dir, opt["arm"], opt["profile_rounds"], opt["label_rounds"],
                      opt["count_rounds"], opt["draft_marks"], opt["block_bytes"])


def run_arm_r7(model, ids, inputs, steps: int, clock, opt, FORK, arm: str, out_dir=None):
    """The I4DC cell's run_arm_d line for line (the runner of runner_choice(arm, check): a probed arm's plain arm's) plus
    the lines marked R7: the probe of a probed arm (before runner.run, removed after it; its evidence after the sync) and
    every probe round's fork phases with its round (the drain-free probe statistic)."""
    import gc
    import torch
    from nosi import ragged_engine as RE
    from nosi import spec_round as SR
    from nosi import spec_tentative_fork as STF
    from nosi import verify_rows as VR
    check = opt["check_every"] == 1
    kind, kw = runner_choice(arm, check)
    Runner = runner_class(kind, STF)
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
    pr = probe_for(runner, clock, out_dir, opt) if arm in PROBED else None   # R7
    try:   # R7 (review 3 source finding: the probe's teardown is unconditional)
        runner.run(steps)   # R7 (the core line, under try)
    finally:   # R7
        if pr is not None:   # R7
            pr.detach()   # R7
    rd = runner.read()
    rd.pop("trace_all", None)
    clock.sync()
    mem.update(decode_peak_GB=round(torch.cuda.max_memory_allocated() / gb, 3),
               decode_peak_reserved_GB=round(torch.cuda.max_memory_reserved() / gb, 3))
    phases = [VR.phase_ms(m, clock.elapsed) for m in runner.marks if m]
    layers = [VR.layer_ms(m, clock.elapsed) for m in runner.marks if m]
    mrounds = [(opt["probe_every"] * (i + 1), VR.phase_ms(m, clock.elapsed)) for i, m in enumerate(runner.marks) if m]   # R7
    res = dict(read=rd, out=runner.out.clone(), fp=(runner.fmax.clone(), runner.fsum.clone()),
               fp_ok=runner.fp_ok.clone(), K=K, probe_every=opt["probe_every"], prefill_s=prefill_s,
               first_token_agreement=first, meta=dict(rs.meta, spec=runner.meta), undo=None, mem=mem,
               joint_phases=I3C.fork_phase_summary(phases, layers), fa_check=runner.vs.fa_check,
               store=dict(X=X, store_GB=round(arena.store_bytes() / 1e9, 4), cb0_GB=round(arena.chain_bytes() / 1e9, 4)),
               compile=I3C._compile_report(FORK), state=IA.state_checksum(runner), runner_kind=kind, runner_kw=kw)
    res.update(marks_rounds=mrounds, probe=(pr.evidence(clock.elapsed) if pr is not None else None))   # R7
    runner.detach()
    del runner, rs, cache, arena
    return res


# ------------------------------------------------------------------ pure parts (CPU-tested)
def arm_gates_r7(arm, rd, rec, cpu, keep, check: bool, B: int, pool: int, steps: int, L: int, pa):
    """The I4DC cell's arm gates; a probed arm: its plain arm's gates under its own name, COUNTS against the plain arm
    (RP.counts_gate_r7: the check-only "tent" counters declared), and -PEQ against the plain arm."""
    if arm not in PROBED:
        return DCC.arm_gates(arm, rd, rec, cpu, keep, check, B, pool, steps, L, pa)
    x = PROBED[arm]
    out = []
    for n, m, ok, d in DCC.arm_gates(x, rd, rec, cpu, keep, check, B, pool, steps, L, pa):
        n2 = arm + n[len(x):] if n.startswith(x + "-") else n
        if n2 != arm + "-COUNTS":
            out.append((n2, m, ok, d))
    ref = keep.get(x)
    out.append(RP.counts_gate_r7(arm, cpu["counts"], ref["counts"] if ref else None, x, GC.COUNT_KEYS))
    out.append(RP.peq_gate_r7(arm, cpu, ref, x, steps))
    return out


def probe_record_r7(res, ref_rec, ref_marks_rounds, ref_rd, pe: int, drain, dims, B: int) -> dict:
    """The probed arm's record from its evidence: the marked passes (drain-free and all), the profiled windows against
    the plain arm (ref_rec / ref_rd after the steady restriction), the commit windows (D6), the work per pass (D5, D7),
    L7 with the fixed statistic, the counted passes (D2, D3, D7)."""
    ev, rd = res["probe"], res["read"]
    js = RP.joint_probe_summary_r7(ev["joint"], rd, drain)
    ds = RP.draft_probe_summary_g(ev["draft"], rd, drain)
    pm = (ref_rec or {}).get("phases_median") or {}
    sj, sd = pm.get("joint"), pm.get("draft")
    passes = ev["profile"]["passes"]
    jd = [p for p in passes if p.get("slot") != "commit"]
    ref_tot = RP.ref_probe_total(ref_marks_rounds, drain)
    return dict(drain=list(drain), finished=ev["finished"], probe_joint=js, probe_draft=ds,
                profile=RP.I4C.prof_record(jd, ref_rd, pe, sj, sd), commit=RP.commit_record(passes),
                work=RP.pass_work_record(passes, dims["block_bytes"]), ref_probe_total=ref_tot,
                l7=RP.l7_r7(sj, ref_tot, js), counted=RP.counted_record(ev["counted"], dims["L"], dims["H"], B,
                                                                        dims["block_bytes"]),
                recorded=ev["recorded"], recording_rounds=ev["recording_rounds"], count_rounds=ev["count_rounds"],
                profile_rounds=ev["profile"]["rounds"], label_rounds=ev["profile"]["label_rounds"],
                profile_done=ev["profile"]["done"], profile_skipped=ev["profile"]["skipped"], keep=ev["keep"])


def probe_gates_r7(arm, res, rec, xr, pe: int, L: int, K: int, draft_marks: bool):
    """The probed arm's probe gates (MUST: PROF-PATH, MARKS-NEUTRAL, MARKS-SHAPE, PHASE-SUM, COUNT; REPORT: PROF)."""
    ev, rd = res["probe"], res["read"]
    pr = ev["profile"]
    marked = sorted({x["round"] for x in ev["joint"]} | {x["round"] for x in ev["draft"]})
    g = [RP.prof_path_gate(arm, pr["rounds"], pr["done"], pr["passes"], pr["skipped"], K - 1),
         RP.marks_neutral_gate_r7(arm, rd, pe, marked, pr["done"], [c["round"] for c in ev["counted"]],
                                  ev["recording_rounds"], xr, rec),
         RP.marks_shape_r7(arm, ev["joint"], ev["draft"], L, draft_marks)]
    pj = (rec.get("probe_joint") or {}).get("all") or {}
    g.append(I4C_phase_sum(arm, pj))
    g.append(RP.count_gate(arm, ev["count_rounds"], ev["counted"], ev["recorded"], L, K - 1))
    p = rec.get("profile") or {}
    js = p.get("joint") or {}
    want = len(pr["rounds"])
    g.append(("%s-PROF" % arm, False, want > 0 and js.get("n_ok", 0) == want,
              "profiled rounds %s (done %s); joint windows %s (busy median %s ms), busy / plain steady joint %s; commit "
              "windows %s; draft busy per round %s; errors %s" % (
                  pr["rounds"], pr["done"], js.get("n_ok"), js.get("busy_ms_median"), js.get("busy_over_steady_median"),
                  (rec.get("commit") or {}).get("n"), [(x["round"], x["busy_ms"]) for x in p.get("draft") or []],
                  [q.get("error") for q in pr["passes"] if q.get("error")][:3])))
    return g


def I4C_phase_sum(arm, js):
    """<arm>-PHASE-SUM (MUST; I1P's bound, I4C's gate) over every fully marked joint pass."""
    return RP.I4C.phase_sum_gate(arm, js)


def decision_lines(name, r, dims) -> list:
    """The REPORT lines that the reconciliation reads (docs/NOTES_recon7.md section 3.1: what each value decides)."""
    out = []
    for w in r.get("work") or []:
        out.append("R7 D5/D7 %s round %s %s: rows %s (active %s); busy %s ms; GPU ops %s (kernels %s, copies %s); GEMM %s, "
                   "fork %s, H2D kernels %s (%s ms); graph launches %s; unranged ops %s; NEW host blocks %s = %s bytes "
                   "-> %s GB/s; path ok %s %s" % (
                       name, w["round"], w["slot"], w["rows_e"], w["active"], w["busy_ms"], w["gpu_ops"], w["kernels"],
                       w["copies"], w["gemm_calls"], w["fork_calls"], w["h2d_kernels"], w["h2d_ms"], w["graph_launches"],
                       w["unranged_ops"], w["new_blocks"], w["h2d_bytes"], w["h2d_GBps"], w["path_ok"], w["path"]))
    c = r.get("commit") or {}
    out.append("R7 D6 %s commit windows %s (rounds %s): busy median %s ms, wall %s ms; labelled device ms %s; commit phase "
               "(decide .. twins) %s, decision (decide + accept) %s, commit + twins %s" % (
                   name, c.get("n"), c.get("rounds"), c.get("busy_ms_median"), c.get("wall_ms_median"), c.get("labels"),
                   c.get("phase_device_ms"), c.get("decision_device_ms"), c.get("commit_device_ms")))
    k = r.get("counted") or {}
    for p in k.get("passes") or []:
        out.append("R7 D2/D3/D7 %s counted round %s: NEW %s (meter %s; window %s, store %s, other %s); flagA %s (share %s; "
                   "window %s, store %s); flagB window %s (share %s), both %s; skips recorded %s; union pages per "
                   "(layer, KV head, request) %s (per L x H x B %s, max %s); pages per row %s; store-fill listed per "
                   "(KV head, request) sum %s max %s over %s streams" % (
                       name, p["round"], p["new"], p["meter_new"], p["new_win"], p["new_store"], p["new_other"],
                       p["flagA"], p["flagA_share"], p["flagA_win"], p["flagA_store"], p["flagB_win"],
                       p["flagB_share_win"], p["flagAB_win"], p["skips"], p["union_pages_per_stream"],
                       p["union_pages_per_lhb"], p["union_max"], p["pages_per_row"], p["store_listed_sum"],
                       p["store_listed_max"], p["store_listed_streams"]))
    out.append("R7 D2 %s total: flagA share %s, flagB share (window) %s; D3 union pages per stream median %s; pages per row "
               "median %s" % (name, k.get("flagA_share"), k.get("flagB_share_win"),
                               k.get("union_pages_per_stream_median"), k.get("pages_per_row_median")))
    js = r.get("probe_joint") or {}
    ds = r.get("probe_draft") or {}
    l7 = r.get("l7") or {}
    out.append("R7 PROBE-STAT %s: drain rounds %s; probe joint median without drain %s, all %s, bias %s (excluded %s); "
               "marked joint total %s; marked one-row replay total %s; plain arm's probe mark sum %s; L7 without drain "
               "%s; L7 all %s" % (
                   name, r.get("drain"), (js.get("probe_joint") or {}).get("nodrain"),
                   (js.get("probe_joint") or {}).get("all"), (js.get("probe_joint") or {}).get("bias"),
                   (js.get("probe_joint") or {}).get("excluded"), js.get("total"), ds.get("total"),
                   r.get("ref_probe_total"), l7.get("nodrain"), l7.get("all")))
    dn = (ds.get("nodrain") or {})
    out.append("R7 R3-D1 %s marked one-row replay passes (without drain): labels median %s; total %s; outside the passes per "
               "round %s" % (name, dn.get("labels_median"), dn.get("total_median"), dn.get("outside_median")))
    return out


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    why = PROT.refusal(a.cell, "r2_recon7_cell", a)
    if why is not None:
        print(why, file=sys.stderr)
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
    if any(x in a.arms for x in tuple(CCC.BASE) + tuple(BASE_D) + tuple(PROBED)) and \
            not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        print("REFUSED: a C, D or probed arm needs TORCHINDUCTOR_CACHE_DIR (the step script sets it; never the home "
              "cache)", file=sys.stderr)
        return 20
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    gates, sh, sp, inputs, keep, rds, caps, res_of = [], None, {}, None, {}, {}, {}, {}
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
        stopped = None
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
                       prefill_hook=a.prefill_hook, profile_rounds=list(a.profile_rounds),
                       label_rounds=list(a.label_rounds), count_rounds=list(a.count_rounds),
                       draft_marks=a.draft_marks == "on", block_bytes=dims["block_bytes"])
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_arm_r7(model, ids, inputs, a.steps, clock, opt, FORK, arm, out_dir=a.out)
                pa, pr = BC.gpu_peak()
                rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, a.batch, a.steps, dims)
                rd = res["read"]
                rec.update(arm=arm, gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt,
                           union=I3C.I2C.union_record(rd, L), fork=I3C.fork_record(rd),
                           joint_phases_fork=res["joint_phases"], store=res["store"], compile=res["compile"],
                           mem_arena=res["mem"], lean=rd.get("lean"), graph=rd.get("graph"), dgraph=rd.get("dgraph"),
                           hoist=rd.get("hoist"), capture_rounds=GC.capture_rounds(rd), runner=res["runner_kind"],
                           runner_kw=res["runner_kw"], cc=rd.get("compile"))
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
                res_of[arm] = dict(marks_rounds=res["marks_rounds"], probe=res["probe"], read=rd)
                if arm in GRAPH_ARMS:
                    caps[arm] = HC.capture_rounds_x(rd)
                gates.append(("%s-RUN" % arm, True, True, "%d rounds of K %d" % (rd["rounds"], a.spec)))
                gates.extend((n.replace("L4PT-", arm + "-", 1), m, ok, d) for n, m, ok, d in
                             WA.arm_gates_w("L4PT", res, rec, a.steps, require_stress=bool(a.require_stress)))
                gates.extend(I3C.fork_gates(arm, res, L, check_t, bool(a.require_stress)))
                gates.append(IA.lean_gate(arm, dict(rd, K=a.spec), L))
                cpu = dict(out=res["out"].cpu(), fp=(res["fp"][0].cpu(), res["fp"][1].cpu()), fp_ok=res["fp_ok"].cpu(),
                           acct=IA.acct_record(rd), state=res["state"], counts=GC.counts_record(rd), keep=rd.get("keep"))
                if arm == "L4PTFD":
                    gates.append(IA.mem_record(arm, res["mem"], pa))
                gates.extend(arm_gates_r7(arm, rd, rec, cpu, keep, check_t, a.batch, a.pool, a.steps, L, pa))
                keep[arm] = cpu
                res = None
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
                stopped = arm                                          # STOP THE RUN: no later arm on leaked state
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
            if stopped is not None:
                for later in a.arms[a.arms.index(arm) + 1:]:
                    gates.append(("%s-RUN" % later, True, False, "NOT_RUN: the run stopped after %s's exception" % arm))
                break
    probed = [p for p in PROBED if p in sp and (res_of.get(p) or {}).get("probe")]
    recr = sorted({g for p in probed for g in res_of[p]["probe"]["recording_rounds"]})
    xr = sorted(set(HC.excluded_rounds(caps)) | set(recr))            # L61 + the recording rounds, every arm
    for arm, rec in sp.items():
        HC.restrict(rec, rds[arm], sh, a.batch, a.spec, a.probe_every, xr)
    drain = res_of[probed[-1]]["probe"]["drain"] if probed else []
    for arm, rec in sp.items():
        rec["probe_total_nodrain"] = RP.ref_probe_total(res_of[arm]["marks_rounds"], drain)
    for p in probed:
        x = PROBED[p]
        r = sp[p]
        r.update(probe_record_r7(res_of[p], sp.get(x), (res_of.get(x) or {}).get("marks_rounds"), rds.get(x),
                                 a.probe_every, drain, dims, a.batch))
        gates.extend(probe_gates_r7(p, res_of[p], r, xr, a.probe_every, L, a.spec, a.draft_marks == "on"))
        jr, jm = (sp.get(x) or {}).get("phases_median", {}).get("joint"), r["phases_median"].get("joint")
        gates.append(("%s-PROF-NEUTRAL" % p, False, bool(jr and jm), "steady joint median %s vs %s's %s (ratio %s)" % (
            jm, x, jr, round(jm / jr, 4) if (jm and jr) else None)))
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, spec_k=a.spec, arms=a.arms, fork_store=a.fork_store,
                fork_store_slots=I3C.fork_store_slots("L4PTF", a.spec, a.fork_store), prefill_hook=a.prefill_hook,
                tu_gemm=a.tu_gemm, spec_journal_blocks=a.spec_journal_blocks, spec_twin=a.spec_twin,
                spec_body=a.spec_body, spec_graph=a.spec_graph, spec_redirect=a.spec_redirect,
                probe_every=a.probe_every, check_every=a.check_every, require_stress=a.require_stress,
                profile_rounds=a.profile_rounds, label_rounds=a.label_rounds, count_rounds=a.count_rounds,
                draft_marks=a.draft_marks, cell=a.cell, grant=PROT.grant(a.cell),
                labels=dict(joint=RP.JOINT_LAYER, jointg=RP.JOINT_G, jointgk=RP.JOINT_GK, draftg=RP.DRAFT_G,
                            commit=RP.COMMIT_LABELS, commit_phase=RP.COMMIT_PHASE),
                capture_rounds=caps, recording_rounds=recr, steady_excluded_rounds=xr, drain_rounds=drain,
                strongest_tok_s=STRONGEST, budget_GB=BUDGET_GB, usable_GB=USABLE_GB, dims=dims, host_mem=mem,
                compile=dict(torch=torch.__version__, inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                             kernel_cache=os.environ.get("TRITON_CACHE_DIR")),
                base_commit=os.environ.get("BASE_COMMIT"), fork_blob=FORK_BLOB)
    summary = dict(meta=meta, SH=sh, SPEC=sp,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-24s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    print("REPORT steady exclusion: capture rounds %s; recording rounds %s; excluded from every arm: %s; drain rounds %s"
          % (caps, recr, xr, drain))
    if sh:
        print("REPORT SH B %d C %d: steady %s ms/step, tok/s median %s mean %s effective %s, peak %s GB" % (
            a.batch, BC.C_BASE + a.pool, sh["rates"]["steady_step_ms"], sh["tok_s"]["median_tok_s"],
            sh["tok_s"]["mean_tok_s"], sh["tok_s"]["effective_tok_s"], sh["gpu_peak_alloc_GB"]))
    for name in a.arms:
        r = sp.get(name)
        if r is None:
            continue
        print("REPORT %s K %d: %d rounds; committed tok/s median %s mean %s effective %s; vs SH %s; round median %s ms; "
              "phases %s; draft step %s ms = %s SH; joint %s SH steps; steady rounds %s (excluded %s); probe mark "
              "total without drain %s; tokens equal %.6f (first %s); peak %s GB (with the prefill %s GB)" % (
                  name, r["K"], r["rounds"], r["rates"]["median_tok_s"], r["rates"]["mean_tok_s"],
                  r["rates"]["effective_tok_s"], r["vs_sh"], r["rates"].get("round_ms_median"), r["phases_median"],
                  r["draft_step_ms"], r["draft_step_sh"], r["joint_sh_steps"], r.get("steady_rounds_used"),
                  r.get("steady_excluded_rounds"), r.get("probe_total_nodrain"), r["tokens"]["equal_fraction"],
                  r["tokens"]["first"], r.get("gpu_peak_alloc_GB"), r.get("peak_with_prefill_GB")))
        if name in PROBED:
            for line in decision_lines(name, r, dims):
                print(line)
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_recon7: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
