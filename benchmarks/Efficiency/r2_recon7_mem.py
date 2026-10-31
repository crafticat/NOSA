"""RECON7 cell (c), R3 discriminator D3 (KS/analysis/plan/RECON7_R3_draft_transfer_capacity_schedule.md section 6; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE C 81 MEMORY PROBE. Question: does our layout fit B 287 (the original model's
anchor) and more at C 81 (pool 18)? R3's MODEL (linear in C through the C 73 and C 128 fit lines) says B_fit 321.5 -
322.8 at the usable 83.75 GB; nothing at C 81 is measured. Second question: the store's PEAK occupancy per (layer, KV
head, request): R3 measured a MEAN of 1.03 / 2.28 listed blocks of 128 slots (> 98 % idle) but no peak.

THE I4M METHOD (benchmarks/Efficiency/r2_i4m_cell.py, a BYTE COPY of nosi-r2-i4m f7fb2c5, imported, not edited; its
WS-M base r2_wsm_cell.py likewise): one process, one model; for each --batches B (ascending): SH at (C, B) (the
same-job baseline and the arm's first token), then L4PTFD through probe_arm_r7 = I4M's probe_arm_x line for line plus
the lines marked "# R7" (the occupancy tap: attached after the runner is built, removed before its detach) for
--probe-rounds rounds at the full run's horizon (--steps 512). The probe's marks split the peak into PREFILL and DECODE
(WM.probe_record); WM.probe_check applies the L53 rule. THE LINES: the prefill line through the probes' prefill peaks,
the decode line through their decode peaks projected to the run's round bound (WM.project_decode), both MEASURED at
this C (two B; I4M had one probe point plus a MEASURED anchor, none exists at C 81); B_fit by I4M.fit_arm_x at the cap
84.0 GB with the margins 0 / 0.10 / 0.25 GB and at the usable ceiling 83.75 GB (lesson L60).
THE OCCUPANCY TAP (OccTap): rs.ops.kernels["gather_k"] wrapped for the probe arm's run; on every store fill (the
destination is vs.ext_k: STF:172-174) the listed host blocks per (KV head, request) = (ids[..., 1:] >= 0).sum(-1): the
per-layer maximum, the histogram, the sum. The probe arm is not timed (I4M's rule): the tap's device ops are allowed.
MUST GATES (per B, prefixed "B<b>-"): SH-RUN, L4PTFD-RUN and I4M's probe-mode arm gates (ERR, D1, D3, SPAN, RESID,
ATTN1, FILL1, LEAN, COMMIT-P), L4PTFD-MEM, PR-ROUNDS, PR-FILL, PR-FLAT, PR-PRE, OCC (the tap saw one store fill per layer
and joint pass); FIT (two distinct B with a probe record each). REPORT: the lines, B_fit, the peak at B 287, the
occupancy. EXIT: the number of failed MUST gates (0-19); 20 refused (cell r7mem not granted: r2_recon7_protocol.py); 30 an exception
outside an arm.
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
import r2_i4m_cell as I4M          # noqa: E402  nosi-r2-i4m f7fb2c5, byte copy (tests check the blob)
import r2_recon7_protocol as PROT  # noqa: E402

WM, I4A, I3C, LN, WA, SC, BC = I4M.WM, I4M.I4A, I4M.I3C, I4M.LN, I4M.WA, I4M.SC, I4M.BC
state_checksum = I4M.state_checksum
ARM = I4M.ARM
BUDGET_GB = I4M.BUDGET_GB
USABLE_R3_GB = 83.75               # R3 row 3 (4): the usable ceiling after SH in the same process (lesson L60)
ANCHOR_B = 287                     # the original model's batch at C 81 (REQ)
R3_MODEL_BFIT = (321.5, 322.8)     # R3 row 3: B_fit(C 81) MODEL (linear in C through the two measured fit lines)
X_SLOTS = 129                      # the store: 1 + (K - 1) x 64 slots per request (slot 0 = CB1)


def log(msg):
    print("[r2_recon7_mem %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default=None, help="r7mem: the granted cell (r2_recon7_protocol.py)")
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batches", default="64,192", help="the probe points (two or more distinct B)")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=18, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--spec", type=int, default=3)
    ap.add_argument("--fork-store", type=int, default=0)
    ap.add_argument("--prefill-hook", default="nocopy", choices=("nocopy", "copy"))
    ap.add_argument("--tu-gemm", default="natural", choices=("natural", "rowgroup"))
    ap.add_argument("--spec-journal-blocks", type=int, default=256)
    ap.add_argument("--spec-twin", default="fused", choices=("torch", "bi", "fused"))
    ap.add_argument("--spec-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--spec-graph", default="on", choices=("on", "off"))
    ap.add_argument("--spec-redirect", default="on", choices=("on", "off"))
    ap.add_argument("--probe-every", type=int, default=16)
    ap.add_argument("--probe-rounds", type=int, default=49)
    ap.add_argument("--margin-gb", type=float, default=0.10)
    a = ap.parse_args(argv)
    try:
        a.batches = sorted({int(x) for x in a.batches.split(",") if x})
    except ValueError:
        ap.error("--batches: integers")
    if len(a.batches) < 2 or a.batches[0] < 1:
        ap.error("--batches: at least two distinct B >= 1 (a line needs two points)")
    if a.spec != 3:
        ap.error("--spec: K 3 only")
    if a.steps > 1000 or a.steps < 8 or a.probe_every < 2 or a.probe_rounds < 2 * a.probe_every + 1:
        ap.error("--steps in 8 .. 1000, --probe-every >= 2, --probe-rounds >= 2 x probe_every + 1 (the flat window)")
    if not 1 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 1 .. %d" % BC.POOL_MAX)
    if a.fork_store != 0 and a.fork_store < I3C.KA.store_slots(I4A.TOPK, a.spec):
        ap.error("--fork-store below the bound")
    return a


def arm_opt(a) -> dict:
    """I4M.arm_opt's keys (check_every 0: the timing cells' value)."""
    return dict(arm=ARM, K=a.spec, journal=a.spec_journal_blocks, tu_gemm=a.tu_gemm, twin=a.spec_twin,
                body=a.spec_body, graph=a.spec_graph == "on", redirect=a.spec_redirect == "on",
                probe_every=a.probe_every, check_every=0, fork_store=a.fork_store, prefill_hook=a.prefill_hook)


# ------------------------------------------------------------------ pure parts (CPU-tested)
def lines_from(points_pre, points_dec):
    """The two lines of (L4PTFD, C) through the MEASURED probe points: dict(prefill, decode) as (a, b) GB."""
    return dict(prefill=WM.fit_line(points_pre), decode=WM.fit_line(points_dec))


def fit_record(C: int, lines, steps: int, margin_GB: float) -> dict:
    """B_fit at the cap with the margins 0 / 0.10 / 0.25 GB and the cell's margin; at the usable ceiling 83.75 GB; the
    predicted peak at the anchor B 287; the R3 MODEL band beside it (REPORT)."""
    out = I4M.fit_arm_x(C, lines, margin_GB, steps=steps)
    out["margins"] = {("%.2f" % m): I4M.fit_arm_x(C, lines, m, steps=steps)["B_fit"] for m in I4M.MARGINS_REPORT}
    out["B_fit_usable"] = I4M.fit_arm_x(C, lines, 0.0, budget_GB=USABLE_R3_GB, steps=steps)["B_fit"]
    out["peak_at_anchor_GB"] = round(I4M.arm_peak_x(ANCHOR_B, lines), 4)
    out["anchor_fits"] = out["peak_at_anchor_GB"] <= USABLE_R3_GB
    out["r3_model_bfit"] = R3_MODEL_BFIT
    return out


class OccTap:
    """The store-fill occupancy tap (module docstring): kernels["gather_k"] wrapped for one run (attach after the runner,
    detach before its detach). Device counters per layer; a host count of the fills per layer."""

    def __init__(self, L: int, X: int = X_SLOTS):
        self.L, self.X = int(L), int(X)
        self.kern = self.orig = self.vs = None
        self.fills = [0] * self.L
        self.layer_of = {}
        self.mx = self.hist = self.sum = self.npos = None

    def attach(self, runner) -> None:
        self.kern = runner.rs.ops.kernels
        self.vs = runner.vs
        self.layer_of = {lay.cache_engine._k_cpu.data_ptr(): l for l, lay in enumerate(runner.rs.cache.layers)}
        self.orig = self.kern["gather_k"]
        self._fn = self._gk                                            # ONE bound method: the identity check below
        self.kern["gather_k"] = self._fn

    def detach(self) -> None:
        if self.kern is None:
            return
        if self.kern["gather_k"] is not self._fn:
            raise RuntimeError("OccTap: gather_k is not the tap")
        self.kern["gather_k"] = self.orig
        self.kern = None

    def _gk(self, gpu, cpu, ids, bs):
        out = self.orig(gpu, cpu, ids, bs)
        ext = getattr(self.vs, "ext_k", None)
        if ext is None or gpu.data_ptr() != ext.data_ptr():
            return out
        import torch
        l = self.layer_of.get(cpu.data_ptr())
        if l is None:
            return out
        if self.mx is None:
            dev = ids.device
            self.mx = torch.zeros(self.L, dtype=torch.int64, device=dev)
            self.hist = torch.zeros(self.X + 1, dtype=torch.int64, device=dev)
            self.sum = torch.zeros(self.L, dtype=torch.int64, device=dev)
            self.npos = torch.zeros(self.L, dtype=torch.int64, device=dev)
        per = (ids[..., 1:] >= 0).sum(-1)                              # (H, B): listed host blocks
        self.mx[l] = torch.maximum(self.mx[l], per.max())
        self.hist += torch.bincount(per.flatten().clamp(max=self.X), minlength=self.X + 1)
        self.sum[l] += per.sum()
        self.npos[l] += (per > 0).sum()
        self.fills[l] += 1
        return out

    def state(self) -> dict:
        if self.mx is None:
            return dict(fills=list(self.fills), max=None)
        h = [int(x) for x in self.hist.cpu()]
        top = max((i for i, v in enumerate(h) if v), default=0)
        return dict(fills=list(self.fills), max_by_layer=[int(x) for x in self.mx.cpu()], max=int(self.mx.max()),
                    sum_by_layer=[int(x) for x in self.sum.cpu()], streams_with_listed=[int(x) for x in self.npos.cpu()],
                    hist=h[:top + 1], X=self.X, idle_share_at_max=round(1.0 - int(self.mx.max()) / (self.X - 1), 4))


def occ_gate(name: str, occ, rd, L: int):
    """<name>-OCC (MUST): one store fill per layer and joint pass (STF:172-174) seen by the tap."""
    jp = ((rd.get("tent") or {}).get("calls") or {}).get("joint_passes", 0)
    f = occ.get("fills") or []
    ok = jp > 0 and len(f) == L and all(x == jp for x in f)
    return ("%s-OCC" % name, True, ok, "store fills per layer %s (joint passes %d); peak listed per (layer, KV head, "
            "request) %s of %d slots" % (sorted(set(f)), jp, occ.get("max"), X_SLOTS - 1))


# ------------------------------------------------------------------ the probe arm (GPU; not a timed path)
def probe_arm_r7(model, ids, inputs, steps: int, clock, opt, FORK, lean: bool, cmem, occ):
    """I4M's probe_arm_x (r2_i4m_cell.py, f7fb2c5) line for line plus the lines marked R7: the occupancy tap attached
    after the runner is built and removed right after the run (before the runner's detach checks the kernels)."""
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
    occ.attach(runner)   # R7
    runner.run(steps, max_rounds=cmem.rounds)
    occ.detach()   # R7
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
    why = PROT.refusal(a.cell, "r2_recon7_mem", a)
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
    try:
        docs, sha = BC.RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    C = BC.C_BASE + a.pool
    gates, per_b = [], {}
    FORK, g0 = I3C.import_fork()
    gates.append(g0)
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    hmem = dict(after_load=BC.host_mem())
    L = len(model.layers)
    lay0 = model.layers[0]
    dims = dict(L=L, H=int(lay0.num_key_value_heads), D=int(lay0.head_dim), topk=I4A.TOPK,
                block_bytes=64 * int(lay0.head_dim) * 2 * 2 + 64 * 2)
    opt = arm_opt(a)
    pp, dp = [], []
    for B in a.batches:
        tag = "B%d" % B
        rows = BC.doc_rows(B)
        ids = docs[rows].contiguous().to("cuda")
        sh, inputs, rec, prec, chk, occ = None, None, None, None, None, None
        with torch.inference_mode():
            BC.free(host=True)
            log("%s arm SH (C %d, %d steps)" % (tag, C, a.steps))
            fp_sh = (torch.zeros((B, a.steps), dtype=torch.float32, device="cuda"),
                     torch.zeros((B, a.steps), dtype=torch.float64, device="cuda"))
            try:
                out = BC.run_shipped(model, ids, a.steps, clock, fp=fp_sh)
                pa, pr = BC.gpu_peak()
                rw = {t for t, _ in out["rewarms"]}
                steady = BC.period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
                sh = dict(steady=steady, rates=BC.rates(B, a.steps, out["periods"], steady), rewarms=out["rewarms"],
                          prefill_s=out["prefill_s"], gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr,
                          device_used_end_GB=out["device_used_end_GB"])
                sh["tok_s"] = SC.sh_rates(B, sh)
                inputs = out["inputs"]
                gates.append(("%s-SH-RUN" % tag, True, True, "%d steps" % a.steps))
                gates.append(I4A.mem_record("%s-SH" % tag, "prefill + decode", pa))
                del out
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-SH-RUN" % tag, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            hmem["after_SH_%s" % tag] = BC.host_mem()
            BC.free()
            if inputs is None or FORK is None:
                gates.append(("%s-%s-RUN" % (tag, ARM), True, False, "NOT_RUN: SH or the fork failed"))
            else:
                BC.free(host=True)
                log("%s arm %s probe (%d rounds)" % (tag, ARM, a.probe_rounds))
                try:
                    cmem = I4M.ProbeMem(a.probe_rounds)
                    tap = OccTap(L, I3C.fork_store_slots("L4PTF", a.spec, a.fork_store))
                    res = probe_arm_r7(model, ids, inputs, a.steps, clock, opt, FORK, True, cmem, tap)
                    pa, pr = BC.gpu_peak()
                    rec = WA.summarize_w("L4PT", res, inputs, fp_sh, sh, B, a.steps, dims)
                    rec.update(arm=ARM, gpu_peak_after_end_GB=pa, opt=opt, mem_arena=res["mem"],
                               store=res["store"], lean=res["read"].get("lean"))
                    prec = WM.probe_record(cmem.marks, cmem.trace)
                    rec.update(gpu_peak_alloc_GB=prec["decode_GB"], probe=prec, peak_with_prefill_GB=prec["arm_GB"])
                    occ = tap.state()
                    rd = res["read"]
                    gates.append(("%s-%s-RUN" % (tag, ARM), True, True, "%d rounds" % rd["rounds"]))
                    gates.extend(("%s-%s" % (tag, n), m, ok, d) for n, m, ok, d in
                                 I4M.arm_gates_x("probe", res, rec, L, a.steps, a.probe_rounds))
                    gates.append(I4M.mem_gate("%s-%s-MEM" % (tag, ARM), rec["peak_with_prefill_GB"],
                                              what="arena %s" % res["mem"]))
                    chk = WM.probe_check(prec, a.probe_rounds, a.probe_every)
                    gates.extend(("%s-%s" % (tag, n), m, ok, d) for n, m, ok, d in
                                 WM.probe_gates(ARM, rd, prec, chk, BC.S0))
                    n, m, ok, d = I4M.pr_pre_gate(prec, rec.get("mem_arena"))
                    gates.append(("%s-%s" % (tag, n), m, ok, d))
                    gates.append(occ_gate(tag, occ, rd, L))
                    pp.append((B, float(prec["prefill_GB"])))
                    dp.append((B, float(WM.project_decode(prec, I4M.rounds_bound(a.steps),
                                                          chk.get("growth_B_per_round")))))
                    res = None
                except Exception as e:
                    traceback.print_exc()
                    gates.append(("%s-%s-RUN" % (tag, ARM), True, False, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                            str(e)[:300])))
                hmem["after_%s_%s" % (ARM, tag)] = BC.host_mem()
                BC.free()
        per_b[B] = dict(SH=sh, ARM=rec, probe_check=chk, occupancy=occ)
    fit = None
    if len({b for b, _ in pp}) >= 2:
        fit = fit_record(C, lines_from(pp, dp), a.steps, a.margin_gb)
        fit.update(points=dict(prefill=pp, decode=dp))
    gates.append(("FIT", True, fit is not None, "probe points prefill %s decode %s" % (pp, dp)))
    meta = dict(batches=a.batches, steps=a.steps, S0=BC.S0, workload_sha256=sha, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=C, opt=opt, probe_rounds=a.probe_rounds, margin_GB=a.margin_gb, budget_GB=BUDGET_GB,
                usable_r3_GB=USABLE_R3_GB, anchor_B=ANCHOR_B, r3_model_bfit=R3_MODEL_BFIT, dims=dims, host_mem=hmem,
                cell=a.cell, grant=PROT.grant(a.cell), base_commit=os.environ.get("BASE_COMMIT"),
                fork_blob=I4A.FORK_BLOB)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, per_batch={str(k): v for k, v in per_b.items()}, fit=fit,
                       gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates]),
                  f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-22s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for B, v in per_b.items():
        p = (v.get("ARM") or {}).get("probe") or {}
        print("R7 D3 C %d B %d: prefill %s GB, decode %s GB (reserved %s); occupancy %s" % (
            C, B, p.get("prefill_GB"), p.get("decode_GB"), p.get("decode_reserved_GB"), v.get("occupancy")))
    if fit:
        print("R7 D3 FIT C %d: prefill line %s, decode line %s through %s; B_fit %s (margin %s GB; by margin %s); B_fit "
              "at the usable %.2f GB %s; peak at B %d %s GB (fits the usable ceiling %s); R3 MODEL B_fit %s" % (
                  C, fit["prefill_line"], fit["decode_line"], fit["points"], fit["B_fit"], fit["margin_GB"],
                  fit["margins"], USABLE_R3_GB, fit["B_fit_usable"], ANCHOR_B, fit["peak_at_anchor_GB"],
                  fit["anchor_fits"], R3_MODEL_BFIT))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_recon7_mem: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
