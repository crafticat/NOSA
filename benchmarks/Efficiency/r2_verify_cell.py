"""Task 1.4 ROUTE 1 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the GPU
cell vr1 of the JOINT PASS (nosi/nosi/verify_rows.py: 1 draft row + K verify rows per request in one forward pass).
A NEW script: r2_base_cell.py is not edited (its functions are imported), so --ragged and the shipped cells keep their
bytes (Gate 0). Step script: benchmarks/Efficiency/r2_verify_step.sh.

ONE PROCESS, ONE MODEL, the arms in this order, each with its own shipped prefill:
  SH   the shipped decode = cell mb1 (r2_base_cell.run_shipped: C 128 = pool 65, B 216, 512 steps, NOSI_ATTN_SPLITS
       4), greedy, free-running; it records its tokens and the logits fingerprint of every (request, step) (the max
       and the float64 sum of the row: two reductions per step, declared, as rg1).
  VRX  EXACT mode, gemm "rowgroup" (R GEMM calls per weight, each the shipped call's shape): the serial residency per
       row. Prediction (TBD): every row's fingerprint equal to SH's (bitwise).
  VRC  EXACT mode, gemm "natural" (ONE GEMM call per weight over the B x R rows): the joint pass with one weight read.
  VRF  FAST mode, gemm "natural": one residency (row 0) + the extras store + the masked attention (two
       flash_attn_nosa calls merged by log-sum-exp). Route 1 of the plan.
Every verify arm is TEACHER-FORCED on SH's tokens: pass g runs the positions S0 + g R .. S0 + g R + K with SH's
inputs; its row r compares with SH's step g R + r. G = steps // R passes (512 // 4 = 128).

MEASURED per verify arm: the joint-pass period (CUDA events; pass 0 and the probe passes excluded from the steady
median), joint / SH steady step, the per-row agreement with SH's next token, the fingerprint equality with SH's, and on
the PROBE passes (every --probe-every-th pass, pass 0 excluded) the phase times per pass (CUDA events at the phase
marks of verify_rows: pre, twin, select, probe, rows / resid, attn, end, post, head) and the ONE-ROW selection of row 0
over the B requests ("probe": the same stage-1 / pooling / top-k with B entries), so select / probe = the selection
cost of R rows against 1 row. The union of the R rows' selections per (head, request): mean and max; FAST: the extras
and the COUNTED overflow (dropped extras).

MUST GATES: SH-RUN; per arm <arm>-RUN, <arm>-ERR (no body refusal bit, no twin overflow flag), <arm>-SEL0 (on every
probe the batched selection's row 0 equals the one-row selection as a SET); VRF-IDX (the first FAST pass, layer 0:
sampled entries of call A bitwise equal a plain one-entry call: the bias rows per entry are read as intended).
REPORT: <arm>-SEL0-ORDER (the same in order: the EXACT premise), VRF-OVF (dropped extras), everything else.
REGISTERED PREDICTIONS PRED_VERIFY: TBD by the lead (a None value gives REPORT).
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
import r2_base_cell as BC          # noqa: E402

ARMS = ("VRX", "VRC", "VRF")
ARM_MODE = dict(VRX=("exact", "rowgroup"), VRC=("exact", "natural"), VRF=("fast", "natural"))
# REGISTERED PREDICTIONS, TBD BY THE LEAD: fill every None and the stamp (`date -u`, L33) in a commit BEFORE the run.
#   VRX_bitwise  True / False: every VRX fingerprint equal to SH's
#   VRC_agree, VRF_agree   the lowest teacher-forced argmax agreement with SH (a fraction)
#   VRX_ratio, VRC_ratio, VRF_ratio   (low, high): the steady joint pass / SH's steady step
#   sel_ratio    (low, high): VRF's selection of R rows / its one-row selection (ms per pass, probe passes)
#   union_mean   (low, high): VRF's union of the R rows' selections per (head, request), mean
PRED_VERIFY = dict(stamp=None, VRX_bitwise=None, VRC_agree=None, VRF_agree=None, VRX_ratio=None, VRC_ratio=None,
                   VRF_ratio=None, sel_ratio=None, union_mean=None)


def log(msg):
    print("[r2_verify %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=216)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=65, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--verify-k", type=int, default=3, help="K verify rows per request (R = K + 1 rows per pass)")
    ap.add_argument("--verify-arms", default="VRX,VRC,VRF", help="a subset of %s, in run order" % ",".join(ARMS))
    ap.add_argument("--verify-extra", type=int, default=16, help="FAST: X extras slots per request (X - 1 + next)")
    ap.add_argument("--verify-twin", default="torch", choices=("torch", "bi", "fused"))
    ap.add_argument("--verify-body", default="fused", choices=("torch", "fused"))
    ap.add_argument("--probe-every", type=int, default=16, help="a probe pass every N passes (pass 0 excluded)")
    a = ap.parse_args(argv)
    a.verify_arms = [x for x in a.verify_arms.split(",") if x]
    if not a.verify_arms or any(x not in ARMS for x in a.verify_arms) or len(set(a.verify_arms)) != len(a.verify_arms):
        ap.error("--verify-arms: distinct names from %s" % (ARMS,))
    if not 1 <= a.verify_k <= 31:
        ap.error("--verify-k in 1 .. 31")
    R = a.verify_k + 1
    if a.batch < 1 or a.steps > 1000 or a.steps < 2 * R:
        ap.error("--batch >= 1, --steps in %d .. 1000 (two passes at least; the shipped total_cis holds S + 1024)"
                 % (2 * R))
    if a.verify_extra < 2:
        ap.error("--verify-extra >= 2")
    if a.probe_every < 2:
        ap.error("--probe-every >= 2")
    if not 0 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 0 .. %d" % BC.POOL_MAX)
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def probe_passes(G: int, every: int):
    """The probe passes: every `every`-th pass, pass 0 excluded."""
    return [g for g in range(1, G) if g % every == 0]


def steady_passes(G: int, every: int):
    """The steady passes: pass 0 (the first launches and compiles) and the probe passes excluded."""
    p = set(probe_passes(G, every))
    return [g for g in range(1, G) if g not in p]


def row_agreement(pred, inputs, R: int):
    """pred (B, G R): the arm's argmax per (request, position); inputs (B, >= G R + 1): SH's tokens. The teacher-forced
    agreement of each row index r (columns g R + r)."""
    ok = pred.detach().cpu() == inputs.detach().cpu()[:, 1:pred.shape[1] + 1]
    return [float(ok[:, r::R].float().mean()) for r in range(R)]


def median_phases(phases):
    """phases: one {label: ms} per probe pass. Returns {label: median ms}."""
    import statistics
    labels = sorted({k for p in phases for k in p})
    return {k: round(statistics.median([p.get(k, 0.0) for p in phases]), 3) for k in labels} if phases else {}


def _band(v, band):
    return band[0] <= v <= band[1]


def verdicts_verify(sh, vr):
    """The registered predictions PRED_VERIFY against the arms' numbers: [(name, verdict, detail)]; a registered None
    gives REPORT; an arm that did not run gives NOT_RUN."""
    P = PRED_VERIFY
    out = []
    shm = (sh or {}).get("rates", {}).get("steady_step_ms") if sh else None
    for arm in ARMS:
        r = (vr or {}).get(arm)
        if r is None:
            out.append(("%s-STEP" % arm, "NOT_RUN", "the arm did not run"))
            continue
        if arm == "VRX":
            fp = r["fingerprint"]
            want = P["VRX_bitwise"]
            got = fp["equal_fraction"] == 1.0
            v = "REPORT" if want is None else ("CONFIRMED" if got == bool(want) else "REFUTED")
            out.append(("VRX-BIT", v, "fingerprints equal to SH's: %.6f (%d unequal, first %s); registered %s" % (
                fp["equal_fraction"], fp["n_unequal"], fp["first"], want)))
        else:
            want = P["%s_agree" % arm]
            got = r["agreement"]["tf_agreement"]
            v = "REPORT" if want is None else ("CONFIRMED" if got >= want else "REFUTED")
            out.append(("%s-AGREE" % arm, v, "teacher-forced agreement %.6f (rows %s); registered >= %s" % (
                got, r["row_agreement"], want)))
        jm = r["steady"]["median_ms"]
        if shm and jm:
            ratio = jm / shm
            band = P["%s_ratio" % arm]
            v = "REPORT" if band is None else ("CONFIRMED" if _band(ratio, band) else "REFUTED")
            out.append(("%s-STEP" % arm, v, "joint pass %.3f ms / SH step %.3f ms = %.4f (R %d: %.3f SH steps per "
                        "position); registered %s" % (jm, shm, ratio, r["R"], ratio / r["R"], band)))
        else:
            out.append(("%s-STEP" % arm, "REPORT", "no steady median (SH %s, joint %s)" % (shm, jm)))
    f = (vr or {}).get("VRF")
    if f is None:
        out.append(("VR-SEL", "NOT_RUN", "VRF did not run"))
        out.append(("VR-UNION", "NOT_RUN", "VRF did not run"))
        return out
    ph = f["phases_median"]
    if ph.get("select") and ph.get("probe"):
        ratio = ph["select"] / ph["probe"]
        band = P["sel_ratio"]
        v = "REPORT" if band is None else ("CONFIRMED" if _band(ratio, band) else "REFUTED")
        out.append(("VR-SEL", v, "selection of R rows %.3f ms / one row %.3f ms = %.3f; registered %s" % (
            ph["select"], ph["probe"], ratio, band)))
    else:
        out.append(("VR-SEL", "REPORT", "no probe phases"))
    um = f["stats"].get("union_mean")
    band = P["union_mean"]
    v = "REPORT" if (band is None or um is None) else ("CONFIRMED" if _band(um, band) else "REFUTED")
    out.append(("VR-UNION", v, "union per (head, request) mean %s max %s; registered %s" % (
        um, f["stats"].get("union_max"), band)))
    return out


def arm_gates(arm, res):
    """The MUST / REPORT gates of one verify arm's result: [(name, must, passed, detail)]."""
    g = []
    g.append(("%s-ERR" % arm, True, res["body_err"] == 0 and not any(res["overflow"]),
              "body refusal bits %s; twin overflow flags %s" % (res["body_err"], sorted(set(res["overflow"])))))
    st = res["stats"]
    n = st["sel0_samples"]
    g.append(("%s-SEL0" % arm, True, n > 0 and st["sel0_set_eq"] == n,
              "row 0 of the batched selection = the one-row selection as a set: %d of %d probe layers" % (
                  st["sel0_set_eq"], n)))
    g.append(("%s-SEL0-ORDER" % arm, False, n > 0 and st["sel0_order_eq"] == n,
              "the same in order (the EXACT premise): %d of %d" % (st["sel0_order_eq"], n)))
    if res["mode"] == "fast":
        fa = res.get("fa_check") or {}
        eq = fa.get("equal")
        g.append(("VRF-IDX", True, bool(eq) and all(eq),
                  "call A entries %s against a plain one-entry call: %s" % (fa.get("entries"), eq)))
        g.append(("VRF-OVF", False, st["overflow"] == 0,
                  "dropped extras (counted): %d of %d extras" % (st["overflow"], st["extras"])))
    return g


# ------------------------------------------------------------------ the verify arm (GPU)
def run_verify(model, ids, inputs, steps: int, clock, opt):
    """ONE verify arm: its own shipped prefill (SH's cache and max_gen_len), ragged_engine.attach, verify_rows.attach,
    then G = steps // R joint passes teacher-forced on SH's tokens (module docstring). opt: K, mode, gemm, extra, twin,
    body, probe_every."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    from nosi import ragged_engine as RE
    from nosi import verify_rows as VR
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
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, _ = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    first_agree = int((logits[:, -1, :].argmax(dim=-1) == inputs[:, 0]).sum())
    del logits
    ops = RE.RaggedOps(RE.default_kernels(), gemm="natural", twin=opt["twin"], body=opt["body"], graph=False)
    rs = RE.attach(model, cache, BC.S0, steps, ops)
    gc.collect()
    torch.cuda.empty_cache()                             # the released shipped tables
    K = int(opt["K"])
    R = K + 1
    lin = RE.exact_linear(Bn * R) if opt["gemm"] == "exact" else None
    vs = VR.attach(model, rs, VR.VerifyOps(K=K, mode=opt["mode"], gemm=opt["gemm"], linear=lin, extra=opt["extra"]))
    vs.clock = clock
    if opt["mode"] == "fast":
        vs.fa_check = {}
    G = steps // R
    every = int(opt["probe_every"])
    probe = set(probe_passes(G, every))
    pred = torch.full((Bn, G * R), -1, dtype=torch.int64, device="cuda")
    fmax = torch.zeros((Bn, G * R), dtype=torch.float32, device="cuda")
    fsum = torch.zeros((Bn, G * R), dtype=torch.float64, device="cuda")
    offs = torch.arange(R, dtype=torch.int64, device="cuda")
    ev = [clock.event() for _ in range(G + 1)]
    marks = []
    for g in range(G):
        vs.probe = g in probe
        vs.marks = [] if vs.probe else None
        clock.record(ev[g])
        ids_g = inputs[:, g * R:(g + 1) * R].contiguous()
        pos_g = (BC.S0 + g * R + offs).to(torch.int32).view(1, R).expand(Bn, R).contiguous()
        lg = VR.joint_pass(model, vs, ids_g, pos_g)
        c = slice(g * R, (g + 1) * R)
        pred[:, c] = lg.argmax(dim=-1)
        fmax[:, c] = lg.amax(dim=-1)
        fsum[:, c] = lg.sum(dim=-1, dtype=torch.float64)
        if vs.probe:
            marks.append(vs.marks)
    clock.record(ev[G])
    clock.sync()
    periods = [clock.elapsed(ev[g], ev[g + 1]) for g in range(G)]
    phases = [VR.phase_ms(m, clock.elapsed) for m in marks]
    body_err, overflow = RE.errors(rs)
    stats = VR.read_stats(vs)
    res = dict(pred=pred, fp=(fmax, fsum), periods=periods, G=G, R=R, steady=steady_passes(G, every),
               phases=phases, body_err=body_err, overflow=overflow, stats=stats, fa_check=vs.fa_check,
               prefill_s=round(prefill_s, 1), first_token_agreement=first_agree, mode=opt["mode"],
               meta=dict(rs.meta, verify=vs.meta))
    del vs, rs, cache
    return res


def summarize(arm, res, inputs, fp_sh, sh, B: int):
    """The arm's record (no tensors) from run_verify's result."""
    G, R = res["G"], res["R"]
    n = G * R
    steady = BC.period_stats(res["periods"], res["steady"])
    jm = steady["median_ms"]
    shm = sh["rates"]["steady_step_ms"] if sh else None
    fpc = BC.fingerprint_compare((fp_sh[0][:, :n], fp_sh[1][:, :n]), res["fp"])
    return dict(arm=arm, mode=res["mode"], G=G, R=R, steady=steady, all_passes=BC.period_stats(res["periods"],
                list(range(1, G))), joint_over_sh=(round(jm / shm, 4) if jm and shm else None),
                committed_tok_s=(round(1e3 * B * R / jm, 1) if jm else None),
                agreement=BC.agreement(res["pred"], inputs[:, :n + 1], min(B, BC.N_DOCS)),
                row_agreement=row_agreement(res["pred"], inputs, R), fingerprint=fpc,
                phases_median=median_phases(res["phases"]), n_probes=len(res["phases"]), stats=res["stats"],
                fa_check=res["fa_check"], body_err=res["body_err"], overflow=res["overflow"],
                prefill_s=res["prefill_s"], first_token_agreement=res["first_token_agreement"], meta=res["meta"])


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
    gates, sh, vr, inputs = [], None, {}, None
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    mem = dict(after_load=BC.host_mem())
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps)" % (a.batch, a.steps))
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
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
            del out
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        mem["after_SH"] = BC.host_mem()
        BC.free()
        for arm in a.verify_arms:
            if inputs is None:
                gates.append(("%s-RUN" % arm, True, False, "NOT_RUN: SH failed (teacher-forced on SH's tokens)"))
                continue
            BC.free(host=True)
            mode, gemm = ARM_MODE[arm]
            opt = dict(K=a.verify_k, mode=mode, gemm=gemm, extra=a.verify_extra, twin=a.verify_twin,
                       body=a.verify_body, probe_every=a.probe_every)
            log("arm %s (%s)" % (arm, opt))
            try:
                res = run_verify(model, ids, inputs, a.steps, clock, opt)
                pa, pr = BC.gpu_peak()
                rec = summarize(arm, res, inputs, fp_sh, sh, a.batch)
                rec.update(gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr, opt=opt)
                vr[arm] = rec
                gates.append(("%s-RUN" % arm, True, True, "%d joint passes of R %d rows" % (res["G"], res["R"])))
                gates.extend(arm_gates(arm, res))
                del res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % arm, True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            mem["after_%s" % arm] = BC.host_mem()
            BC.free()
    preds = verdicts_verify(sh, vr)
    meta = dict(batch=a.batch, steps=a.steps, S0=BC.S0, doc_rows=rows, n_unique_docs=min(a.batch, BC.N_DOCS),
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS,
                capacity_C=BC.C_BASE + a.pool, verify_k=a.verify_k, verify_arms=a.verify_arms,
                verify_extra=a.verify_extra, verify_twin=a.verify_twin, verify_body=a.verify_body,
                probe_every=a.probe_every, predictions=PRED_VERIFY, host_mem=mem,
                base_commit=os.environ.get("BASE_COMMIT"))
    summary = dict(meta=meta, SH=sh, VERIFY=vr,
                   gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
                   predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %-16s %s %s: %s" % (g[0], "MUST" if g[1] else "REPORT", "PASS" if g[2] else "FAIL", g[3]))
    for p in preds:
        print("PREDICTION %-10s %s: %s" % p)
    if sh:
        print("REPORT SH B %d: steady %s ms/step, %s tok/s, peak %s GB, prefill %s s" % (
            a.batch, sh["rates"]["steady_step_ms"], sh["rates"]["steady_tok_s"], sh["gpu_peak_alloc_GB"],
            sh["prefill_s"]))
    for name in a.verify_arms:
        r = vr.get(name)
        if r is None:
            continue
        print("REPORT %s R %d: joint pass %s ms (%d steady), joint / SH %s, committed %s tok/s, agreement %.6f rows %s,"
              " fingerprints equal %.6f, phases %s, union mean %s max %s, extras %s overflow %s, peak %s GB" % (
                  name, r["R"], r["steady"]["median_ms"], r["steady"]["n"], r["joint_over_sh"], r["committed_tok_s"],
                  r["agreement"]["tf_agreement"], r["row_agreement"], r["fingerprint"]["equal_fraction"],
                  r["phases_median"], r["stats"]["union_mean"], r["stats"]["union_max"], r["stats"]["extras"],
                  r["stats"]["overflow"], r["gpu_peak_alloc_GB"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_verify: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
