"""R2-M2 "r2base": the throughput cell of the CAPTURED R2 decode step against the SHIPPED decode, at the same B, in the
same process (benchmarks/Efficiency/r2_base_step.sh runs one B per step). Design: docs/R2_BASE.md.

WORKLOAD: the PG19 94-document 16128-token file (kstep-cpu-runs/workload/pg19-test-c16k, sha256 checked by
r2_ragged_gate.load_docs); request b reads row b % 94 (B > 94 repeats documents; the agreement is also reported over
the first min(B, 94) requests, every document once).

ARMS (in this order, one model):
  SH  the SHIPPED decode (nosa_llama.Llama.decode_inference, offload engine cache_engine.InfLLMv2Cache, the host rows
      capped at S + N + 64 by max_gen_len, as the sustained harness does), greedy, free-running, N steps, RE-WARMED
      where its warm-up-sized buffers require it (rewarm_need: a new 64-token block or a new 128-column stage-1 score
      width; the rule of the sustained harness, nosi-sustained a1c8a24 benchmarks/Efficiency/sustained_core.py:217-235).
      Its tokens are the reference sequence.
  R2  the R2 path (build_ragged: per-request prefill, widths preallocated from the horizon, no re-warm), TEACHER-FORCED
      on SH's input tokens: --warmup eager capture-safe steps (r2_capture.decode_step_fixed), then ONE captured step
      replayed for the rest. Inputs advance ON THE DEVICE (no host sync per step). Its argmax at every step is
      compared with SH's next token.

METRICS per arm (CUDA events; a step's period = start of step t to start of step t + 1):
  steady per-step ms   median period of the steady steps (SH: every step that is not a re-warm; R2: the replayed steps
                       after the capture step)
  steady decode tok/s  B / steady step; per-request rate = 1 / steady step (tok/s of one request)
  effective tok/s      B x N / (sum of all periods): includes SH's re-warms and R2's warm-up + capture
  GPU peak             torch.cuda.max_memory_allocated / reserved over the arm (prefill included)
  prefill s            the arm's prefill wall time
AGREEMENT (R2 vs SH): teacher-forced agreement over (request, step); the free-running FIRST DIVERGENCE per request =
its first teacher-forced disagreement (before it the two runs have equal inputs, so the teacher-forced run IS the
free-running run up to there); the prefill first tokens (REPORT: the logits GEMM runs at M = B in SH, M = 1 in R2's
per-request prefill).

MUST: SH-RUN and R2-RUN (N steps each), R2-ERR (no refusal bit), R2-PTR (no state data_ptr moves across the
replays). REPORT: everything else, and the registered predictions P1-P5 (docs/R2_BASE.md) with their verdicts.
EXIT: the number of failed MUST gates; 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_ragged_gate as RG        # noqa: E402

S0 = 16128
N_DOCS = 94
R_TOK, KERNEL, STRIDE, ROUND = 64, 32, 16, 128
PRED = dict(P3_ratio_b64=0.8, P3_ratio_b219=0.9, P4_eff_ratio_b64=1.5, P5_peak_ratio=1.15)


def log(msg):
    print("[r2_base %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--arms", default="SH,R2")
    a = ap.parse_args(argv)
    a.arms = [x for x in a.arms.split(",") if x]
    if a.arms not in (["SH", "R2"], ["SH"]):
        ap.error("arms must be SH,R2 or SH (R2 is teacher-forced on SH's tokens)")
    if a.batch < 1 or not 1 <= a.warmup < a.steps or a.steps > 1000:
        ap.error("--batch >= 1, --warmup in 1 .. steps - 1, --steps <= 1000 (the shipped total_cis holds S + 1024)")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def doc_rows(B: int, n_docs: int = N_DOCS):
    return [b % n_docs for b in range(B)]


def round_up(x: int, m: int) -> int:
    return (int(x) + m - 1) // m * m


def warm_state(seq: int, n_nocomp: int, m_comp: int):
    """What a warm-up at cache length seq fixes (sustained_core.py:217-221): out_len = ceil((seq + 1) / 64) and the
    stage-1 score width round128(m'), m' = m + (n >= KERNEL) (the step's own compression)."""
    mp = int(m_comp) + (1 if int(n_nocomp) >= KERNEL else 0)
    return dict(seq=int(seq), out_len=-(-(int(seq) + 1) // R_TOK), width=round_up(mp, ROUND), m_next=mp)


def rewarm_need(seq: int, n_nocomp: int, m_comp: int, warm):
    """(re-warm before this step?, why) (sustained_core.py:224-235): 'first', 'out_len', 'width', or '' (steady)."""
    if warm is None:
        return True, "first"
    now = warm_state(seq, n_nocomp, m_comp)
    if now["out_len"] != warm["out_len"]:
        return True, "out_len"
    if now["width"] != warm["width"]:
        return True, "width"
    return False, ""


def expected_rewarms(S: int, steps: int):
    """The shipped decode's re-warm steps for a prompt of S tokens over `steps` steps, from the cache lengths the
    shipped buffers follow (cache_engine.py:672-692: no_compress_k_len starts at 16 + S % 16, a compression when it
    reached 32 before the write; n_comp starts at S // 16 - 1)."""
    seq, n, m = S, STRIDE + S % STRIDE, S // STRIDE - 1
    warm, out = None, []
    for t in range(steps):
        need, why = rewarm_need(seq, n, m, warm)
        if need:
            warm = warm_state(seq, n, m)
            out.append((t, why))
        if n >= KERNEL:
            m, n = m + 1, STRIDE + 1
        else:
            n += 1
        seq += 1
    return out


def period_stats(periods, keep):
    xs = sorted(float(periods[i]) for i in keep)
    if not xs:
        return dict(n=0, median_ms=None, p10_ms=None, p90_ms=None, mean_ms=None)
    q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]   # noqa: E731
    return dict(n=len(xs), median_ms=round(statistics.median(xs), 3), p10_ms=round(q(0.1), 3), p90_ms=round(q(0.9), 3),
                mean_ms=round(statistics.fmean(xs), 3))


def rates(B: int, steps: int, periods, steady):
    total_ms = float(sum(periods))
    med = steady["median_ms"]
    return dict(steady_step_ms=med, steady_tok_s=(round(1e3 * B / med, 1) if med else None),
                per_request_tok_s=(round(1e3 / med, 3) if med else None), total_s=round(total_ms / 1e3, 3),
                effective_tok_s=round(1e3 * B * steps / total_ms, 1) if total_ms > 0 else None)


def agreement(pred, inputs, n_unique: int):
    """pred (B, N): R2's argmax at step t; inputs (B, N + 1): SH's tokens (inputs[:, t + 1] = SH's token after step t).
    The free-running first divergence of request b = its first teacher-forced disagreement (None: none in N steps)."""
    import torch
    ok = pred.detach().cpu() == inputs.detach().cpu()[:, 1:]
    Bn, Nn = ok.shape
    first = []
    for b in range(Bn):
        bad = (~ok[b]).nonzero()
        first.append(int(bad[0, 0]) if bad.numel() else None)
    div = sorted(f for f in first if f is not None)
    return dict(tf_agreement=float(ok.float().mean()), tf_agreement_unique_docs=float(ok[:n_unique].float().mean()),
                tf_disagreements=int((~ok).sum()), min_step_agreement=float(ok.float().mean(0).min()),
                n_requests=Bn, n_diverged=len(div), first_divergence_min=(div[0] if div else None),
                first_divergence_median=(div[len(div) // 2] if div else None), first_divergence=first,
                steps=Nn)


def verdicts(B: int, sh, r2, agree, expected):
    """The registered predictions P1-P5 (docs/R2_BASE.md) against this cell's numbers: [(name, verdict, detail)]."""
    out = []
    if agree is not None:
        ok = agree["n_diverged"] == 0
        out.append(("P1", "CONFIRMED" if ok else "REFUTED", "teacher-forced agreement %.6f, %d of %d requests diverged "
                    "(first at step %s)" % (agree["tf_agreement"], agree["n_diverged"], agree["n_requests"],
                                            agree["first_divergence_min"])))
    if sh is not None:
        got = [tuple(x) for x in sh["rewarms"]]
        ok = got == [tuple(x) for x in expected]
        out.append(("P2", "CONFIRMED" if ok else "REFUTED", "shipped re-warms %s (expected %s); R2 warm-up %s eager + 1 "
                    "capture" % (got, expected, r2["warmup"] if r2 else None)))
    if sh is not None and r2 is not None and sh["rates"]["steady_step_ms"] and r2["rates"]["steady_step_ms"]:
        ratio = r2["rates"]["steady_step_ms"] / sh["rates"]["steady_step_ms"]
        lim = PRED["P3_ratio_b64"] if B <= 64 else (PRED["P3_ratio_b219"] if B >= 219 else None)
        v = "REPORT" if lim is None else ("CONFIRMED" if ratio <= lim else "REFUTED")
        out.append(("P3", v, "R2 / SH steady step time %.3f (limit %s at B %d)" % (ratio, lim, B)))
        er = r2["rates"]["effective_tok_s"] / sh["rates"]["effective_tok_s"]
        v = ("CONFIRMED" if er >= PRED["P4_eff_ratio_b64"] else "REFUTED") if B == 64 else "REPORT"
        out.append(("P4", v, "R2 / SH effective tok/s %.3f (limit >= %s at B 64)" % (er, PRED["P4_eff_ratio_b64"])))
        pr = r2["gpu_peak_alloc_GB"] / sh["gpu_peak_alloc_GB"]
        out.append(("P5", "CONFIRMED" if pr <= PRED["P5_peak_ratio"] else "REFUTED",
                    "R2 / SH peak allocated %.3f (limit %s)" % (pr, PRED["P5_peak_ratio"])))
    return out


class CudaClock:
    def event(self):
        import torch
        return torch.cuda.Event(enable_timing=True)

    def record(self, e):
        e.record()

    def elapsed(self, a, b):
        return a.elapsed_time(b)

    def sync(self):
        import torch
        torch.cuda.synchronize()


class HostClock:
    def event(self):
        return [0.0]

    def record(self, e):
        e[0] = time.perf_counter()

    def elapsed(self, a, b):
        return (b[0] - a[0]) * 1e3

    def sync(self):
        pass


def r2_teacher_forced(RC, model, st, cb, cache, inputs, warmup: int, use_graph: bool, clock):
    """The R2 arm's decode loop (r2_capture module RC, all requests active): step t reads SH's input inputs[:, t] and
    the RoPE position S_b + t, both advanced ON THE DEVICE; steps < warmup run decode_step_fixed eagerly, then (with
    use_graph) ONE captured step is replayed. Returns the predictions (B, N), the periods (ms), the capture time and
    the refusal bits / pointer check."""
    import torch
    Bn, N1 = inputs.shape
    N = N1 - 1
    pred = torch.zeros((Bn, N), dtype=torch.int64, device=inputs.device)
    st.active_host.fill_(True)
    st.active.fill_(True)
    cb.pos_ids.copy_(st.pos_host.view(-1, 1))
    ev = [clock.event() for _ in range(N + 1)]
    graph, ptr0, capture_ms = None, None, None
    for t in range(N):
        clock.record(ev[t])
        cb.ids.copy_(inputs[:, t:t + 1])
        if use_graph and t >= warmup:
            if graph is None:
                clock.sync()
                t0 = time.perf_counter()
                ptr0 = RC.state_pointers(st, cb, cache)
                graph = RC.capture_step(model, st, cb, cache)
                clock.sync()
                capture_ms = (time.perf_counter() - t0) * 1e3
            graph.replay()
        else:
            RC.decode_step_fixed(model, st, cb, cache)
        pred[:, t].copy_(cb.logits[:, -1].argmax(-1))
        cb.pos_ids.add_(1)
        st.pos_host += 1
        st.steps += 1
    clock.record(ev[N])
    clock.sync()
    periods = [clock.elapsed(ev[t], ev[t + 1]) for t in range(N)]
    body_err, twin_err = RC.errors(st, cb)
    moved = None
    if ptr0 is not None:
        after = RC.state_pointers(st, cb, cache)
        moved = sorted(k for k in set(ptr0) | set(after) if ptr0.get(k) != after.get(k))
    return dict(pred=pred, periods=periods, capture_ms=capture_ms, body_err=body_err, twin_err=twin_err,
                pointers_changed=moved, pointers_checked=(len(ptr0) if ptr0 is not None else None), graph=graph)


# ------------------------------------------------------------------ the arms (GPU)
def run_shipped(model, ids, steps: int, clock):
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, pos = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    pos = pos[:, -1:] + 1
    cu = torch.arange(0, Bn + 1, dtype=torch.int, device="cuda")
    inputs = torch.empty((Bn, steps + 1), dtype=torch.int64, device="cuda")
    inputs[:, 0:1].copy_(tok)
    model.has_buffers = False
    warm, rewarms = None, []
    ev = [clock.event() for _ in range(steps + 1)]
    for t in range(steps):
        l0 = cache.layers[0]
        need, why = rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen, warm)
        if need:
            warm = warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
            rewarms.append((t, why))
            model.has_buffers = False
        clock.record(ev[t])
        lg = model.decode_inference(tok, cu, pos, cache)
        tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        inputs[:, t + 1:t + 2].copy_(tok)
        pos = pos + 1
    clock.record(ev[steps])
    clock.sync()
    periods = [clock.elapsed(ev[t], ev[t + 1]) for t in range(steps)]
    del cache
    return dict(inputs=inputs, periods=periods, rewarms=rewarms, prefill_s=round(prefill_s, 1))


def run_r2(model, ids, inputs, steps: int, warmup: int, clock):
    import torch
    from nosi import r2_capture as RC
    from nosi.r2_decode_path import build_ragged
    Bn = ids.shape[0]
    prompts = [ids[b].cpu() for b in range(Bn)]
    clock.sync()
    t0 = time.time()
    cache, st, first = build_ragged(model, prompts, steps, f4_fix=False)
    clock.sync()
    prefill_s = time.time() - t0
    first_agree = int((first.view(-1) == inputs[:, 0]).sum())
    kernels, wb = RC.default_kernels()
    cb = RC.CaptureBuffers(model, st, False, kernels, wb)
    res = r2_teacher_forced(RC, model, st, cb, cache, inputs, warmup, True, clock)
    res.update(prefill_s=round(prefill_s, 1), first_token_agreement=first_agree, widths=st.meta.get("widths"))
    del res["graph"], cb, st, cache
    return res


def gpu_peak():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 3), round(torch.cuda.max_memory_reserved() / 1e9, 3)


def free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


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
    if NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != 0:
        print("REFUSED: needs NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS=0", file=sys.stderr)
        return 20
    try:
        docs, sha = RG.load_docs(a.workload, N_DOCS, S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    rows = doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    gates, sh, r2, agree = [], None, None, None
    expected = expected_rewarms(S0, a.steps)
    with torch.inference_mode():
        free()
        log("arm SH (B %d, %d steps)" % (a.batch, a.steps))
        try:
            out = run_shipped(model, ids, a.steps, clock)
            pa, pr = gpu_peak()
            rw = {t for t, _ in out["rewarms"]}
            steady = period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
            sh = dict(rewarms=out["rewarms"], prefill_s=out["prefill_s"], steady=steady,
                      rates=rates(a.batch, a.steps, out["periods"], steady), gpu_peak_alloc_GB=pa,
                      gpu_peak_reserved_GB=pr, rewarm_ms=[round(out["periods"][t], 1) for t, _ in out["rewarms"]])
            inputs = out["inputs"]
            gates.append(("SH-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            inputs = None
        free()
        if "R2" in a.arms:
            if inputs is None:
                gates.append(("R2-RUN", True, False, "NOT_RUN: SH failed (R2 is teacher-forced on SH's tokens)"))
            else:
                log("arm R2")
                try:
                    res = run_r2(model, ids, inputs, a.steps, a.warmup, clock)
                    pa, pr = gpu_peak()
                    steady = period_stats(res["periods"], list(range(a.warmup + 1, a.steps)))
                    r2 = dict(prefill_s=res["prefill_s"], warmup=a.warmup, capture_ms=res["capture_ms"], steady=steady,
                              rates=rates(a.batch, a.steps, res["periods"], steady), gpu_peak_alloc_GB=pa,
                              gpu_peak_reserved_GB=pr, first_token_agreement=res["first_token_agreement"],
                              widths=res["widths"], body_err=res["body_err"], twin_err=res["twin_err"])
                    agree = agreement(res["pred"], inputs, min(a.batch, N_DOCS))
                    gates.append(("R2-RUN", True, True, "%d steps" % a.steps))
                    errs = res["body_err"] or any(res["twin_err"])
                    gates.append(("R2-ERR", True, not errs, "refusal bits body %s twins %s" % (
                        res["body_err"], sorted(set(res["twin_err"])))))
                    ch = res["pointers_changed"]
                    gates.append(("R2-PTR", True, ch == [], "%s of %s state pointers moved: %s" % (
                        len(ch) if ch is not None else "?", res["pointers_checked"], (ch or [])[:10])))
                except Exception as e:
                    traceback.print_exc()
                    gates.append(("R2-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            free()
    preds = verdicts(a.batch, sh, r2, agree, expected)
    meta = dict(batch=a.batch, steps=a.steps, warmup=a.warmup, S0=S0, doc_rows=rows, n_unique_docs=min(a.batch, N_DOCS),
                workload_sha256=sha, expected_rewarms=expected, attn_splits=NL._ATTN_SPLITS, predictions=PRED,
                nosi_commit=os.environ.get("BASE_COMMIT", "unknown"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0))
    out = dict(meta=meta, SH=sh, R2=r2, agreement=agree,
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:300]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    for name, arm in (("SH", sh), ("R2", r2)):
        if arm:
            print("REPORT %s B %d: steady %s ms/step, %s tok/s, %s tok/s per request, effective %s tok/s, peak %s GB, "
                  "prefill %s s" % (name, a.batch, arm["rates"]["steady_step_ms"], arm["rates"]["steady_tok_s"],
                                    arm["rates"]["per_request_tok_s"], arm["rates"]["effective_tok_s"],
                                    arm["gpu_peak_alloc_GB"], arm["prefill_s"]))
    if agree:
        print("REPORT agreement: teacher-forced %.6f, diverged %d of %d (first at %s)" % (
            agree["tf_agreement"], agree["n_diverged"], agree["n_requests"], agree["first_divergence_min"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_base: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
