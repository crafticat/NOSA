"""R2-M2 r2base P3: the PROFILING CELL M-P3 (docs/R2_BASE_P3.md section 4). Where do the +6 ms of the captured R2 step
against the shipped step at B 64 go? One process, one model, the r2base workload (PG19 94 documents at 16128 tokens,
request b = row b % 94), B 64.

ARMS AND WINDOWS (a window = 16 steady steps; every arm's steps are timed with CUDA events in one window and profiled
with torch.profiler (CPU + CUDA activity, Kineto chrome trace) in the next, so the profiler never inflates a period):
  SH   the shipped decode (as r2_base_cell.run_shipped: offload engine, re-warmed by rewarm_need), SH_STEPS steps;
       timed SH_TIMED, profiled SH_PROF (no re-warm inside: MUST SH-STEADY). Per-instance record_function ranges on
       the cache's own methods: sh.k (update_no_compress_k_decode, update_compress_k_decode), sh.cis
       (update_uncompressed_cis, update_cis), sh.kv (decode_update_kv). Its tokens are the R2 arm's inputs.
  R2G  the captured R2 step (r2_capture: build_ragged, CaptureBuffers, 2 eager steps, the capture at step 2), teacher-
       forced on SH's tokens: replays timed R2G_TIMED, profiled R2G_PROF (no ranges: one graph launch per step).
  R2E  the same state continued EAGERLY (decode_step_fixed: the same kernels): timed R2E_TIMED, profiled R2E_PROF with
       ranges r2.twin (twin.step_fixed), r2.body (body.decode_fixed), r2.stage1, r2.pool (the module functions).
OUTPUT per arm (per step, from the trace): GPU ops (kernels, memcpy, memset) and their summed device ms, by NAME CLASS
(classify: GEMM, STAGE1, ATTN, POOL, SELECT, H2D, DIFF, WB, FUSED, SMALL, COPY, OTHER) and by RANGE (attribution:
kernel -> its launch (correlation id, cuda_runtime / cuda_driver) -> the innermost range on the launching thread);
every kernel name with its class; the period statistics and the gap (period - device ms). The traces are kept
(gzip). The registered predictions Q1-Q6 (docs/R2_BASE_P3.md, commit 0ebf84f) get a verdict each (REPORT).
MUST: SH-RUN, SH-STEADY, R2-RUN (refusal bits 0), PROF (the profiler saw GPU work in SH and R2E: else blind).
EXIT: the number of failed MUST gates; 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import re
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_base_cell as BC          # noqa: E402
import r2_ragged_gate as RG        # noqa: E402

N_WIN = 16
SH_TIMED, SH_PROF = (2, 2 + N_WIN), (2 + N_WIN, 2 + 2 * N_WIN)
R2_WARM = 2                                                  # eager steps 0, 1; the capture at step 2
R2G_TIMED = (R2_WARM + 1, R2_WARM + 1 + N_WIN)
R2G_PROF = (R2G_TIMED[1], R2G_TIMED[1] + N_WIN)
R2E_TIMED = (R2G_PROF[1], R2G_PROF[1] + N_WIN)
R2E_PROF = (R2E_TIMED[1], R2E_TIMED[1] + N_WIN)
R2_STEPS = R2E_PROF[1]                                       # 67
SH_STEPS = R2_STEPS + 1                                      # the R2 arm reads SH's inputs 0 .. R2_STEPS - 1
RANGE_PREFIXES = ("sh.", "r2.")
SH_RANGES = dict(update_no_compress_k_decode="sh.k", update_compress_k_decode="sh.k", update_uncompressed_cis="sh.cis",
                 update_cis="sh.cis", decode_update_kv="sh.kv")

# kernel-name classes, first match wins (case-insensitive); gpu_memcpy / gpu_memset are COPY
CLASS_RULES = (
    ("GEMM", r"gemm|gemv|xmma|cutlass|cublas"),
    ("STAGE1", r"stage1|infllm"),
    ("ATTN", r"flash_fwd|flash_attn|splitkv"),
    ("POOL", r"pool"),
    ("SELECT", r"topk|sort|radix"),
    ("H2D", r"h2d"),
    ("DIFF", r"diff"),
    ("WB", r"_wb_kernel|writeback"),
    ("FUSED", r"rmsnorm|rotary|rope|act_and_mul|silu|nosa_linear"),
    ("SMALL", r"elementwise|index|reduce|catarray|copy|fill|where|gather|scatter|arange|clamp|compare"),
)
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")
PRED = dict(Q1=(2000, 4500), Q2_twin_kernels=2000, Q2_twin_ms=3.0, Q2_sh_ms=1.0, Q3_ms=1.0, Q4_rel=0.05, Q5_ms=1.5,
            Q6=(4.0, 8.0))


def log(msg):
    print("[r2_p3 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(argv)
    if not 1 <= a.batch <= 219:
        ap.error("--batch in 1 .. 219")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested: tests/test_r2_p3.py)
def classify(name: str, cat: str = "kernel") -> str:
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "COPY"
    for cls, rx in CLASS_RULES:
        if re.search(rx, name, re.IGNORECASE):
            return cls
    return "OTHER"


def parse_trace(trace, prefixes=RANGE_PREFIXES):
    """Every GPU op of a Kineto chrome trace: [dict(name, cat, cls, dur_us, range)]. range = the innermost
    user_annotation whose name starts with one of `prefixes`, on the thread that launched the op (the launch is the
    cuda_runtime / cuda_driver event with the op's correlation id), containing the launch's start; None otherwise (a
    graph replay's kernels share its launch: no range)."""
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    X = [e for e in ev if e.get("ph") == "X"]
    by_thread = {}
    for e in X:
        if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith(tuple(prefixes)):
            s = float(e["ts"])
            by_thread.setdefault((e.get("pid"), e.get("tid")), []).append((s, s + float(e.get("dur", 0.0)), e["name"]))
    index = {}
    for key, rs in by_thread.items():
        rs.sort()
        index[key] = ([r[0] for r in rs], rs)
    launch = {}
    for e in X:
        if e.get("cat") in LAUNCH_CATS:
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launch[c] = (e.get("pid"), e.get("tid"), float(e["ts"]))
    ops = []
    for e in X:
        cat = e.get("cat")
        if cat not in GPU_CATS:
            continue
        name = str(e.get("name", ""))
        rng = None
        c = (e.get("args") or {}).get("correlation")
        if c in launch:
            pid, tid, ts = launch[c]
            if (pid, tid) in index:
                starts, rs = index[(pid, tid)]
                best = None
                j = bisect.bisect_right(starts, ts) - 1
                for k in range(j, max(-1, j - 64), -1):                 # nested ranges start earlier: look back
                    s, t_end, nm = rs[k]
                    if s <= ts <= t_end and (best is None or t_end - s < best[1] - best[0]):
                        best = (s, t_end, nm)
                rng = best[2] if best else None
        ops.append(dict(name=name, cat=cat, cls=classify(name, cat), dur_us=float(e.get("dur", 0.0)), range=rng))
    return ops


def aggregate(ops, steps: int):
    """Per step: the total, per class, per range ('-' = none) GPU ops and device ms; every name (by device ms)."""
    classes, ranges, names = {}, {}, {}
    tot_us = 0.0
    for o in ops:
        tot_us += o["dur_us"]
        for d, k in ((classes, o["cls"]), (ranges, o["range"] or "-")):
            x = d.setdefault(k, [0, 0.0])
            x[0] += 1
            x[1] += o["dur_us"]
        x = names.setdefault(o["name"], [0, 0.0, o["cls"]])
        x[0] += 1
        x[1] += o["dur_us"]

    def per(d):
        return {k: dict(kernels=round(v[0] / steps, 3), ms=round(v[1] / 1e3 / steps, 4)) for k, v in sorted(d.items())}
    top = sorted(names.items(), key=lambda kv: -kv[1][1])
    return dict(steps=steps, total=dict(kernels=round(len(ops) / steps, 3), ms=round(tot_us / 1e3 / steps, 4)),
                classes=per(classes), ranges=per(ranges),
                names=[dict(name=n[:200], cls=v[2], kernels=round(v[0] / steps, 3), ms=round(v[1] / 1e3 / steps, 4))
                       for n, v in top])


def q_verdicts(agg):
    """The registered Q1-Q6 against the aggregates of the arms SH, R2G, R2E: [(name, verdict, detail)]. R2's side
    of Q1 / Q4 / Q5 / Q6 is the graph replays when the profiler saw their kernels, else the eager steps (the same
    kernels)."""
    sh, r2e, r2g = agg.get("SH"), agg.get("R2E"), agg.get("R2G")
    if not sh or not r2e or sh["total"]["kernels"] <= 0 or r2e["total"]["kernels"] <= 0:
        return [(q, "UNDETERMINED", "the profiler saw no GPU work in SH or R2E") for q in
                ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")]
    r2, src = (r2g, "R2G") if r2g and r2g["total"]["kernels"] > 0 else (r2e, "R2E (graph kernels not seen)")

    def c(a, k, f="ms"):
        return a["classes"].get(k, {}).get(f, 0.0)

    def r(a, k, f="ms"):
        return a["ranges"].get(k, {}).get(f, 0.0)

    def v(ok):
        return "CONFIRMED" if ok else "REFUTED"
    out = []
    d1 = r2["total"]["kernels"] - sh["total"]["kernels"]
    lo, hi = PRED["Q1"]
    out.append(("Q1", v(lo <= d1 <= hi), "%s - SH GPU ops per step %.1f (predicted %d .. %d)" % (src, d1, lo, hi)))
    tk, tm, shm = r(r2e, "r2.twin", "kernels"), r(r2e, "r2.twin"), r(sh, "sh.k") + r(sh, "sh.cis")
    out.append(("Q2", v(tk >= PRED["Q2_twin_kernels"] and tm >= PRED["Q2_twin_ms"] and shm <= PRED["Q2_sh_ms"]),
                "r2.twin %.1f ops / %.3f ms per step (>= %d / >= %.1f); sh.k + sh.cis %.3f ms (<= %.1f)" % (
                    tk, tm, PRED["Q2_twin_kernels"], PRED["Q2_twin_ms"], shm, PRED["Q2_sh_ms"])))
    d3 = r(r2e, "r2.body") - r(sh, "sh.kv")
    out.append(("Q3", v(d3 >= PRED["Q3_ms"]), "r2.body - sh.kv %.3f ms per step (>= %.1f)" % (d3, PRED["Q3_ms"])))
    parts, ok4 = [], True
    for k in ("GEMM", "ATTN"):
        s, x = c(sh, k), c(r2, k)
        ok4 = ok4 and s > 0 and abs(x - s) <= PRED["Q4_rel"] * s
        parts.append("%s %.3f / %.3f" % (k, x, s))
    out.append(("Q4", v(ok4), "%s / SH ms per step: %s (within %.0f %%)" % (src, ", ".join(parts), 100 * PRED["Q4_rel"])))
    sel = ("STAGE1", "POOL", "SELECT")
    d5 = sum(c(r2, k) for k in sel) - sum(c(sh, k) for k in sel)
    out.append(("Q5", v(abs(d5) <= PRED["Q5_ms"]), "%s - SH STAGE1 + POOL + SELECT %.3f ms per step (|.| <= %.1f)" % (
        src, d5, PRED["Q5_ms"])))
    d6 = r2["total"]["ms"] - sh["total"]["ms"]
    lo, hi = PRED["Q6"]
    out.append(("Q6", v(lo <= d6 <= hi), "%s - SH device ms per step %.3f (predicted %.1f .. %.1f; the period "
                                         "difference of rb1 was 6.03)" % (src, d6, lo, hi)))
    return out


# ------------------------------------------------------------------ GPU helpers
def ranged(name, fn):
    import torch

    def w(*a, **k):
        with torch.profiler.record_function(name):
            return fn(*a, **k)
    return w


class Window:
    """CUDA-event timing of the steps in [t0, t1) and a torch.profiler session over [p0, p1), driven by on_step(t)
    BEFORE step t and done(t) AFTER it."""

    def __init__(self, timed, prof, out_dir, tag, on_start=None, on_stop=None):
        self.timed, self.prof, self.out_dir, self.tag = timed, prof, out_dir, tag
        self.on_start, self.on_stop = on_start, on_stop
        self.ev, self.p, self.trace_path = {}, None, None

    def before(self, t, clock):
        import torch
        if self.timed[0] <= t < self.timed[1]:
            self.ev[t] = clock.event()
            clock.record(self.ev[t])
        if t == self.prof[0]:
            if self.on_start:
                self.on_start()
            clock.sync()
            self.p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA])
            self.p.start()

    def after(self, t, clock):
        if t == self.timed[1] - 1:
            self.ev[self.timed[1]] = clock.event()
            clock.record(self.ev[self.timed[1]])
        if t == self.prof[1] - 1 and self.p is not None:
            clock.sync()
            self.p.stop()
            if self.on_stop:
                self.on_stop()
            self.trace_path = os.path.join(self.out_dir, "trace_%s.json" % self.tag)
            self.p.export_chrome_trace(self.trace_path)
            self.p = None

    def periods(self, clock):
        t0, t1 = self.timed
        return [clock.elapsed(self.ev[t], self.ev[t + 1]) for t in range(t0, t1) if t in self.ev and t + 1 in self.ev]


def load_and_pack(path):
    """The trace's GPU ops (parse_trace), then the trace gzip-compressed beside it (the .json removed)."""
    with open(path) as f:
        trace = json.load(f)
    ops = parse_trace(trace)
    del trace
    with open(path, "rb") as fi, gzip.open(path + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(path)
    return ops


def run_sh(model, ids, out_dir, clock):
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = SH_STEPS + 64
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
    inputs = torch.empty((Bn, SH_STEPS + 1), dtype=torch.int64, device="cuda")
    inputs[:, 0:1].copy_(tok)
    saved = {}

    def wrap():
        for meth, rng in SH_RANGES.items():
            saved[meth] = getattr(cache, meth)
            setattr(cache, meth, ranged(rng, saved[meth]))

    def unwrap():
        for meth in SH_RANGES:
            delattr(cache, meth)
    win = Window(SH_TIMED, SH_PROF, out_dir, "SH", on_start=wrap, on_stop=unwrap)
    model.has_buffers = False
    warm, rewarms = None, []
    for t in range(SH_STEPS):
        l0 = cache.layers[0]
        need, why = BC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen, warm)
        if need:
            warm = BC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
            rewarms.append((t, why))
            model.has_buffers = False
        win.before(t, clock)
        lg = model.decode_inference(tok, cu, pos, cache)
        tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        inputs[:, t + 1:t + 2].copy_(tok)
        pos = pos + 1
        win.after(t, clock)
    clock.sync()
    periods = win.periods(clock)
    del cache
    return dict(inputs=inputs, periods=periods, rewarms=rewarms, prefill_s=round(prefill_s, 1), trace=win.trace_path)


def run_r2(model, ids, inputs, out_dir, clock):
    import torch
    from nosi import r2_capture as RC
    from nosi.r2_decode_path import build_ragged
    Bn = ids.shape[0]
    prompts = [ids[b].cpu() for b in range(Bn)]
    clock.sync()
    t0 = time.time()
    cache, st, _ = build_ragged(model, prompts, R2_STEPS, f4_fix=False)
    clock.sync()
    prefill_s = time.time() - t0
    kernels, wb = RC.default_kernels()
    cb = RC.CaptureBuffers(model, st, False, kernels, wb)
    st.active_host.fill_(True)
    st.active.fill_(True)
    cb.pos_ids.copy_(st.pos_host.view(-1, 1))
    saved = {}

    def wrap():
        saved["stage1"], saved["pool"] = RC.stage1_seqused, RC.nosa_pooling_per_request
        RC.stage1_seqused = ranged("r2.stage1", saved["stage1"])
        RC.nosa_pooling_per_request = ranged("r2.pool", saved["pool"])
        for t in st.twins:
            t.step_fixed = ranged("r2.twin", t.step_fixed)
        cb.body.decode_fixed = ranged("r2.body", cb.body.decode_fixed)

    def unwrap():
        RC.stage1_seqused, RC.nosa_pooling_per_request = saved["stage1"], saved["pool"]
        for t in st.twins:
            del t.step_fixed
        del cb.body.decode_fixed
    wg = Window(R2G_TIMED, R2G_PROF, out_dir, "R2G")
    we = Window(R2E_TIMED, R2E_PROF, out_dir, "R2E", on_start=wrap, on_stop=unwrap)
    graph, ptr0, capture_ms = None, None, None
    for t in range(R2_STEPS):
        wg.before(t, clock)
        we.before(t, clock)
        cb.ids.copy_(inputs[:, t:t + 1])
        if R2_WARM <= t < R2E_TIMED[0]:
            if graph is None:
                clock.sync()
                c0 = time.perf_counter()
                ptr0 = RC.state_pointers(st, cb, cache)
                graph = RC.capture_step(model, st, cb, cache)
                clock.sync()
                capture_ms = (time.perf_counter() - c0) * 1e3
            graph.replay()
        else:
            RC.decode_step_fixed(model, st, cb, cache)
        cb.pos_ids.add_(1)
        st.pos_host += 1
        st.steps += 1
        wg.after(t, clock)
        we.after(t, clock)
    clock.sync()
    body_err, twin_err = RC.errors(st, cb)
    moved = sorted(k for k, v in RC.state_pointers(st, cb, cache).items() if ptr0.get(k) != v) if ptr0 else None
    res = dict(periods_graph=wg.periods(clock), periods_eager=we.periods(clock), trace_graph=wg.trace_path,
               trace_eager=we.trace_path, body_err=body_err, twin_err=twin_err, pointers_moved=moved,
               capture_ms=capture_ms, prefill_s=round(prefill_s, 1), widths=st.meta.get("widths"))
    del graph, cb, st, cache
    return res


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
        docs, sha = RG.load_docs(a.workload, BC.N_DOCS, BC.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    rows = BC.doc_rows(a.batch)
    ids = docs[rows].contiguous().to("cuda")
    clock = BC.CudaClock()
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    gates, arms, agg = [], {}, {}
    with torch.inference_mode():
        BC.free()
        log("arm SH (B %d, %d steps; timed %s, profiled %s)" % (a.batch, SH_STEPS, SH_TIMED, SH_PROF))
        inputs = None
        try:
            sh = run_sh(model, ids, a.out, clock)
            inputs = sh.pop("inputs")
            arms["SH"] = sh
            bad = [x for x in sh["rewarms"] if SH_TIMED[0] <= x[0] < SH_PROF[1]]
            gates.append(("SH-RUN", True, True, "%d steps, re-warms %s" % (SH_STEPS, sh["rewarms"])))
            gates.append(("SH-STEADY", True, not bad, "re-warms inside the windows: %s" % (bad,)))
        except Exception as e:
            traceback.print_exc()
            gates.append(("SH-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        BC.free(host=True)
        if inputs is None:
            gates.append(("R2-RUN", True, False, "NOT_RUN: SH failed"))
        else:
            log("arm R2 (capture at %d; graph timed %s profiled %s; eager timed %s profiled %s)" % (
                R2_WARM, R2G_TIMED, R2G_PROF, R2E_TIMED, R2E_PROF))
            try:
                r2 = run_r2(model, ids, inputs, a.out, clock)
                arms["R2"] = r2
                errs = r2["body_err"] or any(r2["twin_err"])
                gates.append(("R2-RUN", True, not errs and r2["pointers_moved"] == [],
                              "%d steps; refusal bits body %s twins %s; state pointers moved %s" % (
                                  R2_STEPS, r2["body_err"], sorted(set(r2["twin_err"])), r2["pointers_moved"])))
            except Exception as e:
                traceback.print_exc()
                gates.append(("R2-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        BC.free(host=True)
    for tag, path in (("SH", arms.get("SH", {}).get("trace")), ("R2G", arms.get("R2", {}).get("trace_graph")),
                      ("R2E", arms.get("R2", {}).get("trace_eager"))):
        if path and os.path.exists(path):
            log("parsing %s" % path)
            agg[tag] = aggregate(load_and_pack(path), N_WIN)
    seen = {k: (agg[k]["total"]["kernels"] if k in agg else 0) for k in ("SH", "R2G", "R2E")}
    gates.append(("PROF", True, seen["SH"] > 0 and seen["R2E"] > 0, "GPU ops per profiled step %s (R2G 0 = the "
                  "graph's kernels not seen: REPORT)" % seen))
    timing = {}
    for tag, per in (("SH", arms.get("SH", {}).get("periods")), ("R2G", arms.get("R2", {}).get("periods_graph")),
                     ("R2E", arms.get("R2", {}).get("periods_eager"))):
        if per:
            s = BC.period_stats(per, list(range(len(per))))
            dev = agg.get(tag, {}).get("total", {}).get("ms")
            timing[tag] = dict(s, device_ms=dev, gap_ms=(round(s["median_ms"] - dev, 3) if dev is not None else None))
    preds = q_verdicts(agg)
    meta = dict(batch=a.batch, S0=BC.S0, doc_rows=rows, workload_sha256=sha, windows=dict(
        SH_TIMED=SH_TIMED, SH_PROF=SH_PROF, R2G_TIMED=R2G_TIMED, R2G_PROF=R2G_PROF, R2E_TIMED=R2E_TIMED,
        R2E_PROF=R2E_PROF), class_rules=CLASS_RULES, predictions=PRED, attn_splits=NL._ATTN_SPLITS,
        nosi_commit=os.environ.get("P3_COMMIT", "unknown"), torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    out = dict(meta=meta, arms=arms, timing=timing, profile=agg,
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:300]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    for tag in ("SH", "R2G", "R2E"):
        if tag in agg:
            cls = ", ".join("%s %.1f/%.2f" % (k, v["kernels"], v["ms"]) for k, v in agg[tag]["classes"].items())
            rng = ", ".join("%s %.1f/%.2f" % (k, v["kernels"], v["ms"]) for k, v in agg[tag]["ranges"].items())
            print("REPORT %s per step: %s ops / %s ms; classes (ops/ms) %s; ranges %s; timing %s" % (
                tag, agg[tag]["total"]["kernels"], agg[tag]["total"]["ms"], cls, rng, timing.get(tag)))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_p3: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
