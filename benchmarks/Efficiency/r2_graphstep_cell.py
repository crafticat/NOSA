"""Task 1.6 rung 1, GPU cell gs1 (NOTES_graphstep.md section 5): the cached-NOSA step EAGER (arm "sh": the shipped
decode exactly as r2_base_cell.run_shipped runs it, re-warmed where the shipped buffers need it) against the SAME step
under --graph-step (arm "gs": nosi/nosi/graph_step.py, one CUDA graph per length bucket, every capture before the timed
loop), in ONE process, one model. --order sets which arm runs first; the second arm is TEACHER-FORCED on the first
arm's tokens, so step t of both arms reads the same input. Workload: r2_base_cell's (PG19, 94 documents of 16128
tokens, request b = row b % 94).

PER ARM (CUDA events; a period = start of step t to start of step t + 1, as r2_base_cell):
  wall per step        the steady median: sh = every step that is not a re-warm, gs = the replays after the first;
                       both without the profiled window [prof_start - 1, prof_start + 3)
  effective tok/s      B x steps / (sum of the periods + the gs capture time)
  device busy / step   torch.profiler (CPU + CUDA) over the 3 steps [prof_start, prof_start + 3): the summed GPU op
                       time (kernels, memcpy, memset; graph kernels included) / 3, its classes (r2_p3_profile.classify)
  host gap / step      the steady wall median - the device busy per step
  launches / step      the profiled cuda_runtime / cuda_driver launch calls (LaunchKernel, cuLaunch*, GraphLaunch,
                       Memcpy*, Memset*) / 3
  capture (gs)         the ms of every bucket's capture and the total; sh: the re-warm periods and, from a probe that
                       times gc.collect / torch.cuda.empty_cache / torch.cuda.synchronize inside each re-warm step,
                       where a re-warm's time goes (the torch.cuda.graph context calls all three on entry; the source
                       of graph.__enter__ in this image is recorded)
BITWISE COMPARISON over ALL steps: per (step, request) the int64 sums of the fp32 logits' bit patterns in 16 chunks of
the vocabulary, and the argmax; the full logits rows of the first 4 requests at the block, compression and width
boundary steps (torch.equal); the prefill digests; the final cache state (bit sums of every state tensor of every
layer, after graph_step.export) and the host counters.
MUST gates: SH-RUN, GS-RUN, PREFILL-EQ (both prefills equal: else GS-EQ is meaningless), GS-EQ, GS-STATE, GS-PTR (no
state pointer moves across the replays), GS-CAP (every planned graph captured before the timed loop, every later step
a replay). REPORT: SH-P2 (the shipped re-warms), PROF (the profiler saw GPU ops in both arms), and the registered
predictions PRED_GS: TBD by the lead before any run (UNREGISTERED while a value is None).
EXIT: the number of failed MUST gates (<= 19); 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import inspect
import json
import os
import re
import shutil
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_base_cell as BC          # noqa: E402
import r2_p3_profile as P3         # noqa: E402
import r2_ragged_gate as RG        # noqa: E402

S0 = BC.S0
N_DOCS = BC.N_DOCS
DIGEST_CHUNKS = 16
KEEP_ROWS = 4
PROF_STEPS = 3
BOUNDARY = (0, 1, 2, 3, 15, 16, 17, 62, 63, 64, 65, 127, 128, 255, 256, 287, 288, 289, 319, 320, 511)
LAUNCH_RX = r"LaunchKernel|cuLaunch|GraphLaunch|Memcpy|Memset"
ENGINE_STATE = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_block_map", "_load_mask", "_cache_lens")
POOL_STATE = ("_pool_map", "_pool_age", "_pool_target", "_pool_action")
LAYER_STATE = ("no_compress_k_cache", "total_cis", "tail_cis", "compressed_cis", "compress_k_cache_varlen",
               "cached_compressed_cu_seqlens")
HOST_COUNTERS = ("seq_length", "no_compress_k_len", "cached_compressed_max_seqlen", "cis_len", "tail_cis_len")
ENGINE_COUNTERS = ("seq_length", "_tail_block_len_on_gpu", "_pool_stamp_base")
# ---- REGISTERED PREDICTIONS: TBD. The lead fills every value and the stamp (`date -u`, L33) in a commit BEFORE any
# run; a None value prints UNREGISTERED. Bands are (low, high); NOTES_graphstep.md section 6 has the builder's
# suggestions (not registered).
PRED_GS = dict(
    stamp=None,                 # UTC stamp of the registration
    steady_ratio=None,          # gs / sh steady wall per step, (low, high)
    effective_ratio=None,       # gs / sh effective tok/s, (low, high)
    launches_gs=None,           # gs launch calls per step, (low, high)
    capture_s=None,             # gs total capture seconds, (low, high)
    busy_ratio=None,            # gs / sh device busy per step, (low, high)
    rewarm_hostcall_share=None,  # sh: (gc + empty_cache + synchronize ms) / re-warm period, mean over re-warms 2..9
)


def log(msg):
    print("[r2_gs %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--pool", type=int, default=65)
    ap.add_argument("--order", default="sh,gs", choices=("sh,gs", "gs,sh"))
    ap.add_argument("--prof-start", type=int, default=100)
    a = ap.parse_args(argv)
    if a.batch < 1 or not 1 <= a.warmup < a.steps or a.steps > 1000:
        ap.error("--batch >= 1, --warmup in 1 .. steps - 1, --steps <= 1000")
    if not 0 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 0 .. %d" % BC.POOL_MAX)
    if not (a.warmup + 2 <= a.prof_start and a.prof_start + PROF_STEPS + 1 <= a.steps):
        ap.error("--prof-start must leave the eager warm-up, the first replay and one step after the window")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested: tests_cpu/test_graph_step.py)
def boundary_steps(steps: int):
    """The steps whose full logits rows are kept: BOUNDARY below `steps`, and the last step."""
    return sorted({t for t in BOUNDARY if t < steps} | {steps - 1})


def steady_steps(mode: str, steps: int, warmup: int, prof_start: int, rewarms):
    """The steps of the steady statistic: sh drops its re-warms, gs its eager steps and the first replay; both drop the
    profiled window and the step before it."""
    skip = set(range(prof_start - 1, prof_start + PROF_STEPS))
    if mode == "eager":
        skip |= {t for t, _ in rewarms}
    else:
        skip |= set(range(warmup + 1))
    return [t for t in range(steps) if t not in skip]


def special_steps(S: int, steps: int):
    """tf = the steps where the tail block fills ((S + t + 1) % 64 == 0); kc = the key-compression steps."""
    tf = [t for t in range(steps) if (S + t + 1) % 64 == 0]
    kc = [t for t in range(1, steps) if (S + t) % 16 == 0]
    return dict(tf=tf, kc=kc)


def count_launches(events):
    """{name: count} of the host launch calls of a chrome trace (cuda_runtime / cuda_driver X events)."""
    out = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in P3.LAUNCH_CATS and re.search(LAUNCH_RX, str(e.get("name", ""))):
            n = str(e.get("name"))
            out[n] = out.get(n, 0) + 1
    return out


def device_span_us(ops):
    if not ops:
        return 0.0
    return max(o["ts"] + o["dur_us"] for o in ops) - min(o["ts"] for o in ops)


def rewarm_breakdown(periods, rewarms, probe_rows):
    """Per re-warm step: its period and the host ms inside gc.collect / empty_cache / synchronize (the probe)."""
    out = []
    for t, why in rewarms:
        d = probe_rows.get(t, {})
        calls = sum(d.get(k, 0.0) for k in ("gc_ms", "empty_cache_ms", "sync_ms"))
        out.append(dict(t=t, why=why, period_ms=round(periods[t], 1), gc_ms=round(d.get("gc_ms", 0.0), 1),
                        empty_cache_ms=round(d.get("empty_cache_ms", 0.0), 1), sync_ms=round(d.get("sync_ms", 0.0), 1),
                        n_gc=d.get("n_gc", 0), n_empty_cache=d.get("n_empty_cache", 0), n_sync=d.get("n_sync", 0),
                        other_ms=round(periods[t] - calls, 1),
                        hostcall_share=(round(calls / periods[t], 4) if periods[t] > 0 else None)))
    return out


def compare_arms(x, y):
    """The bitwise comparison of two arms' records (CPU tensors / dicts): counts and first positions."""
    import torch
    out = dict(prefill_equal=bool(torch.equal(x["prefill_digest"], y["prefill_digest"])))
    neq = (x["digests"] != y["digests"]).any(dim=-1)
    bad = neq.nonzero().tolist()
    out.update(digest_mismatches=len(bad), first_digest_mismatch=(bad[0] if bad else None),
               steps_with_mismatch=sorted({t for t, _ in bad})[:20])
    tb = (x["tokens"][:, 1:] != y["tokens"][:, 1:]).nonzero().tolist()
    out.update(token_mismatches=len(tb), first_token_mismatch=(tuple(tb[0]) if tb else None))
    out["rows_unequal_steps"] = [t for i, t in enumerate(x["keep_steps"]) if not torch.equal(x["rows"][i], y["rows"][i])]
    keys = sorted(set(x["state"]) | set(y["state"]))
    out["state_mismatches"] = [k for k in keys if x["state"].get(k) != y["state"].get(k)]
    out["steps_compared"] = int(x["digests"].shape[0])
    out["requests"] = int(x["digests"].shape[1])
    return out


def gs_verdicts(res, pred):
    """The registered predictions against this cell (REPORT only). A None value: UNREGISTERED."""
    out = []

    def band(name, val, rng, what):
        if val is None:
            out.append((name, "NOT_RUN", "%s: no value" % what))
        elif rng is None:
            out.append((name, "UNREGISTERED", "%s %.4f (TBD by the lead)" % (what, val)))
        else:
            ok = rng[0] <= val <= rng[1]
            out.append((name, "CONFIRMED" if ok else "REFUTED", "%s %.4f (band %s)" % (what, val, tuple(rng))))
    band("GS-P1", res.get("steady_ratio"), pred.get("steady_ratio"), "gs / sh steady wall per step")
    band("GS-P2", res.get("effective_ratio"), pred.get("effective_ratio"), "gs / sh effective tok/s")
    band("GS-P3", res.get("launches_gs"), pred.get("launches_gs"), "gs launch calls per step")
    band("GS-P4", res.get("capture_s"), pred.get("capture_s"), "gs total capture s")
    band("GS-P5", res.get("busy_ratio"), pred.get("busy_ratio"), "gs / sh device busy per step")
    band("GS-P6", res.get("rewarm_hostcall_share"), pred.get("rewarm_hostcall_share"),
         "sh re-warm share in gc.collect + empty_cache + synchronize")
    return out


class Probe:
    """Times gc.collect, torch.cuda.empty_cache and torch.cuda.synchronize while installed, per decode step (self.t;
    None = not counted). Module attributes, so torch.cuda.graph's own calls (graphs.py, graph.__enter__) are seen."""

    def __init__(self):
        self.t, self.rows, self.orig = None, {}, {}

    def _wrap(self, name, fn):
        def w(*a, **k):
            if self.t is None:
                return fn(*a, **k)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                d = self.rows.setdefault(self.t, {})
                d[name + "_ms"] = d.get(name + "_ms", 0.0) + (time.perf_counter() - t0) * 1e3
                d["n_" + name] = d.get("n_" + name, 0) + 1
        return w

    def install(self):
        import torch
        for name, mod, attr in (("gc", gc, "collect"), ("empty_cache", torch.cuda, "empty_cache"),
                                ("sync", torch.cuda, "synchronize")):
            orig = getattr(mod, attr)
            self.orig[name] = (mod, attr, orig)
            setattr(mod, attr, self._wrap(name, orig))

    def remove(self):
        for mod, attr, orig in self.orig.values():
            setattr(mod, attr, orig)
        self.orig = {}
        self.t = None


# ------------------------------------------------------------------ GPU parts
def digest_into(dst, lg, chunks: int):
    """dst (B, chunks) int64 <- the int64 sums of lg's fp32 bit patterns in `chunks` equal parts of the vocabulary."""
    import torch
    x = lg.reshape(lg.shape[0], -1).view(torch.int32)
    dst.copy_(x.view(x.shape[0], chunks, x.shape[1] // chunks).sum(dim=-1, dtype=torch.int64))


def bits_sum(x):
    import torch
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.view(torch.int16)
    elif x.dtype == torch.float32:
        x = x.view(torch.int32)
    return x.sum(dtype=torch.int64)


def state_digest(cache):
    """{name: int} bit sums of every state tensor of every layer and the host counters (one sync)."""
    import torch
    names, vals, host = [], [], {}
    for l, L in enumerate(cache.layers):
        e = L.cache_engine
        for n in ENGINE_STATE + (POOL_STATE if e.pool_blocks > 0 else ()):
            names.append("eng%d.%s" % (l, n))
            vals.append(bits_sum(getattr(e, n)))
        for n in LAYER_STATE:
            names.append("layer%d.%s" % (l, n))
            vals.append(bits_sum(getattr(L, n)))
        for n in HOST_COUNTERS:
            host["layer%d.%s" % (l, n)] = int(getattr(L, n))
        for n in ENGINE_COUNTERS:
            host["eng%d.%s" % (l, n)] = int(getattr(e, n))
    out = dict(zip(names, torch.stack(vals).cpu().tolist()))
    out.update(host)
    return out


def profile_summary(path, nsteps: int):
    with open(path) as f:
        trace = json.load(f)
    ops = P3.parse_trace(trace)
    agg = P3.aggregate(ops, nsteps)
    launches = count_launches(trace.get("traceEvents", []) if isinstance(trace, dict) else trace)
    span = device_span_us(ops)
    del trace
    with open(path, "rb") as fi, gzip.open(path + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(path)
    return dict(steps=nsteps, gpu_ops_per_step=agg["total"]["kernels"], device_ms_per_step=agg["total"]["ms"],
                device_span_ms_per_step=round(span / 1e3 / nsteps, 4),
                launches_per_step=round(sum(launches.values()) / nsteps, 2),
                launches_by_name={k: round(v / nsteps, 2) for k, v in sorted(launches.items())},
                classes=agg["classes"], top_names=agg["names"][:25], trace=os.path.basename(path) + ".gz")


def decode_arm(model, ids, a, clock, mode: str, inputs=None, probe=None, out_dir=None):
    """One arm (module docstring). mode 'eager' = run_shipped's loop; 'graph' = run_graph_step's loop. inputs (B,
    steps + 1) = the first arm's tokens (teacher forcing) or None (free-running)."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    steps, Bn = a.steps, ids.shape[0]
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    clock.sync()
    t0 = time.time()
    logits, pos = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    V = logits.shape[-1]
    C = DIGEST_CHUNKS if V % DIGEST_CHUNKS == 0 else 1
    pre = torch.zeros((Bn, C), dtype=torch.int64, device="cuda")
    digest_into(pre, logits, C)
    tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    pos = pos[:, -1:] + 1
    cu = torch.arange(0, Bn + 1, dtype=torch.int, device="cuda")
    own = torch.empty((Bn, steps + 1), dtype=torch.int64, device="cuda")
    own[:, 0:1].copy_(tok)
    if inputs is not None:
        tok = inputs[:, 0:1]
    dig = torch.zeros((steps, Bn, C), dtype=torch.int64, device="cuda")
    keep = boundary_steps(steps)
    kidx = {t: i for i, t in enumerate(keep)}
    R = min(KEEP_ROWS, Bn)
    rows = torch.zeros((len(keep), R, V), dtype=torch.float32, device="cuda")
    p0, p1 = a.prof_start, a.prof_start + PROF_STEPS
    gs, capture_ms, ptr0, prof = None, None, None, None
    if mode == "graph":
        from nosi import graph_step as GSM
        gs = GSM.GraphStep(model, cache, cu, steps, warmup=a.warmup, pos_dtype=pos.dtype)
    else:
        model.has_buffers = False
    warm, rewarms = None, []
    ev = [clock.event() for _ in range(steps + 1)]
    ev_w = clock.event()
    for t in range(steps):
        if gs is not None and t == gs.warmup:
            clock.record(ev_w)
            clock.sync()
            capture_ms = gs.capture_all()
            ptr0 = gs.state_pointers()
        if t == p0:
            clock.sync()
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.start()
        if gs is None:
            l0 = cache.layers[0]
            need, why = BC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen,
                                       warm)
            if need:
                warm = BC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
                rewarms.append((t, why))
                model.has_buffers = False
        clock.record(ev[t])
        if probe is not None:
            probe.t = t
        lg = model.decode_inference(tok, cu, pos, cache) if gs is None else gs.step(t, tok, pos)
        if probe is not None:
            probe.t = None
        nxt = lg[:, -1, :].argmax(dim=-1, keepdim=True)
        own[:, t + 1:t + 2].copy_(nxt)
        digest_into(dig[t], lg, C)
        if t in kidx:
            rows[kidx[t]].copy_(lg[:R, -1, :])
        tok = inputs[:, t + 1:t + 2] if inputs is not None else nxt
        pos = pos + 1
        if prof is not None and t == p1 - 1:
            clock.sync()
            prof.stop()
    clock.record(ev[steps])
    clock.sync()
    periods = [clock.elapsed(ev[t], ev[t + 1]) for t in range(steps)]
    if capture_ms is not None:
        periods[gs.warmup - 1] = clock.elapsed(ev[gs.warmup - 1], ev_w)
    prof_res = None
    if prof is not None and out_dir:
        path = os.path.join(out_dir, "trace_%s.json" % mode)
        prof.export_chrome_trace(path)
        del prof
        prof_res = profile_summary(path, PROF_STEPS)
    gsr = None
    if gs is not None:
        moved = None
        if ptr0 is not None:
            after = gs.state_pointers()
            moved = sorted(k for k in set(ptr0) | set(after) if ptr0.get(k) != after.get(k))
        gsr = gs.report()
        gsr.update(capture_ms_total=(round(capture_ms, 1) if capture_ms is not None else None),
                   pointers_changed=moved)
        gs.export()
        del gs
    state = state_digest(cache)
    fr, tot = torch.cuda.mem_get_info()
    res = dict(mode=mode, prefill_s=round(prefill_s, 1), periods=periods, rewarms=rewarms, prefill_digest=pre.cpu(),
               tokens=own.cpu(), digests=dig.cpu(), rows=rows.cpu(), keep_steps=keep, state=state, graph_step=gsr,
               capture_ms=capture_ms, profile=prof_res, device_used_end_GB=round((tot - fr) / 1e9, 3), inputs=own)
    del cache
    return res


def arm_summary(name, res, a, probe_rows=None):
    """The JSON-safe summary of one arm."""
    mode = res["mode"]
    keep = steady_steps(mode, a.steps, a.warmup, a.prof_start, res["rewarms"])
    st = BC.period_stats(res["periods"], keep)
    extra = [res["capture_ms"]] if res["capture_ms"] is not None else []
    rt = BC.rates(a.batch, a.steps, res["periods"] + extra, st)
    sp = special_steps(S0, a.steps)
    prof = res["profile"]
    rw = {r for r, _ in res["rewarms"]}
    kc = [res["periods"][t] for t in sp["kc"] if t not in rw and t > a.warmup]
    out = dict(mode=mode, prefill_s=res["prefill_s"], steady=st, rates=rt, steady_steps=len(keep),
               periods=[round(p, 3) for p in res["periods"]],
               tf_periods=[(t, round(res["periods"][t], 2)) for t in sp["tf"]],
               kc_median_ms=(round(statistics.median(kc), 3) if kc else None),
               rewarms=res["rewarms"], rewarm_ms=[round(res["periods"][t], 1) for t, _ in res["rewarms"]],
               profile=prof, host_gap_ms=(round(st["median_ms"] - prof["device_ms_per_step"], 3)
                                          if prof and st["median_ms"] else None),
               gpu_peak_alloc_GB=res.get("gpu_peak_alloc_GB"), gpu_peak_reserved_GB=res.get("gpu_peak_reserved_GB"),
               device_used_end_GB=res["device_used_end_GB"], graph_step=res["graph_step"])
    if probe_rows is not None:
        out["rewarm_breakdown"] = rewarm_breakdown(res["periods"], res["rewarms"], probe_rows)
    return out


def graph_enter_source():
    try:
        import torch
        return inspect.getsource(torch.cuda.graphs.graph.__enter__)[:2000]
    except Exception as e:                               # the record never stops the cell
        return "unavailable: %s: %s" % (type(e).__name__, e)


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
        docs, sha = RG.load_docs(a.workload, N_DOCS, S0)
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
    expected = BC.expected_rewarms(S0, a.steps)
    gates, arms, probe, first = [], {}, Probe(), None
    with torch.inference_mode():
        for name in a.order.split(","):
            mode = "eager" if name == "sh" else "graph"
            BC.free()
            log("arm %s (%s, B %d, %d steps%s)" % (name, mode, a.batch, a.steps,
                                                   ", teacher-forced" if first is not None else ""))
            try:
                if mode == "eager":
                    probe.install()
                res = decode_arm(model, ids, a, clock, mode, inputs=(first["inputs"] if first else None),
                                 probe=(probe if mode == "eager" else None), out_dir=a.out)
                res["gpu_peak_alloc_GB"], res["gpu_peak_reserved_GB"] = BC.gpu_peak()
                arms[name] = res
                gates.append(("%s-RUN" % name.upper(), True, True, "%d steps (%s)" % (a.steps, mode)))
                if first is None:
                    first = res
            except Exception as e:
                traceback.print_exc()
                gates.append(("%s-RUN" % name.upper(), True, False, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                       str(e)[:300])))
            finally:
                probe.remove()
            BC.free(host=True)
            if name not in arms:
                break                                    # the second arm needs the first arm's tokens
    summ = {n: arm_summary(n, r, a, probe.rows if n == "sh" else None) for n, r in arms.items()}
    cmp_ = None
    if "sh" in arms and "gs" in arms:
        sh, gs = arms["sh"], arms["gs"]
        cmp_ = compare_arms(sh, gs)
        gates.append(("PREFILL-EQ", True, cmp_["prefill_equal"], "the two prefills' logits digests are %s" % (
            "equal" if cmp_["prefill_equal"] else "DIFFERENT (GS-EQ is then not about the graph step)")))
        ok = not cmp_["digest_mismatches"] and not cmp_["token_mismatches"] and not cmp_["rows_unequal_steps"]
        gates.append(("GS-EQ", True, ok, "%d steps x %d requests: %d digest mismatches (first %s), %d argmax "
                      "mismatches (first (request, step) %s), rows unequal at steps %s" % (
                          cmp_["steps_compared"], cmp_["requests"], cmp_["digest_mismatches"],
                          cmp_["first_digest_mismatch"], cmp_["token_mismatches"], cmp_["first_token_mismatch"],
                          cmp_["rows_unequal_steps"])))
        gates.append(("GS-STATE", True, not cmp_["state_mismatches"], "final state: %d mismatching entries %s" % (
            len(cmp_["state_mismatches"]), cmp_["state_mismatches"][:10])))
        for nm, r in (("sh", sh), ("gs", gs)):
            torch.save(dict(digests=r["digests"], tokens=r["tokens"], prefill=r["prefill_digest"]),
                       os.path.join(a.out, "digests_%s.pt" % nm))
    if "gs" in arms:
        g = arms["gs"]["graph_step"]
        gates.append(("GS-PTR", True, g["pointers_changed"] == [], "%s of the state pointers moved: %s" % (
            len(g["pointers_changed"] or []) if g["pointers_changed"] is not None else "?",
            (g["pointers_changed"] or [])[:10])))
        gates.append(("GS-CAP", True, g["n_graphs"] == g["planned_graphs"] and g["replays"] == a.steps - a.warmup,
                      "%d of %d planned graphs captured before the timed loop (%s ms); %d replays, %d eager steps" % (
                          g["n_graphs"], g["planned_graphs"], g["capture_ms_total"], g["replays"], g["eager_steps"])))
    if "sh" in arms:
        got = [tuple(x) for x in arms["sh"]["rewarms"]]
        gates.append(("SH-P2", False, got == [tuple(x) for x in expected], "shipped re-warms %s (expected %s)" % (
            got, expected)))
    seen = {n: (s["profile"] or {}).get("gpu_ops_per_step") for n, s in summ.items()}
    gates.append(("PROF", False, all(v for v in seen.values()) and len(seen) == 2, "GPU ops per profiled step %s"
                  % seen))
    res = {}
    if "sh" in summ and "gs" in summ:
        s, g = summ["sh"], summ["gs"]
        if s["steady"]["median_ms"] and g["steady"]["median_ms"]:
            res["steady_ratio"] = g["steady"]["median_ms"] / s["steady"]["median_ms"]
        if s["rates"]["effective_tok_s"] and g["rates"]["effective_tok_s"]:
            res["effective_ratio"] = g["rates"]["effective_tok_s"] / s["rates"]["effective_tok_s"]
        if s["profile"] and g["profile"] and s["profile"]["device_ms_per_step"]:
            res["busy_ratio"] = g["profile"]["device_ms_per_step"] / s["profile"]["device_ms_per_step"]
    if "gs" in summ:
        g = summ["gs"]
        res["launches_gs"] = g["profile"]["launches_per_step"] if g["profile"] else None
        cm = g["graph_step"]["capture_ms_total"] if g["graph_step"] else None
        res["capture_s"] = cm / 1e3 if cm is not None else None
    if "sh" in summ:
        br = [r["hostcall_share"] for r in summ["sh"].get("rewarm_breakdown", [])[1:] if r["hostcall_share"] is not None]
        res["rewarm_hostcall_share"] = statistics.fmean(br) if br else None
    preds = gs_verdicts(res, PRED_GS)
    meta = dict(batch=a.batch, steps=a.steps, warmup=a.warmup, order=a.order, prof_start=a.prof_start, S0=S0,
                doc_rows=rows, workload_sha256=sha, expected_rewarms=expected, attn_splits=NL._ATTN_SPLITS,
                pool_blocks=CEN.POOL_BLOCKS, capacity_C=BC.C_BASE + a.pool, digest_chunks=DIGEST_CHUNKS,
                keep_rows=KEEP_ROWS, boundary_steps=boundary_steps(a.steps), predictions=PRED_GS,
                nosi_commit=os.environ.get("BASE_COMMIT", "unknown"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), graph_enter_source=graph_enter_source())
    out = dict(meta=meta, arms=summ, compare=cmp_, ratios=res,
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else ("FAIL" if g[1] else "REPORT"), g[3][:400]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    for n, s in summ.items():
        pr = s["profile"] or {}
        print("REPORT %s: steady %s ms/step (n %d), effective %s tok/s, device busy %s ms/step, host gap %s ms, "
              "launches %s / step, GPU ops %s / step, prefill %s s, re-warms %s ms, capture %s ms" % (
                  n, s["steady"]["median_ms"], s["steady_steps"], s["rates"]["effective_tok_s"],
                  pr.get("device_ms_per_step"), s["host_gap_ms"], pr.get("launches_per_step"),
                  pr.get("gpu_ops_per_step"), s["prefill_s"], s["rewarm_ms"],
                  (s["graph_step"] or {}).get("capture_ms_total")))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_gs: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
