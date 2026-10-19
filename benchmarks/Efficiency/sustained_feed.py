"""SUSTAINED decode + feeder: paper 4.9 (can CPU packing keep up over a sustained decode run) and 4.4 (interference of
data movement with decode). The pure pieces, units, rules and labels are in sustained_core.py (its docstring).

ONE PROCESS, ONE GPU. Per ARM, a fresh TRAJECTORY: the same B PG19 documents (rows doc_offset .. + B - 1 of the 16128-token
workload, each row's sha256 checked against the manifest) are prefilled into a fresh GPU-resident cache (NOSA's own
InfLLMv2Cache of cache_engine_gpu.py, offload=False; only its unused headroom max_gen_len is set, to --max-gen), then
--steps natural greedy decode steps run (NOSA's decode_inference, warm-up / re-warm by sustained_core.rewarm_need). Arms:
  alone              the decode only (no data movement): the denominator, paired by step with every other arm
  cpu:<MiB>:<frag>   + the CPU-packing feeder (cpupack_core build_desc + Pipe: pack on --pack-threads cores, one H2D per
                     chunk of <= --cap-groups groups on a copy stream, a GPU scatter on a scatter stream)
  gpu:<MiB>:<frag>   + the matched GPU gather (the W8 persistent kernel: --w8-ctas CTAs x --w8-warps warps) of the SAME
                     plans, bytes and destinations, on a gather stream
The feeder starts at step --feed-start, releases one release per step until step steps - deadline, then drains. The
same documents and the same deterministic decode give the same trajectory in every arm (checked: the token sha256), so
decode-step times pair by step index.

LIMITS (stated in summary.json "limits"): the trajectory is at most 1024 decode steps (cache_engine_gpu.py
update_uncompressed_cis allocates total_cis (B, S + 1024, H) at prefill and writes one column per decode step); the
warm-up-sized pooling buffers and the captured pooling graph force a re-warm every 64 tokens and at every 128-column
change of the stage-1 score width (sustained_core.rewarm_need; the shipped benchmark never meets either: it decodes a
few steps); GPU memory: weights + B x (S + max_gen) x 32 KiB of KV per request-token (B 64 at --max-gen 1088: about 36
GB of KV); host memory: the pinned source = 2 x --src-layers x B x S x 512 B (B 64, 32 layers: about 34 GB).

OUTPUT (--out, refused when it exists; created first, so every later outcome is written there):
  summary.json    config, label, limits, environment and module origins (sha256), workload rows and their sha256, per
                  arm: prefill_s, decode (steady period mean / p50 / p95, the re-warm steps and why), slowdown vs alone
                  (paired, block-bootstrap CI), delivery (sustained_core.delivery: delivered / offered GB/s, on-time
                  fractions by count and bytes for --deadline and the DEADLINE_TABLE, lateness quantiles, drain), queue
                  (backlog_host and outstanding_gpu: sustained_core.queue_trend), cores (sustained_core.cores_used over
                  the feeding window), pauses, content check (K and V), token sha256, peak memory, error; verdict
  steps.jsonl     one row per (arm, decode step): sustained_core.STEP_FIELDS
  releases.jsonl  one row per (arm, release): sustained_core.RELEASE_FIELDS
  SHA256SUMS
EXIT: 0 PASS; 1 FAIL (a check failed or an arm raised; summary.json names it); 2 --out exists; 3 another GPU than
--gpu-uuid (before the model loads); 4 a refused configuration (before anything is allocated); 6 a module of the run
(nosi.*, cpupack_core, sustained_*) loaded from outside NOSI_REPO (when set).

  python sustained_feed.py --model <NOSA-8B dir> --tokens <c16k_tokens.bin> --manifest <c16k_manifest.json> --out <dir>
         [--batch 64] [--steps 960] [--arms alone,cpu:64:32,gpu:64:32] [--deadline 3] [--gate gpu|host] [--dry]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import sustained_core as SC  # noqa: E402

CIS_HEADROOM = 1024                          # cache_engine_gpu.py update_uncompressed_cis: total_cis (B, S + 1024, H)
CTX = 16128


# --------------------------------------------------------------------------------------------------------- config
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=960)
    ap.add_argument("--feed-start", type=int, default=8)
    ap.add_argument("--settle", type=int, default=16, help="feeding steps excluded from the steady statistics")
    ap.add_argument("--deadline", type=int, default=3, help="release t is needed by decode step t + deadline")
    ap.add_argument("--arms", default="alone,cpu:64:32,gpu:64:32,cpu:192:96,gpu:192:96")
    ap.add_argument("--gate", choices=SC.GATES, default="gpu")
    ap.add_argument("--pack-threads", type=int, default=8)
    ap.add_argument("--cap-groups", type=int, default=256)
    ap.add_argument("--ring", type=int, default=3)
    ap.add_argument("--slots", type=int, default=64, help="scratch slots per (request, head)")
    ap.add_argument("--src-layers", type=int, default=32)
    ap.add_argument("--max-gen", type=int, default=None, help="KV headroom tokens (default round64(steps + 64))")
    ap.add_argument("--w8-ctas", type=int, default=8)
    ap.add_argument("--w8-warps", type=int, default=4)
    ap.add_argument("--pin", choices=("on", "off"), default="on")
    ap.add_argument("--gpu-uuid", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--dry", action="store_true", help="check the configuration and the workload, write the plan, no CUDA")
    return ap


def check_config(a, rows_available: int, cpus: List[int]) -> Dict[str, Any]:
    """Every refusal before anything is allocated (exit 4). Returns the derived configuration."""
    errs = []
    try:
        arms = SC.parse_arms(a.arms)
    except ValueError as e:
        raise SystemExit4([str(e)])
    if a.steps < 2 or a.steps > CIS_HEADROOM:
        errs.append("--steps %d: a trajectory holds 1 .. %d decode steps (total_cis headroom)" % (a.steps, CIS_HEADROOM))
    if a.batch < 1 or a.doc_offset < 0 or a.doc_offset + a.batch > rows_available:
        errs.append("--batch %d at --doc-offset %d needs rows the workload does not hold (%d)" % (a.batch, a.doc_offset,
                                                                                            rows_available))
    if a.deadline < 1:
        errs.append("--deadline must be >= 1")
    w0, w1 = a.feed_start, a.steps - a.deadline
    if not (1 <= w0 < w1):
        errs.append("feeding window [%d, %d) is empty (feed-start %d, steps %d, deadline %d)" % (w0, w1, w0, a.steps,
                                                                                             a.deadline))
    if w0 + a.settle >= w1:
        errs.append("--settle %d leaves no steady feeding step in [%d, %d)" % (a.settle, w0, w1))
    max_gen = a.max_gen if a.max_gen is not None else SC.round_up(a.steps + 64, 64)
    if max_gen < a.steps:
        errs.append("--max-gen %d < --steps %d (the tail write-back would leave the KV)" % (max_gen, a.steps))
    geom = SC.Geometry(B=a.batch, S=CTX, L=32, M=a.slots)
    nreq = {}
    for arm in arms:
        if arm.method == "alone":
            continue
        try:
            nreq[arm.name] = SC.served_requests(arm.rate_mb, arm.frag, a.batch)
            SC.Schedule(geom, nreq[arm.name], arm.frag)
        except ValueError as e:
            errs.append("arm %s: %s" % (arm.name, e))
    need = 1 + a.pack_threads
    if a.pin == "on" and len(cpus) < need:
        errs.append("--pin on needs %d allowed CPUs (decode 1 + feeder %d), the process has %d" % (need, a.pack_threads,
                                                                                               len(cpus)))
    if not 1 <= a.src_layers <= 32 or a.cap_groups < 1 or a.ring < 2 or a.slots < 1:
        errs.append("--src-layers 1..32, --cap-groups >= 1, --ring >= 2, --slots >= 1")
    if errs:
        raise SystemExit4(errs)
    main_core = cpus[0] if (a.pin == "on" and cpus) else None
    feeder_cores = cpus[1:1 + a.pack_threads] if a.pin == "on" else None
    return dict(arms=arms, w0=w0, w1=w1, max_gen=max_gen, geom=geom, nreq=nreq, main_core=main_core,
                feeder_cores=feeder_cores)


class SystemExit4(Exception):
    def __init__(self, errs):
        super().__init__("; ".join(errs))
        self.errs = errs


class ModuleOrigin(Exception):
    """A module of the run loaded from outside NOSI_REPO (exit 6)."""


def load_workload(tokens: Path, manifest: Path, offset: int, B: int) -> Dict[str, Any]:
    """The B rows [offset, offset + B) of the int32-le (n_docs, 16128) token file; each row's sha256 must equal the
    manifest's (a mismatch is a refused configuration)."""
    man = json.loads(manifest.read_text())
    n, ctx = man["shape"]
    if ctx != CTX or man.get("dtype") != "int32-le":
        raise SystemExit4(["manifest shape %s dtype %s: expected (n, %d) int32-le" % (man["shape"], man.get("dtype"), CTX)])
    raw = tokens.read_bytes()
    if len(raw) != n * ctx * 4:
        raise SystemExit4(["%s holds %d bytes, the manifest says %d x %d int32" % (tokens, len(raw), n, ctx)])
    out, docs = [], []
    for i in range(offset, offset + B):
        if i >= n:
            raise SystemExit4(["row %d is past the %d rows" % (i, n)])
        row = raw[i * ctx * 4:(i + 1) * ctx * 4]
        sha = hashlib.sha256(row).hexdigest()
        if sha != man["docs"][i]["sha256"]:
            raise SystemExit4(["row %d sha256 %s != manifest %s" % (i, sha, man["docs"][i]["sha256"])])
        out.append(torch.frombuffer(bytearray(row), dtype=torch.int32).to(torch.int64))
        docs.append(dict(row=i, index=man["docs"][i]["index"], sha256=sha))
    return dict(ids=torch.stack(out), docs=docs, rows=n, file_sha256=hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------------------------------- the arm runner
class ArmContext:
    """What run_arm needs (the GPU driver builds it from NOSA; the CPU tests from a fake model):
      model        .batch_prefill(ids, cache) -> (logits, position_ids); .decode_inference(ids, cu, pos, cache) -> logits;
                   .has_buffers (False -> the next decode_inference warms up)
      make_cache   () -> a cache with .layers[0].no_compress_k_len, .layers[0].cached_compressed_max_seqlen,
                   .get_seq_length(0) and .layers[i].cache_engine.max_gen_len
      clock        sustained_core.CudaClock / HostClock
      transport    (arm) -> (work fn (sustained_core.cpu_work_fn / gpu_work_fn), prepare () -> None (drain, reset the
                   pipe and the event pool, poison the scratch), finish (schedule) -> {"k": check, "v": check})
      device       where ids, tokens live
      sync         () -> None: a full device synchronize"""

    def __init__(self, model, make_cache, clock, transport, device, ids, sync: Callable[[], None]):
        self.model, self.make_cache, self.clock, self.transport = model, make_cache, clock, transport
        self.device, self.ids, self.sync = device, ids, sync


def run_arm(ctx: ArmContext, arm: SC.ArmSpec, cfg: Dict[str, Any], a) -> Dict[str, Any]:
    """One trajectory (module docstring). Returns the arm record with its raw per-step and per-release rows."""
    N, d, w0, w1 = a.steps, a.deadline, cfg["w0"], cfg["w1"]
    model, clock = ctx.model, ctx.clock
    rec: Dict[str, Any] = dict(arm=arm.name, method=arm.method, rate_mb=arm.rate_mb, frag=arm.frag)
    cache = ctx.make_cache()
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = cfg["max_gen"]
    ctx.sync()
    t0 = time.perf_counter()
    logits, pos = model.batch_prefill(ctx.ids, cache)
    ctx.sync()
    rec["prefill_s"] = time.perf_counter() - t0
    B = ctx.ids.shape[0]
    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    pos = pos[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device=ctx.device)
    toks = torch.empty((B, N), dtype=next_ids.dtype, device=ctx.device)
    model.has_buffers = False
    feeder = None
    sched = None
    releases: Dict[int, SC.ReleaseSpec] = {}
    finish = None
    if arm.method != "alone":
        sched = SC.Schedule(cfg["geom"], cfg["nreq"][arm.name], arm.frag, seed=a.seed)
        for t in range(w0, w1):
            releases[t] = sched.release(t)
        work, prepare, finish = ctx.transport(arm)
        prepare()
        feeder = SC.Feeder(work, affinity=cfg["feeder_cores"], name="feeder-" + arm.method)
        feeder.start()
    ev_s = [clock.event() for _ in range(N + 1)]
    ev_e = [clock.event() for _ in range(N)]
    host_ns = [0] * N
    rewarm = [""] * N
    backlog, outst, pause_ns = [0] * N, [0] * N, [0] * N
    warm = None
    cpu_a = cpu_b = None
    wall_a = wall_b = 0.0
    main_tid = threading.get_native_id()
    aff0 = None
    try:
        if cfg["main_core"] is not None:
            aff0 = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {cfg["main_core"]})
        for t in range(N):
            h0 = time.perf_counter_ns()
            if feeder is not None:
                feeder.raise_if_failed()
                backlog[t] = feeder.backlog()
                outst[t] = SC.outstanding_on_gpu(feeder)
            l0 = cache.layers[0]
            need, why = SC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen, warm)
            if need:
                warm = SC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
                rewarm[t] = why
                if feeder is not None:
                    pause_ns[t] = feeder.pause()
            if t == w0:
                cpu_a, wall_a = SC.cpu_snapshot(), time.perf_counter()
            clock.record(ev_s[t])
            if feeder is not None and w0 <= t < w1:
                feeder.release(SC.Release(spec=releases[t], deadline_t=t + d,
                                          gate=(ev_s[t] if a.gate == "gpu" else None)))
            if need:
                model.has_buffers = False
            logits = model.decode_inference(next_ids, cu, pos, cache)
            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            toks[:, t:t + 1].copy_(next_ids)
            pos = pos + 1
            clock.record(ev_e[t])
            if need and feeder is not None:
                feeder.resume()
            host_ns[t] = time.perf_counter_ns() - h0
        clock.record(ev_s[N])
        cpu_b, wall_b = SC.cpu_snapshot(), time.perf_counter()
    finally:
        if aff0 is not None:
            os.sched_setaffinity(0, aff0)
        if feeder is not None:
            feeder.stop(drain=True)
        ctx.sync()
    if feeder is not None:
        feeder.raise_if_failed()
    ref = ev_s[0]
    ts = [0.0] + [clock.elapsed(ref, e) for e in ev_s[1:]]
    te = [clock.elapsed(ref, e) for e in ev_e]
    per = SC.periods(ts)
    steps_rows = []
    for t in range(N):
        steps_rows.append(dict(arm=arm.name, t=t, rewarm=bool(rewarm[t]), rewarm_why=rewarm[t], start_ms=ts[t],
                               period_ms=per[t], span_ms=te[t] - ts[t], host_iter_ms=host_ns[t] / 1e6,
                               released_groups=(releases[t].n_groups if t in releases else 0), backlog_host=backlog[t],
                               outstanding_host=outst[t], pause_ms=pause_ns[t] / 1e6))
    rec["rewarm_steps"] = [dict(t=t, why=rewarm[t]) for t in range(N) if rewarm[t]]
    rec["tokens_sha256"] = SC.sha256_bytes(toks.cpu().contiguous().numpy().tobytes())
    rec["periods"] = per
    rec["steps_rows"] = steps_rows
    rec["cores"] = SC.cores_used(cpu_a or {}, cpu_b or {}, wall_b - wall_a, main_tid,
                                 feeder.tid if feeder is not None else None)
    if feeder is not None:
        if len(feeder.done_list) != len(releases):
            raise RuntimeError("feeder completed %d of %d releases" % (len(feeder.done_list), len(releases)))
        rows, rel_t, useful, done_ms, gate_ms = [], [], [], [], []
        for r in feeder.done_list:
            t = r.spec.t
            dn = clock.elapsed(ref, r.done)
            g = ts[t]
            rel_t.append(t)
            useful.append(r.spec.useful)
            done_ms.append(dn)
            gate_ms.append(g)
            late = dn - ts[min(t + d, N)]
            rows.append(dict(arm=arm.name, t=t, deadline_t=t + d, groups=r.spec.n_groups, useful=r.spec.useful,
                             layers=r.layers, gate_ms=g, done_ms=dn, lateness_ms=late, on_time=late <= 0.0,
                             queue_wait_ms=(r.pop_ns - r.rel_ns) / 1e6, gate_wait_ms=r.gate_ns / 1e6,
                             service_ms=(r.end_ns - r.start_ns) / 1e6, pause_ms=r.pause_ns / 1e6))
        rec["release_rows"] = rows
        dl = SC.delivery(rel_t, useful, done_ms, gate_ms, ts, d, w0, w1)
        rec["queue_outstanding_gpu"] = SC.queue_trend(dl["outstanding_gpu"][a.settle:])
        rec["queue_backlog_host"] = SC.queue_trend([float(x) for x in backlog[w0 + a.settle:w1]])
        rec["delivery"] = dl
        rec["pauses"] = dict(count=len(feeder.pauses), wait_ms=sum(b - a_ for a_, b in feeder.pauses) / 1e6,
                             paused_ms=feeder.paused_ns_total / 1e6)
        rec["feeder_affinity"] = feeder.observed_affinity
        rec["service_ms"] = dict(p50=SC.pctl([x["service_ms"] for x in rows], 50),
                                 p95=SC.pctl([x["service_ms"] for x in rows], 95))
        rec["content"] = finish(sched)
    return rec


def steady_steps(arm_rec: Dict[str, Any], a, cfg) -> List[int]:
    """Steps of the steady statistics: [w0 + settle, w1) without the re-warm steps and the step before each (its period
    ends at the re-warm step's start, after the feeder's pause)."""
    bad = set()
    for r in arm_rec["rewarm_steps"]:
        bad.add(r["t"])
        bad.add(r["t"] - 1)
    return [t for t in range(cfg["w0"] + a.settle, cfg["w1"]) if t not in bad]


def finalize(arms: List[Dict[str, Any]], a, cfg) -> Dict[str, Any]:
    """Paired slowdowns, trajectory equality and the verdict."""
    alone = next((r for r in arms if r.get("method") == "alone" and "periods" in r), None)
    fails = []
    shas = {r["arm"]: r.get("tokens_sha256") for r in arms if "tokens_sha256" in r}
    ref_sha = alone["tokens_sha256"] if alone else (next(iter(shas.values())) if shas else None)
    for r in arms:
        if "error" in r:
            fails.append("%s raised %s" % (r["arm"], r["error"]["type"]))
            continue
        r["trajectory_equal"] = (r.get("tokens_sha256") == ref_sha)
        if not r["trajectory_equal"]:
            fails.append("%s trajectory differs from %s" % (r["arm"], "alone" if alone else "the first arm"))
        st = steady_steps(r, a, cfg)
        pr = r["periods"]
        r["decode"] = dict(steady_steps=len(st), period_mean_ms=SC.mean([pr[t] for t in st]),
                           period_p50_ms=SC.pctl([pr[t] for t in st], 50), period_p95_ms=SC.pctl([pr[t] for t in st], 95),
                           rewarm_period_ms=[dict(t=x["t"], why=x["why"], ms=pr[x["t"]]) for x in r["rewarm_steps"]])
        if alone is not None and r is not alone:
            r["slowdown"] = SC.paired_slowdown({t: alone["periods"][t] for t in st},
                                               {t: pr[t] for t in st}, st, n_boot=a.boot, seed=a.seed)
        c = r.get("content")
        if c is not None:
            for kv in ("k", "v"):
                if c[kv]["groups_bad"] or c[kv]["untouched_bad"]:
                    fails.append("%s content %s: %s" % (r["arm"], kv, c[kv]))
    return dict(verdict="PASS" if not fails else "FAIL", failures=fails)


# ------------------------------------------------------------------------------------------------------- the GPU side
def gpu_context(a, cfg, wl, rec) -> ArmContext:
    import cpupack_core as K
    import nosi.nosa_llama as nl
    from nosi.cache_engine_gpu import InfLLMv2Cache as NoOffloadCache
    from nosi.flash_cache_engine.flash_h2d_persistent import flash_h2d_persistent
    rec["modules"] = {name: dict(file=str(Path(m.__file__).resolve()), sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
                      for name, m in (("nosa_llama", nl), ("cache_engine_gpu", sys.modules["nosi.cache_engine_gpu"]),
                                      ("flash_h2d_persistent", sys.modules["nosi.flash_cache_engine.flash_h2d_persistent"]),
                                      ("cpupack_core", K), ("sustained_core", SC), ("sustained_feed", sys.modules[__name__]))}
    repo = os.environ.get("NOSI_REPO")
    if repo:
        root = Path(repo).resolve()
        outside = {n: m["file"] for n, m in rec["modules"].items() if not Path(m["file"]).is_relative_to(root)}
        if outside:
            raise ModuleOrigin("modules loaded from outside NOSI_REPO %s: %s" % (root, outside))
    t0 = time.perf_counter()
    model = nl.Llama(model_name=os.path.realpath(a.model), device="cuda", offload=False)
    rec["model_load_s"] = time.perf_counter() - t0
    g = cfg["geom"]
    t0 = time.perf_counter()
    src_k, src_v = [], []
    for l in range(a.src_layers):
        k = torch.empty((g.B, g.S, g.H, g.D), dtype=torch.bfloat16, pin_memory=True)
        v = torch.empty((g.B, g.S, g.H, g.D), dtype=torch.bfloat16, pin_memory=True)
        src_k.append(SC.fill_source(k, l, 0))
        src_v.append(SC.fill_source(v, l, 1))
    rec["source_s"] = time.perf_counter() - t0
    rec["source_bytes"] = 2 * a.src_layers * g.B * g.S * g.H * g.D * 2
    streams = {n: torch.cuda.Stream() for n in ("copy", "scatter", "gather")}
    be = K.CudaBackend(streams, torch.cuda.Stream())
    sb = K.slot_bytes(a.cap_groups)
    stage = torch.zeros((a.ring, sb), dtype=torch.uint8, pin_memory=True)
    land = torch.zeros((a.ring, sb), dtype=torch.uint8, device="cuda")
    pipe = K.Pipe(be, stage, land, a.cap_groups)
    scr_k = torch.empty((g.B, g.M * g.R, g.H, g.D), dtype=torch.bfloat16, device="cuda")
    scr_v = torch.empty_like(scr_k)
    ids_host = [torch.empty((g.H, g.B, g.M), dtype=torch.int32, pin_memory=True) for _ in range(a.ring)]
    ids_dev = [torch.empty((g.H, g.B, g.M), dtype=torch.int32, device="cuda") for _ in range(a.ring)]

    def gather(dst, src, ids):
        flash_h2d_persistent(dst, src, ids, g.R, n_ctas=a.w8_ctas, num_warps=a.w8_warps)

    def reset_all():
        torch.cuda.synchronize()
        be.reset()
        pipe.reset()
        SC.poison(scr_k)
        SC.poison(scr_v)
        torch.cuda.synchronize()

    def transport(arm):
        if arm.method == "cpu":
            work = SC.cpu_work_fn(K, be, pipe, g, src_k, src_v, scr_k, scr_v)
        else:
            work = SC.gpu_work_fn(be, gather, g, src_k, src_v, scr_k, scr_v, ids_host, ids_dev)

        def finish(sched):
            torch.cuda.synchronize()
            return dict(k=SC.check_scratch(scr_k, src_k, sched.last_l, sched.last_blk, g.R),
                        v=SC.check_scratch(scr_v, src_v, sched.last_l, sched.last_blk, g.R))
        return work, reset_all, finish

    # pre-warm both transports (Triton JIT of the W8 kernel, the packer's first calls) on a throwaway release
    t0 = time.perf_counter()
    for method in ("cpu", "gpu"):
        reset_all()
        w = transport(SC.ArmSpec(method, 1.0, 1))[0]
        sch = SC.Schedule(g, 1, min(64, 2 * g.M), seed=a.seed + 99)
        fd = SC.Feeder(w, name="prewarm-" + method)
        fd.start()
        fd.release(SC.Release(spec=sch.release(0), deadline_t=0))
        fd.stop(drain=True)
        fd.raise_if_failed()
        torch.cuda.synchronize()
    rec["prewarm_s"] = time.perf_counter() - t0
    ids = wl["ids"].to("cuda")
    return ArmContext(model, lambda: NoOffloadCache(config=model.config, num_hidden_layers=model.config.num_hidden_layers,
                                                    has_kv_bias=True),
                      SC.CudaClock(), transport, "cuda", ids, torch.cuda.synchronize)


def _err(e: BaseException) -> Dict[str, Any]:
    return dict(type=type(e).__name__, message=str(e)[:4000],
                traceback="".join(traceback.format_exception(type(e), e, e.__traceback__))[-32768:])


def write_outputs(out: Path, rec: Dict[str, Any], arms: List[Dict[str, Any]]) -> None:
    with open(out / "steps.jsonl", "w") as f:
        for r in arms:
            for row in r.get("steps_rows", []):
                f.write(SC.jsonl_line(SC.STEP_FIELDS, row))
    with open(out / "releases.jsonl", "w") as f:
        for r in arms:
            for row in r.get("release_rows", []):
                f.write(SC.jsonl_line(SC.RELEASE_FIELDS, row))
    slim = []
    for r in arms:
        s = {k: v for k, v in r.items() if k not in ("steps_rows", "release_rows", "periods")}
        if "delivery" in s:
            s["delivery"] = {k: v for k, v in s["delivery"].items() if k not in ("delivered_cum_bytes", "outstanding_gpu")}
        slim.append(s)
    rec["arm_results"] = slim
    (out / "summary.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    names = [n for n in ("releases.jsonl", "steps.jsonl", "summary.json") if (out / n).exists()]
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (hashlib.sha256((out / n).read_bytes()).hexdigest(), n)
                                            for n in names))


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print("sustained_feed: --out %s exists" % out, file=sys.stderr)
        return 2
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    try:
        man_rows = json.loads(Path(a.manifest).read_text())["shape"][0]
        cfg = check_config(a, man_rows, cpus)
        wl = load_workload(Path(a.tokens), Path(a.manifest), a.doc_offset, a.batch)
    except SystemExit4 as e:
        print("sustained_feed: refused: %s" % e, file=sys.stderr)
        return 4
    out.mkdir(parents=True)
    rec: Dict[str, Any] = dict(label=SC.LABEL, argv=list(sys.argv if argv is None else argv), config=dict(vars(a)),
                               derived=dict(w0=cfg["w0"], w1=cfg["w1"], max_gen=cfg["max_gen"], nreq=cfg["nreq"],
                                            main_core=cfg["main_core"], feeder_cores=cfg["feeder_cores"],
                                            useful_per_release={x.name: cfg["nreq"].get(x.name, 0) * x.frag * SC.GROUP_USEFUL
                                                                for x in cfg["arms"]}),
                               limits=dict(max_steps=CIS_HEADROOM, rewarm="every 64 tokens and at every 128-column score "
                                           "width change (sustained_core.rewarm_need)",
                                           kv_headroom_tokens=cfg["max_gen"]),
                               workload=dict(rows=wl["docs"], file_sha256=wl["file_sha256"]), cpus=cpus,
                               env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NOSI_ATTN_SPLITS", "OMP_NUM_THREADS",
                                                                    "PYTHONPATH", "TORCH_CUDA_ARCH_LIST")},
                               status="running")
    (out / "summary.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    if a.dry:
        rec["status"] = "dry"
        write_outputs(out, rec, [])
        print("sustained_feed: dry run, plan written to %s" % out)
        return 0
    arms: List[Dict[str, Any]] = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        props = torch.cuda.get_device_properties(0)
        rec["gpu"] = dict(name=props.name, uuid=str(getattr(props, "uuid", "UNOBSERVED")),
                          total_GB=props.total_memory / 1e9, torch=torch.__version__)
        if a.gpu_uuid and rec["gpu"]["uuid"] != a.gpu_uuid:
            rec["status"] = "wrong_gpu"
            write_outputs(out, rec, arms)
            return 3
        torch.set_num_threads(a.pack_threads)
        ctx = gpu_context(a, cfg, wl, rec)
        for arm in cfg["arms"]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                r = run_arm(ctx, arm, cfg, a)
            except Exception as e:                                       # noqa: BLE001 (recorded; the run goes on)
                r = dict(arm=arm.name, method=arm.method, error=_err(e))
            r["peak_alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
            arms.append(r)
            rec["status"] = "arm %s done" % arm.name
            write_outputs(out, rec, arms)
        rec.update(finalize(arms, a, cfg))
        rec["status"] = "done"
    except ModuleOrigin as e:
        rec["error"] = _err(e)
        rec["status"] = "module_origin"
        rec["verdict"] = "FAIL"
        write_outputs(out, rec, arms)
        return 6
    except Exception as e:                                               # noqa: BLE001
        rec["error"] = _err(e)
        rec["status"] = "error"
        rec["verdict"] = "FAIL"
    write_outputs(out, rec, arms)
    return 0 if rec.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
