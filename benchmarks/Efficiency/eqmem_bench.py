"""EQUAL-MEMORY HEADLINE driver (paper 4.1): the rules, units and labels are in eqmem_core.py (its docstring).

  python eqmem_bench.py --mode plan   --budget-gb 75 --cells original,cached:33,cached:50,spec:2 --out <plan.json>
  python eqmem_bench.py --mode arm    --budget-gb 75 --cell cached:50 --model <NOSA-8B> --tokens <c16k_tokens.bin>
                                      --manifest <c16k_manifest.json> --out <new dir> [--rep 0] [--gpu-uuid <uuid>]
  python eqmem_bench.py --mode report --results <run dir> --out <new dir>

ARM (one cell, one process, one GPU): the engine knobs of the cell MUST already be in the environment
(NOSI_POOL_BLOCKS = P, NOSI_VERIFY_ROUND_SLOTS = R; cache_engine.py reads them once at import; checked before and after
the import); the batch is the plan's (eqmem_core.plan_cell over the budget, N, the fit). Steps: NOSA-8B (offload=True);
the cache with max_gen_len = the plan's headroom; the instrument (every engine's decode_update wrapped: two CUDA events
around it and the count of its load mask >= 0, no host sync); prefill of the B requests (the workload rows of
eqmem_core.doc_rows; per-request completion = an event after the last layer's prefill_forward of that request) INSIDE
the wall clock; then the decode of N - 1 tokens per request: original / cached = NOSA's decode_inference step by step
(re-warm by eqmem_core.rewarm_need, lookahead 1); spec = the R2 pilot's run_spec_arm (spec_loop_pilot.py, LOCKSTEP
commit c = 1 + min_b n_acc), its re-warm test replaced by eqmem_core's (lookahead U = K + 1, the width rule included).
Memory: torch peaks (allocated, reserved) and device samples (cudaMemGetInfo) after the model load, after prefill,
every 16 decode steps (original / cached; never inside a step), at the end.
OUTPUT (--out, refused when it exists): result.json (plan, docs, metrics, memory with the budget verdict, re-warm
steps, the pilot's aggregates for spec, module origins, error), tokens.pt (the committed token matrix), fit_row.json
(the measured row for eqmem_core.fit_from_rows), SHA256SUMS.
EXIT: 0 the cell ran and is VALID (inside the budget); 1 an error or OVER_BUDGET; 2 --out exists; 3 another GPU; 4 a
refused configuration (no batch fits, a knob in the environment differs from the cell's); 6 a module from outside
NOSI_REPO.
4.2 ADDITIONS (ctxsweep_bench.py calls run_arm with an S-aware plan; the 4.1 behaviour is unchanged): the workload
context is the plan's S; every device sample carries the host memory (VmRSS / VmHWM, the memory cgroup); result.json
adds memory.phase_peaks (prefill / decode: the peak statistics are reset after the prefill), host (host_record) and
latency (ctxsweep_core.latency_metrics, CUDA-event clock); fit_row.json adds S, max_gen, host_peak_gib and the phase peaks.
REPORT: every result.json under --results (one comparison: the same budget, margin, N, S) -> report.json / report.md:
the cells (repetitions summarized), the best VALID cell per family, the speedups, the prefix agreement of every cell's
tokens with the original cell's (an observation), and the DERIVED time to serve Q requests in static batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import eqmem_core as EC  # noqa: E402
import ctxsweep_core as CS  # noqa: E402  (paper 4.2: the host-memory parsers and the latency metrics)


class Refused(Exception):
    """A refused configuration (exit 4)."""


class ModuleOrigin(Exception):
    """A module loaded from outside NOSI_REPO (exit 6)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="equal-memory headline (eqmem_core.py)")
    ap.add_argument("--mode", choices=("plan", "arm", "report"), required=True)
    ap.add_argument("--budget-gb", type=float, default=None)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--cells", default="original,cached:33,cached:50,spec:2")
    ap.add_argument("--n-new", type=int, default=512, help="generated tokens per request (the first from the prefill)")
    ap.add_argument("--margin-gb", type=float, default=1.0)
    ap.add_argument("--host-budget-gb", type=float, default=220.0)
    ap.add_argument("--b-cap", type=int, default=4096)
    ap.add_argument("--fit", default=None, help="JSON of MemFit fields (default: eqmem_core.MemFit())")
    ap.add_argument("--fit-rows", default=None, help="a file of fit_row.json lines to refit from (eqmem_core.fit_from_rows)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--gpu-uuid", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=None)
    ap.add_argument("--workload-q", type=int, default=None, help="report: the request count of the derived time")
    return ap


def load_fit(a) -> EC.MemFit:
    fit = EC.MemFit()
    if a.fit:
        d = json.loads(Path(a.fit).read_text())
        fit = EC.MemFit(**{k: v for k, v in d.items() if k in EC.MemFit.__dataclass_fields__})
    if a.fit_rows:
        rows = [json.loads(x) for x in Path(a.fit_rows).read_text().splitlines() if x.strip()]
        fit = EC.fit_from_rows(rows, fit)
    return fit


def plan_for(a, cell: EC.Cell, fit: EC.MemFit) -> Dict[str, Any]:
    if a.budget_gb is None:
        raise Refused("--budget-gb is required")
    try:
        return EC.plan_cell(cell, a.budget_gb, a.n_new, fit, a.margin_gb, EC.CTX, a.host_budget_gb, a.b_cap)
    except ValueError as e:
        raise Refused(str(e))


def load_workload(tokens: Path, manifest: Path, rows: List[int], want_ctx: int = EC.CTX):
    """The int32-le rows of the (n, want_ctx) token file (16128 for 4.1; the 4.2 context sweep passes its S); each
    used row's sha256 must equal the manifest's."""
    import torch
    man = json.loads(manifest.read_text())
    n, ctx = man["shape"]
    if ctx != int(want_ctx) or man.get("dtype") != "int32-le":
        raise Refused("manifest shape %s dtype %s" % (man["shape"], man.get("dtype")))
    raw = tokens.read_bytes()
    if len(raw) != n * ctx * 4:
        raise Refused("%s holds %d bytes, the manifest says %d x %d int32" % (tokens, len(raw), n, ctx))
    out, cache = [], {}
    for r in rows:
        if r not in cache:
            row = raw[r * ctx * 4:(r + 1) * ctx * 4]
            sha = hashlib.sha256(row).hexdigest()
            if sha != man["docs"][r]["sha256"]:
                raise Refused("row %d sha256 %s != manifest %s" % (r, sha, man["docs"][r]["sha256"]))
            cache[r] = torch.frombuffer(bytearray(row), dtype=torch.int32).to(torch.int64)
        out.append(cache[r])
    return torch.stack(out), dict(n_rows=n, docs=[dict(row=r, index=man["docs"][r]["index"]) for r in sorted(cache)],
                                  file_sha256=hashlib.sha256(raw).hexdigest())


def check_env(cell: EC.Cell) -> Dict[str, str]:
    """The cell's engine knobs must be in the environment before the engine is imported (read once at import)."""
    want = cell.env()
    got = {k: (os.environ.get(k) or "0") for k in want}
    if got != want:
        raise Refused("environment %s != the cell's %s (start the process with the cell's knobs)" % (got, want))
    return got


# ------------------------------------------------------------------------------------------------------- GPU pieces
class EngineProbe:
    """Wraps every layer engine's decode_update (an instance attribute bound in CacheEngine.__init__): two CUDA events
    around it and the count of its load mask >= 0 (blocks fetched host -> GPU; a pool hit clears the mask, so it counts
    the real H2D on both the shipped and the pooled path) accumulated on the device; no host sync."""

    def __init__(self, torch, cache):
        self.torch, self.cache, self.windows = torch, cache, []
        with torch.inference_mode():
            self.loads = torch.zeros((), dtype=torch.int64, device="cuda")

    def install(self):
        T = self.torch
        for lay in self.cache.layers:
            eng = lay.cache_engine
            orig = eng.decode_update

            def wrapped(*a, _orig=orig, _eng=eng, **kw):
                e0 = T.cuda.Event(enable_timing=True)
                e1 = T.cuda.Event(enable_timing=True)
                e0.record()
                out = _orig(*a, **kw)
                e1.record()
                self.loads += (_eng._load_mask >= 0).sum()
                self.windows.append((e0, e1))
                return out
            eng.decode_update = wrapped

    def harvest(self):
        return int(self.loads), sum(a.elapsed_time(b) for a, b in self.windows), len(self.windows)


def host_now() -> Dict[str, Any]:
    """The process's host memory now, GiB: /proc/self/status (rss, hwm = the peak resident set, lck, pin) and the
    memory cgroup's usage and peak (ctxsweep_core.cgroup_memory_files); a value that cannot be read is absent / None."""
    out: Dict[str, Any] = {}
    try:
        out.update(CS.parse_proc_status(Path("/proc/self/status").read_text()))
    except Exception:                                                         # noqa: BLE001 (best effort)
        pass
    try:
        files = CS.cgroup_memory_files(Path("/proc/self/cgroup").read_text())
    except Exception:                                                         # noqa: BLE001
        files = None
    if files:
        out["cgroup_version"] = files["version"]
        for k in ("usage", "peak"):
            try:
                out["cgroup_%s_gib" % k] = int(Path(files[k]).read_text().split()[0]) / CS.GIB
            except Exception:                                                 # noqa: BLE001
                out["cgroup_%s_gib" % k] = None
    return out


def host_record(torch, samples: List[Dict[str, Any]], pinned_nbytes: List[int]) -> Dict[str, Any]:
    """The cell's host memory: peak_gib = the largest VmHWM seen (the process's peak resident set; the pinned KV is
    resident: ctxsweep_core HOST_EVIDENCE), the memory cgroup's peak, the pinned tensors (nominal and power-of-two
    sizes) and torch's host-allocator statistics when this torch has them."""
    now = host_now()
    hs = [s.get("host") or {} for s in samples] + [now]

    def mx(k):
        v = [h[k] for h in hs if h.get(k) is not None]
        return max(v) if v else None
    stats = None
    f = getattr(torch.cuda, "host_memory_stats", None)
    if f is not None:
        try:
            stats = {k: v for k, v in dict(f()).items() if isinstance(v, (int, float))}
        except Exception as e:                                                # noqa: BLE001 (recorded)
            stats = dict(error=str(e)[:300])
    return dict(peak_gib=mx("hwm_gib"), rss_end_gib=now.get("rss_gib"), cgroup_peak_gib=mx("cgroup_peak_gib"),
                cgroup_usage_max_gib=mx("cgroup_usage_gib"), pinned=CS.pinned_summary(pinned_nbytes),
                torch_host_memory_stats=stats, now=now)


def mem_sample(torch, label: str) -> Dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    return dict(label=label, used_gb=(total - free) / 1e9, reserved_gb=torch.cuda.memory_reserved() / 1e9,
                alloc_gb=torch.cuda.memory_allocated() / 1e9, total_gb=total / 1e9, host=host_now())


def live_rewarm(model, cache, lookahead: int):
    """eqmem_core.rewarm_need on the live state: the last warm-up's shapes are the model's own buffers
    (pooling_buf_all's last dimension = pooled blocks, layer 0's score_buf's last dimension = the score width)."""
    lay = cache.layers[0]
    warm = None
    if model.has_buffers:
        warm = dict(out_len=int(model.pooling_buf_all.shape[-1]), width=int(model.layers[0].score_buf.shape[-1]))
    return EC.rewarm_need(cache.get_seq_length(0), lay.no_compress_k_len, lay.cached_compressed_max_seqlen, warm,
                          lookahead)


def decode_shipped(torch, model, cache, t0, pos0, N: int, samples: List, e_prev=None) -> Dict[str, Any]:
    """original / cached: N - 1 greedy decode steps of NOSA's decode_inference; per-step CUDA events; re-warm by
    live_rewarm (lookahead 1); device samples every 16 steps (between steps). e_prev (the prefill-end event): itl_ms =
    the intervals between consecutive step-end events, the first from e_prev (host gaps included)."""
    B = t0.shape[0]
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    tok, pos = t0, pos0.clone()
    seq = [t0]
    evs, rewarms = [], []
    model.has_buffers = False
    with torch.inference_mode():
        for it in range(N - 1):
            need, why = live_rewarm(model, cache, 1)
            if need:
                model.has_buffers = False
                rewarms.append(dict(step=it, why=why))
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            lg = model.decode_inference(tok, cu, pos, cache)
            tok = lg[:, -1, :].argmax(-1, keepdim=True)
            e1.record()
            seq.append(tok)
            pos = pos + 1
            evs.append((e0, e1))
            if it % 16 == 15:
                samples.append(mem_sample(torch, "decode %d" % it))
    torch.cuda.synchronize()
    ms = [a.elapsed_time(b) for a, b in evs]
    itl = None
    if e_prev is not None:
        ends = [e_prev] + [b for _, b in evs]
        itl = [ends[i].elapsed_time(ends[i + 1]) for i in range(len(ends) - 1)]
    return dict(seq=torch.cat(seq, dim=1).cpu(), unit_tokens=[1.0] * len(ms), unit_ms=ms, rewarms=rewarms,
                gen_tokens=N, rounds=None, itl_ms=itl)


def decode_spec(torch, model, cache, t0, pos0, N: int, cell: EC.Cell, out: Path) -> Dict[str, Any]:
    """spec: the R2 pilot's run_spec_arm (LOCKSTEP) committing >= N - 1 tokens after t0, with its re-warm test
    replaced by live_rewarm(lookahead K + 1)."""
    os.environ["NOSI_SPEC_OUT"] = str(out / "pilot")                 # the pilot creates its output directory at import
    import spec_loop_pilot as pilot
    pilot.needs_rewarm = lambda m, c: live_rewarm(m, c, cell.K + 1)[0]
    res = pilot.run_spec_arm(model, cache, t0, pos0, cell.K, N - 1, False)
    torch.cuda.synchronize()
    units_t, units_ms, gather = [], [], 0.0
    for r in res["rounds"]:
        if r.get("rewarm"):
            units_t.append(1.0)
            units_ms.append(float(r["step_ms"]))
            continue
        units_t.append(float(r["c"]))
        units_ms.append(sum(r["draft_ms"]) + float(r["verify_ms"] or 0.0) + float(r["rollback_ms"] or 0.0))
        vs = r.get("verify_split") or {}
        gather += float(vs.get("gather_ms") or 0.0)
    keep = {k: res.get(k) for k in ("K", "U", "W", "R", "tokens", "counters", "wire", "timely_recall", "host_syncs",
                                     "yield_lockstep_tokens", "yield_ragged_projection_tokens", "agg")}
    return dict(seq=res["seq"], unit_tokens=units_t, unit_ms=units_ms, rewarms=[dict(round=r["r"]) for r in res["rounds"]
                                                                                 if r.get("rewarm")],
                gen_tokens=1 + int(res["tokens"]), rounds=keep, spec_gather_ms=gather,
                spec_wire_blocks=int(res["counters"]["prefetch_blocks"]) + int(res["counters"]["late_fetch_blocks"]))


def run_arm(a, plan: Optional[Dict[str, Any]] = None) -> int:
    """One cell. plan: None = the 4.1 plan (plan_for at 16128); the 4.2 context sweep passes ctxsweep_core's S-aware
    plan (ctxsweep_bench.py). Beyond the 4.1 record: the prefill and decode phase peaks (the peak statistics are reset
    after the prefill; memory_record gets the larger of the two), the host memory (host_record), the latency on the
    CUDA-event clock (ctxsweep_core.latency_metrics), and fit_row.json carries S, max_gen and the host peak."""
    cell = EC.parse_cell(a.cell)
    if plan is None:
        fit = load_fit(a)
        plan = plan_for(a, cell, fit)
    knobs = check_env(cell)
    out = Path(a.out)
    if out.exists():
        print("eqmem: --out %s exists" % out, file=sys.stderr)
        return 2
    sched = EC.doc_rows(plan["B"], json.loads(Path(a.manifest).read_text())["shape"][0], a.doc_offset)
    ids, wl = load_workload(Path(a.tokens), Path(a.manifest), sched["rows"], int(plan["S"]))
    out.mkdir(parents=True)
    rec: Dict[str, Any] = dict(cell=cell.name, family=cell.family, rep=a.rep, plan=plan, plan_key=list(EC.plan_key(plan)),
                               knobs=knobs, docs=dict(sched, workload=wl), argv=sys.argv, status="running",
                               env={k: os.environ.get(k) for k in ("NOSI_ATTN_SPLITS", "PYTORCH_CUDA_ALLOC_CONF",
                                                                    "CUDA_VISIBLE_DEVICES", "PYTHONPATH")})
    write(out, rec)
    import torch
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        props = torch.cuda.get_device_properties(0)
        rec["gpu"] = dict(name=props.name, uuid=str(getattr(props, "uuid", "UNOBSERVED")), total_gb=props.total_memory / 1e9)
        if a.gpu_uuid and rec["gpu"]["uuid"] != a.gpu_uuid:
            rec["status"] = "wrong_gpu"
            write(out, rec)
            return 3
        import nosi.nosa_llama as nl
        import nosi.cache_engine as ce
        mods = dict(nosa_llama=nl, cache_engine=ce, eqmem_core=EC, eqmem_bench=sys.modules[__name__], ctxsweep_core=CS)
        drv = sys.modules.get("__main__")
        if getattr(drv, "__file__", None) and Path(drv.__file__).name in ("eqmem_bench.py", "ctxsweep_bench.py"):
            mods["driver"] = drv
        rec["modules"] = {k: dict(file=str(Path(m.__file__).resolve()),
                                  sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()) for k, m in mods.items()}
        repo = os.environ.get("NOSI_REPO")
        if repo:
            bad = {k: v["file"] for k, v in rec["modules"].items() if not Path(v["file"]).is_relative_to(Path(repo).resolve())}
            if bad:
                raise ModuleOrigin("modules from outside NOSI_REPO: %s" % bad)
        if (ce.POOL_BLOCKS, ce.VERIFY_ROUND_SLOTS) != (cell.P, cell.R):
            raise Refused("the engine read P %d R %d, the cell is P %d R %d" % (ce.POOL_BLOCKS, ce.VERIFY_ROUND_SLOTS,
                                                                                  cell.P, cell.R))
        torch.cuda.reset_peak_memory_stats()
        samples = [mem_sample(torch, "start")]
        t0 = time.time()
        model = nl.Llama(model_name=os.path.realpath(a.model), device="cuda", offload=True)
        rec["model_load_s"] = time.time() - t0
        samples.append(mem_sample(torch, "model"))
        cache = ce.InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
        for lay in cache.layers:
            lay.cache_engine.max_gen_len = plan["max_gen"]
        probe = EngineProbe(torch, cache)
        probe.install()
        x = ids.to("cuda")
        last = model.layers[-1]
        pre_ev = []
        orig_pf = last.prefill_forward

        def pf(*aa, **kw):
            o = orig_pf(*aa, **kw)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            pre_ev.append(e)
            return o
        last.prefill_forward = pf
        torch.cuda.synchronize()
        e_start = torch.cuda.Event(enable_timing=True)
        t_start = time.time()
        e_start.record()
        logits, position_ids = model.batch_prefill(x, cache)
        e_pre = torch.cuda.Event(enable_timing=True)
        e_pre.record()
        torch.cuda.synchronize()
        t_pre = time.time()
        del last.prefill_forward
        if len(pre_ev) != plan["B"]:
            raise RuntimeError("prefill recorded %d request events for B %d" % (len(pre_ev), plan["B"]))
        prefill_done = [e_start.elapsed_time(e) for e in pre_ev]
        first_token = e_start.elapsed_time(e_pre)
        samples.append(mem_sample(torch, "prefill"))
        peak_pre = dict(alloc_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
        torch.cuda.reset_peak_memory_stats()
        t0tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        pos0 = position_ids[:, -1:] + 1
        if cell.family == "spec":
            dec = decode_spec(torch, model, cache, t0tok, pos0, a.n_new, cell, out)
        else:
            dec = decode_shipped(torch, model, cache, t0tok, pos0, a.n_new, samples, e_prev=e_pre)
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        torch.cuda.synchronize()
        t_end = time.time()
        samples.append(mem_sample(torch, "end"))
        peak_dec = dict(alloc_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
        loads, wait_ms, n_windows = probe.harvest()
        if cell.family == "spec":
            loads += dec["spec_wire_blocks"]
            wait_ms += dec["spec_gather_ms"]
        rec["metrics"] = EC.arm_metrics(plan["B"], dec["gen_tokens"], t_start, t_pre, t_end, prefill_done, first_token,
                                        dec["unit_tokens"], dec["unit_ms"], loads, wait_ms, n_boot=a.boot, seed=a.seed)
        rec["memory"] = EC.memory_record(max(peak_pre["alloc_gb"], peak_dec["alloc_gb"]),
                                         max(peak_pre["reserved_gb"], peak_dec["reserved_gb"]), samples, plan["budget_gb"])
        rec["memory"]["phase_peaks"] = dict(prefill=peak_pre, decode=peak_dec,
                                            binding="prefill" if peak_pre["alloc_gb"] >= peak_dec["alloc_gb"] else "decode")
        rec["latency"] = CS.latency_metrics(plan["B"], dec["gen_tokens"], first_token, prefill_done, dec["unit_tokens"],
                                            dec["unit_ms"], dec.get("itl_ms"), e_start.elapsed_time(e_end))
        rec["rewarms"] = dec["rewarms"]
        rec["decode_update_windows"] = n_windows
        rec["spec"] = dec["rounds"]
        pinned = [t.numel() * t.element_size() for lay in cache.layers
                  for t in (lay.cache_engine._k_cpu, lay.cache_engine._v_cpu)]
        rec["host_pinned_gb"] = sum(pinned) / 1e9
        rec["host"] = host_record(torch, samples, pinned)
        torch.save(dict(seq=dec["seq"], rows=sched["rows"]), out / "tokens.pt")
        (out / "fit_row.json").write_text(json.dumps(dict(family=cell.family, cell=cell.name, B=plan["B"], slots=cell.slots,
                                                          alloc_peak_gb=rec["memory"]["alloc_peak_gb"],
                                                          reserved_peak_gb=rec["memory"]["reserved_peak_gb"],
                                                          nonpt_gb=rec["memory"]["nonpt_gb"], S=int(plan["S"]),
                                                          max_gen=int(plan["max_gen"]),
                                                          host_peak_gib=rec["host"]["peak_gib"],
                                                          prefill_alloc_peak_gb=peak_pre["alloc_gb"],
                                                          decode_alloc_peak_gb=peak_dec["alloc_gb"])) + "\n")
        rec["status"] = "done"
    except ModuleOrigin as e:
        rec["error"], rec["status"] = _err(e), "module_origin"
        write(out, rec)
        return 6
    except Refused as e:
        rec["error"], rec["status"] = _err(e), "refused"
        write(out, rec)
        return 4
    except Exception as e:                                                    # noqa: BLE001 (recorded)
        rec["error"], rec["status"] = _err(e), "error"
    write(out, rec)
    return 0 if (rec.get("status") == "done" and rec["memory"]["verdict"] == "VALID") else 1


def _err(e: BaseException) -> Dict[str, Any]:
    return dict(type=type(e).__name__, message=str(e)[:4000],
                traceback="".join(traceback.format_exception(type(e), e, e.__traceback__))[-32768:])


def write(out: Path, rec: Dict[str, Any]) -> None:
    (out / "result.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    names = [n for n in ("fit_row.json", "result.json", "tokens.pt") if (out / n).exists()]
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (hashlib.sha256((out / n).read_bytes()).hexdigest(), n)
                                            for n in names))


# ------------------------------------------------------------------------------------------------------- plan, report
def run_plan(a) -> int:
    fit = load_fit(a)
    out = Path(a.out)
    if out.exists():
        print("eqmem: --out %s exists" % out, file=sys.stderr)
        return 2
    plans = []
    for c in [x for x in a.cells.split(",") if x.strip()]:
        plans.append(plan_for(a, EC.parse_cell(c), fit))
    out.write_text(json.dumps(dict(budget_gb=a.budget_gb, n_new=a.n_new, margin_gb=a.margin_gb, fit=fit.as_dict(),
                                   plans=plans), indent=1) + "\n")
    for p in plans:
        print("%-14s B %4d (bound %-11s) slots %3d usable %3d predicted used %.1f GB alloc %.1f GB host %.0f GB"
              % (p["cell"], p["B"], p["bound_by"], p["slots"], p["usable"], p["predicted_used_gb"], p["predicted_alloc_gb"],
                 p["predicted_host_gb"]))
    return 0


def run_report(a) -> int:
    root = Path(a.results)
    out = Path(a.out)
    if out.exists():
        print("eqmem: --out %s exists" % out, file=sys.stderr)
        return 2
    results, dirs = [], {}
    for p in sorted(root.glob("*/result.json")):
        r = json.loads(p.read_text())
        if r.get("status") in ("running", "refused", "wrong_gpu", "module_origin") or "plan_key" not in r:
            continue
        if r.get("status") != "done":
            r.setdefault("error", dict(type="status", message=r.get("status")))
        results.append(r)
        dirs[(r["cell"], r.get("rep", 0))] = p.parent
    if not results:
        print("eqmem: no results under %s" % root, file=sys.stderr)
        return 1
    rep = EC.comparison(results)
    import torch
    ref = dirs.get(("original", 0))
    agree = {}
    if ref is not None and (ref / "tokens.pt").exists():
        rt = torch.load(ref / "tokens.pt")["seq"].tolist()
        for (cell, r_), d in sorted(dirs.items()):
            if (cell, r_) != ("original", 0) and (d / "tokens.pt").exists():
                agree["%s_rep%d" % (cell, r_)] = EC.prefix_agreement(torch.load(d / "tokens.pt")["seq"].tolist(), rt)
    rep["agreement_vs_original_rep0"] = agree
    valid = [r for r in results if not r.get("error") and r["memory"]["verdict"] == "VALID"]
    Q = a.workload_q or max((r["metrics"]["batch"] for r in valid), default=0)
    derived = {}
    for f, b in rep["best"].items():
        if not b:
            continue
        rs = [r for r in valid if r["cell"] == b["cell"]]
        e2e = EC.mean([r["metrics"]["e2e_s"] for r in rs])
        B = rs[0]["metrics"]["batch"]
        n_batches = math.ceil(Q / B) if B else None
        derived[f] = dict(cell=b["cell"], Q=Q, batches=n_batches, time_s=(n_batches * e2e) if n_batches else None,
                          label="DERIVED: ceil(Q / B) identical static batches x the measured batch time")
    rep["derived_workload"] = derived
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    lines = ["# equal-memory comparison (budget %s GB, N %s)" % (results[0]["plan"]["budget_gb"], results[0]["plan"]["N"]),
             "", "| cell | family | runs | batch | e2e tok/s | decode tok/s | bytes/token | data wait share | latency s | "
                 "used peak GB | over budget |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in sorted(rep["cells"].items()):
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %d |" % (
            name, c["family"], c["runs"], c.get("batch"), _m(c.get("e2e_tok_s")), _m(c.get("decode_tok_s")),
            _f(c.get("bytes_per_committed_token"), "%.0f"), _f(c.get("data_wait_share"), "%.3f"),
            _f(c.get("latency_e2e_s")), _f(c.get("used_peak_est_gb")), c["over_budget"]))
    lines += ["", "best per family: %s" % {f: (b or {}).get("cell") for f, b in rep["best"].items()},
              "speedups e2e: %s" % rep["speedup_e2e_tok_s"], "speedups decode: %s" % rep["speedup_decode_tok_s"],
              "LABEL: the spec family is LOCKSTEP (c = 1 + min_b n_acc per round): a LOWER BOUND of stage 3 until R2-M2"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _m(s) -> str:
    if not s or not s.get("n"):
        return "-"
    if s["n"] == 1:
        return "%.1f" % s["mean"]
    return "%.1f [%.1f, %.1f]" % (s["mean"], s["ci_lo"], s["ci_hi"])


def _f(v, fmt="%.2f") -> str:
    return "-" if v is None else (fmt % v)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.mode == "plan":
            return run_plan(a)
        if a.mode == "report":
            if not a.results:
                raise Refused("--results is required")
            return run_report(a)
        for k in ("cell", "model", "tokens", "manifest"):
            if getattr(a, k) is None:
                raise Refused("--%s is required in arm mode" % k)
        return run_arm(a)
    except Refused as e:
        print("eqmem: refused: %s" % e, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
