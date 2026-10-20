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
v2 (2026-10-20): steps.json, the per-unit record of the decode (eqmem_core.step_record: per step / round the GPU ms,
the data-wait ms, the rest, the blocks, the re-warm flag, the tokens; result.json keeps its summary as steps_summary,
or steps_error when a window could not be charged to its step); result.json "host": the process's host memory
(rss_peak_gb = VmHWM, ru_maxrss_gb, pinned_kv_gb = the engines' pinned K / V, vm_pin_gb, rss_samples[].rss_gb at the
device sample points). Plan mode takes a cell LIST with capacity scans (eqmem_core.expand_cells) and prints the step
commands (--step-nosi / --step-holder / --step-run / --step-reps). Report: the token agreement is keyed on the FIRST
COMPLETED original repetition (agreement_reference), plus the capacity-scan table and the per-step split.
EXIT: 0 the cell ran and is VALID (inside the budget); 1 an error or OVER_BUDGET; 2 --out exists; 3 another GPU; 4 a
refused configuration (no batch fits, a knob in the environment differs from the cell's); 6 a module from outside
NOSI_REPO.
4.2 ADDITIONS (ONE harness for 4.1 and 4.2: ctxsweep_bench.py calls run_arm with an S-aware plan; a 4.1 cell plans as
before): the workload context is the plan's S; result.json adds memory.phase_peaks (prefill / decode: the peak
statistics are reset after the prefill; the budget verdict takes the larger), host += host_extras (pinned_kv_pow2_gb,
cgroup_peak_gb, torch's host-allocator statistics) and latency (ctxsweep_core.latency_metrics, CUDA-event clock);
fit_row.json adds S, max_gen, host_peak_gib (= host.rss_peak_gb in GiB, the unit of srun --mem) and the phase peaks.
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
import ctxsweep_core as CS  # noqa: E402  (paper 4.2: the cgroup reader, the pinned sizes, the latency metrics)


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
    # v2, plan mode: the step commands of a cell list (a capacity scan: cached:P1/P2/... or scan:LO-HI:STEP)
    ap.add_argument("--step-nosi", default=None, help="plan: the full NOSI sha for the printed step commands")
    ap.add_argument("--step-holder", default=None, help="plan: the holder job id for the printed step commands")
    ap.add_argument("--step-run", default=None, help="plan: the run id for the printed step commands")
    ap.add_argument("--step-reps", default="0", help="plan: comma-separated repetitions (default 0)")
    return ap


def step_cell_text(cell: EC.Cell) -> str:
    """The cell as eqmem_step.sh takes it (spec only with the step's R = 62)."""
    if cell.family == "original":
        return "original"
    if cell.family == "cached":
        return "cached:%d" % cell.P
    if cell.R != EC.DEFAULT_R:
        raise Refused("eqmem_step.sh runs spec cells with R %d only, not %s" % (EC.DEFAULT_R, cell.name))
    return "spec:%d" % cell.K


def step_commands(cells: List[EC.Cell], budget: float, nosi: str, holder: str, run: str, reps: List[int]) -> List[str]:
    """One eqmem_step.sh invocation per (rep, cell), reps outer, then the report; the lead runs them one at a time."""
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", nosi or "") or not re.fullmatch(r"[0-9]+", holder or "") \
            or not re.fullmatch(r"[a-z0-9]{1,16}", run or ""):
        raise Refused("--step-nosi (40 hex), --step-holder (digits) and --step-run ([a-z0-9]{1,16}) are required for commands")
    if any(r < 0 or r > 9 for r in reps):
        raise Refused("reps must be 0 .. 9")
    b = ("%.1f" % budget).rstrip("0").rstrip(".")
    head = "bash benchmarks/Efficiency/eqmem_step.sh --nosi %s %s %s %s" % (nosi, holder, run, b)
    cmds = ["%s %s %d" % (head, step_cell_text(c), r) for r in reps for c in cells]
    return cmds + ["%s report" % head]


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
        self.torch, self.cache, self.windows, self.window_loads = torch, cache, [], []
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
                n = (_eng._load_mask >= 0).sum()
                self.loads += n
                self.window_loads.append(n)            # v2: per window, for the per-step split (device scalar, no sync)
                self.windows.append((e0, e1))
                return out
            eng.decode_update = wrapped

    def harvest(self):
        return int(self.loads), sum(a.elapsed_time(b) for a, b in self.windows), len(self.windows)

    def per_window(self):
        """After a synchronize: (ms per window, blocks per window), in launch order."""
        ms = [a.elapsed_time(b) for a, b in self.windows]
        loads = self.torch.stack(self.window_loads).tolist() if self.window_loads else []
        return ms, [int(x) for x in loads]


def proc_status() -> Dict[str, Any]:
    try:
        return EC.parse_proc_status(Path("/proc/self/status").read_text())
    except OSError:
        return dict(VmRSS=None, VmHWM=None, VmPin=None, VmLck=None)


def cgroup_memory() -> Dict[str, Any]:
    """The memory cgroup of this process (ctxsweep_core.cgroup_memory_files): version, usage and PEAK in GB (v1
    memory.max_usage_in_bytes, v2 memory.peak: the step's peak, page cache included); None where unreadable."""
    try:
        files = CS.cgroup_memory_files(Path("/proc/self/cgroup").read_text())
    except OSError:
        files = None
    if not files:
        return dict(cgroup_version=None, cgroup_usage_gb=None, cgroup_peak_gb=None)
    out: Dict[str, Any] = dict(cgroup_version=files["version"])
    for k in ("usage", "peak"):
        try:
            out["cgroup_%s_gb" % k] = int(Path(files[k]).read_text().split()[0]) / 1e9
        except (OSError, ValueError, IndexError):
            out["cgroup_%s_gb" % k] = None
    return out


def host_extras(torch, pinned_nbytes: List[int]) -> Dict[str, Any]:
    """4.2 additions to the host record (eqmem_core.host_record keeps B's fields: rss_peak_gb, rss_end_gb, ru_maxrss_gb,
    pinned_kv_gb, vm_pin_gb, rss_samples): the pinned K / V tensors at their POWER-OF-TWO size (what the CUDA caching
    host allocator reserves: ctxsweep_core HOST_EVIDENCE), the memory cgroup's peak, torch's host-allocator statistics
    when this torch has them."""
    ps = CS.pinned_summary(pinned_nbytes)
    stats = None
    f = getattr(torch.cuda, "host_memory_stats", None)
    if f is not None:
        try:
            stats = {k: v for k, v in dict(f()).items() if isinstance(v, (int, float))}
        except Exception as e:                                                # noqa: BLE001 (recorded)
            stats = dict(error=str(e)[:300])
    out = dict(pinned_kv_tensors=ps["tensors"], pinned_kv_pow2_gb=ps["pow2_gib"] * CS.GIB / 1e9,
               torch_host_memory_stats=stats)
    out.update(cgroup_memory())
    return out


def mem_sample(torch, label: str) -> Dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    rss = proc_status().get("VmRSS")
    return dict(label=label, used_gb=(total - free) / 1e9, reserved_gb=torch.cuda.memory_reserved() / 1e9,
                alloc_gb=torch.cuda.memory_allocated() / 1e9, total_gb=total / 1e9,
                host_rss_gb=(rss * 1e3 / 1e9) if rss is not None else None)


def live_rewarm(model, cache, lookahead: int):
    """eqmem_core.rewarm_need on the live state: the last warm-up's shapes are the model's own buffers
    (pooling_buf_all's last dimension = pooled blocks, layer 0's score_buf's last dimension = the score width)."""
    lay = cache.layers[0]
    warm = None
    if model.has_buffers:
        warm = dict(out_len=int(model.pooling_buf_all.shape[-1]), width=int(model.layers[0].score_buf.shape[-1]))
    return EC.rewarm_need(cache.get_seq_length(0), lay.no_compress_k_len, lay.cached_compressed_max_seqlen, warm,
                          lookahead)


def decode_shipped(torch, model, cache, t0, pos0, N: int, samples: List, probe: "EngineProbe",
                   e_prev=None) -> Dict[str, Any]:
    """original / cached: N - 1 greedy decode steps of NOSA's decode_inference; per-step CUDA events; re-warm by
    live_rewarm (lookahead 1); device samples every 16 steps (between steps). v2: the probe's window count is marked
    before every step and after the last, so every decode_update window (one per layer per step, the re-warm steps'
    warm-up path included) and its load count are charged to their own step (eqmem_core.split_step_windows).
    e_prev (the prefill-end event; 4.2): itl_ms = the intervals between consecutive step-end events, the first from
    e_prev (host gaps included)."""
    B = t0.shape[0]
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    tok, pos = t0, pos0.clone()
    seq = [t0]
    evs, rewarms, marks = [], [], []
    model.has_buffers = False
    with torch.inference_mode():
        for it in range(N - 1):
            need, why = live_rewarm(model, cache, 1)
            if need:
                model.has_buffers = False
                rewarms.append(dict(step=it, why=why))
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            marks.append(len(probe.windows))
            e0.record()
            lg = model.decode_inference(tok, cu, pos, cache)
            tok = lg[:, -1, :].argmax(-1, keepdim=True)
            e1.record()
            seq.append(tok)
            pos = pos + 1
            evs.append((e0, e1))
            if it % 16 == 15:
                samples.append(mem_sample(torch, "decode %d" % it))
        marks.append(len(probe.windows))
    torch.cuda.synchronize()
    ms = [a.elapsed_time(b) for a, b in evs]
    itl = None
    if e_prev is not None:
        ends = [e_prev] + [b for _, b in evs]
        itl = [ends[i].elapsed_time(ends[i + 1]) for i in range(len(ends) - 1)]
    steps, steps_error = None, None
    try:
        wms, wl = probe.per_window()
        nL = len(cache.layers)
        rw = set(r["step"] for r in rewarms)
        steps = EC.step_record(ms, EC.split_step_windows(wms, marks, nL), EC.split_step_windows(wl, marks, nL),
                               [1 if i in rw else 0 for i in range(len(ms))], unit="step")
    except ValueError as e:                   # recorded; the cell's metrics stand
        steps_error = str(e)
    return dict(seq=torch.cat(seq, dim=1).cpu(), unit_tokens=[1.0] * len(ms), unit_ms=ms, rewarms=rewarms,
                gen_tokens=N, rounds=None, steps=steps, steps_error=steps_error, itl_ms=itl)


def decode_spec(torch, model, cache, t0, pos0, N: int, cell: EC.Cell, out: Path, probe: "EngineProbe") -> Dict[str, Any]:
    """spec: the R2 pilot's run_spec_arm (LOCKSTEP) committing >= N - 1 tokens after t0, with its re-warm test
    replaced by live_rewarm(lookahead K + 1). v2 per-unit record: a ROUND (its draft + verify + rollback ms; wait =
    the verify's late-fetch gather ms on the main stream (the draft's prefetch runs on a side stream); blocks = the
    round's prefetch + late-fetch counters; tokens = c) or a RE-WARM step (its ms; wait and blocks = the probe's
    windows of that step: only the shipped re-warm steps call decode_update in a spec cell, one window per layer)."""
    os.environ["NOSI_SPEC_OUT"] = str(out / "pilot")                 # the pilot creates its output directory at import
    import spec_loop_pilot as pilot
    pilot.needs_rewarm = lambda m, c: live_rewarm(m, c, cell.K + 1)[0]
    res = pilot.run_spec_arm(model, cache, t0, pos0, cell.K, N - 1, False)
    torch.cuda.synchronize()
    units_t, units_ms, gather = [], [], 0.0
    u_wait, u_blocks, u_rw = [], [], []
    for r in res["rounds"]:
        if r.get("rewarm"):
            units_t.append(1.0)
            units_ms.append(float(r["step_ms"]))
            u_wait.append(None)
            u_blocks.append(None)
            u_rw.append(1)
            continue
        units_t.append(float(r["c"]))
        units_ms.append(sum(r["draft_ms"]) + float(r["verify_ms"] or 0.0) + float(r["rollback_ms"] or 0.0))
        vs = r.get("verify_split") or {}
        g = float(vs.get("gather_ms") or 0.0)
        gather += g
        cn = r.get("counters") or {}
        u_wait.append(g)
        u_blocks.append(int(cn.get("prefetch_blocks", 0)) + int(cn.get("late_fetch_blocks", 0)))
        u_rw.append(0)
    steps, steps_error = None, None
    try:
        wms, wl = probe.per_window()
        nL = len(cache.layers)
        n_rw = sum(u_rw)
        # the probe's windows: the pilot's warm-up step (before the first round; not a round record), then one step per
        # re-warm round, in order
        rw_wait, rw_blocks = EC.rewarm_unit_windows(wms, n_rw, nL), EC.rewarm_unit_windows(wl, n_rw, nL)
        k = 0
        for i, f in enumerate(u_rw):
            if f:
                u_wait[i], u_blocks[i] = rw_wait[k], rw_blocks[k]
                k += 1
        steps = EC.step_record(units_ms, u_wait, u_blocks, u_rw, unit="round", tokens=units_t,
                               wait_label="rounds: the verify's late-fetch gathers (main stream); re-warm steps: their decode_update windows")
    except ValueError as e:
        steps_error = str(e)
    keep = {k: res.get(k) for k in ("K", "U", "W", "R", "tokens", "counters", "wire", "timely_recall", "host_syncs",
                                     "yield_lockstep_tokens", "yield_ragged_projection_tokens", "agg")}
    return dict(seq=res["seq"], unit_tokens=units_t, unit_ms=units_ms, rewarms=[dict(round=r["r"]) for r in res["rounds"]
                                                                                 if r.get("rewarm")],
                gen_tokens=1 + int(res["tokens"]), rounds=keep, spec_gather_ms=gather,
                spec_wire_blocks=int(res["counters"]["prefetch_blocks"]) + int(res["counters"]["late_fetch_blocks"]),
                steps=steps, steps_error=steps_error)


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
            dec = decode_spec(torch, model, cache, t0tok, pos0, a.n_new, cell, out, probe)
        else:
            dec = decode_shipped(torch, model, cache, t0tok, pos0, a.n_new, samples, probe, e_prev=e_pre)
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
        pinned_list = [t.numel() * t.element_size() for lay in cache.layers
                       for t in (lay.cache_engine._k_cpu, lay.cache_engine._v_cpu)]
        pinned = sum(pinned_list)
        rec["host_pinned_gb"] = pinned / 1e9
        # v2: the process's host memory (field names: host.rss_peak_gb, host.ru_maxrss_gb, host.pinned_kv_gb,
        # host.vm_pin_gb, host.rss_samples[].rss_gb) and the per-step timings (steps.json; the summary here)
        import resource
        try:
            rec["host"] = EC.host_record(samples, proc_status(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, pinned)
        except ValueError as e:
            rec["host"] = dict(error=str(e), pinned_kv_gb=pinned / 1e9)
        rec["host"].update(host_extras(torch, pinned_list))                  # 4.2: power-of-two size, cgroup peak
        if dec.get("steps") is not None:
            (out / "steps.json").write_text(json.dumps(dec["steps"], separators=(",", ":")) + "\n")
            rec["steps_summary"] = dec["steps"]["summary"]
        rec["steps_error"] = dec.get("steps_error")
        torch.save(dict(seq=dec["seq"], rows=sched["rows"]), out / "tokens.pt")
        rss_peak = rec["host"].get("rss_peak_gb")
        (out / "fit_row.json").write_text(json.dumps(dict(family=cell.family, cell=cell.name, B=plan["B"], slots=cell.slots,
                                                          alloc_peak_gb=rec["memory"]["alloc_peak_gb"],
                                                          reserved_peak_gb=rec["memory"]["reserved_peak_gb"],
                                                          nonpt_gb=rec["memory"]["nonpt_gb"], S=int(plan["S"]),
                                                          max_gen=int(plan["max_gen"]),
                                                          host_peak_gib=(rss_peak * 1e9 / CS.GIB) if rss_peak is not None else None,
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
    names = [n for n in ("fit_row.json", "result.json", "steps.json", "tokens.pt") if (out / n).exists()]
    (out / "SHA256SUMS").write_text("".join("%s  %s\n" % (hashlib.sha256((out / n).read_bytes()).hexdigest(), n)
                                            for n in names))


# ------------------------------------------------------------------------------------------------------- plan, report
def run_plan(a) -> int:
    fit = load_fit(a)
    out = Path(a.out)
    if out.exists():
        print("eqmem: --out %s exists" % out, file=sys.stderr)
        return 2
    try:
        cells = EC.expand_cells(a.cells)
    except ValueError as e:
        raise Refused(str(e))
    plans = [plan_for(a, c, fit) for c in cells]
    commands = None
    if a.step_nosi or a.step_holder or a.step_run:
        try:
            reps = [int(x) for x in a.step_reps.split(",") if x.strip()]
        except ValueError:
            raise Refused("--step-reps %r" % a.step_reps)
        commands = step_commands(cells, a.budget_gb, a.step_nosi, a.step_holder, a.step_run, reps)
    out.write_text(json.dumps(dict(budget_gb=a.budget_gb, n_new=a.n_new, margin_gb=a.margin_gb, fit=fit.as_dict(),
                                   cells=[c.name for c in cells], plans=plans, commands=commands), indent=1) + "\n")
    for p in plans:
        print("%-14s B %4d (bound %-11s) slots %3d usable %3d predicted used %.1f GB alloc %.1f GB host %.0f GB"
              % (p["cell"], p["B"], p["bound_by"], p["slots"], p["usable"], p["predicted_used_gb"], p["predicted_alloc_gb"],
                 p["predicted_host_gb"]))
    for c in commands or []:
        print(c)
    return 0


def run_report(a) -> int:
    root = Path(a.results)
    out = Path(a.out)
    if out.exists():
        print("eqmem: --out %s exists" % out, file=sys.stderr)
        return 2
    results, dirs, entries = [], {}, []
    for p in sorted(root.glob("*/result.json")):
        r = json.loads(p.read_text())
        entries.append((r.get("cell"), int(r.get("rep", 0)), r.get("status"), (p.parent / "tokens.pt").exists()))
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
    # v2: the agreement reference is the FIRST COMPLETED repetition of the original cell (rep 0 may have failed:
    # h41 original_rep0 was OOM-killed and original_rep1 completed); every other completed (cell, rep) is compared
    ref = EC.agreement_reference(entries)
    agree = {}
    if ref is not None:
        rt = torch.load(dirs[ref] / "tokens.pt")["seq"].tolist()
        for (cell, r_), d in sorted(dirs.items()):
            if (cell, r_) != ref and (d / "tokens.pt").exists():
                agree["%s_rep%d" % (cell, r_)] = EC.prefix_agreement(torch.load(d / "tokens.pt")["seq"].tolist(), rt)
    rep["agreement_reference"] = ("%s_rep%d" % ref) if ref else None
    rep["agreement_vs_original"] = agree
    rep["first_completed_rep"] = EC.first_completed(entries)
    rep["capacity_scan"] = EC.capacity_scan(results)
    rep["steps_summary"] = {"%s_rep%d" % (r["cell"], r.get("rep", 0)): r.get("steps_summary") for r in results}
    rep["host"] = {"%s_rep%d" % (r["cell"], r.get("rep", 0)): r.get("host") for r in results}
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
              "LABEL: the spec family is LOCKSTEP (c = 1 + min_b n_acc per round): a LOWER BOUND of stage 3 until R2-M2",
              "token agreement reference (the first completed original repetition): %s" % rep["agreement_reference"]]
    cs = rep["capacity_scan"]
    if cs["table"]:
        lines += ["", "## capacity scan (MEASURED cells; %s)" % cs["key"], "",
                  "| P | cell | reps | batch | decode tok/s | e2e tok/s | bytes/token |", "|---|---|---|---|---|---|---|"]
        for t in cs["table"]:
            lines.append("| %d | %s | %d | %d | %s | %s | %s |" % (t["P"], t["cell"], t["reps"], t["batch"], _f(t.get(cs["key"]), "%.1f"),
                                                             _f(t["e2e_tok_s"], "%.1f"), _f(t["bytes_per_committed_token"], "%.0f")))
        if cs["best"]:
            lines.append("measured best P %d (%s, %.1f), %s of the scan%s" % (
                cs["best"]["P"], cs["best"]["cell"], cs["best"]["value"], cs["position"],
                ("; %+.1f%% over the original" % (100 * cs["best_vs_original"])) if cs.get("best_vs_original") is not None else ""))
    steps_rows = [(k, s) for k, s in sorted(rep["steps_summary"].items()) if s]
    if steps_rows:
        lines += ["", "## per-step split (MEASURED per unit: CUDA events; steady = units without a re-warm)", "",
                  "| cell | unit | units | re-warm units | steady total ms (median) | steady wait ms (median) | steady rest ms (median) | steady wait share |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, s in steps_rows:
            med = lambda x: _f((x or {}).get("median"), "%.1f")
            lines.append("| %s | %s | %d | %d | %s | %s | %s | %s |" % (k, s["unit"], s["n"], s["rewarm_units"], med(s["steady_total"]),
                                                                    med(s["steady_wait"]), med(s["steady_rest"]),
                                                                    _f(s["steady_wait_share"], "%.3f")))
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
