"""WEIGHTS-VERSUS-KV BATCH SWEEP: HBM traffic and GPU memory of NOSA-8B decode against batch (user authorization
2026-09-23; queued behind the gather-focus job). L = 16128, cache capacity C = 63 (the shipped 64-slot window) and
C = 128 (the fork's physical victim pool, NOSI_POOL_BLOCKS = 65: 63 attended + 65 pool slots, cache_engine.py:71,
:274, :717-842), B = 1, 8, 32, 64, 128, 256 where feasible plus the largest batch that fits.

MODES (per cell, from one matched committed state; teacher-forced PG-19 continuation, verify_alone.py:154-181):
  ordinary        the shipped decode step at its natural misses. C = 63: per gated step, the natural step from the
                  committed state R (light restore, state_snapshot.CounterSnapshot) and repeats from R (exact: the
                  same misses are re-fetched into the same slots; gate = logits torch.equal + equal load count).
                  C = 128: the natural trajectory, one execution per step (no snapshot covers the pool).
  resident        C = 63 only: R with the post-step block map, i.e. every selected block already resident: 0 loads,
                  logits torch.equal the ordinary step (the worker_sweep.py:174-205 light-restore pattern, with the
                  natural step's map instead of a flushed reference). LABEL: "resident target step: the traffic
                  stand-in for resident-only drafting, NOT a draft".
  UNSUPPORTED rows (reported, never substituted): resident at C = 128, resident-only draft and cache-masked draft
                  at both C (see UNSUPPORTED below).

STAGES (the job script runs U, then P):
  U  UNPROFILED: per (C, B) one process: prefill, WARM steps, then timed steps WARM..N-1. Timers: 'gated' = a GPU
     sleep on the stream while the host enqueues the step, clock from the end of the sleep (worker_sweep.py:147-172;
     host enqueue time recorded, gate_valid = the host finished before the sleep did); 'eager' = two CUDA events
     around the enqueue + run (verify_alone.py:496-504; host launch gaps included). Memory: synchronized points,
     allocator stats (allocated / reserved / requested, current and window peak), cudaMemGetInfo device-used,
     nvidia-smi, an INVENTORY of every owned GPU tensor + a gc sweep, deduplicated by untyped storage, the
     allocator snapshot (graph private pools), host pinned bytes and NUMA pages. Peak windows are reset at
     synchronized boundaries and never added. After the timed steps: one torch.profiler(record_shapes) step per
     mode (tensor shapes for the kernel join; never timed).
     LARGEST BATCH: implementation limits are checked first (hbm_sweep_lib.refuse_reasons); a batch predicted to
     exceed GPU memory in the PREFILL phase is first run as a quick fit probe (full-batch allocation + request 0's
     prefill, nosa_llama.py:1043-1060 for t = 0); probes bisect between the largest fit and the smallest failure.
  P  MINIMAL PROFILE (after U): Nsight Compute on a subset, the LAST decode step only, inside
     cudaProfilerStart/Stop windows (gemm_ncu_probe.py:46-49 pattern), --replay-mode kernel --cache-control none
     --clock-control none --nvtx, a single-pass metric set chosen by a pass probe; per-kernel DRAM read/write,
     duration, launch geometry, L2 device-write sectors (write ARRIVALS) and sysmem read sectors when single-pass.
     Windows are delimited by runs of spin_kernel markers (torch.cuda._sleep). Attribution: NVTX module range
     (nosa_llama.py:547-667) or kernel family; analytic bytes in separate columns.

UNSUPPORTED (file:line):
  resident @ C=128: the light restore omits the pool state (_pool_map/_pool_age/_pool_target/_pool_action and the
      host _pool_stamp_base; state_snapshot.py:74-80, cache_engine.py:180, :305-316, :814): a re-run would advance
      the pool LRU stamp and perturb the natural trajectory.
  resident-only draft (both C): no standalone resident-only DRAFT step exists on this base: mech=stale suppresses
      fetches but attends stale occupants unmasked, mech=mask masks but still fetches (avail_policy.py:21-47,
      :577-591); the resident target step is the labelled stand-in at C = 63.
  cache-masked draft @ C=63: NOSI_AVAIL mech=mask fetches every miss physically, adds a per-layer device-to-host
      sync and host loops (avail_policy.py:486-584); its driver is B = 1 only (acceptance_pilot.py:468).
  cache-masked draft @ C=128: NOSI_AVAIL with NOSI_POOL_BLOCKS is refused at construction (cache_engine.py:192-198);
      the acceptance pilot's C = 128 is a simulated LRU on the 64-slot engine (avail_policy.py:168-246).

Env: HBM_MODE (orchestrate | cell | probe | ncu_cell | passprobe | table), HBM_STAGE (U | P), HBM_OUT, HBM_C, HBM_B,
HBM_TAG, HBM_L (16128), HBM_N (63), HBM_WARM (20), HBM_REPS_GATED (2), HBM_REPS_EAGER (1), HBM_REPS_RES_GATED (3),
HBM_SLEEP_MS (50), HBM_CAPTURE (1), HBM_MEM_HISTORY_MAX_B (64), HBM_CS ('63 128'), HBM_BATCHES ('1 8 32 64 128 256'),
HBM_GPU_BUDGET_GB (80), HBM_HOST_BUDGET_GB (480), HBM_HOST_RESERVE_GB (40), HBM_DEADLINE (epoch s, stage end),
HBM_LADDER (1), HBM_LADDER_RES (8), HBM_LADDER_MAX_FULL (2), HBM_P_CELLS, HBM_NCU, HBM_NCU_SETS, HBM_FORCE_PROBES (0),
NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_ATTN_SPLITS (4), NOSI_COMMIT.
"""
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hbm_sweep_lib as HL  # noqa: E402

MODE = os.environ.get("HBM_MODE", "orchestrate")
STAGE = os.environ.get("HBM_STAGE", "U")
OUT = os.path.abspath(os.environ.get("HBM_OUT", "hbm_sweep_out"))
C = int(os.environ.get("HBM_C", "63"))
B = int(os.environ.get("HBM_B", "8"))
L = int(os.environ.get("HBM_L", "16128"))
N = int(os.environ.get("HBM_N", "63"))
WARM = int(os.environ.get("HBM_WARM", "20"))
REPS_GATED = int(os.environ.get("HBM_REPS_GATED", "2"))
REPS_EAGER = int(os.environ.get("HBM_REPS_EAGER", "1"))
REPS_RES_GATED = int(os.environ.get("HBM_REPS_RES_GATED", "3"))
SLEEP_MS = float(os.environ.get("HBM_SLEEP_MS", "50"))
CAPTURE = os.environ.get("HBM_CAPTURE", "1") == "1"
MEM_HISTORY_MAX_B = int(os.environ.get("HBM_MEM_HISTORY_MAX_B", "64"))
P_MULTIPASS_MAX_B = int(os.environ.get("HBM_P_MULTIPASS_MAX_B", "64"))   # stage P cap when no metric set is single-pass
TAG = os.environ.get("HBM_TAG") or ("c%d_b%d" % (C, B))
CELL_DIR = os.path.join(OUT, TAG)
P_POOL = C - 63
NCU_WINDOW_MODES = ("ordinary", "resident") if C == 63 else ("ordinary",)
UNSUPPORTED = {
    (128, "resident"): "light restore omits the pool state (_pool_map/_pool_age/_pool_target/_pool_action, host _pool_stamp_base; state_snapshot.py:74-80, cache_engine.py:180,305-316,814): a re-run would advance the pool LRU stamp",
    (63, "resident_only_draft"): "no standalone resident-only DRAFT step on this base: mech=stale attends stale occupants unmasked, mech=mask still fetches (avail_policy.py:21-47,577-591); stand-in = the resident target step",
    (128, "resident_only_draft"): "no resident-only draft path with the pool: every draft path refuses NOSI_POOL_BLOCKS (cache_engine.py:192-198; verify_alone.py:452-453)",
    (63, "cache_masked_draft"): "NOSI_AVAIL mech=mask fetches every miss physically and adds a per-layer D2H sync + host loops (avail_policy.py:486-584); driver is B = 1 only (acceptance_pilot.py:468): traffic = ordinary + mask overhead, timing not an eager step",
    (128, "cache_masked_draft"): "NOSI_AVAIL with NOSI_POOL_BLOCKS refused at construction (cache_engine.py:192-198); the pilot's C = 128 is a simulated LRU on the 64-slot engine (avail_policy.py:168-246) = a silent proxy, not substituted",
}
MODE_LABELS = {"ordinary": "ordinary decode (shipped step, natural misses)",
               "resident": "resident target step: 0 loads, all selected blocks already resident; traffic stand-in for resident-only drafting, NOT a draft"}

if MODE in ("cell", "probe", "ncu_cell"):
    # verify_alone reads its knobs at import (verify_alone.py:88-104): map ours onto them (worker_sweep.py:29-34 pattern)
    os.environ["NOSI_ALONE_MODE"] = "points"
    os.environ["NOSI_ALONE_L"] = str(L)
    os.environ["NOSI_ALONE_N"] = str(N)
    os.environ["NOSI_ALONE_WARM"] = str(min(max(WARM, 1), N - 1))
    os.environ["NOSI_ALONE_B"] = str(B)
    os.environ["NOSI_ALONE_OUT"] = CELL_DIR
    os.environ["NOSI_ALONE_DOCS"] = "1"
    os.environ["NOSI_ALONE_D"] = "0"


def log(msg):
    print("[hbm] %s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# =====================================================================================================================
# GPU side
# =====================================================================================================================
def _torch():
    import torch
    return torch


def cuda_mem(torch):
    """Synchronized allocator stats + cudaMemGetInfo (device-used includes the context and everything outside the
    caching allocator)."""
    torch.cuda.synchronize()
    st = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=int(st.get("allocated_bytes.all.current", 0)), reserved=int(st.get("reserved_bytes.all.current", 0)),
                requested=st.get("requested_bytes.all.current"), peak_allocated=int(st.get("allocated_bytes.all.peak", 0)),
                peak_reserved=int(st.get("reserved_bytes.all.peak", 0)), peak_requested=st.get("requested_bytes.all.peak"),
                device_used=int(total - free), device_total=int(total), num_alloc_retries=st.get("num_alloc_retries"), num_ooms=st.get("num_ooms"))


def smi():
    """nvidia-smi memory / clock / power of the visible GPU (raw strings kept)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,clocks.sm,power.draw,clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        line = (r.stdout or "").strip().splitlines()[0]
        f = [x.strip() for x in line.split(",")]
        return dict(used_mib=HL.to_number(f[0]), total_mib=HL.to_number(f[1]), sm_mhz=HL.to_number(f[2]), power_w=HL.to_number(f[3]), throttle=f[4] if len(f) > 4 else "", raw=line)
    except Exception as e:
        return dict(error=str(e)[:200])


def nvml_sample(torch):
    out = {}
    try:
        out["clock_mhz"] = torch.cuda.clock_rate()
    except Exception:
        pass
    try:
        out["power_w"] = torch.cuda.power_draw() / 1000.0
    except Exception:
        pass
    return out


class Recorder:
    """Memory points + windows (HL.MemLedger) with the raw rows flushed to JSONL as they happen."""

    def __init__(self, torch, cell_dir):
        self.t = torch
        self.dir = cell_dir
        self.ledger = HL.MemLedger()
        os.makedirs(cell_dir, exist_ok=True)
        for f in ("memory_points.jsonl", "memory_windows.jsonl", "logits_sha.jsonl", "steps.jsonl"):
            if os.path.exists(os.path.join(cell_dir, f)):
                os.remove(os.path.join(cell_dir, f))          # a cell directory holds ONE run

    def point(self, name, **extra):
        m = cuda_mem(self.t)
        s = smi()
        row = self.ledger.point(name, m["allocated"], m["reserved"], m["device_used"], m["device_total"], m["requested"], s.get("used_mib"),
                                peak_allocated_since_reset=m["peak_allocated"], peak_reserved_since_reset=m["peak_reserved"],
                                peak_requested_since_reset=m["peak_requested"], num_alloc_retries=m["num_alloc_retries"], num_ooms=m["num_ooms"],
                                smi=s, t=time.time(), **extra)
        HL.append_jsonl(os.path.join(self.dir, "memory_points.jsonl"), row)
        return row

    def reset(self):
        self.t.cuda.synchronize()
        self.t.cuda.reset_peak_memory_stats()

    def window(self, name, **extra):
        m = cuda_mem(self.t)
        row = self.ledger.add_window(name, m["peak_allocated"], m["peak_reserved"], allocated_after=m["allocated"], reserved_after=m["reserved"],
                                     device_used_after=m["device_used"], peak_requested=m["peak_requested"], **extra)
        HL.append_jsonl(os.path.join(self.dir, "memory_windows.jsonl"), row)
        return m


def load_model(path):
    """The model, once per process. Unlike verify_alone.load_model (verify_alone.py:448-461) the pool is ALLOWED
    (C = 128 ordinary decode); the knobs the engine read at import must match this cell."""
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    from nosi import avail_policy as _ap
    from nosi import transfer_trace as _tt
    if _ce.POOL_BLOCKS != P_POOL:
        raise SystemExit("REFUSED: the engine read NOSI_POOL_BLOCKS=%d at import, the cell is C=%d (P=%d)" % (_ce.POOL_BLOCKS, C, P_POOL))
    if _ce.VERIFY_ROUND_SLOTS != 0:
        raise SystemExit("REFUSED: NOSI_VERIFY_ROUND_SLOTS must be 0")
    if _ap.SPEC not in ("", "0", "off") or _ap.POLICY is not None:
        raise SystemExit("REFUSED: NOSI_AVAIL must be unset (the availability hook is not a mode of this sweep)")
    if _tt.MODE not in ("0", "", "off") or _tt.TRACE is not None:
        raise SystemExit("REFUSED: NOSI_TRANSFER_TRACE must be unset/0 (the trace is instrumentation)")
    splits = int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0)
    log("model load: C=%d (pool %d) B=%d L=%d N=%d warm=%d ATTN_SPLITS=%d KV_BIAS_SCALE=%s" % (C, P_POOL, B, L, N, WARM, splits, _ce.KV_BIAS_SCALE))
    return Llama(model_name=path, device="cuda", offload=True)


def owned_entries(torch, model, cache, snap, driver):
    """(kind, owner, attr, tensor) of every GPU tensor the driver can name, in the counting order: weights,
    engine window before its views, layer tables before the decode buffers that alias them, snapshot, driver."""
    ents = []

    def add(kind, owner, attr, t):
        if torch.is_tensor(t) and t.is_cuda:
            ents.append((kind, owner, attr, t))
    for a in ("embed_tokens", "lm_head", "norm_weight"):
        add("model", "model", a, getattr(model, a, None))
    for i, lay in enumerate(model.layers):
        for a in ("wqkv", "wo", "gate_up_proj", "down_proj", "input_layernorm_weight", "post_attention_layernorm_weight", "A"):
            add("layer", "layer%d" % i, a, getattr(lay, a, None))
        d = getattr(lay, "delta", None)
        if d is not None and hasattr(d, "named_parameters"):
            for pn, p in d.named_parameters():
                add("layer", "layer%d" % i, "delta." + pn, p)
    add("model", "model", "cos_sin_cache", getattr(model, "cos_sin_cache", None))
    if cache is not None:
        for i, cl in enumerate(cache.layers):
            e = cl.cache_engine
            for a in HL.ENGINE_WINDOW + HL.ENGINE_META + HL.ENGINE_VIEWS:
                add("engine", "engine%d" % i, a, getattr(e, a, None))
            for a in HL.LAYER_TABLES:
                add("cache_layer", "cache_layer%d" % i, a, getattr(cl, a, None))
    for i, lay in enumerate(model.layers):
        for a in HL.LAYER_BUFFERS:
            add("layer_buffer", "layer%d" % i, a, getattr(lay, a, None))
    for a in HL.MODEL_BUFFERS:
        add("model_buffer", "model", a, getattr(model, a, None))
    if snap is not None and getattr(snap, "layers", None):
        for i, slot in enumerate(snap.layers):
            for part in ("engine", "layer"):
                for k, v in slot[part].items():
                    add("snapshot", "snap%d.%s" % (i, part), k, v)
    for k, v in (driver or {}).items():
        add("driver", "driver", k, v)
    return ents


def gc_extras(torch, known):
    """CUDA tensors reachable by the garbage collector that no named owner holds (label: python_visible_unowned)."""
    out, seen = [], set(known)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                k = HL.storage_key(obj)
                if k not in seen:
                    seen.add(k)
                    out.append(("gc", "gc", "%s%s" % (str(obj.dtype).replace("torch.", ""), list(obj.shape)), obj))
        except Exception:
            continue
    return out


def inventory(torch, rec, name, model, cache, snap, driver):
    """Inventory rows + allocator snapshot + breakdown at a synchronized point; files <name>_*.csv/json."""
    torch.cuda.synchronize()
    ents = owned_entries(torch, model, cache, snap, driver)
    keys = set()
    for _, _, _, t in ents:
        try:
            keys.add(HL.storage_key(t))
        except Exception:
            pass
    ents += gc_extras(torch, keys)
    slots = 64 + P_POOL
    rows = HL.build_inventory(ents, slots_total=slots, topk=64)
    try:
        segs = HL.summarize_segments(torch.cuda.memory._snapshot())
    except Exception as e:
        segs = dict(pools={}, segments=[], keys=[], error=str(e)[:300])
    vis = [(int(r["storage_ptr"], 16), r["storage_bytes"]) for r in rows if r["counted_bytes"] and not r["name"].endswith(("[tail]", "[pool]"))]
    gp = HL.graph_pool_bytes(segs, vis)
    m = cuda_mem(torch)
    br = HL.memory_breakdown(rows, m["allocated"], m["reserved"], m["device_used"], m["device_total"], m["requested"], gp["graph_pool_bytes"])
    for r in br:
        r.update(point=name, C=C, B=B)
    HL.write_csv(os.path.join(CELL_DIR, "inventory_%s.csv" % name), rows)
    HL.write_csv(os.path.join(CELL_DIR, "breakdown_%s.csv" % name), br)
    HL.write_json(os.path.join(CELL_DIR, "segments_%s.json" % name), dict(segments=segs, graph_pools=gp, stats=m))
    tot = HL.breakdown_totals(br)
    log("inventory %s: device used %.2f GB = %s" % (name, m["device_used"] / 1e9, ", ".join("%s %.2f" % (k, v / 1e9) for k, v in tot.items())))
    return rows, br, m


def host_record(torch, cache):
    """Pinned host K/V: requested bytes, the power-of-two allocation estimate, NUMA pages of all 64 buffers
    (numa_maps.range_pages, every VMA a buffer touches), process status lines."""
    import numa_maps
    out = dict(buffers=[], pinned_estimate=HL.pinned_host_bytes(B, L))
    try:
        vmas = numa_maps.parse()
    except Exception as e:
        vmas, out["numa_error"] = [], str(e)[:200]
    for i, cl in enumerate(cache.layers):
        e = cl.cache_engine
        for name in ("_k_cpu", "_v_cpu"):
            t = getattr(e, name, None)
            if t is None:
                continue
            nb = t.numel() * t.element_size()
            rec = numa_maps.range_pages(vmas, t.data_ptr(), nb) if vmas else None
            out["buffers"].append(dict(layer=i, name=name, nbytes=nb, pinned=bool(t.is_pinned()), node=numa_maps.node_of(rec), numa=rec))
    try:
        out["status"] = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM", "VmLck", "VmPin", "VmSize"))}
    except OSError:
        pass
    try:
        out["cpu_affinity"] = sorted(os.sched_getaffinity(0))[:1] + sorted(os.sched_getaffinity(0))[-1:] + [len(os.sched_getaffinity(0))]
    except Exception:
        pass
    return out


def make_timers(torch):
    sleep_cycles = int(SLEEP_MS * 1e-3 * 1.41e9)

    def gated(fn):
        """worker_sweep.py:147-172 bracket(fn_main, None): GPU sleep -> gate -> t0 -> the step enqueued while the GPU
        sleeps -> tm. host_enqueue_ms = the host's enqueue time of the step; gate_valid = it finished before the
        sleep ended (otherwise device_ms may include host gaps)."""
        torch.cuda.synchronize()
        ev = {k: torch.cuda.Event(enable_timing=True) for k in ("pre", "gate", "t0", "tm")}
        ev["pre"].record()
        torch.cuda._sleep(sleep_cycles)
        ev["gate"].record()
        ev["t0"].record()
        h0 = time.perf_counter()
        out = fn()
        host_ms = 1000.0 * (time.perf_counter() - h0)
        ev["tm"].record()
        torch.cuda.synchronize()
        sl = ev["pre"].elapsed_time(ev["gate"])
        return out, dict(device_ms=ev["t0"].elapsed_time(ev["tm"]), host_enqueue_ms=host_ms, sleep_ms=sl,
                         host_lag_ms=ev["gate"].elapsed_time(ev["t0"]), gate_valid=bool(host_ms < 0.9 * sl))

    def eager(fn):
        """verify_alone.py:496-504 _timed_call: sync, event, enqueue + run, event, sync (host launch gaps included)."""
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        h0 = time.perf_counter()
        out = fn()
        host_ms = 1000.0 * (time.perf_counter() - h0)
        e1.record()
        torch.cuda.synchronize()
        return out, dict(device_ms=e0.elapsed_time(e1), host_enqueue_ms=host_ms, sleep_ms=0.0, host_lag_ms=0.0, gate_valid=None)
    return gated, eager


def logits_sha(lg):
    return hashlib.sha256(lg.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]   # verify_alone.py:138-139


def prefill_first_request(torch, model, ids_row, total_bsz, cache):
    """Request 0's prefill at the FULL batch's allocation (nosa_llama.py:1043-1060 with t = 0, T = 1): every
    full-batch GPU tensor (cache_engine.py:263-297, :951, :979, :1046-1048, :1086) and the host pinned cache are
    allocated, and the single-request prefill transient is live, i.e. the prefill-phase peak."""
    import torch.nn.functional as F
    x = ids_row[:L].view(1, L).to("cuda")
    pos = torch.arange(0, L, device="cuda", dtype=torch.long).unsqueeze(0)
    hs = F.embedding(x, model.embed_tokens)
    cu = torch.arange(0, L + 1, L, dtype=torch.int, device="cuda")
    for idx in range(model.num_layers):
        hs = model.layers[idx].prefill_forward(hs, pos, model.cos_sin_cache, cu, L, cache, total_bsz=total_bsz, current_batch_pos=0)
    torch.cuda.synchronize()
    return hs


class CellContext:
    """Shared by 'cell' and 'ncu_cell': load, prefill, warm steps, counters."""

    def __init__(self, torch, rec, ph):
        self.t = torch
        self.rec = rec
        self.ph = ph

    def setup(self):
        torch, rec = self.t, self.rec
        import verify_alone as VA
        self.VA = VA
        VA.check_budget()
        path = os.environ["NOSI_MODEL_PATH"]
        self.ph["phase"] = "corpus"
        corpus = VA.load_corpus(path, B)
        self.ids, self.docs, self.distinct = VA.pick_batch(corpus, B, 0)
        del corpus
        self.ph["phase"] = "load"
        rec.reset()
        self.model = load_model(path)
        rec.window("load")
        rec.point("after_load")
        self.ph["phase"] = "prefill"
        rec.reset()
        t0 = time.time()
        model, cache, logits, position_ids, forced, meta = VA._setup(self.model, self.ids)
        self.prefill_s = time.time() - t0
        del logits
        self.cache, self.forced, self.meta = cache, forced, meta
        self.position_ids = position_ids[:, -1:] + 1
        del position_ids
        gc.collect()
        rec.window("prefill", seconds=self.prefill_s)
        rec.point("after_prefill")
        self.engines = [lay.cache_engine for lay in cache.layers]
        self.cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
        self.ph["phase"] = "warm"

    def step_fn(self, tok):
        return lambda: self.model.decode_inference(tok, self.cu, self.position_ids, self.cache)

    def counts(self):
        torch = self.t
        loads = torch.stack([(e._load_mask >= 0).sum() for e in self.engines]).sum().view(1)
        if P_POOL > 0:
            acts = torch.stack([torch.bincount(e._pool_action.flatten().to(torch.int64), minlength=4)[:4] for e in self.engines]).sum(0)
            v = torch.cat([loads, acts]).tolist()
            return dict(loads=int(v[0]), pool_none=int(v[1]), pool_swap=int(v[2]), pool_move_in=int(v[3]), pool_move_out=int(v[4]))
        return dict(loads=int(loads.item()), pool_none=0, pool_swap=0, pool_move_in=0, pool_move_out=0)

    def ck(self):
        return int(self.cache.layers[0].compress_k_cache_varlen.shape[1])

    def tail_rows(self):
        return int(self.engines[0]._tail_block_len_on_gpu)

    def warm_steps(self, record_row=None):
        torch, rec = self.t, self.rec
        for it in range(WARM):
            tok = self.forced[:, it:it + 1]
            rec.reset()
            ck0 = self.ck()
            self.model.decode_inference(tok, self.cu, self.position_ids, self.cache, warmup=(it == 0))
            torch.cuda.synchronize()
            rec.window("warmup_step0" if it == 0 else "warm_steps")
            if record_row is not None:
                record_row(dict(step=it, mode="warm", rep=0, timer="none", timed=False, compress_keys=self.ck(), burst=self.ck() != ck0,
                                tail_rows=self.tail_rows(), **self.counts()))
            if it == 0:
                rec.point("after_warmup_step0")
            self.position_ids = self.position_ids + 1
        rec.point("after_warm_steps")


def snapshot_post_step_map(torch, snap, engines):
    """Point the light snapshot at the POST-step block map (worker_sweep.py:194-197), so that the next restore gives
    the committed state with every selected block resident (the resident target step). INSIDE inference_mode: the
    snapshot's saved tensors are clones made under CounterSnapshot.take's @torch.inference_mode
    (state_snapshot.py:166), i.e. inference tensors, and an in-place copy_ on an inference tensor outside
    inference_mode raises RuntimeError (job 2174640; worker_sweep.run is @torch.inference_mode, worker_sweep.py:81)."""
    with torch.inference_mode():
        for i, e in enumerate(engines):
            slot = snap.layers[i]["engine"]
            for name in ("_block_map", "_new_block_map_buf"):
                slot[name].copy_(getattr(e, name))


def write_result(status, cls="", detail="", **extra):
    HL.write_json(os.path.join(CELL_DIR, "result.json"), dict(C=C, B=B, L=L, N=N, warm=WARM, mode=MODE, tag=TAG, status=status, cls=cls, detail=detail,
                                                              nosi_commit=os.environ.get("NOSI_COMMIT"), t=time.time(), **extra))


def run_guarded(fn):
    """Run a GPU mode; every failure is classified into result.json before the process exits."""
    os.makedirs(CELL_DIR, exist_ok=True)
    phase_holder = {}
    try:
        rc = fn(phase_holder)
        return rc
    except SystemExit:
        raise
    except BaseException as e:
        tb = traceback.format_exc()
        status, detail = HL.classify_failure(1, tb)
        extra = {}
        try:
            torch = _torch()
            extra["mem_at_failure"] = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                                           max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved())
            f, t = torch.cuda.mem_get_info()
            extra["mem_at_failure"].update(device_used=t - f, device_total=t)
        except Exception:
            pass
        write_result(status, status, detail, phase=phase_holder.get("phase"), traceback=tb[-4000:], **extra)
        print(tb, flush=True)
        log("FAILED (%s) in phase %s: %s" % (status, phase_holder.get("phase"), detail))
        return 10 if status == "CUDA_OOM" else (11 if status.startswith("IMPLEMENTATION_LIMIT") else 12)


# ------------------------------------------------------------------ stage U cell
def mode_cell(ph):
    torch = _torch()
    torch.cuda.init()
    from nosi import state_snapshot as ss
    rec = Recorder(torch, CELL_DIR)
    rec.point("process_start")
    ctx = CellContext(torch, rec, ph)
    ctx.setup()
    gated, eager = make_timers(torch)
    steps_path = os.path.join(CELL_DIR, "steps.jsonl")
    if os.path.exists(steps_path):
        os.remove(steps_path)
    fails = []

    def record_row(r):
        r.update(C=C, B=B)
        HL.append_jsonl(steps_path, r)
    ph["phase"] = "warm"
    ctx.warm_steps(record_row)
    trans = ss.transient_ids(ctx.model)
    snap = ss.CounterSnapshot(ctx.cache) if C == 63 else None
    timers = dict(gated=gated, eager=eager)

    def timed(tok, timer, mode, it, rep, first_exec, ref=None, expect_loads=None):
        rec.reset()
        ck0 = ctx.ck()
        lg, t = timers[timer](ctx.step_fn(tok))
        m = rec.window("decode:%s:step%d" % (mode, it), rep=rep)
        cnt = ctx.counts()
        row = dict(step=it, mode=mode, rep=rep, timer=timer, timed=True, first_exec=first_exec, compress_keys=ctx.ck(), burst=ctx.ck() != ck0,
                   tail_rows=ctx.tail_rows(), step_peak_allocated=m["peak_allocated"], step_peak_reserved=m["peak_reserved"], allocated_after=m["allocated"], **t, **cnt)
        row.update(nvml_sample(torch))
        if ref is not None:
            row["logits_equal"] = bool(torch.equal(lg, ref))
            row["loads_ok"] = (cnt["loads"] == expect_loads) if expect_loads is not None else None
            if not row["logits_equal"] or row["loads_ok"] is False:
                fails.append(dict(step=it, mode=mode, rep=rep, logits_equal=row["logits_equal"], loads=cnt["loads"], expected=expect_loads))
                log("GATE FAIL step %d %s rep %d: logits_equal=%s loads=%d expected=%s" % (it, mode, rep, row["logits_equal"], cnt["loads"], expect_loads))
        record_row(row)
        return lg, row

    def restore():
        rec.reset()
        snap.restore()
        ss.assert_transients_intact(ctx.model, trans)
        rec.window("restore")

    def capture(tok, mode, it):
        """One torch.profiler(record_shapes=True) step (never timed) + an allocator history dump at small B."""
        from torch.profiler import ProfilerActivity, profile, record_function
        hist = B <= MEM_HISTORY_MAX_B
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):   # kineto's first session is a warm-up (verify_alone.py:769-770)
            (torch.zeros(1, device="cuda") + 1).sum().item()
        if hist:
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=200000)
        rec.reset()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("decode_call"):
                torch.cuda.nvtx.range_push(HL.STEP_RANGE_PREFIX + mode)
                lg = ctx.step_fn(tok)()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
        m = rec.window("capture:%s" % mode)
        if hist:
            try:
                torch.cuda.memory._dump_snapshot(os.path.join(CELL_DIR, "memhist_%s_step%d.pickle" % (mode, it)))
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
        tmp = os.path.join(CELL_DIR, "profile_%s_step%d.json" % (mode, it))
        prof.export_chrome_trace(tmp)
        import gzip
        with open(tmp, "rb") as fi, gzip.open(tmp + ".gz", "wb") as fo:
            fo.write(fi.read())
        os.remove(tmp)
        cnt = ctx.counts()
        record_row(dict(step=it, mode=mode, rep=-1, timer="profiler", timed=False, capture=True, compress_keys=ctx.ck(), tail_rows=ctx.tail_rows(),
                        step_peak_allocated=m["peak_allocated"], trace="profile_%s_step%d.json.gz" % (mode, it), memhist=hist, **cnt))
        return lg

    ph["phase"] = "decode"
    nat_reps = ["gated"] * REPS_GATED + ["eager"] * REPS_EAGER
    res_reps = ["gated"] * REPS_RES_GATED + ["eager"] * REPS_EAGER
    t_dec = time.time()
    for it in range(WARM, N):
        tok = ctx.forced[:, it:it + 1]
        last = it == N - 1
        first_timer = "gated" if it % 2 == 0 else "eager"
        if C == 63:
            snap.take()
            lg_nat, r0 = timed(tok, first_timer, "ordinary", it, 0, True)
            loads0 = r0["loads"]
            HL.append_jsonl(os.path.join(CELL_DIR, "logits_sha.jsonl"), dict(C=C, B=B, step=it, mode="ordinary", sha=logits_sha(lg_nat)))
            for rep, tm in enumerate(nat_reps, start=1):
                restore()
                timed(tok, tm, "ordinary", it, rep, False, ref=lg_nat, expect_loads=loads0)
            if last and CAPTURE:
                restore()
                capture(tok, "ordinary", it)
            snapshot_post_step_map(torch, snap, ctx.engines)   # worker_sweep.py:194-197: the light restore returns to the POST-step map
            for rep, tm in enumerate(res_reps):
                restore()
                timed(tok, tm, "resident", it, rep, False, ref=lg_nat, expect_loads=0)
            if last and CAPTURE:
                restore()
                capture(tok, "resident", it)
            del lg_nat
        else:
            if last and CAPTURE:
                lg = capture(tok, "ordinary", it)
            else:
                lg, r0 = timed(tok, first_timer, "ordinary", it, 0, True)
            HL.append_jsonl(os.path.join(CELL_DIR, "logits_sha.jsonl"), dict(C=C, B=B, step=it, mode="ordinary", sha=logits_sha(lg)))
            del lg
        ctx.position_ids = ctx.position_ids + 1
        if it in (WARM, WARM + 1) or it % 10 == 0 or last:
            log("step %d done (%.0f s into decode, gate fails %d)" % (it, time.time() - t_dec, len(fails)))
    ph["phase"] = "inventory"
    driver = dict(forced=ctx.forced, cu=ctx.cu, position_ids=ctx.position_ids)
    rows, br, m = inventory(torch, rec, "steady", ctx.model, ctx.cache, snap, driver)
    rec.point("steady_end")
    host = host_record(torch, ctx.cache)
    HL.write_json(os.path.join(CELL_DIR, "host.json"), host)
    peaks = rec.ledger.peak_summary()
    fam = {}
    for w in rec.ledger.windows:
        f = HL.window_family(w["window"])
        fam[f] = max(fam.get(f, 0), w["peak_allocated"])
    HL.write_csv(os.path.join(CELL_DIR, "memory_points.csv"), rec.ledger.points)
    HL.write_csv(os.path.join(CELL_DIR, "memory_windows.csv"), rec.ledger.windows)
    steps = HL.read_jsonl(steps_path)
    HL.write_csv(os.path.join(CELL_DIR, "steps.csv"), steps)
    summ = {}
    for mode in ("ordinary", "resident"):
        for timer in ("gated", "eager"):
            xs = [r["device_ms"] for r in steps if r.get("timed") and r["mode"] == mode and r["timer"] == timer and (timer != "gated" or r.get("gate_valid"))]
            if xs:
                s = HL.step_stats(xs)
                s["tokens_per_s"] = HL.tokens_per_s(B, s["median"])
                summ["%s_%s" % (mode, timer)] = s
    status = "OK" if not fails else "GATE_FAIL"
    unsupported = [dict(C=C, mode=k[1], reason=v) for k, v in UNSUPPORTED.items() if k[0] == C]
    write_result(status, "" if not fails else "GATE_FAIL", "%d gate failures" % len(fails), gate_failures=fails, step_summary=summ,
                 peaks=peaks, window_family_peaks=fam, prefill_s=ctx.prefill_s, docs=ctx.docs[:8], distinct_docs=ctx.distinct,
                 meta=ctx.meta, predicted=HL.predict_gpu_peak_gb(C, B, light_restore=(C == 63)), unsupported=unsupported,
                 modes={k: MODE_LABELS[k] for k in (("ordinary", "resident") if C == 63 else ("ordinary",))},
                 settings=dict(sleep_ms=SLEEP_MS, reps_gated=REPS_GATED, reps_eager=REPS_EAGER, reps_res_gated=REPS_RES_GATED, warm=WARM, N=N,
                               attn_splits=os.environ.get("NOSI_ATTN_SPLITS"), alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF")))
    log("cell C=%d B=%d %s: %s" % (C, B, status, json.dumps({k: round(v["median"], 2) for k, v in summ.items()})))
    return 0 if not fails else 1


# ------------------------------------------------------------------ quick fit probe
def mode_probe(ph):
    torch = _torch()
    torch.cuda.init()
    rec = Recorder(torch, CELL_DIR)
    rec.point("process_start")
    import verify_alone as VA
    from nosi.cache_engine import InfLLMv2Cache
    path = os.environ["NOSI_MODEL_PATH"]
    ph["phase"] = "corpus"
    ids, _ = VA.load_corpus(path, 1)
    ph["phase"] = "load"
    model = load_model(path)
    rec.point("after_load")
    ph["phase"] = "prefill_request0"
    rec.reset()
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    model.has_buffers = False
    t0 = time.time()
    prefill_first_request(torch, model, ids[0], B, cache)
    m = rec.window("prefill_request0", seconds=time.time() - t0)
    rec.point("after_prefill_request0")
    pred = HL.predict_gpu_peak_gb(C, B, light_restore=False)
    write_result("OK", "FIT_PREFILL_PHASE", "the full-batch allocation + request 0's prefill fit; the decode phase is not tested by a probe",
                 probe=True, peak_allocated=m["peak_allocated"], peak_reserved=m["peak_reserved"], device_used=m["device_used"], predicted=pred,
                 host_pinned=HL.pinned_host_bytes(B, L))
    log("probe C=%d B=%d fits the prefill phase: peak allocated %.2f GB, reserved %.2f GB, device used %.2f GB (predicted %.2f)"
        % (C, B, m["peak_allocated"] / 1e9, m["peak_reserved"] / 1e9, m["device_used"] / 1e9, pred["prefill_gb"]))
    return 0


# ------------------------------------------------------------------ stage P cell (runs under ncu)
def mode_ncu_cell(ph):
    torch = _torch()
    torch.cuda.init()
    from nosi import state_snapshot as ss
    rec = Recorder(torch, CELL_DIR)
    rec.point("process_start")
    ctx = CellContext(torch, rec, ph)
    ctx.setup()
    ph["phase"] = "warm"
    ctx.warm_steps(None)
    cudart = torch.cuda.cudart()
    windows = []
    trans = ss.transient_ids(ctx.model)
    ph["phase"] = "decode"
    for it in range(WARM, N - 1):                                    # untimed, unprofiled natural steps up to the last one
        ctx.model.decode_inference(ctx.forced[:, it:it + 1], ctx.cu, ctx.position_ids, ctx.cache)
        ctx.position_ids = ctx.position_ids + 1
    torch.cuda.synchronize()
    it = N - 1
    tok = ctx.forced[:, it:it + 1]
    snap = ss.CounterSnapshot(ctx.cache) if C == 63 else None
    if snap is not None:
        snap.take()

    def window(k, mode):
        rec.reset()
        ck0 = ctx.ck()
        torch.cuda.synchronize()
        cudart.cudaProfilerStart()
        for _ in range(k):                                              # k spin_kernel markers open window k
            torch.cuda._sleep(1000)
        torch.cuda.nvtx.range_push(HL.STEP_RANGE_PREFIX + mode)
        lg = ctx.step_fn(tok)()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        m = rec.window("ncu:%s" % mode)
        w = dict(window=k, markers=k, mode=mode, step=it, compress_keys=ctx.ck(), burst=ctx.ck() != ck0, tail_rows=ctx.tail_rows(),
                 peak_allocated=m["peak_allocated"], device_used=m["device_used"], **ctx.counts())
        windows.append(w)
        return lg
    ph["phase"] = "ncu_windows"
    lg_nat = window(1, "ordinary")
    windows[-1]["logits_sha"] = logits_sha(lg_nat)
    if C == 63:
        snapshot_post_step_map(torch, snap, ctx.engines)
        snap.restore()
        ss.assert_transients_intact(ctx.model, trans)
        lg_res = window(2, "resident")
        windows[-1]["logits_equal_ordinary"] = bool(torch.equal(lg_res, lg_nat))
        windows[-1]["loads_ok"] = windows[-1]["loads"] == 0
    rec.point("after_ncu_windows")
    HL.write_json(os.path.join(CELL_DIR, "windows.json"), dict(C=C, B=B, L=L, N=N, step=it, windows=windows, marker=HL.MARKER_KERNEL,
                                                                 modes={w["mode"]: MODE_LABELS[w["mode"]] for w in windows}))
    HL.write_csv(os.path.join(CELL_DIR, "memory_points.csv"), rec.ledger.points)
    HL.write_csv(os.path.join(CELL_DIR, "memory_windows.csv"), rec.ledger.windows)
    bad = [w for w in windows if w["mode"] == "resident" and not (w.get("logits_equal_ordinary") and w.get("loads_ok"))]
    write_result("OK" if not bad else "GATE_FAIL", "" if not bad else "GATE_FAIL", "profiled windows %d" % len(windows), windows=windows)
    return 0 if not bad else 1


# ------------------------------------------------------------------ pass-count probe (runs under ncu, no model)
def mode_passprobe(ph):
    """A GEMM, an elementwise kernel and the engine's Triton gather reading pinned host memory, inside one
    cudaProfilerStart/Stop window: ncu's '- N passes' lines tell whether a metric set is single-pass."""
    torch = _torch()
    import torch.nn.functional as F
    import importlib.util
    spec = importlib.util.spec_from_file_location("hbm_flash_h2d_mask", os.path.join(HERE, "..", "..", "nosi", "nosi", "flash_cache_engine", "flash_h2d_mask.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flash_h2d_from_mask = mod.flash_h2d_from_mask
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(4608, 4096, dtype=torch.bfloat16, device="cuda")
    host = torch.randn(1, 64 * 64, 2, 128, dtype=torch.bfloat16).pin_memory()
    gpu = torch.zeros(1, 64 * 64, 2, 128, dtype=torch.bfloat16, device="cuda")
    ids = torch.full((2, 1, 64), -1, dtype=torch.int64, device="cuda")
    ids[:, :, :4] = torch.arange(4, device="cuda")
    for _ in range(2):
        F.linear(x, w)
        flash_h2d_from_mask(gpu, host, ids, 64)
        (gpu * 2).sum()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    torch.cuda._sleep(1000)
    F.linear(x, w)
    flash_h2d_from_mask(gpu, host, ids, 64)
    (gpu * 2).sum()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    ok = torch.equal(gpu[0, :256].cpu(), host[0, :256])
    write_result("OK" if ok else "FAIL", "", "gather copied correctly: %s" % ok)
    return 0 if ok else 1


# =====================================================================================================================
# orchestration (no CUDA in this process)
# =====================================================================================================================
def _env_list(name, default):
    return [int(x) for x in os.environ.get(name, default).split()]


class Orchestrator:
    def __init__(self, stage):
        self.stage = stage
        self.dir = os.path.join(OUT, stage)
        os.makedirs(self.dir, exist_ok=True)
        self.deadline = float(os.environ.get("HBM_DEADLINE", "0") or 0) or (time.time() + 6 * 3600)
        self.gpu_budget = float(os.environ.get("HBM_GPU_BUDGET_GB", "80"))
        hb = os.environ.get("HBM_HOST_BUDGET_GB", "480")
        self.host_budget = float(hb) * 1e9 if hb else None
        self.host_reserve = float(os.environ.get("HBM_HOST_RESERVE_GB", "40")) * 1e9
        self.results = []
        self.aborted = None
        self.jsonl = os.path.join(self.dir, "orchestrator.jsonl")
        # every knob is read ONCE, here (a run cannot change its plan halfway)
        self.cs = _env_list("HBM_CS", "63 128")
        self.batches = _env_list("HBM_BATCHES", "1 8 32 64 128 256")
        self.ladder_on = os.environ.get("HBM_LADDER", "1") == "1"
        self.ladder_res = int(os.environ.get("HBM_LADDER_RES", "8"))
        self.ladder_max_full = int(os.environ.get("HBM_LADDER_MAX_FULL", "2"))
        self.force_probes = os.environ.get("HBM_FORCE_PROBES", "0") == "1"
        self.p_cells = os.environ.get("HBM_P_CELLS", "63:1 63:64 128:64 128:1 63:1:all 128:max|128:128 63:max|63:256")
        self.ncu_sets = os.environ.get("HBM_NCU_SETS", "full mid min").split()
        self.ncu = os.environ.get("HBM_NCU", "/opt/nvidia/nsight-compute/2025.1.0/ncu")

    def remaining(self):
        return self.deadline - time.time()

    def record(self, **row):
        row.setdefault("t", time.time())
        self.results.append(row)
        HL.append_jsonl(self.jsonl, row)
        log("%s C=%s B=%s kind=%s -> %s %s" % (self.stage, row.get("C"), row.get("B"), row.get("kind"), row.get("status"), (row.get("detail") or "")[:160]))
        return row

    def _env(self, mode, c, b, tag, extra=None):
        env = dict(os.environ)
        env.update(HBM_MODE=mode, HBM_C=str(c), HBM_B=str(b), HBM_TAG=tag, HBM_OUT=self.dir, NOSI_POOL_BLOCKS=str(c - 63), NOSI_VERIFY_ROUND_SLOTS="0",
                   NOSI_TRANSFER_TRACE="0")
        env.pop("NOSI_AVAIL", None)
        env.setdefault("NOSI_ATTN_SPLITS", "4")
        env.update(extra or {})
        return env

    def launch(self, kind, mode, c, b, tag, est_s, cmd_prefix=None, extra_env=None):
        """Run one cell process (fresh CUDA context). Returns the recorded row."""
        if self.remaining() < est_s:
            return self.record(kind=kind, C=c, B=b, tag=tag, status="DEFERRED", cls="IMPLEMENTATION_LIMIT:time_budget",
                               detail="planned %.0f s > remaining %.0f s of the stage budget (not run)" % (est_s, self.remaining()))
        cell = os.path.join(self.dir, tag)
        os.makedirs(cell, exist_ok=True)
        res = os.path.join(cell, "result.json")
        if os.path.exists(res):
            os.remove(res)
        cmd = (cmd_prefix or []) + [sys.executable, os.path.abspath(__file__)]
        logp = os.path.join(cell, "cell.log")
        t0 = time.time()
        timed_out = False
        tmo = max(60.0, min(self.remaining(), 2.5 * est_s + 300))
        with open(logp, "w") as lf:
            lf.write("# %s\n" % " ".join(cmd))
            lf.flush()
            proc = subprocess.Popen(cmd, env=self._env(mode, c, b, tag, extra_env), stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, start_new_session=True)
            try:
                rc = proc.wait(timeout=tmo)
            except subprocess.TimeoutExpired:
                timed_out = True
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGKILL)       # ncu AND its python child
                except OSError:
                    pass
                rc = proc.wait()
        wall = time.time() - t0
        text = open(logp, errors="replace").read()
        out = json.load(open(res)) if os.path.exists(res) else None
        if out is not None and not timed_out:
            status, cls, detail = out.get("status"), out.get("cls", ""), out.get("detail", "")
            if status not in ("OK",) and not cls:
                cls = status
        else:
            status, detail = HL.classify_failure(rc, text[-20000:], timed_out)
            cls = status
        return self.record(kind=kind, C=c, B=b, tag=tag, status=status, cls=cls, detail=detail, rc=rc, wall_s=wall, planned_s=est_s,
                           result=(out or {}).get("peak_allocated") if kind == "probe" else None, dir=cell)

    # ------------------------------------------------------------------ stage U
    def cell_U(self, c, b):
        refuse = HL.refuse_reasons(c, b, L, self.host_budget, self.host_reserve)
        if refuse:
            return self.record(kind="cell", C=c, B=b, tag="c%d_b%d" % (c, b), status="REFUSED", cls=refuse[0][0], detail="; ".join(d for _, d in refuse))
        pred = HL.predict_gpu_peak_gb(c, b, light_restore=(c == 63))
        if pred["peak_gb"] > self.gpu_budget - 1.0 and pred["binding"] == "prefill" and not self.force_probes:
            p = self.probe(c, b)
            if p["status"] != "OK":
                return self.record(kind="cell", C=c, B=b, tag="c%d_b%d" % (c, b), status="NOT_RUN", cls=p["cls"],
                                   detail="quick probe %s (%s); predicted %.1f GB" % (p["status"], p["detail"], pred["peak_gb"]))
        return self.launch("cell", "cell", c, b, "c%d_b%d" % (c, b), HL.cell_estimate_s("U", b))

    def probe(self, c, b):
        return self.launch("probe", "probe", c, b, "probe_c%d_b%d" % (c, b), HL.cell_estimate_s("probe", b))

    def ladder(self, c):
        """Largest batch at capacity c. Implementation limits come from the pre-check (hbm_sweep_lib.refuse_reasons);
        GPU memory from measurement: when the PREDICTED binding phase is the prefill, quick probes bisect between the
        largest fit and the smallest CUDA OOM (seeded at the PREDICTED maximum), then full cells step down from the
        largest probe fit; when it is the decode (C = 63 with the light restore), full cells only (at most
        HBM_LADDER_MAX_FULL). The stop is recorded as CUDA_OOM (measured) or the implementation limit (pre-check)."""
        light = c == 63
        res = self.ladder_res
        max_full = self.ladder_max_full
        mine = [r for r in self.results if r.get("C") == c and r.get("B")]
        full_ok = [r["B"] for r in mine if r["kind"] == "cell" and r["status"] in ("OK", "GATE_FAIL")]
        fit = [r["B"] for r in mine if r["status"] in ("OK", "GATE_FAIL")]
        oom = [r["B"] for r in mine if r.get("cls") == "CUDA_OOM"]
        fit_max = max(fit) if fit else 0
        fail_min = min(oom) if oom else None
        impl_max, impl_reasons = HL.implementation_max_batch(c, L, self.host_budget, self.host_reserve)
        pred_max = HL.predicted_max_batch(c, self.gpu_budget, light)
        binding = HL.predict_gpu_peak_gb(c, max(min(impl_max, pred_max), fit_max, 1), light)["binding"]   # the phase that binds AT the candidate maximum
        tried = {r["B"] for r in mine}
        if fail_min is None and binding == "prefill":
            cand = min(impl_max, pred_max) // res * res
            if cand > fit_max and cand not in tried:
                tried.add(cand)
                p = self.probe(c, cand)
                if p["status"] == "OK":
                    fit_max = cand
                elif p.get("cls") == "CUDA_OOM":
                    fail_min = cand
            if fail_min is None and fit_max < impl_max:
                up = min(impl_max, fit_max + 4 * res)
                if up not in tried and up > fit_max:
                    tried.add(up)
                    p = self.probe(c, up)
                    if p["status"] == "OK":
                        fit_max = up
                    elif p.get("cls") == "CUDA_OOM":
                        fail_min = up
        while binding == "prefill" and fail_min is not None and self.remaining() > 600:
            nxt = HL.next_probe(fit_max, fail_min, seed=pred_max, resolution=res, tried=tried)
            if nxt is None:
                break
            tried.add(nxt)
            p = self.probe(c, nxt)
            if p["status"] == "OK":
                fit_max = nxt
            elif p.get("cls") == "CUDA_OOM":
                fail_min = nxt
            else:
                break
        if binding == "decode":
            fit_max = max(fit_max, min(impl_max, pred_max) if fail_min is None else fit_max)
        best_full = max(full_ok) if full_ok else 0
        cand, n_full = fit_max, 0
        while cand > best_full and n_full < max_full:
            r = self.launch("cell", "cell", c, cand, "c%d_b%d" % (c, cand), HL.cell_estimate_s("U", cand))
            n_full += 1
            if r["status"] in ("OK", "GATE_FAIL"):
                best_full = cand
                break
            if r.get("cls") == "CUDA_OOM":
                fail_min = cand if fail_min is None else min(fail_min, cand)
                cand = max(best_full, (cand - 1) // res * res)
                continue
            break
        if fail_min is not None and fail_min <= impl_max:
            stop = dict(B_stop=fail_min, cls="CUDA_OOM", detail="smallest measured CUDA OOM (probe or full cell)", measured=True)
        else:
            stop = dict(B_stop=impl_max + 1, cls=impl_reasons[0][0] if impl_reasons else "none", detail="; ".join(d for _, d in impl_reasons),
                        measured=False, note="implementation limit from the pre-check (not run); PREDICTED GPU-memory maximum %d" % pred_max)
        return self.record(kind="ladder", C=c, B=best_full, tag="ladder_c%d" % c, status="LARGEST", cls=stop["cls"],
                           detail="largest full cell B=%d; largest prefill-phase fit %d; stop at B=%s (%s) %s" % (best_full, fit_max, stop["B_stop"], stop["cls"], stop["detail"]),
                           largest_full=best_full, largest_fit=fit_max, stop=stop, impl_max=impl_max, predicted_gpu_max=pred_max, binding_phase_predicted=binding)

    def preflight_failed(self, rows):
        """PREFLIGHT GATE (standing rule: one inspected miniature before the larger run): the first cell of each
        capacity must finish OK; a harness failure or an exactness-gate failure stops the stage."""
        bad = [r for r in rows if r.get("status") in ("FAIL", "GATE_FAIL") or r.get("cls") in ("FAIL", "HOST_OOM_KILL")]
        return bad

    def abort(self, why, remaining_plan):
        for (c, b) in remaining_plan:
            self.record(kind="cell", C=c, B=b, tag="c%d_b%d" % (c, b), status="ABORTED", cls="preflight", detail=why)
        self.aborted = why
        return self.finish()

    def run_U(self):
        cs, bs = self.cs, self.batches
        small = [b for b in bs if b <= 128]
        big = [b for b in bs if b > 128]
        plan = [(c, b) for b in small for c in cs] + [(c, b) for b in big for c in cs]
        first = []
        for i, (c, b) in enumerate(plan):              # the matched trend cells first, both C at each B
            r = self.cell_U(c, b)
            if i < len(cs):
                first.append(r)
                if i == len(cs) - 1:
                    bad = self.preflight_failed(first)
                    if bad:
                        return self.abort("preflight failed at %s" % ", ".join("C=%s B=%s %s" % (x["C"], x["B"], x["status"]) for x in bad), plan[i + 1:])
        if self.ladder_on:
            for c in sorted(cs, reverse=True):          # C = 128 (probes, memory-bound) before C = 63 (the 15-min 344 cell)
                self.ladder(c)
        for c in cs:
            for k, v in UNSUPPORTED.items():
                if k[0] == c:
                    self.record(kind="unsupported", C=c, B=None, tag="-", status="UNSUPPORTED", cls="unsupported", detail="%s: %s" % (k[1], v), mode=k[1])
        return self.finish()

    # ------------------------------------------------------------------ stage P
    def ncu_cmd(self, rep_base, metrics, cache_control):
        return [self.ncu, "--profile-from-start", "off", "--replay-mode", "kernel", "--cache-control", cache_control, "--clock-control", "none",
                "--nvtx", "--metrics", metrics, "--target-processes", "application-only", "-f", "-o", rep_base]

    def export(self, cell, rep_base):
        ncu = self.ncu
        rep = rep_base + ".ncu-rep"
        outs = {}
        for name, args in (("raw.csv", ["--csv", "--page", "raw", "--print-units", "base"]),
                           ("raw_nvtx.csv", ["--csv", "--page", "raw", "--print-units", "base", "--print-nvtx-rename", "kernel"]),
                           ("session.csv", ["--csv", "--page", "session"])):
            p = os.path.join(cell, name)
            try:
                with open(p, "w") as f:
                    r = subprocess.run([ncu, "--import", rep] + args, stdout=f, stderr=subprocess.PIPE, text=True, timeout=1800)
                outs[name] = dict(rc=r.returncode, stderr=(r.stderr or "")[-500:])
            except Exception as e:
                outs[name] = dict(rc=None, error=str(e)[:300])
        return outs

    def passprobe(self, sets):
        chosen, lines = None, []
        for name, metrics in sets:
            tag = "passprobe_%s" % name
            cell = os.path.join(self.dir, tag)
            os.makedirs(cell, exist_ok=True)
            r = self.launch("passprobe", "passprobe", 63, 1, tag, HL.cell_estimate_s("passprobe", 1), cmd_prefix=self.ncu_cmd(os.path.join(cell, "rep"), metrics, "none"))
            text = open(os.path.join(cell, "cell.log"), errors="replace").read()
            passes = HL.parse_ncu_passes(text)
            sp = HL.single_pass(text)
            lines.append(dict(set=name, metrics=metrics, status=r["status"], single_pass=sp, passes=[p["passes"] for p in passes], kernels=[p["name"] for p in passes]))
            self.record(kind="passprobe_result", C=63, B=1, tag=tag, status="SINGLE_PASS" if sp else ("MULTI_PASS" if sp is False else "UNKNOWN"),
                        cls="", detail="%s: passes %s" % (name, [p["passes"] for p in passes]))
            if sp and chosen is None:
                chosen = (name, metrics)
                break
        self.p_single_pass = chosen is not None
        if chosen is None:
            chosen = sets[-1]
        HL.write_json(os.path.join(self.dir, "passprobe.json"), dict(sets=lines, chosen=dict(name=chosen[0], metrics=chosen[1])))
        return chosen

    def cell_P(self, c, b, metrics, cache_control="none", tag=None):
        tag = tag or "c%d_b%d" % (c, b)
        refuse = HL.refuse_reasons(c, b, L, self.host_budget, self.host_reserve)
        if refuse:
            return self.record(kind="ncu_cell", C=c, B=b, tag=tag, status="REFUSED", cls=refuse[0][0], detail="; ".join(d for _, d in refuse))
        cell = os.path.join(self.dir, tag)
        os.makedirs(cell, exist_ok=True)
        rep_base = os.path.join(cell, "rep")
        r = self.launch("ncu_cell", "ncu_cell", c, b, tag, HL.cell_estimate_s("P", b), cmd_prefix=self.ncu_cmd(rep_base, metrics, cache_control))
        if os.path.exists(rep_base + ".ncu-rep"):
            ex = self.export(cell, rep_base)
            HL.write_json(os.path.join(cell, "export.json"), ex)
            try:
                self.postprocess(cell)
            except Exception as e:
                self.record(kind="postprocess", C=c, B=b, tag=tag, status="FAIL", cls="FAIL", detail="postprocess: %s" % str(e)[:300])
        text = open(os.path.join(cell, "cell.log"), errors="replace").read()
        HL.write_json(os.path.join(cell, "ncu_settings.json"), dict(command=self.ncu_cmd(rep_base, metrics, cache_control), metrics=metrics.split(","),
                                                                     replay_mode="kernel", cache_control=cache_control, clock_control="none",
                                                                     passes=HL.parse_ncu_passes(text), single_pass=HL.single_pass(text),
                                                                     why=dict(replay="kernel replay; single-pass metric set so no save/restore replay perturbs caches",
                                                                              cache_control="none = in-situ L2 (each kernel inherits the previous kernel's L2); a control cell uses 'all' (flush before each kernel) to bound the L2 effect",
                                                                              clock_control="none = natural boost clocks, matching the unprofiled stage U")))
        return r

    def postprocess(self, cell):
        raw = open(os.path.join(cell, "raw.csv"), errors="replace").read()
        nv = open(os.path.join(cell, "raw_nvtx.csv"), errors="replace").read() if os.path.exists(os.path.join(cell, "raw_nvtx.csv")) else None
        rows = HL.ncu_kernel_rows(raw, nv)
        wins = HL.split_windows(rows)
        meta = json.load(open(os.path.join(cell, "windows.json"))) if os.path.exists(os.path.join(cell, "windows.json")) else dict(windows=[])
        bym = {w["markers"]: w for w in meta.get("windows", [])}
        krows, summ = [], []
        for w in wins:
            m = bym.get(w["markers"], {})
            for i, r in enumerate(w["rows"]):
                a = HL.attribute(r)
                krows.append(dict(r, window=w["markers"], mode=m.get("mode"), order=i, **a))
            s = HL.summarize_window(w["rows"])
            summ.append(dict(window=w["markers"], mode=m.get("mode"), step=m.get("step"), loads=m.get("loads"), pool_swap=m.get("pool_swap"),
                             pool_move_in=m.get("pool_move_in"), pool_move_out=m.get("pool_move_out"), compress_keys=m.get("compress_keys"),
                             tail_rows=m.get("tail_rows"), burst=m.get("burst"), **s))
        HL.write_csv(os.path.join(cell, "kernels.csv"), krows)
        HL.write_json(os.path.join(cell, "window_summary.json"), dict(windows=summ, n_windows_found=len(wins), n_windows_expected=len(meta.get("windows", []))))

    def run_P(self):
        full = ("full", "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_aperture_device_op_read.sum,"
                        "lts__t_sectors_aperture_device_op_write.sum,lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum,pcie__read_bytes.sum,pcie__write_bytes.sum")
        mid = ("mid", "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_aperture_device_op_write.sum,"
                      "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum")
        mini = ("min", "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum")
        sets = [s for s in (full, mid, mini) if s[0] in self.ncu_sets]
        up = os.path.join(OUT, "U", "orchestrator.json")
        if os.path.exists(up):
            u = json.load(open(up))
            if u.get("aborted"):
                self.record(kind="ncu_cell", C=None, B=None, tag="-", status="ABORTED", cls="preflight", detail="stage U aborted: %s" % u["aborted"])
                self.aborted = "stage U aborted"
                return self.finish()
        name, metrics = self.passprobe(sets)
        largest = self.largest_from_U()
        spec = self.p_cells
        done = set()
        for item in spec.split():
            for k, alt in enumerate(item.split("|")):
                parts = alt.split(":")
                c = int(parts[0])
                if parts[1] == "max":
                    b = largest.get(c)
                    if not b:
                        self.record(kind="ncu_cell", C=c, B=None, tag="c%d_max" % c, status="NOT_RUN", cls="no_stage_U_largest", detail="stage U recorded no largest full cell at C=%d" % c)
                        continue
                else:
                    b = int(parts[1])
                cc = parts[2] if len(parts) > 2 else "none"
                tag = "c%d_b%d" % (c, b) + ("" if cc == "none" else "_cache%s" % cc)
                if tag in done:
                    break
                if not getattr(self, "p_single_pass", True) and b > P_MULTIPASS_MAX_B:
                    # review NB (2026-09-23): no metric set is single-pass, so kernel replay would save/restore
                    # device memory at 60-77 GB allocated; keep stage P at B <= 64 and record the rest, never silently.
                    self.record(kind="ncu_cell", C=c, B=b, tag=tag, status="NOT_RUN", cls="multi_pass_metric_set",
                                detail="no single-pass metric set; profiled cells capped at B <= %d" % P_MULTIPASS_MAX_B)
                    done.add(tag)
                    break
                if self.remaining() < HL.cell_estimate_s("P", b) and k + 1 < len(item.split("|")):
                    self.record(kind="ncu_cell", C=c, B=b, tag=tag, status="DEFERRED", cls="IMPLEMENTATION_LIMIT:time_budget",
                                detail="planned %.0f s > remaining %.0f s; trying the fallback %s" % (HL.cell_estimate_s("P", b), self.remaining(), item.split("|")[k + 1]))
                    continue
                r = self.cell_P(c, b, metrics, cc, tag)
                done.add(tag)
                if len(done) == 1 and r is not None and self.preflight_failed([r]):
                    self.record(kind="ncu_cell", C=c, B=b, tag=tag, status="ABORTED", cls="preflight", detail="the first profiled cell failed: the remaining stage P cells are not run")
                    self.aborted = "first profiled cell failed"
                    return self.finish()
                break
        return self.finish()

    def largest_from_U(self):
        p = os.path.join(OUT, "U", "orchestrator.jsonl")
        out = {}
        for r in HL.read_jsonl(p):
            if r.get("kind") == "ladder" and r.get("largest_full"):
                out[int(r["C"])] = int(r["largest_full"])
        return out

    def finish(self):
        HL.write_csv(os.path.join(self.dir, "cells.csv"), self.results)
        unexpected = [r for r in self.results if r.get("kind") not in ("passprobe", "passprobe_result") and (r.get("status") in ("FAIL",) or r.get("cls") in ("FAIL", "HOST_OOM_KILL"))]
        gate = [r for r in self.results if r.get("status") == "GATE_FAIL"]
        HL.write_json(os.path.join(self.dir, "orchestrator.json"), dict(stage=self.stage, results=self.results, unexpected=len(unexpected), gate_failures=len(gate), aborted=self.aborted))
        log("stage %s done: %d rows, %d unexpected failures, %d gate failures" % (self.stage, len(self.results), len(unexpected), len(gate)))
        return min(len(unexpected) + len(gate) + (1 if self.aborted else 0), 100)


# =====================================================================================================================
# table (CPU): convenience summaries; Codex aggregates from the raw files
# =====================================================================================================================
def table():
    import hbm_sweep_table as T
    return T.main(OUT, L=L)


if __name__ == "__main__":
    if MODE == "orchestrate":
        o = Orchestrator(STAGE)
        sys.exit(o.run_U() if STAGE == "U" else o.run_P())
    if MODE == "table":
        sys.exit(table())
    fn = dict(cell=mode_cell, probe=mode_probe, ncu_cell=mode_ncu_cell, passprobe=mode_passprobe).get(MODE)
    if fn is None:
        raise SystemExit("unknown HBM_MODE %r" % MODE)
    sys.exit(run_guarded(fn))
