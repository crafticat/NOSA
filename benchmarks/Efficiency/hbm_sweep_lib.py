"""Pure helpers of the weights-versus-KV batch sweep (hbm_batch_sweep.py). No CUDA, no model: CPU-tested in
retroinfer-eval tests/test_hbm_batch_sweep.py. Every number that is a model fact is cited; every number that is a
prediction is labelled PREDICTED and is never written into a measured column.

Sections:
  1. model facts (NOSA-8B) and byte units
  2. implementation limits, predictions, failure classes, the ladder (largest batch)
  3. step statistics
  4. GPU memory: tensor categories, inventory with alias dedup, window split, graph pools, breakdown, windows
  5. ANALYTIC HBM bytes per decode step (kept apart from every counter column)
  6. Nsight Compute CSV / log parsing, profiled-window split, kernel attribution (method labelled)
  7. torch.profiler chrome trace -> kernel rows with tensor shapes; the order join to ncu rows
  8. CSV / JSON export
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------------------------
# 1. model facts (NOSA-8B: /mnt/beegfs/ojerbi/models/NOSA-8B/config.json; nosa_llama.py:153-183, :993-1000)
# ---------------------------------------------------------------------------------------------------------------
LAYERS = 32
HIDDEN = 4096
Q_HEADS = 32
KV_HEADS = 2
HEAD_DIM = 128
INTER = 16384
VOCAB = 73448
BLOCK = 64                      # tokens per block (nosa_llama.py:960)
TOPK = 64                       # attended slots per (layer, KV head, request); slot 63 = the tail (cache_engine.py:404)
MAX_GEN = 8192                  # host cache head-room (cache_engine.py:172)
BF16 = 2
SPLITS = 4                      # NOSI_ATTN_SPLITS pinned by the job script (nosa_llama.py:21-27)

# GEMM weights (N, K, calls per step): retroinfer-eval scripts/gemm_ncu_probe.py:25-27
GEMM_SHAPES = OrderedDict(wqkv=(4608, 4096, 32), wo=(4096, 4096, 32), gate_up=(32768, 4096, 32), down=(4096, 16384, 32), lm_head=(73448, 4096, 1))
GEMM_WEIGHT_BYTES = sum(n * k * c * BF16 for n, k, c in GEMM_SHAPES.values())          # 15,768,289,280 B (REPRODUCE.md E5)

ROW_KV_BYTES = KV_HEADS * HEAD_DIM * BF16 * 2          # K + V of one token row, both KV heads: 1024 B
ROW_BIAS_BYTES = KV_HEADS * BF16                       # kv_bias of one token row, both heads: 4 B
ROW_BYTES = ROW_KV_BYTES + ROW_BIAS_BYTES              # 1028 B (verify_alone.py:107 ROW_BYTES_ATTN = 64 x 1028)
SLOT_BYTES = BLOCK * ROW_BYTES                         # 65,792 B: one slot, both heads, K + V + bias
LOAD_WIRE_BYTES = BLOCK * HEAD_DIM * BF16 * 2          # 32,768 B over PCIe per loaded (layer, head, request, slot) (transfer_trace.py:52-54)
LOAD_BIAS_BYTES = BLOCK * BF16                         # 128 B kv_bias per load, read on the device from total_cis (transfer_trace.py:55)
LOAD_HBM_WRITE_BYTES = LOAD_WIRE_BYTES + LOAD_BIAS_BYTES   # 32,896 B written into the window per load
POOL_SWAP_BYTES = 2 * LOAD_HBM_WRITE_BYTES             # SWAP reads 2 and writes 2 block-heads (flash_pool_swap.py action table)
POOL_MOVE_BYTES = LOAD_HBM_WRITE_BYTES                 # MOVE_IN / MOVE_OUT read 1 and write 1
POOL_NONE, POOL_SWAP, POOL_MOVE_IN, POOL_MOVE_OUT = 0, 1, 2, 3   # transfer_trace.py:50-51

MODEL_RANGES = ("linear", "rope", "compression", "stage 1", "max pooling", "offloading update", "stage 2", "ffn")   # nosa_llama.py:547-667
STEP_RANGE_PREFIX = "hbm_step:"
MARKER_KERNEL = "spin_kernel"      # torch.cuda._sleep -> at::cuda::sleep -> spin_kernel (present in the image's libtorch_cuda.so)


def next_pow2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------------------------------------------
# 2. implementation limits, predictions, failure classes, ladder
# ---------------------------------------------------------------------------------------------------------------
def pinned_host_bytes(B: int, L: int, max_gen: int = MAX_GEN, heads: int = KV_HEADS, dim: int = HEAD_DIM, layers: int = LAYERS) -> Dict:
    """Host pinned K and V caches: 2 x layers tensors of (B, L + max_gen, heads, dim) bf16 (cache_engine.py:263-264).
    The pinned caching allocator rounds each block up to a power of two (observed: 65,536 pages = 256 MiB at B = 16,
    1,048,576 pages = 4 GiB at B = 336 in docs/evidence/worker_sweep_preflight_2175723/sweep_b*.json numa pages)."""
    per = B * (L + max_gen) * heads * dim * BF16
    alloc = next_pow2(per)
    return dict(per_tensor=per, per_tensor_alloc=alloc, requested=2 * layers * per, allocated=2 * layers * alloc, tensors=2 * layers)


# Component model of the persistent GPU footprint (retroinfer-eval ENGINE map, 2026-09-23; reproduces the 7 traced
# peaks of the pool jobs within 0.1-0.25 GB). PREDICTED; used only to order ladder probes and to label predictions.
PRED_FIXED_GB = 16.4382             # unique weights 16.3705 GB + RoPE table 0.0676 GB
PRED_PREFILL_TRANSIENT_GB = 2.6     # single-request prefill transient (fitted)
PRED_PER_REQ_GB = {63: 0.15433, 128: 0.29128}
PRED_LIGHT_RESTORE_PER_REQ_GB = 0.019450112
PRED_DECODE_TRANSIENT_GB = 0.3


def per_req_state_gb(C: int) -> float:
    if C in PRED_PER_REQ_GB:
        return PRED_PER_REQ_GB[C]
    # linear in the physical slots: 63 -> 64 slots, 128 -> 129 slots (2.105 MB per slot per request over 32 layers)
    return PRED_PER_REQ_GB[63] + (C - 63) * (PRED_PER_REQ_GB[128] - PRED_PER_REQ_GB[63]) / 65.0


def predict_gpu_peak_gb(C: int, B: int, light_restore: bool) -> Dict:
    """PREDICTED peak GB per phase and the binding phase (never a measurement)."""
    s = per_req_state_gb(C)
    prefill = PRED_FIXED_GB + PRED_PREFILL_TRANSIENT_GB + s * B
    decode = PRED_FIXED_GB + PRED_DECODE_TRANSIENT_GB + (s + (PRED_LIGHT_RESTORE_PER_REQ_GB if light_restore else 0.0)) * B
    return dict(prefill_gb=prefill, decode_gb=decode, peak_gb=max(prefill, decode), binding="prefill" if prefill >= decode else "decode", label="PREDICTED (component model)")


def predicted_max_batch(C: int, budget_gb: float, light_restore: bool, hi: int = 4096) -> int:
    lo = 0
    for b in range(1, hi + 1):
        if predict_gpu_peak_gb(C, b, light_restore)["peak_gb"] <= budget_gb:
            lo = b
        else:
            break
    return lo


def refuse_reasons(C: int, B: int, L: int, host_budget_bytes: Optional[float], host_reserve_bytes: float = 40e9) -> List[Tuple[str, str]]:
    """Implementation limits checked BEFORE any GPU time (a refusal is a recorded result, not a failure).
    (a) host pinned memory after power-of-two rounding vs the job's host memory (cache_engine.py:263-264);
    (b) the pooled path's int32 asserts (cache_engine.py:289-294; P = 0 runs the upstream path without them);
    (c) the pool bookkeeping kernel's shared memory 28 P + 4 M <= 48 KB (pool_update_kernel.cu:276-285)."""
    out = []
    P = C - (TOPK - 1)
    if P < 0:
        out.append(("CONFIG", "C = %d < 63: the engine has 63 usable attended slots at minimum" % C))
        return out
    ph = pinned_host_bytes(B, L)
    if host_budget_bytes is not None and ph["allocated"] + host_reserve_bytes > host_budget_bytes:
        out.append(("IMPLEMENTATION_LIMIT:host_pinned",
                    "host pinned K/V = %d tensors x %.3f GB rounded to %.3f GB (power of two) = %.1f GB, + %.0f GB reserve > job host memory %.0f GB (cache_engine.py:263-264)"
                    % (ph["tensors"], ph["per_tensor"] / 1e9, ph["per_tensor_alloc"] / 1e9, ph["allocated"] / 1e9, host_reserve_bytes / 1e9, host_budget_bytes / 1e9)))
    if P > 0:
        hd = KV_HEADS * HEAD_DIM
        if B * (L + MAX_GEN) * hd >= 2 ** 31:
            out.append(("IMPLEMENTATION_LIMIT:int32_assert_host", "pooled path asserts B x (L + 8192) x 256 < 2^31: %d (cache_engine.py:289-291)" % (B * (L + MAX_GEN) * hd)))
        if B * (TOPK + P) * BLOCK * hd >= 2 ** 31:
            out.append(("IMPLEMENTATION_LIMIT:int32_assert_gpu", "pooled path asserts B x (64 + P) x 64 x 256 < 2^31: %d (cache_engine.py:292-294)" % (B * (TOPK + P) * BLOCK * hd)))
        if 28 * P + 4 * TOPK > 48 * 1024:
            out.append(("IMPLEMENTATION_LIMIT:pool_smem", "28 P + 4 M = %d > 48 KB shared memory (pool_update_kernel.cu:276-285)" % (28 * P + 4 * TOPK)))
    return out


def implementation_max_batch(C: int, L: int, host_budget_bytes: Optional[float], host_reserve_bytes: float = 40e9, hi: int = 4096) -> Tuple[int, List[Tuple[str, str]]]:
    """Largest B with no refusal, and the refusal reasons at B + 1 (the stop classification)."""
    best = 0
    for b in range(1, hi + 1):
        if refuse_reasons(C, b, L, host_budget_bytes, host_reserve_bytes):
            return best, refuse_reasons(C, b, L, host_budget_bytes, host_reserve_bytes)
        best = b
    return best, []


STATUS_OK = "OK"
FAILURE_CLASSES = ("CUDA_OOM", "HOST_OOM_KILL", "IMPLEMENTATION_LIMIT:assert", "IMPLEMENTATION_LIMIT:index_overflow",
                   "IMPLEMENTATION_LIMIT:host_pinned", "IMPLEMENTATION_LIMIT:time_budget", "FAIL")


def classify_failure(rc: Optional[int], text: str = "", timed_out: bool = False) -> Tuple[str, str]:
    """(status, detail) of a finished cell process. rc 0 -> OK. A CUDA caching-allocator OOM is CUDA_OOM; a kill by
    the host OOM killer (rc 137 / -9) is HOST_OOM_KILL; a pinned-host allocation failure (traceback through
    pin_memory / cudaHostAlloc) is IMPLEMENTATION_LIMIT:host_pinned; an int32 guard assertion is
    IMPLEMENTATION_LIMIT:assert; an illegal address / index error is IMPLEMENTATION_LIMIT:index_overflow; a timeout is
    IMPLEMENTATION_LIMIT:time_budget; anything else is FAIL."""
    t = text or ""
    if timed_out:
        return "IMPLEMENTATION_LIMIT:time_budget", "the cell exceeded its time budget and was stopped"
    if rc == 0:
        return STATUS_OK, ""
    if "pin_memory" in t or "cudaHostAlloc" in t or "pinned memory" in t.lower() or "CachingHostAllocator" in t:
        return "IMPLEMENTATION_LIMIT:host_pinned", _last_line(t, ("RuntimeError", "Error"))
    if "OutOfMemoryError" in t or "CUDA out of memory" in t:
        return "CUDA_OOM", _last_line(t, ("OutOfMemoryError", "CUDA out of memory"))
    if rc in (137, -9) or any(l.strip() == "Killed" or l.strip().endswith(" Killed") for l in t.splitlines()[-3:]):
        return "HOST_OOM_KILL", "rc=%s (SIGKILL: host cgroup OOM killer or scancel)" % rc
    if re.search(r"int32|overflow|2\*\*31", t) and "AssertionError" in t:
        return "IMPLEMENTATION_LIMIT:assert", _last_line(t, ("AssertionError",))
    if "illegal memory access" in t or "device-side assert" in t or "index out of bounds" in t.lower():
        return "IMPLEMENTATION_LIMIT:index_overflow", _last_line(t, ("illegal memory access", "device-side assert", "index"))
    return "FAIL", _last_line(t, ("Error", "error", "REFUSED")) or ("rc=%s" % rc)


def _last_line(text: str, keys: Sequence[str]) -> str:
    for line in reversed(text.splitlines()):
        if any(k in line for k in keys):
            return line.strip()[:400]
    return ""


def next_probe(fit_max: int, fail_min: Optional[int], seed: Optional[int] = None, resolution: int = 8, tried: Iterable[int] = ()) -> Optional[int]:
    """The next batch to probe between the largest measured fit and the smallest measured failure (None = done).
    The first probe inside an unexplored interval is the seed (the PREDICTED maximum) aligned down to `resolution`;
    later probes bisect. Never returns a batch already tried or outside (fit_max, fail_min)."""
    tried = set(int(t) for t in tried)
    if fail_min is None or fail_min - fit_max <= resolution:
        return None
    lo, hi = fit_max, fail_min
    cands = []
    if seed is not None:
        s = int(seed) // resolution * resolution
        if lo < s < hi:
            cands.append(s)
    mid = (lo + hi) // 2 // resolution * resolution
    if mid <= lo:
        mid = lo + resolution
    cands.append(mid)
    for c in cands:
        if lo < c < hi and c not in tried:
            return c
    for c in range(lo + resolution, hi, resolution):
        if c not in tried:
            return c
    return None


def cell_estimate_s(stage: str, B: int, prefill_s_per_req: float = 2.75, overhead_s: float = 75.0, n_steps: int = 63) -> float:
    """Wall seconds a cell is PLANNED to take (deferral decisions only). prefill 2.7 s / request at L = 16128
    (job 2175723: 908.5 s for 336 requests); decode ~0.4 s per step at large B incl. reps and sleep gates."""
    if stage == "probe":
        return overhead_s + 60.0 + 3.0 + (B >= 128) * 60.0
    if stage == "passprobe":
        return 60.0
    decode = n_steps * (0.15 + 0.004 * B)
    prof = 0.0
    if stage == "P":
        prof = 120.0 + 0.6 * B
    return overhead_s + prefill_s_per_req * B + decode + prof


# ---------------------------------------------------------------------------------------------------------------
# 3. step statistics
# ---------------------------------------------------------------------------------------------------------------
def percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    v = sorted(float(x) for x in xs)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    return v[int(k)] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def step_stats(xs: Sequence[float]) -> Dict:
    v = [float(x) for x in xs]
    if not v:
        return dict(n=0, median=float("nan"), p95=float("nan"), mean=float("nan"), min=float("nan"), max=float("nan"))
    return dict(n=len(v), median=percentile(v, 50), p95=percentile(v, 95), mean=sum(v) / len(v), min=min(v), max=max(v))


def tokens_per_s(B: int, median_ms: float) -> float:
    return 1000.0 * B / median_ms if median_ms and median_ms > 0 else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# 4. GPU memory
# ---------------------------------------------------------------------------------------------------------------
MEM_CATEGORIES = ("weights", "resident_historical_kv", "active_compressed_kv", "staging_provisional", "workspaces_graph_pools",
                  "other_overhead", "unclassified", "allocator_slack")
WEIGHT_ATTRS = {"wqkv": "gemm_weight", "wo": "gemm_weight", "gate_up_proj": "gemm_weight", "down_proj": "gemm_weight", "lm_head": "gemm_weight",
                "embed_tokens": "embedding_table", "norm_weight": "norm", "input_layernorm_weight": "norm", "post_attention_layernorm_weight": "norm",
                "delta.weight": "nosa_cis_gate", "delta.bias": "nosa_cis_gate", "A": "nosa_cis_gate"}
MODEL_CONSTANTS = {"cos_sin_cache": "rope_cos_sin_table (model constant)"}
ENGINE_WINDOW = ("_k_gpu", "_v_gpu", "_kv_bias_gpu")                   # split by slot: historical / tail / pool (cache_engine.py:295-297)
ENGINE_VIEWS = ("_k_gpu_att", "_v_gpu_att", "_kv_bias_gpu_att")         # aliases of the window (cache_engine.py:390-393)
ENGINE_META = ("_block_map", "_new_block_map_buf", "_load_mask", "_cache_lens", "_pool_map", "_pool_age", "_pool_target", "_pool_action", "_round_map")
LAYER_TABLES = {"compress_k_cache_varlen": "compressed_k", "no_compress_k_cache": "uncompressed_k_ring", "compressed_cis": "compressed_cis",
                "tail_cis": "compressed_cis", "total_cis": "per_token_kv_bias_table", "cached_compressed_cu_seqlens": "selection_metadata",
                "cached_compressed_cu_seqlens_adder": "selection_metadata", "no_rope_keys": "prefill_leftover"}
MODEL_BUFFERS = ("pooling_buf_all", "max_pooling_buf", "max_pooling_buf_cis", "topk_buf_val", "topk_buf_indices", "topk_val_buf_q",
                 "topk_idx_buf_q", "topk_val_buf", "topk_idx_buf", "mask_buf")                       # nosa_llama.py:1098-1108
LAYER_BUFFERS = ("score_buf", "compressed_cis_buf")                                                  # nosa_llama.py:450-451


def classify_tensor(kind: str, attr: str) -> Tuple[str, str]:
    """(category, subcategory) of one owned tensor. kind: model | layer | engine | cache_layer | model_buffer |
    layer_buffer | snapshot | driver | gc."""
    if kind in ("model", "layer"):
        if attr in WEIGHT_ATTRS:
            return "weights", WEIGHT_ATTRS[attr]
        if attr in MODEL_CONSTANTS:
            return "other_overhead", MODEL_CONSTANTS[attr]
        if attr in LAYER_BUFFERS or attr in MODEL_BUFFERS:
            return "workspaces_graph_pools", "decode_buffers"
        return "unclassified", "model_attr:" + attr
    if kind == "engine":
        if attr in ENGINE_WINDOW or attr in ENGINE_VIEWS:
            return "resident_historical_kv", "window (split by slot)"
        if attr in ENGINE_META:
            return "active_compressed_kv", "selection_metadata"
        return "unclassified", "engine_attr:" + attr
    if kind == "cache_layer":
        if attr in LAYER_TABLES:
            sub = LAYER_TABLES[attr]
            if sub == "prefill_leftover":
                return "unclassified", "prefill_leftover:" + attr
            return "active_compressed_kv", sub
        return "unclassified", "cache_layer_attr:" + attr
    if kind in ("model_buffer", "layer_buffer"):
        return "workspaces_graph_pools", "decode_buffers"
    if kind == "snapshot":
        return "staging_provisional", "light_restore_snapshot"
    if kind == "driver":
        if attr in ("input_ids", "forced", "prompt"):
            return "other_overhead", "driver_inputs"
        return "staging_provisional", "driver:" + attr
    return "unclassified", "python_visible_unowned"


def split_window(nbytes: int, slots_total: int, topk: int = TOPK) -> Dict[str, int]:
    """Bytes of one window tensor by slot class: historical = slots 0..topk-2, tail = slot topk-1, pool = slots >= topk.
    Exact when nbytes is a multiple of slots_total (every window tensor is (B, slots x 64, ...))."""
    if slots_total <= 0 or nbytes % slots_total:
        raise ValueError("window bytes %d not divisible by %d slots" % (nbytes, slots_total))
    per = nbytes // slots_total
    return dict(historical=per * (topk - 1), tail=per, pool=per * (slots_total - topk))


def storage_key(t) -> Tuple[str, int]:
    return storage_info(t)[0]


def storage_info(t) -> Tuple[Tuple[str, int], int, bool]:
    """((device, storage base pointer), storage bytes, fallback). Falls back to (data_ptr - offset, numel x element
    size) if the untyped storage is not accessible (flagged, so an alias could then be counted twice)."""
    try:
        st = t.untyped_storage()
        return (str(t.device), int(st.data_ptr())), int(st.nbytes()), False
    except Exception:
        base = int(t.data_ptr()) - int(t.storage_offset()) * int(t.element_size())
        return (str(t.device), base), int(t.numel()) * int(t.element_size()), True


def build_inventory(entries: Iterable[Tuple[str, str, str, object]], slots_total: int = TOPK, topk: int = TOPK) -> List[Dict]:
    """One row per (owner, attr); a storage is COUNTED ONCE (first owner in walk order), later owners of the same
    storage (views, tied weights, aliases) are rows with counted_bytes = 0 and alias_of = the first owner. Window
    tensors (_k_gpu / _v_gpu / _kv_bias_gpu) are split into historical / tail / pool rows (analytic_split = True).
    entries: (kind, owner, attr, tensor) with any torch tensor (CPU tensors in tests)."""
    first: Dict[Tuple[str, int], str] = {}
    rows: List[Dict] = []
    for kind, owner, attr, t in entries:
        if t is None:
            continue
        try:
            key, nbytes, fallback = storage_info(t)
        except Exception:
            continue
        name = "%s.%s" % (owner, attr)
        cat, sub = classify_tensor(kind, attr)
        base = dict(kind=kind, owner=owner, attr=attr, name=name, device=str(t.device), storage_ptr=hex(key[1]), storage_bytes=nbytes,
                    shape=list(t.shape), dtype=str(t.dtype), category=cat, subcategory=sub, alias_of="", analytic_split=False, storage_fallback=fallback)
        if nbytes == 0 or key[1] == 0:
            rows.append(dict(base, counted_bytes=0, alias_of="(empty)"))
            continue
        if key in first:
            rows.append(dict(base, counted_bytes=0, alias_of=first[key]))
            continue
        first[key] = name
        if kind == "engine" and attr in ENGINE_WINDOW:
            parts = split_window(nbytes, slots_total, topk)
            rows.append(dict(base, category="resident_historical_kv", subcategory="window_slots_0_%d" % (topk - 2), counted_bytes=parts["historical"], analytic_split=True))
            rows.append(dict(base, name=name + "[tail]", category="active_compressed_kv", subcategory="active_tail_slot_%d" % (topk - 1), counted_bytes=parts["tail"], analytic_split=True))
            if parts["pool"]:
                rows.append(dict(base, name=name + "[pool]", category="resident_historical_kv", subcategory="victim_pool_slots", counted_bytes=parts["pool"], analytic_split=True))
            continue
        rows.append(dict(base, counted_bytes=nbytes))
    return rows


def inventory_totals(rows: Iterable[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["category"]] += int(r["counted_bytes"])
    return dict(out)


def inventory_by_sub(rows: Iterable[Dict]) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r["counted_bytes"]:
            out[(r["category"], r["subcategory"])] += int(r["counted_bytes"])
    return dict(out)


def summarize_segments(snapshot: Dict) -> Dict:
    """torch.cuda.memory._snapshot() -> per pool: segments, reserved, allocated, active, requested (frames dropped)."""
    pools: Dict[str, Dict] = {}
    segs = []
    for s in snapshot.get("segments", []):
        pid = s.get("segment_pool_id", (0, 0))
        pid = tuple(pid) if isinstance(pid, (list, tuple)) else (pid,)
        key = "%s" % (pid,)
        p = pools.setdefault(key, dict(pool_id=list(pid), segments=0, total_size=0, allocated_size=0, active_size=0, requested_size=0))
        p["segments"] += 1
        p["total_size"] += int(s.get("total_size", 0))
        p["allocated_size"] += int(s.get("allocated_size", 0))
        p["active_size"] += int(s.get("active_size", 0))
        req = sum(int(b.get("requested_size", 0)) for b in s.get("blocks", []) if b.get("state") == "active_allocated")
        p["requested_size"] += req
        segs.append(dict(address=int(s.get("address", 0)), total_size=int(s.get("total_size", 0)), allocated_size=int(s.get("allocated_size", 0)),
                         active_size=int(s.get("active_size", 0)), requested_size=req, pool_id=list(pid), stream=s.get("stream"),
                         segment_type=s.get("segment_type"), is_expandable=s.get("is_expandable"),
                         blocks=[dict(address=int(b.get("address", 0)), size=int(b.get("size", 0)), requested_size=int(b.get("requested_size", 0)), state=b.get("state"))
                                 for b in s.get("blocks", [])]))
    return dict(pools=pools, segments=segs, keys=sorted({k for s in snapshot.get("segments", []) for k in s.keys()}))


def graph_pool_bytes(seg_summary: Dict, visible: Iterable[Tuple[int, int]]) -> Dict:
    """Active bytes held in NON-default private pools (CUDA graph pools: segment_pool_id != (0, 0)) that no
    python-visible storage accounts for. visible: (address, nbytes) of the inventory's counted storages."""
    vis = sorted((int(a), int(n)) for a, n in visible)
    starts = [a for a, _ in vis]
    import bisect
    held, visible_in, found = 0, 0, False
    for s in seg_summary.get("segments", []):
        pid = tuple(s.get("pool_id") or (0, 0))
        if pid == (0, 0) or pid == (0,):
            continue
        found = True
        for b in s.get("blocks", []):
            if b.get("state") != "active_allocated":
                continue
            a, n = int(b["address"]), int(b["size"])
            i = bisect.bisect_right(starts, a) - 1
            if i >= 0 and vis[i][0] <= a < vis[i][0] + max(vis[i][1], 1):
                visible_in += n
            else:
                held += n
    return dict(graph_pool_bytes=held, visible_in_private_pools=visible_in, private_pools_found=found)


def memory_breakdown(inv_rows: Sequence[Dict], allocated: int, reserved: int, device_used: int, device_total: int,
                     requested: Optional[int] = None, graph_pools: int = 0) -> List[Dict]:
    """Occupied GPU memory by category, summing EXACTLY to device_used:
      inventory categories (python-visible storages, counted once) + CUDA-graph private pools not visible
      + unclassified (= requested - visible - graph pools; allocated when requested is unknown)
      + allocator slack (= reserved - allocated, plus block rounding allocated - requested)
      + other overhead outside the caching allocator (= device_used - reserved: context, cuBLAS/cuDNN handles, ...)."""
    by_sub = inventory_by_sub(inv_rows)
    visible = sum(by_sub.values())
    rows = []

    def add(cat, sub, b, basis):
        rows.append(dict(category=cat, subcategory=sub, bytes=int(b), gb=b / 1e9, pct_of_device_used=100.0 * b / device_used if device_used else float("nan"),
                         pct_of_device_total=100.0 * b / device_total if device_total else float("nan"), basis=basis))
    for (cat, sub), b in sorted(by_sub.items(), key=lambda kv: (MEM_CATEGORIES.index(kv[0][0]) if kv[0][0] in MEM_CATEGORIES else 99, kv[0][1])):
        add(cat, sub, b, "inventory (untyped storage bytes, alias-deduplicated)")
    if graph_pools:
        add("workspaces_graph_pools", "cuda_graph_private_pools (allocator snapshot, not python-visible)", graph_pools, "memory snapshot: active blocks in non-default pools")
    base = requested if requested is not None else allocated
    add("unclassified", "allocator-held, not attributed (%s - visible - graph pools)" % ("requested" if requested is not None else "allocated"),
        base - visible - graph_pools, "allocator stats minus inventory")
    if requested is not None:
        add("allocator_slack", "block rounding (allocated - requested)", allocated - requested, "memory_stats")
    add("allocator_slack", "cached free (reserved - allocated)", reserved - allocated, "memory_stats")
    add("other_overhead", "outside the caching allocator (device used - reserved)", device_used - reserved, "cudaMemGetInfo - memory_stats")
    return rows


def breakdown_totals(rows: Sequence[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["category"]] += int(r["bytes"])
    return dict(out)


class MemLedger:
    """Synchronized memory points and PEAK WINDOWS. The caller synchronizes and resets the allocator peak at the
    start of each window; a window's peak is its own max_memory_allocated. Peaks of different windows are never
    added: the process peak is the MAX over windows (with the window that set it)."""

    def __init__(self):
        self.points: List[Dict] = []
        self.windows: List[Dict] = []
        self._open: Optional[str] = None

    def point(self, name: str, allocated: int, reserved: int, device_used: int, device_total: int, requested: Optional[int] = None,
              smi_used_mib: Optional[float] = None, **extra) -> Dict:
        row = dict(point=name, allocated=int(allocated), reserved=int(reserved), requested=None if requested is None else int(requested),
                   device_used=int(device_used), device_total=int(device_total), nvidia_smi_used_mib=smi_used_mib, **extra)
        self.points.append(row)
        return row

    def begin(self, name: str):
        if self._open is not None:
            raise RuntimeError("window %r opened while %r is still open (windows must not overlap)" % (name, self._open))
        self._open = name

    def end(self, name: str, peak_allocated: int, peak_reserved: int, **extra) -> Dict:
        if self._open != name:
            raise RuntimeError("window %r closed but %r is open" % (name, self._open))
        self._open = None
        row = dict(window=name, peak_allocated=int(peak_allocated), peak_reserved=int(peak_reserved), **extra)
        self.windows.append(row)
        return row

    def add_window(self, name: str, peak_allocated: int, peak_reserved: int, **extra) -> Dict:
        """A window measured by the caller in one piece (reset -> work -> read), e.g. one timed step."""
        self.begin(name)
        return self.end(name, peak_allocated, peak_reserved, **extra)

    def peak_summary(self) -> Dict:
        by = OrderedDict()
        for w in self.windows:
            k = w["window"]
            if k not in by or w["peak_allocated"] > by[k]["peak_allocated"]:
                by[k] = dict(peak_allocated=w["peak_allocated"], peak_reserved=w["peak_reserved"], n=0)
        for w in self.windows:
            by[w["window"]]["n"] += 1
            by[w["window"]]["peak_reserved"] = max(by[w["window"]]["peak_reserved"], w["peak_reserved"])
        if not by:
            return dict(windows={}, max_peak_allocated=None, max_window=None, max_peak_reserved=None)
        kmax = max(by, key=lambda k: by[k]["peak_allocated"])
        return dict(windows=dict(by), max_peak_allocated=by[kmax]["peak_allocated"], max_window=kmax,
                    max_peak_reserved=max(v["peak_reserved"] for v in by.values()),
                    rule="peaks of different windows are never added; the process peak is the max over windows")


def window_family(name: str) -> str:
    """'decode:ordinary:step20' -> 'decode:ordinary' (per-step windows grouped by mode)."""
    parts = name.split(":")
    return ":".join(parts[:2]) if len(parts) > 2 else name


# ---------------------------------------------------------------------------------------------------------------
# 5. ANALYTIC HBM bytes per decode step (separate from every measured column)
# ---------------------------------------------------------------------------------------------------------------
# Activation traffic per request per layer, UPPER BOUND (no L2 reuse), bytes: (read, write). Split-K / cuBLAS
# workspaces are NOT included (unknown from code). Shapes: nosa_llama.py:545-667; out_accum (splits, B, 2, 16, 128)
# fp32 + lse from flash_api.cpp:217-246 at NOSI_ATTN_SPLITS = 4.
H2 = HIDDEN * BF16
ACC = SPLITS * Q_HEADS * HEAD_DIM * 4
LSE = SPLITS * Q_HEADS * 4
ACT_PER_REQ_LAYER = OrderedDict([
    ("input_rmsnorm", (H2, H2)),
    ("wqkv_act", (H2, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM * BF16)),
    ("nosa_linear", ((Q_HEADS + 2 * KV_HEADS) * HEAD_DIM * BF16, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM * BF16 + KV_HEADS * BF16)),
    ("rope", ((Q_HEADS + KV_HEADS) * HEAD_DIM * BF16, (Q_HEADS + KV_HEADS) * HEAD_DIM * BF16)),
    ("stage1_q", (Q_HEADS * HEAD_DIM * BF16, 0)),
    ("attention_q_accum", (Q_HEADS * HEAD_DIM * BF16, ACC + LSE)),
    ("attention_combine", (ACC + LSE, H2)),
    ("wo_act", (H2, H2)),
    ("residual_1", (2 * H2, H2)),
    ("post_rmsnorm", (H2, H2)),
    ("gate_up_act", (H2, 2 * INTER * BF16)),
    ("silu_and_mul", (2 * INTER * BF16, INTER * BF16)),
    ("down_act", (INTER * BF16, H2)),
    ("residual_2", (2 * H2, H2)),
])
LOGITS_PER_REQ = OrderedDict([("final_rmsnorm", (H2, H2)), ("lm_head_out", (H2, VOCAB * BF16)), ("logits_float", (VOCAB * BF16, VOCAB * 4)),
                              ("embedding_out", (0, H2))])
# selection metadata per request per layer: diff reads old map + topk_idx and writes new map + load mask; the copy_
# reads and writes one map (verify_alone.py:204-210: 6 x H x 64 x 8 B); the gathers read the load mask twice.
MAP_BYTES = KV_HEADS * TOPK * 8
META_PER_REQ_LAYER = OrderedDict([("diff_offload", (2 * MAP_BYTES, 2 * MAP_BYTES)), ("block_map_copy", (MAP_BYTES, MAP_BYTES)),
                                  ("gather_load_mask_reads", (2 * MAP_BYTES, 0))])
POOL_META_PER_REQ_LAYER = lambda P: (2 * MAP_BYTES + 2 * KV_HEADS * P * 8, MAP_BYTES + KV_HEADS * TOPK + MAP_BYTES + 2 * KV_HEADS * P * 8)   # pool_update maps/ages/target/action


def scoring_bytes_per_req_layer(M_c: int) -> Tuple[int, int]:
    """Stage-1 output (2, ~ceil128(M_c)) bf16, its copy into score_buf, the captured pooling / top-k graph
    (nosa_llama.py:465-497), compressed_cis reads. APPROXIMATE (+-30%); read, write."""
    cols = int(math.ceil(M_c / 128.0) * 128)
    score = KV_HEADS * cols * BF16
    nblk = int(math.ceil((M_c * 16 + 64) / 64.0))       # pooled blocks of the context (~253 at 16K)
    pool_rw = (score + KV_HEADS * M_c * BF16 + 3 * KV_HEADS * nblk * BF16, 4 * KV_HEADS * nblk * BF16)
    topk = (2 * KV_HEADS * nblk * BF16 + KV_HEADS * nblk, KV_HEADS * (33 + 64) * (BF16 + 8) + 2 * KV_HEADS * nblk)
    return score + pool_rw[0] + topk[0], 2 * score + pool_rw[1] + topk[1]


def analytic_step_bytes(B: int, C: int, loads: float, tail_rows: int, compress_keys: int, burst: bool = False,
                        pool_swaps: float = 0.0, pool_moves: float = 0.0, weight_read_bytes: Optional[int] = None,
                        small_weight_bytes: int = LAYERS * (2 * H2 + 2 * 256 * BF16 + 4) + H2, layers: int = LAYERS) -> List[Dict]:
    """ANALYTIC HBM bytes of ONE decode step, by category (never mixed with counter columns).
      loads = measured host arrivals (block-heads, from the engines' load masks); pool_swaps / pool_moves = measured
      pool actions (SWAP; MOVE_IN + MOVE_OUT); tail_rows = rows in the tail slot when attention runs (after the tail
      write); compress_keys = compressed keys per request (stage-1 reads them all); burst = the step re-concatenated the
      compressed-K cache (torch.cat, cache_engine.py:970)."""
    P = C - (TOPK - 1)
    rows = []

    def add(cat, sub, rd, wr, basis):
        rows.append(dict(category=cat, subcategory=sub, read_bytes=float(rd), write_bytes=float(wr), basis=basis))
    w = GEMM_WEIGHT_BYTES if weight_read_bytes is None else weight_read_bytes
    add("weights", "gemm_weights_read_once", w, 0, "wqkv/wo/gate_up/down x 32 + lm_head, bf16, each read once per step")
    add("weights", "norms_and_cis_gate", small_weight_bytes, 0, "RMSNorm weights, delta, A")
    add("weights", "embedding_rows", B * H2, 0, "B rows of the embedding table")
    add("kv_historical", "window_read_63_slots", B * layers * (TOPK - 1) * SLOT_BYTES, 0, "attention reads K + V + bias of slots 0..62 (cache_seqlens)")
    add("kv_historical", "arrival_writes", 0, loads * LOAD_HBM_WRITE_BYTES, "loads x (32 KiB K+V + 128 B bias) written by the gathers")
    add("kv_historical", "bias_source_reads", loads * LOAD_BIAS_BYTES, 0, "gather_bias reads 64 bf16 per load from total_cis (device)")
    if P > 0:
        add("kv_historical", "victim_pool_moves_d2d", pool_swaps * POOL_SWAP_BYTES + pool_moves * POOL_MOVE_BYTES,
            pool_swaps * POOL_SWAP_BYTES + pool_moves * POOL_MOVE_BYTES, "SWAP 2r+2w, MOVE 1r+1w block-heads of 32,896 B (flash_pool_swap)")
    add("kv_active_tail", "tail_rows_read", B * layers * tail_rows * ROW_BYTES, 0, "attention reads the tail slot's filled rows")
    add("kv_active_tail", "tail_row_write", 0, B * layers * (ROW_BYTES + 2 * KV_HEADS * BF16), "new K/V/bias row + total_cis and tail_cis rows")
    ck = B * layers * compress_keys * KV_HEADS * HEAD_DIM * BF16
    add("kv_compressed_selection", "stage1_compressed_k_read", ck, 0, "stage 1 reads every compressed key of every request")
    sr, sw = scoring_bytes_per_req_layer(compress_keys)
    add("kv_compressed_selection", "scoring_and_pooling_buffers", B * layers * sr, B * layers * sw, "APPROXIMATE: score output, score_buf copy, pooling/top-k graph")
    add("kv_compressed_selection", "uncompressed_k_ring_write", 0, B * layers * KV_HEADS * HEAD_DIM * BF16, "one key row into no_compress_k_cache")
    if burst:
        add("kv_compressed_selection", "compress_cat_burst", ck + B * layers * 32 * KV_HEADS * HEAD_DIM * BF16 * 2,
            ck + B * layers * KV_HEADS * HEAD_DIM * BF16 * (1 + 32 + 16), "torch.cat of the compressed-K cache + ring clone/shift on this step")
    ar = sum(r for r, _ in ACT_PER_REQ_LAYER.values())
    aw = sum(w_ for _, w_ in ACT_PER_REQ_LAYER.values())
    add("activations_metadata_workspace", "activations_upper_bound", B * layers * ar, B * layers * aw, "UPPER BOUND, no L2 reuse; split-K workspaces excluded")
    lr = sum(r for r, _ in LOGITS_PER_REQ.values())
    lw = sum(w_ for _, w_ in LOGITS_PER_REQ.values())
    add("activations_metadata_workspace", "logits_and_embedding_out", B * lr, B * lw, "final norm, lm_head output, .float()")
    mr = sum(r for r, _ in META_PER_REQ_LAYER.values())
    mw = sum(w_ for _, w_ in META_PER_REQ_LAYER.values())
    add("activations_metadata_workspace", "block_maps_and_load_masks", B * layers * mr, B * layers * mw, "diff_offload, map copy, gather mask reads")
    if P > 0:
        pr, pw = POOL_META_PER_REQ_LAYER(P)
        add("activations_metadata_workspace", "pool_bookkeeping", B * layers * pr, B * layers * pw, "pool_update maps/ages/target/action (APPROXIMATE)")
    return rows


def analytic_totals(rows: Sequence[Dict]) -> Dict:
    by = OrderedDict()
    for r in rows:
        c = by.setdefault(r["category"], dict(read_bytes=0.0, write_bytes=0.0))
        c["read_bytes"] += r["read_bytes"]
        c["write_bytes"] += r["write_bytes"]
    tr = sum(v["read_bytes"] for v in by.values())
    tw = sum(v["write_bytes"] for v in by.values())
    for v in by.values():
        v["pct_of_reads"] = 100.0 * v["read_bytes"] / tr if tr else float("nan")
        v["pct_of_writes"] = 100.0 * v["write_bytes"] / tw if tw else float("nan")
        v["pct_of_total"] = 100.0 * (v["read_bytes"] + v["write_bytes"]) / (tr + tw) if (tr + tw) else float("nan")
    return dict(categories=by, read_bytes=tr, write_bytes=tw, total_bytes=tr + tw,
                weight_share_of_reads=by.get("weights", {}).get("read_bytes", 0.0) / tr if tr else float("nan"))


def pcie_step_bytes(loads: float) -> Dict:
    """CPU-GPU transfer per step (separate table): the gathers read loads x 32 KiB of pinned host memory over PCIe."""
    return dict(h2d_bytes=loads * LOAD_WIRE_BYTES, d2h_bytes=0.0, basis="loads x 32,768 B (engines' load masks)")


# ---------------------------------------------------------------------------------------------------------------
# 6. Nsight Compute parsing and attribution
# ---------------------------------------------------------------------------------------------------------------
_UNIT = {"byte": 1.0, "bytes": 1.0, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12,
         "kibyte": 1024.0, "mibyte": 1024.0 ** 2, "gibyte": 1024.0 ** 3,
         "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0,
         "sector": 1.0, "ksector": 1e3, "msector": 1e6, "gsector": 1e9,
         "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "cycle": 1.0, "kcycle": 1e3, "mcycle": 1e6}


def unit_scale(unit: str) -> float:
    """Multiplier to base units (bytes, seconds, sectors). ncu 2025.1 prints scaled units ('Mbyte', 'ms', 'us',
    'Kbyte', 'Gbyte') even in CSV exports (E5 raw.csv; scripts/gemm_ncu_probe.py:65-69 fixed 'ms')."""
    u = (unit or "").strip().lower()
    if u in _UNIT:
        return _UNIT[u]
    if u.endswith("/second") or u.endswith("/s") or u in ("%", ""):
        return 1.0
    return 1.0


def to_number(v) -> float:
    s = str(v).strip().replace(",", "")
    if s in ("", "n/a", "N/A", "-"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_ncu_csv(text: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """ncu --import rep --csv --page raw: row 0 = column names, row 1 = units (may be all empty), then one row per
    kernel. Lines ncu prints before the table (==PROF==, warnings) are skipped."""
    lines = [l for l in text.splitlines() if l.startswith('"')]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if not rows:
        return [], [], []
    hdr = rows[0]
    units, data = [""] * len(hdr), rows[1:]
    if data and ("ID" in hdr) and (data[0][hdr.index("ID")] == ""):
        units, data = data[0], data[1:]
    return hdr, units, data


METRIC_COLUMNS = OrderedDict([
    ("dram_read_bytes", "dram__bytes_read.sum"), ("dram_write_bytes", "dram__bytes_write.sum"), ("gpu_time_s", "gpu__time_duration.sum"),
    ("l2_device_read_sectors", "lts__t_sectors_aperture_device_op_read.sum"), ("l2_device_write_sectors", "lts__t_sectors_aperture_device_op_write.sum"),
    ("l2_sysmem_tex_read_sectors", "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum"),
    ("pcie_read_bytes", "pcie__read_bytes.sum"), ("pcie_write_bytes", "pcie__write_bytes.sum"),
])
LAUNCH_COLUMNS = ("launch__grid_size", "launch__block_size", "launch__registers_per_thread", "launch__waves_per_multiprocessor", "launch__stream_id")


def ncu_kernel_rows(raw_text: str, nvtx_text: Optional[str] = None) -> List[Dict]:
    """Normalized per-kernel rows (base units) + the raw metric values kept. The NVTX-renamed export (same report
    imported with --print-nvtx-rename kernel) is joined by ID: its 'Kernel Name' is the innermost push/pop range."""
    hdr, units, data = parse_ncu_csv(raw_text)
    col = {h: i for i, h in enumerate(hdr)}
    ren = {}
    if nvtx_text:
        h2, _, d2 = parse_ncu_csv(nvtx_text)
        c2 = {h: i for i, h in enumerate(h2)}
        kn = "Kernel Name" if "Kernel Name" in c2 else ("Function Name" if "Function Name" in c2 else None)
        if kn and "ID" in c2:
            ren = {r[c2["ID"]]: r[c2[kn]] for r in d2 if len(r) > max(c2["ID"], c2[kn])}
    kname = "Kernel Name" if "Kernel Name" in col else ("Function Name" if "Function Name" in col else None)
    out = []
    for r in data:
        if len(r) < len(hdr):
            continue
        rid = r[col["ID"]] if "ID" in col else str(len(out))
        name = r[col[kname]] if kname else "?"
        row = OrderedDict(id=int(to_number(rid)) if to_number(rid) == to_number(rid) else len(out), kernel=name,
                          nvtx_name=ren.get(rid, ""), grid=r[col["Grid Size"]] if "Grid Size" in col else "",
                          block=r[col["Block Size"]] if "Block Size" in col else "", stream=r[col["Stream"]] if "Stream" in col else "")
        for key, metric in METRIC_COLUMNS.items():
            if metric in col:
                i = col[metric]
                row[key] = to_number(r[i]) * unit_scale(units[i] if i < len(units) else "")
                row[key + "__raw"] = r[i] + ((" " + units[i]) if i < len(units) and units[i] else "")
            else:
                row[key] = float("nan")
        for c in LAUNCH_COLUMNS:
            if c in col:
                row[c] = r[col[c]]
        out.append(row)
    return out


_PASS_RE = re.compile(r'==PROF== Profiling "(?P<name>[^"]*)"(?: - (?P<idx>\d+))?:.*?(?P<passes>\d+) pass(?:es)?')


def parse_ncu_passes(log_text: str) -> List[Dict]:
    """'==PROF== Profiling "kernel" - 3: 0%....50%....100% - 6 passes' -> [{name, index, passes}]."""
    out = []
    for line in log_text.splitlines():
        m = _PASS_RE.search(line)
        if m:
            out.append(dict(name=m.group("name"), index=int(m.group("idx")) if m.group("idx") else None, passes=int(m.group("passes"))))
    return out


def single_pass(log_text: str) -> Optional[bool]:
    p = parse_ncu_passes(log_text)
    if not p:
        return None
    return all(x["passes"] == 1 for x in p)


def split_windows(rows: Sequence[Dict], marker: str = MARKER_KERNEL) -> List[Dict]:
    """Profiled windows delimited by runs of marker kernels: a run of k markers opens window k's rows (the driver
    launches k = window index + 1 markers at the start of each cudaProfilerStart/Stop window). Marker rows are
    dropped from the windows."""
    wins, cur = [], None
    run = 0
    for r in rows:
        if marker in (r.get("kernel") or ""):
            if cur is not None and cur["rows"]:
                wins.append(cur)
                cur = None
            run += 1
            continue
        if run:
            cur = dict(markers=run, rows=[])
            run = 0
        if cur is None:
            cur = dict(markers=0, rows=[])
        cur["rows"].append(r)
    if cur is not None and (cur["rows"] or cur["markers"]):
        wins.append(cur)
    return wins


# Kernel families: retroinfer-eval scripts/verify_timeline_report.py:75-84 (FIRST MATCH WINS), plus the marker.
FAMILY_TABLE = (
    ("marker", re.compile(r"spin_kernel")),
    ("gather", re.compile(r"flash_h2d")),
    ("diff", re.compile(r"diff_kernel")),
    ("scoring", re.compile(r"stage1_kernel|nosa_pool_kernel|gatherTopK|radixSelect|topk(?!_to_uint64)|bitonic", re.I)),
    ("pool", re.compile(r"pool_\w*kernel")),
    ("attention", re.compile(r"flash_fwd|topk_to_uint64|mha_|attn", re.I)),
    ("gemm", re.compile(r"gemm|gemv|matmul|cutlass|cublas|nvjet|splitkreduce", re.I)),
    ("norm_rope_act", re.compile(r"RMSNorm|Rotary|rope|act_and_mul|nosa_linear_kernel|silu", re.I)),
    ("elementwise", re.compile(r"elementwise|reduce_kernel|indexSelect|index_put|scatter|fill|copy|CatArray|softmax|exp|where|masked|arange|cumsum|unique|iota|sum|argmax|compare|clamp|sort|nonzero|Functor", re.I)),
)


def kernel_family(name: str) -> str:
    for fam, rx in FAMILY_TABLE:
        if rx.search(name or ""):
            return fam
    return "other"


# What the DRAM bytes of a (range, family) can contain. A LABEL for the reader, never used to split bytes.
CONTENT_HINTS = {
    ("linear", "gemm"): "wqkv weights + activations (+ split-K partials): counters cannot separate",
    ("ffn", "gemm"): "gate_up / down weights + activations (+ split-K partials): counters cannot separate",
    ("stage 2", "attention"): "KV window rows (historical + tail) + q + split accumulators: counters cannot separate",
    ("offloading update", "gather"): "arrival writes + load-mask / kv_bias-source reads (host reads are sysmem, not DRAM)",
    ("offloading update", "pool"): "victim-pool block moves (D2D) + pool maps",
    ("offloading update", "diff"): "block maps and load masks",
    ("stage 1", "scoring"): "compressed keys + q + scores",
    ("step", "gemm"): "wo or lm_head weights + activations (range outside the model's NVTX ranges)",
}


def attribute(row: Dict, model_ranges: Sequence[str] = MODEL_RANGES) -> Dict:
    """range + family + ATTRIBUTION METHOD of one ncu kernel row:
      nvtx-module    the innermost NVTX range is one of the model's decode ranges (nosa_llama.py:547-667)
      kernel-family  the kernel ran outside the model's ranges (inside the driver's 'hbm_step:*' range or none) and
                     only its name classifies it (wo, residual, pooling-graph replay, embedding, final norm, lm_head)
      unknown        neither"""
    nv = (row.get("nvtx_name") or "").strip()
    fam = kernel_family(row.get("kernel") or "")
    if nv in model_ranges:
        rng, method = nv, "nvtx-module"
    elif nv.startswith(STEP_RANGE_PREFIX) or nv == "" or nv == row.get("kernel"):
        rng, method = "step", ("kernel-family" if fam != "other" else "unknown")
    else:
        rng, method = "nvtx:" + nv, ("nvtx-other" if fam != "other" else "unknown")
    hint = CONTENT_HINTS.get((rng, fam), "")
    return dict(range=rng, family=fam, attribution_method=method, content_hint=hint)


def summarize_window(rows: Sequence[Dict]) -> Dict:
    """Totals of one profiled window: all kernels, and per (range, family). NaN metrics (not collected) stay NaN."""
    def tot(rs, k):
        vals = [r.get(k, float("nan")) for r in rs]
        if all(v != v for v in vals):
            return float("nan")
        return sum(v for v in vals if v == v)
    groups: Dict[Tuple[str, str, str], List[Dict]] = OrderedDict()
    for r in rows:
        a = attribute(r)
        groups.setdefault((a["range"], a["family"], a["attribution_method"]), []).append(r)
    by = []
    for (rng, fam, meth), rs in groups.items():
        d = dict(range=rng, family=fam, attribution_method=meth, n_kernels=len(rs), content_hint=CONTENT_HINTS.get((rng, fam), ""))
        for k in METRIC_COLUMNS:
            d[k] = tot(rs, k)
        by.append(d)
    total = dict(n_kernels=len(rows))
    for k in METRIC_COLUMNS:
        total[k] = tot(rows, k)
    total["l2_device_write_arrival_bytes"] = total["l2_device_write_sectors"] * 32 if total["l2_device_write_sectors"] == total["l2_device_write_sectors"] else float("nan")
    total["sysmem_read_bytes"] = total["l2_sysmem_tex_read_sectors"] * 32 if total["l2_sysmem_tex_read_sectors"] == total["l2_sysmem_tex_read_sectors"] else float("nan")
    return dict(total=total, by_range_family=by)


# ---------------------------------------------------------------------------------------------------------------
# 7. torch.profiler chrome trace -> kernel rows with shapes; order join to ncu
# ---------------------------------------------------------------------------------------------------------------
def load_trace(path: str) -> Dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def profiler_kernel_rows(trace: Dict) -> List[Dict]:
    """Kernels of a chrome trace in launch order, each with the innermost host op (cpu_op) that enclosed its launch
    and that op's 'Input Dims' (torch.profiler record_shapes=True). Correlation rule as in retroinfer-eval
    scripts/verify_timeline_report.py:30-45 (kernel args.correlation -> the runtime launch -> innermost enclosing
    cpu_op on the launching thread). Memcpy / memset events are returned with family memcpy_* / memset."""
    ev = [e for e in trace.get("traceEvents", []) if isinstance(e, dict) and e.get("ph") == "X"]
    launches = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                launches[c] = e
    ops_by_tid: Dict = defaultdict(list)
    for e in ev:
        if e.get("cat") == "cpu_op":
            ops_by_tid[(e.get("pid"), e.get("tid"))].append(e)
    for v in ops_by_tid.values():
        v.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))

    starts = {k: [o["ts"] for o in v] for k, v in ops_by_tid.items()}

    def innermost(tid, ts):
        """The latest-starting cpu_op on the thread that still contains ts (nested ops start after their parents)."""
        import bisect
        ops = ops_by_tid.get(tid, ())
        i = bisect.bisect_right(starts.get(tid, []), ts) - 1
        while i >= 0:
            o = ops[i]
            if o["ts"] <= ts <= o["ts"] + o.get("dur", 0):
                return o
            i -= 1
        return None
    out = []
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    gpu.sort(key=lambda e: e["ts"])
    for e in gpu:
        a = e.get("args") or {}
        l = launches.get(a.get("correlation"))
        op = innermost((l.get("pid"), l.get("tid")), l["ts"]) if l else None
        oa = (op or {}).get("args") or {}
        fam = kernel_family(e.get("name", "")) if e.get("cat") == "kernel" else ("memset" if e.get("cat") == "gpu_memset" else "memcpy_" + _mc_dir(e.get("name", "")))
        out.append(dict(name=e.get("name", ""), cat=e.get("cat"), ts=e["ts"], dur_us=e.get("dur", 0), family=fam, correlation=a.get("correlation"),
                        grid=a.get("grid"), block=a.get("block"), bytes=a.get("bytes"), launch=(l or {}).get("name"),
                        op=(op or {}).get("name"), input_dims=oa.get("Input Dims"), input_types=oa.get("Input type")))
    return out


def _mc_dir(n: str) -> str:
    for d in ("HtoD", "DtoD", "DtoH"):
        if d in n:
            return d.lower()
    return "other"


def short_name(name: str, width: int = 96) -> str:
    n = re.sub(r"^void ", "", name or "")
    n = re.sub(r"<.*", "", n)
    n = re.sub(r"\(.*", "", n)
    n = n.split("::")[-1] if "::" in n else n
    return n.strip()[:width]


def join_shapes(ncu_rows: Sequence[Dict], prof_rows: Sequence[Dict]) -> Dict:
    """Join ncu kernel rows (one profiled window, markers dropped) to the torch.profiler kernels of the same step
    type BY ORDER, gated on the kernel names matching 1:1 (short names). Memcpy / memset profiler rows are skipped
    (ncu does not profile copy-engine work). Returns the joined rows only when the gate passes."""
    pk = [r for r in prof_rows if r.get("cat") == "kernel" and r.get("family") != "marker"]
    nk = [r for r in ncu_rows if kernel_family(r.get("kernel", "")) != "marker"]
    if len(pk) != len(nk):
        return dict(gate="FAIL", reason="kernel counts differ: ncu %d vs profiler %d" % (len(nk), len(pk)), rows=[])
    bad = [i for i, (a, b) in enumerate(zip(nk, pk)) if short_name(a.get("kernel", "")) != short_name(b.get("name", ""))]
    if bad:
        i = bad[0]
        return dict(gate="FAIL", reason="%d name mismatches; first at %d: ncu %r vs profiler %r" % (len(bad), i, short_name(nk[i].get("kernel", "")), short_name(pk[i].get("name", ""))), rows=[])
    rows = []
    for a, b in zip(nk, pk):
        rows.append(dict(a, op=b.get("op"), input_dims=b.get("input_dims"), input_types=b.get("input_types"), shape_source="torch.profiler record_shapes, joined by order"))
    return dict(gate="PASS", reason="", rows=rows)


# ---------------------------------------------------------------------------------------------------------------
# 8. export
# ---------------------------------------------------------------------------------------------------------------
def _cell(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    if v is None:
        return ""
    return v


def write_csv(path: str, rows: Sequence[Dict], columns: Optional[Sequence[str]] = None) -> str:
    cols = list(columns) if columns else []
    if not cols:
        seen = OrderedDict()
        for r in rows:
            for k in r:
                seen.setdefault(k, None)
        cols = list(seen)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in cols})
    return path


def read_csv(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sanitize(obj):
    """NaN / inf -> None (strict JSON for other tools), recursively; tuples become lists."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def num(x) -> float:
    """None / '' / non-numeric -> NaN (the inverse of sanitize for reading back)."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def write_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sanitize(obj), f, indent=1, default=_json_default, allow_nan=False)
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, float) and o != o:
        return None
    try:
        return float(o)
    except Exception:
        return str(o)


def append_jsonl(path: str, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(sanitize(obj), default=_json_default, allow_nan=False) + "\n")
        f.flush()


def read_jsonl(path: str) -> List[Dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
