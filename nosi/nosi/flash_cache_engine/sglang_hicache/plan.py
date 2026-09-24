"""Index plans, grid formulas and CPU twins for the two vendored SGLang HiCache IO kernels (sglang 87db743).
Pure torch and device-agnostic: the same builders run on the CPU (tests) and on the GPU (the harness).

THE KERNELS (line numbers: retroinfer-eval scripts/sglang_hicache/{transfer_upstream_87db743.cu (:NNN),
hicache_upstream_87db743.cuh (H:NNN)})
  AOT  transfer_kernel_impl (:260-308), launcher (:310-397). CLOSEST RELEASED IMPLEMENTATION of Strata's IO
       kernel (Strata's first author; the paper releases no code). grid: items_per_warp = ceil(N / (quota * warps)),
       blocks = ceil(N / (items_per_warp * warps)) <= quota (:338-341); warp w copies items
       [w * ipw, (w + 1) * ipw) (STATIC CONTIGUOUS split, :283-289); one warp per item, lane j moves 8-B words
       j, j + 32, ... (:28-31); K, then V at the same indices (:293-305). Offsets:
         lf  (layer_first, :90-100)  record = base + index * item_size
         pf  (page_first,  :102-112) record = base + index * page_dim + layer_id * item_size
  JIT  hicache_transfer_per_layer (H:181-218), SGLang's DEFAULT load at 87db743 (not a Strata author).
       threads per item = 32 / unroll (unroll 4 for elements <= 512 B, hicache.py :124-132); workers per block =
       block / threads-per-item; grid = min(ceil(N / workers_per_block), quota) (H:333-334); worker i copies items
       i, i + workers, ... (ROUND-ROBIN grid-stride, H:203-204); record = base + index * row_stride on both sides
       (H:207-208), K then V.

THE PAYLOADS
  native (SGLang's own form, the N0 control): page_size 1, one item = one token row of BOTH KV heads (512 B at NOSA-8B:
       H = 2, Dh = 128, bf16), K and V at the same index; per request `count` runs of `block_rows` consecutive HOST
       slots (a prefix's host slots are allocated consecutively) into scattered DEVICE rows; the SAME index tensors
       serve every layer (SGLang builds one plan per load op and reuses it for all layers, l2_transfer.py :88-110).
  NOSI adapter (the S2 arms): NOSI's miss descriptor load_ids (H, B, M): slot m of (head h, request b) receives host
       block blk. Host cache per layer (B, S_cpu, H, Dh), device window / scratch (B, S_dst, H, Dh).
         i256 (EXACT for any descriptor, per-head items): flatten (token, head) into 256-B rows,
              src = (b * S_cpu + blk * R + r) * H + h,  dst = (b * S_dst + m * R + r) * H + h
         i512 (token rows of both heads, SGLang's native item): src = b * S_cpu + blk * R + r, dst = b * S_dst + m * R + r;
              EXACT ONLY when every head loads the same block into the same slot (coupled); otherwise refused.
       Items are in HOST ADDRESS ORDER within a (request, slot): (b, m, r, h) for i256, (b, m, r) for i512.

THE PLAN COST: every plan here is PREBUILT outside the timed side (the 'prebuilt-plan copy microbenchmark, NOT an
integrated LRU getter baseline'). timed_native_build / timed_nosi_build time the construction separately.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch

WARP = 32
AOT_DEFAULT_QUOTA = 2              # kvcacheio.py :39 (CUDA)
AOT_DEFAULT_WARPS = 32             # kvcacheio.py :40 (CUDA): 32 warps = 1024 threads
JIT_DEFAULT_QUOTA = 2              # ops/kvcache/hicache.py :21 (CUDA)
JIT_BLOCK = 1024                   # ops/kvcache/hicache.py :30
LAYOUTS = ("lf", "pf")
ITEMS = {"i256": 256, "i512": 512}
PLAN_COST_LABEL = "plan EXCLUDED from the timed side (prebuilt indices; construction timed separately)"
BENCH_LABEL = "prebuilt-plan copy microbenchmark, NOT an integrated LRU getter baseline"


class PlanRefused(ValueError):
    """The requested item is not an exact copy of this descriptor."""


# --------------------------------------------------------------------------- grids and walks
def aot_grid(num_items: int, block_quota: int = AOT_DEFAULT_QUOTA, num_warps: int = AOT_DEFAULT_WARPS) -> Tuple[int, int]:
    """(items_per_warp, num_blocks) of transfer_kv_launcher (:338-341)."""
    if num_items <= 0 or block_quota <= 0 or num_warps <= 0:
        raise ValueError("positive arguments required")
    ipw = -(-num_items // (block_quota * num_warps))
    blocks = -(-num_items // (ipw * num_warps))
    return ipw, blocks


def default_unroll(element_size: int) -> int:
    """ops/kvcache/hicache.py :124-132."""
    return 4 if element_size <= 512 else (2 if element_size <= 1024 else 1)


def jit_grid(num_items: int, block_quota: int = JIT_DEFAULT_QUOTA, unroll: int = 4, block: int = JIT_BLOCK) -> int:
    """num_blocks of HiCacheKernel::run_one (H:333-334)."""
    if num_items <= 0:
        raise ValueError("length must be positive")
    per_block = block // (WARP // unroll)
    return min(-(-num_items // per_block), block_quota)


def aot_walk(num_items: int, block_quota: int = AOT_DEFAULT_QUOTA, num_warps: int = AOT_DEFAULT_WARPS) -> Iterator[Tuple[int, int]]:
    """(global warp id, item id) pairs transfer_kernel_impl visits (:279-289), in each warp's order."""
    ipw, blocks = aot_grid(num_items, block_quota, num_warps)
    for w in range(blocks * num_warps):
        for i in range(ipw):
            item = w * ipw + i
            if item >= num_items:
                break
            yield w, item


def jit_walk(num_items: int, block_quota: int = JIT_DEFAULT_QUOTA, unroll: int = 4, block: int = JIT_BLOCK) -> Iterator[Tuple[int, int]]:
    """(worker id, item id) pairs hicache_transfer_per_layer visits (H:203-204)."""
    blocks = jit_grid(num_items, block_quota, unroll, block)
    per_block = block // (WARP // unroll)
    workers = per_block * block_quota          # kNumWorkers uses the QUOTA, not the launched grid (H:195)
    for wid in range(blocks * per_block):
        for i in range(wid, num_items, workers):
            yield wid, i


def walk_covers_once(pairs: Iterator[Tuple[int, int]], num_items: int) -> bool:
    seen = [0] * num_items
    for _, i in pairs:
        seen[i] += 1
    return all(c == 1 for c in seen)


# --------------------------------------------------------------------------- CPU twins (byte level)
def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """The bytes of a CONTIGUOUS tensor's storage from its first element (the kernels see base = data_ptr)."""
    if not t.is_contiguous():
        raise ValueError("the byte view needs a contiguous tensor")
    return t.view(-1).view(torch.uint8)


def simulate_aot(src_k, dst_k, src_v, dst_v, src_idx, dst_idx, item_size: int, block_quota: int = AOT_DEFAULT_QUOTA,
                 num_warps: int = AOT_DEFAULT_WARPS, layout: str = "lf", layer_id: int = 0, src_layout_dim: int = 0) -> None:
    """CPU twin of transfer_kernel_impl for the per-layer entry points: walk exactly as the kernel (aot_walk) and move
    item_size bytes per (item, tensor) at the offsets of get_global_offset_lf / _pf (dst always lf)."""
    if layout not in LAYOUTS:
        raise ValueError("layout must be lf or pf")
    sk, dk, sv, dv = (_flat_u8(t) for t in (src_k, dst_k, src_v, dst_v))
    s_idx, d_idx = src_idx.tolist(), dst_idx.tolist()
    if len(s_idx) != len(d_idx):
        raise ValueError("index lengths differ")
    for _, item in aot_walk(len(s_idx), block_quota, num_warps):
        s = s_idx[item] * (src_layout_dim if layout == "pf" else item_size) + (layer_id * item_size if layout == "pf" else 0)
        d = d_idx[item] * item_size
        dk[d:d + item_size] = sk[s:s + item_size]
        dv[d:d + item_size] = sv[s:s + item_size]


def simulate_jit(k_dst2d, v_dst2d, idx_dst, k_src2d, v_src2d, idx_src, block_quota: int = JIT_DEFAULT_QUOTA) -> None:
    """CPU twin of hicache_transfer_per_layer on 2-D (-1, D) views (any row stride, unit inner stride): item i copies
    row idx_src[i] of the source into row idx_dst[i] of the destination, K then V, in the kernel's walk order."""
    for t in (k_dst2d, v_dst2d, k_src2d, v_src2d):
        if t.dim() != 2 or t.stride(1) != 1:
            raise ValueError("2-D views with unit inner stride required (H:298-309)")
    es = k_src2d.shape[1] * k_src2d.element_size()
    s_idx, d_idx = idx_src.tolist(), idx_dst.tolist()
    for _, i in jit_walk(len(s_idx), block_quota, default_unroll(es)):
        k_dst2d[d_idx[i]] = k_src2d[s_idx[i]]
        v_dst2d[d_idx[i]] = v_src2d[s_idx[i]]


# --------------------------------------------------------------------------- native payload (N0)
@dataclass
class NativePlan:
    host_idx: torch.Tensor           # int64 [N] host token slot per item
    dev_idx: torch.Tensor            # int64 [N] device row per item
    counts: List[int]                # host runs per request
    block_rows: int
    item_size: int
    n_items: int
    host_tokens_per_request: int
    dev_rows_per_request: int
    dev_mode: str

    @property
    def bytes_per_layer(self) -> int:
        return 2 * self.n_items * self.item_size          # K and V

    def per_request_items(self) -> List[int]:
        return [c * self.block_rows for c in self.counts]


def native_counts(B: int, runs_per_request: float, seed: int) -> List[int]:
    """floor(runs) + 1 runs for round(frac * B) requests (seeded choice), floor(runs) for the rest: the registration's
    3.5 runs of 64 tokens per request (the nosi16 miss volume, coupled heads)."""
    g = torch.Generator().manual_seed(int(seed))
    base = int(math.floor(runs_per_request))
    extra = int(round((runs_per_request - base) * B))
    counts = torch.full((B,), base, dtype=torch.int64)
    counts[torch.randperm(B, generator=g)[:extra]] += 1
    return [int(c) for c in counts]


def build_native(B: int, host_tokens_per_request: int, dev_rows_per_request: int, runs_per_request: float = 3.5,
                 block_rows: int = 64, item_size: int = 512, seed: int = 0, dev_mode: str = "scattered") -> NativePlan:
    """SGLang-native load plan (see the module docstring). Request b owns host slots [b * S_h, (b + 1) * S_h) and device
    rows [b * S_d, (b + 1) * S_d). Host: `count` distinct runs of block_rows consecutive slots, run starts at
    multiples of block_rows (uniform, seeded). Device: 'scattered' = distinct rows drawn uniformly from the request's
    range (SGLang's page_size-1 allocator after churn), 'runs' = distinct block_rows-aligned runs."""
    if host_tokens_per_request % block_rows or dev_rows_per_request % block_rows:
        raise ValueError("host tokens and device rows per request must be whole blocks")
    if dev_mode not in ("scattered", "runs"):
        raise ValueError("dev_mode must be scattered or runs")
    counts = native_counts(B, runs_per_request, seed)
    g = torch.Generator().manual_seed(int(seed) + 7919)
    n_host_blocks, n_dev_blocks = host_tokens_per_request // block_rows, dev_rows_per_request // block_rows
    if max(counts) > n_host_blocks or max(counts) * block_rows > dev_rows_per_request:
        raise ValueError("%d runs do not fit %d host blocks / %d device rows" % (max(counts), n_host_blocks, dev_rows_per_request))
    hs, ds = [], []
    r = torch.arange(block_rows, dtype=torch.int64)
    for b, c in enumerate(counts):
        blks = torch.randperm(n_host_blocks, generator=g)[:c].to(torch.int64)
        hs.append((b * host_tokens_per_request + blks[:, None] * block_rows + r[None, :]).reshape(-1))
        if dev_mode == "scattered":
            rows = torch.randperm(dev_rows_per_request, generator=g)[:c * block_rows].to(torch.int64)
        else:
            rows = (torch.randperm(n_dev_blocks, generator=g)[:c].to(torch.int64)[:, None] * block_rows + r[None, :]).reshape(-1)
        ds.append(b * dev_rows_per_request + rows)
    host_idx, dev_idx = torch.cat(hs), torch.cat(ds)
    return NativePlan(host_idx=host_idx, dev_idx=dev_idx, counts=counts, block_rows=block_rows, item_size=item_size, n_items=int(host_idx.numel()),
                      host_tokens_per_request=host_tokens_per_request, dev_rows_per_request=dev_rows_per_request, dev_mode=dev_mode)


def timed_native_build(**kw) -> Tuple[NativePlan, Dict]:
    """The released caller's plan cost, once per load op (NEVER per layer): SGLang concatenates the prefix nodes' CPU
    int64 host indices (hiradix_cache.py :1410, cache_controller.py :157-158) and uploads them with
    .to(device, non_blocking=True) from PAGEABLE memory (cache_controller.py :862-865; synchronous in effect, :889-892).
    Here: host build (the seeded draw + cat) and the two pageable uploads, timed separately."""
    t0 = time.perf_counter()
    p = build_native(**kw)
    t1 = time.perf_counter()
    dev = torch.device("cuda")
    h = p.host_idx.to(dev, non_blocking=True)
    d = p.dev_idx.to(dev, non_blocking=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return p, dict(build_host_ms=1000 * (t1 - t0), upload_ms=1000 * (t2 - t1), n_items=p.n_items, once_per="load op (all layers)",
                   host_idx_gpu=h, dev_idx_gpu=d)


# --------------------------------------------------------------------------- NOSI adapter (S2)
def _loads(load_ids: torch.Tensor):
    if load_ids.dim() != 3:
        raise ValueError("load_ids must be (H, B, M)")
    h, b, m = (load_ids >= 0).nonzero(as_tuple=True)
    blk = load_ids[h, b, m].to(torch.int64)
    return h.to(torch.int64), b.to(torch.int64), m.to(torch.int64), blk


def nosi_items(load_ids: torch.Tensor, *, s_cpu: int, s_dst: int, block_rows: int = 64, item: str = "i256") -> Tuple[torch.Tensor, torch.Tensor]:
    """(src_idx, dst_idx) int64 in the item's record units, host address order within a (request, slot).
    i256: per-head rows, exact for any descriptor. i512: token rows of both heads, refused unless coupled."""
    if item not in ITEMS:
        raise ValueError("item must be one of %s" % sorted(ITEMS))
    H, B, M = load_ids.shape
    if M * block_rows > s_dst:                                 # per-row addressing: no whole-block requirement on s_cpu / s_dst
        raise ValueError("M slots of %d rows exceed s_dst %d" % (block_rows, s_dst))
    if bool(((load_ids.to(torch.int64) + 1) * block_rows > s_cpu).any()):
        raise ValueError("a host block id is past the host cache")
    r = torch.arange(block_rows, dtype=torch.int64, device=load_ids.device)
    if item == "i512":
        ref = load_ids[0]
        if not bool((load_ids == ref.unsqueeze(0)).all()):
            raise PlanRefused("i512 (token rows of both heads) is exact only when every head loads the same block into the same slot")
        b, m = (ref >= 0).nonzero(as_tuple=True)
        blk = ref[b, m].to(torch.int64)
        b, m = b.to(torch.int64), m.to(torch.int64)
        order = torch.argsort(b * M + m)
        b, m, blk = b[order], m[order], blk[order]
        src = (b[:, None] * s_cpu + blk[:, None] * block_rows + r[None, :]).reshape(-1)
        dst = (b[:, None] * s_dst + m[:, None] * block_rows + r[None, :]).reshape(-1)
        return src, dst
    h, b, m, blk = _loads(load_ids)
    src = ((b[:, None] * s_cpu + blk[:, None] * block_rows + r[None, :]) * H + h[:, None])
    dst = ((b[:, None] * s_dst + m[:, None] * block_rows + r[None, :]) * H + h[:, None])
    # order (b, m, r, h): the H heads of one token sit side by side, so this is host address order when every head of
    # a slot loads the same block; per-head slots stay exact either way (each item carries its own (b, blk, r, h))
    key = ((b * M + m)[:, None] * block_rows + r[None, :]) * H + h[:, None]
    order = torch.argsort(key.reshape(-1))
    return src.reshape(-1)[order], dst.reshape(-1)[order]


def nosi_bytes(load_ids: torch.Tensor, block_rows: int, head_dim: int, elem: int) -> int:
    """K + V bytes of one layer: every (head, request, slot) load moves block_rows x Dh x elem per tensor."""
    return 2 * int((load_ids >= 0).sum()) * block_rows * head_dim * elem


def validate_indices(src_idx: torch.Tensor, dst_idx: torch.Tensor, src_records: int, dst_records: int, item_size: int) -> None:
    """What the kernels would misread: dtype, equal length, range, and duplicate destinations (two items into one row
    race: the result is not a copy)."""
    if src_idx.dtype != torch.int64 or dst_idx.dtype != torch.int64:
        raise ValueError("indices must be int64 (the AOT kernel requires it; the JIT kernel accepts it)")
    if src_idx.numel() != dst_idx.numel() or src_idx.numel() == 0:
        raise ValueError("indices must be non-empty and of one length")
    if int(src_idx.min()) < 0 or int(src_idx.max()) >= src_records:
        raise ValueError("a source index is out of range [0, %d)" % src_records)
    if int(dst_idx.min()) < 0 or int(dst_idx.max()) >= dst_records:
        raise ValueError("a destination index is out of range [0, %d)" % dst_records)
    if int(torch.unique(dst_idx).numel()) != int(dst_idx.numel()):
        raise ValueError("duplicate destination rows")
    if item_size % 8:
        raise ValueError("item_size must be a multiple of 8")


def timed_nosi_build(load_ids_gpu: torch.Tensor, *, s_cpu: int, s_dst: int, block_rows: int, item: str, reps: int = 3) -> Dict:
    """The NOSI-adapter plan cost an INTEGRATED getter would pay per LAYER per step (NOSI's miss plans differ per layer):
    compaction of the GPU miss mask + the device -> host count sync the released launchers need (both size their grid
    from the host-visible index length, transfer.cu :339-341, H:327) + the index arithmetic. Timed on the GPU path,
    median over reps, OUTSIDE every bracket; reported as a separate component, never added to the copy rows."""
    out = []
    for _ in range(max(1, reps)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = int((load_ids_gpu >= 0).sum())                                         # the count sync
        t1 = time.perf_counter()
        s, d = nosi_items(load_ids_gpu, s_cpu=s_cpu, s_dst=s_dst, block_rows=block_rows, item=item)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        out.append((1000 * (t1 - t0), 1000 * (t2 - t1), 1000 * (t2 - t0)))
    out.sort(key=lambda x: x[2])
    c, b, t = out[len(out) // 2]
    return dict(item=item, count_sync_ms=c, index_build_ms=b, per_layer_ms=t, n_loads=n, n_items=int(s.numel()),
                note="ISOLATED plan cost on an IDLE GPU = a LOWER BOUND of the per-layer per-step cost if integrated (inside a real step the count sync would also drain the launch queue; not measured); NOSI plans differ per layer; EXCLUDED from every copy row")
