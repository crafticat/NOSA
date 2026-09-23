"""HiSparse miss plans from NOSI miss descriptors: the LAYOUT ADAPTER between NOSI's per-head block
gather and SGLang HiSparse's item copy (copy_cache_planned_kernel, sglang 87db743, hisparse.cuh
:837-892). Pure torch and device-agnostic: the same code builds a plan on the CPU (tests, the host
path) and on the GPU (the device path); only the timing differs.

WHAT EACH SIDE COPIES
  NOSI     (flash_h2d_persistent.py / flash_h2d_mask.py): for every (h, b, m) with
           load_ids[h, b, m] = blk >= 0, the 64-row tile of ONE KV head
             dst[b, m*64 : m*64+64, h, :] = src[b, blk*64 : blk*64+64, h, :]
           with src = the pinned host cache (B, S_cpu, H, Dh) and dst = the device window
           (B, S_dst, H, Dh). Slots are assigned PER HEAD, so head 0 and head 1 of one request may
           load different blocks into different slots.
  HiSparse (copy_miss_item generic path :212-224): one warp moves one ITEM of item_size_bytes,
             dev_k[dst_loc * item : +item] = host_k[src_loc * item : +item], then the same for V,
           with host and device viewed as flat arrays of equal-size records.

THE ADAPTERS (item kinds; labels are the byte sizes at NOSA-8B's geometry H = 2, Dh = 128, bf16,
64-row blocks)
  head_row  "i256"  one token row of ONE head: 256 B. EXACT for any NOSI descriptor. A per-head block
                    miss (b, h, blk, m) becomes 64 items, r = 0..63:
                      src_loc = (b*S_cpu + blk*64 + r)*H + h,  dst_loc = (b*S_dst + m*64 + r)*H + h
                    (units of 256 B: a token row holds H heads of Dh elements).
  row       "i512"  one token row of ALL heads: 512 B = HiSparse's native token item for this head count.
  block     "i32k"  one (request, block) with both heads: 64 rows x 512 B = 32 KB, contiguous on both
                    sides. src_loc = b*(S_cpu/64) + blk, dst_loc = b*(S_dst/64) + m.
  row and block are exact ONLY when, for every (b, m), either no head loads or EVERY head loads the SAME
  host block into slot m (true for the worker sweep's payload: ids 0..D-1 of both heads into scratch
  slots 0..D-1). Otherwise build_plan raises PlanRefused; it never approximates. block also needs
  S_cpu and S_dst to be multiples of the block rows (records are addressed from the tensor base).

THE PLAN (the tensors the kernel reads, hisparse_coordinator.py:296-307 shapes)
  src   int64 [B, P]  host record index per item; plan row = request b; P = plan_stride = max items per
                      request; unused tail entries = pad (-1): the kernel never reads past miss_counts[b]
  dst   int32 [B, P]  device record index per item
  counts int32 [B]    items of request b
  num_real int32 [1]  = B
Items inside a row are in NOSI's load_ids memory order: head, then slot, then row (head_row); slot,
then row (row); slot (block).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterator, Optional, Tuple

import torch

KINDS = ("head_row", "row", "block")
LABEL_KIND = {"i256": "head_row", "i512": "row", "i32k": "block"}
LABEL_BYTES = {"i256": 256, "i512": 512, "i32k": 32768}
WARP_SIZE = 32


class PlanRefused(ValueError):
    """The requested item kind is not an exact copy of this NOSI descriptor."""


def item_label(nbytes: int) -> str:
    return "i%dk" % (nbytes // 1024) if nbytes >= 1024 and nbytes % 1024 == 0 else "i%d" % nbytes


def item_bytes(kind: str, n_heads: int, head_dim: int, elem_size: int, block_rows: int) -> int:
    row = n_heads * head_dim * elem_size
    return {"head_row": head_dim * elem_size, "row": row, "block": block_rows * row}[kind]


def resolve_kind(item: str) -> Tuple[str, Optional[int]]:
    """'i256' / 'i512' / 'i32k' (labels, with the byte size they promise) or a kind name."""
    if item in LABEL_KIND:
        return LABEL_KIND[item], LABEL_BYTES[item]
    if item in KINDS:
        return item, None
    raise ValueError("unknown item %r (labels %s, kinds %s)" % (item, sorted(LABEL_KIND), KINDS))


@dataclass
class Plan:
    src: torch.Tensor
    dst: torch.Tensor
    counts: torch.Tensor
    num_real: torch.Tensor
    kind: str
    item_size_bytes: int
    n_items: int
    host_records: int
    dev_records: int

    @property
    def plan_stride(self) -> int:
        return int(self.src.stride(0))

    @property
    def label(self) -> str:
        return item_label(self.item_size_bytes)

    @property
    def bytes_per_launch(self) -> int:
        """K and V: copy_miss_item moves the K item, then the V item at the same locs (:213-222)."""
        return 2 * self.n_items * self.item_size_bytes

    def to(self, device, non_blocking: bool = False) -> "Plan":
        return replace(self, src=self.src.to(device, non_blocking=non_blocking), dst=self.dst.to(device, non_blocking=non_blocking),
                       counts=self.counts.to(device, non_blocking=non_blocking), num_real=self.num_real.to(device, non_blocking=non_blocking))


def build_plan(load_ids: torch.Tensor, *, s_cpu: int, s_dst: int, n_heads: int, head_dim: int, elem_size: int,
               block_rows: int = 64, item: str = "i256", pad: int = -1) -> Plan:
    """NOSI (H, B, M) load ids (-1 = nothing to load; slot m of request b, head h receives host block
    load_ids[h, b, m]) -> the HiSparse plan for `item`. Raises PlanRefused when the item kind cannot copy
    exactly the bytes NOSI's per-head tile copy moves, ValueError on malformed input."""
    kind, promised = resolve_kind(item)
    if load_ids.dim() != 3:
        raise ValueError("load_ids must be (H, B, M), got %s" % (tuple(load_ids.shape),))
    if load_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("load_ids must be int32 or int64, got %s" % load_ids.dtype)
    H, B, M = load_ids.shape
    if H != n_heads:
        raise ValueError("load_ids has %d heads, the cache %d" % (H, n_heads))
    if B < 1:
        raise ValueError("empty batch")
    if M * block_rows > s_dst:
        raise ValueError("M = %d slots of %d rows exceed the %d destination rows" % (M, block_rows, s_dst))
    nbytes = item_bytes(kind, n_heads, head_dim, elem_size, block_rows)
    if promised is not None and nbytes != promised:
        raise PlanRefused("%s promises %d-byte items; this geometry (H=%d, Dh=%d, elem=%d, rows=%d) gives %d"
                          % (item, promised, n_heads, head_dim, elem_size, block_rows, nbytes))
    ids = load_ids.to(torch.int64)
    valid = ids >= 0
    if bool((ids < -1).any()):
        raise ValueError("load_ids entries must be -1 or a host block id")
    n_src_blocks = s_cpu // block_rows
    if bool((ids >= n_src_blocks).any()):
        raise ValueError("a host block id >= %d: its %d rows would pass S_cpu = %d" % (n_src_blocks, block_rows, s_cpu))
    dev = ids.device
    r = torch.arange(block_rows, device=dev, dtype=torch.int64)
    if kind == "head_row":
        b_i, h_i, m_i = valid.permute(1, 0, 2).nonzero(as_tuple=True)          # row-major (b, h, m)
        blk = ids[h_i, b_i, m_i]
        src = ((b_i * s_cpu + blk * block_rows)[:, None] + r) * H + h_i[:, None]
        dst = ((b_i * s_dst + m_i * block_rows)[:, None] + r) * H + h_i[:, None]
        per_miss = block_rows
    else:
        any_h, all_h = valid.any(0), valid.all(0)
        if not torch.equal(any_h, all_h):
            n_bad = int((any_h & ~all_h).sum())
            raise PlanRefused("%s needs every head of a (request, slot) to load or none; %d (request, slot) pairs are "
                              "loaded by only some heads (NOSI assigns slots per head)" % (item, n_bad))
        same = ((ids == ids[:1]).all(0)) | ~all_h
        if not bool(same.all()):
            n_bad = int((~same).sum())
            raise PlanRefused("%s needs all heads of a slot to load ONE host block; %d (request, slot) pairs load "
                              "different blocks per head" % (item, n_bad))
        b_i, m_i = all_h.nonzero(as_tuple=True)                                  # row-major (b, m)
        blk = ids[0, b_i, m_i]
        if kind == "row":
            src = (b_i * s_cpu + blk * block_rows)[:, None] + r
            dst = (b_i * s_dst + m_i * block_rows)[:, None] + r
            per_miss = block_rows
        else:
            if s_cpu % block_rows or s_dst % block_rows:
                raise PlanRefused("%s addresses whole blocks from the tensor base: S_cpu = %d and S_dst = %d must be "
                                  "multiples of %d" % (item, s_cpu, s_dst, block_rows))
            src = (b_i * (s_cpu // block_rows) + blk)[:, None]
            dst = (b_i * (s_dst // block_rows) + m_i)[:, None]
            per_miss = 1
    host_records = B * s_cpu * n_heads * head_dim * elem_size // nbytes
    dev_records = B * s_dst * n_heads * head_dim * elem_size // nbytes
    if dev_records >= 2 ** 31:
        raise ValueError("%d device records exceed miss_dst_locs' int32" % dev_records)
    miss_per_req = torch.bincount(b_i, minlength=B)
    counts = miss_per_req * per_miss
    P = max(int(counts.max()), 1)
    first = torch.cumsum(miss_per_req, 0) - miss_per_req
    rank = torch.arange(b_i.numel(), device=dev, dtype=torch.int64) - first[b_i]
    pos = rank[:, None] * per_miss + torch.arange(per_miss, device=dev, dtype=torch.int64)
    rows = b_i[:, None].expand_as(pos)
    plan_src = torch.full((B, P), pad, dtype=torch.int64, device=dev)
    plan_dst = torch.full((B, P), pad, dtype=torch.int32, device=dev)
    plan_src[rows, pos] = src
    plan_dst[rows, pos] = dst.to(torch.int32)
    return Plan(src=plan_src, dst=plan_dst, counts=counts.to(torch.int32), num_real=torch.tensor([B], dtype=torch.int32, device=dev),
                kind=kind, item_size_bytes=nbytes, n_items=int(counts.sum()), host_records=host_records, dev_records=dev_records)


def validate_plan(plan: Plan) -> None:
    """Every value the kernel will dereference, checked once (syncs; never inside a timed region):
    num_real <= rows, 0 <= counts <= P, every used src in [0, host_records), every used dst in
    [0, dev_records) and unique (two items writing one record would race)."""
    R, P = plan.src.shape
    real = int(plan.num_real.reshape(-1)[0])
    assert 1 <= real <= R, "num_real %d outside [1, %d]" % (real, R)
    c = plan.counts.to(torch.int64)
    assert c.numel() >= R and bool((c >= 0).all()) and int(c.max()) <= P, "counts outside [0, %d]" % P
    used = torch.arange(P, device=plan.src.device)[None, :] < c[:R, None]
    used[real:] = False
    s, d = plan.src[used], plan.dst[used].to(torch.int64)
    assert int(used.sum()) == plan.n_items or real < R, "n_items %d != used entries %d" % (plan.n_items, int(used.sum()))
    if s.numel():
        assert int(s.min()) >= 0 and int(s.max()) < plan.host_records, "src outside [0, %d)" % plan.host_records
        assert int(d.min()) >= 0 and int(d.max()) < plan.dev_records, "dst outside [0, %d)" % plan.dev_records
        assert torch.unique(d).numel() == d.numel(), "two items write one device record"


def plans_equal(a: Plan, b: Plan) -> bool:
    return (a.kind == b.kind and a.item_size_bytes == b.item_size_bytes and a.n_items == b.n_items
            and torch.equal(a.src.cpu(), b.src.cpu()) and torch.equal(a.dst.cpu(), b.dst.cpu())
            and torch.equal(a.counts.cpu(), b.counts.cpu()) and torch.equal(a.num_real.cpu(), b.num_real.cpu()))


def _sync(device) -> None:
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed_build(load_ids_cpu: torch.Tensor, device=None, **kw):
    """Build the plan on the HOST (CPU torch) and, when `device` is given, upload it and ALSO build it
    on the DEVICE from the same ids; the two must be equal. Returns (plan on `device` or CPU, timing).
    Timings are wall-clock ms: build_host_ms (CPU build), upload_ms (plan H2D), build_device_ms (ids
    H2D excluded; includes the syncs nonzero() needs). None of this is inside a measured bracket."""
    t0 = time.perf_counter()
    p_cpu = build_plan(load_ids_cpu, **kw)
    t = dict(build_host_ms=1e3 * (time.perf_counter() - t0), n_items=p_cpu.n_items, plan_stride=p_cpu.plan_stride,
             item_size_bytes=p_cpu.item_size_bytes, kind=p_cpu.kind, label=p_cpu.label, plan_bytes=_plan_bytes(p_cpu))
    if device is None:
        return p_cpu, t
    _sync(device)
    t1 = time.perf_counter()
    p_dev = p_cpu.to(device)
    _sync(device)
    t["upload_ms"] = 1e3 * (time.perf_counter() - t1)
    ids_dev = load_ids_cpu.to(device)
    _sync(device)
    t2 = time.perf_counter()
    p_dev2 = build_plan(ids_dev, **kw)
    _sync(device)
    t["build_device_ms"] = 1e3 * (time.perf_counter() - t2)
    t["device_build_equal"] = plans_equal(p_dev, p_dev2)
    return p_dev, t


def _plan_bytes(p: Plan) -> int:
    return sum(x.numel() * x.element_size() for x in (p.src, p.dst, p.counts, p.num_real))


# --------------------------------------------------------------------------- CPU twins of the kernel
def kernel_walk(counts, num_real: int, num_blocks: int, block_size: int, warp_size: int = WARP_SIZE) -> Iterator[Tuple[int, int, int]]:
    """Python model of copy_cache_planned_kernel's index walk (hisparse.cuh :853-891): yields
    (warp_global, request r, miss m) for every item the kernel copies. Each warp walks every request in
    warp-sized windows (:864-868), keeps `start` = flat index of request r's first miss (:863, :889),
    and takes the misses m = m0, m0 + total_warps, ... with m0 = (warp_global - start) mod total_warps
    (:871-872, :876). Used by the tests to prove every planned item is copied exactly once."""
    counts = [int(c) for c in counts]
    nw = block_size // warp_size
    total = num_blocks * nw
    for warp_global in range(total):
        start = 0
        for base in range(0, num_real, warp_size):
            window = min(num_real - base, warp_size)
            for j in range(window):
                cnt = counts[base + j]
                if cnt == 0:
                    continue
                m0 = (warp_global - start) % total
                for m in range(m0, cnt, total):
                    yield warp_global, base + j, m
                start += cnt


def _records(t: torch.Tensor, item: int) -> torch.Tensor:
    assert t.is_contiguous()
    flat = t.reshape(-1).view(torch.uint8)
    assert flat.numel() % item == 0
    return flat.view(-1, item)


def simulate_planned_copy(plan: Plan, host_k, host_v, dev_k, dev_v, num_blocks: Optional[int] = None,
                          block_size: Optional[int] = None) -> int:
    """CPU twin of one copy_cache_planned_kernel launch with IsMLA = false (:879-887 -> :212-222): for every
    planned item, dev[dst] = host[src] for K and for V, byte for byte. With num_blocks / block_size the
    items are visited in the kernel's own warp walk (kernel_walk); otherwise vectorized. Returns the number
    of items copied."""
    item = plan.item_size_bytes
    real = int(plan.num_real.reshape(-1)[0])
    pairs = ((_records(host_k, item), _records(dev_k, item)), (_records(host_v, item), _records(dev_v, item)))
    src, dst = plan.src.cpu(), plan.dst.cpu().to(torch.int64)
    if num_blocks is not None:
        n = 0
        for _, r, m in kernel_walk(plan.counts.cpu().tolist(), real, num_blocks, block_size):
            s, d = int(src[r, m]), int(dst[r, m])
            for h, dv in pairs:
                dv[d] = h[s]
            n += 1
        return n
    R, P = src.shape
    used = torch.arange(P)[None, :] < plan.counts.cpu().to(torch.int64)[:R, None]
    used[real:] = False
    s, d = src[used], dst[used]
    for h, dv in pairs:
        dv[d] = h[s]
    return int(used.sum())
