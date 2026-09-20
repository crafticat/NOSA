"""STAGE E4: per-layer prefetch of the S row's selection through the COPY ENGINE
(DESIGN.md section 9; REPRODUCE.md 'E3d RESULT', 'E3d TIMELINES', 'HOST PACK
THROUGHPUT': the only transfer that overlaps a decode step on this device is a
CONTIGUOUS cudaMemcpyAsync from pinned memory, kappa 0.006-0.05 at 25 GB/s;
every SM-side gather slows the co-running kernels 2.5-3x).

THE CADENCE (one behind). At layer l of tick t, after S's scoring and V's
update, the launch thread hands (sel_S(l, t), _block_map(l) after the update)
to a CPU worker: the worker plans want = sel_S minus resident minus in-flight
(plan_prefetch), PACKS the wanted blocks from the pinned window into a pinned
staging buffer (pack_rows + index_select: contiguous 32 KB pieces per
(request, block), both KV heads, K then V), and issues ONE cudaMemcpyAsync
on the side stream into the device ring half of (l, t) with an event. At
layer l of tick t + 1, before V's update, the launch thread polls that event
(event.query(): no host sync); if complete, the ring half is handed to the
S == 1 body: diff assigns the slots as today, the RING-HIT kernel serves the
loads the ring holds by a D2D scatter (K, V from the ring; the bias rows from
the GPU-resident total_cis) and clears those _load_mask entries, then the
shipped gathers fetch the residual misses (blocking, counted). The window is
still the decode's 64 slots in the decode's slot order: V stays bit-exact.

WHY PER (request, block) WITH BOTH HEADS. The pinned window is (B, S, H, D):
a block's 64 rows x both heads are ONE contiguous 32 KB run (row stride
H*D*2 = 512 B); a per-head piece would be 64 strided 256 B runs. So a piece =
(b, blk), K and V = two contiguous 32 KB copies; the other head's half is
charged as extra bytes when only one head needed the block.

ACCOUNTING per (layer, tick): pieces / bytes issued, arrived in time or
LATE (event pending at need time: the whole half is unusable and its loads
go through the blocking path), served pieces (hit by V's diff), wrong pieces
(arrived, never needed by V), extra-head bytes, residual misses (blocking
bytes), unrequested (cap), ring occupancy, host pack ms, deadline slack
(need time - arrival time, CUDA events).

Pure-torch parts (CPU-tested): plan_prefetch, pack_rows, RingBook (the
readiness state machine over a double-buffered ring per layer),
diff_reference (the CUDA diff kernel's rule), ring_hit_reference,
v_update_with_ring (the S == 1 body with the ring step, over injected ops),
PrefetchAccount. GPU parts (guarded): the Triton ring-hit kernel and
PrefetchEngine (worker pool, pinned staging, side stream, events).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, NamedTuple, Optional

import torch

try:                      # module-level (fused_bias.py precedent: the JIT resolves tl annotations through the globals)
    import triton
    import triton.language as tl
except Exception:         # pragma: no cover
    triton = None
    tl = None

BLOCK_TOKENS = 64
HEAD_DIM = 128
BYTES_HEAD_BLOCK = BLOCK_TOKENS * HEAD_DIM * 2           # one head's rows of one block, K or V: 16 KB
FREE, INFLIGHT, ARRIVED, CONSUMED = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# the plan (host side, pure torch)
# ---------------------------------------------------------------------------
class Pieces(NamedTuple):
    b: torch.Tensor            # (P,) int64 request per piece, sorted by (b, blk)
    blk: torch.Tensor          # (P,) int64 block id
    heads: torch.Tensor        # (P, H) bool: which heads wanted it
    n_want: torch.Tensor       # (H, B) int64: wanted (not resident, not in flight, fetchable) per stream
    n_unrequested: torch.Tensor  # (H, B) int64: wanted but beyond the cap
    n_inflight_hit: torch.Tensor  # (H, B) int64: wanted but already in flight (not re-issued)


def plan_prefetch(sel_s: torch.Tensor, req_idx: torch.Tensor, window: torch.Tensor, tail_slot: int, forbid_ge: int,
                  inflight: Optional[torch.Tensor], cap: int) -> Pieces:
    """sel_s (H, n, K) int64 (S's selection of the n call requests req_idx (n,)),
    window (H, B, topk) int64 (_block_map AFTER V's update = the resident set at
    the need time), forbid_ge: block ids >= it are not on the host (the tail
    block T: the host copy of T is written at the fill only), inflight (B, C)
    int64 ids in flight per request (-1 padded) or None, cap: pieces per
    request. Canonical order (b, blk) ascending; a piece is (b, blk) over the
    heads that want it (packed with both heads)."""
    H, n, K = sel_s.shape
    B, topk = window.shape[1], window.shape[2]
    dev = sel_s.device
    b_of = req_idx.view(1, n, 1).expand(H, n, K)
    ids = sel_s
    valid = (ids >= 0) & (ids < int(forbid_ge))
    win = window[:, req_idx, :tail_slot]                                                      # (H, n, topk-1)
    resident = (ids.unsqueeze(-1) == win.unsqueeze(-2)).any(-1)                               # (H, n, K)
    want = valid & ~resident
    if inflight is not None:
        inf = inflight[req_idx]                                                               # (n, C)
        in_flight = (ids.unsqueeze(-1) == inf.view(1, n, 1, -1)).any(-1) & (inf.view(1, n, 1, -1) >= 0).any(-1)
        n_inflight_hit = (want & in_flight).sum(-1)
        want = want & ~in_flight
    else:
        n_inflight_hit = torch.zeros((H, n), dtype=torch.int64, device=dev)
    n_want = want.sum(-1)                                                                     # (H, n)
    h_of = torch.arange(H, dtype=torch.int64, device=dev).view(H, 1, 1).expand(H, n, K)
    key = (b_of * (1 << 40) + ids)[want]
    hs = h_of[want]
    if key.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.int64, device=dev)
        return Pieces(b=empty, blk=empty, heads=torch.zeros((0, H), dtype=torch.bool, device=dev),
                      n_want=_scatter_hb(n_want, req_idx, H, B), n_unrequested=torch.zeros((H, B), dtype=torch.int64, device=dev),
                      n_inflight_hit=_scatter_hb(n_inflight_hit, req_idx, H, B))
    ukey, inv = torch.unique(key, return_inverse=True)                                       # sorted: (b, blk) ascending
    P = ukey.numel()
    heads = torch.zeros((P, H), dtype=torch.bool, device=dev)
    heads[inv, hs] = True
    pb = ukey // (1 << 40)
    pblk = ukey % (1 << 40)
    # the cap: the first `cap` pieces of every request in (b, blk) order
    first = torch.searchsorted(pb, pb, right=False)
    rank = torch.arange(P, dtype=torch.int64, device=dev) - first
    keep = rank < int(cap)
    dropped_heads = heads[~keep]                                                              # (Pd, H)
    n_unreq = torch.zeros((H, B), dtype=torch.int64, device=dev)
    if dropped_heads.numel():
        n_unreq.index_put_((torch.arange(H, device=dev).view(H, 1).expand(H, dropped_heads.shape[0]).reshape(-1),
                            pb[~keep].view(1, -1).expand(H, -1).reshape(-1)),
                           dropped_heads.t().reshape(-1).to(torch.int64), accumulate=True)
    return Pieces(b=pb[keep].contiguous(), blk=pblk[keep].contiguous(), heads=heads[keep].contiguous(),
                  n_want=_scatter_hb(n_want, req_idx, H, B), n_unrequested=n_unreq, n_inflight_hit=_scatter_hb(n_inflight_hit, req_idx, H, B))


def _scatter_hb(v: torch.Tensor, req_idx: torch.Tensor, H: int, B: int) -> torch.Tensor:
    out = torch.zeros((H, B), dtype=torch.int64, device=v.device)
    out[:, req_idx] = v
    return out


def pack_rows(piece_b: torch.Tensor, piece_blk: torch.Tensor, S_host: int, bs: int = BLOCK_TOKENS) -> torch.Tensor:
    """(P*bs,) int64 flat rows into the pinned window viewed as (B*S, H, D):
    piece p = rows b*S + blk*bs .. + bs - 1 (one contiguous 32 KB run each)."""
    base = piece_b * int(S_host) + piece_blk * bs
    return (base.view(-1, 1) + torch.arange(bs, dtype=torch.int64, device=piece_b.device).view(1, bs)).reshape(-1)


def pack_pieces(k_cpu: torch.Tensor, v_cpu: torch.Tensor, rows: torch.Tensor, stage: torch.Tensor) -> int:
    """K then V of the pieces into ``stage`` (2, P_max*bs, H, D) pinned, rows
    from pack_rows; index_select writes the contiguous 32 KB runs (the HOST
    PACK THROUGHPUT path: 21-30 GB/s with 8-16 threads). Returns the rows."""
    B, S, H, D = k_cpu.shape
    n = rows.numel()
    if n > stage.shape[1]:
        raise ValueError("stage holds %d rows, pack needs %d" % (stage.shape[1], n))
    torch.index_select(k_cpu.view(B * S, H, D), 0, rows, out=stage[0, :n])
    torch.index_select(v_cpu.view(B * S, H, D), 0, rows, out=stage[1, :n])
    return n


def id_table(piece_b: torch.Tensor, piece_blk: torch.Tensor, B: int, cap: int) -> (torch.Tensor, torch.Tensor):
    """(B, cap) int64 ids (-1 padded) and (B, cap) int32 piece index per request,
    the ring-hit kernel's lookup table (pieces are sorted by (b, blk))."""
    dev = piece_b.device
    ids = torch.full((B, cap), -1, dtype=torch.int64, device=dev)
    pidx = torch.full((B, cap), -1, dtype=torch.int32, device=dev)
    if piece_b.numel():
        first = torch.searchsorted(piece_b, piece_b, right=False)
        col = torch.arange(piece_b.numel(), dtype=torch.int64, device=dev) - first
        ids[piece_b, col] = piece_blk
        pidx[piece_b, col] = torch.arange(piece_b.numel(), dtype=torch.int32, device=dev)
    return ids, pidx


# ---------------------------------------------------------------------------
# the CUDA diff kernel's rule, as pure torch (flash_cache_engine/diff_offload_kernel.cu)
# ---------------------------------------------------------------------------
def diff_reference(old_map: torch.Tensor, new_act: torch.Tensor):
    """new_map / load_map per (h, b): a slot whose old block is in the new
    selection keeps it (load -1); the k-th new block that is not in the old
    map goes to the k-th slot whose old block is not in the new selection, in
    slot order, and is loaded (diff_offload_kernel.cu:34-85). Slots the kernel
    does not write keep the buffer's previous content: here new_map starts as
    old_map and load_map as -1 (the steady state, |new| = M distinct ids)."""
    H, B, M = old_map.shape
    old_hit = (old_map.unsqueeze(-1) == new_act.unsqueeze(-2)).any(-1)                       # (H, B, M) slot's old id in new
    new_hit = (new_act.unsqueeze(-1) == old_map.unsqueeze(-2)).any(-1)                       # new id already resident
    new_map = old_map.clone()
    load_map = torch.full_like(old_map, -1)
    free_rank = torch.cumsum((~old_hit).to(torch.int64), dim=-1) - 1                         # rank of a free slot among free slots
    new_rank = torch.cumsum((~new_hit).to(torch.int64), dim=-1) - 1                          # rank of a new id among new ids
    for h in range(H):
        for b in range(B):
            free_slots = (~old_hit[h, b]).nonzero().squeeze(-1)
            new_ids = new_act[h, b][~new_hit[h, b]]
            k = min(free_slots.numel(), new_ids.numel())
            if k:
                new_map[h, b, free_slots[:k]] = new_ids[:k]
                load_map[h, b, free_slots[:k]] = new_ids[:k]
    return new_map, load_map


# ---------------------------------------------------------------------------
# the ring-hit step: D2D scatter of arrived pieces into the slots diff assigned
# ---------------------------------------------------------------------------
def ring_hit_reference(k_gpu, v_gpu, kv_bias_gpu, total_cis, load_mask, ring_k, ring_v, ring_ids, ring_piece, served_p, served_hb, bs: int):
    """Pure torch twin of the kernel. ring_k / ring_v (P*bs, H, D) = the pieces'
    K / V rows (both heads); ring_ids / ring_piece (B, C) from id_table.
    For every (h, b, m) with load_mask >= 0 whose block is in the ring: copy
    head h's rows of the piece into slot m, the bias rows from total_cis,
    clear the load, count."""
    H, B, M = load_mask.shape
    for h in range(H):
        for b in range(B):
            for m in range(M):
                blk = int(load_mask[h, b, m])
                if blk < 0:
                    continue
                hit = (ring_ids[b] == blk).nonzero()
                if hit.numel() == 0:
                    continue
                p = int(ring_piece[b, hit[0, 0]])
                k_gpu[b, m * bs:(m + 1) * bs, h, :] = ring_k[p * bs:(p + 1) * bs, h, :]
                v_gpu[b, m * bs:(m + 1) * bs, h, :] = ring_v[p * bs:(p + 1) * bs, h, :]
                kv_bias_gpu[b, m * bs:(m + 1) * bs, h] = total_cis[b, blk * bs:(blk + 1) * bs, h]
                load_mask[h, b, m] = -1
                served_p[p] += 1
                served_hb[h, b] += 1


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if triton is None or tl is None:
        raise RuntimeError("triton is not importable here: the ring-hit kernel needs it")

    @triton.jit
    def ring_hit_kernel(k_ptr, v_ptr, bias_ptr, cis_ptr, load_ptr, rk_ptr, rv_ptr, ids_ptr, piece_ptr, sp_ptr, shb_ptr,
                        B, M, C, S_GPU, S_CIS,
                        BS: tl.constexpr, H: tl.constexpr, D: tl.constexpr, CP: tl.constexpr):
        h = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        m = tl.program_id(2)
        blk = tl.load(load_ptr + (h * B + b) * M + m)
        if blk >= 0:
            offs_c = tl.arange(0, CP)
            cmask = offs_c < C
            ids = tl.load(ids_ptr + b * C + offs_c, mask=cmask, other=-2)
            hit = ids == blk
            found = tl.max(hit.to(tl.int32), 0) > 0
            if found:
                pc = tl.max(tl.where(hit, offs_c, -1), 0)
                p = tl.load(piece_ptr + b * C + pc).to(tl.int64)
                offs_i = tl.arange(0, BS)[:, None]
                offs_d = tl.arange(0, D)[None, :]
                src = (p * BS + offs_i) * (H * D) + h * D + offs_d
                dst = (b * S_GPU + m * BS + offs_i) * (H * D) + h * D + offs_d
                tl.store(k_ptr + dst, tl.load(rk_ptr + src))
                tl.store(v_ptr + dst, tl.load(rv_ptr + src))
                offs_r = tl.arange(0, BS)
                cis = tl.load(cis_ptr + (b * S_CIS + blk * BS + offs_r) * H + h)
                tl.store(bias_ptr + (b * S_GPU + m * BS + offs_r) * H + h, cis)
                tl.store(load_ptr + (h * B + b) * M + m, -1)
                tl.atomic_add(sp_ptr + p, 1)
                tl.atomic_add(shb_ptr + h * B + b, 1)

    _KERNEL = ring_hit_kernel
    return _KERNEL


def ring_hit_triton(k_gpu, v_gpu, kv_bias_gpu, total_cis, load_mask, ring_k, ring_v, ring_ids, ring_piece, served_p, served_hb, bs: int):
    """Grid (H, B, M) like the shipped gathers; every argument contiguous; D2D only."""
    H, B, M = load_mask.shape
    Bk, S_GPU, Hk, D = k_gpu.shape
    for name, t in (("k_gpu", k_gpu), ("v_gpu", v_gpu), ("kv_bias_gpu", kv_bias_gpu), ("total_cis", total_cis), ("load_mask", load_mask),
                    ("ring_k", ring_k), ("ring_v", ring_v), ("ring_ids", ring_ids), ("ring_piece", ring_piece)):
        if not t.is_contiguous():
            raise ValueError("%s must be contiguous" % name)
    if Hk != H or Bk != B or tuple(ring_ids.shape) != tuple(ring_piece.shape) or ring_ids.shape[0] != B:
        raise ValueError("ring_hit: shapes k_gpu %s load_mask %s ring_ids %s" % (tuple(k_gpu.shape), tuple(load_mask.shape), tuple(ring_ids.shape)))
    if ring_ids.dtype != torch.int64 or ring_piece.dtype != torch.int32 or load_mask.dtype != torch.int64 or served_p.dtype != torch.int32 or served_hb.dtype != torch.int32:
        raise ValueError("ring_hit dtypes: ring_ids int64, ring_piece int32, load_mask int64, served_* int32")
    C = ring_ids.shape[1]
    CP = 1
    while CP < C:
        CP *= 2
    _kernel()[(H, B, M)](k_gpu, v_gpu, kv_bias_gpu, total_cis, load_mask, ring_k, ring_v, ring_ids, ring_piece, served_p, served_hb,
                         B, M, C, S_GPU, total_cis.shape[1], BS=bs, H=H, D=D, CP=CP)


# ---------------------------------------------------------------------------
# the S == 1 body with the ring step (cache_engine.decode_update_has_kv_bias, replayed over injected ops)
# ---------------------------------------------------------------------------
class EngineOps(NamedTuple):
    diff: Callable            # diff.diff_offload(old_map, topk_idx, new_map_buf, load_mask)
    gather_k: Callable        # flash_h2d_from_mask(k_gpu, k_cpu, load_mask, bs)
    gather_v: Callable        # flash_h2d_from_mask_bias(v_gpu, v_cpu, kv_bias_gpu, kv_bias, load_mask, bs)
    ring_hit: Callable        # ring_hit_triton / ring_hit_reference
    bias_rows: Callable       # cache_engine._bias_rows


class RingHalfDevice(NamedTuple):
    ring_k: torch.Tensor      # (P_max*bs, H, D) device
    ring_v: torch.Tensor
    ids: torch.Tensor         # (B, C) int64 device
    piece: torch.Tensor       # (B, C) int32 device
    served_p: torch.Tensor    # (P_max,) int32 device, zeroed before use
    served_hb: torch.Tensor   # (H, B) int32 device, zeroed before use


def v_update_with_ring(engine, ops: EngineOps, key_states, value_states, kv_bias, topk_idx, ring: Optional[RingHalfDevice]):
    """cache_engine.py decode_update_has_kv_bias (:640-716), verbatim with
    ``self`` -> ``engine`` and the kernels -> ``ops``, plus the RING step
    between diff and the gathers (where the victim pool sits, :785-800):
    arrived pieces serve their loads D2D and clear _load_mask, the shipped
    gathers fetch the rest. No line of the S == 1 body is changed otherwise;
    tests/test_paired_core.py replays it on a fake engine against the plain
    body (same maps, same bytes, fewer host loads)."""
    B, S, H, D = key_states.shape
    assert H == engine.head_num and D == engine.head_dim and S == 1
    engine.seq_length += 1                                                                              # :645
    engine._cache_lens[...] += 1                                                                        # :646
    _tail_write_pos = engine._tail_block_idx_on_gpu * engine.block_size + engine._tail_block_len_on_gpu  # :649
    engine._k_gpu[:, _tail_write_pos:_tail_write_pos + 1, :, :].copy_(key_states, non_blocking=True)    # :650
    engine._v_gpu[:, _tail_write_pos:_tail_write_pos + 1, :, :].copy_(value_states, non_blocking=True)  # :651
    engine._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos + 1, :].copy_(ops.bias_rows(kv_bias[:, engine.seq_length - 1:engine.seq_length, :]), non_blocking=True)  # :654
    engine._tail_block_len_on_gpu += 1                                                                  # :657
    tail_full = engine._tail_block_len_on_gpu == engine.block_size                                      # :659
    ops.diff(engine._block_map, topk_idx, engine._new_block_map_buf, engine._load_mask)                # :663
    if ring is not None:
        # RING HIT (E4): the loads the arrived ring half holds are served D2D and cleared from _load_mask;
        # the map assignment above is untouched (the slot order is diff's), so V stays the decode's row
        ops.ring_hit(engine._k_gpu, engine._v_gpu, engine._kv_bias_gpu, kv_bias, engine._load_mask,
                     ring.ring_k, ring.ring_v, ring.ids, ring.piece, ring.served_p, ring.served_hb, engine.block_size)
    engine._block_map.copy_(engine._new_block_map_buf, non_blocking=True)                              # :677
    ops.gather_k(engine._k_gpu, engine._k_cpu, engine._load_mask, engine.block_size)                    # :681
    ops.gather_v(engine._v_gpu, engine._v_cpu, engine._kv_bias_gpu, kv_bias, engine._load_mask, engine.block_size)   # :682
    if tail_full:                                                                                       # :701
        tail_block_base_pos = engine._tail_block_idx_on_gpu * engine.block_size
        cpu_block_base_pos = engine.seq_length - engine.block_size
        engine._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos + engine.block_size, :, :].copy_(engine._k_gpu[:, tail_block_base_pos:tail_block_base_pos + engine.block_size, :, :], non_blocking=True)
        engine._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos + engine.block_size, :, :].copy_(engine._v_gpu[:, tail_block_base_pos:tail_block_base_pos + engine.block_size, :, :], non_blocking=True)
        engine._block_map[..., engine._tail_block_idx_on_gpu] += 1
        engine._tail_block_len_on_gpu = 0
        engine._cache_lens[...] = (engine.topk - 1) * engine.block_size + engine._tail_block_len_on_gpu


# ---------------------------------------------------------------------------
# the ring bookkeeping (host side, pure python): a double-buffered ring per layer
# ---------------------------------------------------------------------------
class HalfState:
    __slots__ = ("status", "tick", "n_pieces", "n_rows", "ev_done", "ev_consumed", "t_issue", "t_ready", "pack_ms", "late", "heads", "n_unreq")

    def __init__(self):
        self.n_unreq = 0
        self.status = FREE
        self.tick = -1
        self.n_pieces = 0
        self.n_rows = 0
        self.ev_done = None
        self.ev_consumed = None
        self.t_issue = None
        self.t_ready = None
        self.pack_ms = None
        self.late = False
        self.heads = None


class RingBook:
    """Per layer two halves; tick t's issue goes into half t % 2, is needed at
    tick t + 1 (one-behind), and its half is free again once its scatter on
    the main stream is done (ev_consumed) and its copy completed. A half that
    is still INFLIGHT at need time is LATE: nothing of it is used, its pieces
    are charged as late, and it is freed when its copy completes. An issue
    into a half that is not FREE is refused (ring full: counted)."""

    def __init__(self, num_layers: int):
        self.halves = [[HalfState(), HalfState()] for _ in range(num_layers)]
        self.ring_full = 0
        self.lock = threading.Lock()

    def half_for(self, tick: int) -> int:
        return int(tick) % 2

    def try_issue(self, l: int, tick: int, n_pieces: int, heads=None) -> Optional[int]:
        with self.lock:
            hs = self.halves[l][self.half_for(tick)]
            if hs.status != FREE:
                self.ring_full += 1
                return None
            hs.status, hs.tick, hs.n_pieces, hs.late, hs.heads = INFLIGHT, int(tick), int(n_pieces), False, heads
            hs.t_issue = time.perf_counter()
            hs.ev_done = hs.ev_consumed = None
            return self.half_for(tick)

    def set_event(self, l: int, half: int, ev, pack_ms: float, n_rows: int):
        with self.lock:
            hs = self.halves[l][half]
            hs.ev_done, hs.pack_ms, hs.n_rows = ev, float(pack_ms), int(n_rows)
            hs.t_ready = time.perf_counter()

    def check_need(self, l: int, tick_need: int, query: Callable[[object], bool]) -> Optional[int]:
        """At layer l of tick ``tick_need``: the half issued at tick_need - 1, if
        its copy is complete (query(ev) True) -> ARRIVED and returned; if the
        copy is still pending -> LATE (None); if nothing was issued -> None."""
        with self.lock:
            half = self.half_for(tick_need - 1)
            hs = self.halves[l][half]
            if hs.status != INFLIGHT or hs.tick != tick_need - 1:
                return None
            if hs.ev_done is None or not query(hs.ev_done):
                hs.late = True
                return None
            hs.status = ARRIVED
            return half

    def consume(self, l: int, half: int, ev_consumed):
        with self.lock:
            hs = self.halves[l][half]
            hs.status, hs.ev_consumed = CONSUMED, ev_consumed

    def release_done(self, l: int, query: Callable[[object], bool]) -> List[int]:
        """Free every half whose copy AND consumption are complete (or whose
        copy completed after being late). Returns the freed half indices."""
        freed = []
        with self.lock:
            for half, hs in enumerate(self.halves[l]):
                if hs.status == CONSUMED and (hs.ev_consumed is None or query(hs.ev_consumed)):
                    hs.status = FREE
                    freed.append(half)
                elif hs.status == INFLIGHT and hs.late and hs.ev_done is not None and query(hs.ev_done):
                    hs.status = FREE
                    freed.append(half)
        return freed

    def occupancy(self, l: int) -> int:
        return sum(1 for hs in self.halves[l] if hs.status != FREE)


class PrefetchAccount:
    """Per (layer, tick) records, pure python; ``summary`` turns them into
    bytes per committed token and the exposed / wrong / extra fractions."""

    def __init__(self):
        self.rows: List[dict] = []

    def add(self, **kw):
        self.rows.append(dict(kw))

    @staticmethod
    def piece_bytes(n: int) -> int:
        return int(n) * 2 * 2 * BYTES_HEAD_BLOCK          # K + V, both heads

    def summary(self, committed_tokens: int) -> dict:
        r = self.rows
        n = len(r)
        tot = lambda k: sum(int(x.get(k, 0) or 0) for x in r)
        out = dict(records=n, issued_pieces=tot("issued"), issued_bytes=self.piece_bytes(tot("issued")),
                   in_time_layers=sum(1 for x in r if x.get("issued", 0) and x.get("arrived")), late_layers=sum(1 for x in r if x.get("issued", 0) and not x.get("arrived")),
                   late_pieces=tot("late_pieces"), served_pieces=tot("served"), wrong_pieces=tot("wrong"),
                   extra_head_bytes=tot("extra_head_bytes"), residual_misses=tot("residual"), residual_bytes=tot("residual") * 2 * BYTES_HEAD_BLOCK,
                   unrequested=tot("unrequested"), ring_full=tot("ring_full"),
                   pack_ms=[x.get("pack_ms") for x in r if x.get("pack_ms") is not None], slack_ms=[x.get("slack_ms") for x in r if x.get("slack_ms") is not None],
                   occupancy_max=max([int(x.get("occupancy", 0)) for x in r] or [0]))
        out["late_bytes"] = self.piece_bytes(out["late_pieces"])
        out["wrong_bytes"] = self.piece_bytes(out["wrong_pieces"])
        c = max(1, int(committed_tokens))
        out["issued_bytes_per_token"] = out["issued_bytes"] / c
        out["residual_bytes_per_token"] = out["residual_bytes"] / c
        out["exposed_bytes_per_token"] = (out["residual_bytes"]) / c            # every blocking gather byte is exposed
        out["pack_ms_mean"] = (sum(out["pack_ms"]) / len(out["pack_ms"])) if out["pack_ms"] else None
        out["slack_ms_min"] = min(out["slack_ms"]) if out["slack_ms"] else None
        out["slack_ms_mean"] = (sum(out["slack_ms"]) / len(out["slack_ms"])) if out["slack_ms"] else None
        return out


def account_half(heads: torch.Tensor, served_p: torch.Tensor) -> dict:
    """heads (P, H) bool (packed = both heads), served_p (P,) counts of heads
    served from the piece -> served / wrong pieces and the extra-head bytes
    (K + V halves of a head the piece carried but no slot needed)."""
    P = int(served_p.numel())
    sp = served_p.to(torch.int64)
    used = sp > 0
    H = int(heads.shape[1]) if heads.dim() == 2 else 2
    extra_heads = int((H - sp[used]).clamp(min=0).sum())
    return dict(served=int(used.sum()), wrong=int(P - int(used.sum())), extra_head_bytes=extra_heads * 2 * BYTES_HEAD_BLOCK)


# ---------------------------------------------------------------------------
# the GPU driver (guarded: constructed only with a CUDA cache)
# ---------------------------------------------------------------------------
class PrefetchEngine:
    """Per process: the worker pool, the pinned staging (2 halves per layer),
    the device ring (2 halves per layer), the side stream and the book.
    ``submit`` is called by the launch thread at layer l of tick t after V's
    update; ``check_need`` / ``consumed`` around V's update at tick t + 1."""

    def __init__(self, cache, B: int, cap: int, workers: int, ops: EngineOps, device):
        eng0 = cache.layers[0].cache_engine
        H, D, bs = eng0.head_num, eng0.head_dim, eng0.block_size
        self.L = len(cache.layers)
        self.B, self.cap, self.bs, self.H, self.D = int(B), int(cap), int(bs), H, D
        self.P_max = self.B * self.cap
        self.ops = ops
        self.device = device
        self.S_host = eng0._k_cpu.shape[1]
        self.book = RingBook(self.L)
        self.account = PrefetchAccount()
        dt = eng0._k_gpu.dtype
        self.stage_h = [[torch.empty((2, self.P_max * bs, H, D), dtype=dt).pin_memory() for _ in range(2)] for _ in range(self.L)]
        self.ring_k = [[torch.empty((self.P_max * bs, H, D), dtype=dt, device=device) for _ in range(2)] for _ in range(self.L)]
        self.ring_v = [[torch.empty((self.P_max * bs, H, D), dtype=dt, device=device) for _ in range(2)] for _ in range(self.L)]
        self.ids_h = [[torch.full((self.B, cap), -1, dtype=torch.int64).pin_memory() for _ in range(2)] for _ in range(self.L)]
        self.piece_h = [[torch.full((self.B, cap), -1, dtype=torch.int32).pin_memory() for _ in range(2)] for _ in range(self.L)]
        self.ids_d = [[torch.full((self.B, cap), -1, dtype=torch.int64, device=device) for _ in range(2)] for _ in range(self.L)]
        self.piece_d = [[torch.full((self.B, cap), -1, dtype=torch.int32, device=device) for _ in range(2)] for _ in range(self.L)]
        self.served_p = [[torch.zeros((self.P_max,), dtype=torch.int32, device=device) for _ in range(2)] for _ in range(self.L)]
        self.served_hb = [[torch.zeros((H, self.B), dtype=torch.int32, device=device) for _ in range(2)] for _ in range(self.L)]
        self.sel_h = [[torch.empty((H, self.B, int(eng0.topk)), dtype=torch.int64).pin_memory() for _ in range(2)] for _ in range(self.L)]
        self.map_h = [[torch.empty((H, self.B, int(eng0.topk)), dtype=torch.int64).pin_memory() for _ in range(2)] for _ in range(self.L)]
        self.inflight_ids = [torch.full((self.B, 2 * cap), -1, dtype=torch.int64) for _ in range(self.L)]   # host: ids in both halves
        self.side = torch.cuda.Stream(device=device)
        self.pool = ThreadPoolExecutor(max_workers=int(workers))
        self.pending: Dict[tuple, object] = {}
        self.ev_need: Dict[tuple, object] = {}
        self.tail_ids = [0] * self.L
        self.errors: List[str] = []
        self.pinned_gb = sum(t.numel() * t.element_size() for hh in self.stage_h for t in hh) / 1e9
        self.device_gb = sum(t.numel() * t.element_size() for hh in self.ring_k + self.ring_v for t in hh) / 1e9

    # -- the launch thread, layer l of tick t, after V's update and S's scoring --
    def submit(self, l: int, tick: int, sel_s: torch.Tensor, block_map: torch.Tensor, tail_id: int, req_idx: Optional[torch.Tensor] = None):
        self.book.release_done(l, lambda e: e.query())            # free the halves whose copy and scatter completed, then issue
        half = self.book.try_issue(l, tick, 0)
        if half is None:
            self.account.add(layer=l, tick=tick, issued=0, arrived=False, ring_full=1)
            return
        sel_h, map_h = self.sel_h[l][half], self.map_h[l][half]
        n = sel_s.shape[1]
        sel_h[:, :n].copy_(sel_s, non_blocking=True)
        map_h.copy_(block_map, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        consumed_ev = self.book.halves[l][half].ev_consumed
        req = torch.arange(self.B) if req_idx is None else req_idx.cpu()
        self.pending[(l, tick)] = self.pool.submit(self._job, l, tick, half, ev, n, req, int(tail_id), consumed_ev)

    def _job(self, l, tick, half, ev, n, req, tail_id, consumed_ev):
        try:
            ev.synchronize()
            t0 = time.perf_counter()
            sel = self.sel_h[l][half][:, :n]
            other = 1 - half
            infl = self.inflight_ids[l][:, other * self.cap:(other + 1) * self.cap]
            tail_slot = self.map_h[l][half].shape[2] - 1
            pieces = plan_prefetch(sel, req, self.map_h[l][half], tail_slot, tail_id, infl, self.cap)
            P = pieces.b.numel()
            rows = pack_rows(pieces.b, pieces.blk, self.S_host, self.bs)
            n_rows = pack_pieces(self.k_cpu[l], self.v_cpu[l], rows, self.stage_h[l][half]) if P else 0
            ids, pidx = id_table(pieces.b, pieces.blk, self.B, self.cap)
            self.ids_h[l][half].copy_(ids)
            self.piece_h[l][half].copy_(pidx)
            self.inflight_ids[l][:, half * self.cap:(half + 1) * self.cap] = ids
            pack_ms = 1e3 * (time.perf_counter() - t0)
            with torch.cuda.stream(self.side):
                if consumed_ev is not None:
                    self.side.wait_event(consumed_ev)
                if n_rows:
                    self.ring_k[l][half][:n_rows].copy_(self.stage_h[l][half][0, :n_rows], non_blocking=True)
                    self.ring_v[l][half][:n_rows].copy_(self.stage_h[l][half][1, :n_rows], non_blocking=True)
                self.ids_d[l][half].copy_(self.ids_h[l][half], non_blocking=True)
                self.piece_d[l][half].copy_(self.piece_h[l][half], non_blocking=True)
                self.served_p[l][half].zero_()
                self.served_hb[l][half].zero_()
                done = torch.cuda.Event(enable_timing=True)
                done.record(self.side)
            with self.book.lock:
                hs = self.book.halves[l][half]
                hs.n_pieces = int(P)
                hs.heads = pieces.heads
                hs.n_unreq = int(pieces.n_unrequested.sum())
            self.book.set_event(l, half, done, pack_ms, n_rows)
        except Exception as e:   # never take the tick down: recorded, the half stays INFLIGHT without an event (= late)
            self.errors.append("layer %d tick %d: %s: %s" % (l, tick, type(e).__name__, e))

    def bind_host(self, cache):
        self.cache_layers = cache.layers
        self.k_cpu = [lay.cache_engine._k_cpu for lay in cache.layers]
        self.v_cpu = [lay.cache_engine._v_cpu for lay in cache.layers]

    # -- the launch thread, layer l of tick t, BEFORE V's update --
    def check_need(self, l: int, tick: int) -> Optional[RingHalfDevice]:
        half = self.book.check_need(l, tick, lambda e: e.query())
        ev_need = torch.cuda.Event(enable_timing=True)
        ev_need.record(torch.cuda.current_stream())
        self.ev_need[(l, tick)] = ev_need
        if half is None:
            return None
        return RingHalfDevice(ring_k=self.ring_k[l][half], ring_v=self.ring_v[l][half], ids=self.ids_d[l][half], piece=self.piece_d[l][half],
                              served_p=self.served_p[l][half], served_hb=self.served_hb[l][half])

    def consumed(self, l: int, tick: int):
        half = self.book.half_for(tick - 1)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self.book.consume(l, half, ev)

    # -- after the tick (synchronized): the records of the halves needed at this tick --
    def close_tick(self, tick: int, residual: torch.Tensor):
        """residual (L, H, B) int64: the blocking loads per layer of this tick."""
        for l in range(self.L):
            half = self.book.half_for(tick - 1)
            hs = self.book.halves[l][half]
            rec = dict(layer=l, tick=tick, residual=int(residual[l].sum()), occupancy=self.book.occupancy(l))
            if hs.tick == tick - 1 and hs.status in (ARRIVED, CONSUMED, INFLIGHT, FREE) and hs.n_pieces >= 0 and hs.t_issue is not None:
                rec.update(issued=hs.n_pieces, arrived=(hs.status in (ARRIVED, CONSUMED)), pack_ms=hs.pack_ms, unrequested=getattr(hs, "n_unreq", 0))
                if hs.status in (ARRIVED, CONSUMED):
                    rec.update(account_half(hs.heads, self.served_p[l][half][:hs.n_pieces].cpu()))   # only the issued pieces, not the half's capacity
                    evn = self.ev_need.get((l, tick))
                    if evn is not None and hs.ev_done is not None:
                        rec["slack_ms"] = hs.ev_done.elapsed_time(evn)      # + = arrived before it was needed
                else:
                    rec["late_pieces"] = hs.n_pieces
            self.account.add(**rec)
            self.book.release_done(l, lambda e: e.query())

    def shutdown(self):
        self.pool.shutdown(wait=True)
