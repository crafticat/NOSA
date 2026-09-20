"""FUSED per-row bias / mask build: ONE Triton kernel per layer (E2c; the E2
measurement made it a precondition: the pure-torch build cost 1.03 ms per
token at B = 128 against a total room of 0.084 ms).

What the kernel computes, per program (query row r, slot s), for every KV
head h -- the SAME rule as core.v_visible_rows / core.s_visible_rows /
core.paired_bias (the torch path, NOSI_PAIRED_BIAS=torch, kept as the
reference; tests/test_paired_core.py gates torch.equal on random rounds
through Triton's interpreter, TRITON_INTERPRET=1):
  V row:  vis = bs for a window slot, tail_rows_v for the tail slot, 0 else.
  S row:  the slot's block id (block_map for the window, block_map[tail] -
          content_off for the tail slot, ring_ids for the ring) must be in
          S's selection AND the slot must be ready: vis = bs (tail_rows_s
          for the tail slot); the OWN ROW (S's provisional K/V, flat row
          own_row_s) is always visible: it extends the prefix when it sits
          right after it (the tail slot in the narrow layout, row 0 of the
          provisional slot in the union layout) and is an EXTRA visible row
          otherwise (the tail block not selected: rows below it stay masked).
  bias[r, s*bs + i, h] = cis[req[r], s*bs + i, h] where visible, MASKED else;
  prefix[r, s, h] = vis (the prefix count, what the ledger reads);
  extent[r] = max over slots of (s*bs + vis, own_row + 1): the exact
  cache_seqlens (atomic max; the buffer is zeroed before the launch).
Rows: r = i*U + u for the i-th request of the call (req[r] = its request id
for cis / block_map / ready; srow[r] = i for the selection rows, which are
(H, n, K) over the call's requests); role[r] = 0 (V) / 1 (S).
Memory: the kernel writes R x W*bs x H bf16 once (4 MB at R = 256, W = 64)
and reads the same volume of cis plus K int64 per (program, head) for S rows.
Launches per layer: one memset (extent) + one kernel. PREDICTION (registered
in DESIGN.md section 8): <= 1 ms per 32-layer call at B = 128.

``fused_bias_reference`` is the same arithmetic in vectorized torch (no
per-row Python loop, no host sync): the CPU twin of the kernel that the tests
compare against both the torch path and the interpreted kernel.
No Triton import at module level: ``fused_bias_triton`` imports it on first use.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from ..verify import rows_attention as _ra


class FusedArgs(NamedTuple):
    cis: torch.Tensor         # (B, W*bs, H) store dtype: the engine's _kv_bias_gpu
    sel: torch.Tensor         # (H, n, K) int64: S's selection per call request (any int64 (H, >= 1, K) when the call has no S rows)
    bmap: torch.Tensor        # (H, B, topk) int64: _block_map
    ring_ids: torch.Tensor    # (H, B, max(NR, 1)) int64: ring occupants (E4); a (H, B, 1) tensor of -1 when NR = 0
    ready: torch.Tensor       # (H, B, W) uint8: core.ready_mask as uint8
    req: torch.Tensor         # (R,) int32: request id per row
    srow: torch.Tensor        # (R,) int32: selection row per row (the call-local request index)
    role: torch.Tensor        # (R,) int8: 0 = V, 1 = S
    W: int
    bs: int
    topk: int
    tail_slot: int
    ring_lo: int
    n_ring: int
    tail_rows_v: int          # V's rows of the tail slot (= tail_len_after)
    tail_rows_s: int          # S's rows of the tail slot when the tail block is selected (bs after a fill, else tail_len_after)
    own_row_s: int            # flat row of S's provisional K/V, -1 when the call has no S rows
    content_off: int          # 1 when slot topk-1 physically holds the completed block T while _block_map says T+1
    masked: float             # ALREADY rounded to the store dtype (rounded_masked)


def rounded_masked(masked: float, dtype: torch.dtype) -> float:
    """The masked value as the store dtype holds it (bf16(-3e4) = -29952), so
    the kernel's fp32 -> bf16 store and the torch path's bf16 constant agree."""
    return float(torch.tensor(float(masked), dtype=dtype))


def check_args(a: FusedArgs, out: torch.Tensor, prefix: torch.Tensor, extent: torch.Tensor) -> int:
    """Every shape / dtype fact both implementations rely on, asserted where
    the arguments are produced; no device value is read. Returns R."""
    B, rows, H = a.cis.shape
    if rows != a.W * a.bs or not a.cis.is_contiguous():
        raise ValueError("cis must be a contiguous (B, W*bs=%d, H), got %s" % (a.W * a.bs, tuple(a.cis.shape)))
    R = a.req.numel()
    for name, t, dt, shape in (("req", a.req, torch.int32, (R,)), ("srow", a.srow, torch.int32, (R,)), ("role", a.role, torch.int8, (R,))):
        if t.dtype != dt or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError("%s must be %s contiguous %s, got %s %s" % (name, dt, shape, t.dtype, tuple(t.shape)))
    if a.sel.dtype != torch.int64 or a.sel.dim() != 3 or a.sel.shape[0] != H or not a.sel.is_contiguous():
        raise ValueError("sel must be int64 contiguous (H=%d, n, K), got %s %s" % (H, a.sel.dtype, tuple(a.sel.shape)))
    if a.bmap.dtype != torch.int64 or tuple(a.bmap.shape) != (H, B, a.topk) or not a.bmap.is_contiguous():
        raise ValueError("bmap must be int64 contiguous (H, B, topk=%d), got %s %s" % (a.topk, a.bmap.dtype, tuple(a.bmap.shape)))
    if a.ring_ids.dtype != torch.int64 or tuple(a.ring_ids.shape) != (H, B, max(a.n_ring, 1)) or not a.ring_ids.is_contiguous():
        raise ValueError("ring_ids must be int64 contiguous (H, B, max(NR, 1)=%d), got %s %s" % (max(a.n_ring, 1), a.ring_ids.dtype, tuple(a.ring_ids.shape)))
    if a.ready.dtype != torch.uint8 or tuple(a.ready.shape) != (H, B, a.W) or not a.ready.is_contiguous():
        raise ValueError("ready must be uint8 contiguous (H, B, W), got %s %s" % (a.ready.dtype, tuple(a.ready.shape)))
    if out.dim() != 3 or out.shape[0] < max(R, B) or tuple(out.shape[1:]) != (rows, H) or out.dtype != a.cis.dtype or not out.is_contiguous():
        raise ValueError("out must be a contiguous (>= max(R, B), W*bs, H) %s, got %s %s" % (a.cis.dtype, out.dtype, tuple(out.shape)))
    if prefix.dtype != torch.int32 or prefix.dim() != 3 or prefix.shape[0] < R or tuple(prefix.shape[1:]) != (a.W, H) or not prefix.is_contiguous():
        raise ValueError("prefix must be int32 contiguous (>= R, W, H)")
    if extent.dtype != torch.int32 or extent.dim() != 1 or extent.shape[0] < R or not extent.is_contiguous():
        raise ValueError("extent must be int32 contiguous (>= R,)")
    if not (0 <= a.tail_rows_v < a.bs) or not (0 <= a.tail_rows_s <= a.bs):
        raise ValueError("tail_rows_v=%d must be in [0, bs), tail_rows_s=%d in [0, bs]" % (a.tail_rows_v, a.tail_rows_s))
    if a.own_row_s >= a.W * a.bs:
        raise ValueError("own_row_s=%d outside the allocation" % a.own_row_s)
    return R


def _result(a: FusedArgs, out, prefix, extent, R: int) -> _ra.RowsBias:
    return _ra.RowsBias(bias=out, selected=prefix[:R] > 0, visible_rows=prefix[:R], cache_batch_idx=a.req,
                        cache_seqlens=extent[:R], U=1, block_size=a.bs)


def fused_bias_reference(a: FusedArgs, out: torch.Tensor, prefix: torch.Tensor, extent: torch.Tensor) -> _ra.RowsBias:
    """The kernel's arithmetic in vectorized torch. out (>= max(R, B), W*bs, H)
    contiguous; prefix (>= R, W, H) int32; extent (>= R,) int32."""
    R = check_args(a, out, prefix, extent)
    W, bs, H = a.W, a.bs, a.cis.shape[2]
    dev = a.cis.device
    req = a.req.to(torch.int64)
    srow = a.srow.to(torch.int64)
    is_v = (a.role == 0).view(R, 1, 1)
    slot = torch.arange(W, dtype=torch.int64, device=dev).view(1, W, 1)
    in_win = slot < a.tail_slot
    is_tail = slot == a.tail_slot
    in_ring = (slot >= a.ring_lo) & (slot < a.ring_lo + a.n_ring)
    bmap_r = a.bmap[:, req, :].permute(1, 2, 0)                                   # (R, topk, H)
    ids = torch.full((R, W, H), -1, dtype=torch.int64, device=dev)
    ids[:, :a.tail_slot, :] = bmap_r[:, :a.tail_slot, :]
    ids[:, a.tail_slot, :] = bmap_r[:, a.tail_slot, :] - a.content_off
    if a.n_ring > 0:
        ids[:, a.ring_lo:a.ring_lo + a.n_ring, :] = a.ring_ids[:, req, :a.n_ring].permute(1, 2, 0)
    # V rows never read the selection (the kernel masks that load by role): clamp so a V-only call's 1-row sel_dummy indexes
    sel_r = a.sel[:, srow.clamp(max=a.sel.shape[1] - 1), :].permute(1, 0, 2)      # (R, H, K)
    found = (ids.permute(0, 2, 1).unsqueeze(-1) == sel_r.unsqueeze(2)).any(-1).permute(0, 2, 1) & (ids >= 0)
    ready_r = a.ready[:, req, :].permute(1, 2, 0) != 0                            # (R, W, H)
    z = torch.zeros((), dtype=torch.int64, device=dev)
    full_rows = torch.where(is_tail, torch.full((), a.tail_rows_s, dtype=torch.int64, device=dev), torch.full((), bs, dtype=torch.int64, device=dev))
    vis_s = torch.where(found & ready_r & (in_win | is_tail | in_ring), full_rows, z)
    vis_v = torch.where(in_win, torch.full((), bs, dtype=torch.int64, device=dev),
                        torch.where(is_tail, torch.full((), a.tail_rows_v, dtype=torch.int64, device=dev), z))
    vis = torch.where(is_v, vis_v, vis_s)                                         # (R, W, H)
    own_slot = a.own_row_s // bs if a.own_row_s >= 0 else -1
    own_i = a.own_row_s % bs if a.own_row_s >= 0 else 0
    has_own = (~is_v) & (slot == own_slot)
    fold = has_own & (vis == own_i)
    extra = has_own & (vis < own_i)
    vis = vis + fold.to(torch.int64)
    i = torch.arange(bs, dtype=torch.int64, device=dev).view(1, 1, bs, 1)
    visible = (i < vis.unsqueeze(2)) | (extra.unsqueeze(2) & (i == own_i))       # (R, W, bs, H)
    cis_r = a.cis.index_select(0, req).view(R, W, bs, H)
    neg = torch.tensor(a.masked, dtype=a.cis.dtype, device=dev)
    torch.where(visible, cis_r, neg, out=out[:R].view(R, W, bs, H))
    prefix[:R].copy_(vis.to(torch.int32))
    cand = torch.where(vis > 0, slot * bs + vis, torch.zeros_like(vis))
    cand = torch.maximum(cand, torch.where(extra, torch.full_like(vis, a.own_row_s + 1), torch.zeros_like(vis)))
    extent[:R].copy_(cand.amax(dim=(1, 2)).to(torch.int32))
    return _result(a, out, prefix, extent, R)


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def paired_bias_kernel(cis_ptr, bias_ptr, prefix_ptr, extent_ptr, sel_ptr, bmap_ptr, ring_ptr, ready_ptr, req_ptr, srow_ptr, role_ptr,
                           B, W, K, NSEL, NR, NRP, TOPK, TAIL, RING_LO, TAIL_ROWS_V, TAIL_ROWS_S, OWN_ROW_S, CONTENT_OFF, MASKED,
                           BS: tl.constexpr, H: tl.constexpr, KP: tl.constexpr):
        r = tl.program_id(0)
        slot = tl.program_id(1)
        b = tl.load(req_ptr + r).to(tl.int64)
        srow = tl.load(srow_ptr + r).to(tl.int64)
        role = tl.load(role_ptr + r).to(tl.int32)
        is_v = role == 0
        offs_i = tl.arange(0, BS)
        offs_k = tl.arange(0, KP)
        kmask = (offs_k < K) & (role != 0)
        in_win = slot < TAIL
        is_tail = slot == TAIL
        in_ring = (slot >= RING_LO) & (slot < RING_LO + NR)
        own_slot = OWN_ROW_S // BS
        own_i = OWN_ROW_S % BS
        has_own = (role == 1) & (OWN_ROW_S >= 0) & (slot == own_slot)
        rows = W * BS
        for h in tl.static_range(H):
            hb = h * B + b
            bid_win = tl.load(bmap_ptr + hb * TOPK + tl.minimum(slot, TOPK - 1))
            bid_tail = tl.load(bmap_ptr + hb * TOPK + TAIL) - CONTENT_OFF
            bid_ring = tl.load(ring_ptr + hb * NRP + tl.maximum(slot - RING_LO, 0), mask=in_ring, other=-1)
            bid = tl.where(in_win, bid_win, tl.where(is_tail, bid_tail, tl.where(in_ring, bid_ring, -1)))
            sel = tl.load(sel_ptr + (h * NSEL + srow) * K + offs_k, mask=kmask, other=-2)
            found = (tl.max((sel == bid).to(tl.int32), 0) > 0) & (bid >= 0)
            rd = tl.load(ready_ptr + hb * W + slot) != 0
            full_rows = tl.where(is_tail, TAIL_ROWS_S, BS)
            vis_s = tl.where(found & rd & (in_win | is_tail | in_ring), full_rows, 0)
            vis_v = tl.where(in_win, BS, tl.where(is_tail, TAIL_ROWS_V, 0))
            vis = tl.where(is_v, vis_v, vis_s)
            fold = has_own & (vis == own_i)
            extra = has_own & (vis < own_i)
            vis = vis + fold.to(tl.int32)
            visible = (offs_i < vis) | (extra & (offs_i == own_i))
            flat = slot * BS + offs_i
            cis = tl.load(cis_ptr + (b * rows + flat) * H + h)
            outv = tl.where(visible, cis.to(tl.float32), MASKED)
            tl.store(bias_ptr + (r.to(tl.int64) * rows + flat) * H + h, outv.to(cis.dtype))
            tl.store(prefix_ptr + (r * W + slot) * H + h, vis)
            cand = tl.where(vis > 0, slot * BS + vis, 0)
            cand = tl.maximum(cand, tl.where(extra, OWN_ROW_S + 1, 0))
            tl.atomic_max(extent_ptr + r, cand)

    _KERNEL = paired_bias_kernel
    return _KERNEL


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def fused_bias_triton(a: FusedArgs, out: torch.Tensor, prefix: torch.Tensor, extent: torch.Tensor) -> _ra.RowsBias:
    """ONE kernel launch over (R, W) programs (+ the extent memset)."""
    R = check_args(a, out, prefix, extent)
    H = a.cis.shape[2]
    K = a.sel.shape[2]
    extent[:R].zero_()
    kern = _kernel()
    kern[(R, a.W)](a.cis, out, prefix, extent, a.sel, a.bmap, a.ring_ids, a.ready, a.req, a.srow, a.role,
                   a.cis.shape[0], a.W, K, a.sel.shape[1], a.n_ring, max(a.n_ring, 1), a.topk, a.tail_slot, a.ring_lo,
                   a.tail_rows_v, a.tail_rows_s, a.own_row_s, a.content_off, float(a.masked),
                   BS=a.bs, H=H, KP=_pow2(K))
    return _result(a, out, prefix, extent, R)
