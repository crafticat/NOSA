"""Pure-torch core of the PAIRED TICK (CPU-runnable; spec
retroinfer-eval docs/superpowers/specs/2026-09-20-paired-tick.md sections 1-3).

Everything here is integer / boolean torch or a small copy; no CUDA extension
is imported. tests/test_paired_core.py executes every function on CPU tensors.
The GPU body (tick.py) adds only the kernels.

THE TICK (spec section 1), per request r at committed position tau:
  V row = the exact row at tau (input: the committed token); its K/V go to
          the tail through the engine's own S == 1 decode body
          (cache_engine.decode_update_has_kv_bias: tail write, diff, the two
          blocking Triton gathers, the write-back and rename on a fill).
  S row = the draft row at tau + 1 (input: S's own output from the previous
          tick, or the restart's); its K/V go to a PROVISIONAL slot that only
          S's mask names in this tick; never persisted.
Rows are ADJACENT: row 2r = V, row 2r + 1 = S (RowPlan), so the two rows of a
request are neighbours in the kernel's grid (L2 reuse of the request's K/V).

SLOT LAYOUT of one request (W = topk + R + 2 slots of block_size rows; the
allocation the engine makes under NOSI_VERIFY_ROUND_SLOTS = R > 0,
cache_engine.py prefill_update: _gpu_slots = topk + R + 2):
    0 .. topk-2      the decode's window (diff-managed: ids in _block_map)
    topk-1           the LIVE TAIL, block T = _block_map[..., topk-1]
    topk             PROVISIONAL slot: row 0 = S's K/V at tau + 1 (this module)
    topk+1 .. W-3    the STAGING RING of stage E4 (unused in E2: ids -1, not ready)
    W-2, W-1         Path 1's tail mirrors (unused by the paired tick)
The provisional slot sits right after the window so that the S row's extent
is (topk*bs + 1) rows, not W*bs: the rows kernel loads every row below a
row's cache_seqlens (rows_attention.py, PRECONDITION), so the extent IS the
read volume.

WHAT ONE ROW SEES (visible_rows[r, slot, h] = rows of the slot row r attends):
  V: slots 0..topk-2 in full, the tail slot up to tail_len_after (the rows the
     shipped decode reads: _cache_lens = (topk-1)*bs + tail_len, cache_engine
     S == 1 body), nothing else. After a fill (tail_len_after == 0) the tail
     slot shows 0 rows, exactly as the decode at the fill step attends
     (topk-1)*bs rows (the just-completed block is on the host until the next
     diff fetches it into the window) -- G6 follows the S == 1 body.
  S: a window / ring slot in full iff its block id is in S's selection AND the
     slot is READY (resident, event complete: G4); the tail slot up to
     tail_rows (bs when slot topk-1 physically holds the just-completed block
     T, else tail_len_after) iff T is in S's selection; its own provisional
     row, always. Never the mirrors.
cache_seqlens[r] = the EXACT extent = max over visible slots of
(slot*bs + visible_rows): the last n-block is predicated by the kernel at
that bound (the shipped decode's rule), so rows above it are never loaded.

CANONICAL ORDER (G2). A row's bias is a dense per-slot tensor; the kernel's
fp32 reduction runs over PHYSICAL columns (descending n-block inside a split,
splits combined in index order: flash_fwd_kernel.h:594-597, :1331-1339), and
which block sits in which slot is decided by diff per (head, request) from
that request's own history (float-order law: a function of request i's own
state). ``canonical_order`` lists a row's visible slots in ascending
(block id, slot) order: the deterministic NAMING every mask is built from
(tests: two builds from equal state give torch.equal masks; the order never
depends on the batch or on neighbours).

MASKED = -3.0e4, finite on purpose (rows_attention.py module docstring:
split-KV rescale with two consecutive fully-masked blocks; exp2 underflows
to exactly 0 for any scale >= 0.08) -- G7.

ACCOUNTING per (layer, head, request), never assumed: V's residual misses
(blocks diff fetched this tick = _load_mask >= 0), S's selection size and its
resident hits, and the DIVERGENCE between S's selection last tick and V's
selection this tick at the same position: |S \\ V| (predicted, not needed),
|V \\ S| (needed, not predicted), |S ∩ V|. Bytes = blocks x BYTES_PER_BLOCK
(K + V rows of one block: 2 x 64 x 128 x 2 B, spec_loop.BYTES_WIRE_PER_BLOCK).

COMMIT LAW (G8): lockstep; exactly one committed token per request per tick
(``tick_end`` asserts it). accept iff the draft S consumed equals the token
the next V row consumes (in production argmax(V); under teacher forcing the
forced token). On accept S's output becomes the next draft; on reject the
chain is broken and the request joins the RESTART set: one resident S-row
forward at tau + 1 on the committed token (spec section 3 option (a),
sequential, batched over the rejected requests), whose output is the draft
for tau + 2. Nothing persistent changes on a reject (G5): the provisional row
is overwritten by the next S write (and NaN-poisoned in between under
NOSI_PAIRED_POISON=1, G1/G7), the layer tables the S position touched are
restored from the journal (spec_loop.LayerJournal), the draft token is cleared.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

import torch

from ..verify import rows_attention as _ra
from ..verify import tail_write as _tw
from .. import spec_loop as _sl

MASKED = float(_ra.MASKED)
BLOCK_TOKENS = 64
HEAD_DIM = 128
BYTES_PER_BLOCK = 2 * BLOCK_TOKENS * HEAD_DIM * 2
if BYTES_PER_BLOCK != _sl.BYTES_WIRE_PER_BLOCK:
    raise AssertionError("BYTES_PER_BLOCK %d != spec_loop.BYTES_WIRE_PER_BLOCK %d" % (BYTES_PER_BLOCK, _sl.BYTES_WIRE_PER_BLOCK))

ROLE_V, ROLE_S = 0, 1


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
class Layout(NamedTuple):
    topk: int
    R: int            # NOSI_VERIFY_ROUND_SLOTS (0 = the NARROW layout)
    bs: int
    W: int            # topk (narrow) or topk + R + 2 (union)
    tail_slot: int    # topk - 1
    prov_slot: int    # narrow: the tail slot (the provisional row is ROW tail_len of it); union: slot topk, row 0
    ring_lo: int      # union: topk + 1 (inclusive); narrow: W (empty)
    ring_hi: int      # union: W - 2 (exclusive); narrow: W (empty)
    mirror_lo: int    # union: W - 2; narrow: -1 (none)
    mirror_hi: int    # union: W - 1; narrow: -1
    narrow: bool


def paired_layout(topk: int, round_slots: int, block_size: int = 64) -> Layout:
    """R = 0: the NARROW layout (E2c) = the SHIPPED allocation, W = topk slots.
    S's provisional K/V is written to ROW tail_len_after of the tail slot: the
    row the next V write overwrites, which the shipped decode never reads
    (cache_seqlens = (topk-1)*bs + tail_len: cache_engine.py S == 1 body) and
    a fill's write-back copies only after V overwrote it. Both rows' extents
    stay <= topk*bs, so ONE call over the allocation the decode itself uses
    keeps the decode's split seams: n_blocks_per_split = ceil(ceil(seqlen_k /
    kBlockN) / num_splits) with seqlen_k = the ALLOCATED length
    (flash_fwd_kernel.h:594, flash_api.cpp:338) -- any extra slot (W = 65 ->
    33 n-blocks -> 9 per split instead of 8) would move the seams.
    R >= 1: the UNION layout (W = topk + R + 2, union_store.union_layout with
    no pool): provisional slot topk row 0, ring topk+1 .. W-3, mirrors W-2/W-1."""
    topk, R, bs = int(topk), int(round_slots), int(block_size)
    if topk < 2 or R < 0 or bs < 1:
        raise ValueError("paired_layout: topk=%d round_slots=%d block_size=%d (need topk >= 2, R >= 0)" % (topk, R, bs))
    if R == 0:
        return Layout(topk=topk, R=0, bs=bs, W=topk, tail_slot=topk - 1, prov_slot=topk - 1, ring_lo=topk, ring_hi=topk,
                      mirror_lo=-1, mirror_hi=-1, narrow=True)
    W = topk + R + 2
    return Layout(topk=topk, R=R, bs=bs, W=W, tail_slot=topk - 1, prov_slot=topk, ring_lo=topk + 1, ring_hi=W - 2,
                  mirror_lo=W - 2, mirror_hi=W - 1, narrow=False)


def check_layout_against_engine(lay: Layout, engine) -> None:
    """Guard where the value is produced: the engine's allocation and counters
    must be the layout's (cache_engine.py: _gpu_slots, _tail_block_idx_on_gpu)."""
    if int(engine.topk) != lay.topk or int(engine.block_size) != lay.bs:
        raise ValueError("engine topk/block_size (%d, %d) != layout (%d, %d)" % (engine.topk, engine.block_size, lay.topk, lay.bs))
    if int(engine.verify_round_slots) != lay.R:
        raise ValueError("engine verify_round_slots=%d != layout R=%d" % (engine.verify_round_slots, lay.R))
    if int(engine.pool_blocks) != 0:
        raise ValueError("the paired tick runs without the victim pool (pool_blocks=%d)" % engine.pool_blocks)
    if engine._k_gpu.shape[1] != lay.W * lay.bs:
        raise ValueError("allocation has %d rows, layout W=%d x bs=%d" % (engine._k_gpu.shape[1], lay.W, lay.bs))
    if int(engine._tail_block_idx_on_gpu) != lay.tail_slot:
        raise ValueError("_tail_block_idx_on_gpu=%d != tail_slot=%d" % (engine._tail_block_idx_on_gpu, lay.tail_slot))


# ---------------------------------------------------------------------------
# row layout: adjacent [V | S] rows
# ---------------------------------------------------------------------------
class RowPlan(NamedTuple):
    req: torch.Tensor    # (R,) int64: the request of each row
    role: torch.Tensor   # (R,) int8: ROLE_V / ROLE_S
    U: int               # rows per request: 2 (paired), 1 (twin: V only; srows: S only)
    has_v: bool
    has_s: bool


def row_plan(req_idx: torch.Tensor, has_v: bool, has_s: bool) -> RowPlan:
    """req_idx (n,) int64 = the requests taking part (arange(B) for a whole
    batch; the rejected subset for a restart). Paired: rows 2i = V, 2i + 1 = S
    of req_idx[i] (adjacent, spec section 1). Exactly one of has_v / has_s may
    be False (twin / restart); both False is refused."""
    if req_idx.dim() != 1 or req_idx.dtype != torch.int64 or req_idx.numel() < 1:
        raise ValueError("req_idx must be a non-empty int64 vector, got %s %s" % (req_idx.dtype, tuple(req_idx.shape)))
    if not (has_v or has_s):
        raise ValueError("row_plan: a row layout needs V rows, S rows or both")
    roles = ([ROLE_V] if has_v else []) + ([ROLE_S] if has_s else [])
    U = len(roles)
    req = req_idx.repeat_interleave(U)
    role = torch.tensor(roles, dtype=torch.int8, device=req_idx.device).repeat(req_idx.numel())
    return RowPlan(req=req, role=role, U=U, has_v=bool(has_v), has_s=bool(has_s))


# ---------------------------------------------------------------------------
# slot ids, readiness (G4), visible rows
# ---------------------------------------------------------------------------
def slot_ids(block_map: torch.Tensor, tail_content_id: torch.Tensor, lay: Layout,
             ring_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(H, B, W) int64 block id per slot of the whole allocation, -1 where
    nothing addressable sits. Window slots from _block_map; the tail slot
    carries the id of the block whose rows PHYSICALLY sit there
    (tail_content_id (H, B): T, or T after a fill although _block_map already
    says T+1); the provisional slot and the mirrors are -1 (never matched by
    id: the provisional row is S's own, added explicitly); the ring carries
    ring_ids (H, B, ring_hi - ring_lo) in E4, -1 in E2."""
    H, B, topk = block_map.shape
    if topk != lay.topk or block_map.dtype != torch.int64:
        raise ValueError("block_map must be int64 (H, B, topk=%d), got %s %s" % (lay.topk, block_map.dtype, tuple(block_map.shape)))
    if tuple(tail_content_id.shape) != (H, B) or tail_content_id.dtype != torch.int64:
        raise ValueError("tail_content_id must be int64 (H, B), got %s %s" % (tail_content_id.dtype, tuple(tail_content_id.shape)))
    ids = torch.full((H, B, lay.W), -1, dtype=torch.int64, device=block_map.device)
    ids[..., :lay.tail_slot] = block_map[..., :lay.tail_slot]
    ids[..., lay.tail_slot] = tail_content_id
    if ring_ids is not None:
        n_ring = lay.ring_hi - lay.ring_lo
        if tuple(ring_ids.shape) != (H, B, n_ring) or ring_ids.dtype != torch.int64:
            raise ValueError("ring_ids must be int64 (H, B, %d), got %s %s" % (n_ring, ring_ids.dtype, tuple(ring_ids.shape)))
        ids[..., lay.ring_lo:lay.ring_hi] = ring_ids
    return ids


def ready_mask(H: int, B: int, lay: Layout, device=None, ring_ready: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(H, B, W) bool: which slots an S row may READ this tick (G4).
    Window and tail: True (the V row's gathers ran BLOCKING on the main stream
    before the attention is enqueued: stream order = complete). Provisional:
    True (S writes it before attending). Ring: ``ring_ready`` (E4: one flag per
    slot, True only after the host observed that slot's gather event complete
    -- spec_loop.Store.mark_arrived semantics), False in E2. Mirrors: False."""
    ready = torch.zeros((H, B, lay.W), dtype=torch.bool, device=device)
    ready[..., :lay.tail_slot + 1] = True
    ready[..., lay.prov_slot] = True
    if ring_ready is not None and lay.ring_hi > lay.ring_lo:
        n_ring = lay.ring_hi - lay.ring_lo
        if tuple(ring_ready.shape) != (H, B, n_ring) or ring_ready.dtype != torch.bool:
            raise ValueError("ring_ready must be bool (H, B, %d), got %s %s" % (n_ring, ring_ready.dtype, tuple(ring_ready.shape)))
        ready[..., lay.ring_lo:lay.ring_hi] = ring_ready
    return ready


def v_visible_rows(H: int, B: int, lay: Layout, tail_len_after: int, device=None) -> torch.Tensor:
    """(B, W, H) int64: what a V row attends = what the shipped decode reads
    after the S == 1 update: the window in full, the tail slot up to
    tail_len_after (0 at a fill step), nothing above the window."""
    tl = int(tail_len_after)
    if not (0 <= tl < lay.bs):
        raise ValueError("tail_len_after=%d outside [0, bs=%d) (the S == 1 body resets it to 0 on a fill)" % (tl, lay.bs))
    vis = torch.zeros((B, lay.W, H), dtype=torch.int64, device=device)
    vis[:, :lay.tail_slot, :] = lay.bs
    vis[:, lay.tail_slot, :] = tl
    return vis


def s_visible_rows(ids: torch.Tensor, ready: torch.Tensor, sel: torch.Tensor, lay: Layout, tail_rows: int) -> torch.Tensor:
    """(B, W, H) int64: the PREFIX rows an S row attends per slot. ids / ready
    (H, B, W) from slot_ids / ready_mask; sel (H, B, K) int64 = S's own
    selection (block ids, -1 padded); tail_rows = rows of the tail slot that
    hold exact KV of the block named by ids[..., tail_slot] (bs after a fill,
    else tail_len_after). Rule: slot visible in full iff (id in sel) and
    ready, for every slot but the tail (tail_rows instead of bs); the
    mirrors never. S's OWN provisional row is not part of this prefix: it is
    given to paired_bias as ``own_rows`` (it extends the prefix when it sits
    right after it, the common case, and is an extra visible row otherwise)."""
    H, B, W = ids.shape
    if W != lay.W or tuple(ready.shape) != (H, B, W) or ready.dtype != torch.bool:
        raise ValueError("ids %s / ready %s must be (H, B, W=%d), ready bool" % (tuple(ids.shape), tuple(ready.shape), lay.W))
    if sel.dim() != 3 or tuple(sel.shape[:2]) != (H, B) or sel.dtype != torch.int64:
        raise ValueError("sel must be int64 (H, B, K), got %s %s" % (sel.dtype, tuple(sel.shape)))
    tr = int(tail_rows)
    if not (0 <= tr <= lay.bs):
        raise ValueError("tail_rows=%d outside [0, bs=%d]" % (tr, lay.bs))
    valid_id = ids >= 0
    named = ((ids.unsqueeze(-1) == sel.unsqueeze(-2)) & (sel >= 0).unsqueeze(-2)).any(-1) & valid_id & ready   # (H, B, W)
    rows = torch.full((H, B, W), lay.bs, dtype=torch.int64, device=ids.device)
    rows[..., lay.tail_slot] = tr
    vis = torch.where(named, rows, torch.zeros_like(rows))
    if not lay.narrow:
        vis[..., lay.prov_slot] = 0
        vis[..., lay.mirror_lo] = 0
        vis[..., lay.mirror_hi] = 0
    return vis.permute(1, 2, 0).contiguous()   # (B, W, H)


def own_rows(plan: RowPlan, own_row_s: int) -> torch.Tensor:
    """(R,) int64: the flat row of S's provisional K/V for the S rows of the
    plan (provisional_row), -1 for V rows."""
    return torch.where(plan.role == ROLE_S, torch.full_like(plan.req, int(own_row_s)), torch.full_like(plan.req, -1))


def assemble_rows(plan: RowPlan, vis_v: Optional[torch.Tensor], vis_s: Optional[torch.Tensor], req_idx: torch.Tensor):
    """Interleave the per-request visible-row tensors into the row order of
    ``plan``: (R, W, H) int64 and cache_batch_idx (R,) int32. vis_v / vis_s are
    (n, W, H) over the n requests of req_idx (row i <-> req_idx[i])."""
    n = req_idx.numel()
    parts = []
    if plan.has_v:
        if vis_v is None or vis_v.shape[0] != n:
            raise ValueError("vis_v must be (n=%d, W, H)" % n)
        parts.append(vis_v)
    if plan.has_s:
        if vis_s is None or vis_s.shape[0] != n:
            raise ValueError("vis_s must be (n=%d, W, H)" % n)
        parts.append(vis_s)
    W, H = parts[0].shape[1:]
    vis = torch.stack(parts, dim=1).reshape(n * plan.U, W, H)   # row i*U + u
    if plan.req.numel() != n * plan.U:
        raise AssertionError("row plan has %d rows, req_idx x U = %d" % (plan.req.numel(), n * plan.U))   # no device read: plan.req was built from req_idx
    return vis, plan.req.to(torch.int32)


# ---------------------------------------------------------------------------
# the rows bias (the ONE thing the attention call takes besides q / K / V)
# ---------------------------------------------------------------------------
def paired_bias(cis_rows: torch.Tensor, visible_rows: torch.Tensor, cache_batch_idx: torch.Tensor,
                U: int, masked_value: float = MASKED, out: Optional[torch.Tensor] = None, *, check_values: bool,
                own_rows: Optional[torch.Tensor] = None) -> _ra.RowsBias:
    """cis_rows (B, W*bs, H): the engine's _kv_bias_gpu (store dtype);
    visible_rows (R, W, H) int64; cache_batch_idx (R,) int32. Returns a
    rows_attention.RowsBias: bias[r, slot*bs + i, h] = cis of request
    cache_batch_idx[r] where i < visible_rows[r, slot, h], MASKED elsewhere;
    cache_seqlens = the exact extent. ``out`` (>= R rows, contiguous, the
    allocation's row shape) receives the bias in place (no allocation on the
    steady path); its storage must hold >= B rows for the kernel's host check
    (rows_attention.py: the narrowed view). ``own_rows`` (R,) int64: a flat row
    that is visible for the row whatever the prefix says (S's provisional
    K/V; -1 = none): when it sits right after the slot's prefix the prefix is
    extended by one (visible_rows reports it); when the prefix is shorter (the
    tail block not selected) it is an EXTRA visible row that visible_rows does
    not report (the bias and the extent do). ``check_values`` reads the index /
    row tensors on the host (4 device syncs): the CPU tests pass True; the GPU
    body passes False because both come from its own bounded rules
    (v_visible_rows / s_visible_rows / row_plan) -- job 2175550 showed the
    syncs inside every layer's fetch bracket."""
    if cis_rows.dim() != 3 or not cis_rows.is_floating_point():
        raise ValueError("cis_rows must be a floating (B, W*bs, H), got %s %s" % (cis_rows.dtype, tuple(cis_rows.shape)))
    B, rows, H = cis_rows.shape
    if visible_rows.dim() != 3 or visible_rows.dtype != torch.int64 or visible_rows.shape[2] != H:
        raise ValueError("visible_rows must be int64 (R, W, H=%d), got %s %s" % (H, visible_rows.dtype, tuple(visible_rows.shape)))
    R, W, _ = visible_rows.shape
    if rows % W != 0:
        raise ValueError("cis_rows has %d rows, not a multiple of W=%d" % (rows, W))
    bs = rows // W
    if cache_batch_idx.dtype != torch.int32 or tuple(cache_batch_idx.shape) != (R,):
        raise ValueError("cache_batch_idx must be int32 (R=%d,), got %s %s" % (R, cache_batch_idx.dtype, tuple(cache_batch_idx.shape)))
    if check_values:
        if R > 0 and (int(cache_batch_idx.min()) < 0 or int(cache_batch_idx.max()) >= B):
            raise ValueError("cache_batch_idx must index requests 0..B-1=%d" % (B - 1))
        if bool((visible_rows < 0).any()) or bool((visible_rows > bs).any()):
            raise ValueError("visible_rows must lie in [0, bs=%d]" % bs)
    dev = cis_rows.device
    extra_extent = None
    if own_rows is not None:
        if own_rows.dtype != torch.int64 or tuple(own_rows.shape) != (R,):
            raise ValueError("own_rows must be int64 (R=%d,), got %s %s" % (R, own_rows.dtype, tuple(own_rows.shape)))
        visible_rows = visible_rows.clone()
        has = own_rows >= 0
        own_slot = torch.where(has, own_rows // bs, torch.zeros_like(own_rows))
        own_i = torch.where(has, own_rows % bs, torch.zeros_like(own_rows))
        cur = visible_rows.gather(1, own_slot.view(R, 1, 1).expand(R, 1, H)).squeeze(1)          # (R, H)
        fold = has.view(R, 1) & (cur == own_i.view(R, 1))
        extra = has.view(R, 1) & (cur < own_i.view(R, 1))
        visible_rows.scatter_add_(1, own_slot.view(R, 1, 1).expand(R, 1, H), fold.to(torch.int64).view(R, 1, H))
        extra_extent = torch.where(extra.any(1), own_rows + 1, torch.zeros_like(own_rows))
    cis = cis_rows.index_select(0, cache_batch_idx.to(torch.int64)).view(R, W, bs, H)
    i_idx = torch.arange(bs, dtype=torch.int64, device=dev).view(1, 1, bs, 1)
    vis = i_idx < visible_rows.unsqueeze(2)                                                             # (R, W, bs, H)
    if own_rows is not None:
        # the EXTRA own rows (prefix shorter than the own row): a broadcast mask, no host sync, no per-row loop
        s_idx = torch.arange(W, dtype=torch.int64, device=dev).view(1, W, 1, 1)
        vis = vis | (extra.view(R, 1, 1, H) & (s_idx == own_slot.view(R, 1, 1, 1)) & (i_idx == own_i.view(R, 1, 1, 1)))
    neg = torch.tensor(float(masked_value), dtype=cis_rows.dtype, device=dev)
    if out is None:
        bias = torch.where(vis, cis, neg).view(R, rows, H)
    else:
        if out.dim() != 3 or out.shape[0] < max(R, B) or tuple(out.shape[1:]) != (rows, H) or out.dtype != cis_rows.dtype or not out.is_contiguous():
            raise ValueError("out must be a contiguous (>= max(R, B)=%d, %d, %d) %s tensor, got %s %s" % (max(R, B), rows, H, cis_rows.dtype, out.dtype, tuple(out.shape)))
        torch.where(vis, cis, neg, out=out[:R].view(R, W, bs, H))
        bias = out
    slot = torch.arange(W, dtype=torch.int64, device=dev).view(1, W, 1)
    extent = torch.where(visible_rows > 0, slot * bs + visible_rows, torch.zeros_like(visible_rows)).amax(dim=(1, 2))
    if extra_extent is not None:
        extent = torch.maximum(extent, extra_extent)
    return _ra.RowsBias(bias=bias, selected=visible_rows > 0, visible_rows=visible_rows,
                        cache_batch_idx=cache_batch_idx.contiguous(), cache_seqlens=extent.to(torch.int32).contiguous(),
                        U=int(U), block_size=bs)


def canonical_order(ids_row: torch.Tensor, visible_row: torch.Tensor) -> torch.Tensor:
    """G2: the visible slots of one (row, head), as a deterministic list in
    ascending (block id, slot) order. ids_row (W,) int64 (-1 = no id: the
    provisional slot sorts by slot among the id-less), visible_row (W,) int64."""
    W = ids_row.numel()
    slots = torch.arange(W, dtype=torch.int64, device=ids_row.device)
    keep = visible_row > 0
    above = int(ids_row.max()) + 1 if W else 0                                   # larger than every real id: id-less slots sort last
    key = torch.where(ids_row >= 0, ids_row, torch.full_like(ids_row, above)) * W + slots   # (id, slot) lexicographic
    return slots[keep][torch.argsort(key[keep], stable=True)]


# ---------------------------------------------------------------------------
# accounting: misses, divergence, bytes
# ---------------------------------------------------------------------------
def set_counts(a: torch.Tensor, b: torch.Tensor):
    """a, b (..., K) int64 block ids (-1 padded, distinct per row as top-k
    emits them). Returns (|a \\ b|, |b \\ a|, |a ∩ b|), each (...,) int64, over
    the valid ids. Elementwise over the leading dims: stream (h, b) never
    reads stream (h', b')."""
    if a.dtype != torch.int64 or b.dtype != torch.int64 or a.shape[:-1] != b.shape[:-1]:
        raise ValueError("set_counts: a %s %s, b %s %s" % (a.dtype, tuple(a.shape), b.dtype, tuple(b.shape)))
    va, vb = a >= 0, b >= 0
    a_in_b = ((a.unsqueeze(-1) == b.unsqueeze(-2)) & vb.unsqueeze(-2)).any(-1) & va
    b_in_a = ((b.unsqueeze(-1) == a.unsqueeze(-2)) & va.unsqueeze(-2)).any(-1) & vb
    return (va & ~a_in_b).sum(-1), (vb & ~b_in_a).sum(-1), a_in_b.sum(-1)


def blocks_to_bytes(n_blocks) -> int:
    return int(n_blocks) * BYTES_PER_BLOCK


class LayerAccount(NamedTuple):
    """One layer's counts of one tick, (H, n) int64 each (n = requests in the
    call); None where the call has no such row."""
    v_miss: Optional[torch.Tensor]        # V: blocks diff fetched (residual misses)
    s_sel: Optional[torch.Tensor]         # S: distinct valid ids in its selection
    s_hit: Optional[torch.Tensor]         # S: selected slots that were resident+ready (attended)
    div_s_not_v: Optional[torch.Tensor]   # |S_prev \ V|: predicted, not needed
    div_v_not_s: Optional[torch.Tensor]   # |V \ S_prev|: needed, not predicted
    div_both: Optional[torch.Tensor]      # |S_prev ∩ V|
    tail_len_after: int
    filled: bool


def layer_account(v_miss, sel_s, vis_s, sel_v, sel_s_prev, lay: Layout, tail_len_after: int, filled: bool) -> LayerAccount:
    """v_miss (H, n) or None; sel_s (H, n, K) / vis_s (n, W, H) or None;
    sel_v (H, n, K) and sel_s_prev (H, n, K) (last tick's S selection for the
    same requests) or None."""
    s_sel = s_hit = None
    if sel_s is not None:
        s_sel = (sel_s >= 0).sum(-1)
        if vis_s is None:
            raise ValueError("vis_s is needed with sel_s")
        # attended = window slots + ring slots the row sees in full (the tail and the provisional row are not selections)
        s_hit = (vis_s[:, :lay.tail_slot, :] > 0).sum(1).t() + (vis_s[:, lay.ring_lo:lay.ring_hi, :] > 0).sum(1).t()   # (H, n)
    d1 = d2 = d3 = None
    if sel_v is not None and sel_s_prev is not None:
        d1, d2, d3 = set_counts(sel_s_prev, sel_v)
    return LayerAccount(v_miss=v_miss, s_sel=s_sel, s_hit=s_hit, div_s_not_v=d1, div_v_not_s=d2, div_both=d3,
                        tail_len_after=int(tail_len_after), filled=bool(filled))


class Ledger:
    """Per-tick accumulation of the per-layer accounts, kept as tensors
    (Lyr, H, B) on whatever device the calls ran on; ``summary`` moves the
    totals to the host once."""

    FIELDS = ("v_miss", "s_sel", "s_hit", "div_s_not_v", "div_v_not_s", "div_both")

    def __init__(self):
        self.ticks: List[dict] = []
        self.restarts: List[dict] = []

    def add_tick(self, tick_index: int, layers: Sequence[LayerAccount], accepted: torch.Tensor,
                 greedy_agree: Optional[torch.Tensor], n_restart: int, ms: Optional[float] = None):
        rec = dict(tick=int(tick_index), n_restart=int(n_restart), ms=ms,
                   accepted=int(accepted.sum()), n=int(accepted.numel()),
                   greedy_agree=(int(greedy_agree.sum()) if greedy_agree is not None else None),
                   tail_len_after=[la.tail_len_after for la in layers], filled=[la.filled for la in layers])
        for f in self.FIELDS:
            vals = [getattr(la, f) for la in layers]
            rec[f] = torch.stack(vals, 0) if all(v is not None for v in vals) else None
        self.ticks.append(rec)

    def add_restart(self, tick_index: int, rows: int, ms: Optional[float], layers: Sequence[LayerAccount]):
        rec = dict(tick=int(tick_index), rows=int(rows), ms=ms)
        vals = [la.s_sel for la in layers]
        rec["s_sel"] = torch.stack(vals, 0) if (vals and all(v is not None for v in vals)) else None
        self.restarts.append(rec)

    def summary(self) -> dict:
        n_ticks = len(self.ticks)
        out = dict(ticks=n_ticks, restarts=len(self.restarts),
                   restart_rows=sum(r["rows"] for r in self.restarts),
                   restart_ms=[r["ms"] for r in self.restarts],
                   tick_ms=[t["ms"] for t in self.ticks],
                   accepted=sum(t["accepted"] for t in self.ticks), rows=sum(t["n"] for t in self.ticks),
                   greedy_agree=(sum(t["greedy_agree"] for t in self.ticks) if all(t["greedy_agree"] is not None for t in self.ticks) and n_ticks else None))
        out["acceptance_rate"] = (out["accepted"] / out["rows"]) if out["rows"] else None
        for f in self.FIELDS:
            ts = [t[f] for t in self.ticks if t[f] is not None]
            if ts:
                stack = torch.stack(ts, 0).to(torch.float64)          # (ticks, Lyr, H, B)
                out[f + "_total"] = int(stack.sum())
                out[f + "_mean_per_layer_head_req"] = float(stack.mean())
                out[f + "_per_layer_mean"] = stack.mean(dim=(0, 2, 3)).tolist()
            else:
                out[f + "_total"] = None
        if out.get("v_miss_total") is not None:
            out["v_miss_bytes"] = blocks_to_bytes(out["v_miss_total"])
            out["v_miss_bytes_per_committed_token"] = out["v_miss_bytes"] / max(1, out["rows"])
        if out.get("div_both_total") is not None and out.get("div_s_not_v_total") is not None:
            pred = out["div_both_total"] + out["div_s_not_v_total"]
            need = out["div_both_total"] + out["div_v_not_s_total"]
            out["prefetch_precision"] = (out["div_both_total"] / pred) if pred else None
            out["prefetch_recall"] = (out["div_both_total"] / need) if need else None
        return out


# ---------------------------------------------------------------------------
# the state machine (G8)
# ---------------------------------------------------------------------------
class TickOutcome(NamedTuple):
    committed: torch.Tensor     # (B,) int64: exactly one token per request (G8)
    accepted: torch.Tensor      # (B,) bool: the draft S consumed equals the committed token
    next_draft: torch.Tensor    # (B,) int64: S's output where accepted, -1 elsewhere
    restart_idx: torch.Tensor   # (n_rej,) int64: requests that need the restart forward
    had_draft: torch.Tensor     # (B,) bool: the request entered the tick with a draft


class DraftState:
    """Per-request draft token (-1 = none), the only cross-tick state of S."""

    def __init__(self, B: int, device=None):
        if B < 1:
            raise ValueError("B must be >= 1")
        self.B = int(B)
        self.draft = torch.full((self.B,), -1, dtype=torch.int64, device=device)

    def s_inputs(self, fallback: torch.Tensor) -> torch.Tensor:
        """The S rows' input tokens: the draft, or ``fallback`` (the V input)
        where no draft exists (the row still runs, lockstep; it is rejected at
        tick end because had_draft is False)."""
        if tuple(fallback.shape) != (self.B,) or fallback.dtype != torch.int64:
            raise ValueError("fallback must be int64 (B,)")
        return torch.where(self.draft >= 0, self.draft, fallback)

    def tick_end(self, committed: torch.Tensor, s_argmax: torch.Tensor) -> TickOutcome:
        """committed (B,) int64: the token every request commits this tick (the
        next V input); s_argmax (B,) int64: argmax of the S rows. Lockstep:
        every request commits exactly one token; accept iff it had a draft and
        the draft equals the committed token; S's output becomes the next
        draft only then."""
        for name, t in (("committed", committed), ("s_argmax", s_argmax)):
            if tuple(t.shape) != (self.B,) or t.dtype != torch.int64:
                raise ValueError("%s must be int64 (B=%d,), got %s %s" % (name, self.B, t.dtype, tuple(t.shape)))
        if bool((committed < 0).any()):
            raise ValueError("G8: every request commits exactly one token per tick; a negative token is not a token")
        had = self.draft >= 0
        accepted = had & (self.draft == committed)
        next_draft = torch.where(accepted, s_argmax, torch.full_like(s_argmax, -1))
        restart_idx = (~accepted).nonzero().squeeze(-1)
        self.draft = next_draft
        return TickOutcome(committed=committed, accepted=accepted, next_draft=next_draft, restart_idx=restart_idx, had_draft=had)

    def apply_restart(self, restart_idx: torch.Tensor, restart_argmax: torch.Tensor) -> None:
        """The restart forward's outputs become the drafts of the restarted
        requests (spec section 3 option (a))."""
        if restart_idx.dim() != 1 or tuple(restart_argmax.shape) != tuple(restart_idx.shape) or restart_argmax.dtype != torch.int64:
            raise ValueError("restart_idx %s / restart_argmax %s must be int64 vectors of one length" % (tuple(restart_idx.shape), tuple(restart_argmax.shape)))
        if restart_idx.numel() and bool((self.draft[restart_idx] >= 0).any()):
            raise AssertionError("apply_restart over a request that still has a draft")
        self.draft[restart_idx] = restart_argmax


ACTIVE, CATCHING = 0, 1


class CatchupOutcome(NamedTuple):
    committed: torch.Tensor     # (B,) int64: one token per request (G8)
    accepted: torch.Tensor      # (B,) bool: ACTIVE requests whose draft equalled the committed token
    rejoined: torch.Tensor      # (B,) bool: CATCHING requests whose catch-up token 1 equalled the committed token (they rejoin next tick)
    catch_idx: torch.Tensor     # (n,) int64: requests that need a catch-up forward DURING the next tick
    catch_in: torch.Tensor      # (n,) int64: its input token = the token they just committed (position tau + 1)
    had_draft: torch.Tensor     # (B,) bool


class CatchupState:
    """Spec section 3 option (b), the OVERLAPPED restart. A request rejected at
    tick t (V at tau) goes CATCHING: during tick t + 1 a two-position catch-up
    forward runs on a side stream for it -- position tau + 1 on the committed
    token x_{tau+1} gives tok1 = x^_{tau+2}, position tau + 2 on tok1 gives
    tok2 = x^_{tau+3}; its S row in tick t + 1 is a dummy. At tick t + 1's end
    tok1 is judged against the committed x_{tau+2}: equal -> the request
    rejoins at tick t + 2 with draft tok2 (S at tau + 3); else it stays
    CATCHING with a new catch-up on x_{tau+2}. Lockstep: every request commits
    one token per tick whatever its mode (G8)."""

    def __init__(self, B: int, device=None):
        if B < 1:
            raise ValueError("B must be >= 1")
        self.B = int(B)
        z = lambda: torch.full((self.B,), -1, dtype=torch.int64, device=device)
        self.draft, self.catch_in, self.tok1, self.tok2 = z(), z(), z(), z()
        self.mode = torch.full((self.B,), CATCHING, dtype=torch.int8, device=device)   # no draft yet: every request needs the first catch-up

    def s_inputs(self, fallback: torch.Tensor) -> torch.Tensor:
        return torch.where(self.draft >= 0, self.draft, fallback)

    def catch_set(self):
        """The requests whose catch-up must run during the coming tick and their inputs."""
        idx = (self.mode == CATCHING).nonzero().squeeze(-1)
        return idx, self.catch_in[idx]

    def apply_catchup(self, idx: torch.Tensor, tok1: torch.Tensor, tok2: torch.Tensor) -> None:
        """The side forward's results, read before the tick's end."""
        if idx.dim() != 1 or tuple(tok1.shape) != tuple(idx.shape) or tuple(tok2.shape) != tuple(idx.shape):
            raise ValueError("apply_catchup: idx %s tok1 %s tok2 %s" % (tuple(idx.shape), tuple(tok1.shape), tuple(tok2.shape)))
        if idx.numel() and bool((self.mode[idx] != CATCHING).any()):
            raise AssertionError("apply_catchup over a request that is not catching up")
        self.tok1[idx] = tok1
        self.tok2[idx] = tok2

    def tick_end(self, committed: torch.Tensor, s_argmax: torch.Tensor) -> CatchupOutcome:
        for name, t in (("committed", committed), ("s_argmax", s_argmax)):
            if tuple(t.shape) != (self.B,) or t.dtype != torch.int64:
                raise ValueError("%s must be int64 (B=%d,), got %s %s" % (name, self.B, t.dtype, tuple(t.shape)))
        if bool((committed < 0).any()):
            raise ValueError("G8: every request commits exactly one token per tick")
        active = self.mode == ACTIVE
        had = active & (self.draft >= 0)
        accepted = had & (self.draft == committed)
        rejoined = (~active) & (self.tok1 >= 0) & (self.tok1 == committed)
        new_draft = torch.where(accepted, s_argmax, torch.where(rejoined, self.tok2, torch.full_like(self.draft, -1)))
        now_catching = ~(accepted | rejoined)
        self.draft = new_draft
        self.mode = torch.where(now_catching, torch.full_like(self.mode, CATCHING), torch.full_like(self.mode, ACTIVE))
        self.catch_in = torch.where(now_catching, committed, torch.full_like(self.catch_in, -1))
        self.tok1 = torch.full_like(self.tok1, -1)
        self.tok2 = torch.full_like(self.tok2, -1)
        idx = now_catching.nonzero().squeeze(-1)
        return CatchupOutcome(committed=committed, accepted=accepted, rejoined=rejoined, catch_idx=idx, catch_in=committed[idx], had_draft=had)


def catchup_schedule(t_reject: int, tau: int) -> dict:
    """Which tick sees which rows for a request rejected at tick t_reject (V at
    tau): the catch-up runs DURING tick t_reject + 1 at positions tau + 1
    (input x_{tau+1}, the committed token; attends rows <= tau and its own row
    above V's exact tau + 1 row) and tau + 2 (input tok1; attends rows <= tau + 1
    incl. V's exact row); judged at tick t_reject + 1's end against x_{tau+2};
    rejoins at tick t_reject + 2 with S at tau + 3."""
    return dict(run_during=t_reject + 1, pos1=tau + 1, pos2=tau + 2, dummy_s_tick=t_reject + 1, judged_at=t_reject + 1,
                rejoin_tick=t_reject + 2, rejoin_pos=tau + 3, pos1_attends_upto=tau, pos2_attends_upto=tau + 1)


def lockstep_check(committed_per_tick: Sequence[torch.Tensor], B: int) -> int:
    """G8 accounting identity: T ticks commit exactly T tokens per request."""
    n = 0
    for c in committed_per_tick:
        if tuple(c.shape) != (B,):
            raise AssertionError("a tick committed %s tokens, not (B=%d,)" % (tuple(c.shape), B))
        n += 1
    return n


# ---------------------------------------------------------------------------
# the writes: V's tail (the S == 1 twin, G6), S's provisional row (G1, G5, G7)
# ---------------------------------------------------------------------------
def v_tail_write(engine, key_states: torch.Tensor, value_states: torch.Tensor, kv_bias: torch.Tensor):
    """CPU twin of the V row's tail path = the S == 1 body's lines :345-359
    and :401-409 (tail_write._write_one_token / _write_back_full_tail, verbatim
    replays of cache_engine.decode_update_has_kv_bias). key/value (B, 1, H, D),
    kv_bias (B, L, H) the total_cis table. Returns (row written, filled).
    On the GPU tick.py does NOT call this: it calls the engine's own body
    (cache.decode_update_kv), which also runs diff and the gathers."""
    row, tail_full = _tw._write_one_token(engine, key_states, value_states, kv_bias)
    if tail_full:
        _tw._write_back_full_tail(engine)
    return int(row), bool(tail_full)


def provisional_row(lay: Layout, tail_len_after: int) -> int:
    """The flat row of S's provisional K/V: narrow = row tail_len_after of
    the tail slot (the next V write's row); union = row 0 of slot topk."""
    if lay.narrow:
        tl = int(tail_len_after)
        if not (0 <= tl < lay.bs):
            raise ValueError("tail_len_after=%d outside [0, bs=%d)" % (tl, lay.bs))
        return lay.tail_slot * lay.bs + tl
    return lay.prov_slot * lay.bs


def s_provisional_write(engine, key_states: torch.Tensor, value_states: torch.Tensor, bias_rows: torch.Tensor,
                        lay: Layout, req_idx: Optional[torch.Tensor], tail_len_after: int) -> int:
    """S's K/V/bias at tau + 1 -> the provisional row (provisional_row) of the
    given requests (all when req_idx is None). key/value (n, H, D) or
    (n, 1, H, D); bias_rows (n, H) or (n, 1, H), already through the engine's
    _bias_rows (KV_BIAS_SCALE). Touches nothing the exact path reads: in the
    narrow layout the row is above the decode's cache_seqlens and is the next
    V write's own row; never the window, the host or the counters (G5).
    Returns the flat row written."""
    r = provisional_row(lay, tail_len_after)
    k = key_states.reshape(key_states.shape[0], -1, engine.head_num, engine.head_dim)
    v = value_states.reshape(value_states.shape[0], -1, engine.head_num, engine.head_dim)
    b = bias_rows.reshape(bias_rows.shape[0], -1, engine.head_num)
    if k.shape[1] != 1 or v.shape[1] != 1 or b.shape[1] != 1:
        raise ValueError("one provisional token per request: key %s value %s bias %s" % (tuple(k.shape), tuple(v.shape), tuple(b.shape)))
    n = k.shape[0]
    if req_idx is None:
        if n != engine._k_gpu.shape[0]:
            raise ValueError("%d rows for %d requests (pass req_idx for a subset)" % (n, engine._k_gpu.shape[0]))
        engine._k_gpu[:, r:r + 1].copy_(k, non_blocking=True)
        engine._v_gpu[:, r:r + 1].copy_(v, non_blocking=True)
        engine._kv_bias_gpu[:, r:r + 1].copy_(b, non_blocking=True)
    else:
        if req_idx.numel() != n:
            raise ValueError("req_idx has %d entries for %d rows" % (req_idx.numel(), n))
        engine._k_gpu[req_idx, r] = k[:, 0]
        engine._v_gpu[req_idx, r] = v[:, 0]
        engine._kv_bias_gpu[req_idx, r] = b[:, 0]
    return r


def poison_provisional(engine, lay: Layout, req_idx: Optional[torch.Tensor], tail_len_after: int) -> int:
    """NaN-poison the provisional row after a tick (G1 / G7 control under
    NOSI_PAIRED_POISON=1): a read of the row by any later attention surfaces
    as a non-finite output. Every S row writes the row before attending; no
    V row's extent reaches it (narrow: V's extent ends one row below it;
    union: at the tail slot), and the next V write overwrites it."""
    r = provisional_row(lay, tail_len_after)
    nan = float("nan")
    if req_idx is None:
        engine._k_gpu[:, r:r + 1].fill_(nan)
        engine._v_gpu[:, r:r + 1].fill_(nan)
        engine._kv_bias_gpu[:, r:r + 1].fill_(nan)
    else:
        engine._k_gpu[req_idx, r] = nan
        engine._v_gpu[req_idx, r] = nan
        engine._kv_bias_gpu[req_idx, r] = nan
    return r


def exact_store_digest(engine, lay: Layout) -> dict:
    """The persistent, exact state of one engine (G5): the window + tail rows
    of the three GPU tensors, the maps, the counters, and the host window
    around seq_length. Compared with torch.equal by the tests."""
    hi = (lay.tail_slot + 1) * lay.bs
    seq = int(engine.seq_length)
    lo_h, hi_h = max(0, seq - 2 * lay.bs), min(engine._k_cpu.shape[1], seq + 2 * lay.bs)
    return dict(k=engine._k_gpu[:, :hi].clone(), v=engine._v_gpu[:, :hi].clone(), bias=engine._kv_bias_gpu[:, :hi].clone(),
                block_map=engine._block_map.clone(), cache_lens=engine._cache_lens.clone(),
                seq_length=seq, tail_len=int(engine._tail_block_len_on_gpu),
                k_cpu=engine._k_cpu[:, lo_h:hi_h].clone(), v_cpu=engine._v_cpu[:, lo_h:hi_h].clone())


def digests_equal(a: dict, b: dict) -> List[str]:
    """Names of the entries that differ (empty = torch.equal everywhere)."""
    bad = []
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            if not torch.equal(x, y):
                bad.append(k)
        elif x != y:
            bad.append(k)
    return bad


# ---------------------------------------------------------------------------
# the tail content rule (what physically sits in slot topk-1)
# ---------------------------------------------------------------------------
class TailView(NamedTuple):
    tail_len_after: int       # _tail_block_len_on_gpu after the V update
    filled: bool              # the V write filled the block this tick (S == 1: write-back + rename)
    slot_complete: bool       # slot topk-1 physically holds a complete exact block
    tail_rows_for_s: int      # rows of slot topk-1 an S row may read: bs if slot_complete else tail_len_after
    content_offset: int       # id of the block in slot topk-1 = _block_map[..., topk-1] - content_offset (1 after a fill)


def tail_view(tail_len_before: int, wrote_v: bool, slot_complete_before: bool, lay: Layout) -> TailView:
    """After a V write (wrote_v) from tail_len_before: tail_len_after =
    tail_len_before + 1, or 0 on a fill (tail_len_before == bs - 1); the slot
    then holds block T complete while _block_map says T+1 (the S == 1 body's
    rename, cache_engine.py :407-409). Without a V write (a restart / seed
    forward) the previous view carries over: the slot is complete iff the
    last V write filled it and nothing was written since (tail_len == 0)."""
    tl0 = int(tail_len_before)
    if not (0 <= tl0 < lay.bs):
        raise ValueError("tail_len_before=%d outside [0, bs=%d)" % (tl0, lay.bs))
    if wrote_v:
        filled = (tl0 + 1 == lay.bs)
        after = 0 if filled else tl0 + 1
        complete = filled
    else:
        filled = False
        after = tl0
        complete = bool(slot_complete_before) and after == 0
    return TailView(tail_len_after=after, filled=filled, slot_complete=complete,
                    tail_rows_for_s=(lay.bs if complete else after), content_offset=(1 if complete else 0))
