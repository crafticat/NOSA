"""Union-store construction for one verify round, integer torch ops only.

Spec (retroinfer-eval ``docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md``):
section 1 (the ``topk_idx`` rows, C5: ``(b, u)`` order; slot 63 never named;
tail via the mirror), section 2(b) (layout), section 2(c) (union formation:
``new_j = sel_j \\ union_map``, bump allocation, ``tail id T -> W-2, T+1 -> W-1``,
overflow detected and never truncated, pool members need no fetch), section
5(b) (what a test must show).

LAYOUT per (KV head h, request b), in SLOTS of ``block_size`` rows (see
tail_write.py for the row view):

    0 .. topk-2              attended window     ids: window_map[h, b, s]
    topk-1                   live tail (id T)    NEVER named by a verify mask
    topk .. topk+P-1         victim pool         ids: pool_map[h, b, q]; no fetch
    topk+P .. topk+P+R-1     ROUND REGION        bump-allocated here; every occupant is fetched
    W-2                      mirror of T         a selection naming T maps here
    W-1                      mirror of T+1       only for positions u >= rollover_at

Spec 2(c) places the round region at ``64*64:(64+R)*64`` for v1 (no pool); with
a pool the region follows it, because ``flash_pool_swap`` addresses pool slots
as ``(topk+q)*block_size`` (cache_engine.py:202 and the pool section). That is
the one layout choice this module makes; ``union_layout`` is its single source.

ORDER. The round region is assigned position-major, then in the order the ids
appear in that position's selection, first occurrence only. It is a function of
the request's own selections and window (float-order law is moot: integers;
isolation holds: every op below is elementwise over (h, b) or reduces over the
last dim only, so stream (h, b) never reads stream (h', b')).

OVERFLOW (> R new blocks in a round for some (h, b)) sets ``overflow[h, b]``;
that stream's mask rows are all -1 (poisoned, never a plausible partial mask)
and the caller falls back to sequential verification (spec 2c). The same holds
for ``invalid[h, b]``: a selection named a block id above the live tail at that
position (T before the rollover, T+1 after) or an id below -1.

No CUDA, no extension import. The overflow/invalid flags are tensors (no host
sync to read them); the one host sync is the ``tail_id`` consistency guard.

``rollover_at`` comes from ``tail_write.write_tail``: positions ``>= rollover_at``
see T+1 as the live tail; ``== U`` when no position does (no rollover, or a
rollover on the round's last token, whose T+1 is then empty).
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class Layout(NamedTuple):
    topk: int
    pool_blocks: int
    round_blocks: int
    tail_slot: int   # topk - 1
    pool_base: int   # topk
    round_base: int  # topk + P
    mirror_lo: int   # W - 2
    mirror_hi: int   # W - 1
    W: int           # topk + P + R + 2


def union_layout(topk: int, pool_blocks: int, round_blocks: int) -> Layout:
    if topk < 1 or pool_blocks < 0 or round_blocks < 0:
        raise ValueError("union_layout: topk=%d pool_blocks=%d round_blocks=%d" % (topk, pool_blocks, round_blocks))
    tail_slot = topk - 1
    pool_base = topk
    round_base = topk + pool_blocks
    mirror_lo = round_base + round_blocks
    mirror_hi = mirror_lo + 1
    return Layout(topk, pool_blocks, round_blocks, tail_slot, pool_base, round_base, mirror_lo, mirror_hi, mirror_hi + 1)


class UnionRound(NamedTuple):
    layout: Layout
    union_map: torch.Tensor   # (H, B, W) int64: slot -> block id, -1 free
    mask: torch.Tensor        # (H, B*U, K) int32: union slot ids, -1 padded, row b*U+u
    round_map: torch.Tensor   # (H, B, R) int64: the fetch list (a prefix), -1 free
    round_load: torch.Tensor  # (U, H, B, R) int64: per position, the ids assigned (fetched) at that position
    n_new: torch.Tensor       # (H, B) int64: new blocks in the round (may exceed R)
    overflow: torch.Tensor    # (H, B) bool
    invalid: torch.Tensor     # (H, B) bool
    pinned: torch.Tensor      # (H, B, W) bool: slots that must not change occupant during the round


def build_union(window_map, tail_id, sel, pool_map, round_blocks: int, rollover_at: int) -> UnionRound:
    """Build the union store, the per-query masks, the fetch list and the pin set.

    window_map: ``(H, B, topk)`` int64, ``_block_map`` at round start.
    tail_id: ``(H, B)`` int64, ``_block_map[..., topk-1]`` captured at round
        start, BEFORE any tail write (a rollover renames that slot, :407).
    sel: ``(U, H, B, K)`` int64, the per-position selections (block ids, -1 padded).
    pool_map: ``(H, B, P)`` int64, ``_pool_map`` (-1 empty); P may be 0.
    round_blocks: R, the round region's capacity.
    rollover_at: from ``tail_write.write_tail``: positions ``>= rollover_at`` see
        T+1 as the live tail; ``== U`` when the round did not roll over.
    """
    if window_map.dim() != 3 or sel.dim() != 4 or pool_map.dim() != 3 or tail_id.dim() != 2:
        raise ValueError("shapes: window_map %s tail_id %s sel %s pool_map %s" % (
            tuple(window_map.shape), tuple(tail_id.shape), tuple(sel.shape), tuple(pool_map.shape)))
    H, B, topk = window_map.shape
    U, Hs, Bs, K = sel.shape
    if (Hs, Bs) != (H, B) or tuple(pool_map.shape[:2]) != (H, B) or tuple(tail_id.shape) != (H, B):
        raise ValueError("(H, B) mismatch: window %s sel %s pool %s tail_id %s" % (
            (H, B), (Hs, Bs), tuple(pool_map.shape[:2]), tuple(tail_id.shape)))
    if U < 1 or K < 1:
        raise ValueError("sel must have U >= 1 positions and K >= 1 entries, got (U=%d, K=%d)" % (U, K))
    if not (1 <= rollover_at <= U):
        raise ValueError("rollover_at=%d must be in [1, U=%d] (U means no rollover)" % (rollover_at, U))
    for name, t in (("window_map", window_map), ("tail_id", tail_id), ("sel", sel), ("pool_map", pool_map)):
        if t.dtype != torch.int64:
            raise TypeError("%s must be int64 (the engine's _block_map dtype, cache_engine.py:227), got %s" % (name, t.dtype))
    P = pool_map.shape[-1]
    R = int(round_blocks)
    lay = union_layout(topk, P, R)
    dev = window_map.device

    live_tail_slot = window_map[..., lay.tail_slot]
    # the map given is the round-start map (slot 63 = T) or the post-rename map
    # (slot 63 = T+1); slots 0..62 are identical in both (:377 does not run in a round)
    ok_tail = (live_tail_slot == tail_id) | (live_tail_slot == tail_id + 1)
    if not bool(ok_tail.all()):
        raise ValueError("window_map[..., tail_slot] is neither tail_id nor tail_id+1 for some stream")

    neg2 = torch.full((), -2, dtype=torch.int64, device=dev)
    # static members: the window without its tail slot, then the pool; empties -> -2
    static_ids = torch.cat([window_map[..., :lay.tail_slot], pool_map], dim=-1)
    static_ids = torch.where(static_ids >= 0, static_ids, neg2)

    round_map = torch.full((H, B, R), -1, dtype=torch.int64, device=dev)
    round_load = torch.full((U, H, B, R), -1, dtype=torch.int64, device=dev)
    ptr = torch.zeros((H, B), dtype=torch.int64, device=dev)
    overflow = torch.zeros((H, B), dtype=torch.bool, device=dev)
    invalid = torch.zeros((H, B), dtype=torch.bool, device=dev)
    below_diag = torch.ones((K, K), dtype=torch.int64, device=dev).tril(-1).to(torch.bool)
    dump = torch.full((H, B, 1), -1, dtype=torch.int64, device=dev)

    for u in range(U):
        ids = sel[u]                                   # (H, B, K)
        valid = ids >= 0
        live_tail = tail_id + (1 if u >= rollover_at else 0)
        invalid |= (valid & (ids > live_tail.unsqueeze(-1))).any(-1) | (ids < -1).any(-1)
        is_tail = ids == tail_id.unsqueeze(-1)
        is_tail_next = (ids == (tail_id + 1).unsqueeze(-1)) if u >= rollover_at else torch.zeros_like(valid)
        in_static = (ids.unsqueeze(-1) == static_ids.unsqueeze(-2)).any(-1)
        round_ids = torch.where(round_map >= 0, round_map, neg2)
        in_round = (ids.unsqueeze(-1) == round_ids.unsqueeze(-2)).any(-1)
        # first occurrence inside this selection row
        repeated = ((ids.unsqueeze(-1) == ids.unsqueeze(-2)) & below_diag).any(-1)
        new = valid & ~is_tail & ~is_tail_next & ~in_static & ~in_round & ~repeated
        rank = torch.cumsum(new.to(torch.int64), dim=-1) - 1
        dst = ptr.unsqueeze(-1) + rank                 # (H, B, K)
        n_new_u = new.sum(-1)                          # (H, B)
        overflow |= (ptr + n_new_u) > R
        fits = new & (dst < R)
        dst_c = torch.where(fits, dst, torch.full_like(dst, R))   # non-fitting entries go to the dump column
        src = torch.where(fits, ids, torch.full_like(ids, -1))
        ext = torch.cat([round_map, dump], dim=-1)
        ext.scatter_(-1, dst_c, src)
        round_map = ext[..., :R]
        ext_u = torch.cat([torch.full((H, B, R), -1, dtype=torch.int64, device=dev), dump], dim=-1)
        ext_u.scatter_(-1, dst_c, src)
        round_load[u] = ext_u[..., :R]
        ptr = ptr + n_new_u

    n_new = ptr

    # slot -> id lookup for the masks: slot 63 and every free slot are -2 (never match)
    lookup = torch.full((H, B, lay.W), -2, dtype=torch.int64, device=dev)
    lookup[..., :lay.tail_slot] = static_ids[..., :lay.tail_slot]
    lookup[..., lay.pool_base:lay.round_base] = static_ids[..., lay.tail_slot:]
    lookup[..., lay.round_base:lay.mirror_lo] = torch.where(round_map >= 0, round_map, neg2)
    lookup[..., lay.mirror_lo] = tail_id

    union_map = torch.full((H, B, lay.W), -1, dtype=torch.int64, device=dev)
    union_map[..., :topk] = window_map
    union_map[..., lay.pool_base:lay.round_base] = pool_map
    union_map[..., lay.round_base:lay.mirror_lo] = round_map
    union_map[..., lay.mirror_lo] = tail_id
    rolled = rollover_at < U
    if rolled:
        union_map[..., lay.mirror_hi] = tail_id + 1

    bad = overflow | invalid
    masks = []
    pinned_ext = torch.zeros((H, B, lay.W + 1), dtype=torch.bool, device=dev)
    for u in range(U):
        lk = lookup
        if u >= rollover_at:
            lk = lookup.clone()
            lk[..., lay.mirror_hi] = tail_id + 1
        eq = sel[u].unsqueeze(-1) == lk.unsqueeze(-2)   # (H, B, K, W)
        found = eq.any(-1)
        slot = eq.to(torch.int64).argmax(-1)            # first match; ids are distinct across slots
        mask_u = torch.where(found & ~bad.unsqueeze(-1), slot, torch.full_like(slot, -1))
        masks.append(mask_u)
        idx = torch.where(mask_u >= 0, mask_u, torch.full_like(mask_u, lay.W))
        pinned_ext.scatter_(-1, idx, torch.ones_like(idx, dtype=torch.bool))
    mask = torch.stack(masks, dim=2).reshape(H, B * U, K).to(torch.int32)   # row = b*U + u (spec C5)

    pinned = pinned_ext[..., :lay.W]
    pinned[..., lay.tail_slot] = True
    pinned[..., lay.mirror_lo] = True
    if rolled:
        pinned[..., lay.mirror_hi] = True

    return UnionRound(lay, union_map, mask, round_map, round_load, n_new, overflow, invalid, pinned)
