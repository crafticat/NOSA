"""END-TO-END speculative decode loop on NOSI: the CPU-provable core (retroinfer-eval
fork, 2026-09-20; driver benchmarks/Efficiency/spec_loop_pilot.py; tests
retroinfer-eval tests/test_nosi_spec_loop.py).

Pure torch. This module imports nothing but torch and typing so that every
function here runs on a CPU tensor exactly as it runs on the device; the GPU
engine (cache_engine.py, spec_* methods) and the model (nosa_llama.py,
draft_forward / spec_verify_forward) call these functions and add only the
kernels: the shipped decode kernel with a bias (the draft), the two shipped
Triton gathers (prefetch and late fetch), the Path-1 varlen call (the verify).

THE ROUND (greedy, exact by construction; K drafted positions, U = K + 1)

  DRAFT    K decode steps that attend ONLY to blocks resident in the store
           (window + round region), through the shipped decode kernel over
           the WHOLE allocation with a per-row bias of MASK_BIAS on every row
           the step may not see (avail_policy.py's mechanism, same value).
           The blocks a step's selection misses are REQUESTED: their gathers
           are launched on a side stream into free (or LRU-evicted) slots,
           bounded by the store; an event per issue records arrival.
  VERIFY   one Path-1 round over [x_n, d_1 .. d_K] (U positions), each
           attending its EXACT selection over the store; blocks not yet
           arrived are waited for (late_wait) or fetched now (late_fetch).
  ACCEPT   n_acc(b) = longest prefix with argmax(verify[b, i]) == d_{i+1};
           the batch commits c = 1 + min_b n_acc(b) tokens =
           argmax(verify[b, 0:c]) (accept_commit: why this is exact).
  ROLLBACK truncate_tail (the tail rows for positions >= c undone, a
           rollover past c undone from the mirror) + LayerJournal.restore +
           replay(c) (the compressed-key / cis tables of a decode of the c
           committed tokens).

WHY THE COMMIT IS LOCKSTEP. NOSI's engine has ONE seq_length and ONE
_tail_block_len_on_gpu per engine (cache_engine.py), one no_compress_k_len /
tail_cis_len / cis_len per layer, and grows compress_k_cache_varlen by a
batch-wide torch.cat every 16 tokens; the pooling graph is captured with a
fixed max_seqlen_comp. A per-request committed length (ragged tails) is an
engine change, not a pilot: this loop commits the batch minimum and MEASURES
the per-request n_acc so the ragged yield (1 + mean n_acc) is computable from
the same run. At batch B with per-token acceptance a the lockstep yield is
1 + a^B (K = 1): meaningful at small B or few distinct documents, ~1 at 128
distinct documents. The pilot reports both.

PER-REQUEST ROUNDS (R2 milestone 2): the section at the end of this file
keeps one counter per request (pos (B,), the positional twin's counter) and
commits keep(b) = 1 + n_acc(b) per request: a counter reset plus one masked
copy of the tail slot. The lockstep functions above are unchanged; the
per-request ones are tested against them. The one intended difference: at a
findings-F4 step (p % 64 == 63) the per-request draft and verify skip the tail
block as the shipped decode does, unless PerReqSpecContext.f4 = "fixed"
(section F4 PARITY).

STORE (per layer, KV head, request; slots of block_size rows; W = topk + R + 2):
    0 .. topk-2        window, evictable      ids[..., s] >= 0 when occupied
    topk-1             LIVE TAIL, FIXED       id T = tail_id (an int, all layers
                                              and requests advance together)
    topk .. topk+R-1   round region, evictable
    W-2, W-1           TAIL MIRROR, FIXED     T at W-2, T+1 at W-1 (tail_write.py)
State per slot: FREE / INFLIGHT (gather issued, event pending) / RESIDENT /
FIXED. A slot is RESIDENT only after the host observed its issue event
complete (mark_arrived: an integer op over the stacked store), so a kernel
enqueued afterwards on the main stream reads a finished write. The side
stream waits on a main-stream event recorded after the previous readers of
the slot were enqueued, so an eviction never races an attention kernel
(the engine records those events; the queue here is the bookkeeping).

ISOLATION / FLOAT-ORDER. Every op below is elementwise over (h, b) or reduces
over the last dimension only; all are integer or boolean. Stream (h, b) never
reads stream (h', b'). The one float constant is MASK_BIAS.

HOST SYNCS, counted by the driver: plan_draft has none (the issue map is
consumed on the device); plan_round's overflow/invalid read is one per layer
per round (pilot quality); accept_commit's c is one per round (unavoidable:
it drives the host loop).
"""
from __future__ import annotations

from collections import deque
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import torch

MASK_BIAS = -3.0e4              # avail_policy.MASK_BIAS: finite (split-KV recombination), fully masking
FREE, INFLIGHT, RESIDENT, FIXED = 0, 1, 2, 3
BIG = 1 << 40                   # sort key of a slot that may not be assigned

# wire bytes of one (layer, KV head, request, block): K + V rows over PCIe;
# the 64 bias values come from the GPU-resident total_cis (transfer_trace.py)
BLOCK_TOKENS = 64
HEAD_DIM = 128
BYTES_WIRE_PER_BLOCK = 2 * BLOCK_TOKENS * HEAD_DIM * 2


class StoreLayout(NamedTuple):
    topk: int
    R: int
    tail_slot: int    # topk - 1
    round_base: int   # topk
    mirror_lo: int    # W - 2
    mirror_hi: int    # W - 1
    W: int            # topk + R + 2


def store_layout(topk: int, R: int) -> StoreLayout:
    """union_store.union_layout with no victim pool (v1 of the verifier, spec 2c)."""
    if topk < 2 or R < 1:
        raise ValueError("store_layout: topk=%d R=%d (need topk >= 2, R >= 1)" % (topk, R))
    W = topk + R + 2
    return StoreLayout(topk, R, topk - 1, topk, W - 2, W - 1, W)


class Store:
    """The bookkeeping of every layer's store, stacked: (Lyr, H, B, W).

    ids     int64  block id in the slot, -1 when FREE; FIXED slots carry -1
                   (the tail and the mirror are matched by id explicitly)
    state   int8   FREE / INFLIGHT / RESIDENT / FIXED
    gen     int64  the issue id of the gather that filled the slot
    stamp   int64  the tick of the last selection that named the slot (LRU)
    used    bool   named by a draft attention or a verify mask since filled
    Counters (Lyr, H, B) int64: prefetch_blocks (issued on the side stream),
    late_fetch_blocks (fetched at the round), late_wait_blocks (in flight at
    verify start, waited for), timely_blocks, hit_old_blocks, wasted_blocks
    (evicted or dropped before any use), unrequested_blocks (misses that found
    no slot), forbidden_blocks (misses of block ids >= forbid_ge, see plan_draft).
    """

    COUNTERS = ("prefetch_blocks", "late_fetch_blocks", "late_wait_blocks", "timely_blocks",
                "hit_old_blocks", "wasted_blocks", "unrequested_blocks", "forbidden_blocks",
                "draft_sel_blocks", "draft_hit_blocks", "draft_inflight_blocks")

    def __init__(self, num_layers: int, H: int, B: int, layout: StoreLayout, device=None):
        self.lay = layout
        self.num_layers, self.H, self.B = int(num_layers), int(H), int(B)
        shape = (self.num_layers, self.H, self.B, layout.W)
        self.ids = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.state = torch.full(shape, FREE, dtype=torch.int8, device=device)
        self.gen = torch.zeros(shape, dtype=torch.int64, device=device)
        self.stamp = torch.zeros(shape, dtype=torch.int64, device=device)
        self.used = torch.zeros(shape, dtype=torch.bool, device=device)
        for s in (layout.tail_slot, layout.mirror_lo, layout.mirror_hi):
            self.state[..., s] = FIXED
        self.evictable = torch.ones((layout.W,), dtype=torch.bool, device=device)
        for s in (layout.tail_slot, layout.mirror_lo, layout.mirror_hi):
            self.evictable[s] = False
        for n in self.COUNTERS:
            setattr(self, n, torch.zeros((self.num_layers, self.H, self.B), dtype=torch.int64, device=device))

    # -- state transitions ------------------------------------------------------
    def sync_window(self, l: int, block_map: torch.Tensor, stamp_now: int):
        """After a SHIPPED decode step (diff + synchronous gathers into the window,
        cache_engine.decode_update_has_kv_bias): slots 0..topk-2 hold exactly
        block_map's ids and are RESIDENT (FREE where -1); the tail slot stays
        FIXED. A round-region copy of a block the window now also holds is
        freed (dedupe: the draft must not attend a block twice)."""
        lay = self.lay
        if tuple(block_map.shape) != (self.H, self.B, lay.topk):
            raise ValueError("block_map %s != (H=%d, B=%d, topk=%d)" % (tuple(block_map.shape), self.H, self.B, lay.topk))
        win = block_map[..., :lay.tail_slot]
        self.ids[l, ..., :lay.tail_slot] = win
        self.state[l, ..., :lay.tail_slot] = torch.where(win >= 0, torch.full_like(win, RESIDENT), torch.full_like(win, FREE)).to(torch.int8)
        self.stamp[l, ..., :lay.tail_slot] = torch.where(win >= 0, torch.full_like(win, stamp_now), self.stamp[l, ..., :lay.tail_slot])
        self.used[l, ..., :lay.tail_slot] = win >= 0
        self.ids[l, ..., lay.tail_slot] = -1
        rr = slice(lay.round_base, lay.mirror_lo)
        rids = self.ids[l, ..., rr]
        dup = (rids.unsqueeze(-1) == win.unsqueeze(-2)).any(-1) & (rids >= 0)
        self.wasted_blocks[l] += (dup & ~self.used[l, ..., rr]).sum(-1)
        self.ids[l, ..., rr] = torch.where(dup, torch.full_like(rids, -1), rids)
        self.state[l, ..., rr] = torch.where(dup, torch.full_like(self.state[l, ..., rr], FREE), self.state[l, ..., rr])

    def mark_arrived(self, done_gen: int):
        """Every INFLIGHT slot whose issue id <= done_gen becomes RESIDENT. The
        side stream is one in-order stream, so 'the event of issue g is
        complete' implies every issue <= g is complete."""
        arrived = (self.state == INFLIGHT) & (self.gen <= int(done_gen))
        self.state = torch.where(arrived, torch.full_like(self.state, RESIDENT), self.state)

    def reset(self):
        lay = self.lay
        self.ids.fill_(-1)
        self.state.fill_(FREE)
        for s in (lay.tail_slot, lay.mirror_lo, lay.mirror_hi):
            self.state[..., s] = FIXED
        self.gen.zero_()
        self.stamp.zero_()
        self.used.zero_()
        for n in self.COUNTERS:
            getattr(self, n).zero_()

    def snapshot(self) -> dict:
        d = {n: getattr(self, n).clone() for n in ("ids", "state", "gen", "stamp", "used")}
        d.update({n: getattr(self, n).clone() for n in self.COUNTERS})
        return d

    def restore(self, d: dict):
        for n in ("ids", "state", "gen", "stamp", "used"):
            getattr(self, n).copy_(d[n])
        for n in self.COUNTERS:
            getattr(self, n).copy_(d[n])

    def resident_count(self) -> torch.Tensor:
        return (self.state == RESIDENT).sum(-1)

    def counters(self) -> dict:
        return {n: getattr(self, n).clone() for n in self.COUNTERS}


class PrefetchQueue:
    """The in-flight issues, oldest first: (gen, event). ``done_gen`` scans
    from the oldest and pops every event whose ``query()`` is True; it stops
    at the first pending one (in-order stream). No host sync: ``query`` is a
    non-blocking poll. ``event`` is duck-typed (torch.cuda.Event on the
    device; any object with ``query()`` in tests)."""

    def __init__(self):
        self.pending: deque = deque()
        self.last_done = 0
        self.last_issued = 0
        self.last_event = None

    def push(self, gen: int, event):
        if gen <= self.last_issued:
            raise ValueError("issue ids must increase: got %d after %d" % (gen, self.last_issued))
        self.last_issued = int(gen)
        self.last_event = event
        self.pending.append((int(gen), event))

    def done_gen(self) -> int:
        while self.pending and self.pending[0][1].query():
            self.last_done = self.pending.popleft()[0]
        return self.last_done

    def drain(self) -> int:
        """After the caller synchronised the side stream: everything issued is done."""
        self.pending.clear()
        self.last_done = self.last_issued
        return self.last_done


# ---------------------------------------------------------------------------
# slot assignment (shared by the draft's prefetch and the round's late fetch)
# ---------------------------------------------------------------------------
def _assign(want: torch.Tensor, ids: torch.Tensor, cand: torch.Tensor, key: torch.Tensor):
    """Assign the wanted entries (H, B, N) to candidate slots (H, B, W) in
    ascending ``key`` order (stable). Returns (issue (H, B, W) int64: id per
    assigned slot else -1; fits (H, B, N) bool; n_cand (H, B) int64)."""
    H, B, W = cand.shape
    order = torch.argsort(key, dim=-1, stable=True)
    n_cand = cand.sum(-1)
    rank = torch.cumsum(want.to(torch.int64), dim=-1) - 1
    fits = want & (rank < n_cand.unsqueeze(-1))
    dst = torch.gather(order, -1, torch.where(fits, rank, torch.zeros_like(rank)))
    dump = torch.full_like(dst, W)
    dst_c = torch.where(fits, dst, dump)
    ext = torch.full((H, B, W + 1), -1, dtype=torch.int64, device=ids.device)
    ext.scatter_(-1, dst_c, torch.where(fits, ids, torch.full_like(ids, -1)))
    # CONTIGUOUS (2026-10-20 fix): the map goes to the two Triton gathers, which read entry (h, b, m) at
    # h * B * M + b * M + m from the data pointer (flash_h2d_mask.py:34, flash_h2d_mask_bias.py:34). The narrowed
    # view ext[..., :W] has rows of W + 1: the kernels read shifted entries and put blocks into other slots, other
    # requests' slots and the tail slot (runs r2m1_g2 / r2m1_p1: rollback probe 'layer 0 ... k_tail differs').
    return ext[..., :W].contiguous(), fits, n_cand


def _first_occurrence(ids: torch.Tensor) -> torch.Tensor:
    """(..., N) -> bool: True where the id is the first of its value in the row."""
    N = ids.shape[-1]
    below = torch.ones((N, N), dtype=torch.bool, device=ids.device).tril(-1)
    return ~((ids.unsqueeze(-1) == ids.unsqueeze(-2)) & below).any(-1)


def _slot_key(state: torch.Tensor, stamp: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """FREE candidates first (by slot), then RESIDENT candidates oldest stamp
    first (then by slot); non-candidates last."""
    W = state.shape[-1]
    w = torch.arange(W, dtype=torch.int64, device=state.device)
    free_key = w.expand_as(stamp)
    res_key = W + (stamp + 1) * W + w
    key = torch.where(state == FREE, free_key, res_key)
    return torch.where(cand, key, torch.full_like(key, BIG))


# ---------------------------------------------------------------------------
# the draft step
# ---------------------------------------------------------------------------
class DraftPlan(NamedTuple):
    deny: torch.Tensor      # (H, B, W) bool: slot-level denial (tail and mirror_hi rows are the caller's, draft_row_limits)
    issue: torch.Tensor     # (H, B, W) int64: block id to gather NOW into the slot on the side stream, -1 elsewhere
    n_sel: torch.Tensor     # (H, B): valid non-tail selection entries
    n_hit: torch.Tensor     # (H, B): entries served by a RESIDENT slot (attended)
    n_inflight: torch.Tensor  # (H, B): entries whose block is INFLIGHT (not attended, not re-requested)
    n_issue: torch.Tensor   # (H, B): entries requested now
    n_unrequested: torch.Tensor  # (H, B): fetchable misses that found no slot
    n_forbidden: torch.Tensor    # (H, B): misses of ids >= forbid_ge (not on the host exactly yet)
    n_evicted_unused: torch.Tensor  # (H, B): evicted slots that were never used


def draft_row_limits(tail_len: int, rolled: bool, block_size: int) -> Tuple[int, int]:
    """Rows the draft may see (exclusive bounds) in the tail slot and in the
    mirror_hi slot: before the draft's own rollover the tail slot holds T's
    ``tail_len`` rows and mirror_hi nothing; after it the tail slot holds T
    complete and mirror_hi holds T+1's ``tail_len`` rows (draft_tail_write)."""
    if not (0 <= tail_len <= block_size):
        raise ValueError("tail_len=%d outside [0, %d]" % (tail_len, block_size))
    if rolled:
        return block_size, tail_len
    return tail_len, 0


def plan_draft(store: Store, l: int, sel: torch.Tensor, tail_id: int, rolled: bool,
               gen_now: int, stamp_now: int, forbid_ge: int) -> DraftPlan:
    """One draft step's availability and prefetch plan for layer ``l``.

    sel: (H, B, K) int64 block ids (-1 padded), the step's own selection
        (topk_idx_buf: the decode's, never restricted).
    tail_id: T, the live tail's block id at ROUND start (an int: every layer
        and request advances together). After the draft's own rollover the
        tail slot is labelled T+1 in _block_map; both T and T+1 are served by
        the tail slot / mirror_hi rows (draft_row_limits), never fetched.
    forbid_ge: block ids >= this are NOT fetchable from the host during the
        draft: the host copy of block T (and later) is written by the draft's
        own write-back with APPROXIMATE rows until the verify rewrites it.
        Always T; a selection can name at most T+1.
    Mutates the store: issued slots become INFLIGHT with this gen, stamps
    of named slots move to stamp_now, attended slots become used.
    """
    lay = store.lay
    ids, state, gen, stamp, used = store.ids[l], store.state[l], store.gen[l], store.stamp[l], store.used[l]
    if sel.dim() != 3 or tuple(sel.shape[:2]) != (store.H, store.B) or sel.dtype != torch.int64:
        raise ValueError("sel must be int64 (H=%d, B=%d, K), got %s %s" % (store.H, store.B, sel.dtype, tuple(sel.shape)))
    H, B, K = sel.shape
    W = lay.W
    valid = sel >= 0
    t = torch.full_like(sel, int(tail_id))
    is_tail = valid & ((sel == t) | (sel == t + 1)) if rolled else valid & (sel == t)
    present = (state == INFLIGHT) | (state == RESIDENT)
    first = _first_occurrence(sel)                 # every count below is over DISTINCT block ids of the row
    eq = (sel.unsqueeze(-1) == ids.unsqueeze(-2)) & present.unsqueeze(-2)     # (H, B, K, W)
    found = eq.any(-1) & valid & ~is_tail & first
    slot = eq.to(torch.int64).argmax(-1)
    slot_state = torch.gather(state.to(torch.int64), -1, slot)
    hit = found & (slot_state == RESIDENT)
    inflight = found & (slot_state == INFLIGHT)
    named = torch.zeros((H, B, W + 1), dtype=torch.bool, device=sel.device)
    named.scatter_(-1, torch.where(found, slot, torch.full_like(slot, W)), torch.ones_like(slot, dtype=torch.bool))
    named = named[..., :W]
    attended = named & (state == RESIDENT)
    deny = ~attended
    deny[..., lay.tail_slot] = False
    deny[..., lay.mirror_hi] = False
    deny[..., lay.mirror_lo] = True

    miss = valid & ~is_tail & ~found & first
    fetchable = miss & (sel < int(forbid_ge))
    forbidden = miss & ~fetchable
    cand = store.evictable.view(1, 1, W) & ~named & (state != INFLIGHT) & ((state == FREE) | (state == RESIDENT))
    key = _slot_key(state, stamp, cand)
    issue, fits, _ = _assign(fetchable, sel, cand, key)
    newly = issue >= 0
    evicted_unused = newly & (state == RESIDENT) & ~used

    store.ids[l] = torch.where(newly, issue, ids)
    store.state[l] = torch.where(newly, torch.full_like(state, INFLIGHT), state)
    store.gen[l] = torch.where(newly, torch.full_like(gen, int(gen_now)), gen)
    store.stamp[l] = torch.where(named | newly, torch.full_like(stamp, int(stamp_now)), stamp)
    store.used[l] = torch.where(newly, torch.zeros_like(used), used | attended)

    n_issue = fits.sum(-1)
    n_unreq = (fetchable & ~fits).sum(-1)
    store.prefetch_blocks[l] += n_issue
    store.unrequested_blocks[l] += n_unreq
    store.forbidden_blocks[l] += forbidden.sum(-1)
    store.wasted_blocks[l] += evicted_unused.sum(-1)
    n_sel = (valid & ~is_tail & first).sum(-1)
    store.draft_sel_blocks[l] += n_sel
    store.draft_hit_blocks[l] += hit.sum(-1)
    store.draft_inflight_blocks[l] += inflight.sum(-1)
    return DraftPlan(deny=deny, issue=issue, n_sel=n_sel, n_hit=hit.sum(-1),
                     n_inflight=inflight.sum(-1), n_issue=n_issue, n_unrequested=n_unreq,
                     n_forbidden=forbidden.sum(-1), n_evicted_unused=evicted_unused.sum(-1))


def apply_draft_bias(scratch: torch.Tensor, kv_bias_gpu: torch.Tensor, deny: torch.Tensor,
                     deny_buf: torch.Tensor, layout: StoreLayout, block_size: int,
                     tail_len: int, rolled: bool, mask_bias: float = MASK_BIAS) -> torch.Tensor:
    """The per-row bias the draft's attention gets: a copy of the allocation's
    bias with MASK_BIAS on every row of a denied slot, on the tail rows above
    what the draft may see and on the mirror_hi rows above its tail_len
    (draft_row_limits). ``scratch`` has the allocation's shape (B, W*bs, H)
    (allocated once per layer, never the allocation itself: avail_policy.py,
    NEVER MASK _kv_bias_gpu IN PLACE); ``deny_buf`` is (B, W, 1, H) bool."""
    B, S, H = kv_bias_gpu.shape
    W = layout.W
    if S != W * block_size or tuple(scratch.shape) != (B, S, H) or tuple(deny.shape) != (H, B, W):
        raise ValueError("apply_draft_bias: bias %s scratch %s deny %s W=%d bs=%d" % (
            tuple(kv_bias_gpu.shape), tuple(scratch.shape), tuple(deny.shape), W, block_size))
    if tuple(deny_buf.shape) != (B, W, 1, H):
        raise ValueError("deny_buf must be (B, W, 1, H)=%s, got %s" % ((B, W, 1, H), tuple(deny_buf.shape)))
    scratch.copy_(kv_bias_gpu)
    deny_buf.copy_(deny.permute(1, 2, 0).unsqueeze(2))
    scratch.view(B, W, block_size, H).masked_fill_(deny_buf, mask_bias)
    tail_ok, hi_ok = draft_row_limits(tail_len, rolled, block_size)
    ts, hs = layout.tail_slot * block_size, layout.mirror_hi * block_size
    if tail_ok < block_size:
        scratch[:, ts + tail_ok:ts + block_size] = mask_bias
    if hi_ok < block_size:
        scratch[:, hs + hi_ok:hs + block_size] = mask_bias
    return scratch


def draft_tail_write(engine, key_states, value_states, kv_bias, mirror_hi: int, rolled: bool) -> bool:
    """The draft's ONE-token tail write: the S == 1 semantics of
    cache_engine.decode_update_has_kv_bias (:345-357 and :401-409, the twin
    tail_write._write_one_token / _write_back_full_tail replays) with one
    difference after the draft's own rollover: the tokens of block T+1 go to
    MIRROR W-1, not to slot topk-1, so the exact rows of block T (rows below
    the round-start tail_len are committed KV) are never overwritten by an
    approximate draft row. The verify's write_tail later starts from the
    round-start counters, rewrites rows >= tail_len_0 of slot topk-1 and the
    mirror, and re-does the write-back with exact rows.

    key_states / value_states: (B, 1, H, D); kv_bias: the layer's total_cis
    (B, L, H), row seq_length-1 after the increment (:354).
    Returns the new ``rolled`` flag. Refuses a second rollover in one draft.
    """
    bs = int(engine.block_size)
    if key_states.dim() != 4 or key_states.shape[1] != 1:
        raise ValueError("draft_tail_write takes one token (B, 1, H, D), got %s" % (tuple(key_states.shape),))
    W = engine._k_gpu.shape[1] // bs
    if int(mirror_hi) != W - 1:
        raise ValueError("mirror_hi=%d must be W-1=%d" % (mirror_hi, W - 1))
    engine.seq_length += 1                                                                              # :345
    engine._cache_lens[...] += 1                                                                        # :346
    if not rolled:
        pos = engine._tail_block_idx_on_gpu * engine.block_size + engine._tail_block_len_on_gpu          # :349
    else:
        if engine._tail_block_len_on_gpu >= bs:
            raise ValueError("a second rollover inside one draft is not supported (tail_len=%d)" % engine._tail_block_len_on_gpu)
        pos = int(mirror_hi) * bs + engine._tail_block_len_on_gpu
    engine._k_gpu[:, pos:pos+1, :, :].copy_(key_states, non_blocking=True)                              # :350
    engine._v_gpu[:, pos:pos+1, :, :].copy_(value_states, non_blocking=True)                            # :351
    engine._kv_bias_gpu[:, pos:pos+1, :].copy_(kv_bias[:, engine.seq_length-1:engine.seq_length, :], non_blocking=True)  # :354
    engine._tail_block_len_on_gpu += 1                                                                  # :357
    if not rolled and engine._tail_block_len_on_gpu == engine.block_size:                               # :359, :401
        tail_block_base_pos = engine._tail_block_idx_on_gpu * engine.block_size                         # :402
        cpu_block_base_pos = engine.seq_length - engine.block_size                                      # :403
        engine._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+engine.block_size, :, :].copy_(engine._k_gpu[:, tail_block_base_pos:tail_block_base_pos+engine.block_size, :, :], non_blocking=True)  # :404
        engine._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+engine.block_size, :, :].copy_(engine._v_gpu[:, tail_block_base_pos:tail_block_base_pos+engine.block_size, :, :], non_blocking=True)  # :405
        engine._block_map[..., engine._tail_block_idx_on_gpu] += 1                                      # :407
        engine._tail_block_len_on_gpu = 0                                                               # :408
        engine._cache_lens[...] = (engine.topk - 1) * engine.block_size + engine._tail_block_len_on_gpu # :409
        return True
    return rolled


class TailCounters(NamedTuple):
    """The engine-level tail state at round start (host ints + what a fill
    changes), restored before the verify (restore_tail_counters)."""
    seq_length: int
    tail_len: int
    tail_id: int


def capture_tail_counters(engine, tail_id: int) -> TailCounters:
    return TailCounters(int(engine.seq_length), int(engine._tail_block_len_on_gpu), int(tail_id))


def restore_tail_counters(engine, tc: TailCounters, rolled: bool):
    """Undo the draft's K single-row writes on the counters: seq_length,
    tail_len, _cache_lens, and the tail slot's rename if the draft rolled
    over. The rows the draft wrote (slot topk-1 above tail_len_0, mirror_hi)
    are overwritten by the verify's write_tail; nothing else moved."""
    engine.seq_length = int(tc.seq_length)
    engine._tail_block_len_on_gpu = int(tc.tail_len)
    engine._cache_lens[...] = (engine.topk - 1) * engine.block_size + int(tc.tail_len)
    if rolled:
        engine._block_map[..., engine._tail_block_idx_on_gpu] -= 1


# ---------------------------------------------------------------------------
# the verify round over the persistent store
# ---------------------------------------------------------------------------
class RoundPlan(NamedTuple):
    mask: torch.Tensor          # (H, B*U, K) int32 union slot ids, -1 padded, row b*U + u (spec C5); slot topk-1 never named
    fetch: torch.Tensor         # (H, B, W) int64: block id to gather NOW into the slot on the main stream (late misses)
    overflow: torch.Tensor      # (H, B) bool: the union does not fit; that stream's mask rows are -1
    invalid: torch.Tensor       # (H, B) bool: a selection named an id above the live tail or below -1
    n_new: torch.Tensor         # (H, B): blocks fetched at the round (= late_fetch); the name verify_pilot's union_stats reads
    n_need: torch.Tensor        # (H, B): distinct non-tail blocks the U selections name
    n_hit_old: torch.Tensor     # (H, B): of those, resident since before the round
    n_timely: torch.Tensor      # (H, B): prefetched this round and arrived by verify start
    n_late_wait: torch.Tensor   # (H, B): prefetched this round, still in flight at verify start (waited for)
    n_late_fetch: torch.Tensor  # (H, B): not in the store: fetched at the round
    union_size: torch.Tensor    # (H, B): RESIDENT slots after the round


def plan_round(store: Store, l: int, sel: torch.Tensor, tail_id: int, rollover_at: int,
               gen_round0: int, gen_now: int, stamp_now: int, was_inflight: torch.Tensor) -> RoundPlan:
    """The verify's union over the persistent store for layer ``l``.

    sel: (U, H, B, K) int64 per-position selections. tail_id: T at round
    start. rollover_at: from tail_write.write_tail (positions >= rollover_at
    see T+1 as the live tail; == U when none does). gen_round0: the issue
    counter at round start (a slot with gen >= gen_round0 was filled this
    round). was_inflight: (H, B, W) bool, the INFLIGHT slots at verify start
    BEFORE the main stream waited for the side stream; the caller marks them
    RESIDENT after the wait, so here every present slot is RESIDENT.
    Mutates the store: fetched slots RESIDENT with gen_now; every named slot
    stamped and used. The mask's tail entries name the mirror slots.
    """
    lay = store.lay
    ids, state, gen, stamp, used = store.ids[l], store.state[l], store.gen[l], store.stamp[l], store.used[l]
    if sel.dim() != 4 or tuple(sel.shape[1:3]) != (store.H, store.B) or sel.dtype != torch.int64:
        raise ValueError("sel must be int64 (U, H=%d, B=%d, K), got %s %s" % (store.H, store.B, sel.dtype, tuple(sel.shape)))
    U, H, B, K = sel.shape
    W = lay.W
    if not (1 <= rollover_at <= U):
        raise ValueError("rollover_at=%d must be in [1, U=%d]" % (rollover_at, U))
    if tuple(was_inflight.shape) != (H, B, W):
        raise ValueError("was_inflight must be (H, B, W)")
    if bool((state == INFLIGHT).any()):
        raise RuntimeError("plan_round: INFLIGHT slots remain; the caller must wait for the side stream and mark_arrived first")
    T = int(tail_id)
    valid = sel >= 0
    live_tail = torch.tensor([T + (1 if u >= rollover_at else 0) for u in range(U)], dtype=torch.int64, device=sel.device).view(U, 1, 1, 1)
    invalid = ((valid & (sel > live_tail)) | (sel < -1)).any(-1).any(0)          # (H, B)
    is_tail = valid & (sel == T)
    is_tail_next = valid & (sel == T + 1) & (torch.arange(U, device=sel.device).view(U, 1, 1, 1) >= rollover_at)

    # the distinct non-tail blocks the round needs, position-major, first occurrence
    need = sel.permute(1, 2, 0, 3).reshape(H, B, U * K)
    need_valid = (valid & ~is_tail & ~is_tail_next).permute(1, 2, 0, 3).reshape(H, B, U * K)
    need_ids = torch.where(need_valid, need, torch.full_like(need, -2))
    first = _first_occurrence(need_ids) & need_valid
    present = state == RESIDENT
    eq = (need_ids.unsqueeze(-1) == ids.unsqueeze(-2)) & present.unsqueeze(-2)   # (H, B, U*K, W)
    found = eq.any(-1) & first
    slot = eq.to(torch.int64).argmax(-1)
    slot_gen = torch.gather(gen, -1, slot)
    slot_infl = torch.gather(was_inflight, -1, slot)
    late_wait = found & slot_infl
    timely = found & ~slot_infl & (slot_gen >= int(gen_round0))
    hit_old = found & ~slot_infl & (slot_gen < int(gen_round0))
    new = first & ~found
    named = torch.zeros((H, B, W + 1), dtype=torch.bool, device=sel.device)
    named.scatter_(-1, torch.where(found, slot, torch.full_like(slot, W)), torch.ones_like(slot, dtype=torch.bool))
    named = named[..., :W]

    cand = store.evictable.view(1, 1, W) & ~named & ((state == FREE) | (state == RESIDENT))
    key = _slot_key(state, stamp, cand)
    fetch, fits, n_cand = _assign(new, need_ids, cand, key)
    n_new = new.sum(-1)
    overflow = n_new > n_cand
    bad = overflow | invalid
    newly = fetch >= 0
    evicted_unused = newly & (state == RESIDENT) & ~used

    ids2 = torch.where(newly, fetch, ids)
    state2 = torch.where(newly, torch.full_like(state, RESIDENT), state)
    store.ids[l] = ids2
    store.state[l] = state2
    store.gen[l] = torch.where(newly, torch.full_like(gen, int(gen_now)), gen)
    store.stamp[l] = torch.where(named | newly, torch.full_like(stamp, int(stamp_now)), stamp)
    store.used[l] = used | named | newly
    # a bad stream fetches nothing (its mask is poisoned; the caller refuses the round)
    fetch = torch.where(bad.unsqueeze(-1), torch.full_like(fetch, -1), fetch)

    # the masks over the updated store
    present2 = state2 == RESIDENT
    masks = []
    for u in range(U):
        eq_u = (sel[u].unsqueeze(-1) == ids2.unsqueeze(-2)) & present2.unsqueeze(-2)   # (H, B, K, W)
        found_u = eq_u.any(-1)
        slot_u = eq_u.to(torch.int64).argmax(-1)
        m = torch.where(found_u, slot_u, torch.full_like(slot_u, -1))
        m = torch.where(is_tail[u], torch.full_like(m, lay.mirror_lo), m)
        m = torch.where(is_tail_next[u], torch.full_like(m, lay.mirror_hi), m)
        m = torch.where(bad.unsqueeze(-1), torch.full_like(m, -1), m)
        masks.append(m)
    mask = torch.stack(masks, dim=2).reshape(H, B * U, K).to(torch.int32)

    store.late_fetch_blocks[l] += fits.sum(-1)
    store.late_wait_blocks[l] += late_wait.sum(-1)
    store.timely_blocks[l] += timely.sum(-1)
    store.hit_old_blocks[l] += hit_old.sum(-1)
    store.wasted_blocks[l] += evicted_unused.sum(-1)
    return RoundPlan(mask=mask, fetch=fetch, overflow=overflow, invalid=invalid, n_new=n_new,
                     n_need=first.sum(-1), n_hit_old=hit_old.sum(-1), n_timely=timely.sum(-1),
                     n_late_wait=late_wait.sum(-1), n_late_fetch=fits.sum(-1), union_size=present2.sum(-1))


# ---------------------------------------------------------------------------
# accept / commit
# ---------------------------------------------------------------------------
class AcceptResult(NamedTuple):
    n_acc: torch.Tensor      # (B,) int64 per request: accepted draft tokens
    c: int                   # the lockstep commit length, 1 + min_b n_acc
    committed: torch.Tensor  # (B, c) int64: the tokens every request commits this round
    next_token: torch.Tensor # (B, 1) int64: committed[:, c-1], fed first next round


def accept_lengths(verify_argmax: torch.Tensor, drafted: torch.Tensor) -> torch.Tensor:
    """(B, U) verify argmaxes and (B, K) drafted tokens, U >= K + 1 ->
    (B,) n_acc: the longest prefix with verify_argmax[b, i] == drafted[b, i]."""
    if verify_argmax.dim() != 2 or drafted.dim() != 2 or verify_argmax.shape[0] != drafted.shape[0]:
        raise ValueError("shapes: verify_argmax %s drafted %s" % (tuple(verify_argmax.shape), tuple(drafted.shape)))
    K = drafted.shape[1]
    if verify_argmax.shape[1] < K + 1:
        raise ValueError("U=%d must be >= K+1=%d" % (verify_argmax.shape[1], K + 1))
    if K == 0:
        return torch.zeros((drafted.shape[0],), dtype=torch.int64, device=verify_argmax.device)
    match = (verify_argmax[:, :K] == drafted).to(torch.int64)
    return torch.cumprod(match, dim=1).sum(1)


def accept_commit(verify_argmax: torch.Tensor, drafted: torch.Tensor) -> AcceptResult:
    """WHY committed = argmax(verify[b, 0:c]) is exact for every request.
    verify[b, 0] is the exact-attention distribution after feeding x_n (the
    last committed token), so argmax(verify[b, 0]) is the shipped greedy token
    t_{n+1}. For 1 <= i < c: c - 1 <= n_acc(b), so d_1 .. d_i were all
    accepted, i.e. d_j = t_{n+j}; verify[b, i] is then the exact distribution
    after feeding t_{n+1} .. t_{n+i}, so its argmax is t_{n+i+1}. The batch
    commits argmax(verify[b, 0:c]) for every b: c-1 accepted draft tokens
    plus the verify's own next token as the bonus. Any c <= 1 + min n_acc is
    exact; the loop takes the largest.
    One host sync: the min. (Path-1 numerics vs the decode kernel: argmax
    agreement is the measured gate, ledger 2026-09-19; the pilot's sequence
    identity gate is the end-to-end check.)"""
    n_acc = accept_lengths(verify_argmax, drafted)
    c = 1 + int(n_acc.min())
    committed = verify_argmax[:, :c]
    return AcceptResult(n_acc=n_acc, c=c, committed=committed, next_token=committed[:, c - 1:c])


# ---------------------------------------------------------------------------
# rollback: the tail (in place) and the layer tables (journal)
# ---------------------------------------------------------------------------
class TruncateResult(NamedTuple):
    keep: int
    tail_len: int
    seq_length: int
    tail_id: int              # the live tail's id after the truncation
    undo_rollover: bool
    rows_restored: int        # rows of slot topk-1 copied back from mirror_lo


def truncate_tail(engine, tail, keep: int, mirror_lo: int) -> TruncateResult:
    """Roll the engine's tail back from ``tail.U`` written tokens to the first
    ``keep`` (0 <= keep <= U), IN PLACE, after tail_write.write_tail.

    ``tail`` is the TailWriteResult of that write (U, tail_len_before,
    rolled_over, mirror_lo/hi). The token that filled block T is index
    f = block_size - 1 - tail_len_before (valid iff rolled_over).
      keep >= f + 1  the fill is kept: tail_len = keep - (f + 1) rows of T+1
                     in slot topk-1, _block_map[..., topk-1] stays T+1.
      otherwise      tail_len = tail_len_before + keep; if the round rolled
                     over, the rename is undone (_block_map -= 1) and the
                     U - (f + 1) rows of slot topk-1 that T+1's tokens
                     overwrote are copied back from mirror_lo, which holds T
                     complete after write_tail (spec 2b).
    seq_length = seq0 + keep; _cache_lens = (topk-1)*bs + tail_len. The host
    write-back of a kept fill stays (exact rows); of an undone fill it is
    beyond the committed length and is rewritten at the next fill; the tail
    block is always served from slot topk-1, never from the host.
    What is NOT touched, by design: the store / round region (a cache of
    exact blocks), mirror_hi, rows of slot topk-1 at or above tail_len (stale,
    never read: the decode kernel reads rows < _cache_lens, the verify masks
    name the mirror), the layer tables (LayerJournal).
    """
    U = int(tail.U)
    if not (0 <= keep <= U):
        raise ValueError("keep=%d outside [0, U=%d]" % (keep, U))
    bs = int(engine.block_size)
    tail_slot = int(engine._tail_block_idx_on_gpu)
    if int(mirror_lo) != int(tail.mirror_lo):
        raise ValueError("mirror_lo=%d != the write's %d" % (mirror_lo, tail.mirror_lo))
    tl0 = int(tail.tail_len_before)
    seq0 = int(engine.seq_length) - U
    rolled = bool(tail.rolled_over)
    f = bs - 1 - tl0
    undo = False
    restored = 0
    if rolled and keep >= f + 1:
        tail_len = keep - (f + 1)
    else:
        tail_len = tl0 + keep
        if rolled:
            engine._block_map[..., tail_slot] -= 1
            undo = True
            restored = U - (f + 1)
            if restored > 0:
                ts, ms = tail_slot * bs, int(mirror_lo) * bs
                engine._k_gpu[:, ts:ts + restored].copy_(engine._k_gpu[:, ms:ms + restored], non_blocking=True)
                engine._v_gpu[:, ts:ts + restored].copy_(engine._v_gpu[:, ms:ms + restored], non_blocking=True)
                engine._kv_bias_gpu[:, ts:ts + restored].copy_(engine._kv_bias_gpu[:, ms:ms + restored], non_blocking=True)
    if not (0 <= tail_len < bs):
        raise AssertionError("truncate_tail: tail_len=%d after keep=%d (tl0=%d, U=%d, rolled=%s)" % (tail_len, keep, tl0, U, rolled))
    engine.seq_length = seq0 + keep
    engine._tail_block_len_on_gpu = tail_len
    engine._cache_lens[...] = (engine.topk - 1) * bs + tail_len
    tail_id = (seq0 + keep) // bs
    return TruncateResult(keep=keep, tail_len=tail_len, seq_length=seq0 + keep, tail_id=tail_id,
                          undo_rollover=undo, rows_restored=restored)


class LayerJournal:
    """Snapshot of one InfLLMv2CacheLayer's per-token tables at round start and
    the exact replay of the committed positions after the verify.

    The verify's scoring loop (nosa_llama.verify_forward) advances, per
    position, update_no_compress_k_decode, update_compress_k_decode,
    update_uncompressed_cis and update_cis; the draft steps advance the same
    four. Two of the tables are SHIFTED on a compress event (no_compress_k_cache
    and tail_cis, every 16 tokens), one is REBOUND by torch.cat
    (compress_k_cache_varlen: the old tensor is intact, so its reference is
    the snapshot), one is incremented in place (cached_compressed_cu_seqlens);
    compressed_cis and total_cis are only ever appended at a length counter,
    so their rows beyond the restored length are stale and unread. The
    scalars are python ints.

    restore() puts the layer back at round start; replay(c) applies the four
    updates for the first c positions with the SAME key/cis tensors the
    verify computed (kept by spec_verify_forward), in the same order the
    decode makes them. Exact by construction: it is that computation.
    """

    SCALARS = ("cached_compressed_max_seqlen", "no_compress_k_len", "comp_cis_len", "tail_cis_len", "cis_len", "seq_length")
    CLONED = ("no_compress_k_cache", "tail_cis", "cached_compressed_cu_seqlens")

    def __init__(self):
        self.scalars: dict = {}
        self.cloned: dict = {}
        self.compress_k_ref = None
        self.taken = False

    def take(self, layer):
        for n in self.SCALARS:
            self.scalars[n] = int(getattr(layer, n))
        for n in self.CLONED:
            src = getattr(layer, n)
            dst = self.cloned.get(n)
            if torch.is_tensor(dst) and dst.shape == src.shape and dst.dtype == src.dtype and dst.device == src.device:
                dst.copy_(src)
            else:
                self.cloned[n] = src.detach().clone()
        self.compress_k_ref = layer.compress_k_cache_varlen
        self.taken = True
        return self

    def restore(self, layer):
        if not self.taken:
            raise RuntimeError("LayerJournal.restore before take")
        for n in self.SCALARS:
            setattr(layer, n, self.scalars[n])
        for n in self.CLONED:
            getattr(layer, n).copy_(self.cloned[n])
        layer.compress_k_cache_varlen = self.compress_k_ref
        return self

    @staticmethod
    def replay(layer, key_states, cis, c: int, kernel_size: int, kernel_stride: int):
        """key_states (B, U, H, D), cis (B, U, H): the verify's per-position
        tensors; the first c positions are replayed (verify_forward's order)."""
        B = key_states.shape[0]
        if not (0 <= c <= key_states.shape[1]) or cis.shape[1] != key_states.shape[1]:
            raise ValueError("replay: c=%d, key_states %s, cis %s" % (c, tuple(key_states.shape), tuple(cis.shape)))
        for u in range(c):
            no_compress_k = layer.update_no_compress_k_decode(key_states[:, u:u+1], kernel_size, kernel_stride)
            new_compressed_k = no_compress_k.mean(dim=1, keepdim=True) if no_compress_k is not None else None
            layer.update_compress_k_decode(new_compressed_k, None)
            layer.update_uncompressed_cis(cis[:, u:u+1], 0, B)
            layer.update_cis(cis[:, u:u+1].permute(2, 0, 1), 0, B)
            layer.seq_length += 1                     # InfLLMv2CacheLayer.decode_update_kv (:899)


# ---------------------------------------------------------------------------
# the round context the engine and the model share
# ---------------------------------------------------------------------------
def _no_mark(key: str):
    return None


class SpecContext:
    """Per-process state of the loop, handed to the engine's spec_* methods.
    ``side`` is the side CUDA stream (duck-typed: wait_event(ev)); ``make_event``
    a factory of events with record(stream) / query(); ``current_stream`` returns
    the main stream; ``stream_ctx(side)`` returns the context manager that makes
    ``side`` current (torch.cuda.stream on the device, a no-op in tests).
    ``rolled[l]`` says whether layer l's draft filled its tail block this round
    (every layer fills at the same draft step, but each engine fills ITS OWN
    tail when its write runs, so the flag is per layer)."""

    def __init__(self, store: Store, side, make_event, current_stream, stream_ctx, num_layers: int,
                 block_size: int, kernel_size: int, kernel_stride: int):
        self.store = store
        self.side = side
        self.make_event = make_event
        self.current_stream = current_stream
        self.stream_ctx = stream_ctx
        self.num_layers = int(num_layers)
        self.queue = PrefetchQueue()
        self.gen = 0                 # issue counter (side prefetch and late fetch share it)
        self.tick = 0                # LRU stamp: one per draft step and per verify
        self.round_idx = 0
        self.block_size = int(block_size)
        self.kernel_size, self.kernel_stride = int(kernel_size), int(kernel_stride)
        self.journals = [LayerJournal() for _ in range(self.num_layers)]
        self.tail0: Optional[TailCounters] = None
        self.tail_id = 0
        self.rolled: List[bool] = [False] * self.num_layers
        self.gen_round0 = 0
        self.was_inflight = None
        self.host_syncs = 0
        self._scratch: dict = {}
        self._deny_buf: dict = {}
        self.full_len = None
        self.seen0 = 0
        self.mark = _no_mark          # the driver binds verify_trace.TRACE.rec (finer brackets); a no-op otherwise

    def on_side(self):
        return self.stream_ctx(self.side)

    def next_gen(self) -> int:
        self.gen += 1
        return self.gen

    def next_tick(self) -> int:
        self.tick += 1
        return self.tick

    def begin_round(self, cache, tail_id: int):
        eng0 = cache.layers[0].cache_engine
        self.round_idx += 1
        self.tail_id = int(tail_id)
        self.tail0 = capture_tail_counters(eng0, tail_id)
        self.rolled = [False] * self.num_layers
        self.gen_round0 = self.gen + 1
        self.seen0 = int(cache._seen_tokens)
        for j, lay in zip(self.journals, cache.layers):
            j.take(lay)

    def restore_before_verify(self, cache):
        """The draft's K steps are undone on the counters and the layer
        tables (its rows are overwritten by the verify's write_tail)."""
        for l, lay in enumerate(cache.layers):
            restore_tail_counters(lay.cache_engine, self.tail0, self.rolled[l])
        for j, lay in zip(self.journals, cache.layers):
            j.restore(lay)
        cache._seen_tokens = self.seen0

    def bias_scratch(self, l: int, kv_bias_gpu: torch.Tensor):
        s = self._scratch.get(l)
        if s is None or s.shape != kv_bias_gpu.shape or s.dtype != kv_bias_gpu.dtype:
            s = torch.empty_like(kv_bias_gpu)
            self._scratch[l] = s
        B, S, H = kv_bias_gpu.shape
        W = self.store.lay.W
        d = self._deny_buf.get(l)
        if d is None or tuple(d.shape) != (B, W, 1, H):
            d = torch.empty((B, W, 1, H), dtype=torch.bool, device=kv_bias_gpu.device)
            self._deny_buf[l] = d
        return s, d

    def full_lengths(self, B: int, device) -> torch.Tensor:
        if self.full_len is None or self.full_len.shape[0] != B:
            self.full_len = torch.full((B,), self.store.lay.W * self.block_size, dtype=torch.int32, device=device)
        return self.full_len


# ---------------------------------------------------------------------------
# accounting helpers (pure arithmetic, tested)
# ---------------------------------------------------------------------------
def lockstep_yield(n_acc_rounds: Sequence[Sequence[int]]) -> Tuple[int, float]:
    """Per round a sequence of n_acc over the batch -> (tokens committed under
    the lockstep policy, the ragged-engine projection = sum of per-request
    means of 1 + n_acc)."""
    lock = 0
    ragged = 0.0
    for row in n_acc_rounds:
        row = [int(x) for x in row]
        if not row:
            raise ValueError("empty round")
        lock += 1 + min(row)
        ragged += 1.0 + sum(row) / len(row)
    return lock, ragged


def wire_bytes(counters: dict, bytes_per_block: int = BYTES_WIRE_PER_BLOCK) -> dict:
    """Store counters (sums over layers, heads, requests as ints) -> bytes:
    prefetch (issued on the side), late (fetched at the round), wasted
    (evicted before use, a subset of prefetch); total on the wire = prefetch
    + late (late_wait blocks are already inside prefetch)."""
    pre = int(counters["prefetch_blocks"]) * bytes_per_block
    late = int(counters["late_fetch_blocks"]) * bytes_per_block
    wasted = int(counters["wasted_blocks"]) * bytes_per_block
    return dict(prefetch_bytes=pre, late_bytes=late, wasted_bytes=wasted, wire_bytes=pre + late)


def timely_recall(counters: dict) -> Optional[float]:
    """timely / (timely + late_wait + late_fetch): of the blocks the verify
    needed that were not resident at round start, the fraction the draft's
    prefetch had landed by verify start. None when nothing was needed."""
    t, w, f = int(counters["timely_blocks"]), int(counters["late_wait_blocks"]), int(counters["late_fetch_blocks"])
    n = t + w + f
    return None if n == 0 else t / n


# ===========================================================================
# PER-REQUEST ROUNDS (R2 milestone 2, 2026-10-20; SOURCE ONLY; CPU tests
# benchmarks/Efficiency/tests_cpu/test_spec_perreq.py)
# ===========================================================================
# ONE COUNTER PER REQUEST. pos (B,) int64 is the number of tokens that
# request b holds: the shipped seq_length of one request, and the ``pos`` row
# of the positional twin (nosi-r2-m2, r2_positional_twin.py at 65b7341:
# counters() is (4, B) = pos, n_comp, n_ccis, epoch). The tail state is
# DERIVED from pos, as the twin derives it (tail_row, tail_block, cache_lens):
#     tail_len   = pos % bs                 shipped _tail_block_len_on_gpu
#     tail id    = pos // bs                shipped _block_map[..., topk-1]
#     cache_lens = (topk-1)*bs + pos % bs   shipped _cache_lens, int32 (findings
#                                           F4 kept on purpose: parity target)
# The engine carries the counter as ``engine._pos`` (to_perreq). There,
# seq_length and _tail_block_len_on_gpu are None, so a lockstep function
# that gets a per-request engine raises.
#
# THE ROUND, per request (the lockstep schedule; K drafted positions, U = K+1):
#   DRAFT    draft_tail_write_perreq writes one row per request at ITS
#            position. After ITS rollover the rows go to mirror_hi, so the
#            exact rows of block T_b stay. draft_tail_state gives (tail_len,
#            rolled) per request. plan_draft_perreq and
#            apply_draft_bias_perreq are the lockstep plans with T_b,
#            rolled_b and forbid_ge = T_b per request.
#   RESTORE  restore_tail_counters_perreq: pos = pos0, a counter reset (the
#            map's tail column and cache_lens follow). The tables commit to
#            their round-start counters (keep 0).
#   VERIFY   write_tail_perreq writes U rows per request from its own
#            tail_len. It returns rollover_at and seqused_k per request (the
#            varlen verify call already takes seqused_k as (B,)) and the
#            counter trajectory ctr[u] = pos0 + u. plan_round_perreq is the
#            lockstep union; the live tail of request b at position u is
#            T_b + (u >= rollover_at_b).
#   ACCEPT   accept_perreq: keep (B,) = 1 + n_acc(b). No host sync.
#   COMMIT   commit_layer_perreq: the tables commit ctr[keep_b, :, b] (the
#            twin: a counter reset; no table row is copied, because the
#            twin's key ring has 64 >= 32 + U rows: a rollback over at most
#            32 positions, its MAX_ROLLBACK, never overwrites a live key; the
#            compressed tables and total_cis are only written at or above
#            their counters). The tail:
#            truncate_perreq sets pos = ctr[keep_b, b] (a counter reset) and
#            does ONE masked copy. For each request whose kept prefix undoes
#            a rollover, rows [0, U - (f_b + 1)) of slot topk-1 come back
#            from mirror_lo (torch.where); f_b = bs - 1 - tail_len0_b.
# F4 PARITY (2026-10-20; run p2: every above-tolerance identity mismatch was
# at a position p with p % 64 == 63). The per-request attention does not read
# _cache_lens: the draft attends the whole allocation through a masked bias,
# and the verify attends a mask of union slots with seqused_k. So the cache_lens
# formula above does not give findings F4 to the attention. The shipped decode,
# at a step that writes p % bs == bs - 1, reads rows [0, (topk-1)*bs) only: it
# skips the tail block, the current token included (docs/R2_ENGINE_FINDINGS.md
# F4). PerReqSpecContext.f4 selects the behaviour of the per-request round:
#   "shipped" (the default, the parity target): the verify drops the mirror
#            ids from the mask rows of the F4 positions (drop_tail_on_f4,
#            f4_steps of ctr[u] = pos0 + u); the draft masks the whole tail
#            slot at a step whose write filled T_b (apply_draft_bias_perreq,
#            f4 = f4_steps of the position written; mirror_hi has no row yet).
#   "fixed"  the tail block is read on every step, as the lockstep functions
#            read it (they are unchanged).
# The store, the fetches, the counters and the tables are the same in both
# modes: only the attention masks of the F4 positions differ.
# EXACTNESS. The argument of accept_commit holds per request: any c_b <=
# 1 + n_acc(b) is exact for request b. No request reads the rows or the
# counters of another request (every op below is per (h, b) or per b).
# HOST SYNCS. accept_perreq: none. The host write-back of a filled block
# reads the (B,) fill mask (_host_write_back). commit_layer_perreq compares
# the two trajectories (a guard). The engine can keep pos on the host: the
# loop reads keep (B,) once per round, as the lockstep loop reads c.
# NOT HERE: the engine and model wiring (cache_engine spec_*, the twin's
# tables in the model, the per-request kernels of nosi-r2-m2-kernels), the
# per-request pooling width (findings F1), the ragged outputs of the host
# loop.
# ---------------------------------------------------------------------------
def _bvec(x, B: int, name: str, dtype=torch.int64) -> torch.Tensor:
    """A (B,) tensor, checked, as ``dtype``. The per-request functions take
    tensors only: an int is a lockstep value."""
    if not torch.is_tensor(x) or tuple(x.shape) != (B,):
        got = tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
        raise ValueError("%s must be a (B=%d,) tensor, got %s" % (name, B, got))
    return x.to(dtype)


def _batch(engine) -> int:
    return int(engine._k_gpu.shape[0])


F4_SHIPPED, F4_FIXED = "shipped", "fixed"
F4_MODES = (F4_SHIPPED, F4_FIXED)


def check_f4_mode(mode) -> str:
    """The F4 mode of the per-request round (see F4 PARITY above): "shipped"
    or "fixed". Raises ValueError for any other value."""
    if mode not in F4_MODES:
        raise ValueError("f4 mode %r: must be one of %s" % (mode, F4_MODES))
    return mode


def f4_steps(p: torch.Tensor, block_size: int) -> torch.Tensor:
    """Elementwise bool: the step that writes position p is a findings-F4
    step (p % bs == bs - 1: the write fills the tail block, and the shipped
    attention reads rows [0, (topk-1)*bs) only)."""
    bs = int(block_size)
    return p % bs == bs - 1


def drop_tail_on_f4(mask: torch.Tensor, f4: torch.Tensor, layout: StoreLayout) -> torch.Tensor:
    """The verify's F4 parity. mask: (H, B*U, K) int32 union slot ids, row
    b*U + u (RoundPlan.mask). f4: (U, B) bool (f4_steps of the positions
    pos0_b + u). On every row with f4[u, b], each entry that names a tail
    mirror (mirror_lo: block T_b; mirror_hi: T_b + 1) becomes -1, so that
    position reads only its non-tail selected blocks, as the shipped decode
    reads rows [0, (topk-1)*bs) at that step. Every other row and entry is
    unchanged. Pure; returns a new contiguous int32 mask."""
    if mask.dim() != 3 or f4.dim() != 2 or f4.dtype != torch.bool:
        raise ValueError("drop_tail_on_f4: mask (H, B*U, K) and f4 (U, B) bool, got %s and %s %s"
                         % (tuple(mask.shape), f4.dtype, tuple(f4.shape)))
    U, B = f4.shape
    if mask.shape[1] != B * U:
        raise ValueError("drop_tail_on_f4: mask has %d rows, f4 names B*U = %d" % (mask.shape[1], B * U))
    rows = f4.t().reshape(1, B * U, 1).to(mask.device)               # row b*U + u
    tail = (mask == int(layout.mirror_lo)) | (mask == int(layout.mirror_hi))
    return torch.where(rows & tail, torch.full_like(mask, -1), mask).contiguous()


def _set_tail_state(engine, pos: torch.Tensor, tail_ids: torch.Tensor, tail_len: torch.Tensor):
    """Write the counter and the state derived from it: the map's tail column
    (every KV head) and cache_lens = (topk-1)*bs + tail_len."""
    engine._pos.copy_(pos)
    engine._block_map[..., int(engine._tail_block_idx_on_gpu)] = tail_ids.view(1, -1).to(engine._block_map.dtype)
    engine._cache_lens.copy_(((int(engine.topk) - 1) * int(engine.block_size) + tail_len).to(engine._cache_lens.dtype))


def check_tail_state(engine, pos: torch.Tensor):
    """The shipped invariants against per-request counters: the map's tail
    column (every head) == pos // bs and _cache_lens == (topk-1)*bs +
    pos % bs. Raises ValueError (a guard; one host read)."""
    bs = int(engine.block_size)
    B = _batch(engine)
    pos = _bvec(pos, B, "pos")
    tail = engine._block_map[..., int(engine._tail_block_idx_on_gpu)]
    if not bool((tail == (pos // bs).view(1, B)).all()):
        raise ValueError("the block map's tail column is not pos // %d" % bs)
    want = ((int(engine.topk) - 1) * bs + pos % bs).to(engine._cache_lens.dtype)
    if not torch.equal(engine._cache_lens, want):
        raise ValueError("_cache_lens is not (topk-1)*%d + pos %% %d" % (bs, bs))


def to_perreq(engine) -> torch.Tensor:
    """Move a lockstep engine to per-request counters: ``engine._pos`` = (B,)
    int64, every request at seq_length. seq_length and
    _tail_block_len_on_gpu become None, so a lockstep reader raises. Refuses
    an engine whose tail state is not its counters' (check_tail_state).
    Returns engine._pos."""
    bs = int(engine.block_size)
    if getattr(engine, "seq_length", None) is None:
        raise ValueError("to_perreq: the engine has no lockstep seq_length (per-request already?)")
    seq, tl = int(engine.seq_length), int(engine._tail_block_len_on_gpu)
    if tl != seq % bs:
        raise ValueError("to_perreq: tail_len %d != seq_length %d %% %d" % (tl, seq, bs))
    pos = torch.full((_batch(engine),), seq, dtype=torch.int64, device=engine._block_map.device)
    check_tail_state(engine, pos)
    engine._pos = pos
    engine.seq_length = None
    engine._tail_block_len_on_gpu = None
    return engine._pos


def _host_write_back(engine, mask: torch.Tensor, block_ids: torch.Tensor, src_slot: int):
    """cache_engine.py:402-405 for the requests in ``mask``: block
    block_ids[b] of request b goes to the host from slot ``src_slot``. One
    host read of the (B,) mask, then one contiguous copy per filled request
    (the shipped write-back is one copy for the whole batch)."""
    bs = int(engine.block_size)
    s = int(src_slot) * bs
    for b in mask.nonzero().flatten().tolist():
        base = int(block_ids[b]) * bs
        engine._k_cpu[b, base:base + bs].copy_(engine._k_gpu[b, s:s + bs], non_blocking=True)
        engine._v_cpu[b, base:base + bs].copy_(engine._v_gpu[b, s:s + bs], non_blocking=True)


def _masked_slot_copy(engine, dst_slot: int, src_slot: int, nrows: torch.Tensor):
    """Per request, rows [0, nrows[b]) of slot ``dst_slot`` <- the same rows
    of slot ``src_slot``, for K, V and the bias: ONE torch.where over the
    slot. Rows at or above nrows[b] keep their values."""
    bs = int(engine.block_size)
    B = _batch(engine)
    m = torch.arange(bs, device=nrows.device).view(1, bs) < nrows.view(B, 1)
    d, s = int(dst_slot) * bs, int(src_slot) * bs
    for name in ("_k_gpu", "_v_gpu"):
        t = getattr(engine, name)
        t[:, d:d + bs].copy_(torch.where(m.view(B, bs, 1, 1), t[:, s:s + bs], t[:, d:d + bs]))
    t = engine._kv_bias_gpu
    t[:, d:d + bs].copy_(torch.where(m.view(B, bs, 1), t[:, s:s + bs], t[:, d:d + bs]))


def _masked_tail_restore(engine, tail_slot: int, mirror_lo: int, nrows: torch.Tensor):
    """The restore of truncate_perreq. It has its own name so that the
    negative control can remove it."""
    _masked_slot_copy(engine, tail_slot, mirror_lo, nrows)


# -- the draft, per request ----------------------------------------------------
class PerReqTailCounters(NamedTuple):
    """The tail state per request at round start: pos0 (B,) and tail_id =
    pos0 // bs (B,)."""
    pos: torch.Tensor
    tail_id: torch.Tensor


def capture_tail_counters_perreq(engine) -> PerReqTailCounters:
    pos = engine._pos.clone()
    return PerReqTailCounters(pos, pos // int(engine.block_size))


def restore_tail_counters_perreq(engine, tc: PerReqTailCounters):
    """restore_tail_counters per request: pos = pos0. The map's tail column
    and cache_lens follow, so a rename by the draft's rollover is undone with
    them. The verify's write overwrites the draft's rows."""
    bs = int(engine.block_size)
    pos = _bvec(tc.pos, _batch(engine), "pos0")
    _set_tail_state(engine, pos, pos // bs, pos % bs)


def draft_tail_state(pos: torch.Tensor, tail_id0: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(tail_len, rolled) per request inside a draft that started with tail
    id T = tail_id0. rolled: the draft filled block T (pos >= (T+1)*bs).
    tail_len: the rows of the live block (T before the rollover, T+1 after),
    in [0, bs]. This is the lockstep counter, which reaches bs when the draft
    writes the last row of T+1. Refuses pos outside [T*bs, (T+2)*bs]."""
    bs = int(block_size)
    lo = tail_id0 * bs
    if bool(((pos < lo) | (pos > lo + 2 * bs)).any()):
        raise ValueError("draft_tail_state: a position is outside [T*%d, (T+2)*%d]" % (bs, bs))
    rolled = pos >= lo + bs
    tail_len = pos - lo - rolled.to(pos.dtype) * bs
    return tail_len, rolled


def draft_row_limits_perreq(tail_len: torch.Tensor, rolled: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """draft_row_limits per request: the rows of the tail slot and the rows
    of mirror_hi that the draft may see (exclusive bounds), each (B,)."""
    bs = int(block_size)
    if bool(((tail_len < 0) | (tail_len > bs)).any()):
        raise ValueError("tail_len outside [0, %d]" % bs)
    return (torch.where(rolled, torch.full_like(tail_len, bs), tail_len),
            torch.where(rolled, tail_len, torch.zeros_like(tail_len)))


def draft_tail_write_perreq(engine, key_states, value_states, kv_bias, mirror_hi: int, tail_id0: torch.Tensor) -> torch.Tensor:
    """draft_tail_write per request. Request b writes its token at position
    p_b = engine._pos[b]: to slot topk-1, row p_b % bs, while p_b is in block
    T_b = tail_id0[b]; to mirror_hi, row p_b % bs, after the draft filled
    T_b. The bias row is kv_bias[b, p_b] (the shipped row seq_length - 1
    after the increment, :354). A request whose write fills T_b writes the
    block back to the host from slot topk-1 (:402-405), and its map tail
    column becomes T_b + 1 (:407). A write into block T_b + 2 (a second
    rollover) is refused, as in the lockstep function. Returns rolled (B,)
    bool after the write."""
    bs = int(engine.block_size)
    B = _batch(engine)
    if key_states.dim() != 4 or tuple(key_states.shape[:2]) != (B, 1):
        raise ValueError("draft_tail_write_perreq takes one token (B=%d, 1, H, D), got %s" % (B, tuple(key_states.shape)))
    if tuple(value_states.shape) != tuple(key_states.shape):
        raise ValueError("value_states %s != key_states %s" % (tuple(value_states.shape), tuple(key_states.shape)))
    W = engine._k_gpu.shape[1] // bs
    if int(mirror_hi) != W - 1:
        raise ValueError("mirror_hi=%d must be W-1=%d" % (mirror_hi, W - 1))
    T = _bvec(tail_id0, B, "tail_id0")
    p = engine._pos.clone()
    tail_len, rolled = draft_tail_state(p, T, bs)
    if bool((rolled & (tail_len >= bs)).any()):
        raise ValueError("a second rollover inside one draft is not supported")
    if kv_bias.shape[0] != B or kv_bias.shape[1] <= int(p.max()):
        raise ValueError("kv_bias %s has no row %d" % (tuple(kv_bias.shape), int(p.max())))
    ts = int(engine._tail_block_idx_on_gpu)
    row = torch.where(rolled, int(mirror_hi) * bs + tail_len, ts * bs + tail_len)
    bi = torch.arange(B, device=p.device)
    engine._k_gpu[bi, row] = key_states[:, 0].to(engine._k_gpu.dtype)
    engine._v_gpu[bi, row] = value_states[:, 0].to(engine._v_gpu.dtype)
    engine._kv_bias_gpu[bi, row] = kv_bias[bi, p].to(engine._kv_bias_gpu.dtype)
    _host_write_back(engine, ~rolled & (tail_len + 1 == bs), T, ts)
    p1 = p + 1
    tl1, rolled1 = draft_tail_state(p1, T, bs)
    _set_tail_state(engine, p1, T + rolled1.to(T.dtype), tl1)
    return rolled1


def plan_draft_perreq(store: Store, l: int, sel: torch.Tensor, tail_id: torch.Tensor, rolled: torch.Tensor,
                      gen_now: int, stamp_now: int, forbid_ge: torch.Tensor) -> DraftPlan:
    """plan_draft with the tail per request: tail_id, rolled and forbid_ge
    are (B,) tensors (T_b at round start, the draft's rollover of request b,
    T_b). The lines that use them are the only change; every other line is
    plan_draft's."""
    lay = store.lay
    ids, state, gen, stamp, used = store.ids[l], store.state[l], store.gen[l], store.stamp[l], store.used[l]
    if sel.dim() != 3 or tuple(sel.shape[:2]) != (store.H, store.B) or sel.dtype != torch.int64:
        raise ValueError("sel must be int64 (H=%d, B=%d, K), got %s %s" % (store.H, store.B, sel.dtype, tuple(sel.shape)))
    H, B, K = sel.shape
    W = lay.W
    t = _bvec(tail_id, B, "tail_id").view(1, B, 1)
    r = _bvec(rolled, B, "rolled", torch.bool).view(1, B, 1)
    fg = _bvec(forbid_ge, B, "forbid_ge").view(1, B, 1)
    valid = sel >= 0
    is_tail = valid & ((sel == t) | (r & (sel == t + 1)))
    present = (state == INFLIGHT) | (state == RESIDENT)
    first = _first_occurrence(sel)
    eq = (sel.unsqueeze(-1) == ids.unsqueeze(-2)) & present.unsqueeze(-2)     # (H, B, K, W)
    found = eq.any(-1) & valid & ~is_tail & first
    slot = eq.to(torch.int64).argmax(-1)
    slot_state = torch.gather(state.to(torch.int64), -1, slot)
    hit = found & (slot_state == RESIDENT)
    inflight = found & (slot_state == INFLIGHT)
    named = torch.zeros((H, B, W + 1), dtype=torch.bool, device=sel.device)
    named.scatter_(-1, torch.where(found, slot, torch.full_like(slot, W)), torch.ones_like(slot, dtype=torch.bool))
    named = named[..., :W]
    attended = named & (state == RESIDENT)
    deny = ~attended
    deny[..., lay.tail_slot] = False
    deny[..., lay.mirror_hi] = False
    deny[..., lay.mirror_lo] = True

    miss = valid & ~is_tail & ~found & first
    fetchable = miss & (sel < fg)
    forbidden = miss & ~fetchable
    cand = store.evictable.view(1, 1, W) & ~named & (state != INFLIGHT) & ((state == FREE) | (state == RESIDENT))
    key = _slot_key(state, stamp, cand)
    issue, fits, _ = _assign(fetchable, sel, cand, key)
    newly = issue >= 0
    evicted_unused = newly & (state == RESIDENT) & ~used

    store.ids[l] = torch.where(newly, issue, ids)
    store.state[l] = torch.where(newly, torch.full_like(state, INFLIGHT), state)
    store.gen[l] = torch.where(newly, torch.full_like(gen, int(gen_now)), gen)
    store.stamp[l] = torch.where(named | newly, torch.full_like(stamp, int(stamp_now)), stamp)
    store.used[l] = torch.where(newly, torch.zeros_like(used), used | attended)

    n_issue = fits.sum(-1)
    n_unreq = (fetchable & ~fits).sum(-1)
    store.prefetch_blocks[l] += n_issue
    store.unrequested_blocks[l] += n_unreq
    store.forbidden_blocks[l] += forbidden.sum(-1)
    store.wasted_blocks[l] += evicted_unused.sum(-1)
    n_sel = (valid & ~is_tail & first).sum(-1)
    store.draft_sel_blocks[l] += n_sel
    store.draft_hit_blocks[l] += hit.sum(-1)
    store.draft_inflight_blocks[l] += inflight.sum(-1)
    return DraftPlan(deny=deny, issue=issue, n_sel=n_sel, n_hit=hit.sum(-1),
                     n_inflight=inflight.sum(-1), n_issue=n_issue, n_unrequested=n_unreq,
                     n_forbidden=forbidden.sum(-1), n_evicted_unused=evicted_unused.sum(-1))


def apply_draft_bias_perreq(scratch: torch.Tensor, kv_bias_gpu: torch.Tensor, deny: torch.Tensor,
                            deny_buf: torch.Tensor, layout: StoreLayout, block_size: int,
                            tail_len: torch.Tensor, rolled: torch.Tensor, mask_bias: float = MASK_BIAS,
                            f4: Optional[torch.Tensor] = None) -> torch.Tensor:
    """apply_draft_bias with the row limits per request
    (draft_row_limits_perreq): MASK_BIAS on the rows of the tail slot at or
    above tail_ok[b] and on the rows of mirror_hi at or above hi_ok[b].
    f4 (B,) bool, optional: the requests whose step is a findings-F4 step
    (f4_steps of the position just written): tail_ok and hi_ok are 0 for
    them, so the whole tail slot is masked, as the shipped decode reads rows
    [0, (topk-1)*bs) only at that step. None: no request (the lockstep
    behaviour; F4 PARITY above)."""
    B, S, H = kv_bias_gpu.shape
    W = layout.W
    if S != W * block_size or tuple(scratch.shape) != (B, S, H) or tuple(deny.shape) != (H, B, W):
        raise ValueError("apply_draft_bias_perreq: bias %s scratch %s deny %s W=%d bs=%d" % (
            tuple(kv_bias_gpu.shape), tuple(scratch.shape), tuple(deny.shape), W, block_size))
    if tuple(deny_buf.shape) != (B, W, 1, H):
        raise ValueError("deny_buf must be (B, W, 1, H)=%s, got %s" % ((B, W, 1, H), tuple(deny_buf.shape)))
    tail_ok, hi_ok = draft_row_limits_perreq(_bvec(tail_len, B, "tail_len"), _bvec(rolled, B, "rolled", torch.bool), block_size)
    if f4 is not None:
        f4 = _bvec(f4, B, "f4", torch.bool)
        tail_ok = torch.where(f4, torch.zeros_like(tail_ok), tail_ok)
        hi_ok = torch.where(f4, torch.zeros_like(hi_ok), hi_ok)
    scratch.copy_(kv_bias_gpu)
    deny_buf.copy_(deny.permute(1, 2, 0).unsqueeze(2))
    scratch.view(B, W, block_size, H).masked_fill_(deny_buf, mask_bias)
    rows = torch.arange(block_size, device=scratch.device).view(1, block_size, 1)
    ts, hs = layout.tail_slot * block_size, layout.mirror_hi * block_size
    scratch[:, ts:ts + block_size].masked_fill_(rows >= tail_ok.view(B, 1, 1), mask_bias)
    scratch[:, hs:hs + block_size].masked_fill_(rows >= hi_ok.view(B, 1, 1), mask_bias)
    return scratch


# -- the verify, per request ---------------------------------------------------
class PerReqTail(NamedTuple):
    """What write_tail_perreq wrote: TailWriteResult's fields per request."""
    U: int
    pos0: torch.Tensor              # (B,) tokens held before the round
    tail_len_before: torch.Tensor   # (B,) pos0 % bs
    tail_id: torch.Tensor           # (B,) T_b = pos0 // bs
    rolled_over: torch.Tensor       # (B,) bool: a token of the round filled T_b (tail_len_before + U >= bs)
    fill_at: torch.Tensor           # (B,) bs - 1 - tail_len_before: the token that fills T_b (where rolled_over)
    rollover_at: torch.Tensor       # (B,) the first token in T_b + 1; U when no token is (no fill, or a fill on the last token)
    tail_len_after: torch.Tensor    # (B,) (pos0 + U) % bs
    mirror_lo: int
    mirror_hi: int
    last_used_slot: torch.Tensor    # (B,) mirror_hi where rolled_over, else mirror_lo
    seqused_k: torch.Tensor         # (B,) int32: last_used_slot * bs + tail_len_after
    ctr: torch.Tensor               # (U + 1, B) int64: pos after u tokens, pos0 + u (row 0 of the twin's ctr)


def write_tail_perreq(engine, key_states, value_states, kv_bias, mirror_slot: int) -> PerReqTail:
    """tail_write.write_tail per request: request b writes its U tokens from
    its own position pos0_b = engine._pos[b]. Token i of request b (position
    p = pos0_b + i) goes to slot topk-1, row p % bs, and to the mirror
    (mirror_lo row p % bs while p is in T_b, mirror_hi row p % bs after). Its
    bias row is kv_bias[b, p]. At round start, rows [0, tail_len0_b) of slot
    topk-1 are copied to mirror_lo (one masked copy), so mirror_lo holds T_b
    complete after the round. The token that fills T_b writes the block back
    to the host from slot topk-1 (:402-405) before the tokens of T_b + 1
    reuse its rows. After the round pos = pos0 + U, and the map's tail column
    and cache_lens follow. The refusals are those of write_tail, per
    request; a refused call changes nothing."""
    k_gpu = engine._k_gpu
    bs = int(engine.block_size)
    topk = int(engine.topk)
    ts = int(engine._tail_block_idx_on_gpu)
    if key_states.dim() != 4:
        raise ValueError("key_states must be (B, U, H, D), got %s" % (tuple(key_states.shape),))
    B, U, H, D = key_states.shape
    if tuple(value_states.shape) != (B, U, H, D):
        raise ValueError("value_states %s != key_states %s" % (tuple(value_states.shape), (B, U, H, D)))
    if U < 1:
        raise ValueError("write_tail_perreq: U must be >= 1, got %d" % U)
    if k_gpu.shape[0] != B or k_gpu.shape[2] != H or k_gpu.shape[3] != D or k_gpu.shape[1] % bs != 0:
        raise ValueError("_k_gpu %s does not match key_states %s (block_size %d)" % (tuple(k_gpu.shape), (B, U, H, D), bs))
    W = k_gpu.shape[1] // bs
    if ts != topk - 1:
        raise ValueError("_tail_block_idx_on_gpu=%d but topk-1=%d" % (ts, topk - 1))
    mirror_lo = int(mirror_slot)
    mirror_hi = mirror_lo + 1
    if mirror_lo != W - 2 or mirror_lo < topk:
        raise ValueError("mirror_slot=%d must be W-2=%d and outside the decode window (topk=%d)" % (mirror_lo, W - 2, topk))
    pos0 = engine._pos.clone()
    tl0 = pos0 % bs
    T = pos0 // bs
    if bool((tl0 + U >= 2 * bs).any()):
        raise ValueError("tail_len + U >= 2*block_size for a request: more than one rollover per round is not supported")
    top = int(pos0.max()) + U
    if kv_bias.shape[0] != B or kv_bias.shape[1] < top or kv_bias.shape[2] != H:
        raise ValueError("kv_bias %s must be (B=%d, >= max pos + U = %d, H=%d)" % (tuple(kv_bias.shape), B, top, H))
    if engine._k_cpu.shape[1] < top:
        raise ValueError("host window has %d rows, need %d" % (engine._k_cpu.shape[1], top))

    _masked_slot_copy(engine, mirror_lo, ts, tl0)                 # round start: mirror_lo <- rows [0, tail_len0_b)
    bi = torch.arange(B, device=pos0.device)
    for i in range(U):
        p = pos0 + i
        r = p % bs
        in_T = (p // bs) == T
        kk = key_states[:, i].to(engine._k_gpu.dtype)
        vv = value_states[:, i].to(engine._v_gpu.dtype)
        bb = kv_bias[bi, p].to(engine._kv_bias_gpu.dtype)
        for row in (ts * bs + r, torch.where(in_T, mirror_lo * bs + r, mirror_hi * bs + r)):
            engine._k_gpu[bi, row] = kk
            engine._v_gpu[bi, row] = vv
            engine._kv_bias_gpu[bi, row] = bb
        _host_write_back(engine, in_T & (r == bs - 1), T, ts)
    pos1 = pos0 + U
    tl1 = pos1 % bs
    _set_tail_state(engine, pos1, pos1 // bs, tl1)
    rolled = tl0 + U >= bs
    fill_at = bs - 1 - tl0
    rollover_at = torch.where(rolled & (fill_at + 1 < U), fill_at + 1, torch.full_like(fill_at, U))
    last_used = torch.where(rolled, torch.full_like(T, mirror_hi), torch.full_like(T, mirror_lo))
    ctr = pos0.view(1, B) + torch.arange(U + 1, dtype=pos0.dtype, device=pos0.device).view(U + 1, 1)
    return PerReqTail(U=U, pos0=pos0, tail_len_before=tl0, tail_id=T, rolled_over=rolled, fill_at=fill_at,
                      rollover_at=rollover_at, tail_len_after=tl1, mirror_lo=mirror_lo, mirror_hi=mirror_hi,
                      last_used_slot=last_used, seqused_k=(last_used * bs + tl1).to(torch.int32), ctr=ctr)


def plan_round_perreq(store: Store, l: int, sel: torch.Tensor, tail_id: torch.Tensor, rollover_at: torch.Tensor,
                      gen_round0: int, gen_now: int, stamp_now: int, was_inflight: torch.Tensor) -> RoundPlan:
    """plan_round with the live tail per request: tail_id and rollover_at
    are (B,) tensors (PerReqTail.tail_id / rollover_at), so the live tail of
    request b at position u is T_b + (u >= rollover_at_b) (the lockstep
    live-tail list, :511, per request). The lines that use them are the only
    change; every other line is plan_round's."""
    lay = store.lay
    ids, state, gen, stamp, used = store.ids[l], store.state[l], store.gen[l], store.stamp[l], store.used[l]
    if sel.dim() != 4 or tuple(sel.shape[1:3]) != (store.H, store.B) or sel.dtype != torch.int64:
        raise ValueError("sel must be int64 (U, H=%d, B=%d, K), got %s %s" % (store.H, store.B, sel.dtype, tuple(sel.shape)))
    U, H, B, K = sel.shape
    W = lay.W
    T = _bvec(tail_id, B, "tail_id")
    ra = _bvec(rollover_at, B, "rollover_at")
    if bool(((ra < 1) | (ra > U)).any()):
        raise ValueError("rollover_at must be in [1, U=%d] for every request" % U)
    if tuple(was_inflight.shape) != (H, B, W):
        raise ValueError("was_inflight must be (H, B, W)")
    if bool((state == INFLIGHT).any()):
        raise RuntimeError("plan_round_perreq: INFLIGHT slots remain; the caller must wait for the side stream and mark_arrived first")
    valid = sel >= 0
    after = torch.arange(U, device=sel.device).view(U, 1, 1, 1) >= ra.view(1, 1, B, 1)     # (U, 1, B, 1)
    Tv = T.view(1, 1, B, 1)
    live_tail = Tv + after.to(torch.int64)
    invalid = ((valid & (sel > live_tail)) | (sel < -1)).any(-1).any(0)          # (H, B)
    is_tail = valid & (sel == Tv)
    is_tail_next = valid & (sel == Tv + 1) & after

    need = sel.permute(1, 2, 0, 3).reshape(H, B, U * K)
    need_valid = (valid & ~is_tail & ~is_tail_next).permute(1, 2, 0, 3).reshape(H, B, U * K)
    need_ids = torch.where(need_valid, need, torch.full_like(need, -2))
    first = _first_occurrence(need_ids) & need_valid
    present = state == RESIDENT
    eq = (need_ids.unsqueeze(-1) == ids.unsqueeze(-2)) & present.unsqueeze(-2)   # (H, B, U*K, W)
    found = eq.any(-1) & first
    slot = eq.to(torch.int64).argmax(-1)
    slot_gen = torch.gather(gen, -1, slot)
    slot_infl = torch.gather(was_inflight, -1, slot)
    late_wait = found & slot_infl
    timely = found & ~slot_infl & (slot_gen >= int(gen_round0))
    hit_old = found & ~slot_infl & (slot_gen < int(gen_round0))
    new = first & ~found
    named = torch.zeros((H, B, W + 1), dtype=torch.bool, device=sel.device)
    named.scatter_(-1, torch.where(found, slot, torch.full_like(slot, W)), torch.ones_like(slot, dtype=torch.bool))
    named = named[..., :W]

    cand = store.evictable.view(1, 1, W) & ~named & ((state == FREE) | (state == RESIDENT))
    key = _slot_key(state, stamp, cand)
    fetch, fits, n_cand = _assign(new, need_ids, cand, key)
    n_new = new.sum(-1)
    overflow = n_new > n_cand
    bad = overflow | invalid
    newly = fetch >= 0
    evicted_unused = newly & (state == RESIDENT) & ~used

    ids2 = torch.where(newly, fetch, ids)
    state2 = torch.where(newly, torch.full_like(state, RESIDENT), state)
    store.ids[l] = ids2
    store.state[l] = state2
    store.gen[l] = torch.where(newly, torch.full_like(gen, int(gen_now)), gen)
    store.stamp[l] = torch.where(named | newly, torch.full_like(stamp, int(stamp_now)), stamp)
    store.used[l] = used | named | newly
    fetch = torch.where(bad.unsqueeze(-1), torch.full_like(fetch, -1), fetch)

    present2 = state2 == RESIDENT
    masks = []
    for u in range(U):
        eq_u = (sel[u].unsqueeze(-1) == ids2.unsqueeze(-2)) & present2.unsqueeze(-2)   # (H, B, K, W)
        found_u = eq_u.any(-1)
        slot_u = eq_u.to(torch.int64).argmax(-1)
        m = torch.where(found_u, slot_u, torch.full_like(slot_u, -1))
        m = torch.where(is_tail[u], torch.full_like(m, lay.mirror_lo), m)
        m = torch.where(is_tail_next[u], torch.full_like(m, lay.mirror_hi), m)
        m = torch.where(bad.unsqueeze(-1), torch.full_like(m, -1), m)
        masks.append(m)
    mask = torch.stack(masks, dim=2).reshape(H, B * U, K).to(torch.int32)

    store.late_fetch_blocks[l] += fits.sum(-1)
    store.late_wait_blocks[l] += late_wait.sum(-1)
    store.timely_blocks[l] += timely.sum(-1)
    store.hit_old_blocks[l] += hit_old.sum(-1)
    store.wasted_blocks[l] += evicted_unused.sum(-1)
    return RoundPlan(mask=mask, fetch=fetch, overflow=overflow, invalid=invalid, n_new=n_new,
                     n_need=first.sum(-1), n_hit_old=hit_old.sum(-1), n_timely=timely.sum(-1),
                     n_late_wait=late_wait.sum(-1), n_late_fetch=fits.sum(-1), union_size=present2.sum(-1))


STORE_FIELDS = ("ids", "state", "gen", "stamp", "used")


def store_layer_copy(store: Store, l: int) -> dict:
    """Layer l's store state and counters (clones): what isolate_bad_streams puts back."""
    d = {n: getattr(store, n)[l].clone() for n in STORE_FIELDS}
    d.update({n: getattr(store, n)[l].clone() for n in store.COUNTERS})
    return d


def isolate_bad_streams(store: Store, l: int, plan: RoundPlan, bad: torch.Tensor, before: dict,
                        layout: StoreLayout) -> RoundPlan:
    """THE PER-REQUEST OVERFLOW FALLBACK of one layer's round. ``bad`` (H, B) bool: the streams whose union overflowed
    or named an invalid id (plan.overflow | plan.invalid). Every bad stream gets its store state and counters back
    from ``before`` (store_layer_copy taken before plan_round_perreq: plan_round marks the slots it assigned RESIDENT
    before the caller knows whether the round is refused), fetches nothing, and its mask rows name only mirror_lo (a
    finite placeholder: block T's rows up to each query's causal limit, so attention stays finite; its request
    commits nothing this round). The other streams are untouched. Returns the plan with the new mask, fetch and
    union_size. Pure, per (h, b)."""
    H, B = bad.shape
    for n in STORE_FIELDS:
        t = getattr(store, n)
        t[l] = torch.where(bad.unsqueeze(-1), before[n], t[l])
    for n in store.COUNTERS:
        t = getattr(store, n)
        t[l] = torch.where(bad, before[n], t[l])
    K = plan.mask.shape[2]
    U = plan.mask.shape[1] // B
    m = plan.mask.view(H, B, U, K)
    ph = torch.full_like(m, -1)
    ph[..., 0] = layout.mirror_lo
    mask = torch.where(bad.view(H, B, 1, 1), ph, m).reshape(H, B * U, K).contiguous()
    fetch = torch.where(bad.unsqueeze(-1), torch.full_like(plan.fetch, -1), plan.fetch)
    return plan._replace(mask=mask, fetch=fetch, union_size=(store.state[l] == RESIDENT).sum(-1))


# -- accept and commit, per request --------------------------------------------
class PerReqAccept(NamedTuple):
    n_acc: torch.Tensor       # (B,) accepted draft tokens per request
    keep: torch.Tensor        # (B,) 1 + n_acc: the tokens request b commits this round
    committed: torch.Tensor   # (B, U) int64: argmax(verify[b, u]) for u < keep[b], -1 after
    next_token: torch.Tensor  # (B, 1) int64: committed[b, keep[b] - 1], fed first next round


def accept_perreq(verify_argmax: torch.Tensor, drafted: torch.Tensor) -> PerReqAccept:
    """accept_commit per request: request b commits keep_b = 1 + n_acc(b)
    tokens, argmax(verify[b, 0:keep_b]). This is the argument of
    accept_commit applied to request b alone (any c_b <= 1 + n_acc(b) is
    exact for b). No host sync."""
    n_acc = accept_lengths(verify_argmax, drafted)
    keep = 1 + n_acc
    B, U = verify_argmax.shape
    u = torch.arange(U, device=verify_argmax.device).view(1, U)
    committed = torch.where(u < keep.view(B, 1), verify_argmax, torch.full_like(verify_argmax, -1))
    next_token = verify_argmax.gather(1, (keep - 1).view(B, 1))
    return PerReqAccept(n_acc=n_acc, keep=keep, committed=committed, next_token=next_token)


class PerReqTruncate(NamedTuple):
    keep: torch.Tensor           # (B,)
    pos: torch.Tensor            # (B,) pos0 + keep
    tail_len: torch.Tensor       # (B,) pos % bs
    tail_id: torch.Tensor        # (B,) pos // bs
    undo_rollover: torch.Tensor  # (B,) bool: the round filled T_b and keep_b does not reach the fill
    rows_restored: torch.Tensor  # (B,) rows of slot topk-1 copied back from mirror_lo


def _check_truncate(engine, tail: PerReqTail, keep, mirror_lo: int) -> torch.Tensor:
    keep = _bvec(keep, _batch(engine), "keep")
    if bool(((keep < 0) | (keep > tail.U)).any()):
        raise ValueError("keep outside [0, U=%d]" % tail.U)
    if int(mirror_lo) != int(tail.mirror_lo):
        raise ValueError("mirror_lo=%d != the write's %d" % (mirror_lo, tail.mirror_lo))
    if not torch.equal(engine._pos, tail.ctr[tail.U]):
        raise ValueError("the engine is not at the end of this round's write (truncated already, or another round)")
    return keep


def _reset_counters(ctr: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """ctr (U + 1, B), keep (B,) -> ctr[keep[b], b]: the counter reset."""
    return ctr.gather(0, keep.view(1, -1)).view(-1)


def truncate_perreq(engine, tail: PerReqTail, keep: torch.Tensor, mirror_lo: int) -> PerReqTruncate:
    """truncate_tail per request: request b keeps the first keep[b] of the
    round's U tokens (0 <= keep[b] <= U), after write_tail_perreq.
    pos = ctr[keep_b, b], a counter reset; the map's tail column and
    cache_lens follow. With f_b = fill_at[b]: a request whose round filled
    T_b and whose keep_b < f_b + 1 undoes the fill. Rows [0, U - (f_b + 1))
    of slot topk-1, which the tokens of T_b + 1 overwrote, come back from
    mirror_lo (complete T_b after the write): one masked copy for the batch
    (_masked_tail_restore). The host write-back of an undone fill stays
    beyond the committed length, as in truncate_tail. Not touched: the
    store, mirror_hi, the rows of slot topk-1 at or above the tail length."""
    keep = _check_truncate(engine, tail, keep, mirror_lo)
    bs = int(engine.block_size)
    U = int(tail.U)
    f = tail.fill_at
    undo = tail.rolled_over & (keep < f + 1)
    restored = torch.where(undo, U - (f + 1), torch.zeros_like(f))
    _masked_tail_restore(engine, int(engine._tail_block_idx_on_gpu), int(mirror_lo), restored)
    pos = _reset_counters(tail.ctr, keep)
    _set_tail_state(engine, pos, pos // bs, pos % bs)
    return PerReqTruncate(keep=keep, pos=pos, tail_len=pos % bs, tail_id=pos // bs, undo_rollover=undo, rows_restored=restored)


def commit_layer_perreq(engine, tail: PerReqTail, tables, ctr: torch.Tensor, keep: torch.Tensor, mirror_lo: int) -> PerReqTruncate:
    """The per-request commit of one layer. The tables commit ctr[keep_b, :, b]
    (``tables`` has the twin's interface: commit(ctr, keep), a counter reset),
    then the tail is truncated to keep_b (truncate_perreq). ctr is the
    tables' trajectory of the verify (the twin's advance(): (U + 1, C, B),
    row 0 = pos). Row 0 must equal the tail's trajectory: the two counters
    are one counter. Every check runs before either side changes."""
    keep = _check_truncate(engine, tail, keep, mirror_lo)
    B = _batch(engine)
    if ctr.dim() != 3 or ctr.shape[0] != tail.U + 1 or ctr.shape[2] != B:
        raise ValueError("ctr must be (U + 1 = %d, C, B = %d), got %s" % (tail.U + 1, B, tuple(ctr.shape)))
    if not torch.equal(ctr[:, 0].to(device=tail.ctr.device, dtype=tail.ctr.dtype), tail.ctr):
        raise ValueError("the tables' pos trajectory is not the tail's: the two counters of the layer differ")
    tables.commit(ctr, keep)
    return truncate_perreq(engine, tail, keep, mirror_lo)


class PerReqSpecContext(SpecContext):
    """SpecContext with the round state per request (the lockstep fields
    tail0 / tail_id / rolled, per request): tail0 is a PerReqTailCounters,
    tail_id the (B,) round-start tail ids, rolled[l] a (B,) bool per layer,
    tables0[l] the round-start counters of layer l's tables ((1, C, B), the
    twin's counters()). The cache is duck-typed: ``cache.layers[l]`` has
    ``cache_engine`` (per-request: _pos) and ``tables`` (counters(),
    commit(ctr, keep)). The journals of the lockstep context are not used:
    the tables roll back by a counter reset. ``f4`` (keyword; default
    "shipped", the parity target): how the engine's per-request draft and
    verify treat a findings-F4 step (F4 PARITY in the section header;
    check_f4_mode refuses any other value)."""

    def __init__(self, *args, f4: str = F4_SHIPPED, **kwargs):
        self.f4 = check_f4_mode(f4)
        super().__init__(*args, **kwargs)

    def begin_round(self, cache, tail_id=None):
        eng0 = cache.layers[0].cache_engine
        B = _batch(eng0)
        tc = capture_tail_counters_perreq(eng0)
        tables0 = []
        for lay in cache.layers:
            if not torch.equal(lay.cache_engine._pos, tc.pos):
                raise ValueError("begin_round: the layers' tail counters differ")
            c0 = lay.tables.counters().unsqueeze(0)
            if not torch.equal(c0[0, 0].to(device=tc.pos.device, dtype=tc.pos.dtype), tc.pos):
                raise ValueError("begin_round: the tables' pos is not the tail's")
            tables0.append(c0)
        if tail_id is not None and not torch.equal(_bvec(tail_id, B, "tail_id"), tc.tail_id):
            raise ValueError("begin_round: tail_id is not pos // block_size")
        self.round_idx += 1
        self.tail0 = tc
        self.tail_id = tc.tail_id
        self.rolled = [torch.zeros((B,), dtype=torch.bool, device=tc.pos.device) for _ in range(self.num_layers)]
        self.gen_round0 = self.gen + 1
        self.tables0 = tables0
        # the requests whose union overflowed (or named an invalid id) on some layer this round
        # (cache_engine.spec_round_update_perreq, isolate_bad_streams): they commit nothing this round
        self.round_bad = torch.zeros((B,), dtype=torch.bool, device=tc.pos.device)

    def restore_before_verify(self, cache):
        """The draft's K steps undone: the tail counters (restore_tail_
        counters_perreq) and the tables (commit to the round start, keep 0)."""
        B = int(self.tail0.pos.shape[0])
        zeros = torch.zeros((B,), dtype=torch.int64, device=self.tail0.pos.device)
        for l, lay in enumerate(cache.layers):
            restore_tail_counters_perreq(lay.cache_engine, self.tail0)
            lay.tables.commit(self.tables0[l], zeros)
