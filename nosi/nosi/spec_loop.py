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
it drives the host loop). (Path-1 round, by reading: plan_round's INFLIGHT
guard and its pageable torch.tensor of the live-tail ids, the engine's two
overflow / invalid reads, verify_step's two guards: SIX per layer.)

EXACT VERIFY (R2 verify branch, the section at the end of this file): every
verified position is the shipped decode step, so its logits are bitwise the
shipped decode's (conditions E1-E7 there); no host sync per layer; one
decision sync per round (round mode) or one per computed position but the
last (early mode).
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
                "draft_sel_blocks", "draft_hit_blocks", "draft_inflight_blocks",
                # the EXACT verify (section at the end of this file): window loads of the verify's shipped steps
                # served D2D from the round region / left to the shipped host gathers, and the same for the
                # window restore of a rollback
                "verify_served_blocks", "verify_host_blocks", "rollback_served_blocks", "rollback_host_blocks")

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

    def sync_window_stacked(self, block_maps: torch.Tensor, stamp_now: int):
        """sync_window for EVERY layer at once (one set of ops over the stacked store instead of one set per layer:
        the exact round re-syncs after every round). block_maps: (Lyr, H, B, topk), e.g. torch.stack of the engines'
        _block_map. Same transitions as sync_window, element for element (tests_cpu: test_spec_exact.py)."""
        lay = self.lay
        if tuple(block_maps.shape) != (self.num_layers, self.H, self.B, lay.topk):
            raise ValueError("block_maps %s != (Lyr=%d, H=%d, B=%d, topk=%d)" % (
                tuple(block_maps.shape), self.num_layers, self.H, self.B, lay.topk))
        ts = lay.tail_slot
        win = block_maps[..., :ts]
        self.ids[..., :ts] = win
        self.state[..., :ts] = torch.where(win >= 0, torch.full_like(win, RESIDENT), torch.full_like(win, FREE)).to(torch.int8)
        self.stamp[..., :ts] = torch.where(win >= 0, torch.full_like(win, stamp_now), self.stamp[..., :ts])
        self.used[..., :ts] = win >= 0
        self.ids[..., ts] = -1
        rr = slice(lay.round_base, lay.mirror_lo)
        rids = self.ids[..., rr]
        dup = (rids.unsqueeze(-1) == win.unsqueeze(-2)).any(-1) & (rids >= 0)
        self.wasted_blocks += (dup & ~self.used[..., rr]).sum(-1)
        self.ids[..., rr] = torch.where(dup, torch.full_like(rids, -1), rids)
        self.state[..., rr] = torch.where(dup, torch.full_like(self.state[..., rr], FREE), self.state[..., rr])

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
    return ext[..., :W], fits, n_cand


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
# EXACT VERIFY (R2 verify branch, 2026-10-20; driver benchmarks/Efficiency/spec_exact_gate.py; CPU tests
# benchmarks/Efficiency/tests_cpu/test_spec_exact.py).
#
# WHY. The Path-1 verify above (plan_round + verify_step.verify_attention, the prefill varlen op over the union
# store, M = B*U GEMMs) is not bit-exact with the decode: its logits differ by up to 0.41 (REPRODUCE.md, G-MPV-2),
# so the committed tokens part from the shipped decode at near-ties (R2 run g2: every request, first at position 6).
#
# WHAT. Every verified position IS the shipped decode step: the round calls the model's own
# decode_inference(tok_u (B, 1), cu, pos + u, cache) once per position u = 0 .. U-1, teacher-forced with
# [x_n, d_1 .. d_K]. One "row" per (request, position): the call for position u holds exactly the B rows of the
# shipped step at that position, in the same order. Nothing in the decode text changes. The ONE device-side
# difference is WHERE some window loads come from: StoreServeHook, called by the shipped body at its existing hook
# point (cache_engine.decode_update_has_kv_bias, after diff_offload and before the two gathers: the on_diff site of
# avail_policy), copies the blocks that the draft prefetched into the round region to the slots diff assigned,
# device to device, with the SHIPPED gather kernels and the allocation itself as the source, and clears their
# load-mask entries; the shipped gathers then fetch the rest from the host.
#
# THE CONDITIONS THAT MAKE POSITION u BITWISE EQUAL TO THE SHIPPED STEP AT THE SAME (request, position):
#   E1 same code: decode_inference at (B, 1): the five GEMMs at M = B on contiguous (B, 1, K) inputs (cuBLAS picks
#      its kernel from M, N, K and the strides, so an M = B*U GEMM is NOT bit-equal per row: the paired tick
#      measured 0.72-1.4 max |dlogit| at M = 2B, REPRODUCE.md E2), the same norm / rope / nosa_linear / silu
#      launches, the stage-1 call and the captured pooling / top-k graph, the lm_head at M = B. has_buffers stays
#      True: no position of a round needs a re-warm (round_span), so the pooling grid (out_len) is the shipped one.
#   E2 same attention call: flash_attn_nosa_with_kvcache over the 64-SLOT VIEWS (size(1) = topk * bs = 4096 rows:
#      the split partition is a function of the allocated length, cache_engine.py prefill section 3c, job 2173299;
#      decode_update_has_kv_bias_verify_slots returns those views at NOSI_VERIFY_ROUND_SLOTS > 0), the window's
#      bias view, num_splits = NOSI_ATTN_SPLITS (the same knob value in both arms), the same B requests in the same
#      order (the engine is lockstep: every request is at the same position), so even the split heuristic of
#      NOSI_ATTN_SPLITS = 0 sees the same (batch, heads, length). The gate pins 4.
#   E3 same cache_seqlens, F4 included: _cache_lens = (topk-1)*bs + tail_len after the step's own write, and at a
#      block-completing step (p % 64 == 63) the shipped body resets it to (topk-1)*bs (attention skips the whole
#      tail block, the current token included: R2_ENGINE_FINDINGS F4). The verify runs that body, so it reproduces
#      F4 at exactly the shipped positions. An engine that "fixes" F4 is no longer bitwise equal.
#   E4 same window bytes in the same slot order: the window of position u is made by the shipped diff from the
#      window of position u-1; the draft never writes a window slot (pin_window: its prefetch goes to the round
#      region only); a round starts at the committed state (round mode: rollback_exact restores the map, the
#      window rows, the tail rows, the counters and the tables; early mode: nothing to undo). A store-served load
#      is byte-equal to the host gather's: the store slot was filled by the same two Triton gathers from the same
#      host rows (_k_cpu / _v_cpu) and total_cis rows, for a block < T (plan_draft forbid_ge): complete, and never
#      rewritten on the host or in total_cis after its write-back.
#   E5 same tail rows: position u writes row tail_len0 + u (the shipped write) over the draft's approximate row and
#      attends rows < tail_len0 + u + 1 only. The draft's approximate write-back of block T at its own fill is
#      rewritten by the verify's fill at the same position, or T stays the live tail (never gathered) until a
#      later exact fill rewrites it.
#   E6 same inputs: the token ids (teacher-forced, or accepted drafts: by induction the accepted prefix is the
#      shipped greedy sequence) and the position ids (pos + u, a fresh contiguous (B, 1) tensor).
#   E7 same layer tables: decode_forward advances the compress-k / cis tables per position; the draft's advances
#      are undone by the round-start LayerJournal; a rollback restores the journal taken right after the last
#      committed position.
#   PRECONDITIONS: (a) after the warm-up step every window slot 0 .. topk-2 holds a block (the selection names topk
#      distinct valid ids, L >= topk * bs; the driver checks it once, outside the clock); (b) the stage-1 score
#      width round_multiple(max_seqlen_comp, 128) does not change inside a run (at L = 16128 it changes after 272
#      tokens; a change raises in both arms alike); (c) U <= block_size (at most one tail fill per round).
#      KV_BIAS_SCALE need not be 1: the store copies the bias bytes the gather wrote, the mirror the tail rows as
#      written.
#   NOT bitwise, by design and never read: rows at or above cache_seqlens, the host rows of the live tail block
#   beyond the committed length, the round region and the mirror slots.
#
# HOST SYNCS. The exact round has none per layer or per position: the hook, the marks, the rollback and the store
# sync are device ops and host ints. Round mode: ONE decision sync per round (accept_commit's min). Early mode
# (stop_after): one per computed position except the last, at most U - 1, and no rollback.
# ===========================================================================
EXACT_MODES = ("round", "early")


class GatherOps(NamedTuple):
    """The two shipped Triton gathers (flash_cache_engine.flash_h2d_mask / flash_h2d_mask_bias), or their CPU
    twins. gather_k(gpu_data, cpu_data, load_ids, block_size); gather_v(gpu_data, cpu_data, kv_bias_buf, kv_bias,
    load_ids, block_size). Program (b, h, m) copies block load_ids[h, b, m] (rows id*bs .. id*bs+bs-1 of the
    SOURCE tensor it is handed) into slot m of the destination; ids < 0 are skipped."""
    gather_k: Any
    gather_v: Any


def pin_window(store: Store):
    """EXACT MODE: the draft's prefetch may assign round-region slots only (plan_draft's candidates are
    store.evictable): a window slot is the SHIPPED window's, owned by diff, and a prefetch into it would change
    the bytes the next exact step attends (E4). The tail slot and the mirrors are FIXED already."""
    store.evictable[:store.lay.topk] = False
    return store


def round_span(seq_length: int, out_len: int, K: int, remaining: int, block_size: int) -> int:
    """U for an exact round that starts at engine seq_length: at most K + 1, at most the remaining tokens, at most
    block_size (one tail fill per round), and NO position needs a re-warm. Position u runs at seq_length + u; the
    captured pooling grid (out_len blocks, nosa_llama.decode_inference) serves it iff
    ceil((seq_length + u + 1) / bs) <= out_len (spec_loop_pilot.needs_rewarm_len), i.e. u < out_len*bs - seq_length.
    0 = the next step itself needs a re-warm (a plain shipped step); 1 = a plain step without re-warm."""
    for name, v in (("seq_length", seq_length), ("out_len", out_len), ("K", K), ("remaining", remaining), ("block_size", block_size)):
        if not isinstance(v, int):
            raise TypeError("round_span: %s must be a host int, got %r" % (name, type(v)))
    if K < 0 or block_size < 1:
        raise ValueError("round_span: K=%d block_size=%d" % (K, block_size))
    lim = out_len * block_size - seq_length
    return max(0, min(K + 1, remaining, block_size, lim))


def fill_position(tail_len0: int, U: int, block_size: int) -> Optional[int]:
    """The position of the round whose shipped step fills the tail block (its write is row block_size-1): u_f =
    block_size - 1 - tail_len0, or None when the round ends before it."""
    if not (0 <= tail_len0 < block_size):
        raise ValueError("tail_len0=%d outside [0, %d)" % (tail_len0, block_size))
    f = block_size - 1 - tail_len0
    return f if f < U else None


def store_sources(ids_l: torch.Tensor, state_l: torch.Tensor, load_mask: torch.Tensor, lo: int, hi: int):
    """For every load diff assigned (load_mask (H, B, M) int64, block id >= 0 at the slot it goes to): the
    round-region slot in [lo, hi) that holds that block RESIDENT, else -1. Returns (src (H, B, M) int64, hit bool).
    Integer ops only, no host sync."""
    rids = ids_l[..., lo:hi]                                                        # (H, B, R)
    rres = state_l[..., lo:hi] == RESIDENT
    want = load_mask >= 0
    eq = (load_mask.unsqueeze(-1) == rids.unsqueeze(-2)) & rres.unsqueeze(-2) & want.unsqueeze(-1)   # (H, B, M, R)
    hit = eq.any(-1)
    src = int(lo) + eq.to(torch.int64).argmax(-1)
    src = torch.where(hit, src, torch.full_like(src, -1)).contiguous()
    return src, hit


def serve_from_store(engine, ops: GatherOps, store: Store, l: int, load_mask: torch.Tensor, stamp_now: int,
                     served_counter: str = "verify_served_blocks", host_counter: str = "verify_host_blocks"):
    """Serve the loads of ``load_mask`` that the round region holds: K, V and bias rows copied from the store slot
    to the slot diff assigned, by the shipped gathers with the ALLOCATION as their source (the kernels are pointer
    arithmetic over the tensors they are handed: flash_h2d_mask.py, flash_h2d_mask_bias.py; source slots are >=
    topk, destinations < topk, so a launch never reads what it writes); then those entries are cleared IN PLACE,
    so the caller's host gathers fetch only the rest. Stamps the source slots (LRU) and counts. No host sync."""
    lay = store.lay
    src, hit = store_sources(store.ids[l], store.state[l], load_mask, lay.round_base, lay.mirror_lo)
    bs = int(engine.block_size)
    ops.gather_k(engine._k_gpu, engine._k_gpu, src, bs)
    ops.gather_v(engine._v_gpu, engine._v_gpu, engine._kv_bias_gpu, engine._kv_bias_gpu, src, bs)
    load_mask.masked_fill_(hit, -1)
    H, B, _ = src.shape
    W = lay.W
    named = torch.zeros((H, B, W + 1), dtype=torch.bool, device=src.device)
    named.scatter_(-1, torch.where(hit, src, torch.full_like(src, W)), torch.ones_like(hit))
    named = named[..., :W]
    store.stamp[l] = torch.where(named, torch.full_like(store.stamp[l], int(stamp_now)), store.stamp[l])
    store.used[l] = store.used[l] | named
    getattr(store, served_counter)[l] += hit.sum(-1)
    getattr(store, host_counter)[l] += (load_mask >= 0).sum(-1)
    return hit


class StoreServeHook:
    """The store service, installed as avail_policy.POLICY for the verify's shipped steps ONLY (hook_installed).
    Duck-typed to every call site of that global: decode_inference -> begin_step(); decode_forward ->
    begin_layer(l), mask_bias(kv_bias) (returns kv_bias itself: attention gets the shipped bias view);
    decode_update_has_kv_bias -> on_diff(old map, new map, load mask, topk_idx, tail slot) between diff_offload
    and the gathers; with the transfer trace on -> pre_denial_mask(), denied() (None: nothing is denied). It never
    writes the selection, the maps, or a slot that diff did not assign a load to."""

    def __init__(self, engines: Sequence[Any], store: Store, ops: GatherOps):
        self.engines = list(engines)
        self.store = store
        self.ops = ops
        self.layer = None
        self.stamp = 0
        self.calls = 0

    def begin_step(self):
        self.layer = None

    def begin_layer(self, layer_idx: int):
        self.layer = int(layer_idx)

    def on_diff(self, block_map, new_block_map_buf, load_mask, topk_idx, tail_slot):
        if self.layer is None:
            raise RuntimeError("StoreServeHook.on_diff before begin_layer")
        eng = self.engines[self.layer]
        if load_mask is not eng._load_mask or block_map is not eng._block_map:
            raise RuntimeError("StoreServeHook: on_diff tensors are not layer %d's engine buffers" % self.layer)
        serve_from_store(eng, self.ops, self.store, self.layer, load_mask, self.stamp)
        self.calls += 1

    def mask_bias(self, kv_bias):
        return kv_bias

    def pre_denial_mask(self):
        return None

    def denied(self):
        return None


class hook_installed:
    """with hook_installed(avail_policy_module, hook): ... -- binds the module's POLICY to the hook and restores
    None on exit (also on an exception). Refuses to replace an installed availability policy."""

    def __init__(self, module, hook):
        self.module = module
        self.hook = hook

    def __enter__(self):
        if getattr(self.module, "POLICY", None) is not None:
            raise RuntimeError("an availability policy is installed (%r); the store hook refuses to replace it" % (self.module.POLICY,))
        self.module.POLICY = self.hook
        return self.hook

    def __exit__(self, *exc):
        self.module.POLICY = None
        return False


def mirror_tail(engine, mirror_lo: int):
    """Copy the tail slot's block_size rows (K, V, bias, as written) into slot mirror_lo, device to device. The
    round calls it right after the shipped step that filled the tail (fill_position) when later positions of the
    same round will overwrite rows 0.. of the tail slot with block T+1, so that a rollback to before the fill can
    put block T's rows back (rollback_exact B)."""
    bs = int(engine.block_size)
    ts = int(engine._tail_block_idx_on_gpu) * bs
    ms = int(mirror_lo) * bs
    if ms + bs > engine._k_gpu.shape[1] or ms == ts:
        raise ValueError("mirror slot %d is not a separate slot of the allocation" % mirror_lo)
    engine._k_gpu[:, ms:ms + bs].copy_(engine._k_gpu[:, ts:ts + bs])
    engine._v_gpu[:, ms:ms + bs].copy_(engine._v_gpu[:, ts:ts + bs])
    engine._kv_bias_gpu[:, ms:ms + bs].copy_(engine._kv_bias_gpu[:, ts:ts + bs])


class EngineMark(NamedTuple):
    seq_length: int
    tail_len: int


class RoundMarks:
    """ROUND MODE: right after position u's shipped step (u = 0 .. U-2), per layer: a device clone of the engine's
    block map, its two host counters, the layer tables (LayerJournal), and the cache's token counter. Buffers are
    allocated once and reused. rollback_exact(c) restores the marks of position c - 1."""

    def __init__(self, num_layers: int, max_positions: int):
        self.num_layers = int(num_layers)
        self.max_positions = int(max_positions)
        self.maps: dict = {}
        self.counters: dict = {}
        self.journals = [[LayerJournal() for _ in range(self.num_layers)] for _ in range(self.max_positions)]
        self.seen: dict = {}
        self.taken: set = set()

    def reset(self):
        self.taken = set()

    def take(self, u: int, cache):
        if not (0 <= u < self.max_positions):
            raise ValueError("RoundMarks.take: position %d outside [0, %d)" % (u, self.max_positions))
        for l, lay in enumerate(cache.layers):
            eng = lay.cache_engine
            dst = self.maps.get((u, l))
            if dst is None or dst.shape != eng._block_map.shape or dst.device != eng._block_map.device:
                dst = torch.empty_like(eng._block_map)
                self.maps[(u, l)] = dst
            dst.copy_(eng._block_map)
            self.counters[(u, l)] = EngineMark(int(eng.seq_length), int(eng._tail_block_len_on_gpu))
            self.journals[u][l].take(lay)
        self.seen[u] = int(cache._seen_tokens)
        self.taken.add(u)


def rollback_exact(cache, ops: GatherOps, store: Store, marks: RoundMarks, c: int, U: int, u_fill: Optional[int],
                   mirror_lo: int, stamp_now: int) -> dict:
    """ROUND MODE, 1 <= c < U: put every layer back at the state right after position c - 1 (= the shipped state
    after the c committed tokens), from the state after position U - 1. Per layer:
      A. every window slot (not the tail slot) whose block id changed after position c - 1 gets the target block
         back: from the round region D2D when it is there, else by the shipped host gathers (K, V from the host
         window, the bias rows from the layer's total_cis, as diff's loads). A complete block's bytes are a
         function of its id, so slots whose id did not change need nothing;
      B. a REJECTED fill (c - 1 < u_fill < U - 1): positions u_fill+1 .. U-1 wrote rows 0 .. U-2-u_fill of the
         tail slot with block T+1; those rows of block T come back from the mirror (mirror_tail);
      C. the block map (the tail rename included), seq_length, tail_len and _cache_lens = (topk-1)*bs + tail_len;
      D. the layer tables (the journal of position c - 1); then the cache's token counter.
    Untouched, never read: tail rows at or above tail_len, the host rows of the live tail beyond the committed
    length (rewritten by its next exact fill), the round region. No host sync: c, U, u_fill are host ints."""
    if not (1 <= c < U):
        raise ValueError("rollback_exact: c=%d outside [1, U=%d)" % (c, U))
    t = c - 1
    if t not in marks.taken:
        raise RuntimeError("rollback_exact: no marks for position %d (taken %s)" % (t, sorted(marks.taken)))
    rows_restored = 0
    if u_fill is not None and t < u_fill < U - 1:
        rows_restored = U - 1 - u_fill
    for l, layer in enumerate(cache.layers):
        eng = layer.cache_engine
        bs = int(eng.block_size)
        ts = int(eng._tail_block_idx_on_gpu)
        tgt = marks.maps[(t, l)]
        # A. window slots
        changed = eng._block_map != tgt
        changed[..., ts] = False
        restore = torch.where(changed, tgt, torch.full_like(tgt, -1)).contiguous()
        serve_from_store(eng, ops, store, l, restore, stamp_now,
                         served_counter="rollback_served_blocks", host_counter="rollback_host_blocks")
        ops.gather_k(eng._k_gpu, eng._k_cpu, restore, bs)
        ops.gather_v(eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, layer.total_cis, restore, bs)
        # B. tail rows of a rejected fill
        if rows_restored > 0:
            r0, m0 = ts * bs, int(mirror_lo) * bs
            eng._k_gpu[:, r0:r0 + rows_restored].copy_(eng._k_gpu[:, m0:m0 + rows_restored])
            eng._v_gpu[:, r0:r0 + rows_restored].copy_(eng._v_gpu[:, m0:m0 + rows_restored])
            eng._kv_bias_gpu[:, r0:r0 + rows_restored].copy_(eng._kv_bias_gpu[:, m0:m0 + rows_restored])
        # C. map and counters
        eng._block_map.copy_(tgt)
        mk = marks.counters[(t, l)]
        eng.seq_length = int(mk.seq_length)
        eng._tail_block_len_on_gpu = int(mk.tail_len)
        eng._cache_lens[...] = (int(eng.topk) - 1) * bs + int(mk.tail_len)
        # D. tables
        marks.journals[t][l].restore(layer)
    cache._seen_tokens = marks.seen[t]
    return dict(c=c, U=U, rows_restored=rows_restored, undo_fill=(u_fill is not None and u_fill > t))


def stop_after(argmax_u: torch.Tensor, next_input: torch.Tensor) -> bool:
    """EARLY MODE's decision after position u: True when some request's exact token differs from the drafted
    input of position u + 1 (then positions > u would verify a wrong prefix). ONE host sync (the bool)."""
    return bool((argmax_u.reshape(-1) != next_input.reshape(-1)).any())


class ExactRoundResult(NamedTuple):
    U: int                    # positions planned (K + 1)
    computed: int             # positions run (U in round mode; c in early mode)
    c: int                    # committed tokens
    logits: torch.Tensor      # (B, computed, V) the exact logits of the computed positions
    committed: torch.Tensor   # (B, c)
    next_token: torch.Tensor  # (B, 1)
    n_acc: torch.Tensor       # (B,) accepted drafts per request (early mode: censored at the stop)
    drafted: torch.Tensor     # (B, K) the draft's own argmax tokens
    u_fill: Optional[int]
    rollback: Optional[dict]
    host_syncs: int


def _no_timer(key: str, idx: int = 0):
    return None


def exact_round(model, cache, ctx: "SpecContext", hook: StoreServeHook, policy_module, marks: Optional[RoundMarks],
                ops: GatherOps, next_tok: torch.Tensor, pos: torch.Tensor, cu: torch.Tensor, U: int, mode: str,
                forced_tokens: Optional[torch.Tensor] = None, forced_c: Optional[int] = None,
                timer=None) -> ExactRoundResult:
    """ONE exact round: K = U - 1 draft steps (the loop's draft, prefetching into the round region only:
    pin_window), the draft undone (ctx.restore_before_verify), the prefetches waited for on the main stream, then
    U shipped steps (model.decode_inference, one per position, the store hook installed), the commit, and in round
    mode the rollback of the rejected positions.

    next_tok (B, 1): x_n, the last committed token (not yet in the cache); pos (B, 1) its position.
    forced_tokens (B, >= U): TEACHER FORCING (the gate): the draft's inputs after step 0 and the verify's inputs
    are forced_tokens[:, 0:U] (forced_tokens[:, 0] must be x_n) instead of the draft's own tokens.
    forced_c (round mode only): commit this many positions instead of 1 + min n_acc (the gate's rollback schedule;
    exact only when the forced inputs are the shipped tokens). timer(key, idx): optional event hook
    (draft_begin/draft_end j, verify_begin/verify_end u, rollback_begin/rollback_end 0).
    Host syncs: round mode 1 (the min of n_acc; 0 with forced_c), early mode <= U - 1."""
    if mode not in EXACT_MODES:
        raise ValueError("mode %r not in %s" % (mode, EXACT_MODES))
    if not (2 <= U):
        raise ValueError("exact_round needs U >= 2 (U=%d): a 1-position round is a plain shipped step" % U)
    if mode == "round" and (marks is None or marks.max_positions < U - 1):
        raise ValueError("round mode needs RoundMarks for U - 1 = %d positions" % (U - 1))
    if forced_c is not None and (mode != "round" or not (1 <= forced_c <= U)):
        raise ValueError("forced_c=%r needs round mode and 1 <= forced_c <= U=%d" % (forced_c, U))
    if not getattr(model, "has_buffers", False):
        raise RuntimeError("exact_round needs the warm-up step's buffers and graph (a re-warm is a plain step)")
    timer = timer or _no_timer
    K = U - 1
    eng0 = cache.layers[0].cache_engine
    bs = int(eng0.block_size)
    if U > bs:
        raise ValueError("U=%d > block_size=%d: two tail fills in one round" % (U, bs))
    store = ctx.store
    syncs0 = ctx.host_syncs
    tail_id = int(eng0.seq_length) // bs
    ctx.begin_round(cache, tail_id)
    # DRAFT: K approximate steps; their only products are the drafted tokens and the side-stream prefetch
    drafted = []
    tok, p = next_tok, pos
    for j in range(K):
        ctx.next_tick()
        store.mark_arrived(ctx.queue.done_gen())
        timer("draft_begin", j)
        lg = model.draft_inference(tok, cu, p, cache, ctx)
        timer("draft_end", j)
        d = lg[:, -1, :].argmax(-1, keepdim=True)
        drafted.append(d)
        tok = d if forced_tokens is None else forced_tokens[:, j + 1:j + 2].contiguous()
        p = p + 1
    ctx.restore_before_verify(cache)
    # arrivals: whatever the host saw complete, then the main stream WAITS for the rest (no host sync)
    store.mark_arrived(ctx.queue.done_gen())
    if ctx.queue.last_event is not None:
        ctx.current_stream().wait_event(ctx.queue.last_event)
    ctx.queue.drain()
    store.mark_arrived(ctx.gen)
    hook.stamp = ctx.next_tick()
    # VERIFY: U shipped steps
    drafted_t = torch.cat(drafted, dim=1)
    if forced_tokens is None:
        toks = torch.cat([next_tok, drafted_t], dim=1)
    else:
        toks = forced_tokens[:, :U]
    tail_len0 = int(eng0._tail_block_len_on_gpu)
    u_fill = fill_position(tail_len0, U, bs)
    mirror_lo = store.lay.mirror_lo
    if marks is not None:
        marks.reset()
    rows = []
    with hook_installed(policy_module, hook):
        for u in range(U):
            timer("verify_begin", u)
            lg = model.decode_inference(toks[:, u:u + 1].contiguous(), cu, (pos + u).contiguous(), cache)
            timer("verify_end", u)
            rows.append(lg[:, -1, :])
            if u == U - 1:
                break
            if mode == "round":
                if u == u_fill:
                    for lay in cache.layers:
                        mirror_tail(lay.cache_engine, mirror_lo)
                marks.take(u, cache)
            else:
                ctx.host_syncs += 1
                if stop_after(rows[-1].argmax(-1), toks[:, u + 1:u + 2]):
                    break
    computed = len(rows)
    logits = torch.stack(rows, dim=1)                                          # (B, computed, V)
    am = logits.argmax(-1)                                                     # (B, computed)
    rb = None
    if mode == "round":
        if forced_c is None:
            acc = accept_commit(am, toks[:, 1:U])                              # the ONE decision sync
            ctx.host_syncs += 1
            c, n_acc = acc.c, acc.n_acc
        else:
            c, n_acc = int(forced_c), accept_lengths(am, toks[:, 1:U])
        if c < U:
            timer("rollback_begin", 0)
            rb = rollback_exact(cache, ops, store, marks, c, U, u_fill, mirror_lo, hook.stamp)
            timer("rollback_end", 0)
    else:
        c = computed
        n_acc = accept_lengths(am, toks[:, 1:computed])
    committed = am[:, :c]
    stamp = ctx.next_tick()
    store.sync_window_stacked(torch.stack([lay.cache_engine._block_map for lay in cache.layers], dim=0), stamp)
    return ExactRoundResult(U=U, computed=computed, c=c, logits=logits, committed=committed,
                            next_token=committed[:, c - 1:c], n_acc=n_acc, drafted=drafted_t, u_fill=u_fill,
                            rollback=rb, host_syncs=ctx.host_syncs - syncs0)


def plain_step(model, cache, ctx: "SpecContext", tok: torch.Tensor, pos: torch.Tensor, cu: torch.Tensor,
               rewarm: bool) -> torch.Tensor:
    """One exact shipped step outside a round (a re-warm at a 64-token boundary, or a 1-position span), with no
    hook. Nothing is in flight: every round drains the queue at verify start. Re-syncs the store's window."""
    if ctx.queue.pending:
        raise RuntimeError("plain_step with %d prefetches in flight" % len(ctx.queue.pending))
    if rewarm:
        model.has_buffers = False
    lg = model.decode_inference(tok, cu, pos, cache)
    stamp = ctx.next_tick()
    ctx.store.sync_window_stacked(torch.stack([lay.cache_engine._block_map for lay in cache.layers], dim=0), stamp)
    return lg


EXACT_ARMS = ("tf", "round", "early")


def tf_commit(r: int, U: int, u_fill: Optional[int]) -> int:
    """The TEACHER-FORCED arm's commit schedule: c = 1 + r mod U, except that a round whose fill position u_fill
    is >= 1 commits c = u_fill, so that EVERY such fill is rolled back once (positions 0 .. u_fill-1 kept; the next
    round starts at the fill, u_fill = 0, which is always committed). Covers the rejected fill (with the tail rows
    restored when positions follow it) and the committed fill."""
    if U < 2:
        raise ValueError("tf_commit: U=%d" % U)
    if u_fill is not None and u_fill >= 1:
        return int(u_fill)
    return 1 + (int(r) % int(U))


def _no_logits(row, n, u, kind):
    return None


def exact_loop(model, cache, ctx: "SpecContext", hook: StoreServeHook, policy_module, marks: Optional[RoundMarks],
               ops: GatherOps, t0: torch.Tensor, pos0: torch.Tensor, cu: torch.Tensor, K: int, total: int, arm: str,
               ref_seq: Optional[torch.Tensor] = None, on_logits=None, timer_factory=None, after_warmup=None) -> dict:
    """The exact loop until ``total`` tokens are committed (the GPU gate's driver and the CPU tests run this same
    function). Step 0 is the shipped warm-up step (t0 -> t1). Then per iteration: U = round_span(...); U = 0 a
    re-warm plain step, U = 1 a plain step, else exact_round (arm 'tf' and 'round': round mode; 'early': early
    mode). arm 'tf' needs ref_seq (B, >= total + 1): every round's inputs are ref_seq[:, n : n+U] and its commit
    is tf_commit's schedule. on_logits(row (B, V), n, u, kind) gets the logits of every step whose INPUT is token
    index n (its argmax is token n + 1): the plain steps (kind 'plain', u = 0) and, of a round (kind 'round',
    u = its position), every computed position in arm 'tf' (their inputs are the reference tokens) and the
    committed positions otherwise. after_warmup() runs once after step 0 (the driver's precondition check and
    its clock start).
    timer_factory() -> a timer for one round or plain step (kept in the record). Returns committed (list of
    (B, k) tensors: t0, t1, then each commit), the records, and n."""
    if arm not in EXACT_ARMS:
        raise ValueError("arm %r not in %s" % (arm, EXACT_ARMS))
    if arm == "tf" and (ref_seq is None or ref_seq.shape[1] < total + 1):
        raise ValueError("arm tf needs ref_seq with >= total + 1 = %d tokens" % (total + 1))
    mode = "early" if arm == "early" else "round"
    on_logits = on_logits or _no_logits
    eng0 = cache.layers[0].cache_engine
    bs = int(eng0.block_size)
    pos = pos0.clone()
    lg = plain_step(model, cache, ctx, t0, pos, cu, rewarm=True)
    on_logits(lg[:, -1, :], 0, 0, "plain")
    next_tok = lg[:, -1, :].argmax(-1, keepdim=True)
    pos = pos + 1
    committed = [t0.clone(), next_tok.clone()]
    n = 1
    if after_warmup is not None:
        after_warmup()
    rounds, r = [], 0
    while n < total:
        out_len = int(model.pooling_buf_all.shape[-1])
        U = round_span(int(eng0.seq_length), out_len, int(K), int(total - n), bs)
        if arm == "tf":
            next_tok = ref_seq[:, n:n + 1].contiguous()
        tm = timer_factory() if timer_factory is not None else None
        if U <= 1:
            if tm is not None:
                tm("plain_begin", 0)
            lg = plain_step(model, cache, ctx, next_tok, pos, cu, rewarm=(U == 0))
            if tm is not None:
                tm("plain_end", 0)
            on_logits(lg[:, -1, :], n, 0, "plain")
            next_tok = lg[:, -1, :].argmax(-1, keepdim=True)
            committed.append(next_tok)
            rounds.append(dict(kind="plain", rewarm=(U == 0), n=n, c=1, timer=tm))
            pos = pos + 1
            n += 1
            continue
        forced, fc = None, None
        if arm == "tf":
            forced = ref_seq[:, n:n + U].contiguous()
            fc = tf_commit(r, U, fill_position(int(eng0._tail_block_len_on_gpu), U, bs))
        res = exact_round(model, cache, ctx, hook, policy_module, marks, ops, next_tok, pos, cu, U, mode,
                          forced_tokens=forced, forced_c=fc, timer=tm)
        upto = res.computed if arm == "tf" else res.c
        for u in range(upto):
            on_logits(res.logits[:, u], n + u, u, "round")
        rounds.append(dict(kind="round", n=n, U=U, computed=res.computed, c=res.c, u_fill=res.u_fill,
                           rollback=res.rollback, host_syncs=res.host_syncs, n_acc=res.n_acc, timer=tm))
        committed.append(res.committed)
        next_tok = res.next_token
        pos = pos + res.c
        n += res.c
        r += 1
    return dict(committed=committed, rounds=rounds, n=n, mode=mode)


def wire_bytes_exact(counters: dict, bytes_per_block: int = BYTES_WIRE_PER_BLOCK) -> dict:
    """Exact mode: prefetch (side stream) + the verify's host loads + the rollback's host loads; served = loads
    the round region served D2D (no wire)."""
    pre = int(counters["prefetch_blocks"]) * bytes_per_block
    vh = int(counters["verify_host_blocks"]) * bytes_per_block
    rh = int(counters["rollback_host_blocks"]) * bytes_per_block
    return dict(prefetch_bytes=pre, verify_host_bytes=vh, rollback_host_bytes=rh, wire_bytes=pre + vh + rh,
                verify_served_bytes=int(counters["verify_served_blocks"]) * bytes_per_block,
                rollback_served_bytes=int(counters["rollback_served_blocks"]) * bytes_per_block)


def served_fraction(counters: dict) -> Optional[float]:
    """Of the window loads the verify's shipped steps needed, the fraction the round region served D2D."""
    s, h = int(counters["verify_served_blocks"]), int(counters["verify_host_blocks"])
    return None if s + h == 0 else s / (s + h)
