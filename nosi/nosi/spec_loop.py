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
