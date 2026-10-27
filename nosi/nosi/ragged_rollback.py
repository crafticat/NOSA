"""Task 1.5 of KS/analysis/plan/PLAN_speckv_on_cached_nosa.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ACCEPT, CORRECT
AND ROLL BACK, PER REQUEST, inside the cached-NOSA engine (paper section 3.4), on top of the per-request engine of
Task 1.3 (nosi/nosi/ragged_engine.py). The VICTIM POOL is part of the state that rolls back. OPT-IN: no module imports
this one; a caller makes a Rollback (benchmarks/Efficiency/r2_base_cell.py --rollback). ragged_engine.py,
cache_engine.py, nosa_llama.py and every kernel are the bytes of the Task 1.3 tree. NO NEW KERNEL: torch ops and the
shipped kernels as they are (flash_pool_swap, flash_h2d_from_mask, flash_h2d_from_mask_bias, r2_writeback).

THE GROUP
  rb = Rollback(rs, K)            attaches the hooks to the RaggedState rs (below). Steps outside a group are the
                                  Task 1.3 step, bit for bit (the hooks pass through).
  rb.begin()                      opens a group.
  ragged_engine.decode_step(...)  up to K times: TENTATIVE steps. The step itself does not change; the hooks record
                                  what it changes (the journal) and defer the host write-back of a filled block.
  rb.commit(n_acc)                request b keeps its first n_acc[b] tentative steps and loses the others
                                  (n_acc: a (B,) device tensor, clamped to [0, steps run]). NO HOST READ.
  rb.reject_all()                 commit(0) for every request; the LRU clock also goes back (below).
accept_perreq(verify_argmax, drafted) gives n_acc from a greedy verify (ported, PR spec_loop.py:1513-1524).

THE JOURNAL (a port of EQ LayerJournal / truncate_tail, nosi-eqmem dde91d4 spec_loop.py:588-765, and of PR
truncate_perreq / commit_layer_perreq, nosi-r2-perreq e58011c spec_loop.py:1527-1589: per request and per step instead
of lockstep). Before tentative step u writes layer l, the hook records what that step can change. commit() undoes the
steps in REVERSE order, u = n_run - 1 .. 0, and each undo touches only the requests that reject step u (n_acc[b] <= u):
every undo op is a torch.where over the request axis, or a masked kernel. After the loop, request b holds its state from
before step n_acc[b], which is the state of n_acc[b] steps.

EVERY FIELD, per layer (the shipped names; cache_engine.py at ded9d41):
  field                                what a step changes                 journal (before step u)    undo
  twin pos, n_comp, n_ccis             +1 per event (step_unchecked)       (3, B) copy                where
  twin comp_k, ring, total_cis, ccis   one row per request at the          old rows + their indices   where, same index
                                       counters' index (r2_twin_fused
                                       writes the same rows)
  _k_gpu/_v_gpu/_kv_bias_gpu tail row  write_tail (DecodeBody or the       old rows + row index       where
  63 * 64 + pos % 64                   FusedBody kernel)
  window and pool SLOTS (64 rows       flash_pool_swap (SWAP, MOVE_IN,     the plan (target, action)  see SLOTS
  each)                                MOVE_OUT) and the host gathers      + the OLD DATA of every
                                                                           slot the step overwrites
  _block_map, _cache_lens              select, tail id, lengths            whole copy                 where
  _new_block_map_buf, _load_mask       diff, select (per-step scratch)     whole copy                 where
  _pool_map, _pool_age                 pool_update                         whole copy                 where
  _pool_target, _pool_action           pool_update (the per-step plan)     whole copy                 where
  _pool_stamp_base (host int)          += topk per step (the LRU clock)    host value at begin()      CLOCK
  host rows _k_cpu / _v_cpu            the tail write-back of a fill       -                          DEFERRED
  RaggedState.pos_host (host mirror)   += active_host per step             begin value, activity      commit(n_acc_host)
NOT RESTORED, with the reason: the sticky refusal flags (twin overflow, RaggedState err: a set flag voids the run,
ragged_engine.errors); the per-step scratch of RaggedState (pos_before, n_used, M_b, score_buf, the pooling / top-k
buffers, the captured graphs) and FusedBody's write-back mask (rewritten by every step before any read); st.steps (a
count of the steps that ran).

SLOTS. A step overwrites a slot in three ways (pool_update_kernel.cu:223-254, flash_pool_swap.py:131-176): SWAP
exchanges window slot m and pool slot q (no data is lost); MOVE_OUT copies the displaced block Y into pool slot q (the
LRU victim there is lost), then the gather fills slot m; MOVE_IN (an empty window slot m) copies pool slot q into m (the
old bytes of m are lost); a load without a pool slot (pool overflow, or an empty slot m) lets the gather overwrite m. The
pool hook runs after pool_update and before flash_pool_swap: it copies the plan and saves the OLD DATA of each lost
slot (pool slots of MOVE_OUT, window slots of MOVE_IN, window slots loaded with no action) into a flat list of at most
N = capacity slots per layer and step (index_select; the list is built with cumsum + scatter, no host read). The undo
of step u: (1) flash_pool_swap with the inverse plan, masked to the rejected requests: SWAP -> SWAP, MOVE_OUT ->
MOVE_IN (the parked block Y goes back into window slot m and overwrites the gathered block), MOVE_IN -> NONE; (2) the
saved data goes back (index_copy_); (3) a lost slot past N that held a block (its id from the journal's _block_map /
_pool_map) comes back from the HOST through the shipped gathers: exact, because every non-tail block on the GPU is a
copy of its host rows and of its total_cis rows. A past-N slot that held no block keeps other bytes: no reader exists
(attention reads mapped window slots only, a pool slot is read only when _pool_map names it), and the counter
"overflow" counts these slots. Without a pool, the gather hook saves the loaded window slots before the gather.

CLOCK. _pool_stamp_base is a per-layer step clock (pool_update_kernel.cu:37-43: stamps stamp_base + m, compared inside
one (head, request) row). commit() does NOT move it back: every stamp that a request keeps is below it, and a reference
that runs the kept steps with the other requests inactive moves it by the same amount (topk per step). reject_all()
sets it back to its begin() value: then no stamp of the group stays anywhere, and the state is the snapshot, bit for bit.

DEFERRED WRITE-BACK (Review Focus: test_reject_at_block_edge_keeps_host_rows). A tentative step that fills its tail
block (pos % 64 == 63) does not write the host rows. The write-back hook copies slot 63 into a per-layer GPU STAGING
block (B, 64, H, D) and records the step, the position and the block id. Rejected tokens therefore never reach the host
rows. commit() writes the staged blocks of the KEPT fills to the host with the shipped device-driven write-back kernel
(r2_writeback.masked_writeback: device mask and positions, no host read) and drops the others. REDIRECT: a later
tentative step of the same group loads that block (it is in the forced local window, so the next step selects it);
its host rows are not written yet, so the gather hooks copy K and V of that block from the staging block after the
shipped gathers (one more masked gather each, block id 0 of the staging block); the bias is right already (the gathers
read it from twin.total_cis on the device).
ROWS OF REJECTED STEPS IN A PART-FILLED TAIL (SCORING_dr1.md section 7: in the Task 1.1 draft flow nothing rolls the
draft rows back, so the next shipped EDGE step writes them to the host with the whole block). Here the tail-row journal
puts back the rows of every rejected step, and pos goes back. The kept steps then write those positions again before the
block can fill, so the write-back at an edge (deferred in a group, shipped outside a group) copies only rows of kept
steps. In a draft + verify flow the DRAFT steps are always rejected (commit(0) or reject_all) and the exact steps (the
verify) write the rows that are kept: the draft K / V are never kept, even for an accepted token.

RELEASE EVENTS (Task 1.2's rule, "a writer waits the last reader"). commit() runs the device work of each layer through
an executor (stream.run) and records one event per layer after it (stream.record): after it, every earlier reader of
that layer's discarded storage (the tentative steps' attention, the undo itself) is done. Storage of discarded tokens
(the journal of the group, the staging blocks, the tail rows and slots that the undo restored) is reusable only after
that event. The next step waits the events of every layer before its first op that touches the engine: the hook is
the step's first GEMM (rs.ops.linear), because ragged_engine reads twin.pos into pos_before before the twin step. The
wait is on the device (wait_event; no host sync). Another writer (a background prefetch into pool slots, Task 1.2) waits
release_event(l). With the default executor everything is on one stream, so the waits are already satisfied; they
matter when commit()'s work runs on another stream.

HOST SYNCS: none in the hooks and none in commit() / reject_all() (benchmarks/Efficiency/tests_cpu/test_rollback.py
scans the source of COMMIT_PATH and traps tensor-to-host reads at run time). stats() reads the counters (one sync; for
tests and the cell).

COST (INFERRED, to measure in cell rb1): per tentative step and layer about 45 small ops more (the copies, the pool hook
and its 3 index_select, the redirect gathers, the staging copy); per commit and layer, per step run, about 30 small ops,
one flash_pool_swap, two index_copy_ per tensor and the two fallback gathers. The ops have fixed shapes and no host read,
so a captured commit (Task 1.6) is possible.

NEGATIVE CONTROLS (tests only; neg): no_defer (the write-back goes to the host during the group), no_redirect (the
staged block is not copied after the gathers), no_slot_restore, no_host_fallback, no_pool_restore, no_meta_restore,
no_tail_restore, no_twin_restore (each removes one part of the undo), no_rollback (commit undoes nothing: the dr1
flow), skip_release_wait (the next step does not wait the release event), host_sync (commit reads n_acc on the host
once).

PHASE 2 ADDITIONS (the integration tree nosi-r2-spec; the speculative round runner is the only caller of the new parts).
Without a call of truncate / fast_begin / fast_end a group behaves as above, bit for bit.
  SPLIT JOURNAL. The pre-step journal has two parts. The TWIN part (counters, the four twin rows, the tail row) is
      taken in the twin hook, as before. The ENGINE part (_block_map, _new_block_map_buf, _load_mask, _cache_lens and
      the four pool tables) is taken in a hook on the body's decode_fixed (rs.body.decode_fixed), just before that
      step writes the engine, and the hook sets the step index that the pool, gather and write-back hooks use. In the
      ragged step nothing writes these tables between the twin step and decode_fixed (the stage-1 / pooling / top-k
      work writes RaggedState buffers only), so the journal holds the same values as before. The split lets a joint
      pass (verify_rows.py) advance the twin R times first and then run R residencies, each with its own journal.
  JOINT PASS, FAST MODE (verify_rows FAST has one residency for R rows and no decode_fixed): verify_rows calls
      fast_begin(l, eng, R) before the residency (the engine journal of the first row) and fast_end(l, eng, p0, full,
      R) after the pass. fast_end writes the engine journal of rows 1 .. R - 1 as "the state after r rows": the maps
      and the pool after the residency, the tail id (p0 + r) // 64 and the attention length of position p0 + r - 1;
      their slot journals are empty (the residency stays: it is a valid cache state for any kept prefix); and the
      deferred fill of a request that fills in the pass gets its own row, u0 + 63 - p0 % 64.
  TRUNCATE. truncate(n_keep) inside an open group: request b with n_keep[b] < the steps run loses its steps
      n_keep[b] .. now (the undo of commit, these requests only), and the group stays open for every request. A kept
      fill of a truncated request goes to the host rows now; its staged block is dropped either way. The undone steps
      are marked (_undone): a later commit / truncate never undoes them again. Use: the joint pass of a speculative
      round keeps the draft row of the requests that accepted every token in the group (it is the first draft step of
      their next round), and the requests that rejected a token go back to their accepted prefix at once (they draft
      again from the correction in the same group).
  Every group's commit / truncate checks that each layer journaled the engine of every step (_ures == _u), so a
  joint pass without these hooks is refused, not silently wrong.
  Negative control (tests only): no_truncate_mark (commit undoes the truncated steps a second time: the inverse pool
  moves run twice).
"""
from __future__ import annotations

import functools

import torch

from .r2_positional_twin import RING

NONE_, SWAP, MOVE_IN, MOVE_OUT = 0, 1, 2, 3            # pool_update_kernel.cu:79-82, flash_pool_swap.py:83-86
UNDO_ACTION = (NONE_, SWAP, NONE_, MOVE_IN)            # the inverse data move of each action (SLOTS)
META3 = ("_block_map", "_new_block_map_buf", "_load_mask")
META4 = META3 + ("_cache_lens",)
POOL4 = ("_pool_map", "_pool_age", "_pool_target", "_pool_action")
TWIN_CTR = ("pos", "n_comp", "n_ccis")
# destroyed: lost slots; saved: those in the journal; overflow: lost slots past N that HELD A BLOCK (the host fallback
# reloads them); overflow_dead: lost slots past N that held no block (their bytes are not restored; no reader); the first
# decode step after a prefill has 63 empty window slots per (head, request); redirected: loads served by a staging block
COUNTERS = ("destroyed", "saved", "overflow", "overflow_dead", "redirected")
NEG_KINDS = (None, "no_defer", "no_redirect", "no_slot_restore", "no_host_fallback", "no_pool_restore",
             "no_meta_restore", "no_tail_restore", "no_twin_restore", "no_rollback", "skip_release_wait",
             "host_sync", "no_truncate_mark")
# the functions of the group's path: no host sync allowed (tests_cpu/test_rollback.py, test_no_host_sync_in_commit_path)
COMMIT_PATH = ("commit", "reject_all", "_mirror", "_commit_layer", "_undo_step", "_undo_slots", "_restore_slots",
               "_fallback", "_redirect_ids", "_redirect", "_writeback_deferred", "_where_rows", "_wait_release_all",
               "_linear_hook", "_twin_hook", "_pool_update_hook", "_gather_k_hook", "_gather_v_hook",
               "_writeback_hook", "_save_slots", "_slot_rows",
               "_decode_fixed_hook", "_journal_engine", "truncate", "_truncate_layer", "fast_begin", "fast_end")


class RollbackError(RuntimeError):
    pass


# ------------------------------------------------------------------ acceptance (ported)
def accept_lengths(verify_argmax: torch.Tensor, drafted: torch.Tensor) -> torch.Tensor:
    """EQ spec_loop.py:588-599: (B, U) verify argmaxes and (B, K) drafted tokens, U >= K + 1 -> (B,) n_acc, the longest
    prefix with verify_argmax[b, i] == drafted[b, i]. Device ops only."""
    if verify_argmax.dim() != 2 or drafted.dim() != 2 or verify_argmax.shape[0] != drafted.shape[0]:
        raise ValueError("shapes: verify_argmax %s drafted %s" % (tuple(verify_argmax.shape), tuple(drafted.shape)))
    K = drafted.shape[1]
    if verify_argmax.shape[1] < K + 1:
        raise ValueError("U=%d must be >= K+1=%d" % (verify_argmax.shape[1], K + 1))
    if K == 0:
        return torch.zeros((drafted.shape[0],), dtype=torch.int64, device=verify_argmax.device)
    match = (verify_argmax[:, :K] == drafted).to(torch.int64)
    return torch.cumprod(match, dim=1).sum(1)


def accept_perreq(verify_argmax: torch.Tensor, drafted: torch.Tensor):
    """PR spec_loop.py:1513-1524: request b commits keep_b = 1 + n_acc(b) tokens, argmax(verify[b, 0:keep_b]): its
    accepted drafts and the verify's own next token (the correction; after a full accept, the bonus of the PR rule).
    Returns (n_acc (B,), keep (B,), committed (B, U) with -1 past keep, next_token (B, 1)). No host sync. The rollback
    takes n_acc: the tentative steps that request b keeps."""
    n_acc = accept_lengths(verify_argmax, drafted)
    keep = 1 + n_acc
    B, U = verify_argmax.shape
    u = torch.arange(U, device=verify_argmax.device).view(1, U)
    committed = torch.where(u < keep.view(B, 1), verify_argmax, torch.full_like(verify_argmax, -1))
    next_token = verify_argmax.gather(1, (keep - 1).view(B, 1))
    return n_acc, keep, committed, next_token


# ------------------------------------------------------------------ the executor of commit()
class _Done:
    """A completed event (CPU)."""
    done = True

    def query(self) -> bool:
        return True

    def synchronize(self, by=None) -> None:
        return None


class ImmediateStream:
    """The default executor of commit(): the device work runs at once on the current stream. record(): a CUDA event on
    the current stream (a completed _Done on the CPU). wait(ev): the current stream waits ev ON THE DEVICE (no host
    sync); on the CPU, ev.synchronize() (a test executor's pending event then runs its work)."""

    def run(self, fn) -> None:
        fn()

    def record(self, device):
        if device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(device))
            return ev
        return _Done()

    def wait(self, ev, device) -> None:
        if device.type == "cuda":
            torch.cuda.current_stream(device).wait_event(ev)
        else:
            ev.synchronize()


# ------------------------------------------------------------------ the rollback
class Rollback:
    """Per-request accept / roll back of tentative steps on one RaggedState (module docstring).
      rs        ragged_engine.RaggedState (attach or build), pool on or off
      K         the most tentative steps in one group (1 .. 64: one tail fill per request and group)
      capacity  N, the saved slots per layer and step (SLOTS; a lost slot past N comes back from the host)
      stream    the executor of commit()'s device work (ImmediateStream; tests pass a deferred one)
      neg       a negative control (NEG_KINDS), tests only"""

    def __init__(self, rs, K: int, capacity: int = 256, stream=None, neg=None):
        if neg not in NEG_KINDS:
            raise RollbackError("unknown negative control %r" % (neg,))
        if not 1 <= int(K) <= 64:
            raise RollbackError("K must be in 1 .. 64 (one tail fill per request and group: one staging block)")
        if int(capacity) < 1:
            raise RollbackError("capacity must be >= 1")
        if getattr(rs, "rollback", None) is not None:
            raise RollbackError("this RaggedState has a Rollback already")
        self.rs, self.K, self.N, self.neg = rs, int(K), int(capacity), neg
        self.stream = stream if stream is not None else ImmediateStream()
        eng0 = rs.cache.layers[0].cache_engine
        self.L, self.B = int(rs.L), int(rs.B)
        self.dev = eng0._k_gpu.device
        self.pool = bool(rs.pool)
        self.bs, self.M = int(eng0.block_size), int(eng0.topk)
        self.P = int(eng0.pool_blocks) if self.pool else 0
        self.S = self.M + self.P                        # slots per (head, request): the window, then the pool
        self.H, self.D = int(eng0.head_num), int(eng0.head_dim)
        if int(eng0._tail_block_idx_on_gpu) != self.M - 1:
            raise RollbackError("the tail slot must be topk - 1")
        self.tail0 = (self.M - 1) * self.bs
        self.R = self.S * self.bs                       # GPU rows per request
        for l, lay in enumerate(rs.cache.layers):
            e = lay.cache_engine
            for name, shape in (("_k_gpu", (self.B, self.R, self.H, self.D)), ("_v_gpu", (self.B, self.R, self.H, self.D)),
                                ("_kv_bias_gpu", (self.B, self.R, self.H))):
                t = getattr(e, name)
                if tuple(t.shape) != shape or not t.is_contiguous():
                    raise RollbackError("layer %d: %s is %s (contiguous %s); the rollback needs a contiguous %s"
                                        % (l, name, tuple(t.shape), t.is_contiguous(), shape))
        dev, B, i64 = self.dev, self.B, torch.int64
        self.bshape = torch.Size([B])
        self.all_b = torch.arange(B, dtype=i64, device=dev)
        self.rows = torch.arange(self.bs, dtype=i64, device=dev)
        self.flat = torch.arange(self.H * B * self.S, dtype=i64, device=dev)
        self.dummy = torch.tensor(self.M - 1, dtype=i64, device=dev)    # (head 0, request 0, the tail slot): never saved
        self.zero = torch.tensor(0, dtype=i64, device=dev)
        self.minus1 = torch.tensor(-1, dtype=i64, device=dev)
        self.zero8 = torch.tensor(0, dtype=torch.int8, device=dev)
        self.ustep = [torch.tensor(u, dtype=i64, device=dev) for u in range(self.K)]
        self.undo_lut = torch.tensor(UNDO_ACTION, dtype=torch.int8, device=dev)
        self.zeros_b = torch.zeros(B, dtype=i64, device=dev)
        self.zeros_b_host = torch.zeros(B, dtype=i64)
        self.count = {n: torch.zeros(self.L, dtype=i64, device=dev) for n in COUNTERS}
        self.J = [[self._alloc_step(l) for _ in range(self.K)] for l in range(self.L)]
        self.T = [self._alloc_defer(l) for l in range(self.L)]
        self.tentative = False
        self.groups = 0
        self.truncations = 0
        self._u = [0] * self.L
        self._ures = [0] * self.L                       # SPLIT JOURNAL: the steps whose engine part is journaled
        self._undone = torch.zeros((self.K, B), dtype=torch.bool, device=dev)    # TRUNCATE: undone (step, request)
        self._undone_host = torch.zeros((self.K, B), dtype=torch.bool)
        self.kbig = torch.tensor(self.K, dtype=i64, device=dev)                  # "no staged fill" (T["u"])
        self._cur = [0] * self.L
        self._stamp0 = [0] * self.L
        self._release = [None] * self.L
        self._need_wait = False
        self._act_host = []
        self._pos_host0 = None
        self._lay = {}
        self._orig = None
        self.attach()

    # -------------------------------------------------------------- allocation and hooks
    def _alloc_step(self, l: int) -> dict:
        """The journal of one layer and one step (preallocated once)."""
        eng, tw = self.rs.cache.layers[l].cache_engine, self.rs.st.twins[l]
        B, H, D, N, dev = self.B, self.H, self.D, self.N, self.dev
        j = {n: torch.empty_like(getattr(eng, n)) for n in META4 + (POOL4 if self.pool else ())}
        if self.pool:
            j["plan_target"] = torch.empty_like(eng._pool_target)
            j["plan_action"] = torch.empty_like(eng._pool_action)
        j["ctr"] = torch.empty((3, B), dtype=torch.int64, device=dev)
        j["idx"] = torch.empty((4, B), dtype=torch.int64, device=dev)
        j["comp_k"] = torch.empty((B, H, D), dtype=tw.comp_k.dtype, device=dev)
        j["ring"] = torch.empty((B, H, D), dtype=tw.ring.dtype, device=dev)
        j["total_cis"] = torch.empty((B, H), dtype=tw.total_cis.dtype, device=dev)
        j["ccis"] = torch.empty((H, B), dtype=tw.ccis.dtype, device=dev)
        j["trow"] = torch.empty((B,), dtype=torch.int64, device=dev)
        j["tk"] = torch.empty((B, H, D), dtype=eng._k_gpu.dtype, device=dev)
        j["tv"] = torch.empty((B, H, D), dtype=eng._v_gpu.dtype, device=dev)
        j["tb"] = torch.empty((B, H), dtype=eng._kv_bias_gpu.dtype, device=dev)
        j["ent"] = torch.full((N + 1,), -1, dtype=torch.int64, device=dev)      # entry N: the dummy of the scatter
        j["rows"] = torch.empty((N * self.bs,), dtype=torch.int64, device=dev)
        j["unsaved"] = torch.zeros((H, B, self.S), dtype=torch.bool, device=dev)
        j["sk"] = torch.empty((N * self.bs, D), dtype=eng._k_gpu.dtype, device=dev)
        j["sv"] = torch.empty((N * self.bs, D), dtype=eng._v_gpu.dtype, device=dev)
        j["sb"] = torch.empty((N * self.bs,), dtype=eng._kv_bias_gpu.dtype, device=dev)
        return j

    def _alloc_defer(self, l: int) -> dict:
        """The deferred write-back of one layer: the staging block, the step (K = none), the position, the block id."""
        eng = self.rs.cache.layers[l].cache_engine
        shape = (self.B, self.bs, self.H, self.D)
        return dict(k=torch.zeros(shape, dtype=eng._k_gpu.dtype, device=self.dev),
                    v=torch.zeros(shape, dtype=eng._v_gpu.dtype, device=self.dev),
                    u=torch.full((self.B,), self.K, dtype=torch.int64, device=self.dev),
                    pos=torch.full((self.B,), -1, dtype=torch.int64, device=self.dev),
                    id=torch.full((self.B,), -1, dtype=torch.int64, device=self.dev),
                    m2=None)

    def attach(self) -> None:
        """Install the hooks: the ops' linear (the release wait, RELEASE EVENTS), each twin's step_unchecked (the
        pre-step journal), the ops' gather_k / gather_v_bias (redirect; without a pool, the slot save) and pool_update
        (the slot save), the body's writeback_kernel (the deferred write-back). The originals are kept; detach() puts
        them back."""
        rs, kern, twins = self.rs, self.rs.ops.kernels, self.rs.st.twins
        self._orig = dict(gather_k=kern["gather_k"], gather_v_bias=kern["gather_v_bias"],
                          pool_update=kern.get("pool_update"), pool_swap=kern.get("pool_swap"),
                          writeback=rs.body.writeback_kernel, linear=rs.ops.linear,
                          twin=[t.step_unchecked for t in twins],
                          twin_own=["step_unchecked" in t.__dict__ for t in twins],
                          decode_fixed=rs.body.decode_fixed, decode_fixed_own="decode_fixed" in rs.body.__dict__)
        rs.ops.linear = self._linear_hook
        rs.body.decode_fixed = self._decode_fixed_hook
        for l, lay in enumerate(rs.cache.layers):
            e = lay.cache_engine
            self._lay[e._k_gpu.data_ptr()] = l
            self._lay[e._v_gpu.data_ptr()] = l
            if self.pool:
                self._lay[e._pool_map.data_ptr()] = l
        kern["gather_k"] = self._gather_k_hook
        kern["gather_v_bias"] = self._gather_v_hook
        if self.pool:
            kern["pool_update"] = self._pool_update_hook
        rs.body.writeback_kernel = self._writeback_hook
        for l, t in enumerate(twins):
            t.step_unchecked = functools.partial(self._twin_hook, l)
        rs.rollback = self

    def detach(self) -> None:
        if self.tentative:
            raise RollbackError("detach inside a group: commit first")
        rs, kern, o = self.rs, self.rs.ops.kernels, self._orig
        if rs.body.__dict__.get("decode_fixed") != self._decode_fixed_hook:
            raise RollbackError("detach: another hook wraps the body's decode_fixed (detach that hook first)")
        kern["gather_k"], kern["gather_v_bias"] = o["gather_k"], o["gather_v_bias"]
        if self.pool:
            kern["pool_update"] = o["pool_update"]
        rs.body.writeback_kernel = o["writeback"]
        rs.ops.linear = o["linear"]
        if o["decode_fixed_own"]:
            rs.body.decode_fixed = o["decode_fixed"]
        else:
            del rs.body.decode_fixed
        for t, f, own in zip(rs.st.twins, o["twin"], o["twin_own"]):
            if own:
                t.step_unchecked = f
            else:
                del t.step_unchecked
        rs.rollback = None

    # -------------------------------------------------------------- the group
    def begin(self) -> None:
        if self.tentative:
            raise RollbackError("begin inside a group: commit first")
        self.tentative = True
        self._u = [0] * self.L
        self._ures = [0] * self.L
        self._undone.zero_()
        self._undone_host.zero_()
        self._act_host = []
        self._pos_host0 = self.rs.st.pos_host.clone()
        for l in range(self.L):
            self._stamp0[l] = self.rs.cache.layers[l].cache_engine._pool_stamp_base if self.pool else 0

    def commit(self, n_acc: torch.Tensor, n_acc_host=None) -> None:
        """Request b keeps its first n_acc[b] tentative steps (module docstring). n_acc: a (B,) integer tensor on the
        engine's device (no host read); n_acc_host: the same values on the CPU, or None (then RaggedState.pos_host is
        not updated)."""
        if not self.tentative:
            raise RollbackError("commit outside a group: begin first")
        n_run = self._u[0]
        same = self._u.count(n_run)
        if same != self.L:
            raise RollbackError("the layers ran different numbers of tentative steps: %s" % (self._u,))
        if n_acc.shape != self.bshape or n_acc.device != self.dev:
            raise RollbackError("n_acc must be a (%d,) tensor on %s" % (self.B, self.dev))
        if self._ures != self._u:
            raise RollbackError("a step changed the engine without its engine journal (layers: twin %s, engine %s): "
                                "a joint pass without the rollback's hooks?" % (self._u, self._ures))
        na = None                                                     # every check before any work (no partial commit)
        if n_acc_host is not None:
            na = torch.as_tensor(n_acc_host, dtype=torch.int64).clamp(0, n_run)
            if na.device.type != "cpu" or na.shape != self.bshape:
                raise RollbackError("n_acc_host must be a (%d,) CPU tensor or list" % self.B)
        nacc = n_acc.to(torch.int64).clamp(0, n_run)
        if self.neg == "host_sync":                                   # NEGATIVE CONTROL (tests only)
            self._neg_host_sync(nacc)
        und = self._undone[:n_run].clone()                            # TRUNCATE: these steps are undone already
        if self.neg == "no_truncate_mark":                            # NEGATIVE CONTROL (tests only): undo them twice
            und.zero_()
        for l in range(self.L):
            self.stream.run(functools.partial(self._commit_layer, l, nacc, n_run, und))
            self._release[l] = self.stream.record(self.dev)
        self._need_wait = True
        if na is not None:
            self._mirror(na, n_run)
        self.tentative = False
        self._u = [0] * self.L
        self.groups += 1

    def reject_all(self) -> None:
        """commit(0) for every request, then the LRU clock of every layer back to its begin() value (CLOCK)."""
        if not self.tentative:
            raise RollbackError("reject_all outside a group: begin first")
        stamps = self._stamp0[:]
        self.commit(self.zeros_b, self.zeros_b_host)
        if self.pool:
            for l in range(self.L):
                self.rs.cache.layers[l].cache_engine._pool_stamp_base = stamps[l]

    def release_event(self, l: int):
        """The event after commit()'s last access of layer l (None once a step of layer l waited it)."""
        return self._release[l]

    def stats(self) -> dict:
        """ONE host sync (tests and the cell; never on the group's path): the journal counters per layer."""
        v = torch.stack([self.count[n] for n in COUNTERS]).cpu().tolist()
        return {n: [int(x) for x in row] for n, row in zip(COUNTERS, v)}

    def _neg_host_sync(self, nacc) -> int:
        """NEGATIVE CONTROL (tests only): one host read of n_acc. Not in COMMIT_PATH."""
        return int(nacc.max())

    def _mirror(self, na, n_run: int) -> None:
        """RaggedState.pos_host = its value at begin() + the active tentative steps that na keeps (na: the checked (B,)
        CPU copy of n_acc, clamped; CPU tensors only: no device read)."""
        adv = torch.zeros(self.B, dtype=torch.int64)
        for u in range(n_run):
            adv += ((na > u) & self._act_host[u] & ~self._undone_host[u]).to(torch.int64)
        self.rs.st.pos_host.copy_(self._pos_host0 + adv)

    # -------------------------------------------------------------- the undo
    def _commit_layer(self, l: int, nacc, n_run: int, und) -> None:
        if self.neg != "no_rollback":                                # NEGATIVE CONTROL (tests only): the dr1 flow
            for u in range(n_run - 1, -1, -1):
                self._undo_step(l, u, (nacc <= u) & ~und[u])
        self._writeback_deferred(l, nacc)

    def _where_rows(self, dst, saved, mask) -> None:
        dst.copy_(torch.where(mask, saved, dst))

    def _undo_step(self, l: int, u: int, rej) -> None:
        """Undo tentative step u of layer l for the requests in rej (B,) bool (module docstring, the table)."""
        eng, tw, j, ab = self.rs.cache.layers[l].cache_engine, self.rs.st.twins[l], self.J[l][u], self.all_b
        r3 = rej.view(1, self.B, 1)
        r1 = rej.view(self.B, 1)
        r2 = rej.view(self.B, 1, 1)
        if self.neg != "no_slot_restore":
            self._undo_slots(l, eng, j, rej, r3)
        if self.neg != "no_meta_restore":
            for n in META3:
                self._where_rows(getattr(eng, n), j[n], r3)
            self._where_rows(eng._cache_lens, j["_cache_lens"], rej)
        if self.pool and self.neg != "no_pool_restore":
            for n in POOL4:
                self._where_rows(getattr(eng, n), j[n], r3)
        if self.neg != "no_tail_restore":
            row = j["trow"]
            eng._k_gpu[ab, row] = torch.where(r2, j["tk"], eng._k_gpu[ab, row])
            eng._v_gpu[ab, row] = torch.where(r2, j["tv"], eng._v_gpu[ab, row])
            eng._kv_bias_gpu[ab, row] = torch.where(r1, j["tb"], eng._kv_bias_gpu[ab, row])
        if self.neg != "no_twin_restore":
            ix = j["idx"]
            tw.comp_k[ab, ix[0]] = torch.where(r2, j["comp_k"], tw.comp_k[ab, ix[0]])
            tw.ring[ab, ix[1]] = torch.where(r2, j["ring"], tw.ring[ab, ix[1]])
            tw.total_cis[ab, ix[2]] = torch.where(r1, j["total_cis"], tw.total_cis[ab, ix[2]])
            tw.ccis[:, ab, ix[3]] = torch.where(rej.view(1, self.B), j["ccis"], tw.ccis[:, ab, ix[3]])
            for i in range(3):
                self._where_rows(getattr(tw, TWIN_CTR[i]), j["ctr"][i], rej)

    def _undo_slots(self, l: int, eng, j, rej, r3) -> None:
        """SLOTS (1)-(3): the inverse plan, the saved data, the host for the slots past N."""
        if self.pool:
            act = torch.where(r3, self.undo_lut[j["plan_action"].to(torch.int64)], self.zero8)
            self._orig["pool_swap"](eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu, j["plan_target"], act, self.M, self.bs)
        self._restore_slots(eng, j, rej)
        if self.neg != "no_host_fallback":
            self._fallback(l, eng, j, r3)

    def _restore_slots(self, eng, j, rej) -> None:
        """The saved rows back where their request rejects the step. An entry that does not (or the dummy) writes back
        the value it reads: one index_copy_ per tensor, fixed shapes."""
        ent = j["ent"][:self.N]
        b = (torch.where(ent >= 0, ent, self.dummy) // self.S) % self.B
        valid = (ent >= 0) & rej[b]
        vr = valid.view(-1, 1).expand(-1, self.bs).reshape(-1)
        idx = j["rows"]
        kf = eng._k_gpu.view(-1, self.D)
        kf.index_copy_(0, idx, torch.where(vr.view(-1, 1), j["sk"], kf.index_select(0, idx)))
        vf = eng._v_gpu.view(-1, self.D)
        vf.index_copy_(0, idx, torch.where(vr.view(-1, 1), j["sv"], vf.index_select(0, idx)))
        bf = eng._kv_bias_gpu.view(-1)
        bf.index_copy_(0, idx, torch.where(vr, j["sb"], bf.index_select(0, idx)))

    def _fallback(self, l: int, eng, j, r3) -> None:
        """A lost slot past N that held a block: the block again from the host (the shipped gathers over the whole
        slot range, the bias from twin.total_cis), then the staging redirect."""
        owner = torch.cat((j["_block_map"], j["_pool_map"]), dim=2) if self.pool else j["_block_map"]
        ids = torch.where(j["unsaved"] & r3, owner, self.minus1)
        self._orig["gather_k"](eng._k_gpu, eng._k_cpu, ids, self.bs)
        self._orig["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, self.rs.st.twins[l].total_cis, ids,
                                    self.bs)
        m2 = self._redirect_ids(l, ids)
        self._redirect(eng._k_gpu, self.T[l]["k"], m2)
        self._redirect(eng._v_gpu, self.T[l]["v"], m2)

    def _writeback_deferred(self, l: int, nacc) -> None:
        """DEFERRED WRITE-BACK: the staged blocks of the kept fills (fill step < n_acc[b]) to the host rows with the
        shipped device-driven kernel; then no block is staged."""
        eng, T = self.rs.cache.layers[l].cache_engine, self.T[l]
        keep = T["u"] < nacc
        self._orig["writeback"](T["k"], T["v"], eng._k_cpu, eng._v_cpu, keep, T["pos"], 0, self.bs)
        T["u"].fill_(self.K)
        T["pos"].fill_(-1)
        T["id"].fill_(-1)

    def _wait_release_all(self) -> None:
        """The current stream waits the release event of every layer (RELEASE EVENTS); then the events are done with."""
        self._need_wait = False
        for l in range(self.L):
            ev = self._release[l]
            self._release[l] = None
            if ev is None:
                continue
            if self.neg == "skip_release_wait":                      # NEGATIVE CONTROL (tests only)
                continue
            self.stream.wait(ev, self.dev)

    # -------------------------------------------------------------- the hooks (inside a step)
    def _linear_hook(self, x, w):
        """The ops' GEMM. The first GEMM after a commit is the first op of a step: the release wait goes here."""
        if self._need_wait:
            self._wait_release_all()
        return self._orig["linear"](x, w)

    def _twin_hook(self, l: int, keys, cis, active):
        """The TWIN part of the pre-step journal of layer l (the first write of a step is the twin's: the counters, the
        twin rows, the tail row), then the twin step. The ENGINE part is journaled by _decode_fixed_hook (SPLIT
        JOURNAL)."""
        if not self.tentative:
            return self._orig["twin"][l](keys, cis, active)
        if self._need_wait:                                          # a joint pass reaches the engine here first
            self._wait_release_all()
        u = self._u[l]
        if u >= self.K:
            raise RollbackError("more than K = %d tentative steps in one group (layer %d)" % (self.K, l))
        self._cur[l] = u
        self._u[l] = u + 1
        if l == 0:
            self._act_host.append(self.rs.st.active_host.clone())
        eng, tw, j, ab = self.rs.cache.layers[l].cache_engine, self.rs.st.twins[l], self.J[l][u], self.all_b
        c, ix = j["ctr"], j["idx"]
        c[0].copy_(tw.pos)
        c[1].copy_(tw.n_comp)
        c[2].copy_(tw.n_ccis)
        ix[0].copy_(tw.n_comp.clamp(max=tw.lk_max - 1))              # the row indices of step_unchecked (:381-397)
        ix[1].copy_(tw.pos % RING)
        ix[2].copy_(tw.pos.clamp(max=tw.n_max - 1))
        ix[3].copy_(tw.n_ccis.clamp(max=tw.lk_max - 1))
        j["comp_k"].copy_(tw.comp_k[ab, ix[0]])
        j["ring"].copy_(tw.ring[ab, ix[1]])
        j["total_cis"].copy_(tw.total_cis[ab, ix[2]])
        j["ccis"].copy_(tw.ccis[:, ab, ix[3]])
        row = j["trow"]
        row.copy_(tw.pos % self.bs + self.tail0)                     # DecodeBody.write_tail's row (:248)
        j["tk"].copy_(eng._k_gpu[ab, row])
        j["tv"].copy_(eng._v_gpu[ab, row])
        j["tb"].copy_(eng._kv_bias_gpu[ab, row])
        return self._orig["twin"][l](keys, cis, active)

    def _decode_fixed_hook(self, eng, *args, **kwargs):
        """SPLIT JOURNAL: in a group, the ENGINE part of the pre-step journal of the next step of this layer whose
        twin step ran (the ragged step: the same step; a joint pass in EXACT mode: its rows in order), and the step
        index of the pool / gather / write-back hooks; then the body's decode_fixed."""
        if self.tentative:
            if self._need_wait:
                self._wait_release_all()
            l = self._lay[eng._k_gpu.data_ptr()]
            u = self._ures[l]
            uu = self._u[l]
            if u >= uu:
                raise RollbackError("layer %d: decode_fixed without its twin step (engine journal %d, twin %d)"
                                    % (l, u, uu))
            self._ures[l] = u + 1
            self._journal_engine(l, u)
        return self._orig["decode_fixed"](eng, *args, **kwargs)

    def _journal_engine(self, l: int, u: int) -> None:
        """The ENGINE part of the journal of step u, layer l (the maps, the lengths and the pool tables before the step
        writes them); the pool / gather / write-back hooks then record into step u."""
        eng, j = self.rs.cache.layers[l].cache_engine, self.J[l][u]
        for n in META4:
            j[n].copy_(getattr(eng, n))
        if self.pool:
            for n in POOL4:
                j[n].copy_(getattr(eng, n))
        self._cur[l] = u

    # -------------------------------------------------------------- the joint pass, FAST mode (verify_rows.py)
    def fast_begin(self, l: int, eng, R: int) -> None:
        """verify_rows FAST, before the residency of layer l (one residency for the R rows whose twin steps ran): the
        engine journal of the first row. Outside a group: nothing."""
        if not self.tentative:
            return
        if self._need_wait:
            self._wait_release_all()
        u0 = self._u[l] - R
        uu = self._ures[l]
        if u0 != uu:
            raise RollbackError("layer %d: fast_begin for rows %d .. %d, the engine journal is at %d" % (
                l, u0, u0 + R - 1, uu))
        self._journal_engine(l, u0)
        self._ures[l] = u0 + 1

    def fast_end(self, l: int, eng, p0, full, R: int) -> None:
        """verify_rows FAST, after the pass of layer l: the engine journal of rows 1 .. R - 1 = the state after r rows
        (module docstring, PHASE 2 ADDITIONS), empty slot journals for them, and the row of each request's fill
        (full (B,) bool: the request filled its tail block in this pass, at row 63 - p0 % 64). p0 (B,) int64: the
        position of row 0."""
        if not self.tentative:
            return
        u0 = self._u[l] - R
        bs, tail = self.bs, self.M - 1
        for r in range(1, R):
            j = self.J[l][u0 + r]
            for n in META4:
                j[n].copy_(getattr(eng, n))
            if self.pool:
                for n in POOL4:
                    j[n].copy_(getattr(eng, n))
                j["plan_action"].zero_()                             # no pool move of its own: the undo moves nothing
                j["plan_target"].fill_(-1)
            j["_block_map"][:, :, tail].copy_(((p0 + r) // bs).view(1, self.B).expand(self.H, self.B))
            j["_cache_lens"].copy_(self.rs.body.cache_lens_value(eng, p0 + (r - 1)).to(torch.int32))
            j["ent"].fill_(-1)                                       # no saved slot (the residency stays)
            j["rows"].copy_(self._slot_rows(j["ent"][:self.N]))      # valid row ids (the dummy) for the masked copy
            j["unsaved"].zero_()
        T = self.T[l]
        T["u"].copy_(torch.where(full, u0 + (bs - 1) - p0 % bs, T["u"]))
        self._ures[l] = u0 + R

    # -------------------------------------------------------------- TRUNCATE (an open group, some requests)
    def truncate(self, n_keep: torch.Tensor, n_keep_host=None) -> None:
        """Inside an open group: request b with n_keep[b] < the steps run loses its steps n_keep[b] .. n_run - 1 now;
        the group stays open (module docstring, PHASE 2 ADDITIONS). n_keep: a (B,) integer tensor on the engine's
        device (no host read). n_keep_host: the same values on the CPU, or None (then RaggedState.pos_host is not
        updated; a later commit's mirror then counts the truncated steps as kept)."""
        if not self.tentative:
            raise RollbackError("truncate outside a group: begin first")
        n_run = self._u[0]
        same = self._u.count(n_run)
        if same != self.L:
            raise RollbackError("the layers ran different numbers of tentative steps: %s" % (self._u,))
        if n_keep.shape != self.bshape or n_keep.device != self.dev:
            raise RollbackError("n_keep must be a (%d,) tensor on %s" % (self.B, self.dev))
        if self._ures != self._u:
            raise RollbackError("a step changed the engine without its engine journal (layers: twin %s, engine %s)"
                                % (self._u, self._ures))
        nh = None
        if n_keep_host is not None:
            nh = torch.as_tensor(n_keep_host, dtype=torch.int64).clamp(0, n_run)
            if nh.device.type != "cpu" or nh.shape != self.bshape:
                raise RollbackError("n_keep_host must be a (%d,) CPU tensor or list" % self.B)
        nk = n_keep.to(torch.int64).clamp(min=0, max=n_run)
        tr = nk < n_run
        rejs = [(nk <= u) & ~self._undone[u] for u in range(n_run)]
        for l in range(self.L):
            self.stream.run(functools.partial(self._truncate_layer, l, rejs, nk, tr, n_run))
            self._release[l] = self.stream.record(self.dev)
        self._need_wait = True
        for u in range(n_run):
            self._undone[u] |= rejs[u]
        if nh is not None:
            for u in range(n_run):
                self._undone_host[u] |= nh <= u
            adv = torch.zeros(self.B, dtype=torch.int64)
            for u in range(n_run):
                adv += (self._act_host[u] & ~self._undone_host[u]).to(torch.int64)
            self.rs.st.pos_host.copy_(self._pos_host0 + adv)
        self.truncations += 1

    def _truncate_layer(self, l: int, rejs, nk, tr, n_run: int) -> None:
        """The undo of the truncated steps of layer l (reverse order), then the staged fill of each truncated request:
        to the host rows if its step is kept (fill step < n_keep), dropped either way."""
        for u in range(n_run - 1, -1, -1):
            self._undo_step(l, u, rejs[u])
        eng, T = self.rs.cache.layers[l].cache_engine, self.T[l]
        keep = tr & (T["u"] < nk)
        self._orig["writeback"](T["k"], T["v"], eng._k_cpu, eng._v_cpu, keep, T["pos"], 0, self.bs)
        T["u"].copy_(torch.where(tr, self.kbig, T["u"]))
        T["pos"].copy_(torch.where(tr, self.minus1, T["pos"]))
        T["id"].copy_(torch.where(tr, self.minus1, T["id"]))

    def _pool_update_hook(self, old_map, load_mask, pool_map, pool_age, pool_target, pool_action, stamp_base,
                          tail_slot):
        """pool_update, then (in a group) the plan and the old data of every slot this step will lose (SLOTS)."""
        self._orig["pool_update"](old_map, load_mask, pool_map, pool_age, pool_target, pool_action, stamp_base,
                                  tail_slot)
        if not self.tentative:
            return
        l = self._lay[pool_map.data_ptr()]
        j = self.J[l][self._cur[l]]
        j["plan_target"].copy_(pool_target)
        j["plan_action"].copy_(pool_action)
        a = pool_action.to(torch.int64)
        win = ((load_mask >= 0) & (a == NONE_)) | (a == MOVE_IN)
        pq = torch.zeros((self.H, self.B, self.P + 1), dtype=torch.bool, device=self.dev)
        pq.scatter_(2, torch.where(a == MOVE_OUT, pool_target, self.P), True)
        self._save_slots(l, j, torch.cat((win, pq[:, :, :self.P]), dim=2))

    def _gather_k_hook(self, gpu, cpu, load_ids, block_size):
        if not self.tentative:
            return self._orig["gather_k"](gpu, cpu, load_ids, block_size)
        l = self._lay.get(gpu.data_ptr(), -1)
        if l < 0:                                                    # not an engine table (verify_rows FAST's extras
            return self._orig["gather_k"](gpu, cpu, load_ids, block_size)   # store: scratch, no journal, no redirect)
        if not self.pool:                                            # no pool: the gather is the only loser of slots
            self._save_slots(l, self.J[l][self._cur[l]], load_ids >= 0)
        self._orig["gather_k"](gpu, cpu, load_ids, block_size)
        T = self.T[l]
        T["m2"] = self._redirect_ids(l, load_ids)
        self._redirect(gpu, T["k"], T["m2"])

    def _gather_v_hook(self, gpu_v, cpu_v, bias_buf, kv_bias, load_ids, block_size):
        self._orig["gather_v_bias"](gpu_v, cpu_v, bias_buf, kv_bias, load_ids, block_size)
        if not self.tentative:
            return
        l = self._lay.get(gpu_v.data_ptr(), -1)
        if l < 0:                                                    # the extras store (see _gather_k_hook)
            return
        T = self.T[l]
        self._redirect(gpu_v, T["v"], T["m2"])

    def _redirect_ids(self, l: int, ids):
        """(H, B, M') load ids -> 0 where the id is the staged block of its request, else -1."""
        d = self.T[l]["id"].view(1, self.B, 1)
        m2 = torch.where((ids == d) & (d >= 0), self.zero, self.minus1)
        self.count["redirected"][l].add_((m2 >= 0).sum())
        return m2

    def _redirect(self, dst, staged, m2) -> None:
        if self.neg == "no_redirect":                                # NEGATIVE CONTROL (tests only)
            return
        self._orig["gather_k"](dst, staged, m2, self.bs)

    def _writeback_hook(self, k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size):
        """Outside a group: the shipped write-back. In a group: slot 63 of each filling request into the staging
        block (DEFERRED WRITE-BACK); the host rows are not touched."""
        if not self.tentative or self.neg == "no_defer":
            return self._orig["writeback"](k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size)
        l = self._lay[k_gpu.data_ptr()]
        T, u = self.T[l], self._cur[l]
        f4 = full.view(self.B, 1, 1, 1)
        T["k"].copy_(torch.where(f4, k_gpu[:, tail0:tail0 + block_size], T["k"]))
        T["v"].copy_(torch.where(f4, v_gpu[:, tail0:tail0 + block_size], T["v"]))
        T["pos"].copy_(torch.where(full, pos, T["pos"]))
        T["u"].copy_(torch.where(full, self.ustep[u], T["u"]))
        T["id"].copy_(torch.where(full, pos // block_size, T["id"]))

    # -------------------------------------------------------------- the slot journal
    def _save_slots(self, l: int, j, lost) -> None:
        """lost (H, B, S) bool: the slots this step overwrites. The first N in (head, request, slot) order go to the
        flat journal (index_select); the others are marked unsaved. No host read."""
        eng, N = self.rs.cache.layers[l].cache_engine, self.N
        f = lost.reshape(-1)
        rank = torch.cumsum(f.to(torch.int64), 0) - 1
        ent = j["ent"]
        ent.fill_(-1)
        ent.scatter_(0, torch.where(f & (rank < N), rank, N), self.flat)
        uns = j["unsaved"]
        uns.copy_((f & (rank >= N)).view(self.H, self.B, self.S))
        held = torch.cat((j["_block_map"], j["_pool_map"]), dim=2) >= 0 if self.pool else j["_block_map"] >= 0
        n = rank[-1] + 1
        self.count["destroyed"][l].add_(n)
        self.count["saved"][l].add_(n.clamp(max=N))
        self.count["overflow"][l].add_((uns & held).sum())
        self.count["overflow_dead"][l].add_((uns & ~held).sum())
        idx = j["rows"]
        idx.copy_(self._slot_rows(ent[:N]))
        torch.index_select(eng._k_gpu.view(-1, self.D), 0, idx, out=j["sk"])
        torch.index_select(eng._v_gpu.view(-1, self.D), 0, idx, out=j["sv"])
        torch.index_select(eng._kv_bias_gpu.view(-1), 0, idx, out=j["sb"])

    def _slot_rows(self, ent):
        """(N,) flat (head, request, slot) ids, -1 = none -> (N * 64,) row ids of the flat (B * R * H) row space of
        _k_gpu.view(-1, D) (the dummy, -1: head 0, request 0, the tail slot)."""
        f = torch.where(ent >= 0, ent, self.dummy)
        s = f % self.S
        hb = f // self.S
        b = hb % self.B
        h = hb // self.B
        r = (b * self.R + s * self.bs).view(-1, 1) + self.rows.view(1, -1)
        return (r * self.H + h.view(-1, 1)).reshape(-1)
