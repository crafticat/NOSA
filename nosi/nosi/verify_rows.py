"""Task 1.4, ROUTE 1 of KS/analysis/plan/PLAN_speckv_on_cached_nosa.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the
VERIFY ROWS IN THE SAME FORWARD PASS (paper section 3.2). Opt-in, on top of the per-request engine of Task 1.3
(ragged_engine.py). Only benchmarks/Efficiency/r2_verify_cell.py and the CPU tests import this module. The shipped
engine, ragged_engine.py and r2_base_cell.py do not import it (Gate 0).

WHAT ONE JOINT PASS DOES. Each request runs R = K + 1 rows at the positions p, p + 1, ..., p + K (K = 3: rows 0-2 are
the verify rows, row K is the draft row; in this route every row attends its full selection). Entry e = b * R + r.
Per layer, in the shipped order (nosa_llama.py:539-662):
  1. rmsnorm, wqkv, nosa_linear, rope over ALL E = B * R rows: ONE GEMM call per weight (one weight read).
  2. the positional twin advances R times (r2_positional_twin.step_unchecked, B rows per call, the serial shapes).
     Row r records its own lengths: n_used = n_comp after its step, M_b = pos_before // 64 + 1.
  3. ONE BATCHED SELECTION for the E rows: ONE stage-1 call (stage1_seqused, cu_seqlens_q = arange(E + 1),
     cu_seqlens_k[e] = (e // R) * Lk_max, seqused_k[e] = row e's n_used), ONE per-request pooling call over E entries
     (the compressed cis of request b copied to its R entries), ONE top-k pair. No loop over positions.
  4. residency and attention, by mode (below).
  5. wo, rmsnorm, gate_up, SiLU, down over all E rows: ONE GEMM call per weight.
Final rmsnorm and lm_head over the E rows (one call). Returns logits (B, R, V).

TWO MODES (plan, Global Constraints: EXACT for the proofs, FAST for the timing).
  "exact" (the proof form). For r = 0 .. K: the body's decode_fixed for row r (tail row p + r, diff + victim pool +
      gathers, write-back, tail id, attention length), then the shipped attention of row r on the attended window
      (B entries). Every op of row r gets the inputs of the serial step at position p + r, so row r is BITWISE the
      serial step when the batched ops are row-invariant (gemm "rowgroup" or "exact"; stage 1, the pooling, top-k,
      nosa_linear, rope and rmsnorm are per row). The engine ends in the serial state. The window holds row r's
      selection when row r attends; the pool keeps the blocks that a later row evicts.
  "fast" (the timing form; route 1 of the plan). ONE residency for row 0's selection (the shipped diff, victim pool
      and gathers), then the EXTRAS: the blocks of rows 1 .. K that the window does not hold. A pool hit is copied
      device to device (torch indexing) into a small extras store (B, X * 64, H, D); a miss is fetched from the host
      with the shipped gathers into the same store. The last extras slot holds the rows of the NEXT block when the
      group crosses a 64-token block edge. Then TWO attention calls over all E entries (flash_attn_nosa, the decode
      kernel, seqlen_q = 1 per entry, the K / V of request b through cache_batch_idx[e] = b):
        call A on the attended window: a per-entry bias row set (the window bias where the slot's block is in row
               r's selection, MASK_BIAS elsewhere; the tail slot always visible); the causal cut inside the tail
               block through cache_seqlens (the body's attention length of position p + r: findings F4 kept);
        call B on the extras store: the bias masked to row r's selected extras and, for a row past the block edge,
               to the next-block rows at or before its own position.
      The two partial results are merged by their log-sum-exp (return_softmax_lse). Then the write-back of a filled
      block, the next-block rows into the tail slot, the tail id and the attention length.
      FAST is NOT bitwise the serial step: the window slot order and the 128-column tile pairs differ, so the online
      softmax sums in another order. A union layout can never be bitwise, with or without splits (NOTES_verify1.md).

WHY flash_attn_nosa AND NOT NOSA'S PREFILL VARLEN (the task's choice). Both need no kernel change. flash_attn_nosa
already computes the shipped decode math, (q.k + bias) x scale, with the bias inside the kernel. The InfLLM-v2 varlen
kernel has no bias input: it needs V x exp(bias x scale) and a second call for the denominator (the EQ verify_step.py
two-call trick, 33-58 ms of copies in the pilot), a pre-scaled bias, and a contiguous per-request union store (a copy
of the K / V). Neither route shares K / V loads between rows (both tile one token x 16 heads).

THE BIAS ROWS PER ENTRY (no kernel change). flash_api.cpp:366 checks bcache against the CACHE batch (batch_size_c =
k_cache.size(0) = B), but the kernel reads the bias with the QUERY entry index bidb (flash_fwd_kernel.h:968-970),
while K / V use cache_batch_idx[bidb] (:649-659). So the bias is a contiguous (E, L, H) buffer, passed as its
first-B-rows view (shape (B, L, H), batch stride L * H): the kernel then reads row e for entry e. The GPU cell gates
it (VRF-IDX: sampled entries equal a plain one-entry call, bitwise).

MASK_BIAS = -3e4, finite, as avail_policy.py:113 (a whole split can be masked; -inf would give NaN there).

NEGATIVE CONTROLS (tests only; VerifyOps.neg): "no_init_force" (the pooling forces no init block), "no_extras" (call B
sees no extras), "no_row_mask" (call A shows every window slot to every row), "causal_leak" (call A's tail cut is the
last tail row's for every row), "shared_row_lengths" (every row uses row 0's n_used and M_b), "drop_spill" (mode
"union": ONE store fill and no spill, so the listed blocks past X - 1 are not served: the old FAST overflow of vr1,
5.57 % of the extras; caught by the presence recount and by test_wsb.py's byte check), "union_attention" (mode
"union": every row sees every window block and every listed store block, the union; the contract forbids it; caught
by the visible count).

NO NEW KERNEL. The kernels are the shipped ones and the ones of Task 1.3: stage1_seqused (the installed infllm_v2.C
op), nosa_pooling_per_request, torch.topk, diff_offload, pool_update, flash_pool_swap, flash_h2d_from_mask(_bias),
r2_writeback.masked_writeback, flash_attn_nosa_with_kvcache. Everything else is torch.

PHASE 2 (the integration tree nosi-r2-spec, the speculative round runner; every default keeps Task 1.4's behaviour):
  - VerifyOps(draft_last=True), FAST: row K is a resident-only draft row (the extras store holds the verify rows'
    union only; nothing is fetched for row K alone). EXACT has no such option: the runner's DraftResident hook makes
    row K's decode_fixed a draft residency.
  - VerifyState.hooks (default None): an open rollback group (Task 1.5's Rollback) journals a FAST layer through
    hooks.fast_begin / hooks.fast_end. The EXACT rows need no hook (the rollback wraps the body's decode_fixed).
  - The FAST write-back calls rs.body.writeback_kernel (the same kernel as kernels["writeback"]; in an open rollback
    group, the deferred write-back).

WS-B (tree nosi-r2-wsb, KS/analysis/plan/PLAN_L4P_faithful.md, departure D2; docs/NOTES_wsb.md): MODE "union", the
FAST joint pass with ONE residency per layer for the UNION of the rows' selections. The modes "fast" and "exact" are
the bytes of 40ac348 (test_wsb.py Gate 0 runs both sources). Per layer, in this order (_union_rows):
  1. the tentative rows: the tail block's rows into the tail slot, the next block's rows into STORE SLOT 0.
  2. THE RESIDENCY (one per layer and pass; vs.calls["resid"] counts it): row 0's selection into the window with the
     shipped diff, victim pool, map commit and the two host gathers (the shipped residency of position p).
  3. THE UNION LIST (_union_list): per (head, request), in id order, every block that a FULL row selects (a verify row:
     its whole selection must be present) and every POOL HIT of a RESIDENT row (a draft row reads what is on the GPU and
     never fetches), minus the window's blocks and the next block. Nothing is dropped: the list holds up to cap =
     C x (X - 1) ids, and C is sized from the bound (R - 1) x topk (a row's selection has topk ids and row 0's are in
     the window), so cap covers any union.
  4. THE STORE (B, X x 64, H, D) holds X - 1 listed blocks per (head, request) at a time (slots 1 .. X - 1; slot 0 is
     the next block). Fill c (c = 0 .. C - 1) loads the listed ids c (X - 1) .. (c + 1)(X - 1) - 1 with the SHIPPED
     gathers: a pool hit device to device (flash_h2d_from_mask with the engine's own slot rows as the source, the slot
     topk + q as the "block id"), a miss from the host rows (the urgent GPU gather). Fill 0 is the extra store; fills
     1 .. C - 1 are the SPILL: the same store again, counted (stat spill), never a drop.
  5. ATTENTION: call A on the window (as FAST: the per-entry bias rows, the tail cut by cache_seqlens) and one call per
     store fill, each a flash_attn_nosa call over the E one-row entries with cache_batch_idx (route 1) and a per-entry
     bias (VRF-IDX), the per-entry length = the highest slot that the row reads (0: the kernel's early exit, out 0,
     lse +inf; the merge gives that part the weight 0). The parts merge by their log-sum-exp after each call (a
     running numerator, denominator and maximum; no stack of the parts).
  6. the end state of FAST (write-back of a filled block, the next block's rows into the tail slot, tail id, length).
There is no serial per-row residency. Row 0 of a pass whose window state equals the serial one attends exactly the
serial row-0 window (its whole selection is in the window, its store parts are empty), so its attention has the
serial bits (test_wsb.py). Rows 1 .. R - 1 read their blocks in other slots and parts: FAST (float order only).
CHECK-ONLY (VerifyOps(check=True); the check cells; OFF in the timing cells): the per-layer copy of the selections
(vs.chk_sel), the PRESENCE RECOUNT (every selected block of a FULL row is in the window, a served store slot, the tail
block or the next block: stat dropped, a MUST counter), the VISIBLE COUNT (union residency must not become union
attention: a full row SEES exactly its selection size, a resident row its selection size minus its skipped blocks;
stat vis_bad, a MUST counter) and the per-row counts. ACCOUNTING (ops.stats, device counters; off in the timing
cells): row roles, the window's ready part and residual host loads, the store's pool / host blocks, the spill, the
rows that the attention calls stream. HOST COUNTS (always, no device op): vs.calls (GEMM, attention, residency and
gather calls, store fills). PRODUCTION (always): the store fills, the spill, the merge; the resident rows' miss ids
and the pin record only when the caller sets vs.miss_ids / vs.pin (the paper schedule with rung L5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import ragged_engine as RE
from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK
from .stage1_seqused import stage1_seqused

MODES = ("fast", "exact", "union")
GEMMS = ("natural", "rowgroup", "exact")
NEG_KINDS = (None, "no_init_force", "no_extras", "no_row_mask", "causal_leak", "shared_row_lengths", "drop_spill",
             "union_attention")
UNION_NEG = ("drop_spill", "union_attention")
UNION_STATS = ("extras", "extras_pool", "extras_host", "extras_max", "spill", "unserved", "fills_used", "dropped",
               "skipped", "union_sum", "union_max", "vis_bad", "win_sel", "win_host", "rows_full", "rows_res",
               "streamed_rows", "visible_blocks")
UNION_ROW_STATS = ("dropped", "skipped", "vis_bad", "visible", "selected")
MASK_BIAS = -3.0e4                     # avail_policy.py:113: finite on purpose
K_MAX = 31                             # R <= 32 rows: the twin's MAX_ROLLBACK and one block edge per group


class VerifyError(RuntimeError):
    pass


class VerifyOps:
    """The run parameters of the joint pass (all recorded in VerifyState.meta).
      K       verify rows per request (R = K + 1 rows: K verify + 1 draft), 1 .. K_MAX
      mode    "fast" (one residency + extras store + masked attention) or "exact" (the serial residency per row)
      gemm    "natural" (one call over the E rows), "rowgroup" (R calls, each over the B rows of one row index: the
              shipped call's shape, bitwise the shipped GEMM; R weight reads), "exact" (one call of a batch-invariant
              `linear`, e.g. r2_spec_core.PaddedLinear on the GPU, row_linear on the CPU)
      linear  the GEMM callable (default F.linear; required for gemm "exact")
      extra   X, the extras store slots per request (FAST): X - 1 extras + 1 next-block slot; >= 2
      attn    the attention callable (default: nosa_llama's flash_attn_nosa_with_kvcache)
      neg     a negative control (NEG_KINDS), tests only
      record  keep the per-layer evidence in VerifyState.log (tests only)
      stats   accumulate the union statistics on the device
      draft_last  FAST only (Phase 2, the speculative round runner): row K is a RESIDENT-ONLY DRAFT row. The extras store
              loads the union of rows 0 .. K - 1 only; row K attends what is present for it (the window and the
              extras that the verify rows loaded) and nothing is fetched for it alone. Default False (Task 1.4).
              Mode "union": row K is a resident row (its pool hits are brought, its misses are skipped).
      spill   mode "union": the store fills per layer (None: C = ceil((R - 1) topk / (X - 1)), the bound, so nothing can
              be dropped; an int: tests only)
      check   mode "union": the CHECK-ONLY operations (module docstring; OFF in the timing cells)"""

    def __init__(self, K=3, mode="fast", gemm="natural", linear=None, extra=16, attn=None, neg=None, record=False,
                 stats=True, draft_last=False, spill=None, check=False):
        if not 1 <= int(K) <= K_MAX:
            raise VerifyError("K must be in 1 .. %d (got %r)" % (K_MAX, K))
        if mode not in MODES or gemm not in GEMMS:
            raise VerifyError("mode %r / gemm %r: one of %s / %s" % (mode, gemm, MODES, GEMMS))
        if gemm == "exact" and linear is None:
            raise VerifyError("gemm 'exact' takes a batch-invariant linear")
        if int(extra) < 2:
            raise VerifyError("extra must be >= 2 (one extras slot and the next-block slot)")
        if neg not in NEG_KINDS:
            raise VerifyError("unknown negative control %r" % (neg,))
        if mode != "union" and (spill is not None or check or neg in UNION_NEG):
            raise VerifyError("spill / check / %s belong to mode 'union'" % (UNION_NEG,))
        if spill is not None and int(spill) < 1:
            raise VerifyError("spill: at least one store fill")
        self.spill = None if spill is None else int(spill)
        self.check = bool(check)
        self.K, self.R = int(K), int(K) + 1
        self.mode, self.gemm = mode, gemm
        self.linear = F.linear if linear is None else linear
        self.extra = int(extra)
        self.attn = attn
        self.neg = neg
        self.record = bool(record)
        self.stats = bool(stats)
        self.draft_last = bool(draft_last)

    def describe(self) -> dict:
        d = dict(K=self.K, R=self.R, mode=self.mode, gemm=self.gemm,
                 linear=getattr(self.linear, "__name__", type(self.linear).__name__),
                 m_fix=getattr(self.linear, "m_fix", None), extra=self.extra, neg=self.neg,
                 draft_last=self.draft_last)
        if self.mode == "union":
            d.update(spill=self.spill, check=self.check)
        return d


class VerifyState:
    """The buffers of the joint pass for one RaggedState: the E-entry selection buffers, the entry maps, the extras
    store (FAST), the device statistics, the test log and the timing marks."""

    def __init__(self, model, rs, ops: VerifyOps):
        st = rs.st
        self.rs, self.ops = rs, ops
        self.B, self.R = int(rs.B), int(ops.R)
        self.E = self.B * self.R
        B, R, E = self.B, self.R, self.E
        lay0 = model.layers[0]
        eng0 = rs.cache.layers[0].cache_engine
        H, D, bs, topk = int(eng0.head_num), int(eng0.head_dim), int(eng0.block_size), int(eng0.topk)
        if bs != BLOCK or int(eng0._tail_block_idx_on_gpu) != topk - 1:
            raise VerifyError("the engine must have 64-token blocks and the tail slot topk - 1")
        if R > bs:
            raise VerifyError("R = %d rows: more than one block edge per group" % R)
        dev = st.cu_q.device
        bdt = lay0.wqkv.dtype
        self.H, self.D, self.bs, self.topk = H, D, bs, topk
        nl = rs.nl
        self.pq = torch.empty((H, E, st.M_alloc), dtype=bdt, device=dev)
        self.pc = torch.empty((H, E, st.M_alloc), dtype=bdt, device=dev)
        self.tv_q = torch.empty((H, E, nl.QK_SELECT), dtype=bdt, device=dev)
        self.ti_q = torch.empty((H, E, nl.QK_SELECT), dtype=torch.int64, device=dev)
        self.tv = torch.empty((H, E, lay0.topk_blocks), dtype=bdt, device=dev)
        self.ti = torch.empty((H, E, lay0.topk_blocks), dtype=torch.int64, device=dev)
        self.mask = torch.empty((H, E, st.M_alloc), dtype=torch.bool, device=dev)
        self.ccis_rows = torch.empty((H, B, R, st.lk_max), dtype=st.twins[0].ccis.dtype, device=dev)
        self.cu_q = torch.arange(E + 1, dtype=torch.int32, device=dev)
        self.cu_k = (torch.arange(E + 1, dtype=torch.int64, device=dev) // R * st.lk_max).to(torch.int32)
        self.cbi = (torch.arange(E, dtype=torch.int64, device=dev) // R).to(torch.int32)
        self.n_used = torch.zeros((B, R), dtype=torch.int32, device=dev)
        self.M_b = torch.zeros((B, R), dtype=torch.int32, device=dev)
        self.offs = torch.arange(R, dtype=torch.int64, device=dev)
        self.bidx = torch.arange(B, dtype=torch.int64, device=dev).view(B, 1).expand(B, R)
        self.ar_bs = torch.arange(bs, dtype=torch.int64, device=dev)
        self.full = torch.zeros(B, dtype=torch.bool, device=dev)
        self.wpos = torch.zeros(B, dtype=torch.int64, device=dev)
        X = ops.extra
        self.X = X
        if ops.mode == "fast":
            kdt = eng0._k_gpu.dtype
            self.ext_k = torch.zeros((B, X * bs, H, D), dtype=kdt, device=dev)
            self.ext_v = torch.zeros((B, X * bs, H, D), dtype=kdt, device=dev)
            self.ext_bias = torch.zeros((B, X * bs, H), dtype=eng0._kv_bias_gpu.dtype, device=dev)
            self.ids_buf = torch.full((H, B, X + 1), -1, dtype=torch.int64, device=dev)
            self.load_x = torch.full((H, B, X), -1, dtype=torch.int64, device=dev)
            self.lens_x = torch.full((E,), X * bs, dtype=torch.int32, device=dev)
            self.hh = torch.arange(H, device=dev).view(H, 1, 1).expand(H, B, X - 1)
            self.bb = torch.arange(B, device=dev).view(1, B, 1).expand(H, B, X - 1)
            self.xx = torch.arange(X - 1, device=dev).view(1, 1, X - 1).expand(H, B, X - 1)
        # mode "union" (WS-B): the store, the fills, the per-pass row masks (module docstring)
        self.full_rows = self.res_rows = self.miss_rows = None   # (R, B) bool per pass; None: full_default / ~full
        self.miss_ids = None                # a list of L (H, B, topk) int64: the miss rows' skipped ids (rung L5 seam)
        self.pin = None                     # an object with record(l, sel, mask) (the requested-KV pin of the schedule)
        self.C = self.cap = 0
        self.chk_sel = None
        if ops.mode == "union":
            kdt = eng0._k_gpu.dtype
            if ops.neg == "drop_spill":                              # NEGATIVE CONTROL: no spill (the old overflow)
                C = 1
            elif ops.spill is not None:
                C = ops.spill
            else:                                                    # the bound: (R - 1) rows x topk ids
                C = max(1, -(-((R - 1) * topk) // (X - 1)))
            self.C, self.cap = C, C * (X - 1)
            self.ext_k = torch.zeros((B, X * bs, H, D), dtype=kdt, device=dev)
            self.ext_v = torch.zeros((B, X * bs, H, D), dtype=kdt, device=dev)
            self.ext_bias = torch.zeros((B, X * bs, H), dtype=eng0._kv_bias_gpu.dtype, device=dev)
            self.lst_buf = torch.full((H, B, self.cap + 1), -1, dtype=torch.int64, device=dev)
            self.served = torch.full((H, B, self.cap), -1, dtype=torch.int64, device=dev)
            self.dev_x = torch.full((H, B, X), -1, dtype=torch.int64, device=dev)    # slot 0 stays -1 (next block)
            self.host_x = torch.full((H, B, X), -1, dtype=torch.int64, device=dev)
            self.ar_x1 = torch.arange(1, X, dtype=torch.int64, device=dev)
            self.minus1 = torch.tensor(-1, dtype=torch.int64, device=dev)
            self.zero_i = torch.zeros((), dtype=torch.int64, device=dev)
            full = torch.ones((R, B), dtype=torch.bool, device=dev)
            if ops.draft_last:
                full[R - 1] = False
            self.full_default = full
            if ops.check:                                            # CHECK-ONLY: the selections of every layer
                self.chk_sel = torch.full((len(model.layers), H, B, R, topk), -1, dtype=torch.int64, device=dev)
        z = lambda: torch.zeros((), dtype=torch.int64, device=dev)    # noqa: E731
        self.stat = dict(union_sum=z(), union_max=z(), extras=z(), overflow=z(), sel0_order_eq=z(), sel0_set_eq=z())
        if ops.mode == "union":
            self.stat.update({k: z() for k in UNION_STATS if k not in self.stat})
            self.stat_row = {k: torch.zeros(R, dtype=torch.int64, device=dev) for k in UNION_ROW_STATS}
        self.stat_n = dict(union=0, sel0=0)
        # host invocation counts (every mode; no device op): GEMM calls, attention calls, residency calls, gather calls
        self.calls = dict(gemm=0, attn=0, resid=0, gather_dev=0, gather_host=0, fills=0)
        self.log = []
        self.marks = None                   # a list -> (layer, label, event) per phase (the cell's probe passes)
        self.clock = None                   # the cell's CudaClock (event / record)
        self.probe = False                  # True: the one-row selection after the batched one (timing + SEL0)
        self.fa_check = None                # a dict -> the VRF-IDX gate on the next FAST pass, layer 0
        self.hooks = None                   # Phase 2: the rollback (fast_begin / fast_end around a FAST layer)
        self._rec_fills = []                # mode union, record: the store after each fill of the current layer
        self.passes = 0
        self.meta = dict(ops=ops.describe(), B=B, R=R, E=E, X=X, mask_bias=MASK_BIAS)
        if ops.mode == "union":
            self.meta.update(fills=self.C, cap=self.cap, bound=(R - 1) * topk)


def attach(model, rs, ops: VerifyOps) -> VerifyState:
    """The joint-pass state of a RaggedState (ragged_engine.attach or build). It allocates buffers only: no engine
    table and no twin table changes (test_flag_off_is_shipped checks it)."""
    vs = VerifyState(model, rs, ops)
    if ops.attn is None:
        ops.attn = rs.nl.flash_attn_nosa_with_kvcache
    return vs


# ------------------------------------------------------------------ helpers
def _mark(vs: VerifyState, l: int, label: str) -> None:
    if vs.marks is not None:
        e = vs.clock.event()
        vs.clock.record(e)
        vs.marks.append((l, label, e))


def _lin(vs: VerifyState, x, w):
    """The GEMM of every weight. natural / exact: ONE call over the E rows. rowgroup: R calls, each over the B rows of
    one row index, with the shipped call's shape (B, 1, K)."""
    ops = vs.ops
    vs.calls["gemm"] += 1 if ops.gemm != "rowgroup" else vs.R            # host count (WS-B: invocation counts)
    if ops.gemm != "rowgroup":
        return ops.linear(x, w)
    B, R = vs.B, vs.R
    xs = x.reshape((B, R) + tuple(x.shape[1:]))
    outs = [ops.linear(xs[:, r].contiguous(), w) for r in range(R)]
    y = torch.stack(outs, 1)
    return y.reshape((B * R,) + tuple(y.shape[2:]))


def _twin_rows(vs: VerifyState, twin, k4, c3):
    """The twin advances R times; row r records n_used = n_comp after its step and M_b = pos_before // 64 + 1 (the
    ragged engine's _lengths, per row). Returns pos_before of every row (a list of R (B,) int64 tensors)."""
    st = vs.rs.st
    pos_before = []
    for r in range(vs.R):
        pb = twin.pos.clone()
        twin.step_unchecked(k4[:, r].contiguous(), c3[:, r].contiguous(), st.active)
        vs.n_used[:, r].copy_(twin.n_comp)
        vs.M_b[:, r].copy_(pb // BLOCK + 1)
        pos_before.append(pb)
    if vs.ops.neg == "shared_row_lengths":                          # NEGATIVE CONTROL: row 0's lengths for every row
        vs.n_used.copy_(vs.n_used[:, :1].clone().expand(vs.B, vs.R))
        vs.M_b.copy_(vs.M_b[:, :1].clone().expand(vs.B, vs.R))
    return pos_before


def _select_rows(vs: VerifyState, layer, q, twin):
    """ONE batched selection of the E rows (module docstring, step 3). Returns the selections (H, B, R, topk) int64,
    a view of vs.ti (row r of request b at [:, b, r])."""
    rs, ops = vs.rs, vs.ops
    H, B, R, E = vs.H, vs.B, vs.R, vs.E
    ck = twin.comp_k_flat()
    n_used, M_b = vs.n_used.view(E), vs.M_b.view(E)
    score = stage1_seqused(q, ck, ck, vs.cu_q, vs.cu_k, 1, twin.lk_max, seqused_k=n_used,
                           ext=rs.ops.kernels.get("stage1_ext"))
    vs.ccis_rows.copy_(twin.ccis.unsqueeze(2).expand(H, B, R, twin.lk_max))
    init = 0 if ops.neg == "no_init_force" else layer.init_blocks
    nosa_pooling_per_request(score, vs.ccis_rows.view(H, E, twin.lk_max), n_used, M_b, vs.pq, vs.pc,
                             stride=layer.pooling_stride, block_size=layer.block_size,
                             local_blocks=layer.window_blocks, init_blocks=init)
    torch.topk(vs.pq, rs.nl.QK_SELECT, dim=-1, sorted=False, out=(vs.tv_q, vs.ti_q))
    vs.mask.zero_()
    vs.mask.scatter_(2, vs.ti_q, True)
    vs.pc.masked_fill_(vs.mask, float("inf"))
    torch.topk(vs.pc, layer.topk_blocks, dim=-1, sorted=False, out=(vs.tv, vs.ti))
    return vs.ti.view(H, B, R, layer.topk_blocks)


def _probe_row0(vs: VerifyState, layer, q, twin, sel) -> None:
    """The ONE-ROW selection of row 0 (B entries: the ragged engine's buffers and its stage-1 / pooling call), for the
    cell's timing (selection of R rows against 1 row) and its SEL0 gate: the batched selection's row 0 against it,
    in order (the EXACT premise) and as a set (correctness). Device counters only."""
    rs = vs.rs
    st = rs.st
    B, R, H = vs.B, vs.R, vs.H
    rs.n_used.copy_(vs.n_used[:, 0])
    rs.M_b.copy_(vs.M_b[:, 0])
    q0 = q.view(B, R, q.shape[1], q.shape[2])[:, 0].contiguous()
    ck = twin.comp_k_flat()
    score = stage1_seqused(q0, ck, ck, st.cu_q, twin.cu, 1, twin.lk_max, seqused_k=rs.n_used,
                           ext=rs.ops.kernels.get("stage1_ext"))
    nosa_pooling_per_request(score, twin.ccis, rs.n_used, rs.M_b, st.pq, st.pc, stride=layer.pooling_stride,
                             block_size=layer.block_size, local_blocks=layer.window_blocks,
                             init_blocks=layer.init_blocks)
    torch.topk(st.pq, rs.nl.QK_SELECT, dim=-1, sorted=False, out=(st.tv_q, st.ti_q))
    st.mask.zero_()
    st.mask.scatter_(2, st.ti_q, True)
    st.pc.masked_fill_(st.mask, float("inf"))
    torch.topk(st.pc, layer.topk_blocks, dim=-1, sorted=False, out=(st.tv, st.ti))
    s0 = sel[:, :, 0]
    vs.stat["sel0_order_eq"] += (st.ti == s0).all().to(torch.int64)
    vs.stat["sel0_set_eq"] += (st.ti.sort(-1).values == s0.sort(-1).values).all().to(torch.int64)
    vs.stat_n["sel0"] += 1


def _union(vs: VerifyState, sel):
    """The union of the R rows' selections per (head, request): the sorted ids and the first-occurrence mask (ids >=
    0); the union size goes into the device statistics."""
    H, B, R, T = sel.shape
    s = torch.sort(sel.reshape(H, B, R * T), dim=-1).values
    first = torch.ones_like(s, dtype=torch.bool)
    first[..., 1:] = s[..., 1:] != s[..., :-1]
    first &= s >= 0
    if vs.ops.stats:
        n = first.sum(-1)
        vs.stat["union_sum"] += n.sum()
        vs.stat["union_max"] = torch.maximum(vs.stat["union_max"], n.max())
        vs.stat_n["union"] += H * B
    return s, first


# ------------------------------------------------------------------ EXACT: the serial residency per row
def _exact_rows(vs: VerifyState, l: int, layer, eng, twin, q, k4, v4, sel, pos_before):
    rs, ops = vs.rs, vs.ops
    kern = rs.ops.kernels
    B, R = vs.B, vs.R
    q4 = q.view(B, R, q.shape[1], q.shape[2])
    diff = RE._diff_pool(rs, eng) if rs.pool else kern["diff"]
    outs = []
    for r in range(R):
        ti_r = sel[:, :, r].contiguous()
        _mark(vs, l, "resid")                                          # WS-B: row r's residency (probe passes only)
        key_states, value_states, kv_bias, cache_lens = rs.body.decode_fixed(
            eng, k4[:, r].unsqueeze(1).contiguous(), v4[:, r].unsqueeze(1).contiguous(), twin.total_cis, ti_r,
            pos_before[r], rs.st.active, rs.err, diff, kern["gather_k"], kern["gather_v_bias"])
        if rs.pool:
            key_states, value_states, kv_bias = eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att
        if ops.record:
            vs.log.append(dict(kind="exact", layer=l, row=r, sel=ti_r.clone(), win=eng._block_map.clone(),
                               pos=pos_before[r].clone()))
        _mark(vs, l, "attn")                                           # WS-B: row r's attention (probe passes only)
        o = ops.attn(q4[:, r].unsqueeze(1).contiguous(), key_states, value_states, kv_bias, cache_seqlens=cache_lens,
                     num_splits=rs.nl._ATTN_SPLITS)
        outs.append(o.reshape(B, -1))
        vs.calls["resid"] += 1                                       # host counts (WS-B): one residency, two host
        vs.calls["gather_host"] += 2                                 # gathers and one attention call per row
        vs.calls["attn"] += 1
    return torch.stack(outs, 1)                                        # (B, R, Hq * D)


# ------------------------------------------------------------------ FAST: one residency, extras, masked attention
def _extras(vs: VerifyState, eng, twin, sel, m, crossing):
    """The extras of rows 1 .. K: the union's blocks that the window does not hold, minus the next block (it has no
    host rows yet). Up to X - 1 per (head, request), in id order; more are COUNTED (stat overflow), never silent.
    A pool hit is copied device to device into its extras slot; a miss is fetched with the shipped gathers. Returns
    the extras ids (H, B, X): slot X - 1 = the next block id where the group crosses a block edge, else -1."""
    rs = vs.rs
    kern = rs.ops.kernels
    H, B, X, bs, topk = vs.H, vs.B, vs.X, vs.bs, vs.topk
    s, first = _union(vs, sel[:, :, :vs.R - 1] if vs.ops.draft_last else sel)   # draft_last: the verify rows' union
    in_win = (s.unsqueeze(-1) == eng._block_map.unsqueeze(-2)).any(-1)        # (H, B, R * topk)
    nxt = (m + 1).view(1, B, 1)
    extra = first & ~in_win & (s != nxt)
    rank = extra.to(torch.int64).cumsum(-1) - 1
    keep = extra & (rank < X - 1)
    n_extra = extra.to(torch.int64).sum(-1)                                   # (H, B)
    vs.stat["extras"] += n_extra.sum()
    vs.stat["overflow"] += (n_extra - (X - 1)).clamp(min=0).sum()
    buf = vs.ids_buf
    buf.fill_(-1)
    buf.scatter_(-1, torch.where(keep, rank, torch.full_like(rank, X)), torch.where(keep, s, torch.full_like(s, -1)))
    buf[..., X - 1] = torch.where(crossing.view(1, B), (m + 1).view(1, B), torch.full_like(buf[..., X - 1], -1))
    ids = buf[..., :X]
    head = ids[..., :X - 1]                                                   # (H, B, X - 1)
    if rs.pool:
        pm = eng._pool_map                                                    # (H, B, P)
        hit = (head.unsqueeze(-1) == pm.unsqueeze(-2)) & (head >= 0).unsqueeze(-1)
        has = hit.any(-1)
        qslot = hit.to(torch.int64).argmax(-1)
    else:
        has = torch.zeros_like(head, dtype=torch.bool)
        qslot = torch.zeros_like(head)
    vs.load_x[..., :X - 1] = torch.where(has, torch.full_like(head, -1), head)
    vs.load_x[..., X - 1] = -1
    if rs.pool:                                                               # pool hits: device to device
        nslot = int(eng._k_gpu.shape[1]) // bs
        src = (topk + qslot).clamp(max=nslot - 1)
        hh, bb, xx = vs.hh, vs.bb, vs.xx
        for gpu, ext in ((eng._k_gpu, vs.ext_k), (eng._v_gpu, vs.ext_v)):
            g5 = gpu.view(B, nslot, bs, H, vs.D)
            e5 = ext.view(B, X, bs, H, vs.D)
            e5[bb, xx, :, hh, :] = torch.where(has[..., None, None], g5[bb, src, :, hh, :], e5[bb, xx, :, hh, :])
        g4 = eng._kv_bias_gpu.view(B, nslot, bs, H)
        e4 = vs.ext_bias.view(B, X, bs, H)
        e4[bb, xx, :, hh] = torch.where(has[..., None], g4[bb, src, :, hh], e4[bb, xx, :, hh])
    kern["gather_k"](vs.ext_k, eng._k_cpu, vs.load_x, bs)                    # misses: the shipped gathers
    kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.load_x, bs)
    return ids


def _fast_rows(vs: VerifyState, l: int, layer, eng, twin, q, k4, v4, sel, pos_before):
    rs, ops = vs.rs, vs.ops
    kern = rs.ops.kernels
    B, R, E, X, H, D, bs, topk = vs.B, vs.R, vs.E, vs.X, vs.H, vs.D, vs.bs, vs.topk
    Hq = q.shape[1]
    tail0 = (topk - 1) * bs
    p0 = pos_before[0]
    if vs.hooks is not None:                                                  # Phase 2: the rollback's FAST journal
        vs.hooks.fast_begin(l, eng, R)
    p_rows = p0.view(B, 1) + vs.offs.view(1, R)                              # (B, R)
    m = p0 // bs                                                              # the tail block at the start
    in_tail = (p_rows // bs) == m.view(B, 1)
    crossing = (~in_tail).any(1)                                              # (B,)
    row_in = p_rows % bs
    bi = vs.bidx
    kd, vd, bd = eng._k_gpu.dtype, eng._v_gpu.dtype, eng._kv_bias_gpu.dtype
    k4c, v4c = k4.to(kd), v4.to(vd)
    cis_rows = twin.total_cis[bi, p_rows].to(bd)                              # (B, R, H): written by the twin
    # 1. the tentative rows: the tail block's into slot topk - 1, the next block's into the last extras slot
    r63 = tail0 + row_in
    it4, it3 = in_tail.view(B, R, 1, 1), in_tail.view(B, R, 1)
    eng._k_gpu[bi, r63] = torch.where(it4, k4c, eng._k_gpu[bi, r63])
    eng._v_gpu[bi, r63] = torch.where(it4, v4c, eng._v_gpu[bi, r63])
    eng._kv_bias_gpu[bi, r63] = torch.where(it3, cis_rows, eng._kv_bias_gpu[bi, r63])
    rx = (X - 1) * bs + row_in
    vs.ext_k[bi, rx] = torch.where(it4, vs.ext_k[bi, rx], k4c.to(vs.ext_k.dtype))
    vs.ext_v[bi, rx] = torch.where(it4, vs.ext_v[bi, rx], v4c.to(vs.ext_v.dtype))
    vs.ext_bias[bi, rx] = torch.where(it3, vs.ext_bias[bi, rx], cis_rows.to(vs.ext_bias.dtype))
    # 2. the residency of row 0's selection: the shipped diff, victim pool, map commit and gathers
    diff = RE._diff_pool(rs, eng) if rs.pool else kern["diff"]
    diff(eng._block_map, sel[:, :, 0].contiguous(), eng._new_block_map_buf, eng._load_mask)
    eng._block_map.copy_(eng._new_block_map_buf)
    kern["gather_k"](eng._k_gpu, eng._k_cpu, eng._load_mask, bs)
    kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, eng._load_mask, bs)
    # 3. the extras of rows 1 .. K
    ids = _extras(vs, eng, twin, sel, m, crossing)
    if ops.record:
        vs.log.append(dict(kind="fast", layer=l, sel=sel.clone(), win=eng._block_map.clone(), ids=ids.clone(),
                           m=m.clone(), in_tail=in_tail.clone(), p_rows=p_rows.clone(),
                           k_gpu=eng._k_gpu.clone(), v_gpu=eng._v_gpu.clone(), bias_gpu=eng._kv_bias_gpu.clone(),
                           ext_k=vs.ext_k.clone(), ext_v=vs.ext_v.clone(), ext_bias=vs.ext_bias.clone(),
                           k_cpu=eng._k_cpu.clone(), v_cpu=eng._v_cpu.clone(), total_cis=twin.total_cis.clone()))
    _mark(vs, l, "attn")
    # 4. call A: the window, a bias row set per entry (row r's selection), the causal cut by cache_seqlens
    win = eng._block_map                                                      # (H, B, topk); slot topk - 1 = m
    vis = (win.unsqueeze(2).unsqueeze(-1) == sel.unsqueeze(3)).any(-1)        # (H, B, R, topk)
    vis[..., topk - 1] = True
    if ops.neg == "no_row_mask":
        vis.fill_(True)
    if rs.pool:
        katt, vatt, batt = eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att
    else:
        katt, vatt, batt = eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu
    rows_a = topk * bs
    m_a = vis.permute(1, 2, 3, 0).unsqueeze(3).expand(B, R, topk, bs, H).reshape(B, R, rows_a, H)
    bias_a = torch.where(m_a, batt.unsqueeze(1), torch.full((), MASK_BIAS, dtype=batt.dtype, device=batt.device))
    bias_a = bias_a.reshape(E, rows_a, H)
    lens_tail = rs.body.cache_lens_value(eng, p_rows)                         # the body's length of position p + r
    if ops.neg == "causal_leak":                                              # NEGATIVE CONTROL: the last tail row's
        last = torch.where(in_tail, p_rows, torch.full_like(p_rows, -1)).amax(1, keepdim=True)
        lens_tail = rs.body.cache_lens_value(eng, last.expand(B, R))
    lens_a = torch.where(in_tail, lens_tail, torch.full_like(lens_tail, rows_a)).to(torch.int32).reshape(E)
    q_e = q.reshape(E, 1, Hq, q.shape[2])
    nsplit = rs.nl._ATTN_SPLITS
    o_a, lse_a = ops.attn(q_e, katt, vatt, bias_a[:B], cache_seqlens=lens_a, cache_batch_idx=vs.cbi,
                          num_splits=nsplit, return_softmax_lse=True)
    if vs.fa_check is not None and l == 0 and "equal" not in vs.fa_check:
        _fa_index_check(vs, q_e, katt, vatt, bias_a, lens_a, o_a)
    # 5. call B: the extras store, row r's selected extras and the next-block rows at or before its own position
    vis_x = (ids[..., :X - 1].unsqueeze(2).unsqueeze(-1) == sel.unsqueeze(3)).any(-1)   # (H, B, R, X - 1)
    vis_x &= (ids[..., :X - 1] >= 0).unsqueeze(2)
    if ops.neg == "no_extras":
        vis_x.fill_(False)
    m_x = vis_x.permute(1, 2, 3, 0).unsqueeze(3).expand(B, R, X - 1, bs, H).reshape(B, R, (X - 1) * bs, H)
    m_n = (vs.ar_bs.view(1, 1, bs) <= row_in.view(B, R, 1)) & (~in_tail).view(B, R, 1)     # (B, R, bs)
    m_b = torch.cat([m_x, m_n.unsqueeze(-1).expand(B, R, bs, H)], dim=2)                   # (B, R, X * bs, H)
    bias_b = torch.where(m_b, vs.ext_bias.unsqueeze(1),
                         torch.full((), MASK_BIAS, dtype=vs.ext_bias.dtype, device=vs.ext_bias.device))
    bias_b = bias_b.reshape(E, X * bs, H)
    o_b, lse_b = ops.attn(q_e, vs.ext_k, vs.ext_v, bias_b[:B], cache_seqlens=vs.lens_x, cache_batch_idx=vs.cbi,
                          num_splits=nsplit, return_softmax_lse=True)
    vs.calls["resid"] += 1                                                    # host counts (WS-B)
    vs.calls["attn"] += 2
    vs.calls["gather_host"] += 4
    # 6. the merge of the two partial softmaxes by their log-sum-exp (float32)
    la = lse_a.reshape(E, Hq, 1).float()
    lb = lse_b.reshape(E, Hq, 1).float()
    mx = torch.maximum(la, lb)
    wa, wb = torch.exp(la - mx), torch.exp(lb - mx)
    o = (o_a.reshape(E, Hq, -1).float() * wa + o_b.reshape(E, Hq, -1).float() * wb) / (wa + wb)
    attn = o.to(q.dtype).reshape(B, R, -1)
    _mark(vs, l, "end")
    # 7. the end state: write-back of a filled block, the next block's rows into the tail slot, tail id, length
    p_last = p0 + (R - 1)
    vs.full.copy_((p0 % bs) + (R - 1) >= bs - 1)
    vs.wpos.copy_(m * bs + (bs - 1))
    # the body's write-back callable: the same kernel as kern["writeback"] (RaggedOps.make_body), or the rollback's
    # deferred write-back while a group is open (Phase 2)
    rs.body.writeback_kernel(eng._k_gpu, eng._v_gpu, eng._k_cpu, eng._v_cpu, vs.full, vs.wpos, tail0, bs)
    eng._k_gpu[bi, r63] = torch.where(it4, eng._k_gpu[bi, r63], k4c)
    eng._v_gpu[bi, r63] = torch.where(it4, eng._v_gpu[bi, r63], v4c)
    eng._kv_bias_gpu[bi, r63] = torch.where(it3, eng._kv_bias_gpu[bi, r63], cis_rows)
    eng._block_map[..., topk - 1] = ((p_last + 1) // bs).view(1, B).expand(H, B)
    eng._cache_lens.copy_(rs.body.cache_lens_value(eng, p_last).to(torch.int32))
    if vs.hooks is not None:
        vs.hooks.fast_end(l, eng, p0, vs.full, R)
    return attn


def _fa_index_check(vs: VerifyState, q_e, katt, vatt, bias_a, lens_a, o_a) -> None:
    """VRF-IDX: sampled entries of call A against a PLAIN one-entry call (no cache_batch_idx, the entry's own bias
    row, the same split count). Equal bits mean the kernel read the entry's bias row and the request's K / V. One
    host sync (once per run)."""
    E, R = vs.E, vs.R
    picks = sorted({0, E // 3, (2 * E) // 3, E - 1})
    eq = []
    for e in picks:
        b = e // R
        o1 = vs.ops.attn(q_e[e:e + 1], katt[b:b + 1], vatt[b:b + 1], bias_a[e:e + 1],
                         cache_seqlens=lens_a[e:e + 1].contiguous(), num_splits=vs.rs.nl._ATTN_SPLITS)
        eq.append(bool(torch.equal(o1, o_a[e:e + 1])))
    vs.fa_check.update(entries=picks, equal=eq)


# ------------------------------------------------------------------ UNION (WS-B): one residency, the union store
def _row_masks(vs: VerifyState):
    """The rows of this pass: full (R, B) bool (a verify row: its whole selection must be present) and res (R, B) bool
    (a resident row: its pool hits are brought, its misses are skipped). Defaults: vs.full_default and ~full."""
    full = vs.full_rows if vs.full_rows is not None else vs.full_default
    res = vs.res_rows if vs.res_rows is not None else ~full
    return full, res


def _union_list(vs: VerifyState, eng, sel, full, res, m):
    """THE UNION LIST of one layer, after the window residency (module docstring, step 3). sel (H, B, R, T); m (B,)
    the tail block. Returns lst (H, B, cap): the listed ids in id order (-1 past the end); src_dev (H, B, cap): the
    engine slot topk + q of a pool hit, else -1; src_host (H, B, cap): the id of a host miss, else -1."""
    rs = vs.rs
    H, B, R, T = sel.shape
    cap, topk = vs.cap, vs.topk
    cand = torch.where(full.t().reshape(1, B, R, 1), sel, vs.minus1)          # every block of a full row
    if rs.pool:
        pm = eng._pool_map                                                     # (H, B, P), after the residency
        hit = (sel.unsqueeze(-1) == pm.unsqueeze(2).unsqueeze(2)).any(-1) & (sel >= 0)
        rr = res.t().reshape(1, B, R, 1)
        cand = torch.cat((cand, torch.where(rr & hit, sel, vs.minus1)), dim=2)  # a resident row's pool hits
    s = torch.sort(cand.reshape(H, B, -1), dim=-1).values
    first = torch.ones_like(s, dtype=torch.bool)
    first[..., 1:] = s[..., 1:] != s[..., :-1]
    first &= s >= 0
    in_win = (s.unsqueeze(-1) == eng._block_map.unsqueeze(-2)).any(-1)
    extra = first & ~in_win & (s != (m + 1).view(1, B, 1))
    rank = extra.to(torch.int64).cumsum(-1) - 1
    keep = extra & (rank < cap)
    buf = vs.lst_buf
    buf.fill_(-1)
    buf.scatter_(-1, torch.where(keep, rank, torch.full_like(rank, cap)), torch.where(keep, s, torch.full_like(s, -1)))
    lst = buf[..., :cap]
    if rs.pool:
        pm = eng._pool_map
        hl = (lst.unsqueeze(-1) == pm.unsqueeze(-2)) & (lst >= 0).unsqueeze(-1)  # (H, B, cap, P)
        has = hl.any(-1)
        src_dev = torch.where(has, hl.to(torch.int64).argmax(-1) + topk, vs.minus1)
        src_host = torch.where(has, vs.minus1, lst)
    else:
        has = torch.zeros_like(lst, dtype=torch.bool)
        src_dev = torch.full_like(lst, -1)
        src_host = lst
    if vs.ops.stats:                                                           # accounting (device counters)
        st, X = vs.stat, vs.X
        n_extra = extra.to(torch.int64).sum(-1)                                # (H, B)
        st["extras"] += n_extra.sum()
        st["extras_max"] = torch.maximum(st["extras_max"], n_extra.max())
        st["extras_pool"] += has.sum()
        st["extras_host"] += (src_host >= 0).sum()
        st["spill"] += (n_extra.clamp(max=cap) - (X - 1)).clamp(min=0).sum()
        st["unserved"] += (n_extra - cap).clamp(min=0).sum()
        n_u = first.to(torch.int64).sum(-1)
        st["union_sum"] += n_u.sum()
        st["union_max"] = torch.maximum(st["union_max"], n_u.max())
        vs.stat_n["union"] += H * B
    return lst, src_dev, src_host


def _store_fill(vs: VerifyState, eng, twin, src_dev, src_host, lo: int, hi: int) -> None:
    """Store fill: slots 1 .. X - 1 <- the listed ids lo .. hi - 1 (slot 0, the next block, is never written here).
    The SHIPPED gathers: a pool hit from the engine's own slot rows (the slot index as the "block id"), a miss from the
    host rows (the urgent GPU gather; the bias from twin.total_cis)."""
    kern, bs = vs.rs.ops.kernels, vs.bs
    if vs.rs.pool:
        vs.dev_x[..., 1:].copy_(src_dev[..., lo:hi])
        kern["gather_k"](vs.ext_k, eng._k_gpu, vs.dev_x, bs)
        kern["gather_v_bias"](vs.ext_v, eng._v_gpu, vs.ext_bias, eng._kv_bias_gpu, vs.dev_x, bs)
        vs.calls["gather_dev"] += 2
    vs.host_x[..., 1:].copy_(src_host[..., lo:hi])
    kern["gather_k"](vs.ext_k, eng._k_cpu, vs.host_x, bs)
    kern["gather_v_bias"](vs.ext_v, eng._v_cpu, vs.ext_bias, twin.total_cis, vs.host_x, bs)
    vs.calls["gather_host"] += 2
    vs.calls["fills"] += 1


def _store_call(vs: VerifyState, c: int, lst_c, sel_sorted, q_e, row_in, in_tail):
    """One attention call over the store after fill c: row r reads slot x >= 1 where the listed id is in its selection,
    and (fill 0 only, a row past the block edge) slot 0's rows at or before its own position. The per-entry length is
    the highest slot the row reads (0: nothing; the kernel exits early, out 0 and lse +inf). Returns (o (E, Hq, D)
    float, lse (E, Hq, 1) float, lens (E,) int32, vis (H, B, R, X - 1) bool); an entry with length 0 gives o 0 and
    lse -inf (weight 0)."""
    H, B, R, T = sel_sorted.shape
    X, bs, E = vs.X, vs.bs, vs.E
    Hq = q_e.shape[2]
    ids = lst_c.unsqueeze(2).expand(H, B, R, X - 1).contiguous()
    at = torch.searchsorted(sel_sorted, ids).clamp(max=T - 1)
    vis = (sel_sorted.gather(-1, at) == ids) & (ids >= 0)                    # (H, B, R, X - 1)
    if vs.ops.neg == "union_attention":                                       # NEGATIVE CONTROL: every listed block
        vis = ids >= 0
    m_x = vis.permute(1, 2, 3, 0).unsqueeze(3).expand(B, R, X - 1, bs, H).reshape(B, R, (X - 1) * bs, H)
    top = torch.where(vis, vs.ar_x1, vs.zero_i).amax(-1).amax(0)              # (B, R): the highest slot read
    if c == 0:
        m_n = (vs.ar_bs.view(1, 1, bs) <= row_in.view(B, R, 1)) & (~in_tail).view(B, R, 1)
        top = torch.where(in_tail, top, top.clamp(min=1))
    else:
        m_n = torch.zeros((B, R, bs), dtype=torch.bool, device=vis.device)
    m_b = torch.cat([m_n.unsqueeze(-1).expand(B, R, bs, H), m_x], dim=2)    # (B, R, X * bs, H): slot 0 first
    eb = vs.ext_bias
    bias = torch.where(m_b, eb.unsqueeze(1), torch.full((), MASK_BIAS, dtype=eb.dtype, device=eb.device))
    bias = bias.reshape(E, X * bs, H)
    lens = (top * bs).to(torch.int32).reshape(E)
    o, lse = vs.ops.attn(q_e, vs.ext_k, vs.ext_v, bias[:B], cache_seqlens=lens, cache_batch_idx=vs.cbi,
                         num_splits=vs.rs.nl._ATTN_SPLITS, return_softmax_lse=True)
    vs.calls["attn"] += 1
    if vs.fa_check is not None and c == 0 and "B" not in vs.fa_check:
        nz = (lens > 0).nonzero().view(-1).tolist()          # one host read per layer until a layer has store entries
        if nz:
            _fa_check_u(vs, "B", q_e, vs.ext_k, vs.ext_v, bias, lens, o, sorted({nz[0], nz[len(nz) // 2], nz[-1]}))
    has = (lens > 0).view(E, 1, 1)
    o = torch.where(has, o.reshape(E, Hq, -1).float(), torch.zeros((), dtype=torch.float32, device=o.device))
    lse = torch.where(has, lse.reshape(E, Hq, 1).float(),
                      torch.full((), float("-inf"), dtype=torch.float32, device=lse.device))
    if vs.ops.record:
        vs._rec_fills.append(dict(c=c, lst=lst_c.clone(), ext_k=vs.ext_k.clone(), ext_v=vs.ext_v.clone(),
                                  ext_bias=vs.ext_bias.clone(), lens=lens.clone(), vis=vis.clone()))
    return o, lse, lens, vis


def _fa_check_u(vs: VerifyState, tag: str, q_e, kc, vc, bias, lens, o, picks) -> None:
    """VRF-IDX of mode union (tag "A": the window call; "B": the first store call): sampled entries against a PLAIN
    one-entry call (no cache_batch_idx; the entry's own bias row; the K / V of its request). One host sync, once."""
    eq = []
    for e in picks:
        b = e // vs.R
        o1 = vs.ops.attn(q_e[e:e + 1], kc[b:b + 1], vc[b:b + 1], bias[e:e + 1],
                         cache_seqlens=lens[e:e + 1].contiguous(), num_splits=vs.rs.nl._ATTN_SPLITS)
        eq.append(bool(torch.equal(o1, o[e:e + 1])))
    vs.fa_check[tag] = dict(entries=list(picks), equal=eq)


def _presence(vs: VerifyState, eng, sel, lst, m):
    """(H, B, R, T) bool: each selected id is in the window, in the union list (the ids that the store fills serve), the
    tail block m or the next block m + 1 (store slot 0). Sorted search; no host read."""
    H, B, R, T = sel.shape
    mm = m.view(1, B, 1).expand(H, B, 1)
    ps = torch.sort(torch.cat((eng._block_map, lst, mm, mm + 1), dim=2), dim=-1).values.contiguous()
    v = sel.reshape(H, B, R * T).contiguous()
    at = torch.searchsorted(ps, v).clamp(max=ps.shape[-1] - 1)
    return (ps.gather(-1, at) == v).view(H, B, R, T)


def _union_rows(vs: VerifyState, l: int, layer, eng, twin, q, k4, v4, sel, pos_before):
    """Mode "union", one layer (module docstring, WS-B, steps 1-6). The phase marks (probe passes): "union" after the
    residency, "fetch" (fill 0), "attn" (call A, the fill-0 call and its merge step), "spill" (fills 1 .. C - 1, their
    calls and merge steps), "merge" (the final division), "check" (CHECK-ONLY and the rung-L5 seams), "end"."""
    rs, ops = vs.rs, vs.ops
    kern = rs.ops.kernels
    B, R, E, X, H, D, bs, topk = vs.B, vs.R, vs.E, vs.X, vs.H, vs.D, vs.bs, vs.topk
    Hq = q.shape[1]
    tail0 = (topk - 1) * bs
    p0 = pos_before[0]
    if vs.hooks is not None:                                                  # the rollback's FAST journal
        vs.hooks.fast_begin(l, eng, R)
    full, res = _row_masks(vs)
    p_rows = p0.view(B, 1) + vs.offs.view(1, R)                              # (B, R)
    m = p0 // bs                                                              # the tail block at the start
    in_tail = (p_rows // bs) == m.view(B, 1)
    row_in = p_rows % bs
    bi = vs.bidx
    kd, vd, bd = eng._k_gpu.dtype, eng._v_gpu.dtype, eng._kv_bias_gpu.dtype
    k4c, v4c = k4.to(kd), v4.to(vd)
    cis_rows = twin.total_cis[bi, p_rows].to(bd)                              # (B, R, H): written by the twin
    # 1. the tentative rows: the tail block's into slot topk - 1, the next block's into store slot 0
    r63 = tail0 + row_in
    it4, it3 = in_tail.view(B, R, 1, 1), in_tail.view(B, R, 1)
    eng._k_gpu[bi, r63] = torch.where(it4, k4c, eng._k_gpu[bi, r63])
    eng._v_gpu[bi, r63] = torch.where(it4, v4c, eng._v_gpu[bi, r63])
    eng._kv_bias_gpu[bi, r63] = torch.where(it3, cis_rows, eng._kv_bias_gpu[bi, r63])
    vs.ext_k[bi, row_in] = torch.where(it4, vs.ext_k[bi, row_in], k4c.to(vs.ext_k.dtype))
    vs.ext_v[bi, row_in] = torch.where(it4, vs.ext_v[bi, row_in], v4c.to(vs.ext_v.dtype))
    vs.ext_bias[bi, row_in] = torch.where(it3, vs.ext_bias[bi, row_in], cis_rows.to(vs.ext_bias.dtype))
    # 2. THE RESIDENCY (one per layer and pass): row 0's selection, the shipped diff, victim pool, map commit, gathers
    diff = RE._diff_pool(rs, eng) if rs.pool else kern["diff"]
    diff(eng._block_map, sel[:, :, 0].contiguous(), eng._new_block_map_buf, eng._load_mask)
    eng._block_map.copy_(eng._new_block_map_buf)
    kern["gather_k"](eng._k_gpu, eng._k_cpu, eng._load_mask, bs)
    kern["gather_v_bias"](eng._v_gpu, eng._v_cpu, eng._kv_bias_gpu, twin.total_cis, eng._load_mask, bs)
    vs.calls["resid"] += 1
    vs.calls["gather_host"] += 2
    if ops.stats:                       # accounting: row roles; the window's READY part and its RESIDUAL host loads
        vs.stat["rows_full"] += full.sum()
        vs.stat["rows_res"] += res.sum()
        vs.stat["win_sel"] += (sel[:, :, 0] >= 0).sum()
        vs.stat["win_host"] += (eng._load_mask >= 0).sum()                   # after pool_update: the host misses
    _mark(vs, l, "union")
    # 3. the union list (every full row's blocks and the resident rows' pool hits, minus the window and m + 1)
    lst, src_dev, src_host = _union_list(vs, eng, sel, full, res, m)
    sel_sorted = torch.sort(sel, dim=-1).values.contiguous()                 # (H, B, R, T): the rows' sets
    _mark(vs, l, "fetch")
    # 4. store fill 0 (the extra store)
    _store_fill(vs, eng, twin, src_dev, src_host, 0, X - 1)
    if ops.record:
        vs._rec_fills = []
        vs.log.append(dict(kind="union", layer=l, sel=sel.clone(), win=eng._block_map.clone(), lst=lst.clone(),
                           m=m.clone(), in_tail=in_tail.clone(), p_rows=p_rows.clone(), full=full.clone(),
                           res=res.clone(), pool_map=(eng._pool_map.clone() if rs.pool else None),
                           k_gpu=eng._k_gpu.clone(), v_gpu=eng._v_gpu.clone(), bias_gpu=eng._kv_bias_gpu.clone(),
                           k_cpu=eng._k_cpu.clone(), v_cpu=eng._v_cpu.clone(), total_cis=twin.total_cis.clone(),
                           fills=vs._rec_fills))
    _mark(vs, l, "attn")
    # 5. call A: the window, a bias row set per entry (row r's selection), the causal cut by cache_seqlens (as FAST)
    win = eng._block_map                                                      # (H, B, topk); slot topk - 1 = m
    winR = win.unsqueeze(2).expand(H, B, R, topk).contiguous()
    at = torch.searchsorted(sel_sorted, winR).clamp(max=topk - 1)
    vis = (sel_sorted.gather(-1, at) == winR) & (winR >= 0)                  # (H, B, R, topk)
    if ops.neg == "union_attention":                                          # NEGATIVE CONTROL: every window block
        vis = winR >= 0
    vis[..., topk - 1] = True
    if rs.pool:
        katt, vatt, batt = eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att
    else:
        katt, vatt, batt = eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu
    rows_a = topk * bs
    m_a = vis.permute(1, 2, 3, 0).unsqueeze(3).expand(B, R, topk, bs, H).reshape(B, R, rows_a, H)
    bias_a = torch.where(m_a, batt.unsqueeze(1), torch.full((), MASK_BIAS, dtype=batt.dtype, device=batt.device))
    bias_a = bias_a.reshape(E, rows_a, H)
    lens_tail = rs.body.cache_lens_value(eng, p_rows)                         # the body's length of position p + r
    lens_a = torch.where(in_tail, lens_tail, torch.full_like(lens_tail, rows_a)).to(torch.int32).reshape(E)
    q_e = q.reshape(E, 1, Hq, q.shape[2])
    nsplit = rs.nl._ATTN_SPLITS
    o_a, lse_a = ops.attn(q_e, katt, vatt, bias_a[:B], cache_seqlens=lens_a, cache_batch_idx=vs.cbi,
                          num_splits=nsplit, return_softmax_lse=True)
    vs.calls["attn"] += 1
    if vs.fa_check is not None and l == 0 and "A" not in vs.fa_check:
        _fa_check_u(vs, "A", q_e, katt, vatt, bias_a, lens_a, o_a, sorted({0, E // 3, (2 * E) // 3, E - 1}))
    # THE MERGE runs after each call (log-sum-exp, float32): num / den / mx; a part of length 0 has the weight 0, so an
    # entry that reads only the window keeps call A's output exactly (num * 1 + 0, den 1)
    num = o_a.reshape(E, Hq, -1).float()
    mx = lse_a.reshape(E, Hq, 1).float()
    den = torch.ones_like(mx)

    def merge(num, den, mx, o_c, l_c):
        new = torch.maximum(mx, l_c)
        a, b_ = torch.exp(mx - new), torch.exp(l_c - new)
        return num * a + o_c * b_, den * a + b_, new
    # CHECK-ONLY: the blocks each row SEES (mask level): window slots, store slots, slot 0 past the block edge
    vis_cnt = vis.sum(-1) + (~in_tail).view(1, B, R).to(torch.int64) if ops.check else None
    o_c, l_c, lens_c, vis_c = _store_call(vs, 0, lst[..., :X - 1], sel_sorted, q_e, row_in, in_tail)
    num, den, mx = merge(num, den, mx, o_c, l_c)
    used = (lens_c > 0).any().to(torch.int64) if ops.stats else None
    streamed = lens_a.to(torch.int64).sum() + lens_c.to(torch.int64).sum() if ops.stats else None
    if vis_cnt is not None:
        vis_cnt = vis_cnt + vis_c.sum(-1)
    _mark(vs, l, "spill")
    # the SPILL: fills 1 .. C - 1 of the same store (counted in _union_list; never a drop)
    for c in range(1, vs.C):
        lo, hi = c * (X - 1), (c + 1) * (X - 1)
        _store_fill(vs, eng, twin, src_dev, src_host, lo, hi)
        o_c, l_c, lens_c, vis_c = _store_call(vs, c, lst[..., lo:hi], sel_sorted, q_e, row_in, in_tail)
        num, den, mx = merge(num, den, mx, o_c, l_c)
        if used is not None:
            used = used + (lens_c > 0).any().to(torch.int64)
            streamed = streamed + lens_c.to(torch.int64).sum()
        if vis_cnt is not None:
            vis_cnt = vis_cnt + vis_c.sum(-1)
    _mark(vs, l, "merge")
    # 6. the end of the merge (call A is never empty: the tail slot, so mx is finite)
    o = num / den
    attn = o.to(q.dtype).reshape(B, R, -1)
    if used is not None:
        vs.stat["fills_used"] += used
        vs.stat["streamed_rows"] += streamed                    # rows the attention calls stream (per entry, per head)
    _mark(vs, l, "check")
    # CHECK-ONLY (ops.check) and the rung-L5 seams (vs.miss_ids, vs.pin): the presence of each selected block
    if ops.check or vs.miss_ids is not None or vs.pin is not None:
        found = _presence(vs, eng, sel, lst, m)
        miss = ~found & (sel >= 0)
        if ops.check:
            vs.chk_sel[l].copy_(sel)
            f4 = full.t().reshape(1, B, R, 1)
            r4 = res.t().reshape(1, B, R, 1)
            d = miss & f4
            s_ = miss & r4
            vs.stat["dropped"] += d.sum()
            vs.stat["skipped"] += s_.sum()
            vs.stat_row["dropped"] += d.sum((0, 1, 3))
            vs.stat_row["skipped"] += s_.sum((0, 1, 3))
            # UNION RESIDENCY MUST NOT BECOME UNION ATTENTION: a full row sees exactly its selection size; a resident
            # row sees its selection size minus its skipped blocks (seen and selected are counted per (head, request))
            n_sel = (sel >= 0).sum(-1)                                        # (H, B, R)
            want = n_sel - s_.sum(-1)
            bad = (f4[..., 0] | r4[..., 0]) & (vis_cnt != want)
            vs.stat["vis_bad"] += bad.sum()
            vs.stat["visible_blocks"] += vis_cnt.sum()
            vs.stat_row["vis_bad"] += bad.sum((0, 1))
            vs.stat_row["visible"] += vis_cnt.sum((0, 1))
            vs.stat_row["selected"] += n_sel.sum((0, 1))
        if vs.miss_rows is not None and vs.miss_ids is not None:              # one miss row per request (the draft row)
            mr = vs.miss_rows.t().reshape(1, B, R, 1)
            vs.miss_ids[l].copy_(torch.where(miss & mr, sel, vs.minus1).amax(2))
        if vs.miss_rows is not None and vs.pin is not None:
            for r in range(R):
                vs.pin.record(l, sel[:, :, r], vs.miss_rows[r].view(1, B, 1))
    if ops.record:
        vs.log[-1].update(q_e=q_e.clone(), out=o.clone(), lens_a=lens_a.clone(), bias_a=bias_a.clone())
    _mark(vs, l, "end")
    # 7. the end state (as FAST): write-back of a filled block, the next block's rows into the tail slot, tail id, length
    p_last = p0 + (R - 1)
    vs.full.copy_((p0 % bs) + (R - 1) >= bs - 1)
    vs.wpos.copy_(m * bs + (bs - 1))
    rs.body.writeback_kernel(eng._k_gpu, eng._v_gpu, eng._k_cpu, eng._v_cpu, vs.full, vs.wpos, tail0, bs)
    eng._k_gpu[bi, r63] = torch.where(it4, eng._k_gpu[bi, r63], k4c)
    eng._v_gpu[bi, r63] = torch.where(it4, eng._v_gpu[bi, r63], v4c)
    eng._kv_bias_gpu[bi, r63] = torch.where(it3, eng._kv_bias_gpu[bi, r63], cis_rows)
    eng._block_map[..., topk - 1] = ((p_last + 1) // bs).view(1, B).expand(H, B)
    eng._cache_lens.copy_(rs.body.cache_lens_value(eng, p_last).to(torch.int32))
    if vs.hooks is not None:
        vs.hooks.fast_end(l, eng, p0, vs.full, R)
    return attn


# ------------------------------------------------------------------ the layer and the pass
def _layer(vs: VerifyState, l: int, layer, hidden, pos_flat, cos_sin_cache):
    rs, ops = vs.rs, vs.ops
    NL, st = rs.nl, rs.st
    B, R, E = vs.B, vs.R, vs.E
    Hkv, D = layer.num_key_value_heads, layer.head_dim
    twin = st.twins[l]
    eng = rs.cache.layers[l].cache_engine
    _mark(vs, l, "pre")
    residual = hidden
    h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = _lin(vs, h, layer.wqkv)
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
    q = q.view(E, -1)
    k = k.view(E, -1)
    NL.apply_rope_with_cos_sin_cache_inplace(pos_flat, q, k, D, cos_sin_cache, True)
    q = q.reshape(E, layer.num_heads, D)
    k4 = k.reshape(B, R, Hkv, D)
    v4 = v.reshape(B, R, Hkv, D)
    c3 = cis.reshape(B, R, Hkv)
    _mark(vs, l, "twin")
    pos_before = _twin_rows(vs, twin, k4, c3)
    _mark(vs, l, "select")
    sel = _select_rows(vs, layer, q, twin)
    if vs.probe:
        _mark(vs, l, "probe")
        _probe_row0(vs, layer, q, twin, sel)
    if ops.mode == "exact":
        _mark(vs, l, "rows")
        if ops.stats:
            _union(vs, sel)
        attn = _exact_rows(vs, l, layer, eng, twin, q, k4, v4, sel, pos_before)
    elif ops.mode == "union":
        _mark(vs, l, "resid")
        attn = _union_rows(vs, l, layer, eng, twin, q, k4, v4, sel, pos_before)
    else:
        _mark(vs, l, "resid")
        attn = _fast_rows(vs, l, layer, eng, twin, q, k4, v4, sel, pos_before)
    _mark(vs, l, "post")
    attn = attn.reshape(E, 1, -1)
    hidden = residual + _lin(vs, attn, layer.wo)
    residual = hidden
    h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    h = _lin(vs, h, layer.gate_up_proj)
    out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
    NL.silu_and_mul(h, out)
    return residual + _lin(vs, out, layer.down_proj)


@torch.inference_mode()
def joint_pass(model, vs: VerifyState, input_ids, position_ids):
    """ONE joint pass: request b runs its R rows at the positions position_ids[b] (p_b .. p_b + K, the twin's
    positions) with the tokens input_ids[b]. input_ids (B, R) int64, position_ids (B, R). Every request must be active.
    Returns logits (B, R, V) float. Host bookkeeping: steps += R, pos_host += R. No host read of a device value."""
    rs = vs.rs
    st = rs.st
    B, R, E = vs.B, vs.R, vs.E
    if tuple(input_ids.shape) != (B, R) or tuple(position_ids.shape) != (B, R):
        raise VerifyError("input_ids and position_ids must be (%d, %d), got %s and %s"
                          % (B, R, tuple(input_ids.shape), tuple(position_ids.shape)))
    if not bool(st.active_host.all()):
        raise VerifyError("the joint pass needs every request active")
    rs.all_active = True
    hidden = F.embedding(input_ids.reshape(E, 1), model.embed_tokens)
    pos_flat = position_ids.reshape(E)
    for l, layer in enumerate(model.layers):
        hidden = _layer(vs, l, layer, hidden, pos_flat, model.cos_sin_cache)
    _mark(vs, len(model.layers), "head")
    hidden = rs.nl.layer_norm(hidden, w=model.norm_weight, eps=model.norm_variance_epsilon)
    logits = _lin(vs, hidden, model.lm_head).float()
    _mark(vs, len(model.layers), "done")
    st.steps += R
    st.pos_host += R * st.active_host.to(torch.int64)
    vs.passes += 1
    return logits.reshape(B, R, -1)


def read_stats(vs: VerifyState) -> dict:
    """ONE host sync: the union, extras, overflow and SEL0 counters."""
    keys = sorted(vs.stat)
    vals = torch.stack([vs.stat[k] for k in keys]).cpu().tolist()
    d = dict(zip(keys, (int(x) for x in vals)))
    nu = vs.stat_n["union"]
    d.update(union_mean=(d["union_sum"] / nu if nu else None), union_samples=nu, sel0_samples=vs.stat_n["sel0"])
    if vs.ops.mode == "union":                     # WS-B: per-row counts (CHECK-ONLY), the fills
        d.update({"row_" + k: [int(x) for x in v.cpu()] for k, v in vs.stat_row.items()})
        d.update(fills=vs.C, cap=vs.cap, check=vs.ops.check)
    d.update(calls=dict(vs.calls), passes=vs.passes)            # host invocation counts (every mode)
    return d


def phase_ms(marks, elapsed) -> dict:
    """The phase times of one probe pass: marks [(layer, label, event)] in record order; elapsed(a, b) in ms. The time
    from one mark to the next goes to the first mark's label. Returns {label: ms summed over the layers}."""
    out = {}
    for (_, lab, e0), (_, _, e1) in zip(marks[:-1], marks[1:]):
        out[lab] = out.get(lab, 0.0) + float(elapsed(e0, e1))
    return out


def layer_ms(marks, elapsed) -> list:
    """WS-B: the COMPLETE time of each layer of one probe pass, from its first mark to the first mark of the next layer
    (the last layer: to the "head" mark). Returns [ms] per layer (the phases of phase_ms nest inside these spans)."""
    firsts = []
    for lay, _, e in marks:
        if not firsts or firsts[-1][0] != lay:
            firsts.append((lay, e))
    return [float(elapsed(a[1], b[1])) for a, b in zip(firsts[:-1], firsts[1:])]
