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
last tail row's for every row), "shared_row_lengths" (every row uses row 0's n_used and M_b).

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
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import ragged_engine as RE
from .max_pooling_per_request import nosa_pooling_per_request
from .r2_positional_twin import BLOCK
from .stage1_seqused import stage1_seqused

MODES = ("fast", "exact")
GEMMS = ("natural", "rowgroup", "exact")
NEG_KINDS = (None, "no_init_force", "no_extras", "no_row_mask", "causal_leak", "shared_row_lengths")
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
              extras that the verify rows loaded) and nothing is fetched for it alone. Default False (Task 1.4)."""

    def __init__(self, K=3, mode="fast", gemm="natural", linear=None, extra=16, attn=None, neg=None, record=False,
                 stats=True, draft_last=False):
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
        return dict(K=self.K, R=self.R, mode=self.mode, gemm=self.gemm,
                    linear=getattr(self.linear, "__name__", type(self.linear).__name__),
                    m_fix=getattr(self.linear, "m_fix", None), extra=self.extra, neg=self.neg,
                    draft_last=self.draft_last)


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
        z = lambda: torch.zeros((), dtype=torch.int64, device=dev)    # noqa: E731
        self.stat = dict(union_sum=z(), union_max=z(), extras=z(), overflow=z(), sel0_order_eq=z(), sel0_set_eq=z())
        self.stat_n = dict(union=0, sel0=0)
        self.log = []
        self.marks = None                   # a list -> (layer, label, event) per phase (the cell's probe passes)
        self.clock = None                   # the cell's CudaClock (event / record)
        self.probe = False                  # True: the one-row selection after the batched one (timing + SEL0)
        self.fa_check = None                # a dict -> the VRF-IDX gate on the next FAST pass, layer 0
        self.hooks = None                   # Phase 2: the rollback (fast_begin / fast_end around a FAST layer)
        self.passes = 0
        self.meta = dict(ops=ops.describe(), B=B, R=R, E=E, X=X, mask_bias=MASK_BIAS)


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
        key_states, value_states, kv_bias, cache_lens = rs.body.decode_fixed(
            eng, k4[:, r].unsqueeze(1).contiguous(), v4[:, r].unsqueeze(1).contiguous(), twin.total_cis, ti_r,
            pos_before[r], rs.st.active, rs.err, diff, kern["gather_k"], kern["gather_v_bias"])
        if rs.pool:
            key_states, value_states, kv_bias = eng._k_gpu_att, eng._v_gpu_att, eng._kv_bias_gpu_att
        if ops.record:
            vs.log.append(dict(kind="exact", layer=l, row=r, sel=ti_r.clone(), win=eng._block_map.clone(),
                               pos=pos_before[r].clone()))
        o = ops.attn(q4[:, r].unsqueeze(1).contiguous(), key_states, value_states, kv_bias, cache_seqlens=cache_lens,
                     num_splits=rs.nl._ATTN_SPLITS)
        outs.append(o.reshape(B, -1))
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
    return d


def phase_ms(marks, elapsed) -> dict:
    """The phase times of one probe pass: marks [(layer, label, event)] in record order; elapsed(a, b) in ms. The time
    from one mark to the next goes to the first mark's label. Returns {label: ms summed over the layers}."""
    out = {}
    for (_, lab, e0), (_, _, e1) in zip(marks[:-1], marks[1:]):
        out[lab] = out.get(lab, 0.0) + float(elapsed(e0, e1))
    return out
