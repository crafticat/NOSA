"""The PAIRED TICK's model-level body and driver methods (GPU; spec
retroinfer-eval docs/superpowers/specs/2026-09-20-paired-tick.md section 1,
gates G1-G8; stage E2 of section 4: RESIDENT, no per-layer prefetch).

ONE per-layer body, ``layer_body``, serves three row layouts (core.RowPlan):
  paired   rows [V | S] per request, adjacent (2B rows): the tick;
  twin     V rows only (B rows): the reference twin of gate G3 (twin.py);
  srows    S rows only over a subset of requests (n rows): the RESTART after
           a reject and the SEED before the first tick (spec section 3 (a)).
Per layer, in order (spec section 1, audit D.2):
  [1] prenorm, ONE qkv GEMM over all rows (M = n*U), nosa_linear, rope with
      per-row positions (tau for V, tau + 1 for S): verify_forward's text
      (nosa_llama.py:697-708) at U = 2.
  [3] scoring per position through the DECODE's kernels and the captured
      pooling / top-k graph, one position at a time on the table state that
      position sees (verify_forward :714-739): V's table update persists;
      S's is bracketed by spec_loop.LayerJournal.take / restore, so the S
      position's state follows V's and nothing of it survives (G5).
  [4] the V row's engine update = the SHIPPED S == 1 body, called, not
      copied: cache.decode_update_kv -> cache_engine.decode_update_has_kv_bias
      (tail write :345-357, diff :363, the two BLOCKING Triton gathers
      :381-382, the write-back and rename on a fill :401-409). Its 64-slot
      return views are not used: the rows call reads the whole allocation.
      V's residual misses = _load_mask >= 0 after it (bytes counted).
      The S row's K/V/bias -> row 0 of the provisional slot (core).
  [5] the per-row bias (core.paired_bias) and ONE rows-attention call over
      all rows: the shipped decode kernel through cache_batch_idx and a
      per-row bias (verify/rows_attention.py, reuse ledger (ii-a)), explicit
      num_splits (refused at 0: the heuristic depends on the row count).
  [6] wo, FFN over all rows.
The driver methods: ``paired_tick`` (one tick: (B, 2) rows -> V and S logits,
then the poison of the provisional rows under NOSI_PAIRED_POISON=1),
``s_rows_forward`` (restart / seed over a subset), ``setup`` (the scratch:
layout checks, the bias storage, the E2 readiness mask, the journals, the
one-time zero-fill of the tail slot's never-written rows).

No CUDA extension is imported at module level: ``_gpu`` binds the model
module's names (nosa_llama.py:1-42) and the shipped kernel entry
``flash_attn_nosa.flash_attn_with_kvcache`` on first use, so this module and
its text are importable by the CPU tests.
"""
from __future__ import annotations

import os
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F

from . import core
from . import fused_bias as _fb
from .. import spec_loop as _sl


class TickConfig(NamedTuple):
    num_splits: int        # explicit split-KV count of every rows call (> 0)
    poison: bool           # NaN-poison the provisional rows after every S write's tick (G1 / G7 control)
    masked: float          # the finite masked bias value
    refuse_compress: bool  # refuse a 16-token compress event inside a call (the captured pooling graph is fixed-shape)
    gemm_pad_rows: int     # twin2b diagnostic (DESIGN.md blocker 5): zero rows appended to EVERY GEMM's M so cuBLAS sees the tick's M; 0 = off
    s_off: bool            # NOSI_PAIRED_S_OFF=1 debug: S rows stay in the GEMMs / attention call but score nothing, write nothing, attend V's mask
    bias_impl: str         # NOSI_PAIRED_BIAS: 'fused' (one Triton kernel per layer, fused_bias.py) or 'torch' (the core.py path)
    gemm: str              # NOSI_PAIRED_GEMM: 'fused' (one GEMM at M = n*U rows) or 'split' (one M = n call per row kind: V's call is the decode's)


def config_from_env(gemm_pad_rows: int = 0) -> TickConfig:
    """Resolved by the DRIVER at dispatch time (never at import): NOSI_ATTN_SPLITS
    (default 4; 0 refused), NOSI_PAIRED_POISON (1 = on). ``gemm_pad_rows`` is
    the driver's (the twin2b arm passes B); it is not an environment knob."""
    raw = os.environ.get("NOSI_ATTN_SPLITS", "4")
    try:
        splits = int(raw)
    except ValueError:
        raise SystemExit("NOSI_ATTN_SPLITS=%r is not an integer" % raw)
    if not (1 <= splits <= 128):
        raise SystemExit("NOSI_ATTN_SPLITS=%d: the paired tick needs an EXPLICIT split count in [1, 128] "
                         "(0 = the library heuristic, which depends on the number of rows and breaks the twin identity)" % splits)
    if int(gemm_pad_rows) < 0:
        raise SystemExit("gemm_pad_rows=%d must be >= 0" % gemm_pad_rows)
    bias_impl = os.environ.get("NOSI_PAIRED_BIAS", "fused")
    if bias_impl not in ("fused", "torch"):
        raise SystemExit("NOSI_PAIRED_BIAS=%r: 'fused' (the Triton kernel) or 'torch' (the core path)" % bias_impl)
    gemm = os.environ.get("NOSI_PAIRED_GEMM", "fused")
    if gemm not in ("fused", "split"):
        raise SystemExit("NOSI_PAIRED_GEMM=%r: 'fused' (M = n*U) or 'split' (one M = n call per row kind)" % gemm)
    if gemm == "split" and int(gemm_pad_rows) > 0:
        raise SystemExit("NOSI_PAIRED_GEMM=split cannot combine with the twin2b padding")
    return TickConfig(num_splits=splits, poison=os.environ.get("NOSI_PAIRED_POISON", "0") == "1",
                      masked=core.MASKED, refuse_compress=True, gemm_pad_rows=int(gemm_pad_rows),
                      s_off=os.environ.get("NOSI_PAIRED_S_OFF", "0") == "1", bias_impl=bias_impl, gemm=gemm)


def linear_rows(x: torch.Tensor, w: torch.Tensor, cfg: TickConfig) -> torch.Tensor:
    """The GEMM over x (n, U, K): cfg.gemm == 'fused' -> ONE call at M = n*U
    (linear_padded, the twin2b padding applies); 'split' -> one M = n call per
    row kind u (U calls, U <= 2: a loop over row KINDS, never over rows), each
    on a contiguous (n, 1, K) input -- V's call is then byte-identical to the
    shipped decode's F.linear(hidden (B, 1, K), w) and cuBLAS picks the same
    kernel (E2: twin vs twin2b showed the M = B vs 2B kernel term is real)."""
    n, U, K = x.shape
    if cfg.gemm == "split" and U > 1:
        return torch.cat([F.linear(x[:, u:u + 1].contiguous(), w) for u in range(U)], dim=1)
    return linear_padded(x, w, cfg.gemm_pad_rows)


def rope_positions(position_ids: torch.Tensor) -> torch.Tensor:
    """The flat, CONTIGUOUS position vector the flashinfer rope takes
    (rope.cu:122 CHECK_INPUT(pos_ids): a strided view such as
    position_ids[:, 0] is refused -- job 2175550, the equiv arm's crash).
    reshape(-1) of a strided view copies; of a contiguous tensor it is a view."""
    p = position_ids.reshape(-1)
    return p if p.is_contiguous() else p.contiguous()


def linear_padded(x: torch.Tensor, w: torch.Tensor, pad_rows: int) -> torch.Tensor:
    """F.linear over x (n, U, K) whose flattened row count is padded with
    ``pad_rows`` ZERO rows (appended, discarded from the output): the cuBLAS
    call then has M = n*U + pad_rows, so a U = 1 twin with pad_rows = B runs
    every GEMM at the tick's M = 2B and cuBLAS selects the tick's kernel
    (DESIGN.md blocker 5). Every output row depends on its own input row and
    the kernel only, so the first n*U rows are what the unpadded rows would
    be under that kernel. pad_rows = 0 is the plain call."""
    if pad_rows <= 0:
        return F.linear(x, w)
    n, U, K = x.shape
    xf = x.reshape(n * U, K)
    xp = torch.cat([xf, xf.new_zeros((int(pad_rows), K))], dim=0)
    return F.linear(xp, w)[:n * U].reshape(n, U, -1)


class _GPU(NamedTuple):
    NL: object          # nosi.nosa_llama: layer_norm, nosa_linear, apply_rope_with_cos_sin_cache_inplace, infllmv2_attn_stage1_fast, silu_and_mul
    fa: object          # flash_attn_nosa.flash_attn_with_kvcache: THE shipped decode kernel entry (nosa_llama.py:42)
    bias_rows: object   # cache_engine._bias_rows: KV_BIAS_SCALE applied exactly as the engine's four write sites do


_G: Optional[_GPU] = None


def _gpu() -> _GPU:
    global _G
    if _G is None:
        import nosi.nosa_llama as NL
        from nosi import cache_engine as CE
        from flash_attn_nosa import flash_attn_with_kvcache
        _G = _GPU(NL=NL, fa=flash_attn_with_kvcache, bias_rows=CE._bias_rows)
    return _G


# ---------------------------------------------------------------------------
# the per-process scratch
# ---------------------------------------------------------------------------
class Scratch:
    """Everything the tick needs beyond the model and the cache, allocated once."""

    def __init__(self, model, cache, lay: core.Layout, B: int):
        self.lay = lay
        self.B = int(B)
        eng0 = cache.layers[0].cache_engine
        dev, dtype = eng0._k_gpu.device, eng0._k_gpu.dtype
        H, D = eng0.head_num, eng0.head_dim
        self.device, self.dtype, self.H, self.D = dev, dtype, H, D
        rows = lay.W * lay.bs
        # the rows bias storage: >= max(2B, B) rows (the kernel's host check wants bias.size(0) == B on the narrowed view)
        self.bias = torch.empty((2 * self.B, rows, H), dtype=dtype, device=dev)
        self.bias_twin = torch.empty((self.B, rows, H), dtype=dtype, device=dev)
        self.ready = core.ready_mask(H, self.B, lay, device=dev)                      # E2: window + tail + provisional; ring not ready
        self.ready_u8 = self.ready.to(torch.uint8).contiguous()                        # the fused kernel's readiness input (E4 updates both)
        self.prefix = torch.zeros((2 * self.B, lay.W, H), dtype=torch.int32, device=dev)
        self.extent = torch.zeros((2 * self.B,), dtype=torch.int32, device=dev)
        self.ring_dummy = torch.full((H, self.B, max(lay.ring_hi - lay.ring_lo, 1)), -1, dtype=torch.int64, device=dev)  # E2: no ring occupants (E4 fills it)
        self.sel_dummy = torch.full((H, 1, int(model.topk_blocks)), -1, dtype=torch.int64, device=dev)   # a V-only call has no S selection (K = topk_blocks)
        self.masked_rounded = _fb.rounded_masked(core.MASKED, dtype)
        self.journals = [_sl.LayerJournal() for _ in range(model.num_layers)]
        self.slot_complete = [False] * model.num_layers                            # slot topk-1 holds a complete block (after a fill)
        self.prev_sel_s: List[Optional[torch.Tensor]] = [None] * model.num_layers  # (H, B, K) int64: S's selection for the NEXT V position
        self.req_all = torch.arange(self.B, dtype=torch.int64, device=dev)
        self.cu_full = torch.arange(self.B + 1, dtype=torch.int32, device=dev)
        self.key_pad = torch.zeros((self.B, 1, H, D), dtype=dtype, device=dev)
        self.cis_pad = torch.zeros((self.B, 1, H), dtype=dtype, device=dev)
        self.q_pad = torch.zeros((self.B, int(model.num_heads), D), dtype=dtype, device=dev)   # the subset's q rows scattered into a full-batch stage-1 call


@torch.inference_mode()   # the engine's tensors are inference tensors: the zero-fill below is an in-place write on them (state_snapshot.py, job 2174640)
def setup(model, cache, B: int) -> Scratch:
    """Build the scratch and check every assumption where it is produced:
    the model has the warm-up buffers and the captured graph; every layer's
    engine has the layout's allocation (R >= 1, no pool); the window is fully
    occupied (every row the V rows read has been written by a gather); the
    never-written rows of the tail slot are zero-filled ONCE (the rows call
    loads every row below a row's extent: rows_attention.py PRECONDITION; the
    shipped decode never reads them, so this changes no served logit); the
    compressed length is uniform over the batch (the subset scoring of the
    restart builds its cu_seqlens from it)."""
    if not getattr(model, "has_buffers", False):
        raise RuntimeError("the paired tick needs the warm-up decode step's buffers and captured pooling graph (decode_inference warmup) first")
    eng0 = cache.layers[0].cache_engine
    if int(eng0._k_gpu.shape[0]) != int(B):
        raise RuntimeError("engine batch %d != B=%d" % (eng0._k_gpu.shape[0], B))
    lay = core.paired_layout(eng0.topk, eng0.verify_round_slots, eng0.block_size)   # R = 0 -> narrow, R >= 1 -> union
    sc = Scratch(model, cache, lay, B)
    for l, clayer in enumerate(cache.layers):
        eng = clayer.cache_engine
        core.check_layout_against_engine(lay, eng)
        if not bool((eng._block_map[..., :lay.tail_slot] >= 0).all()):
            raise RuntimeError("layer %d: the window has unoccupied slots; the V rows read the whole window (run the warm-up decode step first, "
                               "and use a context long enough for 64 selected blocks)" % l)
        tl = int(eng._tail_block_len_on_gpu)
        lo, hi = lay.tail_slot * lay.bs + tl, (lay.tail_slot + 1) * lay.bs
        if hi > lo:
            eng._k_gpu[:, lo:hi].zero_()
            eng._v_gpu[:, lo:hi].zero_()
            eng._kv_bias_gpu[:, lo:hi].zero_()
        if model.layers[l].compressed_cis_buf.data_ptr() != clayer.compressed_cis.data_ptr():
            # nosa_llama.py:451 binds the warm-up's compressed_cis_buf to update_cis's return value, which IS the
            # layer's persistent table (cache_engine.py:1081). score_position relies on that being a self-copy.
            raise RuntimeError("layer %d: compressed_cis_buf does not alias the layer's compressed_cis table; score_position's "
                               "full-batch self-copy assumption (DESIGN.md A8) no longer holds" % l)
    return sc


def compress_budget(cache, n_positions_ahead: int) -> int:
    """How many more positions (V appends + one provisional S append) fit
    before a 16-token compress event: pooling_block_size - no_compress_k_len.
    A compress event changes the stage-1 score's column count while the
    captured graph's score_buf is fixed-shape (nosa_llama.py:465-497,
    :613-618): the call would fail at score_buf.copy_. The driver sizes N
    from this once, after the warm-up step."""
    layer0 = cache.layers[0]
    kernel_size = int(layer0.no_compress_k_cache.shape[1])
    return kernel_size - int(layer0.no_compress_k_len) - int(n_positions_ahead)


# ---------------------------------------------------------------------------
# scoring: the decode's per-position chain (stage 1 + the captured pooling / top-k graph)
# ---------------------------------------------------------------------------
def score_position(G, model, cache, layer, l: int, q_u: torch.Tensor, k_u: torch.Tensor, cis_u: torch.Tensor,
                   sc: Scratch, cfg: TickConfig, req_idx: Optional[torch.Tensor]):
    """One position's table updates and selection, exactly decode_forward's /
    verify_forward's text (nosa_llama.py:715-739) over the WHOLE batch. For a
    SUBSET of requests (the restart) the subset's q / key / cis rows are
    scattered into zero-padded full-batch tensors (the layer tables are
    batch-wide; the caller journals and restores them) and the selection is
    read off topk_idx_buf at req_idx.

    NEVER a slice write into score_buf / compressed_cis_buf. compressed_cis_buf
    is NOT a scratch buffer: the warm-up bound it to update_cis's return value
    (nosa_llama.py:451), which is the layer's PERSISTENT compressed-cis table
    (cache_engine.py:1081), so the decode's compressed_cis_buf.copy_(
    compressed_cis) is a self-copy. A compacted scoring that wrote the
    gathered rows of req_idx into [:, :n] overwrote the table rows of requests
    0..n-1 with other requests' cis, un-journaled (job 2175550: after every
    restart of n rows exactly requests 0..n-1 diverged from the twin).
    Returns (sel (H, n, K) int64, ucis = the layer's total_cis table, score)."""
    NL = G.NL
    B = sc.B
    if req_idx is None:
        key_full = k_u.unsqueeze(1).contiguous()
        cis_full = cis_u.unsqueeze(1).contiguous()
        q_full = q_u.contiguous()
    else:
        key_full, cis_full, q_full = sc.key_pad, sc.cis_pad, sc.q_pad
        key_full.zero_(); cis_full.zero_(); q_full.zero_()
        key_full[req_idx, 0] = k_u
        cis_full[req_idx, 0] = cis_u
        q_full[req_idx] = q_u
    no_compress_k = cache.update_no_compress_k_decode(key_full, l, layer.pooling_block_size, layer.pooling_stride)
    if no_compress_k is not None:
        if cfg.refuse_compress:
            raise RuntimeError("layer %d: a 16-token compress event fired inside a paired call (no_compress_k_len reached %d); the captured "
                               "pooling graph's score_buf is fixed-shape (nosa_llama.py:465-497): shorten the run or move L (tick.compress_budget)"
                               % (l, layer.pooling_block_size))
        new_compressed_k = no_compress_k.mean(dim=1, keepdim=True)
    else:
        new_compressed_k = None
    compressed_k, cu_comp, max_seqlen_comp = cache.update_compress_k_decode(new_compressed_k, l)
    ucis = cache.update_uncompressed_cis(cis_full, l, 0, B)
    compressed_cis = cache.update_cis(cis_full.permute(2, 0, 1), l, 0, B)          # (H, B, M): the layer's table itself
    ck = compressed_k.contiguous().flatten(0, 1)
    score = NL.infllmv2_attn_stage1_fast(q_full, ck, ck, cu_seqlens_q=sc.cu_full, cu_seqlens_k=cu_comp,
                                         max_seqlen_q=1, max_seqlen_k=max_seqlen_comp, causal=False)
    layer.score_buf.copy_(score)                        # nosa_llama.py:613
    layer.compressed_cis_buf.copy_(compressed_cis)      # nosa_llama.py:614: a self-copy (the alias above), the decode's own line
    layer.after_pooling_graph.replay()
    if req_idx is None:
        sel = model.topk_idx_buf.clone()                # (H, B, K) int64; the buffer is rewritten by the next replay
    else:
        sel = model.topk_idx_buf[:, req_idx].clone()    # (H, n, K)
    return sel, ucis, score


# ---------------------------------------------------------------------------
# the attention call: the shipped decode kernel over R rows
# ---------------------------------------------------------------------------
def attend_rows(G, q: torch.Tensor, k_gpu: torch.Tensor, v_gpu: torch.Tensor, rb, num_splits: int) -> torch.Tensor:
    """q (R, Hq, D) contiguous; K/V = the WHOLE allocation (B, W*bs, Hkv, D);
    rb = core.paired_bias's RowsBias whose storage has >= max(R, B) rows.
    Every kernel fact rows_attention.rows_attention_args asserts is asserted
    here for a row count R that need not equal B*U (the restart's subset):
    the narrowed view bias[:B] satisfies flash_api.cpp:366, its stride(0) is
    what the kernel multiplies the query row by, and the storage holds every
    row < R. Returns (R, Hq, D) in the store dtype."""
    if k_gpu.dim() != 4:
        raise ValueError("k_gpu must be (B, W*bs, Hkv, D), got %s" % (tuple(k_gpu.shape),))
    B, rows, Hkv, D = k_gpu.shape
    if tuple(v_gpu.shape) != (B, rows, Hkv, D) or not (k_gpu.is_contiguous() and v_gpu.is_contiguous()):
        raise ValueError("v_gpu must match k_gpu and both must be the contiguous allocation")
    if q.dim() != 3 or q.shape[2] != D or not q.is_contiguous():
        raise ValueError("q must be a contiguous (R, Hq, D=%d), got %s" % (D, tuple(q.shape)))
    R, Hq, _ = q.shape
    if R < 1 or Hq % Hkv != 0 or D % 8 != 0:
        raise ValueError("R=%d Hq=%d Hkv=%d D=%d: need R >= 1, Hq %% Hkv == 0, D %% 8 == 0 (the ngroups swap)" % (R, Hq, Hkv, D))
    bias = rb.bias
    if bias.dim() != 3 or tuple(bias.shape[1:]) != (rows, Hkv) or bias.shape[0] < max(R, B) or not bias.is_contiguous():
        raise ValueError("bias storage must be a contiguous (>= max(R, B)=%d, %d, %d), got %s" % (max(R, B), rows, Hkv, tuple(bias.shape)))
    if not (q.dtype == k_gpu.dtype == v_gpu.dtype == bias.dtype):
        raise TypeError("q/k/v/bias dtypes differ: %s %s %s %s" % (q.dtype, k_gpu.dtype, v_gpu.dtype, bias.dtype))
    cbi, csl = rb.cache_batch_idx, rb.cache_seqlens
    if cbi.dtype != torch.int32 or tuple(cbi.shape) != (R,) or not cbi.is_contiguous():
        raise ValueError("cache_batch_idx must be int32 contiguous (R=%d,), got %s %s" % (R, cbi.dtype, tuple(cbi.shape)))
    if csl.dtype != torch.int32 or tuple(csl.shape) != (R,) or not csl.is_contiguous():
        raise ValueError("cache_seqlens must be int32 contiguous (R=%d,), got %s %s" % (R, csl.dtype, tuple(csl.shape)))
    if not (q.device == k_gpu.device == v_gpu.device == bias.device == cbi.device == csl.device):
        raise ValueError("q/k/v/bias/cache_seqlens/cache_batch_idx are not on one device")
    ns = int(num_splits)
    if not (1 <= ns <= 128):
        raise ValueError("num_splits=%d must be explicit in [1, 128]" % ns)
    bias_view = bias[:B]
    if bias_view.stride(0) != rows * Hkv or bias_view.stride(-2) != Hkv or bias_view.stride(-1) != 1 or bias_view.data_ptr() != bias.data_ptr():
        raise AssertionError("bias[:B] strides %s; the kernel needs (W*bs*Hkv, Hkv, 1) over the full storage" % (bias_view.stride(),))
    out = G.fa(q.unsqueeze(1), k_gpu, v_gpu, bias_view, cache_seqlens=csl, cache_batch_idx=cbi, num_splits=ns)
    if tuple(out.shape) != (R, 1, Hq, D):
        raise AssertionError("kernel output %s != (R, 1, Hq, D) = %s" % (tuple(out.shape), (R, 1, Hq, D)))
    return out.squeeze(1)


def fused_args(sc: Scratch, eng, plan: core.RowPlan, n: int, sel_s, tv: core.TailView, own_row: int) -> "_fb.FusedArgs":
    """The fused kernel's argument tuple for one layer call, from the engine's
    own tensors and the scratch (ring_dummy: (H, B, max(NR, 1)) of -1 until E4
    fills the ring; sel_dummy for a V-only call). tests/test_paired_core.py
    builds it through a real Scratch at ROUND_SLOTS = 62 and 0 (job 2175563's
    mixed-code crash was a (H, B, 1) placeholder against NR = 61)."""
    lay = sc.lay
    req_i32 = plan.req.to(torch.int32)
    srow_i32 = torch.arange(int(n), dtype=torch.int32, device=sc.device).repeat_interleave(plan.U)
    return _fb.FusedArgs(cis=eng._kv_bias_gpu, sel=(sel_s.contiguous() if sel_s is not None else sc.sel_dummy), bmap=eng._block_map,
                         ring_ids=sc.ring_dummy, ready=sc.ready_u8, req=req_i32, srow=srow_i32, role=plan.role,
                         W=lay.W, bs=lay.bs, topk=lay.topk, tail_slot=lay.tail_slot, ring_lo=lay.ring_lo, n_ring=lay.ring_hi - lay.ring_lo,
                         tail_rows_v=tv.tail_len_after, tail_rows_s=tv.tail_rows_for_s, own_row_s=int(own_row),
                         content_off=tv.content_offset, masked=sc.masked_rounded)


# ---------------------------------------------------------------------------
# the per-layer body
# ---------------------------------------------------------------------------
class LayerOut(NamedTuple):
    hidden: torch.Tensor
    account: core.LayerAccount
    insitu: Optional[dict]


def layer_body(G, model, cache, l: int, hidden: torch.Tensor, position_ids: torch.Tensor, plan: core.RowPlan,
               req_idx: torch.Tensor, whole_batch: bool, sc: Scratch, cfg: TickConfig, trace=None, insitu=None) -> LayerOut:
    NL = G.NL
    layer = model.layers[l]
    clayer = cache.layers[l]
    eng = clayer.cache_engine
    lay = sc.lay
    if trace is not None:
        trace.begin_layer(l)
    n, U, _ = hidden.shape
    if U != plan.U or n != int(req_idx.numel()):
        raise ValueError("hidden %s does not match the row plan (n=%d, U=%d)" % (tuple(hidden.shape), req_idx.numel(), plan.U))
    Hq, Hk, D = layer.num_heads, layer.num_key_value_heads, layer.head_dim
    sub = None if whole_batch else req_idx
    residual = hidden

    # [1] prenorm, qkv GEMM over all rows, rope at per-row positions (verify_forward :697-708)
    hs = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
    qkv = linear_rows(hs, layer.wqkv, cfg).contiguous()
    q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hk)
    q = q.view(n * U, -1)
    k = k.view(n * U, -1)
    q_pre = k_pre = None
    if insitu is not None and plan.has_v and plan.has_s:
        q_pre, k_pre = q.clone(), k.clone()                   # pre-rope copies for the in-situ 'qkv' vs 'rope' terms (equiv arm only)
    NL.apply_rope_with_cos_sin_cache_inplace(rope_positions(position_ids), q, k, D, model.cos_sin_cache, True)
    q = q.reshape(n, U, Hq, D)
    k = k.reshape(n, U, Hk, D)
    v = v.reshape(n, U, Hk, D)
    s_off = bool(cfg.s_off) and plan.has_v and plan.has_s   # the S rows stay in the GEMMs and the attention call only

    # [3] scoring, one position at a time, on the table state that position sees
    if trace is not None:
        trace.rec("score_begin")
    u = 0
    sel_v = sel_s = ucis = score_v = None
    if plan.has_v:
        sel_v, ucis, score_v = score_position(G, model, cache, layer, l, q[:, u], k[:, u], cis[:, u], sc, cfg, None)
        u += 1
    if plan.has_s and not s_off:
        j = sc.journals[l]
        j.take(clayer)
        sel_s, _, _ = score_position(G, model, cache, layer, l, q[:, u], k[:, u], cis[:, u], sc, cfg, sub)
        j.restore(clayer)                                     # the S position's table state never survives (G5)

    # [4] the V row's engine update = the shipped S == 1 body; the S row's provisional write
    if trace is not None:
        trace.rec("fetch_begin")
    tail_len_before = int(eng._tail_block_len_on_gpu)
    v_miss = None
    if plan.has_v:
        cache.decode_update_kv(k[:, 0], v[:, 0], ucis, l, sel_v)      # returns the 64-slot views; the rows call reads the allocation
        v_miss = (eng._load_mask >= 0).sum(-1)                        # (H, B): V's residual misses, fetched BLOCKING above
    tv = core.tail_view(tail_len_before, plan.has_v, sc.slot_complete[l], lay)
    sc.slot_complete[l] = tv.slot_complete
    if int(eng._tail_block_len_on_gpu) != tv.tail_len_after:
        raise AssertionError("layer %d: engine tail_len %d != the S == 1 rule's %d" % (l, eng._tail_block_len_on_gpu, tv.tail_len_after))
    own_row = core.provisional_row(lay, tv.tail_len_after) if (plan.has_s and not s_off) else -1
    if plan.has_s and not s_off:
        core.s_provisional_write(eng, k[:, u], v[:, u], G.bias_rows(cis[:, u].unsqueeze(1)), lay, sub, tv.tail_len_after)

    # [5] the per-row bias and ONE rows-attention call
    vis_v = vis_s = None
    if cfg.bias_impl == "fused" and not s_off:
        # ONE kernel per layer (fused_bias.py); prefix rows for the ledger come out of the same launch
        rb = _fb.fused_bias_triton(fused_args(sc, eng, plan, n, sel_s, tv, own_row), sc.bias, sc.prefix, sc.extent)
        pre = rb.visible_rows.view(n, plan.U, lay.W, Hk)
        if plan.has_v:
            vis_v = pre[:, 0]
        if plan.has_s:
            vis_s = pre[:, plan.U - 1]
    else:
        if plan.has_v:
            vis_v = core.v_visible_rows(Hk, n, lay, tv.tail_len_after, device=sc.device)
        if plan.has_s and not s_off:
            content_id = eng._block_map[..., lay.tail_slot] - tv.content_offset
            ids = core.slot_ids(eng._block_map, content_id, lay)
            ids_n, ready_n = (ids, sc.ready) if whole_batch else (ids[:, req_idx], sc.ready[:, req_idx])
            vis_s = core.s_visible_rows(ids_n, ready_n, sel_s, lay, tv.tail_rows_for_s)
        elif s_off:
            vis_s = vis_v                                     # S rows attend exactly what V attends: no S contribution anywhere
        vis, cbi = core.assemble_rows(plan, vis_v, vis_s, req_idx)
        rb = core.paired_bias(eng._kv_bias_gpu, vis, cbi, plan.U, cfg.masked, out=sc.bias, check_values=False,
                              own_rows=(core.own_rows(plan, own_row) if own_row >= 0 else None))
    if trace is not None:
        trace.rec("fetch_end")
    q_rows = q.reshape(n * U, Hq, D).contiguous()
    attn = attend_rows(G, q_rows, eng._k_gpu, eng._v_gpu, rb, cfg.num_splits)
    if trace is not None:
        trace.rec("attn_end")
    attn = attn.view(n, U, -1)

    # [6] wo, FFN over all rows (decode_forward :652-667)
    hidden = residual + linear_rows(attn, layer.wo, cfg)
    residual = hidden
    hs = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    gu = linear_rows(hs, layer.gate_up_proj, cfg)
    dd = gu.shape[-1] // 2
    act = torch.empty(gu.shape[:-1] + (dd,), dtype=gu.dtype, device=gu.device)
    NL.silu_and_mul(gu, act)
    hidden = residual + linear_rows(act, layer.down_proj, cfg)

    # accounting (core): misses, S's selections and hits, the divergence vs S's prediction for this V position
    # (the twin arm has no S rows and no prediction: its accounts carry no divergence; the paired
    # driver asserts the seed ran before the first tick, so every V position there has one)
    sel_s_prev = sc.prev_sel_s[l] if plan.has_v else None
    acct = core.layer_account(v_miss, sel_s, (None if s_off else vis_s), sel_v, sel_s_prev, lay, tv.tail_len_after, tv.filled)
    if plan.has_s and not s_off:
        if sc.prev_sel_s[l] is None:
            sc.prev_sel_s[l] = torch.full((Hk, sc.B, sel_s.shape[-1]), -1, dtype=torch.int64, device=sc.device)
        if whole_batch:
            sc.prev_sel_s[l].copy_(sel_s)
        else:
            sc.prev_sel_s[l][:, req_idx] = sel_s
    rec = None
    if insitu is not None and plan.has_v and plan.has_s:
        rec = insitu.compare_layer(G, model, cache, layer, l, position_ids=position_ids, q_pre=q_pre, k_pre=k_pre, q=q, k=k, v=v, cis=cis,
                                   score_v=score_v, sel_v=sel_v, vis_v=vis_v, rb=rb, attn=attn, hidden_out=hidden, eng=eng, sc=sc, cfg=cfg)
    return LayerOut(hidden=hidden, account=acct, insitu=rec)


# ---------------------------------------------------------------------------
# driver methods
# ---------------------------------------------------------------------------
class TickResult(NamedTuple):
    logits_v: torch.Tensor           # (n, V) fp32
    logits_s: Optional[torch.Tensor] # (n, V) fp32 or None
    accounts: List[core.LayerAccount]
    insitu: Optional[list]           # per-layer in-situ twin records (+ the head record) or None


def _forward(G, model, cache, sc: Scratch, cfg: TickConfig, tokens: torch.Tensor, position_ids: torch.Tensor,
             plan: core.RowPlan, req_idx: torch.Tensor, whole_batch: bool, trace=None, insitu=None) -> TickResult:
    NL = G.NL
    n, U = tokens.shape
    if tuple(position_ids.shape) != (n, U) or U != plan.U:
        raise ValueError("tokens %s / position_ids %s / plan U=%d disagree" % (tuple(tokens.shape), tuple(position_ids.shape), plan.U))
    hidden = F.embedding(tokens, model.embed_tokens)
    if trace is not None:
        trace.begin_call()
    accts, recs = [], []
    for l in range(model.num_layers):
        if insitu is not None:
            insitu.capture_input(l, hidden)
        out = layer_body(G, model, cache, l, hidden, position_ids, plan, req_idx, whole_batch, sc, cfg, trace=trace, insitu=insitu)
        hidden = out.hidden
        accts.append(out.account)
        if out.insitu is not None:
            recs.append(out.insitu)
    pre = hidden
    hidden = NL.layer_norm(hidden, model.norm_variance_epsilon, model.norm_weight)
    logits = linear_rows(hidden, model.lm_head, cfg).float()                          # (n, U, V)
    if trace is not None:
        trace.end_call()
    if insitu is not None and plan.has_v and plan.has_s:
        recs.append(insitu.compare_head(G, model, pre, hidden, logits))
    if plan.has_s and cfg.poison and not cfg.s_off:
        for clayer in cache.layers:
            eng = clayer.cache_engine
            core.poison_provisional(eng, sc.lay, None if whole_batch else req_idx, int(eng._tail_block_len_on_gpu))
    lv = logits[:, 0] if plan.has_v else None
    ls = logits[:, U - 1] if plan.has_s else None
    return TickResult(logits_v=lv if lv is not None else ls, logits_s=ls if plan.has_v else None, accounts=accts, insitu=(recs if insitu is not None else None))


@torch.inference_mode()
def paired_tick(model, cache, sc: Scratch, cfg: TickConfig, v_tokens: torch.Tensor, s_tokens: torch.Tensor,
                position: torch.Tensor, trace=None, insitu=None) -> TickResult:
    """ONE TICK over the whole batch: v_tokens / s_tokens (B,) int64, position
    (B,) the committed position tau (any integer dtype the rope accepts).
    Returns V's and S's fp32 logits (B, V) and the per-layer accounts."""
    G = _gpu()
    B = sc.B
    for name, t in (("v_tokens", v_tokens), ("s_tokens", s_tokens), ("position", position)):
        if tuple(t.shape) != (B,):
            raise ValueError("%s must be (B=%d,), got %s" % (name, B, tuple(t.shape)))
    plan = core.row_plan(sc.req_all, True, True)
    toks = torch.stack([v_tokens, s_tokens], dim=1)                # (B, 2): row 2b = V, 2b + 1 = S
    pos = torch.stack([position, position + 1], dim=1)
    return _forward(G, model, cache, sc, cfg, toks, pos, plan, sc.req_all, True, trace=trace, insitu=insitu)


@torch.inference_mode()
def s_rows_forward(model, cache, sc: Scratch, cfg: TickConfig, tokens: torch.Tensor, position: torch.Tensor,
                   req_idx: torch.Tensor, trace=None) -> TickResult:
    """The RESTART (after a reject: position tau + 1 on the committed token)
    and the SEED (before the first tick: position tau on the V input) over a
    subset of requests, spec section 3 option (a): S rows only, resident-only
    attention, provisional writes, journaled tables. tokens / position /
    req_idx (n,). Returns the rows' fp32 logits (n, V) as logits_v."""
    G = _gpu()
    n = int(req_idx.numel())
    if n < 1 or tuple(tokens.shape) != (n,) or tuple(position.shape) != (n,):
        raise ValueError("tokens %s / position %s must be (n=%d,) with n >= 1" % (tuple(tokens.shape), tuple(position.shape), n))
    whole = (n == sc.B) and bool(torch.equal(req_idx, sc.req_all))
    plan = core.row_plan(req_idx, False, True)
    return _forward(G, model, cache, sc, cfg, tokens.unsqueeze(1), position.unsqueeze(1), plan, req_idx, whole, trace=trace)


def restart_after(model, cache, sc: Scratch, cfg: TickConfig, state: core.DraftState, outcome: core.TickOutcome,
                  committed: torch.Tensor, position_next: torch.Tensor, trace=None) -> int:
    """Spec section 3 (a): the sequential restart for outcome.restart_idx:
    S rows at position tau + 1 on the committed token; their argmax becomes
    the draft for tau + 2. Returns (rows run, their per-layer accounts); (0, []) when none."""
    idx = outcome.restart_idx
    n = int(idx.numel())
    if n == 0:
        return 0, []
    res = s_rows_forward(model, cache, sc, cfg, committed[idx], position_next[idx], idx, trace=trace)
    state.apply_restart(idx, res.logits_v.argmax(-1))
    return n, res.accounts
