"""Move whole blocks between an attended slot and a victim-pool slot, in place,
inside NOSI's GPU cache tensors (retroinfer-eval fork).

The pool lives at slot indices ``topk .. topk+P-1`` of the SAME
``_k_gpu`` / ``_v_gpu`` / ``_kv_bias_gpu`` tensors, i.e. at row offsets
``(topk+q)*block_size``. Those rows sit BEYOND ``_cache_lens``
(``cache_engine.py:231``, and again at :285/:399/:451:
``(topk-1)*block_size + tail_len``, at most ``topk*block_size - 1`` because
``tail_len <= block_size``), and ``flash_attn_nosa_with_kvcache`` takes no block
table -- it reads rows ``0 .. cache_seqlens-1`` as one flat sequence
(``nosa_llama.py:610-616``, ``cache_seqlens=_cache_lens``). So the pool is
invisible to attention and the attended set is still exactly slots
``0..topk-1``. That is the hard constraint
this whole design is built around: THE CACHE IS THE ATTENDED SET.

Written in the style of ``flash_h2d_mask.py`` (grid ``(B, H, M)``, strides
taken from ``.stride()``, a scalar early return on an inactive entry) so the
two can be read side by side.

ACTIONS, decided by ``pool_update_kernel.cu`` and mirrored in
``scripts/nosi_pool_reference.py``:

===========  ==============================================================
NONE (0)     nothing to do (no load at this slot, the tail slot, a pool
             overflow, or a miss with no block worth parking)
SWAP (1)     attended slot s and pool slot q exchange contents. The wanted
             block was IN the pool, so no host fetch happens at all; the
             block leaving slot s lands in q, which is the eviction, free.
MOVE_IN (2)  pool slot q -> attended slot s, one way. Same as SWAP except
             slot s held nothing worth keeping, so q is left empty.
MOVE_OUT (3) attended slot s -> pool slot q, one way. The wanted block was
             NOT in the pool; the host gather will overwrite slot s a moment
             later, so copying the pool victim back into s would be pure
             waste (~65 kB per missed entry, ~0.2 GB per step at 64K x 64).
===========  ==============================================================

DISJOINTNESS (why there is no race across programs in one launch). Program
``(b, h, s)`` touches attended rows ``[s*bs, (s+1)*bs)`` and pool rows
``[(topk+q)*bs, ...)`` at batch ``b``, head ``h``. Distinct ``s`` give
distinct attended rows; distinct ``s`` give distinct ``q`` because the
bookkeeping kernel claims each pool slot with an atomicCAS and demotes the
loser to an ordinary miss, so one slot cannot be handed to two attended slots
even if its inputs are malformed; and the attended and pool regions never
overlap. Distinct ``(b, h)`` index disjoint memory. So every program writes
memory no other program touches.

INDEXING, both envelopes, because the pool TIGHTENS one of them.

* HOST GATHER (``flash_h2d_mask.py`` / ``flash_h2d_mask_bias.py``, UNCHANGED by
  the pool and NOT int64-cast): ``pid_b * stride_b_cpu`` is int32 x int32, so
  the CPU-side tensor bounds it at ``B * (S + max_gen_len) < 2**23``
  (max_gen_len = 8192; REPRODUCE.md: 64K x 112 passes, 64K x 120 aborts).
* GPU SIDE of the same gathers: ``pid_b * stride_b_gpu`` with
  ``stride_b_gpu = (topk + P) * block_size * H * D``. For NOSA-8B
  ``block_size * H * D = 64 * 2 * 128 = 16384`` (config.json: two KV heads,
  head_dim 128; the CacheEngine defaults agree), so the bound is
  ``B * (topk + P) < 2**31 / 16384 = 131072``. THE POOL DIVIDES THAT HEADROOM
  BY ``(topk + P) / topk`` -- 3.0x at P=129 -- and the constant is
  MODEL-SPECIFIC: an 8-KV-head model has ``block_size * H * D = 65536`` and a
  4x tighter bound.
* This kernel is immune to both by construction: ``pid_b``/``pid_h``/``pid_m``
  and the row/column offsets are ``.to(tl.int64)`` before any multiply.

At the cells this job actually runs: 64K x 64 at P=129 gives 4,718,592 of
8,388,608 (1.8x margin) on the host side and 12,352 of 131,072 (10.6x) on the
GPU side; 16K x 128 at P=17 gives 3,145,728 (2.7x) and 10,368 (12.6x). The pool
becomes the binding constraint only when ``S + 8192 < block_size * (topk + P)``,
i.e. below about 4K of context at P=129. ``cache_engine.prefill_update`` asserts
both bounds at allocation time so an out-of-envelope cell aborts loudly instead
of indexing a wrapped address.
"""
import torch
import triton
import triton.language as tl

POOL_NONE = 0
POOL_SWAP = 1
POOL_MOVE_IN = 2
POOL_MOVE_OUT = 3


@triton.jit
def pool_swap_kernel(
    k_ptr,           # (B, S_GPU, H, D)
    v_ptr,           # (B, S_GPU, H, D)
    bias_ptr,        # (B, S_GPU, H)
    target_ptr,      # (H, B, M) int64: the pool slot q
    action_ptr,      # (H, B, M) int8:  NONE / SWAP / MOVE_IN / MOVE_OUT
    stride_b_k, stride_m_k, stride_h_k, stride_d_k,
    stride_b_v, stride_m_v, stride_h_v, stride_d_v,
    stride_b_bias, stride_m_bias, stride_h_bias,
    S_GPU, B, H, M, TOPK, P,
    block_size: tl.constexpr,
    D: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(2).to(tl.int64)

    idx = pid_h * B * M + pid_b * M + pid_m
    act = tl.load(action_ptr + idx)

    # the overwhelmingly common case: ~3.6 of 64 slots move per (h, b) step
    if act == POOL_NONE:
        return

    q = tl.load(target_ptr + idx).to(tl.int64)

    # Belt and braces, BOTH ENDS of the range. The bookkeeping kernel only ever
    # pairs a non-NONE action with 0 <= q < P, but the row masks below are
    # upper-bound only (the same shape as flash_h2d_mask.py:56-57), so a
    # negative q would slip past them and read out of bounds, and a q >= P
    # would make `mask_pool` all-false -- which does NOT skip the store: on a
    # SWAP `mask_att` is still true, so `tl.load(..., other=0)` would write 64
    # tokens of ZEROED K, V and bias into an ATTENDED slot. Finite, wrong, and
    # with no NaN or OOB to trip on. Return the way the gather returns on an
    # invalid block id (flash_h2d_mask.py:32-33). Two statements rather than
    # one `|`: a scalar early return is the form this file already uses.
    if q < 0:
        return
    if q >= P:
        return

    read_att = (act == POOL_SWAP) | (act == POOL_MOVE_OUT)   # attended -> pool
    read_pool = (act == POOL_SWAP) | (act == POOL_MOVE_IN)   # pool -> attended

    offset_s = tl.arange(0, block_size)[:, None].to(tl.int64)   # (bs,1)
    offset_d = tl.arange(0, D)[None, :].to(tl.int64)            # (1,D)

    att_row = pid_m * block_size + offset_s
    pool_row = (TOPK + q) * block_size + offset_s

    mask_att = att_row < S_GPU
    mask_pool = pool_row < S_GPU

    # ---- K ------------------------------------------------------------------
    k_att = (k_ptr + pid_b * stride_b_k + att_row * stride_m_k
             + pid_h * stride_h_k + offset_d * stride_d_k)
    k_pool = (k_ptr + pid_b * stride_b_k + pool_row * stride_m_k
              + pid_h * stride_h_k + offset_d * stride_d_k)
    # Both sides are loaded BEFORE either is stored, so a SWAP is a real
    # exchange and not a double-write. The pool store follows the pool load of
    # the same address in program order, which the compiler must preserve
    # because the two provably alias.
    ka = tl.load(k_att, mask=mask_att & read_att, other=0)
    kp = tl.load(k_pool, mask=mask_pool & read_pool, other=0)
    tl.store(k_pool, ka, mask=mask_pool & read_att)
    tl.store(k_att, kp, mask=mask_att & read_pool)

    # ---- V ------------------------------------------------------------------
    v_att = (v_ptr + pid_b * stride_b_v + att_row * stride_m_v
             + pid_h * stride_h_v + offset_d * stride_d_v)
    v_pool = (v_ptr + pid_b * stride_b_v + pool_row * stride_m_v
              + pid_h * stride_h_v + offset_d * stride_d_v)
    va = tl.load(v_att, mask=mask_att & read_att, other=0)
    vp = tl.load(v_pool, mask=mask_pool & read_pool, other=0)
    tl.store(v_pool, va, mask=mask_pool & read_att)
    tl.store(v_att, vp, mask=mask_att & read_pool)

    # ---- kv_bias: (B, S_GPU, H), NO head_dim axis, so the row column alone
    # indexes it -- exactly the shape flash_h2d_mask_bias.py:45-56 uses.
    b_att = (bias_ptr + pid_b * stride_b_bias + att_row * stride_m_bias
             + pid_h * stride_h_bias)
    b_pool = (bias_ptr + pid_b * stride_b_bias + pool_row * stride_m_bias
              + pid_h * stride_h_bias)
    ba = tl.load(b_att, mask=mask_att & read_att, other=0)
    bp = tl.load(b_pool, mask=mask_pool & read_pool, other=0)
    tl.store(b_pool, ba, mask=mask_pool & read_att)
    tl.store(b_att, bp, mask=mask_att & read_pool)


def flash_pool_swap(
    k_gpu,          # (B, S_GPU, H, D)
    v_gpu,          # (B, S_GPU, H, D)
    kv_bias_buf,    # (B, S_GPU, H)
    pool_target,    # (H, B, M) int64
    pool_action,    # (H, B, M) int8
    topk: int,
    block_size: int = 64,
):
    """Apply one decode step's pool plan to the cache tensors, in place."""
    B, S_GPU, H, D = k_gpu.shape
    H2, B2, M = pool_target.shape
    assert H == H2 and B == B2, (k_gpu.shape, pool_target.shape)
    assert M == topk, f"pool_target must be [H,B,topk]; got M={M}, topk={topk}"
    assert pool_action.shape == pool_target.shape
    # P is DERIVED from the tensor the kernel will actually index, not passed
    # in, so a mismatch between the pool state's width and the cache tensor's
    # width cannot open the q >= P hole the kernel guards.
    P = S_GPU // block_size - topk
    assert P >= 1, f"no pool rows in the cache tensor: S_GPU={S_GPU}, topk={topk}"

    stride_b_k, stride_m_k, stride_h_k, stride_d_k = k_gpu.stride()
    stride_b_v, stride_m_v, stride_h_v, stride_d_v = v_gpu.stride()
    stride_b_bias, stride_m_bias, stride_h_bias = kv_bias_buf.stride()

    grid = (B, H, M)

    pool_swap_kernel[grid](
        k_gpu, v_gpu, kv_bias_buf, pool_target, pool_action,
        stride_b_k, stride_m_k, stride_h_k, stride_d_k,
        stride_b_v, stride_m_v, stride_h_v, stride_d_v,
        stride_b_bias, stride_m_bias, stride_h_bias,
        S_GPU, B, H, M, topk, P,
        block_size, D,
        num_warps=8,
    )
