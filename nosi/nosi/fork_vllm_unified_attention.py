# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

# [fork] FORK of vLLM 0.20.2 vllm/v1/attention/ops/triton_unified_attention.py (Apache-2.0; the header above is kept)
# [fork] with the @triton.jit helpers it imports from vllm/v1/attention/ops/triton_attention_helpers.py (the same
# [fork] revision) copied in. Pristine copies: benchmarks/Efficiency/tests_cpu/fixtures_vllm_0_20_2/. Every line that is
# [fork] not upstream text carries the tag "[fork]" (benchmarks/Efficiency/tests_cpu/test_r2_fia4.py FV2 checks it).
# [fork] Why (skill reusing-before-building, rung 4): FlashInfer 0.5.3 BatchPrefillWithPagedKVCacheWrapper + a custom
# [fork] mask was tried on our code (record /mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_f1a_2195573). Measured blocker:
# [fork] 1.68-1.70x the device time of piece 3 (2 CTAs per SM from 255 registers + 81 KB shared memory, one wave,
# [fork] latency-bound) and 77-100 ms of plan + mask build per tick. Plan: kstep-cpu-runs/analysis/plan/
# [fork] PLAN_speckv_on_cached_nosa.md, Task 0.3, survey option 2. The human's rule: fork the closest kernel, no other.
# [fork] What changes (docs/NOTES_fia4.md lists every region):
# [fork]  1. BLOCK_Q = the rows of one request (4: 1 draft + 3 verify) x the query heads of one KV head (16) = a 64-row
# [fork]     tile. Each KV tile (TILE_SIZE = one 64-token NOSA block) is loaded once and multiplied for all 64 rows.
# [fork]  2. The block table of a program = the UNION of its rows' selected blocks for (request, KV head), in ascending
# [fork]     block-id order. The program builds it on the device (BUILD_UNION), or it reads a table built before.
# [fork]  3. Per row and union entry: the number of tokens of that block the row sees (8 bits; 0 = not selected). The
# [fork]     kernel applies it to the SCORE tile (-inf), never to K / V copies. K / V / bias loads stop at the largest.
# [fork]  4. The CIS bias: logit = (q.k + bias) x scale, the shipped decode math (FA2-NOSA decode, piece 3).
# [fork]  5. The 2D path (no split-KV) is the default. The 3D split path stays behind its flag (the segment buffers).
# [fork] Removed (they need token positions; the union tile space has none): sliding window, chunked attention, ALiBi,
# [fork] the QQ bias, the PrefixLM ranges. The launcher refuses them. No vllm import (the image has no vllm).

from typing import Any

import torch

import enum  # [fork] KVQuantMode below
import triton  # [fork] plain Triton (vllm.triton_utils re-exports these two names)
import triton.language as tl  # [fork]


class KVQuantMode(enum.IntEnum):  # [fork] vllm.v1.kv_cache_interface.KVQuantMode (0.20.2): the values used here
    NONE = 0  # [fork]
    FP8_PER_TENSOR = 1  # [fork]
    INT8_PER_TOKEN_HEAD = 2  # [fork]
    FP8_PER_TOKEN_HEAD = 3  # [fork]


# [fork] the @triton.jit helpers of vllm/v1/attention/ops/triton_attention_helpers.py (vLLM 0.20.2) follow
# ===========================================================================
# Scalar helpers (reused by every kernel + reduce_segments)
# ===========================================================================


@triton.jit
def cdiv_fn(x, y):
    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    """Softcap (aka tanh-style clamp) used to bound attention scores.

    ``x * tanh(S / x)`` rewritten to avoid a direct ``tanh`` call.
    """
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


# ===========================================================================
# Attention loop
# ===========================================================================


@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).
    """
    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Binary search over the cumulative query-length prefix.

    When ``use_q_block_mode`` is True, the prefix values are reshaped
    into units of ``BLOCK_Q`` plus one entry per boundary — matching
    the q-block grid laid out by the attention kernels.  When False
    we search the plain cumulative-length prefix (used by
    ``reduce_segments`` which iterates over raw query tokens).
    """
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Initial row-max ``M`` for the online softmax.

    Without sinks: ``-inf``.  With sinks: load the per-head sink bias
    once.  In 3D mode only segment 0 loads — ``reduce_segments`` adds
    the sink contribution exactly once across segments, so other
    segments must start from ``-inf``.

    ``segm_idx_or_0`` is the 3D segment index or 0 for 2D (caller
    passes ``0`` when ``IS_3D`` is False).
    """
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


# [fork] removed: compute_tile_loop_bounds (positions, sliding window); the fork loops over the union entries


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Store per-segment ``M`` and ``L`` for ``reduce_segments`` to
    combine into the final softmax.

    Shared across every 3D attention epilogue; the per-token output
    stripes are mode-specific (flat / 2-stream split / 4-stream split)
    and stay inlined.
    """
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# [fork] removed: compute_kv_seq_mask (the causal / window mask by position); the fork masks by the rows' extents


# [fork] removed: apply_alibi_to_score and load_qq_bias_tile (both read token positions)


@triton.jit
def softmax_step(S, M, L):
    """Online softmax update for one tile.

    Returns ``(M_new, L_new, P, alpha)``.  Caller is responsible for
    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside so
    kernels with a different number / shape of accumulators can reuse
    the same step.
    """
    # compute running maximum
    # m_j : (BLOCK_M,)
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # For sliding window there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    # P : (BLOCK_M, TILE_SIZE)
    P = tl.exp(S - m_j[:, None])
    # l_j : (BLOCK_M,)
    l_j = tl.sum(P, axis=1)
    # alpha : (BLOCK_M, )
    alpha = tl.exp(M - m_j)
    # update constants
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha

# [fork] removed: logger (unused)
is_batch_invariant = False  # [fork] no vllm envs here; the fork has one schedule (no split unless asked)
float8_info = torch.finfo(torch.float8_e4m3fn)  # [fork] = current_platform.fp8_dtype() on CUDA (platforms/interface.py)


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


@triton.jit  # [fork] new: the union table of one program, built by the program itself (no host plan, no launch)
def fork_build_union(  # [fork]
    row_key_ptr,  # [fork] (R, Hkv, P) int32: the block id of each selected page of each row (canonical lists)
    row_slot_ptr,  # [fork] (R, Hkv, P) int32: the slot of that page (the strides of row_key)
    row_npages_ptr,  # [fork] (R, Hkv) int32: the listed pages per (row, KV head)
    row_last_ptr,  # [fork] (R,) int32: the tokens the row sees in its LAST listed page (1 .. BLOCK_SIZE)
    row_entry_ptr,  # [fork] (R,) int32: the row's storage entry; physical block = entry * slots_per_entry + slot
    table_ptr,  # [fork] out: this program's row of the union table (physical blocks)
    ext_ptr,  # [fork] out: this program's row of the packed extents (byte u = row u)
    len_ptr,  # [fork] out: this program's union length
    row0,  # [fork] the request's first row
    n_rows,  # [fork] the request's rows (<= BLOCK_Q)
    kv_head_idx,  # [fork]
    stride_rk_0,  # [fork] row_key / row_slot: the row stride
    stride_rk_1,  # [fork] row_key / row_slot: the KV-head stride (the last stride is 1)
    stride_rn_0,  # [fork] row_npages: the row stride (the last stride is 1)
    row_width,  # [fork] P, the list width
    slots_per_entry,  # [fork] S
    BLOCK_Q: tl.constexpr,  # [fork]
    P_PAD: tl.constexpr,  # [fork] a power of two >= P
    STEPS: tl.constexpr,  # [fork] binary-search steps: log2(P_PAD) + 1
    BLOCK_SIZE: tl.constexpr,  # [fork]
):  # [fork]
    # [fork] List entry e = (row u, position i) of the BLOCK_Q rows; its value v = block id << 32 | slot (the order and
    # [fork] the de-duplication of the bench's torch union, fp_union). The union entry of e = the number of DISTINCT
    # [fork] values below v_e (ascending block-id order: union entry 0 is block 0). The row-r byte of that entry = the
    # [fork] tokens row r sees in the block: BLOCK_SIZE, last_len for the row's last listed page, 0 when row r does not
    # [fork] list it. PRECONDITION (piece 3's canonical order, build_page_table): each row lists its pages in strictly
    # [fork] ascending block id. Then a lower bound by binary search gives the position of v_e in every row (no sort,
    # [fork] no pairwise compare, no host read): entry e is FIRST when no earlier row lists v_e; rank_e = the sum over
    # [fork] rows r of the FIRST entries of row r below v_e (an inclusive count C, kept in this program's ext row as
    # [fork] scratch until the final stores). All entries of one value store equal values (cf. our sharedkv3 union
    # [fork] kernel, nosi/nosi/bi_paged_attention_shared.py _union_kernel: the same first / rank definitions).
    NE: tl.constexpr = BLOCK_Q * P_PAD  # [fork]
    offs_e = tl.arange(0, NE)  # [fork]
    u_e = offs_e // P_PAD  # [fork]
    i_e = offs_e % P_PAD  # [fork]
    row_e = row0 + u_e  # [fork]
    np_e = tl.load(row_npages_ptr + row_e * stride_rn_0 + kv_head_idx, mask=u_e < n_rows, other=0)  # [fork]
    ok_e = (u_e < n_rows) & (i_e < np_e) & (i_e < row_width)  # [fork]
    lst_e = row_e * stride_rk_0 + kv_head_idx * stride_rk_1 + i_e  # [fork]
    slot_e = tl.load(row_slot_ptr + lst_e, mask=ok_e, other=0)  # [fork]
    v_e = (tl.load(row_key_ptr + lst_e, mask=ok_e, other=0).to(tl.int64) << 32) | slot_e.to(tl.int64)  # [fork]
    word_e = tl.zeros([NE], dtype=tl.int32)  # [fork] the packed extents of e's union entry
    dup_e = tl.zeros([NE], dtype=tl.int32)  # [fork] earlier rows that list v_e
    for r in tl.static_range(BLOCK_Q):  # [fork] pass 1: the bytes and the FIRST flags
        pos, hit, ext = fork_row_lower_bound(  # [fork]
            v_e, row_key_ptr, row_slot_ptr, row_npages_ptr, row_last_ptr, row0 + r, r < n_rows, kv_head_idx,  # [fork]
            stride_rk_0, stride_rk_1, stride_rn_0, NE, STEPS, BLOCK_SIZE)  # [fork]
        word_e += tl.where(hit, ext << (8 * r), 0)  # [fork] row r's byte (disjoint bytes: the sum is the or)
        dup_e += tl.where(hit & (u_e > r), 1, 0)  # [fork]
    first_e = (ok_e & (dup_e == 0)).to(tl.int32)  # [fork]
    # [fork] The count runs in int64 and is cast to int32 before the store. Triton 3.2's interpreter gives the cumsum of
    # [fork] an int32 vector as int64 numpy data still typed int32 (np.cumsum promotes; interpreter.py:729-735,
    # [fork] :849-850), and its store copies the data's 8 bytes per element: one int32 past this row was written (the
    # [fork] next row's entry 0 became 0; fia4 CPU run 02bc4c4). An int64 -> int32 cast is a real cast in both backends.
    c_first = tl.cumsum(first_e.to(tl.int64), axis=0).to(tl.int32)  # [fork]
    tl.store(ext_ptr + offs_e, c_first)  # [fork] scratch: C[e] = FIRST entries at or before e
    tl.debug_barrier()  # [fork]
    rank_e = tl.zeros([NE], dtype=tl.int32)  # [fork]
    for r in tl.static_range(BLOCK_Q):  # [fork] pass 2: rank_e = sum over r of C[r P_PAD + pos - 1] - C[r P_PAD - 1]
        pos, hit, ext = fork_row_lower_bound(  # [fork]
            v_e, row_key_ptr, row_slot_ptr, row_npages_ptr, row_last_ptr, row0 + r, r < n_rows, kv_head_idx,  # [fork]
            stride_rk_0, stride_rk_1, stride_rn_0, NE, STEPS, BLOCK_SIZE)  # [fork]
        c_hi = tl.load(ext_ptr + r * P_PAD + pos - 1, mask=pos > 0, other=0)  # [fork]
        c_lo = tl.load(ext_ptr + (r * P_PAD - 1) + 0 * offs_e, mask=(pos > 0) & (offs_e * 0 + r > 0), other=0)  # [fork]
        rank_e += c_hi - c_lo  # [fork]
    tl.debug_barrier()  # [fork] every read of the scratch is done before the final stores
    entry_e = tl.load(row_entry_ptr + row_e, mask=u_e < n_rows, other=0)  # [fork]
    keep = ok_e & (rank_e < NE)  # [fork] never outside this program's row (only an unsorted list could reach NE)
    tl.store(table_ptr + rank_e, entry_e * slots_per_entry + slot_e, mask=keep)  # [fork]
    tl.store(ext_ptr + rank_e, word_e, mask=keep)  # [fork]
    num_union = tl.sum(first_e, axis=0)  # [fork]
    tl.store(len_ptr, num_union)  # [fork]
    return num_union  # [fork]


@triton.jit  # [fork] new: the lower bound of v_e in row r's sorted list (and the row's byte when it lists v_e)
def fork_row_lower_bound(  # [fork]
    v_e, row_key_ptr, row_slot_ptr, row_npages_ptr, row_last_ptr, row_r, valid_r, kv_head_idx,  # [fork]
    stride_rk_0, stride_rk_1, stride_rn_0,  # [fork]
    NE: tl.constexpr, STEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr,  # [fork]
):  # [fork]
    np_r = tl.load(row_npages_ptr + row_r * stride_rn_0 + kv_head_idx, mask=valid_r, other=0)  # [fork]
    base = row_r * stride_rk_0 + kv_head_idx * stride_rk_1  # [fork]
    lo = tl.zeros([NE], dtype=tl.int32)  # [fork]
    hi = tl.zeros([NE], dtype=tl.int32) + np_r  # [fork]
    for _ in tl.static_range(STEPS):  # [fork] while lo < hi: the first position whose value is >= v_e
        mid = (lo + hi) // 2  # [fork]
        live = mid < hi  # [fork]
        v_m = (tl.load(row_key_ptr + base + mid, mask=live, other=0).to(tl.int64) << 32) | tl.load(  # [fork]
            row_slot_ptr + base + mid, mask=live, other=0).to(tl.int64)  # [fork]
        right = live & (v_m < v_e)  # [fork]
        lo = tl.where(right, mid + 1, lo)  # [fork]
        hi = tl.where(live & (v_m >= v_e), mid, hi)  # [fork]
    at = lo < np_r  # [fork]
    v_a = (tl.load(row_key_ptr + base + lo, mask=at, other=0).to(tl.int64) << 32) | tl.load(  # [fork]
        row_slot_ptr + base + lo, mask=at, other=0).to(tl.int64)  # [fork]
    hit = at & (v_a == v_e)  # [fork]
    last_r = tl.load(row_last_ptr + row_r, mask=valid_r, other=0)  # [fork]
    ext = tl.where(lo == np_r - 1, last_r, BLOCK_SIZE)  # [fork]
    return lo, hit, ext  # [fork]


@triton.jit
def kernel_unified_attention(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into the
    # three ``segm_*`` tensors and ``output_ptr`` is unused (callers may
    # pass any non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Per-(token, head) scale caches (used iff KV_QUANT_MODE in {2, 3}).
    # For other modes callers may pass any non-null pointer.
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    cis_bias_ptr,  # [fork] CIS bias [num_blocks, block_size, num_kv_heads] (the per-token-head scale layout)
    union_ext_ptr,  # [fork] per union entry: the packed extents of the BLOCK_Q rows (block_tables_ptr's layout)
    union_len_ptr,  # [fork] [num_kv_heads, num_seqs] int32: the union entries per (KV head, request)
    row_key_ptr,  # [fork] BUILD_UNION inputs (fork_build_union); any pointer otherwise
    row_slot_ptr,  # [fork]
    row_npages_ptr,  # [fork]
    row_last_ptr,  # [fork]
    row_entry_ptr,  # [fork]
    # Scalars
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    stride_b_blk: tl.int64,  # [fork] cis_bias strides
    stride_b_slot: tl.int64,  # [fork]
    stride_b_head: tl.int64,  # [fork]
    stride_rk_0: tl.int64,  # [fork] row_key / row_slot strides (row, KV head)
    stride_rk_1: tl.int64,  # [fork]
    stride_rn_0: tl.int64,  # [fork] row_npages row stride
    row_width,  # [fork] P, the width of the row lists
    slots_per_entry,  # [fork] S: physical block = storage entry * S + slot
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    BUILD_UNION: tl.constexpr,  # [fork] True: build the union table in the program; False: read it
    USE_CIS_BIAS: tl.constexpr,  # [fork]
    P_PAD: tl.constexpr,  # [fork] a power of two >= row_width (BUILD_UNION)
    BUILD_STEPS: tl.constexpr,  # [fork] fork_build_union's binary-search steps
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3).
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # [fork] The program's KV = the UNION of its rows' selected blocks for (seq_idx, kv_head_idx): one TILE per union
    # [fork] entry (TILE_SIZE == BLOCK_SIZE), ascending block id. Its table row: (kv_head_idx, seq_idx). seq_len above is
    # [fork] not used (no positions here).
    union_row = kv_head_idx * num_seqs + seq_idx  # [fork]
    block_table_offset = union_row.to(tl.int64) * block_table_stride  # [fork] (moved up from below: the union table)
    if BUILD_UNION:  # [fork] the program builds its own table: no host plan, no extra launch
        num_union = fork_build_union(  # [fork]
            row_key_ptr, row_slot_ptr, row_npages_ptr, row_last_ptr, row_entry_ptr,  # [fork]
            block_tables_ptr + block_table_offset, union_ext_ptr + block_table_offset,  # [fork]
            union_len_ptr + union_row, cur_batch_in_all_start_index, cur_batch_query_len, kv_head_idx,  # [fork]
            stride_rk_0, stride_rk_1, stride_rn_0, row_width, slots_per_entry,  # [fork]
            BLOCK_Q, P_PAD, BUILD_STEPS, BLOCK_SIZE,  # [fork]
        )  # [fork]
        tl.debug_barrier()  # [fork] the table stores are visible to every thread of the program (as _union_kernel)
    else:  # [fork]
        num_union = tl.load(union_len_ptr + union_row)  # [fork] a table built before the launch

    if IS_3D:
        tiles_per_segment = cdiv_fn(num_union, NUM_SEGMENTS_PER_SEQ)  # [fork] tiles = union entries
        if segm_idx * tiles_per_segment >= num_union:  # [fork]
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # [fork] block_table_offset is set above (the union table row of (kv_head_idx, seq_idx))

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # [fork] removed: context_len and the ALiBi / QQ-bias set-up (token positions)
    row_shift = 8 * (offs_m // num_queries_per_kv)  # [fork] the bit offset of each tile row's byte in a union_ext word

    if IS_3D:  # [fork] the tile loop runs over the union entries (was compute_tile_loop_bounds, by position)
        loop_lo = segm_idx * tiles_per_segment  # [fork]
        loop_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, num_union)  # [fork]
    else:  # [fork]
        loop_lo = 0  # [fork]
        loop_hi = num_union  # [fork]

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        ext_word = tl.load(union_ext_ptr + block_table_offset + j)  # [fork] union entry j: the rows' packed extents
        row_ext = (ext_word >> row_shift) & 0xFF  # [fork] (BLOCK_M,): the tokens of this block each tile row sees
        reach = tl.maximum(  # [fork] the largest of the (<= 4) bytes, as scalar arithmetic (no cross-thread reduce)
            tl.maximum(ext_word & 0xFF, (ext_word >> 8) & 0xFF),  # [fork]
            tl.maximum((ext_word >> 16) & 0xFF, (ext_word >> 24) & 0xFF),  # [fork]
        )  # [fork]
        tile_mask = offs_t < reach  # [fork] load only tokens some row sees (was: seq_offset < max_seq_prefix_len)

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        if USE_CIS_BIAS:  # [fork] the CIS bias of each token, read through the block like the per-token-head scales
            cis_idx = (  # [fork]
                physical_block_idx * stride_b_blk  # [fork]
                + (seq_offset % BLOCK_SIZE) * stride_b_slot  # [fork]
                + kv_head_idx * stride_b_head  # [fork]
            )  # [fork]
            cis = tl.load(cis_bias_ptr + cis_idx, mask=tile_mask, other=0.0).to(tl.float32)  # [fork]
        else:  # [fork]
            cis = tl.zeros([TILE_SIZE], dtype=tl.float32)  # [fork]

        seq_mask = offs_t[None, :] < row_ext[:, None]  # [fork] the row's block bit + its extent (was the causal mask)

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += (tl.dot(Q, K) * k_token_head_scales[None, :] + cis[None, :]) * scale  # [fork] + the CIS bias
        else:
            S += (tl.dot(Q, K) + cis[None, :]) * scale  # [fork] (q.k + bias) x scale: the shipped decode math

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # [fork] removed: the ALiBi and QQ-bias terms (token positions)

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        # [fork] removed: the sliding-window V mask (token positions)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by ``reduce_segments``.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    union_len_ptr,  # [fork] [num_kv_heads, num_seqs] int32: the union entries (the tiles) per (KV head, request)
    num_queries_per_kv: tl.constexpr,  # [fork]
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # [fork] the tiles of this (KV head, request) = its union entries (was: seq_len tokens)
    num_union = tl.load(union_len_ptr + (query_head_idx // num_queries_per_kv) * num_seqs + seq_idx)  # [fork]

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(num_union, num_segments)  # [fork] as the kernel's IS_3D branch

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(num_union, tiles_per_segment)  # [fork]
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32


def _fork_inputs(q, k, block_table, union_ext, union_len, cis_bias, build_union, block_q, block_m,  # [fork]
                 max_seqlen_q, num_seqs):  # [fork]
    """[fork] Metadata checks (no device read) and the extra kernel arguments of the fork: the bias strides and the
    [fork] build inputs (stand-in pointers when the kernel reads a table built before; never dereferenced then)."""
    num_kv_heads, block_size = k.shape[2], k.shape[1]  # [fork]
    assert block_q in (1, 2, 4) and max_seqlen_q <= block_q, "[fork] one q block (<= 4 rows) per request"  # [fork]
    assert block_m >= 16 and block_m == triton.next_power_of_2(block_m), "[fork] the stacked tile rows"  # [fork]
    assert block_size == triton.next_power_of_2(block_size) and block_size <= 128, "[fork] extents fit 8 bits"  # [fork]
    w = block_table.shape[-1]  # [fork]
    for t in (block_table, union_ext):  # [fork]
        assert t.dtype == torch.int32 and t.is_contiguous(), "[fork] the union tables: contiguous int32"  # [fork]
        assert tuple(t.shape) == (num_kv_heads, num_seqs, w), "[fork] the union tables: [Hkv, num_seqs, W]"  # [fork]
    assert union_len.dtype == torch.int32 and union_len.is_contiguous(), "[fork] union_len: contiguous int32"  # [fork]
    assert tuple(union_len.shape) == (num_kv_heads, num_seqs), "[fork] union_len: [Hkv, num_seqs]"  # [fork]
    b_strides = (0, 0, 0)  # [fork]
    if cis_bias is not None:  # [fork]
        assert tuple(cis_bias.shape) == (k.shape[0], block_size, num_kv_heads), "[fork] cis_bias: k's pages"  # [fork]
        b_strides = cis_bias.stride()  # [fork]
    out = dict(b_strides=b_strides, key=block_table, slot=block_table, npages=union_len, last=union_len,  # [fork]
               entry=union_len, rk=(0, 0), rn=0, width=1, slots=1, p_pad=16, steps=1)  # [fork]
    if build_union is not None:  # [fork]
        key, slot, npg = build_union["page_key"], build_union["page_table"], build_union["num_pages"]  # [fork]
        last, entry = build_union["last_len"], build_union["seq_idx"]  # [fork]
        r_, hkv, p_ = key.shape  # [fork]
        for t in (key, slot, npg, last, entry):  # [fork]
            assert t.dtype == torch.int32 and t.is_contiguous(), "[fork] the row lists: contiguous int32"  # [fork]
        assert tuple(slot.shape) == (r_, hkv, p_) and hkv == num_kv_heads and r_ == q.shape[0]  # [fork]
        assert tuple(npg.shape) == (r_, hkv) and tuple(last.shape) == (r_,) == tuple(entry.shape)  # [fork]
        p_pad = triton.next_power_of_2(p_)  # [fork]
        assert w >= block_q * p_pad, "[fork] the table width W must hold BLOCK_Q x P_PAD entries"  # [fork]
        out.update(key=key, slot=slot, npages=npg, last=last, entry=entry, rk=(key.stride(0), key.stride(1)),  # [fork]
                   rn=npg.stride(0), width=p_, slots=int(build_union["slots_per_entry"]), p_pad=p_pad,  # [fork]
                   steps=p_pad.bit_length())  # [fork] log2(P_PAD) + 1: a lower bound over 0 .. P
        assert key.stride(2) == 1 and slot.stride() == key.stride() and npg.stride(1) == 1  # [fork]
    return out  # [fork]


def _fork_launch_options(num_warps, num_stages):  # [fork]
    """[fork] Triton launch options; empty = Triton's defaults (upstream passes none)."""
    opts = {}  # [fork]
    if num_warps is not None:  # [fork]
        opts["num_warps"] = int(num_warps)  # [fork]
    if num_stages is not None:  # [fork]
        opts["num_stages"] = int(num_stages)  # [fork]
    return opts  # [fork]


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
    # [fork] block_table above is the UNION TABLE [num_kv_heads, num_seqs, W] int32 (physical blocks; contiguous):
    # [fork] written by the kernel when build_union is given, else read. Causality is in the rows' extents.
    union_ext=None,  # [fork] [num_kv_heads, num_seqs, W] int32: per entry the BLOCK_Q rows' visible tokens, 8 bits each
    union_len=None,  # [fork] [num_kv_heads, num_seqs] int32: the union entries per (KV head, request)
    cis_bias=None,  # [fork] [num_blocks, block_size, num_kv_heads] (k's dtype): the CIS bias; None = no bias
    build_union=None,  # [fork] None, or dict(page_key, page_table, num_pages, last_len, seq_idx, slots_per_entry)
    rows_per_q_block=4,  # [fork] BLOCK_Q: the rows of one request (1 draft + 3 verify)
    num_warps=None,  # [fork] launch overrides; None = Triton's defaults (as upstream)
    num_stages=None,  # [fork]
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"
    assert window_size[0] < 0 and chunk_lookback == -1, "[fork] no sliding window / chunked attention"  # [fork]
    assert alibi_slopes is None and qq_bias is None and mm_prefix_range is None, "[fork] position-based"  # [fork]
    assert kv_quant_mode == KVQuantMode.NONE, "[fork] only an unquantized KV cache is tested here"  # [fork]
    assert union_ext is not None and union_len is not None, "[fork] the union tables are required"  # [fork]

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_Q = rows_per_q_block  # [fork] all rows of a request in ONE program: each KV tile is loaded once for them
    BLOCK_M = BLOCK_Q * num_queries_per_kv  # [fork] 4 x 16 = 64 tile rows: one tl.dot over the stacked tile
    fk = _fork_inputs(q, k, block_table, union_ext, union_len, cis_bias, build_union, BLOCK_Q, BLOCK_M,  # [fork]
                      max_seqlen_q, num_seqs)  # [fork]

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = block_size  # [fork] one tile = one block = one union entry (a tile never spans two blocks)
    TILE_SIZE_DECODE = block_size  # [fork]

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > BLOCK_Q  # [fork] the verify rows: up to BLOCK_Q tokens of a request in one program
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are required arguments; non-per-token-head
    # modes pass dummy zeros (the code path is dead-code eliminated by
    # the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch in the kernel).
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = 0
        vs_blk = vs_slot = vs_head = 0
        # Pass the K cache as a stand-in pointer; never dereferenced.
        k_scale_ptr = k
        v_scale_ptr = v

    # 3D needs real segm tensors; 2D never touches them but Triton wants
    # a non-null pointer.  Reuse ``out`` as the placeholder.
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        cis_bias_ptr=cis_bias if cis_bias is not None else k,  # [fork] a stand-in pointer without bias (never read)
        union_ext_ptr=union_ext,  # [fork]
        union_len_ptr=union_len,  # [fork]
        row_key_ptr=fk["key"],  # [fork]
        row_slot_ptr=fk["slot"],  # [fork]
        row_npages_ptr=fk["npages"],  # [fork]
        row_last_ptr=fk["last"],  # [fork]
        row_entry_ptr=fk["entry"],  # [fork]
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(1),  # [fork] [num_kv_heads, num_seqs, W]: the (head, request) row stride
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        stride_b_blk=fk["b_strides"][0],  # [fork]
        stride_b_slot=fk["b_strides"][1],  # [fork]
        stride_b_head=fk["b_strides"][2],  # [fork]
        stride_rk_0=fk["rk"][0],  # [fork]
        stride_rk_1=fk["rk"][1],  # [fork]
        stride_rn_0=fk["rn"],  # [fork]
        row_width=fk["width"],  # [fork]
        slots_per_entry=fk["slots"],  # [fork]
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        BUILD_UNION=build_union is not None,  # [fork]
        USE_CIS_BIAS=cis_bias is not None,  # [fork]
        P_PAD=fk["p_pad"],  # [fork]
        BUILD_STEPS=fk["steps"],  # [fork]
        KV_QUANT_MODE=kv_quant_mode,
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        **_fork_launch_options(num_warps, num_stages),  # [fork]
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
            union_len_ptr=union_len,  # [fork]
            num_queries_per_kv=num_queries_per_kv,  # [fork]
        )
