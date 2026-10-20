"""R2 milestone 2, step 4 (KERNELS), item 2: NOSA's block pooling with PER-REQUEST lengths.

The shipped kernel (max_pooling_fused.py:7-75 nosa_pool_kernel, launched by nosa_pooling :79-107; captured in the
decode's after-pooling graph at nosa_llama.py:462) takes ONE scalar max_seq_len (the compressed length, frozen when
the graph is captured) and takes M, the pooled width that places the local window (:61, M - m <= local_blocks), from
the output buffer's shape: every request of the batch is one length. This variant reads, per request b, n_comp[b]
(its compressed length: the mask of the window loads) and M_b[b] (its pooled width: the local window and the valid
range) from DEVICE tensors, so a captured graph can serve per-request lengths that change between replays. Pooled
blocks m >= M_b[b] are written -inf (both outputs). Everything else is the shipped kernel's arithmetic in the shipped
order: the same 8-lane loads with the same mask (set_size 5, padding 1), the same tl.max, the same +inf boundary
where, the same stores. With every n_comp[b] equal to the shipped max_seq_len and every M_b[b] equal to the buffer
width M, its outputs are bitwise the shipped kernel's (the extra where keeps every stored value); the GPU gate
(benchmarks/Efficiency/r2_kernels_gate.py G4) checks torch.equal.

pool_reference is the same arithmetic in pure torch (CPU or GPU; the CPU tests' reference and the gate's G6): a max
over the gathered window values (no arithmetic), so it is exact in any dtype.

Constraints kept from the shipped launcher: q_score / cis_score (H, B, N), pooled (H, B, M); the constexprs
kernel_size / stride are passed and unused by the body (as shipped); local_blocks + 1 is passed (as shipped). The
launcher cannot check M_b[b] <= M without a device read; a larger M_b[b] is the caller's error (the gate checks it).
"""
from __future__ import annotations

import torch

try:                                                   # triton is needed only to launch the kernel (not for the
    import triton                                      # reference): a CPU test without triton imports this module
    import triton.language as tl
except ImportError:                                    # pragma: no cover - environment dependent
    triton = tl = None

SET_SIZE, PADDING = 5, 1                               # nosa_pooling's defaults (max_pooling_fused.py:79)


def pool_reference(q_score, cis_score, n_comp, M_b, M: int, init_blocks: int = 1, local_blocks: int = 16,
                   set_size: int = SET_SIZE, padding: int = PADDING, out_dtype=None):
    """The kernel's arithmetic in torch. q_score / cis_score (H, B, N); n_comp, M_b (B,) int (any device). Returns
    (pooled_q, pooled_cis) (H, B, M) in out_dtype (default the input dtype): for m < M_b[b], the max over the loads
    n = m (set_size - 1) - padding + i, 0 <= i < set_size, 0 <= n < n_comp[b] (-inf when none), then +inf when
    m < init_blocks or M_b[b] - m <= local_blocks + 1; -inf for m >= M_b[b]."""
    H, B, N = q_score.shape
    dev = q_score.device
    out_dtype = out_dtype or q_score.dtype
    nc = torch.as_tensor(n_comp, device=dev).to(torch.int64).view(1, B, 1, 1)
    mb = torch.as_tensor(M_b, device=dev).to(torch.int64).view(1, B, 1)
    m = torch.arange(M, device=dev).view(1, 1, M)
    n = m.unsqueeze(-1) * (set_size - 1) - padding + torch.arange(set_size, device=dev).view(1, 1, 1, set_size)
    load = (n >= 0) & (n < nc)                                                       # (1, B, M, set_size)
    idx = n.clamp(0, N - 1).expand(H, B, M, set_size).reshape(H, B, M * set_size)
    neg = torch.tensor(float("-inf"), dtype=q_score.dtype, device=dev)
    pos = torch.tensor(float("inf"), dtype=out_dtype, device=dev)
    out = []
    for s in (q_score, cis_score):
        g = torch.gather(s, 2, idx).view(H, B, M, set_size)
        acc = torch.where(load.expand(H, B, M, set_size), g, neg).max(dim=-1).values.to(out_dtype)
        boundary = (m < init_blocks) | (mb - m <= local_blocks + 1)
        acc = torch.where(boundary.expand(H, B, M), pos, acc)
        acc = torch.where((m < mb).expand(H, B, M), acc, -pos)
        out.append(acc)
    return out[0], out[1]


if triton is not None:
    @triton.jit
    def nosa_pool_kernel_per_request(
        q_ptr,              # [H, B, N]
        cis_ptr,            # [H, B, N]
        n_comp_ptr,         # [B] int32: request b's compressed length (the shipped scalar max_seq_len)
        m_b_ptr,            # [B] int32: request b's pooled width (the shipped M, per request)
        pooled_q_ptr,       # [H, B, M]
        pooled_cis_ptr,     # [H, B, M]
        q_stride_h,
        q_stride_b,
        q_stride_n,
        cis_stride_h,
        cis_stride_b,
        cis_stride_n,
        pooled_q_stride_h,
        pooled_q_stride_b,
        pooled_q_stride_m,
        pooled_cis_stride_h,
        pooled_cis_stride_b,
        pooled_cis_stride_m,
        H,
        B,
        M,                  # the buffer width (the grid); the valid width of request b is M_b[b] <= M
        kernel_size: tl.constexpr,
        stride: tl.constexpr,
        padding: tl.constexpr,
        set_size: tl.constexpr,
        init_blocks: tl.constexpr,
        local_blocks: tl.constexpr,
    ):
        # grid: (H, B, M)
        tidx_h = tl.program_id(0)  # head
        tidx_b = tl.program_id(1)  # batch idx
        tidx_m = tl.program_id(2)  # window idx
        max_seq_len = tl.load(n_comp_ptr + tidx_b)                # per request (shipped: a scalar argument)
        M_b = tl.load(m_b_ptr + tidx_b)                           # per request (shipped: the buffer width M)

        block_idx = tl.arange(0, 8)
        n_offset = tidx_m * (set_size - 1) - padding
        q_beg_pos = q_ptr + tidx_h * q_stride_h + tidx_b * q_stride_b + n_offset * q_stride_n
        q_block_ptrs = q_beg_pos + block_idx * q_stride_n
        cis_beg_pos = cis_ptr + tidx_h * cis_stride_h + tidx_b * cis_stride_b + n_offset * cis_stride_n
        cis_block_ptrs = cis_beg_pos + block_idx * cis_stride_n
        mask = (block_idx + n_offset < max_seq_len) & (block_idx + n_offset >= 0) & (block_idx < set_size)
        q_block_scores = tl.load(
            q_block_ptrs,
            mask=mask,
            other=-float("inf"),
        )
        acc_q = tl.max(q_block_scores, axis=0)
        cis_block_scores = tl.load(
            cis_block_ptrs,
            mask=mask,
            other=-float("inf"),
        )
        acc_cis = tl.max(cis_block_scores, axis=0)

        boundary_mask = (tidx_m < init_blocks) | (M_b - tidx_m <= local_blocks)
        out_q = tl.where(boundary_mask, float("inf"), acc_q)
        out_cis = tl.where(boundary_mask, float("inf"), acc_cis)
        valid = tidx_m < M_b                                      # per request: beyond M_b[b] nothing is a block
        out_q = tl.where(valid, out_q, -float("inf"))
        out_cis = tl.where(valid, out_cis, -float("inf"))

        tl.store(
            pooled_q_ptr + tidx_b * pooled_q_stride_b + tidx_h * pooled_q_stride_h + tidx_m * pooled_q_stride_m,
            out_q,
            mask=tidx_m < M,
        )

        tl.store(
            pooled_cis_ptr + tidx_b * pooled_cis_stride_b + tidx_h * pooled_cis_stride_h + tidx_m * pooled_cis_stride_m,
            out_cis,
            mask=tidx_m < M,
        )


def check_lengths(t, B: int, name: str, device) -> None:
    if not torch.is_tensor(t) or t.dtype != torch.int32 or not t.is_contiguous() or tuple(t.shape) != (B,):
        raise ValueError("nosa_pooling_per_request: %s must be a contiguous int32 tensor of shape (%d,) (got %s %s)"
                         % (name, B, getattr(t, "dtype", type(t)), tuple(getattr(t, "shape", ()))))
    if t.device != device:
        raise ValueError("nosa_pooling_per_request: %s is on %s, the scores on %s" % (name, t.device, device))


def nosa_pooling_per_request(q_score, cis_score, n_comp, M_b, q_score_pooled, cis_score_pooled, block_size=32,
                             stride=16, padding=PADDING, set_size=SET_SIZE, init_blocks=1, local_blocks=16):
    """nosa_pooling (max_pooling_fused.py:79-107) with per-request n_comp / M_b (B,) int32 device tensors in place of
    the scalar max_seqlen and the buffer-width M. The grid is the shipped one (H, B, M), M = the pooled buffer width."""
    if triton is None:
        raise RuntimeError("nosa_pooling_per_request needs triton (pool_reference is the torch twin)")
    H, B = q_score.shape[:2]
    M = q_score_pooled.shape[-1]
    check_lengths(n_comp, B, "n_comp", q_score.device)
    check_lengths(M_b, B, "M_b", q_score.device)
    grid = (H, B, M)
    nosa_pool_kernel_per_request[grid](
        q_score,
        cis_score,
        n_comp,
        M_b,
        q_score_pooled,
        cis_score_pooled,
        q_score.stride(0),
        q_score.stride(1),
        q_score.stride(2),
        cis_score.stride(0),
        cis_score.stride(1),
        cis_score.stride(2),
        q_score_pooled.stride(0),
        q_score_pooled.stride(1),
        q_score_pooled.stride(2),
        cis_score_pooled.stride(0),
        cis_score_pooled.stride(1),
        cis_score_pooled.stride(2),
        H, B, M, block_size, stride, padding, set_size, init_blocks, local_blocks + 1,
    )
    return
