"""R2 milestone 2, step 4 (KERNELS), item 1: the InfLLM-v2 stage-1 call WITH PER-REQUEST KEY LENGTHS (seqused_k).

WHY. The C++ op infllm_cuda.varlen_fwd_stage1 already accepts seqused_k (dependencies/infllmv2_cuda_impl/csrc/
flash_attn/flash_api.cpp:791 in the signature, :901-906 its checks: int32, CUDA, contiguous, shape (batch,)); it reaches
params.seqused_k (:958), BlockInfo honours it (src/block_info.h:23: actual_seqlen_k = seqused_k[b] - leftpad_k), the
kernel bounds its key blocks by it (src/flash_fwd_kernel.h:1230: n_block_max = min(ceil(actual_seqlen_k / kBlockN),
...)), and stage 1 never splits (src/flash_fwd_launch_template.h:189: Split = false; flash_api.cpp:1013 num_splits = 1).
The shipped Python wrapper infllmv2_attn_stage1_fast (dependencies/infllmv2_cuda_impl/infllm_v2/
infllmv2_sparse_attention.py:625-729) passes None at that position (:711). This module is a NEW wrapper: the shipped
wrapper's body unchanged, plus seqused_k at the op's 7th argument. It calls the INSTALLED extension (infllm_v2.C, the
module the shipped wrapper uses), imported lazily, never at module import; the existing wrapper is not touched.

    stage1_seqused(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k=None, ...)

With seqused_k None it issues EXACTLY the shipped wrapper's op call (the same arguments in the same order); the GPU
gate (benchmarks/Efficiency/r2_kernels_gate.py G1) checks torch.equal against the shipped wrapper, and the CPU test
checks the argument plumbing against the shipped source by AST.

THE USE SITE (plan; nosa_llama.py:592-601, LlamaLayer.decode_forward; the warm-up twin :424): today
    score = infllmv2_attn_stage1_fast(query_states, compressed_k, compressed_k, cu_seqlens_q=cu_seqlens,
                                      cu_seqlens_k=cu_seqlens_comp, max_seqlen_q=max_seqlen,
                                      max_seqlen_k=max_seqlen_comp, causal=False)
where compressed_k is the engine's batch-wide, cat-grown table (all requests one length; cache_engine
update_compress_k_decode) and cu_seqlens_comp its cumulative lengths. Per request:
  - the compressed keys live in a PADDED table (B, Mcap, H, D) (flattened to (B * Mcap, H, D)), Mcap fixed at warm-up;
  - cu_seqlens_k = arange(B + 1) * Mcap (int32, fixed: the padded starts), seqused_k = n_comp (B,) int32 on the device
    (request b's own compressed length), max_seqlen_k = Mcap (fixed);
  - score = stage1_seqused(query_states, ck_padded, ck_padded, cu_seqlens, cu_k_padded, max_seqlen, Mcap,
                           seqused_k=n_comp)  -> (HKV, B, round128(Mcap)) bf16; row b is valid for columns < n_comp[b],
    the columns >= n_comp[b] stay at the op's zero fill (flash_api.cpp:936 torch::full(.., 0), :945 zero_) and are
    masked by the per-request pooling (max_pooling_per_request.py: it reads n_comp[b] per row).
  The scalars passed to the op (max_seqlen_q, Mcap) are fixed for a run, so the call can be captured; only seqused_k
  changes, through device memory.
GRAPH CAPTURE. The op ALLOCATES its output inside the extension (flash_api.cpp:936 p = torch::full(...); also out =
torch::empty({0}) :912 and rng_state :997). Capturing the call needs either
  (a) a p_out argument in the C++ op (the commented-out out_ path at flash_api.cpp:913-924 is the pattern; an extension
      rebuild), or
  (b) capture into the graph's PRIVATE memory pool (torch.cuda.graph's default): the allocations become fixed
      addresses owned by the graph, the returned tensor must be kept as the graph's output and read after each replay
      (or copied, as decode_forward copies score into score_buf at nosa_llama.py:607 today), and nothing else may
      free it.
  (b) needs no extension change; it fixes Mcap (the output width) at capture. Either way, the shape of every argument
  is frozen at capture, which is why the per-request lengths go through seqused_k and not through cu_seqlens_k.
"""
from __future__ import annotations

import importlib
from typing import Any, Optional

import torch


def _ext(ext: Any = None) -> Any:
    """The C extension: the given object (a CPU test's stub), else the installed infllm_v2.C (lazy import)."""
    return ext if ext is not None else importlib.import_module("infllm_v2.C")


def check_seqused(seqused_k: torch.Tensor, batch_size: int, device: Optional[torch.device] = None) -> None:
    """The op's own checks of seqused_k (flash_api.cpp:901-906), raised in Python BEFORE the call: int32, contiguous,
    shape (batch,), and on q's device (the op requires CUDA; a CPU test passes CPU tensors and a stub op)."""
    if not torch.is_tensor(seqused_k) or seqused_k.dtype != torch.int32:
        raise TypeError("stage1_seqused: seqused_k must be an int32 tensor (got %r)" % (getattr(seqused_k, "dtype", type(seqused_k)),))
    if not seqused_k.is_contiguous() or tuple(seqused_k.shape) != (batch_size,):
        raise ValueError("stage1_seqused: seqused_k must be contiguous with shape (%d,) (got %s)" % (batch_size, tuple(seqused_k.shape)))
    if device is not None and seqused_k.device != device:
        raise ValueError("stage1_seqused: seqused_k is on %s, q on %s" % (seqused_k.device, device))


def stage1_seqused(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k=None, dropout_p=0.0,
                   softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                   block_table=None, ext: Any = None):
    """infllmv2_attn_stage1_fast (infllmv2_sparse_attention.py:625-729) with seqused_k passed to the op. q (total_q,
    nheads, head_dim), k / v (total_k, nheads_k, head_dim); returns the op's first result (nheads_k, total_q, round128(
    max_seqlen_k)) exactly as the shipped wrapper (no slicing, no NaN replacement)."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    total_q, nheads, head_dim = q.shape
    batch_size = cu_seqlens_q.numel() - 1
    nheads_k = k.shape[1]
    nheads_per_group = nheads // nheads_k
    if seqused_k is not None:
        check_seqused(seqused_k, batch_size, q.device)
    q = q.view(total_q, nheads_k, nheads_per_group, head_dim).permute(0, 2, 1, 3).reshape(
        total_q * nheads_per_group, nheads_k, head_dim)
    cu_seqlens_q_adjusted = cu_seqlens_q * nheads_per_group
    max_seqlen_q_adjusted = max_seqlen_q * nheads_per_group
    result = _ext(ext).varlen_fwd_stage1(
        q,
        k,
        v,
        None,
        cu_seqlens_q_adjusted,
        cu_seqlens_k,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q_adjusted,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        True,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        True,
        None,
    )
    return result[0]
