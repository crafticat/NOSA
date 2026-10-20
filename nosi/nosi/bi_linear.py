# Adapted from SGLang python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py (matmul_kernel_persistent and
# _matmul_persistent_triton; SGLang commit 87db743, read in the local checkout
# /mnt/beegfs/ojerbi/scratch_research/hisparse_sglang_kyFDRo/sglang), which is itself adapted from Thinking Machines
# Lab's batch_invariant_ops (https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/
# batch_invariant_ops/batch_invariant_ops.py). Both are licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0); this file is distributed under the same license.
# Modifications (R2-M2 step 7, 2026-10-20): a non-persistent grid (one program per output tile and K-split); the
# weight read in place as F.linear's (N, K); tile sizes chosen per WEIGHT SHAPE (never per M) by a fixed table; an
# optional FIXED split-K per weight shape with a fixed-order fp32 reduction kernel; the row count M is not
# specialized; a torch twin with the same tiled reduction order; no DeepGEMM path, no bias, no autotune.
"""R2-M2 step 7: the M-INVARIANT linear (design note docs/R2_STEP7_SPEC_ROUND.md section 3.3, option G-a) for
NOSA-8B's decode linears: y = x @ W.T, exactly F.linear(x, W) in meaning, with a row's output bits a function of that
row, W and the weight shape only, NEVER of the number of rows M (AGENTS.md law 1).

WHY cuBLAS IS NOT ENOUGH. cuBLAS picks its kernel (tile shape, split-K) from (M, N, K): an M = 2B call is not
row-equal to an M = B call (paired run 2175550: 0 of 11 rows equal at M = 2B). The flattened tick of step 7 runs
B x (1 + U) rows, so its GEMM rows must be bitwise the serial decode's at M = B.

WHY THIS IS M-INVARIANT (the claim, by construction):
  * a program computes one (BLOCK_M x BLOCK_N) output tile for one K-split; its K loop walks the split's
    BLOCK_K chunks in ascending order and accumulates with tl.dot (bf16 operands, fp32 accumulator). A tensor-core
    dot computes C[i, j] from row i of A and column j of B only, so the rows sharing a tile never mix, and rows of A
    beyond M are loaded as zero and never stored (a GPU tensor-core property; the CPU interpreter and the twin
    show the tiling and the invariance logic, the GPU gate shows the hardware);
  * every constexpr (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) and the launch constants (num_warps, num_stages) come from
    config_for(N, K): a fixed table keyed on the WEIGHT SHAPE. Nothing reads M except the grid size and the row mask;
    M is in do_not_specialize, so M = 1 or M % 16 == 0 does not compile another variant. No autotune;
  * the K-split is FIXED per weight shape (never per M). Each split writes an fp32 partial; _splitk_sum adds them in
    the fixed order 0, 1, ..., S-1 per element and rounds once to the output dtype. With SPLIT_K = 1 the epilogue is
    the single rounding of the fp32 accumulator. The same epilogue for every row.
  The result is NOT bitwise cuBLAS's: the serial reference (the BI serial decode) must use this op too.

INTERFACE (fixed; C's microgate B3 imports it):
    config_for(N, K) -> GemmConfig(block_m, block_n, block_k, split_k, num_warps, num_stages)
    bi_linear(x, weight, out=None, config=None, workspace=None) -> (..., N)
        x        (..., K)  bf16 (fp32 / fp16 accepted; fp32 for the CPU interpreter only: on the GPU an fp32 tl.dot
                 runs as TF32), last stride 1 after the reshape to (M, K)
        weight   (N, K)    x's dtype, contiguous (F.linear's layout: no transpose is made)
        out      optional (M, N) / (..., N) in x's dtype
        config   None = config_for(N, K). Tests and the gate's NEGATIVE control may pass another config; a production
                 caller never does (a config chosen by M breaks the law).
        workspace optional fp32 (split_k, M_cap >= M, N) for split_k > 1 (a fixed buffer, CUDA-graph friendly)
    bi_linear_reference(x, weight, config=None) -> the CPU TWIN: the same tiled order per row (per K-split, BLOCK_K
        chunks in order, fp32 partials summed in split order, one rounding).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

try:                                                   # triton is needed only to launch; the twin, the config table
    import triton                                      # and the tests import this module without it
    import triton.language as tl
except ImportError:                                    # pragma: no cover - environment dependent
    triton = tl = None


class GemmConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int


# NOSA-8B's decode linears (nosa_llama.py on the integration line: F.linear on the fused weights; no bias)
NOSA8B_SHAPES = {
    "wqkv": (4608, 4096),       # q 32 x 128 + k 2 x 128 + v 2 x 128, hidden 4096
    "wo": (4096, 4096),
    "gate_up": (32768, 4096),   # 2 x intermediate 16384
    "down": (4096, 16384),
    "lm_head": (73448, 4096),   # vocab 73448
}
BLOCK_M = 64                    # one row tile size for every shape and every M
BLOCK_K = 64


def config_for(N: int, K: int) -> GemmConfig:
    """The FIXED tile table: a function of the weight shape (N, K) only. Wide weights (N >= 16384: gate_up, lm_head)
    already give >= 256 tiles of 128 columns: no split. Narrow ones (N <= 8192: wqkv, wo, down) use 64 columns and a
    fixed K-split (2, or 4 when K > 8192) so a decode-sized M still launches >= 128 programs on 108 SMs."""
    N, K = int(N), int(K)
    if N <= 0 or K <= 0:
        raise ValueError("config_for(N=%d, K=%d)" % (N, K))
    if N >= 16384:
        return GemmConfig(BLOCK_M, 128, BLOCK_K, 1, 4, 4)
    split = 2 if K <= 8192 else 4
    return GemmConfig(BLOCK_M, 64, BLOCK_K, split, 4, 4)


def k_per_split(K: int, cfg: GemmConfig) -> int:
    """K columns per split, rounded up to whole BLOCK_K chunks (the last split may run past K: masked)."""
    per = -(-K // cfg.split_k)
    return -(-per // cfg.block_k) * cfg.block_k


if triton is not None:
    @triton.jit(do_not_specialize=["M", "s_ps"])
    def _bi_matmul_kernel(X, W, P, M, N, K, s_xm, s_xk, s_wn, s_wk, s_ps, s_pm, s_pn, K_SPLIT,
                          BLOCK_M_C: tl.constexpr, BLOCK_N_C: tl.constexpr, BLOCK_K_C: tl.constexpr,
                          FP32_OUT: tl.constexpr):
        pid = tl.program_id(0)
        pid_s = tl.program_id(1)
        num_pid_m = tl.cdiv(M, BLOCK_M_C)
        pid_m = pid % num_pid_m                       # the M tiles of one column tile are adjacent (weight reuse)
        pid_n = pid // num_pid_m
        offs_m = pid_m * BLOCK_M_C + tl.arange(0, BLOCK_M_C)
        offs_n = pid_n * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
        m_ok = offs_m < M
        n_ok = offs_n < N
        rows = tl.where(m_ok, offs_m, 0).to(tl.int64)
        cols = tl.where(n_ok, offs_n, 0).to(tl.int64)
        k0 = pid_s * K_SPLIT
        acc = tl.zeros((BLOCK_M_C, BLOCK_N_C), dtype=tl.float32)
        for kk in range(0, K_SPLIT, BLOCK_K_C):
            offs_k = k0 + kk + tl.arange(0, BLOCK_K_C)
            k_ok = offs_k < K
            x = tl.load(X + rows[:, None] * s_xm + offs_k[None, :] * s_xk, mask=m_ok[:, None] & k_ok[None, :],
                        other=0.0)
            w = tl.load(W + cols[None, :] * s_wn + offs_k[:, None] * s_wk, mask=n_ok[None, :] & k_ok[:, None],
                        other=0.0)
            acc = tl.dot(x, w, acc)
        p_ptr = P + pid_s * s_ps + rows[:, None] * s_pm + cols[None, :] * s_pn
        if FP32_OUT:
            tl.store(p_ptr, acc, mask=m_ok[:, None] & n_ok[None, :])
        else:
            tl.store(p_ptr, acc.to(P.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])

    @triton.jit(do_not_specialize=["M", "s_ps"])
    def _splitk_sum_kernel(P, OUT, M, N, s_ps, s_pm, s_pn, s_om, s_on,
                           SPLIT_K: tl.constexpr, BLOCK_M_C: tl.constexpr, BLOCK_N_C: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M_C + tl.arange(0, BLOCK_M_C)
        offs_n = pid_n * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
        ok = (offs_m < M)[:, None] & (offs_n < N)[None, :]
        rows = offs_m.to(tl.int64)
        cols = offs_n.to(tl.int64)
        base = P + rows[:, None] * s_pm + cols[None, :] * s_pn
        acc = tl.load(base, mask=ok, other=0.0)
        for s in tl.static_range(1, SPLIT_K):                # the fixed order 0, 1, ..., S-1
            acc = acc + tl.load(base + s * s_ps, mask=ok, other=0.0)
        tl.store(OUT + rows[:, None] * s_om + cols[None, :] * s_on, acc.to(OUT.dtype.element_ty), mask=ok)
else:                                                  # pragma: no cover
    _bi_matmul_kernel = _splitk_sum_kernel = None


def _check(x2, weight):
    if weight.dim() != 2 or x2.dim() != 2 or x2.shape[1] != weight.shape[1]:
        raise ValueError("x (..., K) and weight (N, K): got %s and %s" % (tuple(x2.shape), tuple(weight.shape)))
    if x2.dtype != weight.dtype:
        raise TypeError("x %s and weight %s dtypes differ" % (x2.dtype, weight.dtype))
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous (N, K): F.linear's layout")
    if x2.device != weight.device:
        raise ValueError("x and weight on different devices")


def bi_linear(x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None,
              config: Optional[GemmConfig] = None, workspace: Optional[torch.Tensor] = None) -> torch.Tensor:
    """y = x @ weight.T, M-invariant (module docstring). No host sync; shape checks only."""
    if _bi_matmul_kernel is None:
        raise RuntimeError("triton is not installed: use bi_linear_reference")
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    _check(x2, weight)
    M, N = x2.shape[0], weight.shape[0]
    cfg = config_for(N, K) if config is None else config
    if out is None:
        out = torch.empty((*lead, N), dtype=x.dtype, device=x.device)
    elif out.shape != (*lead, N) or out.dtype != x.dtype:
        raise ValueError("out must be %s in %s" % (((*lead, N),), x.dtype))
    o2 = out.view(-1, N)
    if M == 0:
        return out
    ks = k_per_split(K, cfg)
    grid = (triton.cdiv(M, cfg.block_m) * triton.cdiv(N, cfg.block_n), cfg.split_k)
    if cfg.split_k == 1:
        _bi_matmul_kernel[grid](x2, weight, o2, M, N, K, x2.stride(0), x2.stride(1), weight.stride(0),
                                weight.stride(1), 0, o2.stride(0), o2.stride(1), ks,
                                BLOCK_M_C=cfg.block_m, BLOCK_N_C=cfg.block_n, BLOCK_K_C=cfg.block_k,
                                FP32_OUT=(x.dtype == torch.float32), num_warps=cfg.num_warps,
                                num_stages=cfg.num_stages)
        return out
    if workspace is None:
        ws = torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=x.device)
    else:
        if workspace.dtype != torch.float32 or workspace.dim() != 3 or workspace.shape[0] != cfg.split_k \
                or workspace.shape[1] < M or workspace.shape[2] != N or workspace.stride(-1) != 1:
            raise ValueError("workspace must be fp32 (split_k=%d, >= M=%d, N=%d) with a unit last stride" % (
                cfg.split_k, M, N))
        ws = workspace
    _bi_matmul_kernel[grid](x2, weight, ws, M, N, K, x2.stride(0), x2.stride(1), weight.stride(0), weight.stride(1),
                            ws.stride(0), ws.stride(1), ws.stride(2), ks,
                            BLOCK_M_C=cfg.block_m, BLOCK_N_C=cfg.block_n, BLOCK_K_C=cfg.block_k, FP32_OUT=True,
                            num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    _splitk_sum_kernel[(triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n))](
        ws, o2, M, N, ws.stride(0), ws.stride(1), ws.stride(2), o2.stride(0), o2.stride(1),
        SPLIT_K=cfg.split_k, BLOCK_M_C=cfg.block_m, BLOCK_N_C=cfg.block_n, num_warps=4)
    return out


def bi_linear_reference(x: torch.Tensor, weight: torch.Tensor, config: Optional[GemmConfig] = None) -> torch.Tensor:
    """The CPU TWIN (tests only; host loops): per row, per K-split in split order, the split's BLOCK_K chunks in
    ascending order accumulated in fp32 (each chunk one (1 x BK) @ (BK x N) product: the same call shape for every
    row and every M, so the row's bits cannot depend on M), then the split partials added in split order and rounded
    once to x's dtype. config: as bi_linear (the negative control passes one chosen by M)."""
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    _check(x2, weight)
    M, N = x2.shape[0], weight.shape[0]
    cfg = config_for(N, K) if config is None else config
    ks = k_per_split(K, cfg)
    f32 = torch.float32
    wf = weight.to(f32)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    for i in range(M):
        xi = x2[i].to(f32)
        total = None
        for s in range(cfg.split_k):
            acc = torch.zeros(N, dtype=f32, device=x.device)
            for kk in range(0, ks, cfg.block_k):
                lo, hi = s * ks + kk, min(s * ks + kk + cfg.block_k, K)
                if lo >= K:
                    break
                acc = acc + xi[lo:hi].unsqueeze(0).matmul(wf[:, lo:hi].t()).squeeze(0)
            total = acc if total is None else total + acc
        out[i] = total.to(x.dtype)
    return out.view(*lead, N)
