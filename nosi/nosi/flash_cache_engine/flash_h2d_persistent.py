"""Throttled ('persistent') host->device block gather: the same copy as
flash_h2d_from_mask_kernel, but a FIXED number of CTAs loop over the (h, b, m)
items instead of one CTA per item. Purpose (E3d, ledger 2026-09-20): a gather
that occupies a few SMs while keeping enough loads in flight to saturate PCIe,
so that a decode step can run beside it. n_ctas is the only knob; the item
order and the copied bytes are identical to the shipped kernel's.

Same layout contract as flash_h2d_mask.py: gpu (B, S_GPU, H, D), cpu (B, S_CPU, H, D)
pinned, load_ids (H, B, M) int32 with -1 = nothing to load.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def flash_h2d_persistent_kernel(
    gpu_ptr, cpu_ptr, load_ids_ptr,
    stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu,
    stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu,
    S_GPU, S_CPU, B, H, M, n_items, n_ctas,
    block_size: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset_s = tl.arange(0, block_size)[:, None]
    offset_d = tl.arange(0, D)[None, :]
    n_iter = (n_items + n_ctas - 1) // n_ctas
    for k in range(0, n_iter):
        idx = pid + k * n_ctas                       # item order = load_ids memory order: h-major, then b, then m
        if idx < n_items:
            h = idx // (B * M)
            rem = idx - h * (B * M)
            b = rem // M
            m = rem - b * M
            cpu_block_id = tl.load(load_ids_ptr + idx)
            if cpu_block_id >= 0:
                src_ptrs = cpu_ptr + b * stride_b_cpu + (cpu_block_id * block_size + offset_s) * stride_m_cpu + h * stride_h_cpu + offset_d * stride_d_cpu
                dst_ptrs = gpu_ptr + b * stride_b_gpu + (m * block_size + offset_s) * stride_m_gpu + h * stride_h_gpu + offset_d * stride_d_gpu
                mask_src = (cpu_block_id * block_size + offset_s) < S_CPU
                mask_dst = (m * block_size + offset_s) < S_GPU
                vals = tl.load(src_ptrs, mask=mask_src)
                tl.store(dst_ptrs, vals, mask=mask_dst)


@triton.jit
def flash_h2d_persistent_cv_kernel(
    gpu_ptr, cpu_ptr, load_ids_ptr,
    stride_b_cpu, stride_m_cpu, stride_h_cpu, stride_d_cpu,
    stride_b_gpu, stride_m_gpu, stride_h_gpu, stride_d_gpu,
    S_GPU, S_CPU, B, H, M, n_items, n_ctas,
    block_size: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset_s = tl.arange(0, block_size)[:, None]
    offset_d = tl.arange(0, D)[None, :]
    n_iter = (n_items + n_ctas - 1) // n_ctas
    for k in range(0, n_iter):
        idx = pid + k * n_ctas                       # item order = load_ids memory order: h-major, then b, then m
        if idx < n_items:
            h = idx // (B * M)
            rem = idx - h * (B * M)
            b = rem // M
            m = rem - b * M
            cpu_block_id = tl.load(load_ids_ptr + idx)
            if cpu_block_id >= 0:
                src_ptrs = cpu_ptr + b * stride_b_cpu + (cpu_block_id * block_size + offset_s) * stride_m_cpu + h * stride_h_cpu + offset_d * stride_d_cpu
                dst_ptrs = gpu_ptr + b * stride_b_gpu + (m * block_size + offset_s) * stride_m_gpu + h * stride_h_gpu + offset_d * stride_d_gpu
                mask_src = (cpu_block_id * block_size + offset_s) < S_CPU
                mask_dst = (m * block_size + offset_s) < S_GPU
                vals = tl.load(src_ptrs, mask=mask_src, cache_modifier=".cv")   # E5: bypass L1 / L2 for the host (UVA) loads
                tl.store(dst_ptrs, vals, mask=mask_dst)


def flash_h2d_persistent(gpu_data, cpu_data, load_ids, block_size: int = 64, n_ctas: int = 32, num_warps: int = 4, bypass_cache: bool = False):
    B, S_GPU, H, D = gpu_data.shape
    _, S_CPU, _, _ = cpu_data.shape
    H2, B2, M = load_ids.shape
    assert H == H2 and B == B2 and load_ids.dtype == torch.int32 and load_ids.is_contiguous()
    sb_c, sm_c, sh_c, sd_c = cpu_data.stride()
    sb_g, sm_g, sh_g, sd_g = gpu_data.stride()
    n_items = H * B * M
    kernel = flash_h2d_persistent_cv_kernel if bypass_cache else flash_h2d_persistent_kernel
    kernel[(int(n_ctas),)](
        gpu_data, cpu_data, load_ids,
        sb_c, sm_c, sh_c, sd_c, sb_g, sm_g, sh_g, sd_g,
        S_GPU, S_CPU, B, H, M, n_items, int(n_ctas),
        block_size, D, num_warps=num_warps,
    )
