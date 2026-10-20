"""R2-M2 (step 8 prerequisite): the DEVICE-DRIVEN, CUDA-graph-capturable MASKED WRITE-BACK of the tail slot.

WHAT IT DOES. When request b's decode step writes the last row of its tail block (position p_b with
(p_b + 1) % 64 == 0), the 64 rows of the tail slot (GPU rows tail0 .. tail0 + 63 of _k_gpu / _v_gpu) are copied to
request b's host rows p_b - 63 .. p_b of _k_cpu / _v_cpu. The shipped engine does this for the whole batch at once,
from the host, outside any graph (cache_engine.py:401-405; prior-art table row 13: "write-back not in graph", a
POSSIBLE delta). The R2 body's first backend (r2_decode_body.DecodeBody.writeback) loops on the HOST over a host copy
of the mask: one copy_(non_blocking=True) per full request and tensor; it cannot be captured.

THIS KERNEL reads the mask full (B,) bool and the positions pos (B,) int64 from DEVICE memory and stores through the
pinned host buffer's pointer: the reverse direction of the shipped gather, which loads through that pointer
(flash_cache_engine/flash_h2d_mask.py:11-62, `cpu_ptr` + tl.load; prior-art table row 11). Grid (B, H): one program
per (request, KV head) moves 64 x D elements of K and of V. A request whose mask bit is 0 loads and stores nothing
(the mask is part of every load and store mask; there is no host branch and no data-dependent shape), so a captured
launch serves any mask and any positions written into the same device tensors before a replay. Rows outside
[0, N_host) are never stored (a guard; the body refuses such positions on the host before the launch).

masked_writeback_torch is the same contract in plain torch with a host loop: the reference of the CPU tests and the
GPU gate. It equals r2_decode_body.DecodeBody.writeback (the torch backend) on every in-range case.

Constraints: block_size and head_dim are powers of two (tl.arange); K and V share one stride set on each side; the
host tensors must be pinned when the GPU tensors are on CUDA (checked, except while a graph is being captured, where
the check would be a CUDA API call inside the capture). Gate and commands: docs/R2_WRITEBACK.md.
"""
import torch            # no `from __future__ import annotations`: the kernel's constexpr annotations must be objects

try:                                                   # triton is needed only to launch the kernel; the reference and
    import triton                                      # the checks work without it (CPU tests without triton)
    import triton.language as tl
except Exception:                                      # pragma: no cover - not installed, or a broken install
    triton = tl = None


if triton is not None:
    @triton.jit
    def _wb_kernel(gk_ptr, gv_ptr, ck_ptr, cv_ptr, full_ptr, pos_ptr,
                   s_gb, s_gs, s_gh, s_gd, s_cb, s_cs, s_ch, s_cd,
                   tail0, n_host,
                   BLOCK: tl.constexpr, D: tl.constexpr, ROW_SHIFT: tl.constexpr, MASKED: tl.constexpr):
        pid_b = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        p = tl.load(pos_ptr + pid_b).to(tl.int64)
        rows = tl.arange(0, BLOCK).to(tl.int64)[:, None]                    # (BLOCK, 1)
        cols = tl.arange(0, D).to(tl.int64)[None, :]                        # (1, D)
        dst_row = p - (BLOCK - 1) + ROW_SHIFT + rows                        # host rows p - 63 .. p
        ok = (dst_row >= 0) & (dst_row < n_host) & (cols < D)               # (BLOCK, D)
        if MASKED:
            ok = ok & (tl.load(full_ptr + pid_b) != 0)
        src = pid_b * s_gb + (tail0 + rows) * s_gs + pid_h * s_gh + cols * s_gd
        dst = pid_b * s_cb + dst_row * s_cs + pid_h * s_ch + cols * s_cd
        tl.store(ck_ptr + dst, tl.load(gk_ptr + src, mask=ok, other=0.0), mask=ok)
        tl.store(cv_ptr + dst, tl.load(gv_ptr + src, mask=ok, other=0.0), mask=ok)


def check_args(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0: int, block_size: int) -> None:
    """Metadata checks only (no device read, no sync). Raises ValueError."""
    if k_gpu.dim() != 4 or tuple(v_gpu.shape) != tuple(k_gpu.shape):
        raise ValueError("k_gpu / v_gpu must be the same (B, S_GPU, H, D), got %s / %s"
                         % (tuple(k_gpu.shape), tuple(v_gpu.shape)))
    B, S_GPU, H, D = k_gpu.shape
    if k_cpu.dim() != 4 or tuple(v_cpu.shape) != tuple(k_cpu.shape) or k_cpu.shape[0] != B \
            or tuple(k_cpu.shape[2:]) != (H, D):
        raise ValueError("k_cpu / v_cpu must be the same (%d, N_host, %d, %d), got %s / %s"
                         % (B, H, D, tuple(k_cpu.shape), tuple(v_cpu.shape)))
    if len({k_gpu.dtype, v_gpu.dtype, k_cpu.dtype, v_cpu.dtype}) != 1:
        raise ValueError("all four tables must have one dtype")
    if tuple(v_gpu.stride()) != tuple(k_gpu.stride()) or tuple(v_cpu.stride()) != tuple(k_cpu.stride()):
        raise ValueError("K and V must share one stride set on each side")
    if v_gpu.device != k_gpu.device or v_cpu.device != k_cpu.device:
        raise ValueError("K and V must be on the same device on each side")
    if tuple(full.shape) != (B,) or full.dtype != torch.bool or full.device != k_gpu.device or full.stride(0) != 1:
        raise ValueError("full must be a contiguous (%d,) bool on %s" % (B, k_gpu.device))
    if tuple(pos.shape) != (B,) or pos.dtype != torch.int64 or pos.device != k_gpu.device or pos.stride(0) != 1:
        raise ValueError("pos must be a contiguous (%d,) int64 on %s" % (B, k_gpu.device))
    if not isinstance(tail0, int) or not isinstance(block_size, int) or tail0 < 0 or tail0 + block_size > S_GPU:
        raise ValueError("tail rows [%r, %r + %r) outside the %d GPU rows" % (tail0, tail0, block_size, S_GPU))
    if block_size < 1 or block_size & (block_size - 1) or D & (D - 1):
        raise ValueError("block_size (%d) and head_dim (%d) must be powers of two" % (block_size, D))
    if k_gpu.is_cuda:
        if k_cpu.device.type != "cpu":
            raise ValueError("the host tables must be CPU tensors")
        if not torch.cuda.is_current_stream_capturing() and not (k_cpu.is_pinned() and v_cpu.is_pinned()):
            raise ValueError("the host tables must be pinned: the kernel stores through their device-visible pointer")
    elif k_cpu.device != k_gpu.device:
        raise ValueError("off CUDA (the Triton interpreter) all tables must be on one device")


def _launch(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0: int, block_size: int, row_shift: int, masked: bool) -> None:
    """The launch. row_shift != 0 or masked = False exist ONLY for the negative controls of the tests and the gate."""
    if triton is None:
        raise RuntimeError("r2_writeback: triton is not installed")
    B, _, H, D = k_gpu.shape
    _wb_kernel[(B, H)](k_gpu, v_gpu, k_cpu, v_cpu, full.view(torch.uint8), pos,
                       *k_gpu.stride(), *k_cpu.stride(), tail0, k_cpu.shape[1],
                       BLOCK=block_size, D=D, ROW_SHIFT=row_shift, MASKED=masked)


def masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0: int, block_size: int) -> None:
    """For every b with full[b]: host rows [pos[b] - block_size + 1, pos[b]] of k_cpu / v_cpu <- GPU rows
    [tail0, tail0 + block_size) of k_gpu / v_gpu. Device-driven: full and pos are read on the device; no host sync;
    capturable (the four tables and full / pos keep their addresses between replays)."""
    check_args(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size)
    _launch(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size, 0, True)


def masked_writeback_torch(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0: int, block_size: int) -> None:
    """The same contract in plain torch with a HOST loop (it reads full and pos on the host): the reference. Rows
    outside [0, N_host) are dropped, as the kernel drops them."""
    check_args(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size)
    n_host = k_cpu.shape[1]
    for b, (f, p) in enumerate(zip(full.tolist(), pos.tolist())):
        if not f:
            continue
        r0 = p - (block_size - 1)
        lo, hi = max(r0, 0), min(r0 + block_size, n_host)
        if lo >= hi:
            continue
        k_cpu[b, lo:hi].copy_(k_gpu[b, tail0 + lo - r0:tail0 + hi - r0].to(k_cpu.device))
        v_cpu[b, lo:hi].copy_(v_gpu[b, tail0 + lo - r0:tail0 + hi - r0].to(v_cpu.device))
