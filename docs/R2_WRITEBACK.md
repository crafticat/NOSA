# R2-M2: the device-driven masked write-back (step 8 prerequisite)

Branch `feature/nosi-r2-m2-wb` (from the integration line `feature/nosi-r2-m2-int` 54ea1d6).
Status: SOURCE ONLY. Nothing here comes from a GPU run.

Code: `nosi/nosi/r2_writeback.py` (the kernel, its launcher, its torch reference),
`nosi/nosi/r2_decode_body.py` (`DeviceWritebackBody`).
CPU tests: `tests/test_r2_writeback.py`, `tests/test_r2_decode_body.py` (the
`test_device_writeback_body_*` nodes). GPU gate: `benchmarks/Efficiency/r2_wb_gate.py`,
step script `benchmarks/Efficiency/r2_wb_step.sh`.

## What and why

When request b's decode step writes the last row of its tail block
(`(p_b + 1) % 64 == 0`), the 64 tail-slot rows (GPU rows `4032 .. 4095` of
`_k_gpu`, `_v_gpu`) go to request b's host rows `[p_b - 63, p_b]` of `_k_cpu`,
`_v_cpu`.

* The shipped engine does this for the whole batch at once, from the host,
  outside any graph: `cache_engine.py:401-405` in this tree. Prior-art table
  (main repository, `docs/superpowers/specs/2026-09-02-prior-art-table.md`)
  row 13: "write-back not in graph", a POSSIBLE delta, unbuilt and unmeasured.
* The R2 body's first backend (`DecodeBody.writeback`) loops on the host over a
  host copy of the mask: one `copy_(non_blocking=True)` per full request and
  tensor. It cannot be captured.
* This kernel reads the mask `full` (B,) bool and the positions `pos` (B,)
  int64 in DEVICE memory and STORES through the pinned host buffer's pointer.
  That is the reverse direction of the shipped gather, which LOADS through
  that pointer (`flash_cache_engine/flash_h2d_mask.py:11-62`, `cpu_ptr` +
  `tl.load`; prior-art row 11, "nothing" new in the read direction). What is
  ours: the store direction, the device mask, and capture.

## The kernel contract (`r2_writeback.masked_writeback`)

`masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, tail0, block_size)`:

| argument | shape / dtype / device |
|---|---|
| `k_gpu`, `v_gpu` | (B, S_GPU, H, D), one dtype, CUDA, one stride set |
| `k_cpu`, `v_cpu` | (B, N_host, H, D), the same dtype, CPU, PINNED, one stride set |
| `full` | (B,) bool, on the GPU tensors' device |
| `pos` | (B,) int64, same device |
| `tail0`, `block_size` | Python ints; `tail0 + block_size <= S_GPU`; `block_size` and `D` powers of two |

* For every b with `full[b]`: host rows `[pos[b] - block_size + 1, pos[b]]`
  <- GPU rows `[tail0, tail0 + block_size)`, for K and V. Nothing else is
  written. Rows outside `[0, N_host)` are dropped (a guard; the body refuses
  such positions on the host before any launch).
* Grid `(B, H)`; each program moves `block_size x D` elements of K and of V.
  The mask bit is part of every load and store mask: no host branch, no
  data-dependent shape, no host sync (`test_W3` scans the launch path).
* Capturable: a captured launch serves any mask, positions and tail rows
  written INTO the same device tensors before a replay. The pinned check
  (`is_pinned`) is skipped while a capture is in progress (it is a CUDA API
  call); it runs on every eager call.
* `masked_writeback_torch` is the same contract with a host loop: the reference.
* `_launch(..., row_shift, masked)` exists only for the negative controls.

## In the body

`r2_decode_body.DeviceWritebackBody(f4_fix, writeback_kernel)` is
`DecodeBody` with `writeback()` replaced: it calls the injected kernel with
the device mask `full` and the device positions `p` that `decode()` already
computes. `decode()`'s signature does not change. `r2_decode_body.py` still
imports only torch.

To switch the R2 path (C's `R2Body` in `nosi/nosi/r2_decode_path.py`, not
edited here), after the GPU gate passes:

    from . import r2_writeback as WB
    self.body = RB.DeviceWritebackBody(bool(f4_fix), WB.masked_writeback)

## Gates

CPU (inside a CPU job):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2-wb python -m pytest -q tests/test_r2_writeback.py tests/test_r2_decode_body.py

* W1: the reference == the body's torch backend, bitwise over the whole
  NaN-poisoned host buffers, 20 random cases.
* W2: argument refusals. W3: no host sync in the launch path (AST).
* W5: the KERNEL under `TRITON_INTERPRET=1` in a fresh process (the pattern
  of `benchmarks/Efficiency/tests_cpu/test_r2_kernels.py` K5): == the
  reference and == `DecodeBody.writeback` on four cases (one is the row
  guard); the negative controls row shift -1, row shift +1 and unmasked are
  each caught (3 of 3). Skipped without triton.
* In `tests/test_r2_decode_body.py`: `DeviceWritebackBody` (with the torch
  reference as its kernel) passes G-EQ (against the shipped body), G-RAG and
  G-POISON; a counting kernel shows `decode()` hands it the device mask
  `active & (p + 1) % 64 == 0` and the positions, once per step.

GPU (`r2_wb_gate.py`; B 64, H 2, D 128, bf16, N_host 24384 pinned):

* W0: the body's torch backend on the GPU == the reference.
* W1: the kernel (eager) == the reference; n_full 0, 1, 7, 64 and a guard case.
* W2: one captured launch, 12 replays with new masks, positions and tail rows
  written into the captured tensors: each replay == an eager launch on a
  second pinned pair == the reference.
* W3: the kernel, then the shipped `flash_h2d_from_mask` reading the written
  block on the same stream: the read-back rows == the tail rows (eager, and
  both kernels captured in one graph and replayed).
* W4: row shift -1, row shift +1, unmasked: each caught by the W1 comparison.
* W5 REPORT: median microseconds of the kernel, the graph replay and the torch
  backend for n_full 1, 8, 64. No claim is made from it; the plan's open
  question (the kernel's PCIe store efficiency against the copy engine) is
  answered by these numbers, not predicted here.

Every comparison is bitwise over the whole host buffers (int16 views of
bf16), NaN-poisoned before every case.

Step (the lead runs it in the holder; DRYRUN on the login node first):

    DRYRUN=1 bash benchmarks/Efficiency/r2_wb_step.sh --nosi <sha> <holder job> <run id>
    bash benchmarks/Efficiency/r2_wb_step.sh --nosi <sha> <holder job> <run id>

The GPU uuid comes from `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/<holder job>.gpu_uuid`
(lead-owned; read before any claim; recorded in `command.txt`); the child
still checks it with `nvidia-smi` (exit 3). Claim directory
`/mnt/beegfs/ojerbi/r2-runs/wb/wb_active`; `STEP_MIN` 10.

## Not covered

The write-back inside a captured WHOLE decode step (the body still validates
on the host and uses `torch.where` temporaries); its effect on the decode's
step time (W5 times the kernel alone); writes from more than one stream;
memory ordering with a reader on ANOTHER stream (W3 covers the same stream,
which is how the engine runs).
