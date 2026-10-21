# R2: the fused twin step and the lean body write path

Branch `feature/nosi-r2-fused`, worktree `/mnt/beegfs/ojerbi/nosi-r2-fused`.

**Base.** The base is the merge e8097b4 of two lines:
- the tick-capture line 90ecdcf (`step_unchecked`, `mean_fn`, the BI mode, the tick);
- the r2base P3 re-analysis line 622088f (`r2_capture`, `decode_fixed`, `step_fixed`, M-P3).

The lines share one file, `r2_positional_twin.py`. Its two conflicts kept both sides.

## 1. What the M-P3 data say (run 2190876, re-analysis job 2190953)

The data are per step at B 64, from the captured step (R2G). The ranges in the graph come from
the name alignment, 99.96 % matched.

| part                             | R2 graph ms | shipped ms | difference |
|----------------------------------|-------------|------------|------------|
| twin (`step_fixed`, 72 ops/layer) | 5.359      | 0.510 (`sh.k` + `sh.cis`) | +4.85 |
| body, the two host gathers       | 19.276      | 19.364     | -0.09      |
| body, the other ops              | 3.960       | 1.123      | +2.84      |

**The twin.** Each layer runs 72 small kernels. There are 2304 per step, each 1.5-10 us. The
work is a few KB per request. The time is the launch count, not the data.

**The body: the gathers are not slower in the graph.** Per step,
`flash_h2d_from_mask` takes 9.634 ms (shipped 9.677) and `_bias` takes 9.642 ms (shipped
9.687). The kernels, the grid and the per-block addressing are the same. Only the host buffer's
row count (the batch stride) is different.

The +0.78 ms per gather that the eager R2E window shows (10.467 ms) is not a shape or a stride
effect. The same kernels on the same state take 9.634 ms in the graph window. R2E is the later
window (steps 51-66, against 19-34 for R2G and 18-33 for SH). The number of blocks a step loads
depends on the selection churn at that position. Eager mode and the window cannot be separated
from this run. A check: count the loaded blocks (`_load_mask >= 0`) per step in each window. The
fusion does not need it.

**The body: the other ops.** `decode_fixed` runs 48 ops per layer. 44 of them are small torch
ops. The table gives their cost in the graph, per step, against `sh.kv`:

| ops (positions in the per-layer sequence, `r2_p3_reclass` SEQ) | per layer | ms per step |
|---|---|---|
| index gathers (`_k_gpu[bi, row]`, `_v_gpu[bi, row]`, `kv_bias[bi, prow]`, `_kv_bias_gpu[bi, row]`) | 4 | 0.650 |
| `index_put` (the three tail rows) | 3 | 0.410 |
| `where` (three tail rows, the map, the tail id, the lengths) | 6 | 0.515 |
| the refusal check (positions 0-10: compare, and, any, cast, mul, or, any, not, and) | 11 | |
| index arithmetic (the tail row, the bias row clamp, full, the tail id, the lengths) | 12 | 1.69 together |
| copies, casts, not, masked_fill, the two map memcpy | 8 | |
| `diff_kernel` (kept) | 1 | 0.604 |
| `_wb_kernel` (kept) | 1 | 0.088 |

## 2. The fused twin step (`nosi/nosi/r2_twin_fused.py`)

**The kernel.** It is ONE Triton kernel, `_twin_kernel`, with one program per bank. It runs T
steps in program order. `step_fused` is T = 1, and `advance_fused` is T steps in one launch.
Each step does this:
1. It reads the counters, the key-summary window (ring rows `(16 n_comp + w) % 64`) and the
   cis-summary window (total_cis rows `min(16 n_ccis + w, N - 1)`). In the cis window, the row
   that step 3 writes takes the new value in registers.
2. It forms both summaries as float32 chains, in window order, then multiplies by 2^-5. This is
   the same result as / 32, because both are the correctly rounded value.
3. A barrier, then the stores: comp_k, the ring row, the total_cis row, ccis, the overflow flag
   and the counters.
4. A second barrier ends the step.

The window loads are masked by "due and written". This is exact: the per-element order does not
depend on B.

**Rounding to bfloat16.** It uses integer ops: round to nearest even, and NaN becomes 0x7FC0. So
the Triton interpreter, whose cast truncates, and the GPU give the same bits.

**The result.** It is BITWISE `step_unchecked` / `advance_unchecked` with `mean_fn =
fixed_order_mean`: every table, counter, the sticky overflow flag and `ctr`.

**attach(twin, methods).** It binds the fused functions on the instance. It refuses a twin whose
`mean_fn` is not `fixed_order_mean`, unless the change is declared (`require_fixed_order=False`).

**`attach(twin, ("step_fixed",), require_fixed_order=False)`** is the declared change for the
capture step. The refusal is per bank, through the overflow flag, not the whole batch through
`err`. The mean is the fixed-order mean.

**Planted faults (negative controls).** PLANT 1: the key sum in reverse order. PLANT 2: the cis
window read before the write. PLANT 3: the ring row written for an inactive bank.

## 3. The lean body (`nosi/nosi/r2_body_fused.py`)

**The path.** `FusedBody(DeviceWritebackBody).decode_fixed` runs these ops per layer:
1. `diff_offload`;
2. `_body_kernel` (one program per request);
3. the two host gathers;
4. the Triton write-back.

That is 5 GPU ops per layer, where `decode_fixed` runs 48.

**What `_body_kernel` does:**
- the whole-batch position refusal (every program reads all B positions; program 0 ORs ERR_POS
  into err);
- the three tail rows;
- the selection's where / masked_fill / copy;
- slot 63's block id;
- the attention length;
- the write-back mask, into a (B,) buffer made once.

**The result.** It is BITWISE `decode_fixed`: `_k_gpu`, `_v_gpu`, `_kv_bias_gpu`, the three
maps, `_cache_lens`, err and the host rows.

**The two order moves are exact:**
- The diff runs before the tail rows. The diff reads neither.
- The tail id and the lengths are written before the gathers. The gathers read neither.

**Refusal of overrides.** `FusedBody` refuses a subclass that overrides a piece of `DecodeBody`
(`positions`, `full_mask`, `cache_lens_value`, `write_tail`, `select`, `writeback`).

**Planted faults (negative controls).** PLANT 1: the tail rows written for an inactive request.
PLANT 2: no tail id. PLANT 3: the load mask of an inactive request kept.

## 4. Tests and the GPU cell

**CPU** (`benchmarks/Efficiency/tests_cpu/test_r2_fused.py`; the interpreter runs in a fresh
process):
- F1: the constants, `check_twin`, `attach`, the override refusal, and no host sync in the
  launch paths.
- F2: the twin kernel against `step_unchecked` / `advance_unchecked`, float32 and bfloat16.
  - It runs 30 ragged steps and 12 advance ticks with T 3, then an overflow run (an unprefilled
    bank, N_max and Lk_max reached).
  - It counts the key / cis events and the block crossings.
  - The three planted faults must be caught.
- F3: `FusedBody` against `decode_fixed`, f4_fix False and True.
  - It runs 150 ragged steps with write-backs, then a refused step.
  - It uses the torch stand-ins of `tests/test_r2_decode_body.py`.
  - The three planted faults must be caught.

**GPU** (`benchmarks/Efficiency/r2_fused_cell.py`, `r2_fused_step.sh`):
- GT: the fused twin is bitwise equal on the GPU, bfloat16, at the R2 sizes. MUST GT-EQ,
  GT-COVER, GT-NEG.
- GB: the lean body is bitwise equal with the real kernels at B 64. MUST GB-EQ, GB-COVER, GB-NEG.
- MF: the measurement of the prediction.
  - It runs SH, and two graphs on one R2 state: G as built, and F with the fused twin on
    `step_fixed` and `FusedBody`.
  - The order is blocks G F G F of 16 timed steps, then 16 profiled steps of each.
  - MUST: SH-RUN, SH-STEADY, R2-RUN, PROF.

## 5. Registered prediction (MF, B 64, device ms per step from the profiled steps)

**The basis.** The basis is M-P3: G 52.97 ms, SH 45.46 ms.
- The twin goes from 5.36 ms to 0.15-0.30 ms (32 launches of a kernel that loads about 2 KB per
  request, plus 64 KB for the 1-in-16 requests whose summary is due).
- The body's other ops go from 3.96 ms to 0.79-0.94 ms (the diff 0.60, the write-back 0.09, the
  fused kernel 0.10-0.25).
- The gathers, GEMM, attention, stage 1 and the pooling do not change.

| prediction | value |
|---|---|
| F device ms | 44.0 - 46.0 |
| F - SH device ms | -1.5 to +0.5 (the R2 step is about as fast as the shipped one, or faster) |
| G - F device ms | 7.0 - 9.5 |
| F GPU ops | 1150 - 1260 per step (G 4841, SH 1269) |
| F median period - SH median period | <= +0.5 ms |

## 6. Not done here

The tick (the tick-capture worktree is pinned) can take `attach(twin)`:
- It would make `advance_unchecked` one launch per layer, instead of T_cap x 72 ops.
- `cb.cnt` and the strided key / cis cells are accepted as they are.
