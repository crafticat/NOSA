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
- F4: the pure parts of the tick profile M-T (section 7), with a refused sequence as the
  negative control.

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

## 6. The tick takes the fused twin as it is

The tick-capture worktree is pinned. Its code is in this worktree through the merge.
- `attach(twin)` on the tick's twins makes `advance_unchecked` ONE launch per layer.
- `cb.cnt` and the strided key / cis cells of `_layer_fixed` are accepted as they are.
- The lean body is for `DecodeBody`. The tick has its own store and page path; the M-T profile
  shows its cost (`t.pages`, `-`).

## 7. M-T: the device profile of the captured tick (TC run tc1, holder 2190876)

**The TC timing.** At B 8 the captured tick takes 143.68 ms per tick, of which the host part is
7.68 ms. The tick is device-bound.

**Why the twin is large in the tick.** The tick's twins use `fixed_order_mean` (the BI mode).
Each summary is 65 small kernels: a cast of the first row, 31 casts and 31 adds, a divide and a
cast. A twin step is about 190 kernels, against 72 for `step_fixed` with torch's `.mean`. Per
layer, `advance_unchecked` runs T steps plus the counter stacks:
- G_v (T 3): about 578 kernels per layer, 18,500 per tick;
- G_d (T 1): about 194 kernels per layer, 6,200 per tick.

At B 8 (16 banks) each kernel is 1.2-2.4 us.

**The cell** (`benchmarks/Efficiency/r2_tick_profile.py`, `r2_tick_profile_step.sh`):
- TE: the reference and the plan.
- TF: eager, with ranges t.twin / t.stage1 / t.pool / t.select / t.pages / t.attn / t.gemm.
  '-' is the rest of the region (the store writes, the row cells, rmsnorm, nosa_linear, rope,
  silu, the residuals, the embedding, lm_head, the token log). 'out' is the work between regions
  (the upload, the decision read, the events).
- TG: captured; the ranges come from TF by tick and by name.
- TGF: captured, with the fused twin.
  - MUST TGF-EQ: it equals TE bitwise at every tick and at the end.
  - Its profile takes TG's labels with the twin removed.

All arms use the same window of 6 ticks, with at least 2 of each capacity. The output is ops and
device ms per tick, per capacity, per range, per class and per range x class, plus the period.

**Registered predictions (M-T, B 8, device per tick, the mean over the window's ticks of a
capacity):**

| prediction | G_v | G_d |
|---|---|---|
| TK2: TG t.twin ops | 16,000 - 21,000 | 5,500 - 7,200 |
| TK2: TG t.twin ms | 22 - 45 | 7 - 16 |
| TK3: TGF t.twin | 32 - 40 ops, <= 1.0 ms | 32 - 40 ops, <= 0.5 ms |
| TK4: TG - TGF device ms | 21 - 44.5 | 6.5 - 15.5 |
| TK1 (MUST TGF-EQ) | bitwise | bitwise |

## 8. M-F f1 (holder 2190876) and what follows

**The result.** All 5 registered predictions were CONFIRMED:
- F is 44.936 ms per step.
- F - SH is -0.532 ms device and -1.400 ms in the median period.
- G - F is 7.588 ms.
- F has 1193 GPU ops per step.

GT-EQ, GT-COVER, GB-EQ and GB-NEG passed. GT-NEG failed: PLANT 1 (the key sum in reverse order)
was not caught.

**Why PLANT 1 was blind (proof: test F5).**

The kernel's key summary works per bank and per (head, dim) column:
- It sums 32 bfloat16 ring rows, the positions 16 n_comp .. 16 n_comp + 31.
- Each row is decoded exactly to float32.
- One float32 chain adds them in window order (PLANT 1: in reverse order).
- It multiplies by 2^-5 and rounds the result to bfloat16.

A bfloat16 value has 8 significant bits. Let e_min and e_max be the smallest and the largest
exponent of the nonzero values in a column. If e_max - e_min <= 11, every partial sum is a
multiple of 2^(e_min - 7) and is below 2^(e_max + 6). That is at most 24 bits, so it is exact in
float32. The sum is then exact in any order, and the reversed chain gives the same bits. When the
spread is wider, a reordering changes only the last float32 bits. The bfloat16 rounding of the
mean removes that change, unless the sum is within about 2^-16 of a rounding boundary.

The GT gate's data was synthetic N(0, 1) keys in bfloat16, not model keys. On that data more than
95 % of the column sums are exact, and at most 1e-4 of the bfloat16 means change. So PLANT 1 did
not change the output: the control was blind, and the twin has no defect. The CPU test F2 caught
PLANT 1 because its plant runs use float32 tables (24-bit inputs, whose sums do round).

**The fix (GT, unchanged thresholds elsewhere).**
- Each plant runs on three copies fed the same inputs:
  - the torch ops (a);
  - the unplanted kernel (u);
  - the planted kernel (p).
- A plant is EFFECTIVE at the first step where p differs from u, and CAUGHT at the first step where
  p differs from a.
- A plant that is never effective is "no-op on this data". It is REPORTED, not counted as caught.
- GT-NEG (the run's data): plants 2 and 3 must be caught. No effective plant may be missed. u must
  equal a throughout. Plant 1 may be caught or no-op.
- GT-NEG-ORDER (new, MUST): the ORDER PROBE (`order_probe_keys` / `order_probe_ring`) puts every key
  summary on order-sensitive rows. Per 16 positions there are 14 rows of 2^-30, then 1, then -1.
  - In window order the sum is 0. Reversed, it is 14 x 2^-30.
  - Plant 1 must be effective AND caught, and u must equal a.

**The next job: r2base-fused** (`r2_base_cell.py --r2 fused`, `r2_base_fused_step.sh`).
- It is the r2base cell of cd77de2: B 64 / 128 / 219, S 16128, N 512, agreement against shipped.
- The R2 arm has the fused twin on `step_fixed` and `FusedBody`, attached before the warm-up.
- R2-ERR also requires every overflow flag clear.
- The cell `gt64` runs GT first (its gates are MUST gates of the cell), then B 64. So ONE job holds
  the GT f2 gate and the B 64 row. b128 and b219 are separate steps.

**Registered predictions (r2base-fused).** The basis is rb1 2190876 (R2 / SH steady 1.129 / 1.090
/ 1.068, effective 2.784 / 2.170 / 1.755) and M-F f1 (the fusion removes about 7.4 ms of period at
B 64, and F - SH = -1.40 ms).

| B | P3F steady R2 / SH | P4F effective R2 / SH |
|---|---|---|
| 64 | 0.95 - 0.99 | 2.6 - 3.2 |
| 128 | 0.95 - 1.00 | 2.05 - 2.5 |
| 219 | 0.95 - 1.00 | 1.65 - 1.95 |

- P1F: the teacher-forced agreement is >= 0.999 at each B. The point prediction is 1.000000, as
  rb1. The shipped compression and torch's `.mean` add in float32, and float32 sums of 32 bfloat16
  keys are exact almost always (above), so the fixed-order mean gives the same bits.
- P2 and P5 are unchanged.
- **Falsifiers:**
  - a steady ratio above the band at any B (the fused R2 step is not at parity);
  - an agreement below 0.999 (the fixed-order mean differs from torch's on model keys);
  - an effective ratio below the band.

## 9. r2-mbase: the matched baseline and the GEMM scan (SOURCE ONLY, not run)

Purpose: two missing inputs of the +17 % model (paper-notes/reconciliation-17pct.md rows 2 and 7;
recon-rows-2-3.md). The reference of the model is 3,160.82 tok/s: C 128, B 216, eager median
68.336639 ms (job 2175791). Q is 3,703.54 tok/s.

### 9.1 What the code can and cannot run

- C = 63 + P. P is the victim pool (NOSI_POOL_BLOCKS).
- The R2 decode body has NO victim pool. `DecodeBody.allocate` raises when `pool_blocks` is not 0
  (nosi/nosi/r2_decode_body.py:99-100). So a captured R2 fused step at C > 63 does not exist.
- At P > 0 the cells run the SH arm only. The SH steady statistic excludes every re-warm period
  (a re-warm re-captures the shipped after-pooling graph). So the SH steady step is the shipped decode
  WITHOUT the re-capture cost. The recon accepts this definition (row 2: "or the SH steady step").
- The fused step at C > 63 is a MODEL only: SH steady x 0.990 (the C 63 P3F ratio at B 219). The cells
  report it as MB-R2C and never as a measurement.

### 9.2 Cells (benchmarks/Efficiency/r2_base_fused_step.sh; 512 steps, 16,128 tokens, splits 4)

| cell | C | P | B | arms | STEP_MIN | --mem | estimated wall |
|---|---|---|---|---|---|---|---|
| mb1 | 128 | 65 | 216 | SH | 35 | 240G | 16-18 min |
| mb2c113 | 113 | 50 | 240 | SH | 35 | 240G | 18-20 min |
| mb2c96 | 96 | 33 | 280 | SH | 40 | 340G | 20-23 min |
| mb2c79 | 79 | 16 | 328 | SH | 45 | 340G | 23-26 min |
| mb2c63 | 63 | 0 | 344 | SH + R2 fused | 70 | 340G | 40-45 min |
| fit<C>b<B> | any of the five | | B | SH, 16 steps | 30 | 240G / 340G | 5-20 min |

- Wall estimate: load + pool JIT build (about 4 min) + SH prefill 3.03 s per request (basef b219:
  664 s at B 219) + 512 steps + 9 re-warms (about 0.0275 s per request each). mb2c63 adds the R2 arm
  (prefill 2.85 s per request). basef b219 (SH + R2, B 219) took 26 min 18 s.
- B per C: `mbase_batch(C)`, the largest multiple of 8 whose modeled prefill peak is <= 82.04 GB.
  Model: peak = 19.0 + B x (0.155 + (C - 63) x 0.002105) GB (2175791 output 2; basef b219 52.925 GB
  against 52.945). C 128 gives 216 (2175791: 216 fits, 224 OOM). C 63 uses B 344, 2175791's measured
  point, not the modeled GPU limit (400): it calibrates the harness at a second C.
- Host: pinned host blocks round up to a power of two: 128 GiB up to B 251, 256 GiB above.
- gm1 (benchmarks/Efficiency/r2_bi_step.sh ... gm1): r2_bi_microgate.py --gemm-scan. STEP_MIN 15,
  estimated wall 3-6 min. It needs no model and no workload.

### 9.3 Registered predictions (MB)

- MB-STEP, the SH steady step (ms), point and band:
  - C 128 B 216: 68.0, [64.92, 70.39] (2175791 eager 68.34 x [0.95, 1.03]).
  - C 113 B 240: 74.5, [68.0, 81.0].
  - C 96 B 280: 85.6, [78.0, 94.0].
  - C 79 B 328: 106.0, [92.0, 118.0].
  - C 63 B 344: 180.6, [174.5, 185.3] (2175791 eager 179.87 x [0.97, 1.03]).
  - Basis for C 79 / 96 / 113: the C 63 resident step R(B) of 2175791 + 1.26 us per load (64 x the
    misses per request-step of cache_engine.py:31-34, x 0.7-1.0) + the pool's device copies.
- MB-FIT: the SH peak allocated is within 5 % of the model.
- MB-F: at C 63 B 344, R2 fused / SH steady is in (0.95, 1.00).
- ACROSS CELLS (the lead scores it; no cell can): the best SH steady tok/s over the five C is in
  [3,100, 3,450], at C 96 or C 113. C 128 is within 5 % of the best.
- Falsifiers:
  - any cell's SH steady step outside its band;
  - the best C at or above 3,703.54 tok/s: then the model's gain is gone;
  - the best C is C 63 or C 79;
  - a registered B does not fit (SH-RUN fails with an OOM): the peak model is wrong. Then run a fit
    probe at B - 8 and B - 16, and report the C at the B that fits.

### 9.4 Registered predictions (gm1)

- GM-NAT, cuBLAS natural per step (32 x four linears + lm_head), ms, point and band: M 64 14.46
  [13.0, 16.0]; 128 16.12 [14.5, 17.8]; 256 22.13 [19.9, 24.4]; 320 28.13 [25.3, 31.0]; 512 38.26
  [34.4, 42.1]; 768 57.54 [51.8, 63.3]; 861 67.2 [57.5, 77.0]; 1024 76.8 [65.0, 88.0]; 1148 96.1
  [77.0, 106.0]. Basis: the old M-scan (2175525, graph timing) x 1.105 (bigemm_v1 eager / old scan
  at M 128 / 256 / 320); past 768 by 256-row tiles.
- GM-EXACT at B 287 (M_fix 1148): EXACT / natural = 3.8 [2.8, 4.6] at 287 live rows, 1.43 [1.1, 1.8]
  at 861 live rows.
- GM-DENSE at B 287, K 3: FAST = 3 natural(287) + natural(861) = 142.5 [120, 165] ms, NOT binding
  (HBM 213.90 ms). EXACT = 3 padded(287, 1148) + padded(861, 1148) = 385 [300, 430] ms, BINDING.
- Falsifiers: a natural per-step total outside its band; FAST dense above 213.90 ms (then the GEMM
  binds even in FAST); EXACT dense below 213.90 ms (then EXACT does not bind).

### 9.5 Doubts

- The fused step at C > 63 is not measurable without a victim pool in the R2 body.
- The C 79 / 96 / 113 bands come from a step model, not from a run at those C.
- The holder's memory must allow --mem 340G. Read the holder's allocation before the queue entry.
- The pool path has run at 63 steps (2175791). It has not run for 512 steps.
