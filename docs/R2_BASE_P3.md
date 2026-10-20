# r2base P3: why the captured R2 step is slower than the shipped step at B 64

Branch `feature/nosi-r2-base-p3`, from `feature/nosi-r2-base` cd77de2 (the commit that the
running r2base cells pin; that worktree is not changed). Status: ANALYSIS AND MEASUREMENT
DESIGN. Nothing new has run.

## 1. The data (run rb1, holder 2190876, cell b64)

Source: `/mnt/beegfs/ojerbi/r2-runs/base/base_rb1_2190876_b64/summary.json`.

| Arm | Steady ms per step (median) | p10 / p90 | n |
|---|---|---|---|
| SH (shipped, eager, re-warmed) | 46.789 | 45.596 / 48.370 | 503 |
| R2 (captured, replayed) | 52.818 | 51.667 / 54.222 | 509 |

R2 / SH = 1.129. The registered P3 was <= 0.8, so P3 is REFUTED. The difference is +6.03 ms
per step. The two distributions do not overlap: SH's p90 is below R2's p10.

The other predictions held:

- P1: agreement 1.000, no divergence in 512 steps;
- P2: the re-warm schedule;
- P4: effective tok/s 2.784x;
- P5: peak memory 1.008x.

The cell records no per-op split, only the step periods.

## 2. Why P3's basis was wrong

P3 assumed a shipped step of about 167 ms at B 64 with a fixed part of about 135 ms. The
fixed part was host launch work that a capture would remove. The cell measured 46.8 ms. So
the shipped step at this setting is not launch-bound:

- Its pooling and top-k already run in a CUDA graph per layer
  (`nosa_llama.py`, `after_pooling_graph`).
- ORDINARY FIX A (2026-09-19) removed its per-step pageable H2D sync.
- The GPU work of a step at B 64 is longer than the host time to launch it.

The 167 ms / 288 ms basis numbers (eqmem) come from another configuration. I did not check
which one. They are not this cell's shipped step. With no large host part to remove, the
capture cannot win by itself. What remains is the difference in DEVICE work between the two
steps.

## 3. The explanation (to be measured: section 4)

Both arms run the same GEMMs, the same attention (`flash_attn_nosa_with_kvcache`, 4 splits),
the same diff kernel and the same Triton gathers. The R2 step runs much more bookkeeping
work per layer, as many small kernels:

| Part | R2 (captured step, per layer) | Shipped (per layer) |
|---|---|---|
| positional tables | `PositionalTwin.step_fixed` (`r2_positional_twin.py:276-325`): every op on ALL B rows at EVERY step (shape-stable means, masked `torch.where` writes of `comp_k`, `ring`, `total_cis`, `ccis`, the refusal masks, the `err` word): about 72 kernels by count | the compression only every 16th step, otherwise one or two copies (`cache_engine.py` `update_no_compress_k_decode`, `update_compress_k_decode`, `update_cis`) |
| decode body | `DecodeBody.decode_fixed` (`r2_decode_body.py`): masked tail writes (gather old + where + index_put, x3), the refusal masks, `select`'s masking, the full mask, the device write-back kernel, the masked `block_map` and `_cache_lens` updates: about 47 kernels | `decode_update_kv`: the tail copies, diff, the gathers, the map and the lengths: about 10 |
| copies into the capture buffers | `pos_before`, `n_used`, `M_b`, `topk_log`: about 6 | none |

That is about 110 extra kernels per layer, about 3,500 per step. A tiny kernel inside a graph
on the A100 costs about 2 µs, including the gap between graph nodes. 3,500 x 2 µs = 7 ms, the
size of the measured 6.0 ms.

Other candidates and why they are smaller:

- **Per-request stage 1 and pooling against the shipped calls.** The shapes are close:
  M_alloc = 261 against out_len 253-260; Lk_max = 1041 against about 1008-1040; score width
  1152 against 1024 / 1152. At most a few percent of those kernels.
- **The write-back.** One masked kernel per layer per step that writes almost nothing,
  against the shipped write-back once per 64 steps: about 32 kernels per step.
- **The teacher-forced driver's per-step copy / argmax / add outside the graph.** 3 kernels
  per step.

The same cost is in the step-7 tick (`docs/R2_TICK_CAPTURE_PLAN.md`, branch
`feature/nosi-r2-tick-capture`). C's `step_unchecked` also computes both means at every step
(its own note in 4fe5fc9). There the twin runs on 2B banks for up to K steps per tick.

## 4. The measurement that locates the 6 ms (M-P3)

One GPU step in the holder, B 64, the cell's workload and lengths (S 16128). One process,
prefill and warm-up as in `r2_base_cell.py`. About 15 minutes.

**SH arm.** 16 steady steps (no re-warm among them) under `torch.profiler`
(activities CPU + CUDA). The cache's own methods are wrapped, per instance in the
measurement script, with `record_function` ranges:

- `sh.k` (`update_no_compress_k_decode` + `update_compress_k_decode`);
- `sh.cis` (`update_uncompressed_cis` + `update_cis`);
- `sh.kv` (`decode_update_kv`).

The source is not changed.

**R2 arm.**

- (a) 16 graph replays under the profiler: the kernel records of the graph, if this stack's
  CUPTI reports them; the step period by CUDA events.
- (b) 16 EAGER `decode_step_fixed` steps that continue the same state. These are the same
  kernels and the same bits as the replays. The measurement script wraps them per instance
  with ranges:
  - `r2.twin` (`twin.step_fixed`);
  - `r2.body` (`cb.body.decode_fixed`);
  - `r2.stage1` (`RC.stage1_seqused`);
  - `r2.pool` (`RC.nosa_pooling_per_request`).

**Output.** Per arm and per range, and per kernel-name class: kernels per step and summed
device ms per step. The kernel-name classes are GEMM, ATTN, STAGE1, POOL, SELECT
(topk / sort / scatter / masked_fill), H2D, DIFF, WB, SMALL (elementwise / index / reduce /
copy / fill / cat / where) and OTHER. Also each mode's step period, and the period minus the
summed durations (the gaps). Then the R2 - SH difference per class and per range.

**REGISTERED PREDICTIONS for M-P3** (made before it runs; each one is CONFIRMED or REFUTED by
its numbers):

- Q1: R2 runs 2,000 to 4,500 more kernels per step than SH.
- Q2: R2's `r2.twin` runs >= 2,000 kernels per step (32 layers) and takes >= 3.0 ms of device
  time per step. SH's `sh.k` + `sh.cis` take <= 1.0 ms.
- Q3: `r2.body` takes >= 1.0 ms more device time per step than `sh.kv`.
- Q4: GEMM and ATTN device time agree within +-5 % between the arms.
- Q5: STAGE1 + POOL + SELECT differ by <= 1.5 ms per step between the arms.
- Q6: the sum of the class differences is within 4 to 8 ms per step. The measured period
  difference is 6.0 ms.

If Q2 holds, the next change is a FUSED TWIN STEP: one Triton kernel per layer for the masked
per-request update (the means, the four table writes, the counters, the refusal bits),
bitwise equal to `step_fixed`, so that about 72 launches per layer become 1. The same fused
step serves the tick (2B banks, `step_unchecked`). If Q3 holds, the same applies to the
body's masked bookkeeping. Prediction for the fused twin alone: R2's step at B 64 drops by
>= 3 ms. That prediction is registered when the kernel is built, with its own gate.

## 5. Built (source only): the cell, its step, its CPU test

- `benchmarks/Efficiency/r2_p3_profile.py`: the M-P3 cell. The shipped step is SH. The captured
  R2 step's replays are R2G, and its eager steps on the same state are R2E. Each arm has 16 timed
  steps (CUDA events), then 16 profiled steps (torch.profiler, Kineto chrome trace, gzipped beside
  the summary).
  - The GPU ops are classified by name, and each kernel is attributed to the innermost `sh.` /
    `r2.` range of its launch (by correlation id).
  - The output is ops and device ms per step, per class and per range; every kernel name; the
    periods and the gaps; and a verdict for each of Q1-Q6.
  - MUST: SH-RUN, SH-STEADY, R2-RUN, PROF.
  - torch.profiler is the tool because the HBM sweep (job 2175791, `nosa-env.sif`) already
    produced Kineto traces of NOSI on this image.
- `benchmarks/Efficiency/r2_p3_step.sh`: the holder step. It reads the GPU uuid from the holders
  file; STEP_MIN 20, `--mem` 140G, claim `/mnt/beegfs/ojerbi/r2-runs/p3/p3_active`.
- `tests/test_r2_p3.py`: P1-P5 on the pure parts. It covers the classification, the attribution
  on a synthetic trace (with negative controls), the per-step aggregation, the verdicts (each Q
  refuted alone), and the windows.

CPU (inside a CPU job):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-base-p3 python -m pytest -q tests/test_r2_p3.py tests/test_r2_base.py

GPU (from /mnt/beegfs/ojerbi/nosi-r2-base-p3; DRYRUN on the login node first):

    DRYRUN=1 bash benchmarks/Efficiency/r2_p3_step.sh --nosi <sha> <holder job> <run id>
    bash benchmarks/Efficiency/r2_p3_step.sh --nosi <sha> <holder job> <run id>
