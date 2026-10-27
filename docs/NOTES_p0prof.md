# NOTES p0prof: Task 0.1 (baseline profile + ceiling) and the Task 0.4 rule

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md` Task 0.1, Task 0.4, and the
  coordinator's addition from `P0_PILOT_AUTOPSY.md` 7.4 / 9.2 (the verify's extra selection goes into the ceiling).
- Worktree: `/mnt/beegfs/ojerbi/nosi-r2-p0prof`, branch `r2-p0prof`, parent `ded9d41` (the mb1 code).
- Status: SOURCE ONLY. Not run. No CPU test ran (the lead runs pytest). No GPU job ran.
- Style: ASD-STE100.

## 1. Files

| file | what |
|---|---|
| `benchmarks/Efficiency/r2_base_profile_cell.py` | the cell: arm H (headline), arm T (`--stage-timers`), arm N (`--nsys-window`) |
| `benchmarks/Efficiency/r2_base_profile_step.sh` | the step script: cells mb1, p0c128, p0c63b344, p0c63b216, p0k128b63, p0nsys, p0all |
| `benchmarks/Efficiency/tests_cpu/test_r2_p0prof.py` | CPU tests PP1-PP15 |
| `docs/NOTES_p0prof.md` | this file |

No file of the engine changes. `nosi/`, `r2_base_cell.py`, `r2_ragged_gate.py` and `r2_g0_compare.py` are the bytes of
`ded9d41`. The step script stops if they differ (`git diff --quiet ded9d41 <sha> -- ...`). Test PP13 checks the same.

## 2. How the cell works

- Arm H = `r2_base_cell.run_shipped` with `r2_base_cell.CudaClock`. This is the function and the clock of the mb1
  record. The cell installs no wrapper, no callback and no class before H ends. Gate H-UNPATCHED compares every patch
  target by identity before and after H. So FLAG OFF = BASELINE holds by construction.
- Arm T runs only with `--stage-timers`, after H, with a new prefill. It uses `decode_loop` = `run_shipped` with hook
  calls (test PP13 compares the device lines of the two functions).
- Stage timers: CUDA events at stage boundaries. Wrappers on the names that the shipped step reads at call time
  (`nosa_llama.nosa_linear`, `.infllmv2_attn_stage1_fast`, `.flash_attn_nosa_with_kvcache`,
  `LlamaLayer.decode_forward`; `cache_engine.diff`, `._pool`, `._flash_pool_swap`, `.flash_h2d_from_mask`,
  `.flash_h2d_from_mask_bias`; the cache instance's `decode_update_kv`). All operations run on the current stream,
  so the stages of one step partition its period. Gate T-CHAIN checks the label sequence of every timed step. The
  events go into a ring of 4 sets; the host reads a step's times 3 steps later, so the timers add no device wait.
- Stages (per layer, summed over the 32 layers per step): `stage1`, `pool_topk` (= selection), `diff`, `pool_resolve`
  (= diff + pool resolve), `pool_swap`, `fetch_k`, `fetch_v` (= gathers), `attention`, `qkv_gemm`, `ffn_gemm`, `final`
  (= GEMMs / MLP), and the rest (`embed`, `qkv_prep`, `tail_write`, `map_commit`, `writeback`, `inter`, `sample`).
  At pool 0, `diff` also holds the map commit copy.
- Launches per step: one torch.profiler window, steps 8-10, with the wrappers removed. The cell counts the CUDA runtime
  and driver calls by name and kind. Enqueued launches = kernel + graph + memcpy + memset. Device busy = the union of
  the GPU op intervals. Host gap = arm H steady median - device busy.
- Re-captures: a subclass of `torch.cuda.graph` times each graph enter (sync + gc.collect + empty_cache +
  capture_begin) and exit; a `gc.callbacks` entry times each collection. One row per re-warm step, with the arm H
  period of the same re-warm.
- Selection probe (the coordinator's addition): steps 24-31 (step 24 = warm-up, not used). After the shipped stage-1
  call of each layer, the probe runs the selection again with the SAME existing ops, eagerly, into its own buffers:
  1 row per request, and R = 4 rows per request (1 draft + 3 verify) in ONE `infllmv2_attn_stage1_fast` call with
  `cu_seqlens_q` = 0, 4, 8, ... and `max_seqlen_q` 4 (the varlen route; the rows share the compressed-key reads).
  Then `nosa_pooling` + `topk(33)` + scatter + masked_fill + `topk(64)` over B and 4 B rows. No new kernel.
  The rows repeat the request's query (timing only; `causal=False`, as the shipped decode call).
  Outputs: shipped selection, 1-row eager, 4-row eager, extra of a round joint (4 rows - 1 row) and serial (3 x 1 row).
- Arm N (`p0nsys`): the same decode for 40 steps under `nsys profile -t cuda,nvtx --cuda-graph-trace=node
  --capture-range=cudaProfilerApi --capture-range-end=stop`; the cell calls cudaProfilerStart before step 8 and
  cudaProfilerStop after step 27, with one NVTX range per step. NSYS =
  `/usr/local/cuda-12.8/NsightSystems-cli-2024.6.2/target-linux-x64/nsys` in the image. Evidence (static read, not run):
  `retroinfer-eval/env/slurm/nosi_gather_focus.sbatch:53,60` uses this image and this path; job 2175759 made a
  `.nsys-rep` with it (`docs/evidence/gather_focus_2175759/nsys/nsys_nosi16/nsys_nosi16.log:98-102`); the image file
  dates from 2026-09-03, before that job.
- After arm H, the cell writes `H.json`. If arm T ends the process (for example a CUDA fault in the probe), the
  headline numbers stay.

## 3. The ceiling (in the REGISTERED block, `CEILING_FORMULA`)

Speculation pays only if `draft_ms + (verify_extra_ms incl. K extra selections) / accepted_per_round < baseline_ms`
per committed token. K = R - 1 = 3 verify positions per round. The draft can save at most the fetch stages:
`draft_ms >= baseline_ms - save_ms`, `save_ms = baseline_ms x save_frac`, with two sets:

- `gather_only`: fetch_k + fetch_v.
- `fetch_path`: diff + pool_resolve + pool_swap + map_commit + fetch_k + fetch_v.

The cell reports per set: draft floor, step speedup ceiling, tok/s ceiling (and / mb1), save / rest (the plan's
"fetch x of decode"), `verify_budget_ms(a) = a x save_ms`, and after the selection:
`a x save_ms - sel_extra_joint_ms` and `a x save_ms - sel_extra_serial_ms`, plus the smallest `a` in 1-4 with a
positive budget. baseline_ms = arm H steady median of the same process.

## 4. The REGISTERED block (the lead stamps)

`r2_base_profile_cell.py`, section "REGISTERED": `REGISTERED_UTC = "TBD"` and `PRED_P0`. Every "TBD" gives the verdict
UNREGISTERED. Stamp the bands and `REGISTERED_UTC` (from `date -u` in the same command) in one commit before the run.
Cell keys: `128_216` (p0c128), `63_344` (p0c63b344), `63_216` (p0c63b216), `128_63_s130816` (p0k128b63).
`head_band` (0.98, 1.02) is the plan's Global Constraint, not a new prediction. Test PP11 accepts "TBD" or a stamp of
the form `YYYY-MM-DDTHH:MM:SSZ`.

| name | value | band to stamp |
|---|---|---|
| P0-HEAD | H steady median / same-job mb1 median (p0c128 only) | (0.98, 1.02), set |
| P0-OVH | T steady median / H steady median | `ovh_band` |
| P0-STAGE-<group> | T median ms per step of each group, per cell | `stage_ms[cell][group]` |
| P0-LAUNCH | enqueued launches per step, per cell | `launches[cell]` |
| P0-GAP | H median - device busy, per cell | `gap_ms[cell]` |
| P0-RECAP | T period of a later re-capture step | `recap_ms` |
| P0-RECAP-GC | gc share of a later re-capture step | `recap_gc_share` |
| P0-SEL | extra selection of a round, joint, ms per step, per cell | `sel_extra_ms[cell]` |
| P0-CEIL | gather_only step speedup ceiling, per cell | `ceil_speedup[cell]` |

Basis for the stamps (MAP_cached_nosa.md 2.1, ncu step 62, C 128 B 216): selection 8.513 ms, resolve 1.178, pool swap
5.447, fetch 7.585, attention 19.245, GEMMs 19.882 of 64.790 ms; P0_PILOT_AUTOPSY.md 7.4: 26.07 ms per verify
position at 128K B 63 (C 113, NOSI_ATTN_SPLITS 0, the pilot's per-position route).

## 5. CPU tests for the lead (exact)

Run on a CPU node, from the worktree root `/mnt/beegfs/ojerbi/nosi-r2-p0prof`, at the commit of this file (a clean
checkout: PP15's full DRYRUN needs it; else PP15 skips its second half):

```
env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_p0prof.py
```

Regression (code this commit reuses, unchanged files):

```
env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_mbase.py
env PYTHONDONTWRITEBYTECODE=1 NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-p0prof python -m pytest -q -p no:cacheprovider tests/test_r2_p3.py
```

The tests need torch (CPU) and numpy, as test_r2_mbase.py does. PP14 reads the 128K token file (12.5 MB) and hashes
it; PP15 runs the step script in DRYRUN (sha256 of both workload files, `git hash-object`). PP15 strips SLURM_*.
In test_r2_mbase.py, MB4 / GM5 skip their full DRYRUN outside `/mnt/beegfs/ojerbi/nosi-r2-mbase` (expected).

DRYRUN by hand (prints every srun command; no claim, no write, no squeue):

```
DRYRUN=1 bash benchmarks/Efficiency/r2_base_profile_step.sh --nosi <HEAD sha> <holder> <run id> p0all
```

## 6. Expected GPU time per cell

Basis: the sacct rows and summaries of basef 2190876 (mb1 14:31, prefill 657.7 s, SH decode 92.1 s; mb2c63 SH prefill
994.1 s, decode 97.2 s; b219 SH prefill 664.0 s, decode 120.5 s) and ctxsweep cs2 2190876 (128K C 113 B 63: one arm
30:54, prefill 1680-1764 s). Load + JIT build is about 2 min per cell (mb1: 871 s - 658 s - 92 s = 121 s). Arms H and T
each prefill once.

| cell | C, P, B, S | arms | expected | STEP_MIN | --mem |
|---|---|---|---|---|---|
| mb1 | 128, 65, 216, 16K | SH | 15 min | 35 | 240G |
| p0c128 | 128, 65, 216, 16K | H + T | 28 min (2 + 2 x 12.5 + 1) | 45 | 240G |
| p0c63b344 | 63, 0, 344, 16K | H + T | 39 min (2 + 2 x 18.2 + 1) | 60 | 340G |
| p0c63b216 | 63, 0, 216, 16K | H + T | 28 min (2 + 2 x 13.1 + 1) | 45 | 240G |
| p0k128b63 | 128, 65, 63, 128K | H + T | 64 min (2.5 + 2 x 30.8 + 1) | 100 | 440G |
| p0nsys | 128, 65, 216, 16K | N, 40 steps | 15 min | 30 | 240G |
| p0all | mb1 + the four p0c / p0k cells | | 174 min (2.9 h) | 285 | |

p0all needs 285 x 60 + 600 s = 4.9 h left on the holder before it starts (MIN_LEFT). The order: mb1 first; then
p0c128 p0c63b344 p0c63b216 p0k128b63 on an even holder id, the reverse on an odd one. A single p0 cell stops unless the
same run id and holder have a complete mb1 record (Task 0.4).

## 7. The 128K cell

- `p0k128b63`: C 128 (P 65), B 63, S 130816 (`pg19-test-ctx130816`, 24 documents, request b = row b % 24; the step
  script pins its sha256 from its SHA256SUMS). B 63 is the largest B at 128K for any pool > 0: the pooled engine's
  int32 gather envelope `B x (S + 576) x 256 < 2^31` (cache_engine.py:216) gives 2,119,090,176 at B 63 and refuses
  B 64. GPU memory: C 113 B 63 used 62.24 GB at 128K (fit_row.json); C 128 adds about 1.98 GB
  (63 x 15 slots x 64 rows x 2 heads x 128 x 2 B x 2 x 32 layers). NOT MEASURED at C 128.
- The autopsy's 128K cell was C 113 with NOSI_ATTN_SPLITS 0. This cell is C 128 with NOSI_ATTN_SPLITS 4 (the plan's
  base). The selection does not depend on C (stage 1 reads the compressed keys only).
- No --mbase at 128K (the mbase bands and the peak model are 16K fits).

## 8. Risks and open points

1. T-EQ (T's tokens = H's) needs a run-to-run deterministic shipped decode. No record shows that it is. A T-EQ
   failure means "a timer changed the decode" OR "the decode is not deterministic"; the detail says so.
2. The probe's R-row stage-1 call is the existing varlen kernel (the same `varlen_fwd_stage1` as prefill and decode),
   but this argument set (decode, max_seqlen_q 4) has not run before. A Python error disables the probe (T-SEL
   fails, the decode goes on). A CUDA fault ends the process; `H.json` keeps arm H.
3. A stage time includes device idle time when the host falls behind. MAP 2.1 says the step is device-bound at B 216.
   The timers add about 450 event records per step and the wrapper calls (host work). P0-OVH reports the cost.
4. The profiler window is in arm T, with the wrappers removed (the shipped code), not in arm H.
5. The `torch.cuda.graph` subclass uses the torch 2.6 class (its capture stream comes from the base class, which arm H
   already set).
6. The probe's pooling is eager; the shipped pooling is a graph replay. Compare `one_row` with `shipped` for the
   graph benefit; the joint extra uses the eager route on both sides.
7. Output for `analysis/plan/P0_PROFILE.md`: `summary.json` keys `results.stages.groups`, `results.selection`,
   `results.launches`, `results.gap_ms`, `results.recapture`, `results.ceiling`, `results.baseline_ratios`; the SCORE
   and REPORT lines in `stdout.txt`.
