# NOTES p0sweep: the speculation ceiling where it could still win (Task 0.1 continued)

- Inputs: `kstep-cpu-runs/analysis/plan/SCORING_p0all.md` (no measured (C, B) beats cached NOSA; the fetch saving is
  large only at small C, where B is large and the verify rows cost nearly full compute-bound GEMM rows),
  `PLAN_speckv_on_cached_nosa.md` Task 0.1, `docs/NOTES_p0prof.md`, `SCORING_dr1.md`, `SCORING_vr1.md`.
- Question: is there a (C, B, S) where the GEMMs stay near their weight-read floor (small B) AND the fetch is a large
  share (small C or a long context), so that a joint verify pass is cheap and the draft saves much?
- Worktree: `/mnt/beegfs/ojerbi/nosi-r2-p0sweep`, branch `r2-p0sweep`, parent `3351726` (p0prof, stamped PRED_P0).
- Status: SOURCE ONLY. Not run. No CPU test ran (the lead runs pytest). No GPU job ran. PRED_SWEEP is all TBD.
- Style: ASD-STE100. Labels: MEASURED (a record), DERIVED (arithmetic on MEASURED values), MODEL (an estimate).

## 1. Files

| file | change |
|---|---|
| `benchmarks/Efficiency/r2_base_profile_cell.py` | `--context 65280`; VerifyProbe (the verify window of arm T); gemm_rows_summary, vfy_summary, verify_ceiling, verdicts_sweep; REGISTERED block PRED_SWEEP (all TBD); gate T-VFY |
| `benchmarks/Efficiency/r2_base_profile_step.sh` | ROOT = this worktree; 64K workload pin; cells p0c63b64, p0c63b128, p0c128b64, p0k64c63, p0k64c128; sequence p0sweep |
| `benchmarks/Efficiency/tests_cpu/test_r2_p0sweep.py` | new: PS1-PS12 |
| `benchmarks/Efficiency/tests_cpu/test_r2_p0prof.py` | two edits: PP1 (65280 is now a valid context: the refused value is 32768), STEP_ROOT (the step script's ROOT moved here) |
| `docs/NOTES_p0sweep.md` | this file |

No file of the engine changes. `nosi/`, `r2_base_cell.py`, `r2_ragged_gate.py`, `r2_g0_compare.py` are the bytes of
`ded9d41`. The step script stops if they differ; PP13 checks the same. NO NEW KERNEL: every probe op is an existing
torch / flashinfer op that the shipped decode already calls (`F.linear`, `layer_norm` = flashinfer rmsnorm,
`silu_and_mul`, `torch.sort`, `torch.cat`, `repeat_interleave`, compares and sums).

## 2. The cells

Every sweep cell = arm H (the plain headline, `r2_base_cell.run_shipped`, nothing installed) + arm T (`--stage-timers`:
the stage timers, the profiler window, the 1-row vs 4-row selection probe, and the new verify window). The same-job mb1
rule of p0all stays: a single sweep cell STOPS unless the same run id and holder have a complete mb1 record; p0sweep
runs mb1 first, and a failed mb1 stops the sequence.

| cell | C (P) | B | S | why | --mbase |
|---|---|---|---|---|---|
| p0c63b64 | 63 (0) | 64 | 16128 | small B, no pool: GEMM near the floor, fetch a large share | yes |
| p0c63b128 | 63 (0) | 128 | 16128 | the middle of the GEMM curve at C 63 | yes |
| p0c128b64 | 128 (65) | 64 | 16128 | the same B with the pool: the fetch share falls | yes |
| p0k64c63 | 63 (0) | 127 | 65280 | long context, no pool, the largest B | no |
| p0k64c128 | 128 (65) | 127 | 65280 | long context with the pool, the largest B | no |
| p0sweep | | | | mb1 + the five; even holder: the order above; odd holder: the reverse (mb1 first) | |

### 2.1 B at 64K (DERIVED from the code and the 128K arithmetic of NOTES_p0prof.md 7)

- The int32 envelope: `B x (S + max_gen_len) x H_kv x D < 2^31` (cache_engine.py:216; flash_h2d_mask.py does not cast
  its program ids). `max_gen_len = steps + 64 = 576` (decode_loop, run_shipped); H_kv = 2, D = 128 (config.json).
- 64K: (65280 + 576) x 256 = 16,859,136 per request. B 127: 2,141,110,272 < 2,147,483,648. B 128: 2,157,969,408. So
  B 127 is the largest B. 128K (check): B 63 = 2,119,090,176; B 64 refused (as NOTES_p0prof.md 7).
- The engine asserts the envelope only with a pool (cache_engine.py:212-214: "Guarded to the pooled path so p=0 stays
  upstream byte for byte"). At pool 0 the same gathers overflow silently above B 127. The step script uses B 127 for
  both 64K cells. PS11 checks the arithmetic and the script.
- Memory does not bind at B 127 (section 7): ctxsweep cs2 2190876 also ran B 127 at 64K with `bound_by: int32_host`
  (memory bounds there: 202 at P 0 and 135 at P 50, for a 75 GB budget).

### 2.2 The 64K workload pin

`WLM_SHA = f2fb36f441ec12b7a79239d5e86263c6454b7713436f53f81086ef2bedcc34b3` comes from the workload's own
`pg19-test-ctx65280/SHA256SUMS` (the same source as the 128K pin, "PIN (its SHA256SUMS)"). I computed no hash. The
step script checks it with `sha256sum`, as it checks the 16K and 128K pins (the existing pattern). The cell's
`load_workload` checks the file against SHA256SUMS on the compute node and records the sha in `meta.workload_sha256`.
The manifest gives shape 55 x 65280, int32-le (request b = row b % 55; B 127 uses all 55 documents).

## 3. The verify window (VerifyProbe; arm T, steps 34-45 by default)

`--vfy-start 34 --vfy-steps 12 --gemm-mults 1,2,3,4`. The window has no re-warm (re-warms at 64K are 0, 64, 128, 192,
256, 288, 320, 384, 448, the same as 16K; PS1) and it lies apart from the profiler window (7-11 with its neighbours)
and the selection window (24-31). Kind 'vfy'. The steady set loses these 12 steps (478 steady timed steps instead of
490). The probe reads shipped tensors and writes only its own tensors, so T-EQ still holds.

Per layer, in stream order:

| label | where | what |
|---|---|---|
| `union` | decode_update_kv entry (before `tail_write`) | copy `topk_idx` (H x B x 64 block ids) into a ring of R = 4 per layer; when the ring is full, `sort` the 4 x 64 ids of each (head, request) and count the distinct ids |
| `load_count` | flash_h2d_from_mask entry (before `fetch_k`) | host loads of the layer: `(load mask >= 0).sum()`; the pool lookup has already cleared the pool hits |
| `gemm_prep` | decode_forward exit (after `ffn_gemm`) | the layer input repeated m times per request (`repeat_interleave`) |
| `gemm_x<m>` | after `gemm_prep` | the GEMM stack of the shipped step on m x B rows: rmsnorm + wqkv; wo + residual; rmsnorm + gate_up + SiLU-and-mul + down + residual; the last layer adds the final rmsnorm + lm_head + float. m = 1 repeats the group gemm_mlp eagerly. |

The sums stay on the device; the host reads them once after the arm. The first window step is a warm-up for the GEMM
and fetch times (new GEMM shapes). The union counts from window index 3 (the ring holds 4 steps): 9 union steps.

The union is the union of 4 CONSECUTIVE shipped selections (t - 3 .. t). A joint pass verifies 4 consecutive positions
(t .. t + 3). INFERRED: the two have the same distribution. Check: vr1 MEASURED 72.14 per (head, request) at C 128 B
216 (SCORING_vr1.md 8, extras 8.137); the selection does not depend on C (stage 1 reads the compressed keys only).

## 4. GEMM ms vs rows

`results.gemm_rows`: per m, the median ms per step (32 layers + final) at m x B rows, ms per row, the excess over the
floor (10.6 ms, SCORING_p0all.md 3), the marginal ms per row between neighbour points, the least-squares line
(intercept = the floor estimate of this cell), and probe(B rows) / shipped gemm_mlp (eager vs eager; expect about 1).
Together the five cells give probe points at 64, 127, 128, 192, 254, 256, 381, 384, 508 and 512 rows. Earlier points:
p0all's shipped gemm_mlp at 63, 216 and 344 rows (the same op set) and vr1's 864-row GEMM phase (it also holds
nosa_linear and rope). The curve shows where the GEMM leaves the floor.

## 5. The ceiling with the measured verify model (`results.verify_ceiling`, VERIFY_FORMULA)

- Round: K = 3 draft steps + 1 joint pass of R = 4 rows per request. Committed tokens per request at per-token
  acceptance a: E(a) = 1 + a + a^2 + a^3 (3.9701 at a = 0.995; 4 at a = 1).
- `draft_ms = baseline_ms - share x gather_ms`; gather_ms = H median x the gather_only fraction (as `ceiling`). Shares
  (DRAFT_RECOVERY): `floor` 1.0; `dr1_69` 0.6926 (dr1 saved 3.368 ms of p0all's 4.863 ms gather; SCORING_p0all.md 8.1,
  applied at C 63 in 8.4); `dr1_44` 0.4440 (3.368 of the ncu gather 7.585 ms; SCORING_dr1.md 9). `dr1_for_pool`: dr1_69
  at pool 0, dr1_44 with the pool (the lead's mapping). Every cell reports all three.
- `joint_ms = baseline_ms + (gemm_x4 - gemm_x1) + (selection 4 rows - selection 1 row) + union_fetch_ms`: the shipped
  step with its GEMM stack and its selection on 4 rows per request, ONE attention (perfect K / V sharing), and the
  union extras. The lead's form is printed too: GEMM(4B) + selection(4 rows) + one attention + union fetch + the rest
  of the shipped step (row 0's fetch, resolve, swap, rest).
- `union_fetch_ms = extras x streams x ms per load`; extras = union - 64 per (head, request); streams = 32 x 2 x B; ms
  per load = window fetch ms / window host loads. Every extra counts as a host load: exact at pool 0 (the GPU holds only
  the current 64 blocks), an upper bound with the pool.
- Variants: `measured` (union fetch on the critical path) and `hidden` (0: a background prefetch hides it; the
  SCORING_p0all 8.3 assumption).
- Per (fetch, draft, a): committed tok/s = 1000 x B x E / (3 x draft + joint); / 3,387.7 (cached NOSA, mb1 2190876);
  / own H tok/s; / same-job mb1; break-even joint_ms against both rates and the margin.
- NOT in the model (each MEASURED as tens of ms in vr1 at C 128 B 216): the positional twin (35.6 ms), the extras
  residency copies (VRF resid 215.7 ms), serial attention, rollback. So the model is optimistic on these terms.

## 6. The REGISTERED block PRED_SWEEP (the lead stamps)

`r2_base_profile_cell.py`, section "REGISTERED p0sweep": `REGISTERED_SWEEP_UTC = "TBD"` and `PRED_SWEEP`. Every "TBD"
gives UNREGISTERED. Stamp the bands (low, high) and `REGISTERED_SWEEP_UTC` (from `date -u` in the same command, L33)
in one commit before the run. Keys: `63_64`, `63_128`, `128_64`, `63_127_s65280`, `128_127_s65280`.

| line | PRED_SWEEP name | value |
|---|---|---|
| SW-STAGE-<group> | stage_ms | T steady median ms per group |
| SW-GEMM-X1 / -XR | gemm_x1_ms / gemm_xr_ms | probe GEMM stack at B / 4B rows, ms per step |
| SW-GEMM-SLOPE | gemm_slope | least-squares ms per row |
| SW-SEL | sel_extra_ms | joint extra selection of a round (4 rows - 1 row) |
| SW-UNION | union_extras | union of 4 consecutive selections - 64, per (head, request) |
| SW-LOADS | loads_per_stream | host loads per (layer, head, request) per step |
| SW-UFETCH | union_fetch_ms | union-extras fetch per joint pass |
| SW-JOINT | joint_steps | joint pass (measured fetch) / H median |
| SW-CACHED-FLOOR / -DR1 | vs_cached_floor / vs_cached_dr1 | committed tok/s / 3,387.7, a 0.995, measured fetch |
| SW-OWN-FLOOR | vs_own_floor | committed tok/s / own H tok/s, a 0.995, draft floor, measured fetch |

The P0-* lines of the sweep cells read PRED_P0. Its per-cell bands have no key for these cells (UNREGISTERED). Its
global bands (ovh_band, recap_ms, recap_gc_share) were stamped for p0all: read them as REPORT for the sweep cells.

## 7. CPU tests for the lead (exact)

Run on a CPU node, from the worktree root `/mnt/beegfs/ojerbi/nosi-r2-p0sweep`, at the commit of this file (a clean
checkout: PS12's and PP15's full DRYRUN need it; else their second half skips):

```
env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_p0sweep.py
env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_p0prof.py
```

Regression (code this commit reuses, unchanged files):

```
env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_mbase.py
env PYTHONDONTWRITEBYTECODE=1 NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-p0sweep python -m pytest -q -p no:cacheprovider tests/test_r2_p3.py
```

The tests need torch (CPU) and numpy. PS10 reads the 64K token file (14.4 MB) and hashes it; PP14 the 128K file.
PS12 / PP15 run the step script in DRYRUN (sha256 of the three workload files, `git hash-object`); they strip SLURM_*.

DRYRUN by hand (prints every srun command; no claim, no write, no squeue):

```
DRYRUN=1 bash benchmarks/Efficiency/r2_base_profile_step.sh --nosi <HEAD sha> <holder> <run id> p0sweep
```

## 8. GPU time per cell (STEP_MIN) and the holder time

Basis: p0all 2195574 (mb1 14:43; p0c63b216 25:55), mb1 prefill 657.7 s for 216 requests (3.045 s each), mb2c63
prefill 994.1 s for 344 (2.89 s each), ctxsweep cs2 2190876 at 64K B 127 (prefill 1,478.6 s at P 0 and 1,480.9 s at
P 50; decode 575.5 and 678.5 tok/s, so about 221 and 187 ms per step; another tree, MEASURED there). Load + JIT about
2 min (2.5 at 64K); 9 re-warms about 5.6 s each per arm (SCORING_p0all 5). The 16K step times are MODEL: linear in B
from p0all C 63 (117.530 ms at B 216, 180.778 at B 344: 0.494 ms per request), and about 30 ms at C 128 B 64.

| cell | prefill per arm | decode per arm | expected | STEP_MIN | --mem |
|---|---|---|---|---|---|
| mb1 | 658 s | 1.5 min | 15 min (MEASURED 14:43) | 35 | 240G |
| p0c63b64 | 185 s | 503 x 42 ms + 50 s = 1.2 min | 12 min | 30 | 240G |
| p0c63b128 | 370 s | 503 x 74 ms + 50 s = 1.5 min | 19 min | 35 | 240G |
| p0c128b64 | 195 s | 503 x 30 ms + 50 s = 1.1 min | 12 min | 30 | 240G |
| p0k64c63 | 1,479 s | 512 x 221 ms + 50 s = 2.7 min | 59 min | 90 | 440G |
| p0k64c128 | 1,481 s | 512 x 187 ms + 50 s = 2.4 min | 58 min | 90 | 440G |
| p0sweep | | | 175 min (2.9 h) | 310 | |

p0sweep needs 310 x 60 + 600 = 19,200 s = 5.3 h left on the holder before it starts (MIN_LEFT). The probes add about
0.1-0.3 s per verify step (12 steps per arm): negligible.

## 9. Memory per cell

GPU peak allocated (the prefill peak binds; the probes run in the decode, below it):

| cell | GB | basis |
|---|---|---|
| p0c63b64 | 28.9 | MEASURED 2175791 output 2 (C 63 B 64 28.93); mbase model 28.92 |
| p0c63b128 | 38.8 | MODEL (mbase peak model 19.0 + 128 x 0.155) |
| p0c128b64 | 37.7 | MEASURED 2175791 (C 128 B 64 37.69); model 37.68 |
| p0k64c63 | 53.9 | MEASURED ctxsweep cs2 2190876 s65280 original bmax, fit_row.json (another tree) |
| p0k64c128 | 71.3 | MODEL: C 113 B 127 67.270 (MEASURED, cs2 cached_p50 bmax) + 127 x 15 slots x 0.002105 GB = 4.01 |

0.002105 GB per slot and request = (64 rows x 2 heads x 128 x 2 B x 2 + 64 x 2 x 2 B) x 32 layers = 2,105,344 B
(MBASE_PEAK slope_per_block_GB; the 128K C 128 check in p0all: 64.227 MEASURED vs 64.22 predicted). p0k64c128 reserved
about 74.2 GB, used about 75.3 GB with the allocator reserve and the non-PyTorch 1.1 GB: below 84 GB. The int32 limit
binds first (section 2.1).

Probe memory in the decode: the ring 4 x 32 x 2 x B x 64 x 8 B (16.6 MB at B 127, 45 MB at B 344); the GEMM transients
at 4B rows: the lm_head output 4B x 73,448 x 6 B (bf16 + float; 0.22 GB at B 127, 0.61 GB at B 344) and gate_up 4B x
32,768 x 2 B.

Host memory: pinned K / V = B x (S + 576) x 256 x 2 B x 2 x 32 layers: 0.55 GB per request at 16K (35 GB at B 64, 70
GB at B 128: 240G as mb1); 2.16 GB per request at 64K (274 GB at B 127: 440G as the ctxsweep 64K cells, host peak
266.7 GiB there).

## 10. Risks and open points

1. The union proxy uses the past 4 selections, not the 4 future rows of a joint pass (section 3). The C 128 B 64 value
   can be compared with vr1's 72.14 (C 128 B 216).
2. ms per load includes the fixed cost of the gather launches. At C 128 the loads are few, so ms per load is high and
   union_fetch_ms is overstated. The scorer can recompute it with the C 63 cell's ms per load at the same B and S
   (p0c63b64 for p0c128b64; p0k64c63 for p0k64c128).
3. All extras as host loads: an upper bound with the pool; `hidden` gives the other end.
4. The model leaves out the twin, the extras residency copies, serial attention and rollback (section 5).
5. The dr1 shares come from one cell (C 128 B 216) on another GPU. Their transfer to other (C, B, S) is INFERRED.
6. No measured step exists for this engine at 64K with NOSI_ATTN_SPLITS 4. The time model uses ctxsweep (another tree).
7. The step script's ROOT is now this worktree. p0all from this tree also runs the verify window (12 steps fewer in the
   steady set, a new MUST gate T-VFY).
8. A probe error disables the probe; T-VFY fails; the summaries use only complete steps. A CUDA fault in a probe ends
   the process; H.json keeps arm H.
9. T-EQ needs a deterministic decode (p0all: 0 token differences in all 4 cells). The verify probe only reads shipped
   tensors.
10. `p0nsys` is still not in any sequence (as p0all).
