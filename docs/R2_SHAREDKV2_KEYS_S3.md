# R2 shared ready-KV reads, part 2: Residency page keys and cell S3 (SOURCE ONLY, NOT RUN)

Branch `r2-sharedkv2`, worktree `/mnt/beegfs/ojerbi/nosi-r2-sharedkv2`, base `f79d0df` (r2-sharedkv; its plan:
`docs/R2_SHAREDKV_PLAN.md`). No CPU test and no GPU cell ran (no interpreter on the login node).

## 1. Residency page keys (the S2 blocker)

The shared kernel orders a request's union by (block id, slot). The ResidentStore gives block ids
(`TickEngine.page_keys`); the Residency backend refused. Now:

* `Plan.page_key` (r2_residency.py): `(R, H, P)` int32 on the device, the block id of every `page_table` entry, -1
  padded; `None` unless the service builds it.
* `r2_residency_sc.page_keys_of(dec, listed, hb, nt)`: device ops only. The listed blocks (`valid` for
  `device_verify`, `ready` for `device_draft`) sorted ascending: both decisions order their page slots by the same
  sorted keys (`vkey` / `rkey`), so position j of the keys is position j of the table. Then tail page u = block
  `hist_bound + u`. It reads the decision's `blk` / `valid` / `ready` columns, NOT its slot choices, so the X-pool
  branch's slot allocation change does not touch it.
* `ResidencySC.want_keys` (default False: the default path is unchanged), `_page_keys`, `_check_tail_keys` (host
  `Row` objects only, once per tick: refuses a row whose tail blocks are not `hist_bound, hist_bound + 1, ...`, the
  rule of `ResidencyBackend.prepare`). B2 (`OwnResidency`) and the sparse record inherit S-C's `resolve`, so both get
  keys. v1 and S-B build none: `ResidencyBackend.page_keys` refuses (TickError), never guesses.
* `ResidencyBackend.__init__` sets `svc.want_keys` when `ops.attn.needs_page_key`; `page_keys(..., plan=)` returns
  `plan.page_key`. `TickEngine._layer` passes `plan=plan` (ResidentStore ignores it).
* Negative control `key_plant = (h, j)`: + 1 on the first verify row's key (h, j); `key_plant_live` (device count)
  is the L34 precondition: the plant hit a listed entry.

### Merge notes (lines at this commit; another builder changes slot allocation in nosi-r2-xpool and resolve in
nosi-r2-fusedres)

| file | touched | what |
|---|---|---|
| r2_residency.py | Plan, +2 lines after `refused` (107-108) | the `page_key` field |
| r2_residency_sc.py | +27 before `class ResidencySC` (924-950) | `page_keys_of` (new function) |
| r2_residency_sc.py | class docstring end + 3 class attributes (958-964) | `want_keys`, `key_plant`, `key_plant_live` |
| r2_residency_sc.py | resolve: +2 after `_check_rows` (1161-1162), +2 before `return plan` (1205-1206) | the tail check, `plan.page_key` |
| r2_residency_sc.py | +29 after resolve (1209-1237) | `_check_tail_keys`, `_page_keys` |
| r2_tick.py | docstring 26-27, `_layer` 715, ResidentStore.page_keys 985-986, ResidencyBackend.__init__ 1036-1037, ResidencyBackend.page_keys 1075-1083 | the plumbing |

If fusedres replaces S-C's `resolve`, keep the two hook pairs (or call `_page_keys` with the same `step` / `tk`).

### CPU tests (`benchmarks/Efficiency/tests_cpu/test_r2_sharedkv2_keys.py`)

| node | what |
|---|---|
| `test_K1_residency_keys_equal_the_resident_store[sc]`, `[b2]` | every keyed call of the teacher-forced Residency tick: verify keys = the ResidentStore's keys (`TickEngine.page_keys` on the same `sel`); draft keys = those filtered to the blocks READY at the tick start (host state after the poll) + tails; every historical entry resident at the tick start is read from that slot; `validate_keys` passes; the tick = serial. Preconditions: keyed calls, verify and draft lists, a MASKED draft list, a draft list with historical blocks, slot checks done |
| `test_K2_wrong_key_is_caught_with_its_precondition[sc]`, `[b2]` | plant (0, 0): precondition `key_plant_live > 0`, mismatches > 0, all at verify rows head 0, at most one per hit |
| `test_K3_shared_attention_on_the_residency_plan[split-sc]` ... `[whole-b2]` | the shared twin on the plan's keys: probe 0 diffs; shared output drives the tick = serial; the drop-page plant detected |
| `test_K4_refusals_and_the_default_path` | v1 / S-B refuse; no keys without `needs_page_key`; the tail rule refuses |
| `test_K5_the_key_path_has_no_host_transfer` | F12d / R5e lint on resolve, `page_keys_of`, `_page_keys` (+ negative) |

Also run the existing `test_r2_tick.py` (T9, T15) and `test_r2_residency.py` (F12d, R5e) and
`test_bi_paged_attention_shared.py` (its `test_tick_page_keys_follow_the_table` still expects the refusal of a call
without a plan).

## 2. Cell S3: DRAM bytes and L2 hit rate (ncu)

`benchmarks/Efficiency/r2_sharedkv_ncu.py` (driver) and `r2_sharedkv_ncu_step.sh` (the holder step, a copy of
`r2_sharedkv_step.sh`). ncu: `/opt/nvidia/nsight-compute/2025.1.0/ncu` inside `nosa-env.sif` (used by
`r2-runs/jhbm/jh_j2_2188818_b64`, job 2175791). The tick here is EAGER (no captured tick in this tree).

* cell: `SharedKVProbe(..., ncu_window=(32, 64))`: among keyed calls with 4 rows of one request, calls 32 .. 95 are
  marked (the first 32 compile the kernels). Each marked call: one `cudaProfilerStart / Stop` window around piece
  3's launch (NVTX `skv|pr|c<j>`: `_bi_split_kernel` + `_bi_combine_kernel`) and the shared launch (`skv|sh|c<j>`:
  `_bi_shared_kernel` + combine), on the same inputs. The union build and the counting launch are outside.
  `--profile-from-start off`, so nothing else is profiled.
* arms: `b32-split`, `b32-whole` (ResidentStore, B 32), `b64-split`, `b64-whole` (B2 on the SG5 path with the page
  keys of section 1, C 96 X 32 R_ver 16, gather transport). 16 committed tokens per request, free-running,
  draft_drop 0 on the ResidentStore.
* metrics: `gpu__time_duration.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum`,
  `lts__t_sectors_srcunit_tex_op_read.sum` (required: a missing one stops before profiling; checked by
  `ncu --query-metrics --chip ga100`), `lts__t_sector_op_read_hit_rate.pct`, `lts__t_sector_hit_rate.pct`
  (optional; never observed on this rig, so dropped and reported if absent; then the hit rate is derived as
  1 - DRAM read / (32 B x L2 read sectors)).
* variants: `--cache-control all` (each kernel from a flushed L2: the kernel's OWN reuse; PRIMARY) and `none` (the
  tick's L2 state; REPORT; the A/B alternates which kernel runs first).

### Byte model (per marked call)

SUM = sum over rows and heads of num_pages x 32768 B (K and V of 64 rows, D 128, bf16): what piece 3 reads with no
reuse. UNION = sum over groups and heads of the union size x 32768 B: the least any kernel must read from HBM. Bias
and q are about 1 % on top.

### 3. REGISTERED prediction (written before any run; cache-control all; `S3_PRED` in the driver)

Basis: the 4.5 harness's attention DRAM 0.363 (joint_shared vs joint_private at B 64, r2-runs/jhbm/jh_j2_2188818_b64
summary.md line 47): the union window vs PRIVATE windows (separate copies, so no L2 sharing between rows). Logical
union / sum 0.282 there. Piece 3 v2 measured 1.72 x the time of 1 row for 4 rows (not 4 x): rows of one request read
the SAME addresses, so adjacent programs may already hit L2; if time follows bytes, piece 3 reads about
1.72 / 4 = 0.43 of the sum.

| quantity | split: central [band] | whole: central [band] |
|---|---|---|
| x_pr = piece 3 DRAM / SUM | 0.43 [0.30, 0.65] | same |
| ratio = shared DRAM / piece 3 DRAM (main kernels) | 0.75 [0.55, 0.95] | 0.70 [0.50, 0.95] |
| sh_over_union = shared DRAM / UNION | 1.15 [0.95, 1.60] | 1.05 [0.95, 1.30] |
| f = (piece 3 DRAM - UNION) / (SUM - UNION) | 0.20 [0.05, 0.50] | same |
| L2 read hit rate, piece 3 (%) | 55 [30, 80] | same |
| L2 read hit rate, shared (%) | 15 [0, 40] | 10 [0, 35] |

B 64 vs B 32 (REPORT): more programs in flight compete for L2, so x_pr(B 64) >= x_pr(B 32) - 0.05.

Decision (the -29 % reconciliation gap, paper row 6; break-even from the plan's P4: tiles / (4 x 64) <= 0.466):
* D1: x_pr <= 0.466: piece 3 already reads at most the break-even bytes from HBM; the gap is a logical count, not an
  HBM-byte gap (its time cost must come from elsewhere).
* D2: x_pr > 0.466 >= x_sh: the gap is real in HBM bytes and the shared kernel closes it.
* D3: x_sh > 0.466: real, and not closed by the shared kernel.
* f is the real fraction of the logical excess (SUM - UNION) that piece 3 pays in HBM: "how much of the gap is real".
The prediction (x_pr 0.43) sits just under the break-even: D1 is the central expectation, D2 is inside the band.

### Falsifiers

* F-S3-0 (MUST, the measurement): `O_windows` (each marked call has exactly its two kernels per range, nothing
  untagged), `O_bounds_all` (cold L2: 0.9 x UNION <= DRAM read <= 1.15 x SUM for both kinds), `O_cell_bitwise`
  (0 differing rows over all keyed calls). Any failure: no ratio is read.
* F-S3-1 x_pr outside [0.30, 0.65]: below = piece 3 dedups more than the 1.72 x time suggests (the gap is mostly
  not in bytes); above = little L2 reuse, the 1.72 x was not a byte effect.
* F-S3-2 ratio > 0.95: the shared kernel saves no HBM bytes over piece 3 (L2 already dedups); ratio < 0.55: piece 3
  gets less reuse than predicted.
* F-S3-3 sh_over_union outside its band: the shared kernel does not read the union (the byte model of P2 is wrong;
  check the counter `tile_ratio` in the same summary).
* F-S3-4 x_pr(all) and x_pr(none) more than 20 % apart: the cold measurement does not represent the tick.
* Device kernel time under ncu (`R_S3_kernel_time`): serialized and cold, a REPORT, not the tick's time.

### CPU tests (`benchmarks/Efficiency/tests_cpu/test_r2_sharedkv_ncu.py`)

| node | what |
|---|---|
| `test_N1_summary_on_a_synthetic_export` | parser, window check, bounds, exact quantities, D2, last line |
| `test_N2_summary_negatives[missing_kernel]`, `[untagged]`, `[below_union]`, `[above_sum]`, `[bitwise]`, `[no_hit]` | each defect fails its MUST gate; a dropped hit metric is not read as 0 |
| `test_N3_probe_marks_calls_with_four_rows_in_the_tick` | marking through the production tick (CPU tiny model) |
| `test_N4_parse_args_refusals` | window / variants |
| `test_N5_step_dryrun_refusals_and_pins` | the step: usage, uuid file, TICK_HOLDERS, HEAD; full DRYRUN prints both blobs and the arm's arguments (clean checkout of this worktree only, else skipped) |

### Step

`bash benchmarks/Efficiency/r2_sharedkv_ncu_step.sh --nosi <sha> <holder> <run> b32-split|b32-whole|b64-split|b64-whole`
(OUTROOT `/mnt/beegfs/ojerbi/r2-runs/sharedkv2`, claim `skv2_active`, STEP_MIN 90, 200 GB host memory). Order:
b32-split, b64-split, b32-whole, b64-whole; stop at the first MUST failure. Not reviewed.
