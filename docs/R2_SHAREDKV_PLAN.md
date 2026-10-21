# R2 shared ready-KV reads: design and GPU cell plan (SOURCE ONLY, NOT RUN)

Branch `r2-sharedkv`, worktree `/mnt/beegfs/ojerbi/nosi-r2-sharedkv`, base `cf35536` (SG5 p6 line).
Status: source and CPU tests only. No GPU run. No CPU run by the author (no interpreter on the login node): the lead
runs the CPU tests on a compute node.

## 1. Why

The paper's +17% forecast (`kstep-cpu-runs/paper-notes/forecast-oct12-17pct.md`) charges the verifier the UNION of the
rows' selected KV groups (U = 69.16 per layer / head per cohort). The audit (`recon-rows-5-6.md`, row 6) found that
piece 3 v2 (`nosi/nosi/bi_paged_attention.py`, blob `39ed5e18` of `7c826c1`) runs one program per (row, KV head,
split). Each row loads its own pages, so the tick reads the SUM (about 189 + 63 groups). By the author's formula that
gives Q = 2242 tok/s (-29.1%). Break-even needs V <= 56.4 verifier groups per cohort.

## 2. Prior-art gate (done before the build)

| candidate | what it shares | why not taken as is |
|---|---|---|
| FlashInfer cascade / MultiLevelCascade, Hydragen, ChunkAttention | a shared PREFIX read once for many queries; partial states merged by log-sum-exp | the shared set is a prefix, not a per-request union of sparse pages; the merge order is the library's, not piece 3's per-row split law |
| tree / speculative-verify attention (SpecInfer, Medusa tree kernels), FlashDecoding with q_len > 1 | several query rows of one sequence over one KV list | one KV list for all rows; per-row sparse masks and per-row split partitions are absent; reduction order depends on the tile M |
| in-house 4.5 joint-HBM harness (`nosi-joint-hbm` 1c4af3e) | one union window per request, every row a query of one FA2-NOSA entry, per-row masks; attention DRAM 0.363x (b64) | FA2 split-KV, NOT batch-invariant; builds a window copy |
| in-house T10 block-diagonal kernel, piece 3 v2 | fixed tiling, no autotune, twin + gate discipline | (used: the discipline, and piece 3's own step and combine) |

Choice: a NEW kernel, because no prior design keeps a per-row split partition and a per-row reduction order. Reused:
piece 3's step formulas (verbatim, in `_sk_row_step`), piece 3's combine KERNEL (`base._bi_combine_kernel`, the same
launch), piece 3's constants (NUM_WARPS, NUM_STAGES, BLOCK_H, PAGES_PER_SPLIT) and address arithmetic. Piece 3's file
is in the tree byte-identical (blob `39ed5e18`, checked by a test); it is not edited.

## 3. Design

`nosi/nosi/bi_paged_attention_shared.py`:

* GROUP = the rows of one seq_idx (one request: its draft row and its verify rows), ascending row index, chunks of
  `rows_max` (<= 4 = 1 + K). Built on the device (`build_union`: sort, cummax, cumsum, scatter; no host read).
* UNION = the distinct (block id, slot) pairs of the group's rows, ascending. Each row lists its pages in ascending
  block id, so each row's list is a subsequence of the union. The block ids come from the tick (`page_key`), because
  ring slots do not give the order (`TickEngine.page_keys`, compacted like the table by the shared `_draft_keep`).
* one program per (group, KV head, split k) ("split", default) walks the union window that holds the rows' split-k
  pages. It loads each 64-row K / V / bias tile ONCE (mask = OR of the rows' masks), then runs each row's piece-3
  step on the row's own masked copy of the tile (`tl.where(ok, tile, 0)` = piece 3's masked load, other = 0).
  A page outside a row's split is that row's exact identity step. The row's partial goes to piece 3's workspace slot
  (row, head, k); piece 3's combine kernel runs unchanged.
* "whole": one program per (group, head) walks the whole union once and flushes each row's partial at the row's own
  split boundary (store, then reset to -inf / 0 / 0). Loads = the union exactly, but R/4 x Hkv programs (128 at B 64).
* `SharedKV` (the selector object, `ops.attn`): calls with `page_key` and rows_max >= 2 use the shared kernel; other
  calls (the BI serial REF, draft-only ticks) use piece 3. `.stats` counts each kind. `SharedKVProbe` runs both on
  every keyed call, returns piece 3's output, counts differing rows on the device, times both with CUDA events
  (alternating order), times the union build separately, and counts the kernel's own tile loads (an untimed extra
  launch with the COUNT constexpr).
* tick (`nosi/nosi/r2_tick.py`): default unchanged. Only when `ops.attn.needs_page_key` is set does `run_rows` count
  rows per request (host, no device read) and `_layer` pass `page_key` / `rows_max`. `ResidentStore.page_keys` gives
  the keys; `ResidencyBackend.page_keys` refuses (`TickError`): its plan compacts draft rows by readiness and S-C has
  no device block-id tensor (`Plan.blocks` is empty there). Building it is a follow-up with C.
* gate (`benchmarks/Efficiency/r2_tick_gate.py`): `--attn per_row|shared|shared-whole|probe|probe-whole|probe-plant`
  (default `per_row` returns piece 3 itself), `--sharedkv <path>`; after the parts: `SKV_shared_kernel_ran` (MUST),
  `SKV_bitwise_shared_equals_per_row` (MUST, probes), `SKV_plant_detected` (MUST, probe-plant), `R_SKV_time_ratio`,
  `R_SKV_tile_ratio` (REPORT, with the bands of section 5).

### Why the rows are NOT stacked into one M tile

Stacking 4 rows x 16 GQA heads into an M = 64 dot would change the tl.dot / MMA layout (warpsPerCTA) and with it the
row-sum tree of `tl.sum(pr, 1)` (and possibly the dot's K split). Then a row's bits would depend on its group: the law
breaks. So each row keeps piece 3's dot shapes (16, D) x (D, 64) and (16, 64) x (64, D) with num_warps 4. Only the
tile LOAD is shared. This is the brief's fallback design, chosen up front.

### Exactness argument and its limits

Per row, every float op has the same operands as in piece 3: the same q tile, the same masked K / V / bias values
(the select reproduces the masked load), the same mask, the same split partition, the same partial slot, the same
combine kernel. The differences are (1) more identity steps (one per union page the row does not have in the
current split) and (2) the tile is loaded by the OR mask. (2) changes no operand of the row. (1) is exact:
m = max(m, -inf) = m, alpha = 1 (the `m_new == -inf` guard or exp2(0)), l * 1 + 0 = l, acc * 1 + dot(0, 0) = acc,
with ONE known exception: an acc entry of exactly -0.0 becomes +0.0. Piece 3 runs the same identity step past a row's
last page, so it has the same exposure at the end of a row's last split, but not in the middle. The effect is only the
sign of an exact zero in the output and needs a negative acc entry that underflowed to -0.0. The bitwise gate uses
int views, so it would count such a row.

Not provable statically (the GPU gate decides): that the Triton compiler gives each row's step the same layouts and
the same FMA contraction inside the larger kernel body as in piece 3's body. L36: the interpreter does not show this.

## 4. CPU tests (`benchmarks/Efficiency/tests_cpu/test_bi_paged_attention_shared.py`)

Command (compute node):
`env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_bi_paged_attention_shared.py`
plus the existing `benchmarks/Efficiency/tests_cpu/test_r2_tick.py` (T7 / T9 cover the `_draft_keep` refactor and the
no-host-read rule of `_layer` / `run_rows` / `page_table`).

Cases (`CASES`): overlap same (100%), half (about 50%), disjoint (0%), random; rows per request 1..4 and 6 (two groups);
ring tail slots ((key, slot) pairs, not slots, are shared); rows shuffled; P = 20 (up to 3 splits per row).

| node | what |
|---|---|
| `test_build_union_invariants[...]` | groups, union, rowpos / uidx inverse, subsequence, windows (both modes) |
| `test_twin_bitwise_equals_piece3_twin[...]` | (a) shared twin vs piece 3's twin, every row, int views, fp32 and bf16, rows_max 4 and 2, both modes; preconditions: sharing, multi-split rows, misaligned splits |
| `test_twin_rows_max_one_is_piece3_twin` | the degenerate group |
| `test_negative_drop_one_union_page_is_caught[split/whole]` | (b) plant: precondition (page listed by a row; output changes), caught exactly on the listing rows |
| `test_negative_union_order_must_follow_the_keys` | ordering by slot instead of key is caught |
| `test_twin_loads_union_not_sum[...]` | (c) tiles: whole = union, split = sum over k of split unions (independent oracle); union <= split <= sum; same lists: split = union < sum |
| `test_kernel_under_the_triton_interpreter` | the KERNEL vs PIECE 3'S KERNEL bitwise (all cases, both modes, rows_max 4 / 2); the kernel's own COUNT = oracle = twin; selector stats; kernel plant caught and local; probe returns piece 3's bits, 0 diffs, tiles = oracle; probe-plant detected |
| `test_tick_page_keys_follow_the_table` | `TickEngine.page_keys` vs `page_table` (draft drop), ascending, T - 1 / T; Residency refuses |
| `test_tick_selector_teacher_forced_equals_serial[split/whole]` | SG1 on the CPU tiny model through the production tick: probe 0 diffs with shared calls; shared IN the tick = serial; plant detected in the tick |
| `test_gate_selector_default_is_piece3_and_choices` | `--attn` default returns piece 3 itself; choices; the MUST gates fail on a diff / a blind plant |
| `test_step_script_shape_and_its_sg1_schedule_is_not_blind` | the step's B / deltas / M_FIX through the gate's parse_args; SG1 coverage by `simulate_schedule` |
| `test_base_is_the_pinned_piece3_blob` | `git hash-object` = 39ed5e18 |
| `test_jit_bodies_resolve_their_globals_in_the_module[...]` | L36 lint (+ negative: a factory-local name is flagged) |
| `test_shared_source_keeps_piece3_constants_and_no_host_read` | constants, the combine launch, no host read, the step lines equal piece 3's |

## 5. GPU cells (SOURCE ONLY; step `benchmarks/Efficiency/r2_sharedkv_step.sh`, a copy of `r2_tick_step.sh`)

Batch: the brief asks for 16K B 64. That does NOT fit `r2_tick_gate.py`: the ResidentStore holds every block of every
layer, (NB + 4) x 64 rows x Hkv 2 x D 128 x 2 B x (K, V) x 32 layers = about 34.6 GB at B 64, and each part runs on a
CLONE of BASE (BASE + clone = 69 GB, + the 16.4 GB model). At B 32 it is 34.6 + 16.4 GB (and BASE + the REF cache
during REF, about 51 GB). The ratios measured here are per request (a group never mixes requests), so B enters only
through occupancy. Cells:

* S1-probe (B 32, 16K, `--attn probe`, `--parts SG1`, `--draft-drop 0.0`: full draft selections as in the 4.5 harness
  and the forecast): every keyed call of every layer of every SG1 tick runs both kernels.
* S1-probe-whole: the same with the whole-union schedule.
* S1-plant (`--attn probe-plant`): the negative control in the tick.
* S1-shared and S1-shared-whole (`--parts SG0,SG1,SG2,SG3`, `--draft-drop 0.5`): the shared kernel's output drives the
  tick; SG1-SG3 gate it bitwise against the BI serial REF (the existing MUST gates).
* S2 (B 64, NOT scripted): needs either (a) `ResidencyBackend.page_keys` (C: a device block-id tensor in the plan) to
  probe inside the SG5 / captured tick, or (b) a gate variant that does not clone BASE. Run S2 only after S1 passes.
* S3 (ncu, NOT scripted): `dram__bytes_read.sum` of `_bi_split_kernel` vs `_bi_shared_kernel` on one verify tick, to
  separate HBM bytes from logical tile loads (see P4).

### Registered predictions (written before any run; the gate prints IN / OUT for P2 / P3)

[sharedkv2] P2's fixed tile bands, P3's kernel-only time band, F3, F4 and F5 below are REVISED by section 8 (the
review of f79d0df, `kstep-cpu-runs/paper-notes/review-sharedkv-f79d0df.md`); the original text stays for the record.

* P1 (MUST, exactness). S1-probe and S1-probe-whole: 0 differing rows (int views) over all keyed calls;
  `SKV_shared_kernel_ran`. S1-shared / S1-shared-whole: every SG0-SG3 MUST gate PASS. S1-plant: diff rows > 0.
* P2 (tiles; the kernel's own counter over all keyed calls; shared / per-row = sum of num_pages). Basis: the 4.5
  harness at b64 measured union-of-4-rows / sum = 0.2820 (U 69.1 + draft 63 - I 61 over 4 x 63), verify-only
  U / sum = 69 / 189 = 0.365.
  whole: central 0.29, band [0.24, 0.36]. split: central 0.45, band [0.30, 0.60] (each shared page sits in about 1.6
  distinct split indices across the 4 rows, because the rows' list positions of a shared page differ by the count of
  non-shared pages before it, a few pages against a split of 8; this factor is an estimate, not a measurement).
* P3 (time per keyed call with rows_max 4, shared kernel + combine over piece 3's split + combine; union build
  reported separately). Basis: per-row v2 is NOT a pure SUM reader: the v2 gate measured 1.31 / 1.50 / 2.25 x FA2 at
  1 + U = 1 / 2 / 4 (B 64), i.e. 4 rows cost 1.72 x 1 row, not 4 x (adjacent programs of one request hit L2). With an
  effective per-row DRAM of about 1.72 / 4 = 0.43 of the sum and the harness's union DRAM 0.363, the bandwidth-bound
  ratio is about 0.85 for whole-like loads; split loads more (P2) but keeps piece 3's parallelism.
  split: central 0.80, band [0.55, 1.10]. whole: central 1.6, band [0.80, 3.50] (only R/4 x Hkv = 64 programs at
  B 32; piece 3 v1 with 128 programs was 8.8 x FA2).
* P4 (consequence for the paper; REPORT). The forecast's V = (verify-tick tiles per request and head) - 63. Break-even
  V <= 56.4 means tiles / (4 x 64) <= (63 + 56.4) / 256 = 0.466. split's central 0.45 sits AT the break-even; whole
  (0.29) is far below it but slow (P3). The logical tile ratio is NOT the HBM ratio: S3 must measure DRAM bytes before
  the forecast row is rewritten.

### Falsifiers

* F1 any differing row in S1-probe / S1-probe-whole, or any SG1-SG3 MUST failure in S1-shared: exactness refuted for
  this design. First split by kind: sign-of-zero only (section 3) vs a real value difference. A real difference means
  the compiler treats the per-row step differently inside this kernel: fallback = one launch of piece 3's kernel per
  group with an L2-friendly program order (rows of a group adjacent, same split), no shared load, and S3 to measure
  how much L2 already dedups.
* F2 S1-plant shows 0 differing rows: the comparison is BLIND; P1 is void.
* F3 `SKV_shared_kernel_ran` false or keyed calls with rows_max 4 absent: the cell did not test the kernel.
* F4 P2 outside its band: the byte model (builder or schedule) is wrong; check the counter against the CPU oracle on
  the run's own tables.
* F5 split time ratio > 1.10: sharing does not pay at this schedule (duplication + register pressure of 4 rows'
  accumulators: 4 x (16 x 128) fp32 + 4 q tiles per program, possible spills). Then try rows_max 2 (two half-groups)
  or the whole schedule with more programs per group (a split over KV heads only is already used).
* F6 whole time ratio < 0.80: occupancy is not the bottleneck; whole becomes the default candidate.

## 6. Step script changes (SOURCE ONLY)

`benchmarks/Efficiency/r2_sharedkv_step.sh` = `r2_tick_step.sh` with: NOSI = this worktree; OUTROOT
`/mnt/beegfs/ojerbi/r2-runs/sharedkv`; claim `skv_active`; a 5th argument ARM (probe | probe-whole | probe-plant |
shared | shared-whole) that selects `--parts` / `--draft-drop`; B 32 with 32 deltas, M_FIX 128; STEP_MIN 90; it also
checks that this tree's `nosi/nosi/bi_paged_attention.py` IS the pinned blob and that the shared module is tracked;
it passes `--attn "$ARM" --sharedkv "$NOSI/nosi/nosi/bi_paged_attention_shared.py"`. Order: S1-probe, S1-plant,
S1-probe-whole, S1-shared, S1-shared-whole; stop at the first MUST failure. [sharedkv2] SUPERSEDED by section 8:
the order is per_row, probe-plant, probe, probe-whole, shared, shared-whole, and the shared module is a frozen copy.

## 7. Merge notes (the parallel X-pool branch, `nosi-r2-xpool`)

Piece 3's file is unchanged here. The shared kernel copies piece 3's address arithmetic (module lines 291-294 bases,
311-314 tile rows and loads, 198 q, 236 / 239 workspace, at f79d0df; [sharedkv2] at the review-fix commit + 4:
295-298, 315-318, 202, 240 / 243). When the X-pool's flat-position mode lands in piece 3's kernel, the same constexpr
change is needed at those lines of `_bi_shared_kernel` (a follow-up after both gates).

## 8. [sharedkv2] Review fixes (review of f79d0df; SOURCE ONLY, not run)

* MAJOR-1 (P2, F4). No fixed tile band. The probe records per keyed call the GROUP MIX on the device (per group size
  s = 1 .. 4: groups, their per-row pages, their union pages; `report()["mix_all"]`, `["mix_full"]`, `["per_call"]`),
  and the inputs of the first 4 keyed calls with 4 rows (`report_tables()`; the gate saves `skv_tables.pt`: F4's
  offline check against the CPU twin). REGISTERED FORMULA (`bi_paged_attention_shared.RHO`, `expected_tile_ratio`):
  expected = sum_s RHO[mode][s] x pages_s / sum_s pages_s, with RHO whole {1: 1.00, 2: 0.55, 3: 0.40, 4: 0.29},
  split {1: 1.00, 2: 0.70, 3: 0.55, 4: 0.45}; the measured tile ratio must lie in [0.80, 1.25] x expected
  (`R_SKV_tile_ratio`, which also prints the measured union / sum per size: the direct test of RHO). The counter's
  identities are MUST (`SKV_tile_identity`): whole tiles = the union pages; split union <= tiles <= sum.
* MAJOR-2 (P3, F5). The shared side's timed span includes the union build (NVTX `bd` in the S3 window too).
  REGISTERED with the build: split central 1.30 [0.80, 2.20], whole 2.10 [1.00, 4.00] (the build is about 50 small
  ops per call; at B 32 the kernel is of order 0.1 ms, so we expect the build to cost about as much as the sharing
  saves). F5 now reads: (build + shared) / per_row > 1.10 (split) = the selector does not pay in the tick at this
  schedule; the follow-up is to hoist `grow` (seq_idx only) to once per tick. The kernel-only band of P3 stays as a
  REPORT.
* MAJOR-3. The primary time ratio is the S3 cell's DEVICE kernel time (ncu `gpu__time_duration`, cache-control none
  = the tick's L2 state; `R_S3_device_time_none`, docs/R2_SHAREDKV2_KEYS_S3.md). The gate's CUDA-event ratio
  (`R_SKV_time_ratio`) is REPORT only; the probe now pre-queues the GPU (`torch.cuda._sleep`) so the events time the
  device, and the uncounted launch passes a persistent dummy counter (no `torch.zeros` launch per call).
* MAJOR-4. `test_step_dryrun_refusals_and_pins` runs the step in DRYRUN: usage, the uuid-file rules, TICK_HOLDERS and
  the new TICK_PIN_* overrides refused outside DRYRUN, a wrong --nosi, the per-arm PARTS / DROP / --attn on a clean
  checkout, and both piece-3 blob pins (the second with an older piece 3 blob from git log, c866d7c).
* MINOR-1. The -0.0 exception goes both ways (module docstring corrected).
* MINOR-2 (F1 triage). The probe counts differing ELEMENTS as sign-of-zero only (floats equal) or value, and keeps the
  first differing element per call (flat index, both bit patterns); the bitwise gate prints them.
* MINOR-3 (F3). MUST `SKV_rows_max4_calls`: keyed calls with 4 rows of one request exist (`by_rows_max`).
* MINOR-5. `--attn shared` / `shared-whole` refuse `--no-check` (parse_args).
* MINOR-7. The step freezes the shared module into the record (`git show`, hash checked) and the gate loads that copy.
* MINOR-9 / MINOR-11. Arm order: `per_row` (piece 3 only, SG1, B 32: the miniature that checks memory and SG1
  coverage at this batch), `probe-plant`, `probe`, `probe-whole`, `shared`, `shared-whole`; stop at the first MUST
  failure; a probe result without a passed probe-plant is VOID.
* MINOR-12. `test_negative_drop_one_union_page_is_caught` asserts that the DEFAULT plant (the last union page, the GPU
  variant) changes the output on the CPU data, only on the listing rows.
* NOSI in `r2_sharedkv_step.sh` is now `/mnt/beegfs/ojerbi/nosi-r2-sharedkv2` (the fixes live there).

New / changed CPU nodes (`benchmarks/Efficiency/tests_cpu/test_bi_paged_attention_shared.py`):
`test_probe_records_the_group_mix_and_the_expectation`, `test_probe_triages_sign_of_zero_and_value_differences`,
`test_probe_times_the_build_with_the_shared_side_and_launches_no_extra_zeros`, `test_step_dryrun_refusals_and_pins`;
changed: `test_negative_drop_one_union_page_is_caught[split]` / `[whole]` (MINOR-12),
`test_gate_selector_default_is_piece3_and_choices` (MINOR-3, MINOR-5).
