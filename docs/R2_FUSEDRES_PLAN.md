# R2 fused residency resolve (b2): plan, interface, registered prediction

Branch `r2-fusedres2` (worktree /mnt/beegfs/ojerbi/nosi-r2-fusedres2), from 0e28cf6 (r2-fusedres, base cf35536).
SOURCE ONLY, NOT RUN. Labels: MEASURED (a cited run), MODEL (a derivation in this file), MISSING.
fusedres2 = 0e28cf6 + the fixes of paper-notes/review-fusedres-0e28cf6.md (M1-M5, m1-m5, m8, m9; section 10).

## 1. The problem

- Per draft tick at 16K B 64 (FAST, S-C, eager), the residency decision kernels `res:resolve` take **37.62 ms in 9,774
  kernels** (MEASURED: analysis/bound_4_2/other_res_r1/other_res.md, "S 16128 B 64 res:resolve, G_d"). A draft
  forward takes 26.86 ms (paper-notes/recon-draft-tick.md section 1).
- Break-even at B 287 needs twin + res + end-of-tick records <= 9-16 ms in total (recon-draft-tick.md section 4.3).
- No existing build removes res. b2 (OwnResidency) reuses S-C's device path unchanged.

## 2. Prior art (what exists, grep-checked 2026-10-21)

| item | where | form |
|---|---|---|
| S-C per-layer resolve | nosi/nosi/r2_residency_sc.py `ResidencySC.resolve`, `DeviceMirror.layer`, `device_verify` | torch ops: 2 segmented sorts per layer for the mirror ranks, stable sorts in `rank_in_group` (cub onesweep) and `cummax`, `index` / `index_put_` / `index_add_` scatters |
| S-B draft decision | r2_residency_sb.py `device_draft`, `rank_in_group` | the same torch op families |
| C's sparse record | r2_residency_sparse.py `sparse_layer_record`, `_compact`, `_pack_bits`, `draft_bias_fill` | torch ops; the bias fill allocates and rewrites a whole (B, rows, H) buffer per layer |
| B2 mirror | r2_own2.py `OwnResidency.sync_mirror` | table -> dmirror copy, once per tick (kept) |
| Triton kernels in the R2 line | r2_twin_fused.py, r2_residency_sc.py (exact-wait board kernel), r2_xpool.py (h2d entries, xpool branch) | none decides residency |
| nosi-r2-m2-own2b (3fbf8a7), nosi-r2-m2-sparse (9508109), nosi-r2-sharedkv, nosi-r2-xpool (44d5e95) | same files | no fused resolve; xpool binds lanes AFTER the torch decisions |

The kernel names of the 37.62 ms (other_res.md, G_d at B 64): radix sort onesweep 6.91 ms / 768 launches,
radixSortKVInPlace 4.50 / 160, cummax scan 2.25 / 64, index 2.02 / 292, copies 1.89 / 514, index_put 1.55 / 262,
sort memsets 1.33 / 960, and / where / fill / clone. 9,774 / 32 = **305 kernels per layer**, 3.85 us each on average:
launch-count-bound, not byte-bound.

## 3. Design (nosi/nosi/r2_fusedres.py)

`OwnResidency(..., resolve="torch" | "fused" | "check", fused_plant=0)`. "torch" (default) is unchanged.
SG5: `r2_tick_sg5.py --b2-resolve torch|fused|check`, `--b2-fused-plant 0|1|2` (hidden, FR-g0p), `--b2-res-profile`.

Per layer, in S-C's order, grid S = B x H programs (one per stream) except SCAN (one program):

| launch | when | does |
|---|---|---|
| V1 | the layer has verify rows | the verify rows' classification; per (row, head) the K5 extra-need count and the record count; per stream the usage; program 0 zeroes V2's sums |
| SCAN | idem | the exclusive prefixes of V1's counts, the layer's verify budget X - borrowed, the record total |
| V2 | idem | K5 grants (ONE base per (row, head) from SCAN + the in-row prefix), the q-th FREE slot, Policy E (`tl.sort` of the stream's candidate keys), read slots, page tables, every decision field, load ids, verify record / looks / read bitmap, the post-verify stream state; self-check of its counts against V1's (err bit 16) |
| (S-C's device wait and urgent gather) | idem | unchanged |
| D1 | the layer has draft rows | the draft rows' classification on the post-verify state; counts; usage; program 0 zeroes D2's sums |
| SCAN | idem | D1's prefixes, the draft budget X - borrowed - R_ver, the record total |
| D2 | idem | K5 grants, take slots, page tables, decision fields, the draft record, the bias rows (bit copy, or an in-kernel cast if the dtypes differ), counters, `sel` / `rowsel`; self-check against D1 (bit 16) |

So a draft-only layer is **3 launches**, a layer with verify and draft rows **6**. Per tick (first layer only): B2's
`sync_mirror` (torch, unchanged), `_tick_rows` (one host-to-device copy, unchanged), ONE more host-to-device copy (the
per-batch-slot row lists), the err bookkeeping (2 ops) and, once per tick and width, the verify entries' (b, h). No host
read in the per-layer path (test FR4).

Exactness: every output equals the torch path's (section 6). Assumptions A1-A3 are in the module docstring; the kernels
raise the watched err flag when A1 / A3 / the stamp range / the phase self-check fail (the tick is void). A2 ties
(equal candidate stamps in one stream) are NOT flagged: B2's stamps are fresh clock values (r2_own2_ops.py:398, 788,
848), both picks are legal, and a tie would show as a check-mode difference.

Limits (refused): pin_records True (B1), record_cap set, any sc_mut / sb_mut / sparse_mut negative control, a subclass
that overrides `_layer_record`, C + X > 512, an attached xpool (section 5), non-contiguous / non-int64 mirror or row
tensors (L28, checked once per tick), non-contiguous or non-(B, rows, H) bias tables. Counted, not refused: a
non-int64 / non-contiguous `sel` (one copy per layer: `sel_conversions`), a bias dtype cast (`bias_casts`; 0 at p5b2:
r2_tick.py from_r2 makes the twins (468) and store_bias (477-480) with src.ring.dtype, so cis and store_b agree).

## 4. The slot-allocation interface the kernels assume (for nosi-r2-xpool)

The kernels decide on LOGICAL slots, as S-C does:

- I1 Per (layer, b, h) stream: NPOS = C + X slot indices 0 .. NPOS - 1; the mirror tables
  `dmirror.{code, block, gen, cur, prot, stamp}` are (L, B, H, NPOS) int64, contiguous; the kernels read row
  `[l, b, h, :]` at flat offset ((b H + h) NPOS) of `table[l]`.
- I2 A K5 grant takes the q-th FREE slot of its stream in ascending slot index; q counts the stream's earlier grants
  of the layer (verify rows first, then draft rows on the post-verify FREE set).
- I3 K5: a stream's first (C - usage) first misses are local; later ones take a shared extra; the shared extras of a
  layer number X - borrowed (- R_ver for drafts), borrowed = sum over streams of max(usage - C, 0); grants go in K3X
  entry order (row, head, ascending block).
- I4 Policy E evicts a slot of the SAME stream (LRU by stamp, then the owner's in-flight prefetch by take stamp).
- I5 Outputs name logical slots: `VerifyDecision.load_ids` (H, B, NPOS), `t["take_slot"]`, the page tables, the
  records. The bias rows of the draft takes are written at store_b[l][b, take_slot x 64 + r, h] (D2) - the ONLY
  physical write of the fused path.

## 5. Merge with nosi-r2-xpool (44d5e95)

xpool keeps logical slots and binds lanes after the decisions (`XPool.release_free` after sync_mirror, `XPool.bind`
after device_verify and after device_draft, `draft_bias_fill_flat` instead of `draft_bias_fill`). To merge, in
`FusedResolver`: (1) first layer of a tick: `svc.xpool.release_free(svc.dmirror.code == C_FREE)` after `sync_mirror`;
(2) after V2: `svc.xpool.bind(l, vdec.bh[0], vdec.bh[1], vdec.t["take_slot"], (granted | evict) != 0)` before the
wait; (3) D2 with `BIAS=0`, then `svc.xpool.bind(...)` on the draft takes and `XP.draft_bias_fill_flat(...)`; (4) drop
the xpool refusal in `FusedResolver.__init__`. That costs xpool's own launches on top of 3 / 6.
[integ] Implemented on branch r2-integ (docs/R2_INTEG_PLAN.md section 2): steps (1)-(4), check mode on the flat bias
store, and S-C's [sharedkv2] page-key hooks in `FusedResolver.resolve`. Not run.

## 6. CPU tests (benchmarks/Efficiency/tests_cpu/test_r2_fusedres.py; Triton interpreter in a fresh process, L36)

| node | checks |
|---|---|
| `test_FR0_the_default_is_torch_and_bad_modes_are_refused` | default "torch"; bad mode / plant / negative controls / record_cap / `_layer_record` override refused |
| `test_FR1_fused_equals_torch_at_every_layer_and_b2_equals_v1[f12b\|f20b\|f21b\|f6]` | every tick and layer every output equal; B2 on fused decisions == v1; err 0; no sel copy, no bias cast; preconditions per schedule |
| `test_FR1_the_lockstep_schedules_reach_the_verify_dedup` | v_dup > 0 on a lockstep schedule; no lockstep tick overflows |
| `test_FR2_each_plant_is_caught_at_its_first_effective_layer[1\|2]` | L34 plants; first difference == first effective (tick, layer), also through the report's `first_diff_at` / `first_effective` (FR-g0p's scoring fields) |
| `test_FR2_the_stats_trace_is_cumulative_and_matches_the_report` | the counters are consistent |
| `test_FR3_three_launches_per_draft_only_layer_six_with_verify_rows_and_no_torch_compute` | 3 / 6 launches; no torch compute op on later draft-only layers |
| `test_FR4_the_per_layer_path_has_no_host_read_and_a_planted_one_is_found` | static |
| `test_FR4_every_triton_function_resolves_its_names_in_its_module` | L36 lint incl. `_fr_scan` |
| `test_FR5_a_free_running_round_on_b2_check_equals_torch` | one tick-level round, check == torch |
| `test_FR6a_overflow_dup_behind_overflow_and_an_in_row_duplicate` | M5 hand case: v_over, v_dup_over, sel_dup > 0 asserted first; equal at both layers; then CapacityRefusal |
| `test_FR6b_two_draft_rows_of_one_batch_slot` | M5 hand case, ONE tick: 2 draft rows of one b, d_dup > 0 first; equal at both layers; then the SAME B2 refusal (d_merge) in torch and check mode; proof that production never builds the case: r2_tick.py RoundState.finish_tick (228-230) refuses "two draft rows" (source pin) |
| `test_FR7_the_shared_fixtures_in_check_mode` | every C fixture: B2 (fused) == reference; fused == torch; void_overflow reaches v_over |
| `test_FR8_the_kernel_time_parser_and_its_negative_control` | M3 parser on synthetic Kineto traces; a launch moved out of t.res is not counted; orphans counted; no t.tick refused |
| `test_FR9_the_sg5_flags_for_the_plant_and_the_profile` | the two SG5 flags and their wiring |
| `test_FR10_no_kernel_writes_past_a_buffer_and_the_guard_detector_works` | fusedres2b: guard mode (sentinels after every kernel-written buffer, checked after EVERY launch) on f20b / f21b / f6 / f12b; no hit; negative control: a changed guard is named |

Also re-run test_r2_own2.py, test_r2_tick_b2.py (TB5 pins the SG5 b2 lines) and test_r2_residency.py F23.

## 7. REGISTERED PREDICTION (2026-10-21, before any run; supersedes 0e28cf6's)

Cell **FR-g** at p5b2's point: r2_tick_sg5.py with p5b2's arguments (`--modes exact,fast --backends b2 --C 96 --X 64
--R-ver 16`, pool 100, gilfree, threads 8, unchecked), one A100 holder, invocations in this order. TRITON_CACHE_DIR is
set per job by the step launcher (r2_tick_sg5_step.sh:191, C1).

| arm | flags | role |
|---|---|---|
| FR-g0 | `--b2-resolve check --draft-ticks 4 --steps 48` | GPU equality gate with coverage (M1) |
| FR-g0x | FR-g0 + `--C 24 --X 8 --R-ver 4` | pressure arm: the counts p5b2's point may not reach |
| FR-g0p | `--b2-resolve check --b2-fused-plant 2 --draft-ticks 4 --steps 8` | GPU negative control (M2) |
| FR-g1 | `--b2-resolve torch` | same-holder control (re-measures p5b2) |
| FR-g2 | `--b2-resolve fused` | event-span timing arm |
| FR-g3 | `--b2-resolve fused --b2-res-profile --draft-ticks 12` | KERNEL time of res (M3; its spans are not timings) |
| FR-g3t | `--b2-resolve torch --b2-res-profile --draft-ticks 12` | the same for the torch path (b2) |

GATES (decided before any timing claim):
- G0 (FR-g0, FR-g0x): in `draft_only.fused` and `residency.report.b2.fused`: diffs == 0, compared == layers,
  err == 0 (err_any: every tick's bits), no CapacityRefusal naming fused_res_err. COVERAGE: each of
  `v_evict, pe_flip_effective, v_grant, d_grant, d_refused` (report "gate") must be > 0 in FR-g0 or FR-g0x; a
  counter at 0 in both is reported "path NOT GATED on the GPU", never as a pass.
- G0p (FR-g0p): `draft_only.fused.stats.d_grant > 0` (precondition) and diffs > 0 with
  `first_diff_at == first_effective["d_grant"]` (the CPU rule FR2). This also exercises the GPU atomics behind stats.
- G2 (FR-g2): `draft_only.fused.launches_per_layer == 3.0`, `sel_conversions == 0`, `bias_casts == 0`; the compiled
  variants (`compiled`) are reported. The draft-only stats skip tick 0 (compiles there). The ROUND of FR-g2 compiles the
  verify kernels at its first verify tick inside its mean: its T_tick and ratio are REPORT only, not results.

Geometry: L 32, B 64, H 8, NPOS 160 (tile 256), k 64, nblk ~253 (tile 256), one draft row per request.

| quantity | p5b2 / profile (MEASURED) | prediction (MODEL) | measured by |
|---|---|---|---|
| decision equality on the GPU | - | **MUST: 0 differences** with the coverage of G0 | FR-g0 / FR-g0x |
| live negative control | - | **MUST: caught at its first effective layer** | FR-g0p |
| launches per draft-only layer | ~305 kernels | **3.0** Triton launches | FR-g2 `launches_per_layer` |
| res device events per draft tick | 9,774 (S-C) | **<= 140** (96 fused + B2's per-tick sync_mirror ops and copies) | FR-g3 `draft_only_res_profile.res.n_stats.median` |
| res KERNEL ms per draft tick, B 64, FAST = EXACT | 37.62 (S-C) | **0.8 - 6.0 ms**, point 2.5 | FR-g3 `...res.ms_stats.median` |
| res KERNEL ms, torch path on b2 | 37.62 (S-C) | 30 - 50 ms, >= 8,000 events | FR-g3t |
| res EVENT span per draft-only tick (median), eager | 129.01 FAST / 128.76 EXACT | **4 - 25 ms** both modes | FR-g2 `draft_only.device_ms.res` |
| res event span, FR-g1 control | - | within 15% of 129 ms | FR-g1 |

Derivation (MODEL):
- Bytes: per layer each kernel reads the layer's mirror (6 x 512 x 160 x 8 B = 3.9 MB); D2 writes the 9 decision fields
  (2.4 MB), the record (1.3 MB) and the pages (0.4 MB): about 12 MB per layer, 0.26 ms per tick at 1.5 TB/s; with B2's
  per-tick table copy (~250 MB, ~0.2 ms) a FLOOR of about 0.5 ms.
- Compute: per stream program the register tiles are (64 x 256) lookups (2 in D1, 3 in D2), (64 x 64) sorts, a
  (128 x 64) page compaction and a (256 x 64) bitmap: about 0.21 M (D1) + 0.34 M (D2) element ops, x 512 programs =
  0.28 G per layer; at 2 - 8 T element ops/s effective, 35 - 140 us per layer. SCAN: one program, R x H = 512 counts
  (one chunk) + 512 usages, about 3 - 6 us. Per tick 32 x (40 - 150 us) = **1.3 - 4.8 ms**, + sync_mirror and copies
  (0.2 - 0.5 ms): band rounded to 0.8 - 6.0 ms.
- Event span: host-bound eager tick; per layer about 15 allocations (~5 us), 3 Triton launches (~30-60 us each in
  Triton 3.2's launcher), the rows check and the prologue: 170 - 330 us per layer, 5.5 - 10.5 ms per tick, + the
  tick's first layer (~1 ms); band 4 - 25 ms.
- B 287 (MODEL, break-even context). 0e28cf6 had every phase-2 program read the whole (R x H) count and usage vectors:
  O(S x R x H) reads, 300 MB of L2 per verify V2 launch at B 287 (review M4). fusedres2 removes that term: each program
  loads ONE base; SCAN reads R x H + S integers once (B 287 draft-only: 2,296 + 2,296, 3 chunks, about 5 - 10 us; a
  verify tick with Rv about 4B: 9,200 counts, 9 chunks, about 10 - 20 us). The per-program work does not depend on B.
  So the kernel ms scale with the programs (x 4.5): D1 + D2 160 - 630 us per layer + SCAN, **5.5 - 21 ms per tick**
  (+ the table copy, x 4.5: 0.9 - 2 ms). The low end is inside the 9 - 16 ms budget only together with a fused twin
  and b2's record; the high end is not. If FR-g3 lands in the upper half at B 64, the tile work (section 9) comes
  before any B 287 claim.

Falsifiers (each stops the claim it names):
- F1 any G0 difference, compared != layers, or a NOT GATED path without an explicit statement: no exactness claim for
  that path on the GPU (L36).
- F1p G0p fails (no difference, or the first difference is not at the first effective layer): the GPU check is blind;
  FR-g0's PASS is void.
- F2 err_any != 0 (report "err") or a CapacityRefusal naming fused_res_err in any arm: an invariant fails on real data.
  STOP.
- F3 a compile failure of any of the five kernels on the A100: STOP (L36).
- F4 G2 fails (launches_per_layer != 3.0, sel_conversions > 0 or bias_casts > 0): the timing is not of the design.
- F5 FR-g3 res kernel ms > 6.0 at B 64: the tile cost model is wrong (spills / reductions); re-tile before B 287.
- F6 FR-g2 res event median > 25 ms: the host side dominates; next is the captured tick (TC-R) with the fused resolve.
- F7 FR-g1 not within 15% of 129 ms: the holder differs from p5b2's; compare FR-g2 with FR-g1 only.

## 8. Kernel-time measurement (M3: built)

`--b2-resolve ... --b2-res-profile` runs the draft-only phase under torch.profiler (CPU + CUDA) with every TickTimer
span also a `record_function("t.<cat>")`, writes `<out>/<label>_draft_only_trace.json` and parses it with
benchmarks/Efficiency/r2_fusedres_prof.py (`parse_res_trace`: a device event belongs to t.res when its launch
(cuLaunchKernel / cudaLaunchKernel / memcpy, by correlation id) lies inside a t.res range of the launching thread; the
tick is the enclosing t.tick range). Output `draft_only_res_profile.res` / `.tick`: per tick ms and event counts,
stats without tick 0, per kernel name totals, orphans. The profiled phase's event spans are NOT timings (profiler
overhead), hence FR-g3 is its own arm.

## 9. Doubts and next steps

- Not run anywhere. The first CPU run may show interpreter-level errors (dynamic `range` bounds from loaded values,
  `tl.sort` on int64, scalar atomics, `triton.jit(do_not_specialize=...)` in the interpreter, the pointer dtype in
  the BIAS 2 cast).
- Tile size: (64 x 256) int32 tiles in 8 warps may spill (review m11). If F5 fires: split the lookup over NPOS chunks
  of 64, or sort the stream's slot blocks once per program and use a merge lookup.
- `stats` are compared in no CPU check against a torch twin (they have none); FR2 / FR6 / G0p validate them by effect.
- The kernel-time harness attributes by launch correlation; Kineto event categories (`cuda_driver` for Triton) were
  not observed on this installation (FR-g3 will show orphans / zero counts if they differ: a zero res event count with
  nonzero t.res ranges is a harness failure, not a result).
- The check mode compares on the same inputs per layer; it does not compare a whole GPU run against a torch run (the
  transfer timing differs between runs, p5b2 scoring section 0).

## 10. Review fixes in fusedres2 (paper-notes/review-fusedres-0e28cf6.md)

| item | fix |
|---|---|
| M1 | FR-g0 runs a 48-step round (verify rows) plus the pressure arm FR-g0x; the report's `gate` marks each coverage counter "gated" / "NOT GATED"; G0 requires them |
| M2 | `--b2-fused-plant` (hidden) and FR-g0p; the report carries `first_diff_at` and `first_effective` (trace-based), the CPU FR2 asserts the same fields |
| M3 | the harness exists (`--b2-res-profile`, r2_fusedres_prof.py, FR8); FR-g3 / FR-g3t registered; F5 measurable |
| M4 | one SCAN launch per phase; phase 2 loads one base; the B 287 model updated (section 7) |
| M5 | FR6a (overflow, dup behind overflow, in-row duplicate), FR6b (two draft rows of one b), FR7 (C's fixtures incl. void_overflow), FR1 aggregate v_dup; new stats v_dup_over, d_dup, sel_dup; preconditions asserted before the equality claims |
| m1 | `err_any` (sticky REPORT) next to the per-tick `err`; F2 reworded |
| m2 | contiguity / dtype / device type of every flat-addressed tensor checked once per tick |
| m3 | `sel_conversions` counted, gated 0 in G2 |
| m4 | `do_not_specialize` on every int argument (no re-compile when a count crosses 1 or 16); compiled variants reported; FR-g2's round marked REPORT; TRITON_CACHE_DIR from the launcher |
| m5 | phase 2 re-checks its counts against phase 1's (err bit 16) |
| m6 / m7 | ties and the A3 scratch effects documented (module docstring) |
| m8 | `sel` validated in int64 before the int32 cast |
| m9 | dtypes confirmed equal at p5b2 (code facts above); an in-kernel cast (BIAS 2) replaces the torch fallback |
| m10 | unchanged: stats validated by FR2 / FR6 on CPU and G0p on the GPU |
| m11 | unchanged (section 9) |

### fusedres2b: the interpreter heap corruption of d2be560 (job 2191246)

- Symptom: every interpreter child (lockstep, tick) died with SIGABRT, glibc "corrupted size vs. prev_size"; 0e28cf6
  passed the same tests. Under TRITON_INTERPRET the kernels write straight into the CPU tensors' storage, so a write
  past a buffer's end corrupts the heap.
- Diff 0e28cf6 -> d2be560, kernel stores whose lanes reach past their buffer (static analysis; not confirmed by a
  run). In 0e28cf6 the largest such overshoot was a few elements:
  - r2_fusedres.py:255-256 (d2be560), the new SCAN chunk stores: 1,024 lanes on a buffer of R x H elements (8-64 in
    the tests);
  - :287, the new vlook zeroing: 256 lanes on Rv x 4;
  - :541 and :754, the stats atomics: 16 lanes on an 11-element buffer.
  These lanes are masked. The most likely cause is a masked lane that the interpreter's store / atomic path still
  touched far past the allocation. The pinpoint is open: FR10 names the kernel and the buffer if any write still
  lands past an end.
- Fix by construction:
  - every buffer a kernel writes is allocated with LANE_PAD = 2,048 spare elements (`FusedResolver._buf`; more than
    any lane overshoot);
  - the stats buffer holds all 16 atomic lanes; the draft counters hold all 4;
  - the SCAN chunk is pow2(max(R H, S)) capped at 1,024, the vlook chunk pow2(Rv x 4) capped at 256;
  - a masked bias lane clamps its slot to 0, so its address stays inside the table.
- The detector: OwnResidency(fused_guard=True) fills the spare elements with a sentinel and checks every guard after
  every launch (`_check_guards`); FR10 runs four schedules under it, plus a negative control.

### fusedres2c: the guard hits of 7727e34 (job 2191259)

- FR10 found a real out-of-bounds write: `_fr_scan.d` wrote past `d.bne` and `d.brec` at every (tick, layer) of
  f20b (101 hits). The sites, in 7727e34's r2_fusedres.py: the scan's int32 stores of a cumsum (255-256) and its
  int32 misc store (266).
- Masks and sizes were right (one chunk: RH = 4 lanes on 4 elements, no masked lane) and the values were right. The
  cause (INFERENCE, consistent with every observation): the interpreter keeps a scan / sum of int32 values as
  8-byte numpy data under the declared int32 type. An int32 store then writes 8 bytes per element; the last lane
  spills 4 bytes past the end, and little-endian overlap keeps the values right. On the GPU the values are real
  int32, so this exact spill is interpreter-only. The fix does not rely on that argument.
- Fix: every scratch buffer is int64 (ne, rec, bne, brec, misc, usg, usg2, usgd, post, err, err_any); every stored
  value is cast to int64 explicitly (a real cast from a declared int32); `_put1` casts to its destination's element
  type. The only int32 kernel outputs left are the plan's page table and page counts, stored from int64 values by a
  real cast. The padding stays as defence; FR10's gate is 0 hits.
- FR6b: ONE tick, decision equality, then the same B2 refusal in both modes, plus the production proof (above).
- FR4: the lint names the `_go` call form of all six launches.
