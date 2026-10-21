# R2 shared ready-KV reads, part 3: a cheaper union build (SOURCE ONLY, NOT RUN)

Branch `r2-sharedkv3`, worktree `/mnt/beegfs/ojerbi/nosi-r2-sharedkv3`, base `83bc239` (r2-sharedkv2). No CPU test
and no GPU cell ran (no interpreter on the login node).

## 1. Why

sharedkv2 registered that the shared side, with `build_union` charged to it, is SLOWER than piece 3 in the tick
(split central 1.30, F5 expected to fire): `build_union` is about 50 small device ops per keyed call, 32 layers per
tick, against an attention kernel of order 0.1 ms at B 32.

## 2. (a) What depends on what

| step of `build_union` | depends on | ops (approx.) | now |
|---|---|---|---|
| groups: sort of seq_idx, newg, cummax, m, gid, grow scatter | seq_idx, rows_max | 12 | `build_groups`, ONCE PER TICK (the selector caches it per TICK TOKEN: the tick passes `group_token=ctx`, its per-tick context object, to every layer; plus the seq_idx object and rows_max; no token = rebuilt every call. Fix 2: the first version keyed on the seq_idx version counter, which inference tensors do not have, CPU job 2191241) |
| entries (key << 32 / slot), gather by grow, sort, first, cumsum, ut / rowpos / uidx scatters | the layer's page_table, page_key, num_pages | about 25 | ONE kernel `_union_kernel` |
| windows: pad, view, amin / amax, where, stack | uidx | about 8 | in the same kernel |
| int32 casts / contiguous | - | 6 | the kernel writes int32 into `torch.empty` buffers (no launch) |

So per layer: 1 launch instead of about 50; per tick: + about 12 group ops on the first keyed layer.

## 3. (b) The kernel (`nosi/nosi/bi_paged_attention_shared.py`: `_union_entries`, `_union_kernel`, `build_union_fused`)

One program per (group g, KV head h), grid (R, Hkv) as the shared kernel. The N = NR x P entries of the group's rows
(v = key << 32 | slot, listed iff the row exists and i < num_pages) are walked in tiles of TE (64; 16 in a test):

* pass 0: ut and rowpos of (g, h) := -1; barrier;
* pass A: FIRST(f) = listed and no EARLIER row of the group lists the same v (a TE x TE comparison per tile pair),
  stored to a scratch buffer; barrier;
* pass B: uid(e) = the number of FIRST entries with a smaller v. Since every value has exactly one FIRST occurrence,
  uid is the number of DISTINCT values below v: the ascending, deduplicated order of build_union's sort. It writes
  uidx, ut[uid] (FIRST entries), rowpos[j, uid] = i, ucnt = the FIRST count; whole mode: win = (0, ucnt);
* pass C (split mode, after a barrier; section 7): per split k the stored uidx of positions [k PPS, (k + 1) PPS) of
  all rows, min / max + 1 by unrolled compare-selects (empty: (0, 0)), as build_union.

The outputs are integers, so "the same output" is exact equality, field by field. Cost per program: about
2 x (N / TE)^2 tiles of TE x TE int64 compares (N = 4 x 68 = 272: 25 tile pairs per pass), no sort.

Considered and NOT taken: building the union in the attention kernel's prologue. A split program (g, h, k) needs the
union order of the whole (g, h) list (its window bounds are global union indices), so each of the S split programs
would recompute the whole union (S x the work), and the whole schedule already has too few programs. One extra
launch of a few microseconds is cheaper.

The selector: `SharedKV(..., union="torch" | "fused")` (default torch: sharedkv2 unchanged);
`SharedKVProbe(..., union=...)`; the gate `--union torch|fused`; the S3 driver `--union`; step arms
`probe-fused`, `probe-plant-fused`, `shared-fused` (r2_sharedkv_step.sh) and `b32-split-fused`, `b64-split-fused`
(r2_sharedkv_ncu_step.sh). Both steps now pin NOSI = this worktree. With `impl="reference"` the fused union is its
torch twin `build_union_fused_reference` (the same algorithm in torch; the CPU tick).

## 4. (c) CPU tests (`benchmarks/Efficiency/tests_cpu/test_r2_sharedkv3_union.py`)

| node | what |
|---|---|
| `test_U1_twin_equals_build_union[...]` | 6 cases x {split, whole} x rows_max {4, 2}: build_groups = build_union's grow; the twin = build_union in every field, with and without the cached groups |
| `test_U2_dedup_plant_is_caught_after_its_precondition[...]` | dedup off: caught (ucnt differs) where the case shares a value between rows (precondition from build_union's own output); a NO-OP on the disjoint case, reported, not counted |
| `test_U3_selector_builds_the_groups_once_per_tick_token` | (9a501d8; under inference_mode) one build per (tick token, seq_idx object, rows_max); rebuild for another token, another seq_idx object with the same values, another rows_max, and on every call without a token. An IN-PLACE change of seq_idx under one token is NOT detected (no producer does it; the version-counter guard of a360d86 is gone); bits = piece 3's twin = the torch-union selector |
| `test_U4_fused_union_in_the_tick[split]`, `[whole]` | SG1 on the CPU tiny model with the fused union driving the tick = serial; group builds <= ticks and < shared calls; the fused probe 0 diffs |
| `test_U5_kernel_under_the_triton_interpreter` | the KERNEL = build_union in every field (all cases, both modes, rows_max 4 / 2, TE 16 and 64); the DEDUP plant caught after its precondition; SharedKV(union fused) kernel bits = piece 3's kernel, one group build |
| `test_U6_fused_path_source` | one launch, torch.empty only, no build_union in the fused path, three barriers (9a501d8) |

Changed: `test_bi_paged_attention_shared.py::test_step_script_shape_and_its_sg1_schedule_is_not_blind` (`--attn
"$ATTN" --union "$UNION"`), `::test_step_dryrun_refusals_and_pins` (the fused arms, NOSI sharedkv3),
`test_r2_sharedkv_ncu.py::test_N5_step_dryrun_refusals_and_pins` (the fused arm, NOSI sharedkv3). The L36 lint
(`test_jit_bodies_resolve_their_globals_in_the_module`) covers the two new jit functions.

Not provable on the CPU (L36): `tl.debug_barrier` orders the CTA's global writes on the GPU compiler as intended (the
interpreter runs a program serially, so a missing barrier would not show there). The GPU arms compare outputs
bitwise end to end (the probe), so a race would show as differing rows or a failed SG gate.

## 5. REGISTERED tick-time band (written before any run)

Device kernel time (the S3 cell, ncu `gpu__time_duration`, cache-control none, the primary): (union build + shared
+ combine) / (piece 3 split + combine) over the marked calls with 4 rows, `--union fused`:

* split: central 0.90, band [0.60, 1.25]; whole: central 1.75, band [0.85, 3.70]. [sharedkv4, review MINOR-4:
  SUPERSEDED: split band [0.60, 1.10] (its top IS F5'); the whole fused band is withdrawn (no whole fused arm:
  not tested). The build band is per marked call whose build range holds ONLY the union kernel (MINOR-5).]
* the build alone (`bd` range, per marked call; the group ops only on a tick's first keyed layer): central 15 us,
  band [5, 60].
* (sharedkv2, `--union torch`, unchanged: split 1.30 [0.80, 2.20], whole 2.10 [1.00, 4.00].)

The gate's event ratio (REPORT) uses the same bands (`SKV_PRED[...]["time_with_fused_build"]`).
Falsifier F5' : split above 1.10 with the fused union = sharing does not pay even with a one-launch build at this
schedule (then the kernel itself, not the build, is the cost: see the kernel-only ratio). Fused build above 60 us
per call = the O(N^2) compare passes are slower than predicted (try TE 32, or a per-row merge).

## 6. Merge notes

`bi_paged_attention_shared.py`: new code at 177-305 (build_groups, _windows, build_union_fused_reference,
build_union_fused) and 504-593 (the second triton block); SharedKV gains `union`, `_groups`, `_build`; the probe's
`build()` calls `_build`. The shared kernel's address arithmetic moved by + 130 lines against sharedkv2: q 332,
workspace 370 / 373, bases 425-428, tile rows and loads 445-448 (the X-pool follow-up of docs/R2_SHAREDKV_PLAN.md
section 7).

## 7. Fixes after the CPU suite on a360d86 (job 2191241: 421 passed, 7 skipped, 3 failed)

1. U4[split], U4[whole]: "Inference tensors do not track version counter". The production tick runs under
   torch.inference_mode, and the group cache read `seq_idx._version`. Fix: the cache key is an explicit TICK TOKEN.
   `TickEngine._layer` passes `group_token=ctx` (one SimpleNamespace per tick, built in run_rows, the same object for
   all layers); `SharedKV` / `SharedKVProbe.bi_paged_decode(..., group_token=None)`; the cache holds (token,
   seq_idx object, rows_max, grow) and keeps references to both objects; no token = rebuilt on every call. U3 and U4
   now run under torch.inference_mode (U3 asserts `seq_idx.is_inference()`).
2. U5: the kernel's `win` differed from build_union in all 24 SPLIT cases (whole mode and every other field
   matched, so uid / uidx were right). The split windows were the only quantity computed by a different construct
   than build_union: a masked [TE, SP] min / max over (entry, split) accumulated across the pass-B tile loop
   (a360d86 `_union_kernel`, the `if not WHOLE:` block after the rowpos store, and the final `else:` store with
   `hi1 = hi + 1`). The exact faulty step is NOT proven from the source alone: each line reads correct, and the
   only primitives there that the passing fields did not use are the int tl.min / tl.max reductions (uid and ucnt
   use tl.sum and matched). The interpreter's RuntimeWarning "invalid value encountered in subtract"
   (interpreter.py:417, a binary op) may come from those reductions or from the attention kernels' -inf
   arithmetic in the same child; not decided. Fix: pass C (after a third barrier; `_union_kernel` lines 586-598)
   reads the STORED uidx back per split k over exactly build_union's (row, position) set with NO reduction
   primitive: unrolled scalar loads (static NR x PPS) and compare-selects (tl.where), empty -> (0, 0). The hand
   case pins the values, so a remaining fault will show as concrete numbers.
3. New test `test_U7_hand_case_pins_win`: one request, 2 rows, P 4, PPS 2, hand-derived grow / ucnt / ut / uidx /
   rowpos and windows split [[0, 3], [3, 5]], whole [[0, 5]], empty group (0, 0): build_union and the twin; U5's
   interpreter child runs the KERNEL on it (tile sizes 2 and 16) and prints the windows on a mismatch.
   `test_U6_fused_path_source` now expects 3 barriers and no version-counter use in SharedKV.
