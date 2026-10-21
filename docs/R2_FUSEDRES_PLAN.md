# R2 fused residency resolve (b2): plan, interface, registered prediction

Branch `r2-fusedres` (worktree /mnt/beegfs/ojerbi/nosi-r2-fusedres), from cf35536 (SG5 p6 head). SOURCE ONLY, NOT RUN.
Labels: MEASURED (a cited run), MODEL (a derivation in this file), MISSING.

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
| Triton kernels in the R2 line | r2_twin_fused.py (twin), r2_residency_sc.py (exact-wait board kernel), r2_xpool.py (h2d entries, xpool branch) | none decides residency |
| nosi-r2-m2-own2b (3fbf8a7), nosi-r2-m2-sparse (9508109), nosi-r2-sharedkv, nosi-r2-xpool (44d5e95) | same files | no fused resolve; xpool binds lanes AFTER the torch decisions |

The kernel names of the 37.62 ms (other_res.md, G_d at B 64) confirm "many small torch ops per layer": radix sort
onesweep 6.91 ms / 768 launches, radixSortKVInPlace 4.50 / 160, cummax scan 2.25 / 64, index 2.02 / 292, copies
1.89 / 514, index_put 1.55 / 262, sort memsets 1.33 / 960, and / where / fill / clone. 9,774 / 32 = **305 kernels per
layer**, 3.85 us each on average: launch-count-bound, not byte-bound.

Known techniques used: one program per (request, head) stream with a register-tile lookup of the selected blocks
against the stream's slot blocks (a hash / sorted-merge lookup in its simplest form: k x NPOS compares); a
segment-local rank by `tl.cumsum`; vLLM-style page-table writes on the GPU; a global exclusive prefix done by each
program from per-(row, head) counts (no extra launch).

## 3. Design (nosi/nosi/r2_fusedres.py)

`OwnResidency(..., resolve="torch" | "fused" | "check", fused_plant=0)`. "torch" (default) is unchanged.
SG5: `r2_tick_sg5.py --b2-resolve torch|fused|check`.

Per layer, in S-C's order, grid S = B x H programs (one per stream):

| launch | when | does |
|---|---|---|
| V1 | the layer has verify rows | the verify rows' classification; per (row, head) the K5 extra-need count and the record count; per stream the usage; program 0 zeroes V2's sums |
| V2 | idem | K5 grants (the global prefix from V1's counts), the q-th FREE slot, Policy E (`tl.sort` of the stream's candidate keys), read slots, page tables (plan int32 and decision int64), every decision field, load ids, verify record / looks / read bitmap, the stream's state after its verify rows (a scratch buffer) |
| (S-C's device wait and urgent gather) | idem | unchanged |
| D1 | the layer has draft rows | the draft rows' classification on the post-verify state; counts; usage; program 0 zeroes D2's sums |
| D2 | idem | K5 grants with the R_ver reserve, take slots, page tables, decision fields, the draft record, the bias rows of the granted takes (bit copies), the counters, the selection bitmaps `sel` / `rowsel` |

So a draft-only layer is **2 launches**, a layer with verify rows **4**. Per tick (first layer only): B2's
`sync_mirror` (torch, unchanged), `_tick_rows` (one host-to-device copy, unchanged) and ONE more host-to-device copy
(the per-batch-slot row lists). No host read in the per-layer path (test FR4).

Exactness: every output equals the torch path's (section 6). The assumptions A1-A3 are in the module docstring; the
kernels raise the watched err flag when A1 / A3 / the stamp range fail (the tick is void, never silently different).
A2 (distinct candidate stamps per stream) holds by construction (every stamp is a fresh clock value) and is checked on
CPU by the equality tests, not in the kernel.

Limits (refused at construction): pin_records True (B1), record_cap set, any sc_mut / sb_mut / sparse_mut negative
control, a subclass that overrides `_layer_record`, C + X > 512, and an attached xpool (section 5). If store_b[l] and
cis[l] are not contiguous (B, rows, H) tables of one 2- or 4-byte dtype, D2 skips the bias copy and S-B's torch
`draft_bias_fill` runs after it (counted in `bias_fallbacks`; expected 0 at p5b2's point, where both are the tick
dtype).

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
  records. The bias rows of the draft takes are written at store_b[l][b, take_slot x 64 + r, h] (D2, BIAS) - the
  ONLY physical write of the fused path.

## 5. Merge with nosi-r2-xpool (44d5e95)

xpool keeps logical slots and binds lanes after the decisions (`XPool.release_free` after sync_mirror, `XPool.bind`
after device_verify and after device_draft, `draft_bias_fill_flat` instead of `draft_bias_fill`). The fused path is
compatible with I1-I5 as they stand. To merge, in `FusedResolver`:

1. first layer of a tick: `svc.xpool.release_free(svc.dmirror.code == C_FREE)` after `sync_mirror` (as xpool's
   ResidencySC.resolve);
2. after V2: `svc.xpool.bind(l, vdec.bh[0], vdec.bh[1], vdec.t["take_slot"], (granted | evict) != 0)` before the wait;
3. D2 with `BIAS=False`, then `svc.xpool.bind(...)` on the draft takes and `XP.draft_bias_fill_flat(...)`;
4. drop the xpool refusal in `FusedResolver.__init__`.

That costs xpool's own launches (bind, fill) on top of 2 / 4. A later step can fold the bind into V2 / D2 (the lane is
"the lowest free lane in (b, s) order per head": a global prefix like K5's).

## 6. CPU tests (benchmarks/Efficiency/tests_cpu/test_r2_fusedres.py; Triton interpreter in a fresh process, L36)

| node | checks |
|---|---|
| `test_FR0_the_default_is_torch_and_bad_modes_are_refused` | default "torch"; bad mode / plant / negative controls / record_cap / `_layer_record` override refused |
| `test_FR1_fused_equals_torch_at_every_layer_and_b2_equals_v1[f12b\|f20b\|f21b\|f6]` | (a) resolve="check" on B2's lockstep schedules: at EVERY tick and layer every output equals the torch path's (decision fields, pages, load ids, counters, record, bitmaps, bias rows); the B2 world equals v1; err 0; compared == ticks x L; preconditions: f20b Policy E LRU evictions, f21b in-flight evictions, f12b refusals, f6 ready / in flight / new |
| `test_FR2_each_plant_is_caught_at_its_first_effective_layer[1\|2]` | (b) plant 1 (Policy E newest first) on f20b / f21b, plant 2 (draft take off by one) on f6 / f12b; L34: the kernels count where the plant changes an output; effective -> the first difference is at exactly that (tick, layer); never effective -> no difference, reported as a NO-OP; each plant effective at least once |
| `test_FR2_the_stats_trace_is_cumulative_and_matches_the_report` | the precondition counters are consistent |
| `test_FR3_two_launches_per_draft_only_layer_four_with_verify_rows_and_no_torch_compute` | (c) f12b: 2 launches per draft-only layer, 4 per verify layer; no torch compute op (only allocations / views) on a draft-only layer after the tick's first; the torch path's compute ops per such layer reported (precondition >= 20) |
| `test_FR4_the_per_layer_path_has_no_host_read_and_a_planted_one_is_found` | no host read / host-to-device copy in `_layer`; planted `int(...)` found |
| `test_FR4_every_triton_function_resolves_its_names_in_its_module` | L36 name lint on every @triton.jit of the module (F23 in test_r2_residency.py also covers it); planted factory kernel found |
| `test_FR5_a_free_running_round_on_b2_check_equals_torch` | one free-running greedy round on b2 (tiny model): check == torch (rows, argmax, tokens, decision stats), no layer difference, err 0 |

Run: `env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_fusedres.py`
(also re-run test_r2_own2.py, test_r2_tick_b2.py (TB5 pins the SG5 b2 branch) and test_r2_residency.py F23).

## 7. REGISTERED PREDICTION (2026-10-21, before any run)

Cell **FR-g** at p5b2's point: r2_tick_sg5.py with p5b2's arguments (`--modes exact,fast --backends b2 --C 96 --X 64
--R-ver 16`, pool 100, gilfree, threads 8, unchecked) on one A100 holder, three invocations in this order:

- FR-g0 `--b2-resolve check` (short: `--draft-ticks 4 --steps 8`): the GPU equality gate.
- FR-g1 `--b2-resolve torch`: the same-holder control (re-measures p5b2).
- FR-g2 `--b2-resolve fused`: the timing arm.

Geometry used below: L 32, B 64, H 8, NPOS 160 (tile 256), k 64, nblk ~253 (tile 256), one draft row per request.

| quantity | p5b2 / profile (MEASURED) | prediction FR-g2 (MODEL) | how measured |
|---|---|---|---|
| decision equality on the GPU | - | **MUST: 0 differences**, compared == layers resolved, err 0, in FR-g0 draft-only and round | `draft_only.fused` and `round...report.b2.fused` (diffs, compared, layers, err) |
| launches per draft-only layer | ~305 kernels (S-C profile) | **2.0** Triton launches | `draft_only.fused.launches_per_layer` |
| res kernels per draft tick | 9,774 | **<= 120** (64 fused + B2's per-tick sync_mirror ops and two copies) | profile (section 8) |
| res KERNEL ms per draft tick, B 64, FAST = EXACT | 37.62 | **0.8 - 6.0 ms**, point 2.5 ms | profile (section 8); upper end also bounded by the event span below |
| res EVENT span per draft-only tick (median), eager | 129.01 FAST / 128.76 EXACT | **4 - 25 ms** both modes | `draft_only.device_ms.res` median |
| res event span, FR-g1 control | - | within 15% of p5b2's 129 ms | idem |

Derivation (MODEL):
- Bytes: per layer each kernel reads the layer's mirror (6 x 512 x 160 x 8 B = 3.9 MB); D2 writes the 9 decision fields
  (9 x 32,768 x 8 B = 2.4 MB), the record (1.3 MB) and the pages (0.4 MB). About 12 MB per layer, 380 MB per tick:
  0.26 ms at 1.5 TB/s. With B2's per-tick table copy (~250 MB, ~0.2 ms): FLOOR about 0.5 ms.
- Compute: per stream program, the register tiles are (64 x 256) lookups (2 in D1, 3 in D2), (64 x 64) sorts, a
  (128 x 64) page compaction and a (256 x 64) bitmap: about 0.21 M element ops (D1) + 0.34 M (D2), x 512 programs =
  0.28 G per layer. At 2 - 8 T element ops/s effective (A100 int32 9.7 T peak; reductions and spills cost) that is
  35 - 140 us per layer, **1.1 - 4.5 ms per tick**; + sync_mirror and the copies (0.2 - 0.5 ms); band rounded to
  0.8 - 6.0 ms.
- Event span: host-bound eager tick. Per layer about 15 allocations (~5 us), 2 Triton launches with ~60 arguments
  (~30-60 us each in Triton 3.2's Python launcher), the rows check (64 rows) and the prologue: 150 - 300 us per layer,
  5 - 10 ms per tick, + the tick's first layer (sync_mirror, row tables: ~1 ms); band 4 - 25 ms.
- B 287 (MODEL, for the break-even): the programs grow to 2,296; each program also reads the (R x H) count vectors
  (2 x 2,296 int32) and the usage vector: kernel ms scale about x 4.5 to **5 - 25 ms** (the low end only if the tiles
  do not spill). That alone can exceed the 9 - 16 ms budget; section 9 lists the next steps.

Falsifiers (each one stops the claim it names):
- F1 any FR-g0 difference, or compared != layers: the fused path is not exact on the GPU (L36: the interpreter is not
  the compiler). STOP; no timing claim.
- F2 err != 0 in any arm: an invariant (A1 / A3 / stamp range) fails on real data; the tick was voided. STOP.
- F3 a compile failure of any of the four kernels on the A100: STOP (L36).
- F4 `launches_per_layer` != 2.0 in FR-g2's draft-only phase, or `bias_fallbacks` > 0: the dispatch is not the
  designed one (the timing then includes S-B's torch bias fill).
- F5 res kernel ms > 6.0 at B 64 (profile): the tile cost model is wrong (spills / reductions); re-tile before B 287.
- F6 res event median > 25 ms in FR-g2: the host side dominates; the next step is the captured tick (TC-R) with the
  fused resolve, not more kernel work.
- F7 FR-g1 not within 15% of 129 ms: the holder differs from p5b2's; compare FR-g2 with FR-g1 only.

## 8. Kernel-time measurement (MISSING harness piece)

SG5 records CUDA-event spans only, and the tickprof harness (nosi-ctxsweep-tick ctxsweep_tick.py, parsed by
bound_4_2.py / other_res.py with the innermost `t.*` range) has `--tick-backend sc|sb|v1`, no b2. Two options for the
lead: (a) add `b2` and the resolve option to ctxsweep_tick.py (its `t.res` ranges then give res kernel ms per G_d
tick directly); (b) bound it by FR-g2's event span (kernel <= span). F5 needs (a).

## 9. Doubts and next steps

- Not run anywhere. Every kernel was written without an interpreter run on this login node (rules). The first CPU run
  may show interpreter-level errors (dynamic `range` bounds from loaded values, `tl.sort` on int64, scalar atomics).
- Tile size: (64 x 256) int32 tiles in 8 warps may spill. If F5 fires: split the lookup over NPOS chunks of 64, or
  sort the stream's slot blocks once per program and use a merge lookup.
- At large B the per-program reads of the (R x H) count vectors grow as B^2 in total; replace by a decoupled
  look-back prefix (one launch) if B 287 is the target.
- An independent static review (2026-10-21, before this commit) found and the commit fixes: a duplicate kernel
  parameter name (the module did not compile), the verify record's HOLE flag on a dup entry whose first miss
  overflowed (torch: DUP only), and a sticky err flag (now zeroed at each tick's first layer). Coverage gaps it named,
  open: no lockstep schedule has a dup behind an overflowed first miss, and no row selects one block twice (the
  duplicate branch of the sorted-unique step is untested on CPU).
- The check mode compares on the same inputs per layer; it does not compare a whole GPU run against a torch run (the
  transfer timing differs between runs, p5b2 scoring section 0).
- The event span stays host-bound until the tick is captured; the fused path makes capture simpler (no host read,
  fixed shapes per tick type, 2 / 4 launches per layer).
