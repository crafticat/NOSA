# NOTES: WS-B (departure D2), the joint pass with ONE residency for the union (`--joint-residency union`)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_L4P_faithful.md`, WS-B, and its "Component contracts"
  item 2 (human, 2026-10-28). Paper: `paper-notes/speckv/speckv.txt` section 3.2 ("We run drafting and verification in
  one forward pass ... each query attends to its own selected entries").
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-wsb`, branch `r2-wsb`, from `40ac348` (nosi-r2-spec5). No other worktree was touched.
- Status: SOURCE ONLY. I did not run a test, an interpreter or a job. The lead runs the commands in sections 5 and 6.
- Written 2026-10-28T10:22:33Z (`date -u`). Style: ASD-STE100. Labels: CODE = read at file:line; MEASURED = from a
  scoring file; MODEL = hand arithmetic; INFERRED = my reading.
- Two binding corrections arrived during the work (coordinator, 2026-10-28). Section 3 shows how each one is met:
  (1) FAST is checked on FAST itself against SH (selection sets, presence, logits deltas), not against the EXACT pass;
  check-only work stays out of the timing cells; the phase dependencies and the timer boundaries are stated.
  (2) the WS-B acceptance contract: row roles, actual kernel dispatch, per-row selections / masks / lengths, weight and
  KV reuse; union residency must not become union attention (a MUST visible count); the exposed counters.

## 1. What I made

| file | what |
|---|---|
| `nosi/nosi/verify_rows.py` | mode "union" (`_row_masks`, `_union_list`, `_store_fill`, `_store_call`, `_fa_check_u`, `_presence`, `_union_rows`), VerifyOps(spill, check), negative controls "drop_spill" and "union_attention", host invocation counts `vs.calls` (every mode), per-row EXACT marks "resid" / "attn", `layer_ms`. Modes "exact" and "fast" keep their bits (Gate 0, section 5) |
| `nosi/nosi/spec_round.py` | SpecOps(mode="union", joint_stats) for L4 and L4P; `_round_l4p` sets the rows' roles per pass in union mode (no DraftResident bracket). Modes "exact" / "fast" are 40ac348's code paths |
| `benchmarks/Efficiency/r2_wsb_cell.py` (new) | the GPU cell: SH; timing arms L4X, L4XU; check arms SHR, UC, UCR; MUST gates; REGISTERED block `PRED_WSB` (TBD) |
| `benchmarks/Efficiency/r2_wsb_step.sh` (new) | a copy of `r2_spec5_step.sh`: ROOT this tree, OUTROOT `/mnt/beegfs/ojerbi/r2-runs/wsb`, claim `wb_active_<holder>`, cells wb1m, wb1a, wb1b, wb1ca, wb1cb |
| `benchmarks/Efficiency/tests_cpu/wsb_world.py` (new) | the CPU world (not a test file): `fake_attention_u`, `SelRecorder`, `run_union_vs_sh`, `union_bytes`, `sh_attention_ref`, `run_spec_union`, `old_module`, `joint_run` |
| `benchmarks/Efficiency/tests_cpu/test_wsb.py` (new) | the CPU tests (section 5) |
| `benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py` | Gate 0 parts (a) / (b) read the rung-L5 tip 40ac348 (`SPEC5_TIP`) in place of the working tree; the WS-B change on top is pinned by test_wsb.py |

NO NEW KERNEL. NO KERNEL EDIT. The union pass uses the shipped kernels as they are: `diff_offload`, `pool_update`,
`flash_pool_swap` (the residency), `flash_h2d_from_mask` / `_bias` (the window gathers, the store's host misses, and
the store's POOL HITS: the same Triton kernel with the engine's own `_k_gpu` / `_v_gpu` / `_kv_bias_gpu` as the source
and the slot index `topk + q` as the "block id", CODE flash_h2d_mask.py:30-62, flash_h2d_mask_bias.py:33-81; the
rollback's own staging redirect already uses a GPU source, ragged_rollback.py:817-820), `flash_attn_nosa_with_kvcache`
(the window call and the store calls, cache_batch_idx, return_softmax_lse). Everything else is torch.

## 2. The design

### 2.1 One layer of the union pass (verify_rows.py `_union_rows`, :761-928)

1. THE TENTATIVE ROWS (:782-790): rows in the tail block into the tail slot; rows past the block edge into STORE SLOT 0.
2. THE RESIDENCY, ONE per layer and pass (:791-798): row 0's selection into the window with the shipped diff, victim
   pool, map commit and the two host gathers. This is the shipped residency of position p. `vs.calls["resid"]` counts
   it. There is no serial per-row residency.
3. THE UNION LIST (`_union_list` :627-675): per (head, request), in id order: every block that a FULL row selects (a
   verify row: its whole selection must be present) and every POOL HIT of a RESIDENT row (a draft row reads what is on
   the GPU and never fetches; the same rule as the serial DraftResident step), minus the window's blocks and the next
   block m + 1. Each listed id gets its source: the pool slot `topk + q` (a pool hit after the residency) or the host.
4. THE STORE (B, X x 64, H, D): slot 0 = the next block; slots 1 .. X - 1 take X - 1 listed blocks per fill. FILL c
   loads the listed ids c (X - 1) .. (c + 1)(X - 1) - 1 (`_store_fill` :678-692): pool hits device to device, misses
   with the urgent GPU gather from the host rows. Fill 0 is the extra store; fills 1 .. C - 1 are the SPILL: the same
   store again, counted (stat `spill`), never a drop.
5. THE BOUND (VerifyState :257-264): C = ceil((R - 1) x topk / (X - 1)). Row 0's selection is in the window, so the
   list holds at most (R - 1) x topk ids, and C (X - 1) >= that bound: NO SELECTED BLOCK CAN BE DROPPED. `unserved`
   (listed ids past C (X - 1)) counts the plant only. With X 48 and R 4: C = 5 (fill 0 + 4 spill fills).
6. ATTENTION (:819-874): call A on the window (the FAST call: the per-entry bias marks the window slots of row r's own
   selection, the tail slot is cut by cache_seqlens, F4 kept); then ONE call per store fill (`_store_call` :695-735):
   row r sees store slot x only when the listed id is in its own selection, slot 0 only past the block edge and only
   up to its own position. Each call is flash_attn_nosa over the E = B x R one-row entries with cache_batch_idx (route
   1) and the entry-indexed bias (VRF-IDX). The per-entry length = the highest slot the row reads, so an empty fill
   costs the kernel's early exit (CODE flash_fwd_kernel.h:603-641; the combine :1331-1339: out 0, lse +inf; the merge
   gives that part the weight 0, :728-731).
7. THE MERGE: after each call, a running log-sum-exp merge (numerator, denominator, maximum; float32; no stack of the
   parts, so the peak memory is one part); the label "merge" covers the final division. Call A is never empty (the
   tail slot), so the maximum is finite; a part of length 0 has the weight 0.
8. THE END STATE (:915-928): the FAST end state of 40ac348 (write-back of a filled block, the next block's rows into
   the tail slot, the tail id, the length); the rollback hooks `fast_begin` / `fast_end` as FAST (ragged_rollback.py
   :671-711), so truncate / commit undo a union pass exactly as a FAST pass.

WHY ROW 0 OF THE FIRST PASS HAS SH'S BITS: its whole selection is in the window in the serial slot order (the same
residency call on the same state), its bias row equals the window bias, its store parts are empty (weight 0, exact),
and its length is the body's. With row-invariant GEMMs, every layer's row 0 then has the serial inputs (CPU claim,
test_union_vs_sh). Rows 1 .. R - 1 read their blocks in other slots and parts: the float order differs (FAST).

### 2.2 Row roles (the contract: "exact row roles")

| level | full rows (whole selection present) | resident rows (pool hits brought, misses skipped) | neither |
|---|---|---|---|
| L4XU (spec_round L4, mode union) | rows 0 .. K - 1 (the verify rows) | row K (the draft row) | - |
| L4P union (spec_round L4P) | rows r < v_b (per request) | the deferred row v_b and the draft row u_b | pad rows > u_b |
| UC / UCR (check arms) and the CPU vs-SH tests | every row (teacher-forced: every row's K / V stays committed state, as SH's) | - | - |

L4P union: the draft row's skipped ids go to the rung-L5 seam through `vs.miss_ids` (= DraftResident.miss_ids) and its
selection to the requested-KV pin through `vs.pin` (spec_round.py :917-925).

### 2.3 What is not changed

- The window holds row 0's selection after the pass (as FAST). The next round's drafts and joint pass start from it.
  The store is scratch: a host miss of a store fill is not kept (risk 7.5).
- The draft row of L4XU is RESIDENT, as L4X's DraftResident row K (the plan's "K + 1 rows" read as K verify rows and
  one draft row; FAST must change only the float order, so the draft row keeps its semantics).

## 3. The corrections and the contract: where each one is met

| requirement | how | where |
|---|---|---|
| FAST checked on FAST against SH: each verify row's selected SET = SH's on the same prefix (MUST) | check arms SHR (the serial ragged step, teacher-forced; MUST its fingerprints / argmaxes = SH's) and UC (production GEMM) / UCR (rowgroup GEMM): per pass, layer, row, head and request the sorted sets compared; changed count, by layer, by row, symmetric-difference histogram, first cases | r2_wsb_cell.py `run_shr`, `run_check_union`, gate `<arm>-SEL`; CPU `test_union_vs_sh` |
| every selected block present (MUST) | the presence recount (`_presence` :750-758): window, list, tail, next block; stat `dropped`; `unserved` | gate `<arm>-PRESENT`; CPU byte check `union_bytes` |
| logits deltas vs SH per verified position (REPORT) | per (position, request): max abs delta over SHR's top-16 tokens, the max-logit delta, the argmax against SH's token; quantiles per row index | `run_check_union`, summary CHECK.<arm>.logits |
| an EXACT pass does not certify FAST | no CPU or GPU claim of WS-B compares FAST with EXACT; the CPU tests compare with the serial step (SH) | test_wsb.py |
| check-only work out of the timing cells; marked | VerifyOps(check) and SpecOps(joint_stats) off in L4XU; every check-only line says CHECK-ONLY | verify_rows.py :855-856, :882-912; r2_wsb_cell.py `run_spec` |
| phase dependencies and timer boundaries | section 4 | - |
| exact row roles; counts per pass | section 2.2; stats `rows_full`, `rows_res` (check cells); host `vs.calls` per pass (every arm) | summary `roles`, `calls_per_pass` |
| ACTUAL kernel dispatch | one PROFILED joint pass (UC, UCR) and one profiled SHR step (torch.profiler: kernel names, calls, device ms); CPU stand-ins do not establish dispatch | `_profile`, summary CHECK.<arm>.profile |
| per-row selections / masks / lengths | `chk_sel` (selections), the visible count per row (masks), `streamed_rows` (lengths) | stats |
| union residency must not become union attention (MUST) | the VISIBLE COUNT: a full row sees exactly its selection size; a resident row its selection size minus its skipped blocks; stat `vis_bad` (plant "union_attention" proves the counter) | :855-874, :897-906; gate `<arm>-VIS` |
| real weight and KV reuse | weights: ONE GEMM call per weight over E rows (`calls["gemm"]` per pass); KV: ONE residency per layer (`calls["resid"]`), each listed block fetched once for all rows (`extras_pool`, `extras_host`); attention reads are NOT shared between rows (route 1: each entry streams its own columns, masked or not): `streamed_rows` against `visible_blocks` x 64 shows the masked-column traffic | stats, section 7 risk 7.4 |
| dense / selection / residency / attention service per pass | probe-pass phase marks, grouped (r2_wsb_cell.PHASE_GROUPS) | summary `joint_phases` |
| ready intersection and residual misses, blocks and bytes | ready = window hits + pool hits (the residency's and the store's); residual = the window's and the store's host loads | `ready_record` (check arms); the KVMeter's NEW blocks per round (timing arms) |
| the COMPLETE layer time | `layer_ms`: first mark of layer l to the first mark of layer l + 1 | summary `joint_phases.layer_ms_*` |
| the same representative inputs as L4X / L4P cells | the workload pin, doc rows, C, B, K 3, journal 256, twin / body fused, graph on, redirect on of sp3a / sp3b | r2_wsb_step.sh |

## 4. Phase dependencies and timer boundaries (verify_rows.py; one stream today)

A mark is a CUDA event recorded on the current stream at the line (`_mark` :312). A label's time runs from its mark to
the NEXT mark of the same pass (`phase_ms` :1022); it includes the GPU idle time while the host is late. The labels
PARTITION each layer (no nesting, no overlap); `layer_ms` (:1031) gives the complete layer span; the pass runs from
layer 0's "pre" to "done" (:1001); the round runner's "joint" phase (spec_round round events) brackets the pass plus the
embedding and the meter toggles.

| label (mark) | ops | DATA dependency (what it needs) | could overlap with (not done today) |
|---|---|---|---|
| pre (:939) | rmsnorm, wqkv GEMM (:942), nosa_linear, rope | the previous layer's output | - |
| twin (:951) | R twin advances (`_twin_rows` :333) | k, cis of "pre" | - |
| select (:953) | ONE stage-1 (:357), ONE pooling (:361), two top-k (:364, :368) over E rows | q of "pre"; the twin's comp_k, n_used, M_b | - |
| resid (:964 -> :782-798) | tentative rows; diff + pool update + pool swap (:793); map commit; window gathers (:795-796) | row 0's selection | the tentative-row writes need only "pre" / "twin" |
| union (:804 -> :806-807) | the union list, the sources, the sorted sets | all rows' selections; the window and pool MAPS after the diff / pool (not the gathers) | the window gathers (:795-796) |
| fetch (:808 -> :810) | store fill 0 (device copies + host gathers) | the list; the pool SLOTS after the pool swap; the host rows | call A (:840): independent of fill 0 |
| attn (:819 -> :821-862) | call A (:840), the fill-0 call (:857 -> :721) and its merge step | call A: the window gathers + the tentative rows + q; the fill-0 call: fill 0 | the next fills (with a second store) |
| spill (:863 -> :865-874) | fills 1 .. C - 1, their calls and merge steps | fill c needs the store free: the previous call has read it (write-after-read) | a second store would remove this wait |
| merge (:875 -> :877-878) | the final division of the running merge | every part | - |
| check (:882 -> :884-912) | CHECK-ONLY (and the rung-L5 seams when set) | the list, the selections | - (off in timing cells) |
| end (:915 -> :917-927) | write-back, next-block rows, tail id, length | call A has read the tail slot (write-after-read) | the "post" GEMMs (they need only the merged output) |
| post (:969 -> :971-...) | wo GEMM, rmsnorm, gate_up, SiLU, down | the merged attention | - |
| head (:998) | final rmsnorm, lm_head | the last layer | - |

EXACT (L4X): "rows" (:959) covers the union statistics sort; then per row "resid" (:425: the row's decode_fixed, its
DraftResident / rollback hooks) and "attn" (:434: the row's attention): residency and attention ALTERNATE per row, and
row r + 1's residency waits for row r's attention (the window is rewritten in place).

## 5. CPU tests for the lead

Run on a CPU node, inside a job:

    cd /mnt/beegfs/ojerbi/nosi-r2-wsb
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsb.py

Regressions (the modified modules' suites; test_spec_prefetch.py builds the C++ packer, in the image as before):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_verify_rows.py benchmarks/Efficiency/tests_cpu/test_spec_round.py \
        benchmarks/Efficiency/tests_cpu/test_spec_paper.py benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py \
        benchmarks/Efficiency/tests_cpu/test_rollback.py benchmarks/Efficiency/tests_cpu/test_ragged_engine.py

| test | claim | negative control / preconditions (L34) |
|---|---|---|
| `test_gate0_files` | the tree = 40ac348 except the declared WS-B files; no untracked file in nosi/; test_spec_prefetch.py changes only its Gate-0 reads (SPEC5_TIP); verify_rows.py names no rollback / round / prefetch module (the older Gate-0 scans) | - |
| `test_gate0_verify_rows_unchanged[B1, B8]` | modes exact (gemm exact, rowgroup) and fast: logits of every pass and every engine and twin table BITWISE the 40ac348 source (git show) | - |
| `test_gate0_spec_round_unchanged[L4X, L4F, L4P]` | this round runner = the 40ac348 source: tokens, logits rows, counters, records, state, bit for bit | both accept paths ran |
| `test_union_vs_sh[B1, B8, B1nat, B8nat]` | THE CHECK OF FAST ITSELF AGAINST SH: every row's selection set = SH's (every layer, head, request; 3 passes); dropped 0, unserved 0; visible = selected per row; row 0 of pass 0 BITWISE SH's logits; argmax = SH's, delta <= 1e-4; one residency per layer and pass (module count and an independent count of the diff kernel); attention calls = L (1 + C) per pass; VRF-IDX (window call; store call when extras exist) | a block edge in the run; B 8: extras > 0 |
| `test_union_attention_is_sh_math` | every full row's output = SH's attention math in float64, order-free, within 2e-5 | a row past a block edge, an F4 fill row, a store entry |
| `test_every_selected_block_present[B1, B8]` | X 4 (the union does not fit one fill): every selected block of every full row is in the window / a store slot of some fill with its host rows and cis rows, or it is the tail / next block; dropped 0; selections and logits still SH's | spill > 0 |
| `test_dropped_extras_plant_caught` | plant "drop_spill" (the old 5.57 % behaviour): dropped > 0 and = the byte check's missing count; the cell's UC-PRESENT gate FAILS; clean run PASSES | unserved > 0 |
| `test_union_attention_plant_caught` | plant "union_attention": vis_bad > 0; the logits leave SH's; the cell's UC-VIS gate FAILS | extras > 0 in the clean run |
| `test_one_residency_call_per_layer` | union: L diff calls per pass | EXACT: R L (the serial rows) |
| `test_resident_rows_single_pass` | L4XU's roles (K full + 1 resident): rows_full / rows_res; the full rows = SH; the resident row's skipped count = the byte check's | - |
| `test_fake_attention_u` | the stand-in = verify_world.fake_attention_idx per entry, bitwise; length 0: out 0, lse +inf (the kernel) | verify_world's stand-in raises on length 0 |
| `test_union_ops_and_bound` | refusals; C = ceil((R - 1) topk / (X - 1)), cap >= bound, C minimal; drop_spill C 1; spill override | - |
| `test_no_host_sync_in_union_pass` | verify_rows.py reads no device value in a union pass (check on and off); the only trap hit is joint_pass's check of the HOST mirror active_host (40ac348) | the trap is not blind (read_stats) |
| `test_l4xu_runner[B1, B8]`, `test_l4pu_runner` | spec_round mode union: residency and host-row invariants after every round, no error, one joint pass per round, one residency per layer and pass; L4P: roles set, the rung-L5 seam gets the draft row's misses; agreement with the shipped decode >= 0.9 (printed; FAST: WS-G gives the gate) | both accept paths; a block edge |
| `test_cell_arguments`, `test_cell_pure_parts`, `test_step_script_wb1` | the cell's arguments and refusals, the pure parts, the gates against the CPU runs, the TBD verdicts; the step script's cells, DRYRUN wb1m / wb1ca, wb1a / wb1b STOP while PRED_STAMP_WSB is None | - |

If a precondition fails, the data do not exercise the path: tell me and I change the data, not the claim. The most
data-dependent ones: extras > 0 and spill > 0 at B 1 (S 4923), a store entry in the records.

## 6. The GPU cells

    DRYRUN=1 bash benchmarks/Efficiency/r2_wsb_step.sh --nosi <sha> <holder> <run id> wb1m
    bash benchmarks/Efficiency/r2_wsb_step.sh --nosi <sha> <holder> <run id> wb1m     # the miniature first
    bash benchmarks/Efficiency/r2_wsb_step.sh --nosi <sha> <holder> <run id> wb1ca    # the check, C 128 (not timed)
    bash benchmarks/Efficiency/r2_wsb_step.sh --nosi <sha> <holder> <run id> wb1cb    # the check, C 73
    bash benchmarks/Efficiency/r2_wsb_step.sh --nosi <sha> <holder> <run id> wb1b     # timing C 73 (after PRED_WSB)
    bash benchmarks/Efficiency/r2_wsb_step.sh --nosi <sha> <holder> <run id> wb1a     # timing C 128 (after PRED_WSB)

| cell | C (pool) | B | steps | arms (after SH) | X | STEP_MIN | mem | GPU time (MODEL) |
|---|---|---|---|---|---|---|---|---|
| wb1m | 73 (10) | 8 | 64 | L4X, L4XU, SHR, UC, UCR; --require-stress 1 | 4 (C 64 fills: the spill runs) | 40 | 100G | 15-20 min (6 prefills of about 2 min, sp3m 124.8 s) |
| wb1a | 128 (65) | 216 | 512 | L4X, L4XU | 48 (C 5) | 70 | 240G | about 40 min (3 prefills of 11 min) |
| wb1b | 73 (10) | 318 | 512 | L4X, L4XU | 48 (C 5) | 100 | 340G | about 56 min (3 prefills of 16 min) |
| wb1ca | 128 (65) | 216 | 128 | SHR, UC, UCR (check-steps 128: 32 passes) | 48 | 80 | 240G | about 50 min (4 prefills) |
| wb1cb | 73 (10) | 318 | 128 | SHR, UC, UCR | 48 | 110 | 340G | about 70 min (4 prefills) |

Total about 4 h. MUST gates: SH-RUN; L4X-RUN / -ERR / -COMMIT / -TOK / -FP; L4XU-RUN / -ERR / -COMMIT / -RESID;
SHR-RUN / -FP / -TOK; UC, UCR -RUN / -ERR / -SEL / -PRESENT / -VIS / -RESID / -IDX; wb1m also -STRESS. REPORT: the
round phases, the joint-pass ms in SH steps, committed tok/s vs SH and vs 3,407.2, the probe-pass phase groups (dense,
twin, selection, residency, fetch, attention, spill, check, end) and the complete layer time, the host calls per pass,
the KV at verify (NEW blocks per round: the residual misses of the joint pass), the check arms' logits deltas, union /
extras / spill / streamed rows / ready and residual blocks and MB, the profiled kernels.

REGISTERED BLOCK, PREDICTIONS TBD FOR THE LEAD: `r2_wsb_cell.PRED_WSB` (keys C128 and C73: joint_sh_steps,
ratio_vs_sh, ratio_vs_l4x) and `PRED_STAMP_WSB` (`date -u`), in a commit BEFORE wb1a / wb1b (the step script stops while
the stamp is None). Plan (MODEL): joint pass about 2.85 SH steps at C 128 B 216, about 3.2 at C 73 B 318; FALSIFIER
> 3.5 / > 4.0.

INPUTS FOR THE BANDS (MODEL / MEASURED, not a prediction of mine):
- MEASURED parts (SCORING_vr1.md section 6, VRC/VRF at C 128 B 216, per pass): GEMM + norms 69.5 ms (one call over 864
  rows), twin 35.6 ms (torch twin; the cells here use the FUSED twin: about -32 ms, INFERRED from rg1), selection of 4
  rows 25.6 ms, FAST attention + masks + merge 72.5 ms. The serial residency + attention of 4 rows was 153.8 ms (VRC).
- The union adds per layer: one residency (the shipped one: about 14.8 / 32 ms per layer at C 128, MAP 2.1), the
  union list and sorts (small), fill 0 (device copies of about 8 pool hits per (head, request) and the host misses),
  C - 1 = 4 empty spill fills (about 20 small launches each: about 2,500 launches per pass; MODEL 5-10 ms per pass),
  the running merge of 6 parts (6 steps of about 6 element-wise ops on 864 x 32 x 128 float32: MODEL 2-3 ms per pass).
- vr1's VRF "resid" phase (215.7 ms) is NOT a model of this pass: VRF copied all X - 1 = 15 extras slots of every
  (head, request) with advanced-index torch.where (verify_rows.py :364-370 of 40ac348) and fetched again every pass.
  Here the pool hits go through the shipped gather kernel (programs exit at once for an empty slot), and each fill
  call reads only up to the highest slot a row needs.

## 7. Risks and open points

1. Nothing ran. The CPU tests are my reading of the stand-ins and of the code.
2. THE SELECTION MUST WILL PROBABLY FAIL ON THE GPU (INFERRED, MEASURED basis): UC uses the production GEMM (one call
   over 4 B rows); vr1's VRC (the EXACT residency with that GEMM) disagreed with SH on 0.40 % of the argmaxes from GEMM
   rounding alone (SCORING_vr1.md section 7). A changed q moves a near-tie of the pooled bf16 scores at the top-k
   boundary. UCR removes the GEMM part (rowgroup = the shipped GEMM bits), so a UCR change is the union attention's
   float order alone. The symmetric-difference histogram separates near-ties (size 2: one block swapped) from a
   semantic bug (many blocks). The gate is MUST as instructed; if it fails only with size-2 swaps in layers > 0, that is
   float order, not a missing or wrong block, and the lead / human decide the rule. I did not weaken the gate.
3. MEMORY at wb1a: the store at X 48 is 216 x 48 x 64 x 2 x 128 x 2 x 2 B = 680 MB (K + V) plus 2.6 MB bias; L4F
   (X 16, 226 MB) peaked at 83.007 GB of about 85.1 (SCORING_specp_fix.md section 8). MODEL L4XU peak about 83.5 GB.
   If L4XU stops on memory, X 32 (453 MB, C 7) is the fallback.
4. ROUTE 1 STREAMS MASKED COLUMNS: call A reads the whole window for every row (4096 rows per entry), also the slots of
   other rows' selections; a store call reads up to the highest slot a row needs. `streamed_rows` against
   `visible_blocks` x 64 reports this traffic. Shared loads are WS-C (the fork).
5. The store is scratch: a host miss of a fill is fetched again by the next pass if its row needs it again (the serial
   path would keep it in the window or pool). The KVMeter's NEW blocks per round of L4XU against L4X show the refetch.
6. The empty spill fills are launched on every layer (no host read decides it). With X 48 they serve nothing in the
   common case; the "spill" phase reports their cost. A smaller C would need a host decision or a bound from the data.
7. FAST end state: the window holds row 0's selection after the pass (as FAST of 40ac348); the drafts that follow find
   fewer window hits than after the serial pass (its window = the last row's selection).
8. The visible count is at the MASK level: the F4 fill row has its tail slot visible in the mask but cut to 0 rows by
   cache_seqlens, as SH (F4 kept). A past-edge row always sees the tail slot (block m) and slot 0 (block m + 1): both
   are in its forced local window; if not, vis_bad counts it.
9. The profiler (CHECK-ONLY) may fail in the container (CUPTI); then the record says so and the arm runs on (REPORT).
10. The check arms read SHR's selections from host memory (n x L x H x B x 64 int16: 226 MB at B 216, 333 MB at B 318).
11. L4P union is wired and CPU-tested (invariants, roles, the seam) but has no GPU cell here: the plan's I2 rung uses it.
12. `vs.calls` counts host calls in every mode (an added integer, no device work). r2_verify_cell's VRX phases now
    split into "resid" / "attn" per row (the new EXACT marks; probe passes only).

## 8. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- `date -u` (the start time and the stamp of this file).
- `cd` (shell built-in) in front of read commands; `&&`, `;`, `|`.
- In one read command I wrote the shell variable assignment `sed_unused=0` and ran `head -c 0 /dev/null` (no sed
  program ran; nothing was changed).
- git: `git worktree add` (as instructed), `git add`, `git commit -F` (the message written with the Write tool), and
  read-only git commands.
- No interpreter, no sed, no awk, no heredoc, no `cat >>`, no tr / paste / uniq / xargs, no hashing, no `find`, no
  srun / sbatch / pytest, no sleep. Files were written with the Write and Edit tools.
