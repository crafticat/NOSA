# NOTES: rung I2 = I1 (WS-A, tentative buffers) + WS-B (one residency per layer for the union): arm L4PTU

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_L4P_faithful.md`, the integration ladder (I0 -> I1 -> I2),
  the component contracts 2 and 4, the checkpoint rules (one lead integrates FROZEN versions; each rung reports its
  incremental change against its prediction).
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-i2`, branch `r2-i2`, from r2-wsareg `83dac96` (WS-A + the lead's registration).
  Merge commit `e3827c0` (+ r2-wsb `1a8ce68`), then the I2 commit (this file). No other worktree was touched: the
  pinned trees nosi-r2-wsa (83dac96), nosi-r2-wsb (1a8ce68), nosi-r2-wsb2 (9688cee), nosi-r2-wse, -wsg, -wsm, -wsc were
  only read.
- Status: CPU tests RUN by me on the amd partition (the human's permission of 2026-10-28T13:19Z), section 6.1; no GPU
  job. Commits: e3827c0 (merge), 1fd7c9c (I2), 55551c3 (CPU-run fixes), e6a1cf9 (review fixes), then this notes update.
- Written 2026-10-28T13:10:41Z (`date -u`). Style: ASD-STE100. Labels: CODE = read at file:line; MEASURED = from a
  scoring file; MODEL = hand arithmetic; INFERENCE = my reading. Line numbers come from `grep -n`.

## 1. The merge

`git merge --no-ff 1a8ce68` into 83dac96: NO CONFLICT (commit e3827c0). WS-A adds spec_tentative.py, its cell, step
script, world, tests and one Gate-0 line in four older tests; WS-B edits verify_rows.py and spec_round.py and adds its
cell, step script, world and tests. The one file that both sides touched, `tests_cpu/test_spec_prefetch.py`, auto-merged
with both sides' lines kept verbatim (WS-A: the DECLARED extension and the spec_tentative line filter; WS-B: SPEC5_TIP in
parts (a) and (b)). Checked with git: the set of changed lines of the merged file against 40ac348 equals the union of the
two sides' sets (`test_i2_gate0_files` (c) asserts it).

Gate-0 lists after the merge (each side's Gate 0 compares the tree with 40ac348, so each must now declare the other side's
files; both sides' entries are kept):

| test | change |
|---|---|
| `test_wsa.py` | DECLARED += WS-B's files + I2's files; in the older Gate-0 tests the lines of WS-B's own diff (40ac348 -> 1a8ce68) are allowed; the module scan allows spec_tentative_union.py. Only the function `test_wsa_flag_off_is_spec5` and module constants changed (test_i2 (e), AST) |
| `test_wsb.py` | DECLARED += WS-A's files, WS-A's four Gate-0 test files, I2's files; in test_spec_prefetch.py the lines of WS-A's own diff (40ac348 -> 83dac96) are allowed; the `union_list` scan allows spec_tentative_union.py. Only `test_gate0_files` and constants changed |
| `test_ragged_engine.py`, `test_verify_rows.py`, `test_rollback.py`, `test_spec_round.py` | one added exclude line each: `nosi/nosi/spec_tentative_union.py` (I2, Gate 0: test_i2.py) |

## 2. The seam (task step 1): how TentRunner's joint pass reaches verify_rows, and why WS-B's mode is not selectable

CODE, r2-wsareg 83dac96, `nosi/nosi/spec_tentative.py`:
- the J slot: `TentRunner._ps_round` calls `T.run_pass(self.model, plan)` (:919) with the plan of `_joint_plan` (:873);
  the plan's GEMM is `lin=lambda x, w: VR._lin(self.vs, x, w)` (:899).
- `TentState.run_pass` (:295) runs its OWN layer loop `_layer` (:313). From verify_rows it uses only the per-row lengths
  (`plan.vs.n_used` / `plan.vs.M_b`, :343-344), the batched selection `VR._select_rows` (:355) and `VR._lin`.
- every row then runs `self._row(...)` (:357), and each `_row` (:385) runs its OWN residency (`kern["diff"]` :401, pool
  update, pool swap, gathers): R residencies per layer.
- the runner's VerifyState is attached with `VerifyOps(..., mode="exact", ...)` (:772): only for those buffers.

CODE, r2-wsb 1a8ce68, `nosi/nosi/verify_rows.py`: mode "union" lives in `_union_rows` (:777), reached only through
`verify_rows._layer` (`elif ops.mode == "union"` :984) from `joint_pass` (:1002). TentRunner never calls `joint_pass`
(only PaperJournalRunner does, spec_tentative.py:1245).

ANSWER: WS-B's union mode can NOT be selected through the same ops object. Setting `VerifyOps(mode="union")` on
TentRunner's VerifyState changes nothing, because TentState never enters `verify_rows._layer`. And `_union_rows` cannot be
reused as it is: it writes the tentative rows into the COMMITTED tail slot (:801-803) and store slot 0 (:804-806), writes a
filled block back to the host (:941) and moves the tail id (:945): the FAST end state that a journal must undo (departure
D1, which WS-A removed). The MISSING SEAM is inside TentState: the row loop of a joint pass. I added it:

| seam (spec_tentative.py, this tree) | what |
|---|---|
| `TentState._rows` (:371), called from `_layer` (:362) | the rows of a pass, row by row (the old loop, the same ops) |
| `TentState._resid` (:450), called from `_row` (:410) | step 4 (the residency of one row), moved out of `_row`, the same ops in the same order |
| `TentState._write` (:482), called from `_row` (:412) | step 5 (the buffer write and the layer-0 counters), moved out of `_row` |
| `TentRunner._joint_ops` (:850), `_make_state` (:855), called from `__init__` (:803, :807) | the VerifyOps and the TentState of the runner |

Every arm of spec_tentative.py keeps the bits of 83dac96 (Gate 0, section 4: run against the 83dac96 SOURCE loaded from
git, also with spec_tentative alone so the seams are isolated; the AST outside the seams is 83dac96's).

The other side of the seam is the NEW module `nosi/nosi/spec_tentative_union.py`:
- `TentStateU(TentState)` (:87): `_rows` (:104) sends a JOINT pass (R > 1) to `_rows_union` (:134); a one-row pass (the
  P / D slots, a J slot of one-row requests) keeps TentState's row (one residency per pass anyway).
- `_rows_union`, per layer: ONE residency = row 0's selection with spec_tentative's own step 4 (`self._resid` :157:
  draft skips, the tentative-block skip and the tail-id injection, kept, never undone); every row's K / V / cis into the
  TENTATIVE BUFFER (`self._write` :161; the D3 counters); the union list with WS-B's `VR._union_list` (:171) on the
  selection with every TENTATIVE id (above qb = qc // 64) set to -1 (:170: those rows live in the buffer, not on the
  host); the store fills `VR._store_fill` (:201, :223); attention per entry e = b R + r: call A on the committed window
  (:203; a window slot visible iff its block is in row r's selection; the tail slot iff not past the edge or the head
  selects qb, :187; TentState._row's cut), one call per store fill `VR._store_call` (:224; in_tail all True, so slot 0
  is never used), call B on the tentative buffer (:247; TentState._row's masks per entry, with cache_batch_idx), merged
  by the running log-sum-exp; check-only presence and visible count (`VR._presence` :267). Nothing else: the committed
  state changes only at TentState.commit.
- `TentRunnerU(TentRunner)` (:295): `_joint_ops` (:326) returns VerifyOps mode "union" (store, fills, check / stats
  flags), `_make_state` (:331) a TentStateU; `_ps_round` (:335) records phase marks on probe rounds; `read` (:348) adds
  `union = verify_rows.read_stats`; `run_teacher` (:356) is the teacher-forced driver of the check cell.

NO NEW KERNEL: every kernel call is one that spec_tentative.py or verify_rows.py mode union already makes
(diff_offload, pool_update, flash_pool_swap, flash_h2d_from_mask(_bias) with host and with GPU sources,
flash_attn_nosa_with_kvcache with cache_batch_idx / return_softmax_lse); the rest is torch.

## 3. What L4PTU changes against L4PT (both FAST; both checked against SH)

| item | L4PT (I1) | L4PTU (I2) |
|---|---|---|
| residency calls per joint pass | R L (one per row) | L (row 0) |
| window after the pass | the last row's selection | row 0's selection (WS-B risk 7: the following drafts see other hits) |
| blocks of rows 1 .. R - 1 that the window lacks | loaded INTO the window / pool by their own residency | copied (pool hit) or fetched (host) into the scratch STORE, C fills |
| attention calls per layer | 2 R (A + B per row, B entries each) | 2 + C (A, C store calls, B; E = B R entries each) |
| a draft row r >= 1 | its own draft residency (pool hits swapped in, misses skipped) | reads the window, its pool hits via the store, and blocks a verify row fetched |
| GEMM of the joint pass | rowgroup (r2_wsa_cell.run_arm) | the cell's --tu-gemm, default natural (WS-B production); L4PTUR = rowgroup (optional arm) |
| D1 (no residency undo), D3 (no deferred recompute), commit / reset, schedule of contract 4 | yes | the same code (TentState) |

## 4. CPU tests (task steps 2 and 3)

The world `tests_cpu/i2_world.py` (over wsa_world, wsb_world, spec_world): `tentu_builder` / `run_tentu` (WS-A's
`run_tent` with a TentRunnerU built with `wsb_world.fake_attention_u`: a store call can have length 0), `count_joint_diff`
(an INDEPENDENT count of `kernels["diff"]` inside every joint pass), `run_teacher_cpu` (the check cell on the CPU),
`swapped` / `old_module_as` / `old_chain` / `world_modules` (an older commit's modules as one coherent set in the
synthetic package, also for spec_round's run-time `verify_rows` import), `same_run` (two runs bit for bit). The test file
`tests_cpu/test_i2.py` loads WS-A's test file as a library (`wsa_tests_lib`) for spec_trace, the row checks and the cost
run.

| test | claim | negative control / preconditions (L34) |
|---|---|---|
| `test_i2_gate0_files` | (a) the tree = 40ac348 except WS-A's, WS-B's and I2's declared files; (b) WS-A's kept files = 83dac96 bytes, WS-B's kept files (verify_rows.py, spec_round.py, cell, step script, world, notes) = 1a8ce68 bytes; (c) test_spec_prefetch.py: changed lines = both sides' sets exactly; (d) the four older Gate-0 tests add only lines naming spec_tentative_union; (e) test_wsa.py / test_wsb.py: only their Gate-0 function changed (AST); (f) spec_tentative.py = 83dac96's AST except the declared seams (added: `_rows`, `_resid`, `_write`, `_joint_ops`, `_make_state`; changed: docstring, `_layer`, `_row`, `__init__`); no untracked file in nosi/; (g) spec_tentative_union.py names none of spec_round / ragged_rollback / state_snapshot / spec_prefetch; only it and spec_tentative.py name it; no older cell names it | - |
| `test_i2_gate0_tent_arms[L4PT-B1-full, L4PT-B8-full, L4PT-B8-st, L3T-B8-full, L4PJ-B8-full, L4PJ-B8-st, L4P-B8-full]` | every arm of WS-A's branch (L4PT, L3T, L4PJ, the legacy L4P) BIT FOR BIT against the 83dac96 source: committed tokens, fingerprints, logits rows, fp_ok, read() without timing (an older verify_rows has no WS-B host counts: `calls` / `passes` dropped from read()["verify"] in the full chain), the round records, engine and twin state, the tentative buffers, tpos, qc, ctr_c, the layer-0 row writes. "full": spec_tentative + spec_round + verify_rows of 83dac96 as one set; "st": spec_tentative alone (isolates the seams) | precondition: the old module's SR / VR are the chain's (asserted); rejects and full accepts ran |
| `test_i2_gate0_wsb_arms[L4X, L4XU]` | WS-B's arms bit for bit against the 1a8ce68 source (verify_rows + spec_round) | both accept paths ran |
| `test_l4ptu_commits_sh_tokens[B1, B8, B8nat]` | L4PTU commits SH's tokens (MUST), exactly N per request; logits within TOL 1e-4 (printed); every committed verify / target row's selection per layer and head = SH's (the reference: bitwise SH's logits, asserted); residency and host rows after every round; errors 0; dropped 0, unserved 0, vis_bad 0, per row visible = selected - skipped. B8nat: gemm natural | rejects and full accepts, an edge crossed, rows past an edge, F4 fill rows, draft and forced skips, pool moves; B 8: ragged positions, store entries (extras > 0, fills used) |
| `test_l4ptu_one_residency_per_layer` | per joint pass L diff calls (independent count) = the module's joint_resid / union_layers / vs.calls["resid"] | CONTROL union=False (L4PT's rows): (K + 1) L per pass, SH's tokens, the cell's L4PTU-RESID gate FAILS |
| `test_l4ptu_deferred_and_residency[B1, B8]` | D3: no deferred-row write, K + 1 rows per joint pass, one replay per group from the draft row's entry, the J slots write exactly the trace's fed positions, twin position = rope position, remade 0; D1: residency_undo 0 | PLANT recompute_deferred: K + 2 rows and deferred writes (CAUGHT), SH's tokens |
| `test_l4ptu_rollback_fixed_cost` | WS-A's cost run with L4PTU: one torch-call count per reset and per commit over two runs; a reset writes no engine table; dropped = the recount from the buffer writes; the union pass ran | different dropped counts; the maps changed between resets; PLANT undo_residency: CAUGHT |
| `test_l4ptu_schedule_trace` | the contract-4 trace = WS-A's plain-rule model (spec_trace), exactly N, the rows that ran = the trace, after the final group only target / idle, SH's tokens | corrections, refills, B finals, target rows |
| `test_l4ptu_store_and_union_plants` | X 4: spill > 0, dropped 0, unserved 0, vis_bad 0, SH's tokens; the cell's PRESENT / VIS / USTRESS gates pass and IDX fails without VRF-IDX evidence | PLANTS drop_spill (unserved, dropped > 0), union_attention (vis_bad > 0, logits leave SH's), short_store_len (vis_bad > 0, dropped 0): CAUGHT |
| `test_l4ptu_committed_state[B1, B8]` | WS-A's state classes against SH fed the committed tokens: EXACT fields equal, ROUNDED fields layer-0 bitwise and within TOL; complete committed blocks = SH's host rows | rejects, an edge crossed |
| `test_l4ptu_buffer_lifetime` | dead buffer entries poisoned change no token, logits within TOL | PLANTS no_causal, no_lower (act on the union call B too): CAUGHT |
| `test_l4ptu_teacher_forced_vs_sh[B1, B8]` | the check cell on the CPU: 6 teacher-forced passes; every row's selection SET = SH's (every layer, head, request); argmax = SH's; logits within TOL; dropped 0, unserved 0, vis_bad 0, visible = selected; L residency calls per pass (independent count); VRF-IDX A, T (and B with extras) bitwise; the cell's TUC gates pass (B 8) | an F4 fill row and rows past an edge inside the passes; B 8: extras; CONTROL changed=1: TUC-SEL FAILS |
| `test_l4ptu_store_bytes_and_tentative_plant` | X 4, recorded: every selected block of every row is in the window or a store slot with its host rows and cis rows, or it is block qb / the tentative block (wsb_world.union_bytes) | spill > 0; PLANT list_tentative (the tentative block in the union list, B 1, the 5th pass is the first past the edge): vis_bad > 0 (CAUGHT) |
| `test_i2_wsg_positive_control_and_arms` | WS-G's check on the stand-ins (filter "end": corrections, finals, target rows in D slots): WJ (L4PJ) passes every part incl. BITS (MUST WJ-CHK PASS); WT and WU pass every part (REPORT gates); the capture changes no token, fingerprint or counter of WU; every wrapper is back after the run | D-slot commits ran; slots, REF self-check and SEL were compared |
| `test_i2_wsg_planted_control` | WUP (L4PTU, drop_spill, X 4): MISS > 0 and the check FAILS; WUP-PLANT-CHK-PRE and -CAUGHT pass (MUST) | precondition: unserved > 0 (the plant dropped blocks) |
| `test_l4ptu_refusals_and_no_host_sync` | refusals (level L3, unknown uneg / vneg, spill 0, extra 1, unknown tneg) before the meter attaches; no host read in a round from spec_tentative_union.py or verify_rows.py; spec_tentative.py only in `_run_paper` / `_ps_after` | the trap is not blind (the per-round sync is seen) |
| `test_i2_cell_arguments`, `test_i2_cell_report_and_gates`, `test_step_script_i2` | the cell's arguments and refusals; summarize / gates on CPU runs of L4PT and L4PTU (the WS-A gates renamed L4PTU-*, RESID, IDX not blind); verdicts REPORT while PRED_I2 is None, the per-batch entry overrides it (RATIO, PLAN-15, AGREE, ABOVE-I1); the step script: wrong cells STOP, DRYRUN prints the cell line and arguments, i2a / i2b STOP while PRED_STAMP_I2 is None | - |

If a precondition fails, the data do not exercise the path: tell me and I change the data, not the claim.

## 5. The GPU cells (task step 4)

`benchmarks/Efficiency/r2_i2_cell.py` (SH first; timing arms L4PT = `r2_wsa_cell.run_arm` UNCHANGED (the wa1 arm),
L4PTU = `run_arm_u`; optional L4PTUR (rowgroup GEMM) and L4PJ; check arms SHR = `r2_wsb_cell.run_shr`, TUC / TUCR =
`run_check_tu`), `benchmarks/Efficiency/r2_i2_step.sh` (a copy of r2_wsa_step.sh; ROOT this tree; claim
`/mnt/beegfs/ojerbi/r2-runs/spec/sp_active_<holder>`, the same as the spec / WS-A steps; output `i2_<run>_<holder>_<cell>`).

| cell | C (pool) | B | steps | arms after SH | X | passes | check / stress | STEP_MIN | mem | GPU time (MODEL) |
|---|---|---|---|---|---|---|---|---|---|---|
| i2m | 73 (10) | 8 | 64 | L4PT, L4PTU, SHR, TUC | 4 (C 64: the spill runs) | 15 | 1 / 1 | 45 | 100G | 13-18 min (5 prefills of about 2 min; X 4 is slow by design, wb1m: joint 58 SH steps) |
| i2a | 128 (65) | 216 | 512 | L4PT, L4PTU | 48 (C 5) | - | 0 / 0 | 70 | 240G | about 40 min (3 prefills of 11 min) |
| i2b | 73 (10) | 318 | 512 | L4PT, L4PTU | 48 | - | 0 / 0 | 95 | 340G | about 55 min (3 prefills of 16 min) |
| i2ca | 128 (65) | 216 | 129 | SHR, TUC | 48 | 32 | 0 / 0 | 70 | 240G | about 38 min |
| i2cb | 73 (10) | 318 | 129 | SHR, TUC | 48 | 32 | 0 / 0 | 95 | 340G | about 53 min |

Total about 3.3 h. All cells: `--spec 3 --joint-residency union --tu-gemm natural --spec-journal-blocks 256 --spec-twin
fused --spec-body fused --spec-graph on --spec-redirect on --probe-every 16`, the inputs of sp3a / sp3b / wa1a / wa1b
(workload pin, S0, document rows, prefill). L4PT and L4PTU run in ONE job, so the rung's incremental change is measured
there. Order: i2m first (it exercises every new GPU path: the per-entry call B with cache_batch_idx on the 64-row buffer,
VRF-IDX tag T, the store with spill, the teacher driver), then the check cells, then the timing cells after the
registration.

MUST gates: SH-RUN; <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN (WS-A's, the L4PTU ones renamed); L4PTU-RESID (one
residency per layer and joint pass; the union path ran for every joint pass); i2m also L4PT-STRESS, L4PTU-STRESS,
L4PTU-USTRESS, -PRESENT, -VIS, -IDX, TUC-STRESS; check arms SHR-RUN / -FP / -TOK, TUC -RUN, -ERR, -SEL, -PRESENT, -VIS,
-RESID, -IDX, -IDXT. REPORT per timing arm: WS-A's record (tok/s median / mean / effective vs SH and vs 3,407.2; round
phases reset, joint, commit, draft; rollback per round; calls; bytes; memory and fitting batch; tokens against SH;
schedule record); L4PTU also the joint pass in SH steps, the host calls per joint pass, the union counters, and the
PROBE-PASS PHASES of the union rows (resid, write, union, fetch, attn, spill, buffer, merge, post; groups; the complete
layer time). Check arms: the selection changes with symmetric-difference sizes, logits deltas per row, the stats, ready /
residual blocks, one profiled pass.

REGISTERED BLOCK (r2_i2_cell.py): `PRED_STAMP_I2 = None`, `PRED_I2` (flat, every value None) and `PRED_I2_BY_B` = {216:
all None, 318: all None} (the per-batch entry overrides the flat block key by key, as r2_wsa_cell.PRED_WSA_BY_B). Keys:
I2_ratio, I2_vs_I1, I2_joint_sh, I2_joint_saving, I2_rollback_sh, I1_ratio, I2_agree (floor), I2_plan (the plan's point;
verdict I2-PLAN-15 = within +-15 %), I2_above_I1 (the rung falsifier). The step script STOPS i2a / i2b while
PRED_STAMP_I2 is None.

### 5.0 WS-G's check (cell i2g; scope addition of the lead, 2026-10-28T13:19Z)

`benchmarks/Efficiency/r2_wsg_check.py` is WS-G's module of b1ae579 (nosi-r2-wsgreg), copied UNCHANGED with
`git checkout b1ae579 -- <path>` in this tree (test_i2 (h) checks its bytes); the other tree was not touched.
WS-G's FastCapture maps "rows 0 .. m - 1 of a joint pass, m = the count increase" and wraps verify_rows.joint_pass. That
map is wrong for contract 4 (D-slot target rows also commit) and TentRunner never calls joint_pass. So
`benchmarks/Efficiency/r2_i2_check.py` adds `ContractCapture(FastCapture)`: the same stores and result keys (so
r2_wsg_check.run_ref, chk_parts, chk_gates apply unchanged), but rows are mapped BY POSITION (column = p - S; a row of a
round commits its column if count0 <= column < count1; the last row of a column decides its selection and its logits;
tentative runners give the verify masks). MISS (tentative arms): every committed id of a verify row's residency is in
the window (TentState._resid wrapped) plus L4PTU's presence recount ("dropped"). POS (tentative): T.qc = S + count and
T.ctr_c[0] = T.qc (the twins stand at the frontier between rounds by design). REF and the parts: WS-G's.

Arms (cell i2g, wgcm's shape: C 73, B 100, 64 steps, slot check every 4 rounds; STEP_MIN 90, --mem 200G; MODEL about
45-55 min: wgcm's SH + 2 x (arm + REF) took 26:47, here SH + 4 x (arm + REF)):

| arm | runner | gates |
|---|---|---|
| WJ | L4PJ (PaperJournalRunner, EXACT) | WJ-CHK MUST (the positive control: MAP, REF, SEL, GREEDY, POS, SLOT, BITS) |
| WT | L4PT (TentRunner) | every gate REPORT |
| WU | L4PTU (TentRunnerU, union checks on) | every gate REPORT |
| WUP | L4PTU with VerifyOps neg drop_spill at X 4 (the planted control) | WUP-CHK REPORT; WUP-PLANT-CHK-PRE (MISS > 0) and WUP-PLANT-CHK-CAUGHT MUST |

Plus <arm>-RUN, -ERR, -COMMIT-N (MUST, structural, as wgcm's RUN / ERR / COMMIT). Tie tolerance 0.0 (WS-G's
registered CHK_TIE_LOGIT). The tentative check arms run with check_every 1 (a TentRunner fingerprints its committed rows
only on check rounds; MAP reads them).

    DRYRUN=1 bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2g
    bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2g

### 5.1 Inferred bands (INFERENCE, for the lead to register or change; not registered by me)

MEASURED inputs (SCORING_specp_fix.md section 7, same configuration, same-job SH): SH step 64.484 ms (3,349.7 tok/s) at C
128 B 216, 124.003 ms (2,564.5 tok/s) at C 73 B 318; legacy L4P joint 424.233 / 660.395 ms (5 rows; the deferred row
+93.498 / +120.471), drafts 154.959 / 210.848 ms for 2 (77.48 / 105.42 each, with the journal hooks); L4X joint 330.735 /
539.924 ms (4 EXACT rows, rowgroup); yield 2.947 / 2.905 committed per request-round (contract 4 is -1 to -2 %: NOTES_wsa
section 4, choice 2; I use 2.90 at both C). MEASURED parts of a 4-row joint pass at C 128 B 216 (SCORING_vr1.md, cited by
NOTES_wsb.md section 6): GEMM + norms (natural, 864 rows) 69.5 ms, twin 35.6 ms (torch; fused about 4 ms, INFERRED from
rg1), selection of 4 rows 25.6 ms, FAST attention + masks + merge 72.5 ms (call A + one extras call), one shipped
residency about 14.8 ms per pass; rowgroup GEMM costs +27.0 ms over natural (CHECKPOINTS D2 row). MODEL (NOTES_wsa.md
section 7): L4PT joint 270-330 ms (central 300) at C 128, about 470 ms at C 73; L4PT drafts about 60 / 85 ms each
(no hooks, plus call B); reset + commit 3-7 ms. The plan: I2 = 0.52 (C 128) / 0.63 (C 73); WS-B's joint about 184 ms (2.85
SH steps) / 398 ms (3.2).

L4PTU JOINT PASS, C 128 B 216 (MODEL): GEMM natural 69.5 + twin and the replay about 5 + selection 25.6 + one residency 14.8
+ union list and sorts about 4 + store fills (C 5, about 30 launches per layer) about 7.5 + attention (vr1's 72.5 + the
extra store calls and call B about 10 = 82.5) + buffer writes about 3 = about 212 ms = 3.29 SH steps. Band: 168-271 ms
(2.6-4.2 steps): the attention and the launch-bound small ops are not measured in this layout.

L4PTU ROUND, C 128: reset + joint 212 + commit about 5 + 2 drafts 120 = about 337 ms (band 281-421). Tokens per round
216 x 2.90 = 626.4 -> 626.4 / 0.337 s = 1,859 tok/s -> 0.555 x SH (band 1,488-2,229 tok/s -> 0.44-0.67). L4PT round
about 0.3 + 300 + 5 + 120 = 425 ms -> 1,474 tok/s -> 0.44 (the lead's registered L4PT band 0.32-0.52). I2 / I1 = 425 /
337 = 1.26 (band: joint saving 0-170 ms on a round of 425 -> 1.0-1.67, narrowed by the correlated drafts to 1.02-1.55).

C 73 B 318 (MODEL): GEMM natural over 1,272 rows about 80 (compute-bound 0.056-0.063 ms per row) + twin about 6 + selection
25.6 x 318 / 216 = 38 + residency and host fetch (row 0's misses and the store's misses; the L4P WAIT is 140.4 ms per
round, NEW 101,136 blocks = 3.3 GB per round: PCIe-bound) about 130 + attention 72.5 x 318 / 216 = 107 plus about 11 +
list, fills, writes about 20 = about 392 ms = 3.16 SH steps (band 330-480 ms, 2.6-3.9). Round: 392 + 6 + 2 x 85 = 568 ms
(band 495-690); tokens 318 x 2.90 = 922.2 -> 1,624 tok/s -> 0.633 x SH (band 1,337-1,863 -> 0.52-0.73). L4PT round about
470 + 6 + 170 = 646 ms -> 1,428 tok/s -> 0.557. I2 / I1 = 646 / 568 = 1.14 (band 0.98-1.45: at C 73 the host fetch
dominates both, and the scratch store refetches what the serial residency kept, WS-B risk 5).

| key | B 216 (C 128) | B 318 (C 73) | basis |
|---|---|---|---|
| I2_ratio | (0.43, 0.67) | (0.50, 0.75) | the round MODEL above; the plan's 0.52 / 0.63 inside |
| I2_vs_I1 | (1.02, 1.55) | (0.98, 1.45) | the joint saving on L4PT's MODEL round |
| I2_joint_sh | (2.6, 4.2) | (2.6, 3.9) | the joint MODEL; WS-B plan 2.85 / 3.2 inside |
| I2_joint_saving (ms) | (0.0, 170.0) | (0.0, 200.0) | L4PT joint 270-330 / about 470 minus the L4PTU joint |
| I2_rollback_sh | (0.02, 0.30) | (0.01, 0.30) | D1 is kept: the lead's L4PT bands |
| I1_ratio | (0.32, 0.52) | (0.40, 0.62) | the lead's registered PRED_WSA_BY_B L4PT_ratio |
| I2_agree (floor) | 0.25 | 0.25 | FAST float order (also the union store parts and the natural GEMM); L4F agreed 0.35-0.38 (SCORING_specp_fix.md section 6); WS-A's floor |
| I2_plan | 0.52 | 0.63 | PLAN_L4P_faithful.md ladder, row I2 |
| I2_above_I1 | True | True | the plan's rung falsifier |

If wa1a / wa1b (WS-A's GPU cells, queued) land first, their measured L4PT joint and round replace the L4PT MODEL above;
then I2_vs_I1 and I2_joint_saving should be recomputed before the stamp (no threshold moved after an I2 result).

## 6. Commands for the lead

CPU, on a CPU node inside a job, in the image (`/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif`), from
`/mnt/beegfs/ojerbi/nosi-r2-i2`, one file per line:

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i2.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsa.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsb.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_spec_paper.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_spec_round.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_rollback.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_verify_rows.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_ragged_engine.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_draft_pool.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_mbase.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_tick.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_r2_base.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_r2_capture_step.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_r2_p3.py

test_spec_prefetch.py builds the C++ packer (as before). The heaviest of the new tests (INFERENCE): the Gate-0 arm runs
(9 pairs of B 8 runs) and the X 4 runs (C 64 store fills per layer). test_step_script_i2 runs the full DRYRUN only at a
clean checkout of this tree at HEAD (else it skips).

GPU, after the CPU suite is green (a clean checkout at the I2 commit <sha>; holder <h>; run id <r>):

    DRYRUN=1 bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2m
    bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2m       # the miniature first
    bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2ca      # the check, C 128 (not timed)
    bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2cb      # the check, C 73
    bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2b       # timing C 73 (after PRED_STAMP_I2)
    bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <h> <r> i2a       # timing C 128 (after PRED_STAMP_I2)

### 6.1 CPU runs (MEASURED, amd partition; one srun per line of section 8)

| log (/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/) | commit | file(s) | summary line |
|---|---|---|---|
| i2_1fd7c9c_a.txt | 1fd7c9c | test_i2.py | 3 failed, 28 passed (the name-scan expectation; `--arms` commas escaped by printf %q; WJ MAP / BITS: a stale J-slot logits row at a column that a D-slot step committed) -> fixed in 55551c3 |
| i2_1fd7c9c_b.txt | 1fd7c9c | test_wsa.py | 24 passed, 1 skipped (WS-A's DRYRUN needs its own tree) |
| i2_1fd7c9c_c.txt | 1fd7c9c | test_wsb.py | 26 passed, 1 skipped (WS-B's DRYRUN) |
| i2_1fd7c9c_d.txt | 1fd7c9c | test_spec_prefetch.py | 23 passed, 1 skipped |
| i2_1fd7c9c_e.txt | 1fd7c9c | test_spec_round.py, test_spec_paper.py, test_rollback.py, test_verify_rows.py, test_ragged_engine.py | 118 passed, 4 skipped (other trees' DRYRUNs) |
| i2_55551c3_a.txt | 55551c3 | test_i2.py | see the report |
| i2_e6a1cf9_a.txt | e6a1cf9 | test_i2.py | see the report |

The logs start with container noise (`"sed": executable file not found in $PATH`): the image's start scripts look for
tools that the compute PATH does not hold; no test calls them.

## 7. Risks and open points

1. The GPU cells did not run. The CPU stand-ins do not establish GPU dispatch, ordering, timing or memory.
1b. Review (a read-only reviewer agent on 1fd7c9c's I2 source): one BUG (the name-scan expectation, fixed in 55551c3);
   RISKS taken: the visible count now reads call B's own-block mask; the registration test now accepts a registration
   (e6a1cf9). RISKS kept: (i) in L4PTU only row 0 of a joint pass runs TentState._resid, so the joint draft row's skipped
   misses do not reach T.skipped while draft_rows counts the row: `draft.skip_per_stream_per_call` of L4PTU is lower and
   does not compare with L4PT's; forced / injected count row 0 and the one-row passes only; (ii) VR.attach's own
   refusals come after the KVMeter attaches (as in L4PT). NITS kept: TentState._layer has no closing mark, so layer_ms
   gives L - 1 spans and the last layer's "post" is dropped; the _evu cache key is not normalised (two equal runs).
2. TUC-SEL (MUST) WILL PROBABLY FAIL ON THE GPU, as WS-B's UC-SEL did in wb1m (2,685 of 32,768 sets changed, mostly
   size-2 swaps, none at layer 0; STATUS_LOG 11:20:48Z): float order moves near ties. TUCR (rowgroup) removes the GEMM
   part. WS-B2's near-tie discriminator (9688cee) is NOT merged here (it changes WS-B's cell only); the lead decides the
   rule. I did not weaken the gate.
3. L4PTU's tokens against SH are REPORT (FAST: WS-G's accuracy gate decides).
4. A new GPU combination: call B is flash_attn_nosa over E entries on the 64-row tentative buffer with cache_batch_idx and
   the per-entry bias view (WS-A used it without cache_batch_idx, WS-B on its store). VRF-IDX tag T checks it (i2m and the
   check cells, MUST L4PTU-IDX / TUC-IDXT).
5. The merge of the GEMM change: L4PTU uses gemm natural (WS-B's production), L4PT rowgroup. The I2 change therefore
   includes about -27 ms of GEMM at C 128 (MEASURED in vr1). The optional arm L4PTUR (rowgroup) separates it: add it to
   the arms of i2a / i2b if the attribution is needed (about +15 min per cell).
6. The window after a union pass holds row 0's selection; the following D-slot drafts may see fewer window hits (more
   skipped blocks, possibly lower acceptance). And a draft row r >= 1 reads blocks fetched for verify rows (better drafts).
   Both change acceptance, not correctness. The acceptance record shows it.
7. The store is scratch: a host miss fetched for rows 1 .. R - 1 is not kept; the next pass fetches it again if needed
   (WS-B risk 5). At C 73 this can raise the NEW blocks per round against L4PT (whose serial residency keeps them). The
   KVMeter counts the store's host loads (its source is the host rows), so kv_at_verify shows it.
8. Launches: about 260 small ops per layer and joint pass (MODEL), about 8,300 per pass; at 10-20 us per launch this is
   83-166 ms of host time against about 212 ms of device time at C 128: probably hidden, not at small B (the miniature).
9. The C spill fills run on every layer (C 5 at X 48) even when empty (WS-B risk 6).
10. A store host miss gets no staging redirect (WS-B risk 13). Here it is safe: the union list excludes every block at or
    above qb (the tentative block has no host rows; block qb is in the tail slot); the committed blocks below qb were
    written back by an earlier commit on the same stream.
11. The probe-pass phase "post" is coarse: it covers the dense GEMMs after the attention and the next layer's GEMM, twin
    and selection (TentState._layer has no marks; I did not add any to WS-A's code).
12. Gate 0 loads older modules into the synthetic package under tagged names and swaps sys.modules entries while an old
    run executes (spec_round imports verify_rows at run time). If the image's Python resolved relative imports otherwise,
    the precondition asserts of test_i2_gate0_tent_arms fail (an error, not a silent pass).
13. The miniature's L4PTU-PRESENT / -VIS / -IDX gates read check-only counters of a FREE run (not teacher-forced): they test
    the structure (presence, visibility) on the production schedule.
14. Memory (MODEL): the store at X 48 is 0.68 GB at B 216 and 1.0 GB at B 318; call A's per-entry bias is 14 / 21 MB per
    layer (transient). L4PT's peak is not measured yet (wa1 queued); L4P peaked at 83.187 GB at B 216 with the journal
    (about 2.3 GB more than L4PT, NOTES_wsa section 8).

## 8. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc / date (this task)

- `cd` (shell built-in) in front of read commands; `echo` (built-in) for separators; `&&`, `;`, `|`.
- bash process substitution `<( ... )` in one read-only `diff` of git output (no file written).
- git: `git worktree add` (as instructed), `git merge --no-ff --no-commit`, `git commit -F` (messages written with the
  Write tool), `git checkout b1ae579 -- benchmarks/Efficiency/r2_wsg_check.py` (in this worktree: the exact WS-G blob),
  read-only git commands.
- `squeue -u ojerbi` (read-only, twice).
- `timeout 1500 tail -f /dev/null` in the background, once, to wait for the reviewer agent.
- srun (the human's permission of 13:19Z; the lead's template; `export PATH=...bin-compute:$PATH` first; the output
  redirected to the log by the shell, as the template does), seven times:
  `srun -p amd -c 16 --mem=64G -t 240 apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi --env
  TMPDIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc,TORCH_EXTENSIONS_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i2_ext_<p>
  /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif bash -c "cd /mnt/beegfs/ojerbi/nosi-r2-i2 && env PYTHONDONTWRITEBYTECODE=1
  /venv/nosa/bin/python -m pytest -q [-s] -p no:cacheprovider -rfEs <file(s)>" > .../cpu-logs/i2_<sha7>_<p>.txt 2>&1`
  with (p, files): (a, test_i2.py) at 1fd7c9c, 55551c3, e6a1cf9 (logs i2_<sha7>_a; the e6a1cf9 run used ext dir f);
  (b, test_wsa.py), (c, test_wsb.py), (d, test_spec_prefetch.py), (e, test_spec_round.py test_spec_paper.py
  test_rollback.py test_verify_rows.py test_ragged_engine.py: five CPU stand-in files in one srun) at 1fd7c9c.
- No interpreter on the login node, no sed, no awk, no heredoc, no `cat >>`, no tr / paste / uniq / xargs, no hashing,
  no `find`, no sbatch, no GPU job. Files were written with the Write and Edit tools (and the git checkout above).
