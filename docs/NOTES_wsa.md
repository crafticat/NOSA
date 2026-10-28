# NOTES: WS-A, the tentative buffers (D1 + D3) and the schedule of contract 4 (`--spec-rollback tentative`)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_L4P_faithful.md`, workstream WS-A, the component
  contracts 1 and 4 (the human decisions (a) and (b) of 2026-10-28).
- Paper: `/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/speckv/speckv.txt`, sections 3, 3.3, 3.4 and Figure 3.
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-wsa`, branch `r2-wsa`, from `40ac348` (nosi-r2-spec5). No other worktree was touched
  (nosi-r2-spec5 40ac348, nosi-r2-specp a6c9d10 and nosi-r2-spec fd1a239 keep their HEADs).
- Status: SOURCE ONLY. I did not run a test, an interpreter or a job. The lead runs the commands of sections 6 and 7.
- Style: ASD-STE100. Labels: CODE = read at file:line; MODEL = hand arithmetic; INFERRED = my reading.

## 1. What I made

| file | what |
|---|---|
| `nosi/nosi/spec_tentative.py` (new) | `TentState` (the tentative buffers, `commit`, `reset`, the check-only poison), `PaperSchedule` (the host planner of the schedule of contract 4, the position trace, the token accounting, the run loop), `TentRunner` (FAST: L4P on that schedule, L3 as a reference), `PaperJournalRunner` (EXACT: spec_round's journal on the same schedule) |
| `benchmarks/Efficiency/r2_wsa_cell.py` (new) | the GPU cell `wa1`: SH, L4P (legacy, as sp3a / sp3b), L4PJ (journal, contract 4), L4PT (tentative, contract 4); the contract counters; the REGISTERED block `PRED_WSA` (TBD) |
| `benchmarks/Efficiency/r2_wsa_step.sh` (new) | a copy of `r2_spec5_step.sh`: ROOT this tree; cells wa1m, wa1a, wa1b; the same claim `sp_active_<holder>` |
| `benchmarks/Efficiency/tests_cpu/wsa_world.py` (new) | the CPU world: `tent_runner`, the WS-G `Recorder`, `reference` (SH fed the committed tokens, with its selections), `run_tent`, the state classes, `OpCount`, the journal shim `JOURNAL` |
| `benchmarks/Efficiency/tests_cpu/test_wsa.py` (new) | the CPU tests (section 6) |
| five older Gate 0 tests | only ADDED lines that name spec_tentative: test_ragged_engine, test_verify_rows, test_rollback, test_spec_round (an exclude line; `src.pop("spec_tentative.py")` before its name scan, because spec_tentative imports spec_round), test_spec_prefetch (the WS-A files in its DECLARED list; its own per-file check skips the added spec_tentative lines) |

NO NEW KERNEL. NO KERNEL EDIT. `spec_round.py`, `ragged_rollback.py`, `verify_rows.py`, `ragged_engine.py`, every engine
file and every older cell are the bytes of 40ac348. The new code uses the shipped kernels (diff_offload, pool_update,
flash_pool_swap, flash_h2d_from_mask(_bias), r2_writeback.masked_writeback, flash_attn_nosa_with_kvcache with
`return_softmax_lse`) and torch ops (`torch.where(..., out=)`, `torch._foreach_copy_`).

## 2. The design (paper 3.3, 3.4): FAST, `TentRunner`

THE STATE per request: the committed position `qc` (the committed cache holds positions below it: window tail slot, host
rows, twin rows below the committed counters), the CHAIN of tentative positions `qc .. frontier - 1` (fed in position
order), the committed twin counters `ctr_c` (the same in every layer).

THE BUFFER per layer: K, V `(B, 64, H, D)` and the bias `(B, 64, H)` (the cis row, as the tail row holds it). Position x is
row `x % 64` (its tail-row index). `tpos (B, 64)` (shared by the layers) holds the position of each row. A row at position
p reads the entries with `qc <= tpos <= p`. Every position in `[qc, p]` was written by the current chain before the row at
p reads it, because the chain feeds positions in order; the entries of a dropped chain lie above the frontier. The
buffers start as zeros, not `torch.empty`: a masked entry still enters the kernel's sums with weight 0, and `0 x NaN` is
NaN (the pitfall "undefined torch.empty regions").

THE COMPRESSED SUMMARIES (the plan's "compressed-summary appends"): a tentative row's appends (the twin's key ring row,
its `total_cis` row, a due `comp_k` / `ccis` row) go into the twin's preallocated rows ABOVE the committed counters; the
rows below them are never written by a tentative row (r2_positional_twin, ROLLBACK: a step writes only at indices at or
above the counters it starts from; the ring is exact within MAX_ROLLBACK). These rows are the tentative buffer of the
summaries; "restore the summaries to the accepted prefix" is the counter reset below. The committed counters of the m
accepted positions come from a layer-0 snapshot after each joint-pass row.

ONE TENTATIVE ROW (draft row, verify row, target row), per layer (CODE spec_tentative.py `TentState._layer`, `_row`):
1. the GEMMs, nosa_linear, rope (a one-row pass: the ragged step's GEMM; the joint pass: verify_rows' rowgroup GEMM);
2. the twin step; the draft row of a group first REPLAYS the deferred position (`twin.step_unchecked` with the K and the cis
   row from its buffer entry: no GEMM, no residency, no attention: D3);
3. the selection (one row: ragged_engine's stage 1 and its captured pooling graph; the joint pass: `_select_rows`);
4. the RESIDENCY on the COMMITTED window, in the shipped order (diff_offload, the inactive mask, pool_update,
   flash_pool_swap, the LRU clock, the map). It stays: it is never undone (D1). A draft row skips every miss (Task 1.1's
   draft residency). Every row skips the TENTATIVE BLOCK (an id above `qc // 64`: past a block edge a row selects the block
   it writes, which has no host rows yet). The diff needs 64 distinct ids and must keep slot 63 (the committed tail): if a
   head of a row past the edge does not select `qc // 64`, that id takes the place of the tentative block id, and the head
   masks slot 63. (A `-1` in the selection is not safe: the diff kernel assumes a bijection, CODE
   flash_cache_engine/diff_offload_kernel.cu:35-98.)
5. its K, V and cis row go into the buffer at row `p % 64`, and `tpos` (layer 0). Never into the committed tail slot.
6. attention: call A on the committed window (a scratch bias that masks the skipped slots; the cut `base + min(qc % 64,
   n_tail(p))` in block `qc // 64`, `base + qc % 64` past its edge, with `n_tail(p)` the shipped attention length of the
   step at p minus `base`: findings F4 kept) and call B on the buffer (entry x visible iff `qc <= x <= p`, `x % 64 <
   n_tail(p)` in p's block, and, past the edge, the committed block's entries only for a head that selects it), merged by
   their log-sum-exp in float32 (verify_rows FAST's merge).

ACCEPT (`TentState.commit`): the first m positions of the chain become committed. Per layer: one masked `where(out=)` per
tensor puts their buffer rows of block `qc // 64` into the tail slot (the buffer row and the tail row of a position have
the same index); the shipped device-driven write-back writes a block that the commit filled (only committed rows: the
tail slot holds the old committed rows and the rows just put there); one masked `where(out=)` per tensor puts the next
block's committed rows into the tail slot; the tail id and the attention length. Then `qc += m`, `ctr_c <- the counters
after m`. The same ops for every m (9 calls per layer and one shared set).

REJECT and ROUND START (`TentState.reset`): the twin counters of the selected requests `<- ctr_c` (one fused copy for all
layers). r2_positional_twin, ROLLBACK: a counter reset is exact while the rollback spans at most MAX_ROLLBACK = 32
positions; the chain spans at most K + 2 (the runner refuses K + 2 >= 32). The rejected entries are dropped by `qc` and the
frontier only. No slot byte, map entry, pool entry, LRU age or host row moves (test_wsa.py COST checks it).

THE COST TARGET (MODEL): the rollback per round = reset (2 calls) + commit (about 300 calls: 9 per layer, 32 layers, a
shared set; its device traffic about 2.7 GB at C 128 B 216: two phases x 3 where() x the tail slot `B x 64 x (2 H D + H)`
bf16 x 32 layers, about 1.8 ms) + the reject's counter set (2 calls). No host sync, no per-layer loop over requests or
positions. At about 10-20 us per call this is 3-7 ms per round: 0.05-0.11 SH steps at C 128 (SH 64.5 ms), 0.03-0.06 at
C 73 (124.0 ms). The plan's target is <= 0.1 SH step (falsifier > 0.3). The C 128 cell is near the target.

## 3. FAST against SH: what differs, and why it is not semantic (the binding correction)

FAST is checked on itself against SH (section 6, TOKENS and STATE), not certified by the EXACT path. A verify / target
row of FAST attends the same SET of (key, value, bias) rows as the shipped step at its position (the selected history and
the tail block's rows that F4 lets the shipped step see), with the same selection; only the slot ORDER and the float
order of the two-call merge differ. The committed state after a round equals the state of committing only the accepted
tokens (the reference fed SH's tokens to exactly the committed count), field by field:

| class | fields | why |
|---|---|---|
| EXACT (must be equal) | twin `pos`, `n_comp`, `n_ccis`; the tail id (map slot 63); `_cache_lens`; host rows past the committed block (still the poison); the residency invariant (every named window and pool slot holds its block's host rows and cis rows, no id twice, no id at or above the committed tail block); the committed counters `ctr_c` = the twin counters after the reset | positions and lengths do not depend on float values |
| ROUNDED (layer 0 bitwise SH's; every layer within TOL = 1e-4) | twin `comp_k[:n_comp]`, `ccis[:, :n_ccis]`, `total_cis[:pos]`, the live ring rows `[16 n_comp, pos)`; the tail-slot rows below `qc % 64` (K, V, bias); the host rows of the complete committed blocks | layer 0 has no attention before it (bitwise); from layer 1 on the attention output differs by FLOAT ORDER (the slot order of the split-KV sums and the float32 merge of the two calls) |
| CACHE (may differ; reported, not semantic) | the window's contents (slots 0-62) and their slot order; the pool's contents and LRU order; the LRU clock `_pool_stamp_base`; the per-step scratch (`_new_block_map_buf`, `_load_mask`, `_pool_target`, `_pool_action`); the bytes of unnamed slots; `st.steps` | a cache: the tentative rows' residency on the committed history stays, so other blocks are resident and in other slots; every named slot holds its block's bytes; the next selection loads what it misses |
| TENTATIVE (not committed state) | twin rows above the committed counters; the twin counters between rounds (the chain's frontier); the buffer entries and `tpos` | scratch of the chain: a reset or the next writes replace them |

The DRAFTS of FAST differ from the journal runner's: FAST keeps the draft step's K / V of the deferred position, the
journal runner remakes them in the joint pass (with the verify rows' target K / V in the context). So acceptance and the
trace of the two runners differ; the committed tokens of both are SH's.

THE LEAD'S TASK SAID "float order equal there" (on the CPU stand-ins). It is not: the CPU attention stand-in sums the
rows in slot order (ragged_world.fake_attention), and FAST's slot order and its two-call merge differ from SH's. So the
CPU claims are: tokens equal (MUST), the logits within TOL (MUST, the maximum printed), the selections equal as sets
(MUST), layer 0 bitwise (MUST). RISK: a selection gate at layer 1 can fail on a near tie of two pooled scores (float
noise about 1e-7 relative); a failure there needs the score gap before it is called a bug.

HOOKS for WS-G (TentRunner, check rounds only): `on_select(l, kind, r, pos, verify, sel)` gives the target selection of
every row and layer (before the tail-id injection); `on_logits(round, kind, pos, verify, logits)` the joint pass's (and the
L3 verify steps') logits. test_wsa.py compares them with SH's.

CHECK-ONLY work (TentRunner, only on check rounds: the probe rounds in the GPU cell, which the steady median excludes):
the logits fingerprints, the hooks, the extended counters (forced / injected skips, rows past an edge, F4 fill rows),
the poison (tests only), the buffer-write record (tests only). REQUIRED work (never check-only): every op of the row, the
commit, the reset, the argmax and the acceptance, `tpos`, the deferred-write counter and the draft-skip counter, the host
plan and the trace (host only, after the round's sync).

## 4. The schedule of contract 4 (decisions (a) and (b)), in both runners

THE ROLES per request and round (CODE `PaperSchedule._ps_plan`): `done` (no token left), `target` (after the final group:
ordinary target decoding, one row per slot), `final` (N - count <= K: one final verification group of `min(u, N - count)`
rows, no deferral, no draft row; the D slots of that round idle), `fill` (u < K: one draft row at the J slot, then the D
slots idle), `group` (u >= K: Figure 3: v = u - 1 verify rows, the newest draft deferred, the draft row; a full accept keeps
`[e_u, e_new]`, a reject starts a NEW FILL from the accepted prefix and the correction). One host plan per round from the
state of the round's one host sync (count, q, u); one host-to-device copy of the roles (on the GPU from pinned memory
with `non_blocking=True`, so this copy does not synchronize the stream).

FIGURE 3 AND (a), K = 3 (positions from q; d_i is the token of position q + i; test_paper_spec_model_literal):

| slot | after the initial fill | after a reject at d1 (q' = q + 1) |
|---|---|---|
| P0 P1 P2 / D D | draft d1, d2, d3 | draft d1', d2' (from the correction) |
| J | verify d1, d2; defer d3; draft d4 | fill: draft d3' |
| D D | draft d5, d6 | idle, idle |
| J | verify d3, d4, d5; defer d6; draft d7 | verify d1', d2'; defer d3'; draft d4' |

DECISIONS AND CHOICES (for the lead):
1. "KEEP THE EXACT JOURNAL PATH'S SCHEDULE IDENTICAL": I read it as "the EXACT proof arm runs the same new schedule as
   FAST, so the two compare like with like". So `PaperJournalRunner` runs contract 4 with the journal (arm L4PJ), and the
   legacy L4P (spec_round.py, unchanged) runs in wa1 as the sp3a / sp3b reference. If the lead meant "do not touch the
   EXACT path", L4PJ is an addition only: spec_round.py and the legacy arm are unchanged either way.
2. THE NEW FILL'S K-TH DRAFT. After a reject the round's K - 1 D slots draft d1' .. d_{K-1}'. The K-th fill draft goes into
   the next J slot (a one-row draft), then the request idles for the K - 1 D slots, and its first group is the next J slot.
   The batch keeps one J slot per round. The other choice, one extra batch-wide draft step per round, costs about one draft
   step in almost every round at B 216 (MODEL: with 1.7 % rejects per request-round, some request rejects in about 97 % of
   the rounds). The position trace is the same either way; only the idle slots differ. Cost of my choice (MODEL): a
   rejecting request loses about one round of progress: about 1.7 % (C 128) and 3.1 % (C 73) of request-rounds, about 1-2 %
   of the committed tokens per round against the legacy restart.
3. THE DRAIN TRIGGER: N - count <= K at the round start. A final group commits at most `min(u, N - count)` verified tokens;
   drafts past it are discarded unverified (counted). The final round's D slots idle (the host does not know m before the
   sync); target rows follow from the next round. Every request commits exactly N tokens (MUST gate COMMIT-N). In FAST a
   J slot whose rows are all one-row (fill, target, a final of one row) runs as a one-row pass (the shipped step's cost).
4. THE JOURNAL RUNNER: a group's joint pass remakes the deferred row (row v at q + v), a fill's J slot remakes its earlier
   fill positions as draft rows (rows 0 .. u - 1) and feeds e_u at row u; idle requests are inactive in the D slots (the
   activity comes from the host plan, no extra sync); a target request's rows are kept by the round's commit. The trace's
   `remade` lists these positions (FAST: none).

THE TRACE (both runners; CODE `PaperSchedule._ps_after`): per request and pass: slot (P, J, D), role, the committed
position q (J slots), the fed positions (verify / target rows, draft rows), the verified tokens' positions, the deferred
token's position, the drafted token's position, the accepted count n, the committed count, the remade positions. The cell
prints the first and the last `TRACE_N = 12` passes of every request; the token accounting (proposed, verified, accepted,
corrected, committed by group / final / target, discarded, discarded unverified, groups, fills, finals, targets, remade
rows).

## 5. Which claims need the GPU cell

The CPU stand-ins model the kernels' arithmetic with flat addresses (L28). They do not establish GPU dispatch, streams,
ordering, timing, memory or bf16 rounding. ON THE CPU (test_wsa.py): the tokens against SH, the logits within TOL, the
selections, the state classes, the schedule trace, exact counts, the D1 / D3 call counts, the host rows, the fixed call
counts of reset and commit, the buffer lifetime. ONLY THE GPU CELL CAN SHOW: the token agreement at bf16 (float order can
flip a near tie: REPORT, the FAST accuracy gate is WS-G's); that `flash_attn_nosa_with_kvcache` accepts a strided scratch
bias with `return_softmax_lse` and no `cache_batch_idx` (call A) and a 64-row cache (call B); `torch.where(out=)` on a
strided view and `torch._foreach_copy_` on the GPU build; the timing (rollback ms, phases, tok/s); the launch-bound cost of
the commit; the memory (decode peak, fitting batch). Ordering: everything runs on the current stream; the write-back stores
through the pinned host pointer on that stream (as the shipped body).

## 6. CPU tests for the lead

Run on a CPU node, inside a job, in the image (`/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif`):

    cd /mnt/beegfs/ojerbi/nosi-r2-wsa
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsa.py

Regressions (the spec suites and the hooks; five of them got added lines):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py benchmarks/Efficiency/tests_cpu/test_spec_paper.py \
        benchmarks/Efficiency/tests_cpu/test_spec_round.py benchmarks/Efficiency/tests_cpu/test_rollback.py \
        benchmarks/Efficiency/tests_cpu/test_verify_rows.py benchmarks/Efficiency/tests_cpu/test_ragged_engine.py \
        benchmarks/Efficiency/tests_cpu/test_draft_pool.py benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py \
        benchmarks/Efficiency/tests_cpu/test_r2_mbase.py benchmarks/Efficiency/tests_cpu/test_r2_tick.py \
        tests/test_r2_base.py tests/test_r2_capture_step.py tests/test_r2_p3.py

test_step_script_wa1 runs the full DRYRUN only at a clean checkout of THIS tree at HEAD (else it skips). The `-s` prints
the measured logit deltas, the state classes and the call counts.

| test | claim | negative control / preconditions (L34) |
|---|---|---|
| `test_wsa_flag_off_is_spec5[B1, B8]` | the tree = 40ac348 except the declared WS-A files; the five older Gate 0 tests only ADD lines naming spec_tentative; no other nosi module names spec_tentative; it names none of ragged_rollback / state_snapshot / spec_prefetch; the cell's flag is off by default; TentRunner / PaperJournalRunner are built only under their flags in `run_arm`; an attached TentRunner / PaperJournalRunner and no round: 16 plain steps bitwise the plain world, `detach()` restores every callable | precondition: a block fill in the 16 steps |
| `test_paper_spec_model_literal` | the test's own plain-rule model gives Figure 3 (fill d1 d2 d3; verify d1 d2, defer d3, draft d4; next group d3 d4 d5, defer d6, draft d7), decision (a) (draft d1' d2', fill d3', idle, then verify d1' d2', defer d3', draft d4'), decision (b) (the final group verifies the pending drafts up to the limit; after its reject a target row; exactly N) | - |
| `test_paper_schedule_trace[FAST-B1, FAST-B8, EXACT-B1, EXACT-B8]` | the runner's trace = the model's, driven by the run's own accepted counts (role, q, fed, verified, deferred, drafted, n, committed per pass and request); remade: FAST none, EXACT the group's q + v and the fill's earlier positions; every request commits exactly N; tokens = SH's across rejects, refills and the end; after the final group only target / idle passes; EXACT: logits rows bitwise, residency and host rows after every round; FAST: logits within TOL; THE TRACE AGAINST THE ROWS THAT RAN (not the host plan): FAST every J / P / D slot's layer-0 row writes and the row count per request; EXACT the joint pass's q and the twin position before every active D slot (check-only `dpos`; the prelude excepted) | CONTROLS: one FAST row and one EXACT slot shifted by one position: CAUGHT; the filter `end` (draft 1 wrong in some rounds; every draft for a column >= N - 3 wrong): corrections, refills, B finals, target rows |
| `test_tent_commits_sh_tokens[L4P-B1, L4P-B8, L3-B1, L3-B8]` | FAST commits SH's tokens; the logits within TOL (printed); the selection of every committed verify / target row, layer and head = SH's (the reference ragged world fed SH's tokens; its logits bitwise SH's, asserted); residency and host rows after every round; no refusal; FAST's tokens = the EXACT run's (L4P: PaperJournalRunner; L3: spec_round); L4P: exactly N | rejects and full accepts, commits across a block edge, rows past an edge, F4 fill rows, forced and draft skips, pool moves, (B 8) ragged positions |
| `test_tent_committed_state[B1, B8]` | section 3's classes: EXACT fields equal, ROUNDED fields layer-0 bitwise and within TOL, CACHE fields printed | rejects and a committed edge crossing |
| `test_tent_deferred_not_recomputed[B1, B8]` | the deferred-write counter 0; joint rows = (K + 1) x joint passes; one replay per group from the newest buffer write of its position, a draft row's, unchanged; the J slots write exactly the trace's fed positions; twin position = rope position for every row; remade rows 0 | PLANTS recompute_deferred (K + 2 rows, the deferred row written: CAUGHT by the counters; a cost plant, the tokens stay SH's) and no_replay (CAUGHT by the position check) |
| `test_tent_reject_keeps_host_rows[B1, B8]` | no host row past the committed block after any round; the complete committed blocks = SH's host rows (layer 0 bitwise, TOL) | the first S in S0 .. S0 + 3 with a chain that filled a block that its commit did not; PLANT writeback_tentative (not capped at 12 rounds, so it reaches the clean run's fill): CAUGHT |
| `test_tent_rollback_is_fixed_cost` (B 8) | torch calls per reset and per commit: one value each over every call of two runs (the device write-back counted as its one launch); a reset writes no engine table (maps, pool, ages, slots, host rows, lengths, clock); the dropped entries = a recount from the layer-0 buffer writes (the written positions not committed that a reject or a final drops; not the runner's formula) | different dropped counts in the two runs; the maps changed between resets; PLANT undo_residency: CAUGHT |
| `test_tent_buffer_lifetime[B1, B8]` | the dead (published or dropped) entries poisoned (V = 1e3) change no token and no logit beyond TOL | the poison acted; PLANTS no_causal and no_lower: CAUGHT |
| `test_tent_refusals_hooks_check_rounds` | refusals of both runners before any hook attaches; check_every 0: no hook, no fingerprint past the warm row; check_every 2: logits hooks on even rounds > 0 only, shapes; on_select gives (H, B, topk) in every layer | - |
| `test_tent_no_host_sync_in_round[L4P, L3]` | the run reads host values only in the per-round sync and the host bookkeeping after it (`_run_paper`, `_ps_after`; L3 `_run_l3`) | the trap is not blind |
| `test_wsa_cell_arguments_and_report` | the cell's arguments and refusals; summarize_w / arm_gates_w / verdicts on a FAST and an EXACT CPU run: COMMIT-N replaces COMMIT, TOK / FP (L4PJ), D1 / D3 / SPAN / STRESS (L4PT) pass; D3 and COMMIT-N are not blind | - |
| `test_step_script_wa1` | wrong cells STOP; DRYRUN wa1m prints the cell line and the arguments; wa1a / wa1b STOP while PRED_STAMP_WSA is None | - |

If a precondition fails, the data do not exercise the path: tell me and I change the data, not the claim. The most
data-dependent ones: target rows in the `end` runs (they need a final group that rejects), the uncommitted chain fill
(searched over S0 .. S0 + 3), the different dropped counts of the two cost runs.

## 7. The GPU cell wa1 (REGISTERED block TBD for the lead)

    DRYRUN=1 bash benchmarks/Efficiency/r2_wsa_step.sh --nosi <sha> <holder> <run id> wa1m
    bash benchmarks/Efficiency/r2_wsa_step.sh --nosi <sha> <holder> <run id> wa1m    # the miniature first
    bash benchmarks/Efficiency/r2_wsa_step.sh --nosi <sha> <holder> <run id> wa1b
    bash benchmarks/Efficiency/r2_wsa_step.sh --nosi <sha> <holder> <run id> wa1a

| cell | C (pool) | B | steps | journal | checks | stress gate | STEP_MIN | mem | GPU time (MODEL) |
|---|---|---|---|---|---|---|---|---|---|
| wa1m | 73 (10) | 8 | 64 | 8 | every round | on | 30 | 100G | about 8-12 min |
| wa1a | 128 (65) | 216 | 512 | 256 | probe rounds | off | 75 | 240G | about 50-60 min |
| wa1b | 73 (10) | 318 | 512 | 256 | probe rounds | off | 110 | 340G | about 75-90 min |

All cells: `--spec 3 --spec-arms L4P,L4PJ,L4PT --spec-rollback tentative --spec-twin fused --spec-body fused --spec-graph
on --spec-redirect on --probe-every 16`, SH first, the inputs of sp3a / sp3b (the same workload pin, S0, document rows,
prefill, K, journal, twin and body): extra work is told apart from slower execution. Total about 2.5 h of GPU time. The
step script STOPS wa1a / wa1b while `r2_wsa_cell.PRED_STAMP_WSA` is None at the commit (L33); wa1m needs no registration.

PER ARM the cell reports: committed tok/s (median / mean / effective) against SH and against 3,407.2; phases; ROLLBACK per
round in ms and SH steps with its stated boundaries (module docstring of the cell); the actual undo and remake calls
(journal arms: `_undo_step`, commit, truncate, reject_all calls and the remade rows; L4PT: residency-undo calls, resets,
commits, deferred-row writes, joint passes and rows, replays); bytes per round (L4PT: published rows, the tail-slot
traffic MODEL, write-back, counter restore, residency restore 0; journal arms: the journal's saved slots and the fallback
reloads); memory (start, prefill peak, decode peak, the fitting batch at 84 GB, MODEL); tokens against SH; L4PT's
fingerprint deltas on the check rounds; the schedule record (token accounting, the first and last 12 passes of every
request). MUST gates: SH-RUN; <arm>-RUN, -ERR; L4P -COMMIT, -TOK, -FP; L4PJ -COMMIT-N, -TOK, -FP; L4PT -COMMIT-N, -D1,
-D3, -SPAN; wa1m <arm>-STRESS.

REGISTERED, TBD (`r2_wsa_cell.PRED_WSA` and `PRED_STAMP_WSA`, from `date -u`, in a commit before wa1a / wa1b):

| key | quantity | value |
|---|---|---|
| L4PT_rollback_sh | median steady (reset + commit) ms / SH step, (low, high) | TBD (the plan: <= 0.1; falsifier > 0.3) |
| L4PT_ratio | L4PT / SH median steady tok/s, (low, high) | TBD (the plan's I1: about 0.41 at C 128, 0.51 at C 73, a model of the legacy schedule) |
| L4PT_vs_L4PJ | L4PT / L4PJ median steady tok/s (the same schedule: D1 + D3 only), (low, high) | TBD |
| L4PT_joint_saving | L4PJ joint ms - L4PT joint ms, median, (low, high) | TBD (the deferred row: +93 / +120 ms measured in L4P against L4X) |
| L4PT_agree | the lowest fraction of L4PT's tokens equal to SH's | TBD (REPORT until WS-G) |
| L4PJ_tok | L4PJ's tokens bitwise SH's | TBD (expected True) |
| L4PJ_vs_L4P | L4PJ / L4P median steady tok/s (the schedule's change, EXACT), (low, high) | TBD |

INPUTS FOR THE BANDS (MEASURED / MODEL, not a registration):
- MEASURED (SCORING_specp_fix.md section 7): legacy L4P at C 128 B 216: round 896.3 ms = joint 424.2 + truncate 127.1 +
  draft 155.0 + commit 190.2; at C 73 B 318: 1,267.3 = 660.4 + 127.2 + 210.8 + 271.6. SH step 64.48 / 124.00 ms. The fifth
  (deferred) row: +93.5 / +120.5 ms. RB-STEP (rb1): the journal hooks add 22.1 ms per tentative step at C 128 B 216.
- MODEL L4PT round at C 128 B 216: joint pass of 4 rows without the journal hooks (424.2 - 93.5 = 331 ms for the EXACT
  rows; minus the hooks 4 x ~20 ms; plus call B and the merge per row and layer, about 15 small ops each, about 4 x 5 ms)
  about 270-330 ms; reset + commit 3-7 ms; 2 drafts without hooks (77.5 - 22 = 55 ms each, plus call B, about 60 ms)
  about 120 ms; total about 400-460 ms for about 2.9 tokens per request: about 0.40-0.47 x SH. At C 73 B 318: joint 660.4
  - 120.5 - 4 x ~25 + 4 x ~7 = about 470 ms; drafts 2 x ~85 ms; total about 650 ms: about 0.55 x SH. The schedule change
  costs about 1-2 % (section 4, choice 2). These are central values with wide bands: the joint pass's hook share is not
  measured per row.

## 8. Risks and open points

- Nothing ran. The CPU tests are my reading of the stand-ins.
- The selection gate at layer 1 can trip on a near tie (section 3).
- `torch.where(out=)` on a strided view of the tail slot (in place): TensorIterator allows full overlap; if a build refuses
  it, the fix is `dst.copy_(torch.where(...))` (two calls instead of one).
- Call A passes a strided scratch bias view (batch stride of the full allocation) with `return_softmax_lse=True` and no
  `cache_batch_idx`. DraftResident passes the same view without the lse; verify_rows FAST passes `return_softmax_lse` with
  `cache_batch_idx`. This exact combination is new on the GPU: wa1m shows it first.
- The commit is about 300 launches per round. If the GPU host time is about 20 us per launch (rb1's figure), the rollback
  is about 6 ms = 0.09 SH step at C 128: near the target. A captured commit (fixed shapes, no host read) is the known cure.
- HOST WORK PER ROW (MODEL): a FAST row has about 100 small ops per layer (residency, the buffer write, two attention
  calls, the masks, the merge); the EXACT row has about 80 (decode_fixed, the draft hook, the journal hooks). At B 216 /
  318 the device time of a row (about 60 ms per one-row pass) hides about 3,200 launches per pass; at small B it may not.
  The masks that depend only on the pass (the call-B visibility without the per-head part, the cut, the past-edge flag)
  could be computed once per pass instead of once per layer (about 25 ops fewer per layer); not done here.
- The new fill idles a rejecting request for K - 1 slots (section 4, choice 2). The lead can change the choice; the trace
  of a request does not change, the throughput does.
- MEMORY: L4PT needs no journal (about 2.3 GB less than L4P at B 216, MODEL) and adds the buffers (about 0.45 GB at B 216,
  0.67 GB at B 318) and a scratch bias (7 MB). L4PJ needs the journal as L4P.
- The legacy L4P arm keeps its fixed rounds (it commits up to steps + K - 1 tokens and checks the first steps): its numbers
  compare with sp3a / sp3b, not one to one with the contract-4 arms (they commit exactly `steps` tokens).
- L3 / L3T are not in the step script's cells (the plan names L4P); L3T keeps the legacy L3 rounds (fixed rounds).

## 9. Review

A read-only reviewer agent read the working tree before the commit (spec_tentative.py, wsa_world.py, test_wsa.py,
r2_wsa_cell.py, r2_wsa_step.sh, the Gate 0 edits) against the code that they call. It found no BUG and no LIKELY BUG. It
checked shapes, dtypes and the API of every callee, FAST against F4 and the shipped residency order, the schedule against
the role rules, the EXACT keep counts against the Rollback semantics, the plants and the cell's keys. Its findings and my
fixes:

| # | finding | class | fix |
|---|---|---|---|
| 1 | The D / P slots' trace came from the host plan only; no test compared it with the rows that ran (a D slot fed at a wrong position would only lower the acceptance). EXACT had no position witness at all. | RISK | `test_paper_schedule_trace` now compares the trace with the rows that ran: FAST every J / P / D slot's layer-0 row writes (`_joint_row_errors`, `_slot_row_errors`); EXACT the joint pass's q and the twin position before each active D slot (`dpos`, a check-only record in `PaperJournalRunner`). A row or slot moved by one position: CAUGHT (test-side controls). |
| 2 | The writeback_tentative plant run stopped at 12 rounds; the clean run's uncommitted fill can come later. | RISK | That plant runs without the cap. |
| 3 | The joint pass replayed row K - 2 with an empty mask (a group has v = K - 1 or K): L extra twin steps per pass. | NIT | The replay rows come from the host plan (the v of each group). |
| 4 | The roles' host-to-device copy synchronized the stream on CUDA (pageable memory). | NIT | Pinned memory and `non_blocking=True` on CUDA; the CPU path is unchanged. |
| 5 | The `miscount` control was accepted but did nothing in the contract-4 runners. | NIT | `_ps_stats` applies it as spec_round does (accepted, not committed, into `committed_sum`). |
| 6a | The dropped recount used the runner's own formula. | NIT | The recount now counts the written, uncommitted positions from the layer-0 buffer writes. |
| 6b | The L3 journal arm's rollback time leaves out `reject_all` (it falls in the "draft" phase). | NIT | Marked as a lower bound (`lower_bound` in the cell's rollback record; a comment at `ROLLBACK_J`). L3 is not in wa1. |

The reviewer's own commands: `cd`, and `sed -n 1p /dev/null` once (read /dev/null only; a breach of the no-sed rule).

## 10. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- `git worktree add` (as instructed), `git add`, `git commit -F` (the message written with the Write tool): git.
- `date -u` (the stamps).
- `ls -laL` on the reviewer agent's transcript file (its size). To wait for the reviewer: `tail -c`, and `tail -f`
  piped to `grep -m 1` on its transcript (tail and grep only; I stopped the two waits with the task tool).
- Shell built-ins otherwise (`cd`, `echo`, `;`, `&&`, `|`). One harmless slip: a no-op variable assignment named
  `sed_unused=0` in front of a grep command (an assignment, not a command; no sed ran).
- ONE BREACH: `uniq -c` once, by mistake, inside a read-only grep pipeline over r2_spec_cell.py whose output went to
  `head -0` (nothing printed, nothing changed); the next command did the same listing with `sort -u`.
- No interpreter, no sed, no awk, no heredoc, no `cat >>`, no tr / paste / xargs, no hashing, no `find`, no srun / sbatch
  / pytest, no sleep, no chmod. Files written with the Write and Edit tools.
