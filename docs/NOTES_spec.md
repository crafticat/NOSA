# NOTES: Phase 2, rungs L3 and L4: the speculative round runner (`--spec K`)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`, Phase 2 (rungs L3, L4).
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-spec`, branch `r2-spec`, from `962cd1b` (r2-rollback, Task 1.5).
- Status: SOURCE ONLY. I did not run a test or a job. The lead runs the commands below.
- Style: ASD-STE100. Labels: CODE = read at file:line; INFERRED = my reading; MODEL = hand arithmetic.

## 1. The merge (four hook trees in one tree)

| commit | what | conflicts and how I resolved them |
|---|---|---|
| `40b87fc` | merge `r2-verify1` (`5304d9b`: Task 1.4 route 1 and the per-row nosa_linear stand-in fix `6adf18d`) | `tests_cpu/test_ragged_engine.py`, the Gate 0 exclude list: both sides added one file; I kept both |
| `673f234` | merge `r2-draftpool` (`de3d120`: Task 1.1, cells dr1 / dr1m / dr73 / dr81) | `benchmarks/Efficiency/r2_base_cell.py`, ten hunks, each two independent additions at one place (--ragged / --rollback against --draft-resident). I kept both, in the order ragged, rollback, draft-resident. Two hunks shared context lines between two new functions; each function keeps its own copy (run_draft keeps `logits, pos = ...`). One new line of behaviour: parse_args refuses --draft-resident with --ragged / --rollback (the DR arm runs on the shipped engine). `cache_engine.py` merged without a conflict (the draft block is appended) |
| `3a493d2` | merge `r2-poolpf` (`151355d`: Task 1.2, cell pf1) | none (new files only) |
| (Phase 2 commit) | this work | - |

Every flag stays opt-in and default OFF: `--ragged`, `--rollback`, `--draft-resident`, `--pool-prefetch` / `--fetch`
(r2_poolpf_cell), `--spec` (r2_spec_cell, new). With every flag off, no new code runs.

GATE 0 TESTS IN THE MERGED TREE. Each hook's Gate 0 compared "its" tree with its base, so in the merged tree each one
saw the other hooks' files. I changed the exclude lists only; each excluded file has its own Gate 0 (named in a comment
at each change). No other assertion changed.

| test | change |
|---|---|
| `test_ragged_engine.py` test_gate0 | excludes verify_rows.py, cache_engine.py (draft block), pool_prefetch.py, r2_transport_gilfree_v3.py, spec_round.py |
| `test_rollback.py` test_flag_off_is_shipped | the same files excluded; "no other module names ragged_rollback" now allows spec_round.py (the opt-in caller) |
| `test_verify_rows.py` test_flag_off_is_shipped | the other hooks' nosi files excluded; r2_base_cell.py leaves the byte diff (it carries the --rollback and --draft-resident arms; their Gate 0s check every call sits under its flag); "no flag-off file names verify_rows" still includes r2_base_cell.py |
| `test_pool_prefetch.py` test_flag_off_is_shipped | cache_engine.py: ded9d41's module statements must be a prefix (the draft block is appended); r2_base_cell.py: must not name pool_prefetch; the other files keep the byte diff against ded9d41 |
| `test_draft_pool.py` | no change (its Gate 0 checks only cache_engine.py's prefix and files that no other hook changes) |

THE CPU STAND-IN OF nosa_linear (verify1 fix `6adf18d`: softplus ROW BY ROW, as the kernel's one program per (row,
head)). `tests_cpu/ragged_world.py` has it from the merge, so `rollback_world.py`, `verify_world.py` and the new
`spec_world.py` (they import it) have it. I also carried it into the two older worlds that had a batch-wide softplus:
`tests/r2_fake_world.py` (fake_nosa_linear) and `benchmarks/Efficiency/tests_cpu/test_r2_tick.py` (_nosa_linear). The
draftpool and poolpf worlds have no nosa_linear stand-in.

## 2. What I made (Phase 2)

| file | what |
|---|---|
| `nosi/nosi/spec_round.py` (new) | `SpecRunner` (L3 and L4 rounds), `SpecOps`, `DraftResident` (Task 1.1 on the ragged engine), `KVMeter`, `accept_prefix` |
| `nosi/nosi/ragged_rollback.py` | Phase 2 additions (module docstring "PHASE 2 ADDITIONS"): split journal (engine part in a hook on the body's decode_fixed), `truncate`, `fast_begin` / `fast_end`, the extras store passes through the gather hooks, the `_ures == _u` refusal, NEG `no_truncate_mark` |
| `nosi/nosi/verify_rows.py` | `VerifyOps(draft_last=...)` (FAST: the extras union of the verify rows only), `VerifyState.hooks` (FAST journal calls), the FAST write-back through `rs.body.writeback_kernel` (the same kernel; the deferred one in a group) |
| `benchmarks/Efficiency/r2_spec_cell.py` (new) | the GPU cell: SH, then L3, L4X, L4F; gates, REGISTERED block `PRED_SPEC` (TBD) |
| `benchmarks/Efficiency/r2_spec_step.sh` (new) | a copy of `r2_base_rollback_step.sh`: cells sp3m, sp3a, sp3b; per-holder claim `sp_active_<holder>` |
| `benchmarks/Efficiency/tests_cpu/spec_world.py` (new) | the CPU world of the new tests (not a test file) |
| `benchmarks/Efficiency/tests_cpu/test_spec_round.py` (new) | the CPU tests |

NO NEW KERNEL. NO KERNEL EDIT. The round uses ragged_engine.decode_step, verify_rows.joint_pass, ragged_rollback and
the shipped kernels (diff_offload, pool_update, flash_pool_swap, flash_h2d_from_mask(_bias), masked_writeback,
stage1, pooling, flash_attn_nosa) as they are. Everything new is torch.

## 3. The round (paper sections 3 and 3.4)

L3 (serial verify), per round, request b at committed position p_b with its last committed token t_b:
1. begin; K DRAFT steps (DraftResident "all"): step j feeds x_j at p_b + j; reject_all (the drafts never stay).
2. begin; K VERIFY steps: the ragged step, unchanged (feeds t_b, d_1 .. d_{K-1}).
3. n_b = longest prefix with target argmax == draft; m_b = min(n_b + 1, K); commit the target's tokens g_1 .. g_{m_b}
   (accepted drafts and, on a reject, the correction; full accept: K tokens, NO bonus); rollback commit(m_b).

L4 (joint pass), per round; the rollback group holds the joint pass and the K - 1 drafts after it (2 K steps):
1. begin; ONE joint pass, R = K + 1 rows: rows 0 .. K - 1 verify, row K is the draft row (feeds d_K at p_b + K).
   EXACT: serial residency per row, row K through DraftResident (mode = row K). FAST: one residency, the extras store of
   the verify rows' union, row K resident-only (draft_last).
2. accept as L3; commit the tokens.
3. truncate: a full-accept request keeps all R rows (its draft row stays tentative: it is the first draft step of its
   next round); another request goes back to m_b rows at once.
4. K - 1 draft steps: full accept from e_0 (the draft row's token) at p_b + K + 1; else from the correction at
   p_b + m_b.
5. commit: full accept keeps the K verify rows; another request keeps m_b rows (the truncated steps are marked and
   not undone twice).
K >= 2 for L4.

THE FIRST STEP of a run is a PLAIN step (`warm`, default 1): after the prefill the window slots hold torch.empty
bytes until a step loads them, and a draft reads its skipped slots through the masked bias (0 x NaN = NaN; Task 1.1's
DraftBlockEdge refusal had the same reason).

THE DRAFT (DraftResident): the shipped diff, victim pool and map commit run; the two host gathers do not. Skipped slots
= `_load_mask >= 0` after pool_update; their map entry becomes -1; a scratch copy of the attended bias gets -3e4 on their
rows; attention reads the scratch (`_kv_bias_gpu_att` swapped for the step, restored by `end()`). The live bias is
never written. One exception: the block that a step of the same group filled lives only in the rollback's staging
block; with `redirect` the draft loads it through the rollback's redirect (the gathers restricted to that block; a
host flag computed from the position mirror at the round start). Draft rows never reach the host rows: every fill in
a group is deferred, and the fills of undone steps are dropped.

ROLLBACK, PHASE 2 ADDITIONS (ragged_rollback.py docstring):
- SPLIT JOURNAL: the twin part (counters, twin rows, tail row) at the twin hook, as before; the engine part (maps,
  lengths, pool tables) at a new hook on `rs.body.decode_fixed`. In the ragged step nothing writes those tables between
  the twin step and decode_fixed, so the journal holds the same values (the Task 1.5 tests run this path). A joint pass
  (twin R times, then R residencies) now has one journal per row.
- FAST JOURNAL: verify_rows FAST calls `fast_begin` before its one residency (the journal of row 0) and `fast_end`
  after the pass (rows 1 .. R - 1 = the state after r rows: the residency stays, the tail id and attention length of
  position p0 + r - 1, empty slot journals; the fill row of each request, u0 + 63 - p0 % 64).
- TRUNCATE: undo some requests' later steps inside an open group; a kept fill of a truncated request goes to the host
  rows at once; the undone steps are marked.
- REFUSAL: commit / truncate refuse a group where a layer journaled fewer engine steps than twin steps (a joint pass
  without the hooks): `RollbackError`, never a silent wrong journal.

HOST SYNCS: one per round after the round (committed counts, twin positions), one after the warm step. None inside a
round. The rollback has none (test_no_host_sync_in_round traps both modules).

## 4. A FINDING OF THE DESIGN (L4 as the task defines it), for the lead

After its first reject, a request has K - 1 valid drafts per round (its draft row fed no valid token: the token at
p + K is the correction, unknown before the pass). Its row K - 1 then always gives the target's own token, and its draft
row is never used again. BOTH cases commit at most K tokens per round: full accept = K accepted drafts; the other case =
K - 1 accepted drafts plus the target's token. So in this round structure the draft row adds no committed token. It
only replaces one draft step while a request has accepted every draft so far. With per-token acceptance about 0.995
(dr1: draft 0.99475), a request leaves the full-accept case after about 1 / (1 - 0.995^3) = 67 rounds (MODEL), so in a
512-token run most requests spend most rounds in the second case. The L4 vs L3 difference is then: one joint pass of
K + 1 rows plus K - 1 drafts against K drafts plus K serial verify steps. The paper's schedule (section 3, Figure 3:
"the joint pass drafts d4 and verifies d1, d2; d3 is deferred") is a different structure (verify K - 1, defer the newest
draft); it needs the drafts' tail rows to stay tentative across the joint pass. I did not build it (the task asked for
"verify of K positions + the next draft row").

## 5. Tests for the lead

Run on a CPU node, inside a job:

    cd /mnt/beegfs/ojerbi/nosi-r2-spec
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_spec_round.py

Regressions (the touched modules and the four hooks' suites in the merged tree):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_rollback.py benchmarks/Efficiency/tests_cpu/test_verify_rows.py \
        benchmarks/Efficiency/tests_cpu/test_ragged_engine.py benchmarks/Efficiency/tests_cpu/test_draft_pool.py \
        benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py benchmarks/Efficiency/tests_cpu/test_r2_mbase.py \
        benchmarks/Efficiency/tests_cpu/test_r2_tick.py tests/test_r2_base.py tests/test_r2_capture_step.py \
        tests/test_r2_p3.py

test_pool_prefetch.py needs the image (`/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif`: triton, a C++ compiler, the CUDA
headers) as before. The step-script tests skip outside a clean checkout of their own ROOT (the old hooks' scripts keep
their own trees as ROOT).

| test | claim | negative control (precondition, L34) |
|---|---|---|
| `test_flags_off_is_shipped[B1, B8]` | nosi/nosi + dependencies = `3a493d2` (the merged hooks) except spec_round.py and the declared Phase 2 files; no other module names spec_round; the cell refuses `--spec 0`; an attached SpecRunner (L3 and L4), no round: 16 plain steps bitwise the plain world (logits, every state tensor, a block fill); detach() restores every callable | one committed DRAFT step in the attached world (precondition: it skipped a host load) |
| `test_spec_tokens_equal_shipped_greedy[L3/L4 x B1/B8]` | committed tokens = the SHIPPED greedy decode (Llama.decode_inference, re-warmed), the committed rows' logits bitwise the shipped step's; after every round: residency invariants, host rows; no refusal, no journal overflow | preconditions: rejects AND full accepts, a block edge, draft skips, pool moves, (B 8) ragged positions |
| `test_reject_keeps_accepted_prefix_and_correction[L3, L4]` | per round and request: m = n + 1 (K at most), the accepted prefix = the drafts, the next token = the target's | commit_draft_at_reject, keep_rejected: CAUGHT (preconditions: a real reject, m < K) |
| `test_full_accept_no_bonus_token[L3, L4]` | a full accept commits exactly the K drafts | L4 bonus_on_full_accept (every draft row's token wrong): CAUGHT (precondition: a full accept) |
| `test_no_draft_row_reaches_host[L3/L4 x B1/B8]` | after every round: completed blocks = the shipped host rows; host rows from the committed block on still NaN (S 4925: the first round fills a block in a step it does not keep) | rollback no_defer: CAUGHT (precondition: such a fill) |
| `test_accept_counts_reported[L3, L4]` | device counters = a host recount of the round records | miscount: CAUGHT (precondition: m != n somewhere) |
| `test_truncate_then_commit_equals_reference[B1, B8]` | truncate + 2 more steps + commit = the world of the kept steps (state, kept logits, 3 later steps); a kept fill of a truncated request is written at truncate | no_truncate_mark: CAUGHT (precondition: truncated steps moved slots) |
| `test_fast_rounds_keep_residency[B1, B8]` | L4 FAST: residency / host-row / twin invariants after every round, no overflow; a FAST joint pass without the hooks is REFUSED (RollbackError) | - |
| `test_draft_resident_skips_and_masks[B1, B8]` | the draft runs no host gather; miss list = skip mask; map -1; scratch bias MASK_BIAS on the skipped rows, live bias untouched, view restored; reject_all = snapshot; a draft outside a group is refused | precondition: the last layer skipped a load |
| `test_prefetch_wiring`, `test_no_host_sync_in_round[L3, L4]` | the L5 hook is called per draft step / round and is off by default; only run() and the redirect flag read host values, the rollback none | the trap must see run()'s sync |
| `test_cell_arguments_and_verdicts`, `test_step_script_sp3` | arguments, refusals (`--spec 0`, `--spec-prefetch on`), pure parts, TBD verdicts = REPORT, gates; DRYRUN of the cells | one ULP, one token past the count (ignored) |

THE FILTERS (spec_world.inject / corrupt): the token proof holds for ANY draft tokens. In rounds -1 (the L4 prelude)
and 0 the drafts ARE the shipped tokens (full accepts by construction); then chosen drafts are corrupted to force
rejects at known places; the other rounds keep the natural drafts. If a precondition fails, the data do not exercise
the path: tell me and I change the data, not the claim.

## 6. The GPU cells (REGISTERED block TBD for the lead)

    DRYRUN=1 bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3m
    bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3m    # the miniature first
    bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3a
    bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3b

| cell | C (pool) | B | steps | STEP_MIN | mem |
|---|---|---|---|---|---|
| sp3m | 128 (65) | 8 | 64 | 25 | 100G |
| sp3a | 128 (65) | 216 | 512 | 80 | 240G |
| sp3b | 73 (10) | 318 | 512 | 115 | 340G |

All cells: `--spec 3 --spec-arms L3,L4X,L4F --spec-journal-blocks 256 --spec-twin fused --spec-body fused --spec-graph
on --spec-redirect on --spec-prefetch off --probe-every 16`. THE FUSED TWIN by default (coordinator, from
SCORING_rg1.md: rg1f +0.6 % / +1.7 % against cached NOSA with the logits bitwise SH's; rg1, the torch twin, +13-14 %).
The CPU proofs keep the torch twin (the shipped summary bits).

Per arm: committed tok/s (median over steady rounds, mean, effective) and each against SH's; acceptance (accepted /
valid drafts, histogram, full fraction); phase ms per round (L3: draft, verify, commit; L4: joint, truncate, draft,
commit); KV at verify (NEW blocks and MB per round, READY slots, IN FLIGHT 0, the gather WAIT ms on probe rounds); the
token check and the fingerprint check against SH (bitwise); draft skips per stream; journal counters. MUST gates:
SH-RUN, <arm>-RUN, <arm>-ERR, <arm>-COMMIT.

REGISTERED, TBD: fill `r2_spec_cell.PRED_SPEC` and its stamp (`date -u`, L33) in a commit before the run.

| name | quantity | value |
|---|---|---|
| L3-TOK, L4X-TOK | committed tokens bitwise SH's | TBD |
| L4F-AGREE | committed tokens equal to SH's, fraction >= | TBD |
| L3-RATIO, L4X-RATIO, L4F-RATIO | median steady committed tok/s / SH steady tok/s, (low, high) | TBD |
| SP-ACCEPT | L3 accepted / valid drafts, (low, high) | TBD |
| L4F-VS-L3 | L4F / L3 median steady tok/s, (low, high) (the plan's L4 gate) | TBD |

INPUTS FOR THE BANDS (MODEL / INFERRED, not measured):
- sp3a L3 round = 3 drafts (dr1: 60.2 ms each) + 3 verify steps (+ rollback hooks; RB-STEP registered 1.0-1.4 x) +
  2 commits (RB-COMMIT 5-60 ms each): about 181 + 3 x 70 + 2 x 30 = 450 ms for at most 3 tokens per request; SH: 3 x
  63.6 = 191 ms for 3 tokens. L3-RATIO about 0.4 (0.3-0.5). L3 is the proof rung, not a speed rung.
- sp3a L4F round = the FAST joint pass (vr1 MODEL: VRF about 3.1 SH steps, about 200 ms) + truncate (about one commit) +
  2 drafts (120 ms) + commit of 6 steps (about 2 commits): about 200 + 30 + 120 + 60 = 410 ms for at most 3 tokens:
  L4F-RATIO about 0.45 (0.3-0.6); L4F-VS-L3 about 1.1 (0.9-1.4).
- sp3b: dr73 (draft 0.80 of SH, registered) is the only input; the journal overflows more at C 73 (many more loads):
  the undo then reloads lost blocks from the host (slower, exact).
- SP-ACCEPT: dr1 draft agreement 0.99475 (teacher-forced, drifting state); the rounds' acceptance is n / v over fresh
  drafts: about 0.99 (0.97-1.0).

## 7. Risks and open points

- Nothing ran. The CPU tests are my reading of the stand-ins. The most data-dependent preconditions: a full accept in
  L4-B1 (natural drafts in its first rounds), draft skips on 76-block prompts, an unkept fill in sp-host runs (S 4925
  makes it deterministic).
- MEMORY: L4 holds a 6-step journal (about 2 GB at 256 slots, B 216: MODEL), plus staging 0.45 GB, plus the joint pass's
  E = 4 B rows. mb1's peak was 82.04 GB on an 85.1 GB device. If an arm stops on memory, lower --spec-journal-blocks
  (exact: a lost slot past it comes back from the host).
- COST: truncate and the final commit undo up to 2 K steps per layer with about 30-45 small ops each (host launches);
  the phases report them. A captured commit (Task 1.6) is the known cure.
- FAST draft row: it sees the window and the extras that the verify rows loaded; a pool hit of its own is not visible
  (a draft approximation, timing form only).
- The KVMeter counts the EXACT draft row's redirect gathers (stale host rows of the staged block, overwritten from
  staging) as host loads. Small.
- --spec-prefetch on is refused in the cell: pool_prefetch.PoolPrefetch hooks the shipped engine's decode_update
  (its reader event and commit wait) and reads seq_length; on the per-request engine and inside a rollback group it
  would write pool slots outside the journal. The runner exports the draft miss list (DraftResident.miss_ids) and calls
  ops.prefetch.on_draft / on_round: that is the L5 seam.
- The cell runs ceil((steps - 1) / K) rounds: slow requests commit fewer than `steps` tokens; the token check covers
  each request's committed prefix (count_min is reported).

## 8. Review (a read-only reviewer agent on 2140e1d) and the fixes

| finding | fix |
|---|---|
| BUG: test_truncate compared pos_host and the LRU clock of a 5-step world with a 3-step reference (the diff could never be empty; the plant tested nothing) | truncate / commit get their host copies; the reference runs 5 masked steps (the last two with no active request: the clock moves by topk each, as in x); the plant's precondition needs a SWAP or MOVE_OUT (the inverse of MOVE_IN is no move) |
| LIKELY BUG: the FAST test compared completed host blocks with the shipped run (FAST is not bitwise) | host_rows_errors(compare_complete=False) for FAST: only the "still NaN" part |
| BUG (latent, --pool 0): without a pool the slot journal is written only by the gather hook; a draft with no gather left the journal of an older step, and its undo wrote old bytes | `_journal_engine` starts the slot journal empty when there is no pool |
| RISK: the full-accept preconditions depended on natural drafts in the first rounds | run_vs_shipped(oracle_rounds=(-1, 0)): in those rounds the drafts ARE the shipped tokens (spec_world.inject; the runner exposes `next_col`, the committed column of each draft token, to the test filters) |
| RISK: the runner read twin.pos before the rollback's release wait (an async executor only) | `_sync_release()` before every read of twin.pos |
| NIT: the joint pass's default GEMM was F.linear, not the ragged step's | the rollback's original linear (the step's own GEMM) |
| NIT: KVMeter counts the stale host read of a staged block | documented (it is a real PCIe read) |
| NIT: lru_cache keys of `_ev` | normalised |
| NIT: truncate without host copy + commit with host copy gives a wrong mirror | refused (RollbackError) |

## 9. CPU run sp_a6eb5dd_a (6 failed, 21 passed; regressions sp_a6eb5dd_b / _c green) and the fixes

1. test_no_draft_row_reaches_host[L3-B1, L3-B8, L4-B1, L4-B8]: the L34 precondition (a block filled by a step that its
   round does not keep) was empty. ROOT CAUSE (a runner bug that shows on a CPU device only): after the warm step,
   `SpecRunner.run` set `self.pos_host = twins[0].pos.cpu()`; on a CPU device `Tensor.cpu()` returns the SAME tensor, so
   the host mirror followed the live twin counter for the whole of round 0 (L3) / the L4 prelude. The edge filter
   (`corrupt_before_edge`, applied after draft 0 ran) then saw position 4927 (offset 0, no forced reject), and the round
   record stored the round-END position. Later rounds were right (their mirror comes from `torch.stack(...).cpu()`, a new
   tensor). On the GPU `.cpu()` copies. FIX: `.cpu().clone()` (one line in spec_round.py). With S 4925 the first round
   starts at 4926: the forced reject keeps m = 1 and the fill at 4927 (offset 1) is not kept, in every parametrization.
2. test_truncate_then_commit_equals_reference[B1, B8]: the plant no_truncate_mark was not caught. ROOT CAUSE: in that
   scenario (and in the L4 commit) every truncated request keeps 3 steps, more than its truncated step indices, so commit
   never selects a truncated step again: the mark is never read there, and the plant cannot act. FIX (test only): the
   commit scenario keeps its claim; a second scenario (truncate, 2 more steps, reject_all) needs the mark (keep count 0
   is below every truncated step): claim = the snapshot back bit for bit and the next 3 steps equal; plant
   no_truncate_mark there, precondition a SWAP / MOVE_OUT in a truncated step of a truncated request.

## 10. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- `git worktree add` (as instructed), `git merge --no-commit`, `git add`, `git commit -F` (messages written with the
  Write tool): git.
- `sleep` in the background twice (a timer while the reviewer agent ran).
- Shell built-ins only otherwise (`cd`, a variable assignment, `;`, `&&`, `|`). No interpreter, no sed, no awk, no
  heredoc, no `cat >>`, no tr / paste / uniq / xargs, no hashing, no `find`, no srun / sbatch / pytest. Files written
  with the Write and Edit tools.
- The reviewer agent reported one breach of its own: `sed -n 100,130p test_verify_rows.py | head -0` once (read-only,
  output discarded).
