# NOTES: Phase 2, rung L5: background prefetch of the draft misses, with the soft pin (`--spec-prefetch on`)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`, Phase 2, rung L5 ("background
  prefetch + pin (1.2); faster than L4; verify waits measured").
- Paper: `/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/speckv/speckv.txt`, sections 3.1, 3.3, 4.3.
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-spec5`, branch `r2-spec5`, from `a6c9d10` (nosi-r2-specp: rung L4P with the undo fix
  of fd1a239). No other worktree was touched.
- Status: SOURCE ONLY. I did not run a test, an interpreter or a job. The lead runs the commands in sections 6 and 7.
- Style: ASD-STE100. Labels: CODE = read at file:line; MEASURED = from a scoring file; MODEL = hand arithmetic;
  INFERRED = my reading.

## 1. What I made

| file | what |
|---|---|
| `nosi/nosi/spec_prefetch.py` (new) | `SpecPrefetch(PoolPrefetch)`: the seams `on_draft`, `on_verify`, `on_release`, `on_round`, the outermost hook of the body's `decode_fixed` (the commit wait per layer and the accounting), `report()` |
| `nosi/nosi/spec_round.py` | five guarded seam calls (`bind` in `SpecRunner.__init__`, `unbind` in `detach`, `on_verify` in `_round_l3` and `_round_l4p`, `on_release` in `run`); docstrings. With `SpecOps(prefetch=None)` the code is a6c9d10's (section 6, Gate 0) |
| `benchmarks/Efficiency/r2_spec5_cell.py` (new) | the GPU cell: SH, L3, L3P, L4P, L4PP; MUST gates; the REGISTERED block `PRED_SPEC5` (TBD) |
| `benchmarks/Efficiency/r2_spec5_step.sh` (new) | a copy of `r2_spec_step.sh`: ROOT this tree; cells sp5m, sp5a, sp5b; the same claim `sp_active_<holder>` |
| `benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py` (new) | the CPU tests (section 6) |
| `benchmarks/Efficiency/tests_cpu/spec5_world.py` (new) | the CPU world: `FakeTransport`, `make_pf`, `VerifyRecount`, `run_pf` (not a test file) |
| `benchmarks/Efficiency/tests_cpu/spec_world.py` | two keyword arguments: `make_prefetch` (build the prefetch on the attached state) and `sr` (the round-runner module) |
| four older Gate 0 tests | one added exclude line each (`spec_prefetch.py`): test_ragged_engine, test_verify_rows, test_rollback, test_spec_round |

NO NEW KERNEL. NO KERNEL EDIT. `pool_prefetch.py`, the transport, the C++ packer and every engine file are a6c9d10's
bytes. SpecPrefetch reuses PoolPrefetch unchanged: `_issue` (dedup, lanes, the descriptor job, the delivery probes),
`_commit` and `_commit_layer` (the arrival wait, the last-reader wait, the dedup against the window and the pool, the
LRU slot that is not pinned, the SHIPPED scatter kernels `flash_h2d_from_mask` / `_bias`, the map write and the soft pin
age). New code: torch on small tables and host bookkeeping.

## 2. The design (paper 3.1, 3.3)

ONE ROUND, the calls of the round runner (CODE spec_round.py, `_round_l3`, `_round_l4p`, `run`):

| seam | when | what SpecPrefetch does |
|---|---|---|
| `on_draft(miss_ids)` | after each draft step (L3: the K drafts of the round; L4P: the joint pass's draft row and the K - 1 drafts; the prelude's K drafts) | a SNAPSHOT of the step's skipped blocks (DraftResident.miss_ids, Task 1.1's miss list): one stack, one cast to int16, one D2H copy into a pinned buffer, one event. LAG 1: the host waits the snapshot of the PREVIOUS draft step and issues its blocks as ONE descriptor job to the C++ packer. The host never waits the newest step, so the GPU runs it meanwhile |
| `on_verify()` | OUTSIDE a rollback group, before the verify (L3: after the drafts' reject_all; L4P: at the round start, after RequestPin.apply) | issues the last snapshot; classifies each job of the epoch: landed (READY) or not landed (IN FLIGHT); MERGES the epoch's jobs into ONE commit per layer and enqueues it on the commit stream (PoolPrefetch._commit). The host waits only the SUBMISSION of the newest job; the commit stream waits its ARRIVAL on the device |
| the verify | the first residency of layer l | the main stream waits the commit of layer l (CUDA events A_l / B_l around the wait). The verify finds a prefetched block as an ORDINARY POOL HIT (SWAP / MOVE_IN) and the urgent GPU gather skips it |
| `on_release()` | OUTSIDE a group, after the round's commit, before RequestPin.release | counts the tags that remain (unread / moved); every prefetch pin that is left gets the slot's age from before the commit back |
| `on_round("end")` | after the round | L4P: issues the last draft's snapshot. Its job lands while the GPU runs the commit; the next round's on_verify commits it |

THE EPOCH: the jobs that one on_verify commits. L3: the drafts of the same round. L4P: the draft row and drafts of the
previous round (or the prelude). The verify that needs a block is the one after the commit.

COMPLETE BLOCKS ONLY. A snapshot item (l, h, b, X) goes to the packer only if X < pos_b // 64, with pos_b the round-start
host mirror (CODE spec_round.py `run`: `pos_host` after the round's host sync). Every block below is on the host rows:
a plain step writes it back at once, and a group's fill is written at the group's commit, which ends before the host
sync. A block that a step of the current group filled stays staged (not on the host) and is not fetched (counted
`unsafe`). Each (layer, head, request, block) goes out once per epoch (a host table; counted `dup_epoch`).

ONE COMMIT PER LAYER AND EPOCH. The jobs of an epoch use distinct staging lanes (the lanes only grow in an epoch;
PoolPrefetch.busy is reset because the jobs never share a lane). The merged rows are PoolPrefetch's format [h, b, X,
lane, group start], sorted by (head, request). The commit stream waits the NEWEST job's event (the C++ worker is FIFO,
so every older job has landed then). One commit per layer instead of one per job cuts the host launches at the commit
point by the number of jobs (MODEL: about 55 small ops per layer and commit).

THE WAIT IS PER LAYER (paper 3.2: "both wait at a layer if verification's required KV is not ready"). The hook is the
OUTERMOST wrapper of `rs.body.decode_fixed` (bound after DraftResident, which wraps after the Rollback; CODE
spec_round.py `SpecRunner.__init__`). So the wait runs before the rollback's journal copy of the pool tables.

## 3. Rollback and prefetch: the proof (the task's question "journaled or outside the group: say which")

ANSWER: OUTSIDE THE GROUP. The prefetched pool slots and pins are written only by `on_verify` (the commit) and
`on_release` (the pin end), and both REFUSE to run inside a rollback group (`rb.tentative`). The undo never needs to know
that a block came from the prefetch.

The argument (CODE ragged_rollback.py docstring, THE JOURNAL, SLOTS, THE UNDO READS ONLY NAMED SLOTS):
1. A group journals, for each tentative step u and layer l, the engine part (the maps and the four pool tables) at the
   step's `decode_fixed`, and the old bytes of every slot that the step overwrites.
2. The commit of an epoch ends before the group begins (L3: the verify group; L4P: the joint pass's group). On the
   device: the main stream waits the commit of layer l before the first `decode_fixed` of layer l, so the first journal
   copy of layer l sees the committed tables.
3. So the state S0 from which every journal entry of the group starts already holds the prefetched blocks: their bytes
   in their pool slots, their ids in `_pool_map`, their pin ages in `_pool_age`.
4. The undo of the group gives back, per request, the state after its kept steps, starting from S0 (the rollback's
   claim, proven by test_rollback.py and test_spec_round.py for any S0 that satisfies the residency invariant). A
   prefetched block is a block like any other here: a SWAP / MOVE_IN / MOVE_OUT of it is undone by the inverse move of
   named slots; a lost slot past the journal's N that held it comes back from the HOST through the shipped gathers,
   and that is exact because a prefetched block is a COMPLETE block (section 2) whose host rows and total_cis rows are
   its bytes.
5. Nothing writes the pool between the group's begin and its commit except the group's own steps (the commit stream
   runs only at on_verify, and the main stream waited it).
6. on_release runs after the group's commit. It writes ages only (no slot bytes, no map), and only of slots with a
   prefetch pin age; the next group journals the result.
WHY THE ORDER MATTERS (the negative control `commit_mid_group`, L3, tests only): the commit runs inside the verify group,
at verify step K - 1. The undo of verify step 1 (a request with m = 1) puts back step 1's journal, taken BEFORE the
commit: the pool map names the victim block Y again, but the slot holds the bytes of the prefetched block X (no journal
saved them). The residency check (every named slot holds its host rows) catches it.

CPU PROOF (section 6): test_prefetch_tokens_equal_shipped, test_prefetch_tiny_journal_equals_shipped (a 1-slot journal,
with the precondition that a verify step / row that the round UNDID had hit a prefetched block),
test_commit_inside_group_is_caught (refusals and the plant).

## 4. The soft pin (paper 3.3) and pin conservation

- PoolPrefetch writes `_pool_age = PIN_BASE + n` for each committed slot. SpecPrefetch starts n at `PF_SEQ0 = 2**40`,
  so a prefetch pin is above every RequestPin age (PIN_BASE + [0, H B P)) and the two pins never mix. pool_update takes
  a pinned slot as an LRU victim last; a hit gives the slot a real stamp, so the pin ends at the first read.
- PIN OVERFLOW (counted, never silent): `commit_overflow` (no unpinned slot at the commit: the item is dropped and the
  verify gathers it urgently) and `evict_overflow` (pool_update evicts a pinned prefetched block because its row had
  fewer unpinned unclaimed slots than MOVE_OUTs). `evict_free` counts a prefetched block evicted although an unpinned
  unclaimed slot existed: 0 by the soft-pin rule (MUST gate PF-PIN). The plant `no_pin` gives the slot the victim's old
  age instead of a pin: evict_free > 0, and the recount sees tagged slots without a pin age.
- L4PP: RequestPin.apply pins the draft's pool HITS before the commit; the commit then takes only unpinned slots, so it
  never evicts a requested hit. The prefetch pins the fetched MISSES. That is the paper's "both newly fetched blocks
  and cache hits".
- THE TAGS (one (H, B, P + 1) table per layer; column P a dummy): the class of the job that delivered the slot's block
  (0 ready, 1 in flight) or -1. A verify residency counts each pool hit on a tagged slot (read_ready / read_inflight for
  a verify request, read_draft for a draft request of an L4P joint pass) and each MOVE_OUT of a tagged slot
  (evict_overflow / evict_free); a touched slot loses its tag. on_release counts the tags that remain: `unread` (the
  slot still holds the block) or `moved` (a draft step or an undo moved it).
- PIN CONSERVATION (MUST gate PF-CONS, exact per run): committed == read_ready + read_inflight + read_draft +
  evict_overflow + evict_free + unread + moved; no tag and no prefetch pin is left after a release; requested ==
  committed + dup_dropped + commit_overflow (PoolPrefetch's counters).

## 5. The measurement (paper 4.3: "report bytes AND time")

Per verify residency and verify request (the meter's verify bracket): `sel` = topk selected slots per KV head. The split:

| category | definition | blocks | bytes | wait |
|---|---|---|---|---|
| READY_resident | window hit or a pool hit of a block that the prefetch did not deliver | sel - new - read_ready - read_inflight | x block_bytes | 0 |
| READY_prefetched | pool hit of a prefetched block whose job had LANDED when on_verify started | read_ready | x block_bytes | 0 |
| IN FLIGHT | pool hit of a prefetched block whose job had NOT landed: the verify waited its event | read_inflight | x block_bytes | the per-layer device waits (sum of A_l -> B_l) + the boundary stall (on_verify's entry -> exit events on the main stream: the GPU idle time while the host waits the newest job's submission and enqueues the commit) |
| NEW | a host load of the verify: the urgent GPU gather | new | x block_bytes | the gather spans on the probe rounds (r2_spec_cell kv_at_verify, unchanged) |

block_bytes = 64 x 128 x 2 x 2 + 64 x 2 = 32,896 B (K, V and bias of one (layer, head, request) block, as in SCORING
files); the packer moves 32,768 B per item (K + V; the bias comes from the GPU-resident total_cis). Also per round:
`refetch` (a verify host miss of a block that the prefetch delivered in the same epoch), `victim_named` (commits that
evicted a named pool block), jobs ready / in flight at the verify (the backlog), items per class, the host ms of each
seam (issue, verify, release, accounting, the snapshot waits, the submission waits). Per run: delivery ms (PoolPrefetch's
issue -> arrival probes; a "late" sample is an upper bound, as in pf1), the worker's stats, lane refusals, unsafe and
duplicate items, uncommitted items (the last epoch).

USEFUL PREFETCH: read by the verify (read_ready + read_inflight), read by a draft only (read_draft, and part of moved),
read by nothing (unread + evicted). The CPU test recounts the split from the residency itself (section 6, SPLIT).

## 6. CPU tests for the lead

Run on a CPU node, inside a job, in the image (`/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif`: the PACKER test builds the C++
extension, as test_pool_prefetch.py does):

    cd /mnt/beegfs/ojerbi/nosi-r2-spec5
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py

Regressions (the spec suites and the hooks, as in NOTES_specp.md section 6; four of them got one exclude line):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_spec_paper.py benchmarks/Efficiency/tests_cpu/test_spec_round.py \
        benchmarks/Efficiency/tests_cpu/test_rollback.py benchmarks/Efficiency/tests_cpu/test_verify_rows.py \
        benchmarks/Efficiency/tests_cpu/test_ragged_engine.py benchmarks/Efficiency/tests_cpu/test_draft_pool.py \
        benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py benchmarks/Efficiency/tests_cpu/test_r2_mbase.py \
        benchmarks/Efficiency/tests_cpu/test_r2_tick.py tests/test_r2_base.py tests/test_r2_capture_step.py \
        tests/test_r2_p3.py

test_step_script_sp5 runs the full DRYRUN only at a clean checkout of THIS tree at HEAD; otherwise it skips. The step
scripts' test of nosi-r2-specp (test_step_script_sp3) skips here (its STEP_ROOT is the other tree), as before.

| test | claim | negative control / preconditions (L34) |
|---|---|---|
| `test_flags_off_is_specp[B1, B8]` | the tree = a6c9d10 except the declared files; the four older Gate 0 tests only ADD a line naming spec_prefetch; spec_round.py = a6c9d10's AST plus the guarded seams (docstrings aside); no engine module but spec_round.py names spec_prefetch; with prefetch None, THIS tree's round runner and the a6c9d10 SOURCE (loaded from git into the same synthetic package) give the same tokens, logits rows, counters, round records and final state, bit for bit, for L3 and L4P | both paths of a round ran (rejects and full accepts) |
| `test_prefetch_tokens_equal_shipped[L3-B1, L3-B8, L4P-B1, L4P-B8, L4PP-B8]` | committed tokens = the SHIPPED greedy decode, logits rows bitwise; residency (every named slot holds its host rows and cis rows: the prefetched blocks too) and host rows after every round; no soft pin after a round; pin conservation; evict_free 0 | the prefetch acted (jobs, prefetched verify hits); rejects and full accepts, a block edge, draft skips, pool moves, (B 8) ragged; L4PP: the RequestPin applied pins |
| `test_prefetch_tiny_journal_equals_shipped[L3, L4P]` | the same with a 1-slot journal | journal overflow and overflow_dead > 0; prefetched hits; a prefetched hit in a step / row that the round UNDID (the first world seed in 0 .. 2 that has one) |
| `test_prefetched_block_is_a_pool_hit[L3, L4P]` | in round 0 every prefetched pool hit of the verify is a HOST LOAD of the same verify call (same layer, call, request, head, block) in the run without the prefetch | at least one such block |
| `test_wait_split_adds_up[L3/L4P x ready/inflight]` | an independent recount (VerifyRecount: window W, pool P, prefetched F0 / F1, new N, from the maps and tags before each call) = the device split; sel = READY_resident + READY_pf + IN_FLIGHT + NEW; NEW = the load mask's host loads; ready transport: no IN FLIGHT; held transport: every prefetched hit IN FLIGHT, the submission waited, one commit wait per committed layer | - |
| `test_dropped_pin_is_caught` | every tagged slot has a prefetch pin age before each verify call; evict_free 0; the cell's PF-PIN passes | PLANT no_pin: CAUGHT by the recount (tagged slots without a pin), by evict_free > 0 and by PF-PIN; preconditions: tagged slots, evictions of tagged slots in the plant run |
| `test_commit_inside_group_is_caught` | on_verify / on_release refuse inside a group; the clean run with the same drafts is exact | PLANT commit_mid_group: CAUGHT (residency or tokens); preconditions: committed items, named victims, a request with m = 1 |
| `test_seams_and_refusals` | the seam order per round (L3: begin, K drafts, verify, release, end; L4P: the prelude's K drafts, then begin, verify, K drafts, release, end); refusals (unknown control, no pool, level L4, a second bind, attach / begin_step / prefetch, close before unbind); detach restores every callable | - |
| `test_host_syncs_are_declared[L3, L4P]` | in a run, spec_prefetch.py reads host values only in `_issue_snap` (the lag-1 snapshot); the round runner's syncs stay a6c9d10's (run, _edge_before) | the trap is not blind (run's per-round sync) |
| `test_real_packer_tokens_equal_shipped[L3, L4P]` | the REAL packer v3 (C++ worker, issue_desc, flat lanes) in CPU mode: tokens, residency, conservation, the recount | the prefetch acted |
| `test_cell5_arguments_and_report` | the cell's arguments and refusals; summarize / pf_summary / arm_gates / verdicts on CPU runs; STRESS fails without overflow and passes with a 1-slot journal and 4 lanes; L4PP's gates | a missing prefetch report FAILS the PF gates |
| `test_step_script_sp5` | wrong cells STOP; DRYRUN sp5m prints the cell line and the arguments; sp5a / sp5b STOP while PRED_STAMP5 is None | - |

If a precondition fails, the data do not exercise the path: tell me and I change the data, not the claim. The most
data-dependent ones: prefetched hits at B 1; an undone prefetched hit (tiny journal); evictions of tagged slots in the
no_pin plant; named victims and m = 1 in the commit_mid_group plant.

## 7. The GPU cells (REGISTERED block TBD for the lead)

    DRYRUN=1 bash benchmarks/Efficiency/r2_spec5_step.sh --nosi <sha> <holder> <run id> sp5m
    bash benchmarks/Efficiency/r2_spec5_step.sh --nosi <sha> <holder> <run id> sp5m    # the miniature first
    bash benchmarks/Efficiency/r2_spec5_step.sh --nosi <sha> <holder> <run id> sp5b
    bash benchmarks/Efficiency/r2_spec5_step.sh --nosi <sha> <holder> <run id> sp5a

| cell | C (pool) | B | steps | journal | lanes (staging) | stress gate | STEP_MIN | mem | GPU time (MODEL) |
|---|---|---|---|---|---|---|---|---|---|
| sp5m | 73 (10) | 8 | 64 | 8 | 8 (17 MB) | on | 40 | 100G | about 10-15 min (sp3m: 7 min; + the build, 1-2 min) |
| sp5a | 128 (65) | 216 | 512 | 256 | 128 (268 MB) | off | 110 | 240G | about 65-75 min (sp3a: 62 min, the same arm count) |
| sp5b | 73 (10) | 318 | 512 | 256 | 1024 (2.15 GB) | off | 160 | 340G | about 95-110 min (sp3b specp: 92 min) |

Total about 3 h of GPU time. All cells: `--spec 3 --spec-arms L3,L3P,L4P,L4PP --spec-prefetch on --spec-twin fused
--spec-body fused --spec-graph on --spec-redirect on --spec-pf-threads 8 --spec-pf-cap-groups 256 --probe-every 16`;
SH runs first. The step script STOPS sp5a / sp5b while `r2_spec5_cell.PRED_STAMP5` is None at the commit (L33). sp5m
needs no registration (wiring). The order sp5m, sp5b, sp5a puts the stressed cell (large fetch) before the C 128 cell.

MUST gates: SH-RUN, PF-BUILD; per arm <arm>-RUN, -ERR, -COMMIT, -TOK, -FP (every arm is EXACT); L3P / L4PP: -PF-ACT
(jobs > 0, prefetched verify hits > 0), -PF-CONS (pin conservation and READY_resident >= 0 per round), -PF-PIN
(evict_free 0), -PF-NEW (the split's NEW against the gather meter: L3P equal, L4PP at most; the meter also counts the
L4P draft requests' staged-block reads); sp5m: -STRESS (journal overflow and overflow_dead > 0, and a prefetch overflow
path ran). REPORT verdicts: L5-VS-L4 (L4PP / L4P, the plan's L5 gate) and L5-VS-L0 (L4PP / SH, the stop rule).

REGISTERED, TBD (`r2_spec5_cell.PRED_SPEC5` and `PRED_STAMP5`, from `date -u`, in a commit before sp5a / sp5b):

| key | quantity | value |
|---|---|---|
| L3P_tok, L4PP_tok | committed tokens bitwise SH's | TBD |
| L3P_vs_L3, L4PP_vs_L4P | median steady committed tok/s ratio, (low, high) | TBD |
| L3P_served, L4PP_served | (READY_pf + IN_FLIGHT) / (READY_pf + IN_FLIGHT + NEW) over the steady rounds, (low, high) | TBD |
| L3P_inflight_ms, L4PP_inflight_ms | median per steady round of the IN FLIGHT wait (device waits + stall), ms, (low, high) | TBD |

INPUTS FOR THE BANDS (MODEL / INFERRED, not measured):
- The ceiling of the benefit is the verify's gather WAIT per round (MEASURED): sp3b L3 136.5 ms of a 967 ms round,
  L4P 139.0 of 1,255 ms (SCORING_sp3b_fix.md section 8, SCORING_sp3b_specp.md section 7); sp3a L3 15.0 of 645 ms, L4P
  20.0 of 913 ms (SCORING_sp3a.md section 7). A perfect prefetch gives about +14 % (L3) / +12 % (L4P) at sp5b and
  +2.3 % / +2.2 % at sp5a, before its costs. Even then L4P reaches about 832 tok/s at C 73 (SCORING_sp3b_specp.md
  section 7): far below SH (2,575). THE PLAN'S STOP RULE (L5 must beat L0) WILL TRIGGER (MODEL); this rung measures
  the paper's 4.3 split and the prefetch's own cost.
- The traffic (MODEL from the draft skips, MEASURED in sp3b / sp3a): about 2.2 skipped blocks per stream and draft call
  at C 73 B 318 = about 45,700 items = 1.5 GB per draft step before the epoch dedup; about 0.4 at C 128 B 216 = about
  5,500 items = 180 MB. The packer delivered 192 MiB in 20.6 ms in pf1 (about 10 GB/s end to end; the urgent GPU gather
  moved about 25 GB/s in sp3b). At sp5b the packer may NOT keep up within one draft step: expect IN FLIGHT items and
  host submission waits; the lane cap (1024) may refuse items.
- Costs (MODEL): the snapshot (3 ops and 2.6 MB D2H per draft step); the host issue (5-40 ms per snapshot at B 318,
  hidden while the host is ahead); the commit (about 55 small ops per layer and epoch on the commit stream; at the L4P
  round start the GPU is idle while the host enqueues it: `stall_ms`); the accounting (about 35 small ops per verify
  residency and layer: L3 3, L4P 5 residencies per layer and round: about 3,400 / 5,600 launches per round, a few ms of
  host time each round, reported as `account_ms`). pf1: background packing interferes +0.25 % (64 MiB) / +1.46 % (192
  MiB per step).
- A second effect (INFERRED): prefetched blocks that stay in the pool after the release are pool hits for the next
  drafts, so the drafts skip less and acceptance can change against L3 / L4P.

## 8. Risks and open points

- Nothing ran. The CPU tests are my reading of the stand-ins and of the transport.
- HOST WAITS BY DESIGN: the snapshot of the previous draft step (lag 1) and the submission of the newest job at the
  commit point. If a draft step is host-bound, the lag-1 wait can shorten the host's lead. `snap_wait_ms` and
  `submit_wait_ms` report them.
- THE C++ WORKER AND CUDA GRAPHS: the worker's CUDA calls break a capture in the global mode (PoolPrefetch docstring).
  The ragged engine captures its pooling graphs at the warm step only (CODE ragged_engine.py `_select`); on_draft
  refuses a call while a graph is missing. No other capture runs in the spec path (CODE: verify_rows has none).
- MEMORY: sp5a's L4P peaked at 83.19 GB of 85.1 (SCORING_sp3a.md section 7); L4PP adds 268 MB of staging, 17 MB of
  rings and about 15 MB of tables. If L4PP stops on memory there, lower `--spec-pf-lanes` (64: 134 MB).
- THE POOL AS THE LANDING AREA (C 73: P = 10 slots per (head, request)): a commit evicts the LRU unpinned pool block
  (`victim_named`); the verify may then load that victim from the host. `refetch` counts only refetched DELIVERED
  blocks, not refetched victims.
- In L4P the commit of the drafts' fetches waits until the next round's start (the joint pass is the first verify that
  needs them, and it begins after the round's host sync). The last draft's job has the commit phase (MEASURED 190-265
  ms) to land.
- The IN FLIGHT class is decided per JOB at on_verify's entry (landed or not). A job that lands between that point and
  the layer's need is still IN FLIGHT; its device wait is then about 0 (the waits are reported per layer).
- The accounting's launches are inside the prefetch arms only, so L3P - L3 includes them (`account_ms` reports the host
  part).
- One commit per layer and epoch: a layer's ready items wait for the epoch's newest job too. The layer needs all its
  data anyway, so the verify's wait does not change; only the early commit of ready items is lost.
- The per-round issue counters of L4P belong to the next round's verify (the jobs that a round issues are committed by
  the next round).

## 9. Review (a read-only reviewer agent on the working tree before the commit) and the fixes

The reviewer found no defect in the pin conservation identity, evict_free = 0, the stream order (commit_ev[l] after
every commit-stream write; the main stream waits it before any read), the outermost hook, the AST strip (it removes
exactly the seven changes), the SyncTrap set (`_issue_snap` only) and the DRYRUN line.

| finding | fix |
|---|---|
| BUG: `refetch` counted almost every slot without a host miss: the commit's scatter set the dummy column NB of `delivered` to True | the commit sets column NB back to False; the count also ANDs the host-miss mask |
| BUG (test): Gate 0 part (c) checked that r2_spec_cell.py does not contain "spec_prefetch", but that unchanged file has the argument `a.spec_prefetch` | the check looks for `SpecPrefetch` and for an import of the module |
| RISK: a layer without a commit in an epoch was not accounted, so sel / new (and PF-NEW, the split test) could miss verify calls | every verify residency is accounted (all layers, every round); found by me before the report too |
| RISK: a failed packer job is not waited by the device (gilfree_pack.cpp `stream_wait_job` skips it), so its stale staging could be committed as block X | `_commit` calls `tp.poll()` after the submission wait: the worker's error raises before any commit; found by me before the report too |
| LIKELY BUG: `uncommitted` was 0 in the report (only close() counted it, after the report) | report() also counts the pending jobs of the last epoch |

## 10. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- `git worktree add` (as instructed), `git add`, `git commit -F` (the message written with the Write tool): git.
- `date -u` (the stamps, and a clock read while the reviewer agent ran; its progress was read with `ls -laL` on its
  transcript file only).
- Shell built-ins only otherwise (`cd`, `echo`, variable assignments, `;`, `&&`, `|`). No interpreter, no sed, no awk,
  no heredoc, no `cat >>`, no tr / paste / uniq / xargs, no hashing, no `find`, no srun / sbatch / pytest, no sleep.
  Files written with the Write and Edit tools. Two harmless slips: a no-op variable assignment named `sed_none=1` in
  front of `head -0`, and one named `sleep_unused=0` in front of `ls` (assignments, not commands).
- The read-only reviewer agent (section 9) reported `git status` (read-only), `cd`, `echo` and a `for` loop around `ls`.
