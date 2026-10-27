# NOTES: Task 1.5, accept, correct and roll back per request, the pool included (`--rollback`)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`, Task 1.5 (paper section 3.4).
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-rollback`, branch `r2-rollback`, from `0bdc045` (Task 1.3, `nosi-r2-ragged`).
- Status: SOURCE ONLY. I did not run a test or a job. The lead runs the commands below.
- Style: ASD-STE100.

## 1. What I made

| file | what |
|---|---|
| `nosi/nosi/ragged_rollback.py` (new) | `Rollback(rs, K, capacity, stream, neg)`: `begin`, `commit(n_acc, n_acc_host)`, `reject_all`, `release_event`, `stats`, `detach`; `accept_lengths` / `accept_perreq` (ported); `ImmediateStream` |
| `nosi/nosi/state_snapshot.py` | the pool fields `_pool_map / _pool_age / _pool_target / _pool_action` and the host `_pool_stamp_base` in the snapshot lists (only `benchmarks/Efficiency/acceptance_pilot.py` imports this module) |
| `benchmarks/Efficiency/r2_base_cell.py` | flag `--rollback` (default OFF) and `--rb-k / --rb-groups / --rb-seed / --rb-journal-blocks`; arms RBR (reference) and RBS (rollback) after SH; `rollback_schedule`, `rollback_compare`, `rollback_agreement`, `verdicts_rollback`, `run_rollback`; `PRED_ROLLBACK` (TBD) |
| `benchmarks/Efficiency/r2_base_rollback_step.sh` (new) | a copy of `r2_base_ragged_step.sh`: cells `rb1` (natural GEMM) and `rb1x` (exact GEMM), ROOT `/mnt/beegfs/ojerbi/nosi-r2-rollback`, output `/mnt/beegfs/ojerbi/r2-runs/rollback/rb_*`, claim `rb_active` |
| `benchmarks/Efficiency/tests_cpu/rollback_world.py` (new) | the CPU world of the tests (not a test file), on top of Task 1.3's `ragged_world.py` |
| `benchmarks/Efficiency/tests_cpu/test_rollback.py` (new) | the CPU tests |
| `benchmarks/Efficiency/tests_cpu/test_ragged_engine.py` | Task 1.3's Gate 0 now excludes the two files above from its "nosi/nosi = ded9d41" diff (one call; no assertion weakened otherwise) |

NO NEW KERNEL. NO KERNEL EDIT. `ragged_engine.py`, `cache_engine.py`, `nosa_llama.py`, the R2 modules and every kernel
are the `0bdc045` bytes. The rollback uses torch ops and these shipped kernels as they are: `flash_pool_swap` (the
inverse move), `flash_h2d_from_mask` / `flash_h2d_from_mask_bias` (host fallback, staging redirect),
`r2_writeback.masked_writeback` (the deferred write-back). Ported code: EQ `LayerJournal` / `truncate_tail` /
`accept_commit` (`nosi-eqmem` dde91d4 `spec_loop.py:588-765`), PR `accept_perreq` / `truncate_perreq` /
`commit_layer_perreq` (`nosi-r2-perreq` e58011c `spec_loop.py:1294-1590`), `state_snapshot.py`. The lockstep scalars of
EQ are per-request tensors here; PR's mirror-slot tail restore is a per-step row journal here.

### How it attaches (no edit of ragged_engine.py)

`Rollback(rs, ...)` replaces five callables of ONE RaggedState and keeps the originals (`detach()` puts them back):
`rs.ops.linear` (the release wait), each twin's `step_unchecked` (the pre-step journal), `rs.ops.kernels["pool_update"]`
(the slot journal), `rs.ops.kernels["gather_k" / "gather_v_bias"]` (the staging redirect; without a pool, the slot
journal), `rs.body.writeback_kernel` (the deferred write-back). Outside a group every hook calls the original only, so a
step is the Task 1.3 step bit for bit (`test_flag_off_is_shipped`). Both bodies work: `DeviceWritebackBody` and
`FusedBody` call `self.writeback_kernel` and the gathers through these names, and the tail rows are journaled before the
step (the fused kernel writes them).

## 2. The group, and every state field

`rb.begin()`; up to K tentative steps (`ragged_engine.decode_step`, unchanged); `rb.commit(n_acc)`: request b keeps its
first `n_acc[b]` steps. The undo runs steps u = n_run - 1 .. 0 in reverse; each op of step u touches only the requests
with `n_acc[b] <= u` (a `torch.where` over the request axis, or a masked kernel). Then request b holds its state from
before step `n_acc[b]`.

| field (per layer) | what a step changes | journal (before step u) | undo |
|---|---|---|---|
| twin `pos`, `n_comp`, `n_ccis` | +1 per event | (3, B) copy | where |
| twin `comp_k`, `ring`, `total_cis`, `ccis` | one row per request at the counters' index (`step_unchecked` and `r2_twin_fused` write the same rows) | the old rows and their 4 indices | where, at the same index |
| `_k_gpu / _v_gpu / _kv_bias_gpu`, tail row `4032 + pos % 64` | `write_tail` (torch body) or the FusedBody kernel | old rows (B, H, D) x 2, (B, H), the row index | where |
| `_k_gpu / _v_gpu / _kv_bias_gpu`, window and pool SLOTS | `flash_pool_swap` (SWAP, MOVE_IN, MOVE_OUT) and the host gathers (+ the redirect) | the plan (target, action) and the OLD DATA of every slot the step overwrites: the pool slot of a MOVE_OUT (the victim), the window slot of a MOVE_IN, a window slot loaded with no pool action (pool overflow, empty slot); at most N per layer and step, flat (cumsum + scatter, `index_select`) | (1) `flash_pool_swap` with the inverse plan (SWAP -> SWAP, MOVE_OUT -> MOVE_IN, MOVE_IN -> NONE); (2) the saved rows back (`index_copy_`); (3) a lost slot past N that held a block: the block from the HOST (shipped gathers, bias from `twin.total_cis`) |
| `_block_map`, `_cache_lens` | select, tail id, lengths | whole copy | where |
| `_new_block_map_buf`, `_load_mask` (per-step scratch) | diff, select | whole copy | where |
| `_pool_map`, `_pool_age` | `pool_update` | whole copy | where |
| `_pool_target`, `_pool_action` (the per-step plan) | `pool_update` | whole copy | where |
| `_pool_stamp_base` (host int, the LRU clock) | += 64 per step | value at `begin()` | `commit`: NOT moved back (every kept stamp is below it; the reference of the kept steps moves it the same way); `reject_all`: back to the `begin()` value |
| host rows `_k_cpu / _v_cpu` | the write-back of a filled tail block | - | DEFERRED: a tentative fill goes to a per-layer GPU staging block; `commit` writes the KEPT fills with `masked_writeback` (device mask, no host read) |
| `RaggedState.pos_host` (host mirror) | += active per step | value at `begin()`, host activity per step | `commit(n_acc, n_acc_host)` sets it from the CPU copy (no device read) |

NOT RESTORED, on purpose: the sticky refusal flags (`twin.overflow`, `rs.err`: a set flag voids the run); the RaggedState
scratch (`pos_before`, `n_used`, `M_b`, `score_buf`, the pooling / top-k buffers, the captured graphs) and FusedBody's
write-back mask (each step rewrites them before a read); `st.steps` (a count of the steps that ran).

WHY THE INVERSE MOVE IS EXACT. A SWAP is its own inverse. After a MOVE_OUT the displaced block Y is in pool slot q and
the gather wrote X into window slot m: a MOVE_IN of q into m puts Y back over X, then the journal puts the victim back
into q. Every non-tail block on the GPU is a copy of its host rows and its `total_cis` rows, so the host fallback is
exact. A lost slot past N that held NO block (empty) keeps other bytes: nothing reads it (attention reads mapped window
slots only; a pool slot is read only when `_pool_map` names it). Counters (`stats()`): `destroyed` (lost slots),
`saved` (in the journal), `overflow` (past N and held a block: the host fallback reloads them), `overflow_dead` (past N
and empty: not restored), `redirected` (loads served by a staging block). The first decode step after a prefill loses 63
EMPTY window slots per (head, request): at B 216 that is 27,216 per layer, so `overflow_dead` is large in group 0 of
rb1 by construction (review finding; the exactness does not change).

REDIRECT. A fill is deferred, so the host rows of that block are not written in the group. The next step selects the
block (it is in the forced local window) and the shipped gather reads its stale host rows; the gather hooks then copy K
and V of that block from the staging block (one more shipped gather each, block id 0 of the staging block). The bias is
right already (the gather reads it from `twin.total_cis` on the device).

THE dr1 CASE (`/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/SCORING_dr1.md` section 7). In Task 1.1's draft flow the
draft steps write their K / V into the live tail and nothing rolls them back; the DraftBlockEdge guard stops a draft from
the write-back, but the next SHIPPED edge step writes the whole 64-row block to the host (`nosi-r2-draftpool`
cache_engine.py:523-527 vs :930-936), so 45-47 draft rows of 64 reached the host rows in dr1. Here: (1) a rejected
step's tail row is put back by the tail-row journal and `pos` goes back, so the kept steps write those positions again
before the block can fill: the write-back at an edge (deferred inside a group, shipped outside it) copies rows of kept
steps only; (2) a fill inside a group is not written to the host before the commit, and only when the fill step is kept;
(3) in a draft + verify flow the DRAFT steps are always rejected (`commit(0)` / `reject_all`) and the exact steps (the
verify, Task 1.4) write the rows that are kept: draft K / V are never kept, also for an accepted token. Test: the second
half of `test_reject_at_block_edge_keeps_host_rows` (drafts with wrong tokens fill part of the tail, `reject_all`, then a
plain step fills the block); plant `no_rollback` reproduces dr1 (the draft rows reach the host at the edge: CAUGHT).

RELEASE EVENTS. `commit()` runs each layer's work through `stream.run` and records one event per layer
(`stream.record`). The next step waits every event before its first op that touches the engine: the hook is the step's
first GEMM (`rs.ops.linear`), because `ragged_engine._layer` reads `twin.pos` into `pos_before` before the twin step
(`ragged_engine.py:409`). The wait is `wait_event` (no host sync). A writer on another stream (Task 1.2's background
prefetch into pool slots) waits `release_event(l)`. With the default executor everything is on one stream, so the waits
are already satisfied; they matter when the commit's work runs elsewhere (the CPU test models that).

HOST SYNCS: none in the hooks, `commit()` and `reject_all()` (the static scan and the run-time trap below). `stats()`
reads the counters with one sync (tests and the cell only).

## 3. Tests for the lead

Run on a CPU node, inside a job:

    cd /mnt/beegfs/ojerbi/nosi-r2-rollback
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_rollback.py

Regressions of the touched files (r2_base_cell.py, the Gate 0 of Task 1.3, state_snapshot.py):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_ragged_engine.py benchmarks/Efficiency/tests_cpu/test_r2_mbase.py \
        benchmarks/Efficiency/tests_cpu/test_r2_fused.py tests/test_r2_base.py tests/test_r2_p3.py

| test | claim | negative control (precondition, L34) |
|---|---|---|
| `test_flag_off_is_shipped[B1, B8]` | engine sources = `0bdc045` except `ragged_rollback.py` and `state_snapshot.py`; no other nosi module names them; `--rollback` off parses as before and every rollback call in `main` sits under `if a.rollback` (5 calls); a world with the Rollback attached and no group = the plain world bitwise over 16 steps through a block fill; `detach()` restores the 6 callables | precondition: the 16 steps wrote host rows (a fill) |
| `test_accept_all_equals_shipped[B1, B8]` | S 4607 (fill at the first tentative step), 4 groups x K 3, n_acc = K: logits BITWISE the SHIPPED engine (`Llama.decode_inference`) and the plain ragged world at every step; whole state = the plain ragged world; engine tables (pool, host rows) = the shipped engine's | preconditions: the plain world wrote the block during group 0 while the rollback world did not (deferred); the redirect ran; the shipped pool did a MOVE_OUT; journal overflow 0 |
| `test_reject_all_restores_snapshot_with_pool[B1, B8, B8-nopool]` | n_acc = 0 after 12 warm-up steps: every tensor (pool, clock, host rows, twin, scratch, dead bytes) bitwise the snapshot; two `CacheSnapshot.take()` (the ported module) equal; the next 3 steps equal a world that never ran the group | preconditions: the group changed the slot, meta, twin (and pool) families; lost slots; a SWAP or MOVE_OUT; the clock moved |
| `test_negctl_reject_all_plants[no_meta_restore, no_pool_restore, no_slot_restore, no_tail_restore, no_twin_restore]` | each plant is CAUGHT (the state differs in its family); `no_pool_restore` is also seen by the ported `CacheSnapshot` | precondition: the group changed that family (and, for slots, lost or moved slots) |
| `test_partial_accept_equals_shipped_prefix[B1, B8]` | 4 groups, a different n_acc per request (every value 0 .. 3): kept logits equal the reference that ran ONLY the kept steps; whole state = that reference after every group (scratch excepted: the reference's inactive steps rewrite it); group 0: each request's rows (scratch included) = a world that ran n_acc[b] plain steps | preconditions: every n_acc value; a kept and a rejected fill; rejected steps lost slots and moved pool blocks; (B 8) a rejected MOVE_OUT evicted a LIVE block; the redirect ran; overflow 0 |
| `test_reject_at_block_edge_keeps_host_rows[B1, B8]` | (1) the host rows of each request's tail block do not change at ANY tentative step; after each commit the state = the reference (a kept fill reaches the host at the commit); (2) THE dr1 CASE: 3 draft steps with wrong tokens write part of the tail block (no fill), `reject_all`, then plain steps: a plain step fills the block and writes it back; the host rows of that block = the world that never ran the drafts, bitwise; logits and whole state equal | (1) a rejected and a kept fill; a fill before the last step (redirect). (2) no draft fills; a plain step reaches the edge (6 of 8 requests at B 8) and writes finite host rows; the draft rows differ from the kept rows at the same positions |
| `test_negctl_no_rollback_leaks_draft_rows_at_the_edge[B1, B8]` | plant `no_rollback` (commit undoes nothing = the Task 1.1 flow of dr1): the edge write-back puts draft rows into the host rows: CAUGHT (host rows and state differ) | the same preconditions as (2); the drafts themselves wrote no host row |
| `test_negctl_no_defer_writes_host_rows[B1, B8]` | the plant writes the host rows inside the group: CAUGHT ("host rows of block") | precondition: the host row past S is NaN (poison), the fill writes finite rows |
| `test_negctl_no_redirect_reads_stale_host_rows[B1, B8]` | without the redirect the next step reads the stale (NaN) host rows: CAUGHT (kept logits or state) | precondition: the redirect ran in the normal run |
| `test_rollback_frees_after_last_reader[B1, B8]` | a deferred executor (`DeferredStream`): the next step waits every layer's release event (`ran_by == "wait"`), and the state equals the immediate executor's, also after one more step | plant `skip_release_wait`: preconditions: the events were still pending when the step ran, and the step overwrote the journal that the undo reads; CAUGHT (the state differs) |
| `test_no_host_sync_in_commit_path` | static scan of the 21 functions of `COMMIT_PATH`: 0 sync constructs; run-time trap (`item`, `tolist`, `cpu`, `numpy`, `nonzero`, `__bool__`, `__int__`, `__float__`, `__index__` called directly from the module): 0 during tentative steps, `commit`, `reject_all` (B 1 and B 8) | planted `int(nacc.max())` line: the scan finds exactly it; `neg="host_sync"`: the trap records it |
| `test_journal_overflow_falls_back_to_host[B1, B8]` | capacity 1: every non-slot tensor exact, the LIVE slot bytes exact (host fallback), the next 3 steps equal a world without the group | preconditions: overflow > 0 and an unsaved lost slot held a block; plant `no_host_fallback`: CAUGHT (live bytes differ) |
| `test_fused_body_rollback_under_the_interpreter` | `FusedBody` (the GPU cells' body) under the rollback = the torch-body reference, partial accept (child with `TRITON_INTERPRET=1`; skips without triton) | preconditions: redirect ran, overflow 0 |
| `test_rollback_cell_arguments_schedule_compare_verdicts`, `test_accept_perreq`, `test_refusals`, `test_step_script_rb1` | cell arguments and refusals, the seeded schedule, the bitwise kept-row compare (one ULP caught, a not-kept row ignored), the agreement helper, the TBD verdicts (REPORT) and the filled ones, `accept_perreq` values and no host read, the Rollback refusals, DRYRUN of rb1 / rb1x | - |

In PARTIAL and EDGE a rejected tentative step reads a WRONG token (the true one + 1), as a rejected draft does, so its
rows differ from the rows that the kept step at that position writes later; a leak is then visible.

CPU world: L 2, 2 KV heads, head dim 8, pool P 8, prompts of 4605-4733 tokens (72-74 blocks: the 64-slot selection
leaves blocks out, so the pool moves blocks), K 3, journal capacity 1024 (the first decode step loses 63 slots per
(head, request): 1008 at B 8). KNOWN PITFALL handled: `define_undefined` gives `_new_block_map_buf` / `_load_mask` one
value; every data row is poisoned (NaN) or zero already; each test asserts "two worlds bitwise equal before step 0" on
the whole state.

## 4. GPU cells rb1 / rb1x (same job: SH = mb1, RBR, RBS)

    bash benchmarks/Efficiency/r2_base_rollback_step.sh --nosi <sha> <holder> <run id> rb1
    DRYRUN=1 bash benchmarks/Efficiency/r2_base_rollback_step.sh --nosi <sha> <holder> <run id> rb1

- SH = cell mb1 (`--batch 216 --steps 512 --pool 65 --mbase`, `NOSI_ATTN_SPLITS=4 NOSI_POOL_BLOCKS=65`): the same-job
  baseline, byte for byte the mb1 arguments (no fingerprint is added to SH).
- One schedule: 128 groups, K 3, `n_acc[g, b]` uniform in 0 .. 3 (`--rb-seed 0`, `rollback_schedule`). Request b's step
  u of group g reads SH's token `inputs[b, kc + u]` at position `16128 + kc + u` (kc = its kept steps so far).
- RBR: the ragged engine, plain steps, only the requests with `u < n_acc[g, b]` active (it runs ONLY the kept steps).
- RBS: the same engine with the Rollback: every request active in the K tentative steps, then `commit(n_acc[g])`.
- RB-BIT: the logits fingerprint (max, float64 sum, argmax) of every kept row (g, u, b) of RBS bitwise RBR's. Per group:
  `groups_unequal` and the first unequal (g, u, b). A wrong commit changes the next groups' rows too.
- Cost per group: CUDA events around `commit()` (RB-COMMIT: median / mean / p90 ms); the tentative step against the
  reference step (RB-STEP); journal counters (destroyed, saved, overflow, overflow_dead, redirected) per layer.
- rb1: `--ragged-gemm natural` (the shipped GEMM; the comparison is at one B, as Task 1.3's RGS-BIT). rb1x:
  `--ragged-gemm exact` (PaddedLinear: the plan's EXACT mode). Both `--ragged-twin torch --ragged-body fused`.
- STEP_MIN 75, `--mem 240G` (three prefills of 657.7 s each at mb1, 512 + 384 + 384 steps, 128 commits).

REGISTERED BLOCK, PREDICTIONS TBD FOR THE LEAD. Fill `PRED_ROLLBACK` in `r2_base_cell.py` and its `stamp` (from
`date -u`, L33) in a commit BEFORE the run. Until then each prediction prints REPORT.

| name | quantity | registered value |
|---|---|---|
| RB-BIT | every kept row of RBS bitwise RBR's | TBD |
| RB-STEP | RBS median tentative step / RBR median step | (TBD, TBD) |
| RB-COMMIT | RBS median commit per group, ms | (TBD, TBD) |
| RB-OVF | no lost slot that held a block past `--rb-journal-blocks` (256) (`overflow`; empty ones are `overflow_dead`) | TBD |

MUST gates (not predictions): SH-RUN, RBR-RUN, RBR-ERR, RBR-CAP, RBS-RUN, RBS-ERR, RBS-CAP.

INPUTS FOR THE LEAD'S BANDS (INFERRED, not measured):
- RB-BIT relies on the same GPU facts as RGS-BIT (per-row independence at one B: cuBLAS at the same M, the per-row
  kernels, split-KV attention with forced splits), plus: `flash_pool_swap` with the inverse plan restores the bytes
  (the kernel loads both sides before it stores), `index_select` / `index_copy_` copy bytes, and the staging redirect
  writes the bytes the shipped write-back would have written.
- RB-STEP: per tentative step and layer about 45 small ops more (the copies of the pre-step journal, the pool hook with
  cumsum / scatter / 3 `index_select`, the two redirect gathers over (B, H, 64) programs, the staging `where`). About
  1,400 launches per step more; the step is device-bound at B 216 (64.8 ms device, about 24.5 ms host for mb1), so the
  host part may hide. RBR's inactive requests load nothing, so RBR's step is a little faster than an all-active step.
- RB-COMMIT: per layer and step run (K = 3 always, masked), about 30 small ops, one `flash_pool_swap`, 6
  `index_select` / `index_copy_`, the two fallback gathers over (B, H, 129) programs and two redirect gathers: about
  40 launches x 96 = 3,800 launches per commit. If the commit is host-bound, about 20-40 ms per group; the device work
  is small (about 0.2 GB of copies). A captured commit (Task 1.6) can remove the host part: the ops have fixed shapes
  and no host read.
- RB-OVF: the ncu step of mb1 has 5,256 loads per step over 32 layers (about 164 per layer); a lost slot is about one
  per load after the pool fills. 256 slots per layer and step is about 1.6 x that mean; peaks are not measured. The
  first decode step (group 0, step 0) loses 63 EMPTY slots per (head, request): they go to `overflow_dead`, not to
  `overflow` (the empty pool slots of the early MOVE_OUTs too).
- MEMORY: RBS adds about 1.0 GB of journal (256 slots: about 10.6 MB per layer and step x 32 x 3) and 0.45 GB of
  staging blocks to the ragged engine. mb1's allocator peak was 82.04 GB. If RBS stops on memory, lower
  `--rb-journal-blocks` (RB-OVF then counts the host fallbacks).

## 5. Risks and open points

- I did not run anything. A failing PRECONDITION means the CPU data do not exercise that path; tell me and I change the
  data (W0, prompts, groups), not the claim. The most data-dependent ones: the shipped MOVE_OUT in 12 steps (accept
  test), a LIVE victim after 12 warm-up steps (partial B 8), an unsaved live slot at capacity 1 (overflow B 1).
- The commit's cost is host launches (section 4). The fallback and redirect gathers launch full grids even when no
  entry is set (their programs exit at once); a host read of the overflow counter would skip them but adds a sync.
- The clock: `commit` keeps it monotone; only `reject_all` moves it back. The GPU reference (RBR) moves it by 64 per
  step in the same way, so the absolute `_pool_age` values agree. A caller that needs "the clock of n_acc steps" must
  not use `commit`.
- Dead bytes of a lost EMPTY slot past N are not restored (no reader; counted). The CPU tests that compare whole
  tensors use capacity 1024 and assert overflow 0; the overflow test compares live bytes.
- The release wait is at the step's first GEMM for every layer at once (not per layer), because `ragged_engine` reads
  `twin.pos` before the first hookable call of a layer. `release_event(l)` stays per layer for other writers.
- A tentative step that raises (for example more than K steps) leaves an open group; there is no recovery API. A commit
  after it undoes the steps that the twin hook recorded.
- K <= 64: one staging block per request and layer (one fill per group).
- `FusedBody` with the rollback is tested only under the Triton interpreter on the CPU. `masked_writeback` with a GPU
  staging tensor as `k_gpu` passes its own argument checks by reading (contiguous, same strides, tail0 0 + 64 <= 64).
- `n_acc_host` must be a (B,) CPU tensor (or list); `commit` checks it before any work (review finding: a late check
  could leave a half-done commit). Without it `RaggedState.pos_host` is not updated (no engine path reads it with the
  device write-back body).
- The review also noted that `/mnt/beegfs/ojerbi/r2-runs/rollback` does not exist yet: the step script creates it with
  `mkdir -p "$OUTROOT"` before the claim (r2_base_rollback_step.sh:154), so nothing is needed.
- The CPU fakes assume CUDA's `torch.topk(sorted=False)` order (Task 1.3's assumption).

## 6. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- none. (Shell built-ins `cd`, `;`, `&&`, `|` and redirections only; `git worktree add` as instructed; files written
  with the Write and Edit tools; the commit message with the Write tool and `git commit -F`.)
