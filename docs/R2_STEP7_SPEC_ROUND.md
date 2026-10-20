# R2 step 7: the speculative round on the R2 path (DESIGN NOTE and CPU test plan)

Branch `feature/nosi-r2-m2-spec` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-spec`, from `54ea1d6`).
Status: DESIGN. This note contains no engine code and no GPU result.
Prior-art gate: CLEARED by the lead (retroinfer-eval 3650572). The rows are in
`/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/r2-prior-art-rows.md`; this note cites them as [PA-n] (row n) and
[PA-b n] / [PA-c] (sections b, c).

Sources (read, not copied):

- Paper section 3 (`kstep-cpu-runs/paper-draft.txt:233-400`).
- K3X exact slice (`/mnt/beegfs/ojerbi/nosi-kstep-bc` 91e6e47): `nosi/nosi/kstep/cadence.py` (C1-C4),
  `nosi/nosi/kstep/accept.py` (A2 `accept_greedy`, A1 `accept_cohort` with the S4 normalized rule and `exact_sum`).
  The integration contract (`retroinfer-kstep-bc docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md`)
  is used for section 4 (the numerical rule) and section 5 (the row-stable GEMM).
- The verify agent's EXACT VERIFY (`/mnt/beegfs/ojerbi/nosi-r2-verify/nosi/nosi/spec_loop.py:918-985`, read only).
  Every verified position there is one shipped step. It is the CORRECTNESS reference for commit and rollback
  BOOKKEEPING in the tests of section 7. It is not the throughput path.
- R2 at `54ea1d6`:
  - `r2_decode_path.py` (G0 wiring, `build_ragged`);
  - `r2_positional_twin.py` (shape-stable summaries);
  - `r2_decode_body.py` (`DecodeBody`, `resync` 63f9873).
  - GPU results: G0 PASS (2145 bitwise comparisons; NEGT / NEGF first differ at steps 40 / 63); ragged main and long
    PASS (1100 steps).

## 1. What step 7 builds, and the main performance requirement

### 1.1 Mapping

| Paper section 3 | R2 step 7 | Precedent [PA] |
|---|---|---|
| 3.1 draft over GPU-resident KV, misses requested in the background | draft rows attend only resident blocks of their selection; the other selected blocks are requested into the request's slot store (side stream) | ours [PA-4] |
| 3.1 verification every K steps, newest token deferred, bounded lead | cadence.py C1 cohorts (K = 3: [1,2], [3,5], [6,8], ...), C2 lead bound LAMBDA = 7 | ours (K3X) |
| 3.3 draft and verify rows in one pass | ONE flattened call per tick: the verify rows of every ready cohort (each position its own single-token decode row) and the draft rows, one weight read | vLLM indexer flatten [PA-3]; SparseSpec mixes phases of DIFFERENT requests [PA-7] |
| 3.2 / 3.4 urgent misses, retention | the verify rows' blocks must be resident before the tick (urgent misses gathered, in-flight transfers waited for); a block is HELD by the cohorts that requested it until their decision and reader event | HiSparse resolver idea [PA-c] |
| 3.5 per-request accept / correct / roll back | `accept_greedy` / `accept_cohort` per cohort; per-request counter reset of the exact bank; draft bank reset | vLLM `num_computed_tokens -= num_rejected`, SGLang `kv_committed_len`, DSv4 per-draft banks [PA-1, PA-2, PA-c] |

The difference from SparseSpec [PA-7; code read 2026-10-20, HEAD 5c9616c], stated once:

- SparseSpec batches DIFFERENT requests at different phases in one forward (shared weight pass). Each request is
  NORMAL, DRAFT, VERIFY or STALL per step (`serve/request/request.py:15-19`), never both draft and verify. A verified
  request stalls instead of drafting ahead (`spec_scheduler.py:576-586`).
- Our call batches the SAME request's draft row with its own verify rows of an earlier cohort, plus other requests'
  rows of either kind. SparseSpec is the precedent for the cross-request mixed batch and the shared weight pass; cite
  it there.
- Its attention templates and KV chunking depend on the row count and on the whole batch (FlashInfer fork
  `scheduler.cuh:1085-1166`). So it is not batch-invariant, and its exactness is distributional only. The bitwise
  claim of section 3 has no precedent there.

### 1.2 The main performance requirement: the joint pass replaces the separate verify

B's per-request pilot (B = 64, 2 documents; lead's report 2026-10-20):

- yield per request per round: K1 1.978, K2 2.909, K3 3.810;
- draft step 139 ms, shipped step 167 ms;
- separate verify 229-338 ms per round;
- 272-302 tok/s vs shipped 640.

The separate verify forward is the cost to remove. In step 7 there is no verify forward: a cohort's verify rows ride
in the tick that also carries the draft rows.

Requirement R-PERF, per tick:

- A tick costs T_tick = T_draft-rows + T_verify-rows (a single weight pass, with more GEMM rows and more attention
  rows).
- Each request commits r tokens per tick on average (r = committed / request-ticks, measured by the dispatcher
  counters).
- Throughput vs the shipped decode = r x T_shipped / T_tick. Step 7 must report this ratio with both terms measured,
  and it must exceed 1 for the design to pay.
- With B's numbers (T_shipped 167 ms, a draft-only tick about 139 ms) and r near 0.9, T_tick must stay below about
  150 ms at B = 64. So the flattened rows (up to B x (1 + 3) rows) may add at most about 8 % to a draft-only tick.

The 4.5 harness supplies the GEMM / attention split per row count. At b64 its private group measured 74.84 ms
separate vs 57.66 ms joint (weight sharing alone).

## 2. Per-request state and the round state machine

### 2.1 Notation (cadence.py)

- An epoch of request b starts at committed position E_b with the pending token u_0 = x_b.
- Draft step s feeds u_{s-1} at position E_b + s - 1 and produces d_s with q_s recorded.
- Cohort m verifies d_i, i in I_m. Its verify rows feed u_{i-1} at positions E_b + i - 1 and produce p_i.
- C3: full acceptance commits |I_m| tokens and NO bonus. A rejection at i* commits d_a .. d_{i*-1} and the
  correction c; the next epoch starts at E_b + i* with u_0 = c.
- C4: at max_new_tokens the drafts after the last triggered cohort form a final short cohort.

### 2.2 Per-request record

`epoch`, `E`, `s`, `committed_in_epoch`, the tokens `u`, the draft records `d[i]` / `q[i]`, the cohort queue (m,
[a, b], state, the device argmax `g[i]` or float32 `p[i]`), the held blocks per cohort, and two banks of the
positional tables:

- EXACT bank: committed + verify-provisional;
- DRAFT bank: exact prefix + draft rows.

### 2.3 States

| State | Meaning | Leaves when |
|---|---|---|
| DRAFT | a draft row this tick at E + s | the tick ends: s += 1; at s = mK cohort m is TRIGGERED |
| STALL_LEAD | lead = LAMBDA (C2) | a decision commits |
| STALL_MEM | no space for the next draft step's blocks / rows (paper 3.4) | a release |
| cohort TRIGGERED | waiting for its blocks (every verify row's selected blocks resident or in flight) and for workspace | READY |
| cohort READY | all its rows can run | it joins the next tick as \|I_m\| flattened verify rows |
| cohort DONE | p / g of all its rows computed | the tick's decision (one host sync per tick that has a DONE cohort) |
| ROLLBACK | rejection or cancel | section 6 completes |
| FLUSH / DONE | C4 | the last cohort is decided |

- A request can have one draft row and the verify rows of one cohort in the same tick. Its oldest undecided cohort
  goes first; a later cohort waits for the earlier decision (C3: a rejection cancels it).
- A cohort's selections are computed by its own verify rows inside the tick (section 4.1). So "its blocks resident" is
  checked against the DRAFT rows' selections of the same positions, which requested them (paper 3.1). A verify row
  whose exact selection names a block that is not resident triggers an urgent gather before the attention of that
  layer (paper 3.2, section 5.3).

### 2.4 One tick

1. Host: pick the rows. Draft rows: requests in DRAFT. Verify rows: the READY cohorts, \|I_m\| rows each, in position
   order. Row table (host, then one H2D copy): request, kind, position, bank, token.
2. Per layer, for ALL rows together (flattened), with the batch-invariant (BI) ops of section 3:
   1. rmsnorm, wqkv, nosa_linear, rope;
   2. the positional tables: every row advances its bank by one position in position order (`PositionalTwin.advance`
      semantics per bank); each row gets its own counters (n_comp, n_ccis, pos);
   3. stage 1 (per row: its bank's compressed-key table, its own seqused_k), pooling (per row: its n_comp and
      M_b = pos // 64 + 1), the deterministic top-k (section 3.2);
   4. residency: the verify rows' selected blocks must be resident (urgent gather of the rest); the draft rows' misses
      are queued as background requests;
   5. the rows' own K / V / bias go into the tail pages (section 5.2);
   6. ONE BI attention call: every row its own page list (canonical order) and its own extent;
   7. wo, FFN.
3. Head: verify rows -> p / g (device); draft rows -> d (device).
4. If a cohort is DONE: ONE host sync, the decisions, the rollbacks. No other sync in the tick.

## 3. The batch-invariant (BI) decode: the reference and the exact claim

### 3.1 Why bit-identity needs batch invariance

- R2 g2: the Path-1 verify logits differed from the decode by up to 0.41, and the committed greedy tokens parted at
  near-ties on every request. The contract's numerical rule is `torch.equal` to a declared per-request twin; a
  tolerance gate is not an integration gate.
- AGENTS.md law 1 (the lead's note): the float order of a row must be a function of the request's OWN state, never
  of the batch size. Thinking Machines (blog) states the same principle [PA-3]. vLLM's batch-invariant mode and
  SGLang's batch-invariant ops are the recipe [PA-3, PA-c]: no split-KV, one tile size for decode and verify rows,
  no shape-keyed autotune. The in-house T10 kernel is per-request fixed tiling (G-Kinv) [PA-3].
- A multi-position verifier gated bit-equal to serial decode exists in no code we found [PA-3, PA-b 4]. That
  property and its gate are ours.

### 3.2 The ops, one by one

| Op | Shipped behaviour | BI status | Step 7 action |
|---|---|---|---|
| embedding, rope, silu_and_mul | elementwise | BI | none |
| rmsnorm (flashinfer) | one program per row | BI expected | microgate B1 |
| nosa_linear (Triton, B is a constexpr) | per-row program; a new compile per row count | BI expected (same arithmetic per row) | microgate B1 |
| 5 GEMMs (wqkv, wo, gate_up, down, lm_head) | cuBLAS picks the kernel by M | NOT BI (paired 2175550: 0/11 rows equal at M = 2B) | an M-INVARIANT GEMM (3.3) |
| stage 1 (infllm-v2 varlen, Split = false) | one batch entry per row, tile by head dim, seqused_k per entry | BI expected (no split, no batch-keyed choice) | microgate B2 |
| pooling (`nosa_pooling_per_request`) | one program per (head, row, block) | BI | k1 gate G4 / G5 |
| top-k pair (`torch.topk`, sorted=False) | the algorithm and the tie set can depend on the row count; k1 G7: ties at the cut in 85-105 of 128 rows | NOT guaranteed | a DETERMINISTIC selection: a stable sort by (value descending, index ascending) and the first k; the same rule in the serial reference (k1 FYI) |
| attention (FA2-NOSA, split-KV 4, the 64-slot window in diff slot order) | split partition from the allocated length; reduction order = slot order = the request's diff history | NOT BI for multi-position rows (one window, several positions) | a BI paged attention (3.4) |

### 3.3 The M-invariant GEMM (law 1)

Two candidates. The step measures both against cuBLAS and keeps the cheaper one that passes microgate B3:

- (G-a) a batch-invariant Triton matmul. SGLang's `batch_invariant_ops` (adapted from thinking-machines-lab;
  Apache-2.0; keep the attribution header if adapted) [PA-3]. It has fixed tiles and no split-K, so the per-row
  result is independent of M.
- (G-b) cuBLAS at a FIXED padded M_fix per run: every flattened call pads its rows to M_fix with zero rows.
  - The kernel choice then never changes in a run.
  - A GEMM has no cross-row reduction, so a row's result does not depend on the values of the other rows.
  - The float order is a function of M_fix, a run constant, not of the batch. The serial reference pads to the same
    M_fix.
  - Cost: GEMM work for M_fix rows on every tick.

Microgate B3 (GPU):

- (i) a row's output is `torch.equal` whatever the other rows hold;
- (ii) the same holds at every row index;
- (iii) for G-a, also at every M.

Cost table: wqkv, wo, gate_up, down and lm_head at the tick's row counts (B in {16, 64}, 1 + U in {1, 2, 4}). The
columns are cuBLAS at the natural M, G-a and G-b. This is the coordinator's "cost of the M-invariant GEMM path
against cuBLAS".

### 3.4 The BI paged attention

- Each row reads its KV through its OWN PAGE LIST: the slots that hold its selected blocks, in ASCENDING BLOCK-ID
  order (canonical), then its tail page(s). The last page is cut at the row's own extent.
- The bias of each page row is the CIS value of that token. Masked columns do not exist, because a row reads only its
  own pages.
- One tile size; NO split-KV (num_splits = 1); no autotune. So the reduction order is a function of the row's own
  selection and position only (law 1).
- The page size is the 64-row block. FA2-NOSA's paged mode needs `page_block_size % 256 == 0` and its bias index
  ignores the block table (contract section 5). So the kernel is NEW: a Triton decode kernel with a page-list
  indirection and a per-row bias.
- Candidates to adapt (cite, keep headers):
  - vLLM's unified-attention 2-D path under batch invariance (one tile size) [PA-3];
  - the in-house T10 `blockdiag_attend.py` (per-request fixed tiling, G-Kinv) [PA-3].
- Microgate B4: a row's output is `torch.equal` when it is computed alone, in a batch of 64 rows, and next to rows of
  its own request at other positions.
- Cost against FA2-NOSA split 4 at B = 64 and B = 16. No split-KV costs parallelism at small B; this is measured, not
  assumed.

### 3.5 The declared reference and the claim

- REFERENCE: the BI SERIAL decode. This is the R2 decode in which every op is the BI one of 3.2 to 3.4, one position
  per request per call, from the same committed prefix.
  - Its extent is the row's own position + 1, so F4 is NOT reproduced: the BI reference is a new declared target.
  - Its relation to the shipped decode is a REPORT, gate G0-BI (section 7.2): tolerance, argmax agreement, and every
    greedy difference classified by the shipped top-2 margin.
- CLAIM (the gate's exact claim): for every request and every verified position p, in any tick composition (any
  number of positions per request, other requests, draft rows of the same request), the flattened verify row's
  logits are `torch.equal` to the BI serial decode row of that request at p.
  - The committed greedy sequence equals the BI serial greedy sequence.
  - In sampled mode, `accept_cohort` (S4) gets p bitwise from the reference, so the output law is the reference's.
- WEAKER CLAIM, if the M-invariant GEMM is not adopted (cost): every non-GEMM term is bitwise as above, and the
  logits differ from the reference only by the GEMM kernel choice. This is measured per tick (max \|d\|), and every
  greedy difference is classified by margin. The gate then says so and does not call itself exact.

## 4. Flattened rows and the positional tables

### 4.1 Rows and tables

- The row table for a tick: request, kind (verify / draft), position, bank, token, M_b, last-page length, page-list
  offset.
- The positional tables: ONE `PositionalTwin` per layer with 2B requests. Rows [0, B) are the EXACT banks; rows
  [B, 2B) are the DRAFT banks. This is SGLang DSv4's "bank 0 = committed, banks 1..N = per-draft states, lazy commit"
  [PA-2, PA-c], with one draft bank per request. The twin's tables are per request and shape-stable (step 6), so one
  2B twin holds both banks.
- A verify cohort of \|I_m\| positions advances the exact bank by \|I_m\| positions in one call: `advance(keys, cis,
  n)` per layer gives the counters after each position, ctr (T + 1, 4, 2B).
  - Row u of request b reads `ctr[u + 1]` for its stage-1 seqused_k (n_comp) and its pooling width.
  - The draft rows advance the draft bank by one position.
- Stage 1 shares a request's compressed-key table among its rows: cu_seqlens_k[row] = bank x Lk_max (repeated for
  the rows of one bank), seqused_k[row] = that row's n_comp. This is the shared-prefix alias of SOURCE_MAP 3.4;
  BlockInfo takes the start from cu_seqlens_k and the length from seqused_k.
- Order inside a call: the twin updates of all rows come BEFORE stage 1, as in the shipped decode (a compression that
  a later position needs from an earlier position's raw key is then present).

### 4.2 Draft bank sync

- Epoch start: copy the exact bank's rows above the last sync point into the draft bank, and its counters.
- Full acceptance of cohort m: the exact rows of the published positions replace the draft rows of the same
  positions, and the draft chain is REBUILT (K3X F5 `_rebuild_dchain`). Then the draft rows from position E + b on
  use summaries computed from exact keys where their windows cover published positions.

## 5. Storage: the slot store, the tail pages, residency

### 5.1 Slot store

- Per (request, KV head): S slots of 64 rows (K, V, bias). They hold historical blocks (< the tail block), and each
  slot has a block id or FREE.
- Residency is CONTENT: the set of (block id -> slot). The ORDER of slots never reaches the arithmetic (pages are
  listed in block-id order). So the shipped 64-slot diff window and its slot order (the engine agent's "slot order
  differs after a rollback") stop mattering.
- `DecodeBody`'s diff / gather machinery is reused as the residency mover (the same two gathers, the same
  byte-equality argument as spec_loop E4). Its per-position tail and F4 bookkeeping is not used by BI rows.
- Size: S >= max over live rows of a request of its selection plus the union across its live rows. K3X keeps the
  guard C >= (2K + 1) x max_sel (contract section 4). With K = 3, 7 x 63 = 441 slots per (request, head) is the
  guard's value; the measured union over 4 positions is about 71 blocks (union_K16, the 4.5 harness). The step
  reports the peak.

### 5.2 Tail pages (the slot-63 problem removed)

- Per request: a RING OF TWO EXACT TAIL PAGES and two DRAFT TAIL PAGES. Position p writes row p % 64 of page
  (p // 64) % 2 of its bank.
- A verify cohort that crosses a block boundary writes block T's last rows in one page and block T+1's first rows in
  the other. Both stay readable in the same call.
- A rollback inside or across a boundary needs NO row restore: rows beyond the committed position are simply outside
  every later extent until rewritten. (This removes the engine agent's open slot-63 item for the BI path.)
- Host write-back of block T: when T is COMMITTED complete (all 64 positions committed), its exact tail page is copied
  to the host rows and registered as a historical block. A rejected provisional fill writes nothing to the host.

### 5.3 Background requests and urgent misses

- A draft row's selected blocks that are not resident are requested (side-stream gather into FREE slots, after the
  event of the slot's last reader), held by the cohort that will verify those positions.
- Before a tick: the host polls arrivals (`event.query()`, no sync).
- In the tick, at layer l, a verify row's exact selection can name a block that is neither resident nor in flight.
  That block is fetched by the shipped gather on the main stream before the attention of that layer (urgent, paper
  3.2). A block in flight is waited for (main-stream wait on its event), never fetched twice.
- Release: after the cohort's decision AND its reader event. Ownership is versioned by epoch, so a late arrival or
  late reader of a cancelled epoch cannot touch a slot of the new epoch.

## 6. Rollback per request

- Exact bank: `commit(ctr, keep)` to the counters after position E + i* - 1. The verify rows at E + i* .. E + b - 1
  are undone, a depth of b - i* < \|I_m\| <= K (MAX_ROLLBACK 32).
- Tail pages: nothing to copy (5.2).
- Draft bank: reset to the exact bank at the new epoch (4.2). The draft pages are void (extents).
- Slot store: content only. Blocks held by the cancelled cohorts are released after their events (5.3). A block
  stays resident if another holder or the cache policy keeps it.
- Tables compared after a rollback: BY CONTENT (bank rows below the counters; the set of resident (head, block id)
  and the rows of each block), as the lead asked. Slot order is irrelevant for BI rows by construction.
- Bookkeeping oracle (tests only): the verify agent's exact path. Every verified position is one shipped step. It gives
  the committed tokens and counters that the per-request commit must reproduce in the teacher-forced tests (7.1 S11).

## 7. Gates

### 7.1 CPU test plan

Files: `tests/test_r2_spec_round.py`, `benchmarks/Efficiency/tests_cpu/test_r2_spec_round.py`. The kernels are torch
stand-ins: a BI paged attention reference in float64, stand-in gathers.

| Id | Test | Negative control |
|---|---|---|
| S1 | cadence on the R2 record (cohorts [1,2], [3,5], [6,8]; trigger at s = mK; lead stall at LAMBDA; flush; LAMBDA <= K refused) | a cohort that takes the newest token |
| S2 | state machine over scripted outcomes (all accept, reject at each index, cancel, flush), B = 4 ragged; per tick at most one draft row and one cohort per request | two cohorts of one request in one tick are refused |
| S3 | decisions through `accept_greedy` / `accept_cohort` (imported); one host sync per tick that has a DONE cohort, none otherwise (a counting sync hook) | a second sync in a tick |
| S4 | flattened rows: the 2B twin advanced by a cohort in one call equals \|I_m\| serial single-position advances, per row counters included | the twin advanced in the wrong order (compression before the raw key it needs) |
| S5 | stage-1 alias: cu_seqlens_k repeated per bank, seqused_k per row, against per-row separate entries (stub op recording the arguments) | a row given its bank's final n_comp instead of its own |
| S6 | the deterministic top-k: equal to a reference stable sort on random data with forced ties; independent of the row count and row order | torch.topk(sorted=False) on tie-heavy rows (the set can differ; skip if the CPU kernel happens to agree) |
| S7 | the BI paged attention reference: a row computed alone == in a batch == next to rows of its own request at other positions (float64 and float32 CPU) | a split-KV variant whose partition depends on the batch's longest row |
| S8 | page lists: canonical order, own extent, two tail pages across a boundary; equal to a direct read of the row's units | a page list in slot order (differs after a scripted residency move) |
| S9 | rollback inside and across a block boundary: the exact bank by content equals a never-advanced bank; tail pages need no copy (the next rows read only committed rows) | the one-page tail (slot-63 style) loses committed rows of block T |
| S10 | draft bank sync and the F5 rebuild: the draft chain equals a chain recomputed from the exact rows | a draft bank that keeps its old rows |
| S11 | end to end on a CPU stand-in model (torch R2 path with the BI ops, B = 4 ragged), teacher-forced and free-running: every flattened verify row `torch.equal` to the BI serial decode row; committed tokens and counters equal the bookkeeping oracle | GEMM rows grouped at a varying M with an M-dependent rounding stand-in |
| S12 | slot store: request / arrive / hold / release after decision + reader event, epoch-versioned; urgent miss on the main stream; no double fetch | release before the reader event; a late arrival of an old epoch |
| S13 | the step script: GPU uuid from the lead's holder file (section 8); the gate arguments | a missing or malformed uuid file STOPs |

### 7.2 GPU gates (holder steps)

| Id | Gate | Kind |
|---|---|---|
| B1-B4 | the BI microgates of section 3 (rmsnorm / nosa_linear, stage 1, GEMM row invariance and the cost table, BI attention invariance and its cost against FA2-NOSA split 4) | MUST for B1, B2, B4 and the chosen GEMM's B3; cost REPORT |
| G0-BI | the BI serial decode against the shipped decode, 64 steps, B 1 and 8 | REPORT (tolerance, argmax, margin classes); E / E2 determinism of the BI serial decode MUST |
| SG1 | teacher-forced: the decisions follow a schedule that rejects at every index and across a block boundary; every computed verify row `torch.equal` to the BI serial row at the same (request, position), rejected rows included | MUST; control: row 0 against the next serial step must differ |
| SG2 | free-running greedy, B = 8 ragged: committed sequences `torch.equal` to the BI serial greedy sequences | MUST |
| SG3 | phase mix: ticks with draft rows of the same request and verify rows of other requests; the SG1 equality holds in every composition | MUST |
| SG4 | negatives: (a) cuBLAS at the natural M in the flattened call, (b) split-KV attention, (c) `torch.topk(sorted=False)`. Each must differ from SG1 somewhere, or the gate reports that it could not detect it | MUST to run; detection REPORT |
| SG5 | R-PERF: T_tick, r, r x T_shipped / T_tick at B 16 / 64; the separate-verify time that no longer exists; slot-store peak; urgent misses per tick; decision syncs per tick | REPORT |

## 8. The step script (when written)

- The pattern of `r2_g0_step.sh` (pins, claim, frozen record, sacct row, a main() body).
- GPU_UUID is NOT hard-coded:
  - it is read from the lead-owned file `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/<holder job id>.gpu_uuid` (one
    line, no "GPU-" prefix);
  - the script STOPs (exit 1, before any claim) when the file is missing or the value is not a uuid;
  - the value and the path are recorded in `command.txt`;
  - the child keeps its `nvidia-smi` check (exit 3).
- `r2_ragged_step.sh` is pinned and is not changed.

## 9. Order of work after this note

1. The CPU stand-ins and tests S1-S13 (portable).
2. GPU microgates B1-B4 and the GEMM / attention cost tables, one holder step. Then the decision on G-a vs G-b
   (cost) or the weaker claim.
3. The BI paged attention kernel (Triton) with its CPU interpreter test.
4. The flattened tick on the GPU: SG1, SG2, SG3, then SG4, SG5.

## 10. Open questions and risks

- Q1 Without split-KV, attention at small B has fewer programs (one per row and head). The cost at B = 16 may be the
  limit. vLLM accepts this under batch invariance; measured in B4.
- Q2 The BI reference is not the shipped decode: no F4, canonical page order, the M-invariant GEMM. Shipped equality
  becomes a report. G0 / ragged stay valid for the shipped-order path; they do not cover the BI path.
- Q3 Slot-store size. The K3X guard is large (7 x 63 slots per head); the union over live rows is about 71. The step
  measures the real peak and the memory budget includes it (paper 3.4).
- Q4 The urgent-miss path runs a gather on the main stream inside a layer. It is a host-free device op, but it
  serializes with that layer's attention.
- Q5 SparseSpec [PA-7] code read (2026-10-20): no path puts one request's draft row and its own verify rows in one
  call, and nothing is bitwise. The paper must narrow "no released code batches draft and verify rows in one launch"
  to "the same request, drafting while its earlier draft is verified", and cite SparseSpec for the cross-request
  mixed batch.
