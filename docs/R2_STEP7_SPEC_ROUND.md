# R2 step 7: the speculative round on the R2 path (DESIGN NOTE and CPU test plan)

Branch `feature/nosi-r2-m2-spec` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-spec`, from `54ea1d6`); G0-BI continues on
`feature/nosi-r2-m2-g0bi` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-g0bi`, from eadd87a).
Status: DESIGN, confirmed by the lead at f1f9f56. 972579c / eadd87a add the portable core
(`nosi/nosi/r2_spec_core.py`), the CPU tests S1-S15 and the GPU microgate step B1-B5 (RAN 2026-10-20, holder step
2188818.88, every MUST gate PASS; `/mnt/beegfs/ojerbi/r2-runs/bi/bi_bi1_2188818`). The G0-BI commit adds the BI
serial decode (`nosi/nosi/r2_bi_decode.py`), its gate and step (source only, not run).
The BI paged attention is piece 3 (`/mnt/beegfs/ojerbi/nosi-r2-m2-biattn`, the verify agent; v2 7c826c1 is the pin).
The BI GEMM (G-a) is built in `/mnt/beegfs/ojerbi/nosi-r2-m2-bigemm` (26c45bd, the verify agent); this branch built
G-b, and G-b is the decision (3.3). The flattened tick (piece 4, SG1-SG5) is the verify agent's, on this note.
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

The 8 % is the DESIGN TARGET (lead, 2026-10-20). SG5 measures it (section 7.3).

Taint of the pilot numbers (lead, 2026-10-20):

- The pilot's `spec_loop._assign` gave a non-contiguous issue map to Triton gathers that read `load_ids` flat. The
  fix is nosi-r2-verify 333eb41. The g2 / p1 / 4.1-spec pilot numbers are tainted.
- If the numbers above (139 / 167 ms, 229-338 ms, 272-302 tok/s) come from those runs, they are tainted too. So SG5
  measures T_shipped and the draft-only tick again in its own run and states the budget from them.
- The R2 path does not use `spec_loop._assign`. Static check on this branch (2026-10-20):
  - `git ls-files | grep spec_loop` is empty: the tree has no spec_loop module.
  - `grep -n "_assign\|spec_loop"` over `nosi/nosi/r2_*.py`, `benchmarks/Efficiency/r2_*` and
    `benchmarks/Efficiency/tests_cpu/test_r2_*` finds one line only: `r2_kernels_step.sh:5`, a comment that names the
    step-script pattern `spec_loop_r2_step.sh`.
  - The R2 modules import only `r2_positional_twin`, `r2_decode_body`, `stage1_seqused`, `max_pooling_per_request`,
    `nosa_llama` and `cache_engine` from the package. None of these files contains `spec_loop` or `_assign`.

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
  - OWNER: the verify agent, `/mnt/beegfs/ojerbi/nosi-r2-m2-bigemm`, with its own GPU gate (lead, 2026-10-20). This
    branch builds no BI matmul. B3 loads that module BY PATH when its pins are set in `r2_bi_step.sh` (empty now).
- (G-b) cuBLAS at a FIXED padded M_fix per run: `r2_spec_core.PaddedLinear(m_fix)` (this branch).
  - Every call copies its live rows into rows 0 .. M-1 of a persistent (M_fix, K) buffer and runs `F.linear` on the
    whole buffer. The rows past M are NOT cleared: B3 proves that their content does not reach a live row.
  - The kernel choice then never changes in a run.
  - A GEMM has no cross-row reduction, so a row's result does not depend on the values of the other rows.
  - The float order is a function of M_fix, a run constant, not of the batch. The serial reference uses the same
    PaddedLinear (one live row).
  - Cost: GEMM work for M_fix rows on every tick, plus the copy into the buffer.

Microgate B3 (GPU):

- (i) a row's output is `torch.equal` whatever the other rows hold (G-b: random rows vs zero rows past M);
- (ii) the same holds at every row index (G-b: the live rows in a shuffled order);
- (iii) for G-a, also at every M;
- natural cuBLAS: REPORT the number of rows that differ between M and M = 4B (the premise of this section).

Cost table (lead, 2026-10-20): wqkv, wo, gate_up and down of NOSA-8B at B 64 and B 128, M in {B, 2B, 3B, 4B}
(the tick has B draft rows plus 0 .. 3B verify rows, U = 3), M_fix = 4B. The columns are cuBLAS at the natural M,
G-b (`padded_ms`, the copy included) and G-a (`bi_ms`, when its pins are set), with the ratios against the natural
M. This is the coordinator's "cost of the M-invariant GEMM path against cuBLAS". The registered predictions are in
the commit message of this step.

MEASURED (2026-10-20; microgate eadd87a, `/mnt/beegfs/ojerbi/r2-runs/bi/bi_bi1_2188818`; the bigemm gate 26c45bd,
`/mnt/beegfs/ojerbi/r2-runs/bigemm/bigemm_v1_2188818/bigemm_gate.md`). Ratios against cuBLAS at the natural M:

| linear | G-b at M = B, B 64 | G-b at M = B, B 128 | G-b at M = 4B (B 64 / 128) | G-a at M = 64 / 128 / 256 |
|---|---|---|---|---|
| wqkv | 2.00 | 2.24 | 1.19 / 1.13 | 2.62 / 2.46 / 2.83 |
| wo | 2.21 | 2.35 | 1.19 / 1.12 | 2.85 / 2.49 / 2.63 |
| gate_up | 1.41 | 2.23 | 1.06 / 1.02 | 1.15 / 1.40 / 1.77 |
| down | 1.71 | 2.70 | 1.07 / 1.05 | 1.55 / 1.86 / 2.33 |

- P1 (cuBLAS at the natural M depends on M) HELD: wqkv, wo and down differ on ALL rows at M = B and 2B (B 64 and
  128) and at 3B for wo / down at B 128. gate_up showed no difference at the tested M (empirical; not relied on).
- P2 (G-b invariance) HELD: 0 differing rows (random vs zero padding, shuffled rows).
- P3 (G-b cost) mostly HELD: at M = B inside 1.4-2.2 (B 64; wo 2.21 at the edge) and 2.0-3.2 (B 128). At M = 4B the
  1.00-1.15 band is MISSED by wqkv / wo at B 64 (1.19): the copy and one launch cost about 13 us on a 67 us GEMM.
- P4 (G-a cost) PARTLY FALSIFIED: wqkv / wo 2.5-2.9 at M <= 2B (band 1.3-2.5), gate_up 1.77 / 1.61 at M >= 256
  (band 1.1-1.5); gate_up and down at M <= 2B inside.

DECISION (lead, 2026-10-20): G-b, PaddedLinear with M_fix = 4B, for ALL FIVE linears in the EXACT mode.
- A per-linear choice is lawful (fixed per linear, not a function of M); G-b wins or ties for every linear at the
  tick's row mix. B 64, ms per layer: wqkv 0.080 (G-b) vs 0.104-0.188 (G-a, M 64-256); wo 0.079 vs 0.099-0.175;
  gate_up 0.380 vs 0.405 (G-a in-phase mean) / 0.372 (M 128); down 0.193 vs 0.252; lm_head 0.74 vs 0.79.
  Per tick: G-b 24.2 ms, G-a 30.0 ms in phase, 28.1 ms at M 128. B 128: G-b about 41 ms, G-a 46 ms or more.
- A per-M switch is NOT lawful: a row's bits would depend on M.
- DECLINED: keeping gate_up on natural cuBLAS after an empirical sweep over M. Equality over the tested M is no
  guarantee (the library may pick another kernel at an untested M), and it is a shape-keyed choice in spirit.
- THE GEMM MODE IS A RUN PARAMETER: EXACT (G-b; the exact claim) and FAST (natural cuBLAS; the weaker claim of 3.5).
  SG5 measures both, so the paper states the measured cost of exactness (about 7 ms per tick at B 64, all of it the
  G-b padding, section 7.3). `r2_bi_decode.BIOps(gemm="exact" | "fast" | "bigemm")`; the piece-4 tick takes the same
  parameter.

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
- Cost against FA2-NOSA split 4 at B = 64 and B = 128 (the lead's sizes). No split-KV costs parallelism at small B;
  this is measured, not assumed. B4 also measures the SG5 attention term (section 7.3).

The kernel is PIECE 3 (the verify agent; `/mnt/beegfs/ojerbi/nosi-r2-m2-biattn`, `nosi/nosi/bi_paged_attention.py`).
v1 (4c1243f, blob 7477ddd) is correct and slow (7.3). The PIN is v2 (7c826c1, blob 39ed5e1894ca1662f92e384a10e01a4eaab99042;
GPU gate PASS): a FIXED 8-page split-KV (PAGES_PER_SPLIT = 8, a module constant), a GQA tile, a fixed-order combine.
Its float order equals v1's for rows of at most 8 pages and differs for longer rows; the reference twin
`r2_spec_core.paged_attention_reference` (v1's order) agrees with v2 to rounding there (CPU test G9). The interface
is unchanged (v2 adds an optional workspace):

- `bi_paged_decode(q, k_cache, v_cache, bias_cache, seq_idx, page_table, num_pages, last_len, softmax_scale=None,
  out=None) -> (R, Hq, D)`.
  - q (R, Hq, D) bf16, one row per draft or verify row of the tick.
  - k_cache / v_cache (N, S*64, Hkv, D); bias_cache (N, S*64, Hkv), the token's CIS.
  - seq_idx (R,) int32, the storage entry of each row.
  - page_table (R, Hkv, P) int32, slots in CANONICAL order (ascending block id), from `build_page_table`.
  - num_pages (R, Hkv) int32 >= 1 (the tail page is always listed); last_len (R,) int32 = position % 64 + 1.
- Two tail pages: for a position in block T+1, the list has T's page (read whole), then T+1's page (cut at
  last_len). `r2_spec_core.page_lists` gives the same list when T is in the row's selection (NOSA's local window
  always selects it).
- One 64-row page per online-softmax step, no split-KV, fixed num_warps 4 / num_stages 2, no autotune, P constant
  per run.

THE BIAS SEMANTICS ARE THE SAME (lead's question, 2026-10-20). The reference `r2_spec_core.paged_attention_reference`
computes `s = (q k^T + bias) * scale`, with the bias added before the scale, as FA2-NOSA does
(`flash_fwd_kernel.h:950-985`) and as piece 3 does. Two executable checks:

- CPU S7b: the reference against piece 3's torch twin and its float64 oracle, within 1e-5. The last_len values are
  1, 2, 9, 37, 63 and 64, with tail pairs and one-page rows. Negative controls: the bias-after-scale semantics and a
  cut one row late both differ by far more than the tolerance.
- GPU B4: piece 3's kernel against the reference, max |d| <= 0.02 (MUST).

`r2_spec_core.paged_attention_reference` stays as the INDEPENDENT twin of the contract. This branch builds no
second kernel. B4 loads piece 3 BY PATH: the step script copies the pinned blob into the step record.

### 3.5 The declared reference and the claim

- REFERENCE: the BI SERIAL decode. This is the R2 decode in which every op is the BI one of 3.2 to 3.4, one position
  per request per call, from the same committed prefix.
  - Its extent is the row's own position + 1, so F4 is NOT reproduced: the BI reference is a new declared target.
  - Its relation to the shipped decode is a REPORT, gate G0-BI (section 7.2): tolerance, argmax agreement, and every
    greedy difference classified by the shipped top-2 margin.
  - The exactness claim is accepted against the BI reference (lead, 2026-10-20). Because the BI reference is not the
    shipped decode, G0-BI must also give a QUALITY number against the shipped decode, so that the paper can say what
    BI costs in FIDELITY, not only in time (section 7.2).
- CLAIM (the gate's exact claim): for every request and every verified position p, in any tick composition (any
  number of positions per request, other requests, draft rows of the same request), the flattened verify row's
  logits are `torch.equal` to the BI serial decode row of that request at p.
  - The committed greedy sequence equals the BI serial greedy sequence.
  - In sampled mode, `accept_cohort` (S4) gets p bitwise from the reference, so the output law is the reference's.
- WEAKER CLAIM, if the M-invariant GEMM is not adopted (cost): every non-GEMM term is bitwise as above, and the
  logits differ from the reference only by the GEMM kernel choice. This is measured per tick (max \|d\|), and every
  greedy difference is classified by margin. The gate then says so and does not call itself exact.
- Both are RUN MODES (3.3): EXACT (G-b) gives the claim; FAST (natural cuBLAS) gives the weaker claim. SG5 and the
  paper report both.

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

File: `benchmarks/Efficiency/tests_cpu/test_r2_spec_round.py` (S1-S15), on `nosi/nosi/r2_spec_core.py`. No CUDA and
no Triton launch. The attention is the torch reference (float32, and float64 inputs where stated). Piece 3 has its
own CPU tests (`test_bi_paged_attention.py` in its branch); S7b checks that the two agree on the contract.

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
| S7b | the contract agreement with piece 3 (its pinned blob, loaded by path in a fresh process): the reference vs piece 3's torch twin and float64 oracle within 1e-5; last_len 1, 2, 9, 37, 63, 64; tail pairs; one-page rows | the bias added after the scale; a cut one row late |
| S13 | the step script: GPU uuid from the lead's holder file (section 8); piece 3's pins (commit, path, blob) and the BI GEMM pins (unset); the gate arguments | a missing or malformed uuid file STOPs |
| S14 | G-b `PaddedLinear`: equal to x @ w.T; lead dims kept; more than m_fix rows refused; one buffer per (K, dtype, device); the live rows do not depend on the buffer rows past them (random vs zero) | a stand-in whose rows read the padding |
| S15 | the microgate's GEMM shapes equal the shapes derived from `/mnt/beegfs/ojerbi/models/NOSA-8B/config.json` (gate_up 32768 x 4096, down 4096 x 16384; 972579c had the Llama-3-8B values) | the Llama-3-8B MLP shapes are refused |

G0-BI's CPU tests are `benchmarks/Efficiency/tests_cpu/test_r2_g0bi.py` (G1-G9): the fixed-order mean (alone vs
batch; the reverse order caught), the twin's mean hook (default unchanged; B = 1 vs 3 bitwise), the page-table glue
(slot order and a short extent caught), BIOps / attach_bi and the GEMM modes, subset_state (the other order caught),
the dispatch, the quality helpers on planted logits, the step script's uuid file and pins, and the contract of the
v2 pin against the reference twin (20-, 9-, 8- and 3-page rows; bias-after-scale caught).

### 7.2 GPU gates (holder steps)

| Id | Gate | Kind |
|---|---|---|
| B1-B5 | the BI microgates of section 3, one step `r2_bi_step.sh` (B1 rmsnorm / nosa_linear / rope; B2 stage 1 with the alias; B3 GEMM row invariance and the cost table: natural, G-b, G-a when pinned; B4 piece 3's attention: invariance, the independent twin, the cost against FA2-NOSA and the SG5 attention term; B5 the deterministic selection) | MUST for B1, B2, B4, B5, G-b's B3 and G-a's B3 when pinned; cost REPORT |
| G0-BI | the BI serial decode against the shipped decode. QUALITY (lead, 2026-10-20): greedy token agreement over >= 256 steps on >= 8 documents (teacher-forced on the shipped tokens, so every step compares the same prefix; and free-running: the first divergence step per document); the distribution of the logit max \|d\| per step and of the shipped top-2 margin (quantiles and histogram), every argmax difference classified by its margin | REPORT (the fidelity cost of BI); E / E2 determinism of the BI serial decode MUST |

G0-BI as built (`benchmarks/Efficiency/r2_g0bi_gate.py`, step `r2_g0bi_step.sh`; `nosi/nosi/r2_bi_decode.py`):

- The BI serial decode is a MODE of the R2 path: `attach_bi(state, BIOps(...))` on a body-'r2' R2State; then
  `r2_decode_path.decode_inference` dispatches to it. Ops: the GEMM mode (EXACT = G-b PaddedLinear(M_fix)), piece 3
  v2 on the body's GPU window (page table = the 64 selected slots in ascending block id, slot 63 named by the row's
  own block, cut at p % 64 + 1), the deterministic selection, and a FIXED-ORDER mean for the twin's two decode
  summaries (torch's `.mean` may pick its reduction strategy from the shape, i.e. from B).
- THE SHIPPED DECODE IS VALID FOR 64 STEPS ONLY (findings F1). So the quality has two parts:
  - part W, against the SHIPPED DECODE ITSELF: 4 windows per document, prefixes S - 192, S - 128, S - 64, S (all
    0 mod 64), each a full F1 window of 64 steps: 8 documents x 256 steps;
  - part C, CONTINUOUS 256 steps against the R2 shipped-order path (body r2, f4_fix False: the shipped ops in the
    shipped order, F4 reproduced; bitwise the shipped decode inside the F1 window by G0F), built by build_ragged.
- MUST (part C): E2 = E again, torch.equal at all 256 steps; I1 = request 0 ALONE (B = 1, cut from E's own build
  by `subset_state` before E's first step), torch.equal to E's row 0 at all steps; I2 = the batch REVERSED, equal row
  by row; the first tokens of every build equal. MUST (part W): the three prefill logits of every window equal.
- REPORT: the quality of both parts (as the table row above), the free-running first divergence per document (and
  its consistency with the teacher-forced first disagreement), the negative control NI (I1 with the GEMMs at the
  natural row count: must be detected), ms per decode step per arm.
- The step runs EXACT, B 8, M_fix 32 (= 4B), S 16128, attention pin 7c826c1.
| SG1 | teacher-forced: the decisions follow a schedule that rejects at every index and across a block boundary; every computed verify row `torch.equal` to the BI serial row at the same (request, position), rejected rows included | MUST; control: row 0 against the next serial step must differ |
| SG2 | free-running greedy, B = 8 ragged: committed sequences `torch.equal` to the BI serial greedy sequences | MUST |
| SG3 | phase mix: ticks with draft rows of the same request and verify rows of other requests; the SG1 equality holds in every composition | MUST |
| SG4 | negatives: (a) cuBLAS at the natural M in the flattened call, (b) split-KV attention, (c) `torch.topk(sorted=False)`. Each must differ from SG1 somewhere, or the gate reports that it could not detect it | MUST to run; detection REPORT |
| SG5 | R-PERF: T_tick, r, r x T_shipped / T_tick at B 64 / 128; T_shipped and the draft-only tick measured again in the same run; the budget terms of 7.3 per tick; the separate-verify time that no longer exists; slot-store peak; urgent misses per tick; decision syncs per tick | REPORT (the 8 % design target) |

### 7.3 SG5: the tick budget (the 8 % design target)

The budget per tick, against the draft-only tick T_d (B rows, the shipped ops):

    T_tick - T_d = dGEMM + dATT + dSEL + dXFER + dHOST

- dGEMM: the M-invariant GEMM at the tick's rows against cuBLAS at the B draft rows (B3's table at the tick's M).
- dATT: piece 3's BI attention over the tick's rows against FA2-NOSA over the B draft rows (B4's SG5 term).
- dSEL: stage 1, pooling and the deterministic top-k for the extra rows.
- dXFER: the verify rows' blocks made resident (urgent misses, waits on transfers in flight).
- dHOST: the plan, the page lists and one decision sync per tick that has a DONE cohort.

Rows per tick. With K = 3 a request adds K verify rows per K draft steps (I_1 has K - 1). So a request has 1 + u = 2
rows per tick on average. When the requests are in phase, one tick in three has 1 + 3 rows and two have 1 row.

The attention term with piece 3's registered ratios (BI over B x (1 + u) rows / FA2-NOSA split 4 over B rows):

| B | 1 + u = 1 | 2 | 4 | mean in phase (4, 1, 1) | out of phase (2) |
|---|---|---|---|---|---|
| 64 | 1.4 | 1.9 | 2.8 | 1.87 | 1.9 |
| 128 | 1.1 | 1.6 | 2.4 | 1.53 | 1.6 |

MEASURED (piece 3's own GPU gate, lead 2026-10-20): BI / FA2-NOSA split 4 = 8.8 / 13.4 / 21.7 at 1 + u = 1 / 2 / 4
(B 64) and 9.0 / 13.3 / 22.1 (B 128). The registered 1.1-2.8 is FALSIFIED (and so is P5 of 972579c for this kernel).

- With these numbers the tick mean is about 13.1 (B 64, in phase) to 13.4 (out of phase), so dATT is about
  12 x a x T_d: the 8 % target holds only if attention is below about 0.65 % of the draft-only tick. a is not
  measured yet (SG5 measures it), but 0.65 % is a very small share for a step that reads 64 blocks per KV head.
- So the one-page-per-step, no-split kernel is not expected to carry the tick. The verify agent builds a FIXED-SIZE
  split-KV version: a split size that is a run constant (not a function of the batch or of the longest row), so a
  row's partition and combine order depend on its own page count only, and law 1 still holds.
- The table above keeps the registered values for reference; SG5 uses the measured ones of the kernel it runs.

- With the registered ratios, dATT is about 0.9 x a x T_d at B 64 and about 0.6 x a x T_d at B 128. Here a is the
  attention share of T_d (FA2-NOSA). The draft rows pay the BI ratio at 1 + u = 1 too, because they ride in the same BI call.
- The attention term alone uses the 8 % target when a >= 0.08 / 0.9 = 9 % (B 64) or a >= 13 % (B 128).
- dGEMM = (rho_pad - 1) x g x T_d for G-b, where g is the GEMM share of T_d and rho_pad is B3's padded ratio at
  M = B (M_fix = 4B on every tick). G-a: rho_bi averaged over the tick's M.
- SG5 measures a and g with per-op CUDA events in the draft-only run, and every term above per tick in the
  flattened run. It reports which term exceeds its share. The target is a design target, not a gate.

MEASURED, v2 and the registered predictions of 972579c (2026-10-20):

- Piece 3 v2 (7c826c1; fixed 8-page split): BI / FA2-NOSA split 4 over the B rows = 1.31 / 1.50 / 2.25 at
  1 + u = 1 / 2 / 4 (B 64) and 1.13 / 1.40 / 2.10 (B 128); back to back 1.07 / 1.27 / 2.04 and 1.06 / 1.32 / 2.20
  (`/mnt/beegfs/ojerbi/r2-runs/biattn/biattn_v2_2188818`).
- P5 (the SG5 attention term): FALSIFIED by v1 (8.85 / 13.5 / 21.9 and 9.4 / 12.0 / 23.2, my microgate); v2 is
  inside the registered bands except B 64 at 1 + u = 2 (1.50 against 1.6-2.6: faster than predicted).
- P6 (attention on the same rows): FALSIFIED by v1 (6.26 / 6.22 against the falsifier 3). Not measured for v2.
- P7 (invariance and twin): HELD; max \|piece 3 - twin\| = 0.00049 (predicted about 0.004, limit 0.02); against the
  float64 oracle 0.00035.
- P8, the budget at B 64, with T_d = 139 ms (B's pilot number; possibly tainted, SG5 measures it again):
  - dGEMM (EXACT, G-b): 32 layers padded to 256 rows = 23.4 ms, plus lm_head about 0.75 ms, against 14.5 ms natural
    at M = 64 (the draft-only tick): about +9.7 ms.
  - dATT (v2): FA2 split 4 over 64 rows = 0.205 ms x 32 = 6.6 ms; the v2 tick mean is 1.50 (out of phase) to 1.62
    (in phase): about +3.3 to +4.1 ms.
  - Together about 13-14 ms = 9.4-10 % of T_d, BEFORE dSEL, dXFER and dHOST: the 8 % target is MISSED by GEMM and
    attention alone in the EXACT mode.
  - FAST (natural cuBLAS): the GEMM term is the extra rows only, +2.5 ms (in-phase mean 17.0 ms), so GEMM and
    attention are about 6 ms = 4.5 %.
  - So EXACTNESS COSTS ABOUT 7 MS PER TICK AT B 64, all of it the G-b padding. SG5 measures both modes.

## 8. The step scripts (`benchmarks/Efficiency/r2_bi_step.sh`, `r2_g0bi_step.sh`)

- The pattern of `r2_g0_step.sh` (pins, claim, frozen record, sacct row, a main() body).
- Modules of OTHER branches (piece 3 now; the BI GEMM when its interface is forwarded) are pinned by commit, path and
  blob id. Before the claim, the script checks `git rev-parse <commit>:<path>` against the blob id. After the claim,
  it copies the blob from the shared object store into the step record (`git show <blob>`), checks the copy with
  `git hash-object`, and the gate loads the copy by path. The pins and the copy path are in `command.txt`.
- GPU_UUID is NOT hard-coded:
  - it is read from the lead-owned file `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/<holder job id>.gpu_uuid` (one
    line, no "GPU-" prefix);
  - the script STOPs (exit 1, before any claim) when the file is missing or the value is not a uuid;
  - the value and the path are recorded in `command.txt`;
  - the child keeps its `nvidia-smi` check (exit 3).
- `r2_ragged_step.sh` is pinned and is not changed.

## 9. Order of work after this note

1. The CPU stand-ins and tests S1-S15 (portable). DONE in the commits after f1f9f56.
2. GPU microgates B1-B5 and the GEMM / attention cost tables, one holder step (`r2_bi_step.sh`). DONE (eadd87a,
   all MUST PASS). The decision: G-b in the EXACT mode, FAST as the second run mode (3.3).
3. The BI paged attention kernel (Triton) with its CPU interpreter test: piece 3, the verify agent. DONE (v2
   7c826c1, GPU gate PASS; the pin).
4. G0-BI with the QUALITY numbers (7.2): the G0-BI commit (source; the lead runs it).
5. The flattened tick on the GPU: SG1, SG2, SG3, then SG4, SG5 (7.3): piece 4, the verify agent, with the GEMM mode
   as a run parameter; this branch owns the interfaces (`r2_spec_core`, `r2_bi_decode`) and reviews the tick.

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
