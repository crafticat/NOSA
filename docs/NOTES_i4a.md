# I4A: lever 1, the draft phase (arm L4PTFD = L4PTF with the lean one-row pass)

Written 2026-10-29T03:56:56Z (`date -u`). Style: ASD-STE100. Labels: MEASURED (in a record), DERIVED (hand arithmetic
on MEASURED values, operands shown), CODE (read in the source at file:line), INFERENCE (my estimate; not measured).

Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_L4P_faithful.md`; lever 1 of
`/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_model_vs_measured.md` revision 5 (R5.3, R5.5 item 1);
scoring of I3: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/SCORING_i3.md` (I3S).
Tree: `/mnt/beegfs/ojerbi/nosi-r2-i4a`, branch `r2-i4a` = I3 `c4c64b2` + this rung.
Abbreviations: ST = `nosi/nosi/spec_tentative.py`, STF = `nosi/nosi/spec_tentative_fork.py`, RE =
`nosi/nosi/ragged_engine.py`, FB = `nosi/nosi/r2_body_fused.py`, LN = `benchmarks/Efficiency/r2_i4a_lean.py`.

Files:

- `benchmarks/Efficiency/r2_i4a_lean.py`: `lean_classes(STF)` -> `TentStateFD`, `TentRunnerFD` (arm L4PTFD).
- `benchmarks/Efficiency/r2_i4a_cell.py`, `benchmarks/Efficiency/r2_i4a_step.sh`: the GPU cell and its step script.
- `benchmarks/Efficiency/tests_cpu/i4a_world.py`, `test_i4a.py`: the CPU world and the tests.
- Gate-0 declarations only (one module-level block each): `tests_cpu/test_wsa.py`, `test_wsb.py`, `test_wsc.py`,
  `test_i2.py`, `test_i3.py`.

NO NEW KERNEL. LN imports only torch. Every launch of L4PTFD is a launch kind that L4PTF already makes. No file of
`nosi/` and no I3 file changes: L4PTF is c4c64b2's code, bit for bit (Gate 0, `test_i4a_gate0_files`).

## 0. Answer in five lines

1. MEASURED (I3S 3.1, LED R5.3): an I3 draft step costs 75.682 ms (C 128 B 216) and 97.275 ms (C 73 B 318); a standalone
   draft step costs about 61.353 / 81.577 ms. Excess: 14.329 / 15.698 ms per step.
2. CODE (section 1): an I3 draft step runs I1's tentative row (ST:395-448) in every layer: 84 aten ops per layer in
   `_rows` plus 2 attention calls and 5 residency kernels. The fused shipped body runs 5 kernels and 1 attention call
   (FB:193-205, RE:417-423). The excess is about 80 small ops per layer, about 2,700 per draft step.
3. INFERENCE: these ops are launch-bound (about 4.6-7.0 us per op from the I1P row labels, section 4). 14.3 ms / 2,700 =
   5.3 us per op: the op count explains the excess.
4. BUILT: L4PTFD removes 37 aten ops per layer and 11 in layer 0 as the independent counter sees them: **1,158 per
   draft step** at L 32 (SAVED_PER_LAYER 37, SAVED_LAYER0 11; the CODE count is 39 / 12 = 1,221, section 1.2). Every
   value stays L4PTF's, bit for bit (tokens, logits, buffers, engine, counters, trace).
5. INFERENCE (section 5): -6.4 ms per draft step (band 3.2-8.1), -12.7 ms per round; L4PTFD / L4PTF = 1.036 (C 128) and
   1.022 (C 73); L4PTFD / SH = 0.531 and 0.606.

## 1. The op inventory of the I3 draft step (measure before build; CODE)

The draft step: `TentRunner._ps_round`'s D slots (ST:1015-1036) call `T.run_pass` with a one-row plan
(`_plan_mixed`, ST:877-884); the prelude's K drafts call `_fast_draft` (ST:891-895). `TentStateF` / `TentStateU` give a
one-row pass to `TentState._rows` (STF:81, `spec_tentative_union.py:104-107`), so **L4PTF's draft step is I1's**
(`TentState._row`, spec_tentative.py:395-448). Per
layer `TentState._layer` (ST:318-369) runs the dense part, the twin, the selection and then `_rows` -> `_row`.

The comparison is the plain ragged step, the code counterpart of the shipped step (RE:393-431 with the fused body
FB:152-206): per layer `pos_before.copy_` (RE:409), the twin step (RE:410), `_lengths` (RE:411), stage 1 + the pooling
graph (RE:412-415), the fused body (diff, ONE body kernel, two gathers, the write-back kernel; with the pool also
pool_update and pool_swap: FB:193-205, RE:379-390) and ONE attention call (RE:422-423).

Counts are aten COMPUTE ops (view ops and empty allocations excluded), per layer after the first, in `_rows`; the
attention calls and the residency kernels are listed apart (the CPU world replaces them by stand-ins).

| # | item (I3 draft row) | where (CODE) | aten ops per layer | tensors | class | L4PTFD |
|---|---|---|---|---|---|---|
| 1 | twin position clone | ST:341 | 1 | twin.pos (B,) | REQUIRED (the ragged step copies it too, RE:409) | kept |
| 2 | qb, the row's block, past | ST:400-403 | 3 | qc, p (B,) | REMOVABLE: the same in every layer | once per pass |
| 3 | hasq, the tail-id injection, ti | ST:404-408 | 6 | sel (H, B, 64) | REQUIRED (D1 residency input; past-edge heads) | kept (6) |
| 4 | the residency's torch ops | ST:458-479 | 17 (+ 5 kernels) | maps, load mask (H, B, 64) | REQUIRED (D1; Task 1.1's draft skips; the skip counter); 2 ops MERGEABLE | 15 |
| 5 | the buffer write: index + where + index_put per tensor, p % 64, prow | ST:491-497 | 12 | buffer K, V (B, 64, H, D), bias (B, 64, H) | REQUIRED (D3: the replay reads it, ST:378-393; the chain images read it, STF:105-121); 8 ops MERGEABLE | 4 (one where per tensor + the cis gather) |
| 6 | layer 0: tpos (3) and the counters | ST:499-509 | +3 (layer 0) | tpos (B, 64) | REQUIRED (D3, span, draft rows) | tpos 1 op |
| 7 | n_tail and the call-A cut | ST:415-417 | 9 | (B,) | REMOVABLE: the same in every layer | once per pass |
| 8 | m63 and the slot mask | ST:418-419 | 4 | (H, B, 64) | REQUIRED (per head); 1 MERGEABLE (~hasq once) | 3 |
| 9 | THE FULL MASK COPY: expanded mask copy, scratch copy_, masked_fill_ | ST:420-426 | 3 (two full passes over B x 4096 x H) | scratch bias (B, 4096, H) | REMOVABLE as three ops; ONE masked select is REQUIRED (skipped slots hold stale rows; m63 is per head) | 1 where into the scratch |
| 10 | call A (the committed window) | ST:429-430 | 1 flash call | window | REQUIRED (= SH's attention) | kept |
| 11 | call B's masks (lo, hi, own, F4, cmt, vis) | ST:431-439 | 16 | tpos (B, 64) -> (B, 64, H) | REMOVABLE except vis (it reads hasq) | 3 |
| 12 | call B (the buffer) | ST:440-441 | 1 flash call | buffer (B, 64, H, D) | REQUIRED (section 3) | kept |
| 13 | the float32 log-sum-exp merge | ST:442-448 | 13 | (B, 32, 128) f32 | REQUIRED with call B; a 6-op form exists but is NOT bitwise | kept (bit for bit) |
| 14 | torch.stack of the single row | ST:375-376 | 1 | (B, 4096) | REMOVABLE | a view |

Sum per layer (`_rows`, layers after the first): I3 3 + 6 + 17 + 12 + 9 + 4 + 3 + 16 + 13 + 1 = **84** aten ops (+ the
twin clone, 2 attention calls, 5 residency kernels). L4PTFD 0 + 6 + 15 + 4 + 0 + 3 + 1 + 3 + 13 + 0 = **45**.
Removed (CODE count): 39 per layer. Layer 0: I3 84 + 3 (tpos) = 87; L4PTFD 45 + 16 (the hoisted tensors, part 1:
qb, bp, past, p % 64, the one-hot mask (2), prow, n_tail (4), cq, minimum, where, + base, int32) + 13 (part 2, after the
write: lo, hi, tb, own, % 64, <, ~own, |, &, &, cmt (2), ~cmt) + 1 (tpos) = 75; removed 12. Per draft step at L 32:
39 x 31 + 12 = 1,221 by CODE (I3: about 84 x 32 + 3 = 2,691 in `_rows`).

### 1.2 The measured count (CPU, an independent counter)

`test_l4ptfd_op_counts` counts aten compute ops with a TorchDispatchMode (`i4a_world.OpCount`; views and empty
allocations excluded) in every one-row pass and layer of the same run of both arms (CPU world, L 2, B 4, 12 one-row
passes). First run (log `/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/i4a_wip0_i4a.txt`, MEASURED): L4PTF - L4PTFD = 11 in
layer 0 and 37 in layer 1, 48 per pass, in every pass; the joint passes' counts equal (118,107 / 110,835 in both arms).
By name per pass (DERIVED, totals / 12 passes): index 7, index_put_ 7, remainder 4, __and__ 4, bitwise_not 4, add 3,
floor_divide 3, ge 3, stack 2, copy_ 2, masked_fill_ 2, clamp, eq, gt, le, lt, minimum, __or__, sub 1 each, where -1.
Every name matches the CODE count except two per layer that the counter does not separate: I1's reshape copy of the
expanded mask (`clone`, ST:420) and its int32 cast of the cut (`_to_copy`, ST:417); in layer 0 the hoist's own int32
cast offsets one of them. The counter shows no difference for `clone` / `_to_copy`; the cause is not resolved (it can
be the counter, not the code). **The stated numbers are the counter's: SAVED_PER_LAYER = 37, SAVED_LAYER0 = 11, 37 x 31
+ 11 = 1,158 per draft step at L 32** (CODE_EXTRA_PER_LAYER 2, CODE_EXTRA_LAYER0 1 in LN). The bands use 1,158.

What a draft step adds over the shipped step (I3, per layer): items 2-9 and 11-14 minus the fused body's 1 kernel: about
80 small ops and ONE extra attention call (call B). INFERENCE: at about 5 us per small op (section 4) this is 80 x 32 x
5.3 us = 13.6 ms per step, close to the MEASURED excess 14.329 ms (C 128) and 15.698 ms (C 73).

### 1.1 The candidates of the task, decided

- **Draft rows straight into the tail slot (one attention call).** CODE: the rows of the tail slot at or above qc % 64 are
  not committed state; every reader cuts at qc % 64 (call A's cut ST:417, CB0 STF:112-116) and the commit overwrites
  rows [q0, q1) from the buffer (ST:272-274). So an in-block row could see its chain rows in the window and skip call B.
  A row PAST the edge must see its own block qb + 1, which has no window slot for a head without qb (the injection puts
  qb in place of qb + 1, ST:407-408). Which rows are past the edge is known on the device only (the accept count m), so a
  per-request choice needs a host sync or a page table (the fork). REQUIRED as built: call B + merge for every row.
- **One attention call over window + buffer via the fork.** The fork reads paged blocks, so one call could serve both.
  INFERENCE: the fork moves 0.90-0.95 TB/s against flash's 1.54 (FIA6, LED R4.3), and at R = 1 a fork tile of 64 query
  rows holds 16 (one row x 16 query heads per KV head). Call A on the window is about 19.3 ms per step (SH's attention
  term, LED R4.1); the fork at R = 1 is INFERRED slower. Not built; it needs FIA6's isolated harness at R 1 first.
- **Avoid the full mask copy.** Done: ONE torch.where writes the scratch (item 9). One full pass over the bias stays:
  a skipped slot holds the bytes of an evicted block and m63 is per head, so call A needs a masked bias.
- **CUDA-graph capture of the draft step.** Not built. (a) The shipped step in these cells is eager (SH's whole-step
  graph saves only 1.75 %, LED R4.1); (b) a whole draft-step capture freezes the pool's LRU stamp, a host int passed by
  value (ST:462-466; graph_step.py's shifted-age frame in nosi-r2-graphstep is the existing remedy); (c) the CPU world
  cannot test a capture. A per-layer graph of items 5-13 (after the residency) is the next step if section 5 holds
  (INFERENCE: about 25 launches per layer to 1 graph launch + about 6 static-input copies).

## 2. The build (CODE: LN)

`lean_classes(STF)` builds the classes on the GIVEN spec_tentative_fork module (as `r2_body_fused.fused_body_class`): the
GPU cell passes `nosi.spec_tentative_fork`, the CPU tests the CPU world's copy. LN imports only torch; no nosi module
names it (the nosi name scans of the older Gate-0 tests stay true).

- `TentStateFD(TentStateF)`: `run_pass` renews the hoisted tensors at a one-row pass's start; `_rows` sends a one-row
  pass to `_row_lean` and a joint pass to L4PTF's code; `_lean_pass` (part 1), `_lean_mask` (part 2, after layer 0's
  buffer write: tpos moves only there), `_row_lean`, `_resid_lean`, `_write_lean`.
- `TentRunnerFD(TentRunnerF)`: `_make_state` gives `TentStateFD`; `read()` adds `lean` (host counts: passes, layers).
  `TentState.calls` stays L4PTF's.
- Precondition of the hoist (CODE): in a one-row pass every layer's twin starts at the same counter (all twins step with
  the same mask, ST:344; `set_twins` writes all layers, ST:217-225; the replay exists only in joint passes, ST:339-340),
  so p is the same in every layer; qc moves only at a commit (ST:285).
- Controls (tests only, `dneg`): `mask_before_write`, `live_bias`, `stale_pass`.

## 3. Contract 4, D1, D3, the tentative state (kept exactly)

The lean pass writes the same bytes to the same tensors in the same order of dependence: the residency (D1, the draft
skips), the buffer row and tpos (D3), the counters. `test_l4ptfd_equals_l4ptf` compares a whole L4PTFD run with an
L4PTF run bit for bit (`i2_world.same_run`: tokens, fingerprints, logits rows, read() incl. the trace and the token
accounting, round records, engine and twin state, buffers, tpos, qc, the layer-0 row writes). `test_l4ptfd_trace_d1_d3`
runs WS-A's trace, row-write, replay and position checks and the cell's -D1 / -D3 / -SPAN / -COMMIT-N / -LEAN gates.

FINDING (MEASURED, first CPU run): a planted error in the DRAFT step (`mask_before_write`: the row does not see its own
entry; `live_bias`: stale slots visible) changed the run (drafts, acceptance, round records, buffers) but left every
committed token SH's and the logits within TOL (max |logit - SH's| 8.006e-06, the clean value). Reason (CODE): the
joint pass verifies every draft and recomputes the verify rows; a wrong draft costs acceptance, not correctness. So
SH's tokens are blind to draft-step errors. The gate that catches them is **L4PTFD-EQF**: tokens, check-round
fingerprints AND the acceptance record (`tokens_acct`, `round_accepted`, `round_committed`) equal to L4PTF's in the same
job. It is MUST in i4am; in the timing cells it is REPORT (run-to-run determinism of two arms is not established).

## 4. The per-op cost (INFERENCE from MEASURED labels)

I1P (record `/mnt/beegfs/ojerbi/r2-runs/spec/i1p_p2a_2195574_i1p`, C 128 B 216, LED R5.3): row-0 labels per 32 layers:
write 2.827, amask 2.839, bmask 2.344, call B 1.610, merge 2.111 ms. Ops per label (section 1): write 12 x 32 + 3 + about
16 counters = about 403 -> 7.0 us per op; amask (items 7-9) 16 x 32 = 512 -> 5.5 us; bmask (item 11) 16 x 32 = 512 ->
4.6 us; merge 13 x 32 = 416 -> 5.1 us. Range 4.6-7.0 us, mean about 5.5 us. These labels come from 4-row joint passes of
L4PT; a 1-row draft pass has the same ops per row (CODE), but no record marks a 1-row pass (LED R5.6 item 2).

## 5. Inferred bands (INFERENCE; for the lead's registration, not a registration)

The registration lives in `r2_i4a_cell.py`: `PRED_I4A_BY_B` (every value None) and `PRED_STAMP_I4A` (None). The step
script stops i4aa / i4ab while `PRED_STAMP_I4A` is None.

Saving per draft step: 1,158 ops x 5.5 us = 6.4 ms (central); x 4.6-7.0 us = 5.3-8.1 ms; lower edge x 0.6 for overlap of
host enqueue with device work = 3.2 ms. Band **3.2-8.1 ms per step, 6.4-16.2 ms per round** (2 draft steps), central
12.7 ms per round, the same at both C (the removed ops are small: their size is B x 64 or (H, B, 64), launch-bound at
B 216 and B 318). If the 2 + 1 ops of section 1.2 are real, add 63 ops (0.3 ms per step).

Tokens per round: L4PTFD computes L4PTF's values bit for bit, so its schedule, tokens and yield are L4PTF's (same job);
the rate ratio is the inverse round-time ratio.

C 128 B 216 (MEASURED I3 i3a: draft 151.363 ms, round 365.423 ms, SH step 64.787 ms, L4PTF / SH 0.5128, joint 207.377):

| key | arithmetic | central | band |
|---|---|---|---|
| I4A_draft_saving (ms per round) | 2 x 6.4; 2 x (3.2 .. 8.1) | 12.7 | (5, 18) |
| I4A_draft_step_sh | (151.363 - 12.7) / 2 / 64.787 = 69.33 / 64.787; (151.363 - 16.2) / 2 = 67.58 -> 1.043; (151.363 - 6.4) / 2 = 72.48 -> 1.119 | 1.070 | (1.02, 1.13) |
| I4A_vs_I3 | 365.423 / (365.423 - 12.7) = 365.423 / 352.72; 365.423 / 359.02 = 1.018; 365.423 / 349.22 = 1.046 | 1.036 | (1.01, 1.06) |
| I4A_ratio | 0.5128 x 1.036; 0.5128 x 1.018 = 0.522; 0.5128 x 1.046 = 0.537 | 0.531 | (0.51, 0.55) |
| I3_ratio | MEASURED 0.5128; job-to-job +-1 % (I3S 1.4: 0.1-0.4 %) | 0.513 | (0.50, 0.525) |
| I4A_joint_ratio | the joint pass is L4PTF's code | 1.000 | (0.98, 1.02) |
| I4A_peak_gb | section 6 | 83.89 | (83.5, 84.25) |
| I4A_agree (floor) | L4PTF's 0.403 (I3S 4.3) | - | >= 0.25 |
| I4A_eq_I3, I4A_above_I3 | bit for bit; saving > 0 | True | True |

C 73 B 318 (MEASURED I3 i3b on nova05: draft 194.550 ms, round 585.463 ms, SH step 123.489 ms, L4PTF / SH 0.5925):

| key | arithmetic | central | band |
|---|---|---|---|
| I4A_draft_saving | as C 128 | 12.7 | (5, 18) |
| I4A_draft_step_sh | (194.550 - 12.7) / 2 / 123.489 = 90.93 / 123.489; (194.550 - 16.2) / 2 = 89.18 -> 0.722; (194.550 - 6.4) / 2 = 94.08 -> 0.762 | 0.736 | (0.70, 0.78) |
| I4A_vs_I3 | 585.463 / 572.76; 585.463 / 579.06 = 1.011; 585.463 / 569.26 = 1.028 | 1.022 | (1.005, 1.04) |
| I4A_ratio | 0.5925 x 1.022; 0.5925 x 1.011 = 0.599; 0.5925 x 1.028 = 0.609 | 0.606 | (0.59, 0.62) |
| I3_ratio | MEASURED 0.5925 (nova05); I2X on nova07 0.5165 vs 0.5167 for L4PTU | 0.593 | (0.58, 0.61) |
| I4A_joint_ratio | as C 128 | 1.000 | (0.98, 1.02) |
| I4A_peak_gb | section 6 | 77.69 | (77.3, 78.1) |
| I4A_agree (floor) | L4PTF's 0.388 | - | >= 0.25 |
| I4A_eq_I3, I4A_above_I3 | as C 128 | True | True |

Against the lever's bound (LED R5.4 L1: 28.657 / 31.396 ms per round): the central saving is 12.7 ms = 44 % / 40 % of
the bound. Against the plan (`own SH` ceiling of LED R5.4: 0.620 / 0.708): this rung moves 0.513 -> 0.531 and 0.593 ->
0.606 (central).

`I4A_peak_gb` reads the MEM gate's value (the prefill included), not the decode peak: SCORING_i3.md 6.1 item 1.

## 6. Memory against the usable ceiling 84.25 GB (lesson L59; GB = 1e9 bytes)

MEASURED (I3S 3.5): L4PTF arm peak with the prefill 83.891 GB (C 128 B 216; prefill 78.13 GiB x 1.073741824) and 77.685
GB (C 73 B 318); decode peaks 82.559 / 76.932 GB. The usable allocation ceiling is 84.25 GB (SCORING_wsc.md:52-54,
DERIVED from two OOM messages; lesson L59).

CODE: L4PTFD allocates the same arena, store (X 129), CB0, buffers and scratch as L4PTF (the same constructors,
STF:85-102, ST:155-203). The lean pass runs after the prefill only; it adds the hoisted tensors of one pass (about B x 64
bools and a few (B,) tensors: under 0.1 MB) and drops two per-layer transients of I3 (the expanded mask B x 4096 x H
bools = 1.77 MB at B 216, 2.61 MB at B 318; the stacked row B x 4096 x 2 B = 1.77 / 2.61 MB). MODEL: the L4PTFD arm
peak = L4PTF's: **83.891 GB at C 128** (0.109 GB under 84 GB, **0.359 GB under the usable 84.25 GB**) and **77.685 GB at
C 73** (6.315 / 6.565 GB). The decode peak changes by under 0.01 GB. RISK: the C 128 margin is I3's; the second fork arm
prefills after the first fork arm's memory is freed (`BC.free`), as i3a's L4PTF prefilled after L4PT and L4PTU.

## 7. Cells (r2_i4a_step.sh; SH runs first; arms L4PTF then L4PTFD)

| cell | C | B | steps | options | notes |
|---|---|---|---|---|---|
| i4am | 73 | 8 | 72 | --check-every 1 --require-stress 1 --eq-must 1 | the MINIATURE: every round a check round; L4PTFD-EQF MUST. STEP_MIN 40, 100G |
| i4aa | 128 | 216 | 512 | timing | STOPS while PRED_STAMP_I4A is None. STEP_MIN 80, 240G |
| i4ab | 73 | 318 | 512 | timing | STOPS while PRED_STAMP_I4A is None. STEP_MIN 110, 340G |

STEP_MIN basis (MEASURED, I3S 1.2): i3a 48:12 and i3b 70:47 for SH + 3 arms; i3m 7:51 for SH + 5 arms at B 8.
MUST gates: C0_FORK, SH-RUN, per arm -RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN, -RESID, -ATTN1, -FILL1, L4PTFD-LEAN; i4am
also -STRESS, -FSTRESS, -PRESENT, -VIS, -VB, -TAIL, -CHAIN, -IDX, -RA and L4PTFD-EQF. REPORT: MEM, L4PTFD-EQF (timing).

## 8. Risks

1. The per-op cost model (section 4) is INFERRED from 4-row labels; the 1-row draft pass is not marked in any record. If
   the draft step is bound by its large kernels, the saving is near the lower edge.
2. L4PTFD-EQF needs run-to-run determinism of two arms with separate prefills in one job. The I3 records show identical
   token accounts across jobs and GPUs (I3S 1.4), not identical tokens: EQF is MUST only in the miniature. It is the
   only gate that sees a draft-step error (section 3): SH's tokens and the structural gates do not.
6. The op counter shows 2 ops per layer less than the CODE count (section 1.2); the cause is not resolved.
3. GPU semantics of `torch.where(..., out=<a strided view>)` (the scratch, LN `_row_lean`) and of `out=` aliasing an
   input (the buffer, LN `_write_lean`; the same form as `TentState.commit`, ST:272): the CPU tests cover the values;
   i4am's EQF covers the GPU.
4. Memory at C 128: 0.359 GB under the usable ceiling (section 6), unchanged from I3.
5. CPU stand-ins model the arithmetic, not GPU dispatch, ordering, timing or memory.

## 9. CPU tests

`benchmarks/Efficiency/tests_cpu/test_i4a.py` (its own srun): Gate 0 (files, declarations, nosi and I3 bytes, the fork
blob, the module's imports); L4PTFD = L4PTF bit for bit (B 1, B 8, B 8 natural) with SH's tokens and TOL, and the
preconditions (one-row rows in the committed block and past the edge, draft skips); the contract-4 trace, D1, D3,
exactly N and the cell's gates; the op counts per pass and layer (SAVED_PER_LAYER, SAVED_LAYER0, 1,158 at L 32); the
plants mask_before_write, live_bias, stale_pass (each changes the run and is CAUGHT by L4PTFD-EQF); no host sync; the cell's
arguments, pure parts, registration state, step script (DRYRUN) and these notes. The older suites (test_i3.py,
test_i2.py, test_wsa.py, test_wsb.py, test_wsc.py) must stay green with the declaration blocks.
