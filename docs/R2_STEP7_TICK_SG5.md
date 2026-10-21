# R2 step 7, piece 4: the SG5 harness (R-PERF) on piece 5's Residency (DESIGN)

Branch `feature/nosi-r2-m2-sg5` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-sg5`, from a35037b with C's 2ec08da).
Status: BUILT (source only, not run): the seam (`r2_tick.ResidentStore` / `ResidencyBackend`), `advance_unchecked`
with the void rule, `TickTimer`, the gate `benchmarks/Efficiency/r2_tick_sg5.py` and its step
`r2_tick_sg5_step.sh`; C's answers Q-S1 .. Q-S5 (2026-10-21) applied as below. The design text follows.
Inputs: `docs/R2_STEP7_SPEC_ROUND.md` 1.2 and 7.3 (R-PERF, the budget); C's piece 5
(`/mnt/beegfs/ojerbi/nosi-r2-m2-fetch` 8ee74bc: `nosi/nosi/r2_residency.py`, `docs/R2_STEP7_FETCH_RETENTION.md`
section 8); C's sync-free twin step (4fe5fc9: `step_unchecked` / `advance_unchecked`); C's answers Q1-Q5
(2026-10-21).

## 1. What SG5 measures (REPORT; the 8 % design target)

Per B in {64, 128} and per GEMM mode in {EXACT (G-b `PaddedLinear(4B)`), FAST (natural cuBLAS)}:

| Term | Definition | How |
|---|---|---|
| T_shipped | one shipped decode step at B | the shipped `decode_inference` (no R2 state), same documents, `NOSI_ATTN_SPLITS` 4, CUDA events per step, median of N steps after a warm-up |
| T_d | the DRAFT-ONLY tick: B draft rows, the tick's ops | the engine with one draft row per request per tick (no cohort runs), same store, CUDA events |
| T_tick | one tick of the free-running round | CUDA events from the first launch of `run_rows` to the end of `apply_events`, per tick |
| r | committed tokens / request-ticks | the round's counters |
| ratio | r x T_shipped / T_tick | must exceed 1 for the design to pay |
| dGEMM, dATT, dSEL, dXFER, dHOST | T_tick - T_d split by op (note 7.3) | per-op CUDA event pairs inside the layer (section 3) |
| separate verify | the time the joint pass removes | the BI serial decode step x the verify rows of the round (the cost of a separate verify forward) |
| memory | peak slots, peak retained | `Residency.memory()` / `report()` (4.6) |
| transfers | 4.3 / 4.6 / 4.8 | `Residency.report()` |
| syncs per tick | decision reads; resolve syncs (v1: one per layer); twin syncs (0 with `advance_unchecked`) | counters |

T_shipped and T_d are measured again in the SAME run (the 139 / 167 ms of the pilot may be tainted: note 1.2).
C's Q2 condition stays: the hashed-drop draft is NOT used in SG5; the draft is the Residency's (paper 3.1).

## 2. The tick on the Residency (the storage seam)

The all-resident store is valid for SG1-SG4 only (C, Q1). SG5 replaces it behind ONE seam in `TickEngine`, a storage
backend with four calls. The arithmetic of the tick does not change.

| Call | ResidentStore (built) | ResidencyBackend (SG5) |
|---|---|---|
| `begin()` (start of a tick) | nothing | `svc.poll()` |
| `pages(l, rows, sel)` | device: the selection renamed (historical slot = block id), T - 1 / T to the row's slots; the draft drop | v1: `sel` to the host (one D2H, one sync: C's D2), `plan = svc.resolve(l, rows_svc, sel_cpu)`, then `plan.page_table`, `plan.num_pages`, `plan.last_len`, `plan.seq_idx` to the device (pinned, non-blocking) |
| `after_attention(l, plan)` | nothing | `ev = torch.cuda.Event(); ev.record(); svc.read_done(l, plan, ev)` |
| `commit_block(b, T)` | exact ring page -> historical slot T (device copy) | exact ring page -> the pinned host rows of block T (D2H on the main stream) + an event (section 4, Q-S1) |
| decisions | nothing | `svc.decided(key)` for a published cohort; `svc.cancel(key)` for every key of a rolled-back epoch (Q-S2); `svc.end_request(uid)` when a request is done |

The store layout is C's: `store_k[l]` (B, (C + X + 4) x 64, H, D), the tail rings at slots C + X .. C + X + 3 (2 exact
pages, then 2 draft pages). The tick already writes its rings there: `NB = C + X`, `RING_EXACT = 0`,
`RING_DRAFT = 2` (`r2_tick.py`). The cis table of the service is the exact banks of the 2B twin:
`cis[l] = twins[l].total_cis[:B]` (committed rows only are read).

The rows the service gets (`r2_residency.Row(b, kind, key, pos, hist_bound, tail)`), from the tick's row table:

- `key`: a verify row: `(b, epoch, m)`, its cohort. A draft row producing d_i: `(b, epoch, m(i))`, the cohort that
  will verify its position: `m(i) = 1` if `i <= K - 1`, else `i // K + 1` (cohort_range).
- `hist_bound`: `T` when block T - 1 is committed (written to the host rows), else `T - 1`.
- `tail`: `[(T - 1, ring slot of T - 1)]` when `hist_bound = T - 1`, then `[(T, ring slot of T)]`; the row's own bank
  (exact or draft ring).

The page table the service returns lists the historical slots in ascending block id, then the tail slots: the
canonical order of the BI attention (C's Q1 conditions hold by construction).

The twin moves to `advance_unchecked` (C's 4fe5fc9): the per-bank counts go to the device with the row table (one
pinned H2D per tick); the tick's host check of capacity (positions and compressed rows against N_max and Lk_max on the
host mirrors) runs once per tick; the sticky `overflow` flag is read with the decision read.

## 3. The instrumented engine (the budget terms)

An `InstrumentedTickEngine` subclass records CUDA event pairs per layer around: the five GEMMs, the attention, the
selection (stage 1 + pooling + `select_blocks`), the twin advance, the store writes, and the residency call (`pages`:
its host time and its sync, from `time.perf_counter_ns`). It sums them per tick. The events are read once at the end
of the run (no sync in the tick). With T_d's per-op sums from the draft-only run:

- dGEMM = GEMM(tick) - GEMM(T_d); dATT = attention(tick) - attention(T_d); dSEL likewise for the selection and the
  twin; dXFER = the urgent gathers and the stream waits (the service's plan: `urgent`, `waited`; their event pairs);
  dHOST = T_tick - (the sum of the device terms), split into the resolve host time, the row table and the decision
  read.
- The event pairs themselves cost time: SG5 also runs the round WITHOUT them and reports both T_tick values.

## 4. Questions for C (the Residency interface)

- Q-S1 WRITE-BACK ORDER. A block reaches the host rows when its 64 positions are committed (note 5.2). The tick copies
  the exact ring page to the pinned host rows on the main stream after the decision. A transport that packs host rows
  on the CPU (gilfree, s0) must not read block T before that copy completed. Proposal: `commit_block` records an event
  and hands it to the service (`svc.host_rows_ready(b, T, ev)`); `resolve` does not take a slot for T before the event
  is done (a draft miss of T is deferred; a verify miss host-waits it). Or does C prefer the tick to host-wait the
  copy at the decision (one more sync per committed block)?
- Q-S2 CANCEL OF DRAFT KEYS. After a rejection at cohort m, the draft rows of the old epoch protected blocks under the
  keys (b, epoch, m') for m' = m .. m(last draft index). Is `svc.cancel(key)` correct for a key that only draft rows
  used (no verify look), and must the tick cancel each such key, or does `end_epoch(b, epoch)` belong in the service?
- Q-S3 `decided`. The design note's section 8 shows `svc.decided(key, accepted_upto, rejected)`; 8ee74bc has
  `decided(key)` and `cancel(key)`. The tick will call `decided((b, epoch, m))` after a full acceptance and
  `cancel` for the rejected cohort and the later keys of its epoch. Confirm.
- Q-S4 S-B. When the draft rows' resolution moves to the device (S-B), does `resolve` keep its signature (rows, sel)?
  The seam above isolates the change to `pages`.
- Q-S5 SIZES. C + X and R_ver for SG5 at B 64 / 128 (the union over 4 positions is about 71 blocks; C, Q3): which
  values does C want swept, and which memory budget applies (the all-resident store of SG1-SG4 is not one)?

## 5. Order of work

1. The seam: `ResidentStore` = today's code moved behind it (SG0-SG4 must stay bitwise as they are).
2. `ResidencyBackend` over `r2_residency.Residency` with `TorchGather` and K3X's adversarial transport on the CPU: a
   CPU test that the tick's verify rows on the Residency are torch.equal to the tick's verify rows on the
   ResidentStore (C's F6 property at the tick level), with a negative control (a read of an in-flight slot).
3. `advance_unchecked` in the tick, with the host capacity check and the overflow read; the T9 pin extended.
4. The instrumented engine and the SG5 gate (T_shipped, T_d, T_tick, the budget terms) at B 64 / 128, EXACT and FAST.

## 6. The p4 step cells, and the p4sc64 registration (2026-10-21, before the run)

The step `benchmarks/Efficiency/r2_tick_sg5_step.sh` takes an optional cell. The default is p3's arguments, token for token. Test T19 (`_t19_cells`) pins every cell's printed arguments and the harness's own reading of them (`parse_args`).

| cell | modes | backends | C / X / R_ver | sweep | equal-memory pool (C + X + 4 - 64) |
|---|---|---|---|---|---|
| (default) | exact, fast | sc, sb, v1 | 96 / 32 / 16 | 9 points, 24 steps | 68 |
| p4a | exact, fast | sc, sb, v1 | 96 / 32 / 16 | none | 68 |
| p4b-<b> | exact | b | 96 / 32 / 16 | 9 points, 24 steps | 68 |
| p4sc64 | exact, fast | sc | 96 / 64 / 16 | none | 100 |

Every cell runs 48 steps with gilfree at 8 threads, the superset wait, the torch twin and STEP_MIN 120. No gate or threshold changes.

**Why p4sc64.** In p4a, the S-C gilfree preflight refused at C 96 / X 32 (record `sg5_p4_2190876_p4a.step/stderr.txt`). C's diagnosis (note 12o in docs/R2_STEP7_FETCH_RETENTION.md):
- The policy is right.
- Layer 16's first verify misses needed at least 48 shared extras. X is ONE pool per layer for its 128 streams, and it held 32.
- Every LRU entry was protected by the topk-64 draft selections.
- v1 refuses the same state (C's CPU test F25).

The lead's decision is to measure S-C at X 64. The step's merge base is C's 7c1387f, which adds the K5 decision report (`_k5_report`) to every refusal.

**Registered for p4sc64, with the scoring rules:**

| ID | registered | key paths | rule |
|---|---|---|---|
| A1' | the S-C gilfree preflight PASSES at X 64; or, if it refuses, the refusal's K5 decision report names the per-layer demand | gate `SG5_preflight_gilfree_sc` (MUST); on a refusal, the FULL message in the record's `stderr.txt` (the gate detail is cut at 400 characters, r2_tick_sg5.py:472) | PASS = CONFIRMED (first branch). Refusal + a report = CONFIRMED (second branch), and the report is tabulated. A refusal without the report, or any other exception = REFUTED. The report must name, per layer: the first misses that need a shared extra, the free extras at the tick start, the granted ones and those with no slot. Per refused stream it must name base + extras + Policy E + no slot. |
| R-T | REPORT: the S-C round tick 5.0-5.9 s and the draft-only tick 3.1-3.2 s (the engine agent's 1d1b956 section 3.2, derived at X 32) | `T_tick_ms` and `draft_only.device_ms.tick.mean` of both S-C points | REPORT only. X 64 changes the store size, so these are not falsifiers. I give the distance to each band and the host split per tick: device, twin, res, end_tick = host backend - res, and the rest. |
| R-EQ | REPORT: the equal-memory shipped baseline at NOSI_POOL_BLOCKS 100 | `results.shipped_equal_memory.pool_blocks` (must be 100) and `.stats.median`; gate `R_shipped_equal_memory` | REPORT. ratio_eq for P-SG5-1 uses this run's own median. |
| P-SG5-1..8 | as frozen for p3 (registry line 98), at both S-C points | as in analysis/sg5_p3/SCORING.md | Unchanged. P-SG5-6's bound (32 x 64 x 2 x 128 slots) was written for C + X = 128, and I score it as frozen. At X 64 there are C + X = 160 historical slots per stream: 32 x 64 x 2 x 160 = 655360. I report the peak against both numbers. |

**Expected record shape (integrity, before any score):**
- command.txt holds `cell=p4sc64 modes=exact,fast backends=sc C=96 X=64 R_ver=16 eq_pool=100`, and `sweep=none`.
- stdout holds 10 gate lines:
  - shipped_baseline and R_shipped_equal_memory;
  - `SG5_preflight_gilfree_sc` (MUST) and `R_SG5_preflight_gather_sc`;
  - for `SG5_exact_sc` and `SG5_fast_sc` each: `_completed`, `_bookkeeping` and `_gilfree_team`.
- `results.sweep` is [] and `results.sweep_backend` is null.
- `meta.args.X` is 64.
- `meta.build_gb` should rise against p3's 17.73 GB, because the store holds C + X + 4 = 164 slots per stream instead of 132. The expected value is about 22 GB (REPORT).

## 7. [b2] The p5b2 cell: B2 as the service, and its registration (agent B, 2026-10-21, before the run)

Branch `feature/nosi-r2-m2-sg5-b2` (worktree `/mnt/beegfs/ojerbi/nosi-r2-m2-sg5-b2`): the verify agent's p4d head 3ae7d3b, B's own2b 3fbf8a7 (B2, and through it C's sparse head 771045d), the verify agent's 825665b (S-C's counters).

**What the cell runs.** `p5b2` = p4sc64's arguments with `--backends b2`: both GEMM modes, C 96 / X 64 / R_ver 16, no sweep, 48 steps, gilfree at 8 threads, the superset wait, NOSI_POOL_BLOCKS 100. T19 pins the arguments.

**The b2 backend (`make_engine`, marked `[b2]`).**
- The service is `OwnResidency` (`nosi/nosi/r2_own2.py`). Its ownership table lives on the store's device. It runs `pin_records=False`.
- It keeps S-C's device path: `device_verify` / `device_draft` on the device mirror, the superset wait and the urgent gather of `CudaDeviceOps` behind `_LazyOps` (as sc), and C's sparse record with rowsel.
- The timed path is UNCHECKED. `--b2-checked` turns on every B2 predicate. The cell does not pass it.
- The gates are sc's: completed, bookkeeping, gilfree team. The record adds `info.b2_checked` and `report()["b2"]`. `report()["k5"]` now holds the demand peak of the passing ticks too: B2 computes it in the apply and reads it with its predicates (no extra device read).
- Three fixes on the GPU path (`r2_own2.py`): `memory()` counts the table (one read; the vendored version built 2.6 M slot objects at every poll and end_tick), `stream()` builds one stream at a time (for `Metrics.report`), and the per-layer looks ride in the one read of the counts. The 4.8 lifetimes get their read counts from the table (at a take, and at the report).

**The model (the lead's, from p4sc64's record `sg5_p4_2190876_p4sc64`, EXACT; ms per round tick).**

| term | S-C (p4sc64) | b2 (expected) | why |
|---|---|---|---|
| poll + end_tick (host backend 5645 - res 936) | 4709 | 80-250 | B2's apply: device ops 11-16 ms at C + X 128 (b2tg2), about 13-20 at 160; the host side per take (Transfer objects, metrics, the arrivals loop: 7.7 k takes per tick, 468010 / 61) 60-200; about 25 host reads |
| res (the service's resolve, host-bound) | 936 | 100-1000 | + the sparse record per layer (10-30); - S-C's host mirror delta, which B2 replaces with a device copy (0 if the lead's model holds; 550-850 by the engine agent's registration, unmeasured) |
| rest (residual - poll - end_tick: finish_tick, decisions, write-backs) | 778 | 778 | the same engine and round |
| twin | 122 | 122 | the same |
| gemm + att + sel + store | 61 | 61 | the same |
| **T_tick** | **6606** | **2100 point** | 1897 (the lead's split) + 150 (B2) + 20 (record); 1400 if the mirror delta is 700 |

**Registered for p5b2 (the scoring rules follow the table):**

| ID | registered | key paths | rule |
|---|---|---|---|
| B-A | the b2 gilfree preflight PASSES | gate `SG5_preflight_gilfree_b2` (MUST) | PASS = CONFIRMED. A refusal or an exception = REFUTED; the full message in `stderr.txt` is tabulated (named risks: a CUDA-only fault in B2's glue, which never ran on a GPU; a key-slot refusal, I13, KEYS 7; an out-of-memory). |
| B-T | the b2 EXACT round tick 1.2-2.6 s (point 2.1 s) | `T_tick_ms` of `SG5_exact_b2` | In the band = CONFIRMED. 2.6-3.3 s or 0.9-1.2 s = NEAR (I give the term that moved). Above 3.3 s or below 0.9 s = FALSIFIED: above, B2 keeps more than a third of S-C's 4.7 s; below, the remaining terms are not what the model says. |
| B-F | the b2 FAST round tick within 10 % of b2 EXACT's | `T_tick_ms` of `SG5_fast_b2` | REPORT (S-C: 6452 vs 6606). |
| B-S | the residual 0.85-1.2 s (S-C 5.49 s) | `round.residual_ms.mean` | In the band = CONFIRMED: the end_tick term is gone and B2's own term is 0.1-0.4 s. Above 1.5 s = FALSIFIED. |
| B-R | twin, gemm, att, sel, store within 15 % of S-C's; res 0.1-1.0 s | `round.device_ms.*.mean` | REPORT. res below 0.6 s means S-C's host mirror delta was a large part of S-C's res (the engine agent's claim). |
| B-D | the b2 draft-only tick 0.5-1.5 s (S-C 4.72 s) | `draft_only.device_ms.tick.mean` | REPORT (S-C's draft-only end_tick is 3.76 s; its rest is about 0). |
| B-Q | ratio r x T_shipped / T_tick about 0.02 (S-C 0.006) | `ratio` | REPORT. B2 alone does not reach the design target: res + rest stay above 0.9 s of host work. |
| B-K | REPORT: `report()["k5"]` (the demand peak at X 64, its tick and layer), `report()["b2"]`, `draft_only.sc` / `residency.sc` (`mirror_slots_uploaded` is 0 for b2: a device copy), `meta.peak_gb` (p4sc64 67.41 GB; B2 adds its table, about 0.25 GB, and its temporaries) | as named | REPORT. |

**Expected record shape (integrity, before any score):** command.txt holds `cell=p5b2 modes=exact,fast backends=b2 C=96 X=64 R_ver=16 eq_pool=100` and `sweep=none`; stdout holds 10 gate lines (shipped_baseline, R_shipped_equal_memory, `SG5_preflight_gilfree_b2`, `R_SG5_preflight_gather_b2`, and for `SG5_exact_b2` and `SG5_fast_b2` each `_completed`, `_bookkeeping`, `_gilfree_team`); every point has `b2_checked` false.

**CPU tests (source only, not run).** `tests_cpu/test_r2_tick_b2.py`: TB1 b2 (checked and unchecked) = sc at every tick of the teacher-forced round, at two capacities (rows, tokens, logits, page tables, B's snapshot of the service state, counters, readers, memory; the committed tokens, the journal and the report at the end); TB2 the free-running round; TB3 a planted record defect (drop_rowsel) differs at its tick, and no_read_harvest changes only k48; TB4 the k5 demand on C's F26 / F25 schedules; TB5 the harness's b2 parts. `test_r2_tick.py`: T15 runs b2 too; T19 has the cell p5b2.
