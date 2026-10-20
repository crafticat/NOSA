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
