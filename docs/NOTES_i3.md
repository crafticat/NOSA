# I3: the tentative buffers + the union residency + ONE fork call per layer (arm L4PTF)

Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_L4P_faithful.md` (rung I3, workstream WS-L).
Tree: `/mnt/beegfs/ojerbi/nosi-r2-i3`, branch `r2-i3` = I2 `63da01d` (nosi-r2-i2) + merge of `r2-wsc2` `db21536`
(WS-C `22c27ee` + the miniature's schedule) + the I3 commit.

Files:

- `nosi/nosi/spec_tentative_fork.py`: `TentStateF`, `TentRunnerF` (arm L4PTF), `teacher_facts`.
- `benchmarks/Efficiency/r2_i3_arena.py`: `ChainArena` (WS-C's arena + the CB0 region), `NoCopyPrefillHook`.
- `benchmarks/Efficiency/r2_i3_cell.py`, `benchmarks/Efficiency/r2_i3_step.sh`: the GPU cell and its step script.
- `benchmarks/Efficiency/tests_cpu/i3_world.py`, `test_i3.py`, `test_i3_interp.py`: the CPU world and the tests.
- Gate-0 declarations only: `tests_cpu/test_wsa.py`, `test_wsb.py`, `test_wsc.py`, `test_i2.py`.

NO NEW KERNEL. The fork is fia6's file, byte for byte (blob `f0dc0c18`). I3 adds torch ops and Python only.

## 1. What changes against I2 (L4PTU)

I2 runs, per layer of a joint pass, ONE residency for the union of the rows (row 0's selection), then three kinds of
attention calls and a log-sum-exp merge:

- call A: the committed window (`spec_tentative_union.py:197-205`);
- C store calls (`spec_tentative_union.py:218-233`; X 193 gives C = 1);
- call B: the tentative buffer (`spec_tentative_union.py:236-252`);
- the merge (`spec_tentative_union.py:225`, `:252`, `:254`).

I3 replaces these calls and the merge with ONE call of the fork per layer for every row of every request. Each row
sees only its own selection. Everything else of I2 and I1 stays: the residency of row 0 (D1 kept), the buffer writes
(D3 counters), the replay of the deferred row, the commit, the reset, the schedule of contract 4.

## 2. The seam (file:line)

1. `TentStateU._rows` (`spec_tentative_union.py:104-120`) calls `self._rows_union` for a joint pass (R > 1).
   `TentStateF` (`spec_tentative_fork.py:81`) overrides only `_rows_union` (`spec_tentative_fork.py:124`).
   The I2 host counts (resid, joint_resid, union_layers) and the one-row passes stay I2's and I1's code.
2. `TentRunnerF._joint_ops` (`spec_tentative_fork.py:408`) gives `VerifyOps(mode="union", joint_attn="fork",
   fork=..., full_max=K)`. `verify_rows._fork_state` (`verify_rows.py:1010`) then makes the store (the arena's store
   views, C = 1) and the fork's tables. `TentRunnerF._make_state` (`spec_tentative_fork.py:414`) gives `TentStateF`.
3. Inside `_rows_union` the WS-C helpers run unchanged: `_fork_locate` (`verify_rows.py:1105`), `_fork_lists`
   (`verify_rows.py:1180`), `_fork_call` (`verify_rows.py:1095`), `_fork_probe` (`verify_rows.py:1270`). The WS-C pass
   itself (`verify_rows.py:1308`) is NOT used: it writes the tentative rows into the committed tail slot and store slot
   0 and writes them back (departure D1, which WS-A removed).

### 2.1 Where the tentative rows go: into the arena (decision)

Two options:

- (a) A separate buffer call, merged with the fork's output by the log-sum-exp.
- (b) The tentative rows in the arena, read by the same fork call.

Decision: (b). Reason: the fork's default 2D launch writes the NORMALISED output only. In the 2D path the segment
pointers are the output itself (`fork :1060-1063`). No log-sum-exp leaves the kernel. So (a) needs the 3D split path or
a kernel edit. A kernel edit is a NEW KERNEL (not allowed).

The fork shows a row the tokens 0 .. n - 1 of each block that the row lists (n = 64, or `last_len` on the row's LAST
listed page: `fork :296`, `:370`). Block qb = qc // 64 is split in two places: its committed rows are in the window's
TAIL SLOT, its tentative rows are in the BUFFER (contract 4, D1: a tentative row never enters the tail slot). No prefix of
one place is the row's view. So per layer and joint pass I3 writes TWO SCRATCH BLOCKS per request (`TentStateF._chain`,
`spec_tentative_fork.py:105`):

- CB0 (the ChainArena's new region, one block per request): rows x < qc % 64 from the tail slot (committed), rows
  x >= qc % 64 from buffer row x (the chain's rows of block qb).
- CB1 (store slot 0, the slot that WS-C uses for "the next block"): the buffer (the chain's rows of block qb + 1).

Every row lists CB0 in place of block qb and CB1 in place of block qb + 1. Its other blocks come from the window, a pool
slot (read in place) or the store's listed host blocks, as in WS-C. The views are exactly I1's:

- a row at p in block qb sees CB0's rows 0 .. n_tail(p) - 1 = TentState._row's call A cut (base + min(qc % 64,
  n_tail)) plus call B's rows qc <= x <= p, x % 64 < n_tail (`spec_tentative.py:415-441`);
- a row past the edge sees CB0 whole if its head selects qb (call A up to qc % 64 and call B's block-qb rows), and
  CB1's rows 0 .. n_tail(p) - 1 (call B's own-block rows);
- F4 stays: n_tail = 0 at a block's last position (the row then sees nothing of its own block).

Nothing is written back. The committed state changes only at the round's commit (`TentState.commit`,
`spec_tentative.py:247-286`). The fork never reads the tail slot (CHECK-ONLY stat `tail_reads`, MUST 0).

Why scratch copies and not the buffer itself in the arena: with the buffer in the arena (L x B blocks) CB1 needs no
copy, but the buffer must then exist BEFORE the prefill (the arena is one allocation made before the prefill). That adds
0.45 GB (B 216) / 0.67 GB (B 318) to the PREFILL peak, which is the binding peak at C 128 B 216 (section 6). The scratch
copies cost about 3.3 GB of device traffic per joint pass at B 318 (6 element-wise ops per layer; section 7: 2-6 ms).
Section 6 chooses memory headroom over these milliseconds.

### 2.2 The store (spill-free)

`_fork_state` refuses a store below the bound (full_max - 1) x 64. In contract 4 a request has at most K = 3 verify
rows per joint pass (group K - 1, final <= K, target 1), and they are rows 0 .. v - 1. Row 0's residency already holds a
verify row's blocks, so the other verify rows add at most 2 x 64 listed blocks: X = 1 + 2 x 64 = 129. C = 1: one fill,
no spill, nothing dropped (MUST: dropped 0, unserved 0). The check arm (TFC: every row a verify row) needs X 193.
`TentRunnerF` refuses a smaller store before the meter attaches (only the planted control fork_small_store passes).

## 3. Contract 4, D1 and D3 (kept exactly)

- D1: no residency undo (calls residency_undo 0); a reset writes only the twin counters; the tail slot holds only
  committed rows (the chain images are scratch).
- D3: the deferred row is replayed from its buffer entry (no GEMM, no residency, no attention); deferred_writes 0;
  K + 1 rows per joint pass.
- Exactly N tokens per request (COMMIT-N).
The CPU tests check each of these on L4PTF (section 5), with the plants recompute_deferred and undo_residency.

## 4. CHECK-ONLY work (off in the timing cells)

- PRESENT: every selected block of a verify row is present (dropped 0).
- VIS / VB: from the tables that the KERNEL wrote, each row sees exactly its own (block id, tokens) set; the tail slot is
  never read (TAIL); no entry of another request.
- CHAIN: CB0's committed rows have the tail slot's bits, CB0's other rows and CB1 have the buffer's bits, and every buffer
  row that a row sees holds the chain's position of its block (tpos = 64 qb + x, resp. 64 (qb + 1) + x).
- "F" / "RA" probes (once, layer 0): requests alone and a row alone give the full launch's bits.

## 5. CPU tests (run before the hand-over)

`benchmarks/Efficiency/tests_cpu/test_i3.py` (stand-ins, wsc_world.fake_fork) and
`benchmarks/Efficiency/tests_cpu/test_i3_interp.py` (the REAL fork under the Triton interpreter; it needs its own srun).
The claims: Gate 0 (files; L4PT, L4PTU, L4PJ bit for bit against 63da01d; L4XU, L4XF against db21536); the arena and
the NoCopyPrefillHook; SH's tokens within TOL; SH's selections; every selected block present; VB; TAIL; CHAIN; rows past
an edge; D1 / D3 counters; exactly N; one residency and ONE fork call per layer and pass (independent counts); the byte
check of every (row, block, token) the kernel shows; the plants fork_union_rows, fork_small_store, fork_probe_swap,
window_tail, no_causal, no_lower, recompute_deferred, undo_residency (each CAUGHT); WS-G's check on the stand-ins (WF
REPORT, WFP planted MUST); the cell's arguments, pure parts and step script.

## 6. Memory (INFERENCE from MEASURED peaks; GB = 1e9 bytes, the cells' gpu_peak_alloc_GB)

MEASURED inputs:

- SH arm peak (the prefill dominates): 82.043 GB at C 128 B 216 (i2a), 74.964 GB at C 73 B 318 (i2bx).
- L4PT arm peak (decode): 80.716 GB at C 128 B 216, 74.218 GB at C 73 B 318 (SCORING_wsa.md:278).
- WS-C, C 73 B 318 (record wc_d1b_2195573_wc1b): L4XF arm peak 79.062 GB = SH 74.964 + 4.098. MODEL: store X 129 2.699
  + the copying re-home's one-layer transient 1.548 = 4.247 (difference -0.15 GB). So the prefill peak adds the store and
  the transient.

Arithmetic (bf16, H 2, D 128: a block = 64 x 2 x 257 x 2 B = 65,792 B):

| term | C 128 B 216 | C 73 B 318 |
|---|---|---|
| store X 129 (L4PTF) | 216 x 129 blocks = 1.8332 | 318 x 129 = 2.6989 |
| CB0 (one block per request) | 0.0142 | 0.0209 |
| fork tables, lists, output (E = 4 B) | about 0.01 | about 0.015 |
| copying re-home transient (one layer, NS = 64 + P) | 1.8332 | 1.5482 |
| L4PTF decode peak = L4PT + store + CB0 + tables | 80.716 + 1.857 = 82.57 | 74.218 + 2.735 = 76.95 |
| L4PTF prefill peak, NoCopyPrefillHook = SH + store + CB0 | 82.043 + 1.847 = 83.89 | 74.964 + 2.720 = 77.68 |
| L4PTF prefill peak, copying hook (+ transient) | 85.72 | 79.23 |
| L4PTF arm peak (NoCopyPrefillHook) | **83.9 (against 84 GB: 0.1 GB left)** | **77.7 (6.3 GB left)** |
| TFC check arm (X 193: store 2.7427 / 4.0379) | 82.043 + 2.757 = 84.80 (above 84 GB) | 74.964 + 4.059 = 79.02 |
| L4PTU X 193 (store after the prefill) | 80.716 + 2.743 = 83.46 | MEASURED i2bx decode 72.885 GiB = 78.26 |

The A100 card holds 80 GiB = 85.90 GB in all. With the copying hook the C 128 B 216 fork arm needs about 85.7 GB: an
out-of-memory error is likely. So the cell uses `--prefill-hook nocopy` (r2_i3_arena.NoCopyPrefillHook: the engine's
three GPU row tensors are allocated as the arena's views; no copy, no transient). RISK: at C 128 B 216 the L4PTF arm
peak is about 83.9 GB, 0.1 GB below 84 GB; the TFC check arm (i3ca) is about 84.8 GB, above 84 GB and 1.1 GB below the
card. An out-of-memory error there is a RUN failure, not a band miss.

## 7. Inferred bands (INFERENCE; MEASURED inputs; for the lead's registration, not a registration)

The registration lives in `r2_i3_cell.py`: `PRED_I3_BY_B` (every value None) and `PRED_STAMP_I3` (None). The step
script stops i3a / i3b while `PRED_STAMP_I3` is None.

### 7.1 C 73 B 318 (the cell i3b; the strong basis: one job each)

MEASURED (record /mnt/beegfs/ojerbi/r2-runs/spec/i2_x2bx_2195574_i2bx, same job): SH step 124.064 ms, SH median 2,563.2
tok/s; L4PT median 1,239.6 tok/s, joint 518.661 ms; L4PTU X 193 median 1,323.9 tok/s (I2X / I1 = 1.068), round =
reset 0.119 + joint 466.248 + commit 8.572 + draft 196.2 = 671.14 ms. L4PTU's probe-pass groups (ms per pass):
residency 80.02, write 11.07, union 26.95, fetch 95.79, attention 119.48 (call A + store call 103.24, call B 14.59,
merge 1.68), spill 0, dense_next 122.50.

MEASURED (record /mnt/beegfs/ojerbi/r2-runs/wsc/wc_d1b_2195573_wc1b, C 73 B 318): the fork call (lists + launch) 56.48
ms per pass (L4XU's window + store calls 96.02 + spill 84.33); the fork's locate 13.03 ms; its fetch (host only, pool
read in place) 91.37 ms; L4XF / L4XU 1.147; joint saving 131.7 ms.

The joint pass of I3 = I2X's, with these terms changed:

| term | I2X (MEASURED) | I3 (INFERENCE) | basis |
|---|---|---|---|
| attention | 119.48 | 56.48 (50-75) | one fork call; CB0 / CB1 take the places of the tail slot / store slot 0 of WS-C, so the union has the same entries |
| union / locate | 26.95 | 13.03 (12-20) | WS-C's locate + the chain overlay |
| fetch | 95.79 | 91.37 (88-96) | host blocks only; pool read in place |
| chain images | 0 | 3.5 (2-6) | 3.34 GB per pass at about 1.3-1.5 TB/s + 192 launches |
| residency, write, dense_next | 213.59 | 213.59 | unchanged |

Joint: 466.248 - 119.48 + 56.48 - 26.95 + 13.03 - 95.79 + 91.37 + 3.5 = 388.4 ms (band 376-421).
Saving against I2X: 466.2 - 388.4 = 77.8 ms (band 45-90).
Round: 671.14 - 77.8 = 593.3 ms (band 581-631). The tokens per round are I2X's (the same schedule and residency).
I3 / I2X = 671.14 / 593.3 = 1.131 (band 1.06-1.155).
I3 / SH = 0.5165 x 1.131 = 0.584 (band 0.55-0.60).
I3 / I1 = 1.068 x 1.131 = 1.208 (band 1.13-1.24).
Joint in SH steps = 388.4 / 124.064 = 3.13 (band 3.03-3.40).
Rollback: about I1's (D1 kept): 0.070 SH steps (band 0.03-0.30).
Arm peak: 77.7 GB (section 6; band 76.5-79.5).

### 7.2 C 128 B 216 (the cell i3a; a weaker basis)

The I2X cell at C 128 (i2ax) and WS-C's C 128 cell (wc1a) were running at the time of writing (no exit record). So
the C 128 terms are scaled from C 73. MEASURED (SCORING_i2.md:331, i2a, X 48): L4PTU joint 322.829 ms with probe
groups resid 21.98, write 11.11, union 30.36, fill 0 17.43, call A + fill-0 call 64.44, spill 65.22, call B 11.04,
division 1.02, dense 96.42; L4PT joint 314.959 ms, round 471.998 ms, 573.8 tokens per round (L4PTU 555.7); L4PT / SH
0.3947; SH step 64.724 ms.

- I2X joint (the lead's MODEL, PRED_I2X_BY_B): 322.8 - 65.2 = 257.6 ms; I2X round 481.194 - 65.22 = 416.0 ms.
- attention 76.50 x (56.48 / 119.48 = 0.473) = 36.2 ms; union 30.36 x 0.483 = 14.7 ms; fetch 17.43 x 0.954 = 16.6 ms;
  chain 2.5 ms (2.27 GB per pass).
- I3 joint = 257.6 - 76.50 + 36.2 - 30.36 + 14.7 - 17.43 + 16.6 + 2.5 = 203.3 ms (band 185-240).
- Saving against I2X: 54 ms (band 25-80). Round 416.0 - 54 = 362 ms.
- I3 / I2X = 416.0 / 362 = 1.149 (band 1.06-1.22).
- I3 / I1 = (471.998 / 362) x (555.7 / 573.8) = 1.304 x 0.9685 = 1.263 (band 1.14-1.37).
- I3 / SH = 0.3947 x 1.263 = 0.499 (band 0.45-0.54). Joint in SH steps 203.3 / 64.724 = 3.14 (band 2.85-3.70).
- Arm peak 83.9 GB (section 6; band 83.0-84.6; a value above 84 is a risk, not a band miss).

### 7.3 The rung falsifier

I3-ABOVE-I2X: L4PTF's median committed tok/s must be above L4PTU's (X 193) in the same job. The bands above put it
6-22 % above.

## 8. Cells (r2_i3_step.sh; SH runs first in every cell)

| cell | C | B | steps | arms | notes |
|---|---|---|---|---|---|
| i3m | 73 | 8 | 72 | L4PT, L4PTU, L4PTF, SHR, TFC | the MINIATURE: check every round, require stress, 16 check passes after 1 + 2 plain steps (3 rows past the edge at 16192, 1 F4 row per request) |
| i3a | 128 | 216 | 512 | L4PT, L4PTU (X 193), L4PTF | timing; STOPS while PRED_STAMP_I3 is None |
| i3b | 73 | 318 | 512 | L4PT, L4PTU (X 193), L4PTF | timing; STOPS while PRED_STAMP_I3 is None |
| i3ca | 128 | 216 | 129 | SHR, TFC | check (32 passes; memory risk, section 6) |
| i3cb | 73 | 318 | 129 | SHR, TFC | check (32 passes) |
| i3g | 73 | 100 | 64 | WJ, WT, WF, WFP | WS-G: WJ MUST, WT / WF REPORT, WFP planted MUST |

## 9. Risks

1. Memory at C 128 B 216 (section 6): the L4PTF arm about 83.9 GB, the TFC arm about 84.8 GB.
2. NoCopyPrefillHook replaces the name `torch` in `prefill_update`'s globals during the first sub-batch call. A change of
   cache_engine's allocation code (other shapes, other names) makes the hook raise ArenaError (it checks every layer);
   it cannot give wrong bits silently.
3. The chain images assume that a request's chain spans at most two blocks (K + 2 = 5 positions < 64) and that a row's
   own block is in its selection (NOSA's local block). VB and CHAIN check both on every check pass.
4. CPU stand-ins model the arithmetic, not GPU dispatch, ordering, timing or memory. The interpreter test runs the real
   kernel's code on the CPU, not its compiled form.
5. The C 128 bands use C 73 ratios (section 7.2). Read them as weaker than the C 73 bands.
