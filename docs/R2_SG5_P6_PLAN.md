# SG5 p6: the batched rebuild and the fused twin on the EAGER SG5 path (DESIGN ONLY)

Branch `feature/nosi-r2-sg5-p6`, worktree `/mnt/beegfs/ojerbi/nosi-r2-sg5-p6`, from B's `c675779` (the SG5 line with
backend b2, the code p5b2 ran). Nothing is built here until the lead's GO.

## 0. The question and the answer in short

p5b2 (b2 as SG5's service, exact, record `/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_p5_2190876_p5b2/summary.json`) has a
round tick of 1347 ms:

| term (mean per round tick) | ms | how it is read |
|---|---|---|
| device: gemm + att + sel + store | 58 | CUDA-event spans |
| `twin` (the torch twin, host-bound) | 119 | span |
| `res` (the per-layer backend calls) | 252 | span |
| `end_tick` (host `backend` - `res`) | 143 | 395 - 252 |
| REST (`residual` - `end_tick`): `apply_events`, `finish_tick`, ops outside the spans | 775 | 918 - 143 |

REST is 58 % of the tick. On the draft-only tick (no decision) REST is 6 ms (359 - 353), so about 770 ms of it is
the decision step. D's 16K profile shows the same on the device: 306.9 ms of kernels outside the named ranges per
verify tick (16.6 ms per draft tick).

The cause (static reads, c675779):
- every publication runs `rebuild_from_own_rows` on every layer (`r2_tick.py:114`, the loop in `apply_events`
  `r2_tick.py:706`); p5b2 has 1003 publications in 61 ticks, so about 527 calls per round tick;
- each call is a CHECKED `twin.advance` (`r2_spec_core.rebuild_draft_bank` -> `PositionalTwin.advance`): the torch
  twin's `step` with host reads (a device sync per call), about 190 eager kernels per replayed row;
- `TickEngine(twin="fused")` (07c4940) does NOT reach it: `attach` binds only `step_unchecked` /
  `advance_unchecked` (PF3b, registered in `docs/R2_SG5_FUSED.md` before any fused run).

So p6 brings two pieces of the captured line to the eager SG5 path:
1. the BATCHED REBUILD (A2, 7f06273): all publications of a tick rebuilt with ONE `advance_unchecked` per layer
   (no host read; 32 calls per tick instead of 527);
2. with the fused twin (`twin="fused"`), that one call per layer is ONE fused kernel.

## 1. The choice: a flag in TickEngine, not FixedTickEngine

Option A (SG5 through FixedTickEngine in eager mode) changes the engine under every gate: padded rows, two
capacities, the merged decision read, the timer's categories (`region` instead of the per-op spans). p6 would not be
comparable with p3 / p4 / p5b2, and B2's CPU lockstep (test_r2_tick_b2.py) runs on TickEngine.

Option B (CHOSEN): `TickEngine(rebuild="per_call" | "batched")`. "per_call" is the default and changes nothing.
"batched" replaces only the publication loop of `apply_events`. Every other step, every span and every gate stays
SG5's, so p6 reads directly against p5b2 (same point, same harness, same categories).

Not in p6 (named as the next term): the ROLLBACK path stays per call and checked (`SC.rollback_bank` per layer with
`PositionalTwin.commit`'s host checks, then `sync_draft_bank` + `copy_ring` per rolled-back bank and layer). E1's
`commit_banks` (host-mirror checks, `commit_unchecked`) would remove it; it needs the epoch host mirror that only
FixedTickEngine keeps today.

## 2. The code (every touched line of `r2_tick.py` marked `[batched-rebuild]`)

`nosi/nosi/r2_tick.py`:
- `rebuild_banks_unchecked(twin, items, B, dev_items=None, swap=False)`: 7f06273's function, unchanged, moved to
  `r2_tick.py` so that TickEngine and (later) r2_tick_fixed share ONE copy. FIRST every draft bank's own unverified
  rows (one gather of the ring and of `total_cis` over all items), THEN `sync_draft_bank` of every item (index copies),
  then ONE `advance_unchecked` over T = max(dpos - p) with n[B + b] = dpos - p. A bank is inactive past its own rows,
  as every other bank is in the per-call path, so the result is bitwise the per-call result per bank.
- `REBUILD_MODES = ("per_call", "batched")`; `TickEngine.__init__(..., rebuild="per_call")`; `clone` keeps it.
- `apply_events`: with "batched", the publication loop becomes: the items and their device tensor ONCE per tick,
  `rows_to_copy` (copy_rows' index lists of every publication) ONCE per tick, then per layer ONE
  `rebuild_banks_unchecked` and ONE `copy_rows_many`. The host checks (the draft bank not behind the publication)
  run for all items BEFORE any write. The epoch / `synced` bookkeeping is unchanged.
- `rows_to_copy`, `copy_rows_many`: 7f06273's methods.

`benchmarks/Efficiency/r2_tick_sg5.py`:
- `--variants twin/rebuild,...` (default `torch/per_call`, so every existing cell is unchanged; `--twin` stays as the
  default variant's twin). Each variant is one more arm per (mode, backend): its own draft-only tick and round, its
  label `SG5_<mode>_<backend>_<twin>_<rebuild>`, its meta. `make_engine` passes `twin` and `rebuild` to
  `TickEngine.from_r2`.
- the step script gets a p6 cell: p5b2's command (b2, C 96 / X 64 / R_ver 16, 48 steps, gilfree 8, superset wait,
  unchecked) with `--modes exact` and `--variants torch/per_call,fused/per_call,torch/batched,fused/batched`. A 2 x 2
  attributes each term to its piece; torch/per_call is p5b2's replica (the control). p5b2's point took 84 s of wall
  per arm, so the four arms take about 10-15 min with the shipped baselines.

## 3. CPU tests (each with its negative control)

- **P6a** `rebuild_banks_unchecked` == SG5's CHECKED per-call `rebuild_from_own_rows`, per bank, bitwise (every table
  and counter; the captured rows; a replay across a key and a cis compression; a bank with nothing to replay).
  NEGATIVE CONTROL: a planted bank-order swap (`swap=True`) differs. Refusals: two items of one bank, a replay longer
  than the ring.
- **P6b** the SG5-shaped rounds with `rebuild="batched"` == `rebuild="per_call"`, bitwise every tick (rows, logits,
  twin counters, the committed tokens, the tables at the end): the teacher-forced round on the ResidentStore (T4's
  check against the serial reference) and on the residency backends v1 / sb / sc / b2 (T15's check), and the
  free-running round on b2. Precondition: ticks with two or more publications occur. NEGATIVE CONTROL: an engine whose
  batched call passes `swap=True` differs at the first tick with two or more publications.
- **P6c** (the Triton interpreter, a child process, as test S2 of 07c4940): the SG5-shaped round on b2 with
  `twin="fused", rebuild="batched"` == `twin="torch", rebuild="per_call"`, bitwise (rows, argmax, twin counters every
  tick, committed tokens, decision counts, tables). PLANT 2 in the fused path effective and caught.
- **P6d** the harness: `--variants` parses, refuses an unknown twin or rebuild, reaches `make_engine` and the meta;
  the default keeps p5b2's labels; the p6 cell's DRYRUN prints the four variants.
- The lint: `rebuild_banks_unchecked`, `rows_to_copy`, `copy_rows_many` and the batched branch of `apply_events`
  contain no host read (`.item`, `.tolist`, `.cpu`, `bool(`, synchronize).

## 4. Registered predictions for p6 (exact mode, per round tick, before any run)

The basis: p5b2's terms (section 0); the batched rebuild is about 44 small ops per layer plus one twin advance; an
eager op costs 8-10 us of host time on this path (p3, p5b2); a rollback tick (about 2 per decision tick in p5b2:
123 rollbacks in 61 ticks) keeps its per-call checked path (SC.rollback_bank per layer with host checks, then about 15
ops per layer and rolled-back bank).

| variant | `twin` span | REST (`residual` - `end_tick`) | T_tick |
|---|---|---|---|
| torch / per_call (p5b2's replica, the control) | 105-135 ms | 0.70-0.85 s | 1.20-1.50 s |
| fused / per_call | 2-6 ms | 0.70-0.85 s (PF3b: the rebuild stays CHECKED torch) | 1.08-1.39 s |
| torch / batched | 105-135 ms | 0.12-0.30 s (32 calls per tick, each 44 ops + T x 190 torch kernels, T the longest replay, 2-3) | 0.62-0.95 s |
| fused / batched (p6's target) | 2-6 ms | 0.03-0.11 s (about 1.4 k ops, 11-15 ms, + the rollback path 15-60 ms on a rollback tick + the read and the write-backs) | 0.45-0.62 s |

- `res`, `end_tick` and the device categories are not changed by either piece: each within 15 % of p5b2's (252, 143,
  58 ms) in every variant.
- The decisions do not change: r, publications and rollbacks per arm within the run-to-run spread of free-running
  gilfree (p5b2's exact and fast arms: r 0.873 / 0.869, publications 1003 / 1002, rollbacks 123 / 126).
- r x T_shipped / T_tick improves by T_tick(p5b2) / T_tick(variant): 2.2-3.0x for fused / batched.
- For D's next kernel-time profile (not a p6 term): OTHER per verify tick falls from 306.9 ms to 20-45 ms with fused /
  batched (the draft tick's 16.6 ms, the batched rebuild's 32 x (one fused kernel + 44 small ops), the rollback ops).
- FALSIFIERS: a REST or twin span outside its band; any change of `res` / `end_tick` / device beyond 15 %; a
  bookkeeping gate failing (twins vs mirrors, a void tick); fused / per_call's REST falling below 0.70 s (PF3b
  refuted: the checked rebuild would reach the fused path).

## 5. What I need

- GO for option B and the 2 x 2 cell (or only fused / batched + the control).
- After p6: E1's unchecked rollback on TickEngine (commit_banks with an epoch host mirror) is the next REST term.
