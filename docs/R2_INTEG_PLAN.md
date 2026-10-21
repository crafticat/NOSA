# R2 integration: every gap-closing build at once (r2-integ)

Branch `r2-integ`, worktree `/mnt/beegfs/ojerbi/nosi-r2-integ`, from sharedkv4 `1c5ff8a` (which descends from p6
`cf35536` via `f79d0df`). SOURCE ONLY, NOT RUN: no CPU test and no GPU step ran on this branch. Written
2026-10-21T11:51:27Z (`date -u`). Labels: MEASURED (a cited record), MODEL (a derivation here or in a cited plan).

## 1. What is merged (in this order)

| merge | branch, commit | what it brings | conflicts |
|---|---|---|---|
| base | r2-sharedkv4 `1c5ff8a` | p6 (batched rebuild, fused twin), shared-KV kernel, Residency page keys, fused union | - |
| 1 | r2-fusedres2 `740ea1d` | `OwnResidency(resolve="fused" / "check")`, `r2_fusedres.py` | none (textual) |
| 2 | r2-xpool `feaa7d6` | the shared X pool layout (`r2_xpool.py`, step S6) | `r2_tick.py` (1 hunk), `r2_tick_sg5.py` (2 hunks) |
| 3 | r2-gfw-warm `000870e` | the gilfree C++ worker, `issue_desc`, the built-in warm-up | `r2_transport_gilfree.py` (2 hunks) |

Conflict resolutions (each merge commit lists them):
- `r2_tick.py` `TickEngine._layer`: sharedkv's `**akw` (page_key, rows_max, group_token) now reaches BOTH attention
  calls: the per-stream one and xpool's flat-address one (`self.att[l]`, stride(0) = 0 views).
- `r2_tick_sg5.py`: fusedres's and xpool's flags and `one_point` info lines, both kept.
- `r2_transport_gilfree.py`: gfw-warm moved the destination checks into `_check` (both workers); xpool had extended
  them in `_release`. xpool's flat checks now live in `_check`.

## 2. The interactions made to work together (the integ commit)

1. FUSED RESOLVE + XPOOL (docs/R2_FUSEDRES_PLAN.md section 5, implemented in `FusedResolver`):
   - the refusal of an attached xpool is gone (an xpool must be in mode "pool");
   - first layer of a tick: after `sync_mirror`, the blocking-decision-read rule, then
     `xpool.release_free(dmirror.code)`, as `ResidencySC.resolve` does;
   - after V2, before the device wait and gather: `xpool.bind(l, vdec.bh, take_slot, granted | evict)`;
   - D2 runs with `BIAS 0` when the pool is attached; after D2: `xpool.bind(...)` on the draft takes, then
     `r2_xpool.draft_bias_fill_flat` (the draft entries' (b, h) from `_bhd`, as `_record_inputs` builds them);
   - check mode compares the bias rows through `draft_bias_fill_flat` on a copy of the flat bias store.
   Cost: xpool's own torch ops (two binds and the audit per layer, one release per tick) on top of 3 / 6 launches;
   so FR3's "no torch compute op on a later draft-only layer" holds only without the pool.
2. PAGE KEYS + FUSED RESOLVE: `FusedResolver.resolve` replaced S-C's resolve without the [sharedkv2] hooks. Both are
   back: `_check_tail_keys(rows)` once per tick (before `sync_mirror`, S-C's order) and
   `plan.page_key = svc._page_keys(...)` from the FUSED decisions' `(blk, valid / ready)` columns (`page_keys_of`).
3. SHARED-KV KERNEL + XPOOL FLAT MODE: no kernel edit. The shared kernel takes its strides from the tensors
   (`k_cache.stride(0)` etc.), so a stride(0) = 0 view adds `n * 0` and the page table's absolute positions address
   `m * 64 + r`: the xpool flat mode by construction (docs/R2_XPOOL_PLAN.md section 3). The tick passes the same
   `self.att[l]` views to both kernels. CPU check: ST5 (below).
4. GILFREE + XPOOL on the C++ worker: `_submit` passes `m' = Transfer.phys` and the flat store as a
   `(1, M * 64, H, D)` view (submit checks 4-D; `release_impl` reads `view(-1, D)`, so `S_dst = M * 64`, as the
   Python worker). `issue_desc` refuses the pool (C++ computes `m'` from logical slots).

## 3. The SG5 stack (r2_tick_sg5.py)

New flags (each default unchanged): `--attn per_row|shared|shared-whole`, `--union torch|fused`, `--sharedkv PATH`,
`--sel-check` (TickEngine's per-layer selection check, one host read per layer), `--shared-unchecked`,
`--gilfree-warm-up on|off` (default on = the transport's production default since 000870e), `--stack`,
`--stack-compare`, `--shipped-batch`.

`--stack full` = twin fused, rebuild batched, b2 resolve fused, layout pool (X_pool from `--X-pool`), attn shared
split + union fused, gilfree warm-up on. `--stack full-no-<c>` = full minus component c; `--stack none` = every
component OFF (the pre-integration path, p5b2's code: warm-up off). A stack refuses an explicit component flag, a
non-b2 backend, and `--attn shared` without `--sel-check` (or an explicit `--shared-unchecked`).
`--stack-compare`: per point three arms none, `<stack>`, none2 (`one_point_stacks`), every arm with the run's
fixed `--arrival-lag`; MUST `<label>_ST_replica`, `_ST_committed`, `_ST_bitwise`; REPORT `R_<label>_ST_tick`,
`R_<label>_ST_peak`. With `--attn shared`, every point has MUST `<label>_SKV_ran` (the shared kernel ran, with the
requested union build only).

## 4. CPU tests (benchmarks/Efficiency/tests_cpu/test_r2_integ_stack.py, NOT RUN)

ST1 full == default bitwise (b2, pressure point C 8 / X 40 / R_ver 4, teacher-forced, arrival lag 2, gilfree in
every arm), with each component's precondition; ST2 full minus each component == default (six arms; the removed
component OFF, the others ON); ST3 the comparison sees the fused resolve's plant 2 (precondition d_grant > 0); ST4
check mode on the full stack (pool + keys): 0 differences at every layer; ST5 zero-stride views in the shared kernel,
piece 3's kernel and the fused union kernel (interpreter), with a shifted-table negative control; ST6 static merge
points; ST7 the harness presets, refusals and `one_point_stacks` gates (stand-in); ST8 the step cells (DRYRUN).
Regression files: test_r2_fusedres.py, test_r2_xpool.py, test_r2_transport_gilfree.py, test_r2_sharedkv2_keys.py,
test_r2_sharedkv3_union.py, test_r2_sharedkv4.py, test_bi_paged_attention_shared.py, test_r2_tick.py,
test_r2_tick_b2.py, test_r2_tick_p6.py, test_r2_twin_fused.py, test_r2_residency.py, test_r2_own2.py.

## 5. REGISTERED GPU cell stack1 (2026-10-21T11:51:27Z, before any run)

Step: `bash benchmarks/Efficiency/r2_tick_sg5_step.sh --nosi <sha> <holder> <run id> <cell>`, cell ROOT
`NOSI_INTEG` (since integ2: /mnt/beegfs/ojerbi/nosi-r2-integ2), in the order stack1a, stack1t, stack1b; stop at the first nonzero exit (stack1t and
stack1b are timing claims only after stack1a passed). Every cell: p5b2's point `--modes exact --backends b2 --C 96
--X 64 --R-ver 16 --X-pool 64 --stack full`, 48 steps, gilfree 8 threads, unchecked b2, the superset wait.

| cell | B | other arguments | role |
|---|---|---|---|
| stack1a | 64 | `--stack-compare --arrival-lag 2 --sel-check`, no `--pool-baseline` | EQUALITY: none / full / none2 |
| stack1t | 64 | `--shared-unchecked --pool-baseline` (free timing, as p5b2) | TIMING at p5b2's point |
| stack1b | 216 (integ2; was 264) | `--shared-unchecked --repeat-docs --shipped-batch 64 --log-slots -1`, `--mem=240G` | TIMING at the largest B (bigb MODEL, 5.4b) |

### 5.1 The basis (MEASURED, p5b2's record `r2-runs/sg5/sg5_p5_2190876_p5b2/summary.json`, exact, per round tick,
as tabulated in docs/R2_SG5_P6_PLAN.md section 0)

T_tick 1347 ms = device (gemm + att + sel + store) 58 + twin 119 + res 252 + end_tick 143 + REST 775; draft-only res
span 129 ms (R2_FUSEDRES_PLAN.md section 7); r 0.873.

### 5.2 Per-term predictions for the full stack at B 64 (stack1t; MODEL, each from its component plan)

| term | p5b2 | full stack | source of the term's prediction |
|---|---|---|---|
| twin | 119 ms | 2-6 ms | p6 plan section 4 (fused / batched) |
| REST (residual - end_tick + rebuild) | 775 ms | 30-110 ms | p6 plan section 4 |
| res | 252 ms | 20-80 ms | below |
| end_tick | 143 ms | 122-170 ms | p6: within 15 %; xpool: + one assign_phys read and its host loop (1-5 ms) |
| device (gemm + att + sel + store) | 58 ms | 45-70 ms | p6: within 15 %; sharedkv split fused (build + shared) / per_row 0.60-1.10 of the att part |
| T_tick | 1347 ms | 0.22-0.45 s, point 0.31 s | p6's fused / batched band 0.45-0.62 s with res 252 replaced by 20-80 |
| r | 0.873 | 0.84-0.90 | no component changes a decision (p6, fusedres, xpool, sharedkv: bitwise claims) |
| ratio r x T_shipped / T_tick | p5b2's | 3.0-6.1 x p5b2's | 1347 / T_tick |
| meta.peak_gb | p5b2's | p5b2's minus 7.9-8.9 GB | xpool X-M (store 22.10 -> 13.61 GB at B 64) |
| transport_warm_calls | - | 1 | gfw1f |

Derivation of res (MODEL): the fused resolve's draft-only span is 4-25 ms against S-C's 129 (fusedres plan,
factor 0.03-0.19); applied to the round's 252 ms that is 8-48 ms. Added: the device wait and the gather (S-C's,
unchanged; about 15 ops per layer: about 5 ms per verify tick), xpool's binds and translate (5-12 ms per tick, xpool
plan section 6), the page keys (about 25-30 torch ops per layer for `_page_keys` / `page_keys_of`, 32 layers, 8-10
us each: 6-10 ms) and the fused union's groups (once per tick) in the res span. Sum 24-75 ms; band 20-80 ms.

### 5.3 stack1a (B 64, the equality cell; arrival lag 2, selection check on in all three arms)

- MUST `SG5_exact_b2_stack_full_ST_replica`, `_ST_committed`, `_ST_bitwise` PASS (the CPU claim ST1 on the GPU);
  `SG5_exact_b2_stack_full_stack_SKV_ran` PASS; the bookkeeping gates (no void tick) PASS in all three arms; the b2
  gilfree preflight on the full stack PASSES.
- REPORT `R_..._ST_tick`: T_tick(full) / mean(T_tick(none), T_tick(none2)) in 0.15-0.40 (`ST_TICK_BAND`): both
  arms carry the same 32 host reads per tick of the selection check and the same arrival-lag waits.
- REPORT `residency.report.b2.fused` of the full arm: err 0, launches_per_layer between 3.0 and 6.0.

### 5.4 stack1b (B 264, the largest B the xpool model admits)

The B (MODEL, docs/R2_XPOOL_PLAN.md section 4.1, X_pool 64 rows): B at 84 GB is 265.6 (case A, F 16.37 GB, the most
conservative X_pool-64 row) to 287.8 (case B, F 13.12). B 264 is below the most conservative value (the largest
multiple of 8 below 265.6). Its predicted device memory: F + B x marginal = 13.12 + 264 x 0.24621 = 78.1 GB (case B)
to 16.37 + 264 x 0.25463 = 83.6 GB (case A). Host memory (MODEL): the pinned host rows are B x n_host x H x D x 2 B x
2 (K, V) x L = 264 x 16352 x 2 x 128 x 2 x 2 x 32 = 141 GB, so the step asks `--mem=240G` (120G at B 64).

| term | full stack at B 264 | derivation (MODEL) |
|---|---|---|
| peak GB | 78-84 | above |
| twin | 8-25 ms | B 64's 2-6 ms x 4.1 (one fused kernel over 2B banks) |
| res | 30-150 ms | host-bound launches unchanged per layer; fusedres kernel 5.5-21 ms per tick at B 287 (its plan); binds and keys over 4.1x the entries |
| end_tick | 0.42-0.70 s | B2's host consumer per entry: 143 ms x 4.1 (+-15 %) |
| REST | 0.05-0.30 s | batched rebuild host ops once per tick; rollbacks per tick x 4.1 |
| device | 150-280 ms | GEMMs compute-bound at 1,056 rows (2.5-4x), attention reads x 4.1 |
| T_tick | 0.7-1.5 s, point 1.0 s | sum |
| tok_per_s | 150-330 | r x B / T_tick = 0.87 x 264 / (0.7-1.5 s) |
| shipped_tok_per_s | REPORT | the shipped decode at B 64 (`--shipped-batch 64`): `ratio` is None |

### 5.4b REVISION in integ2 (2026-10-21T12:12:13Z, before any stack1 run): stack1b at B 216, not B 264

The xpool model of 5.4 has no twins, no build state and no workspace. bigb's one-engine model (docs/R2_BIGB_PLAN.md
section 2, bigb_core.MemModel, its byte terms checked against the code's tensors in BG3, F0 and R_FIX fitted to p5b2,
checked on p4sc64: 67.38 predicted, 67.41 measured) gives, at C 96, pool layout, X_pool 64, b2, 24 steps: B 128
51.78 GB, B 216 75.89 GB (its B_max at a 76 GB budget), 84 GB at B 245. Linear in B between its points (slope
(75.89 - 51.78) / 88 = 0.274 GB per request; it reproduces the model's own 245 at 84 GB), B 264 is 75.89 + 48 x
0.274 = 89.0 GB: above the A100's 80 GB. So stack1b moves to B 216 (`--batch 216`), the same B as bigb96-216 (the
stack OFF, 24 steps, lag 2): a direct with / without comparison at one B, protocol differences named below.
The stack adds no tensor the model lacks at this size (the shared kernel's workspace replaces piece 3's of the same
shape; the fused resolve's per-tick scratch is a few MB); 48 steps instead of 24 add < 0.1 GB (twin widths and the
token log; `--log-slots -1` sizes the log).
Host memory: pinned host rows 216 x 16352 x 2 x 128 x 2 B x 2 x 32 = 115.7 GB + about 17 GiB base (bigb_core
HOST_BASE_GIB); the shipped child runs first and alone at B 64. `--mem=240G` (the holder has 480G).
stack1t (B 64): with the preflight leak fixed (bigb), meta.peak_gb is ONE engine: 42.73 GB per-stream (bigb's side
prediction) minus the pool's 8.49 GB = 34.2 GB, band 33-36 GB (this replaces "p5b2's minus 7.9-8.9 GB": p5b2's 67.49 GB
was two engines).

| term | full stack at B 216 (replaces the B 264 table) | derivation (MODEL) |
|---|---|---|
| peak GB | 74-78 | bigb one-engine model 75.89 |
| twin | 7-20 ms | B 64's 2-6 ms x 3.375 |
| res | 25-130 ms | as 5.4, over 3.375x the entries |
| end_tick | 0.41-0.56 s | 143 ms x 3.375 (+-15 %) |
| REST | 0.05-0.25 s | as 5.4 |
| device | 130-200 ms | GEMMs at 864 rows (2.2-3.4x), attention x 3.375 |
| T_tick | 0.6-1.2 s, point 0.85 s | sum |
| tok_per_s | 155-315 | 0.87 x 216 / (0.6-1.2 s) |
| comparison | REPORT | bigb96-216's T_tick (bigb_core.tick_bands(216)) over stack1b's; protocol differs: stack1b 48 steps, free timing, shipped child at B 64; bigb96-216 24 steps, lag 2, no shipped child |

F-ST4 now reads: an out-of-memory error, or peak above 78 GB (the model is wrong at the stack); T_tick outside
0.6-1.2 s.

### 5.5 Falsifiers (each stops the claim it names)

- F-ST1 any stack1a MUST fails (with the replica passed): a component changes numbers on the GPU when combined; the
  first difference names the tick and row; no timing claim of the stack.
- F-ST2 `_ST_replica` fails: the run is not reproducible under the fixed lag; the bitwise gate is VOID.
- F-ST3 stack1t T_tick outside 0.22-0.45 s; then each term against its band names the component whose model is wrong
  (twin outside 2-6 ms: p6's fused twin; REST outside 30-110 ms: the batched rebuild; res outside 20-80 ms: the
  fused resolve / pool / keys model; device outside 45-70 ms: sharedkv).
- F-ST4 stack1b: an out-of-memory error (device or host), or peak above 84 GB: the B model is wrong at the stack;
  T_tick outside 0.7-1.5 s: the B scaling model is wrong (end_tick is the largest assumed term).
- F-ST5 `transport_warm_calls` != 1 or a gilfree stall (a spin at its bound) in any arm.

## 6. Doubts (not claims)

- Nothing ran. The CPU matrix runs the gilfree transport (pinned packer v2, Python worker, synchronous) inside a
  whole tick round for the FIRST time; until now CPU rounds used K3X's adversarial transport.
- The CPU matrix uses the shared kernel's torch TWIN (`impl="reference"`, as test_r2_sharedkv2_keys K3) in the round,
  not the Triton kernel; ST5 checks the kernels on zero-stride views directly, not inside a round.
- The shared selector needs the selection check (sharedkv4 MINOR-3). The timing cells stack1t / stack1b run without
  it (`--shared-unchecked`): the precondition is checked only in stack1a.
- `--gilfree-warm-up` defaults to on (the transport's default after 000870e), so every SG5 cell run from this branch
  warms up; the stack's OFF value (warm-up off) is p5b2's code path.
- The res and B 264 terms are MODEL with wide bands; end_tick at B 264 (B2's host consumer) is an assumption of
  linear scaling, not a measurement.
- B 264 needs about 141 GB of pinned host rows; whether the holder's allocation admits `--mem=240G` is not checked
  here (srun refuses at launch if not).
- `--repeat-docs` tiles the 94 documents at B 264, so requests repeat; the shared kernel's union is per request, so
  repeats do not share reads.
- test_r2_tick.py T21 checked one_point's gate names with a lowercase regex that could not see `<label>_SKV_ran`.
  FIXED in integ2: the pattern reads any case, the expected list holds SKV_ran, and a second check fails on a gate
  name the pattern cannot read. fusedres2's FR-g gates (fused_gates) are appended through gates.extend, outside it.
- [integ2] fusedres2's fused_gates now also apply to the stack arms with `--b2-resolve fused`: MUST
  `<label>_draft_only_fused_err`, `_round_fused_err` and `_draft_only_fused_dispatch` (3.0 launches per layer, no sel
  conversion, no bias cast; the pool path counts no bias cast). These join stack1a / t / b's MUST gates.
