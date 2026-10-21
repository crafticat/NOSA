# R2 bigb: the tick at B 128 .. 320 at 16K on one A100 80GB (SOURCE ONLY, not run)

Branch r2-bigb from feaa7d6 (xpool; it contains cf35536). Code: `benchmarks/Efficiency/bigb_core.py` (pure model),
`r2_tick_sg5.py` ([bigb] lines), `r2_tick_sg5_step.sh` (cells bigb81-128, bigb81-244, bigb96-128, bigb96-216), CPU
tests `benchmarks/Efficiency/tests_cpu/test_r2_bigb.py` (BG1-BG9) and the cherry-picked `test_r2_tick_log.py` (LOG1-5).

## 1. What limited B (static reads at feaa7d6)

| # | limit | where | effect | change |
|---|---|---|---|---|
| 1 | token log of 1 << 16 slots per run, never reused | r2_tick.py `LOG`, run_rows "token log full" | B x N below about 6.5k-32k rows (N 512: B >= ~60 refuses). At N 24 it does NOT bind below B ~ 190 | D's 7c0f998 cherry-picked (`TickEngine(log_slots=)`, default unchanged), f94fce9 (`make_engine(engine_kw=)`), harness `--log-slots` (0 default; -1 = `log_slots_for`) |
| 2 | the harness kept TWO engines alive | r2_tick_sg5.preflight: `del eng, svc, tp` but not `rnd`; `run_round` returns a TickRound that holds `engine.read_tokens` / `on_publish` / `on_rollback` | the first preflight engine (store, twins, service) lived while the second was built. p5b2 67.49 GB = build 17.73 + 2 x 24.75 + 0.25. One engine: 42.7 GB | preflight and one_point release `rnd` with the engine (unconditional; no computed value changes). Test BG5b with two negative controls |
| 3 | host memory: the shipped child's pinned rows (S + 8192 rows per request, each tensor rounded UP to a power of two) and `--mem=120G` | r2_tick_sg5.shipped_child; step script | B 128 needs ~145 GiB in the child, B >= 129 at 16K ~273 GiB | `--skip-shipped` (opt-in); the bigb cells use `--mem 200G` (parent 82-146 GiB, MODEL) |
| 4 | build phase | build_ragged: 64-slot windows per request | 56.7 GB at B 244 (MODEL): below the decode peak | none |

Checked, not limits up to B 320: int32 extents (`int32_bounds`: the smallest is the host-row tensor, B 515, and the
kernels widen to int64 there); PaddedLinear's m_fix = (1 + K) B (the tick's row bound); the gilfree ring (chunks of 256
groups); B2's reader window (ids per layer read); the sparse record (record_cap None). Not used by SG5 but a limit for
the SG0-SG4 gate: `r2_tick_gate.py` keeps BASE and a `BASE.clone()` alive, each with the RESIDENT store (all blocks).

## 2. Memory model (bigb_core.MemModel; GB = 1e9 B of max_memory_allocated)

peak(B) = F0 + B x build_req + engines x engine(B) + workspace(B) + R_FIX. Every engine term comes from the code and
BG3 checks it against the code's own tensors on "meta" for B 64 .. 320 and both layouts: 2B-bank twins, the store
(per-stream or pool), the token log, S-C's DeviceMirror + GenBoard + sparse pins, B2's Table, the XPool map; workspace =
PaddedLinear buffers + lm_head's padded output and its float copy. F0 (16.449 GB: weights + 0.079) is fitted to p5b2's
build_gb, R_FIX (0.128 GB) to p5b2's peak WITH engines = 2. Independent check: p4sc64 (S-C, no B2 table) predicted
67.38 GB, measured 67.41. One engine at p5b2's point: 42.73 GB (a falsifiable side prediction for any rerun).

## 3. B at 76 GB (MODEL, one engine, pool layout, X 64 / X_pool 64, 24 steps)

| C | B 128 peak | B_max | B_max peak | per-stream at B_max |
|---|---|---|---|---|
| 81 | 47.72 | 244 | 75.82 | above 76 (per-stream B_max ~157) |
| 96 | 51.78 | 216 | 75.89 | above 76 |

At 84 GB the same model gives 277 (C 81), 245 (C 96): below the forecast's 287 (it counts no twins, no build state).

## 4. The cell bigb (registered in the step header; run B 128 before B_max at each C)

EXACT, b2, gilfree 8, `--layout pool --X-pool 64 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped`, 24
steps, no sweep, no pool baseline, `--mem 200G`. Predictions BB-A, BB-M, BB-1, BB-T, BB-D, BB-E in the step header.

## 5. Time bands (bigb_core.tick_bands; MODEL)

Scaled from p5b2's B 64 EXACT categories: each category = x64 (f + (1 - f) B / 64) with a B-independent fraction f
(att 0.57, sel 0.60, store 0.30, twin 0.80, res 0.30, residual 0.15); band f +- 0.3. GEMM: EXACT pads to m_fix = 4B,
G = 25.06 ms + 0.0622 (0.045 .. 0.080) ms per row above 256 (the forecast's GEMM(287) -> GEMM(861) slope). td_dev =
the draft-only tick's gemm + att + sel + store (the reconciliation's td proxy; 60.65 ms at B 64).

## 6. Open points

- K5's X 64 is shared by B requests: at B 244 the shared extras per request fall 4x; a CapacityRefusal in the preflight
  (BB-A) is possible (FG2 refused at C 80 / X 32 / B 64). It would be a configuration result, not a harness fault.
- B > 94 tiles the 94 documents: duplicated requests make the residency demand correlated.
- EXACT pads the draft tick's GEMM to 4B rows, so td_dev over-states a draft step's GEMM; FAST is the fair td.
- The pool layout has not run on the GPU (x1 is first): run bigb after x1 passes.
- Time bands use the per-stream C 96 categories for both C and both layouts.
