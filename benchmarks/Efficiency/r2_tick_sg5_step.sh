#!/bin/bash
# R2 MILESTONE 2, STEP 7, PIECE 4, SG5: ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/
# r2_tick_sg5.py: the R-PERF budget of the flattened tick (nosi/nosi/r2_tick.py) on piece 5's Residency (r2_residency*.py,
# C's 2ec08da: backends sc / sb / v1; [b2] b2 = r2_own2.py; the gilfree transport over agent E's packer v2), in EXACT
# and FAST GEMM modes, at the primary point C 96 / X 32 / R_ver 16 (C's recommendation after holder 2190876: at C 80 the
# verifier's urgent
# misses were refused), then C's sweep C {64, 80, 96} x X {16, 32, 64} (R_ver = X / 2; a refusal there is REPORTED).
# B = 64 documents of pg19-test-c16k, S = 16128, 48 committed tokens per request (24 in the sweep). T_shipped (and the
# equal-memory shipped decode, NOSI_POOL_BLOCKS = C + X + 4 - 64) in child processes. REPORT; MUST = the runs complete.
# The pattern of benchmarks/Efficiency/r2_tick_step.sh: the GPU uuid from HOLDERS/<holder job id>.gpu_uuid; piece 3
# and the packer (gilfree_pack.py / .cpp of f2b981a) pinned by commit, path and blob, copied into the step record and
# loaded by path (the packer is built into the step's JIT directory).
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_tick_sg5_step.sh --nosi <sha> <holder job> <run id> [<cell>] [gilfree|gather]
#   DRYRUN=1 bash benchmarks/Efficiency/r2_tick_sg5_step.sh --nosi <sha> <holder job> <run id> [<cell>] [gilfree|gather]
#        (the transport: gilfree, the default; gather = NOSI's Triton gather on a side stream, r2_tick_sg5.py)
#        (the cell and the transport in either order, at most one of each; any other word: usage, exit 2)
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, SG5_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses SG5_HOLDERS.
#
# CELLS (the lead's two-step p4 plan, 2026-10-21; every other argument, gate and threshold as in the default):
#   (none)   the default: p3's arguments, token for token (--modes exact,fast --backends sc,sb,v1 and the 9-point
#            sweep at 24 steps); T19 pins it against p3's frozen command
#   p4a      the six points (--modes exact,fast x --backends sc,sb,v1) and NO sweep: --sweep is left out, so
#            r2_tick_sg5.py's default "" (no sweep point; sweep_backend None)
#   p4b-<b>  b in sc, sb, v1 (the lead names the first backend that passed p4a): --modes exact --backends <b>, its
#            preflight arms, its EXACT primary point, then the 9-point sweep at 24 steps on <b> if it passed
#   p4sc64   S-C at X 64 (the lead's decision after C's diagnosis of p4a's K5 refusal, note 12o: layer 16's first
#            verify misses needed >= 48 shared extras, X was 32): --modes exact,fast --backends sc --C 96 --X 64
#            --R-ver 16, no sweep, 48 steps, gilfree 8 threads, the superset wait (CudaDeviceOps' default). The
#            harness's equal-memory shipped baseline follows: NOSI_POOL_BLOCKS = C + X + 4 - 64 = 100 (the DRYRUN
#            prints it)
#   p5b2     [b2] B2 as the service (OwnResidency, r2_own2.py: B's array-backed ownership table on the store's device,
#            S-C's device path, UNCHECKED: the harness's default without --b2-checked) at p4sc64's point: --modes
#            exact,fast --backends b2 --C 96 --X 64 --R-ver 16, no sweep, 48 steps, gilfree 8 threads, the superset
#            wait; NOSI_POOL_BLOCKS = 100
#   p6       [batched-rebuild] p5b2's point, EXACT only, the 2 x 2 of docs/R2_SG5_P6_PLAN.md: --modes exact --backends b2
#            --C 96 --X 64 --R-ver 16 --variants torch/per_call,fused/per_call,torch/batched,fused/batched (twin and
#            TickEngine(rebuild=...) per arm; torch/per_call is p5b2's replica); NOSI_POOL_BLOCKS = 100
#   x1       [xpool] the SHARED X POOL (step S6, docs/R2_XPOOL_PLAN.md section 6) at p5b2's point, EXACT only: --modes
#            exact --backends b2 --C 96 --X 64 --R-ver 16 --layout both --X-pool 64 --pool-rule both --arrival-lag 2:
#            per point three arms (stream, pool, stream2; B (C + 4) + 64 positions per layer in the pool), every
#            engine with the fixed arrival lag 2 (note 12k); MUST <label>_XP_replica, _XP_committed, _XP_bitwise; the
#            equal-memory baselines NOSI_POOL_BLOCKS = 100 (stream rule) and 37 (k5 rule: 96 + 4 - 64 + 64 // 64).
#            Its ROOT is NOSI_X1 (nosi-r2-xpool); every other cell keeps NOSI (review m8)
#   stack1a  [integ] THE FULL STACK (docs/R2_INTEG_PLAN.md) at p5b2's point, EXACT, B 64: --stack full --stack-compare
#            --X-pool 64 --arrival-lag 2 --sel-check (arms none, full, none2; MUST _ST_replica / _ST_committed /
#            _ST_bitwise), no --pool-baseline. ROOT NOSI_INTEG (nosi-r2-integ6 since integ6), as every stack1* cell.
#            [integ3] + --require-xregion; per arm MUST _ST_acted_<c> / _ST_off_<c> and _ST_xregion (review M2)
#            [integ5] + --stack-role validate (the diagnostic witnesses ON)
#   stack1a128 [integ5] stack1a at B 128 (--repeat-docs, --log-slots -1, --mem 200G): required before stack1b. Its
#            none arms are per-stream (bigb model at B 128, C 96: pool 51.78 GB, per-stream about 69 GB, one engine
#            at a time); a B 216 comparison does not fit (per-stream about 104 GB), so stack1b is validated at B 64
#            and B 128, NOT at 216. [integ6] SUPERSEDED: X = X_pool 896, R_ver 848 and the base arms only-layout
#            (pool layout; no per-stream store fits at X 896); stack1b is B 192 (see the integ6 block below)
#   [integ5] THE TIME CELLS (stack1t, stack1b), the lead's decisions of 2026-10-21: the HEADLINE arm runs the
#            validated schedule (--arrival-lag 2) with --bare (tick events only), the instrumented round of the same
#            arm gives the category split; --stack-role time turns the diagnostic witnesses OFF (MUST
#            <label>_ST_witness_off) and keeps the pool's invariant checks ON (the production safety refusal, no sync);
#            --free-arm adds one REPORT arm at free arrival, MUST only on the lag-free invariants; --validated-by names
#            stack1a's (and stack1b: stack1a128's) summary.json: the harness refuses (exit 2) unless every MUST gate in
#            each passed, replica and bitwise included
#   stack1p  [integ3] stack1a at the pressure point C 80 (X 64, R_ver 16, X_pool 64): more shared X demand; a
#            DIAGNOSTIC (a CapacityRefusal there is a configuration result); it never stops the stack1 chain
#   stack1n-<c>  [integ3] c in twin, rebuild, resolve, layout, attn, warmup: stack1a with --stack full-no-<c> (layout:
#            no --X-pool, no --require-xregion): the localisation of an stack1a failure (review M1)
#   stack1t  [integ] the full stack's TIMING at p5b2's point, B 64, free timing: --stack full --X-pool 64
#            --shared-unchecked (the selection precondition is checked in stack1a), --pool-baseline (100);
#            [integ3] --stack-bands 33:36:220:450 writes stack_bands.txt (peak GB and T_tick ms against the bands)
#   stack1   [integ3] THE COMBINED CELL (review M3), each sub-cell its own step, claim and record: stack1a; if it
#            fails, the six stack1n-<c> diagnostics, then STOP (exit = stack1a's); [integ5] else stack1a128 (STOP on
#            a nonzero exit); then stack1p (diagnostic), stack1t
#            (STOP on a nonzero exit), then STOP with exit 6 unless stack1t's stack_bands.txt holds the exact line
#            "STACK_BANDS_VERDICT IN"; then stack1b (exit = stack1b's)
#   NOTE (review m7): the x1, bigb*, frg* and the default / p4* / p5b2 / p6 cells keep their own ROOTs; this script's
#            blob passes their self-blob check only when run from that ROOT's own checkout at its pinned commit
#   stack1b  [integ2] the full stack's TIMING at B 216 (bigb's one-engine B_max at C 96, pool, X_pool 64, 76 GB budget;
#            B 264 of integ e869342 is 89 GB in that model: it would not fit): --stack full --X-pool 64
#            --shared-unchecked --repeat-docs --shipped-batch 64 --log-slots -1, --mem=240G (about 116 GB pinned host
#            rows + 17 GiB base; the holder has 480G). The same B as bigb96-216 (the stack OFF there). R2_INTEG_PLAN 5.4
#            [integ6] SUPERSEDED: X 64 refuses at B >= 128 (BIGB_128_DIAG.md); stack1b is B 192, X = X_pool 1344,
#            R_ver 1296 (pinned host rows 102.9 GB + base: --mem=240G)
#   bigb81-128 | bigb81-244 | bigb96-128 | bigb96-216
#            [bigb] LARGE B on the SHARED POOL layout (docs/R2_BIGB_PLAN.md section 4; bigb_core.cell_plan): C 81 or
#            96, B 128 (the stepping stone) or B_max (the largest B whose MODEL peak <= 76 GB: 244 at C 81, 216 at
#            C 96), EXACT, b2, gilfree 8 threads, X 64 / R_ver 16, --layout pool --X-pool 64 --arrival-lag 2, 24
#            steps, --repeat-docs (B > 94 documents), --log-slots -1 (the round's size), --skip-shipped (no shipped
#            child: its pinned rows would need 145-273 GiB of host memory), no --pool-baseline, no sweep, --mem 200G.
#            Its ROOT is NOSI_BIGB (nosi-r2-bigb); run B 128 before B_max at each C.
#   evalfast [evalsplit] EVAL CELL (a), the FAST TIMING HEADLINE (human decision part 3, 2026-10-22;
#            paper-notes/fast-eval-cells.md): the R2 engine end to end at the steady-baseline workload (pg19-test-c16k,
#            S 16128, B 216 = mb1's batch, --repeat-docs, the 84 GB device cap) with natural cuBLAS in draft and verify
#            rows: --modes fast --backends b2 --C 96 --X 1512 --R-ver 1464 (X sized 7 per request, integ6's rule) --layout
#            pool --X-pool 1512 --arrival-lag 2 --bare --variants fused/batched,torch/per_call (the headline first; the
#            second is bigb2's measured path) --log-slots -1 --skip-shipped (the baseline is mb1 on the same GPU, a
#            separate cell of nosi-r2-mbase) --eval-role fast-time --mem-cap-gb 84 --accuracy-record = eval cell (b)'s
#            summary.json of the same run id and holder (r2_evalacc_step.sh cell evalacc; a refusal, exit 2, without
#            it) --eval-bands 55:125:30:70:27:65. 48 steps, --mem 240G. ROOT NOSI_EVAL (nosi-r2-evalsplit)
#   STEP_MIN 120 in every cell above.
#   [fusedres2] the FR-g cells of docs/R2_FUSEDRES_PLAN.md section 7 (p5b2's point: --modes exact,fast --backends b2
#   --C 96 --X 64 --R-ver 16, no sweep, gilfree 8 threads, unchecked), with NOSI = FRNOSI (the fusedres2 worktree)
#   for THESE cells only (every other cell keeps NOSI):
#     frg0   --b2-resolve check, --draft-ticks 4, 48 steps, no pool baseline        GPU equality gate     STEP_MIN 120
#     frg0x  frg0 at the pressure point --C 24 --X 8 --R-ver 4                      coverage arm          STEP_MIN 120
#     frg0p  --b2-resolve check --b2-fused-plant 2, --draft-ticks 4, 8 steps        GPU negative control  STEP_MIN 60
#     frg1   p5b2's arguments + --b2-resolve torch                                  control (always runs) STEP_MIN 60
#     frg2   p5b2's arguments + --b2-resolve fused                                  event-span timing     STEP_MIN 60
#     frg3   --b2-resolve fused --b2-res-profile, 12 draft ticks, 8 steps, no pool  kernel time of res    STEP_MIN 60
#     frg3t  --b2-resolve torch --b2-res-profile, 12 draft ticks, 8 steps, no pool  control (always runs) STEP_MIN 90
#     frg    the seven above in that order, each its own step (this script per cell, its own claim and record);
#            after the first cell that exits non-zero, the remaining non-control cells are SKIPPED (frg1 and frg3t
#            always run); exit = the number of cells that ran and failed (or were refused); one line per cell
#   (p5b2 took 17.6 min; the check arms add the torch reference and a host compare per layer, about x 2-3.)
#
# REGISTERED for stack1 ([integ], 2026-10-21T11:51:27Z, SOURCE ONLY, not run; docs/R2_INTEG_PLAN.md section 5 has
# every band, its derivation and the falsifiers F-ST1..5). Order stack1a, stack1t, stack1b; stop at the first nonzero
# exit. stack1a: the three ST MUST gates and <label>_stack_SKV_ran PASS; R ST_tick 0.15-0.40. stack1t (B 64): twin
# 2-6 ms, REST 30-110 ms, res 20-80 ms, end_tick 122-170 ms, device 45-70 ms, T_tick 0.22-0.45 s (point 0.31 s), r
# 0.84-0.90, [SUPERSEDED by the integ2 block below: peak and stack1b's B 264 bands].
# REVISED in integ2 (2026-10-21T12:12:13Z, before any stack1 run; bigb's one-engine memory model, docs/R2_BIGB_PLAN.md
# section 2): stack1t meta.peak_gb 33-36 GB (one engine 42.73 GB per-stream minus the pool's 8.49 GB; the old "p5b2's
# minus" counted p5b2's two engines); stack1b moves to B 216: peak 74-78 GB (bigb model 75.89 GB at 24 steps; 48
# steps add < 0.1 GB), T_tick 0.6-1.2 s (point 0.85 s), tok_per_s 155-315; every other stack1 band unchanged.
# ADDED in integ3 (2026-10-21T12:53:53Z, before any stack1 run; review-integ2-be3eea9 M1-M3): the fused dispatch gate
# is REPORT in stack cells; MUST per stack arm: _ST_acted_<c> for each component ON (twin_variant fused; a batched
# call with >= 2 publications; fused launches; pool binds and lanes handed between requests; shared calls with the
# fused union only and row pages saved > 0; warm_calls 1) and _ST_off_<c> for each OFF; stack1a / stack1p / stack1n-*
# MUST _ST_xregion: the per-(layer, head) peak lanes exceed B x C (x1 measured 6208 > 6144 at this point, layer 9
# head 0, SCORING_x1.md item 8). stack1p, stack1n-*: no timing claim. The stack1 chain stops on stack1t's bands.
# REVISED in integ5 (2026-10-21T14:28:52Z, before any stack1 run; the lead's timing-boundary decisions,
# paper-notes/stack1-timing-boundaries.md): stack1t's HEADLINE is the lag-2 --bare round (the schedule stack1a
# validated; no witness, no selection check): T_tick 0.20-0.50 s (was 0.22-0.45 s at free timing, instrumented: the
# lag's forced waits are not modelled, so the band is widened, not re-derived), peak 33-36 GB; --stack-bands
# 33:36:200:500. The free-arrival arm and the instrumented category split are REPORT. stack1a128 MUST the same gates
# as stack1a at B 128. stack1b: headline lag 2 --bare, bands as integ2 (peak 74-78 GB, T_tick 0.6-1.2 s, REPORT),
# labelled "validated at B 64 and B 128, not at 216".
# REVISED in integ6 (2026-10-21T14:33:33Z, before any stack1 run; analysis/bigb/BIGB_128_DIAG.md: bigb96-128 at
# X 64 refused in the preflight, K5's X per layer is shared by B x H streams): stack1a128 runs C 96, X = X_pool 896,
# R_ver 848 (the diagnosis's P2), base arms --stack-base only-layout (the pool layout, every other component OFF):
# the per-stream store at X 896 is (96 + 896 + 4) x 2,105,344 B x 128 = 268 GB, so no per-stream reference exists at
# B 128; the pool's equality to the per-stream layout is validated at B 64 (stack1a, x1). MUST: the three ST gates,
# SKV_ran, acted / off, xregion, bookkeeping; peak 53.5-56 GB (bigb model 54.45 + the measured +0.105 offset).
# stack1b moves to B 192, C 96, X = X_pool 1344, R_ver 1296 (X = 7 per request, the diagnosis's sizing; draft share
# 48): bigb model peak(B, X 64) 51.78 + 0.27398 (B - 128) = 69.31 GB, + (1344 - 64) x (2,105,408 + 8,576 B) B =
# 4.80 GB: 74.11 GB <= 76 (B 196: 75.5; B 200: 76.6 > 76; B 216 with X 1512: 81.6). Headline lag 2 --bare: peak
# 73-76 GB, T_tick 0.5-1.2 s (B 64's terms x 3), tok_per_s 140-335, REPORT; labelled "validated at B 64 (per-stream
# reference) and B 128 (pool-only reference), not at 192". Gated on stack1a and stack1a128 (--validated-by).
#
# REGISTERED for evalfast ([evalsplit], 2026-10-22T14:21:48Z, SOURCE ONLY, not run; derivation in
# paper-notes/fast-eval-cells.md section 1.3; INFERENCE from bigb96x1344-198 (EXACT, 2.976 s mean / 1.669 s median
# tick, 4,554 committed in 35 ticks, wall 114.1 s), the B slope 0.0127 s per request and p5b2's FAST - EXACT -4.4 %):
#   EF-STEADY-MED  steady tok/s (median tick) 83, band [55, 125]
#   EF-STEADY-MEAN steady tok/s (mean tick) 46, band [30, 70]
#   EF-EFF         effective tok/s 42, band [27, 65]
#   EF-MEM         meta.peak_gb 81.6 (integ6 model) + 1.33 (bigb2's measured offset) = 82.9 <= 84 (MUST EF_mem_cap)
#   EF-RATIO       steady (mean) / mb1 on the same GPU (mean step) about 0.014, band [0.008, 0.025] (scored by hand)
#   FALSIFIER      steady (mean) > 140 or < 15 tok/s; MODEL check: steady (median) >= the same-GPU baseline would mean
#                  the implementation reached the model's Q (SCORING_bigc.md section 5: 3,703.5 MODEL); predicted NOT
#
# REGISTERED for p4sc64 (2026-10-21, before the run; the scoring rules are in docs/R2_STEP7_TICK_SG5.md section 6):
#   A1'  the S-C gilfree preflight PASSES at X 64; or, if it refuses, the refusal's K5 decision report (C's 7c1387f,
#        _k5_report) names the per-layer demand (first misses needing a shared extra, free extras at the tick start,
#        granted, no slot) and per refused stream base + extras + Policy E + no slot
#   R-T  REPORT: the S-C round tick 5.0-5.9 s and the draft-only tick 3.1-3.2 s (the engine agent's derivation,
#        1d1b956 section 3.2, made at X 32: X 64 changes the store size, so these are not falsifiers here)
#   R-EQ REPORT: the equal-memory shipped baseline at NOSI_POOL_BLOCKS 100
#   P-SG5-1..8 (frozen for p3) apply to both S-C points
#
# REGISTERED for p5b2 ([b2], 2026-10-21, before the run; docs/R2_STEP7_TICK_SG5.md section 7 has the derivation and
# the scoring rules; the reference is p4sc64's record sg5_p4_2190876_p4sc64, S-C at the same point):
#   B-A  the b2 gilfree preflight PASSES (MUST); a refusal or an exception there is REFUTED
#   B-T  the b2 EXACT round tick 1.2-2.6 s (point 2.1 s; S-C 6.61 s); FALSIFIED above 3.3 s or below 0.9 s
#   B-F  REPORT: the b2 FAST round tick within 10 % of b2 EXACT's
#   B-S  residual (T_tick minus the categorized device time) 0.85-1.2 s (S-C 5.49 s): the end_tick term is gone;
#        FALSIFIED above 1.5 s
#   B-R  REPORT: twin, gemm, att, sel, store within 15 % of S-C's; res 0.1-1.0 s (S-C 0.94 s)
#   B-D  REPORT: the b2 draft-only tick 0.5-1.5 s (S-C 4.72 s)
#   B-Q  REPORT: ratio about 0.02 (S-C 0.006)
#   B-K  REPORT: report()["k5"] (the demand peak), report()["b2"], draft_only.sc / residency.sc, meta.peak_gb
#
# REGISTERED for p6 ([batched-rebuild], 2026-10-21 06:12:33Z, registry "SG5 p6 PREDICTIONS"; docs/R2_SG5_P6_PLAN.md
# section 4; the reference is p5b2's record sg5_p5_2190876_p5b2, exact: T_tick 1347, twin 119, REST 775, end_tick 143,
# res 252, device 58 ms; REST = residual - (host backend - res)), per round tick:
#   torch/per_call  twin 105-135 ms, REST 0.70-0.85 s, T_tick 1.20-1.50 s (the control)
#   fused/per_call  twin 2-6 ms, REST 0.70-0.85 s (PF3b), T_tick 1.08-1.39 s
#   torch/batched   twin 105-135 ms, REST 0.12-0.30 s, T_tick 0.62-0.95 s
#   fused/batched   twin 2-6 ms, REST 0.03-0.11 s, T_tick 0.45-0.62 s
#   the batched arms time their rebuild in the TickTimer category "rebuild" (for D's profile): their REST is scored
#   as residual - end_tick + rebuild (the same work as the per-call arms' REST);
#   res, end_tick, device within 15 % of p5b2's in every arm; r, publications, rollbacks within the run-to-run spread;
#   FALSIFIERS: a band missed; a bookkeeping gate failing; fused/per_call's REST below 0.70 s (PF3b refuted)
#
# REGISTERED for x1 ([xpool], SOURCE ONLY, not run; docs/R2_XPOOL_PLAN.md section 6 has the derivation; the reference
# is the same run's per-stream arm):
#   X-A  the b2 gilfree preflight on the POOL layout PASSES (MUST)
#   X-R  SG5_exact_b2_XP_replica PASSES (MUST): stream2 = stream under the same comparison (else X-B is VOID)
#   X-C  SG5_exact_b2_XP_committed PASSES (MUST): the committed tokens equal
#   X-B  SG5_exact_b2_XP_bitwise PASSES (MUST): every tick's rows and logit fingerprints equal
#   X-M  the round's peak GB drops by 7.9-8.9 GB (store 22.10 -> 13.61 GB: 8.49 GB MODEL); FALSIFIED below 7.5 GB
#   X-T  T_tick pool / mean(stream, stream2) in 0.97-1.05; 1.05-1.15 or 0.85-0.97 is scored "outside the
#        prediction, not falsified" (a REPORT to diagnose); FALSIFIED outside 0.85-1.15
#   X-E  REPORT: both equal-memory shipped baselines (P 100 and P 37), xpool.report()'s peak lanes, arrival forced /
#        held per arm
#
# REGISTERED for the bigb cells ([bigb], SOURCE ONLY, not run; docs/R2_BIGB_PLAN.md sections 3-5 have the derivation;
# MODEL = bigb_core.MemModel / tick_bands, fitted on p5b2 B 64; GB = 1e9 B of torch.cuda.max_memory_allocated):
#   BB-A  the b2 gilfree preflight on the pool layout PASSES at every point (MUST); a CapacityRefusal there is a
#         configuration result (the K5 X 64 is shared by B requests), reported with its K5 decision report
#   BB-M  meta.peak_gb (MODEL, one engine; band +- 1.0 GB, FALSIFIED outside +- 2.0 GB):
#           bigb81-128 47.72   bigb81-244 75.82   bigb96-128 51.78   bigb96-216 75.89
#         meta.build_peak_gb (MODEL, +- 1.5 GB): 38.71 at B 128, 52.33 at B 216, 56.66 at B 244
#   BB-1  meta.preflight_peak_gb <= meta.peak_gb + 0.5 GB (no second engine; FALSIFIED otherwise)
#   BB-T  T_tick (round, EXACT, mean per tick) low / center / high, REPORT (MODEL from p5b2's categories):
#           B 128: 1.97 / 2.36 / 2.62 s   B 216: 2.82 / 3.75 / 4.37 s   B 244: 3.09 / 4.19 / 4.93 s
#   BB-D  the draft-only tick's device time td_dev = gemm + att + sel + store (the reconciliation's td proxy):
#           B 128: 77 / 92 / 107 ms       B 216: 99 / 135 / 171 ms      B 244: 106 / 148 / 191 ms
#         its wall per tick: B 128 0.91 / 1.09 / 1.22 s, B 216 1.28 / 1.71 / 2.03 s, B 244 1.40 / 1.91 / 2.28 s
#         (EXACT pads every GEMM to m_fix = 4B rows: the draft tick's gemm is about 41 / 63 / 70 ms there)
#   BB-E  REPORT: report()["k5"] (the demand peak vs X 64), refusals, Policy E evictions, xpool.report()'s peak lanes,
#         r, decisions; the per-point peak_gb_cum
#
# OUTPUT: OUTROOT/sg5_<run>_<holder>[_<cell>]/summary.json (+ the children's shipped_*.json), and the step record
# OUTROOT/sg5_<run>_<holder>[_<cell>].step/ (command with the uuid, its file, the pins and the cell, the copies, stdout
# with one GATE / REPORT line per gate, stderr, exit + accounting, sacct row), both frozen. The job name is
# sg5-<run>-<holder>[-<cell>]; the default's names are p3's.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed uuid file included); 5 an exit-0 step
# whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-sg5-p6                                                   # ROOT (p6; [b2] p5b2: nosi-r2-m2-sg5-b2; p4sc64: -p4d; p4a: -p4c; p3: nosi-r2-m2-sg5)
NOSI_X1=/mnt/beegfs/ojerbi/nosi-r2-xpool                                                 # ROOT of cell x1 only ([xpool], review m8)
NOSI_INTEG=/mnt/beegfs/ojerbi/nosi-r2-integ6                                             # ROOT of the stack1 cells only ([integ], integ6)
NOSI_BIGB=/mnt/beegfs/ojerbi/nosi-r2-bigb                                                # ROOT of the bigb cells only ([bigb])
FRNOSI=/mnt/beegfs/ojerbi/nosi-r2-fusedres2                                              # ROOT of the frg* cells only [fusedres2]
NOSI_EVAL=/mnt/beegfs/ojerbi/nosi-r2-evalsplit                                           # ROOT of the evalfast cell only [evalsplit]
EVALACC_OUT=/mnt/beegfs/ojerbi/r2-runs/evalacc                                           # ROOT (r2_evalacc_step.sh's OUTROOT)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (r2_g0_step.sh)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/sg5                                                   # ROOT
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN (piece 3 v2)
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
GF_SHA=f2b981ac4ea450a1fc9d993a2c5d2c9a10a7db43                                          # PIN (E's packer v2)
GF_PY=benchmarks/Efficiency/gilfree_pack.py                                              # PIN
GF_PY_BLOB=45c56f679b017436810bea8e16fa2719d642ba54                                      # PIN
GF_CPP=benchmarks/Efficiency/gilfree_pack.cpp                                            # PIN
GF_CPP_BLOB=df3e3c8026fb0a48cc2fe2597013c3f20ab1338b                                     # PIN
SWEEP=64:16,64:32,64:64,80:16,80:32,80:64,96:16,96:32,96:64
SPLITS=4
THREADS=8
SELF=benchmarks/Efficiency/r2_tick_sg5_step.sh
STEP_MIN=120
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 SG5 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_tick_sg5_step.sh --nosi <sha> <holder> <run id> [p4a|p4b-sc|p4b-sb|p4b-v1|p4sc64|p5b2|p6|x1|bigb81-128|bigb81-244|bigb96-128|bigb96-216|stack1|stack1a|stack1a128|stack1p|stack1t|stack1b|stack1n-twin|stack1n-rebuild|stack1n-resolve|stack1n-layout|stack1n-attn|stack1n-warmup|frg0|frg0x|frg0p|frg1|frg2|frg3|frg3t|frg|evalfast] [gilfree|gather]" >&2
          exit 2; }
{ [ $# -ge 4 ] && [ $# -le 6 ]; } && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
shift 4
TRANSPORT= CELL=
for w in "$@"; do
    case $w in
        gilfree|gather) [ -z "$TRANSPORT" ] || usage; TRANSPORT=$w ;;
        p4a|p4b-sc|p4b-sb|p4b-v1|p4sc64|p5b2|p6|x1|bigb81-128|bigb81-244|bigb96-128|bigb96-216|stack1a|stack1t|stack1b)
            [ -z "$CELL" ] || usage; CELL=$w ;;
        frg0|frg0x|frg0p|frg1|frg2|frg3|frg3t|frg) [ -z "$CELL" ] || usage; CELL=$w ;;          # [fusedres2]
        stack1|stack1a128|stack1p|stack1n-twin|stack1n-rebuild|stack1n-resolve|stack1n-layout|stack1n-attn|stack1n-warmup)
            [ -z "$CELL" ] || usage; CELL=$w ;;                                                  # [integ3]
        evalfast) [ -z "$CELL" ] || usage; CELL=$w ;;                                            # [evalsplit]
        *) usage ;;
    esac
done
TRANSPORT=${TRANSPORT:-gilfree}
if [ "$CELL" = frg ]; then                                  # [fusedres2] the combined cell: one step per FR-g cell
    nfail=0 stopped=
    for sub in frg0 frg0x frg0p frg1 frg2 frg3 frg3t; do
        case $sub in frg1|frg3t) ctl=1 ;; *) ctl=0 ;; esac
        if [ -n "$stopped" ] && [ "$ctl" = 0 ]; then
            echo "R2 SG5 frg: cell $sub SKIPPED (after the failure of $stopped)"
            continue
        fi
        bash "$0" --nosi "$NSHA" "$JOB" "$RUN" "$sub" "$TRANSPORT"
        src=$?
        echo "R2 SG5 frg: cell $sub exit $src"
        if [ "$src" != 0 ]; then
            nfail=$((nfail + 1))
            [ "$ctl" = 1 ] || [ -n "$stopped" ] || stopped=$sub
        fi
    done
    echo "R2 SG5 frg: done; cells failed or refused $nfail; first MUST failure ${stopped:-none}"
    [ "$nfail" -le 125 ] || nfail=125
    exit "$nfail"
fi
if [ "$CELL" = stack1 ]; then                               # [integ3] review-integ2 M3: the combined stack1 cell
    sub() { bash "$0" --nosi "$NSHA" "$JOB" "$RUN" "$1" "$TRANSPORT"; }
    sub stack1a
    src=$?
    echo "R2 SG5 stack1: cell stack1a exit $src"
    if [ "$src" != 0 ]; then                                # localise (M1): every full-no-<c> diagnostic, then STOP
        for c in twin rebuild resolve layout attn warmup; do
            sub "stack1n-$c"
            echo "R2 SG5 stack1: diagnostic stack1n-$c exit $?"
        done
        echo "R2 SG5 stack1: STOP after stack1a (exit $src); stack1a128, stack1p, stack1t and stack1b not run"
        [ "$src" -le 125 ] || src=125
        exit "$src"
    fi
    sub stack1a128                                          # [integ5] the comparison at B 128, before stack1b
    src=$?
    echo "R2 SG5 stack1: cell stack1a128 exit $src"
    if [ "$src" != 0 ]; then
        echo "R2 SG5 stack1: STOP after stack1a128 (exit $src); stack1p, stack1t and stack1b not run"
        [ "$src" -le 125 ] || src=125
        exit "$src"
    fi
    sub stack1p
    echo "R2 SG5 stack1: diagnostic stack1p exit $? (the pressure point; it does not stop the chain)"
    sub stack1t
    src=$?
    echo "R2 SG5 stack1: cell stack1t exit $src"
    if [ "$src" != 0 ]; then
        echo "R2 SG5 stack1: STOP after stack1t (exit $src); stack1b not run"
        [ "$src" -le 125 ] || src=125
        exit "$src"
    fi
    bands=$OUTROOT/sg5_${RUN}_${JOB}_stack1t/stack_bands.txt
    if ! grep -qx 'STACK_BANDS_VERDICT IN' "$bands"; then    # the falsifier bands of stack1t (peak, tick), not
        echo "R2 SG5 stack1: STOP: $bands has no 'STACK_BANDS_VERDICT IN' line; stack1b not run"   # the exit only
        exit 6
    fi
    echo "R2 SG5 stack1: stack1t inside its bands ($bands)"
    sub stack1b
    src=$?
    echo "R2 SG5 stack1: cell stack1b exit $src; done"
    [ "$src" -le 125 ] || src=125
    exit "$src"
fi
MODES=exact,fast
BACKENDS=sc,sb,v1
SWEEP_RUN=$SWEEP
SUFFIX=
VARIANTS=
XPOOL=                                                      # [xpool] x1 only
STACKARG=()                                                 # [integ] the stack1 cells only (BATCH / MEMG / POOLB below)
PC=96 PX=32 PRV=16                                          # the primary point (C, X, R_ver)
BATCH=64 STEPS=48 MEMG=120G                                 # [bigb] every other cell: as before
POOLB=(--pool-baseline)                                     # [bigb] every other cell: as before ([fusedres2]'s POOLARG
BIGB=                                                       # is the same array: merged into POOLB in integ2)
BIGBARG=()                                                  # [bigb] the bigb cells only
DTICKS=12                                                   # [fusedres2] --draft-ticks (p3's in every cell but frg*);
FRARG=()                                                    # the frg* cells' extra flags in FRARG
EVALARG=()                                                  # [evalsplit] the evalfast cell only
case $CELL in
    "") ;;                                                  # the default: p3's arguments
    p4a) SWEEP_RUN= ; SUFFIX=_p4a ;;                        # the six points; --sweep left out (its default "": none)
    p4b-*) MODES=exact ; BACKENDS=${CELL#p4b-} ; SUFFIX=_$CELL ;;   # one backend, EXACT, then the 9-point sweep
    p4sc64) BACKENDS=sc ; PX=64 ; SWEEP_RUN= ; SUFFIX=_p4sc64 ;;   # S-C only at X 64, both modes, no sweep
    p5b2) BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_p5b2 ;;       # [b2] B2 only at p4sc64's point, unchecked
    p6) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_p6         # p5b2's point, EXACT, the 2 x 2
        VARIANTS=torch/per_call,fused/per_call,torch/batched,fused/batched ;;
    x1) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_x1 ; XPOOL=64 ; NOSI=$NOSI_X1 ;;   # [xpool] p5b2's point
    stack1a) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_stack1a ; NOSI=$NOSI_INTEG ; POOLB=()
             STACKARG=(--stack full --stack-compare --X-pool 64 --arrival-lag 2 --sel-check --require-xregion
                       --stack-role validate) ;;
    stack1a128) MODES=exact ; BACKENDS=b2 ; PX=896 ; PRV=848 ; SWEEP_RUN= ; SUFFIX=_stack1a128 ; NOSI=$NOSI_INTEG
             POOLB=() ; BATCH=128 ; MEMG=200G                        # [integ6] X sized (BIGB_128_DIAG P2: X 64 refused)
             STACKARG=(--stack full --stack-compare --stack-base only-layout --X-pool 896 --arrival-lag 2 --sel-check
                       --require-xregion --stack-role validate --repeat-docs --log-slots -1) ;;
    stack1p) MODES=exact ; BACKENDS=b2 ; PC=80 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_stack1p ; NOSI=$NOSI_INTEG ; POOLB=()
             STACKARG=(--stack full --stack-compare --X-pool 64 --arrival-lag 2 --sel-check --require-xregion
                       --stack-role validate) ;;
    stack1n-layout) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_${CELL//-/_} ; NOSI=$NOSI_INTEG
             POOLB=() ; STACKARG=(--stack full-no-layout --stack-compare --arrival-lag 2 --sel-check
                                  --stack-role validate) ;;                                         # [integ3]
    stack1n-*) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_${CELL//-/_} ; NOSI=$NOSI_INTEG ; POOLB=()
             STACKARG=(--stack "full-no-${CELL#stack1n-}" --stack-compare --X-pool 64 --arrival-lag 2 --sel-check
                       --require-xregion --stack-role validate) ;;                                  # [integ3] M1
    stack1t) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_stack1t ; NOSI=$NOSI_INTEG    # [integ5] the
             STACKARG=(--stack full --X-pool 64 --shared-unchecked --stack-role time --arrival-lag 2 --bare   # headline
                       --free-arm --validated-by "$OUTROOT/sg5_${RUN}_${JOB}_stack1a/summary.json"   # = stack1a's
                       --stack-bands 33:36:200:500) ;;                                              # validated path
    stack1b) MODES=exact ; BACKENDS=b2 ; PX=1344 ; PRV=1296 ; SWEEP_RUN= ; SUFFIX=_stack1b ; NOSI=$NOSI_INTEG
             POOLB=() ; BATCH=192 ; MEMG=240G                         # [integ6] B 192, X sized 7 per request
             STACKARG=(--stack full --X-pool 1344 --shared-unchecked --repeat-docs --shipped-batch 64 --log-slots -1
                       --stack-role time --arrival-lag 2 --bare --free-arm                          # [integ5]
                       --validated-by "$OUTROOT/sg5_${RUN}_${JOB}_stack1a/summary.json,$OUTROOT/sg5_${RUN}_${JOB}_stack1a128/summary.json") ;;
    bigb*) MODES=exact ; BACKENDS=b2 ; PX=64 ; PRV=16 ; SWEEP_RUN= ; SUFFIX=_${CELL//-/_} ; NOSI=$NOSI_BIGB   # [bigb]
        BIGB=$CELL ; PC=${CELL:4:2} ; BATCH=${CELL#bigb??-} ; STEPS=24 ; MEMG=200G ; POOLB=()
        BIGBARG=(--layout pool --X-pool 64 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped) ;;
    frg*) BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_$CELL ; NOSI=$FRNOSI   # [fusedres2] p5b2's point, the FR root
        case $CELL in
            frg0) DTICKS=4 ; POOLB=() ; FRARG=(--b2-resolve check) ; STEP_MIN=120 ;;
            frg0x) PC=24 ; PX=8 ; PRV=4 ; DTICKS=4 ; POOLB=() ; FRARG=(--b2-resolve check) ; STEP_MIN=120 ;;
            frg0p) DTICKS=4 ; STEPS=8 ; POOLB=() ; FRARG=(--b2-resolve check --b2-fused-plant 2) ; STEP_MIN=60 ;;
            frg1) FRARG=(--b2-resolve torch) ; STEP_MIN=60 ;;
            frg2) FRARG=(--b2-resolve fused) ; STEP_MIN=60 ;;
            frg3) STEPS=8 ; POOLB=() ; FRARG=(--b2-resolve fused --b2-res-profile) ; STEP_MIN=60 ;;
            frg3t) STEPS=8 ; POOLB=() ; FRARG=(--b2-resolve torch --b2-res-profile) ; STEP_MIN=90 ;;
        esac ;;
    evalfast) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast   # [evalsplit]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched,torch/per_call
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands 55:125:30:70:27:65) ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))                           # [fusedres2] after the cell's STEP_MIN
VARIANTARG=()                                               # [batched-rebuild] no --variants in every other cell
[ -z "$VARIANTS" ] || VARIANTARG=(--variants "$VARIANTS")
XPOOLARG=()                                                 # [xpool] no layout argument in every other cell
[ -z "$XPOOL" ] || XPOOLARG=(--layout both --X-pool "$XPOOL" --pool-rule both --arrival-lag 2)
EQ_POOL=$((PC + PX + 4 - 64))                               # r2_tick_sg5.py's equal-memory NOSI_POOL_BLOCKS
SWEEPARG=()
[ -z "$SWEEP_RUN" ] || SWEEPARG=(--sweep "$SWEEP_RUN")
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${SG5_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "SG5_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$SG5_HOLDERS
fi
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] || stop "no GPU uuid file $UFILE (the lead writes it for each holder)"
GPU_UUID=$(head -n 1 "$UFILE")
[ "$(wc -l < "$UFILE")" -le 1 ] || stop "$UFILE has more than one line"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold a uuid (no GPU- prefix): '$GPU_UUID'"
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_tick_sg5.py benchmarks/Efficiency/r2_g0bi_gate.py benchmarks/Efficiency/r2_g0_gate.py \
         benchmarks/Efficiency/r2_g0_compare.py nosi/nosi/r2_tick.py nosi/nosi/r2_bi_decode.py \
         nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py \
         nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py \
         nosi/nosi/r2_residency.py nosi/nosi/r2_residency_sb.py nosi/nosi/r2_residency_sc.py \
         nosi/nosi/r2_residency_sparse.py nosi/nosi/r2_own2.py nosi/nosi/r2_own2_ops.py nosi/nosi/r2_twin_fused.py \
         nosi/nosi/r2_transport_gilfree.py nosi/nosi/kstep/ownership.py nosi/nosi/kstep/transfer.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ -z "$XPOOL" ] || git -C "$NOSI" rev-parse -q --verify "$NSHA:nosi/nosi/r2_xpool.py" > /dev/null ||
    stop "nosi/nosi/r2_xpool.py is not tracked at $NSHA (cell x1)"
if [ "${#STACKARG[@]}" -gt 0 ]; then                        # [integ] the stack's modules, tracked at the pinned commit
    for f in nosi/nosi/r2_xpool.py nosi/nosi/r2_fusedres.py nosi/nosi/bi_paged_attention_shared.py \
             benchmarks/Efficiency/bigb_core.py; do
        git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA (cell $CELL)"
    done
fi
if [ -n "$BIGB" ]; then                                     # [bigb] the pool layout and the token-log size
    for f in nosi/nosi/r2_xpool.py benchmarks/Efficiency/bigb_core.py; do
        git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA (cell $BIGB)"
    done
fi
if [ "${#EVALARG[@]}" -gt 0 ]; then                         # [evalsplit] the eval modules, tracked at the pinned commit
    for f in benchmarks/Efficiency/r2_evalacc.py nosi/nosi/r2_xpool.py benchmarks/Efficiency/bigb_core.py \
             nosi/nosi/r2_twin_fused.py; do
        git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA (cell $CELL)"
    done
fi
case $CELL in frg*)                                         # [fusedres2] the fused resolve and its profile parser
    for f in nosi/nosi/r2_fusedres.py benchmarks/Efficiency/r2_fusedres_prof.py; do
        git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
    done ;;
esac
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$GF_SHA:$GF_PY")" = "$GF_PY_BLOB" ] ||
    stop "packer: $GF_PY at $GF_SHA is not the blob $GF_PY_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$GF_SHA:$GF_CPP")" = "$GF_CPP_BLOB" ] ||
    stop "packer: $GF_CPP at $GF_SHA is not the blob $GF_CPP_BLOB"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
OUT=$OUTROOT/sg5_${RUN}_$JOB$SUFFIX
REC=$OUT.step
JIT=$JITBASE/sg5_${RUN}_$JOB$SUFFIX
NAME=sg5-${RUN}-$JOB${SUFFIX//_/-}
ATTN_MOD=$REC/piece3_bi_paged_attention.py
GF_DIR=$REC/gilfree
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_R2_DECODE=1 --env NOSI_MODEL_PATH="$MODEL" --env TICK_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 SG5 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 SG5 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEMG" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" sg5-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_tick_sg5.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --biattn "$ATTN_MOD" --gilfree "$GF_DIR" --gilfree-build "$JIT/gilfree_build"
      --threads "$THREADS" --transport "$TRANSPORT" --batch "$BATCH" --S 16128 --steps "$STEPS" --K 3 --LAMBDA 7 --modes "$MODES"
      --backends "$BACKENDS" --C "$PC" --X "$PX" --R-ver "$PRV" "${SWEEPARG[@]}" --sweep-steps 24 --draft-ticks "$DTICKS" --shipped-steps 24
      "${POOLB[@]}" --preflight-steps 4 "${VARIANTARG[@]}" "${XPOOLARG[@]}" "${BIGBARG[@]}" "${STACKARG[@]}" "${FRARG[@]}"
      "${EVALARG[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 SG5 step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 SG5 step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; packer $GF_SHA:$GF_PY blob $GF_PY_BLOB, $GF_CPP blob $GF_CPP_BLOB -> $GF_DIR; transport $TRANSPORT; cell ${CELL:-default}"
    echo "R2 SG5 step: primary point C $PC X $PX R_ver $PRV; equal-memory shipped baseline NOSI_POOL_BLOCKS = C + X + 4 - 64 = $EQ_POOL"
    [ -z "$XPOOL" ] || echo "R2 SG5 step: [xpool] root $NOSI, layout both (stream, pool, stream2), X_pool $XPOOL, arrival lag 2; k5 rule NOSI_POOL_BLOCKS = C + 4 - 64 + X_pool / 64 = $((PC + 4 - 64 + XPOOL / 64))"
    [ "${#STACKARG[@]}" -eq 0 ] || echo "R2 SG5 step: [integ] root $NOSI, batch $BATCH, step memory $MEMG, stack arguments ${STACKARG[*]}"
    [ -z "$BIGB" ] || echo "R2 SG5 step: [bigb] root $NOSI, cell $BIGB: layout pool, C $PC, B $BATCH, X_pool 64, arrival lag 2, $STEPS steps, token log sized (-1), no shipped child, mem $MEMG"
    [ "${#EVALARG[@]}" -eq 0 ] || echo "R2 SG5 step: [evalsplit] root $NOSI, batch $BATCH, step memory $MEMG, variants $VARIANTS, eval arguments ${EVALARG[*]}"
    echo "R2 SG5 step: nosi root $NOSI; STEP_MIN $STEP_MIN; MIN_LEFT $MIN_LEFT s"                   # [fusedres2]
    printf '%q ' "${ARGV[@]}"
    echo
    exit 0
fi
line=$(timeout 60 squeue -j "$JOB" -h -o '%u|%T|%L') || stop "squeue failed"
su=${line%%|*} rest=${line#*|}
ss=${rest%%|*} sl=${rest#*|}
[ "$su|$ss" = "ojerbi|RUNNING" ] || stop "holder $JOB is '$line'"
left=0
case $sl in
    *-*:*:*) left=$(( ${sl%%-*} * 86400 + 10#$(echo "${sl#*-}" | cut -d: -f1) * 3600 + 10#$(echo "$sl" | cut -d: -f2) * 60 )) ;;
    *:*:*) left=$(( 10#$(echo "$sl" | cut -d: -f1) * 3600 + 10#$(echo "$sl" | cut -d: -f2) * 60 )) ;;
esac
[ "$left" -ge "$MIN_LEFT" ] || stop "holder $JOB has $sl left (< $MIN_LEFT s)"
nosteps() {
    steps=$(timeout 60 squeue -s -j "$JOB" -h -o '%i') || return 1
    for s in $steps; do case $s in *.batch|*.extern) ;; *) echo "$s"; return 1 ;; esac; done
}
busy=$(nosteps) || stop "a step is running in $JOB (or squeue -s failed): '$busy'"
mkdir -p "$OUTROOT" || stop "cannot create $OUTROOT"
mkdir "$OUTROOT/sg5_active" 2>/dev/null || stop "$OUTROOT/sg5_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (sg5_active stays)"
mkdir "$REC" && mkdir "$GF_DIR" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (sg5_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (sg5_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (sg5_active stays)"
git -C "$NOSI" show "$GF_PY_BLOB" > "$GF_DIR/gilfree_pack.py" && [ "$(git hash-object -- "$GF_DIR/gilfree_pack.py")" = "$GF_PY_BLOB" ] ||
    stop "cannot copy the packer's blob $GF_PY_BLOB (sg5_active stays)"
git -C "$NOSI" show "$GF_CPP_BLOB" > "$GF_DIR/gilfree_pack.cpp" && [ "$(git hash-object -- "$GF_DIR/gilfree_pack.cpp")" = "$GF_CPP_BLOB" ] ||
    stop "cannot copy the packer's blob $GF_CPP_BLOB (sg5_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (sg5_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (sg5_active stays)"
printf 'biattn=%s:%s blob=%s packer=%s py=%s cpp=%s threads=%s transport=%s sweep=%s splits=%s cell=%s modes=%s backends=%s C=%s X=%s R_ver=%s eq_pool=%s variants=%s xpool=%s batch=%s steps=%s mem=%s stack=%s\n' \
    "$BIATTN_SHA" "$BIATTN_PATH" "$BIATTN_BLOB" "$GF_SHA" "$GF_PY_BLOB" "$GF_CPP_BLOB" "$THREADS" "$TRANSPORT" \
    "${SWEEP_RUN:-none}" "$SPLITS" "${CELL:-default}" "$MODES" "$BACKENDS" "$PC" "$PX" "$PRV" "$EQ_POOL" "${VARIANTS:-none}" \
    "${XPOOL:-none}" "$BATCH" "$STEPS" "$MEMG" "${STACKARG[*]:-none}" >> "$REC/command.txt" ||
    stop "cannot write $REC (sg5_active stays)"
printf 'nosi_root=%s step_min=%s steps=%s draft_ticks=%s frarg=%s evalarg=%s\n' "$NOSI" "$STEP_MIN" "$STEPS" \
    "$DTICKS" "${FRARG[*]:-none}" "${EVALARG[*]:-none}" >> "$REC/command.txt" ||
    stop "cannot write $REC (sg5_active stays)"                                                     # [fusedres2]
"${ARGV[@]}" < /dev/null > "$REC/stdout.txt" 2> "$REC/stderr.txt"
rc=$?
ok=1
echo "exit=$rc utc_end=$(date -u +%FT%TZ)" > "$REC/exit.txt" || ok=0
sleep 10
timeout 60 env TZ=UTC sacct -j "$JOB" -n -P -o JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS < /dev/null |
    grep -F "|$NAME|" > "$REC/sacct.txt" || ok=0
acct="no single sacct row"
if [ "$(wc -l < "$REC/sacct.txt")" = 1 ]; then
    IFS='|' read -r _ _ sstate scode _ < "$REC/sacct.txt"
    case "$rc:$sstate:$scode" in
        "0:COMPLETED:0:0") acct=consistent ;;
        0:*) acct="inconsistent: exit 0, sacct $sstate $scode" ;;
        *) acct="exit $rc, sacct $sstate $scode" ;;
    esac
fi
echo "accounting=$acct" >> "$REC/exit.txt" || ok=0
[ "$rc" != 0 ] || [ "$acct" = consistent ] || ok=0
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/sg5_active" || ok=0
fi
echo "R2 SG5 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
