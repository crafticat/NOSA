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
#   In DRYRUN only, SG5_CLAIM_PROBE=<dir>: after the uuid file checks, mkdir <dir>/<the claim's name> (the real claim's
#        mkdir, under <dir> and not OUTROOT), then exit 0 when it took the claim, 1 when the claim is held; no other check
#        or write (the CPU test uses it); a real run refuses SG5_CLAIM_PROBE.
#
# CLAIM ([sg5par], 2026-10-22): ONE CLAIM PER HOLDER, OUTROOT/sg5_active_<holder job id>. Two steps in the same holder
#   refuse each other; steps in different holders (different GPUs) run at the same time. Every path that a step writes
#   holds the run id AND the holder job (OUT, REC, JIT and its TRITON / TORCH_EXTENSIONS / FLASHINFER / TVM_FFI caches
#   and TMPDIR, the job name, the sacct row); every record that a cell reads (--validated-by, --accuracy-record, the
#   stack_bands.txt of stack1) is the SAME holder's. No cell reads another holder's record, so no cross-holder claim
#   is necessary. The global claim OUTROOT/sg5_active of the older ROOTs (p6, x1, integ6, bigb, fusedres2) is not read:
#   a step of an older ROOT and a step of this script in the SAME holder are kept apart by the holder's dispatcher (one
#   job at a time) and by nosteps() before and after the claim.
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
#            _ST_bitwise), no --pool-baseline. ROOT NOSI_INTEG (nosi-r2-stack2 since stack2), as every stack1* cell.
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
#            [hostfix] ROOT NOSI_EVAL is nosi-r2-hostfix now (its defaults keep evalfast's path; the stack2 fixes are in)
#   evalfast2  [hostfix] evalfast with the HOST FIXES (analysis/evalfast/DIAG_tick_host.md section 4, fixes 0-2) and
#            the FUSED resolver: evalfast's point, pool layout, cap, accuracy record and steps, but --variants
#            fused/batched only, --b2-resolve fused, --host-profile --gc-freeze --lean-time, its own --eval-bands
#   evalfast2t [hostfix] evalfast2 with --b2-resolve torch (evalfast's resolver): the host fixes alone, so the resolver
#            change and the host fixes are scored apart (evalfast -> evalfast2t: host fixes; evalfast2t -> evalfast2:
#            the resolver)
#   evalfast2c [hostfix] the PRECONDITION of evalfast2: the fused resolver EQUALS the torch one at this scale (C + X 1608,
#            NP 2048, B 216): --b2-resolve check (fused, and the torch path compared field by field at every layer:
#            MUST <label>_draft_only_fused_equal / _round_fused_equal), --modes fast (the timing cells' decision
#            inputs), the same point and layout, --variants fused/batched, 8 steps, --draft-ticks 4, no --bare, no
#            eval role; run it BEFORE evalfast2
#            The three: ROOT NOSI_EVAL, --mem 240G, STEP_MIN 120. Not in them (the lead's flag list): --wait-index and
#            --b2-fewer-reads (items 3 and 5, opt-in, CPU-tested; one word each to add)
#   evalfast2rp / evalfast2rl [resfix] evalfast2wx + --b2-res-profile (the device floor of res; STEP_MIN 150) / +
#            --res-lean (the lean torch resolve, tests_cpu/test_r2_resfix.py); REGISTERED block below
#   evalfast2rd / evalfast2rdp [resdev] evalfast2rl + --res-dev (the lean resolve's device hot spots with equal integer
#            ops, nosi/r2_resdev.py, tests_cpu/test_r2_resdev.py; FAST) / + --b2-res-profile too (the device time of
#            the --res-dev draft-only tick; STEP_MIN 150); REGISTERED block below
#   evalfast2c3 / evalfast2f3 / evalfast2f3_192 [fr3] the fused v3 kernel (--fused-kernel v3): CHECK at B 216 / FAST
#            at B 216 / FAST at B 192; REGISTERED block below
#   evalfast2f3p / evalfast2c3l / evalfast2f3l [flean] evalfast2f3 + --b2-res-profile (STEP_MIN 150) / evalfast2c3 +
#            --res-lean --res-dev (the CHECK gate) / evalfast2f3 + --res-lean --res-dev (the fused path with the lean
#            host checks, binds and bias fill; tests_cpu/test_r2_flean.py); REGISTERED block below
#   evalfast2ia [admitfill] evalfast2rl + --inflight-array (the poll admits the due draft issues as whole parts: no sort
#            of the transfers in flight, one done / force per issue; tests_cpu/test_r2_admitfill.py; FAST). It is fix 2
#            of analysis/evalfast/DIAG_effective_gap.md section 4 (fix 1, the fill at admission, is not exact here:
#            the first resident set is the tick-0 selection); REGISTERED block below
#   evalfast2f3lia / evalfast2c3lia [f3lia] evalfast2f3l + --inflight-array (FAST timing) / evalfast2c3l +
#            --inflight-array (the CHECK gate, 0 diffs); evalfast2f3l and evalfast2ia are the same-job controls
#            (tests_cpu/test_r2_f3lia.py); REGISTERED block below
#   evalacc216m / evalacc216 [acc216] THE ACCURACY / EXACTNESS PROOF AT B 216 ON THE TIMED PATH
#            (analysis/evalfast/STATUS_accuracy_B216.md section 5): evalfast2rd's command with --modes fast,bi_c_strict
#            (FAST first, then EXACT = G-c strict, on the same prefill and harness), WITHOUT --eval-role fast-time,
#            --accuracy-record, --mem-cap-gb 84, --eval-bands and --bare (a proof cell, not a timing cell), PLUS
#            --store-committed-tokens --serial-rows 8 --gold-workload GOLD (r2_tick_sg5.py acc_gates). evalacc216 also
#            reads --link-record = the evalfast2rd summary.json of the same run id and holder (REPORT only).
#            evalacc216m: --steps 8, no link: the inspected miniature (memory and gate wiring); run it FIRST.
#            ROOT NOSI_EVAL. REGISTERED block below
#   evalacc216f3lia [f3lia] evalacc216 on the combined path: --b2-resolve fused --fused-kernel v3 --res-lean --res-dev
#            --inflight-array in place of --b2-resolve torch; --link-record = the evalfast2f3lia summary.json of the
#            same run id and holder (REPORT only); REGISTERED block below (the F3LIA block)
#   evalfast2eb [endbatch] evalfast2rl + --end-batch (the requests that finish in a tick end with ONE call; one
#            end_owners, one host read, a per-request index of the queued keys; tests_cpu/test_r2_endbatch.py);
#            REGISTERED block below
#   evalfast2ab / evalfast2c3ab [applyev] evalfast2f3l + --apply-batched (timing; no --eval-bands until the lead
#            registers them) / evalfast2c3l + --apply-batched (the CHECK gate); evalfast2f3l is the same-job control
#            (tests_cpu/test_r2_applyev.py); REGISTERED block below
#   evalfast2all / evalfast2c3all [f3all] evalfast2f3lia + --end-batch --apply-batched (FAST timing; --eval-bands
#            "$ALL_BANDS" = 440:500:345:405:295:345, registered in the F3ALL block) / evalfast2c3lia +
#            --end-batch --apply-batched (the CHECK gate, 0 diffs, plus MUST _AB_acted); evalfast2f3lia is the same-job
#            control (tests_cpu/test_r2_f3all.py); REGISTERED block below (the F3ALL block)
#   evalacc216all [f3all] evalacc216f3lia + --end-batch --apply-batched; --link-record = the evalfast2all summary.json
#            of the same run id and holder (REPORT only); REGISTERED block below (the F3ALL block)
#   evalfast2f3d / evalfast2c3d [f3ld] evalfast2f3lia / evalfast2c3lia + --worker cpp --b2-issue desc (b2-desc 47db0c0
#            ported: ONE E issue_desc job per tick; packer v3 of --nosi, GF3 below)
#   evalfast2f3t / evalfast2c3t [f3ld] evalfast2f3d / evalfast2c3d + --no-transfer-mirror, WITHOUT --inflight-array (the
#            table poll replaces the array poll); evalfast2c3t adds --lean-time (the flag needs it)
#   evalfast2f3c / evalfast2c3c [f3ldx] evalfast2f3lia / evalfast2c3lia + --worker cpp --b2-issue object (the C++
#            feeder alone: the object path's per-layer issues on E's C++ worker, which already writes the pool's lanes,
#            m' = Transfer.phys); its share is measured on its own, against evalfast2f3lia
#            The six: ROOT NOSI_EVAL (nosi-r2-f3ldx), the same-job control evalfast2f3lia
#            (tests_cpu/test_r2_b2_desc.py, tests_cpu/test_r2_f3ld.py, tests_cpu/test_r2_f3ldx.py); REGISTERED block
#            below (the F3LD block). [f3ldx] NOT BLOCKED: E's issue_desc writes the pool's lanes (packer v3 DESC 2: the
#            descriptor's lane row, by the object path's rule), so F3LD_BLOCKED is gone and r2_tick_sg5.py takes
#            --b2-issue desc with --layout pool
#   evalfast2tall / evalfast2c3tall / evalacc216tall [f3tall] the f3t flag set (--worker cpp --b2-issue desc
#            --no-transfer-mirror, no --inflight-array; packer v3 of --nosi, GF3) + --end-batch --apply-batched:
#            evalfast2tall = evalfast2f3t + the two flags (FAST timing; --eval-bands "$TALL_BANDS", TBD until the lead
#            registers them: refused outside DRYRUN); evalfast2c3tall = evalfast2c3t + the two flags (the CHECK gate,
#            0 diffs, plus MUST _AB_acted and the desc path acted); evalacc216tall = evalacc216 on the tall flag set,
#            --link-record = the evalfast2tall summary.json of the same run id and holder (REPORT only). Same-job
#            controls: evalfast2f3t (f3ldx), evalfast2all (f3all), evalfast2f3lia (both bases). ROOT NOSI_EVAL
#            (nosi-r2-f3tall = f3all 3a7f51a merged with f3ldx 3769b64); tests_cpu/test_r2_f3tall.py; REGISTERED block
#            below (the TALL block)
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
# REVISED in stack2 (2026-10-22T14:37:32Z, before any stack2 run; analysis/stack1/SCORING_stack1.md; ROOT
# NOSI_INTEG = nosi-r2-stack2). Code changes: (1) the fused resolver's Policy E key holds the slot in max(9, log2 NP)
# bits (MAX_NPOS 4096; 23b8a38's key bit for bit at NP <= 512); (2) make_engine closes the transport when the build
# fails after it (the gilfree feeder thread held the first engine: stack1a128's 82.97 GB preflight); (3) no Triton
# compile inside a timed tick: the fused kernels' row-list / selection pointers are not specialized, LZ and SCAN's
# BLOCK are constants, and warm_up (fused resolve) / SharedKV.warm_up (rows_max 2 .. 4) compile every variant at the
# engine's first resolve (an untimed first tick). The kernels changed, so stack1a reruns too (bitwise equality is per
# build). The chain and every 23b8a38 MUST gate are unchanged. PREDICTIONS (run id stk2):
#   stack1a     MUST: the 36 gates PASS as at 23b8a38. REPORT, registered: (a) the fused report's "compiled" counts
#               after the stack round EQUAL those after its draft-only phase (no new variant inside the round;
#               stack1a had 15 -> 32); (b) res max per tick < 2 s (was 39.06 s) and the stack arm's T_tick max < 3 s
#               (was 42.41 s); (c) R_ST_tick 0.30-0.55 (the compile-free stack median 433.6 ms x the none arms'
#               mean / median 1.17-1.30, over mean(none) about 1274 ms; the shared attention's median 71.9 ms vs 12.6 ms
#               is NOT a compile and is not changed here): the 0.15-0.40 band of 5.4c stays the REPORT band, and a
#               miss above it is EXPECTED (the falsifier reading of R_ST_tick is unchanged: none).
#   stack1a128  MUST: SG5_preflight_gilfree_b2 PASS (C + X 992 <= 4096) and every stack1a gate at B 128 (5.4e).
#               meta.preflight_peak_gb <= meta.peak_gb + 0.5 (one engine at a time; was 82.97 with no round); peak
#               53.5-56 GB (5.4e, unchanged). REPORT: res per tick at NP 1024 (no band: first fused run above NP 512).
#   stack1p, stack1t, stack1b: as registered in integ5 / integ6 (stack1b: peak 73-76 GB; bigb2 measured +1.33 GB
#               over the model at X 1344, so about 75.4 GB, REPORT).
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
# REGISTERED for evalfast2t / evalfast2 / evalfast2c ([hostfix], STAMP 2026-10-22T21:23:24Z,
# SOURCE ONLY, not run). Derivation: analysis/evalfast/DIAG_tick_host.md sections 1.3, 1.4, 4 and 4.1 (INFERRED ranks,
# not measured one by one) on the MEASURED evalfast record (sg5_ev1_2192143_evalfast, nova05: fused/batched TWIN with
# the TORCH resolver; bare tick median 771.62 / mean 946.75 ms, 215.68 / 175.79 / 137.26 tok/s steady median / steady
# mean / effective; 10,152 committed in 61 ticks = 166.43 per tick; res span 395.8 ms median, residual 379.8 ms
# median; mean - median 175.1 ms) and the stack1b fused res median 1213.1 ms at NP 2048, B 192 (SCORING_stack2.md,
# nova07; cross-GPU, cross-arm: INFERRED grade). tok/s = 166.43 / tick s. Bands are wide on purpose: every saving below
# is INFERRED.
#   HF-T-MED   evalfast2t steady tok/s (median tick): point 250, band [220, 300]. Tick median 771.6 - (lean: looks 40-80
#              on a verify tick, metrics dicts 20-50, memory samples / witness / draft_refused 5-15) - (GC young-
#              generation passes 5-20) = 610-700 ms, point 665 ms
#   HF-T-MEAN  evalfast2t steady tok/s (mean tick): point 230, band [195, 290] (the mean also loses the GC spikes:
#              100-175 ms on the mean if the 2-6 s ticks are collections)
#   HF-T-EFF   evalfast2t effective tok/s: point 175, band [140, 220] (61 x the mean tick + 14-16 s outside the ticks)
#   HF-T-GC    evalfast2t bare tick mean / median <= 1.12 (evalfast 1.227) with gc_freeze.bare.collections_in_phase
#              all 0. FALSIFIER: >= 1.20 with no collection in the phase means the spikes are NOT cyclic GC
#   HF-T-HOST  REPORT (bare_host_profile, median per tick): h_hs_looks < 2 ms (lean); h_hs_takes 60-200, h_poll_arrivals
#              50-150, h_superset_wait 20-100, h_et_issue 15-40, h_b2_ops 20-60 ms; the feeder's Python time 30-90 ms
#              per tick. Reading rule: if h_hs_takes + h_poll_arrivals + the feeder < 100 ms median, fix 4 (array-backed
#              transfers) is NOT the next lever
#   HF-T-FALS  FALSIFIER of the host fixes: evalfast2t steady (median) < 216 (no gain over evalfast's 215.68) or > 340
#   HF-F-MED   evalfast2 steady tok/s (median tick): point 105, band [75, 150]. Tick median = evalfast2t's 665 - the
#              torch res span 396 + the fused res 1200-1400 (stack1b's 1213 at NP 2048 with 384 programs; 432 here,
#              both 4 waves of 108 SMs) = 1470-1670 ms
#   HF-F-MEAN  evalfast2 steady tok/s (mean tick): point 95, band [65, 145]
#   HF-F-EFF   evalfast2 effective tok/s: point 85, band [55, 130]
#   HF-F-FALS  FALSIFIER of the "fused is about 3x slower per unit of work at NP 2048" reading (DIAG 2.2): evalfast2
#              steady (median) ABOVE evalfast2t's (the fused resolver faster than torch at this scale)
#   HF-C       evalfast2c MUST: <label>_draft_only_fused_equal and <label>_round_fused_equal PASS (0 differences, every
#              layer compared), _fused_err PASS. FALSIFIER: a difference: fused is NOT equal to torch at C + X 1608, and
#              evalfast2's fused timing has no equality precondition (score it VOID, keep evalfast2t)
#   HF-MEM     evalfast2t meta.peak_gb 83.2 +- 0.5 <= 84 (MUST EF_mem_cap; evalfast 83.19: the flags change no device
#              allocation); gc_freeze.*.cuda_freed_gb_by_collect <= 0.1 (no CUDA tensor kept in a cycle). evalfast2:
#              <= 84 (MUST), no point (the fused resolver's own temporaries). FALSIFIER: EF_mem_cap fails
#   evalfast2 / evalfast2t eval bands (R_EF_bands, REPORT): the HF-*-MED / -MEAN / -EFF bands above
#
# REGISTERED for evalfast2c192 / evalfast2f192 / evalfast2t192 ([b192], STAMP 2026-10-22T22:59:44Z, before any run).
# WHY: at B 216 the fused path does not fit the A100. evalfast2c hf1/hf2: Triton CUDA OOM at the first _fr_v2 launch
# (driver local memory); evalfast2 hf1/hf2: torch OOM in the preflight after that launch (75.66 GiB torch + 3.3 GiB
# non-torch of 79.25 GiB; DIAG_fused_check_oom.md). stack1b launched the same _fr_v2 binary at B 192 (peak 75.45 GB).
# So the RESOLVER comparison moves to B 192 (C 96 + X 1344, X = 7 per request, R_ver 1296, NP 2048), as a matched pair.
# The B 216 headline stays evalfast2t's (torch). The B 192 cells are NOT a headline: mb1 (the baseline) is at B 216.
#   B1-C     evalfast2c192 MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS. FALSIFIER: a
#            difference: then evalfast2f192 is VOID
#   B1-T-MED evalfast2t192 steady tok/s (median tick): point 255, band [220, 300] (evalfast2t's tick 640 ms scaled to
#            B 192: host + other about 220 ms, torch res about 350 ms -> about 570 ms; 166 x 192/216 = 148 tok per tick)
#   B1-T-MEAN [180, 290]; B1-T-EFF [140, 230]
#   B1-F-MED evalfast2f192 steady tok/s (median tick): point 100, band [75, 140] (tick about 220 + stack1b's fused res
#            median 1215 = about 1435 ms; 148 / 1.435 = 103). B1-F-MEAN [60, 135]; B1-F-EFF [50, 120]
#   B1-F-FALS FALSIFIER of "fused is slower than torch at NP 2048": evalfast2f192 median ABOVE evalfast2t192's
#   B1-MEM   both <= 84 GB (MUST EF_mem_cap). REPORT: fused warm_mem_gb (d2b6627 oomfix: free GB before / after the
#            first warm launch; the difference is the _fr_v2 local-memory pool)
#
# REGISTERED for evalfast2w ([hostw], STAMP 2026-10-23T01:03:07Z, before any run): evalfast2t + --wait-index (DIAG
# section 4 fix 3: 40-100 ms per tick, out of res) + --b2-fewer-reads (fix 5: 5-20 ms). Base: evalfast2t hf2 (nova05)
# median tick 640.09 ms / 260.01 tok/s, mean 785.58 / 211.85, effective 187.89; 166.4 committed tokens per tick.
#   HW-MED   steady tok/s (median tick): point 295, band [270, 330] (tick 640 - 45..120 = 520-595 ms)
#   HW-MEAN  [215, 270]; HW-EFF [185, 235]
#   HW-FALS  FALSIFIER: steady (median) below 262 (no gain over evalfast2t's 260.01 beyond 1 % noise)
#   HW-RES   REPORT: the instrumented res share falls (evalfast2t 43.0 %); the gain is in res, not host residual
#   HW-MEM   peak_gb <= 84 (MUST EF_mem_cap); the flags change no device allocation
#
# REGISTERED for evalfast2wi / evalfast2fr ([hostw2], STAMP 2026-10-23T02:01:01Z, before any run). WHY: evalfast2w
# (both flags) ran 210.87 (nova05 hw1) / 211.75 (nova07 hw2) tok/s vs evalfast2t 260.01 / 261.90: HW-FALS fired on both
# nodes. DIAG_evalfast2w.md: +150 ms on the median tick, all in end_tick (h_hs_takes +64, h_hs_merges +70); rank 1 cause
# --b2-fewer-reads under --lean-time (one .tolist() of the FULL records, r2_own2.py:979-997, where lean read only the
# masked entries); --wait-index cut h_superset_wait about 85 -> 6 ms on ticks 6-31 (the mean), not the median.
#   WI-MED   evalfast2wi (evalfast2t + --wait-index): steady median point 264, band [254, 273]; WI-MEAN [200, 230];
#            WI-EFF [175, 200]. REPORT: h_hs_merges 2-8 ms, h_superset_wait about 6 ms on ticks 6-31.
#            FALSIFIER: h_hs_merges median > 15 ms or h_hs_takes median > 40 ms (the index costs end_tick time)
#   FR-MED   evalfast2fr (evalfast2t + --b2-fewer-reads): steady median point 209, band [195, 225]; FR-MEAN [165, 195];
#            FR-EFF [148, 172]. REPORT: h_hs_merges 65-82 ms. FALSIFIER: h_hs_merges median < 20 ms or the bare median
#            tick < 700 ms (then fewer-reads is not the cause)
#   W2-MEM   both peak_gb <= 84 (MUST EF_mem_cap)
#
# REGISTERED for evalfast2fx / evalfast2wx ([frfix], STAMP 2026-10-23T03:52:16Z, before any run). WHY: the fix in r2_own2.py makes
# --b2-fewer-reads select on the device: it removes the .tolist() of the FULL records and the Python filter. evalfast2fr
# ran 216.8 tok/s vs the control 259.7-260.1 (hw3/hw4), at 150 ms/tick extra, all in h_hs_takes / h_hs_merges.
#   FX-MED   evalfast2fx (evalfast2t + --b2-fewer-reads, the device-selected fix): steady median point 262, band [250, 275];
#            FX-MEAN [200, 225]; FX-EFF [178, 198].
#   FX-FALS  FALSIFIED if the evalfast2fx median < 250, or the median tick's h_hs_merges > 15 ms, or its h_hs_takes
#            exceeds the control's by > 20 ms
#   WX-MED   evalfast2wx (evalfast2t + --wait-index --b2-fewer-reads): steady median point 268, band [255, 282];
#            WX-MEAN [205, 235]; WX-EFF [180, 205]. --wait-index alone gave 263.8 / 267.7.
#   WX-FALS  FALSIFIED if the evalfast2wx median < the evalfast2wi median of the same job minus 5 tok/s (if both run in
#            one job), else if it is < 255
#   FX-MEM   both peak_gb <= 84 (MUST EF_mem_cap; same as W2-MEM)
#
# REGISTERED for evalfast2rp / evalfast2rl ([resfix], STAMP 2026-10-23T10:08:04Z, before any run). WHY:
# analysis/evalfast/DIAG_torch_res.md: the torch resolver's host span (h_pages about 385 ms of the 634.9 ms median tick
# in evalfast2wx) is host-bound (about 800 small ops per mixed layer, 8 hidden syncs per mixed layer, per-row Python
# per layer). ROOT NOSI_EVAL = nosi-r2-resfix. Both cells are evalfast2wx (--wait-index --b2-fewer-reads) plus ONE flag.
#   RP       evalfast2rp (evalfast2wx + --b2-res-profile: torch.profiler over the draft-only phase, r2_fusedres_prof):
#            the DEVICE floor of res at B 216. REPORT only, bands wide: about 13,900 device events per draft-only tick
#            (DIAG 2.2: 435 per layer x 32) and 60-120 ms kernel time per draft-only tick inside 't.res'
#            (draft_only_res_profile.res). Its spans are NOT timings; its eval bands are REPORT only (200:340:150:300:
#            120:260, the round is not profiled). STEP_MIN 150 (the trace export).
#   RL-MED   evalfast2rl (evalfast2wx + --res-lean): steady tok/s (median tick) point 300, band [280, 330] (DIAG 4.1
#            estimate 308 with the batched mirror, which --res-lean does NOT do: its memory, about 1.2 GB of (L, B, H,
#            NPOS) mirror fields, does not fit the 84 GB cap at peak 83.2; the rank sorts once per tick replace part of it)
#   RL-MEAN  [225, 265]; RL-EFF [195, 225] (evalfast2wx's mean / effective with 60-95 ms less per tick)
#   RL-FALS  FALSIFIED if the evalfast2rl median < 275, or its bare_host_profile h_pages median is not lower than
#            evalfast2wx's same-job value by >= 40 ms (if both run in one job; else than wx1's 384.4 ms)
#   RL-MEM   evalfast2rl and evalfast2rp peak_gb <= 84 (MUST EF_mem_cap; the lean cache holds about 50 MB per tick)
#
# REGISTERED for evalfast2rd / evalfast2rdp ([resdev], STAMP 2026-10-23T13:17:36Z, before any run). WHY:
# analysis/evalfast/SCORING_resfix.md 2.2-2.3 / 3.4: after --res-lean the draft-only res span (143-144 ms) is about
# 92 % device work (132 ms kernel time per draft-only tick, 17,000 events; profiled on the OLD path). The device hot spots:
# the 2-row cumsum in XPool.bind (25.9 ms), a long scatter (11.6 ms), the sorts (about 16 ms), the 1-D cummax (7.4 ms),
# fills and copies (about 22 ms). --res-dev (needs --res-lean) runs r2_resdev's equal integer ops there: one flat scan,
# group starts without cummax, compactions for the free lists (no sort), int32 radix keys for the stable sorts, own
# dummy targets for every masked scatter / index_put_ / index_add_, the two mirror rank sorts only when read (no decision
# op reads them), bool sums without casts. ROOT NOSI_EVAL = nosi-r2-graph. Both cells are evalfast2rl plus --res-dev.
#   RD-MED   evalfast2rd: steady tok/s (median tick) point 360, band [335, 420] (rl 325-328; DIAG / SCORING 3.3:
#            about 45 ms less per median tick)
#   RD-MEAN  [260, 300]; RD-EFF [222, 255] (DERIVED by the implementer, not given by the lead: evalfast2rl's mean 256 /
#            effective 219 with about 45 ms less per mean tick of about 650 ms)
#   RD-FALS  FALSIFIED if the evalfast2rd median < 330, or the evalfast2rdp profiled draft-only res kernel time
#            (draft_only_res_profile.res, ms per tick) is not lower than 132 ms by >= 25 ms (i.e. not <= 107 ms)
#   RDP      evalfast2rdp (evalfast2rd + --b2-res-profile): REPORT only, bands wide (200:340:150:300:120:260, the round
#            is not profiled; the profiler slows every later launch, SCORING_resfix.md 2.4). Predict per draft-only tick:
#            kernel time 80-107 ms, the scan_innermost cumsum and the 1-D cummax gone from t.res (0 calls), onesweep
#            radix passes about half of rl's, radixSortKVInPlace about 1 per layer (the free-list sort and the two rank
#            sorts are gone). STEP_MIN 150.
#   RD-MEM   evalfast2rd and evalfast2rdp peak_gb <= 84 (MUST EF_mem_cap). Estimate: rl peak 83.04 GB plus under 0.1 GB
#            (per layer, transient: the own-dummy buffers of the mirror maps 3 x 5.6 MB, the free-list compaction about
#            25 MB, the bias fill's dummy rows Rd x H x k x 64 rows (buffer, hit mask, index: about 70 MB at k 64,
#            140 MB at k 128); tick constants up to about 60 MB (the grown arange); minus the two rank sorts' buffers).
#            So about 83.1-83.3 GB: it holds, with a margin of about 0.7 GB. If RD-MEM fails, drop the bias fill's own
#            dummies first (the largest term).
#
# REGISTERED for evalfast2c3 / evalfast2f3 / evalfast2f3_192 ([fr3], STAMP 2026-10-23T11:29:23Z, before any run). WHY: the fused v2
# kernels spill (_fr_v2 at NP 2048: about 3.0 GB of driver local memory, SCORING_b192.md section 6), so fused v2 does
# not fit at B 216 and runs 103 tok/s at B 192 vs torch 217 (DIAG_torch_res.md fix 1). The v3 kernels (r2_fusedres.py
# docstring, V3; --fused-kernel v3) make the same decisions with bounded tiles: no NP-wide vector, no (KP x NP) tile, no
# tl.sort. The CTA barriers of v3 cannot be tested on the CPU: evalfast2c3 is v3's GPU correctness gate.
#   evalfast2c3      evalfast2c (CHECK, B 216, 8 steps, --draft-ticks 4, no --bare, no eval role) + --fused-kernel v3
#   evalfast2f3      evalfast2wx (FAST timing, B 216, --wait-index --b2-fewer-reads) with --b2-resolve fused
#                    --fused-kernel v3
#   evalfast2f3_192  evalfast2f3 at B 192 (C 96 + X 1344, R_ver 1296: evalfast2f192's point), the direct comparison with
#                    fused v2 (evalfast2f192, 103 tok/s) and torch (evalfast2t192, 217 tok/s) at B 192
#   C3       evalfast2c3 MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: fused v3 == torch at every
#            compared layer (0 differences, every layer compared). REPORT: launches 4 per draft-only layer (the
#            draft-only phase's fused.launches_per_layer 4.0) and 7 per mixed layer; warm-up local memory < 0.5 GB
#            (meta.fused_warm_mem_gb before - after, and the [oomfix] stderr line; v2: 3.0 GB). FALSIFIER: a
#            difference: then evalfast2f3 and evalfast2f3_192 are VOID
#   F3-MED   evalfast2f3 steady tok/s (median tick) at B 216: point 450, band [330, 650] (diagnosis estimate 550-600;
#            wide because untested). F3-MEAN [250, 560]; F3-EFF [210, 480] (REPORT: the WX mean / median and effective /
#            median ratios, 0.81 and 0.71, on the median band, widened)
#   F3-192   evalfast2f3_192 steady tok/s (median tick) at B 192: point 420, band [300, 620]. F3-192-MEAN [230, 530];
#            F3-192-EFF [200, 450] (the same ratios)
#   F3-FALS  FALSIFIED if the evalfast2f3 median < 280 at B 216 (then v3 does not beat torch), or the res span at the
#            median tick > 100 ms
#   F3-MEM   both timing cells peak_gb <= 84 (MUST EF_mem_cap)
#
# REGISTERED for evalfast2f3p / evalfast2c3l / evalfast2f3l ([flean], STAMP 2026-10-23T17:13:30Z, before any run). WHY:
# analysis/evalfast/SCORING_fr3.md sections 2-4. F3 fell 24-25 % below its point. The res span at the median tick is
# 239.9 / 256.3 ms. Fused v3 keeps three costs: the bind / audit kernels (2-row cumsum, scatter_add_ on one dummy lane),
# the per-row Python at every layer, and 3 hidden syncs per mixed layer. --res-lean and --res-dev remove them on the
# torch path. This tree (ROOT NOSI_EVAL = nosi-r2-flean: graph 8ccf262 + fr3 e7de351) lets the fused path use them:
# FusedResolver calls _lean_same_rows / _lean_rows_ok / _lean_plan, XPool.bind_lean and draft_bias_fill_flat_lean
# (LeanConsts.dev selects r2_resdev's ops). The kernels do not change. The check mode's shadow pool keeps the REFERENCE
# bind (tests_cpu/test_r2_flean.py).
#   evalfast2f3p     evalfast2f3 + --b2-res-profile (no new code): the device time of the fused v3 draft-only tick.
#                    REPORT only, bands wide (200:340:150:300:120:260; the profiler slows every later launch). STEP_MIN
#                    150. Per draft-only tick (draft_only_res_profile.res, ticks 1-11):
#   F3P-H2D  "Memcpy HtoD (Pageable -> Device)" in res: n = 768 over 12 draft-only ticks (64 per tick, 2 per layer:
#            r2_xpool.py bind and the bias fill). FALSIFIER: any other count
#   F3P-SCAN tensor_kernel_scan_innermost_dim<long, plus> (the 2-row cumsum): n 384; 25.9 ms per tick, band [22, 30]
#   F3P-SADD _scatter_gather_elementwise_kernel<true, long> ReduceAdd (scatter_add_): n 384; 11.6 ms per tick, band
#            [9.5, 14]
#   F3P-TRI  the 4 Triton kernels _fr3_prep, _fr3_d1, _fr_scan, _fr3_d2: n 1536; together 80 ms per tick, band
#            [45, 110] (GUESS level)
#   F3P-RES  all res device time: point 125 ms per tick, band [95, 150]. Below 95: the span is host-bound
#   evalfast2c3l     evalfast2c3 (CHECK, B 216, 8 steps, --draft-ticks 4, no --bare, no eval role) + --res-lean
#                    --res-dev
#   C3L      MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer compared (the
#            lane compare is against the shadow pool's REFERENCE bind). FALSIFIER: a difference: evalfast2f3l is VOID
#   evalfast2f3l     evalfast2f3 + --res-lean --res-dev, B 216 (--eval-bands 335:375:258:292:220:248)
#   REVISED by the lead before any run (SCORING_fr3.md 3.3 gave 415 [365, 460]; that estimate counted 40-72 ms from
#   --res-dev). New evidence (SCORING_resdev.md): --res-dev cut the draft-only res KERNEL time 132 -> 61.9 ms, but the
#   res WALL span did not move (140-147 ms) and evalfast2rd = evalfast2rl within noise (4 same-job pairs: rd 324.8 /
#   333.4 / 321.5 / 327.6 vs rl 324.5 / 327.7 / 326.0 / 324.8). The span is launch-bound (about 11,300 launches per
#   draft-only tick), so --res-dev counts 0 here; only --res-lean's host terms on the fused path count (per-row Python
#   29-35 ms and 1-3 hidden syncs per mixed tick, partly overlapped): 10-30 ms of the 492.5 ms median tick.
#   F3L-MED  steady tok/s (median tick): point 350, band [335, 375] (f3: 335.64 / 340.21)
#   F3L-MEAN steady tok/s (mean tick): point 272, band [258, 292] (f3 mean / median 0.776)
#   F3L-EFF  effective tok/s: point 230, band [220, 248] (f3 effective / median 0.658)
#   F3L-FALS FALSIFIED if the evalfast2f3l median < 340 (no gain over f3's two runs), or the bare stats median h_pages
#            >= 237 ms (f3: 240.534 / 237.760; predicted 210-230)
#   F3L-MEM  peak_gb <= 84 GB (MUST EF_mem_cap; f3 peak 83.16, the dev dummies add per-layer transients of about
#            0.1-0.3 GB). If it fails, drop the bias fill's own dummies first
#
# REGISTERED for evalfast2ia ([admitfill], STAMP 2026-10-23T18:17:11Z, before any run). WHY: analysis/evalfast/DIAG_effective_gap.md
# sections 3-4: effective is 15.26 s per round below steady; 12.96 s of it is ticks 0-4, and 6.58 s of that is the poll
# (h_poll_arrivals g1rl ticks 1-4: 917.7, 4865.3, 442.1, 547.2 ms; median 38.1 ms): a sort of every transfer in flight
# and the arrival rule per transfer, when the first fill arrives. --inflight-array admits the due draft issues as whole
# parts (one done / force per issue, one vectorized I6 check, the host maps in bulk); the same admitted sets, so the same
# decisions. ROOT NOSI_EVAL = nosi-r2-admitfill. The cell is evalfast2rl plus ONE flag. The control is evalfast2rl of the
# same job (else g1rl: W 46748 ms, median tick 512.9 ms, mean tick 655.9 ms, 61 ticks, 10152 committed).
# ARITHMETIC (INFERRED; g1rl): cost per take c = sum h_hs_takes / transfers = (2921.3 + 500.4 + 225.6 + 258.1 + 100.9 +
# 56 x 26.0) ms / 1267097 = 5462.3 / 1267097 = 4.31 us; first fill N0 = 2921.3 / 0.00431 = about 678,000 draft takes
# (tick 0), tick 1 about 116,000, a steady tick about 6,000 (about 12,000-24,000 in flight). Array poll cost per transfer:
# membership map 0.06 us, urgent-set check 0.03 us, bulk forget (in-flight pop, pending get / delete, field maps) about
# 0.4 us; device (column cat, three gathers, O.poll_arrivals) 20-60 ms for N0. So tick 2 = 794,000 x 0.09 us + 678,000 x
# 0.4 us + 20-60 ms = 71 + 271 + 40 = about 380 ms; tick 1 (N0 not due: seqs, membership, check) about 100 ms; ticks 3, 4
# about 70 and 100 ms; a steady tick 5-12 ms (point 8). Saved: ticks 1-4 6772.3 - 650 = 6122 ms; ticks 5-60 2658 - 56 x 8
# = 2210 ms (ticks 51-60 GUESSED at 32 ms each); tick 0 costs about 90 ms more (the part rows, one index copy). Net 8242 ms.
#   IA-POLL2 tick 2 h_poll_arrivals (bare_host_profile.per_tick_ms) point 380 ms, PASS <= 550 ms
#   IA-POLL  tick 1 <= 200 ms; ticks 3 and 4 <= 150 ms each; the median h_poll_arrivals point 8 ms, PASS <= 15 ms
#   IA-MED   steady tok/s (median tick): median tick 512.9 - (38.1 - 8) = 482.8 ms; 10152 / (61 x 0.4828) = 344.7: point
#            345, band [328, 362]
#   IA-MEAN  mean tick (ticks 1-60) (39355.1 - 8332) / 60 = 517.1 ms; 10152 / (61 x 0.5171) = 321.9: point 322,
#            band [305, 338]
#   IA-EFF   W = 46748 - 8242 = 38506 ms; 10152 / 38.506 = 263.6: point 264, band [245, 282]
#   IA-SAME  the bare round's ticks and committed equal evalfast2rl's of the same job (61, 10152): no decision changed
#   IA-T02   NOT expected to pass the lead's "ticks 0-2 each <= 1.5 x median" (fix 1's prediction): tick 0 about 4.1 s
#            (the takes 2.9 s and the issues 0.6 s stay), tick 1 about 2.35 s (the merges 1.1 s stay), tick 2 about 0.92 s
#            (5403.9 - 4865.3 + 380); fix 1's "effective >= 290" and "steady median unchanged" are NOT expected either
#            (point 264; the median tick falls by about 30 ms, the steady poll). No fill moves out of the clock, so there is
#            no admit_fill_ms: effective has one value
#   IA-FALS  FALSIFIED if tick 2 h_poll_arrivals > 1000 ms (then the time is not the scan: GIL waits on the feeder, or
#            the bulk forget), or the median h_poll_arrivals > 20 ms, or IA-SAME fails, or effective < 240
#   IA-MEM   peak_gb <= 84 (MUST EF_mem_cap). Estimate: rl 83.04 GB plus under 0.05 GB (N0 x 3 int64 columns 16 MB, the row
#            index 5 MB, the cat and the check masks of one poll about 25 MB)
#
# REGISTERED for evalfast2f3lia / evalfast2c3lia / evalacc216f3lia ([f3lia], STAMP 2026-10-24T01:50:02Z, before any run; the
# lead filled every PREDICTION from SCORING_flean.md, SCORING_inflight.md, SCORING_acc216.md, PROFILE_f3l_mixed.md). WHAT: this tree (ROOT NOSI_EVAL = nosi-r2-f3lia: flean 665ac69 merged with admitfill
# c714fb2) has the fused v3 resolver with --res-lean --res-dev (evalfast2f3l) and the array poll --inflight-array
# (evalfast2ia). The two parts are independent: the array poll is OwnResidency._poll_arrivals (the take path's issue
# parts), the fused resolve is FusedResolver.resolve; no new code joins them (tests_cpu/test_r2_f3lia.py). Same-job
# controls: evalfast2f3l (the fused path without the array poll) and evalfast2ia (the array poll on the torch resolve).
#   evalfast2f3lia   evalfast2f3l + --inflight-array (FAST timing, B 216, --eval-bands "$F3LIA_BANDS" =
#                    415:460:305:360:252:300, the F3LIA-MED/MEAN/EFF bands below)
#   evalfast2c3lia   evalfast2c3l + --inflight-array (CHECK, B 216, 8 steps, --draft-ticks 4, no --bare, no eval role)
#   C3LIA    MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer compared, with
#            the array poll on. FALSIFIER: a difference: evalfast2f3lia is VOID
#   F3LIA-MED  steady tok/s (median tick). PREDICTION: 435 [415, 460] (f3l 412.05 / 420.92 + ia's +6 % on rl; the
#              profile's estimate about 435)
#   F3LIA-MEAN steady tok/s (mean tick). PREDICTION: 330 [305, 360] (f3l mean 301.59 / 308.44; ia raised rl's mean +15 %)
#   F3LIA-EFF  effective tok/s. PREDICTION: 275 [252, 300] (f3l 249.68 / 254.46; ia raised rl's effective +12 %)
#   F3LIA-POLL h_poll_arrivals (bare_host_profile.per_tick_ms, tick 2 and the median). PREDICTION: tick 2 > 1000 ms
#              (REPORT; ia's own run kept 2705 / 2739 ms, IA-FALS (a) fired, so the poll is not the gain's cause)
#   F3LIA-SAME the bare round's ticks and committed equal evalfast2f3l's of the same job: no decision changed
#   F3LIA-FALS PREDICTION: the median is below the same job's evalfast2f3l median (ia adds nothing on the fused path)
#   F3LIA-MEM  peak_gb <= 84 (MUST EF_mem_cap). PREDICTION: 83.0 to 83.3 GB (f3l 83.18, ia 83.04)
#   evalacc216f3lia  evalacc216 (the acc216 block below: --modes fast,bi_c_strict, no --bare, no eval role,
#                    --store-committed-tokens --serial-rows 8 --gold-workload GOLD) on this path: --b2-resolve fused
#                    --fused-kernel v3 --res-lean --res-dev --inflight-array; --link-record = the evalfast2f3lia
#                    summary.json of the same run id and holder
#   ACC3LIA-BI   ACC_exact_batch_invariance (MUST). PREDICTION: PASS, 0 of 122 pairs differ
#   ACC3LIA-ETD  ACC_exact_vs_serial (MUST). PREDICTION: PASS, 8 of 8 rows x 48 committed tokens equal
#   ACC3LIA-AE   ACC_AE_fast_vs_exact (MUST). PREDICTION: EQUIVALENT (acc216 on rd: accept d -0.00010, gold d -0.00076)
#   ACC3LIA-LINK R_ACC_fast_link (REPORT) against evalfast2f3lia. PREDICTION: EQUAL counts
#   ACC3LIA-MEM  R_ACC_peak (REPORT). PREDICTION: <= 84 GB (acc216 full 83.25)
#   No miniature: evalfast2c3lia (8 steps, CHECK) runs first in the same job and gates every later cell.
#
# REGISTERED for evalfast2f3d / evalfast2c3d / evalfast2f3t / evalfast2c3t ([f3ld], STAMP 2026-10-24T04:16:01Z, before any
# run and before any f3lia GPU result; the lead's absolute bands assume f3lia near its registered points). WHAT: this tree (ROOT NOSI_EVAL = nosi-r2-f3ldx: nosi-r2-f3ld
# a5c8937 = f3lia 89c9169 + b2-desc 47db0c0 ported by hunks + --no-transfer-mirror, + [f3ldx] the descriptor's lane row
# on the X pool; analysis/evalfast/PLAN_admission_fill.md section 4 ranks 1a and 1b, section 5.2). Same-job control:
# evalfast2f3lia (the f3lia path in this tree, flags off).
#   evalfast2f3d  evalfast2f3lia + --worker cpp --b2-issue desc (FAST timing, B 216, --eval-bands "$F3LD_BANDS")
#   evalfast2c3d  evalfast2c3lia + --worker cpp --b2-issue desc (CHECK, B 216, 8 steps, --draft-ticks 4)
#   evalfast2f3t  evalfast2f3d + --no-transfer-mirror - --inflight-array (FAST timing, --eval-bands "$F3LT_BANDS")
#   evalfast2c3t  evalfast2c3d + --no-transfer-mirror --lean-time - --inflight-array (CHECK)
#   evalfast2f3c  [f3ldx] evalfast2f3lia + --worker cpp --b2-issue object (FAST timing, --eval-bands "$F3LC_BANDS"):
#                 the C++ feeder's share alone (no descriptor, no table path)
#   evalfast2c3c  [f3ldx] evalfast2c3lia + --worker cpp --b2-issue object (CHECK)
#   F3LD-BLOCK  CLEARED at this tree: E's issue_desc on the pool sends a (7, n) descriptor, row 6 = each take's lane
#               by the object path's rule (XPool.lanes_of = pos_of at XPool.phys_index, the index of assign_phys);
#               gilfree_pack.cpp (DESC 2) writes m' = lane on flat stores (the m' of _submit). CPU proof:
#               tests_cpu/test_r2_f3ldx.py (desc on pool == object on pool, store-equal every tick). PREDICTION: the CPU proof holds on GPU: C3D / C3T pass
#   C3C         MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer compared.
#               FALSIFIER: a difference: evalfast2f3c is VOID. PREDICTION: PASS, 0 diffs
#   F3C-MED     steady tok/s (median tick). PREDICTION: 437 [415, 465] (f3lia point 435; the C++ feeder is off the
#               steady tick's critical path)
#   F3C-MEAN    steady tok/s (mean tick). PREDICTION: 333 [305, 365]
#   F3C-EFF     effective tok/s. PREDICTION: 279 [252, 305] (PLAN_admission_fill.md: issue is 0.9 s of the 13.3 s)
#   F3C-ISSUE   tick 0 h_et_issue (bare_host_profile.per_tick_ms) and info.issue per tick. PREDICTION: tick 0 h_et_issue
#               below the same job's evalfast2f3lia tick 0 (REPORT)
#   F3C-SAME    the bare round's ticks and committed equal evalfast2f3lia's of the same job. PREDICTION: equal
#   C3D / C3T   MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer compared.
#               FALSIFIER: a difference: the matching timing cell is VOID. PREDICTION: PASS, 0 diffs (both)
#   F3D-MED     steady tok/s (median tick). PREDICTION: 440 [415, 470]
#   F3D-MEAN    steady tok/s (mean tick). PREDICTION: 345 [310, 380]
#   F3D-EFF     effective tok/s. PREDICTION: 315 [275, 360] (one job per issue: the issue and take objects go; the
#               per-take poll and merges stay)
#   F3D-ISSUE   tick 0 h_et_issue (bare_host_profile.per_tick_ms) and info.issue per tick. PREDICTION: tick 0 h_et_issue
#               at most half of the same job's evalfast2f3lia tick 0
#   F3D-SAME    the bare round's ticks and committed equal evalfast2f3lia's of the same job. PREDICTION: equal
#   F3D-FALS    PREDICTION: fires if effective < the same job's evalfast2f3lia effective + 10 tok/s
#   F3T-MED     steady tok/s (median tick). PREDICTION: 445 [415, 480]
#   F3T-MEAN    steady tok/s (mean tick). PREDICTION: 370 [320, 410]
#   F3T-EFF     effective tok/s. PREDICTION: 390 [350, 415] (PLAN_admission_fill.md section 4: wall -14 to -16 s)
#   F3T-T04     ticks 0-4: h_hs_takes, h_hs_merges, h_poll_arrivals, h_et_issue; arrival forced / forced_ns at tick 2.
#               PREDICTION: the sum of the four spans over ticks 0-4 below 4 s (f3l about 13 s); arrival forced 0
#   F3T-SAME    the bare round's ticks and committed equal evalfast2f3lia's of the same job. PREDICTION: equal
#   F3T-FALS    PREDICTION: fires if effective < the same job's evalfast2f3lia effective + 50 tok/s (the plan's
#               mechanism does not hold)
#   F3C-FALS    PREDICTION: fires if effective < the same job's evalfast2f3lia effective - 5 tok/s (the C++ feeder costs)
#   F3LD-MEM    peak_gb <= 84 (MUST EF_mem_cap) in the three timing cells. PREDICTION: 83.0 to 83.4 GB
#
# REGISTERED for evalacc216m / evalacc216 ([acc216], STAMP 2026-10-23T18:04:12Z, before any run; the lead writes the stamp from
# `date -u` in the command that freezes this block, L33). WHY: analysis/evalfast/STATUS_accuracy_B216.md: at B 216 no
# gate proves the exact target distribution (ETD), batch invariance (BI) or accuracy equivalence (AE) on the timed
# path; the evalacc record is batch 48 on another engine path. ROOT NOSI_EVAL = nosi-r2-acc216. Gates:
# r2_tick_sg5.py acc_gates (MUST unless the name starts with R_).
#   ACC-BI    ACC_exact_batch_invariance PASS: 0 differing pairs of 122 (rows b % 94 and b, b = 94 .. 215; 48 tokens per
#             row in evalacc216, 8 in evalacc216m); the prompt rows are equal in every pair. Basis: G-c strict passed
#             bigc G2 at M 1 .. 1148 (r2_evalacc.py EA-E) and the verify ticks use at most (1 + K) B = 864 rows; the
#             pool / b2 path is bitwise equal to the per-stream path under EXACT (x1 XP_bitwise, stack1a ST_bitwise).
#             FALSIFIER: 1 or more differing pairs: under EXACT the timed path at B 216 is NOT batch-invariant (the row
#             position or the co-batch changes the tokens); every bitwise claim at B 216 is VOID until it is localised
#   ACC-ETD   ACC_exact_vs_serial PASS: rows 0 .. 7 of the EXACT round equal the BI serial bi_c_strict decode of the
#             same 8 prompts, 8 of 8 rows, every token. FALSIFIER: a difference: the EXACT round on the timed path does
#             not commit the target model's greedy tokens
#   ACC-AE    ACC_AE_fast_vs_exact EQUIVALENT (evalacc216): |d acceptance| <= 0.01 (allowance about 0.0104 = 3 x the
#             binomial SE 0.0028 at 10,152 decisions per point and acceptance about 0.959, + floor 0.002; FAST noise 0);
#             gold match over the first copies of the 55 gold documents x 48 tokens = 2,640 tokens per point: |d gold|
#             inside the allowance (about 0.018 at gold match about 0.04; cap 0.02). FALSIFIER: AFFECTS: FAST changes
#             accuracy at B 216, so the FAST timing headline loses its accuracy basis (human decision part 3)
#   ACC-LINK  R_ACC_fast_link (REPORT, evalacc216): the FAST point's accepted 9732, rejected 420, committed 10152,
#             ticks 61 EQUAL the same holder's evalfast2rd record (the gate reads the numbers from that record).
#             FALSIFIER: DIFFER: this cell did not run the timed path (or the FAST counts are not reproducible), so
#             ACC-AE does not transfer to the timed runs (score the link VOID; ACC-BI and ACC-ETD keep their meaning)
#   ACC-AGREE R_ACC_fast_vs_exact_tokens (REPORT): fewer than 216 of 216 rows fully equal (batch 48 reference: 26 of
#             48, median first divergence 25). Not a falsifier: FAST is not batch-invariant
#   ACC-FDUP  R_ACC_fast_dup_rows (REPORT): no prediction
#   ACC-MEM   meta.peak_gb 83.1-83.5 GB (R_ACC_peak, REPORT: a proof cell has no --mem-cap-gb). Model: the points run
#             one engine at a time (one_point frees each engine before the next one); FAST is 83.25 GB in evalfast2rd
#             (its bare round was a third engine of the same size, so dropping --bare changes no peak); the EXACT
#             point holds the same store, twins, residency and X pool, and G-c strict allocates only its output (no
#             m_fix padding, no cuBLAS workspace): FAST +- 0.2 GB. The serial reference runs before the build's GPU
#             windows are freed: build peak 52.62 GB (evalfast2rd) + an 8-row copy under 2 GB, so it is not the peak.
#             PHYSICAL LIMIT: the A100 gives 79.25 GiB = 85.10 GB (DIAG_fused_check_oom.md); 83.5 GB allocated =
#             77.8 GiB leaves about 1.4 GiB for the CUDA context, the cuBLAS / Triton modules and the allocator's
#             slack, as FAST at 83.25 GB had on both GPUs. FALSIFIER: an OOM, or peak > 84 GB in evalacc216m (its
#             preflight peak is already the long run's: evalfast2rd preflight_peak_gb 82.76 at 4 steps): do NOT run
#             evalacc216; the fallback is B 188 (two copies of the 94 documents, 94 pairs; X 1316 = 7 per request,
#             R_ver 1268; about 28 x 0.29 = 8 GB less by the bigb slope), a new cell that needs its own review
#   ACC-MINI  evalacc216m: ACC-BI (122 pairs, 8 tokens) and ACC-ETD (8 rows, 8 tokens) PASS; peak inside ACC-MEM.
#             Its ACC-AE is EXPECTED EQUIVALENT, but at 1,512 decisions and 440 gold tokens the allowance is the cap
#             (0.02): an AFFECTS there is a REPORT for the scorer, NOT a falsifier (the full cell decides)
#   SCOPE     every --eval-role fast-time cell now writes R_EF_accuracy_scope (REPORT): with the batch 48 record and
#             B 216 it reads "SCOPE MISMATCH (REPORT, not MUST)"; no current cell stops on it
#   TIME      evalacc216 about 25-40 min (evalfast2rd 21.5 min without its bare round, plus the EXACT point at 1.5-4 x
#             the FAST round and the 8-row serial decode); STEP_MIN 120 in both cells
#
# REGISTERED for evalfast2eb ([endbatch], STAMP 2026-10-23T18:05:38Z, before any run). WHY: analysis/evalfast/DIAG_effective_gap.md
# section 2.4: the host time outside the tick spans (O) is 3335-3525 ms per bare round (mean 3456 ms, 22.6 % of the
# effective gap; 8 records g1-g4 rl / rd), and 98.7-98.9 % of it is the end_request loop in ticks 50-60 (about 3.3-3.5 s;
# 216 requests end there, about 15 ms each): one end_owner with a boolean mask index per request, and a scan of the
# whole queued set, hist_ev and _key_of per request. --end-batch ends the tick's requests with ONE end_owners call (ONE
# host read) and a per-request index of the queued keys (DIAG section 4, fix 3). The same final state as one end_request
# per request (tests_cpu/test_r2_endbatch.py). ROOT NOSI_EVAL = nosi-r2-endbatch. The cell is evalfast2rl (--wait-index
# --b2-fewer-reads --res-lean) plus --end-batch. O = bare_round.wall_s (ms) - bare_host_profile.tick_ms[0] - 60 x
# bare_T_tick.mean (DIAG section 2.1).
#   EB-O     the outside-span term O <= 350 ms (from about 3.4 s); the sum of bare_host_profile.per_tick_ms.h_end_request
#            <= 300 ms (from 3337 ms in g1rl)
#   EB-EFF   effective tok/s about +7 % over evalfast2rl (from about 218): point 233, band [226, 245] (10152 tokens over
#            the wall clock minus about 3.1 s: 46.5 s -> 43.4 s)
#   EB-MED   steady tok/s (median tick) unchanged within noise: 326 +- 6, band [315, 340] (the ends are outside the
#            tick spans; the median tick does not contain them)
#   EB-MEAN  [245, 268] (the mean tick does not contain the ends either: evalfast2rl's mean about 256, unchanged)
#   EB-FALS  FALSIFIED if O > 2 s (then the time is the wait for the queued GPU work at the first host read of the ends,
#            not the per-request Python)
#   EB-MEM   peak_gb <= 84 (MUST EF_mem_cap; the ends allocate a few small (L, n, H, NPOS) int8 tensors only)
#
# REGISTERED for evalfast2ab / evalfast2c3ab ([applyev], SOURCE ONLY, not run; STAMP 2026-10-24T04:59:23Z, before any
# run; CPU suites ab_8ed091c_{a,b,c1,c2,c3} green). WHY: analysis/evalfast/PROFILE_f3l_mixed.md sections 3.2-3.3 and 4 (step 3): on F3L
# (evalfast2f3l) apply_events is 60.5 ms of the 403.9 ms median tick and the one span that is still launch-bound. Per
# rolled-back request and per layer it runs sync_draft_bank (10 small device ops) and copy_ring (3 copies): 13 x 32 =
# 416 launches per rollback. --apply-batched (ROOT NOSI_EVAL = nosi-r2-applyev: flean 665ac69 + this change) puts the
# rollbacks into the publications' batched rebuild (the _publish_batched pattern: per layer ONE rebuild_banks_unchecked
# and ONE copy_rows_many for every rollback and publication of the tick) and makes the keep vector once per tick. The
# state is bitwise the unbatched state (r2_tick.py TickEngine._apply_batched docstring: EXACTNESS (a)-(c)).
#   evalfast2ab      evalfast2f3l + --apply-batched (FAST timing, B 216, --eval-bands 425:475:305:345:248:280, the
#                    AB-MED / AB-MEAN / AB-EFF bands below)
#   evalfast2c3ab    evalfast2c3l (CHECK, B 216, 8 steps, --draft-ticks 4, no --bare, no eval role) + --apply-batched
#   evalfast2f3l     the same-job control (unchanged arguments; this ROOT, --apply-batched off)
#   AB-C3    evalfast2c3ab MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer
#            compared; MUST <label>_AB_acted (apply_calls > 0, one call with >= 2 rollbacks). FALSIFIER: a difference:
#            evalfast2ab is VOID
#   AB-ACT   evalfast2ab MUST <label>_AB_acted (the round's engine ran the batched apply with rollbacks)
#   AB-MED   steady tok/s (median tick). PREDICTION: 450 [425, 475] (f3l 412.05 / 420.92; about 35 of the 60.5 ms
#            apply_events span goes, so the 404 ms median tick falls to about 370 ms)
#   AB-MEAN  steady tok/s (mean tick). PREDICTION: 323 [305, 345] (the same ms saving on a longer mean tick)
#   AB-EFF   effective tok/s. PREDICTION: 262 [248, 280] (the 16 s first-fill / end gap does not change)
#   AB-HAPPLY bare_host_profile h_apply_events median. PREDICTION: below 30 ms (same-job evalfast2f3l about 60 ms)
#   AB-FALS  PREDICTION: fires if the median is below the same job's evalfast2f3l median + 10 tok/s
#   AB-MEM   peak_gb <= 84 (MUST EF_mem_cap). PREDICTION: 83.1 to 83.3 GB (f3l 83.18)
#
# REGISTERED for evalfast2all / evalfast2c3all / evalacc216all ([f3all], STAMP 2026-10-24T05:23:37Z, before any run; the
# lead used the first f3lia record sg5_l3_2192144_evalfast2f3lia 433.94 / 351.43 / 281.21 and SCORING_endbatch.md). WHAT: this tree (ROOT NOSI_EVAL = nosi-r2-f3all:
# f3lia 5cdfc75 merged with endbatch 1ae07fe and applyev a3199da) has the f3lia path (fused v3 + --res-lean --res-dev +
# --inflight-array), the batched ends --end-batch (evalfast2eb) and the batched apply --apply-batched (evalfast2ab). The
# three parts are independent: the ends are run_round's (after the tick span; OwnResidency.end_requests), the batched
# apply is TickEngine.apply_events (_apply_batched), the fused resolve and the array poll are the service's; no new code
# joins them (tests_cpu/test_r2_f3all.py). Same-job control: evalfast2f3lia (this ROOT, both new flags off).
#   evalfast2all     evalfast2f3lia + --end-batch --apply-batched (FAST timing, B 216, --eval-bands "$ALL_BANDS", the
#                    ALL-MED / ALL-MEAN / ALL-EFF bands below; the step refuses the placeholder outside DRYRUN)
#   evalfast2c3all   evalfast2c3lia + --end-batch --apply-batched (CHECK, B 216, 8 steps, --draft-ticks 4, no --bare, no
#                    eval role)
#   evalfast2f3lia   the same-job control (unchanged arguments; this ROOT)
#   C3ALL    evalfast2c3all MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer
#            compared, with the array poll, the batched ends and the batched apply on; MUST <label>_AB_acted (apply_calls
#            > 0, one call with >= 2 rollbacks). FALSIFIER: a difference: evalfast2all and evalacc216all are VOID
#   ALL-ACT    evalfast2all MUST <label>_AB_acted (the round's engine ran the batched apply with rollbacks)
#   ALL-MED    steady tok/s (median tick). PREDICTION: 470 [440, 500] (f3lia 433.94; --apply-batched about -35 ms of
#              the median tick; --end-batch does not touch the median tick)
#   ALL-MEAN   steady tok/s (mean tick). PREDICTION: 375 [345, 405] (f3lia 351.43)
#   ALL-EFF    effective tok/s. PREDICTION: 320 [295, 345] (f3lia 281.21 = about 43 s per round; --end-batch -3.4 s,
#              --apply-batched about -2 s over 61 ticks)
#   ALL-BANDS  --eval-bands 440:500:345:405:295:345 (ALL-MED / ALL-MEAN / ALL-EFF)
#   ALL-HEND   the sum of bare_host_profile.per_tick_ms.h_end_request (ticks 0-60). PREDICTION: below 50 ms (eb 10.5 ms;
#              f3lia about 3400 ms)
#   ALL-HAPPLY bare_host_profile h_apply_events median. PREDICTION: below 30 ms (f3l about 60 ms)
#   ALL-SAME   the bare round's ticks and committed equal evalfast2f3lia's of the same job. PREDICTION: equal
#   ALL-FALS   the falsifier. PREDICTION: fires if effective < the same job's evalfast2f3lia effective + 15 tok/s
#   ALL-MEM    peak_gb <= 84 (MUST EF_mem_cap). PREDICTION: 83.0 to 83.4 GB
#   evalacc216all    evalacc216f3lia (--modes fast,bi_c_strict, no --bare, no eval role, --store-committed-tokens
#                    --serial-rows 8 --gold-workload GOLD, --b2-resolve fused --fused-kernel v3 --res-lean --res-dev
#                    --inflight-array) + --end-batch --apply-batched; --link-record = the evalfast2all summary.json of the
#                    same run id and holder
#   ACCALL-BI   ACC_exact_batch_invariance (MUST). PREDICTION: PASS, 0 of 122 pairs differ
#   ACCALL-ETD  ACC_exact_vs_serial (MUST). PREDICTION: PASS, 8 of 8 rows x 48 committed tokens equal
#   ACCALL-AE   ACC_AE_fast_vs_exact (MUST). PREDICTION: EQUIVALENT
#   ACCALL-LINK R_ACC_fast_link (REPORT) against evalfast2all. PREDICTION: EQUAL counts
#   ACCALL-MEM  R_ACC_peak (REPORT). PREDICTION: <= 84 GB
#   No miniature: evalfast2c3all (8 steps, CHECK) runs first in the same job and gates every later cell.
#
# REGISTERED for evalfast2tall / evalfast2c3tall / evalacc216tall ([f3tall], NOT STAMPED: the lead fills every
# PREDICTION and the bands and writes the stamp from `date -u` before any run, L33). WHAT: this tree (ROOT NOSI_EVAL =
# nosi-r2-f3tall: f3all 3a7f51a merged with f3ldx 3769b64) has the f3t path (fused v3 + --res-lean --res-dev +
# --worker cpp --b2-issue desc --no-transfer-mirror, no --inflight-array) and the f3all flags --end-batch
# --apply-batched. The merge adds one line of joining code: OwnResidency.end_requests drops each ended request's
# write-back marks (_hm[b] = -1), as end_request does since b2-desc (tests_cpu/test_r2_f3tall.py). Same-job controls:
# evalfast2f3t (f3ldx flags only), evalfast2all (f3all flags only), evalfast2f3lia (both off).
#   evalfast2tall    evalfast2f3t + --end-batch --apply-batched (FAST timing, B 216, --eval-bands "$TALL_BANDS" = TBD;
#                    the step refuses the placeholder outside DRYRUN)
#   evalfast2c3tall  evalfast2c3t + --end-batch --apply-batched (CHECK, B 216, 8 steps, --draft-ticks 4, no --bare, no
#                    eval role)
#   C3TALL   evalfast2c3tall MUST _draft_only_fused_equal / _round_fused_equal / _fused_err PASS: 0 diffs, every layer
#            compared; MUST <label>_AB_acted (apply_calls > 0, one call with >= 2 rollbacks); MUST desc acted, scored
#            from the record (no harness gate): residency.report.b2 issue "desc", desc_jobs > 0, desc_takes > 0,
#            transfer_mirror false, no_transfer_mirror.jobs > 0; info.issue desc_calls > 0. FALSIFIER: a difference or
#            a flag that did not act: evalfast2tall and evalacc216tall are VOID. PREDICTION: TBD by lead
#   TALL-ACT   evalfast2tall MUST <label>_AB_acted. PREDICTION: TBD by lead
#   TALL-MED   steady tok/s (median tick). PREDICTION: TBD by lead
#   TALL-MEAN  steady tok/s (mean tick). PREDICTION: TBD by lead
#   TALL-EFF   effective tok/s. PREDICTION: TBD by lead
#   TALL-HEND  the sum of bare_host_profile.per_tick_ms.h_end_request (ticks 0-60). PREDICTION: TBD by lead
#   TALL-SAME  the bare round's ticks and committed equal evalfast2f3t's of the same job. PREDICTION: TBD by lead
#   TALL-FALS  the falsifier against the same job's evalfast2f3t and evalfast2all. PREDICTION: TBD by lead
#   TALL-MEM   peak_gb <= 84 (MUST EF_mem_cap). PREDICTION: TBD by lead
#   evalacc216tall   evalacc216f3lia's proof on the tall flag set; --link-record = the evalfast2tall summary.json of the
#                    same run id and holder
#   ACCTALL    the ACC3LIA gates (BI, ETD, AE, LINK against evalfast2tall, MEM). PREDICTION: TBD by lead
#   No miniature: evalfast2c3tall (8 steps, CHECK) runs first in the same job and gates every later cell.
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
NOSI_INTEG=/mnt/beegfs/ojerbi/nosi-r2-stack2                                             # ROOT of the stack1 cells only ([integ]; [stack2])
NOSI_BIGB=/mnt/beegfs/ojerbi/nosi-r2-bigb                                                # ROOT of the bigb cells only ([bigb])
FRNOSI=/mnt/beegfs/ojerbi/nosi-r2-fusedres2                                              # ROOT of the frg* cells only [fusedres2]
NOSI_EVAL=/mnt/beegfs/ojerbi/nosi-r2-f3tall                                              # ROOT of the evalfast* and evalacc216* cells only [evalsplit] ([sg5par] claim; [hostfix] host fixes; [hostw2] own worktree; [frfix] own worktree; [resfix] own worktree; [resdev] own worktree; [fr3] own worktree; [flean] own worktree; [admitfill] own worktree; [acc216] own worktree; [f3lia] own worktree: flean + admitfill + acc216; [endbatch] own worktree; [applyev] own worktree; [f3all] own worktree: f3lia + endbatch + applyev; [f3ld] own worktree: f3lia + b2-desc + no transfer mirror; [f3ldx] own worktree: f3ld + the descriptor's lane row on the X pool; [f3tall] own worktree: f3all + f3ldx)
EVALACC_OUT=/mnt/beegfs/ojerbi/r2-runs/evalacc                                           # ROOT (r2_evalacc_step.sh's OUTROOT)
GOLD=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280                        # ROOT [acc216] (r2_evalacc_step.sh's)
GOLD_SHA=f2fb36f441ec12b7a79239d5e86263c6454b7713436f53f81086ef2bedcc34b3                 # PIN (ctx65280_tokens.bin) [acc216]
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
usage() { echo "usage: r2_tick_sg5_step.sh --nosi <sha> <holder> <run id> [p4a|p4b-sc|p4b-sb|p4b-v1|p4sc64|p5b2|p6|x1|bigb81-128|bigb81-244|bigb96-128|bigb96-216|stack1|stack1a|stack1a128|stack1p|stack1t|stack1b|stack1n-twin|stack1n-rebuild|stack1n-resolve|stack1n-layout|stack1n-attn|stack1n-warmup|frg0|frg0x|frg0p|frg1|frg2|frg3|frg3t|frg|evalfast|evalfast2|evalfast2t|evalfast2c|evalfast2f192|evalfast2t192|evalfast2c192|evalfast2w|evalfast2wi|evalfast2fr|evalfast2fx|evalfast2wx|evalfast2rp|evalfast2rl|evalfast2rd|evalfast2rdp|evalfast2c3|evalfast2f3|evalfast2f3_192|evalfast2f3p|evalfast2c3l|evalfast2f3l|evalfast2ia|evalacc216m|evalacc216|evalfast2f3lia|evalfast2c3lia|evalacc216f3lia|evalfast2eb|evalfast2ab|evalfast2c3ab|evalfast2all|evalfast2c3all|evalacc216all|evalfast2f3d|evalfast2c3d|evalfast2f3t|evalfast2c3t|evalfast2f3c|evalfast2c3c|evalfast2tall|evalfast2c3tall|evalacc216tall] [gilfree|gather]" >&2
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
        evalfast2|evalfast2t|evalfast2c) [ -z "$CELL" ] || usage; CELL=$w ;;                     # [hostfix]
        evalfast2f192|evalfast2t192|evalfast2c192) [ -z "$CELL" ] || usage; CELL=$w ;;            # [b192]
        evalfast2w) [ -z "$CELL" ] || usage; CELL=$w ;;                                             # [hostw]
        evalfast2wi|evalfast2fr) [ -z "$CELL" ] || usage; CELL=$w ;;                                # [hostw2]
        evalfast2fx|evalfast2wx) [ -z "$CELL" ] || usage; CELL=$w ;;                                # [frfix]
        evalfast2rp|evalfast2rl) [ -z "$CELL" ] || usage; CELL=$w ;;                                # [resfix]
        evalfast2rd|evalfast2rdp) [ -z "$CELL" ] || usage; CELL=$w ;;                               # [resdev]
        evalfast2c3|evalfast2f3|evalfast2f3_192) [ -z "$CELL" ] || usage; CELL=$w ;;                # [fr3]
        evalfast2f3p|evalfast2c3l|evalfast2f3l) [ -z "$CELL" ] || usage; CELL=$w ;;                 # [flean]
        evalfast2ia) [ -z "$CELL" ] || usage; CELL=$w ;;                                            # [admitfill]
        evalacc216m|evalacc216) [ -z "$CELL" ] || usage; CELL=$w ;;                                 # [acc216]
        evalfast2f3lia|evalfast2c3lia) [ -z "$CELL" ] || usage; CELL=$w ;;                          # [f3lia]
        evalacc216f3lia) [ -z "$CELL" ] || usage; CELL=$w ;;                                        # [f3lia] [acc216]
        evalfast2eb) [ -z "$CELL" ] || usage; CELL=$w ;;                                            # [endbatch]
        evalfast2ab|evalfast2c3ab) [ -z "$CELL" ] || usage; CELL=$w ;;                              # [applyev]
        evalfast2all|evalfast2c3all) [ -z "$CELL" ] || usage; CELL=$w ;;                            # [f3all]
        evalacc216all) [ -z "$CELL" ] || usage; CELL=$w ;;                                          # [f3all] [acc216]
        evalfast2f3d|evalfast2c3d|evalfast2f3t|evalfast2c3t) [ -z "$CELL" ] || usage; CELL=$w ;;    # [f3ld]
        evalfast2f3c|evalfast2c3c) [ -z "$CELL" ] || usage; CELL=$w ;;                              # [f3ldx]
        evalfast2tall|evalfast2c3tall) [ -z "$CELL" ] || usage; CELL=$w ;;                          # [f3tall]
        evalacc216tall) [ -z "$CELL" ] || usage; CELL=$w ;;                                         # [f3tall] [acc216]
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
ACC=                                                        # [acc216] the evalacc216* cells only (the gold pin)
LINKARG=()                                                  # [acc216] evalacc216 only
GF3=                                                        # [f3ld] the f3ld cells only: packer v3 of --nosi (DESC 2)
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
    evalfast2|evalfast2t) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL   # [hostfix]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2 ]; then RES=fused ; BANDS=75:150:65:145:55:130
        else RES=torch ; BANDS=220:300:195:290:140:220
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve "$RES" --host-profile --gc-freeze --lean-time) ;;
    evalfast2c) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2c       # [hostfix]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check) ;;
    evalfast2f192|evalfast2t192) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1344 ; PRV=1296 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [b192]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=192 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2f192 ]; then RES=fused ; BANDS=75:140:60:135:50:120
        else RES=torch ; BANDS=220:300:180:290:140:230
        fi
        EVALARG=(--layout pool --X-pool 1344 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve "$RES" --host-profile --gc-freeze --lean-time) ;;
    evalfast2w) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2w        # [hostw]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands 270:330:215:270:185:235 --b2-resolve torch --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads) ;;
    evalfast2wi|evalfast2fr) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [hostw2]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2wi ]; then ONEFLAG=--wait-index ; BANDS=254:273:200:230:175:200
        else ONEFLAG=--b2-fewer-reads ; BANDS=195:225:165:195:148:172
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve torch --host-profile --gc-freeze --lean-time "$ONEFLAG") ;;
    evalfast2fx|evalfast2wx) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [frfix]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2fx ]; then FXFLAGS=(--b2-fewer-reads) ; BANDS=250:275:200:225:178:198
        else FXFLAGS=(--wait-index --b2-fewer-reads) ; BANDS=255:282:205:235:180:205
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve torch --host-profile --gc-freeze --lean-time "${FXFLAGS[@]}") ;;
    evalfast2rp|evalfast2rl) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [resfix]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2rp ]; then RXFLAG=--b2-res-profile ; BANDS=200:340:150:300:120:260 ; STEP_MIN=150
        else RXFLAG=--res-lean ; BANDS=280:330:225:265:195:225
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve torch --host-profile --gc-freeze --lean-time --wait-index
                 --b2-fewer-reads "$RXFLAG") ;;
    evalfast2rd|evalfast2rdp) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [resdev]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2rdp ]; then RDFLAGS=(--res-lean --res-dev --b2-res-profile) ; BANDS=200:340:150:300:120:260 ; STEP_MIN=150
        else RDFLAGS=(--res-lean --res-dev) ; BANDS=335:420:260:300:222:255
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve torch --host-profile --gc-freeze --lean-time --wait-index
                 --b2-fewer-reads "${RDFLAGS[@]}") ;;
    evalfast2ia) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [admitfill]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands 328:362:305:338:245:282 --b2-resolve torch --host-profile --gc-freeze --lean-time --wait-index
                 --b2-fewer-reads --res-lean --inflight-array) ;;
    evalfast2eb) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2eb        # [endbatch]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; BANDS=315:340:245:268:226:245
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve torch --host-profile --gc-freeze --lean-time --wait-index
                 --b2-fewer-reads --res-lean --end-batch) ;;
    evalfast2c192) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1344 ; PRV=1296 ; SWEEP_RUN= ; SUFFIX=_evalfast2c192    # [b192]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=192 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1344 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check) ;;
    evalfast2c3) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2c3      # [fr3]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3) ;;
    evalfast2f3|evalfast2f3_192) MODES=fast ; BACKENDS=b2 ; PC=96 ; SWEEP_RUN= ; SUFFIX=_$CELL                 # [fr3]
        NOSI=$NOSI_EVAL ; POOLB=() ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2f3 ]; then BATCH=216 ; PX=1512 ; PRV=1464 ; BANDS=330:650:250:560:210:480
        else BATCH=192 ; PX=1344 ; PRV=1296 ; BANDS=300:620:230:530:200:450
        fi
        EVALARG=(--layout pool --X-pool "$PX" --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads) ;;
    evalfast2f3p|evalfast2f3l) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [flean]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        if [ "$CELL" = evalfast2f3p ]; then FLFLAGS=(--b2-res-profile) ; BANDS=200:340:150:300:120:260 ; STEP_MIN=150
        else FLFLAGS=(--res-lean --res-dev) ; BANDS=335:375:258:292:220:248
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads "${FLFLAGS[@]}") ;;
    evalfast2c3l) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2c3l    # [flean]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev) ;;
    evalfast2f3lia) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2f3lia  # [f3lia]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        F3LIA_BANDS=415:460:305:360:252:300                 # REGISTERED F3LIA-MED/MEAN/EFF (STAMP 2026-10-24T01:50:02Z)
        [ "${DRYRUN:-0}" = 1 ] || [ "$F3LIA_BANDS" != TBD ] ||
            stop "cell $CELL: --eval-bands TBD (the lead fills and stamps the REGISTERED F3LIA block first)"
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$F3LIA_BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array) ;;
    evalfast2c3lia) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2c3lia  # [f3lia]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev --inflight-array) ;;
    evalacc216m|evalacc216) MODES=fast,bi_c_strict ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [acc216]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; ACC=1
        if [ "$CELL" = evalacc216m ]; then STEPS=8                       # the miniature: memory and gate wiring, no link
        else LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2rd/summary.json")    # the same holder's
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve torch --host-profile --gc-freeze --lean-time --wait-index --b2-fewer-reads --res-lean
                 --res-dev --store-committed-tokens --serial-rows 8 --gold-workload "$GOLD" "${LINKARG[@]}") ;;
    evalacc216f3lia) MODES=fast,bi_c_strict ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3lia]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; ACC=1    # evalacc216 on f3lia's path
        LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2f3lia/summary.json")      # the same holder's
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time --wait-index --b2-fewer-reads
                 --res-lean --res-dev --inflight-array --store-committed-tokens --serial-rows 8 --gold-workload "$GOLD"
                 "${LINKARG[@]}") ;;
    evalfast2ab) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2ab        # [applyev]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        ABFLAGS=(--res-lean --res-dev --apply-batched)              # evalfast2f3l's FLFLAGS + --apply-batched (no bands)
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands 425:475:305:345:248:280 --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads "${ABFLAGS[@]}") ;;
    evalfast2c3ab) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2c3ab  # [applyev]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev --apply-batched) ;;
    evalfast2all) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2all     # [f3all]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched
        ALL_BANDS=440:500:345:405:295:345                   # REGISTERED ALL-MED/MEAN/EFF (STAMP 2026-10-24T05:23:37Z)
        [ "${DRYRUN:-0}" = 1 ] || [ "$ALL_BANDS" != TBD ] ||
            stop "cell $CELL: --eval-bands TBD (the lead fills and stamps the REGISTERED F3ALL block first)"
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$ALL_BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array --end-batch --apply-batched) ;;
    evalfast2c3all) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_evalfast2c3all  # [f3all]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev --inflight-array --end-batch --apply-batched) ;;
    evalacc216all) MODES=fast,bi_c_strict ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL   # [f3all]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; ACC=1    # evalacc216 on f3all's path
        LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2all/summary.json")        # the same holder's
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time --wait-index --b2-fewer-reads
                 --res-lean --res-dev --inflight-array --end-batch --apply-batched --store-committed-tokens --serial-rows 8
                 --gold-workload "$GOLD" "${LINKARG[@]}") ;;
    evalfast2f3d|evalfast2f3t) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3ld]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; GF3=1
        F3LD_BANDS=415:470:310:380:275:360                  # REGISTERED F3D-MED/MEAN/EFF (STAMP 2026-10-24T04:16:01Z)
        F3LT_BANDS=415:480:320:410:350:415                  # REGISTERED F3T-MED/MEAN/EFF (STAMP 2026-10-24T04:16:01Z)
        if [ "$CELL" = evalfast2f3d ]; then F3BANDS=$F3LD_BANDS ; F3FLAGS=(--inflight-array --worker cpp --b2-issue desc)
        else F3BANDS=$F3LT_BANDS ; F3FLAGS=(--worker cpp --b2-issue desc --no-transfer-mirror)
        fi
        [ "${DRYRUN:-0}" = 1 ] || [ "$F3BANDS" != TBD ] ||
            stop "cell $CELL: --eval-bands TBD (the lead fills and stamps the REGISTERED F3LD block first)"
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$F3BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads --res-lean --res-dev "${F3FLAGS[@]}") ;;
    evalfast2c3d|evalfast2c3t) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3ld]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4 ; GF3=1
        if [ "$CELL" = evalfast2c3d ]; then C3FLAGS=(--inflight-array --worker cpp --b2-issue desc)
        else C3FLAGS=(--lean-time --worker cpp --b2-issue desc --no-transfer-mirror)
        fi
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev "${C3FLAGS[@]}") ;;
    evalfast2f3c) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3ldx]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; GF3=1
        F3LC_BANDS=415:465:305:365:252:305                  # REGISTERED F3C-MED/MEAN/EFF (STAMP 2026-10-24T04:16:01Z)
        [ "${DRYRUN:-0}" = 1 ] || [ "$F3LC_BANDS" != TBD ] ||
            stop "cell $CELL: --eval-bands TBD (the lead fills and stamps the REGISTERED F3LD block first)"
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$F3LC_BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array --worker cpp --b2-issue object) ;;
    evalfast2c3c) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3ldx]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4 ; GF3=1
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev --inflight-array --worker cpp
                 --b2-issue object) ;;
    evalfast2tall) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3tall]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; GF3=1
        TALL_BANDS=TBD                                      # REGISTERED TALL-MED/MEAN/EFF (NOT STAMPED: the lead's)
        [ "${DRYRUN:-0}" = 1 ] || [ "$TALL_BANDS" != TBD ] ||
            stop "cell $CELL: --eval-bands TBD (the lead fills and stamps the REGISTERED TALL block first)"
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped
                 --eval-role fast-time --mem-cap-gb 84
                 --accuracy-record "$EVALACC_OUT/eacc_${RUN}_${JOB}_evalacc/summary.json"
                 --eval-bands "$TALL_BANDS" --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time
                 --wait-index --b2-fewer-reads --res-lean --res-dev --worker cpp --b2-issue desc --no-transfer-mirror
                 --end-batch --apply-batched) ;;
    evalfast2c3tall) MODES=fast ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3tall]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; STEPS=8 ; DTICKS=4 ; GF3=1
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve check --fused-kernel v3 --res-lean --res-dev --lean-time --worker cpp --b2-issue desc
                 --no-transfer-mirror --end-batch --apply-batched) ;;
    evalacc216tall) MODES=fast,bi_c_strict ; BACKENDS=b2 ; PC=96 ; PX=1512 ; PRV=1464 ; SWEEP_RUN= ; SUFFIX=_$CELL  # [f3tall]
        NOSI=$NOSI_EVAL ; POOLB=() ; BATCH=216 ; MEMG=240G ; VARIANTS=fused/batched ; ACC=1 ; GF3=1    # on the tall path
        LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2tall/summary.json")       # the same holder's
        EVALARG=(--layout pool --X-pool 1512 --arrival-lag 2 --repeat-docs --log-slots -1 --skip-shipped
                 --b2-resolve fused --fused-kernel v3 --host-profile --gc-freeze --lean-time --wait-index --b2-fewer-reads
                 --res-lean --res-dev --worker cpp --b2-issue desc --no-transfer-mirror --end-batch --apply-batched
                 --store-committed-tokens --serial-rows 8 --gold-workload "$GOLD" "${LINKARG[@]}") ;;
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
CLAIMNAME=sg5_active_$JOB                                   # [sg5par] one claim per holder (header: CLAIM)
CLAIM=$OUTROOT/$CLAIMNAME
if [ -n "${SG5_CLAIM_PROBE:-}" ]; then                      # [sg5par] the CPU test of the claim's name and its mkdir
    [ "$DRY" = 1 ] || stop "SG5_CLAIM_PROBE is for DRYRUN (the CPU test) only"
    if mkdir "$SG5_CLAIM_PROBE/$CLAIMNAME" 2>/dev/null; then
        echo "R2 SG5 step: CLAIM_PROBE took $CLAIMNAME"
        exit 0
    fi
    echo "R2 SG5 step: CLAIM_PROBE $CLAIMNAME is held" >&2
    exit 1
fi
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
             nosi/nosi/r2_twin_fused.py nosi/nosi/r2_fusedres.py; do                  # [hostfix] + the fused resolve
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
PK_PY_BLOB=$GF_PY_BLOB PK_CPP_BLOB=$GF_CPP_BLOB PK_SRC="$GF_SHA (packer v2)"   # [f3ld] the packer this step copies
if [ -n "$GF3" ]; then                                      # [f3ld] packer v3 of --nosi (r2_gfw_step.sh's rule) + DESC
    PK_PY_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$GF_PY") || stop "packer v3: $GF_PY is not tracked at $NSHA"
    PK_CPP_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$GF_CPP") || stop "packer v3: $GF_CPP is not tracked at $NSHA"
    [ "$PK_CPP_BLOB" != "$GF_CPP_BLOB" ] || stop "packer v3: $GF_CPP at $NSHA is v2's blob (no C++ worker)"
    git -C "$NOSI" show "$PK_CPP_BLOB" | grep -q 'mod.attr("VERSION") = 3;' || stop "packer v3: $GF_CPP at $NSHA is not VERSION 3"
    git -C "$NOSI" show "$PK_CPP_BLOB" | grep -q 'mod.attr("DESC") = 2;' ||                    # [f3ldx] the lane row
        stop "packer v3: $GF_CPP at $NSHA is not DESC 2 (no lane row on the X pool)"
    PK_SRC="$NSHA (packer v3 of --nosi, DESC 2)"
fi
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
if [ -n "$ACC" ]; then                                      # [acc216] the gold continuation, pinned as in r2_evalacc_step.sh
    gl=$(sha256sum "$GOLD/ctx65280_tokens.bin" 2>/dev/null) || stop "no $GOLD/ctx65280_tokens.bin"
    [ "${gl%% *}" = "$GOLD_SHA" ] || stop "ctx65280_tokens.bin sha256 is not $GOLD_SHA"
fi
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
    [ -z "$ACC" ] || echo "R2 SG5 step: [acc216] modes $MODES; gold $GOLD (sha256 $GOLD_SHA); link ${LINKARG[*]:-none}"
    [ -z "$GF3" ] || echo "R2 SG5 step: [f3ld] packer copied: $PK_SRC: $GF_PY blob $PK_PY_BLOB, $GF_CPP blob $PK_CPP_BLOB -> $GF_DIR"
    echo "R2 SG5 step: nosi root $NOSI; STEP_MIN $STEP_MIN; MIN_LEFT $MIN_LEFT s"                   # [fusedres2]
    echo "R2 SG5 step: claim $CLAIM (one per holder)"                                               # [sg5par]
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
mkdir "$CLAIM" 2>/dev/null || stop "$CLAIM exists (another step in holder $JOB, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id ($CLAIMNAME stays)"
mkdir "$REC" && mkdir "$GF_DIR" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT ($CLAIMNAME stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIMNAME stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD ($CLAIMNAME stays)"
git -C "$NOSI" show "$PK_PY_BLOB" > "$GF_DIR/gilfree_pack.py" && [ "$(git hash-object -- "$GF_DIR/gilfree_pack.py")" = "$PK_PY_BLOB" ] ||
    stop "cannot copy the packer's blob $PK_PY_BLOB ($CLAIMNAME stays)"                              # [f3ld] PK_*: v2 or v3
git -C "$NOSI" show "$PK_CPP_BLOB" > "$GF_DIR/gilfree_pack.cpp" && [ "$(git hash-object -- "$GF_DIR/gilfree_pack.cpp")" = "$PK_CPP_BLOB" ] ||
    stop "cannot copy the packer's blob $PK_CPP_BLOB ($CLAIMNAME stays)"
printf 'packer_copied=%s py=%s cpp=%s\n' "$PK_SRC" "$PK_PY_BLOB" "$PK_CPP_BLOB" > "$REC/packer.txt" ||
    stop "cannot write $REC/packer.txt ($CLAIMNAME stays)"                                           # [f3ld]
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIMNAME stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIMNAME stays)"
printf 'biattn=%s:%s blob=%s packer=%s py=%s cpp=%s threads=%s transport=%s sweep=%s splits=%s cell=%s modes=%s backends=%s C=%s X=%s R_ver=%s eq_pool=%s variants=%s xpool=%s batch=%s steps=%s mem=%s stack=%s\n' \
    "$BIATTN_SHA" "$BIATTN_PATH" "$BIATTN_BLOB" "$GF_SHA" "$GF_PY_BLOB" "$GF_CPP_BLOB" "$THREADS" "$TRANSPORT" \
    "${SWEEP_RUN:-none}" "$SPLITS" "${CELL:-default}" "$MODES" "$BACKENDS" "$PC" "$PX" "$PRV" "$EQ_POOL" "${VARIANTS:-none}" \
    "${XPOOL:-none}" "$BATCH" "$STEPS" "$MEMG" "${STACKARG[*]:-none}" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIMNAME stays)"
printf 'nosi_root=%s step_min=%s steps=%s draft_ticks=%s frarg=%s evalarg=%s claim=%s\n' "$NOSI" "$STEP_MIN" "$STEPS" \
    "$DTICKS" "${FRARG[*]:-none}" "${EVALARG[*]:-none}" "$CLAIM" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIMNAME stays)"                                                     # [fusedres2]
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
    rmdir "$CLAIM" || ok=0
fi
echo "R2 SG5 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
