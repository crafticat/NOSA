"""R2 step 7, PIECE 4: SG5, the R-PERF BUDGET of the flattened tick on piece 5's Residency (docs/R2_STEP7_TICK_SG5.md;
docs/R2_STEP7_SPEC_ROUND.md 1.2 and 7.3; C's answers Q-S1 .. Q-S5, 2026-10-21). REPORT only: the 8 % design target is a
target, not a gate; the MUST gates below guard the run's validity. Holder step: benchmarks/Efficiency/r2_tick_sg5_step.sh.

THE RUN (B documents of pg19-test-c16k, all S tokens, the same documents for every arm; --repeat-docs tiles the 94
documents when B > 94, labelled):
  SHIPPED  in a CHILD process each (its memory freed at exit; NOSI_POOL_BLOCKS is read at import): the shipped decode
           (batch prefill, decode_inference without an R2 state, NOSI_ATTN_SPLITS 4), --shipped-steps steps after 2
           warm-ups, CUDA events per step: T_shipped. With --pool-baseline also at the EQUAL MEMORY of the primary
           point: NOSI_POOL_BLOCKS = C + X + 4 - 64 (C, Q-S5).
  BUILD    ONE build_ragged (body r2), the R2 path's widths from the horizon; its GPU windows are freed (the tick uses
           its own store); its pinned host rows are the Residency's host rows; its twins are copied into every engine
           (TickEngine.from_r2(hist_slots = C + X): the store of C + X historical slots + 4 tail pages per stream).
  per GEMM mode (--modes: exact = G-b PaddedLinear(4B), fast = natural cuBLAS) and per residency backend (--backends:
  sc = ResidencySC, sb = ResidencySB, v1 = Residency; C's 2ec08da; [b2] b2 = OwnResidency, B's array-backed ownership
  table on the store's device over S-C's device path, unchecked unless --b2-checked: r2_own2.py), at the PRIMARY
  point (--C, --X, --R-ver):
    T_d    the DRAFT-ONLY tick: --draft-ticks ticks of B draft rows (each draft key decided when its cohort's positions
           are drafted), per-op CUDA events.
    TRANSPORT  --transport gilfree (D5, the default: C's adapter over agent E's packer v2) or gather (GatherTransport
           below: NOSI's Triton gather on a side stream), so one transport's failure does not hide the other terms.
    PREFLIGHT  before the long arms, a short round (--preflight-steps tokens) per backend on the run's transport
           (MUST; a failing backend is left out of the long arms) and on gather too when the run's is gilfree (REPORT).
    ROUND  the free-running greedy round (r2_tick.run_round) to --steps committed tokens per request, per-op CUDA
           events (TickTimer): T_tick per tick, r = committed tokens / request-ticks, r x T_shipped / T_tick, the
           budget dGEMM / dATT / dSEL / dTWIN / dSTORE / dRES (tick minus T_d, per category), the host ms per
           category (rows, backend, read: dHOST per backend), the residual (T_tick minus the categorized device time:
           uncategorized ops + device idle), the service's 4.3 / 4.6 / 4.8 report (4.3 layer_wait_ns: the superset
           in-flight wait of S-C, C's Q6), memory (peak retained, slots), refusals, masked fraction, decision reads,
           void ticks, the twins' step cost (the 'twin' category: both shape-stable means every step).
    BARE   with --bare, the same round with only the tick events (the instrumentation's own cost).
  SWEEP    --sweep C:X,... (R_ver = X / 2) with the first mode and the FIRST BACKEND OF --backends THAT PASSED ITS
           PREFLIGHT (recorded as sweep_backend; none passed: a REPORT "sweep skipped: no backend passed"; p3 took the
           first backend of the list and so skipped the whole sweep when S-C's preflight failed), --sweep-steps tokens:
           the same report.
MUST gates: every run completes (no exception; a CapacityRefusal is reported with its configuration), no void tick,
the twins' counters equal the host mirrors after each round, the gilfree team ran on the requested threads.
[xpool] THE SHARED X POOL LAYOUT (step S6, opt-in; nosi/nosi/r2_xpool.py, docs/R2_XPOOL_PLAN.md): --layout stream (the
default: every arm as before) | pool (the store of B (C + 4) + X_pool positions per layer; --X-pool, default X; sc and
b2 only) | both (each point runs three arms, stream, pool, stream2, labelled <label>_<arm>; needs --arrival-lag >= 1
and no sweep; the preflight runs the pool layout; one_point_both: MUST gates <label>_XP_replica (stream2 = stream),
<label>_XP_committed, <label>_XP_bitwise (rows and per-row logit fingerprints; VOID without the replica); REPORTS
<label>_XP_peak and <label>_XP_tick against the mean of the two stream arms). --arrival-lag D (default 0 = free
timing): the residency's fixed arrival lag (note 12k) for every engine of the run. --pool-rule stream (default:
NOSI_POOL_BLOCKS = C + X + 4 - 64) | k5 (r2_xpool.equal_memory_pool: C + 4 - 64 + floor(X_pool / B)) | both.
[integ] THE STACK (docs/R2_INTEG_PLAN.md, opt-in): --attn shared[-whole] (+ --union, --sharedkv; sc / b2 page keys;
needs --sel-check or an explicit --shared-unchecked), --gilfree-warm-up on|off (default on: the transport's default),
--stack full | full-no-<c> | none (presets of twin, rebuild, b2 resolve, layout, attn, warm-up; an explicit component
flag next to --stack is refused), --stack-compare (arms none, <stack>, none2: MUST <label>_ST_replica /
_ST_committed / _ST_bitwise), --shipped-batch (the shipped decode at another batch: ratio None, tok_per_s reported).
[bigb] LARGE B (docs/R2_BIGB_PLAN.md; bigb_core.py): --log-slots 0 (default: the engine's 1 << 16) | -1 (bigb_core.
log_slots_for the round) | N; --skip-shipped (opt-in: no shipped child, a REPORT gate R_shipped_skipped). The preflight
and one_point now release the round (a TickRound holds its engine) with the engine, so no two engines coexist (before:
the preflight's first engine lived while the second was built: p5b2's 67.49 GB peak was two engines). REPORTS:
meta.build_peak_gb, meta.preflight_peak_gb, each point's peak_gb_cum, meta.log_slots.
[evalsplit] EVAL CELL (a), THE FAST TIMING HEADLINE (human decision part 3, 2026-10-22; opt-in --eval-role fast-time):
the R2 engine end to end with natural cuBLAS (FAST) in draft and verify rows, timed only when the accuracy record of
eval cell (b) (r2_evalacc.py: FAST vs EXACT EQUIVALENT, every MUST gate passed) exists: --accuracy-record (a refusal,
exit 2, otherwise). Needs --modes fast[,...] (the first point is the headline), --backends b2, --layout pool, --bare,
--mem-cap-gb > 0; no --stack, no sweep. Per point "eval": EFFECTIVE tok/s = committed / the round's wall clock and
STEADY tok/s = committed / (ticks x the median (and mean) tick), SEPARATELY (human, 2026-10-21), from the bare round
(the headline) and from the instrumented round (REPORT); prefill and engine build excluded. MUST EF_headline_fast,
EF_bare_<label> (the bare round committed B (steps - 1) tokens), EF_mem_cap (meta.peak_gb <= --mem-cap-gb),
EF_accuracy_record; REPORT R_EF_<label> and R_EF_bands (--eval-bands against the headline).
OUTPUT: --out/summary.json (after every run). The last stdout line is R2_TICK_SG5_DONE failed=<n>; exit = n (2 =
refused). ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0 (the parent; a child sets its own).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_gate as G0  # noqa: E402  (load_workload)
import r2_g0bi_gate as GB  # noqa: E402  (load_by_path, free, gate)

BACKENDS = ("sc", "sb", "v1", "b2")                                                     # [b2] b2: OwnResidency
MODES = ("exact", "fast", "bi_c_strict", "bi_c_bucket")      # [bigemmc] G-c strict / research arm: opt-in only
TRANSPORTS = ("gilfree", "gather")
LAYOUTS = ("stream", "pool", "both")                                                    # [xpool]
XPOOL_BACKENDS = ("sc", "b2")                                                           # [xpool] S-C's device path
_PEAK = [0]                                                                             # [xpool] the max before resets
# [integ] the shared-KV selector in the SG5 tick (docs/R2_SHAREDKV_PLAN.md; r2_tick_gate.select_attn's choices that
# drive the tick: the probes stay in r2_tick_gate.py)
ATTN_CHOICES = ("per_row", "shared", "shared-whole")
KEY_BACKENDS = ("sc", "b2")                                                             # the plans that carry page keys
# [integ] THE STACK (docs/R2_INTEG_PLAN.md): every gap-closing build at once. Each component has an OFF value (the
# pre-integration path, p5b2's code) and an ON value; --stack full turns every one ON, full-no-<c> all but <c>.
STACK_COMPONENTS = ("twin", "rebuild", "resolve", "layout", "attn", "warmup")
STACK_ON = dict(twin=dict(twin="fused"), rebuild=dict(rebuild="batched"), resolve=dict(b2_resolve="fused"),
                layout=dict(layout="pool"), attn=dict(attn="shared", union="fused"),
                warmup=dict(gilfree_warm_up="on"))
STACK_OFF = dict(twin=dict(twin="torch"), rebuild=dict(rebuild="per_call"), resolve=dict(b2_resolve="torch"),
                 layout=dict(layout="stream"), attn=dict(attn="per_row", union="torch"),
                 warmup=dict(gilfree_warm_up="off"))
STACK_CHOICES = ("", "none", "full") + tuple("full-no-" + c for c in STACK_COMPONENTS)
STACK_BASES = ("none",) + tuple("only-" + c for c in STACK_COMPONENTS)   # [integ6] --stack-compare's base arms
# the flags a stack sets: each must be at its parse default when --stack is given (never silently overridden)
STACK_FLAG_DEFAULTS = dict(twin="torch", variants="", b2_resolve="torch", layout="stream", attn="per_row",
                           union="torch", gilfree_warm_up="on")
# [hostfix] --host-profile: the host sub-timers per tick (TickTimer.host, ms per tick in the record; host clocks only,
# no device read). DIAG_tick_host.md section 1.2 names each item; "h_" + the part. The engine's (r2_tick.py), the
# service's (r2_residency_sc.py / r2_own2.py, through svc.host_prof) and the wait's (CudaDeviceOps.wait).
HOST_PROF_CATS = dict(
    h_plan="run_round: plan_tick (item 0; before the tick's span)",
    h_begin_tick="run_rows: backend.begin_tick = the service's poll (item 1)",
    h_poll_arrivals="B2 poll: _poll_arrivals (item 1a)", h_poll_drops="B2 poll: _poll_drops (1b)",
    h_poll_releases="B2 poll: _poll_releases (1c)", h_poll_memory="B2 poll: metrics.sample_memory (1d)",
    h_row_table="run_rows: row_table (item 2)", h_prepare="run_rows: backend.prepare (item 3)",
    h_layers="run_rows: the 32 layers' host time, embedding to lm_head (items 4-6: launches, pages, waits)",
    h_pages="per layer, summed: backend.pages = resolve + translate (5b, 5c, 5d)",
    h_superset_wait="per verify layer, summed: CudaDeviceOps._superset_wait (5c)",
    h_after_attention="per layer, summed: backend.after_attention = read_done (5e)",
    h_end_tick="run_rows: backend.end_tick (item 7)",
    h_et_apply="end_tick: apply_tick = the decision read + the B2 consumer (7a-7f)",
    h_apply_read="B2 apply_tick: the ONE decision read and its parse (7a)",
    h_b2_ops="B2 _apply_b2: parse .. read_done, device ops and their syncs (7b)",
    h_hs_victims="B2 _host_side: Policy E victims (7c)", h_hs_takes="B2 _host_side: the takes loop (7d)",
    h_hs_merges="B2 _host_side: demand merges, refusals, recency journal (7e)",
    h_hs_looks="B2 _host_side: the 4.3 / 4.6 looks and the release queue (7f)",
    h_hs_plans="B2 _host_side: the layer plans (masked, pairs, refusals)",
    h_et_issue="end_tick: Residency.end_tick = assign_phys + the draft issues (7g, 7h)",
    h_et_memory="end_tick: metrics.sample_memory (7i)",
    h_finish_tick="run_round: finish_tick = read_tokens + the host decisions (item 9)",
    h_apply_events="run_round: apply_events (item 10)",
    h_on_tick="run_round: on_tick (the gates' callback; outside the tick's span)",
    h_end_request="run_round: end_request loop (item 11; outside the tick's span)")
# [hostfix] --lean-time (item 2): what a time cell drops. REPORT-only work: no decision, page table, transfer, release
# or MUST-gate input depends on it (test_r2_hostfix.py HF2). B2 only. KEPT: the decisions and their refusals, the
# release queue (_queue), the transfers and their demand sets, the journal, masked / draft pairs / refusals of the
# plans, the bookkeeping (twin counters vs mirrors), the transport's stats, the X pool's invariant checks.
LEAN_DROPPED = (
    "Metrics.issued / Metrics.arrived: the 4.8 lifetimes, transferred and arrival_ns dicts (one record per take)",
    "the 4.8 read-count harvest of the take slots (their old generation / read counts: one stack and one read)",
    "the 4.3 / 4.6 looks: open_call, k43, k46, the firsts lists and their reads (the release queue is KEPT, same calls)",
    "metrics.sample_memory at poll and at end_tick (two table reductions and two reads per tick; peak_retained)",
    "the draft_refused REPORT count (one read per tick)",
    "the X-pool witness (lane handoffs, peak lanes: XPool.witness False), as in a --stack-role time cell")


def log(msg):
    print("[r2_tick_sg5 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy; v2 7c826c1)")
    ap.add_argument("--gilfree", required=True, help="the directory of the pinned gilfree_pack.py / .cpp (E's f2b981a)")
    ap.add_argument("--gilfree-build", required=True, help="the extension's build directory")
    ap.add_argument("--threads", type=int, default=8, help="the gilfree packer's OpenMP team")
    ap.add_argument("--transport", default="gilfree", choices=TRANSPORTS,
                    help="gilfree (D5, the default) or gather (GatherTransport: NOSI's Triton gather on a side stream)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--repeat-docs", action="store_true")
    ap.add_argument("--steps", type=int, default=48, help="committed tokens per request in a round")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--m-fix", type=int, default=0, help="G-b's row count; 0 = (1 + K) B")
    ap.add_argument("--modes", default="exact,fast")
    ap.add_argument("--backends", default="sc,sb,v1")
    ap.add_argument("--C", type=int, default=96, help="C's recommendation after holder 2190876 (132 slots, 17.79 GB)")
    ap.add_argument("--X", type=int, default=32)
    ap.add_argument("--R-ver", type=int, default=16)
    ap.add_argument("--sweep", default="", help="C:X,... (R_ver = X / 2); empty: none")
    ap.add_argument("--sweep-steps", type=int, default=24)
    ap.add_argument("--draft-ticks", type=int, default=12)
    ap.add_argument("--shipped-steps", type=int, default=24)
    ap.add_argument("--pool-baseline", action="store_true", help="also the shipped decode at the primary point's memory")
    ap.add_argument("--bare", action="store_true", help="also each round with the tick events only")
    ap.add_argument("--preflight-steps", type=int, default=4,
                    help="PREFLIGHT: a short round per backend and transport before the long arms (0: none)")
    ap.add_argument("--twin", default="torch", choices=("torch", "fused"),                       # [twin-variant]
                    help="the twins' unchecked step: torch ops (default) or r2_twin_fused (one kernel per layer)")
    ap.add_argument("--variants", default="",                                                    # [batched-rebuild]
                    help="twin/rebuild,... (twin torch|fused, rebuild per_call|batched): one arm per variant at every "
                         "(mode, backend), labelled SG5_<mode>_<backend>_<twin>_<rebuild>; empty (the default): "
                         "the one variant --twin/per_call with p3's labels")
    ap.add_argument("--b2-checked", action="store_true",                                         # [b2]
                    help="b2: every B2 predicate (checked mode); default: unchecked (the timed path)")
    ap.add_argument("--b2-resolve", default="torch", choices=("torch", "fused", "check"),        # [fusedres]
                    help="b2: the per-layer resolve: S-C's torch ops (default), r2_fusedres (fused), or fused with "
                         "the torch path compared at every layer (check: an equality gate, not a timing)")
    ap.add_argument("--b2-fused-plant", type=int, default=0, choices=(0, 1, 2),                 # [fusedres2] M2
                    help=argparse.SUPPRESS)                     # NEGATIVE CONTROL (FR-g0p): r2_fusedres.PLANTS
    ap.add_argument("--b2-res-profile", action="store_true",                                    # [fusedres2] M3
                    help="b2: torch.profiler over the draft-only phase; the device time per draft tick of the kernels "
                         "launched inside the 't.res' spans (r2_fusedres_prof.py); its spans are NOT timings")
    ap.add_argument("--layout", default="stream", choices=LAYOUTS,                               # [xpool]
                    help="the store: stream (per-stream, the default), pool (the shared X pool), both (compare)")
    ap.add_argument("--X-pool", type=int, default=-1,                                            # [xpool]
                    help="the pool's positions per layer (H lanes each); -1: X (K5's own bound)")
    ap.add_argument("--pool-rule", default="stream", choices=("stream", "k5", "both"),          # [xpool]
                    help="the equal-memory shipped baseline's NOSI_POOL_BLOCKS rule (with --pool-baseline)")
    ap.add_argument("--arrival-lag", type=int, default=0,                                        # [xpool] review B1
                    help="the residency's fixed arrival lag D (note 12k; 0: free timing, the default); --layout both "
                         "needs D >= 1, so the arms are comparable tick by tick")
    ap.add_argument("--attn", default="per_row", choices=ATTN_CHOICES,                           # [integ] sharedkv
                    help="the tick's attention: per_row = piece 3 (default, unchanged); shared[-whole] = the "
                         "shared-read kernel (bi_paged_attention_shared.SharedKV) on the plan's page keys (sc / b2)")
    ap.add_argument("--union", default="torch", choices=("torch", "fused"),                     # [integ] sharedkv3
                    help="the shared side's union build: torch (build_union) or fused (one kernel per call)")
    ap.add_argument("--sharedkv", default=os.path.join(os.path.dirname(os.path.dirname(HERE)),   # [integ]
                                                       "nosi", "nosi", "bi_paged_attention_shared.py"),
                    help="the shared-read module (loaded by path; it imports nothing from nosi)")
    ap.add_argument("--sel-check", action="store_true",                                         # [integ]
                    help="TickEngine(check=True): the selection's last two blocks T - 1, T are checked every layer "
                         "(one host read per layer; the shared union build assumes it: sharedkv4 MINOR-3)")
    ap.add_argument("--shared-unchecked", action="store_true",                                  # [integ]
                    help="allow --attn shared* WITHOUT --sel-check (a timing arm; the precondition must be checked "
                         "in another invocation of the same cell; recorded in meta)")
    ap.add_argument("--gilfree-warm-up", default="on", choices=("on", "off"),                   # [integ] gfw1f
                    help="GilfreeTransport(warm_up=...): on = the transport's production default since 000870e; "
                         "off = the pre-warm-up path (p5b2's code; gfw1f's control)")
    ap.add_argument("--stack", default="", choices=STACK_CHOICES,                                # [integ]
                    help="a preset of the gap-closing builds (docs/R2_INTEG_PLAN.md): full = twin fused, rebuild "
                         "batched, b2 resolve fused, layout pool (X_pool from --X-pool), attn shared split + union "
                         "fused, gilfree warm-up; full-no-<c> = full minus one component; none = every one OFF")
    ap.add_argument("--stack-base", default="none", choices=STACK_BASES,                        # [integ6]
                    help="--stack-compare's base arms (none and none2): none = every component OFF (the default); "
                         "only-<c> = only component c ON (only-layout: the pool layout as the reference where the "
                         "per-stream store cannot be built, e.g. X 896 at B 128)")
    ap.add_argument("--stack-compare", action="store_true",                                      # [integ]
                    help="three arms per point: none (every component OFF), the --stack arm, none again (the "
                         "replica); MUST gates <label>_ST_replica / _ST_committed / _ST_bitwise (needs "
                         "--arrival-lag >= 1)")
    ap.add_argument("--stack-role", default="", choices=("", "validate", "time"),               # [integ5]
                    help="validate: the equality cells (needs --stack-compare; the witnesses ON); time: the timing "
                         "cells (needs --validated-by, --arrival-lag >= 1 and --bare; no --sel-check, no "
                         "--stack-compare; the diagnostic witnesses OFF: XPool.witness False, SharedKV.count_saved "
                         "False; the pool's invariant checks stay ON, they are the production safety refusal)")
    ap.add_argument("--validated-by", default="",                                               # [integ5]
                    help="comma list of the passing validation cells' summary.json (stack1a, stack1a128); a time "
                         "cell refuses unless each exists and every MUST gate in it passed, the ST replica and "
                         "bitwise gates included; recorded in meta.validated_by")
    ap.add_argument("--free-arm", action="store_true",                                          # [integ5]
                    help="time cells: also one REPORT arm at free arrival (--arrival-lag 0) after the headline arm, "
                         "MUST only on the lag-free invariants (completed, bookkeeping / void ticks 0, err flags, "
                         "acted); bitwise equality is proven only at the validated lag")
    ap.add_argument("--require-xregion", action="store_true",                                   # [integ3] M2
                    help="make <label>_ST_xregion MUST: the pool's per-(layer, head) peak lanes exceed B x C (the "
                         "shared X region was bound); default REPORT")
    ap.add_argument("--stack-bands", default="",                                                # [integ3] M3
                    help="PEAK_LO:PEAK_HI:TICK_LO_MS:TICK_HI_MS: after the run, <out>/stack_bands.txt holds the "
                         "verdict of meta.peak_gb and the point's T_tick_ms against the bands (the combined cell "
                         "stack1 reads it before stack1b); REPORT gate R_stack_bands")
    ap.add_argument("--shipped-batch", type=int, default=0,                                     # [integ]
                    help="the shipped decode's batch (0: --batch); another value makes 'ratio' None and reports "
                         "tokens per second on both sides")
    ap.add_argument("--log-slots", type=int, default=0,                                          # [bigb]
                    help="the engines' token-log slots: 0 (default) the engine's 1 << 16; -1 bigb_core.log_slots_for("
                         "B, max(steps, sweep-steps), K, LAMBDA); N > 0: N")
    ap.add_argument("--skip-shipped", action="store_true",                                       # [bigb]
                    help="no shipped child (T_shipped None, ratio None; a REPORT gate says so); not with "
                         "--pool-baseline")
    ap.add_argument("--eval-role", default="", choices=("", "fast-time"),                     # [evalsplit]
                    help="fast-time: eval cell (a), the FAST timing headline (module docstring); default: none")
    ap.add_argument("--accuracy-record", default="",                                           # [evalsplit]
                    help="eval cell (b)'s summary.json (r2_evalacc.py): FAST vs EXACT EQUIVALENT, every MUST passed")
    ap.add_argument("--mem-cap-gb", type=float, default=0.0,                                   # [evalsplit]
                    help="the declared device memory cap (GB = 1e9 B, max allocated): MUST EF_mem_cap")
    ap.add_argument("--eval-bands", default="",                                                # [evalsplit]
                    help="SMED_LO:SMED_HI:SMEAN_LO:SMEAN_HI:EFF_LO:EFF_HI tok/s: REPORT R_EF_bands on the headline")
    ap.add_argument("--host-profile", action="store_true",                                     # [hostfix] item 0
                    help="host sub-timers per tick (HOST_PROF_CATS), the feeder thread's Python time per release and "
                         "gc.callbacks (collection start / duration per tick) in the timed rounds; host clocks only")
    ap.add_argument("--gc-freeze", action="store_true",                                       # [hostfix] item 1
                    help="every timed phase (draft-only, round, bare round): gc.collect(), gc.freeze(), gc.disable() "
                         "before it; gc.unfreeze(), gc.enable(), one gc.collect() after it (round_guard); the counts in "
                         "each point's gc_freeze record")
    ap.add_argument("--lean-time", action="store_true",                                       # [hostfix] item 2
                    help="b2 only: skip the REPORT-only host work in every engine (LEAN_DROPPED: the 4.3 / 4.6 / 4.8 "
                         "metrics, the memory samples, the X-pool witness); the decisions and every MUST input stay")
    ap.add_argument("--wait-index", action="store_true",                                      # [hostfix] item 3
                    help="b2 only: the in-flight map keeps a per-layer index (r2_residency_sc.InflightByLayer), so the "
                         "superset wait visits its layer's transfers only (the same waited events, in the same order)")
    ap.add_argument("--b2-fewer-reads", action="store_true",                                  # [hostfix] item 5
                    help="b2 only: the consumer's host-known values in one pinned non-blocking copy and its device "
                         "values in one read per place (OwnResidency(fewer_reads=True)); the same values")
    ap.add_argument("--baseline-only", type=int, default=-1, help=argparse.SUPPRESS)   # the child: NOSI_POOL_BLOCKS
    ap.add_argument("--baseline-out", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.modes = [m for m in a.modes.split(",") if m]
    a.backends = [b for b in a.backends.split(",") if b]
    if not a.modes or any(m not in MODES for m in a.modes) or not a.backends or any(b not in BACKENDS for b in a.backends):
        ap.error("--modes in %s, --backends in %s" % (MODES, BACKENDS))
    a.sweep_pts = []
    for p in [p for p in a.sweep.split(",") if p]:
        c, x = p.split(":")
        a.sweep_pts.append((int(c), int(x), int(x) // 2))
    if a.m_fix == 0:
        a.m_fix = (1 + a.K) * a.batch
    if a.m_fix < (1 + a.K) * a.batch or a.C < 1 or a.X < 0 or not 0 <= a.R_ver <= a.X:
        ap.error("m_fix >= (1 + K) B, C >= 1, 0 <= R_ver <= X")
    if a.S < 64 * 64 or a.steps < 4 or a.draft_ticks < 2 or a.shipped_steps < 2:
        ap.error("S >= 4096, steps >= 4, draft-ticks >= 2, shipped-steps >= 2")
    if a.stack:                                                                                  # [integ]
        given = sorted(k for k, v in STACK_FLAG_DEFAULTS.items() if getattr(a, k) != v)
        if given:
            ap.error("--stack %s sets the components itself: do not pass %s" % (a.stack, ", ".join(
                "--" + k.replace("_", "-") for k in given)))
        apply_stack(a, a.stack)
    elif a.stack_compare:
        ap.error("--stack-compare needs --stack")
    a.variant_list = parse_variants(a.variants, a.twin)                                          # [batched-rebuild]
    if a.variant_list is None:                                                                   # [batched-rebuild]
        ap.error("--variants: twin/rebuild,... with twin in torch|fused and rebuild in per_call|batched, distinct")
    if a.stack:                                                                                  # [integ] the preset's
        a.variant_list = [(a.twin, a.stack_rebuild)]                                             # one variant
    a.rebuild = a.variant_list[0][1]                                                             # [batched-rebuild]
    if a.attn != "per_row":                                                                      # [integ] sharedkv
        if any(b not in KEY_BACKENDS for b in a.backends):
            ap.error("--attn %s needs page keys: --backends in %s (v1 / sb give none)" % (a.attn, KEY_BACKENDS))
        if not a.sel_check and not a.shared_unchecked:
            ap.error("--attn %s needs --sel-check (the union build assumes the selection's tail T - 1, T: sharedkv4 "
                     "MINOR-3), or an explicit --shared-unchecked for a timing arm" % a.attn)
    if a.b2_resolve != "torch" and any(b != "b2" for b in a.backends):                          # [integ]
        ap.error("--b2-resolve %s (or a --stack with it) runs on --backends b2 only" % a.b2_resolve)
    if a.stack_compare and (a.layout == "both" or a.arrival_lag < 1 or a.sweep_pts or a.bare):  # [integ]
        ap.error("--stack-compare: no --layout both, no sweep, no --bare, and --arrival-lag >= 1 (free timing makes "
                 "two runs differ, note 12k)")
    if a.shipped_batch < 0:
        ap.error("--shipped-batch >= 0")
    a.stack_bands_v = None                                                                       # [integ3] M3
    if a.stack_bands:
        try:
            v = [float(x) for x in a.stack_bands.split(":")]
        except ValueError:
            v = []
        if len(v) != 4 or not (v[0] < v[1] and v[2] < v[3]) or not a.stack or a.stack_compare:
            ap.error("--stack-bands PEAK_LO:PEAK_HI:TICK_LO_MS:TICK_HI_MS needs --stack without --stack-compare")
        a.stack_bands_v = v
    if a.require_xregion and not (a.stack and a.layout == "pool"):                             # [integ3] M2
        ap.error("--require-xregion needs a --stack with the pool layout")
    if a.stack_role and not a.stack:                                                             # [integ5]
        ap.error("--stack-role needs --stack")
    if a.stack_base != "none" and not a.stack_compare:                                          # [integ6]
        ap.error("--stack-base needs --stack-compare")
    if a.stack_base == "only-layout" and a.layout != "pool":
        ap.error("--stack-base only-layout compares pool-layout arms: the --stack must keep the pool layout")
    if a.stack_role == "validate" and not a.stack_compare:
        ap.error("--stack-role validate needs --stack-compare")
    if a.stack_role == "time":
        if a.sel_check or a.stack_compare or not a.validated_by or a.arrival_lag < 1 or not a.bare \
                or a.require_xregion:
            ap.error("--stack-role time: --validated-by, --arrival-lag >= 1 (the validated schedule) and --bare; no "
                     "--sel-check, --stack-compare or --require-xregion (the witnesses are OFF here)")
    if a.free_arm and a.stack_role != "time":
        ap.error("--free-arm needs --stack-role time")
    if a.validated_by and a.stack_role != "time":
        ap.error("--validated-by needs --stack-role time")
    if a.layout != "stream" and any(b not in XPOOL_BACKENDS for b in a.backends):                # [xpool]
        ap.error("--layout %s: the shared pool runs on %s only (S-C's device path binds the lanes)"
                 % (a.layout, XPOOL_BACKENDS))
    if a.X_pool != -1 and (a.layout == "stream" or a.X_pool < a.X):                              # [xpool]
        ap.error("--X-pool needs --layout pool / both and X_pool >= X (K5's bound per head)")
    if a.arrival_lag < 0 or (a.layout == "both" and a.arrival_lag < 1):                         # [xpool] review B1
        ap.error("--arrival-lag >= 0; --layout both needs --arrival-lag >= 1 (free timing makes two runs differ, "
                 "note 12k)")
    if a.layout == "both" and a.sweep_pts:                                                       # [xpool] review m4
        ap.error("--layout both runs no sweep (the sweep's points would run the pool arm only)")
    if a.log_slots < -1:                                                                         # [bigb]
        ap.error("--log-slots: 0 (the engine's default), -1 (log_slots_for) or a positive size")
    if a.skip_shipped and a.pool_baseline:                                                       # [bigb]
        ap.error("--skip-shipped runs no shipped child, so no --pool-baseline either")
    a.eval_bands_v = None                                                                        # [evalsplit]
    if a.eval_role == "fast-time":
        if a.modes[0] != "fast" or a.backends != ["b2"] or a.layout != "pool" or not a.bare or a.mem_cap_gb <= 0 \
                or not a.accuracy_record or a.stack or a.stack_compare or a.sweep_pts:
            ap.error("--eval-role fast-time: --modes fast[,...] (the headline first), --backends b2, --layout pool, "
                     "--bare, --mem-cap-gb > 0 and --accuracy-record; no --stack, --stack-compare or --sweep")
        if a.eval_bands:
            try:
                v = [float(x) for x in a.eval_bands.split(":")]
            except ValueError:
                v = []
            if len(v) != 6 or not all(v[i] < v[i + 1] for i in (0, 2, 4)):
                ap.error("--eval-bands SMED_LO:SMED_HI:SMEAN_LO:SMEAN_HI:EFF_LO:EFF_HI (each LO < HI)")
            a.eval_bands_v = v
    elif a.accuracy_record or a.mem_cap_gb or a.eval_bands:
        ap.error("--accuracy-record, --mem-cap-gb and --eval-bands need --eval-role fast-time")
    if a.lean_time and (a.backends != ["b2"] or a.stack_compare or a.stack_role == "validate"):   # [hostfix] item 2
        ap.error("--lean-time: --backends b2 only (OwnResidency(lean=True)), and not in a validation cell "
                 "(--stack-compare / --stack-role validate need the witnesses)")
    if a.wait_index and a.backends != ["b2"]:                                                    # [hostfix] item 3
        ap.error("--wait-index: --backends b2 only (OwnResidency(wait_index=True))")
    if a.b2_fewer_reads and a.backends != ["b2"]:                                                # [hostfix] item 5
        ap.error("--b2-fewer-reads: --backends b2 only (OwnResidency(fewer_reads=True))")
    a.log_slots_n = resolve_log_slots(a)                                                         # [bigb]
    return a


def resolve_log_slots(a):                                                                       # [bigb]
    """[bigb] The engines' token-log size: None (the engine's default 1 << 16) for --log-slots 0; for -1 the round's
    size bigb_core.log_slots_for(B, max(steps, sweep-steps, preflight-steps), K, LAMBDA); else the given size."""
    if a.log_slots == 0:
        return None
    if a.log_slots > 0:
        return int(a.log_slots)
    import bigb_core
    return bigb_core.log_slots_for(a.batch, max(a.steps, a.sweep_steps, a.preflight_steps), a.K, a.LAMBDA)


def pool_positions(a, X):                                                                       # [xpool]
    """[xpool] The pool's positions per layer at a point with K5 bound X: --X-pool, or X when it is -1."""
    return X if a.X_pool < 0 else a.X_pool


def pool_rule_value(a, rule):                                                                   # [xpool]
    """[xpool] NOSI_POOL_BLOCKS of the equal-memory shipped baseline at the primary point under `rule` (before max(0,
    .)): "stream" C + X + 4 - 64; "k5" C + 4 - 64 + floor(X_pool / B) = r2_xpool.equal_memory_pool's k5 rule with
    pool_lanes = H X_pool (H cancels: a position has H lanes, one per head; review m5). Rounded DOWN (the audit's Q4.4
    wrote ceil; at x1 both give 37). The CPU test XP4 pins the two together; the parent does not import nosi here."""
    if rule == "stream":
        return a.C + a.X + 4 - 64
    return a.C + 4 - 64 + pool_positions(a, a.X) // a.batch


def stack_values(name):                                                                         # [integ]
    """[integ] The attribute values of a stack: "none" every component OFF, "full" every one ON, "full-no-<c>" every
    one ON except <c>. Returns a dict of argparse attributes (twin, stack_rebuild, b2_resolve, layout, attn, union,
    gilfree_warm_up)."""
    if name == "none":
        on = set()
    elif name == "full":
        on = set(STACK_COMPONENTS)
    elif name.startswith("full-no-") and name[len("full-no-"):] in STACK_COMPONENTS:
        on = set(STACK_COMPONENTS) - {name[len("full-no-"):]}
    elif name.startswith("only-") and name[len("only-"):] in STACK_COMPONENTS:      # [integ6] one component ON
        on = {name[len("only-"):]}
    else:
        raise ValueError("stack %r (one of %s)" % (name, STACK_CHOICES[1:]))
    out = {}
    for c in STACK_COMPONENTS:
        out.update((STACK_ON if c in on else STACK_OFF)[c])
    out["stack_rebuild"] = out.pop("rebuild")
    return out


def apply_stack(a, name):                                                                       # [integ]
    """[integ] Set a stack's values on the parsed arguments (parse_args checked that none was passed explicitly)."""
    for k, v in stack_values(name).items():
        setattr(a, k, v)
    a.stack_applied = name


def parse_variants(text, twin):                                                                 # [batched-rebuild]
    """[batched-rebuild] --variants -> [(twin, rebuild)]; "" -> [(twin, "per_call")] (p3's arm, its labels unchanged);
    None for a malformed list (an unknown twin or rebuild, a duplicate, an empty item)."""
    if not text:
        return [(twin, "per_call")]
    out = []
    for item in text.split(","):
        parts = item.split("/")
        if len(parts) != 2 or parts[0] not in ("torch", "fused") or parts[1] not in ("per_call", "batched"):
            return None
        if tuple(parts) in out:
            return None
        out.append(tuple(parts))
    return out


def variant_label(a, mode, kind, variant):                                                       # [batched-rebuild]
    """[batched-rebuild] the point's label: p3's SG5_<mode>_<backend> without --variants, else with _<twin>_<rebuild>.
    [integ] with --stack: SG5_<mode>_<backend>_stack_<name> (e.g. SG5_exact_b2_stack_full)."""
    if getattr(a, "stack", ""):
        return "SG5_%s_%s_stack_%s" % (mode, kind, a.stack.replace("-", "_"))
    if not a.variants:
        return "SG5_%s_%s" % (mode, kind)
    return "SG5_%s_%s_%s_%s" % (mode, kind, variant[0], variant[1])


def stats(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    n = len(xs)
    return dict(n=n, mean=sum(xs) / n, median=xs[n // 2], p90=xs[min(n - 1, int(0.9 * n))], min=xs[0], max=xs[-1])


def load_docs(a):
    """B documents of S tokens; with --repeat-docs the workload's documents are tiled (labelled)."""
    import torch
    import json as _j
    man = _j.load(open(os.path.join(a.workload, "c16k_manifest.json")))
    n = int(man["shape"][0])
    if a.batch <= n:
        docs, sha = G0.load_workload(a.workload, a.batch, a.S)
        return docs, sha, False
    if not a.repeat_docs:
        raise SystemExit("REFUSED: B %d > %d documents (pass --repeat-docs to tile them, labelled)" % (a.batch, n))
    docs, sha = G0.load_workload(a.workload, n, a.S)
    reps = -(-a.batch // n)
    return torch.cat([docs] * reps, 0)[:a.batch].contiguous(), sha, True


# ------------------------------------------------------------------ the shipped decode (a child process)
def shipped_child(a) -> int:
    import torch
    import nosi.cache_engine as CE
    from nosi import NOSALlama as Llama
    from nosi.cache_engine import InfLLMv2Cache
    if CE.POOL_BLOCKS != a.baseline_only:
        print("REFUSED: NOSI_POOL_BLOCKS %d != %d" % (CE.POOL_BLOCKS, a.baseline_only), file=sys.stderr)
        return 2
    docs, _, rep = load_docs(a)
    model = Llama(model_name=a.model, device="cuda", offload=True)
    B = a.batch
    with torch.inference_mode():
        cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
        lg, pos = model.batch_prefill(docs[:B, :a.S].to("cuda"), cache)
        model.has_buffers = False                                    # the shipped warm-up captures for this batch
        tok = lg[:, -1].argmax(-1, keepdim=True)
        p = (pos[:, -1:] + 1).to(torch.int32)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
        ms = []
        for s in range(2 + a.shipped_steps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            out = model.decode_inference(tok, cu, p, cache)[:, -1]
            e1.record()
            torch.cuda.synchronize()
            if s >= 2:
                ms.append(e0.elapsed_time(e1))
            tok = out.argmax(-1, keepdim=True)
            p = p + 1
    json.dump(dict(pool_blocks=a.baseline_only, ms=ms, stats=stats(ms), batch=B, S=a.S, repeated_docs=rep,
                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)), open(a.baseline_out, "w"), indent=1)
    return 0


def run_child(a, argv, pool, path):
    env = dict(os.environ, NOSI_POOL_BLOCKS=str(pool))
    cmd = [sys.executable, os.path.abspath(__file__)] + list(argv) + ["--baseline-only", str(pool), "--baseline-out", path]
    if getattr(a, "shipped_batch", 0) > 0:                          # [integ] the shipped decode at its own batch
        cmd += ["--batch", str(a.shipped_batch)]                    # (argparse: the last --batch wins)
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0 or not os.path.exists(path):
        return None, "child exit %d" % r.returncode
    d = json.load(open(path))
    d["wall_s"] = round(time.time() - t0, 1)
    return d, None


# ------------------------------------------------------------------ the engine on the Residency
class GatherTransport:
    """SG5's "gather" transport (C's note, section 4: the GPU-gather baseline of paper 4.4), so the tick's arms can run
    while another transport is fixed: K3X's interface issue / poll / done / force, plus stream_wait (the verifier waits
    on the device). One issue = ONE launch per tensor of NOSI's zero-copy Triton gather flash_h2d_from_mask (it reads
    the pinned host rows) on a SIDE stream, with a load map (H, B, C + X) over the store's historical slots; the side
    stream first waits (device waits) on every transfer's hist_ev (T4: the tick's write-back of the host rows) and
    reader_ev (T3: the destination slot's last reader). One completion event per issue. The bias rows are not moved: the
    service fills them from the GPU cis table at take."""

    name = "gather"

    def __init__(self, host_k, host_v, store_k, store_v, n_slots: int, bs: int = 64, xpool=None):
        import torch
        from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
        self.h2d = flash_h2d_from_mask
        self.host_k, self.host_v, self.store_k, self.store_v = host_k, host_v, store_k, store_v
        self.n_slots, self.bs = int(n_slots), int(bs)
        self.xpool = xpool                          # [xpool] the flat store: one entry per transfer, its lane (phys)
        self.side = torch.cuda.Stream(device=store_k[0].device)
        self.stats = dict(releases=0, transfers=0, threads_min=None, threads_max=None)

    def _issue_flat(self, layer, transfers):                                                     # [xpool]
        """[xpool] ONE launch of r2_xpool.h2d_entries (entries b, h, block, lane) on the side stream, after the same
        waits; the bias rows are not moved (the service fills them at take)."""
        import torch
        from nosi.r2_xpool import h2d_entries
        dst = set()
        for tr in transfers:
            if tr.phys is None or (tr.h, tr.phys) in dst:
                raise ValueError("gather (xpool): transfer seq %d has no lane or shares (h %d, lane %r)"
                                 % (tr.seq, tr.h, tr.phys))
            dst.add((tr.h, tr.phys))
        ent = torch.tensor([[tr.b, tr.h, tr.block, tr.phys, 1] for tr in transfers], dtype=torch.int64).t().contiguous()
        with torch.cuda.stream(self.side):
            for tr in transfers:
                for ev in (tr.hist_ev, tr.reader_ev):
                    if ev is not None:
                        self.side.wait_event(ev)
            d = ent.pin_memory().to(self.store_k[layer].device, non_blocking=True)
            h2d_entries(self.store_k[layer], self.store_v[layer], None, self.host_k[layer], self.host_v[layer], None, d,
                        self.bs)
            ev = torch.cuda.Event()
            ev.record(self.side)
        for tr in transfers:
            tr.ev = ev
        self.stats["releases"] += 1
        self.stats["transfers"] += len(transfers)

    def issue(self, layer, transfers):
        import torch
        if not transfers:
            return
        if self.xpool is not None:                                                               # [xpool]
            return self._issue_flat(layer, transfers)
        B, H = int(self.store_k[layer].shape[0]), int(self.store_k[layer].shape[2])
        ids = torch.full((H, B, self.n_slots), -1, dtype=torch.int64)
        for tr in transfers:
            if ids[tr.h, tr.b, tr.slot] != -1:
                raise ValueError("gather: two transfers into one slot (b %d h %d slot %d)" % (tr.b, tr.h, tr.slot))
            ids[tr.h, tr.b, tr.slot] = tr.block
        with torch.cuda.stream(self.side):
            for tr in transfers:
                for ev in (tr.hist_ev, tr.reader_ev):
                    if ev is not None:
                        self.side.wait_event(ev)
            d = ids.pin_memory().to(self.store_k[layer].device, non_blocking=True)
            self.h2d(self.store_k[layer], self.host_k[layer], d, self.bs)
            self.h2d(self.store_v[layer], self.host_v[layer], d, self.bs)
            ev = torch.cuda.Event()
            ev.record(self.side)
        for tr in transfers:
            tr.ev = ev
        self.stats["releases"] += 1
        self.stats["transfers"] += len(transfers)

    def poll(self):
        pass

    def done(self, tr):
        return tr.ev is not None and tr.ev.query()

    def force(self, tr):
        if tr.ev is None:
            raise RuntimeError("gather: transfer seq %d was never issued" % tr.seq)
        tr.ev.synchronize()

    def stream_wait(self, tr):
        import torch
        torch.cuda.current_stream().wait_event(tr.ev)

    def threads_ok(self):
        return True

    def close(self):
        pass


class _LazyOps:
    """S-C's device operations bound after the service exists (CudaDeviceOps takes the service)."""

    def __init__(self):
        self.impl = None

    def wait(self, l, dec):
        return self.impl.wait(l, dec)

    def gather(self, l, dec):
        return self.impl.gather(l, dec)


class _CloseOnError:
    """[stack2] make_engine's guard: an exception inside the block closes the transport, then propagates. The gilfree
    transport's feeder (background=True) is a thread whose target is a bound method of the transport, so the running
    thread holds the transport and its store (eng.store_k / store_v): no `del` and no gc.collect() frees them until
    close() ends the thread. A close() that raises is printed; the first exception propagates."""

    def __init__(self, tp):
        self.tp = tp

    def __enter__(self):
        return self.tp

    def __exit__(self, et, ev, tb):
        if et is not None:
            try:
                self.tp.close()
            except Exception:                                    # noqa: BLE001 - the original exception wins
                traceback.print_exc()
        return False


def cuda_event():
    import torch
    e = torch.cuda.Event()
    e.record()
    return e


def select_attn(a, attn):                                                                       # [integ] sharedkv
    """[integ] The tick's attention object for --attn (r2_tick_gate.select_attn's tick choices): per_row = piece 3's
    module itself (default, unchanged); shared / shared-whole = a NEW SharedKV over the same loaded piece 3 module
    (its stats count the shared calls: the gate <label>_SKV_ran), union --union."""
    if getattr(a, "attn", "per_row") == "per_row":
        return attn
    sk = GB.load_by_path(a.sharedkv, "sharedkv_bi_paged_attention")
    obj = sk.SharedKV(attn, mode="whole" if a.attn.endswith("whole") else "split", union=a.union)
    obj.count_saved = witness_on(a)      # [integ3] review M2: the union's saved row pages, a device counter (no read);
    return obj                           # [integ5] OFF in a time cell (a diagnostic)


def witness_on(a):                                                                              # [integ5]
    """[integ5] The diagnostic witnesses (XPool handoffs / peak, SharedKV.pages_saved) run except in a time cell.
    [hostfix] --lean-time turns them OFF too (LEAN_DROPPED)."""
    return getattr(a, "stack_role", "") != "time" and not getattr(a, "lean_time", False)


def skv_report(obj):                                                                            # [integ]
    """[integ] REPORT of a SharedKV selector (None for piece 3): stats, calls per rows_max, union builds per kind."""
    if not getattr(obj, "needs_page_key", False):
        return None
    return dict(stats=dict(obj.stats), by_rows_max={str(k): v for k, v in dict(obj.by_rows_max).items()},
                union_builds=dict(getattr(obj, "union_builds", {})), mode=obj.mode, union=getattr(obj, "union", None),
                pages_saved=(int(obj.pages_saved) if getattr(obj, "pages_saved", None) is not None else None),  # [integ3]
                count_saved=bool(getattr(obj, "count_saved", False)))                                     # [integ5]


def arm_layout(a, layout=None):                                                                 # [xpool]
    """[xpool] The store layout of an arm: the caller's, else --layout ("both": the pool, e.g. in the preflight)."""
    lay = layout or getattr(a, "layout", "stream")
    return "pool" if lay == "both" else lay


def make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, timer_cats=None, transport=None, layout=None,
                engine_kw=None):
    """A fresh engine on the Residency layout with its service and gilfree transport (module docstring). [xpool]
    layout (default --layout): "pool" builds the shared-pool store (sc / b2 only) and hands its XPool to the
    transports and the service. engine_kw: extra TickEngine arguments through from_r2 (the context sweep's tick
    family passes log_slots); None = as before. [bigb] --log-slots (a.log_slots_n) sizes the token log when
    engine_kw does not name it (--log-slots 0: the engine's default, as before)."""
    import torch
    from nosi.r2_residency import Residency, TorchGather
    from nosi.r2_residency_sb import ResidencySB
    from nosi.r2_residency_sc import CudaDeviceOps, ResidencySC
    from nosi.r2_tick import ResidencyBackend, TickEngine, TickTimer
    from nosi.r2_transport_gilfree import GilfreeTransport
    ekw = dict(engine_kw or {})
    if getattr(a, "log_slots_n", None) is not None:                                              # [bigb]
        ekw.setdefault("log_slots", int(a.log_slots_n))                                          # [bigb]
    xspec = None                                                                                 # [xpool]
    if arm_layout(a, layout) == "pool":                                                          # [xpool]
        if kind not in XPOOL_BACKENDS:                                                           # [xpool]
            raise RuntimeError("the shared pool runs on %s only (got %s)" % (XPOOL_BACKENDS, kind))
        xspec = (C, pool_positions(a, X))                                                        # [xpool]
    if xspec is None:                                                                            # the default layout
        eng = TickEngine.from_r2(model, cache, st, ops, hist_slots=C + X, twin=a.twin, **ekw)    # [twin-variant]
    else:                                                                                        # [xpool]
        eng = TickEngine.from_r2(model, cache, st, ops, hist_slots=C + X, twin=a.twin, xpool=xspec, **ekw)
    eng.check = bool(getattr(a, "sel_check", False))        # [integ] --sel-check (TickEngine's check; default False)
    if eng.xpool is not None:                               # [integ5] the diagnostics OFF in a time cell
        eng.xpool.witness = witness_on(a)
    eng.rebuild = getattr(a, "rebuild", "per_call")         # [batched-rebuild] the arm's rebuild (parse_variants checks)
    B, L, H = eng.B, eng.L, eng.H
    xp = eng.xpool                                                                               # [xpool] or None
    host_k = [lay.cache_engine._k_cpu for lay in cache.layers]
    host_v = [lay.cache_engine._v_cpu for lay in cache.layers]
    n_host = int(host_k[0].shape[1])
    cis = [t.total_cis[:B] for t in eng.twins]
    hw = lambda ev: ev.synchronize()                                 # noqa: E731
    if (transport or a.transport) == "gilfree":
        if ext is None:
            raise RuntimeError("the gilfree transport needs the packer extension (not built in this run)")
        tp = GilfreeTransport(ext, host_k, host_v, eng.store_k, eng.store_v, 64, eng.D, threads=a.threads, host_wait=hw,
                              background=True, xpool=xp,                                         # [xpool] xp or None
                              warm_up=getattr(a, "gilfree_warm_up", "on") == "on")              # [integ] gfw1f
    else:
        tp = GatherTransport(host_k, host_v, eng.store_k, eng.store_v, C + X, xpool=xp)          # [xpool] xp or None
    # [stack2] a failure after the transport exists closes it: the gilfree feeder thread holds the store (job 386
    # stack1a128: OwnResidency refused C + X 992 here, the thread kept the first engine, the gather arm built a second:
    # preflight peak 82.97 GB). The exception still propagates.
    with _CloseOnError(tp):
        gather = TorchGather(host_k, host_v, eng.store_k, eng.store_v)
        args = (L, B, H, C, X, R_ver, tp, gather)
        kw = dict(store_b=eng.store_bias, cis=cis, block_bytes=64 * 2 * eng.D * 2, host_wait=hw)
        if xp is not None:                                                                           # [xpool]
            kw["xpool"] = xp
        if getattr(a, "arrival_lag", 0) > 0:                                                         # [xpool] review B1
            kw["arrival_lag"] = a.arrival_lag
        if kind == "v1":
            svc = Residency(*args, **kw)
        elif kind == "sb":
            svc = ResidencySB(*args, device="cuda", nblk=n_host // 64, **kw)
        elif kind == "b2":                                               # [b2] B2's table on the store's device; S-C's
            from nosi.r2_own2 import OwnResidency                        # [b2] device path: CudaDeviceOps (the superset
            lazy = _LazyOps()                                            # [b2] wait) behind _LazyOps, as sc
            svc = OwnResidency(*args, device=eng.store_k[0].device, nblk=n_host // 64, devops=lazy,             # [b2]
                               checked=bool(getattr(a, "b2_checked", False)),                                     # [b2]
                               resolve=getattr(a, "b2_resolve", "torch"),                                         # [fusedres]
                               fused_plant=int(getattr(a, "b2_fused_plant", 0)),                                  # [fusedres2]
                               lean=bool(getattr(a, "lean_time", False)),                                         # [hostfix]
                               wait_index=bool(getattr(a, "wait_index", False)),                                  # [hostfix]
                               fewer_reads=bool(getattr(a, "b2_fewer_reads", False)), **kw)                       # [hostfix]
            lazy.impl = CudaDeviceOps(svc, host_k, host_v, eng.store_k, eng.store_v, eng.store_bias, cis)       # [b2]
        else:
            lazy = _LazyOps()
            svc = ResidencySC(*args, device="cuda", nblk=n_host // 64, devops=lazy, **kw)
            lazy.impl = CudaDeviceOps(svc, host_k, host_v, eng.store_k, eng.store_v, eng.store_bias, cis)
        eng.attach_backend(ResidencyBackend(eng, svc, host_k, host_v, cuda_event, a.K))
        eng.timer = TickTimer(cuda=True)
        if getattr(a, "host_profile", False):                            # [hostfix] item 0: host clocks only
            eng.host_prof = True
            svc.host_prof = lambda cat, ns, t=eng.timer: t.host(cat, ns * 1e-9)
            if getattr(tp, "worker", None) == "python":                  # the gilfree Python feeder's releases
                tp.prof = []
        if timer_cats is not None:                                       # BARE: the tick events only
            base_span = eng.timer.span
            import contextlib

            def span(cat):
                return base_span(cat) if cat in timer_cats else contextlib.nullcontext()
            eng.timer.span = span
        return eng, svc, tp


def fused_gates(a, label, phase, rep):
    """[fusedres2] the FR-g gates of docs/R2_FUSEDRES_PLAN.md section 7 as harness gates (a failed MUST counts in the
    exit, so r2_tick_sg5_step.sh's combined cell frg can stop on it). phase: "draft_only" or "round".
      F2   err_any == 0 (MUST, every fused mode)
      G0   check, no plant: 0 differences and every layer compared (MUST); the coverage gate (REPORT: a counter at 0
           is "NOT GATED"; FR-g0 and FR-g0x together decide it)
      G0p  check, plant 2: d_grant > 0, differences > 0, first_diff_at == first_effective["d_grant"] (MUST in the
           draft-only phase, REPORT in the round)
      G2   fused, not profiled, draft-only phase: launches_per_layer == 3.0, sel_conversions == 0, bias_casts == 0
           (MUST)"""
    mode, plant = getattr(a, "b2_resolve", "torch"), int(getattr(a, "b2_fused_plant", 0))
    tag = "%s_%s" % (label, phase)
    out = [GB.gate(tag + "_fused_err", rep["err"] == 0, True, "err_any %s %s" % (rep["err"], rep["err_bits"]))]
    if mode == "check" and not plant:
        ok = rep["diffs"] == 0 and rep["compared"] == rep["layers"] > 0
        out.append(GB.gate(tag + "_fused_equal", ok, True, "diffs %d compared %d layers %d first %s"
                           % (rep["diffs"], rep["compared"], rep["layers"], rep["first_diffs"][:2])))
        out.append(GB.gate(tag + "_fused_coverage", True, False, "gate %s stats %s" % (rep["gate"], rep["stats"])))
    if mode == "check" and plant == 2:
        fe = rep["first_effective"].get("d_grant")
        ok = rep["stats"]["d_grant"] > 0 and rep["diffs"] > 0 and fe is not None and rep["first_diff_at"] == fe
        out.append(GB.gate(tag + "_fused_plant_caught", ok, phase == "draft_only",
                           "d_grant %d diffs %d first_diff_at %s first_effective %s"
                           % (rep["stats"]["d_grant"], rep["diffs"], rep["first_diff_at"], fe)))
    if mode == "fused" and phase == "draft_only" and not getattr(a, "b2_res_profile", False):
        ok = rep["launches_per_layer"] == 3.0 and rep["sel_conversions"] == 0 and rep["bias_casts"] == 0
        stack = bool(getattr(a, "stack", ""))                # [integ3] review-integ2 M1: REPORT in the stack cells
        out.append(GB.gate(("R_" if stack else "") + tag + "_fused_dispatch", ok, not stack,      # until FR-g
                           "launches/layer %s sel_conversions %d bias_casts %d compiled %s%s"     # measures it
                           % (rep["launches_per_layer"], rep["sel_conversions"], rep["bias_casts"], rep["compiled"],
                              " (REPORT in a stack cell until FR-g2 measured it)" if stack else "")))
    return out


def stack_bands_verdict(v, peak_gb, ticks_ms, failed_must):                                     # [integ3] M3
    """[integ3] IN when the run's peak GB and its ONE point's T_tick (ms) lie in v = (peak lo, hi, tick lo, hi) and no
    MUST gate failed; OUT otherwise (a missing value is OUT, never IN)."""
    if failed_must or peak_gb is None or len(ticks_ms) != 1 or ticks_ms[0] is None:
        return "OUT"
    t = ticks_ms[0]
    return "IN" if (v[0] <= peak_gb <= v[1] and v[2] <= t <= v[3]) else "OUT"


def check_validated(paths):                                                                    # [integ5]
    """[integ5] The time cells' evidence chain: each path is a validation cell's summary.json; every MUST gate in it
    passed, and it holds a passed *_ST_replica and *_ST_bitwise gate. Returns [(path, record)] or raises SystemExit
    ("REFUSED ...": the harness exits 2 before any build)."""
    out = []
    for p in [x for x in paths.split(",") if x]:
        try:
            s = json.load(open(p))
            gates = s["gates"]
        except Exception as e:                                       # noqa: BLE001 - a refusal names the cause
            raise SystemExit("REFUSED: --validated-by %s: not a readable summary (%s: %s)" % (p, type(e).__name__, e))
        bad = [g["name"] for g in gates if g.get("must") and not g.get("passed")]
        rep = [g["name"] for g in gates if g["name"].endswith("_ST_replica") and g.get("passed")]
        bit = [g["name"] for g in gates if g["name"].endswith("_ST_bitwise") and g.get("passed")]
        if bad or not rep or not bit:
            raise SystemExit("REFUSED: --validated-by %s: failed MUST %s, passed replica %s, passed bitwise %s"
                             % (p, bad[:5], rep, bit))
        m = s.get("meta", {})
        out.append(dict(path=p, nosi_commit=m.get("nosi_commit"), batch=m.get("args", {}).get("batch"),
                        stack=m.get("stack"), bitwise=bit, n_must=sum(1 for g in gates if g.get("must"))))
    if not out:
        raise SystemExit("REFUSED: --validated-by names no summary")
    return out


def check_accuracy_record(path):                                                               # [evalsplit]
    """[evalsplit] Eval cell (a)'s evidence: r2_evalacc.check_record (FAST vs EXACT EQUIVALENT, every MUST passed).
    SystemExit ("REFUSED ...") otherwise; the harness exits 2 before any build."""
    import r2_evalacc
    return r2_evalacc.check_record(path)


def _rates(committed, ticks, tick_stats, wall_s, source):                                       # [evalsplit]
    if not ticks or not tick_stats or not committed:
        return None
    med, mean = tick_stats.get("median"), tick_stats.get("mean")
    return dict(source=source, committed=committed, ticks=ticks, wall_s=wall_s, tick_ms_median=med, tick_ms_mean=mean,
                steady_tok_s_median=(committed / (ticks * med / 1e3)) if med else None,
                steady_tok_s_mean=(committed / (ticks * mean / 1e3)) if mean else None,
                effective_tok_s=(committed / wall_s) if wall_s else None)


def eval_metrics(p):                                                                           # [evalsplit]
    """[evalsplit] A point's throughput, EFFECTIVE and STEADY SEPARATELY (human, 2026-10-21): steady = committed /
    (ticks x the median or mean tick: no out-of-tick time), effective = committed / the round's wall clock (every
    tick, first-use compiles and the host time between ticks; prefill and the engine build excluded). headline = the
    bare round (tick events only), instrumented = the per-op-event round (REPORT)."""
    br = p.get("bare_round") or {}
    rs = (p.get("round") or {}).get("device_ms", {}).get("tick")
    return dict(headline=_rates(br.get("committed"), br.get("ticks"), p.get("bare_T_tick"), br.get("wall_s"),
                                "bare round (tick events only)"),
                instrumented=_rates(p.get("committed"), p.get("ticks"), rs, p.get("wall_s"),
                                    "instrumented round (per-op events)"))


def eval_gates(a, meta, res, gates):                                                           # [evalsplit]
    """[evalsplit] The gates of --eval-role fast-time (module docstring)."""
    pts = res.get("points") or []
    head = pts[0] if pts else {}
    gates.append(GB.gate("EF_headline_fast", head.get("gemm") == "fast", True,               # one_point: ops.mode
                         "the headline point's GEMM: %s" % head.get("gemm")))
    want = a.batch * (a.steps - 1)
    for p in pts:
        lab = p.get("label") or "%s_%s" % (p.get("gemm", "?")[:4], p.get("rebuild"))
        p["eval"] = eval_metrics(p)
        br = p.get("bare_round") or {}
        gates.append(GB.gate("EF_bare_%s" % lab, br.get("committed") == want, True,
                             "the bare round committed %s tokens (B (steps - 1) = %d) in %s ticks"
                             % (br.get("committed"), want, br.get("ticks"))))
        h, ins = p["eval"]["headline"], p["eval"]["instrumented"]

        def fmt(x):
            return "none" if not x else ("steady %.2f tok/s (median tick) / %.2f (mean tick); effective %.2f tok/s"
                                         % (x["steady_tok_s_median"] or 0, x["steady_tok_s_mean"] or 0,
                                            x["effective_tok_s"] or 0))
        gates.append(GB.gate("R_EF_%s" % lab, True, False, "HEADLINE (bare): %s | instrumented: %s"
                             % (fmt(h), fmt(ins))))
    peak = meta.get("peak_gb")
    gates.append(GB.gate("EF_mem_cap", peak is not None and peak <= a.mem_cap_gb, True,
                         "peak allocated %s GB vs the declared cap %g GB" % (peak, a.mem_cap_gb)))
    gates.append(GB.gate("EF_accuracy_record", bool(meta.get("accuracy_record")), True,
                         "FAST timed under the accuracy record %s" % (meta.get("accuracy_record") or {}).get("path")))
    if a.eval_bands_v is not None:
        h = (head.get("eval") or {}).get("headline")
        v = a.eval_bands_v
        if not h:
            text = "no headline rates"
        else:
            parts = [("steady_median", h["steady_tok_s_median"], v[0], v[1]),
                     ("steady_mean", h["steady_tok_s_mean"], v[2], v[3]),
                     ("effective", h["effective_tok_s"], v[4], v[5])]
            text = "; ".join("%s %.2f in [%g, %g]: %s" % (n, x or 0, lo, hi, "IN" if x is not None and lo <= x <= hi
                                                            else "OUT") for n, x, lo, hi in parts)
        gates.append(GB.gate("R_EF_bands", True, False, text))


def headline_tick_ms(p):                                                                        # [integ5]
    """[integ5] A point's headline T_tick: the --bare round's tick mean (no per-category events) when present, else
    the instrumented round's."""
    bare = p.get("bare_T_tick")
    return bare["mean"] if isinstance(bare, dict) and bare.get("mean") is not None else p.get("T_tick_ms")


def write_stack_bands(a, meta, res, gates):                                                     # [integ3] M3
    """[integ3] <out>/stack_bands.txt: one line per value, then STACK_BANDS_VERDICT IN|OUT (the combined step cell
    stack1 greps that exact line before it runs stack1b; no interpreter in the step). REPORT gate R_stack_bands."""
    pts = res.get("points", [])
    ticks = [headline_tick_ms(p) for p in pts]                       # [integ5] the --bare round's when present
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    v = a.stack_bands_v
    verdict = stack_bands_verdict(v, meta.get("peak_gb"), ticks, nfail)
    text = ("STACK_BANDS peak_gb=%s band=%s:%s\nSTACK_BANDS T_tick_ms=%s band=%s:%s points=%d failed_must=%d\n"
            "STACK_BANDS_VERDICT %s\n" % (meta.get("peak_gb"), v[0], v[1], ticks[0] if len(ticks) == 1 else ticks,
                                          v[2], v[3], len(pts), nfail, verdict))
    with open(os.path.join(a.out, "stack_bands.txt"), "w") as f:
        f.write(text)
    gates.append(GB.gate("R_stack_bands", True, False, text.replace("\n", "; ")))


def stack_components_on(a):                                                                     # [integ3]
    """[integ3] The stack components an arm has ON, from its own arguments (STACK_COMPONENTS order)."""
    return dict(twin=getattr(a, "twin", "torch") == "fused", rebuild=getattr(a, "rebuild", "per_call") == "batched",
                resolve=getattr(a, "b2_resolve", "torch") == "fused", layout=arm_layout(a) == "pool",
                attn=getattr(a, "attn", "per_row") != "per_row", warmup=getattr(a, "gilfree_warm_up", "on") == "on")


def stack_acted(w, B, C):                                                                       # [integ3]
    """[integ3] review-integ2 M2: per component, did it ACT in this arm (its own GPU witness)? w: the arm's witnesses
    (one_point: twin_variant, batched_calls, batched_max, fused, xpool, skv, warm_calls). Returns {c: (acted, detail)}
    and the shared-region witness of the pool (the per-(layer, head) peak above B x C)."""
    fu, xp, skv = w.get("fused"), w.get("xpool"), w.get("skv")
    ub = (skv or {}).get("union_builds", {})
    wit = w.get("witness", True)          # [integ5] a time cell: no handoff / saved-pages witness (it is not run there)
    out = dict(
        twin=(w.get("twin_variant") == "fused", "twin_variant %s" % w.get("twin_variant")),
        rebuild=(w.get("batched_calls", 0) > 0 and w.get("batched_max", 0) >= 2,
                 "batched calls %s, most publications in one call %s" % (w.get("batched_calls"), w.get("batched_max"))),
        resolve=(fu is not None and fu.get("launches", 0) > 0 and fu.get("layers", 0) > 0,
                 "fused launches %s layers %s" % (None if fu is None else fu.get("launches"),
                                                 None if fu is None else fu.get("layers"))),
        layout=(xp is not None and xp.get("binds", 0) > 0 and (xp.get("handoffs", 0) > 0 or not wit),
                "xpool binds %s, lanes handed between requests %s%s"
                % (None if xp is None else xp.get("binds"), None if xp is None else xp.get("handoffs"),
                   "" if wit else " (witness OFF in a time cell: binds only)")),
        attn=(skv is not None and skv["stats"].get("shared", 0) > 0 and ub.get("fused", 0) > 0
              and ub.get("torch", 0) == 0 and ((skv.get("pages_saved") or 0) > 0 or not wit),
              "shared calls %s, union builds %s, row pages saved by the union %s"
              % (None if skv is None else skv["stats"].get("shared"), ub, None if skv is None else skv.get("pages_saved"))),
        warmup=(w.get("warm_calls") == 1, "transport warm_calls %s" % w.get("warm_calls")))
    region = (xp is not None and xp.get("max_peak_lanes", 0) > B * C,
              "per-(layer, head) peak lanes max %s vs B x C = %d; pairs above %s"
              % (None if xp is None else xp.get("max_peak_lanes"), B * C, None if xp is None else xp.get("pairs_over_base")))
    return out, region


def stack_acted_gates(a, label, w, B, C):                                                       # [integ3]
    """[integ3] review-integ2 M2: MUST <label>_ST_acted_<c> for every component ON in a stack arm (its witness shows
    it acted) and MUST <label>_ST_off_<c> for every component OFF (its witness shows nothing: a full-no-<c> cell
    really removed c); the pool's shared X region (<label>_ST_xregion) is MUST with --require-xregion, else REPORT."""
    if not getattr(a, "stack_applied", None):
        return []
    on = stack_components_on(a)
    acted, region = stack_acted(w, B, C)
    gates = []
    for c in STACK_COMPONENTS:
        ok, detail = acted[c]
        if on[c]:
            gates.append(GB.gate("%s_ST_acted_%s" % (label, c), ok, True, detail))
        else:
            gates.append(GB.gate("%s_ST_off_%s" % (label, c), not ok, True, "OFF in this arm: " + detail))
    if on["layout"] and w.get("witness", True):                 # [integ5] measured only where the witness runs
        req = bool(getattr(a, "require_xregion", False))
        gates.append(GB.gate(("" if req else "R_") + "%s_ST_xregion" % label, region[0], req,
                             "the shared X region bound: " + region[1]))
    if getattr(a, "stack_role", "") == "time":                  # [integ5] a time cell refuses a diagnostic in its path
        xp, skv = w.get("xpool"), w.get("skv")
        on_now = [n for n, v in (("xpool.witness", xp is not None and xp.get("witness")),
                                 ("SharedKV.count_saved", skv is not None and skv.get("count_saved"))) if v]
        gates.append(GB.gate("%s_ST_witness_off" % label, not on_now, True,
                             "time cell: diagnostics in the timed path: %s" % (on_now or "none")))
    return gates


def profile_draft_only(a, eng, first, label):
    """[fusedres2] M3 of review-fusedres-0e28cf6: the draft-only phase under torch.profiler (CPU + CUDA), every
    TickTimer span also a record_function range "t.<cat>"; the trace is written to --out and parsed by
    r2_fusedres_prof.parse_res_trace: per draft tick the device ms and the device events launched inside "t.res"
    (tick 0 skipped in the stats: compiles). The event spans of a profiled phase are NOT timings."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    import r2_fusedres_prof as FP
    base_span = eng.timer.span
    import contextlib

    @contextlib.contextmanager
    def span(cat):
        with record_function("t." + cat):
            with base_span(cat):
                yield
    eng.timer.span = span
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_draft_only(eng, first, a.draft_ticks, a.K)
            torch.cuda.synchronize()
    finally:
        eng.timer.span = base_span
    path = os.path.join(a.out, "%s_draft_only_trace.json" % label)
    prof.export_chrome_trace(path)
    out = {}
    for cat in ("res", "tick"):
        p = FP.parse_res_trace(json.load(open(path)), cat=cat)
        out[cat] = FP.summarize(p, skip=1)
    out["trace"] = path
    return out


def run_draft_only(eng, first, n, K):
    """n ticks of B draft rows (the draft-only tick T_d); a draft key is decided once its cohort's positions are all
    drafted (as a verification would)."""
    from types import SimpleNamespace
    tick = sys.modules[type(eng).__module__]                         # the engine's own module: nosi.r2_tick on the
    Ref, cohort_of = tick.Ref, tick.cohort_of                        # GPU, the CPU tests' path-loaded copy (job 2190926)
    B = eng.B
    toks = list(first)
    for t in range(n):
        i = t + 1
        rows = [SimpleNamespace(b=b, kind="d", pos=eng.bank_pos[B + b], epoch=0, i=i, m=0) for b in range(B)]
        eng.timer.next_tick()
        with eng._span("tick"):
            out = eng.run_rows(rows, toks)
        toks = [Ref(s) for s in out.slots]
        if cohort_of(i + 1, K) != cohort_of(i, K):
            for b in range(B):
                eng.backend.svc.decided((b, 0, cohort_of(i, K)))
    eng.check_overflow()


def summarize(rep, skip=1):
    """Per category: the mean per tick (device ms and host ms), skipping the first `skip` ticks (warm-up)."""
    out = dict(ticks=rep["ticks"])
    for kind in ("device_ms", "host_ms"):
        out[kind] = {c: stats(v[skip:]) for c, v in rep[kind].items()}
    tick = rep["device_ms"].get("tick", [])
    cats = [c for c in rep["device_ms"] if c != "tick"]
    resid = [tick[t] - sum(rep["device_ms"][c][t] for c in cats) for t in range(len(tick))]
    out["residual_ms"] = stats(resid[skip:])
    return out


class GcLog:
    """[hostfix] --host-profile: a gc.callbacks hook over a timed round. Each collection is one event (the timer's tick
    at its start, the generation, the start clock ns, the pause ns, the objects collected, the thread that ran it: the
    main thread or the gilfree feeder; a collection holds the GIL, so it pauses both). Collections are serialized by
    the interpreter, so one start clock is enough. Installed by `with`, removed at the exit (an exception too)."""

    def __init__(self, timer):
        self.timer = timer
        self.events = []
        self._t0 = None

    def __call__(self, phase, info):
        if phase == "start":
            self._t0 = time.perf_counter_ns()
        elif self._t0 is not None:
            t1 = time.perf_counter_ns()
            self.events.append((int(self.timer.tick), int(info.get("generation", -1)), self._t0, t1 - self._t0,
                                int(info.get("collected", 0)), threading.current_thread().name))
            self._t0 = None

    def __enter__(self):
        gc.callbacks.append(self)
        return self

    def __exit__(self, et, ev, tb):
        try:
            gc.callbacks.remove(self)
        except ValueError:
            pass
        return False

    def report(self, n_ticks):
        """Per tick the pause ms (sum; a collection before the first tick counts at tick 0), per generation the count
        and ms, the ten longest pauses (tick, generation, ms, collected, thread)."""
        per = [0.0] * max(1, int(n_ticks))
        by_gen = {}
        for (t, g, _, dur, col, _th) in self.events:
            per[min(max(t, 0), len(per) - 1)] += dur / 1e6
            x = by_gen.setdefault(str(g), dict(n=0, ms=0.0, collected=0))
            x["n"] += 1
            x["ms"] += dur / 1e6
            x["collected"] += col
        top = sorted(self.events, key=lambda e: -e[3])[:10]
        return dict(per_tick_ms=[round(x, 3) for x in per], by_generation=by_gen, n=len(self.events),
                    total_ms=round(sum(e[3] for e in self.events) / 1e6, 3),
                    longest=[dict(tick=e[0], gen=e[1], ms=round(e[3] / 1e6, 3), collected=e[4], thread=e[5])
                             for e in top])


@contextlib.contextmanager
def round_guard(a, eng, rec, tag):
    """[hostfix] The host guards of ONE timed phase (the draft-only phase, the round, the bare round) of one engine:
    --host-profile installs a GcLog on the engine's timer for the phase (yielded; None without the flag).
    --gc-freeze (item 1): before the phase gc.collect(), gc.freeze() (every object alive now goes to the permanent
    generation: no later collection walks it), gc.disable(); after it (an exception too) gc.unfreeze(), gc.enable()
    (only when it was enabled before), then ONE gc.collect() outside the phase: the garbage the phase left in cycles.
    rec["gc_freeze"][tag]: the generation counts and the collections per generation (gc.get_stats) before and after,
    the frozen count, the objects collected before and after, the CUDA bytes that the after-collect freed (a CUDA
    tensor held in a cycle stays allocated while the collector is off: the MUST EF_mem_cap sees the phase's peak).
    Without any [hostfix] flag it does nothing (every phase as before). rec: the point's info (the guards' records)."""
    gcl = GcLog(eng.timer) if getattr(a, "host_profile", False) and eng.timer is not None else None
    frz = None
    if getattr(a, "gc_freeze", False):
        frz = dict(enabled_before=gc.isenabled(), count_before=list(gc.get_count()))
        frz["collected_before"] = gc.collect()
        gc.freeze()
        frz["frozen"] = gc.get_freeze_count()
        gc.disable()
        frz["stats_before"] = [dict(s) for s in gc.get_stats()]
    if gcl is not None:
        gc.callbacks.append(gcl)
    try:
        yield gcl
    finally:
        if gcl is not None:
            try:
                gc.callbacks.remove(gcl)
            except ValueError:
                pass
        if frz is not None:
            frz["count_after"] = list(gc.get_count())
            frz["stats_after"] = [dict(s) for s in gc.get_stats()]
            frz["collections_in_phase"] = [s1["collections"] - s0["collections"]
                                           for s0, s1 in zip(frz["stats_before"], frz["stats_after"])]
            gc.unfreeze()
            if frz["enabled_before"]:
                gc.enable()
            tm = sys.modules.get("torch")
            cuda = tm is not None and tm.cuda.is_available()
            m0 = tm.cuda.memory_allocated() if cuda else None
            frz["collected_after"] = gc.collect()
            frz["cuda_freed_gb_by_collect"] = round((m0 - tm.cuda.memory_allocated()) / 1e9, 4) if cuda else None
            rec.setdefault("gc_freeze", {})[tag] = frz


def feeder_report(prof):
    """[hostfix] --host-profile: the gilfree feeder's releases (GilfreeTransport.prof records: layer, transfers, Python
    ns, native ns, end clock): totals and the per-transfer Python time (the GIL-held part)."""
    if prof is None:
        return None
    recs = list(prof)
    py = [r[2] for r in recs]
    nat = [r[3] for r in recs]
    n_tr = sum(r[1] for r in recs)
    return dict(releases=len(recs), transfers=n_tr, python_ms=round(sum(py) / 1e6, 3),
                native_ms=round(sum(nat) / 1e6, 3), python_us_per_transfer=(sum(py) / 1e3 / n_tr) if n_tr else None,
                python_ms_per_release=stats([x / 1e6 for x in py]))


def host_profile_record(rep, gclog=None, tp=None, skip=1):
    """[hostfix] --host-profile: a round's host record from TickTimer.report() (one per round): every host category
    per tick (ms; the "h_" sub-timers of HOST_PROF_CATS and backend / rows / read), the tick span per tick, the
    stats per category (the first `skip` ticks left out, as summarize), the GC pauses (GcLog) and the feeder's
    Python time (feeder_report)."""
    cats = sorted(c for c in rep["host_ms"] if c.startswith("h_") or c in ("backend", "rows", "read"))
    per_tick = {c: [round(x, 3) for x in rep["host_ms"][c]] for c in cats}
    out = dict(ticks=rep["ticks"], skip=skip, per_tick_ms=per_tick,
               tick_ms=[round(x, 3) for x in rep["device_ms"].get("tick", [])],
               stats={c: stats(v[skip:]) for c, v in per_tick.items()},
               missing=sorted(c for c in HOST_PROF_CATS if c not in per_tick))
    if gclog is not None:
        out["gc"] = gclog.report(rep["ticks"])
    if tp is not None:
        out["feeder"] = feeder_report(getattr(tp, "prof", None))
    return out


def budget(rnd_sum, d_sum):
    """dX = the round's mean per tick minus the draft-only tick's, per device category (docs/R2_STEP7_TICK_SG5.md 3)."""
    out = {}
    for c, v in rnd_sum["device_ms"].items():
        dv = d_sum["device_ms"].get(c)
        if v is not None and dv is not None:
            out["d_" + c] = v["mean"] - dv["mean"]
    if rnd_sum["residual_ms"] and d_sum["residual_ms"]:
        out["d_residual"] = rnd_sum["residual_ms"]["mean"] - d_sum["residual_ms"]["mean"]
    return out


SC_COUNTERS = ("mirror_uploads", "mirror_slots_uploaded", "row_uploads")   # ResidencySC's own (r2_residency_sc.py)
DEV_COUNTERS = ("events_waited", "submission_waits", "waited", "gathered")  # CudaDeviceOps; CpuDeviceSim (CPU tests)


def sc_counters(svc):
    """REPORT only (no gate; analysis/sg5_p4/RES_SC.md section 6): a SNAPSHOT of the counters the service itself
    keeps, read once after its run: mirror_uploads (the host-to-device mirror deltas), mirror_slots_uploaded (the slot
    records in them), row_uploads, journal_counts (the stream transitions per kind), and its device operations' wait
    counters (devops: events_waited / submission_waits of CudaDeviceOps behind _LazyOps; waited / gathered of the CPU
    stand-in), each with the object's class name. Cumulative over the service's life (the draft-only engine: its
    --draft-ticks ticks, tick 0 included; the round: every tick). None for a service that keeps none of them (sb,
    v1); a subclass of S-C (e.g. b2) reports whichever it has."""
    out = {k: int(getattr(svc, k)) for k in SC_COUNTERS if hasattr(svc, k)}
    if hasattr(svc, "journal_counts"):
        out["journal_counts"] = {str(k): int(v) for k, v in dict(svc.journal_counts).items()}
    dev = getattr(svc, "dev", None)
    dev = getattr(dev, "impl", dev)                                 # _LazyOps -> its CudaDeviceOps
    d = {k: int(getattr(dev, k)) for k in DEV_COUNTERS if dev is not None and hasattr(dev, k)}
    if d:
        out["devops"] = dict(d, kind=type(dev).__name__)
    if out:
        out["service"] = type(svc).__name__
    return out or None


def logit_fingerprint(logits):                                                                  # [xpool]
    """[xpool] Per row, sum_i bits(logit_i) * (2 i + 1) over the raw float32 bits as int64 (wrapping): device ops, no
    read. One differing element always changes it (odd weights: w 2^k != 0 mod 2^64 for k < 32); several could cancel
    in principle, so equal fingerprints are strong evidence of bitwise rows, not a proof."""
    import torch
    V = logits.shape[-1]
    w = 2 * torch.arange(V, device=logits.device, dtype=torch.int64) + 1
    return (logits.float().contiguous().view(torch.int32).to(torch.int64) * w).sum(-1)


def _reset_peak():                                                                              # [xpool]
    """[xpool] Remember the run's peak so far (meta.peak_gb keeps its meaning), then reset the counter for one arm."""
    import torch
    _PEAK[0] = max(_PEAK[0], torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()


def one_point(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates, label, bare=False,
              layout=None, fp=None):
    """One point (module docstring). [xpool] layout None: --layout (the default arm, unchanged); "stream" / "pool":
    that arm, with the round's peak GB after a reset (peak_gb_round) and fp (a list): per tick (rows, the rows' logit
    fingerprints on the device) and info["committed_tokens"]."""
    import torch
    from nosi.r2_tick import run_round
    info = dict(backend=kind, C=C, X=X, R_ver=R_ver, gemm=ops.mode, steps=steps)
    if kind == "b2":                                                 # [b2] the mode (report()["b2"] repeats it)
        info["b2_checked"] = bool(getattr(a, "b2_checked", False))
        info["b2_resolve"] = getattr(a, "b2_resolve", "torch")                    # [fusedres]
    if layout is not None or getattr(a, "layout", "stream") != "stream":                         # [xpool]
        info["layout"] = arm_layout(a, layout)
        if info["layout"] == "pool":
            info["X_pool"] = pool_positions(a, X)
    # T_d: the draft-only tick
    eng, svc, tp = make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, layout=layout)
    hprof = bool(getattr(a, "host_profile", False))                  # [hostfix] item 0
    try:
        if kind == "b2" and getattr(a, "b2_res_profile", False):    # [fusedres2] M3: the kernel time of res
            info["draft_only_res_profile"] = profile_draft_only(a, eng, first, label)
        else:
            with round_guard(a, eng, info, "draft_only") as gcl:       # [hostfix] (nothing without its flags)
                run_draft_only(eng, first, a.draft_ticks, a.K)
            if hprof:
                info["draft_only_host_profile"] = host_profile_record(eng.timer.report(), gcl, tp)
        info["draft_only"] = summarize(eng.timer.report())
        info["draft_only"]["d2h_reads"] = getattr(svc, "d2h_reads", None)
        info["draft_only"]["sc"] = sc_counters(svc)                    # REPORT (RES_SC.md 6)
        if getattr(svc, "fusedres", None) is not None:                 # [fusedres] REPORT: launches, stats, err,
            info["draft_only"]["fused"] = svc.fusedres.report()        # check-mode differences
            gates.extend(fused_gates(a, label, "draft_only", info["draft_only"]["fused"]))   # [fusedres2] FR-g
    finally:
        tp.close()
        del eng, svc, tp
        GB.free()
    # the round
    if layout is not None:                                                                       # [xpool] this arm's peak
        _reset_peak()
    eng, svc, tp = make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, layout=layout)
    rnd = None                                                       # [bigb] released with the engine (the TickRound
    req_ticks = [0]                                                  # holds its bound methods)

    def on_tick(rows, tokens, out):
        req_ticks[0] += len({r.b for r in rows})
        if fp is not None:                                                                       # [xpool] no read here
            fp.append(([(r.b, r.kind, r.pos) for r in rows], logit_fingerprint(out.logits)))
    try:
        with round_guard(a, eng, info, "round") as gcl:                # [hostfix] (nothing without its flags)
            t0 = time.time()
            rnd = run_round(eng, first, [steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA, on_tick=on_tick)
            torch.cuda.synchronize()
            info["wall_s"] = round(time.time() - t0, 1)
        if hprof:                                                      # [hostfix] after the round's clock
            info["host_profile"] = host_profile_record(eng.timer.report(), gcl, tp)
        if layout is not None:                                                                   # [xpool]
            info["peak_gb_round"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            info["store_gb"] = round(sum(t.numel() * t.element_size() for t in eng.store_k + eng.store_v
                                         + eng.store_bias) / 1e9, 3)
            info["arrival"] = dict(lag=getattr(svc, "arrival_lag", None),                       # review B1: REPORT
                                   forced=getattr(svc, "arrival_forced", None),
                                   held=getattr(svc, "arrival_held", None))
        if eng.xpool is not None:                       # [xpool]; [integ3] every pool arm (the M2 witnesses)
            info["xpool"] = eng.xpool.report()
        if fp is not None:                                                                       # [xpool]
            info["committed_tokens"] = [list(r.out) for r in rnd.reqs]
        rs = summarize(eng.timer.report())
        info["round"] = rs
        committed = sum(len(r.out) - 1 for r in rnd.reqs)
        r = committed / max(1, req_ticks[0])
        T_tick = rs["device_ms"]["tick"]["mean"] if rs["device_ms"].get("tick") else None
        info.update(ticks=rnd.ticks, committed=committed, request_ticks=req_ticks[0], r=r, T_tick_ms=T_tick,
                    ratio=(r * T_shipped / T_tick) if (T_shipped and T_tick) else None,
                    budget=budget(rs, info["draft_only"]), stats=dict(eng.stats), round_stats=dict(rnd.stats),
                    decision_reads_per_verify_tick=eng.stats["decision_reads"] / max(1, rnd.syncs),
                    residency=dict(k43=dict(svc.metrics.k43), k46=dict(svc.metrics.k46),
                                   peak_retained_slots=svc.metrics.peak_retained_slots, memory=svc.memory(),
                                   refused=svc.metrics.refused, masked=svc.metrics.masked,
                                   draft_pairs=svc.metrics.draft_pairs,
                                   masked_fraction=svc.metrics.masked / max(1, svc.metrics.draft_pairs),
                                   counts=dict(svc.metrics.counts), d2h_reads=getattr(svc, "d2h_reads", None),
                                   sc=sc_counters(svc)),                 # REPORT (RES_SC.md 6)
                    transport=dict(tp.stats), threads_ok=tp.threads_ok())
        info["transport_warm_calls"] = getattr(tp, "warm_calls", None)                           # [integ] gfw1f
        sb_ = getattr(a, "shipped_batch", 0) or eng.B                                            # [integ]
        if sb_ != eng.B:                                       # the shipped decode ran at another batch: no ratio
            info["ratio"] = None
        if T_tick:                                                                               # [integ] REPORT
            info["tok_per_s"] = committed / (rnd.ticks * T_tick / 1e3) if rnd.ticks else None
        info["shipped_tok_per_s"] = (sb_ / (T_shipped / 1e3)) if T_shipped else None
        info["shipped_batch"] = sb_
        info["shipped_source"] = "shipped_pool0.json (NOSI_POOL_BLOCKS 0)"                       # [integ] T_shipped's child
        skv = skv_report(ops.attn)                                                               # [integ] sharedkv
        if skv is not None:
            info["skv"] = skv
            want = getattr(a, "union", "torch")
            other = "fused" if want == "torch" else "torch"
            ub = skv["union_builds"]
            gates.append(GB.gate("%s_SKV_ran" % label, skv["stats"].get("shared", 0) > 0 and ub.get(want, 0) > 0
                                 and ub.get(other, 0) == 0, True,
                                 "the shared kernel ran on keyed calls (a silent fallback to piece 3 would pass every "
                                 "gate): stats %s, union builds %s (--union %s)" % (skv["stats"], ub, want)))
        try:
            info["residency"]["report"] = svc.report()
        except Exception as e:                                        # the report is a REPORT
            info["residency"]["report_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        if getattr(svc, "fusedres", None) is not None:                 # [fusedres2] FR-g gates of the round
            gates.extend(fused_gates(a, label, "round", svc.fusedres.report()))
        wit = dict(twin_variant=getattr(eng, "twin_variant", None),                          # [integ3] review M2
                   batched_calls=getattr(eng, "batched_calls", 0), batched_max=getattr(eng, "batched_max", 0),
                   fused=svc.fusedres.report() if getattr(svc, "fusedres", None) is not None else None,
                   xpool=info.get("xpool"), skv=skv, warm_calls=info["transport_warm_calls"],
                   witness=witness_on(a))                     # [integ5] the arm's role (the actual state: below)
        info["stack_witness"] = {k: v for k, v in wit.items() if k not in ("fused", "xpool", "skv")}
        gates.extend(stack_acted_gates(a, label, wit, eng.B, C))
        mirror = eng.mirror_failures()
        gates.append(GB.gate("%s_completed" % label, True, True, "ticks %d, T_tick %s ms, r %.3f, ratio %s"
                             % (rnd.ticks, None if T_tick is None else round(T_tick, 2), r,
                                None if info["ratio"] is None else round(info["ratio"], 3))))
        gates.append(GB.gate("%s_bookkeeping" % label, not mirror and not eng.void and eng.stats["void_ticks"] == 0, True,
                             "twin counters vs mirrors: %d differ; void ticks %d" % (len(mirror), eng.stats["void_ticks"])))
        gates.append(GB.gate("%s_gilfree_team" % label, tp.threads_ok() or tp.stats["releases"] == 0, True,
                             "requested %d threads; teams %s .. %s over %d releases" % (
                                 a.threads, tp.stats["threads_min"], tp.stats["threads_max"], tp.stats["releases"])))
        info["peak_gb_cum"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() \
            else None                                                # [bigb] REPORT: the run's peak so far
    finally:
        tp.close()
        del eng, svc, tp, rnd                                        # [bigb] rnd too (it held the round's engine)
        GB.free()
    if bare:
        eng, svc, tp = make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, timer_cats=("tick",),
                                   layout=layout)
        try:
            with round_guard(a, eng, info, "bare") as gcl:             # [hostfix] (nothing without its flags)
                tb = time.time()
                rb = run_round(eng, first, [steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA)
                if getattr(a, "eval_role", ""):                      # [evalsplit] the headline's own counts
                    torch.cuda.synchronize()
                    info["bare_round"] = dict(ticks=rb.ticks, committed=sum(len(r.out) - 1 for r in rb.reqs),
                                              wall_s=round(time.time() - tb, 3))
            del rb                                                   # (it holds the engine)
            bs = summarize(eng.timer.report())
            info["bare_T_tick"] = bs["device_ms"].get("tick")
            if hprof:                                                # [hostfix] the headline round's host record
                info["bare_host_profile"] = host_profile_record(eng.timer.report(), gcl, tp)
        finally:
            tp.close()
            del eng, svc, tp
            GB.free()
    return info


XP_TICK_BAND = 0.15                                                                             # [xpool] registered


def compare_layouts(recs, a_key="stream", b_key="pool"):                                       # [xpool]
    """[xpool] recs[arm] = (per-tick rows, per-tick fingerprint lists, committed tokens). Returns (equal, compared
    rows, the first difference or None) of arms a_key and b_key."""
    s, p = recs[a_key], recs[b_key]
    if s[2] != p[2]:
        return False, 0, "committed tokens differ"
    if len(s[0]) != len(p[0]):
        return False, 0, "tick counts differ (%d / %d)" % (len(s[0]), len(p[0]))
    n = 0
    for t, (rs, rp, fs, fq) in enumerate(zip(s[0], p[0], s[1], p[1])):
        if rs != rp:
            return False, n, "tick %d: rows differ" % t
        for k, (x, y) in enumerate(zip(fs, fq)):
            if x != y:
                return False, n, "tick %d row %d %s: fingerprint %d / %d" % (t, k, rs[k], x, y)
            n += 1
    return True, n, None


def one_point_both(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates, label,
                   bare=False, point_fn=None):
    """[xpool] --layout both (docs/R2_XPOOL_PLAN.md section 6; review B1, m6a): three arms in the order stream, pool,
    stream2 (one_point with a layout and fingerprints; every arm with the run's fixed --arrival-lag). Gates:
      <label>_XP_replica    MUST: stream2 equals stream under the SAME comparison (the gate can pass at all: L34)
      <label>_XP_committed  MUST: the pool arm's committed tokens equal the stream arm's (whatever the replica says)
      <label>_XP_bitwise    MUST: rows and every row's fingerprint equal, stream vs pool; VOID (not passed) when the
                            replica failed
      R_<label>_XP_peak / R_<label>_XP_tick  REPORTS: the pool vs the MEAN of the two stream arms (no order
                            confound); the stream arms' own spread is printed with it.
    point_fn: the CPU test's stand-in of one_point."""
    import torch
    point_fn = point_fn or one_point
    out, recs = {}, {}
    for arm, lay in (("stream", "stream"), ("pool", "pool"), ("stream2", "stream")):
        fp = []
        info = point_fn(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates,
                        "%s_%s" % (label, arm), bare=bare, layout=lay, fp=fp)
        vals = torch.cat([v for (_, v) in fp]).tolist() if fp else []             # ONE read per arm, after it
        per_tick, at = [], 0
        for (rows, v) in fp:
            per_tick.append(vals[at:at + len(rows)])
            at += len(rows)
        recs[arm] = ([r for (r, _) in fp], per_tick, info.pop("committed_tokens", None))
        out[arm] = info
    rep_ok, rep_n, rep_diff = compare_layouts(recs, "stream", "stream2")
    rep_ok = rep_ok and rep_n > 0
    gates.append(GB.gate(label + "_XP_replica", rep_ok, True, "stream vs stream2 (the same layout twice): %s, %d rows "
                         "compared" % ("EQUAL" if rep_ok else "DIFFER (%s)" % (rep_diff or "no row compared"), rep_n)))
    com_ok = recs["stream"][2] is not None and recs["stream"][2] == recs["pool"][2]
    gates.append(GB.gate(label + "_XP_committed", com_ok, True, "committed tokens stream vs pool: %s"
                         % ("equal" if com_ok else "DIFFER")))
    same, n, diff = compare_layouts(recs)
    ok = rep_ok and same and n > 0
    if not rep_ok:
        detail = "VOID: the replica differs, so this comparison cannot pass under the run's timing (pool vs stream: %s)" \
                 % ("equal" if same and n > 0 else diff)
    else:
        detail = "%s: %d rows of %d ticks compared" % ("EQUAL" if ok else "DIFFER (%s)" % (diff or "no row compared"), n,
                                                      len(recs["stream"][0]))
    gates.append(GB.gate(label + "_XP_bitwise", ok, True, detail))

    def mean2(key):
        x, y = out["stream"].get(key), out["stream2"].get(key)
        return None if x is None or y is None else (x + y) / 2.0
    ps, pp = mean2("peak_gb_round"), out["pool"].get("peak_gb_round")
    ss, sp = out["stream"].get("store_gb"), out["pool"].get("store_gb")
    drop = None if ps is None or pp is None else round(ps - pp, 3)
    gates.append(GB.gate("R_" + label + "_XP_peak", True, False, "round peak GB stream (mean of 2) %s pool %s (drop %s); "
                         "store GB stream %s pool %s (drop %s)" % (ps, pp, drop, ss, sp,
                                                                   None if ss is None or sp is None else round(ss - sp, 3))))
    ts, tq = mean2("T_tick_ms"), out["pool"].get("T_tick_ms")
    t1, t2 = out["stream"].get("T_tick_ms"), out["stream2"].get("T_tick_ms")
    ratio = None if not ts or tq is None else tq / ts
    spread = None if not t1 or t2 is None else t2 / t1
    gates.append(GB.gate("R_" + label + "_XP_tick", True, False, "T_tick stream %s / %s ms (stream2 / stream %s) pool %s "
                         "ms: ratio pool / mean(stream) %s (registered: prediction 0.97-1.05, falsified outside 1 +- "
                         "%.2f: %s)" % (t1, t2, None if spread is None else round(spread, 4), tq,
                                        None if ratio is None else round(ratio, 4), XP_TICK_BAND,
                                        "inside" if ratio is not None and abs(ratio - 1) <= XP_TICK_BAND
                                        else "OUTSIDE")))
    return dict(backend=kind, layouts=out, xp=dict(replica=rep_ok, replica_difference=rep_diff, committed=com_ok,
                                                    bitwise=ok, rows_compared=n, first_difference=diff,
                                                    peak_drop_gb=drop, tick_ratio=ratio, stream_spread=spread))


ST_TICK_BAND = (0.15, 0.40)                                                                     # [integ] registered


def stack_args(a, name):                                                                        # [integ]
    """[integ] A copy of the parsed arguments with stack `name` applied (the arm's own twin / rebuild / resolve /
    layout / attn / union / warm-up); everything else (point, lag, transport, sel_check) is the run's."""
    import copy
    aa = copy.copy(a)
    apply_stack(aa, name)
    aa.rebuild = aa.stack_rebuild
    aa.variant_list = [(aa.twin, aa.rebuild)]
    return aa


def one_point_stacks(a, model, cache, st, first, ops_for, kind, C, X, R_ver, ext, steps, T_shipped, gates, label,
                     point_fn=None):
    """[integ] --stack-compare (docs/R2_INTEG_PLAN.md, cell stack1a): three arms in the order none, <--stack>, none2
    (one_point with the arm's layout and fingerprints; every arm with the run's fixed --arrival-lag; each arm builds
    its own ops (ops_for(arm args): a fresh SharedKV when the arm's attn is shared)). Gates, as one_point_both's:
      <label>_ST_replica    MUST: none2 equals none (rows, every row's logit fingerprint, committed tokens)
      <label>_ST_committed  MUST: the stack arm's committed tokens equal the none arm's
      <label>_ST_bitwise    MUST: rows and fingerprints equal, none vs stack; VOID when the replica failed
      R_<label>_ST_tick / R_<label>_ST_peak  REPORTS against the MEAN of the two none arms.
    point_fn: the CPU test's stand-in of one_point."""
    import torch
    point_fn = point_fn or one_point
    out, recs = {}, {}
    base = getattr(a, "stack_base", "none") or "none"                   # [integ6] the base arms' stack
    for arm, name in (("none", base), ("stack", a.stack), ("none2", base)):
        aa = stack_args(a, name)
        fp = []
        info = point_fn(aa, model, cache, st, first, ops_for(aa), kind, C, X, R_ver, ext, steps, T_shipped, gates,
                        "%s_%s" % (label, arm), layout=aa.layout, fp=fp)
        vals = torch.cat([v for (_, v) in fp]).tolist() if fp else []             # ONE read per arm, after it
        per_tick, at = [], 0
        for (rows, v) in fp:
            per_tick.append(vals[at:at + len(rows)])
            at += len(rows)
        recs[arm] = ([r for (r, _) in fp], per_tick, info.pop("committed_tokens", None))
        info["stack"] = name
        out[arm] = info
    rep_ok, rep_n, rep_diff = compare_layouts(recs, "none", "none2")
    rep_ok = rep_ok and rep_n > 0
    gates.append(GB.gate(label + "_ST_replica", rep_ok, True, "none vs none2 (the same stack twice): %s, %d rows "
                         "compared" % ("EQUAL" if rep_ok else "DIFFER (%s)" % (rep_diff or "no row compared"), rep_n)))
    com_ok = recs["none"][2] is not None and recs["none"][2] == recs["stack"][2]
    gates.append(GB.gate(label + "_ST_committed", com_ok, True, "committed tokens none vs %s: %s"
                         % (a.stack, "equal" if com_ok else "DIFFER")))
    same, n, diff = compare_layouts(recs, "none", "stack")
    ok = rep_ok and same and n > 0
    if not rep_ok:
        detail = "VOID: the replica differs, so this comparison cannot pass under the run's timing (stack vs none: %s)" \
                 % ("equal" if same and n > 0 else diff)
    else:
        detail = "%s: %d rows of %d ticks compared (stack %s)" % (
            "EQUAL" if ok else "DIFFER (%s)" % (diff or "no row compared"), n, len(recs["none"][0]), a.stack)
    gates.append(GB.gate(label + "_ST_bitwise", ok, True, detail))

    def mean2(key):
        x, y = out["none"].get(key), out["none2"].get(key)
        return None if x is None or y is None else (x + y) / 2.0
    ps, pp = mean2("peak_gb_round"), out["stack"].get("peak_gb_round")
    gates.append(GB.gate("R_" + label + "_ST_peak", True, False, "round peak GB none (mean of 2) %s stack %s (drop %s)"
                         % (ps, pp, None if ps is None or pp is None else round(ps - pp, 3))))
    ts, tq = mean2("T_tick_ms"), out["stack"].get("T_tick_ms")
    ratio = None if not ts or tq is None else tq / ts
    lo, hi = ST_TICK_BAND
    gates.append(GB.gate("R_" + label + "_ST_tick", True, False, "T_tick none %s / %s ms, stack %s ms: ratio stack / "
                         "mean(none) %s (registered band %.2f-%.2f: %s)"
                         % (out["none"].get("T_tick_ms"), out["none2"].get("T_tick_ms"), tq,
                            None if ratio is None else round(ratio, 4), lo, hi,
                            "inside" if ratio is not None and lo <= ratio <= hi else "OUTSIDE")))
    return dict(backend=kind, stacks=out, st=dict(replica=rep_ok, replica_difference=rep_diff, committed=com_ok,
                                                   bitwise=ok, rows_compared=n, first_difference=diff,
                                                   tick_ratio=ratio))


def preflight(a, model, cache, st, first, ops, ext, gates, res, engine_fn=None, round_fn=None):
    """PREFLIGHT (lead, 2026-10-22: an integration fault must cost minutes, not the whole step): before the long arms,
    a SHORT free-running round (--preflight-steps committed tokens per request: draft rows, the first verify ticks,
    their decisions) per backend on the run's transport (MUST; a backend that fails is not run in the long arms), and
    on the gather transport too when the run's is gilfree (REPORT: does the fault follow the transport?). Returns the
    backends that passed on the run's transport (all of them with --preflight-steps 0). engine_fn / round_fn: the
    CPU test's stand-ins of make_engine / r2_tick.run_round (default: the real ones)."""
    import torch
    if round_fn is None:
        from nosi.r2_tick import run_round as round_fn
    engine_fn = engine_fn or make_engine
    if a.preflight_steps <= 0:
        return list(a.backends)
    tps = [a.transport] + (["gather"] if a.transport == "gilfree" else [])
    ok, out = [], []
    for tpname in tps:
        for kind in a.backends:
            label = "SG5_preflight_%s_%s" % (tpname, kind)
            log(label)
            must = tpname == a.transport
            t0 = time.time()
            eng = svc = tp = rnd = None     # [bigb] rnd too: a TickRound holds the engine (read_tokens, on_publish)
            try:
                eng, svc, tp = engine_fn(a, model, cache, st, ops, kind, a.C, a.X, a.R_ver, ext, timer_cats=("tick",),
                                         transport=tpname)
                rnd = round_fn(eng, first, [a.preflight_steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mirror = eng.mirror_failures()
                good = not mirror and rnd.syncs > 0
                detail = "%d ticks, %d decision ticks, twin counters %s, %.0f s" % (
                    rnd.ticks, rnd.syncs, "OK" if not mirror else "DIFFER", time.time() - t0)
            except Exception as e:
                traceback.print_exc()
                good, detail = False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])
            finally:
                if tp is not None:
                    tp.close()
                del eng, svc, tp, rnd       # [bigb] before: rnd kept this engine alive while the next one was built
                GB.free()
            gates.append(GB.gate(label if must else "R_" + label, good or not must, must,
                                 detail if good else "FAILED: " + detail))
            out.append(dict(transport=tpname, backend=kind, passed=good, detail=detail))
            if must and good:
                ok.append(kind)
    res["preflight"] = out
    return ok


def sweep_backend(order, passed):
    """The sweep's backend: the FIRST backend of the step's --backends order that PASSED its preflight; None when none
    did. (p3 used order[0] whatever its preflight: its S-C preflight failed, so the whole sweep was skipped.)"""
    for kind in order:
        if kind in passed:
            return kind
    return None


def run_sweep(a, make_ops, passed, gates, res, save, point):
    """The sweep (--sweep C:X,...; R_ver = X / 2) on sweep_backend(a.backends, passed), recorded as
    res["sweep_backend"]; with no backend passed, a REPORT "sweep skipped: no backend passed". point(ops, kind, C, X,
    R_ver, label) runs one sweep point (one_point); its failures are REPORTS (a configuration result, e.g. a
    CapacityRefusal), as before."""
    res["sweep"] = []
    if not a.sweep_pts:
        res["sweep_backend"] = None
        return
    kind = sweep_backend(a.backends, passed)
    res["sweep_backend"] = kind
    if kind is None:
        gates.append(GB.gate("R_SG5sweep_skipped", True, False, "sweep skipped: no backend passed (preflight order %s)"
                             % (list(a.backends),)))
        save()
        return
    log("SWEEP on %s: the first backend of %s that passed its preflight" % (kind, list(a.backends)))
    ops = make_ops()
    for (C, X, R_ver) in a.sweep_pts:
        label = "SG5sweep_%s_C%d_X%d" % (kind, C, X)
        log(label)
        try:
            res["sweep"].append(point(ops, kind, C, X, R_ver, label))
        except Exception as e:
            traceback.print_exc()
            gates.append(GB.gate("R_" + label, True, False, "EXCEPTION %s: %s (a configuration result, "
                                 "e.g. a CapacityRefusal)" % (type(e).__name__, str(e)[:400])))
        GB.free()
        save()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    if a.baseline_only >= 0:
        return shipped_child(a)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    validated = None
    if a.validated_by:                                               # [integ5] the evidence chain, before any work
        try:
            validated = check_validated(a.validated_by)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 2
    acc_rec = None
    if a.eval_role == "fast-time":                                   # [evalsplit] the accuracy evidence, before any work
        try:
            acc_rec = check_accuracy_record(a.accuracy_record)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 2
    try:
        os.makedirs(a.out)
        gates, res = [], {}
        meta = dict(args={k: v for k, v in vars(a).items() if k not in ("sweep_pts",)}, sweep=a.sweep_pts,
                    nosi_commit=os.environ.get("TICK_COMMIT", "unknown"),
                    draft_note="the drafts read the Residency's READY blocks (paper 3.1): this is the representative "
                               "draft, unlike SG1-SG4's hashed stand-in")

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)
        if a.skip_shipped:                                                                     # [bigb] opt-in
            base = None
            res["shipped"] = None
            gates.append(GB.gate("R_shipped_skipped", True, False, "--skip-shipped: no shipped child; T_shipped None, "
                                 "ratio None"))
        else:
            log("SHIPPED baselines in child processes")
            base, err = run_child(a, argv, 0, os.path.join(a.out, "shipped_pool0.json"))
            res["shipped"] = base
            gates.append(GB.gate("shipped_baseline", base is not None, True, err or "T_shipped median %.2f ms"
                                 % base["stats"]["median"]))
        meta["log_slots"] = a.log_slots_n                                                      # [bigb] None: 1 << 16
        if a.pool_baseline:
            pool = a.C + a.X + 4 - 64
            if a.pool_rule in ("stream", "both"):                                              # [xpool] the default
                eq, err = run_child(a, argv, max(0, pool), os.path.join(a.out, "shipped_pool_eq.json"))
                res["shipped_equal_memory"] = eq
                gates.append(GB.gate("R_shipped_equal_memory", True, False, err or "NOSI_POOL_BLOCKS %d: median %.2f ms"
                                     % (pool, eq["stats"]["median"])))
            if a.pool_rule in ("k5", "both"):                                                  # [xpool] the K5 rule
                pk = pool_rule_value(a, "k5")
                eq, err = run_child(a, argv, max(0, pk), os.path.join(a.out, "shipped_pool_eq_k5.json"))
                res["shipped_equal_memory_k5"] = eq
                gates.append(GB.gate("R_shipped_equal_memory_k5", True, False, err or "NOSI_POOL_BLOCKS %d (k5 rule): "
                                     "median %.2f ms" % (pk, eq["stats"]["median"])))
        save()
        T_shipped = base["stats"]["median"] if base else None
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        from nosi.r2_bi_decode import BIOps
        from nosi.r2_decode_path import build_ragged
        if not NL.R2_DECODE or NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS 0 are needed")
        docs, sha, rep = load_docs(a)
        meta.update(workload_sha256=sha, repeated_docs=rep, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
        ext = None
        if a.transport == "gilfree":
            gp = GB.load_by_path(os.path.join(a.gilfree, "gilfree_pack.py"), "gilfree_pack_pinned")
            ext = gp.load(a.gilfree_build)
            meta["gilfree_openmp"] = int(getattr(ext, "OPENMP", 0))
        meta["transport"] = a.transport
        meta["layout"] = a.layout                                                              # [xpool]
        meta["twin"] = a.twin                                                                  # [twin-variant]
        meta.update(stack=a.stack or None, stack_compare=a.stack_compare, attn=a.attn, union=a.union,   # [integ]
                    sel_check=a.sel_check, shared_unchecked=a.shared_unchecked, gilfree_warm_up=a.gilfree_warm_up,
                    b2_resolve=a.b2_resolve, shipped_batch=a.shipped_batch or a.batch,
                    sharedkv=a.sharedkv if a.attn != "per_row" else None)
        meta.update(stack_base=a.stack_base if a.stack_compare else None)                        # [integ6]
        if a.host_profile:                                                                     # [hostfix] item 0
            meta.update(host_profile=True, host_prof_cats=HOST_PROF_CATS,
                        host_profile_note="host clocks only (perf_counter); per point: draft_only_host_profile, "
                                          "host_profile (instrumented round), bare_host_profile (the headline)")
        if a.lean_time:                                                                        # [hostfix] item 2
            meta.update(lean_time=True, lean_dropped=list(LEAN_DROPPED),
                        lean_note="REPORT-only: residency k43 / k46 / k48, peak_retained and draft_refused are NOT "
                                  "measured in this record (zeros); every MUST gate's input is kept")
        if a.wait_index:                                                                       # [hostfix] item 3
            meta.update(wait_index=True)
        if a.b2_fewer_reads:                                                                   # [hostfix] item 5
            meta.update(b2_fewer_reads=True)
        if a.gc_freeze:                                                                        # [hostfix] item 1
            meta.update(gc_freeze=True, gc_freeze_note="each timed phase: collect, freeze, disable before; unfreeze, "
                                                       "enable, collect after (per point: gc_freeze[phase])")
        if a.eval_role:                                                                        # [evalsplit]
            meta.update(eval_role=a.eval_role, accuracy_record=acc_rec, mem_cap_gb=a.mem_cap_gb,
                        eval_bands=a.eval_bands_v,
                        eval_note="FAST (natural cuBLAS) in draft and verify rows; accuracy equivalence from the "
                                  "record, at its scope (batch / GPU); bitwise claims are EXACT's (G-c strict) cells")
        meta.update(stack_role=a.stack_role or None, validated_by=validated, free_arm=a.free_arm,   # [integ5]
                    witness=witness_on(a), validated_note=(None if a.stack_role != "time" else
                    "bitwise equality proven at the validated cells' B and lag only (see validated_by)"))
        if a.variants:                                                                         # [batched-rebuild]
            meta["variants"] = ["%s/%s" % v for v in a.variant_list]                            # [batched-rebuild]
        a.twin, a.rebuild = a.variant_list[0]           # [batched-rebuild] the preflight / sweep arms: the first one
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s" % (time.time() - t0))
        with torch.inference_mode():
            prompts = [docs[b, :a.S] for b in range(a.batch)]
            horizon = max(a.steps, a.sweep_steps) + a.LAMBDA + 64
            cache, st, first_t = build_ragged(model, prompts, horizon, f4_fix=True)
            for lay in cache.layers:                                   # the tick's own store replaces the windows
                e = lay.cache_engine
                e._k_gpu = e._v_gpu = e._kv_bias_gpu = None
            GB.free()
            first = [int(x) for x in first_t.view(-1).tolist()]
            meta["build_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            meta["build_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)             # [bigb] REPORT
            backends = preflight(a, model, cache, st, first, BIOps(select_attn(a, attn), a.m_fix, gemm=a.modes[0]),
                                 ext, gates, res)                                              # [integ] --attn
            meta["preflight_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)         # [bigb] REPORT
            save()
            if not backends:
                log("PREFLIGHT: no backend ran a short round on the %s transport: the long arms are skipped"
                    % a.transport)
            res["points"] = []
            for mode in a.modes:
                ops = BIOps(select_attn(a, attn), a.m_fix, gemm=mode)                          # [integ] --attn
                for kind in backends:
                    for variant in a.variant_list:                                             # [batched-rebuild]
                        a.twin, a.rebuild = variant                                            # [batched-rebuild]
                        label = variant_label(a, mode, kind, variant)                          # [batched-rebuild]
                        log("%s at C %d X %d R_ver %d" % (label, a.C, a.X, a.R_ver))
                        try:
                            if a.stack_compare:                                                # [integ]
                                pt = one_point_stacks(a, model, cache, st, first,
                                                      lambda aa, m=mode: BIOps(select_attn(aa, attn), a.m_fix, gemm=m),
                                                      kind, a.C, a.X, a.R_ver, ext, a.steps, T_shipped, gates, label)
                            else:
                                point = one_point_both if a.layout == "both" else one_point            # [xpool]
                                pt = point(a, model, cache, st, first, ops, kind, a.C, a.X, a.R_ver, ext, a.steps,
                                           T_shipped, gates, label, bare=a.bare)
                            pt["twin"], pt["rebuild"] = variant                                # [batched-rebuild]
                            if a.eval_role:                                                    # [evalsplit]
                                pt["label"] = label
                            res["points"].append(pt)
                            if a.free_arm:                         # [integ5] REPORT: free arrival, after the headline
                                import copy
                                af = copy.copy(a)
                                af.arrival_lag = 0
                                log("%s_free: the same stack at free arrival (REPORT timing)" % label)
                                pf = one_point(af, model, cache, st, first,
                                               BIOps(select_attn(af, attn), a.m_fix, gemm=mode), kind, a.C, a.X,
                                               a.R_ver, ext, a.steps, T_shipped, gates, label + "_free", bare=False)
                                res["free_arm"] = pf
                                gates.append(GB.gate("R_%s_free_tick" % label, True, False,
                                                     "free arrival T_tick %s ms (instrumented) vs the headline (lag "
                                                     "%d, bare) %s ms; bitwise equality is proven only at lag %d"
                                                     % (pf.get("T_tick_ms"), a.arrival_lag, headline_tick_ms(pt),
                                                        a.arrival_lag)))
                        except Exception as e:
                            traceback.print_exc()
                            gates.append(GB.gate(label + "_completed", False, True, "EXCEPTION %s: %s"
                                                 % (type(e).__name__, str(e)[:400])))
                        GB.free()
                        save()
            a.twin, a.rebuild = a.variant_list[0]                                              # [batched-rebuild]
            run_sweep(a, lambda: BIOps(select_attn(a, attn), a.m_fix, gemm=a.modes[0]), backends, gates, res, save,
                      lambda ops, kind, C, X, R_ver, label: one_point(a, model, cache, st, first, ops, kind, C, X, R_ver,
                                                                      ext, a.sweep_steps, T_shipped, gates, label))
        meta["peak_gb"] = round(max(_PEAK[0], torch.cuda.max_memory_allocated()) / 1e9, 2)     # [xpool] resets
        if a.stack_bands_v is not None:                                                        # [integ3] M3
            write_stack_bands(a, meta, res, gates)
        if a.eval_role == "fast-time":                                                         # [evalsplit]
            eval_gates(a, meta, res, gates)
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:400]), flush=True)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("R2_TICK_SG5_DONE failed=%d" % nfail, flush=True)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
