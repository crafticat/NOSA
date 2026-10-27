"""Task 0.1 "p0prof" (kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md): the profile of the cached-NOSA
decode step and the ceiling for speculation, with the verify's extra selection (P0_PILOT_AUTOPSY.md 7.4, 9.2). One
process, one model, one (C, B, S) per process (cache_engine reads NOSI_POOL_BLOCKS once, at import). Workload: the PG19
94-document 16128-token file (S 16128, request b = row b % 94), the 55-document 65280-token file (--context 65280,
request b = row b % 55) or the 24-document 130816-token file (--context 130816, request b = row b % 24).
benchmarks/Efficiency/r2_base_profile_step.sh runs it.
p0sweep (the second part, PLAN Task 0.1 continued): the map of the speculation ceiling where it could still win. Small B
keeps the GEMMs near their weight-read floor; small C or a long context makes the fetch a large share. Arm T also runs
the verify window (VerifyProbe). The cell reports the GEMM time against the rows and the ceiling with the MEASURED
verify model (VERIFY_FORMULA, verify_ceiling). The registered block is PRED_SWEEP.

ARMS (in this order, one model):
  H  THE HEADLINE. r2_base_cell.run_shipped (the function of the mb1 record, 3,387.7 tok/s) with r2_base_cell.CudaClock.
     No timer and no patch: this cell installs nothing before H ends. MUST H-UNPATCHED compares every patch target by
     identity before and after H. The steady statistic is r2_base_cell's (the median period of the non-re-warm steps).
  T  THE TIMER ARM. It runs only with the OPT-IN flag --stage-timers, after H and after a new prefill. The same decode
     (decode_loop = run_shipped with hook calls). CUDA events at the stage boundaries of the shipped step
     (StageRecorder). The boundaries come from wrappers on the module-level names that the shipped step calls
     (Patcher); no file of the engine changes. Three step windows (no re-warm inside any of them):
       profiler  --prof-start, --prof-steps (default 8, 3): the wrappers removed, torch.profiler on: launches per step,
                 device busy time, kernel classes (r2_p3_profile.parse_trace / aggregate). The steps before and after
                 it are not timed per stage (the profiler start / stop syncs fall in them).
       selection --sel-start, --sel-steps, --sel-rows (default 24, 8, 4): SelectionProbe. After the shipped stage-1
                 call of each layer, the selection runs again with the SAME existing ops (infllmv2_attn_stage1_fast,
                 nosa_pooling, torch.topk / scatter / masked_fill), eagerly, into the probe's own buffers: once for 1
                 row per request and once for R rows per request (1 draft + R - 1 verify positions) in ONE stage-1
                 call with cu_seqlens_q = 0, R, 2R, ... (the varlen route; the rows repeat the request's query). No new
                 kernel. The first step of the window is a warm-up (Triton specialization, buffers) and is not used.
       verify    --vfy-start, --vfy-steps, --gemm-mults (default 34, 12, 1,2,3,4): VerifyProbe, the measured inputs
                 of the verify model. Per layer, with existing ops only, into the probe's own tensors: (1) 'union': a
                 copy of the shipped selection (topk_idx) into a ring of R steps; when the ring is full, the number of
                 distinct block ids of each (head, request) over R consecutive steps (sort, compare, sum); (2)
                 'load_count': the host loads of the layer (load mask >= 0 after the pool lookup); (3) 'gemm_x<m>':
                 the GEMM stack of the shipped step (the group gemm_mlp) on m x B rows, for each m. The first step
                 is a warm-up for the GEMM and fetch times; the union counts from the step where the ring is full.
       steady    every other non-re-warm step: the stage timers.
     The host time of each torch.cuda.graph enter and exit (CaptureTimer) and the gc time per step (GcTimer).
  N  THE NSYS ARM. It runs only with --nsys-window W, and it replaces H and T. The same decode for --steps steps;
     cudaProfilerStart before step W and cudaProfilerStop after step W + 19 (the step script starts the process under
     `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi`); one NVTX range per step of the window.

STAGES (arm T). A boundary opens the stage that has its label. The stage ends at the next boundary. All operations of
the shipped step run on the current stream (cache_engine.py:490-492), so the stages of one step partition its period
(closure). Per layer, in stream order (pool 0 has no pool_resolve, pool_swap or map_commit):
  qkv_gemm      LlamaLayer.decode_forward entry: input rmsnorm + wqkv GEMM (nosa_llama.py:553-556)
  qkv_prep      nosa_linear entry: q/k/v split + cis, rope, key compression, cis bookkeeping (:558-585)
  stage1        infllmv2_attn_stage1_fast entry: the stage-1 score kernel (:592-601)
  [sel steps]   sel_prep (the R-row query / cis copies), sel1_stage1, sel1_pool, selr_stage1, selr_pool (the probe)
  pool_topk     the stage-1 exit: score / cis copies + the pooling and top-k graph replay (:607-612)
  tail_write    InfLLMv2Cache.decode_update_kv entry: the tail K / V / bias write (cache_engine.py:453-476)
  diff          diff.diff_offload entry: the diff kernel (:480); at pool 0 also the map commit copy (:377)
  pool_resolve  _pool.pool_update entry: the victim-pool lookup (:504-507)
  pool_swap     _flash_pool_swap entry: the D2D swap (:508-510)
  map_commit    _flash_pool_swap exit: the LRU stamp and the map commit copy (:511-514)
  fetch_k       flash_h2d_from_mask entry: the K gather from pinned host memory (:518)
  fetch_v       flash_h2d_from_mask_bias entry: the V + bias gather (:519)
  writeback     its exit: the tail write-back (one step in 64) and the return (:523-538)
  attention     flash_attn_nosa_with_kvcache entry (nosa_llama.py:634-641)
  ffn_gemm      its exit: wo GEMM, residual, rmsnorm, gate_up GEMM, SiLU, down GEMM, residual (:645-660)
  inter / final decode_forward exit: the Python loop to the next layer / the final rmsnorm + lm_head (:839-843)
  [vfy steps]   'union' before tail_write, 'load_count' before fetch_k, 'gemm_prep' and 'gemm_x<m>' after ffn_gemm
Per step: embed (step start: embedding + loop set-up), sample (decode_inference return: argmax, the token copy, the
position, the next re-warm check). GROUPS map the labels to the stages of the plan: selection, resolve (diff + pool
resolve), pool_swap, fetch, attention, gemm_mlp, rest.

MUST: H-RUN, H-UNPATCHED; with T: T-RUN, T-EQ (T's tokens equal H's), T-CHAIN (every steady timed step has the
expected boundary sequence), T-SEL (every selection step has the probe's sequence and no probe error), T-VFY (every
verify step has the probe's sequence, no probe error, the expected number of probe calls), T-RESTORED (every patch
target is the original object after T), T-PROF (the profiler saw GPU ops and launches); with N: N-RUN.
REPORT: everything else, the REGISTERED predictions (PRED_P0 and PRED_SWEEP, the lead stamps them), the ceiling
(CEILING_FORMULA) and the ceiling with the measured verify model (VERIFY_FORMULA).
EXIT: the number of failed MUST gates; 20 refused; 30 an exception outside an arm.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_base_cell as BC          # noqa: E402
import r2_p3_profile as P3         # noqa: E402
import r2_ragged_gate as RG        # noqa: E402

NSYS_STEPS = 20
PROF_START, PROF_STEPS = 8, 3
SEL_START, SEL_STEPS, SEL_ROWS = 24, 8, 4
CONTEXTS = (16128, 65280, 130816)
ACCEPTED = (1, 2, 3, 4)
VFY_START, VFY_STEPS = 34, 12
GEMM_MULTS = (1, 2, 3, 4)

# ------------------------------------------------------------------ REGISTERED (Task 0.1; THE LEAD STAMPS)
# Every "TBD" is a placeholder. The lead writes the values and REGISTERED_UTC (from `date -u` in the same command, L33)
# in ONE commit BEFORE the run. A "TBD" band gives the verdict UNREGISTERED, never CONFIRMED or REFUTED. head_band is
# not a new prediction: it is the plan's Global Constraint "flags off = baseline: step time within +-2 % of the same-job
# baseline" (arm H at the mb1 cell against the same job's mb1), so it carries its band now.
# Cell keys (cell_key): 128_216 (p0c128), 63_344 (p0c63b344), 63_216 (p0c63b216), 128_63_s130816 (p0k128b63).
# STAMP 2026-10-27T16:25:29Z (the lead). BASIS: ncu step of job 2175791 at C 128 B 216 (selection 8.9, resolve 1.2,
# pool swap 5.4, gather 7.6, attention 19.2, GEMM 19.9 ms; MAP_cached_nosa.md 2.1); original NOSA B 344 C 63 1,908.8
# tok/s (180 ms per step) with fetch about 1.36x the rest (paper Figure 2); P0_PILOT_AUTOPSY.md (128K B 63: selection
# about 26 ms per position). Stage bands are wide on purpose: event timers on a device-bound step add their own cost.
# The 63_* cells run without the pool, so their pool_swap band is (0, 1).
REGISTERED_UTC = "2026-10-27T16:25:29Z"
_CELL_KEYS = ("128_216", "63_344", "63_216", "128_63_s130816")
PRED_P0 = {
    "head_band": (0.98, 1.02),     # P0-HEAD: H steady median / same-job mb1 steady median (same C, B, S only)
    "ovh_band": (1.0, 1.15),       # P0-OVH: T steady median / H steady median (the cost of the timers)
    "stage_ms": {
        "128_216": {"selection": (6, 13), "resolve": (0.7, 2.5), "pool_swap": (3.5, 8), "fetch": (5, 11),
                    "attention": (15, 24), "gemm_mlp": (16, 25), "rest": (1, 8)},
        "63_344": {"selection": (9, 20), "resolve": (1, 4), "pool_swap": (0, 1), "fetch": (70, 130),
                   "attention": (24, 38), "gemm_mlp": (25, 40), "rest": (2, 14)},
        "63_216": {"selection": (6, 13), "resolve": (0.7, 3), "pool_swap": (0, 1), "fetch": (40, 80),
                   "attention": (15, 24), "gemm_mlp": (16, 25), "rest": (1, 10)},
        "128_63_s130816": {"selection": (15, 40), "resolve": (0.5, 3), "pool_swap": (1, 6), "fetch": (2, 12),
                           "attention": (8, 20), "gemm_mlp": (6, 12), "rest": (1, 10)},
    },
    #                                P0-STAGE-<group>: {group: (low, high)} ms per step, T steady median, per cell
    "launches": {k: (800, 4000) for k in _CELL_KEYS},
    #                                P0-LAUNCH: (low, high) enqueued launches per step (kernel + graph + memcpy + memset)
    "gap_ms": {"128_216": (0, 6), "63_344": (0, 6), "63_216": (0, 6), "128_63_s130816": (0, 10)},
    #                                P0-GAP: (low, high) host gap = H steady median - device busy (ms per step)
    "recap_ms": (4500, 7500),      # P0-RECAP: (low, high) T period of a later re-capture step (not the first), ms
    "recap_gc_share": (0.5, 1.0),  # P0-RECAP-GC: hypothesis: gc dominates a re-capture
    "sel_extra_ms": {"128_216": (2, 10), "63_344": (3, 15), "63_216": (2, 10), "128_63_s130816": (5, 40)},
    #                                P0-SEL: (low, high) extra selection of a round, joint (R rows - 1 row), ms per step
    "ceil_speedup": {"128_216": (1.07, 1.2), "63_344": (1.6, 3.2), "63_216": (1.4, 2.6),
                     "128_63_s130816": (1.02, 1.2)},
    #                                P0-CEIL: (low, high) max speedup of the draft, gather_only
}
CEILING_FORMULA = (
    "Speculation pays only if draft_ms + (verify_extra_ms incl. K extra selections) / accepted_per_round < baseline_ms "
    "per committed token (plan Task 0.1 + P0_PILOT_AUTOPSY.md 7.4). K = R - 1 verify positions per round (R selection "
    "rows per request: 1 draft + K verify). The draft step can save at most the fetch stages: draft_ms >= baseline_ms - "
    "save_ms, save_ms = baseline_ms x save_frac (gather_only: fetch_k + fetch_v; fetch_path: diff + pool_resolve + "
    "pool_swap + map_commit + fetch_k + fetch_v; each the median per-step fraction of the T period). The K extra "
    "selections of a round cost sel_extra_joint_ms = selR_ms - sel1_ms (one stage-1 call with R rows per request, "
    "measured) or sel_extra_serial_ms = K x sel1_ms (one call per position, the pilot's way). So the step ceiling is "
    "baseline_ms / (baseline_ms - save_ms), the tok/s ceiling is B x 1000 / (baseline_ms - save_ms), and a round of `a` "
    "accepted tokens leaves verify_budget_ms(a) = a x save_ms - sel_extra_ms for all other verify work; a budget <= 0 "
    "means that no round with `a` accepted tokens can pay. baseline_ms = arm H steady median (same process); the "
    "selection times are arm T's (CUDA events, the probe window).")

# ------------------------------------------------------------------ REGISTERED p0sweep (THE LEAD STAMPS)
# Every "TBD" is a placeholder, as in PRED_P0. The lead writes the values and REGISTERED_SWEEP_UTC (from `date -u` in
# the same command, L33) in ONE commit BEFORE the run. A "TBD" gives the verdict UNREGISTERED. A band is (low, high).
# Cell keys (cell_key): 63_64 (p0c63b64), 63_128 (p0c63b128), 128_64 (p0c128b64), 63_127_s65280 (p0k64c63),
# 128_127_s65280 (p0k64c128). The P0-* lines of these cells read PRED_P0: its per-cell bands have no key for them
# (UNREGISTERED); its global bands (ovh_band, recap_ms, recap_gc_share) were stamped for p0all, so the scorer reads
# them as REPORT for the sweep cells.
REGISTERED_SWEEP_UTC = "TBD"
_SWEEP_KEYS = ("63_64", "63_128", "128_64", "63_127_s65280", "128_127_s65280")
PRED_SWEEP = {
    "stage_ms": {
        "63_64": {"selection": "TBD", "resolve": "TBD", "pool_swap": "TBD", "fetch": "TBD", "attention": "TBD",
                  "gemm_mlp": "TBD", "rest": "TBD"},
        "63_128": {"selection": "TBD", "resolve": "TBD", "pool_swap": "TBD", "fetch": "TBD", "attention": "TBD",
                   "gemm_mlp": "TBD", "rest": "TBD"},
        "128_64": {"selection": "TBD", "resolve": "TBD", "pool_swap": "TBD", "fetch": "TBD", "attention": "TBD",
                   "gemm_mlp": "TBD", "rest": "TBD"},
        "63_127_s65280": {"selection": "TBD", "resolve": "TBD", "pool_swap": "TBD", "fetch": "TBD",
                          "attention": "TBD", "gemm_mlp": "TBD", "rest": "TBD"},
        "128_127_s65280": {"selection": "TBD", "resolve": "TBD", "pool_swap": "TBD", "fetch": "TBD",
                           "attention": "TBD", "gemm_mlp": "TBD", "rest": "TBD"},
    },
    #   SW-STAGE-<group>: {group: (low, high)} ms per step, T steady median, per cell
    "gemm_x1_ms": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                   "128_127_s65280": "TBD"},
    #   SW-GEMM-X1: the probe's GEMM stack on B rows, ms per step (32 layers + the final rmsnorm and lm_head)
    "gemm_xr_ms": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                   "128_127_s65280": "TBD"},
    #   SW-GEMM-XR: the same on R x B rows (R = --sel-rows, 4), ms per step
    "gemm_slope": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                   "128_127_s65280": "TBD"},
    #   SW-GEMM-SLOPE: ms per row, the least-squares slope over the probe points (B, 2B, 3B, 4B rows)
    "sel_extra_ms": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                     "128_127_s65280": "TBD"},
    #   SW-SEL: the extra selection of a round, joint (R rows - 1 row), ms per step
    "union_extras": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                     "128_127_s65280": "TBD"},
    #   SW-UNION: extras per (head, request) = the mean union of R consecutive selections - topk (vr1: 8.137)
    "loads_per_stream": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                         "128_127_s65280": "TBD"},
    #   SW-LOADS: host loads per (layer, head, request) per step, in the verify window
    "union_fetch_ms": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                       "128_127_s65280": "TBD"},
    #   SW-UFETCH: the union-extras fetch (all extras from the host), ms per joint pass
    "joint_steps": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                    "128_127_s65280": "TBD"},
    #   SW-JOINT: the joint pass (measured fetch) / the H steady median
    "vs_cached_floor": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                        "128_127_s65280": "TBD"},
    #   SW-CACHED-FLOOR: committed tok/s / 3,387.7 at acceptance 0.995, draft floor, measured fetch
    "vs_cached_dr1": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                      "128_127_s65280": "TBD"},
    #   SW-CACHED-DR1: the same with dr1's share of the gather for the pool (dr1_for_pool)
    "vs_own_floor": {"63_64": "TBD", "63_128": "TBD", "128_64": "TBD", "63_127_s65280": "TBD",
                     "128_127_s65280": "TBD"},
    #   SW-OWN-FLOOR: committed tok/s / the cell's own H steady tok/s at acceptance 0.995, draft floor, measured fetch
}
# The verify model (not predictions; the basis of each number is in the comment).
VERIFY_ACCEPT = (0.995, 1.0)       # per-token acceptance; dr1 measured 0.99475 (SCORING_dr1.md 7)
REF_CACHED_TOK_S = 3387.7          # cached NOSA, mb1 2190876 (C 128 B 216, 16K): the same-memory best baseline
WEIGHT_FLOOR_MS = 10.6             # gemm_mlp intercept (SCORING_p0all.md 3: the C 63 slope from B 216 to B 344)
DRAFT_RECOVERY = (("floor", 1.0), ("dr1_69", 0.6926), ("dr1_44", 0.4440))
#   the share of the gather that one draft step saves. floor: all of it (draft = baseline - gather). dr1 saved 3.368 ms:
#   69.26 % of p0all's C 128 gather (4.863 ms; SCORING_p0all.md 8.1, applied at C 63 in 8.4) and 44.4 % of the ncu
#   gather (7.585 ms; SCORING_dr1.md 9). dr1_for_pool: dr1_69 at pool 0 (C 63), dr1_44 with a pool (C 128).
VERIFY_FORMULA = (
    "A round = K = R - 1 draft steps + 1 joint pass of R rows per request (1 draft row + K verify rows); it commits "
    "E(a) = 1 + a + ... + a^(R-1) tokens per request at per-token acceptance a. draft_ms = baseline_ms - recovery x "
    "gather_ms (gather_ms = baseline_ms x the gather_only fraction; recovery 1 = the floor). joint_ms = baseline_ms + "
    "(gemm(R x B rows) - gemm(B rows)) + (selection of R rows - selection of 1 row) + union_fetch_ms: the shipped step "
    "with its GEMM stack and its selection on R rows per request, ONE attention (perfect K / V sharing over the R "
    "rows), and the fetch of the union extras. union_fetch_ms = extras x streams x ms per load: extras = the mean "
    "union of R consecutive shipped selections per (head, request) - topk; streams = layers x KV heads x B; ms per "
    "load = the fetch time / the host loads of the same steps (all extras taken as host loads: exact at pool 0, an "
    "upper bound with a pool). Variant 'hidden': union_fetch_ms = 0 (a background prefetch hides it). Committed tok/s "
    "= 1000 x B x E(a) / (K x draft_ms + joint_ms). Break-even joint_ms against a rate X = 1000 x B x E(a) / X - K x "
    "draft_ms (X = 3,387.7, cached NOSA; or the cell's own 1000 x B / baseline_ms). The GEMM and selection times are "
    "the medians of arm T's probes (eager); baseline_ms = the arm H steady median.")


def dr1_for_pool(pool):
    """The dr1 share of the gather for a cell (DRAFT_RECOVERY name): dr1_69 at pool 0, dr1_44 with a pool."""
    return "dr1_69" if int(pool) == 0 else "dr1_44"


# ------------------------------------------------------------------ stages (pure parts, CPU-tested)
LAYER_SEQ_POOL = ("qkv_gemm", "qkv_prep", "stage1", "pool_topk", "tail_write", "diff", "pool_resolve", "pool_swap",
                  "map_commit", "fetch_k", "fetch_v", "writeback", "attention", "ffn_gemm")
LAYER_SEQ_NOPOOL = tuple(x for x in LAYER_SEQ_POOL if x not in ("pool_resolve", "pool_swap", "map_commit"))
SEL_LABELS = ("sel_prep", "sel1_stage1", "sel1_pool", "selr_stage1", "selr_pool")
STEP_LABELS = frozenset(("embed", "sample", "rewarm", "window", "edge"))
GROUPS = (("selection", ("stage1", "pool_topk")),
          ("resolve", ("diff", "pool_resolve")),
          ("pool_swap", ("pool_swap",)),
          ("fetch", ("fetch_k", "fetch_v")),
          ("attention", ("attention",)),
          ("gemm_mlp", ("qkv_gemm", "ffn_gemm", "final")),
          ("rest", ("embed", "qkv_prep", "tail_write", "map_commit", "writeback", "inter", "sample")))
SAVES = (("gather_only", ("fetch_k", "fetch_v")),
         ("fetch_path", ("diff", "pool_resolve", "pool_swap", "map_commit", "fetch_k", "fetch_v")))


def log(msg):
    print("[r2_p0prof %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def cell_key(C, B, S=None):
    """The key of a cell in PRED_P0: '<C>_<B>' at S 16128, '<C>_<B>_s<S>' otherwise."""
    S = BC.S0 if S is None else int(S)
    return "%d_%d" % (int(C), int(B)) + ("" if S == BC.S0 else "_s%d" % S)


def vfy_labels(mults=GEMM_MULTS):
    """The verify probe's labels per layer: union (before tail_write), load_count (before fetch_k), gemm_prep and one
    gemm_x<m> per multiplier (after ffn_gemm, before inter / final)."""
    return ("union", "load_count", "gemm_prep") + tuple("gemm_x%d" % int(m) for m in mults)


def expected_sequence(n_layers: int, pool: bool, sel: bool = False, vfy=None):
    """The boundary labels of one timed step, in stream order (sel: a selection-probe step; vfy: the GEMM multipliers
    of a verify-probe step)."""
    body = list(LAYER_SEQ_POOL if pool else LAYER_SEQ_NOPOOL)
    if sel:
        i = body.index("stage1") + 1
        body[i:i] = SEL_LABELS
    if vfy is not None:
        body.insert(body.index("tail_write"), "union")
        body.insert(body.index("fetch_k"), "load_count")
        body.extend(vfy_labels(vfy)[2:])
    seq = ["embed"]
    for layer in range(int(n_layers)):
        seq.extend(body)
        seq.append("final" if layer == int(n_layers) - 1 else "inter")
    seq.append("sample")
    return tuple(seq)


class EvClock(BC.CudaClock):
    """r2_base_cell.CudaClock plus wait(event): the host waits for one recorded event."""

    def wait(self, e):
        e.synchronize()


class StageRecorder:
    """CUDA events at the stage boundaries, in stream order. A boundary opens the stage named by its label; the stage
    ends at the next boundary (for the last boundary of a step: the first boundary of the next step, or the final
    event). RING sets of events: the host reads a step's times only when its set comes round again (ring - 1 steps
    later), so the timers add no device wait in the steady state. Only `inner` steps record the inner boundaries; every
    step records its first boundary, so every step has its period. `expected`: {kind: label sequence}; each inner step
    of a kind is checked against it (the chain)."""

    def __init__(self, clock, n_layers, expected=None, ring=4, cap=512):
        if int(ring) < 2:
            raise ValueError("ring >= 2: a step ends at the first event of the next step")
        self.clock, self.n_layers, self.ring = clock, int(n_layers), int(ring)
        self.expected = {k: tuple(v) for k, v in (expected or {}).items()}
        self.ev = [[clock.event() for _ in range(cap)] for _ in range(self.ring)]
        self.lab = [[None] * cap for _ in range(self.ring)]
        self.lay = [[-1] * cap for _ in range(self.ring)]
        self.n = [0] * self.ring
        self.held = [None] * self.ring              # (k, t, inner, kind) of the step a set holds
        self.succ = [None] * self.ring              # the set of the next step (-1: the final event)
        self.k = 0
        self.cur = None
        self.inner = False
        self.capturing = False
        self.layer_idx = -1
        self.steps = {}
        self.layer_sum = {}
        self.n_inner = {}
        self.chain = {}
        self.closure_max_ms = 0.0

    def step_begin(self, t, inner, label, kind="main"):
        s = self.k % self.ring
        self.k += 1
        if self.held[s] is not None:
            self._harvest(s, self.ev[self.succ[s]][0])
        if self.cur is not None:
            self.succ[self.cur] = s
        self.held[s] = (self.k, int(t), bool(inner), kind)
        self.succ[s] = None
        self.n[s] = 0
        self.cur = s
        self.inner = bool(inner)
        self.layer_idx = -1
        self._rec(label)

    def layer(self, idx):
        self.layer_idx = int(idx)

    def mark(self, label):
        if self.inner and not self.capturing:
            self._rec(label)

    def _rec(self, label):
        s = self.cur
        i = self.n[s]
        if i == len(self.ev[s]):
            self.ev[s].append(self.clock.event())
            self.lab[s].append(None)
            self.lay[s].append(-1)
        self.clock.record(self.ev[s][i])
        self.lab[s][i] = label
        self.lay[s][i] = -1 if label in STEP_LABELS else self.layer_idx
        self.n[s] = i + 1

    def finish(self):
        """The final event, then every held step (oldest first)."""
        end = self.clock.event()
        self.clock.record(end)
        if self.cur is not None:
            self.succ[self.cur] = -1
        for _, s in sorted((h[0], s) for s, h in enumerate(self.held) if h is not None):
            self._harvest(s, end if self.succ[s] == -1 else self.ev[self.succ[s]][0])
        self.cur = None
        self.inner = False

    def _harvest(self, s, nxt):
        _, t, inner, kind = self.held[s]
        self.held[s] = None
        self.clock.wait(nxt)
        n, ev, lab, lay = self.n[s], self.ev[s], self.lab[s], self.lay[s]
        el = self.clock.elapsed
        row = dict(period_ms=el(ev[0], nxt), inner=inner, label=lab[0], kind=kind)
        if inner:
            stages = {}
            for i in range(n):
                ms = el(ev[i], ev[i + 1] if i + 1 < n else nxt)
                stages[lab[i]] = stages.get(lab[i], 0.0) + ms
                if kind == "main" and 0 <= lay[i] < self.n_layers:
                    acc = self.layer_sum.get(lab[i])
                    if acc is None:
                        acc = self.layer_sum[lab[i]] = [0.0] * self.n_layers
                    acc[lay[i]] += ms
            row["stages"] = stages
            self.n_inner[kind] = self.n_inner.get(kind, 0) + 1
            self.closure_max_ms = max(self.closure_max_ms, abs(sum(stages.values()) - row["period_ms"]))
            exp = self.expected.get(kind)
            if exp is not None:
                c = self.chain.setdefault(kind, dict(ok=0, bad=0, examples=[]))
                got = tuple(lab[:n])
                if got == exp:
                    c["ok"] += 1
                else:
                    c["bad"] += 1
                    if len(c["examples"]) < 8:
                        m = min(len(got), len(exp))
                        j = next((i for i in range(m) if got[i] != exp[i]), m)
                        c["examples"].append(dict(t=t, index=j, got=got[j] if j < len(got) else None,
                                                  expected=exp[j] if j < len(exp) else None, n_got=len(got),
                                                  n_expected=len(exp)))
        self.steps[t] = row

    def chain_of(self, kind):
        return self.chain.get(kind, dict(ok=0, bad=0, examples=[]))

    def per_layer_mean(self):
        n = max(self.n_inner.get("main", 0), 1)
        return {k: [round(x / n, 4) for x in v] for k, v in sorted(self.layer_sum.items())}


# ------------------------------------------------------------------ the wrappers and probes (arm T only)
class _Proxy:
    """An extension module with some attributes replaced; every other attribute reads the original module."""

    def __init__(self, base, **over):
        object.__setattr__(self, "_base", base)
        for k, v in over.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)


def _entry(rec, fn, label):
    def w(*a, **k):
        rec.mark(label)
        return fn(*a, **k)
    w.__wrapped__ = fn
    return w


def _around(rec, fn, entry, leave):
    def w(*a, **k):
        rec.mark(entry)
        r = fn(*a, **k)
        rec.mark(leave)
        return r
    w.__wrapped__ = fn
    return w


def _stage1(rec, fn, probe):
    def w(*a, **k):
        rec.mark("stage1")
        r = fn(*a, **k)
        if probe is not None and probe.active and rec.inner and not rec.capturing:
            probe.run(a, k)
        rec.mark("pool_topk")
        return r
    w.__wrapped__ = fn
    return w


def _vfy_on(rec, vfy):
    return vfy is not None and vfy.active and rec.inner and not rec.capturing


def _layer(rec, fn, last, vfy=None):
    def w(self, *a, **k):
        rec.layer(self.layer_idx)
        rec.mark("qkv_gemm")
        r = fn(self, *a, **k)
        if _vfy_on(rec, vfy):
            vfy.gemm(self, a, self.layer_idx == last)
        rec.mark("final" if self.layer_idx == last else "inter")
        return r
    w.__wrapped__ = fn
    return w


def _probe_entry(rec, fn, label, vfy, method):
    """_entry, and in a verify-probe step the probe's method first (it marks its own label)."""
    def w(*a, **k):
        if _vfy_on(rec, vfy):
            getattr(vfy, method)(a, k)
        rec.mark(label)
        return fn(*a, **k)
    w.__wrapped__ = fn
    return w


class Patcher:
    """The stage wrappers of arm T. install(cache) puts them in place; restore() puts back the ORIGINAL objects (by
    identity) and removes the cache's instance attribute. nl / cen = the modules nosi.nosa_llama / nosi.cache_engine
    (the CPU tests pass fakes with the same names). The shipped code reads every one of these names at call time
    (module globals, a class attribute, an instance attribute), so a wrapper sees every call; T-CHAIN proves it.
    probe: the SelectionProbe (stage-1 wrapper); vfy: the VerifyProbe (decode_forward exit, decode_update_kv entry,
    flash_h2d_from_mask entry)."""

    def __init__(self, rec, nl, cen, n_layers, probe=None, vfy=None):
        self.rec, self.nl, self.cen, self.n_layers, self.probe = rec, nl, cen, int(n_layers), probe
        self.vfy = vfy
        self.saved = []
        self.cache = None
        self.installs = 0

    def _set(self, owner, name, new):
        own = vars(owner)
        had = name in own
        self.saved.append((owner, name, own[name] if had else None, had))
        setattr(owner, name, new)

    def install(self, cache):
        if self.saved:
            raise RuntimeError("Patcher.install: the wrappers are already in place")
        rec, nl, cen = self.rec, self.nl, self.cen
        self.cache = cache
        self._set(nl.LlamaLayer, "decode_forward",
                  _layer(rec, vars(nl.LlamaLayer)["decode_forward"], self.n_layers - 1, self.vfy))
        self._set(nl, "nosa_linear", _entry(rec, nl.nosa_linear, "qkv_prep"))
        self._set(nl, "infllmv2_attn_stage1_fast", _stage1(rec, nl.infllmv2_attn_stage1_fast, self.probe))
        self._set(nl, "flash_attn_nosa_with_kvcache",
                  _around(rec, nl.flash_attn_nosa_with_kvcache, "attention", "ffn_gemm"))
        self._set(cache, "decode_update_kv", _probe_entry(rec, cache.decode_update_kv, "tail_write", self.vfy, "union"))
        self._set(cen, "diff", _Proxy(cen.diff, diff_offload=_entry(rec, cen.diff.diff_offload, "diff")))
        if getattr(cen, "_pool", None) is not None:
            self._set(cen, "_pool", _Proxy(cen._pool, pool_update=_entry(rec, cen._pool.pool_update, "pool_resolve")))
            self._set(cen, "_flash_pool_swap", _around(rec, cen._flash_pool_swap, "pool_swap", "map_commit"))
        self._set(cen, "flash_h2d_from_mask", _probe_entry(rec, cen.flash_h2d_from_mask, "fetch_k", self.vfy, "count"))
        self._set(cen, "flash_h2d_from_mask_bias", _around(rec, cen.flash_h2d_from_mask_bias, "fetch_v", "writeback"))
        self.installs += 1

    def restore(self):
        while self.saved:
            owner, name, orig, had = self.saved.pop()
            if had:
                setattr(owner, name, orig)
            else:
                delattr(owner, name)

    def suspend(self):
        self.restore()

    def resume(self):
        self.install(self.cache)


class SelectionProbe:
    """The verify's extra selection (P0_PILOT_AUTOPSY.md 7.4, U6), measured with existing ops only. run() is called by
    the stage-1 wrapper right after the shipped stage-1 call of a layer, in a selection step. It reads the shipped
    call's arguments (query, compressed keys, cu_seqlens, max lengths) and the layer's compressed cis; it writes only
    its own buffers, so the decode does not change (T-EQ). Per layer, under the labels SEL_LABELS:
      sel_prep     q repeated R times per request (repeat_interleave), cis repeated R times, cu_seqlens_q 0, R, 2R, ...
      sel1_stage1  stage 1, 1 row per request (the shipped call again)
      sel1_pool    nosa_pooling + topk(33) + scatter + masked_fill + topk(64), eager, 1 row per request
      selr_stage1  stage 1, R rows per request in ONE varlen call (max_seqlen_q R): the rows share the key reads
      selr_pool    the pooling and top-k of the R x B rows, eager
    A Python error disables the probe (T-SEL fails, the decode goes on)."""

    def __init__(self, rec, layers, pool_shape, stage1, pooling, torch_mod, rows=SEL_ROWS, qk_select=33):
        self.rec, self.layers, self.pool_shape = rec, layers, pool_shape
        self.stage1, self.pooling, self.torch = stage1, pooling, torch_mod
        self.rows, self.qk_select = int(rows), int(qk_select)
        self.active = False
        self.disabled = False
        self.errors = []
        self.calls = 0
        self.bufs = {}
        self.cu = {}

    def _cu(self, B, like):
        key = (int(B), str(like.device))
        c = self.cu.get(key)
        if c is None:
            c = self.cu[key] = self.torch.arange(0, int(B) * self.rows + 1, self.rows, dtype=like.dtype,
                                                 device=like.device)
        return c

    def _bufs(self, tag, H, rows, out_len, topk, dtype, dev):
        key = (tag, int(H), int(rows), int(out_len), int(topk))
        b = self.bufs.get(key)
        if b is None:
            self.bufs = {k: v for k, v in self.bufs.items() if k[0] != tag}
            T = self.torch
            b = dict(pq=T.empty((H, rows, out_len), dtype=dtype, device=dev),
                     pc=T.empty((H, rows, out_len), dtype=dtype, device=dev),
                     vq=T.empty((H, rows, self.qk_select), dtype=dtype, device=dev),
                     iq=T.empty((H, rows, self.qk_select), dtype=T.int64, device=dev),
                     mask=T.empty((H, rows, out_len), dtype=T.bool, device=dev),
                     v=T.empty((H, rows, topk), dtype=dtype, device=dev),
                     i=T.empty((H, rows, topk), dtype=T.int64, device=dev))
            self.bufs[key] = b
        return b

    def _select(self, layer, score, cis, max_k, rows, tag):
        out_len, dtype = self.pool_shape()
        b = self._bufs(tag, score.shape[0], rows, out_len, layer.topk_blocks, dtype, score.device)
        self.pooling(score, cis, max_k, b["pq"], b["pc"], stride=layer.pooling_stride, block_size=layer.block_size,
                     local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
        self.torch.topk(b["pq"], self.qk_select, dim=-1, sorted=False, out=(b["vq"], b["iq"]))
        b["mask"].zero_()
        b["mask"].scatter_(2, b["iq"], True)
        b["pc"].masked_fill_(b["mask"], float("inf"))
        self.torch.topk(b["pc"], layer.topk_blocks, dim=-1, sorted=False, out=(b["v"], b["i"]))
        return b

    def run(self, a, k):
        if self.disabled:
            return
        rec, R = self.rec, self.rows
        try:
            layer = self.layers[rec.layer_idx]
            q, kk, vv = a[0], a[1], a[2]
            cu_q, cu_k = k["cu_seqlens_q"], k["cu_seqlens_k"]
            mq, mk = k["max_seqlen_q"], k["max_seqlen_k"]
            B = q.shape[0]
            cis = layer.compressed_cis_buf
            rec.mark("sel_prep")
            qr = q.repeat_interleave(R, dim=0)
            cisr = cis.repeat_interleave(R, dim=1)
            cur = self._cu(B, cu_q)
            rec.mark("sel1_stage1")
            s1 = self.stage1(q, kk, vv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=mq, max_seqlen_k=mk,
                             causal=False)
            rec.mark("sel1_pool")
            self._select(layer, s1, cis, mk, B, "one")
            rec.mark("selr_stage1")
            sr = self.stage1(qr, kk, vv, cu_seqlens_q=cur, cu_seqlens_k=cu_k, max_seqlen_q=mq * R, max_seqlen_k=mk,
                             causal=False)
            rec.mark("selr_pool")
            self._select(layer, sr, cisr, mk, B * R, "rows")
            self.calls += 1
        except Exception as e:      # the probe is a measurement: an error disables it, T-SEL reports it
            self.disabled = True
            self.errors.append("layer %s: %s: %s" % (rec.layer_idx, type(e).__name__, str(e)[:300]))


class VerifyProbe:
    """The measured inputs of the verify model (VERIFY_FORMULA), with existing ops only. It reads the shipped tensors
    and writes only its own, so the decode does not change (T-EQ). Active in the verify steps (kind 'vfy'); `step` is
    the index of the step in the window (TimerHooks sets it). Device sums only: the host reads them after the arm.
      union       (decode_update_kv entry) copy the shipped selection topk_idx (H x B x topk block ids, a[4]) into a
                  ring of R per layer. When the ring holds R consecutive steps: sort the R x topk ids of each (head,
                  request) and count the distinct ids. That is the union of R consecutive selections; the rows of a
                  joint pass are R consecutive positions.
      count       (flash_h2d_from_mask entry) the host loads of the layer: load mask (a[2], H x B x topk) >= 0. The
                  pool lookup has already cleared the pool hits.
      gemm        (decode_forward exit) the layer input (a[0], B x 1 x hidden) repeated m times per request
                  (repeat_interleave, label gemm_prep), then for each m (label gemm_x<m>) the GEMM stack of the shipped
                  step on m x B rows: input rmsnorm + wqkv; wo + residual; rmsnorm + gate_up + SiLU-and-mul + down +
                  residual (nosa_llama.py:553-556, 647-660); the last layer adds the final rmsnorm + lm_head + float
                  (:839-843). So the probe's m = 1 repeats the group gemm_mlp (qkv_gemm + ffn_gemm + final) eagerly.
                  The input of wo is the first wo.in_features columns of the probe's own wqkv output (timing only).
    A Python error disables the probe (T-VFY fails; the decode goes on)."""

    def __init__(self, rec, n_steps, torch_mod, F, norm, silu_and_mul, final=None, mults=GEMM_MULTS, rows=SEL_ROWS):
        self.rec, self.n_steps, self.torch = rec, int(n_steps), torch_mod
        self.F, self.norm, self.silu, self.final = F, norm, silu_and_mul, final
        self.mults, self.rows = tuple(int(m) for m in mults), int(rows)
        self.active, self.disabled, self.step = False, False, None
        self.errors = []
        self.calls = dict(union=0, count=0, gemm=0)
        self.ring, self.filled = {}, {}
        self.acc = None                              # device int64 (3, n_steps): union sum, union max, loads
        self.union_n = [0] * self.n_steps            # (head, request) samples of the union, per step
        self.streams = [0] * self.n_steps            # (head, request) streams of the load count, per step

    def _acc(self, like):
        if self.acc is None:
            self.acc = self.torch.zeros((3, self.n_steps), dtype=self.torch.int64, device=like.device)
        return self.acc

    def _fail(self, what, e):
        self.disabled = True
        self.errors.append("%s layer %s: %s: %s" % (what, self.rec.layer_idx, type(e).__name__, str(e)[:300]))

    def union(self, a, k):
        if self.disabled:
            return
        T = self.torch
        try:
            self.rec.mark("union")
            sel = a[4] if len(a) > 4 else k["topk_idx"]
            i, li = int(self.step), self.rec.layer_idx
            ring = self.ring.get(li)
            if ring is None:
                ring = self.ring[li] = [T.empty_like(sel) for _ in range(self.rows)]
                self.filled[li] = 0
            ring[self.filled[li] % self.rows].copy_(sel)
            self.filled[li] += 1
            if self.filled[li] >= self.rows:
                s = T.sort(T.cat(ring, dim=-1), dim=-1).values
                u = (s[..., 1:] != s[..., :-1]).sum(dim=-1) + 1
                acc = self._acc(u)
                acc[0, i].add_(u.sum())
                acc[1, i].copy_(T.maximum(acc[1, i], u.max()))
                self.union_n[i] += int(u.numel())
            self.calls["union"] += 1
        except Exception as e:      # the probe is a measurement: an error disables it, T-VFY reports it
            self._fail("union", e)

    def count(self, a, k):
        if self.disabled:
            return
        try:
            self.rec.mark("load_count")
            mask = a[2] if len(a) > 2 else k["load_ids"]
            i = int(self.step)
            self._acc(mask)[2, i].add_((mask >= 0).sum())
            self.streams[i] += int(mask.shape[0]) * int(mask.shape[1])
            self.calls["count"] += 1
        except Exception as e:
            self._fail("count", e)

    def gemm(self, layer, a, last):
        if self.disabled:
            return
        rec = self.rec
        try:
            rec.mark("gemm_prep")
            x = a[0]
            xs = [x if m == 1 else x.repeat_interleave(m, dim=0) for m in self.mults]
            for m, xm in zip(self.mults, xs):
                rec.mark("gemm_x%d" % m)
                self._stack(layer, xm, last)
            self.calls["gemm"] += 1
        except Exception as e:
            self._fail("gemm", e)

    def _stack(self, layer, x, last):
        F, norm = self.F, self.norm
        h = norm(x, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = F.linear(h, layer.wqkv)
        h = x + F.linear(qkv[..., :layer.wo.shape[1]], layer.wo)
        g = F.linear(norm(h, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight),
                     layer.gate_up_proj)
        out = self.torch.empty(g.shape[:-1] + (g.shape[-1] // 2,), dtype=g.dtype, device=g.device)
        self.silu(g, out)
        h = h + F.linear(out, layer.down_proj)
        if last and self.final is not None:
            w, eps, head = self.final
            h = F.linear(norm(h, eps, w), head).float()
        return h

    def take(self):
        """The device sums as host lists (after the arm's last sync): [union sums, union maxima, loads]."""
        if self.acc is None:
            return [[0] * self.n_steps for _ in range(3)]
        return [[int(x) for x in row] for row in self.acc.tolist()]


def snapshot_targets(nl, cen, cuda):
    """Every object a patch of this cell could replace, for the identity gates H-UNPATCHED and T-RESTORED."""
    return {"NL.LlamaLayer.decode_forward": vars(nl.LlamaLayer).get("decode_forward"),
            "NL.nosa_linear": nl.nosa_linear,
            "NL.infllmv2_attn_stage1_fast": nl.infllmv2_attn_stage1_fast,
            "NL.flash_attn_nosa_with_kvcache": nl.flash_attn_nosa_with_kvcache,
            "CEN.InfLLMv2Cache.decode_update_kv": vars(cen.InfLLMv2Cache).get("decode_update_kv"),
            "CEN.diff": cen.diff,
            "CEN._pool": getattr(cen, "_pool", None),
            "CEN._flash_pool_swap": getattr(cen, "_flash_pool_swap", None),
            "CEN.flash_h2d_from_mask": cen.flash_h2d_from_mask,
            "CEN.flash_h2d_from_mask_bias": cen.flash_h2d_from_mask_bias,
            "torch.cuda.graph": cuda.graph,
            "gc.callbacks": tuple(gc.callbacks)}


def moved_targets(a, b):
    """The names whose object is not the same (identity) in the two snapshots."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if k == "gc.callbacks":
            same = len(x or ()) == len(y or ()) and all(p is q for p, q in zip(x or (), y or ()))
        else:
            same = x is y
        if not same:
            out.append(k)
    return out


class CaptureTimer:
    """The host time of each torch.cuda.graph enter (torch 2.6: synchronize, gc.collect, empty_cache, capture_begin)
    and exit (capture_end). A SUBCLASS of torch.cuda.graph takes its place for arm T only. rec.capturing is True inside
    the context, so no stage boundary goes into a graph."""

    def __init__(self, rec=None, now=time.perf_counter):
        self.rec, self.now = rec, now
        self.enter_ms, self.exit_ms = [], []
        self.owner, self.orig = None, None

    def make(self, base):
        timer = self

        class TimedGraph(base):
            def __enter__(self):
                t0 = timer.now()
                if timer.rec is not None:
                    timer.rec.capturing = True
                try:
                    r = super().__enter__()
                except BaseException:
                    if timer.rec is not None:
                        timer.rec.capturing = False
                    raise
                timer.enter_ms.append((timer.now() - t0) * 1e3)
                return r

            def __exit__(self, *exc):
                t0 = timer.now()
                try:
                    return super().__exit__(*exc)
                finally:
                    if timer.rec is not None:
                        timer.rec.capturing = False
                    timer.exit_ms.append((timer.now() - t0) * 1e3)
        return TimedGraph

    def install(self, owner):
        if self.owner is not None:
            raise RuntimeError("CaptureTimer.install: already in place")
        self.owner, self.orig = owner, owner.graph
        owner.graph = self.make(self.orig)

    def restore(self):
        if self.owner is not None:
            self.owner.graph = self.orig
            self.owner, self.orig = None, None

    def take(self):
        e, x = self.enter_ms, self.exit_ms
        self.enter_ms, self.exit_ms = [], []
        return dict(n_captures=len(e), graph_enter_ms=round(sum(e), 3),
                    graph_enter_first_ms=(round(e[0], 3) if e else None), graph_exit_ms=round(sum(x), 3))


class GcTimer:
    """A gc.callbacks entry: the time and the count per generation of the garbage collections."""

    def __init__(self, now=time.perf_counter):
        self.now = now
        self.t0 = None
        self.ms = 0.0
        self.count = [0, 0, 0]

    def __call__(self, phase, info):
        if phase == "start":
            self.t0 = self.now()
        elif phase == "stop" and self.t0 is not None:
            self.ms += (self.now() - self.t0) * 1e3
            g = int(info.get("generation", 2))
            self.count[min(max(g, 0), 2)] += 1
            self.t0 = None

    def take(self):
        r = (round(self.ms, 3), list(self.count))
        self.ms, self.count = 0.0, [0, 0, 0]
        return r


class TimerHooks:
    """Arm T's hooks in decode_loop. Steps p0 .. p1 - 1 = the profiler window (the wrappers removed, the profiler on);
    steps p0 - 1 .. p1 are not timed per stage (`edge`). Steps s0 .. s1 - 1 = the selection window (kind 'sel', the
    probe on). Steps v0 .. v1 - 1 = the verify window (kind 'vfy', the VerifyProbe on, its `step` = t - v0)."""

    def __init__(self, rec, patcher, cap, gct, window, clock, profiler_factory, cuda, steps, now=time.perf_counter,
                 gc_mod=gc, sel_window=(0, 0), probe=None, vfy_window=(0, 0), vfy=None):
        self.rec, self.patcher, self.cap, self.gct = rec, patcher, cap, gct
        self.p0, self.p1 = window
        self.s0, self.s1 = sel_window
        self.v0, self.v1 = vfy_window
        self.probe, self.vfy = probe, vfy
        self.clock, self.profiler_factory, self.cuda, self.now, self.gc = clock, profiler_factory, cuda, now, gc_mod
        self.prof, self.prof_running = None, False
        self.host_ms = [None] * steps
        self.gc_ms = [None] * steps
        self.recap = []
        self.h0 = None
        self.cache = None
        self.instance_left = None

    def start(self, cache):
        self.cache = cache
        self.cap.install(self.cuda)
        self.gc.callbacks.append(self.gct)
        self.patcher.install(cache)

    def before(self, t, need, why):
        win = self.p0 <= t < self.p1
        edge = self.p0 - 1 <= t <= self.p1
        if t == self.p0:
            self.patcher.suspend()
            self.clock.sync()
            self.prof = self.profiler_factory()
            self.prof.start()
            self.prof_running = True
        inner = not need and not edge
        sel = inner and self.s0 <= t < self.s1
        vf = inner and self.v0 <= t < self.v1
        label = "rewarm" if need else ("window" if win else ("edge" if edge else "embed"))
        self.rec.step_begin(t, inner, label, "sel" if sel else ("vfy" if vf else "main"))
        if self.probe is not None:
            self.probe.active = sel
        if self.vfy is not None:
            self.vfy.active = vf
            self.vfy.step = (t - self.v0) if vf else None
        self.cap.take()
        self.gct.take()
        self.h0 = self.now()

    def after_decode(self, t):
        self.host_ms[t] = round((self.now() - self.h0) * 1e3, 3)
        self.rec.mark("sample")

    def after(self, t, need, why):
        if self.probe is not None:
            self.probe.active = False
        if self.vfy is not None:
            self.vfy.active = False
        g = self.gct.take()
        self.gc_ms[t] = g[0]
        if need:
            self.recap.append(dict(t=t, why=why, host_ms=self.host_ms[t], gc_ms=g[0], gc_count=g[1], **self.cap.take()))
        if t == self.p1 - 1 and self.prof_running:
            self.clock.sync()
            self.prof.stop()
            self.prof_running = False
            self.patcher.resume()

    def end(self, steps):
        self.rec.finish()

    def stop(self):
        if self.probe is not None:
            self.probe.active = False
        if self.vfy is not None:
            self.vfy.active = False
        if self.prof_running:
            try:
                self.prof.stop()
            except Exception:       # the arm already failed; restore the patches below in every case
                traceback.print_exc()
            self.prof_running = False
        self.patcher.restore()
        self.cap.restore()
        if self.gct in self.gc.callbacks:
            self.gc.callbacks.remove(self.gct)
        self.instance_left = self.cache is not None and "decode_update_kv" in getattr(self.cache, "__dict__", {})


class NsysHooks:
    """Arm N's hooks: cudaProfilerStart / Stop around steps w0 .. w1 - 1 and one NVTX range per window step."""

    def __init__(self, rec, window, clock, cuda_profiler, nvtx):
        self.rec, self.clock, self.prof, self.nvtx = rec, clock, cuda_profiler, nvtx
        self.w0, self.w1 = window
        self.running = False

    def start(self, cache):
        pass

    def before(self, t, need, why):
        if t == self.w0:
            self.clock.sync()
            self.prof.start()
            self.running = True
        win = self.w0 <= t < self.w1
        self.rec.step_begin(t, False, "rewarm" if need else ("window" if win else "embed"))
        if win:
            self.nvtx.range_push("p0.step%d" % t)

    def after_decode(self, t):
        pass

    def after(self, t, need, why):
        if self.w0 <= t < self.w1:
            self.nvtx.range_pop()
        if t == self.w1 - 1 and self.running:
            self.clock.sync()
            self.prof.stop()
            self.running = False

    def end(self, steps):
        self.rec.finish()

    def stop(self):
        if self.running:
            self.prof.stop()
            self.running = False


def drive(steps, rewarm_check, decode, sample, hooks):
    """The step loop of r2_base_cell.run_shipped with the hook calls (pure; CPU-tested). rewarm_check() -> (need, why)
    and arms the re-warm; decode() = model.decode_inference; sample(t) = argmax, the token copy, the position + 1."""
    rewarms = []
    for t in range(steps):
        need, why = rewarm_check()
        if need:
            rewarms.append((t, why))
        hooks.before(t, need, why)
        decode()
        hooks.after_decode(t)
        sample(t)
        hooks.after(t, need, why)
    hooks.end(steps)
    return rewarms


def decode_loop(model, ids, steps, clock, hooks):
    """r2_base_cell.run_shipped (r2_base_cell.py:393-433), the same device operations in the same order, with the hook
    calls of drive(). The hooks own the step events (run_shipped's period events are at hooks.before)."""
    import torch
    from nosi.cache_engine import InfLLMv2Cache
    cfg = model.config
    cache = InfLLMv2Cache(config=cfg, num_hidden_layers=cfg.num_hidden_layers, has_kv_bias=True)
    for layer in cache.layers:
        layer.cache_engine.max_gen_len = steps + 64
    Bn = ids.shape[0]
    clock.sync()
    t0 = time.time()
    logits, pos = model.batch_prefill(ids, cache)
    clock.sync()
    prefill_s = time.time() - t0
    tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits
    pos = pos[:, -1:] + 1
    cu = torch.arange(0, Bn + 1, dtype=torch.int, device="cuda")
    inputs = torch.empty((Bn, steps + 1), dtype=torch.int64, device="cuda")
    inputs[:, 0:1].copy_(tok)
    model.has_buffers = False
    st = dict(tok=tok, pos=pos, lg=None, warm=None)

    def rewarm_check():
        l0 = cache.layers[0]
        need, why = BC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen,
                                   st["warm"])
        if need:
            st["warm"] = BC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
            model.has_buffers = False
        return need, why

    def decode():
        st["lg"] = model.decode_inference(st["tok"], cu, st["pos"], cache)

    def sample(t):
        st["tok"] = st["lg"][:, -1, :].argmax(dim=-1, keepdim=True)
        inputs[:, t + 1:t + 2].copy_(st["tok"])
        st["pos"] = st["pos"] + 1

    try:                                                 # a partial start is undone by stop() as well
        hooks.start(cache)
        rewarms = drive(steps, rewarm_check, decode, sample, hooks)
    finally:
        hooks.stop()
    clock.sync()
    fr, tot = torch.cuda.mem_get_info()
    st.clear()
    del cache
    return dict(inputs=inputs, rewarms=rewarms, prefill_s=round(prefill_s, 1),
                device_used_end_GB=round((tot - fr) / 1e9, 3))


# ------------------------------------------------------------------ analysis (pure parts, CPU-tested)
def _stats(xs):
    return BC.period_stats(xs, list(range(len(xs))))


def stage_summary(steps, keep):
    """Per stage label and per GROUP: ms per step (median, mean, p10, p90) over the kept timed steps; the median
    per-step fraction of the period per group and per SAVES set; the closure (max |sum of stages - period|)."""
    rows = [steps[t] for t in keep if t in steps and steps[t].get("inner") and "stages" in steps[t]
            and steps[t]["period_ms"] > 0]
    if not rows:
        return dict(n=0)
    labels = sorted(set().union(*(r["stages"] for r in rows)))
    grouped = {x for _, ls in GROUPS for x in ls}
    per_label = {x: _stats([r["stages"].get(x, 0.0) for r in rows]) for x in labels}
    per_group, frac_group = {}, {}
    for g, ls in GROUPS:
        xs = [sum(r["stages"].get(x, 0.0) for x in ls) for r in rows]
        per_group[g] = _stats(xs)
        frac_group[g] = round(statistics.median([x / r["period_ms"] for x, r in zip(xs, rows)]), 5)
    save_frac = {}
    for name, ls in SAVES:
        save_frac[name] = round(statistics.median([sum(r["stages"].get(x, 0.0) for x in ls) / r["period_ms"]
                                                   for r in rows]), 5)
    closure = max(abs(sum(r["stages"].values()) - r["period_ms"]) for r in rows)
    return dict(n=len(rows), period=_stats([r["period_ms"] for r in rows]), labels=per_label, groups=per_group,
                group_frac=frac_group, save_frac=save_frac, closure_max_ms=round(closure, 6),
                ungrouped=sorted(set(labels) - grouped))


def selection_summary(steps, keep, rows):
    """The selection probe's steps: ms per step (32 layers) of the shipped selection (stage1 + pool_topk, graph), the
    eager 1-row selection, the eager R-row selection, their stage-1 / pooling parts and the prep copies; the extra
    selection of a round of K = R - 1 verify positions: joint (R rows - 1 row, per step) and serial (K x 1 row)."""
    rs = [steps[t] for t in keep if t in steps and steps[t].get("inner") and steps[t].get("kind") == "sel"
          and "stages" in steps[t]]
    if not rs:
        return dict(n=0, rows=int(rows), K=int(rows) - 1)

    def col(ls):
        return [sum(r["stages"].get(x, 0.0) for x in ls) for r in rs]
    k = int(rows) - 1
    one, many = col(("sel1_stage1", "sel1_pool")), col(("selr_stage1", "selr_pool"))
    out = dict(n=len(rs), rows=int(rows), K=k, shipped=_stats(col(("stage1", "pool_topk"))), one_row=_stats(one),
               r_rows=_stats(many), prep=_stats(col(("sel_prep",))), stage1_one=_stats(col(("sel1_stage1",))),
               stage1_rows=_stats(col(("selr_stage1",))), pool_one=_stats(col(("sel1_pool",))),
               pool_rows=_stats(col(("selr_pool",))), extra_joint=_stats([m - o for m, o in zip(many, one)]),
               period=_stats([r["period_ms"] for r in rs]))
    out["extra_joint_ms"] = out["extra_joint"]["median_ms"]
    out["extra_serial_ms"] = round(k * out["one_row"]["median_ms"], 3)
    out["extra_serial_shipped_ms"] = round(k * out["shipped"]["median_ms"], 3)
    out["per_position_joint_ms"] = round(out["extra_joint_ms"] / k, 3) if k else None
    o = out["one_row"]["median_ms"]
    out["rows_over_one"] = round(out["r_rows"]["median_ms"] / o, 4) if o else None
    return out


def pays(draft_ms, verify_extra_ms, accepted_per_round, baseline_ms, extra_sel_ms=0.0):
    """The ceiling condition: draft + (verify_extra + the K extra selections) / accepted < baseline, per token."""
    return draft_ms + (verify_extra_ms + extra_sel_ms) / accepted_per_round < baseline_ms


def ceiling(base_ms, B, save_frac, sel=None, accepted=ACCEPTED, ref_tok_s=None):
    """CEILING_FORMULA per SAVES set: the draft floor, the step speedup, the tok/s ceiling, save / floor (the plan's
    "fetch x of decode"), the verify budget per round of `a` accepted tokens before and after the K extra selections
    (joint, serial), the smallest `a` that leaves a positive budget, the tok/s ceiling / ref_tok_s (mb1)."""
    sel = sel or {}
    out = dict(formula=CEILING_FORMULA, baseline_ms=base_ms, batch=int(B), accepted=list(accepted),
               baseline_tok_s=(round(1e3 * B / base_ms, 1) if base_ms else None), ref_tok_s=ref_tok_s,
               K=sel.get("K"), extra_sel_joint_ms=sel.get("extra_joint_ms"),
               extra_sel_serial_ms=sel.get("extra_serial_ms"))
    for name, f in sorted(save_frac.items()):
        save = base_ms * f
        floor = base_ms - save
        if floor <= 0:
            out[name] = dict(save_frac=f, save_ms=round(save, 3), draft_floor_ms=None)
            continue
        tok = 1e3 * B / floor
        d = dict(save_frac=round(f, 5), save_ms=round(save, 3), draft_floor_ms=round(floor, 3),
                 max_speedup=round(base_ms / floor, 4), ceiling_tok_s=round(tok, 1),
                 save_over_rest=round(save / floor, 4),
                 verify_budget_ms={str(a): round(a * save, 3) for a in accepted},
                 ceiling_vs_ref=(round(tok / ref_tok_s, 4) if ref_tok_s else None))
        for tag in ("joint", "serial"):
            x = out["extra_sel_%s_ms" % tag]
            if x is not None:
                d["verify_budget_after_sel_%s_ms" % tag] = {str(a): round(a * save - x, 3) for a in accepted}
                d["min_accepted_%s" % tag] = next((a for a in accepted if a * save - x > 0), None)
        out[name] = d
    return out


def ls_line(xs, ys):
    """The least-squares line y = intercept + slope x: (slope, intercept); (None, None) for fewer than 2 distinct x."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def gemm_rows_summary(steps, keep, mults, B, shipped_ms=None, floor_ms=WEIGHT_FLOOR_MS):
    """The GEMM probe (kind 'vfy' steps): per multiplier m, the GEMM stack on m x B rows in ms per step (32 layers +
    the final rmsnorm and lm_head; median, p10, p90), ms per row, the excess over floor_ms (the weight-read floor,
    about 10.6 ms: SCORING_p0all.md 3); the marginal ms per row between neighbour points; the least-squares line over
    the points (its intercept estimates the floor); the probe's B rows / the shipped gemm_mlp median (shipped_ms).
    Only steps with every gemm_x<m> label count (a probe error stops the labels)."""
    labels = ["gemm_x%d" % int(m) for m in mults]
    rs = [steps[t] for t in keep if t in steps and steps[t].get("inner") and steps[t].get("kind") == "vfy"
          and all(x in steps[t].get("stages", {}) for x in labels)]
    out = dict(n=len(rs), B=int(B), mults=[int(m) for m in mults], floor_ms=floor_ms, shipped_gemm_mlp_ms=shipped_ms)
    if not rs:
        return out
    pts = []
    for m in mults:
        st = _stats([r["stages"].get("gemm_x%d" % int(m), 0.0) for r in rs])
        rows, med = int(m) * int(B), st["median_ms"]
        pts.append(dict(mult=int(m), rows=rows, median_ms=med, p10_ms=st["p10_ms"], p90_ms=st["p90_ms"],
                        ms_per_row=round(med / rows, 5), over_floor_ms=round(med - floor_ms, 3)))
    slope, icpt = ls_line([p["rows"] for p in pts], [p["median_ms"] for p in pts])
    one = next((p for p in pts if p["mult"] == 1), None)
    out.update(points=pts, prep=_stats([r["stages"].get("gemm_prep", 0.0) for r in rs]),
               marginal=[dict(rows=[p["rows"], q["rows"]],
                              ms_per_row=round((q["median_ms"] - p["median_ms"]) / (q["rows"] - p["rows"]), 5))
                         for p, q in zip(pts, pts[1:])],
               slope_ms_per_row=(round(slope, 5) if slope is not None else None),
               intercept_ms=(round(icpt, 3) if icpt is not None else None),
               x1_over_shipped=(round(one["median_ms"] / shipped_ms, 4) if one and shipped_ms else None))
    return out


def vfy_summary(steps, ts, acc, union_n, streams, topk, warm=1):
    """The verify window. ts: its steps in order; acc: [union sums, union maxima, loads] per window index
    (VerifyProbe.take); union_n / streams: the (head, request) samples per index; topk: the blocks of one selection.
    Union: mean and max over the indices with a full ring; extras_per_stream = mean - topk. Loads and fetch: the
    indices from `warm` on (the first is the warm-up): host loads per stream-step and per step, the fetch (fetch_k +
    fetch_v) ms per step and ms per load in the same steps. union_fetch_ms = extras x streams per step x ms per load
    (every extra a host load). Only steps with the full count of samples count (a probe error stops the counts)."""
    sums, maxes, loads = acc
    ufull, sfull = max(union_n, default=0), max(streams, default=0)
    ui = [i for i in range(len(ts)) if union_n[i] > 0 and union_n[i] == ufull]
    un = sum(union_n[i] for i in ui)
    mean = sum(sums[i] for i in ui) / un if un else None
    li = [i for i in range(int(warm), len(ts)) if streams[i] > 0 and streams[i] == sfull
          and "stages" in steps.get(ts[i], {})]
    ld, sn, n = sum(loads[i] for i in li), sum(streams[i] for i in li), len(li)
    fetch = sum(steps[ts[i]]["stages"].get("fetch_k", 0.0) + steps[ts[i]]["stages"].get("fetch_v", 0.0) for i in li)
    per_load = fetch / ld if ld else None
    extras = mean - int(topk) if mean is not None else None
    sps = sn / n if n else None
    uf = extras * sps * per_load if extras is not None and per_load is not None and sps else None
    return dict(topk=int(topk), union_steps=len(ui), union_samples=un,
                union_mean=(round(mean, 4) if mean is not None else None),
                union_max=(max(maxes[i] for i in ui) if ui else None),
                extras_per_stream=(round(extras, 4) if extras is not None else None),
                load_steps=n, loads=ld, streams_per_step=(round(sps, 1) if sps else None),
                loads_per_stream=(round(ld / sn, 5) if sn else None), loads_per_step=(round(ld / n, 1) if n else None),
                fetch_ms_per_step=(round(fetch / n, 3) if n else None),
                ms_per_load=(round(per_load, 7) if per_load is not None else None),
                union_fetch_ms=(round(uf, 3) if uf is not None else None),
                extras_over_loads=(round(extras * sn / ld, 3) if extras is not None and ld else None))


def verify_case(vc, fetch, draft, accept):
    """One row of verify_ceiling's cases (None if absent)."""
    return next((c for c in (vc or {}).get("cases", []) if c["fetch"] == fetch and c["draft"] == draft
                 and c["accept"] == accept), None)


def verify_ceiling(base_ms, B, gather_ms, gemm, sel, vfy, rows=SEL_ROWS, attention_ms=None, accept=VERIFY_ACCEPT,
                   recovery=DRAFT_RECOVERY, ref_tok_s=REF_CACHED_TOK_S, mb1_tok_s=None):
    """VERIFY_FORMULA. gemm: gemm_rows_summary (needs the points m = 1 and m = R); sel: selection_summary (extra_joint_ms,
    r_rows); vfy: vfy_summary (union_fetch_ms). joint_ms per fetch variant ('measured': + union_fetch_ms; 'hidden': 0);
    the lead's form of the measured joint pass (GEMM of R x B rows + selection of R rows + one attention + union fetch
    + the rest of the shipped step); per (fetch, draft, acceptance) case: draft_ms, round_ms, committed tok/s, / ref
    (cached NOSA), / own (1000 x B / base_ms), / mb1 (the same job), the break-even joint_ms and the margin."""
    R = int(rows)
    K = R - 1
    pts = {p["mult"]: p["median_ms"] for p in (gemm or {}).get("points", [])}
    out = dict(formula=VERIFY_FORMULA, baseline_ms=base_ms, batch=int(B), rows=R, K=K, ref_tok_s=ref_tok_s,
               mb1_tok_s=mb1_tok_s, own_tok_s=(round(1e3 * B / base_ms, 1) if base_ms else None),
               gather_ms=(round(gather_ms, 3) if gather_ms is not None else None), accept=list(accept),
               recovery=[list(r) for r in recovery])
    sx = (sel or {}).get("extra_joint_ms")
    miss = [name for name, v in (("baseline_ms", base_ms), ("gather_ms", gather_ms), ("gemm x1", pts.get(1)),
                                 ("gemm x%d" % R, pts.get(R)), ("selection extra", sx)) if v is None]
    if miss:
        out["missing"] = miss
        return out
    g1, gr = pts[1], pts[R]
    uf = (vfy or {}).get("union_fetch_ms")
    sr = _get(sel, "r_rows", "median_ms")
    joint = {"hidden": base_ms + (gr - g1) + sx}
    if uf is not None:
        joint["measured"] = joint["hidden"] + uf
    out.update(gemm_x1_ms=g1, gemm_xr_ms=gr, gemm_extra_ms=round(gr - g1, 3), sel_r_ms=sr, sel_extra_ms=sx,
               union_fetch_ms=uf, attention_ms=attention_ms, joint_ms={k: round(v, 3) for k, v in joint.items()},
               joint_steps={k: round(v / base_ms, 4) for k, v in joint.items()})
    if "measured" in joint and attention_ms is not None and sr is not None:
        out["components"] = dict(gemm_r_rows_ms=gr, selection_r_rows_ms=sr, one_attention_ms=attention_ms,
                                 union_fetch_ms=uf,
                                 rest_of_step_ms=round(joint["measured"] - gr - sr - attention_ms - uf, 3))
    cases = []
    for fetch in ("measured", "hidden"):
        if fetch not in joint:
            continue
        j = joint[fetch]
        for name, r in recovery:
            draft = base_ms - r * gather_ms
            rnd = K * draft + j
            for a in accept:
                e = sum(a ** i for i in range(R))
                tok = 1e3 * B * e / rnd
                be_own = e * base_ms - K * draft
                be_ref = 1e3 * B * e / ref_tok_s - K * draft
                cases.append(dict(fetch=fetch, draft=name, recovery=r, accept=a, tokens_per_request=round(e, 5),
                                  draft_ms=round(draft, 3), joint_ms=round(j, 3), round_ms=round(rnd, 3),
                                  tok_s=round(tok, 1), vs_cached=round(tok / ref_tok_s, 4),
                                  vs_own=round(e * base_ms / rnd, 4),
                                  vs_mb1=(round(tok / mb1_tok_s, 4) if mb1_tok_s else None),
                                  breakeven_joint_own_ms=round(be_own, 3), margin_own_ms=round(be_own - j, 3),
                                  breakeven_joint_cached_ms=round(be_ref, 3), margin_cached_ms=round(be_ref - j, 3)))
    out["cases"] = cases
    return out


LAUNCH_KINDS = (("kernel", ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchCooperativeKernel")),
                ("graph", ("cudaGraphLaunch", "cuGraphLaunch")),
                ("memcpy", ("cudaMemcpy", "cuMemcpy")),
                ("memset", ("cudaMemset", "cuMemset")),
                ("event", ("cudaEventRecord", "cuEventRecord")),
                ("sync", ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
                          "cuStreamSynchronize", "cuCtxSynchronize")))
ENQUEUE_KINDS = ("kernel", "graph", "memcpy", "memset")


def launch_kind(name: str) -> str:
    for kind, prefixes in LAUNCH_KINDS:
        if name.startswith(prefixes):
            return kind
    return "other"


def launch_counts(trace, steps: int):
    """The CUDA runtime / driver calls of a Kineto trace per step: by name, by kind, and the enqueued launches
    (kernel + graph + memcpy + memset)."""
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    names = {}
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in P3.LAUNCH_CATS:
            n = str(e.get("name", ""))
            names[n] = names.get(n, 0) + 1
    kinds = {}
    for n, c in names.items():
        k = launch_kind(n)
        kinds[k] = kinds.get(k, 0) + c

    def per(c):
        return round(c / steps, 3)
    return dict(steps=steps, launches_per_step=per(sum(kinds.get(k, 0) for k in ENQUEUE_KINDS)),
                kinds_per_step={k: per(v) for k, v in sorted(kinds.items())},
                names_per_step={n: per(c) for n, c in sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))})


def busy_union_ms(ops, steps: int) -> float:
    """Device busy ms per step: the union of the GPU op intervals (ts, ts + dur, us)."""
    iv = sorted((o["ts"], o["ts"] + o["dur_us"]) for o in ops if o["dur_us"] > 0)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return round(tot / 1e3 / steps, 4)


def pack_trace(path, steps):
    """The trace's GPU ops (r2_p3_profile.parse_trace, no ranges) and launch counts; then the trace gzip-compressed
    beside it (the .json removed)."""
    with open(path) as f:
        trace = json.load(f)
    ops = P3.parse_trace(trace, prefixes=())
    lc = launch_counts(trace, steps)
    del trace
    with open(path, "rb") as fi, gzip.open(path + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(path)
    return ops, lc


def _median_of(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def recapture_summary(recap, steps, head_rewarm_ms=None):
    """One row per re-warm step of arm T: its period (CUDA events), host time, graph enter / exit host time (32
    captures), gc time, the rest of the host time; the matching arm H period; the median over the later re-warms."""
    out = []
    for i, r in enumerate(recap):
        per = steps.get(r["t"], {}).get("period_ms")
        other = None
        if r.get("host_ms") is not None:
            other = round(r["host_ms"] - r["graph_enter_ms"] - r["graph_exit_ms"], 3)
        out.append(dict(r, period_ms=(round(per, 3) if per is not None else None), host_other_ms=other,
                        headline_period_ms=(head_rewarm_ms[i] if head_rewarm_ms and i < len(head_rewarm_ms) else None),
                        gc_share=(round(r["gc_ms"] / per, 4) if per else None)))
    later = out[1:]
    keys = ("period_ms", "host_ms", "graph_enter_ms", "graph_exit_ms", "gc_ms", "host_other_ms", "gc_share",
            "headline_period_ms")
    return dict(n=len(out), each=out, later_median={k: _median_of([x.get(k) for x in later]) for k in keys})


def load_workload(path, S):
    """(documents (n, S) int64, sha256 of the token file, n). S 16128: r2_ragged_gate.load_docs (c16k_manifest.json,
    c16k_tokens.bin, 94 documents). Another S: the make_workload.py layout ctx<S>_manifest.json / ctx<S>_tokens.bin
    (int32 little-endian), SHA256SUMS checked the same way. SystemExit("REFUSED ...") on a mismatch."""
    if int(S) == BC.S0:
        docs, sha = RG.load_docs(path, BC.N_DOCS, BC.S0)
        return docs, sha, BC.N_DOCS
    import numpy as np
    import torch
    man = json.load(open(os.path.join(path, "ctx%d_manifest.json" % S)))
    n, ctx = man["shape"]
    if int(ctx) != int(S) or man.get("dtype", "int32-le") != "int32-le":
        raise SystemExit("REFUSED: workload %s has shape %s dtype %s, not ctx %d int32-le" % (
            path, man["shape"], man.get("dtype"), S))
    sums = {}
    for line in open(os.path.join(path, "SHA256SUMS")):
        h, fname = line.split()
        sums[fname] = h
    name = "ctx%d_tokens.bin" % S
    raw = open(os.path.join(path, name), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sums.get(name) != sha:
        raise SystemExit("REFUSED: %s sha256 %s is not SHA256SUMS's" % (name, sha))
    return torch.from_numpy(np.frombuffer(raw, dtype="<i4").reshape(n, ctx).astype("int64")), sha, int(n)


def load_baseline(path):
    """The same job's mb1 summary (r2_base_cell): SH steady median / mean, effective / steady tok/s, C, B."""
    try:
        with open(path) as f:
            s = json.load(f)
        sh, meta = s["SH"], s["meta"]
        return dict(path=path, C=int(meta["capacity_C"]), B=int(meta["batch"]), S=int(meta.get("S0", BC.S0)),
                    median_ms=float(sh["steady"]["median_ms"]), mean_ms=float(sh["steady"]["mean_ms"]),
                    effective_tok_s=float(sh["rates"]["effective_tok_s"]),
                    steady_tok_s=float(sh["rates"]["steady_tok_s"]), nosi_commit=meta.get("nosi_commit"))
    except Exception as e:      # REPORT: a missing baseline never stops the cell
        return dict(path=path, error="%s: %s" % (type(e).__name__, str(e)[:200]))


def baseline_ratios(C, B, S, h, base):
    """variant / baseline (Task 0.4): step time (median, mean) and tok/s (effective, steady)."""
    if not base or "error" in base or not h or not h["steady"]["median_ms"]:
        return None
    st, rt = h["steady"], h["rates"]
    return dict(baseline={k: base[k] for k in ("C", "B", "S", "median_ms", "mean_ms", "effective_tok_s",
                                               "steady_tok_s")},
                same_cell=(int(C), int(B), int(S)) == (base["C"], base["B"], base["S"]),
                median_ratio=round(st["median_ms"] / base["median_ms"], 4),
                mean_ratio=round(st["mean_ms"] / base["mean_ms"], 4),
                effective_ratio=round(rt["effective_tok_s"] / base["effective_tok_s"], 4),
                steady_tok_ratio=round(rt["steady_tok_s"] / base["steady_tok_s"], 4))


def _verdict(value, band, registered=True):
    if value is None:
        return "NOT_RUN"
    if not registered or band is None or band == "TBD":
        return "UNREGISTERED"
    lo, hi = band
    return "CONFIRMED" if lo <= value <= hi else "REFUTED"


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def verdicts_p0(C, B, res, pred=None, utc=None, S=None):
    """The REGISTERED block against this cell: [(name, verdict, detail)]. TBD -> UNREGISTERED."""
    pred = PRED_P0 if pred is None else pred
    utc = REGISTERED_UTC if utc is None else utc
    reg = utc != "TBD"
    key = cell_key(C, B, S)

    def per_cell(name):
        v = pred.get(name)
        return v.get(key, "TBD") if isinstance(v, dict) else v
    out = []
    r = res.get("baseline_ratios")
    if r and r["same_cell"]:
        out.append(("P0-HEAD", _verdict(r["median_ratio"], pred["head_band"]),
                    "H steady median / same-job mb1 %.4f (band %s, the plan's +-2 %% rule)" % (
                        r["median_ratio"], pred["head_band"])))
    else:
        out.append(("P0-HEAD", "REPORT" if r else "NOT_RUN",
                    "H / mb1 (another cell): %s" % ({k: r[k] for k in ("median_ratio", "mean_ratio", "effective_ratio",
                                                                       "steady_tok_ratio")} if r else "no baseline")))
    v = res.get("overhead_ratio")
    out.append(("P0-OVH", _verdict(v, pred["ovh_band"], reg), "T steady / H steady %s (band %s)" % (v, pred["ovh_band"])))
    bands = per_cell("stage_ms")
    for g, _ in GROUPS:
        v = _get(res, "stages", "groups", g, "median_ms")
        band = bands.get(g, "TBD") if isinstance(bands, dict) else bands
        out.append(("P0-STAGE-%s" % g, _verdict(v, band, reg), "%s ms per step (T median) %s (band %s)" % (g, v, band)))
    v = _get(res, "launches", "launches_per_step")
    out.append(("P0-LAUNCH", _verdict(v, per_cell("launches"), reg),
                "enqueued launches per step %s (band %s)" % (v, per_cell("launches"))))
    v = res.get("gap_ms")
    out.append(("P0-GAP", _verdict(v, per_cell("gap_ms"), reg),
                "H steady median - device busy %s ms (band %s)" % (v, per_cell("gap_ms"))))
    v = _get(res, "recapture", "later_median", "period_ms")
    out.append(("P0-RECAP", _verdict(v, pred["recap_ms"], reg),
                "later re-capture step %s ms (band %s)" % (v, pred["recap_ms"])))
    v = _get(res, "recapture", "later_median", "gc_share")
    out.append(("P0-RECAP-GC", _verdict(v, pred["recap_gc_share"], reg),
                "gc share of a later re-capture step %s (band %s)" % (v, pred["recap_gc_share"])))
    v = _get(res, "selection", "extra_joint_ms")
    out.append(("P0-SEL", _verdict(v, per_cell("sel_extra_ms"), reg),
                "extra selection of a round, joint (R rows - 1 row) %s ms per step, K %s (band %s)" % (
                    v, _get(res, "selection", "K"), per_cell("sel_extra_ms"))))
    v = _get(res, "ceiling", "gather_only", "max_speedup")
    out.append(("P0-CEIL", _verdict(v, per_cell("ceil_speedup"), reg),
                "max draft speedup, gather_only %s (band %s)" % (v, per_cell("ceil_speedup"))))
    return out


def verdicts_sweep(C, B, res, pred=None, utc=None, S=None):
    """PRED_SWEEP against this cell: [(name, verdict, detail)]. TBD -> UNREGISTERED; no value -> NOT_RUN. The ceiling
    lines read the verify_ceiling case (fetch 'measured', acceptance VERIFY_ACCEPT[0]) with the draft floor and with
    dr1's share of the gather for this cell's pool (dr1_for_pool(C - 63))."""
    pred = PRED_SWEEP if pred is None else pred
    utc = REGISTERED_SWEEP_UTC if utc is None else utc
    reg = utc != "TBD"
    key = cell_key(C, B, S)

    def band(name):
        v = pred.get(name, "TBD")
        return v.get(key, "TBD") if isinstance(v, dict) else v
    out = []
    sb = band("stage_ms")
    for g, _ in GROUPS:
        v = _get(res, "stages", "groups", g, "median_ms")
        b = sb.get(g, "TBD") if isinstance(sb, dict) else sb
        out.append(("SW-STAGE-%s" % g, _verdict(v, b, reg), "%s ms per step (T median) %s (band %s)" % (g, v, b)))
    vc = res.get("verify_ceiling") or {}
    gr = res.get("gemm_rows") or {}
    R = int(vc.get("rows", SEL_ROWS))
    pts = {p["mult"]: p["median_ms"] for p in gr.get("points", [])}
    a0 = VERIFY_ACCEPT[0]
    dr1 = dr1_for_pool(int(C) - BC.C_BASE)
    fl, d1 = verify_case(vc, "measured", "floor", a0), verify_case(vc, "measured", dr1, a0)
    rows = (("SW-GEMM-X1", "gemm_x1_ms", pts.get(1), "GEMM stack on B rows, ms per step"),
            ("SW-GEMM-XR", "gemm_xr_ms", pts.get(R), "GEMM stack on %d x B rows, ms per step" % R),
            ("SW-GEMM-SLOPE", "gemm_slope", gr.get("slope_ms_per_row"), "GEMM least-squares ms per row"),
            ("SW-SEL", "sel_extra_ms", _get(res, "selection", "extra_joint_ms"),
             "extra selection of a round, joint, ms per step"),
            ("SW-UNION", "union_extras", _get(res, "verify_probe", "extras_per_stream"),
             "union extras per (head, request)"),
            ("SW-LOADS", "loads_per_stream", _get(res, "verify_probe", "loads_per_stream"),
             "host loads per (layer, head, request) per step"),
            ("SW-UFETCH", "union_fetch_ms", _get(res, "verify_probe", "union_fetch_ms"),
             "union-extras fetch, ms per joint pass"),
            ("SW-JOINT", "joint_steps", _get(vc, "joint_steps", "measured"), "joint pass (measured fetch) / H median"),
            ("SW-CACHED-FLOOR", "vs_cached_floor", (fl or {}).get("vs_cached"),
             "committed tok/s / %s at a %s, draft floor, measured fetch" % (REF_CACHED_TOK_S, a0)),
            ("SW-CACHED-DR1", "vs_cached_dr1", (d1 or {}).get("vs_cached"),
             "committed tok/s / %s at a %s, draft %s, measured fetch" % (REF_CACHED_TOK_S, a0, dr1)),
            ("SW-OWN-FLOOR", "vs_own_floor", (fl or {}).get("vs_own"),
             "committed tok/s / own H tok/s at a %s, draft floor, measured fetch" % a0))
    for name, k, v, what in rows:
        b = band(k)
        out.append((name, _verdict(v, b, reg), "%s %s (band %s)" % (what, v, b)))
    return out


# ------------------------------------------------------------------ the arms (GPU)
def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--context", type=int, default=BC.S0, choices=CONTEXTS, help="the prompt length S")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pool", type=int, default=0, help="the victim pool P (C = 63 + P); must equal NOSI_POOL_BLOCKS")
    ap.add_argument("--mbase", action="store_true", help="r2_base_cell.verdicts_mbase on arm H (S 16128 only)")
    ap.add_argument("--stage-timers", action="store_true",
                    help="OPT-IN: arm T after arm H (stage timers, the profiler window, the selection probe, the "
                         "re-capture timers)")
    ap.add_argument("--prof-start", type=int, default=PROF_START)
    ap.add_argument("--prof-steps", type=int, default=PROF_STEPS)
    ap.add_argument("--sel-start", type=int, default=SEL_START)
    ap.add_argument("--sel-steps", type=int, default=SEL_STEPS, help="0: no selection probe")
    ap.add_argument("--sel-rows", type=int, default=SEL_ROWS, help="R rows per request (1 draft + R - 1 verify)")
    ap.add_argument("--vfy-start", type=int, default=VFY_START)
    ap.add_argument("--vfy-steps", type=int, default=VFY_STEPS, help="0: no verify probe")
    ap.add_argument("--gemm-mults", default=",".join(str(m) for m in GEMM_MULTS),
                    help="the GEMM probe's rows per request (comma list, increasing, with 1 and --sel-rows)")
    ap.add_argument("--ring", type=int, default=4)
    ap.add_argument("--baseline", default="", help="the same job's mb1 summary.json (Task 0.4)")
    ap.add_argument("--nsys-window", type=int, default=None,
                    help="arm N only: cudaProfilerStart / Stop around the steps W .. W + %d" % (NSYS_STEPS - 1))
    a = ap.parse_args(argv)
    if not 0 <= a.pool <= BC.POOL_MAX:
        ap.error("--pool in 0 .. %d" % BC.POOL_MAX)
    if a.batch < 1 or not 1 <= a.steps <= 1000:
        ap.error("--batch >= 1, --steps in 1 .. 1000 (the shipped total_cis holds S + 1024)")
    if a.ring < 2:
        ap.error("--ring >= 2")
    if a.mbase and a.context != BC.S0:
        ap.error("--mbase: the mbase bands and the peak model are for S %d" % BC.S0)
    rw = {t for t, _ in BC.expected_rewarms(a.context, a.steps)}
    if a.nsys_window is not None:
        if a.stage_timers:
            ap.error("--nsys-window replaces the arms H and T: no --stage-timers")
        w0, w1 = a.nsys_window, a.nsys_window + NSYS_STEPS
        if w0 < 2 or w1 > a.steps or rw & set(range(w0 - 1, w1 + 1)):
            ap.error("--nsys-window: steps %d .. %d and their neighbours must lie in 1 .. steps - 1 with no re-warm "
                     "(re-warms %s)" % (w0, w1 - 1, sorted(rw)))
    if a.stage_timers:
        p0, p1 = a.prof_start, a.prof_start + a.prof_steps
        if not 1 <= a.prof_steps <= 16 or p0 < 2 or p1 >= a.steps or rw & set(range(p0 - 1, p1 + 1)):
            ap.error("--prof-start / --prof-steps: the window %d .. %d and its neighbours must lie in 1 .. steps - 1 "
                     "with no re-warm (re-warms %s)" % (p0, p1 - 1, sorted(rw)))
        if a.sel_steps:
            s0, s1 = a.sel_start, a.sel_start + a.sel_steps
            if (not 2 <= a.sel_steps <= 64 or not 2 <= a.sel_rows <= 16 or s0 < 1 or s1 > a.steps
                    or rw & set(range(s0, s1)) or set(range(s0, s1)) & set(range(p0 - 1, p1 + 1))):
                ap.error("--sel-start / --sel-steps / --sel-rows: 2 .. 64 steps, 2 .. 16 rows, inside 1 .. steps - 1, "
                         "no re-warm (%s), apart from the profiler window %d .. %d and its neighbours" % (
                             sorted(rw), p0, p1 - 1))
    try:
        a.gemm_mults = tuple(int(x) for x in str(a.gemm_mults).split(","))
    except ValueError:
        ap.error("--gemm-mults: a comma list of integers")
    if a.stage_timers and a.vfy_steps:
        m = a.gemm_mults
        if (any(not 1 <= x <= 8 for x in m) or list(m) != sorted(set(m)) or 1 not in m
                or a.sel_rows not in m):
            ap.error("--gemm-mults %s: increasing, each in 1 .. 8, with 1 and --sel-rows %d" % (m, a.sel_rows))
        v0, v1 = a.vfy_start, a.vfy_start + a.vfy_steps
        s0, s1 = (a.sel_start, a.sel_start + a.sel_steps) if a.sel_steps else (0, 0)
        if (not a.sel_rows + 2 <= a.vfy_steps <= 64 or v0 < 1 or v1 > a.steps or rw & set(range(v0, v1))
                or set(range(v0, v1)) & set(range(p0 - 1, p1 + 1)) or set(range(v0, v1)) & set(range(s0, s1))):
            ap.error("--vfy-start / --vfy-steps: %d .. 64 steps (--sel-rows + 2), inside 1 .. steps - 1, no re-warm "
                     "(%s), apart from the profiler window %d .. %d and its neighbours and from the selection window "
                     "%d .. %d" % (a.sel_rows + 2, sorted(rw), p0, p1 - 1, s0, s1 - 1))
    return a


def arm_headline(a, model, ids, gates, NL, CEN, torch):
    before = snapshot_targets(NL, CEN, torch.cuda)
    base_alloc = round(torch.cuda.memory_allocated() / 1e9, 3)
    log("arm H (B %d, %d steps): r2_base_cell.run_shipped, no timer" % (a.batch, a.steps))
    try:
        out = BC.run_shipped(model, ids, a.steps, BC.CudaClock())
    except Exception as e:
        traceback.print_exc()
        gates.append(("H-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        return None, None
    pa, pr = BC.gpu_peak()
    moved = moved_targets(before, snapshot_targets(NL, CEN, torch.cuda))
    rw = {t for t, _ in out["rewarms"]}
    steady = BC.period_stats(out["periods"], [t for t in range(a.steps) if t not in rw])
    H = dict(rewarms=out["rewarms"], prefill_s=out["prefill_s"], steady=steady,
             rates=BC.rates(a.batch, a.steps, out["periods"], steady), gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr,
             gpu_base_alloc_GB=base_alloc, device_used_end_GB=out["device_used_end_GB"],
             rewarm_ms=[round(out["periods"][t], 1) for t, _ in out["rewarms"]],
             periods_ms=[round(p, 3) for p in out["periods"]])
    gates.append(("H-RUN", True, True, "%d steps, %d re-warms" % (a.steps, len(out["rewarms"]))))
    gates.append(("H-UNPATCHED", True, moved == [], "patch targets moved during H: %s (of %d)" % (moved, len(before))))
    return H, out["inputs"]


def arm_timer(a, model, ids, inputs_h, H, gates, NL, CEN, torch):
    n_layers = len(model.layers)
    pool = getattr(CEN, "_pool", None) is not None
    before = snapshot_targets(NL, CEN, torch.cuda)
    clock = EvClock()
    exp = dict(main=expected_sequence(n_layers, pool), sel=expected_sequence(n_layers, pool, sel=True))
    v0, v1 = (a.vfy_start, a.vfy_start + a.vfy_steps) if a.vfy_steps else (0, 0)
    if a.vfy_steps:
        exp["vfy"] = expected_sequence(n_layers, pool, vfy=a.gemm_mults)
    rec = StageRecorder(clock, n_layers, exp, ring=a.ring, cap=max(len(v) for v in exp.values()) + 8)
    probe, vfy = None, None
    s0, s1 = (a.sel_start, a.sel_start + a.sel_steps) if a.sel_steps else (0, 0)
    if a.sel_steps:
        probe = SelectionProbe(rec, model.layers, lambda: (model.pooling_buf_all.shape[-1], model.pooling_buf_all.dtype),
                               NL.infllmv2_attn_stage1_fast, NL.nosa_pooling, torch, rows=a.sel_rows,
                               qk_select=NL.QK_SELECT)
    if a.vfy_steps:
        vfy = VerifyProbe(rec, v1 - v0, torch, NL.F, NL.layer_norm, NL.silu_and_mul,
                          final=(model.norm_weight, model.norm_variance_epsilon, model.lm_head), mults=a.gemm_mults,
                          rows=a.sel_rows)
    patcher = Patcher(rec, NL, CEN, n_layers, probe, vfy)
    cap = CaptureTimer(rec)
    gct = GcTimer()
    p0, p1 = a.prof_start, a.prof_start + a.prof_steps
    act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    hooks = TimerHooks(rec, patcher, cap, gct, (p0, p1), clock, lambda: torch.profiler.profile(activities=act),
                       torch.cuda, a.steps, sel_window=(s0, s1), probe=probe, vfy_window=(v0, v1), vfy=vfy)
    log("arm T (B %d, %d steps; stage timers, ring %d, %d boundaries per step; profiler window %d .. %d; selection "
        "probe %d .. %d, %d rows; verify probe %d .. %d, GEMM rows per request %s)" % (
            a.batch, a.steps, a.ring, len(exp["main"]), p0, p1 - 1, s0, s1 - 1, a.sel_rows, v0, v1 - 1, a.gemm_mults))
    try:
        out = decode_loop(model, ids, a.steps, clock, hooks)
    except Exception as e:
        traceback.print_exc()
        gates.append(("T-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        moved = moved_targets(before, snapshot_targets(NL, CEN, torch.cuda))
        gates.append(("T-RESTORED", True, moved == [], "patch targets not restored after the failed T: %s" % moved))
        return None
    pa, pr = BC.gpu_peak()
    moved = moved_targets(before, snapshot_targets(NL, CEN, torch.cuda))
    rw = {t for t, _ in out["rewarms"]}
    keep = [t for t in range(a.steps) if t not in rw and not p0 - 1 <= t <= p1 and not s0 <= t < s1
            and not v0 <= t < v1]
    sel_keep = list(range(s0 + 1, s1))                        # the first selection step is the warm-up
    vfy_keep = list(range(v0 + 1, v1))                        # the first verify step is the warm-up (times)
    periods = [rec.steps[t]["period_ms"] for t in range(a.steps)]
    steady = BC.period_stats(periods, keep)
    st = stage_summary(rec.steps, keep)
    sel = selection_summary(rec.steps, sel_keep, a.sel_rows) if probe is not None else None
    gemm_rows, vsum = None, None
    if vfy is not None:
        gemm_rows = gemm_rows_summary(rec.steps, vfy_keep, a.gemm_mults, a.batch,
                                      shipped_ms=_get(st, "groups", "gemm_mlp", "median_ms"))
        vsum = vfy_summary(rec.steps, list(range(v0, v1)), vfy.take(), vfy.union_n, vfy.streams,
                           model.layers[0].topk_blocks)
    eq = bool(torch.equal(out["inputs"], inputs_h))
    ndiff = 0 if eq else int((out["inputs"] != inputs_h).sum())
    prof, prof_err = None, None
    if hooks.prof is not None:
        path = os.path.join(a.out, "trace_T.json")
        try:
            hooks.prof.export_chrome_trace(path)
            ops, launches = pack_trace(path, a.prof_steps)
            agg = P3.aggregate(ops, a.prof_steps)
            win = [periods[t] for t in range(p0, p1)]
            prof = dict(window=[p0, p1 - 1], steps=a.prof_steps, launches=launches, device_ms_sum=agg["total"]["ms"],
                        gpu_ops_per_step=agg["total"]["kernels"], device_busy_ms=busy_union_ms(ops, a.prof_steps),
                        classes=agg["classes"], names=agg["names"][:120],
                        window_period_mean_ms=round(statistics.fmean(win), 3), trace=path + ".gz")
        except Exception as e:
            traceback.print_exc()
            prof_err = "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])
    per_step = {}
    for t in range(a.steps):
        r = rec.steps[t]
        row = dict(period_ms=round(r["period_ms"], 3), label=r["label"], kind=r["kind"], host_ms=hooks.host_ms[t],
                   gc_ms=hooks.gc_ms[t])
        if r.get("stages"):
            row["groups"] = {g: round(sum(r["stages"].get(x, 0.0) for x in ls), 3) for g, ls in GROUPS}
            if r["kind"] == "sel":
                row["sel"] = {x: round(r["stages"].get(x, 0.0), 3) for x in SEL_LABELS}
            if r["kind"] == "vfy":
                row["vfy"] = {x: round(r["stages"].get(x, 0.0), 3) for x in vfy_labels(a.gemm_mults)}
        per_step[str(t)] = row
    cm, cs, cv = rec.chain_of("main"), rec.chain_of("sel"), rec.chain_of("vfy")
    T = dict(rewarms=out["rewarms"], prefill_s=out["prefill_s"], steady=steady, keep_n=len(keep),
             rates=BC.rates(a.batch, a.steps, periods, steady), gpu_peak_alloc_GB=pa, gpu_peak_reserved_GB=pr,
             device_used_end_GB=out["device_used_end_GB"], stages=st, per_layer_mean_ms=rec.per_layer_mean(),
             chain=dict(main=cm, sel=cs, vfy=cv, n_inner=rec.n_inner, expected_len={k: len(v) for k, v in exp.items()},
                        closure_max_ms=round(rec.closure_max_ms, 6)),
             selection=sel, selection_probe=(dict(window=[s0, s1 - 1], used=sel_keep, rows=a.sel_rows,
                                                  calls=probe.calls, errors=probe.errors) if probe else None),
             gemm_rows=gemm_rows, vfy=vsum,
             verify_probe=(dict(window=[v0, v1 - 1], used_times=vfy_keep, rows=a.sel_rows, mults=list(a.gemm_mults),
                                calls=vfy.calls, errors=vfy.errors, union_n=vfy.union_n, streams=vfy.streams)
                           if vfy else None),
             recapture=recapture_summary(hooks.recap, rec.steps, H.get("rewarm_ms") if H else None),
             profile=prof, profile_error=prof_err, tokens_equal_H=eq, token_diffs=ndiff,
             patch_installs=patcher.installs, per_step=per_step)
    gates.append(("T-RUN", True, True, "%d steps, %d re-warms, %d steady timed steps" % (
        a.steps, len(out["rewarms"]), len(keep))))
    gates.append(("T-EQ", True, eq, "T's tokens equal H's: %s (%d differ). A difference means a timer or the probe "
                  "changed the decode OR the shipped decode is not run-to-run deterministic (no record shows it is)" % (
                      eq, ndiff)))
    nm = rec.n_inner.get("main", 0)
    ok = cm["bad"] == 0 and cm["ok"] == nm == len(keep) and nm > 0
    gates.append(("T-CHAIN", True, ok, "%d of %d steady timed steps have the expected %d boundaries (%d kept); first "
                  "mismatches %s; closure %.6f ms" % (cm["ok"], nm, len(exp["main"]), len(keep), cm["examples"][:3],
                                                      rec.closure_max_ms)))
    if probe is not None:
        ns = rec.n_inner.get("sel", 0)
        ok = (cs["bad"] == 0 and cs["ok"] == ns == s1 - s0 and not probe.errors
              and probe.calls == n_layers * (s1 - s0))
        gates.append(("T-SEL", True, ok, "%d of %d selection steps have the expected %d boundaries; probe calls %d "
                      "(expected %d); errors %s; first mismatches %s" % (
                          cs["ok"], ns, len(exp["sel"]), probe.calls, n_layers * (s1 - s0), probe.errors[:2],
                          cs["examples"][:3])))
    if vfy is not None:
        nv, want = rec.n_inner.get("vfy", 0), n_layers * (v1 - v0)
        ok = (cv["bad"] == 0 and cv["ok"] == nv == v1 - v0 and not vfy.errors
              and vfy.calls == dict(union=want, count=want, gemm=want) and vsum["union_steps"] > 0)
        gates.append(("T-VFY", True, ok, "%d of %d verify steps have the expected %d boundaries; probe calls %s "
                      "(expected %d each); union steps %d; errors %s; first mismatches %s" % (
                          cv["ok"], nv, len(exp["vfy"]), vfy.calls, want, vsum["union_steps"], vfy.errors[:2],
                          cv["examples"][:3])))
    gates.append(("T-RESTORED", True, moved == [] and not hooks.instance_left,
                  "patch targets moved after T: %s; cache instance attribute left: %s" % (moved, hooks.instance_left)))
    seen = (prof or {}).get("gpu_ops_per_step", 0)
    nl = _get(prof, "launches", "launches_per_step") or 0
    gates.append(("T-PROF", True, prof is not None and seen > 0 and nl > 0,
                  "profiler window: GPU ops per step %s, enqueued launches per step %s; %s" % (seen, nl, prof_err)))
    return T


def arm_nsys(a, model, ids, gates, torch):
    clock = EvClock()
    rec = StageRecorder(clock, len(model.layers), None, ring=a.ring, cap=4)
    w0, w1 = a.nsys_window, a.nsys_window + NSYS_STEPS
    hooks = NsysHooks(rec, (w0, w1), clock, torch.cuda.profiler, torch.cuda.nvtx)
    log("arm N (B %d, %d steps; cudaProfilerStart / Stop around the steps %d .. %d)" % (a.batch, a.steps, w0, w1 - 1))
    try:
        out = decode_loop(model, ids, a.steps, clock, hooks)
    except Exception as e:
        traceback.print_exc()
        gates.append(("N-RUN", True, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        return None
    periods = [rec.steps[t]["period_ms"] for t in range(a.steps)]
    rw = {t for t, _ in out["rewarms"]}
    inside = sorted(rw & set(range(w0, w1)))
    N = dict(window=[w0, w1 - 1], rewarms=out["rewarms"], prefill_s=out["prefill_s"],
             window_period=BC.period_stats(periods, list(range(w0, w1))), periods_ms=[round(p, 3) for p in periods],
             tokens_first_row=out["inputs"][0, :8].tolist())
    gates.append(("N-RUN", True, not inside, "%d steps; re-warms inside the window: %s" % (a.steps, inside)))
    return N


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 20
    base = load_baseline(a.baseline) if a.baseline else None
    import torch
    import nosi.cache_engine as CEN
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    if NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != a.pool:
        print("REFUSED: needs NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS == --pool (%d; the knob is %d)" % (
            a.pool, CEN.POOL_BLOCKS), file=sys.stderr)
        return 20
    try:
        docs, sha, n_docs = load_workload(a.workload, a.context)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    C, S = BC.C_BASE + a.pool, a.context
    rows = BC.doc_rows(a.batch, n_docs)
    ids = docs[rows].contiguous().to("cuda")
    del docs
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    expected = BC.expected_rewarms(S, a.steps)
    gates, H, T, N = [], None, None, None
    mem = dict(after_load=BC.host_mem())
    with torch.inference_mode():
        BC.free()
        if a.nsys_window is None:
            H, inputs_h = arm_headline(a, model, ids, gates, NL, CEN, torch)
            if H is not None:                            # kept even if arm T ends the process (a CUDA fault)
                with open(os.path.join(a.out, "H.json"), "w") as f:
                    json.dump(dict(H=H, baseline=base, gates=gates), f, indent=1, sort_keys=True, default=str)
            mem["after_H"] = BC.host_mem()
            mem["host_cache_released_after_H"] = BC.free(host=True)
            if a.stage_timers:
                if inputs_h is None:
                    gates.append(("T-RUN", True, False, "NOT_RUN: H failed (T's tokens are checked against H's)"))
                else:
                    T = arm_timer(a, model, ids, inputs_h, H, gates, NL, CEN, torch)
                mem["after_T"] = BC.host_mem()
                BC.free(host=True)
        else:
            N = arm_nsys(a, model, ids, gates, torch)
            BC.free(host=True)
    res = dict(baseline_ratios=baseline_ratios(C, a.batch, S, H, base))
    if H is not None and T is not None:
        hm = H["steady"]["median_ms"]
        res["overhead_ratio"] = round(T["steady"]["median_ms"] / hm, 4) if hm and T["steady"]["median_ms"] else None
        res["stages"] = T["stages"]
        res["selection"] = T["selection"]
        res["launches"] = _get(T, "profile", "launches")
        busy = _get(T, "profile", "device_busy_ms")
        res["gap_ms"] = round(hm - busy, 3) if hm and busy is not None else None
        res["gap_T_window_ms"] = (round(T["profile"]["window_period_mean_ms"] - busy, 3) if busy is not None else None)
        res["recapture"] = T["recapture"]
        sf = _get(T, "stages", "save_frac")
        ref = base.get("steady_tok_s") if base and "error" not in base else None
        sel = T["selection"] if T["selection"] and T["selection"].get("n") else None
        res["ceiling"] = ceiling(hm, a.batch, sf, sel=sel, ref_tok_s=ref) if hm and sf else None
        res["gemm_rows"] = T["gemm_rows"]
        res["verify_probe"] = T["vfy"]
        gather = hm * sf["gather_only"] if hm and sf and "gather_only" in sf else None
        res["verify_ceiling"] = (verify_ceiling(hm, a.batch, gather, T["gemm_rows"], sel, T["vfy"], rows=a.sel_rows,
                                                attention_ms=_get(T, "stages", "groups", "attention", "median_ms"),
                                                mb1_tok_s=ref) if T["gemm_rows"] is not None else None)
    preds = verdicts_p0(C, a.batch, res, S=S) + verdicts_sweep(C, a.batch, res, S=S) if a.nsys_window is None else []
    if a.mbase and H is not None:
        preds = preds + BC.verdicts_mbase(C, a.batch, H, None)
    meta = dict(batch=a.batch, steps=a.steps, S0=S, context=S, doc_rows=rows, n_docs=n_docs,
                n_unique_docs=min(a.batch, n_docs), workload=a.workload, workload_sha256=sha,
                expected_rewarms=expected, attn_splits=NL._ATTN_SPLITS, pool_blocks=CEN.POOL_BLOCKS, capacity_C=C,
                cell_key=cell_key(C, a.batch, S), stage_timers=a.stage_timers,
                prof_window=[a.prof_start, a.prof_start + a.prof_steps - 1],
                sel_window=([a.sel_start, a.sel_start + a.sel_steps - 1] if a.sel_steps else None),
                sel_rows=a.sel_rows, ring=a.ring, nsys_window=a.nsys_window, mbase=a.mbase,
                vfy_window=([a.vfy_start, a.vfy_start + a.vfy_steps - 1] if a.vfy_steps else None),
                gemm_mults=list(a.gemm_mults), registered_sweep_utc=REGISTERED_SWEEP_UTC, predictions_sweep=PRED_SWEEP,
                verify_formula=VERIFY_FORMULA, verify_accept=VERIFY_ACCEPT, draft_recovery=DRAFT_RECOVERY,
                ref_cached_tok_s=REF_CACHED_TOK_S, weight_floor_ms=WEIGHT_FLOOR_MS,
                registered_utc=REGISTERED_UTC, predictions=PRED_P0, ceiling_formula=CEILING_FORMULA, groups=GROUPS,
                saves=SAVES, nosi_commit=os.environ.get("BASE_COMMIT", "unknown"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), host_mem=mem)
    out = dict(meta=meta, H=H, T=T, N=N, baseline=base, results=res,
               gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               predictions=[dict(name=p[0], verdict=p[1], detail=p[2]) for p in preds])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    for g in gates:
        print("GATE %s %s: %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:400]))
    for p in preds:
        print("PREDICTION %s %s: %s" % p)
    if H:
        print("REPORT H C %d B %d S %d: steady %s ms/step (mean %s), %s tok/s, effective %s tok/s, peak %s GB, prefill "
              "%s s, re-warm periods %s ms" % (C, a.batch, S, H["steady"]["median_ms"], H["steady"]["mean_ms"],
                                               H["rates"]["steady_tok_s"], H["rates"]["effective_tok_s"],
                                               H["gpu_peak_alloc_GB"], H["prefill_s"], H["rewarm_ms"]))
    if T and T["stages"].get("n"):
        st = T["stages"]
        print("REPORT T stages (ms per step, median of %d steps; share of the period): %s; period %s; closure %s ms" % (
            st["n"], ", ".join("%s %.3f (%.1f %%)" % (g, st["groups"][g]["median_ms"], 100 * st["group_frac"][g])
                               for g, _ in GROUPS), st["period"]["median_ms"], st["closure_max_ms"]))
        print("REPORT T labels (ms per step, median): %s" % ", ".join(
            "%s %.3f" % (x, v["median_ms"]) for x, v in st["labels"].items()))
        print("REPORT T overhead: T / H steady %s" % res.get("overhead_ratio"))
    if T and T["selection"] and T["selection"].get("n"):
        s = T["selection"]
        print("REPORT selection (ms per step, median of %d steps, R %d): shipped %s, eager 1 row %s (stage 1 %s, pool %s), "
              "eager %d rows %s (stage 1 %s, pool %s), prep %s; extra of K %d: joint %s (%s per position), serial %s "
              "(shipped x K %s); R rows / 1 row %s" % (
                  s["n"], s["rows"], s["shipped"]["median_ms"], s["one_row"]["median_ms"],
                  s["stage1_one"]["median_ms"], s["pool_one"]["median_ms"], s["rows"], s["r_rows"]["median_ms"],
                  s["stage1_rows"]["median_ms"], s["pool_rows"]["median_ms"], s["prep"]["median_ms"], s["K"],
                  s["extra_joint_ms"], s["per_position_joint_ms"], s["extra_serial_ms"],
                  s["extra_serial_shipped_ms"], s["rows_over_one"]))
    if T and T["profile"]:
        pr = T["profile"]
        print("REPORT launches per step %s (kinds %s); device busy %s ms, op sum %s ms; host gap H %s ms, window %s ms" % (
            pr["launches"]["launches_per_step"], pr["launches"]["kinds_per_step"], pr["device_busy_ms"],
            pr["device_ms_sum"], res.get("gap_ms"), res.get("gap_T_window_ms")))
    if T:
        for r in T["recapture"]["each"]:
            print("REPORT re-capture step %s (%s): period %s ms (H %s), host %s ms: graph enter %s (first %s), exit %s, "
                  "gc %s ms %s, other %s" % (r["t"], r["why"], r["period_ms"], r["headline_period_ms"], r["host_ms"],
                                             r["graph_enter_ms"], r["graph_enter_first_ms"], r["graph_exit_ms"],
                                             r["gc_ms"], r["gc_count"], r["host_other_ms"]))
    if res.get("ceiling"):
        cg = res["ceiling"]
        for name, _ in SAVES:
            x = cg.get(name, {})
            print("REPORT ceiling %s C %d B %d S %d: base %.3f ms, save %s ms, draft floor %s ms, speedup %s, %s tok/s "
                  "(/ mb1 %s), save / rest %s, verify budget per round %s ms; after the K %s extra selections: joint "
                  "%s (min a %s), serial %s (min a %s)" % (
                      name, C, a.batch, S, cg["baseline_ms"], x.get("save_ms"), x.get("draft_floor_ms"),
                      x.get("max_speedup"), x.get("ceiling_tok_s"), x.get("ceiling_vs_ref"), x.get("save_over_rest"),
                      x.get("verify_budget_ms"), cg.get("K"), x.get("verify_budget_after_sel_joint_ms"),
                      x.get("min_accepted_joint"), x.get("verify_budget_after_sel_serial_ms"),
                      x.get("min_accepted_serial")))
    gr = res.get("gemm_rows")
    if gr and gr.get("n"):
        print("REPORT GEMM rows C %d B %d S %d (median of %d verify steps; ms per step, 32 layers + final): %s; marginal "
              "ms per row %s; least squares %s ms per row, intercept %s ms (floor about %s); shipped gemm_mlp %s ms "
              "(probe B rows / shipped %s); prep %s ms" % (
                  C, a.batch, S, gr["n"], ", ".join("%d rows %.3f (%.5f per row, %+.3f over the floor)" % (
                      p["rows"], p["median_ms"], p["ms_per_row"], p["over_floor_ms"]) for p in gr["points"]),
                  [m["ms_per_row"] for m in gr["marginal"]], gr["slope_ms_per_row"], gr["intercept_ms"],
                  gr["floor_ms"], gr["shipped_gemm_mlp_ms"], gr["x1_over_shipped"], gr["prep"]["median_ms"]))
    vp = res.get("verify_probe")
    if vp:
        print("REPORT union of R %d consecutive selections: mean %s, max %s per (head, request) over %d steps; extras %s "
              "per stream; host loads %s per stream-step (%s per step, %d steps); fetch %s ms per step, %s ms per load; "
              "union fetch (every extra a host load) %s ms per joint pass; extras / loads per stream %s" % (
                  a.sel_rows, vp["union_mean"], vp["union_max"], vp["union_steps"], vp["extras_per_stream"],
                  vp["loads_per_stream"], vp["loads_per_step"], vp["load_steps"], vp["fetch_ms_per_step"],
                  vp["ms_per_load"], vp["union_fetch_ms"], vp["extras_over_loads"]))
    vc = res.get("verify_ceiling")
    if vc and vc.get("missing"):
        print("REPORT verify ceiling: not computed, missing %s" % vc["missing"])
    elif vc:
        print("REPORT verify joint pass C %d B %d S %d: %s ms (%s shipped steps) = base %.3f + GEMM extra %s (x1 %s, x%d %s)"
              " + selection extra %s + union fetch %s; components (GEMM R rows + selection R rows + one attention + "
              "union fetch + rest of the step) %s" % (
                  C, a.batch, S, vc["joint_ms"], vc["joint_steps"], vc["baseline_ms"], vc["gemm_extra_ms"],
                  vc["gemm_x1_ms"], vc["rows"], vc["gemm_xr_ms"], vc["sel_extra_ms"], vc["union_fetch_ms"],
                  vc.get("components")))
        for c in vc["cases"]:
            print("REPORT verify case fetch %s draft %s (share %s) a %s: draft %.3f ms, round %.3f ms, %.1f tok/s, / %s "
                  "%.4f, / own %.4f, / mb1 %s; break-even joint own %.3f (margin %+.3f), cached %.3f (margin %+.3f)" % (
                      c["fetch"], c["draft"], c["recovery"], c["accept"], c["draft_ms"], c["round_ms"], c["tok_s"],
                      vc["ref_tok_s"], c["vs_cached"], c["vs_own"], c["vs_mb1"], c["breakeven_joint_own_ms"],
                      c["margin_own_ms"], c["breakeven_joint_cached_ms"], c["margin_cached_ms"]))
    r = res.get("baseline_ratios")
    if r:
        print("SCORE C %d B %d S %d / mb1 (C %d B %d S %d, same job): median %.4f, mean %.4f, effective %.4f, steady "
              "tok/s %.4f" % (C, a.batch, S, r["baseline"]["C"], r["baseline"]["B"], r["baseline"]["S"],
                              r["median_ratio"], r["mean_ratio"], r["effective_ratio"], r["steady_tok_ratio"]))
    elif base:
        print("SCORE: no ratio (baseline %s)" % base.get("error", "unusable"))
    if N:
        print("REPORT N window %s: period %s; re-warms %s" % (N["window"], N["window_period"], N["rewarms"]))
    nfail = sum(1 for g in gates if g[1] and not g[2])
    print("r2_p0prof: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
