"""R2 FIATTN: a BOUNDED comparison of an EXISTING kernel against piece 3 v2, BEFORE any switch.

Existing kernel: FlashInfer 0.5.3 (installed in nosa-env.sif; NOT upgraded), BatchDecodeWithPagedKVCacheWrapper with
use_tensor_cores=True and backend "fa2" (it runs the FA2 paged prefill kernel with one query row per request), with a
JIT custom attention variant for the CIS bias. Our kernel: piece 3 v2 (nosi-r2-m2-biattn 7c826c1,
nosi/nosi/bi_paged_attention.py, blob 39ed5e18), loaded BY PATH from a pinned copy. Design of the comparison:
/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/attn-reuse-assessment.md (section 5, item 10, open items). Plan and the
REGISTERED predictions: docs/R2_FIATTN_PLAN.md (PRED below is the same table; test FA8 keeps them equal).
Standalone: production, the reference and the tick are NOT touched; nothing here is wired into the tick.
SOURCE ONLY, NOT RUN.

ARMS
  P   piece 3 v2: bi_paged_decode (fixed split, 8 pages per split; its own fp32 partials and combine).
  F8  FlashInfer, OPTION A (two calls per layer: one per KV head, num_kv_heads 1, the K / V view k[..., h:h+1, :]),
      plan(fixed_split_size=8) (= piece 3's PAGES_PER_SPLIT).
  F0  the same with plan(disable_split_kv=True) (one chunk per request: the CUDA-graph-safe form).
  FH  the same with FlashInfer's default split heuristic: a NEGATIVE CONTROL for the invariance gates only.
  S5  optional, rows-per-request 4 cells: sharedkv5 (nosi-r2-sharedkv5 5eb7514, blob 5c2e7b25) tile "reload", mode
      "split" (REPORT only).
  STRIDED (REPORT probe): OPTION B, one FlashInfer call per layer through an OVERLAPPING as_strided view of the store
      (page id = (entry * S + slot) * 64 * Hkv + h, stride_page = D, stride_n = Hkv * D).
  A FlashInfer "request" is one (row, KV head). Page ids of option A: entry * S + slot (the store viewed as
  (N * S, 64, Hkv, D) pages). Packing is device ops only (cumsum + scatter into an R * P buffer; no host read);
  plan() itself reads indptr / last_page_len on the HOST (FlashInfer's design).

THE CIS BIAS IN FLASHINFER: the variant R2FiCisBias (VARIANT_DECL) follows the ALiBi branch of FlashInfer's default
variant (logits arrive unscaled; the variant scales them and sets sm_scale_log2 = log2(e)):
    logit = (q.k + bias[page * page_mul + entry * entry_stride + head]) * sm_scale,
with page = indices[indptr[request] + kv_idx / 64], entry = kv_idx % 64, read only for kv_idx < kv_len (positions past
the request's extent are masked by the kernel; the guard keeps the read in bounds). Option A: page_mul = 64 * Hkv,
entry_stride = Hkv, head = h; option B: 1, Hkv, 0. If the JIT build fails, that is a FINDING (gate C0 fails) and the
run continues BIAS-FREE (bias = 0 for every arm, FlashInfer's default module), every other gate a REPORT.

CASES (fixed, seeded; make_case): NOSA-8B geometry read from --config (q heads, KV heads, head_dim), 16K context:
every row's tail block is 251 (positions 16064 .. 16127), S = 256 slots per storage entry with a seeded block -> slot
map, 64 pages of 64 per (row, KV head): block 0, the 15 recent blocks 236 .. 250, 47 scattered blocks drawn with
weight 1 / sqrt(251 - b) (locality), and the tail. Rows of one request (rows per request 1 or 4) sit at consecutive
positions; row u > 0 swaps 4 scattered blocks of the request's base list. bf16 q / K / V (N(0, 1)), bias uniform in
[-8, 8] (so that the bias moves the logits by about +-0.7: gate C has teeth, witness CW). B in {8, 32} x rows per
request in {1, 4}. Extra batches: "alt" (every odd row has a new q and new lists: siblings changed), "hazard" (the
full batch + 3 short rows of 5, 8 and 9 pages on an extra storage entry: the split_kv switch of the assessment, 2.3
item 5).

GATES (MUST unless noted; P, F8 and F0 each; "bitwise" = equal int16 views of the bf16 outputs)
  C0   the variant JIT build (time charged separately).
  C    max |arm - fp64 reference| <= TOL_C on N_SAMPLE rows (the reference is THIS module's float64 attention, not
       piece 3's code).
  CW   (F8) the bias witness: F8 with bias = 0 is FAR from the biased reference (> TOL_C). Precondition: the biased and
       bias-free references differ by >= WITNESS_FACTOR x TOL_C on the sampled rows, else VOID (= fail).
  I1   alone: each sampled row as a batch of one (its own plan) == the same row in the full batch.
  I2   order: the batch in reversed request order and in a random row order == unpermuted, per row.
  I3   siblings: the even rows of the "alt" batch == the even rows of the full batch.
  I4   the split_kv switch: each short row alone == the same row in the hazard batch, and every full row of the hazard
       batch == its full-batch output. F8 precondition (read from the plans): the rows of <= 8 pages plan split_kv
       false alone and true in the hazard batch, on every KV head; else VOID (= fail).
  I5   poison: every (slot, head) that no row of the batch lists, and every tail row past the largest extent that
       reads it, is NaN in K, V and bias; the full batch stays finite and bitwise equal.
  N    (FH) the negative control: some sampled row alone != batched. Precondition: the heuristic's chunk size (or
       split flag) for the batch of one differs from the full batch's (read from the plans); else VOID. Pass only
       when DETECTED with the precondition met.
  D    REPORT: bits vs piece 3 (rows differing, max |d|); rounding differences never reject.
  T    REPORT: CUDA-event time of the attention call (FlashInfer: both head calls + the output interleave; and the
       runs alone), plan() host time with host indptr and with device indptr (a D2H sync inside plan), packing time,
       JIT build / load time, workspace bytes and the peak allocated delta of one call. Bands from PRED.
  COMPLETE  [fix after fia1] per requested cell: every expected MUST gate of the cell is present (a cell or arm
       that never reported fails here, not by absence).
  REUSE  REPORT [fix after fia1]: one plan, a second layer's lists written in place into the planned indices
       buffer (constant counts), bitwise vs a fresh plan; timed as "layer with the tick's plan reused".
  The FlashInfer float workspace is sized from FlashInfer's own formula for each plan (fi_float_need; fia1's fixed
  256 MiB overflowed the option-B probe at 32x4) and reported as memory (need, allocated, reserved by the plan).
  NCU  REPORT (--ncu): one ncu process per cell (all arms of the cell, NVTX range fia|<arm> each), kernel filter
       KERNEL_FILTER (the attention kernels only), --cache-control all (cold L2 per kernel), hard cap min(600 s, the
       budget left); DRAM bytes read / write per arm vs the per-row ideal and the request-union ideal. Label: every
       arm reuses pages through the HARDWARE cache only (L2; S5: a forced temporal locality inside one program), none
       has explicit shared loads (REUSE_LABEL).

BUDGET: --budget-s for the whole process (the step gives 2400 s inside a 45 min step); a cell that would start too
late is skipped and fails gate BUDGET; ncu cells get min(--ncu-cap, the time left - NCU_MARGIN_S) or are skipped
(REPORT).

OUTPUT: --out/summary.json (gates, reports, plans, timings, predictions, environment), one GATE / REPORT line per gate
on stdout. EXIT: the number of failed MUST gates (<= 125); 2 = refused (--out exists, bad arguments, no CUDA, no
FLASHINFER_WORKSPACE_BASE, a FlashInfer version other than 0.5.3).

[fia3] SHARED-KV VERIFY ARMS (plan kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md Task 0.3; survey
SURVEY_shared_kv_kernel.md option 1; notes docs/NOTES_fia3.md). NO NEW KERNEL: existing FlashInfer and NOSA kernels,
Python glue, and one more JIT variant (FlashInfer's own customization hook, as the F8 arm's CIS variant).
  FP0 FlashInfer BatchPrefillWithPagedKVCacheWrapper (fa2), option A (one wrapper per KV head, as F8). One FlashInfer
      request = (request, KV head); qo_len = rows per request (1 draft + 3 verify); FlashInfer's own GQA packing puts
      the 16 query heads x 4 rows of that KV head in ONE CTA tile of 64 rows (packed_qo_len, CTA_TILE_Q 64), so each
      KV tile is staged in shared memory once for all of them. KV pages = the union of the rows' pages (ascending
      (block id, slot)); a packed custom mask (FlashInfer's layout: bit qo_idx * kv_len + kv_idx of the request's byte
      segment) gives each row only its own tokens, expanded from the block bits (a row's last page cut at last_len).
      The variant R2FiCisBiasMask = R2FiCisBias + a LogitsMask that reads that mask (FlashInfer applies a custom mask
      through the VARIANT: variants.cuh DefaultAttention; a custom variant without LogitsMask ignores the mask).
      plan(disable_split_kv=True, packed_custom_mask=...) (the packed mask selects the kCustom kernels). One plan per
      tick: with no split the plan does not depend on the union sizes, so later layers write their CSR and mask IN
      PLACE into the buffers the plan holds (relayer; REPORT R_REUSE_FP0 checks it bitwise).
  FP8 the same with fixed_split_size 8 (pages of the union list): FAST mode only (the chunks depend on the union).
      Hazard (MODEL from the 0.6.8 source): a row whose first KV tile of a chunk is fully masked starts that chunk's
      online softmax at -inf (exp2(-inf - -inf) = NaN, prefill.cuh update_mdo_states); REPORT R_FP8_chunk_tile0 counts
      such cells, and FP8's C gate is a REPORT.
  FPU FP0's call without the mask (plan without packed_custom_mask: kNone kernels): TIMING ONLY, the mask's cost.
  V1  NOSA's own prefill varlen route: EQ nosi-eqmem nosi/nosi/verify/verify_step.py varlen_attention (pinned blob,
      loaded by path; the internal op infllm_v2 _wrapped_infllmv2_attn_varlen_forward, two calls V x exp(cis) and
      exp(cis), then the quotient) over a CONTIGUOUS union store built here (union page j at slot j, the union's last
      page at slot W - 1, seqused_k = (W - 1) x 64 + the last-page extent, so the causal rule cuts each row at its own
      extent). The bench bias is pre-scaled (exp(bias x D^-0.5)) because the bench logit is (q.k + bias) x scale.
      Precondition V1_pre (consecutive extents, one shared tail); a missing import is a FINDING (REPORT).
  Gates [fia3]: FP_API and C0_FP (MUST when an FP arm is asked); G_block0 (MUST: every row lists block 0, so no row
  starts on a fully masked KV tile); C (fp64: TOL_C, MUST for FP0, REPORT for FP8; V1: TOL_V1, MUST); FI1-FI3 and FI5 (request
  alone, request order, other requests changed, poison: MUST for FP0, REPORT for FP8); CW_FP0 (bias witness) and
  MW_FP0 (mask witness: every row given the whole union must FAIL C); REPORTs: R_X_row_alone (the EXACT-mode probe:
  A100 KV tiles are 128 tokens = 2 pages, so a row's bits depend on its siblings' pages: FP0 is FAST-only), R_REUSE,
  R_T (event time, plan host time, union + mask build per layer eager / host / CUDA graph, the tick overhead), D
  (bits vs P), NCU (DRAM bytes, L2 -> SM sectors, registers, local-memory bytes) and R_DECISION (the registered rule,
  PRED3).
  Cells [fia3]: B up to 256; the suffix s1 takes the lists of the recorded S1 tick tables (--s1-tables; requests =
  the recorded groups of exactly rows-per-request rows, cycled to B; synthetic q / K / V / bias). Cells with B > 64
  draw K / V / bias on the GPU (a seeded CUDA generator); B <= 64 keeps fia1 / fia2's CPU data bitwise.

[fia4] THE FORK OF vLLM's TRITON UNIFIED ATTENTION (plan Task 0.3 option 2; docs/NOTES_fia4.md). FlashInfer FP0 lost
(SCORING_fia3.md: 1.68-1.70x piece 3, 77-100 ms plan + build per tick). The rung-4 row is written; the human allowed ONE
fork: nosi/nosi/fork_vllm_unified_attention.py (vLLM 0.20.2, every changed line tagged [fork]), loaded by path
(--fv-kernel). No other kernel.
  FV  one program per (request, KV head): the request's 4 rows x the 16 query heads = one 64-row tile; each union KV
      tile (one 64-token block) is loaded once and multiplied for all rows (tl.dot); each row sees only its own blocks
      and its extent by a SCORE mask (8 bits per row and union entry); logit = (q.k + bias) x scale. The program BUILDS
      its union table on the device (fork_build_union: binary search over the rows' sorted lists): no plan, no host
      read, no extra launch. No split (vLLM's 2D path).
  FVT the same kernel reading a table that the torch builder made before the launch (fp_union + fv_tables_from_union):
      timing only. ncu FV - FVT = the in-kernel build's device cost; FVT's torch build per layer is reported too.
  Gates [fia4] (MUST unless marked): C0_FV_import; G_block0 (shared with FP0); C_FV (fp64, TOL_C); FI1-FI3 and FI5 for
  FV (FI5 is MUST with no precondition: FV loads only the tokens some row of the request sees); CW_FV (bias witness);
  MW_FV (mask witness, through the prebuilt-table path with every row given the pair's reach); U_FV_table (the
  kernel-built tables == the torch tables, on the full and the altreq batch). REPORTs: X_row_alone_FV (the EXACT probe:
  expected 0 rows differ), R_GRAPH_FV (one launch captured in a CUDA graph, replay bitwise), R_FVT_equals_FV, R_T,
  R_FV_build, R_D, the ncu numbers (+ achieved occupancy, shared memory, occupancy limits) and R_DECISION4 (PRED4).

[fia5] PIPELINE K / V INSIDE THE FORK (docs/NOTES_fia5.md; SCORING_fia4.md sections 5-6). The fork's tile loop loads the
rows' extents as vectors (prefetched with the page ids), K and the bias as whole tiles, V up to the tile's reach; the
bias may be float32 (prefetched); the token tile may be a part of a block; one more constexpr gives a no-mask timing
variant. The FV arms pass the CIS bias as float32 (FV_BIAS_BITS; the same values: bf16 -> fp32 is exact).
  FVN  the FV launch with use_extents=False: no extents at all (every row sees every union token). TIMING ONLY (not
       attention; R_FVN_not_attention says so); ncu FV - FVN = the cost of the extents.
  --fv-sweep w4s2,w8s3t32,w4s3b16,...: one more FV arm per launch configuration, named FVW<warps>S<stages>T<tile>B<bias
       bits> (tile default 64, bias default 32). Each swept arm runs FV's gates under its own name (C, FI1-FI3, FI5,
       CW, MW, U_<arm>_table; REPORTs X_row_alone, R_GRAPH, R_D vs P and vs FV, R_<arm>_compile); a swept arm whose
       first call fails (compile, resources) is a REPORT FINDING and runs no gate. R_FV_sweep_<cell> picks the best
       swept configuration (lowest event median among those whose MUST gates pass); ncu profiles P, FV, FVT, FVN and
       that best configuration.
  R_<arm>_compile (REPORT): the compiled kernel of the arm's first call (Triton's handle, kept by the fork in
  FORK_LAST_LAUNCH): num_warps, num_stages, registers and local memory per thread (ptxas), shared memory, the shapes
  of the loop's multi-buffered shared allocations in the TTGIR (the leading dimension = the buffer depth) and a MODEL
  of programs per SM. R_DECISION5 scores PRED5 (the rule of PRED4) for the default FV and the best swept arm.
  ncu also collects the WarpStateStats section (raw CSV) and prints each arm's long-scoreboard stall share.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback

import torch

PAGE = 64                                   # NOSA's block = one page
CTX = 16384                                 # 16K context
TAIL_BLOCK = CTX // PAGE - 5                # 251: the rows' tail block (positions 16064 .. 16127)
S_SLOTS = CTX // PAGE                       # 256 slots per storage entry (blocks 252 .. 255 are never listed)
N_SEL = 64                                  # pages per (row, KV head), the tail included
N_RECENT = 15                               # the recent blocks before the tail
N_SWAP = 4                                  # blocks a verify row swaps against the request's base list
BIAS_RANGE = 8.0                            # bias uniform in [-BIAS_RANGE, BIAS_RANGE]
SHORT_COUNTS = (5, 8, 9)                    # the hazard rows' page counts (5, 8 <= FIXED_SPLIT; 9: two chunks)
SHORT_LAST = 33                             # their extent in the tail page
FIXED_SPLIT = 8                             # = piece 3's PAGES_PER_SPLIT
CELLS = ((8, 1), (8, 4), (32, 1), (32, 4))  # (B, rows per request); 8x1 first: the inspected miniature
MAX_B = 256                                 # [fia3] B up to 256 (the 216 x 4 target cell)
CPU_DATA_MAX_B = 64                         # [fia3] B <= 64: K / V / bias drawn on the CPU (fia1 / fia2 data, bitwise);
                                            # larger cells draw them on the GPU (a seeded CUDA generator)
ARMS = ("P", "F8", "F0", "FH", "S5", "FP0", "FP8", "FPU", "V1", "FV", "FVT", "FVN")
MUST_ARMS = ("P", "F8", "F0")
FP_ARMS = ("FP0", "FP8", "FPU")             # [fia3] FlashInfer paged PREFILL over the request union
FV_ARMS = ("FV", "FVT", "FVN")              # [fia4] the fork of vLLM's Triton unified attention (in-kernel / torch
                                            # table); [fia5] FVN: the FV launch without extents (timing only)
FV_ROWS = 4                                 # [fia4] BLOCK_Q of the fork: the rows of one request in one program
FV_BIAS_BITS = 32                           # [fia5] the FV arms pass the CIS bias as float32 (the fork prefetches it)
FV_SWEEP_NAME = re.compile(r"FVW(\d+)S(\d+)T(\d+)B(16|32)")          # [fia5] a swept FV arm (canonical name)
FV_SWEEP_ITEM = re.compile(r"w(\d+)s(\d+)(?:t(\d+))?(?:b(16|32))?")  # [fia5] one --fv-sweep item
FV_SWEEP_WARPS = (2, 4, 8, 16)
FV_SWEEP_STAGES = (1, 2, 3, 4, 5, 6)
FV_SWEEP_TILES = (16, 32, 64)
FI_VERSION = "0.5.3"                        # the version every JIT cache of nosa-env.sif names (nosa-env/jit/*/flashinfer)
FI_VERSIONS = ("0.5.3", "0.6.8.post1")      # [fia3] accepted: the sif's 0.5.3 and the 0.6.8.post1 read for this change
TOL_C = 1.0e-3                              # REGISTERED: max |arm - fp64| (absolute), every arm
TOL_V1 = 4.0e-3                             # [fia3] V1 only (builder choice, for the lead to accept): NOSA's route
                                            # rounds two bf16 outputs and divides in bf16 (verify_step.py:200): about
                                            # 3 x 2^-9 relative, i.e. above 1e-3 where |out| > 0.2
WITNESS_FACTOR = 10.0                       # CW precondition: the bias moves the reference by >= 10 x TOL_C
MIN_FLOAT_WS = 1 << 20                      # [fix after fia1] the float workspace starts here and grows to the
                                            # plans' need (fi_float_need; fia1's fixed 256 MiB overflowed option B)
N_SAMPLE = 16                               # rows per cell for C, I1 and N
CELL_RESERVE_S = 300                        # a cell starts only with this much budget left
NCU_ORDER = ((32, 4), (32, 1), (8, 4), (8, 1))
NCU_MARGIN_S = 120
NCU_CAP_S = 600
ATTN_KERNELS = ("BatchPrefillWithPagedKVCacheKernel", "PersistentVariableLengthMergeStatesKernel",
                "_bi_split_kernel", "_bi_combine_kernel", "_bi_shared_kernel",
                "flash_fwd", "kernel_topk_to_uint64",          # [fia3] V1: InfLLM-v2's forward and its mask packer
                "kernel_unified_attention")                    # [fia4] FV / FVT: the fork (one launch per call)
KERNEL_FILTER = "regex:" + "|".join(ATTN_KERNELS)
NCU_REQUIRED = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum")
NCU_OPTIONAL = ("lts__t_sector_hit_rate.pct",
                # [fia3] L2 -> SM traffic (sectors of 32 B), L2 / DRAM / SM throughput, local memory (register spills)
                "lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex.sum",
                "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
                "sm__warps_active.avg.pct_of_peak_sustained_active",     # [fia4] achieved occupancy
                # [fia5] the share of active warps waiting on a global load (long scoreboard): the latency witness
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct")
# [fia3] the LaunchStats section gives launch__registers_per_thread (a section always exists; a metric name may not
# be listed by --query-metrics, and an unknown metric name stops ncu)
NCU_SECTIONS = ("LaunchStats",
                "WarpStateStats")   # [fia5] the warp stall reasons (long / short scoreboard, barrier) in the raw CSV:
                                    # SCORING_fia4.md section 8 lists their absence; REPORT only (not summarized)
NCU_LAUNCH = ("launch__registers_per_thread", "launch__shared_mem_per_block_dynamic", "launch__grid_size",
              # [fia4] the occupancy limits and the static shared memory (the fia3 CSV has these LaunchStats columns)
              "launch__shared_mem_per_block_static", "launch__occupancy_limit_registers",
              "launch__occupancy_limit_shared_mem", "launch__waves_per_multiprocessor")
L2_SECTOR_BYTES = 32
TAG_RE = re.compile(r"fia\|([A-Z0-9]+)")
KERNELS_PER_CALL = dict(P=2, F8=4, F0=2, FH=None, S5=2,   # F8: 2 heads x (prefill + merge); F0: 2 x prefill
                        FP0=2, FP8=4, FPU=2, V1=None,     # [fia3] FP8: 2 x (prefill + merge); V1: 2 calls x (packer +
                                                          # forward, maybe split + combine): not fixed
                        FV=1, FVT=1,                      # [fia4] one launch: grid (2B, Hkv), both KV heads
                        FVN=1)                            # [fia5] (a swept arm: 1 as well, expected_kernels)
REUSE_LABEL = dict(
    P="hardware cache only: per-row loads; the rows of a request are adjacent programs of one grid (L2 reuse)",
    F8="hardware cache only: one CTA per (row, chunk); requests in request-major order (L2 reuse)",
    F0="hardware cache only: one CTA per row walking all its pages (L2 reuse)",
    FH="hardware cache only (negative control)",
    S5="hardware cache only, forced temporal locality: one program per (request, head, split) reloads each tile per "
       "row (tile 'reload'; L1 / L2), no register or shared-memory sharing",
    FP0="EXPLICIT shared tile: one CTA per (request, KV head) stages each union KV tile in shared memory once for all "
        "rows x 16 query heads (FlashInfer's GQA packing, CTA_TILE_Q 64); each row selects its own tokens by the "
        "packed custom mask",
    FP8="as FP0, the union cut in chunks of 8 pages (one CTA per chunk) + a merge kernel",
    FPU="as FP0 without the mask (timing only)",
    V1="hardware cache only: InfLLM-v2 varlen, kBlockM 16 = one token x 16 heads per CTA; the rows of a request reuse "
       "the union pages through L2",
    FV="EXPLICIT shared tile: one Triton program per (request, KV head) loads each union block once (64 tokens, only "
       "the tokens some row sees) for the 4 rows x 16 query heads (BLOCK_M 64, one tl.dot); the program builds its "
       "union table itself; per-row score mask",
    FVT="as FV, the union table built by torch before the launch (timing only)",
    FVN="as FV without the extents (every row sees every union token; timing only, not attention)",
)
VARIANT_NAME = "R2FiCisBias"
VARIANT_URI = "r2_fiattn_cisbias_bf16_d%d"
ROW_FIELDS = ("q", "seq_idx", "page_table", "page_key", "num_pages", "last_len")
CACHE_FIELDS = ("k", "v", "bias")

# The CIS bias variant (C++ in a Python string, compiled by FlashInfer's JIT inside the GPU step; our text). The
# ragged kernels are instantiated with the same variant and have no paged_kv: the trait keeps them compiling.
VARIANT_DECL = r"""
template <typename T, typename = void>
struct r2fi_has_paged_kv : std::false_type {};
template <typename T>
struct r2fi_has_paged_kv<T, std::void_t<decltype(std::declval<T>().paged_kv)>> : std::true_type {};

struct R2FiCisBias : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ R2FiCisBias(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float b = 0.f;
    if constexpr (r2fi_has_paged_kv<Params>::value) {
      if (kv_idx < kv_len) {
        uint32_t pg, e;
        params.paged_kv.page_size.divmod(kv_idx, pg, e);
        const int64_t page = static_cast<int64_t>(params.paged_kv.indices[params.paged_kv.indptr[batch_idx] + pg]);
        b = __bfloat162float(params.cis_bias[page * params.cis_page_mul +
                                             static_cast<int64_t>(e) * params.cis_entry_stride + params.cis_head]);
      }
    }
    return (logits + b) * static_cast<float>(params.cis_sm_scale);
  })
};
"""
VARIANT_TENSORS = (["cis_bias"], ["nv_bfloat16"])
VARIANT_SCALARS = (["cis_page_mul", "cis_entry_stride", "cis_head", "cis_sm_scale"],
                   ["int64_t", "int64_t", "int64_t", "double"])

# [fia3] The FP arms' variant: R2FiCisBias (the same LogitsTransform, verbatim) + a LogitsMask that reads OUR packed
# mask with FlashInfer's own formula (variants.cuh DefaultAttention, 0.6.8 :57-63, :78-87): the custom mask of a
# custom variant lives in the variant (the generated PagedParams of a custom module have no maybe_custom_mask field:
# nosa-env/jit/fiattn_fia2_2191262/.../generated/r2_fiattn_cisbias_bf16_d128/batch_prefill_config.inc). Our tensor
# names do not start with "maybe", so FlashInfer 0.5.3 (run args appended in order) and 0.6.8 (prepare_jit_additional_
# args: unknown names take the user args in order) pass the same arguments. r2_mask_indptr holds ABSOLUTE byte
# offsets into r2_mask, per KV head (one row per wrapper).
VARIANT_NAME_FP = "R2FiCisBiasMask"
VARIANT_URI_FP = "r2_fia3_cisbias_mask_bf16_d%d"
VARIANT_DECL_FP = r"""
template <typename T, typename = void>
struct r2fm_has_paged_kv : std::false_type {};
template <typename T>
struct r2fm_has_paged_kv<T, std::void_t<decltype(std::declval<T>().paged_kv)>> : std::true_type {};

struct R2FiCisBiasMask : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;
  uint8_t* r2_mask_ptr;

  template <typename Params>
  __device__ __host__ R2FiCisBiasMask(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = math::log2e;
    r2_mask_ptr = params.r2_mask + params.r2_mask_indptr[batch_idx];
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float b = 0.f;
    if constexpr (r2fm_has_paged_kv<Params>::value) {
      if (kv_idx < kv_len) {
        uint32_t pg, e;
        params.paged_kv.page_size.divmod(kv_idx, pg, e);
        const int64_t page = static_cast<int64_t>(params.paged_kv.indices[params.paged_kv.indptr[batch_idx] + pg]);
        b = __bfloat162float(params.cis_bias[page * params.cis_page_mul +
                                             static_cast<int64_t>(e) * params.cis_entry_stride + params.cis_head]);
      }
    }
    return (logits + b) * static_cast<float>(params.cis_sm_scale);
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    if (qo_idx >= qo_len || kv_idx >= kv_len) {
      mask = false;
    } else {
      const uint64_t offset = static_cast<uint64_t>(qo_idx) * kv_len + kv_idx;
      mask = ((r2_mask_ptr[offset / 8] >> (offset % 8)) & 1);
    }
    return mask;
  })
};
"""
VARIANT_TENSORS_FP = (["cis_bias", "r2_mask", "r2_mask_indptr"], ["nv_bfloat16", "uint8_t", "int32_t"])

# REGISTERED before any run (docs/R2_FIATTN_PLAN.md section 6 carries the same table; the lead stamps it).
# Bands are [central, low, high]. Times in ms (CUDA events, median), per attention call of one layer.
PRED = {
    "registered": "REGISTERED <stamp-by-lead>",
    "tol_c_abs": 1.0e-3,
    "err_vs_fp64": {"P": [1.2e-4, 5e-5, 4e-4], "F0": [1.5e-4, 5e-5, 5e-4], "F8": [3e-4, 1e-4, 1e-3]},
    "p3_ms": {"8x1": [0.06, 0.03, 0.15], "8x4": [0.15, 0.07, 0.3], "32x1": [0.15, 0.07, 0.3],
              "32x4": [0.4, 0.2, 0.8]},
    "ratio_vs_p3": {"F8": {"8x1": [1.3, 0.7, 2.5], "8x4": [1.1, 0.6, 2.0], "32x1": [1.1, 0.6, 2.0],
                           "32x4": [0.95, 0.6, 1.6]},
                    "F0": {"8x1": [3.0, 1.5, 8.0], "8x4": [2.5, 1.2, 6.0], "32x1": [2.5, 1.2, 6.0],
                           "32x4": [1.8, 1.0, 4.0]},
                    "S5": {"8x4": [0.9, 0.6, 1.3], "32x4": [0.85, 0.5, 1.3]}},
    "plan_ms_per_tick_host_indptr": [0.2, 0.05, 1.0],
    "plan_ms_per_tick_device_indptr": [0.3, 0.08, 1.5],
    "pack_ms_per_layer": [0.05, 0.01, 0.3],
    "jit_build_s": [240, 60, 900],
    "invariance": {"P": "PASS", "F8": "PASS (p 0.7; the split_kv switch I4 is the main risk, p 0.25)",
                   "F0": "PASS (p 0.85)", "FH": "DETECTED with the precondition met (p 0.9)"},
    "bits_vs_p3": {"F8": "most rows differ (> 0.9 of rows), max |d| in [1e-4, 1e-3]",
                   "F0": "most rows differ (> 0.5 of rows), max |d| <= 5e-4"},
    "dram_read_over_ideal": {"1": {"P": [1.0, 0.9, 1.2], "F8": [1.0, 0.9, 1.25], "F0": [1.0, 0.9, 1.25]},
                             "4": {"P": [0.45, 0.28, 0.9], "F8": [0.45, 0.28, 0.9], "F0": [0.45, 0.28, 0.9],
                                   "S5": [0.33, 0.28, 0.7]}},
}

# [fia3] REGISTERED before the run by the LEAD (the stamp is the lead's; docs/NOTES_fia3.md section 6 carries the
# same json; test FC15 keeps them equal). Bands [central, low, high]. Sources: the plan (Task 0.3) and the survey
# (section 7, "Predictions (MODEL, to register)"); the survey gives intervals only for the byte ratios, so their
# central value is the interval midpoint (a builder choice).
PRED3 = {
    "registered": "REGISTERED 2026-10-27T16:35:45Z (lead stamp, before any run)",
    "source": "PLAN_speckv_on_cached_nosa.md Task 0.3; SURVEY_shared_kv_kernel.md section 7",
    "fp0_over_p_device": {"216x4": [0.75, 0.45, 1.3], "216x4s1": [0.75, 0.45, 1.3]},
    "fp0_dram_read_over_union": [1.05, 1.0, 1.1],
    "fp0_l2sm_read_over_union": [1.15, 1.0, 1.3],
    "p_l2sm_read_over_sum": [0.975, 0.9, 1.05],
    "p_device_ms_per_layer_216x4_central": 1.37,
    "thresholds": {"fp0_over_p_max": 0.8, "tick_overhead_ms_max": 1.0, "layers_per_tick": 32,
                   "hbm_bound_l2sm_drop": 3.0},
    "decision": "ADOPT option 1 (FP0) if, at B 216 x 4, FP0 device time <= 0.8 x P device time AND plan + mask "
                "build <= 1 ms per tick amortized (one plan per tick, 32 layer builds) AND the FP0 MUST gates pass. "
                "Else read the ncu split: mask-bound -> a compact mask (FPc); plan-bound -> survey option 2 (a fork of "
                "vLLM's Triton unified attention) with this measured blocker written down as the skill's rung-4 row. "
                "If L2->SM bytes fall 3x (P / FP0) but the time does not move (FP0 / P > 0.8): HBM-bound, stop at the "
                "15 % byte bound.",
}

# [fia4] REGISTERED block for the LEAD (the stamp is the lead's; docs/NOTES_fia4.md section 6 carries the same json;
# test FV12 keeps them equal). Bands [central, low, high]. Source: the lead's T0.3b brief (FV / P device central 0.7,
# band [0.4, 1.1]; the rule).
PRED4 = {
    "registered": "REGISTERED 2026-10-27T18:58:34Z (lead stamp, before any run)",
    "source": "lead's T0.3b brief 2026-10-27 (PLAN_speckv_on_cached_nosa.md Task 0.3 option 2; SCORING_fia3.md)",
    "fv_over_p_device": {"216x4": [0.7, 0.4, 1.1], "216x4s1": [0.7, 0.4, 1.1]},
    "thresholds": {"fv_over_p_max": 0.8, "union_build_ms_per_tick_max": 1.0, "layers_per_tick": 32},
    "decision": "ADOPT FV if, at B 216 x 4, FV device time <= 0.8 x P device time (ncu) AND the union build <= 1 ms "
                "per tick (32 layers; FV builds in the kernel: 32 x (FV - FVT) device time) AND the FV MUST gates "
                "pass. Else NOT MET, with the ncu split (registers, spills, achieved occupancy, DRAM and L2->SM "
                "bytes) written down.",
}

# [fia5] REGISTERED block for the LEAD (the stamp is the lead's; docs/NOTES_fia5.md section 6 carries the same json;
# test F5_7 keeps them equal). Bands [central, low, high]: the lead's suggestion in the fia5 brief. The rule is PRED4's.
PRED5 = {
    "registered": "REGISTERED 2026-10-27T21:11:01Z (lead stamp, before any run)",
    "source": "lead's fia5 brief 2026-10-27 (SCORING_fia4.md sections 5-6: let Triton pipeline K / V inside the fork)",
    "fv_over_p_device": {"216x4": [0.85, 0.6, 1.05], "216x4s1": [0.85, 0.6, 1.05]},
    "best_swept_over_p_device": {"216x4": [0.85, 0.6, 1.05], "216x4s1": [0.85, 0.6, 1.05]},
    "thresholds": {"fv_over_p_max": 0.8, "union_build_ms_per_tick_max": 1.0, "layers_per_tick": 32},
    "decision": "ADOPT a launch configuration C of the fork (the default FV, or the cell's best swept configuration: "
                "the lowest CUDA-event median among the swept arms whose MUST gates pass) if, at B 216 x 4, C's device "
                "time <= 0.8 x P device time (ncu) AND the union build <= 1 ms per tick (32 layers; 32 x (FV - FVT) "
                "device time at the default launch) AND C's MUST gates pass. When both qualify, the lower device time. "
                "Else NOT MET, with the ncu split and the compile witness (registers, spills, shared memory, K / V "
                "buffer depth) written down.",
}


def log(msg):
    print("[r2_fiattn %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail, status=None):
    g = dict(name=name, passed=bool(passed), must=bool(must), detail=detail)
    if status is not None:
        g["status"] = status
    return g


def load_by_path(path, name):
    """A module of another branch, copied into the step record by the step script (pinned by blob id)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


def band_text(value, b):
    c, lo, hi = b
    if value is None:
        return "n/a (band [%s, %s], central %s)" % (lo, hi, c)
    return "%.4g %s [%s, %s] (central %s)" % (value, "IN" if lo <= value <= hi else "OUT", lo, hi, c)


# ------------------------------------------------------------------ geometry, cells, cases
def geometry_from_config(cfg: dict) -> dict:
    """q heads, KV heads, head_dim from a HF config dict (NOSA-8B: 32, 2, 128)."""
    hq, hkv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
    d = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // hq)
    if hq % hkv != 0 or d < 16 or d & (d - 1):
        raise ValueError("geometry Hq %d, Hkv %d, D %d" % (hq, hkv, d))
    return dict(Hq=hq, Hkv=hkv, D=d, G=hq // hkv)


def parse_cells(text):
    """'8x1,8x4,216x4s1' -> [(8, 1), (8, 4), (216, 4, 's1')]; B in 1 .. MAX_B, rows per request in 1 .. 4; no repeats.
    [fia3] the suffix s1 = the lists of the recorded S1 tick tables (--s1-tables); none = the seeded generator."""
    out = []
    for part in [p for p in str(text).split(",") if p]:
        m = re.fullmatch(r"(\d+)x(\d+)(s1)?", part.strip())
        if m is None:
            raise ValueError("cell %r is not BxR or BxRs1" % part)
        b, r = int(m.group(1)), int(m.group(2))
        c = (b, r) if m.group(3) is None else (b, r, "s1")
        if not (1 <= b <= MAX_B and 1 <= r <= 4) or c in out:
            raise ValueError("cell %r (B 1 .. %d, rows per request 1 .. 4, no repeats)" % (part, MAX_B))
        out.append(c)
    if not out:
        raise ValueError("no cells")
    return out


def cell_parts(c):
    """[fia3] (B, rows per request, source): source 'gen' (the seeded generator) or 's1' (the recorded tables)."""
    return int(c[0]), int(c[1]), (c[2] if len(c) > 2 else "gen")


def ncu_key(c):
    """[fia3] the ncu order: generator cells first, then larger B, then more rows (fia2's 32x4, 32x1, 8x4, 8x1)."""
    b, r, src = cell_parts(c)
    return (src != "gen", -b, -r)


def parse_arms(text):
    """[fia5] also a swept FV arm's canonical name (the ncu child gets the best one by name)."""
    arms = [a for a in str(text).split(",") if a]
    if not arms or any(a not in ARMS and fv_sweep_launch(a) is None for a in arms) or len(set(arms)) != len(arms):
        raise ValueError("arms %r (subset of %s, or FVW<w>S<s>T<t>B<b>)" % (text, ARMS))
    return arms


def fv_sweep_name(warps, stages, tile=PAGE, bias_bits=FV_BIAS_BITS):
    """[fia5] The canonical arm name of one launch configuration of the fork."""
    return "FVW%dS%dT%dB%d" % (int(warps), int(stages), int(tile), int(bias_bits))


def fv_sweep_launch(name):
    """[fia5] A swept arm's launch configuration from its canonical name, or None (not a swept arm, or out of range):
    dict(num_warps, num_stages, tile_size, bias_bits)."""
    m = FV_SWEEP_NAME.fullmatch(str(name))
    if m is None:
        return None
    w, s, t, bb = (int(x) for x in m.groups())
    if w not in FV_SWEEP_WARPS or s not in FV_SWEEP_STAGES or t not in FV_SWEEP_TILES or PAGE % t:
        return None
    if name != fv_sweep_name(w, s, t, bb):                              # no leading zeros: one name per configuration
        return None
    return dict(num_warps=w, num_stages=s, tile_size=t, bias_bits=bb)


def parse_fv_sweep(text):
    """[fia5] '--fv-sweep w4s2,w8s3t32,w4s3b16' -> the canonical arm names ['FVW4S2T64B32', 'FVW8S3T32B32',
    'FVW4S3T64B16']: num_warps w, num_stages s, the token tile t (default PAGE: one block per tile), the CIS bias
    passed as float b bits (default 32; 16 = the store's bf16). Empty -> []. Refused: an unknown item, a value out of
    range (FV_SWEEP_WARPS, FV_SWEEP_STAGES, FV_SWEEP_TILES that divide PAGE), a repeated configuration."""
    out = []
    for item in [x.strip() for x in str(text or "").split(",") if x.strip()]:
        m = FV_SWEEP_ITEM.fullmatch(item)
        if m is None:
            raise ValueError("--fv-sweep item %r is not w<warps>s<stages>[t<tile>][b<16|32>]" % item)
        w, s = int(m.group(1)), int(m.group(2))
        t = int(m.group(3)) if m.group(3) else PAGE
        bb = int(m.group(4)) if m.group(4) else FV_BIAS_BITS
        name = fv_sweep_name(w, s, t, bb)
        if fv_sweep_launch(name) is None:
            raise ValueError("--fv-sweep item %r: warps in %s, stages in %s, tile in %s" % (
                item, FV_SWEEP_WARPS, FV_SWEEP_STAGES, FV_SWEEP_TILES))
        if name in out:
            raise ValueError("--fv-sweep item %r repeats %s" % (item, name))
        out.append(name)
    return out


def is_fv_arm(name):
    """[fia5] An arm of the fork: FV, FVT, FVN or a swept configuration."""
    return name in FV_ARMS or fv_sweep_launch(name) is not None


def cell_tag(b, r, src="gen"):
    return "%dx%d%s" % (b, r, "s1" if src == "s1" else "")


def cell_seed(seed, b, r):
    return int(seed) * 1000 + int(b) * 10 + int(r)


def spread(n, k):
    """k row indices spread over 0 .. n - 1 (the first and the last included)."""
    if n <= k:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


def _draw_scattered(g):
    pool = torch.arange(1, TAIL_BLOCK - N_RECENT)
    w = (TAIL_BLOCK - pool).double().rsqrt()                          # more recent -> more likely (locality)
    n_sc = N_SEL - 2 - N_RECENT
    return pool[torch.multinomial(w, n_sc, replacement=False, generator=g)]


def _swap(g, sc):
    pool = torch.arange(1, TAIL_BLOCK - N_RECENT)
    free = pool[~torch.isin(pool, sc)]
    out = sc.clone()
    pos = torch.randperm(sc.numel(), generator=g)[:N_SWAP]
    out[pos] = free[torch.randperm(free.numel(), generator=g)[:N_SWAP]]
    return out


def _full_list(sc):
    lst = torch.cat([torch.tensor([0]), torch.arange(TAIL_BLOCK - N_RECENT, TAIL_BLOCK), sc, torch.tensor([TAIL_BLOCK])])
    return torch.sort(lst).values


def _short_list(c):
    return torch.cat([torch.tensor([0]), torch.arange(TAIL_BLOCK - (c - 2), TAIL_BLOCK), torch.tensor([TAIL_BLOCK])])


def make_case(geom, b, rpr, seed, with_data=True, data_device="cpu"):
    """The cell's fixed, seeded case on the CPU (module docstring). The lists are drawn BEFORE the data, so
    with_data=False (the ncu summary) gives the same lists without the tensors. Storage entries 0 .. B - 1 are the
    requests, entry B holds the hazard rows.
    [fia3] "altreq": every ODD REQUEST gets new lists and a new q (the FP arms' FI3: their unit is the request), drawn
    from a SEPARATE generator after everything else, so the lists and data of fia1 / fia2 stay bitwise the same.
    data_device "cuda": K / V / bias come from a seeded CUDA generator (the 216-request cells: about 3.6 GB)."""
    hq, hkv, d = geom["Hq"], geom["Hkv"], geom["D"]
    if not (1 <= rpr <= 4 and b >= 1):
        raise ValueError("B %r, rows per request %r" % (b, rpr))
    cs = cell_seed(seed, b, rpr)
    g = torch.Generator().manual_seed(cs)
    r_, n_, s_, p_ = b * rpr, b + 1, S_SLOTS, N_SEL
    slot_of_block = torch.stack([torch.randperm(s_, generator=g) for _ in range(n_)])        # (N, S)
    t0 = torch.randint(0, PAGE - rpr + 1, (b,), generator=g)
    seq = torch.arange(b).repeat_interleave(rpr)
    last = t0.repeat_interleave(rpr) + torch.arange(rpr).repeat(b) + 1                       # 1 .. 64
    blocks = torch.full((r_, hkv, p_), -1, dtype=torch.int64)
    for n in range(b):
        for h in range(hkv):
            sc0 = _draw_scattered(g)
            for u in range(rpr):
                blocks[n * rpr + u, h] = _full_list(sc0 if u == 0 else _swap(g, sc0))
    alt_blocks = blocks.clone()
    for r in range(1, r_, 2):
        for h in range(hkv):
            alt_blocks[r, h] = _full_list(_draw_scattered(g))
    sblocks = torch.full((len(SHORT_COUNTS), hkv, p_), -1, dtype=torch.int64)
    for i, c in enumerate(SHORT_COUNTS):
        sblocks[i, :, :c] = _short_list(c)

    def rows(blk, seq_idx, last_len):
        n_idx = seq_idx.view(-1, 1, 1).expand_as(blk)
        slots = torch.where(blk >= 0, slot_of_block[n_idx, blk.clamp(min=0)], torch.full_like(blk, -1))
        return dict(seq_idx=seq_idx.to(torch.int32).contiguous(), page_key=blk.to(torch.int32).contiguous(),
                    page_table=slots.to(torch.int32).contiguous(),
                    num_pages=(blk >= 0).sum(-1).to(torch.int32).contiguous(),
                    last_len=last_len.to(torch.int32).contiguous())
    full = rows(blocks, seq, last)
    alt = rows(alt_blocks, seq, last)
    short = rows(sblocks, torch.full((len(SHORT_COUNTS),), n_ - 1), torch.full((len(SHORT_COUNTS),), SHORT_LAST))
    g2 = torch.Generator().manual_seed(cs + ALTREQ_SEED_OFFSET)             # [fia3] the altreq lists and q only
    areq_blocks = blocks.clone()
    for n in range(1, b, 2):
        for h in range(hkv):
            sc0 = _draw_scattered(g2)
            for u in range(rpr):
                areq_blocks[n * rpr + u, h] = _full_list(sc0 if u == 0 else _swap(g2, sc0))
    altreq = rows(areq_blocks, seq, last)
    case = dict(geom=dict(geom), B=b, rpr=rpr, R=r_, N=n_, S=s_, P=p_, seed=cs, slot_of_block=slot_of_block,
                full=full, alt=alt, short=short, altreq=altreq, k=None, v=None, bias=None, src="gen",
                data_device=data_device)
    if with_data:
        q = torch.randn(r_, hq, d, generator=g).to(torch.bfloat16)
        qa = torch.randn(r_, hq, d, generator=g).to(torch.bfloat16)
        full["q"] = q
        odd = (torch.arange(r_) % 2 == 1).view(-1, 1, 1)
        alt["q"] = torch.where(odd, qa, q).contiguous()
        short["q"] = torch.randn(len(SHORT_COUNTS), hq, d, generator=g).to(torch.bfloat16)
        _store_data(case, g, cs, n_, s_, hkv, d, data_device)
        odd_req = ((torch.arange(r_) // rpr) % 2 == 1).view(-1, 1, 1)
        altreq["q"] = torch.where(odd_req, torch.randn(r_, hq, d, generator=g2).to(torch.bfloat16), q).contiguous()
    return case


ALTREQ_SEED_OFFSET = 1000003                # [fia3] the altreq generator (a seed no cell_seed reaches in practice)
S1_SEED_OFFSET = 500000                     # [fia3] the s1 cells' data generator


def _store_data(case, g, cs, n_, s_, hkv, d, data_device):
    """K / V / bias of the store. CPU: from g, in fia1 / fia2's order (the same bits). [fia3] another device: from a
    CUDA generator seeded with the cell seed (one draw per tensor, the same order)."""
    if str(data_device) == "cpu":
        case["k"] = torch.randn(n_, s_ * PAGE, hkv, d, generator=g).to(torch.bfloat16)
        case["v"] = torch.randn(n_, s_ * PAGE, hkv, d, generator=g).to(torch.bfloat16)
        case["bias"] = ((torch.rand(n_, s_ * PAGE, hkv, generator=g) * 2 - 1) * BIAS_RANGE).to(torch.bfloat16)
        return
    gd = torch.Generator(device=data_device).manual_seed(cs)
    case["k"] = torch.randn(n_, s_ * PAGE, hkv, d, generator=gd, device=data_device).to(torch.bfloat16)
    case["v"] = torch.randn(n_, s_ * PAGE, hkv, d, generator=gd, device=data_device).to(torch.bfloat16)
    case["bias"] = ((torch.rand(n_, s_ * PAGE, hkv, generator=gd, device=data_device) * 2 - 1)
                    * BIAS_RANGE).to(torch.bfloat16)


# ------------------------------------------------------------------ [fia3] the recorded S1 tick tables
S1_FIELDS = ("seq_idx", "page_table", "num_pages", "last_len", "page_key")


def load_s1_tables(path):
    """The recorded inputs of the S1 tick's first keyed calls with rows_max 4 (sharedkv3 SharedKVProbe.report_tables,
    saved by r2_tick_gate.py:128-134): a list of dicts with seq_idx (R,), page_table / page_key (R, Hkv, P), num_pages
    (R, Hkv), last_len (R,). Loaded as weights only (dicts, lists, str, int, tensors)."""
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                                                   # a torch without weights_only
        t = torch.load(path, map_location="cpu")
    if not isinstance(t, list) or not t:
        raise ValueError("%s: not a non-empty list of recorded calls" % path)
    for i, d in enumerate(t):
        miss = [f for f in S1_FIELDS if not (isinstance(d, dict) and torch.is_tensor(d.get(f)))]
        if miss:
            raise ValueError("%s: call %d lacks %s" % (path, i, miss))
    return t


def s1_groups(tables, rpr, hkv):
    """The recorded REQUESTS with exactly rpr rows (rows with one seq_idx in one call, as build_groups groups them),
    pooled over the calls in call order and ascending seq_idx; each request's rows in POSITION order (tail block x 64 +
    last_len; the tail = the last listed page of KV head 0)."""
    out = []
    for c, t in enumerate(tables):
        pt = t["page_table"].to(torch.int64)
        pk = t["page_key"].to(torch.int64)
        npg = t["num_pages"].to(torch.int64)
        ll = t["last_len"].to(torch.int64)
        seq = t["seq_idx"].to(torch.int64)
        r_, h_, p_ = pt.shape
        if h_ != hkv or tuple(pk.shape) != (r_, h_, p_) or tuple(npg.shape) != (r_, h_) or tuple(ll.shape) != (r_,) \
                or tuple(seq.shape) != (r_,):
            raise ValueError("recorded call %d: shapes seq %s table %s key %s num_pages %s last_len %s (Hkv %d)"
                             % (c, tuple(seq.shape), tuple(pt.shape), tuple(pk.shape), tuple(npg.shape),
                                tuple(ll.shape), hkv))
        seql = seq.tolist()
        for s in sorted(set(seql)):
            rows_ = [i for i in range(r_) if seql[i] == s]
            if len(rows_) != rpr:
                continue
            sel = torch.tensor(rows_)
            if bool(((ll[sel] < 1) | (ll[sel] > PAGE)).any()) or bool((npg[sel] < 1).any()):
                continue                                                 # not a valid extent: not a request here
            rows_.sort(key=lambda i: int(pk[i, 0, max(int(npg[i, 0]) - 1, 0)]) * PAGE + int(ll[i]))
            idx = torch.tensor(rows_)
            out.append(dict(call=c, seq=s, rows=rows_, page_key=pk[idx], page_table=pt[idx], num_pages=npg[idx],
                            last_len=ll[idx]))
    return out


def make_case_s1(geom, b, rpr, seed, groups, with_data=True, data_device="cpu"):
    """[fia3] A case with make_case's fields whose LISTS are recorded: request n takes recorded group n mod G (its rows
    in position order, its (block id, slot) lists and extents) on its own storage entry n; q / K / V / bias are
    synthetic (seeded). alt: every odd row takes the same row of the next group; altreq: every odd request takes the
    next group; both get a new q. The hazard rows (5, 8, 9 pages) sit on entry B with a seeded block -> slot map.
    S = max(256, the largest recorded slot + 1)."""
    hq, hkv, d = geom["Hq"], geom["Hkv"], geom["D"]
    if not groups:
        raise ValueError("no recorded request with %d rows" % rpr)
    p_ = int(groups[0]["page_table"].shape[-1])
    for gr in groups:
        if tuple(gr["page_table"].shape) != (rpr, hkv, p_):
            raise ValueError("recorded group shape %s != (%d, %d, %d)" % (tuple(gr["page_table"].shape), rpr, hkv, p_))
    cs = cell_seed(seed, b, rpr) + S1_SEED_OFFSET
    g = torch.Generator().manual_seed(cs)
    G = len(groups)
    r_, n_ = b * rpr, b + 1
    pick = [n % G for n in range(b)]
    nxt = [(i + 1) % G for i in pick]

    def stack(sel_rows):
        """sel_rows: per row r, (group index, row u in the group)."""
        f = {}
        for key in ("page_key", "page_table", "num_pages", "last_len"):
            f[key] = torch.stack([groups[gi][key][u] for gi, u in sel_rows])
        return f
    base_rows = [(pick[n], u) for n in range(b) for u in range(rpr)]
    alt_rows = [(nxt[n], u) if (n * rpr + u) % 2 == 1 else (pick[n], u) for n in range(b) for u in range(rpr)]
    areq_rows = [(nxt[n], u) if n % 2 == 1 else (pick[n], u) for n in range(b) for u in range(rpr)]
    lists = [stack(x) for x in (base_rows, alt_rows, areq_rows)]
    mx = 0
    for f in lists:
        valid = torch.arange(p_).view(1, 1, p_) < f["num_pages"].unsqueeze(-1)
        mx = max(mx, int(torch.where(valid, f["page_table"], torch.full_like(f["page_table"], -1)).max()))
    s_ = max(S_SLOTS, mx + 1)
    slot_of_block = torch.stack([torch.randperm(s_, generator=g) for _ in range(n_)])
    seq = torch.arange(b).repeat_interleave(rpr)

    def rows(f, seq_idx):
        return dict(seq_idx=seq_idx.to(torch.int32).contiguous(), page_key=f["page_key"].to(torch.int32).contiguous(),
                    page_table=f["page_table"].to(torch.int32).contiguous(),
                    num_pages=f["num_pages"].to(torch.int32).contiguous(),
                    last_len=f["last_len"].to(torch.int32).contiguous())
    full, alt, altreq = (rows(f, seq) for f in lists)
    sblocks = torch.full((len(SHORT_COUNTS), hkv, p_), -1, dtype=torch.int64)
    for i, c in enumerate(SHORT_COUNTS):
        sblocks[i, :, :c] = _short_list(c)
    sslots = torch.where(sblocks >= 0, slot_of_block[n_ - 1][sblocks.clamp(min=0)], torch.full_like(sblocks, -1))
    short = dict(seq_idx=torch.full((len(SHORT_COUNTS),), n_ - 1, dtype=torch.int32),
                 page_key=sblocks.to(torch.int32).contiguous(), page_table=sslots.to(torch.int32).contiguous(),
                 num_pages=(sblocks >= 0).sum(-1).to(torch.int32).contiguous(),
                 last_len=torch.full((len(SHORT_COUNTS),), SHORT_LAST, dtype=torch.int32))
    case = dict(geom=dict(geom), B=b, rpr=rpr, R=r_, N=n_, S=s_, P=p_, seed=cs, slot_of_block=slot_of_block,
                full=full, alt=alt, short=short, altreq=altreq, k=None, v=None, bias=None, src="s1",
                data_device=data_device, s1_groups=G, s1_calls=sorted({gr["call"] for gr in groups}))
    if with_data:
        q = torch.randn(r_, hq, d, generator=g).to(torch.bfloat16)
        qa = torch.randn(r_, hq, d, generator=g).to(torch.bfloat16)
        full["q"] = q
        odd = (torch.arange(r_) % 2 == 1).view(-1, 1, 1)
        alt["q"] = torch.where(odd, qa, q).contiguous()
        odd_req = ((torch.arange(r_) // rpr) % 2 == 1).view(-1, 1, 1)
        altreq["q"] = torch.where(odd_req, qa, q).contiguous()
        short["q"] = torch.randn(len(SHORT_COUNTS), hq, d, generator=g).to(torch.bfloat16)
        _store_data(case, g, cs, n_, s_, hkv, d, data_device)
    return case


def case_to(case, device):
    out = dict(case)
    for k in ("full", "alt", "short", "altreq"):
        if k in case:
            out[k] = {f: t.to(device) for f, t in case[k].items()}
    for f in CACHE_FIELDS:
        out[f] = None if case[f] is None else case[f].to(device)
    return out


def batch_of(case, which, caches=None):
    """A batch: the row fields + the caches (k, v, bias). which: 'full', 'alt', 'short' or 'hazard' (full + short)."""
    if which == "hazard":
        rows = {f: torch.cat([case["full"][f], case["short"][f]], 0).contiguous() for f in case["full"]}
    else:
        rows = dict(case[which])
    c = caches if caches is not None else dict(k=case["k"], v=case["v"], bias=case["bias"])
    rows.update({f: c[f] for f in CACHE_FIELDS})
    return rows


def select(b, idx):
    """Rows idx (a list) of batch b, in that order (contiguous copies); the caches are shared."""
    t = torch.as_tensor(list(idx), dtype=torch.long, device=b["page_table"].device)
    out = {f: b[f].index_select(0, t).contiguous() for f in ROW_FIELDS if f in b}
    out.update({f: b[f] for f in CACHE_FIELDS})
    return out


def check_store(*caches):
    """The flat addressing (option A / B views, the bias offsets) assumes contiguous caches."""
    for c in caches:
        if c is None or not c.is_contiguous():
            raise ValueError("the K / V / bias caches must be contiguous")


def reach_map(b, n_entries, s_slots, hkv):
    """(N, S, Hkv) int64: the number of leading rows of each (slot, head) page that some row of b reads (64 for a
    page listed before a row's last page, last_len for its last page; 0 = not listed)."""
    pt = b["page_table"].to("cpu", torch.int64)
    npg = b["num_pages"].to("cpu", torch.int64)
    si = b["seq_idx"].to("cpu", torch.int64)
    ll = b["last_len"].to("cpu", torch.int64)
    r_, h_, p_ = pt.shape
    j = torch.arange(p_).view(1, 1, -1)
    valid = j < npg.unsqueeze(-1)
    is_last = j == (npg - 1).unsqueeze(-1)
    val = torch.where(is_last, ll.view(-1, 1, 1).expand_as(pt), torch.full_like(pt, PAGE))
    n_idx = si.view(-1, 1, 1).expand_as(pt)
    h_idx = torch.arange(h_).view(1, -1, 1).expand_as(pt)
    flat = (n_idx * s_slots + pt.clamp(min=0)) * hkv + h_idx
    reach = torch.zeros(n_entries * s_slots * hkv, dtype=torch.int64)
    reach.scatter_reduce_(0, flat[valid], val[valid], reduce="amax")
    return reach.view(n_entries, s_slots, hkv)


def poison_caches(case, b):
    """K, V, bias copies where every row no row of b reads is NaN, per (slot, head). Returns (caches dict, stats)."""
    n_, s_, hkv, d = case["N"], case["S"], case["geom"]["Hkv"], case["geom"]["D"]
    reach = reach_map(b, n_, s_, hkv)
    mask = torch.arange(PAGE).view(1, 1, -1, 1) >= reach.unsqueeze(2)                     # (N, S, 64, Hkv)
    dev = b["k"].device
    m = mask.to(dev)
    out = {}
    for f in ("k", "v"):
        t = b[f].clone()
        t.view(n_, s_, PAGE, hkv, d)[m] = float("nan")
        out[f] = t
    t = b["bias"].clone()
    t.view(n_, s_, PAGE, hkv)[m] = float("nan")
    out["bias"] = t
    stats = dict(poisoned_rows=int(mask.sum()), total_rows=int(mask.numel()),
                 unlisted_pages=int((reach == 0).sum()), pages=int(reach.numel()))
    return out, stats


def reference_fp64(b, rows, geom, zero_bias=False, scale=None):
    """The float64 attention of rows (this module's own code): per (row, KV head) the listed pages in list order,
    the last one cut at last_len; softmax((q.k + bias) * scale) . v with one softmax over all listed rows."""
    g, d, hkv = geom["G"], geom["D"], geom["Hkv"]
    sc = d ** -0.5 if scale is None else float(scale)
    k, v, bias = b["k"], b["v"], b["bias"]
    dev = k.device
    pt = b["page_table"].to("cpu", torch.int64)
    npg = b["num_pages"].to("cpu", torch.int64)
    si = b["seq_idx"].to("cpu", torch.int64)
    ll = b["last_len"].to("cpu", torch.int64)
    out = torch.empty((len(rows), geom["Hq"], d), dtype=torch.float64, device=dev)
    for i, r in enumerate(rows):
        n, last = int(si[r]), int(ll[r])
        for h in range(hkv):
            c = int(npg[r, h])
            slots = pt[r, h, :c]
            tok = (slots.view(-1, 1) * PAGE + torch.arange(PAGE)).reshape(-1)[:(c - 1) * PAGE + last].to(dev)
            kk = k[n, tok, h].double()
            vv = v[n, tok, h].double()
            bb = torch.zeros(tok.numel(), dtype=torch.float64, device=dev) if zero_bias else bias[n, tok, h].double()
            qq = b["q"][r, h * g:(h + 1) * g].double()
            w = torch.softmax((qq @ kk.t() + bb.unsqueeze(0)) * sc, dim=1)
            out[i, h * g:(h + 1) * g] = w @ vv
    return out


def ideal_bytes(b, geom):
    """Bytes a row-private kernel must read: per (row, head) its listed token rows x (K + V + bias), bf16."""
    npg = b["num_pages"].to("cpu", torch.int64)
    ll = b["last_len"].to("cpu", torch.int64)
    tok = ((npg - 1) * PAGE + ll.view(-1, 1)).sum()
    return int(tok) * (4 * geom["D"] + 2)


def union_bytes(b, geom, n_entries, s_slots):
    """Bytes of the request union: every (slot, head) page row that some row reads, once."""
    return int(reach_map(b, n_entries, s_slots, geom["Hkv"]).sum()) * (4 * geom["D"] + 2)


# ------------------------------------------------------------------ FlashInfer packing and addressing
def pack_all_heads(b, s_slots):
    """[fix after fia1] FlashInfer CSR for EVERY KV head in one pass (the per-head version cost about 12 small
    launches per head: 0.31 ms per layer, launch-bound). Per head h: (indptr (R + 1), indices (an R * P buffer; the
    first indptr[R] entries are the pages: entry * S + slot in the row's canonical order), last_page_len (R)).
    Ragged counts allowed. Device ops only (cumsum + scatter, no host read)."""
    pt = b["page_table"]
    dev = pt.device
    r_, hkv, p_ = pt.shape
    npg = b["num_pages"].to(torch.int64).t()                                       # (Hkv, R)
    indptr = torch.zeros(hkv, r_ + 1, dtype=torch.int64, device=dev)
    indptr[:, 1:] = torch.cumsum(npg, 1)
    j = torch.arange(p_, device=dev).view(1, 1, -1)
    valid = j < npg.unsqueeze(-1)                                                 # (Hkv, R, P)
    gid = b["seq_idx"].to(torch.int64).view(1, -1, 1) * s_slots + pt.permute(1, 0, 2).to(torch.int64)
    row = r_ * p_ + 1                                                             # per head: R * P + a dump slot
    base = (torch.arange(hkv, device=dev) * row).view(-1, 1, 1)
    pos = base + torch.where(valid, indptr[:, :-1].unsqueeze(-1) + j, torch.full_like(gid, r_ * p_))
    buf = torch.zeros(hkv * row, dtype=torch.int32, device=dev)
    buf.scatter_(0, pos.reshape(-1), gid.to(torch.int32).reshape(-1))             # invalid entries -> the dump slot
    bv = buf.view(hkv, row)
    ip = indptr.to(torch.int32)
    last = b["last_len"].to(torch.int32).contiguous()
    return [(ip[h], bv[h, :r_ * p_], last) for h in range(hkv)]


def pack_option_a(b, s_slots, h):
    """KV head h of pack_all_heads (the tests' per-head view)."""
    return pack_all_heads(b, s_slots)[h]


def pack_fixed(b, s_slots, out):
    """[fix after fia1] The constant-count path: every (row, head) lists exactly P pages (NOSA's fixed top-k; the
    caller's structural guarantee, checked from host counts in FIArm.prepare). Then indptr is r * P for all layers and
    a layer's indices are ONE elementwise expression, written IN PLACE into out (Hkv, R * P) int32: the buffer a plan
    already holds, so one plan per tick serves every layer with these counts (plan depends on indptr / last_page_len
    only; the kernel reads the indices buffer at run time)."""
    pt = b["page_table"]
    r_, hkv, p_ = pt.shape
    torch.add(pt.permute(1, 0, 2), b["seq_idx"].view(1, -1, 1) * s_slots, out=out.view(hkv, r_, p_))
    return out


def pack_option_b(b, s_slots):
    """FlashInfer CSR for ONE call over all KV heads, requests = (row, head) row-major (request r * Hkv + h); page id
    = (entry * S + slot) * 64 * Hkv + h (the overlapping as_strided view kv_view_b)."""
    pt = b["page_table"]
    dev = pt.device
    r_, hkv, p_ = pt.shape
    nq = r_ * hkv
    npg = b["num_pages"].to(torch.int64).reshape(-1)
    indptr = torch.zeros(nq + 1, dtype=torch.int64, device=dev)
    indptr[1:] = torch.cumsum(npg, 0)
    j = torch.arange(p_, device=dev)
    valid = j.view(1, -1) < npg.view(-1, 1)
    hh = torch.arange(hkv, device=dev).view(1, -1, 1)
    gid = ((b["seq_idx"].to(torch.int64).view(-1, 1, 1) * s_slots + pt.to(torch.int64)) * (PAGE * hkv) + hh)
    gid = gid.reshape(nq, p_)
    pos = torch.where(valid, indptr[:-1].view(-1, 1) + j.view(1, -1), torch.full_like(gid, nq * p_))
    buf = torch.zeros(nq * p_ + 1, dtype=torch.int32, device=dev)
    buf.scatter_(0, pos.reshape(-1), gid.to(torch.int32).reshape(-1))
    last = b["last_len"].to(torch.int32).repeat_interleave(hkv).contiguous()
    return indptr.to(torch.int32), buf[:nq * p_], last


def kv_view_a(cache, s_slots, h):
    """Option A: the store (N, S*64, Hkv, D) as NHD pages (N*S, 64, Hkv, D), KV head h only: (N*S, 64, 1, D)."""
    check_store(cache)
    n_, sr, hkv, d = cache.shape
    return cache.view(n_ * s_slots, PAGE, hkv, d)[:, :, h:h + 1, :]


def kv_view_b(cache):
    """Option B: an OVERLAPPING view (M, 64, 1, D), strides (D, Hkv*D, D, 1): page p = g * 64 * Hkv + h (g = entry *
    S + slot) starts at the store row (g * 64, h). M is the largest count that stays inside the storage."""
    check_store(cache)
    n_, sr, hkv, d = cache.shape
    m = n_ * sr * hkv - (PAGE - 1) * hkv
    return torch.as_strided(cache, (m, PAGE, 1, d), (d, hkv * d, d, 1), cache.storage_offset())


def fi_offset(view, page, entry, feat):
    """The element offset FlashInfer's paged_kv_t computes (page.cuh get_elem_offset, NHD, head 0)."""
    return view.storage_offset() + page * view.stride(0) + entry * view.stride(1) + feat * view.stride(3)


def store_offset(cache, n, slot, entry, h, feat):
    """The element offset piece 3 reads (k_base + rows * s_ks + d: bi_paged_attention.py:229-239)."""
    return (cache.storage_offset() + n * cache.stride(0) + (slot * PAGE + entry) * cache.stride(1)
            + h * cache.stride(2) + feat * cache.stride(3))


def bias_args(option, hkv, h=0):
    """The variant's scalars (page_mul, entry_stride, head) for option 'a' (KV head h) or 'b'."""
    if option == "a":
        return PAGE * hkv, hkv, h
    if option == "b":
        return 1, hkv, 0
    raise ValueError(option)


def bias_offset(page, entry, page_mul, entry_stride, head):
    """The variant's bias offset (VARIANT_DECL), relative to the bias tensor's data pointer."""
    return page * page_mul + entry * entry_stride + head


def jit_args_for(d):
    """FlashInfer's jit_args for gen_customize_batch_prefill_module('fa2', *jit_args) (decode.py:709-716)."""
    return [VARIANT_URI % d, torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, d, d,
            list(VARIANT_TENSORS[0]), list(VARIANT_TENSORS[1]), list(VARIANT_SCALARS[0]), list(VARIANT_SCALARS[1]),
            VARIANT_NAME, VARIANT_DECL]


def version_ok(v):
    return str(v).split("+")[0] in FI_VERSIONS


def plan_info_of(w):
    """split_kv, chunk size (tokens), padded batch, CTA tile q of a planned FlashInfer wrapper: the plan vector
    (PrefillPlanInfo::ToVector, scheduler.cuh:650-666) and the kv_chunk_size int32 the plan wrote into its pinned
    host buffer (scheduler.cuh:747, offset vec[9])."""
    pi = [int(x) for x in w._plan_info]
    off = pi[9]
    raw = bytes(w._pin_memory_int_workspace_buffer[off:off + 4].tolist())
    return dict(padded_batch_size=pi[0], cta_tile_q=pi[3], split_kv=bool(pi[14]),
                chunk_tokens=int.from_bytes(raw, "little", signed=True))


def cta_tile_for(g):
    """FA2DetermineCtaTileQ (utils.cuh:349-368, sm80+) for one query row per request: packed length = the group."""
    return 16 if g <= 16 else (64 if g <= 64 else 128)


def _a16(x):
    return (int(x) + 15) // 16 * 16


def fi_float_need(counts, split, g, d, sms, fixed=FIXED_SPLIT):
    """[fix after fia1] FlashInfer's float workspace for ONE plan, by its own formula (scheduler.cuh:750-755 in
    PrefillPlan, padded = new_batch_size without CUDA graph from PrefillSplitQOKVIndptr): tmp_v = G * padded * cta *
    D * 4 B and tmp_s = G * padded * cta * 4 B, each 16-aligned. counts = the pages of each request (one query row).
    'fixed': padded = sum ceil(c / F); 'none': 0 (never split); 'heuristic': the bound max(requests, 2 * SMs) (the
    binary search keeps a split batch <= 2 * SMs / num_kv_heads, num_kv_heads 1 here; unsplit = the requests).
    fia1: option B at 32x4 = 256 requests x 8 chunks -> 256 MiB + 2 MiB, above the fixed 256 MiB (the overflow)."""
    if split == "none":
        return 0
    cta = cta_tile_for(g)
    if split == "fixed":
        nb = sum(max(1, -(-int(c) // fixed)) for c in counts)
    elif split == "heuristic":
        nb = max(len(counts), 2 * int(sms))
    else:
        raise ValueError(split)
    return _a16(g * nb * cta * d * 4) + _a16(g * nb * cta * 4)


# ------------------------------------------------------------------ [fia3] FlashInfer PREFILL over the request union
_BIG = torch.iinfo(torch.int64).max
FP_SPLIT = 8                                # FP8: fixed_split_size in pages of the union list (= piece 3's split)
A100_SMEM_PER_SM = 167936                   # cudaDevAttrMaxSharedMemoryPerMultiprocessor on sm80 (164 KiB)
LAYERS_PER_TICK = 32                        # NOSA-8B layers: the plan-reuse amortization


def cta_tile_prefill(qo_len, g):
    """FA2DetermineCtaTileQ (utils.cuh:384-403, sm80+) at the average packed length qo_len x group."""
    packed = int(qo_len) * int(g)
    return 128 if packed > 64 else (64 if packed > 16 else 16)


def kv_tile_tokens(cta_tile_q, d, smem_per_sm=A100_SMEM_PER_SM, dtype_bytes=2):
    """MODEL of the paged prefill kernel's KV tile (0.6.8 prefill.cuh:2556-2601; the 0.5.3 headers live inside the sif
    and are NOT read): NUM_WARPS_Q 4 when CTA_TILE_Q > 16 (else 1), NUM_WARPS_KV = 4 / NUM_WARPS_Q, NUM_MMA_Q 2 when
    CTA_TILE_Q > 64 (else 1), 2 CTAs per SM when two fit, NUM_MMA_KV = min(8 / NUM_MMA_Q, the shared-memory bound); the
    tile = NUM_MMA_KV x NUM_WARPS_KV x 16 tokens. A100, CTA_TILE_Q 64, D 128, bf16: 8 x 1 x 16 = 128 tokens = 2 pages.
    The dispatch reads only the device's shared memory: no public option pins it (the EXACT-mode note)."""
    nwq = 4 if cta_tile_q > 16 else 1
    nwkv = 4 // nwq
    nmq = 2 if cta_tile_q > 64 else 1
    kv_row = 2 * d * 16 * nwkv * dtype_bytes
    per_cta = cta_tile_q * d * dtype_bytes + kv_row
    ctas = 2 if smem_per_sm >= 2 * per_cta else 1
    by_smem = (smem_per_sm // ctas - cta_tile_q * d * dtype_bytes) // kv_row
    return min(8 // nmq, by_smem) * nwkv * 16


KV_TILE_PAGES = kv_tile_tokens(64, 128) // PAGE     # 2 on an A100 (MODEL)


def fp_float_need(counts, split, g, d, qo_len, fixed=FP_SPLIT):
    """FlashInfer's float workspace for ONE prefill plan (0.6.8 scheduler.cuh:771-776): tmp_v = G x padded x
    CTA_TILE_Q x D x 4 B and tmp_s = G x padded x CTA_TILE_Q x 4 B, each 16-aligned, padded = the sum of the chunks
    (counts = the union pages of each request); 0 without a split."""
    if split == "none":
        return 0
    if split != "fixed":
        raise ValueError(split)
    cta = cta_tile_prefill(qo_len, g)
    nb = sum(max(1, -(-int(c) // fixed)) for c in counts)
    return _a16(g * nb * cta * d * 4) + _a16(g * nb * cta * 4)


def fp_union(b, rpr, s_slots, ones=False):
    """Per (KV head h, request n) the UNION of the rows' pages (rows n * rpr .. n * rpr + rpr - 1, one storage entry),
    in ascending (block id, slot) order (each row's canonical list is a subsequence), and per (row, union page) the
    number of leading tokens the row sees: 64 for a listed page before the row's LAST LISTED page, last_len for that
    page (piece 3 / reference_fp64 semantics), 0 for a page the row does not list. ones=True: every row gets the pair's
    reach (the MASK WITNESS). Device ops only (sort, cumsum, scatter, gather): NO host read (CUDA-graph capturable).
    Returns int64 tensors (U = rpr x P):
      ucnt  (Hkv, B)          the union size (pages)
      upage (Hkv, B, U)       page id entry x S + slot of union page j (j < ucnt), 0 past it
      ukey  (Hkv, B, U)       block id of union page j, -1 past it
      ext   (Hkv, B, rpr, U)  tokens row u sees in union page j
      uidx  (Hkv, B, rpr, P)  union index of row u's list position i, -1 past num_pages
      klast (Hkv, B)          the last union page's extent: kv_len = (ucnt - 1) x 64 + klast"""
    pk = b["page_key"].to(torch.int64)
    pt = b["page_table"].to(torch.int64)
    npg = b["num_pages"].to(torch.int64)
    ll = b["last_len"].to(torch.int64)
    r_, hkv, p_ = pt.shape
    if r_ % rpr:
        raise ValueError("%d rows are not requests of %d rows" % (r_, rpr))
    nb = r_ // rpr
    uw = rpr * p_
    dev = pt.device
    pos = torch.arange(p_, device=dev).view(1, 1, p_)
    valid = pos < npg.unsqueeze(-1)
    is_last = pos == (npg - 1).unsqueeze(-1)
    ext_row = torch.where(is_last, ll.view(r_, 1, 1).expand(r_, hkv, p_), torch.full_like(pt, PAGE))
    ext_row = torch.where(valid, ext_row, torch.zeros_like(ext_row))
    comb = torch.where(valid, (pk << 32) | pt, torch.full_like(pt, _BIG))

    def grp(t):                                                      # (R, Hkv, P) -> (Hkv, B, rpr x P), (u, i) order
        return t.reshape(nb, rpr, hkv, p_).permute(2, 0, 1, 3).reshape(hkv, nb, uw)
    sv, si = torch.sort(grp(comb), dim=-1, stable=True)
    okv = sv != _BIG
    first = okv.clone()
    first[..., 1:] &= sv[..., 1:] != sv[..., :-1]
    uid = first.to(torch.int64).cumsum(-1) - 1
    ucnt = first.to(torch.int64).sum(-1)
    dest = torch.where(first, uid, torch.full_like(uid, uw))
    entry = b["seq_idx"].to(torch.int64).reshape(nb, rpr)[:, 0].view(1, nb, 1)
    slot = sv & 0xFFFFFFFF
    upage = torch.zeros((hkv, nb, uw + 1), dtype=torch.int64, device=dev)
    upage.scatter_(-1, dest, torch.where(first, entry * s_slots + slot, torch.zeros_like(slot)))
    ukey = torch.full((hkv, nb, uw + 1), -1, dtype=torch.int64, device=dev)
    ukey.scatter_(-1, dest, torch.where(first, sv >> 32, torch.full_like(sv, -1)))
    uidx = torch.full((hkv, nb, uw), -1, dtype=torch.int64, device=dev)
    uidx.scatter_(-1, si, torch.where(okv, uid, torch.full_like(uid, -1)))
    row = torch.div(si, p_, rounding_mode="floor")
    ext_s = grp(ext_row).gather(-1, si)
    ext = torch.zeros((hkv, nb, rpr * (uw + 1)), dtype=torch.int64, device=dev)
    ext.scatter_(-1, row * (uw + 1) + torch.where(okv, uid, torch.full_like(uid, uw)),
                 torch.where(okv, ext_s, torch.zeros_like(ext_s)))
    ext = ext.view(hkv, nb, rpr, uw + 1)[..., :uw]
    if ones:
        ext = ext.amax(dim=2, keepdim=True).expand(hkv, nb, rpr, uw).contiguous()
    lastj = (ucnt - 1).clamp(min=0).view(hkv, nb, 1, 1).expand(hkv, nb, rpr, 1)
    klast = ext.gather(-1, lastj).amax(dim=2).view(hkv, nb)
    return dict(ucnt=ucnt, upage=upage[..., :uw], ukey=ukey[..., :uw], uidx=uidx.view(hkv, nb, rpr, p_), ext=ext,
                klast=klast)


def fp_csr(u):
    """FlashInfer's CSR for the union, per KV head (one wrapper each): indptr (Hkv, B + 1) int32 = the union sizes'
    prefix sums; indices (Hkv, B x U + 1) int32 (the first indptr[B] entries are the union page ids, request by
    request; the last entry is a dump slot); last_page_len (Hkv, B) int32 = klast. Device ops only."""
    ucnt = u["ucnt"]
    hkv, nb = ucnt.shape
    uw = u["upage"].shape[-1]
    dev = ucnt.device
    indptr = torch.zeros((hkv, nb + 1), dtype=torch.int64, device=dev)
    indptr[:, 1:] = torch.cumsum(ucnt, 1)
    j = torch.arange(uw, device=dev).view(1, 1, uw)
    pos = indptr[:, :-1].unsqueeze(-1) + j
    pos = torch.where(j < ucnt.unsqueeze(-1), pos, torch.full_like(pos, nb * uw))
    ix = torch.zeros((hkv, nb * uw + 1), dtype=torch.int32, device=dev)
    ix.scatter_(1, pos.reshape(hkv, -1), u["upage"].reshape(hkv, -1).to(torch.int32))
    return indptr.to(torch.int32), ix, u["klast"].to(torch.int32)


def fp_mask_cap(hkv, nb, rpr, uw):
    """The packed mask's static capacity in bytes (every request at its widest union: no host read)."""
    return hkv * nb * ((rpr * uw * PAGE + 7) // 8)


def fp_pack(u):
    """The packed custom mask in FlashInfer's OWN layout (variants.cuh DefaultAttention; VARIANT_DECL_FP): request
    (h, n) owns a byte-aligned segment of ceil(rpr x kv_len / 8) bytes, segments in (h, n) order; row u sees token t
    iff bit u x kv_len + t of its segment is set (little bit order). The token bits are EXPANDED FROM THE BLOCK BITS:
    union page j of row u gives a 64-token word whose first ext[u, j] bits are set. Device ops only: each word's 8
    bytes are shifted by the row's bit phase (u x kv_len mod 8) into 9 bytes and ADDED into an int32 accumulator (the
    bits are disjoint, so add = or; bytes past a segment's end carry only zero bits and go to a dump byte).
    Returns (mask uint8 (cap + 1,), the last byte the dump; mask_indptr int32 (Hkv, B + 1): ABSOLUTE byte offsets of
    the segments, row h for KV head h's wrapper; kv_len int64 (Hkv, B))."""
    ext = u["ext"]
    hkv, nb, rpr, uw = ext.shape
    dev = ext.device
    kv_len = (u["ucnt"] - 1) * PAGE + u["klast"]
    seg = torch.div(rpr * kv_len + 7, 8, rounding_mode="floor")
    base = torch.zeros(hkv * nb + 1, dtype=torch.int64, device=dev)
    base[1:] = torch.cumsum(seg.reshape(-1), 0)
    cap = fp_mask_cap(hkv, nb, rpr, uw)
    k8 = torch.arange(8, device=dev, dtype=torch.int64) * 8
    nbits = (ext.unsqueeze(-1) - k8).clamp(0, 8)                                   # (Hkv, B, rpr, U, 8)
    wb = (torch.ones_like(nbits) << nbits) - 1                                     # byte k of the word
    rowbit = torch.arange(rpr, device=dev, dtype=torch.int64).view(1, 1, rpr) * kv_len.unsqueeze(-1)
    sh = (rowbit % 8).view(hkv, nb, rpr, 1, 1)
    z = torch.zeros_like(wb[..., :1])
    lo = torch.cat([wb, z], -1)                                                    # wb[k], k = 0 .. 8
    hi = torch.cat([z, wb], -1)                                                    # wb[k - 1]
    out9 = ((lo << sh) | (hi >> (8 - sh))) & 0xFF
    start = (base[:-1].view(hkv, nb, 1, 1) + torch.div(rowbit, 8, rounding_mode="floor").unsqueeze(-1)
             + torch.arange(uw, device=dev).view(1, 1, 1, uw) * 8)
    idx = start.unsqueeze(-1) + torch.arange(9, device=dev)
    idx = torch.where(idx < base[1:].view(hkv, nb, 1, 1, 1), idx, torch.full_like(idx, cap))
    acc = torch.zeros(cap + 1, dtype=torch.int32, device=dev)
    acc.scatter_add_(0, idx.reshape(-1), out9.reshape(-1).to(torch.int32))
    mi = base[torch.arange(hkv, device=dev).view(-1, 1) * nb + torch.arange(nb + 1, device=dev).view(1, -1)]
    return acc.to(torch.uint8), mi.to(torch.int32), kv_len


def union_stats(u, b):
    """REPORT numbers of one batch's union (ONE host read)."""
    npg = b["num_pages"].to(torch.int64)
    uc = u["ucnt"].to(torch.float64)
    return dict(ucnt_mean=round(float(uc.mean()), 3), ucnt_max=int(u["ucnt"].max()),
                union_over_sum=round(float(uc.sum() / npg.sum().clamp(min=1)), 5),
                kv_len_mean=round(float(((u["ucnt"] - 1) * PAGE + u["klast"]).to(torch.float64).mean()), 1))


def poison_precondition(u):
    """FI5's precondition for the FP arms: FlashInfer loads EVERY token of a union page before the last one (it masks
    them by score, not by load), and p = 0 times a NaN V is NaN. So the poison plant is valid only when every middle
    union page is fully reached by some row. Returns (ok, detail) (one host read)."""
    reach = u["ext"].amax(dim=2)
    j = torch.arange(reach.shape[-1], device=reach.device).view(1, 1, -1)
    mid = j < (u["ucnt"] - 1).unsqueeze(-1)
    bad = int((mid & (reach < PAGE)).sum())
    return bad == 0, "%d middle union pages are not fully reached by any row" % bad


def chunk_tile_hazard(u, split=FP_SPLIT, tile_pages=KV_TILE_PAGES):
    """FP8 hazard (MODEL from the 0.6.8 source): a split chunk starts its own online softmax at m = -inf; a row with no
    visible token in the chunk's FIRST KV tile gets exp2(-inf - -inf) = NaN (prefill.cuh update_mdo_states has no
    guard) and the merge keeps it. Returns (bad (request, head, row, chunk) cells, rows with any bad cell, chunks)."""
    ext = u["ext"]
    hkv, nb, rpr, uw = ext.shape
    nch = -(-uw // split)
    e = torch.nn.functional.pad(ext, (0, nch * split - uw)).view(hkv, nb, rpr, nch, split)
    vis0 = e[..., :tile_pages].sum(-1)
    live = (torch.arange(nch, device=ext.device) * split).view(1, 1, 1, nch) < u["ucnt"].view(hkv, nb, 1, 1)
    bad = live & (vis0 == 0)
    return int(bad.sum()), int(bad.any(-1).any(0).sum()), int(live[:, :, :1, :].sum())


def flat_base(view):
    """The contiguous tensor a view was made from, flat (the flat-address twins read through it)."""
    t = view
    while t._base is not None:
        t = t._base
    if not t.is_contiguous() or t.storage_offset() != 0:
        raise ValueError("the base of the view must be contiguous from offset 0")
    return t.reshape(-1)


def fi_prefill_twin(q, k_view, v_view, bias, qo_indptr, kv_indptr, kv_indices, last_page_len, page_size, mask,
                    mask_indptr, page_mul, entry_stride, head, scale, custom=True):
    """CPU float64 TWIN of one FlashInfer paged-prefill call with R2FiCisBiasMask (lessons L28: the kernel's flat
    addresses). Request i: query rows q[qo_indptr[i] + qo_idx] (G heads each); kv_len = (kv_indptr[i + 1] -
    kv_indptr[i] - 1) x page_size + last_page_len[i]; token t is entry t % page_size of page kv_indices[kv_indptr[i] +
    t // page_size], read at fi_offset of the view (page.cuh get_elem_offset); its bias at bias_offset(page, entry,
    page_mul, entry_stride, head); logit = (q.k + bias) x scale; custom: token t is visible to row qo_idx iff bit
    qo_idx x kv_len + t of the segment at byte mask_indptr[i] is set; else every token below kv_len is visible."""
    kf, vf = flat_base(k_view).double(), flat_base(v_view).double()
    bfl = bias.reshape(-1).double()
    qi, ki = qo_indptr.to("cpu", torch.int64), kv_indptr.to("cpu", torch.int64)
    ix, ll = kv_indices.to("cpu", torch.int64), last_page_len.to("cpu", torch.int64)
    mk = mask.to("cpu", torch.int64) if custom else None
    mi = mask_indptr.to("cpu", torch.int64) if custom else None
    d = q.shape[-1]
    feat = torch.arange(d).view(1, -1)
    out = torch.zeros(tuple(q.shape), dtype=torch.float64)
    for i in range(qi.numel() - 1):
        kv_len = (int(ki[i + 1] - ki[i]) - 1) * page_size + int(ll[i])
        t = torch.arange(kv_len)
        page = ix[int(ki[i]) + torch.div(t, page_size, rounding_mode="floor")].view(-1, 1)
        ent = (t % page_size).view(-1, 1)
        kk = kf[fi_offset(k_view, page, ent, feat)]
        vv = vf[fi_offset(v_view, page, ent, feat)]
        bb = bfl[bias_offset(page.view(-1), ent.view(-1), page_mul, entry_stride, head)]
        for r in range(int(qi[i + 1] - qi[i])):
            if custom:
                o = int(mi[i]) * 8 + r * kv_len + t
                vis = ((mk[torch.div(o, 8, rounding_mode="floor")] >> (o % 8)) & 1).bool()
            else:
                vis = torch.ones(kv_len, dtype=torch.bool)
            qq = q[int(qi[i]) + r].double().cpu()
            lg = ((qq @ kk.t()) + bb.view(1, -1)) * scale
            lg = lg.masked_fill(~vis.view(1, -1), float("-inf"))
            out[int(qi[i]) + r] = torch.softmax(lg, dim=-1) @ vv
    return out


# ------------------------------------------------------------------ [fia3] V1: NOSA's prefill varlen over a union store
def v1_layout(u):
    """V1's contiguous store layout: W = the largest union over (KV head, request) (ONE host read) slots per request;
    union page j < ucnt - 1 at slot j, the union's last page at slot W - 1, so with seqused_k = (W - 1) x 64 + klast
    the kernel's causal rule (rows < seqused_k - rpr + u + 1, verify_step.py:37-42) cuts query u inside the last page.
    Returns (W, slot_of (Hkv, B, U) int64: the slot of union page j, -1 past ucnt)."""
    ucnt = u["ucnt"]
    uw = u["upage"].shape[-1]
    w = int(ucnt.max())
    j = torch.arange(uw, device=ucnt.device).view(1, 1, uw)
    last = (ucnt - 1).unsqueeze(-1)
    slot_of = torch.where(j < last, j, torch.where(j == last, torch.full_like(j, w - 1), torch.full_like(j, -1)))
    return w, slot_of


def v1_args(b, u, rpr, geom):
    """The internal op's tensors (verify_step.VarlenArgs fields) over a CONTIGUOUS union store gathered from the paged
    store: ku / vu (B, W x 64, Hkv, D), bu (B, W x 64, Hkv); each (KV head, request) has its own slot map (the topk_idx
    rows are per KV head, as EQ's union store); empty slots hold token row 0 of the store (never named, never loaded);
    topk_idx (Hkv, B x rpr, P) int32 = the store slots of each row's listed pages (-1 padded), row n x rpr + u;
    seqused_k (B,) int32 = (W - 1) x 64 + klast of KV head 0 (V1_pre: the same on every head)."""
    hkv, d = geom["Hkv"], geom["D"]
    w, slot_of = v1_layout(u)
    ucnt = u["ucnt"]
    nb = ucnt.shape[1]
    uw = slot_of.shape[-1]
    dev = ucnt.device
    pas = torch.full((hkv, nb, w + 1), -1, dtype=torch.int64, device=dev)
    pas.scatter_(-1, torch.where(slot_of >= 0, slot_of, torch.full_like(slot_of, w)),
                 torch.where(slot_of >= 0, u["upage"], torch.full_like(slot_of, -1)))
    pas = pas[..., :w]
    e = torch.arange(PAGE, device=dev)
    rows = (pas.clamp(min=0).unsqueeze(-1) * PAGE + e).reshape(hkv, nb * w * PAGE)
    k, v, bias = b["k"], b["v"], b["bias"]
    kf, vf, bfl = k.reshape(-1, hkv, d), v.reshape(-1, hkv, d), bias.reshape(-1, hkv)
    ku = torch.empty((nb, w * PAGE, hkv, d), dtype=k.dtype, device=dev)
    vu = torch.empty((nb, w * PAGE, hkv, d), dtype=v.dtype, device=dev)
    bu = torch.empty((nb, w * PAGE, hkv), dtype=bias.dtype, device=dev)
    for h in range(hkv):
        ku[:, :, h] = kf[rows[h], h].view(nb, w * PAGE, d)
        vu[:, :, h] = vf[rows[h], h].view(nb, w * PAGE, d)
        bu[:, :, h] = bfl[rows[h], h].view(nb, w * PAGE)
    ui = u["uidx"]
    sl = torch.gather(slot_of.unsqueeze(2).expand(hkv, nb, rpr, uw), -1, ui.clamp(min=0))
    topk = torch.where(ui >= 0, sl, torch.full_like(sl, -1)).reshape(hkv, nb * rpr, -1).to(torch.int32).contiguous()
    return dict(ku=ku, vu=vu, bu=bu, W=w, topk_idx=topk,
                seqused_k=((w - 1) * PAGE + u["klast"][0]).to(torch.int32).contiguous(),
                cu_seqlens_q=(torch.arange(nb + 1, dtype=torch.int32, device=dev) * rpr).contiguous(),
                cu_seqlens_k=(torch.arange(nb + 1, dtype=torch.int32, device=dev) * (w * PAGE)).contiguous(),
                max_seqlen_q=int(rpr), max_seqlen_k=w * PAGE, softmax_scale=float(d) ** -0.5)


def v1_prep(a):
    """NOSA's two value tensors (verify_step.build_varlen_args :167-169, in the store dtype): ucis = exp(cis), v_scaled
    = v x ucis, v_fake = ucis on every feature. cis = the bench bias x softmax_scale: the bench logit is (q.k + bias) x
    scale (piece 3 = FA2-NOSA decode, bi_paged_attention.py:28-30); NOSA's prefill / Path 1 adds cis UNSCALED
    (nosa_llama.py:301, verify_step.py:167; an open item of the notes)."""
    nb, rows_, hkv, d = a["vu"].shape
    ucis = torch.exp(a["bu"] * a["softmax_scale"])
    vs = (a["vu"] * ucis.unsqueeze(-1)).view(nb * rows_, hkv, d)
    vfk = ucis.unsqueeze(-1).expand(nb, rows_, hkv, d).contiguous().view(nb * rows_, hkv, d)
    return vs, vfk


def v1_precondition(u, b, rpr):
    """V1's layout is exact iff, per request: (a) every row's last listed page is the union's last page on every KV
    head; (b) the last page's extent is the same on every KV head; (c) the rows sit at consecutive extents, last_len[n
    x rpr + u] == klast - rpr + 1 + u (query u is cut at seqused_k - rpr + u + 1). ONE host read. (ok, detail)."""
    ui = u["uidx"]
    hkv, nb, rpr_, p_ = ui.shape
    npg = b["num_pages"].to(torch.int64)
    lastpos = (npg - 1).clamp(min=0).t().reshape(hkv, nb, rpr_, 1)
    lu = ui.gather(-1, lastpos).squeeze(-1)
    a_bad = (lu != (u["ucnt"] - 1).unsqueeze(-1)).any(-1).any(0)
    b_bad = (u["klast"] != u["klast"][:1]).any(0)
    ll = b["last_len"].to(torch.int64).reshape(nb, rpr_)
    want = u["klast"][0].view(nb, 1) - rpr_ + 1 + torch.arange(rpr_, device=ll.device).view(1, rpr_)
    c_bad = (ll != want).any(-1)
    na, nbb, nc = int(a_bad.sum()), int(b_bad.sum()), int(c_bad.sum())
    ok = na == 0 and nbb == 0 and nc == 0
    return ok, ("%s: of %d requests, %d have a row whose last page is not the union's last page, %d a per-head extent "
                "difference, %d non-consecutive extents" % ("met" if ok else "NOT met", nb, na, nbb, nc))


def v1_twin(q, a, vs, vfk, rpr):
    """CPU float64 TWIN of V1 (the varlen layout, verify_step.py:20-48): request n, query u, KV head h sees the store
    rows slot x 64 + e of the slots topk_idx[h, n x rpr + u] names, below seqused_k[n] - rpr + u + 1; weights =
    softmax(q.k x softmax_scale) over them; out = (w . v_scaled) / (w . v_fake[:, 0])."""
    hkv = a["topk_idx"].shape[0]
    nb = a["seqused_k"].numel()
    w = a["W"]
    hq, d = q.shape[1], q.shape[2]
    g = hq // hkv
    ku = a["ku"].reshape(nb, w * PAGE, hkv, d).double().cpu()
    vsd = vs.reshape(nb, w * PAGE, hkv, d).double().cpu()
    vfd = vfk.reshape(nb, w * PAGE, hkv, d).double().cpu()
    tk = a["topk_idx"].to("cpu", torch.int64)
    su = a["seqused_k"].to("cpu", torch.int64)
    out = torch.zeros((nb * rpr, hq, d), dtype=torch.float64)
    e = torch.arange(PAGE)
    for n in range(nb):
        for uq in range(rpr):
            cut = int(su[n]) - rpr + uq + 1
            for h in range(hkv):
                sl = tk[h, n * rpr + uq]
                sl = sl[sl >= 0]
                rr = (sl.view(-1, 1) * PAGE + e.view(1, -1)).reshape(-1)
                rr = rr[rr < cut]
                qq = q[n * rpr + uq, h * g:(h + 1) * g].double().cpu()
                wgt = torch.softmax((qq @ ku[n, rr, h].t()) * a["softmax_scale"], dim=-1)
                out[n * rpr + uq, h * g:(h + 1) * g] = (wgt @ vsd[n, rr, h]) / (wgt @ vfd[n, rr, h, 0]).unsqueeze(-1)
    return out


# ------------------------------------------------------------------ arms (GPU)
class P3Arm:
    """Piece 3 v2 with a fixed workspace (R_max rows)."""
    def __init__(self, mod, geom, rmax, p_width, device):
        self.mod, self.geom = mod, geom
        nsp = mod.splits_for(p_width)
        bh = mod.block_h_for(geom["G"])
        self.ws = (torch.empty((rmax, geom["Hkv"], nsp, bh, geom["D"]), dtype=torch.float32, device=device),
                   torch.empty((rmax, geom["Hkv"], nsp, 2, bh), dtype=torch.float32, device=device))
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in self.ws)
        self.info = None
        self._b = self._out = None

    def prepare(self, b):
        self._b = b
        self._out = torch.empty_like(b["q"])

    def launch(self):
        b = self._b
        return self.mod.bi_paged_decode(b["q"], b["k"], b["v"], b["bias"], b["seq_idx"], b["page_table"],
                                        b["num_pages"], b["last_len"], out=self._out, workspace=self.ws)

    def __call__(self, b):
        self.prepare(b)
        return self.launch().clone()


class S5Arm(P3Arm):
    """sharedkv5: bi_paged_decode_shared(base = piece 3, ..., page_key, rows_max 4, mode 'split', tile 'reload')."""
    def __init__(self, skv, base, geom, rmax, p_width, device):
        super().__init__(base, geom, rmax, p_width, device)
        self.skv = skv

    def launch(self):
        b = self._b
        return self.skv.bi_paged_decode_shared(self.mod, b["q"], b["k"], b["v"], b["bias"], b["seq_idx"],
                                               b["page_table"], b["num_pages"], b["last_len"], b["page_key"],
                                               rows_max=4, mode="split", out=self._out, workspace=self.ws,
                                               tile="reload")


class FIArm:
    """FlashInfer option A: one wrapper per KV head (sharing one float workspace: the head calls run in stream
    order), plan per batch, run per head into (R, G, D) buffers, then the interleave into (R, Hq, D)."""
    def __init__(self, fi, name, split, geom, s_slots, jit_args, device):
        if split not in ("fixed", "none", "heuristic"):
            raise ValueError(split)
        self.fi, self.name, self.split, self.geom, self.S = fi, name, split, geom, s_slots
        self.custom = jit_args is not None
        self.scale = geom["D"] ** -0.5
        self.sms = torch.cuda.get_device_properties(torch.device(device)).multi_processor_count
        self.fws = torch.zeros(MIN_FLOAT_WS, dtype=torch.uint8, device=device)     # grown from the plan's need
        self.float_need = 0
        kw = dict(use_tensor_cores=True, backend="fa2")
        if jit_args is not None:
            kw["jit_args"] = jit_args
        self.wr = [fi.BatchDecodeWithPagedKVCacheWrapper(self.fws, "NHD", **kw) for _ in range(geom["Hkv"])]
        self.info = None
        self.fixed_counts = False
        self._b = self._oh = self._ix = None

    @property
    def workspace_bytes(self):
        """Device workspace now held: the float buffer (sized from the plans' need) + the 8 MiB int buffers."""
        return self.fws.numel() + sum(w._int_workspace_buffer.numel() for w in self.wr)

    @property
    def pinned_bytes(self):
        return sum(w._pin_memory_int_workspace_buffer.numel() for w in self.wr)

    def ensure_ws(self, need):
        """Grow (never shrink) the shared float workspace to FlashInfer's need for the coming plans (MiB-rounded)."""
        self.float_need = int(need)
        if need > self.fws.numel():
            size = (int(need) + (1 << 20) - 1) >> 20 << 20
            self.fws = torch.zeros(size, dtype=torch.uint8, device=self.fws.device)
            for w in self.wr:
                w.reset_workspace_buffer(self.fws, w._int_workspace_buffer)

    def plan_kwargs(self):
        if self.split == "fixed":
            return dict(fixed_split_size=FIXED_SPLIT)
        if self.split == "none":
            return dict(disable_split_kv=True)
        return {}

    def pack(self, b):
        return pack_all_heads(b, self.S)

    def plan_packed(self, packed):
        """plan() per head. indptr / last_page_len on the host or on the device (then plan's own .to('cpu') is the
        D2H sync). The caller synchronizes first: plan rewrites its pinned buffer while the previous plan's H2D copy
        may still be queued (a FlashInfer usage hazard, recorded in the plan doc)."""
        g, d = self.geom["G"], self.geom["D"]
        for h, (indptr, indices, last) in enumerate(packed):
            self.wr[h].plan(indptr, indices, last, g, 1, d, PAGE, q_data_type=torch.bfloat16,
                            kv_data_type=torch.bfloat16, sm_scale=self.scale, **self.plan_kwargs())
        self.info = [plan_info_of(w) for w in self.wr]

    def prepare(self, b):
        """Pack + plan for batch b. With constant counts (every (row, head) lists P pages, read from the host copy)
        the indices live in ONE persistent (Hkv, R * P) buffer written by pack_fixed, so relayer() can serve further
        layers with the same plan; ragged counts use pack_all_heads."""
        check_store(b["k"], b["v"], b["bias"])
        torch.cuda.synchronize()
        r_, hkv, p_ = b["page_table"].shape
        counts = b["num_pages"].cpu()
        self.fixed_counts = bool((counts == p_).all())
        if self.fixed_counts:
            self._ix = torch.empty((hkv, r_ * p_), dtype=torch.int32, device=b["q"].device)
            pack_fixed(b, self.S, self._ix)
            ip = torch.arange(r_ + 1, dtype=torch.int32) * p_
            last = b["last_len"].to("cpu", torch.int32)
            packed = [(ip, self._ix[h], last) for h in range(hkv)]
        else:
            self._ix = None
            packed = [(ip.cpu(), ix, ll.cpu()) for ip, ix, ll in self.pack(b)]
        self.ensure_ws(max(fi_float_need(counts[:, h].tolist(), self.split, self.geom["G"], self.geom["D"], self.sms)
                           for h in range(hkv)))
        self.plan_packed(packed)
        self._oh = [torch.empty((r_, self.geom["G"], self.geom["D"]), dtype=torch.bfloat16, device=b["q"].device)
                    for _ in range(hkv)]
        self._b = b

    def relayer(self, b):
        """[fix after fia1] The next LAYER under the same plan: b must have the planned batch's row count, constant
        counts and last_len (the plan depends on those only); its page lists are written in place into the planned
        indices buffer. No plan, no host read."""
        if not self.fixed_counts or self._ix is None or b["page_table"].shape[0] != self._b["page_table"].shape[0]:
            raise ValueError("relayer needs a prepared constant-count batch of the same size")
        pack_fixed(b, self.S, self._ix)
        self._b = b

    def launch_runs(self):
        b = self._b
        g, hkv = self.geom["G"], self.geom["Hkv"]
        for h in range(hkv):
            kv = (kv_view_a(b["k"], self.S, h), kv_view_a(b["v"], self.S, h))
            extra = [b["bias"], *bias_args("a", hkv, h), self.scale] if self.custom else []
            self.wr[h].run(b["q"][:, h * g:(h + 1) * g], kv, *extra, out=self._oh[h])
        return self._oh

    def launch(self):
        return torch.cat(self.launch_runs(), dim=1)

    def __call__(self, b):
        self.prepare(b)
        return self.launch()


class FIStridedArm(FIArm):
    """Option B (REPORT probe): one wrapper, one call over (row, head) requests through kv_view_b."""
    def __init__(self, fi, geom, s_slots, jit_args, device):
        super().__init__(fi, "STRIDED", "fixed", geom, s_slots, jit_args, device)
        self.wr = self.wr[:1]

    def prepare(self, b):
        check_store(b["k"], b["v"], b["bias"])
        torch.cuda.synchronize()
        ip, ix, ll = pack_option_b(b, self.S)
        self.ensure_ws(fi_float_need(b["num_pages"].cpu().reshape(-1).tolist(), "fixed", self.geom["G"],
                                     self.geom["D"], self.sms))
        self.wr[0].plan(ip.cpu(), ix, ll.cpu(), self.geom["G"], 1, self.geom["D"], PAGE, q_data_type=torch.bfloat16,
                        kv_data_type=torch.bfloat16, sm_scale=self.scale, fixed_split_size=FIXED_SPLIT)
        self.info = [plan_info_of(self.wr[0])]
        r_ = b["q"].shape[0]
        self._oh = [torch.empty((r_ * self.geom["Hkv"], self.geom["G"], self.geom["D"]), dtype=torch.bfloat16,
                                device=b["q"].device)]
        self._b = b

    def launch(self):
        b = self._b
        r_ = b["q"].shape[0]
        q2 = b["q"].view(r_ * self.geom["Hkv"], self.geom["G"], self.geom["D"])
        extra = [b["bias"], *bias_args("b", self.geom["Hkv"]), self.scale]
        self.wr[0].run(q2, (kv_view_b(b["k"]), kv_view_b(b["v"])), *extra, out=self._oh[0])
        return self._oh[0].view(r_, self.geom["Hq"], self.geom["D"])


def jit_args_fp(d):
    """[fia3] jit_args of the FP arms' variant (the same 13-item list as jit_args_for)."""
    return [VARIANT_URI_FP % d, torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, d, d,
            list(VARIANT_TENSORS_FP[0]), list(VARIANT_TENSORS_FP[1]), list(VARIANT_SCALARS[0]),
            list(VARIANT_SCALARS[1]), VARIANT_NAME_FP, VARIANT_DECL_FP]


FP_INIT_PARAMS = ("jit_args", "backend")
FP_PLAN_PARAMS = ("packed_custom_mask", "disable_split_kv", "fixed_split_size", "sm_scale", "q_data_type",
                  "kv_data_type")


def check_fp_api(fi):
    """[fia3] gate FP_API: the installed BatchPrefillWithPagedKVCacheWrapper takes what the FP arms pass (read from the
    signatures; 0.6.8.post1 has them all, the 0.5.3 Python inside the sif is not readable from the login node).
    (ok, detail); a **kwargs signature cannot be checked and says so."""
    w = getattr(fi, "BatchPrefillWithPagedKVCacheWrapper", None)
    if w is None:
        return False, "no BatchPrefillWithPagedKVCacheWrapper in flashinfer"
    try:
        pi = inspect.signature(w.__init__).parameters
        pp = inspect.signature(w.plan).parameters
    except (TypeError, ValueError) as e:
        return False, "signatures not readable: %s" % e
    kw = [n for n, ps in (("__init__", pi), ("plan", pp))
          if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values())]
    miss = [p for p in FP_INIT_PARAMS if p not in pi] + [p for p in FP_PLAN_PARAMS if p not in pp]
    if miss and not kw:
        return False, "missing parameters %s" % miss
    return True, "parameters present%s" % ("" if not kw else " (**kwargs in %s: not checkable)" % kw)


class FPArm:
    """[fia3] Option 1 of the survey: FlashInfer BatchPrefillWithPagedKVCacheWrapper (fa2), one wrapper per KV head
    (option A views, as F8). One FlashInfer request = (request, KV head): qo_len = rows per request; num_qo_heads = the
    group (16), num_kv_heads 1, so FlashInfer packs qo_len x 16 = 64 query rows into ONE CTA tile (its own GQA
    support: packed_qo_len, scheduler.cuh:505-510; CTA_TILE_Q 64, utils.cuh:384-403). KV pages = the request's union;
    the packed mask, its byte offsets, the CSR and last_page_len live in persistent buffers written IN PLACE (build), so
    a plan with no split serves later layers (relayer). split "none" (FP0: disable_split_kv) or "fixed" (FP8:
    fixed_split_size 8); mask "rows" (each row its own tokens), "ones" (each row the pair's reach: the MASK WITNESS),
    "off" (FPU: plan without a mask, kNone kernels: timing only)."""
    def __init__(self, fi, name, split, mask, geom, s_slots, rpr, jit_args, device):
        if split not in ("none", "fixed") or mask not in ("rows", "ones", "off"):
            raise ValueError("split %r, mask %r" % (split, mask))
        self.fi, self.name, self.split, self.mask, self.geom, self.S, self.rpr = fi, name, split, mask, geom, s_slots, rpr
        self.scale = geom["D"] ** -0.5
        self.dev = torch.device(device)
        self.fws = torch.zeros(MIN_FLOAT_WS, dtype=torch.uint8, device=device)
        self.float_need = 0
        self.wr = [fi.BatchPrefillWithPagedKVCacheWrapper(self.fws, "NHD", backend="fa2", jit_args=jit_args)
                   for _ in range(geom["Hkv"])]
        self.bufs = self.u = self.info = None
        self._b = self._oh = None
        self.fresh_buffers = False          # TEST PLANT only: build into NEW tensors (the plan keeps the old ones)

    @property
    def workspace_bytes(self):
        own = sum(t.numel() * t.element_size() for t in (self.bufs or {}).values() if torch.is_tensor(t))
        ints = sum(w._int_workspace_buffer.numel() for w in self.wr if hasattr(w, "_int_workspace_buffer"))
        return self.fws.numel() + ints + own

    @property
    def pinned_bytes(self):
        return sum(w._pin_memory_int_workspace_buffer.numel() for w in self.wr
                   if hasattr(w, "_pin_memory_int_workspace_buffer"))

    def _sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def ensure_ws(self, need):
        self.float_need = int(need)
        if need > self.fws.numel():
            size = (int(need) + (1 << 20) - 1) >> 20 << 20
            self.fws = torch.zeros(size, dtype=torch.uint8, device=self.dev)
            for w in self.wr:
                w.reset_workspace_buffer(self.fws, w._int_workspace_buffer)

    def alloc(self, nb, p_):
        hkv, uw, d = self.geom["Hkv"], self.rpr * p_, self.dev
        self.bufs = dict(nb=nb, p=p_, qo=torch.arange(nb + 1, dtype=torch.int32, device=d) * self.rpr,
                         indptr=torch.zeros((hkv, nb + 1), dtype=torch.int32, device=d),
                         ix=torch.zeros((hkv, nb * uw + 1), dtype=torch.int32, device=d),
                         last=torch.ones((hkv, nb), dtype=torch.int32, device=d),
                         mask=torch.zeros(fp_mask_cap(hkv, nb, self.rpr, uw) + 1, dtype=torch.uint8, device=d),
                         mi=torch.zeros((hkv, nb + 1), dtype=torch.int32, device=d))

    def build(self, b):
        """Union + CSR + packed mask of batch b, copied IN PLACE into the buffers the plan holds. No host read."""
        u = fp_union(b, self.rpr, self.S, ones=(self.mask == "ones"))
        ip, ix, last = fp_csr(u)
        m, mi, _ = fp_pack(u)
        bf = self.bufs
        if self.fresh_buffers:
            bf.update(indptr=ip, ix=ix, last=last, mask=m, mi=mi)
        else:
            bf["indptr"].copy_(ip)
            bf["ix"].copy_(ix)
            bf["last"].copy_(last)
            bf["mask"].copy_(m)
            bf["mi"].copy_(mi)
        self.u = u
        return u

    def host_copies(self):
        """Host copies of (qo_indptr, indptr, last_page_len) per KV head (for the host-indptr plan timing)."""
        bf = self.bufs
        return [(bf["qo"].cpu(), bf["indptr"][h].cpu(), bf["last"][h].cpu()) for h in range(self.geom["Hkv"])]

    def plan(self, host=None):
        """plan() per KV head on the held buffers (device indptr: plan's own D2H read). host = host_copies(): the
        wrapper then holds COPIES, so the plan-reuse path needs the device form again (time_fp_extras re-plans)."""
        g, d, bf = self.geom["G"], self.geom["D"], self.bufs
        kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=self.scale)
        kw.update(dict(disable_split_kv=True) if self.split == "none" else dict(fixed_split_size=FP_SPLIT))
        if self.mask != "off":
            kw["packed_custom_mask"] = bf["mask"]                 # selects the kCustom kernels (mask mode CUSTOM)
        for h in range(self.geom["Hkv"]):
            qo, ip, last = host[h] if host is not None else (bf["qo"], bf["indptr"][h], bf["last"][h])
            self.wr[h].plan(qo, ip, bf["ix"][h], last, g, 1, d, PAGE, **kw)
        try:
            self.info = [plan_info_of(w) for w in self.wr]
        except Exception as e:                                    # a fake or another layout: REPORT only
            self.info = [dict(error="%s: %s" % (type(e).__name__, e))]

    def prepare(self, b):
        check_store(b["k"], b["v"], b["bias"])
        r_, hkv, p_ = b["page_table"].shape
        if r_ % self.rpr:
            raise ValueError("%d rows are not requests of %d rows" % (r_, self.rpr))
        nb = r_ // self.rpr
        seq = b["seq_idx"].to("cpu", torch.int64).view(nb, self.rpr)
        if not bool((seq == seq[:, :1]).all()):
            raise ValueError("the rows of a request must be consecutive and share one storage entry")
        if self.bufs is None or self.bufs["nb"] != nb or self.bufs["p"] != p_:
            self.alloc(nb, p_)
        self.build(b)
        self._sync()                                               # plan rewrites its pinned buffer (plan doc 8.1)
        if self.split == "fixed":
            uc = self.u["ucnt"].cpu()
            self.ensure_ws(max(fp_float_need(uc[h].tolist(), "fixed", self.geom["G"], self.geom["D"], self.rpr)
                               for h in range(hkv)))
        self.plan()
        self._oh = [torch.empty((r_, self.geom["G"], self.geom["D"]), dtype=torch.bfloat16, device=self.dev)
                    for _ in range(hkv)]
        self._b = b

    def relayer(self, b):
        """The next LAYER under the same plan (no split only: the plan then does not read the union sizes): b's union,
        CSR and mask are written in place. No plan, no host read."""
        if self.split != "none" or self.bufs is None or self._b is None \
                or tuple(b["page_table"].shape) != tuple(self._b["page_table"].shape):
            raise ValueError("relayer needs a prepared no-split arm and a batch of the same shape")
        self.build(b)
        self._b = b

    def launch_runs(self):
        b, bf = self._b, self.bufs
        g, hkv = self.geom["G"], self.geom["Hkv"]
        for h in range(hkv):
            kv = (kv_view_a(b["k"], self.S, h), kv_view_a(b["v"], self.S, h))
            self.wr[h].run(b["q"][:, h * g:(h + 1) * g], kv, b["bias"], bf["mask"], bf["mi"][h],
                           *bias_args("a", hkv, h), self.scale, out=self._oh[h])
        return self._oh

    def launch(self):
        return torch.cat(self.launch_runs(), dim=1)

    def __call__(self, b):
        self.prepare(b)
        return self.launch()


def import_v1(path):
    """[fia3] V1's module (EQ verify_step.py, a pinned copy, loaded by path: torch only at import) and NOSA's internal
    varlen op (infllm_v2, inside the sif). (module, None) or (None, the reason: a FINDING)."""
    try:
        vm = load_by_path(path, "eq_verify_step")
        if not (hasattr(vm, "varlen_attention") and hasattr(vm, "VarlenArgs")):
            return None, "%s has no varlen_attention / VarlenArgs" % path
        iv = importlib.import_module("infllm_v2.infllmv2_sparse_attention")
        if not hasattr(iv, "_wrapped_infllmv2_attn_varlen_forward"):
            return None, "infllm_v2.infllmv2_sparse_attention has no _wrapped_infllmv2_attn_varlen_forward"
        return vm, None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:300])


class V1Arm:
    """[fia3] NOSA's prefill varlen route (EQ verify_step.varlen_attention) over a CONTIGUOUS union store: stage_gather
    (union + the store copy, one host read for W), stage_prep (exp(cis) values), launch (the two internal-op calls and
    the quotient)."""
    def __init__(self, vmod, geom, rpr, s_slots):
        self.vmod, self.geom, self.rpr, self.S = vmod, geom, rpr, s_slots
        self.info = None
        self.a = self.vs = self.vfk = self.u = self._b = None

    @property
    def workspace_bytes(self):
        ts = [] if self.a is None else [self.a["ku"], self.a["vu"], self.a["bu"]]
        ts += [t for t in (self.vs, self.vfk) if t is not None]
        return sum(t.numel() * t.element_size() for t in ts)

    def stage_gather(self, b):
        self.u = fp_union(b, self.rpr, self.S)
        self.a = v1_args(b, self.u, self.rpr, self.geom)
        self._b = b

    def stage_prep(self):
        self.vs, self.vfk = v1_prep(self.a)

    def prepare(self, b):
        self.stage_gather(b)
        self.stage_prep()

    def varlen_args(self):
        a = self.a
        nb, rows_, hkv, d = a["ku"].shape
        return self.vmod.VarlenArgs(q=self._b["q"], k=a["ku"].view(nb * rows_, hkv, d), v_scaled=self.vs,
                                    v_fake=self.vfk, cu_seqlens_q=a["cu_seqlens_q"], cu_seqlens_k=a["cu_seqlens_k"],
                                    seqused_k=a["seqused_k"], max_seqlen_q=a["max_seqlen_q"],
                                    max_seqlen_k=a["max_seqlen_k"], topk_idx=a["topk_idx"],
                                    softmax_scale=a["softmax_scale"], causal=True)

    def launch(self):
        return self.vmod.varlen_attention(self.varlen_args())

    def __call__(self, b):
        self.prepare(b)
        return self.launch()


# ------------------------------------------------------------------ [fia4] the fork of vLLM's Triton unified attention
def import_fv(path):
    """[fia4] The fork (nosi/nosi/fork_vllm_unified_attention.py of THIS tree, loaded by path; it imports torch and
    triton only). (module, None) or (None, the reason: a FINDING)."""
    try:
        mod = load_by_path(path, "fork_vllm_unified_attention")
        for name in ("unified_attention", "kernel_unified_attention", "fork_build_union"):
            if not hasattr(mod, name):
                return None, "%s has no %s" % (path, name)
        return mod, None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:300])


def fv_width(p_):
    """[fia4] The union-table width W = FV_ROWS x the next power of two >= P (the kernel's scratch and rank range)."""
    pp = 1
    while pp < int(p_):
        pp *= 2
    return FV_ROWS * pp


def fv_tables_from_union(u, width):
    """[fia4] The fork's union tables from fp_union's output: the TORCH builder (FVT's input, the oracle of FV's
    in-kernel build). table (Hkv, B, width) int32 = the union page ids (entry x S + slot) in ascending (block id, slot)
    order; ext (Hkv, B, width) int32 = per union page the rows' visible tokens, one byte per row (row u at bits 8u ..
    8u + 7); length (Hkv, B) int32 = the union sizes. Entries past a union are 0. Device ops only (no host read)."""
    ext = u["ext"]
    hkv, nb, rpr, uw = ext.shape
    if rpr > FV_ROWS or int(width) < uw:
        raise ValueError("rows per request %d (<= %d), width %d (>= %d)" % (rpr, FV_ROWS, width, uw))
    sh = (torch.arange(rpr, device=ext.device, dtype=torch.int64) * 8).view(1, 1, rpr, 1)
    word = (ext << sh).sum(2)                                                    # disjoint bytes: the sum is the or
    table = torch.zeros((hkv, nb, int(width)), dtype=torch.int32, device=ext.device)
    xt = torch.zeros((hkv, nb, int(width)), dtype=torch.int32, device=ext.device)
    table[..., :uw] = u["upage"].to(torch.int32)
    xt[..., :uw] = word.to(torch.int32)
    return table, xt, u["ucnt"].to(torch.int32).contiguous()


def fv_sorted_rows(b):
    """[fia4] The precondition of the fork's in-kernel build (piece 3's canonical order, build_page_table): every
    (row, KV head) lists its pages in strictly ascending block id. (ok, detail); one host read (gates only)."""
    pk = b["page_key"].to("cpu", torch.int64)
    npg = b["num_pages"].to("cpu", torch.int64)
    j = torch.arange(1, pk.shape[-1]).view(1, 1, -1)
    live = j < npg.unsqueeze(-1)
    bad = int((live & (pk[..., 1:] <= pk[..., :-1])).any(-1).sum())
    return bad == 0, "%d (row, KV head) lists are not strictly ascending by block id" % bad


class FVArm:
    """[fia4] The fork of vLLM's Triton unified attention (module docstring [fia4]; docs/NOTES_fia4.md). build True
    (FV): the kernel builds the union table itself (fork_build_union); False (FVT): the torch builder writes it IN PLACE
    before the launch (build); ones (FVT only): every row sees the pair's whole reach (the MASK WITNESS). Persistent
    buffers per batch shape: the union tables (Hkv, B, W), the lengths, cu_seqlens_q, a zero seqused_k (the fork reads
    only its length) and the output. prepare has one host read (the rows of a request share one storage entry, as
    FPArm); launch has none (R_GRAPH_FV captures it).
    [fia5] launch: None (Triton's defaults, as upstream; the token tile = one block) or dict(num_warps, num_stages,
    tile_size, bias_bits) (a swept arm); extents False: the FVN timing arm (use_extents=False); bias_bits 32: prepare
    passes the CIS bias as a float32 copy (the same values), 16: the store's tensor as it is."""
    def __init__(self, mod, name, geom, s_slots, rpr, build, ones=False, device="cuda", launch=None, extents=True):
        if rpr > FV_ROWS or (build and ones):
            raise ValueError("rows per request %d (<= %d), build %s, ones %s" % (rpr, FV_ROWS, build, ones))
        self.mod, self.name, self.geom, self.S, self.rpr = mod, name, geom, s_slots, rpr
        self.build_in_kernel, self.ones = bool(build), bool(ones)
        self.scale = geom["D"] ** -0.5
        self.dev = torch.device(device)
        self.bufs = self.u = self.info = None
        self._b = self._out = self._bias = None
        cfg = dict(launch or {})                                            # [fia5]
        self.bias_bits = int(cfg.pop("bias_bits", FV_BIAS_BITS))
        if self.bias_bits not in (16, 32) or set(cfg) - {"num_warps", "num_stages", "tile_size"}:
            raise ValueError("launch %r" % (launch,))
        self.launch_cfg = dict(cfg, bias_bits=self.bias_bits)
        self.launch_kw = {k: int(v) for k, v in cfg.items() if v is not None}    # none: Triton's defaults (upstream)
        if not extents:
            self.launch_kw["use_extents"] = False

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.bufs or {}).values() if torch.is_tensor(t))

    def alloc(self, nb, p_):
        hkv, d, w = self.geom["Hkv"], self.dev, fv_width(p_)
        self.bufs = dict(nb=nb, p=p_, w=w, table=torch.zeros((hkv, nb, w), dtype=torch.int32, device=d),
                         ext=torch.zeros((hkv, nb, w), dtype=torch.int32, device=d),
                         ulen=torch.zeros((hkv, nb), dtype=torch.int32, device=d),
                         cu=(torch.arange(nb + 1, dtype=torch.int32, device=d) * self.rpr).contiguous(),
                         seqused=torch.zeros(nb, dtype=torch.int32, device=d))

    def build(self, b):
        """The torch build (FVT; FV's oracle), copied IN PLACE into the tables the launch reads. No host read."""
        u = fp_union(b, self.rpr, self.S, ones=self.ones)
        t, x, n = fv_tables_from_union(u, self.bufs["w"])
        self.bufs["table"].copy_(t)
        self.bufs["ext"].copy_(x)
        self.bufs["ulen"].copy_(n)
        self.u = u
        return u

    def prepare(self, b):
        check_store(b["k"], b["v"], b["bias"])
        r_, hkv, p_ = b["page_table"].shape
        if r_ % self.rpr or b["k"].shape[1] != self.S * PAGE:
            raise ValueError("%d rows are not requests of %d rows, or the store is not S x %d rows" % (r_, self.rpr, PAGE))
        nb = r_ // self.rpr
        seq = b["seq_idx"].to("cpu", torch.int64).view(nb, self.rpr)
        if not bool((seq == seq[:, :1]).all()):
            raise ValueError("the rows of a request must be consecutive and share one storage entry")
        if self.bufs is None or self.bufs["nb"] != nb or self.bufs["p"] != p_:
            self.alloc(nb, p_)
        if not self.build_in_kernel:
            self.build(b)
        self._out = torch.empty_like(b["q"])
        bias = b["bias"].view(b["k"].shape[0] * self.S, PAGE, hkv)               # [fia5] the bias dtype of this arm
        self._bias = bias.float().contiguous() if self.bias_bits == 32 else bias
        self._b = b

    def launch(self):
        b, bf = self._b, self.bufs
        n_, sr, hkv, d = b["k"].shape
        nblk = n_ * self.S
        bu = None
        if self.build_in_kernel:
            bu = dict(page_key=b["page_key"], page_table=b["page_table"], num_pages=b["num_pages"],
                      last_len=b["last_len"], seq_idx=b["seq_idx"], slots_per_entry=self.S)
        self.mod.unified_attention(
            b["q"], b["k"].view(nblk, PAGE, hkv, d), b["v"].view(nblk, PAGE, hkv, d), self._out, bf["cu"], self.rpr,
            bf["seqused"], sr, self.scale, True, (-1, -1), bf["table"], 0, None, None, None,
            union_ext=bf["ext"], union_len=bf["ulen"], cis_bias=self._bias, build_union=bu,     # [fia5] self._bias
            rows_per_q_block=FV_ROWS, **self.launch_kw)
        return self._out

    def tables(self):
        """Copies of the tables the last launch read (FVT) or wrote (FV): (table, ext, length)."""
        return self.bufs["table"].clone(), self.bufs["ext"].clone(), self.bufs["ulen"].clone()

    def __call__(self, b):
        self.prepare(b)
        return self.launch().clone()


def make_fv_arm(mod, name, geom, s_slots, rpr, device, ones=False):
    """[fia5] An arm of the fork by name: FV (the kernel builds its table), FVT (a torch table), FVN (FV without the
    extents: timing only) or a swept FVW<w>S<s>T<t>B<b> (FV with that launch). ones=True: the MASK-WITNESS twin of the
    named arm (the same launch, a torch table where every row sees the pair's reach)."""
    cfg = fv_sweep_launch(name)
    if cfg is None and name not in FV_ARMS:
        raise ValueError("not an arm of the fork: %r" % name)
    if ones:
        return FVArm(mod, name + "ones", geom, s_slots, rpr, build=False, ones=True, device=device, launch=cfg)
    return FVArm(mod, name, geom, s_slots, rpr, build=(name != "FVT"), device=device, launch=cfg,
                 extents=(name != "FVN"))


def fv_compile_info(mod, launch_kw=None):
    """[fia5] The compile witness of the fork's LAST launch (FORK_LAST_LAUNCH: Triton's CompiledKernel; None under the
    interpreter or a fork without it): num_warps, num_stages, registers per thread and local memory per thread in
    4-byte words (Triton's n_spills = CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4, driver.c:125-126), shared memory, the
    loop's multi-buffered shared allocations in the TTGIR (local_alloc with no operand: 'DxAxBxdtype', D = the buffer
    depth: the pipeliner's K / V / table buffers), the largest depth among
    the rank-3 ones, and a MODEL of programs per SM (sm80: 65536 registers in units of 256 per warp, 167936 B of
    shared memory with 1 KiB reserved per program, 64 warps). None when there is no handle."""
    ck = (getattr(mod, "FORK_LAST_LAUNCH", None) or {}).get("kernel")
    if ck is None:
        return None
    md = getattr(ck, "metadata", None)
    asm = getattr(ck, "asm", None) or {}
    try:
        ttgir = asm["ttgir"] if "ttgir" in asm else ""
    except Exception:                                                   # an asm dict that cannot give the text
        ttgir = ""
    allocs = re.findall(r"local_alloc\s*:\s*\(\)\s*->\s*!tt\.memdesc<((?:\d+x)+)([a-z]+\d*)", str(ttgir))
    shapes = sorted({"%s%s" % (dims, ty) for dims, ty in allocs})
    depth = max([int(d.split("x")[0]) for d, _ in allocs if d.count("x") == 3] or [0])
    nw, ns = getattr(md, "num_warps", None), getattr(md, "num_stages", None)
    shared = getattr(md, "shared", None)
    regs, spill = getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)
    per_sm = None
    if nw and regs is not None and shared is not None:
        per_warp = -(-int(regs) * 32 // 256) * 256
        by_regs = (65536 // max(per_warp, 1)) // int(nw)
        by_smem = A100_SMEM_PER_SM // (int(shared) + 1024)
        by_warps = 64 // int(nw)
        per_sm = dict(registers=by_regs, shared=by_smem, warps=by_warps, model=min(by_regs, by_smem, by_warps))
    return dict(num_warps=nw, num_stages=ns, regs=regs, local_words_per_thread=spill, shared=shared,
                buffers=shapes, kv_depth=depth, programs_per_sm=per_sm, launch_kw=dict(launch_kw or {}))


def fv_compile_gate(name, tag, mod, arm):
    """[fia5] R_<arm>_compile (REPORT) after the arm's first call; returns (gate, info or None)."""
    info = fv_compile_info(mod, getattr(arm, "launch_kw", None))
    if info is None:
        return gate("R_%s_compile_%s" % (name, tag), False, False, "no compiled-kernel handle (the interpreter, or a "
                    "fork without FORK_LAST_LAUNCH); launch %s" % getattr(arm, "launch_kw", None)), None
    return gate("R_%s_compile_%s" % (name, tag), True, False,
                "num_warps %s, num_stages %s, registers %s, local memory %s words of 4 B per thread, shared memory "
                "%s B; loop buffers %s (largest depth %d); programs per SM (MODEL) %s; launch %s"
                % (info["num_warps"], info["num_stages"], info["regs"], info["local_words_per_thread"],
                   info["shared"], info["buffers"], info["kv_depth"], info["programs_per_sm"],
                   info["launch_kw"])), info


# ------------------------------------------------------------------ gates (arm-agnostic; the CPU tests use fakes)
def bits_equal(a, b):
    """Bitwise equality (a -0.0 / +0.0 or a NaN payload difference counts)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    it = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(it), b.contiguous().view(it))


def changed_rows(b1, b2):
    """Rows whose q or page lists differ between two batches of equal size (the I3 precondition)."""
    out = []
    for r in range(b1["q"].shape[0]):
        if not (torch.equal(b1["q"][r], b2["q"][r]) and torch.equal(b1["page_table"][r], b2["page_table"][r])):
            out.append(r)
    return out


def invariance_suite(name, tag, arm, bf, ba, bh, alone_rows, perms, poisoned, short_idx, hazard_pre, must):
    """I1 - I5 for one arm and cell. arm(batch) -> (R, Hq, D) output; arm.info after a call = the plan info per KV
    head (None for piece 3). Returns (gates, the full-batch output)."""
    gates = []
    full = arm(bf)
    r_ = full.shape[0]
    bad = [r for r in alone_rows if not bits_equal(arm(select(bf, [r]))[0], full[r])]
    gates.append(gate("I1_alone_%s_%s" % (name, tag), not bad, must,
                      "%d rows alone vs in the batch of %d; differing %s" % (len(alone_rows), r_, bad[:10])))
    bad, det = [], []
    for pname, p in perms:
        o = arm(select(bf, p))
        d = [r for j, r in enumerate(p) if not bits_equal(o[j], full[r])]
        det.append("%s %d" % (pname, len(d)))
        bad += d
    gates.append(gate("I2_order_%s_%s" % (name, tag), not bad, must,
                      "rows differing per permutation: %s; first %s" % (", ".join(det), sorted(set(bad))[:10])))
    oa = arm(ba)
    keep = list(range(0, r_, 2))
    changed = changed_rows(bf, ba)
    pre = bool(changed) and all(r % 2 == 1 for r in changed)
    bad = [r for r in keep if not bits_equal(oa[r], full[r])]
    gates.append(gate("I3_siblings_%s_%s" % (name, tag), pre and not bad, must,
                      "%s; %d odd rows changed (q and lists), %d even rows compared; differing %s"
                      % ("precondition met" if pre else "VOID: the alt batch did not change exactly odd rows",
                         len(changed), len(keep), bad[:10]), status=None if pre else "VOID"))
    oh = arm(bh)
    info_b = arm.info
    bad_rest = [r for r in range(r_) if not bits_equal(oh[r], full[r])]
    bad_short, pre_rows = [], []
    for i, s in enumerate(short_idx):
        o = arm(select(bh, [s]))
        info_a = arm.info
        if not bits_equal(o[0], oh[s]):
            bad_short.append(s)
        if SHORT_COUNTS[i] <= FIXED_SPLIT and info_a is not None and info_b is not None:
            pre_rows.append(all(not x["split_kv"] for x in info_a) and all(x["split_kv"] for x in info_b))
    pre4 = (not hazard_pre) or (bool(pre_rows) and all(pre_rows))
    gates.append(gate("I4_split_switch_%s_%s" % (name, tag), pre4 and not bad_short and not bad_rest, must,
                      "%s; short rows (%s pages) alone vs in the hazard batch: differing %s; full rows in the hazard "
                      "batch vs the full batch: differing %s"
                      % ("precondition met (split_kv false alone, true batched, every head)" if hazard_pre and pre4
                         else ("VOID: split_kv precondition not met %s" % pre_rows if hazard_pre else
                               "no split_kv precondition for this arm"),
                         ",".join(str(c) for c in SHORT_COUNTS), bad_short, bad_rest[:10]),
                      status=None if pre4 else "VOID"))
    bp = dict(bf)
    bp.update(poisoned)
    op = arm(bp)
    finite = bool(torch.isfinite(op.float()).all())
    bad = [r for r in range(r_) if not bits_equal(op[r], full[r])]
    gates.append(gate("I5_poison_%s_%s" % (name, tag), finite and not bad, must,
                      "unread rows NaN in K / V / bias: output finite %s; differing %s" % (finite, bad[:10])))
    return gates, full


def correctness_gate(name, tag, out_rows, ref, tol, must):
    d = (out_rows.double() - ref.to(out_rows.device)).abs()
    finite = bool(torch.isfinite(d).all())
    err = float(d.max()) if finite else float("inf")
    return gate("C_%s_%s" % (name, tag), finite and err <= tol, must,
                "max |arm - fp64| %.3e over %d rows (tol %.1e); %s"
                % (err, out_rows.shape[0], tol, band_text(err, PRED["err_vs_fp64"][name])
                   if name in PRED["err_vs_fp64"] else "no band"))


def bias_witness(tag, arm, bf, rows, ref_bias, ref_nobias, tol, must, gname=None):
    """CW: the arm with bias = 0 must FAIL C against the biased reference; VOID when the bias barely moves the
    reference (a no-op plant, lessons L34). [fia3] gname: the gate's name (default the F8 name)."""
    gname = gname or "CW_bias_witness_%s" % tag
    move = float((ref_bias - ref_nobias).abs().max())
    if not move >= WITNESS_FACTOR * tol:
        return gate(gname, False, must, "VOID: the bias moves the reference by %.3e < %.0f x tol"
                    % (move, WITNESS_FACTOR), status="VOID")
    b0 = dict(bf)
    b0["bias"] = torch.zeros_like(bf["bias"])
    o = arm(b0)[torch.as_tensor(rows, device=bf["q"].device)]
    err = float((o.double() - ref_bias.to(o.device)).abs().max())
    st = "DETECTED" if err > tol else "BLIND"
    return gate(gname, st == "DETECTED", must,
                "%s: bias-free arm vs biased reference %.3e (tol %.1e); the bias moves the reference by %.3e"
                % (st, err, tol, move), status=st)


def negative_control(tag, arm, bf, rows, must):
    """N: the heuristic split (FH) alone vs batched must differ on a row whose plan differs (precondition)."""
    full = arm(bf)
    info_b = arm.info
    pre, det, chunks = [], [], set()
    for r in rows:
        o = arm(select(bf, [r]))
        info_a = arm.info
        differs = any(a["chunk_tokens"] != c["chunk_tokens"] or a["split_kv"] != c["split_kv"]
                      for a, c in zip(info_a, info_b))
        chunks.add((tuple(x["chunk_tokens"] for x in info_a), tuple(x["chunk_tokens"] for x in info_b)))
        if differs:
            pre.append(r)
            if not bits_equal(o[0], full[r]):
                det.append(r)
    st = "VOID" if not pre else ("DETECTED" if det else "BLIND")
    return gate("N_heuristic_negative_%s" % tag, st == "DETECTED", must,
                "%s: %d of %d rows with a batch-dependent plan (chunk tokens alone / batched per head: %s); "
                "differing %d" % (st, len(pre), len(rows), sorted(chunks)[:4], len(det)), status=st), full


def plan_reuse_check(name, tag, arm, b_plan, b_layer):
    """[fix after fia1] REPORT (the assessment's open item): one plan serves a second "layer" with the same counts.
    Plan on b_plan, write b_layer's lists in place (relayer), run; compare bitwise with a fresh plan on b_layer.
    b_layer = the alt batch (other q and lists in the odd rows, the same counts and extents)."""
    arm.prepare(b_plan)
    if not arm.fixed_counts:
        return gate("R_REUSE_%s_%s" % (name, tag), False, False, "not run: the batch has ragged counts")
    arm.relayer(b_layer)
    reused = arm.launch()
    fresh = arm(b_layer)
    bad = [r for r in range(fresh.shape[0]) if not bits_equal(reused[r], fresh[r])]
    return gate("R_REUSE_%s_%s" % (name, tag), not bad, False,
                "one plan, the next layer's lists written in place: %d of %d rows differ from a fresh plan %s"
                % (len(bad), fresh.shape[0], bad[:10]))


def completeness_gates(gates, cells, arms, custom, fp=False, v1=False, fv=False):
    """[fix after fia1] MUST per requested cell: every expected gate of that cell is present (whatever its result),
    so a cell or an arm that never reported cannot pass by absence. [fia3] fp (the FP variant built): FP0's gates;
    v1 (V1 importable): V1_pre and C_V1 (C_V1 is a VOID REPORT when the precondition fails). [fia4] fv (the fork
    imported): FV's gates. [fia5] each swept FV arm of arms: its first-call marker R_<arm>_first_call_<tag>, and when
    that call ran, FV's gates under the arm's name (fv_gate_names)."""
    have = {g["name"] for g in gates}
    by_name = {g["name"]: g for g in gates}
    out = []
    for c in cells:
        b, rpr, src = cell_parts(c)
        tag = cell_tag(b, rpr, src)
        want = ["%s_%s_%s" % (p, a, tag) for a in MUST_ARMS if a in arms
                for p in ("I1_alone", "I2_order", "I3_siblings", "I4_split_switch", "I5_poison", "C")]
        if "FH" in arms:
            want.append("N_heuristic_negative_%s" % tag)
        if custom and "F8" in arms:
            want.append("CW_bias_witness_%s" % tag)
        if fp and "FP0" in arms:
            want += ["G_block0_%s" % tag, "C_FP0_%s" % tag, "MW_FP0_%s" % tag]
            want += ["%s_FP0_%s" % (p, tag) for p in ("FI1_request_alone", "FI2_order", "FI3_other_requests",
                                                      "FI5_poison")]
            if custom:
                want.append("CW_FP0_%s" % tag)
        if v1 and "V1" in arms:
            want += ["V1_pre_%s" % tag, "C_V1_%s" % tag]
        if fv and "FV" in arms:                                           # [fia4]
            want += fv_gate_names("FV", tag, custom)
        if fv:                                                            # [fia5] the swept arms
            for x in [s for s in arms if fv_sweep_launch(s) is not None]:
                first = by_name.get("R_%s_first_call_%s" % (x, tag))
                if first is None:
                    want.append("R_%s_first_call_%s" % (x, tag))
                elif first["passed"]:
                    want += [w for w in fv_gate_names(x, tag, custom) if w not in want]
        miss = [w for w in want if w not in have]
        out.append(gate("COMPLETE_%s" % tag, not miss, True, "%d of %d expected gates present; missing %s"
                        % (len(want) - len(miss), len(want), miss[:8])))
    return out


def fv_gate_names(name, tag, custom):
    """[fia4] FV's expected gates in one cell ([fia5] the same set for a swept arm, under its own name)."""
    out = ["G_block0_%s" % tag, "C_%s_%s" % (name, tag), "MW_%s_%s" % (name, tag), "U_%s_table_%s" % (name, tag)]
    out += ["%s_%s_%s" % (p, name, tag) for p in ("FI1_request_alone", "FI2_order", "FI3_other_requests", "FI5_poison")]
    if custom:
        out.append("CW_%s_%s" % (name, tag))
    return out


def bits_vs(name, tag, out, ref_out):
    """D (REPORT): rows / elements of out that differ bitwise from piece 3's, and the max |difference|."""
    it = torch.int16
    neq = out.contiguous().view(it) != ref_out.contiguous().view(it)
    rows = int(neq.flatten(1).any(1).sum())
    diff = float((out.float() - ref_out.float()).abs().max())
    return gate("R_D_bits_vs_P_%s_%s" % (name, tag), True, False,
                "%d of %d rows differ from piece 3 (%.4f of elements); max |d| %.3e; %s"
                % (rows, out.shape[0], float(neq.float().mean()), diff, PRED["bits_vs_p3"].get(name, "no prediction")))


# ------------------------------------------------------------------ [fia3] gates of the request-unit arms
def request_suite(name, tag, arm, bf, bq, rpr, alone_reqs, perms, poisoned, poison_pre, must):
    """FI1 - FI3 and FI5 for an arm whose unit is the REQUEST (its rows share one union, so a row alone is NOT a unit:
    see R_X_row_alone). FI1: each sampled request alone (a batch of one request, its own plan) == the same rows in the
    batch; FI2: the requests in reversed and in random order (rows keep their order inside a request) == unpermuted;
    FI3: the even requests of the altreq batch (odd requests: new q, new lists) == the full batch's (precondition: the
    changed rows are exactly the odd requests' rows); FI5: unread store rows NaN, output finite and bitwise equal, a
    MUST only with poison_pre (no loaded row is unread), else a VOID REPORT. Returns (gates, the full output)."""
    gates = []
    full = arm(bf)
    r_ = full.shape[0]

    def rows_of(n):
        return [n * rpr + u for u in range(rpr)]
    bad = []
    for n in alone_reqs:
        o = arm(select(bf, rows_of(n)))
        bad += [r for j, r in enumerate(rows_of(n)) if not bits_equal(o[j], full[r])]
    gates.append(gate("FI1_request_alone_%s_%s" % (name, tag), not bad, must,
                      "%d requests alone vs in the batch of %d rows; differing rows %s" % (len(alone_reqs), r_, bad[:10])))
    bad, det = [], []
    for pname, preq in perms:
        rows_ = [r for n in preq for r in rows_of(n)]
        o = arm(select(bf, rows_))
        d = [r for j, r in enumerate(rows_) if not bits_equal(o[j], full[r])]
        det.append("%s %d" % (pname, len(d)))
        bad += d
    gates.append(gate("FI2_order_%s_%s" % (name, tag), not bad, must,
                      "rows differing per request permutation: %s; first %s" % (", ".join(det), sorted(set(bad))[:10])))
    oa = arm(bq)
    changed = changed_rows(bf, bq)
    creq = sorted({r // rpr for r in changed})
    pre = bool(changed) and all(n % 2 == 1 for n in creq)
    keep = [r for r in range(r_) if (r // rpr) % 2 == 0]
    bad = [r for r in keep if not bits_equal(oa[r], full[r])]
    gates.append(gate("FI3_other_requests_%s_%s" % (name, tag), pre and not bad, must,
                      "%s; %d odd requests changed (q and lists), %d rows of even requests compared; differing %s"
                      % ("precondition met" if pre else "VOID: altreq did not change exactly odd requests",
                         len(creq), len(keep), bad[:10]), status=None if pre else "VOID"))
    bp = dict(bf)
    bp.update(poisoned)
    op = arm(bp)
    finite = bool(torch.isfinite(op.float()).all())
    bad = [r for r in range(r_) if not bits_equal(op[r], full[r])]
    pre5, txt5 = poison_pre
    gates.append(gate("FI5_poison_%s_%s" % (name, tag), finite and not bad, must and pre5,
                      "%s; unread rows NaN in K / V / bias: output finite %s; differing %s"
                      % ("precondition met (%s)" % txt5 if pre5 else "VOID REPORT: %s" % txt5, finite, bad[:10]),
                      status=None if pre5 else "VOID"))
    return gates, full


def block0_gate(tag, batches, rpr, s_slots, must):
    """G_block0 (the precondition of every FlashInfer arm with a custom mask): every listed (row, KV head) lists block 0,
    so union page 0 of every (request, head) is block 0 and every row sees tokens 0 .. 63 of the FIRST KV tile. The
    kernel starts the online softmax at m = -inf, d = 1 with no -inf guard (0.6.8 prefill.cuh:415-421, :878-907): a
    row whose first tile is fully masked gets exp2(-inf - -inf) = NaN. Also reported: (request, head, row) cells with
    no visible token in union page 0 and in the first 128-token KV tile (A100, MODEL kv_tile_tokens)."""
    miss = f64 = f128 = cells = 0
    for b in batches:
        pk = b["page_key"].to(torch.int64)
        npg = b["num_pages"].to(torch.int64)
        valid = torch.arange(pk.shape[-1], device=pk.device).view(1, 1, -1) < npg.unsqueeze(-1)
        miss += int((~((pk == 0) & valid).any(-1)).sum())
        ext = fp_union(b, rpr, s_slots)["ext"]
        f64 += int((ext[..., 0] == 0).sum())
        e2 = ext[..., 0] + (ext[..., 1] if ext.shape[-1] > 1 else 0)
        f128 += int((e2 == 0).sum())
        cells += int(ext[..., 0].numel())
    ok = miss == 0 and f64 == 0
    return gate("G_block0_%s" % tag, ok, must,
                "%d batches: (row, KV head) lists without block 0: %d; (request, head, row) cells with no visible token "
                "in union page 0: %d, in the first %d-token KV tile: %d (of %d cells)"
                % (len(batches), miss, f64, KV_TILE_PAGES * PAGE, f128, cells))


def mask_witness(tag, arm_ones, bf, rows, ref, u_rows, tol, must, gname=None):
    """MW: FP0 with every row given the pair's whole reach (no holes) must FAIL C against the reference, so FP0's C
    pass comes from the mask the variant applies. VOID when no row misses a union page (a mask with no hole: a no-op
    plant, L34). [fia4] gname: the gate's name (default the FP0 name; FV: MW_FV_<tag>)."""
    gname = gname or "MW_FP0_%s" % tag
    j = torch.arange(u_rows["ext"].shape[-1], device=u_rows["ext"].device).view(1, 1, 1, -1)
    live = j < u_rows["ucnt"].unsqueeze(-1).unsqueeze(-1)
    holes = int((live & (u_rows["ext"] < u_rows["ext"].amax(dim=2, keepdim=True))).sum())
    if holes == 0:
        return gate(gname, False, must, "VOID: no row misses a union token (the mask has no hole)",
                    status="VOID")
    o = arm_ones(bf)[torch.as_tensor(rows, device=bf["q"].device)]
    err = float((o.double() - ref.to(o.device)).abs().max())
    st = "DETECTED" if err > tol else "BLIND"
    return gate(gname, st == "DETECTED", must,
                "%s: every row given the whole union vs the reference %.3e (tol %.1e); %d (request, head, row, page) "
                "cells see fewer tokens than the reach" % (st, err, tol, holes), status=st)


def row_alone_probe(name, tag, arm, bf, rows, rpr, full=None):
    """R_X_row_alone (REPORT; the EXACT-mode note): each sampled row alone, padded to rpr rows by repeating it (its own
    list is then the union, CTA_TILE_Q stays 64), vs the same row in its request. Equal bits would need KV tiles that
    never hold two pages; FlashInfer's A100 tile is 128 tokens = 2 pages (kv_tile_tokens; the dispatch has no public
    option), so a row's bits depend on its siblings' pages: FP0 is FAST-only."""
    full = arm(bf) if full is None else full
    diff = []
    for r in rows:
        o = arm(select(bf, [r] * rpr))
        if not bits_equal(o[0], full[r]):
            diff.append(r)
    return gate("R_X_row_alone_%s_%s" % (name, tag), True, False,
                "%d of %d sampled rows differ alone (padded to %d identical rows) vs in their request; KV tile %d "
                "tokens (MODEL): %s" % (len(diff), len(rows), rpr, KV_TILE_PAGES * PAGE,
                                        "FAST-only, as expected" if diff else "no difference seen on these rows"))


def fp_reuse_check(name, tag, arm, b_plan, b_layer):
    """R_REUSE (REPORT): plan on b_plan, write b_layer's union, CSR and mask IN PLACE (relayer), run; bitwise vs a
    fresh plan on b_layer. Precondition: some request's union size differs between the two (else VOID)."""
    arm.prepare(b_plan)
    uc_a = arm.u["ucnt"].clone()
    arm.relayer(b_layer)
    reused = arm.launch().clone()
    nd = int((uc_a != arm.u["ucnt"]).sum())
    fresh = arm(b_layer)
    bad = [r for r in range(fresh.shape[0]) if not bits_equal(reused[r], fresh[r])]
    return gate("R_REUSE_%s_%s" % (name, tag), nd > 0 and not bad, False,
                "%s; one plan, the next layer's union / CSR / mask written in place: %d of %d rows differ from a fresh "
                "plan %s" % ("precondition met (%d (head, request) union sizes differ)" % nd if nd else
                             "VOID: no union size differs", len(bad), fresh.shape[0], bad[:10]),
                status=None if nd else "VOID")


def decision_report(report, gates, cells):
    """R_DECISION (REPORT; the lead scores it): PRED3's rule per registered cell from the ncu device times (P, FP0, FPU),
    the L2 -> SM bytes and the measured tick overhead (one plan + 32 layer builds: eager events, and CUDA-graph
    replays)."""
    th = PRED3["thresholds"]
    out = []
    for c in cells:
        b, rpr, src = cell_parts(c)
        tag = cell_tag(b, rpr, src)
        if tag not in PRED3["fp0_over_p_device"]:
            continue
        arms = (((report.get("ncu") or {}).get(tag) or {}).get("summary") or {}).get("arms") or {}
        p, f, fu = arms.get("P"), arms.get("FP0"), arms.get("FPU")
        ratio = f["time_ns"] / p["time_ns"] if p and f and p.get("time_ns") else None
        l2 = (p["l2sm_bytes"] / f["l2sm_bytes"]) if p and f and p.get("l2sm_bytes") and f.get("l2sm_bytes") else None
        mshare = (f["time_ns"] - fu["time_ns"]) / f["time_ns"] if f and fu and f.get("time_ns") else None
        t = (((report.get("cells") or {}).get(tag) or {}).get("arms") or {}).get("FP0") or {}
        ove, ovg = t.get("tick_overhead_eager_ms"), t.get("tick_overhead_graph_ms")
        mine = [g for g in gates if g["must"] and g["name"].endswith("_" + tag)
                and ("FP0" in g["name"] or g["name"].startswith("G_block0"))]
        gates_ok = bool(mine) and all(g["passed"] for g in mine)
        time_ok = ratio is not None and ratio <= th["fp0_over_p_max"]
        ok_e = ove is not None and ove <= th["tick_overhead_ms_max"]
        ok_g = ovg is not None and ovg <= th["tick_overhead_ms_max"]
        verdict = "ADOPT" if (time_ok and gates_ok and (ok_e or ok_g)) else "NOT MET"
        flags = []
        if ratio is not None and not time_ok and l2 is not None and l2 >= th["hbm_bound_l2sm_drop"]:
            flags.append("HBM-bound (L2->SM fell %.2fx, time did not): stop at the 15 %% bound" % l2)
        if not (ok_e or ok_g):
            flags.append("plan / build bound (tick overhead above %.1f ms)" % th["tick_overhead_ms_max"])
        if mshare is not None:
            flags.append("mask share of FP0's device time %.3f (FP0 - FPU)" % mshare)
        out.append(gate("R_DECISION_%s" % tag, True, False,
                        "%s: FP0 / P device %s (%s); L2->SM P / FP0 %s; tick overhead eager %s ms, graph %s ms "
                        "(rule <= %.1f); FP0 MUST gates %s; %s"
                        % (verdict, "%.4f" % ratio if ratio is not None else "n/a",
                           band_text(ratio, PRED3["fp0_over_p_device"][tag]),
                           "%.3f" % l2 if l2 is not None else "n/a",
                           "%.4f" % ove if ove is not None else "n/a", "%.4f" % ovg if ovg is not None else "n/a",
                           th["tick_overhead_ms_max"], "PASS" if gates_ok else "NOT ALL PASS",
                           "; ".join(flags) or "no flag")))
    return out


# ------------------------------------------------------------------ [fia4] gates and the decision of the fork
def fv_table_gate(tag, arm, batches, rpr, s_slots, must, gname=None):
    """U_FV_table (MUST): after an FV call on each batch, the tables the KERNEL wrote == the torch tables
    (fv_tables_from_union over fp_union) on each union's valid prefix, and the lengths are equal. Precondition: the
    rows are sorted (fv_sorted_rows; the in-kernel build needs piece 3's canonical order), else VOID (= fail).
    [fia5] gname: the gate's name (default U_FV_table_<tag>; a swept arm: U_<arm>_table_<tag>)."""
    gname = gname or "U_FV_table_%s" % tag
    det, bad, pre = [], 0, True
    for name, b in batches:
        ok, txt = fv_sorted_rows(b)
        pre = pre and ok
        arm(b)
        kt, kx, kn = arm.tables()
        t, x, n = fv_tables_from_union(fp_union(b, rpr, s_slots), arm.bufs["w"])
        live = torch.arange(t.shape[-1], device=t.device).view(1, 1, -1) < n.unsqueeze(-1).to(torch.int64)
        dl, dt, dx = int((kn != n).sum()), int(((kt != t) & live).sum()), int(((kx != x) & live).sum())
        bad += dl + dt + dx
        # [fia4 fix] the first differing entries: (KV head, request, union index) kernel / torch (words in hex)
        first = ["(%d, %d, %d) page %d / %d, word %#x / %#x" % (h, q, j, int(kt[h, q, j]), int(t[h, q, j]),
                                                                int(kx[h, q, j]) & 0xFFFFFFFF, int(x[h, q, j]) & 0xFFFFFFFF)
                 for h, q, j in (((kt != t) | (kx != x)) & live).nonzero().tolist()[:4]]
        det.append("%s: %d lengths, %d pages, %d extent words differ of %d entries (%s)%s"
                   % (name, dl, dt, dx, int(live.sum()), txt, "; first: " + ", ".join(first) if first else ""))
    return gate(gname, pre and bad == 0, must, "%s; %s" % (
        "precondition met (sorted rows)" if pre else "VOID: unsorted rows", "; ".join(det)),
        status=None if pre else "VOID")


def fv_row_alone_gate(tag, arm, bf, rows, rpr, full, gname=None):
    """X_row_alone_FV (REPORT; the EXACT probe): each sampled row alone (rpr copies of it: its own list is then the
    union) vs the same row in its request, bitwise. Expected: no difference (a tile = one block, ascending block
    order; a block the row does not list is an exact identity step once block 0 was seen, G_block0). [fia5] also with
    a tile that is part of a block (a tile the row does not see is an identity step too); gname as fv_table_gate."""
    diff = []
    for r in rows:
        o = arm(select(bf, [r] * rpr))
        if not bits_equal(o[0], full[r]):
            diff.append(r)
    return gate(gname or "X_row_alone_FV_%s" % tag, not diff, False,
                "%d of %d sampled rows differ alone (%d copies) vs in their request (EXACT expected by construction); "
                "differing %s" % (len(diff), len(rows), rpr, diff[:10]))


def fv_graph_probe(tag, arm, b, gname=None):
    """R_GRAPH_FV (REPORT): one FV launch (the in-kernel union build included) captured in a CUDA graph after a
    side-stream warm-up (torch.cuda.graphs); the replay into a NaN-filled output == the eager launch, bitwise; replay
    time. A capture error is a REPORT field. [fia5] gname as fv_table_gate."""
    gname = gname or "R_GRAPH_FV_%s" % tag
    if not (torch.cuda.is_available() and b["q"].is_cuda):
        return gate(gname, False, False, "not run: no CUDA tensors")
    try:
        arm.prepare(b)
        eager = arm.launch().clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                arm.launch()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            arm.launch()
        arm._out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        same = bits_equal(arm._out, eager)
        t = cuda_time(g.replay, 5, 50)
        del g
        return gate(gname, same, False, "one FV launch in a CUDA graph: replay bitwise equal to eager "
                    "%s; replay %.4f ms per layer" % (same, t["median"]))
    except Exception as e:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return gate(gname, False, False, "capture failed: %s: %s" % (type(e).__name__, str(e)[:300]))


def decision_report4(report, gates, cells):
    """R_DECISION4 (REPORT; the lead scores it): PRED4's rule per registered cell. FV / P from the ncu device times;
    the union build per tick = 32 x (FV - FVT) ncu device time (FV builds in the kernel, FVT reads a torch table), or
    the event-time difference when ncu has no FVT; the FV MUST gates of the cell; plus the ncu split for the record."""
    th = PRED4["thresholds"]
    out = []
    for c in cells:
        b, rpr, src = cell_parts(c)
        tag = cell_tag(b, rpr, src)
        if tag not in PRED4["fv_over_p_device"]:
            continue
        arms = (((report.get("ncu") or {}).get(tag) or {}).get("summary") or {}).get("arms") or {}
        p, f, ft = arms.get("P"), arms.get("FV"), arms.get("FVT")
        ratio = f["time_ns"] / p["time_ns"] if p and f and p.get("time_ns") else None
        rt = ft["time_ns"] / p["time_ns"] if p and ft and p.get("time_ns") else None
        b_ncu = (th["layers_per_tick"] * max(f["time_ns"] - ft["time_ns"], 0.0) / 1e6
                 if f and ft and f.get("time_ns") and ft.get("time_ns") else None)
        t = (((report.get("cells") or {}).get(tag) or {}).get("arms") or {}).get("FV") or {}
        b_ev = t.get("build_in_kernel_ms_per_tick_events")
        build = b_ncu if b_ncu is not None else b_ev
        mine = [g for g in gates if g["must"] and g["name"].endswith("_" + tag)
                and ("_FV_" in g["name"] or g["name"].startswith("G_block0"))]
        gates_ok = bool(mine) and all(g["passed"] for g in mine)
        time_ok = ratio is not None and ratio <= th["fv_over_p_max"]
        build_ok = build is not None and build <= th["union_build_ms_per_tick_max"]
        verdict = "ADOPT" if (time_ok and build_ok and gates_ok) else "NOT MET"
        split = "n/a"
        if f:
            split = "registers %s, local memory %s B, achieved occupancy %s %%, shared memory %s B, DRAM / union %s, " \
                    "L2->SM / union %s" % (sorted(set(f.get("regs") or [])), f.get("local_bytes"), f.get("occ"),
                                           f.get("smem"), f.get("read_over_union"), f.get("l2sm_over_union"))
        out.append(gate("R_DECISION4_%s" % tag, True, False,
                        "%s: FV / P device %s (%s); union build per tick %s ms (ncu %s, events %s; rule <= %.1f); FV "
                        "MUST gates %s; FVT / P %s; FV ncu: %s"
                        % (verdict, "%.4f" % ratio if ratio is not None else "n/a",
                           band_text(ratio, PRED4["fv_over_p_device"][tag]),
                           "%.4f" % build if build is not None else "n/a",
                           "%.4f" % b_ncu if b_ncu is not None else "n/a",
                           "%.4f" % b_ev if b_ev is not None else "n/a", th["union_build_ms_per_tick_max"],
                           "PASS" if gates_ok else "NOT ALL PASS", "%.4f" % rt if rt is not None else "n/a", split)))
    return out


def fv_must_gates(gates, name, tag):
    """[fia5] The MUST gates of one fork arm in one cell (its name between underscores, e.g. C_FV_216x4,
    U_FVW4S3T64B32_table_216x4) and G_block0 of the cell."""
    key = "_%s_" % name
    return [g for g in gates if g["must"] and g["name"].endswith("_" + tag)
            and (key in g["name"] or g["name"].startswith("G_block0"))]


def fv_gates_ok(gates, name, tag):
    """[fia5] True when the arm has MUST gates of its OWN in the cell (G_block0 alone does not count: an arm that
    ran no gate is never "passing") and every MUST gate of fv_must_gates passed."""
    mine = fv_must_gates(gates, name, tag)
    own = [g for g in mine if "_%s_" % name in g["name"]]
    return bool(own) and all(g["passed"] for g in mine)


def fv_sweep_report(rep, tag, gates, names):
    """[fia5] R_FV_sweep_<tag> (REPORT): per swept arm of the cell the event median, x of P, its MUST gates and its
    compile witness; rep["fv_best"] = the lowest event median among the swept arms whose MUST gates all pass (None
    when none). The ncu child profiles it next to FV."""
    arms = rep.get("arms") or {}
    p = (arms.get("P") or {}).get("call_ms", {}).get("median")
    comp = rep.get("fv_compile") or {}
    rows, cands = [], []
    for x in ["FV", "FVN"] + list(names):
        t = (arms.get(x) or {}).get("call_ms", {}).get("median")
        ok = x != "FVN" and fv_gates_ok(gates, x, tag)
        c = comp.get(x) or {}
        rows.append("%s %s ms (x%s of P; MUST %s; regs %s, local words %s, shared %s, depth %s)" % (
            x, "%.4f" % t if t is not None else "n/a", "%.3f" % (t / p) if (t is not None and p) else "n/a",
            "n/a" if x == "FVN" else ("PASS" if ok else "NOT ALL PASS"), c.get("regs"),
            c.get("local_words_per_thread"), c.get("shared"), c.get("kv_depth")))
        if x in names and ok and t is not None:
            cands.append((t, x))
    rep["fv_best"] = min(cands)[1] if cands else None
    gates.append(gate("R_FV_sweep_%s" % tag, rep["fv_best"] is not None, False,
                      "best swept arm %s (lowest event median among the swept arms with all MUST gates passing; ncu "
                      "profiles it); %s" % (rep["fv_best"], "; ".join(rows))))
    return rep["fv_best"]


def decision_report5(report, gates, cells):
    """[fia5] R_DECISION5 (REPORT; the lead scores it): PRED5's rule per registered cell for the default FV and the
    cell's best swept arm: X / P from the ncu device times; the union build per tick = 32 x (FV - FVT) ncu (the default
    launch; else the event difference); X's MUST gates; the extents' share of FV's device time ((FV - FVN) / FV);
    the ncu split and the compile witness of each."""
    th = PRED5["thresholds"]
    out = []
    for c in cells:
        b, rpr, src = cell_parts(c)
        tag = cell_tag(b, rpr, src)
        if tag not in PRED5["fv_over_p_device"]:
            continue
        crep = (report.get("cells") or {}).get(tag) or {}
        arms = (((report.get("ncu") or {}).get(tag) or {}).get("summary") or {}).get("arms") or {}
        p, f, ft, fn = arms.get("P"), arms.get("FV"), arms.get("FVT"), arms.get("FVN")
        best = crep.get("fv_best")
        fb = arms.get(best) if best else None
        tp = p.get("time_ns") if p else None

        def ratio(a):
            return a["time_ns"] / tp if (a and tp and a.get("time_ns")) else None
        b_ncu = (th["layers_per_tick"] * max(f["time_ns"] - ft["time_ns"], 0.0) / 1e6
                 if f and ft and f.get("time_ns") and ft.get("time_ns") else None)
        b_ev = ((crep.get("arms") or {}).get("FV") or {}).get("build_in_kernel_ms_per_tick_events")
        build = b_ncu if b_ncu is not None else b_ev
        build_ok = build is not None and build <= th["union_build_ms_per_tick_max"]
        mshare = (f["time_ns"] - fn["time_ns"]) / f["time_ns"] if (f and fn and f.get("time_ns")) else None
        comp = crep.get("fv_compile") or {}
        parts, ok_cfg = [], []
        for x, a, band in (("FV", f, PRED5["fv_over_p_device"][tag]),
                           (best, fb, PRED5["best_swept_over_p_device"].get(tag))):
            if x is None:
                parts.append("best swept: none (no swept arm with all MUST gates passing, or no sweep)")
                continue
            r = ratio(a)
            g_ok = fv_gates_ok(gates, x, tag)
            if r is not None and r <= th["fv_over_p_max"] and g_ok and build_ok:
                ok_cfg.append((r, x))
            cw = comp.get(x) or {}
            parts.append("%s / P device %s (%s); MUST gates %s; ncu registers %s, local memory %s B, achieved "
                         "occupancy %s %%, shared memory %s B, DRAM / union %s; compile: depth %s, programs per SM "
                         "(MODEL) %s" % (
                             x, "%.4f" % r if r is not None else "n/a", band_text(r, band) if band else "no band",
                             "PASS" if g_ok else "NOT ALL PASS", sorted(set((a or {}).get("regs") or [])),
                             (a or {}).get("local_bytes"), (a or {}).get("occ"), (a or {}).get("smem"),
                             (a or {}).get("read_over_union"), cw.get("kv_depth"),
                             (cw.get("programs_per_sm") or {}).get("model")))
        verdict = ("ADOPT %s" % min(ok_cfg)[1]) if ok_cfg else "NOT MET"
        out.append(gate("R_DECISION5_%s" % tag, True, False,
                        "%s: %s; union build per tick %s ms (ncu %s, events %s; rule <= %.1f); extents' share of FV "
                        "(FV - FVN) / FV %s"
                        % (verdict, " | ".join(parts), "%.4f" % build if build is not None else "n/a",
                           "%.4f" % b_ncu if b_ncu is not None else "n/a", "%.4f" % b_ev if b_ev is not None else "n/a",
                           th["union_build_ms_per_tick_max"], "%.4f" % mshare if mshare is not None else "n/a")))
    return out


# ------------------------------------------------------------------ timing (GPU)
def summarize_ms(xs):
    s = sorted(xs)
    n = len(s)
    return dict(median=round(statistics.median(s), 5), q1=round(s[n // 4], 5), q3=round(s[(3 * n) // 4], 5), n=n)


def cuda_time(fn, warm, reps):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        xs.append(e0.elapsed_time(e1))
    return summarize_ms(xs)


def host_time(fn, reps):
    xs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1e3)
    torch.cuda.synchronize()
    return summarize_ms(xs)


def peak_delta(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - base)


def fi_float_used(info, g, d):
    """Float workspace bytes a plan reserves (scheduler.cuh:750-755: the allocator sizes tmp_v with sizeof(float),
    although the kernel stores DTypeO partials into it, plus tmp_s); 0 without split_kv."""
    if not info["split_kv"]:
        return 0
    return g * info["padded_batch_size"] * info["cta_tile_q"] * d * 4 + g * info["padded_batch_size"] * info["cta_tile_q"] * 4


def time_arm(name, arm, b, warm, reps, geom):
    rep = dict(arm=name)
    arm.prepare(b)
    torch.cuda.synchronize()
    rep["call_ms"] = cuda_time(arm.launch, warm, reps)
    if isinstance(arm, FIArm) and not isinstance(arm, FIStridedArm):
        rep["runs_only_ms"] = cuda_time(arm.launch_runs, warm, reps)
        packed = arm.pack(b)
        host = [(ip.cpu(), ix, ll.cpu()) for ip, ix, ll in packed]
        rep["plan_host_indptr_ms_per_tick"] = host_time(lambda: arm.plan_packed(host), 50)
        rep["plan_device_indptr_ms_per_tick"] = host_time(lambda: arm.plan_packed(packed), 50)
        rep["pack_ms_per_layer"] = cuda_time(lambda: arm.pack(b), 5, 50)          # the ragged path, both heads
        rep["plan_info"] = arm.info
        rep["float_ws_used_bytes"] = max(fi_float_used(i, geom["G"], geom["D"]) for i in arm.info)
        rep["pinned_host_bytes"] = arm.pinned_bytes
        arm.prepare(b)
        rep["float_ws_need_bytes"] = arm.float_need                              # FlashInfer's formula (the cost)
        rep["float_ws_alloc_bytes"] = arm.fws.numel()
        rep["float_ws_need_covers_use"] = arm.float_need >= rep["float_ws_used_bytes"]
        if arm.fixed_counts:                                                     # one plan per tick, every layer
            rep["pack_fixed_ms_per_layer"] = cuda_time(lambda: pack_fixed(b, arm.S, arm._ix), 5, 50)

            def layer():
                arm.relayer(b)
                return arm.launch()
            rep["layer_with_plan_reuse_ms"] = cuda_time(layer, warm, reps)
    if isinstance(arm, FPArm):
        time_fp_extras(rep, arm, b, warm, reps)
    if isinstance(arm, FVArm):                                                  # [fia4]
        time_fv_extras(rep, arm, b, warm, reps)
    if isinstance(arm, V1Arm):
        rep["gather_ms"] = cuda_time(lambda: arm.stage_gather(b), 2, 20)       # union + store copy (one host read)
        rep["prep_ms"] = cuda_time(arm.stage_prep, 2, 20)                     # exp(cis) values
        rep["store_W"] = arm.a["W"]
    rep["workspace_bytes"] = arm.workspace_bytes
    rep["peak_alloc_delta_bytes"] = peak_delta(arm.launch)
    return rep


def build_graph_ms(arm, b):
    """The union + CSR + mask build captured in ONE CUDA graph (it has no host read), replay time by CUDA events. The
    capture follows torch.cuda.graphs (warm-up on a side stream first). Returns (summary, None) or (None, error)."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                arm.build(b)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            arm.build(b)
        res = cuda_time(g.replay, 5, 50)
        torch.cuda.synchronize()
        del g
        return res, None
    except Exception as e:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return None, "%s: %s" % (type(e).__name__, str(e)[:300])


def time_fp_extras(rep, arm, b, warm, reps):
    """[fia3] FP arm host costs: the build per layer (events, host wall, CUDA graph), plan() per tick (device indptr:
    the tick's form; host indptr), the layer under the tick's plan (no split), the tick overhead (no split: one plan +
    32 builds; split: 32 x (plan + build)), the union size, the mask bytes, FlashInfer's mask mode."""
    rep["runs_only_ms"] = cuda_time(arm.launch_runs, warm, reps)
    rep["build_ms_per_layer"] = cuda_time(lambda: arm.build(b), 5, 50)
    rep["build_host_ms_per_layer"] = host_time(lambda: arm.build(b), 50)
    hc = arm.host_copies()
    rep["plan_host_indptr_ms_per_tick"] = host_time(lambda: arm.plan(host=hc), 50)
    arm.prepare(b)
    rep["plan_device_indptr_ms_per_tick"] = host_time(arm.plan, 50)
    rep["plan_info"] = arm.info
    rep["mask_mode_custom"] = [getattr(w, "_custom_mask_buf", None) is not None for w in arm.wr]
    rep["union"] = union_stats(arm.u, b)
    rep["mask_bytes"] = int(arm.bufs["mi"][:, -1].max()) - int(arm.bufs["mi"][:, 0].min())
    rep["float_ws_need_bytes"] = arm.float_need
    rep["float_ws_alloc_bytes"] = arm.fws.numel()
    rep["pinned_host_bytes"] = arm.pinned_bytes
    gms, gerr = build_graph_ms(arm, b)
    if gms is not None:
        rep["build_graph_ms_per_layer"] = gms
    else:
        rep["build_graph_error"] = gerr
    arm.prepare(b)                                                # the device-indptr plan on fresh buffers again
    nl = LAYERS_PER_TICK
    pl = rep["plan_device_indptr_ms_per_tick"]["median"]
    be, bh = rep["build_ms_per_layer"]["median"], rep["build_host_ms_per_layer"]["median"]
    bg = gms["median"] if gms is not None else None
    if arm.split == "none":
        def layer():
            arm.relayer(b)
            return arm.launch()
        rep["layer_with_plan_reuse_ms"] = cuda_time(layer, warm, reps)
        rep["tick_overhead_eager_ms"] = round(pl + nl * be, 5)
        rep["tick_overhead_host_ms"] = round(pl + nl * bh, 5)
        rep["tick_overhead_graph_ms"] = round(pl + nl * bg, 5) if bg is not None else None
    else:
        rep["tick_overhead_eager_ms"] = round(nl * (pl + be), 5)
        rep["tick_overhead_host_ms"] = round(nl * (pl + bh), 5)
        rep["tick_overhead_graph_ms"] = round(nl * (pl + bg), 5) if bg is not None else None


def time_fv_extras(rep, arm, b, warm, reps):
    """[fia4] FV / FVT host costs: the host time of one launch (Python + Triton's launcher; a CUDA graph removes it),
    and for FVT the torch build of the tables per layer (events, host wall, one CUDA graph) and per tick (32 layers).
    FV's in-kernel build is FV - FVT (set by run_cell when both ran)."""
    rep["launch_host_ms"] = host_time(arm.launch, 50)
    rep["build"] = "in the kernel (fork_build_union)" if arm.build_in_kernel else "torch (fp_union + tables)"
    if not arm.build_in_kernel:
        rep["build_ms_per_layer"] = cuda_time(lambda: arm.build(b), 5, 50)
        rep["build_host_ms_per_layer"] = host_time(lambda: arm.build(b), 50)
        gms, gerr = build_graph_ms(arm, b)
        if gms is not None:
            rep["build_graph_ms_per_layer"] = gms
        else:
            rep["build_graph_error"] = gerr
        nl = LAYERS_PER_TICK
        rep["tick_build_eager_ms"] = round(nl * rep["build_ms_per_layer"]["median"], 5)
        rep["tick_build_graph_ms"] = round(nl * gms["median"], 5) if gms is not None else None
        rep["union"] = union_stats(arm.u, b)
    rep["table_width"] = arm.bufs["w"]
    arm.prepare(b)


def fv_time_text(t):
    """[fia4] the FV arms' part of the R_T line."""
    txt = "; launch host %.4f ms; build %s" % (t["launch_host_ms"]["median"], t["build"])
    if "build_ms_per_layer" in t:
        g = t.get("build_graph_ms_per_layer")
        txt += (" per layer %.4f ms events, %.4f ms host, graph %s; per tick (%d layers) eager %.4f, graph %s ms; "
                "union %s" % (t["build_ms_per_layer"]["median"], t["build_host_ms_per_layer"]["median"],
                              "%.4f" % g["median"] if g else "n/a (%s)" % t.get("build_graph_error"), LAYERS_PER_TICK,
                              t["tick_build_eager_ms"],
                              "%.4f" % t["tick_build_graph_ms"] if t.get("tick_build_graph_ms") is not None else "n/a",
                              t["union"]))
    return txt + "; table width %d" % t["table_width"]


# ------------------------------------------------------------------ ncu (REPORT)
def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def ncu_rows(text, fields):
    """Kernel rows of an ncu raw CSV (--csv --page raw --print-units base): the header row naming 'Kernel Name', rows
    with a numeric ID; tag = the first fia|<arm> in any cell (the NVTX column)."""
    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if hi is None:
        raise ValueError("no header row with 'Kernel Name'")
    head = rows[hi]
    out = []
    for r in rows[hi + 1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        if not d.get("ID", "").strip().isdigit():
            continue
        tag = None
        for c in r:
            m = TAG_RE.search(c)
            if m:
                tag = m.group(1)
                break
        row = dict(id=int(d["ID"]), name=d["Kernel Name"], tag=tag)
        for f in fields:
            row[f] = _num(d.get(f, ""))
        out.append(row)
    return out


def ncu_summary(rows, arms, ideal, union, rpr):
    """Per arm: DRAM bytes read / written, kernel time, kernel count (vs KERNELS_PER_CALL), ratios to the ideal
    and the union bytes, the band (PRED dram_read_over_ideal), the reuse label."""
    per = {a: dict(read=0.0, write=0.0, time_ns=0.0, kernels=[], label=REUSE_LABEL.get(a) or (    # [fia5] swept
        "as FV, launch %s" % fv_sweep_launch(a) if fv_sweep_launch(a) else ""), l2sm_sectors=None,
                   regs=[], local_bytes=None, occ=[], smem=[], limits=[],                    # [fia4] occ, smem, limits
                   stall_lsb=[]) for a in arms}                                              # [fia5] long scoreboard
    stray = []
    for r in rows:
        if r["tag"] not in per or not any(k in r["name"] for k in ATTN_KERNELS):
            stray.append((r["tag"], r["name"][:60]))
            continue
        p = per[r["tag"]]
        p["read"] += r.get("dram__bytes_read.sum") or 0.0
        p["write"] += r.get("dram__bytes_write.sum") or 0.0
        p["time_ns"] += r.get("gpu__time_duration.sum") or 0.0
        p["kernels"].append(r["name"][:60])
        sec = r.get("lts__t_sectors_srcunit_tex_op_read.sum")              # [fia3] L2 -> SM reads
        if sec is not None:
            p["l2sm_sectors"] = (p["l2sm_sectors"] or 0.0) + sec
        if r.get("launch__registers_per_thread") is not None:
            p["regs"].append(int(r["launch__registers_per_thread"]))
        loc = [r.get(m) for m in ("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
                                  "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum") if r.get(m) is not None]
        if loc:
            p["local_bytes"] = (p["local_bytes"] or 0.0) + sum(loc)
        occ = r.get("sm__warps_active.avg.pct_of_peak_sustained_active")               # [fia4] achieved occupancy
        if occ is not None:
            p["occ"].append(round(occ, 2))
        lsb = r.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct")    # [fia5]
        if lsb is not None:
            p["stall_lsb"].append(round(lsb, 2))
        sm = [r.get(m) for m in ("launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static")
              if r.get(m) is not None]
        if sm:
            p["smem"].append(int(sum(sm)))
        lim = tuple(r.get(m) for m in ("launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
                                       "launch__waves_per_multiprocessor"))
        if any(x is not None for x in lim):
            p["limits"].append(lim)
    for a, p in per.items():
        p["n_kernels"] = len(p["kernels"])
        p["expected_kernels"] = KERNELS_PER_CALL.get(a, 1 if fv_sweep_launch(a) is not None else None)   # [fia5]
        p["read_over_ideal"] = p["read"] / ideal if ideal else None
        p["read_over_union"] = p["read"] / union if union else None
        bands = PRED["dram_read_over_ideal"].get("4" if rpr > 1 else "1", {})
        p["band"] = band_text(p["read_over_ideal"], bands[a]) if a in bands else "no band"
        p["l2sm_bytes"] = p["l2sm_sectors"] * L2_SECTOR_BYTES if p["l2sm_sectors"] is not None else None
        p["l2sm_over_union"] = p["l2sm_bytes"] / union if (p["l2sm_bytes"] is not None and union) else None
        p["l2sm_over_ideal"] = p["l2sm_bytes"] / ideal if (p["l2sm_bytes"] is not None and ideal) else None
        if a == "FP0":                                                      # [fia3] PRED3 bands
            p["band3"] = "DRAM/union %s; L2->SM/union %s" % (
                band_text(p["read_over_union"], PRED3["fp0_dram_read_over_union"]),
                band_text(p["l2sm_over_union"], PRED3["fp0_l2sm_read_over_union"]))
        elif a == "P":
            p["band3"] = "L2->SM/sum %s" % band_text(p["l2sm_over_ideal"], PRED3["p_l2sm_read_over_sum"])
    return dict(arms=per, stray=stray[:20], ideal_bytes=ideal, union_bytes=union)


def ncu_cap(left_s, cap_s):
    """Seconds for one ncu child, or None (skip) when the budget left cannot cover a minimal child + the margin."""
    room = left_s - NCU_MARGIN_S
    if room < 60:
        return None
    return int(min(cap_s, room))


def ncu_argv(a, b, rpr, child_out, rep, metrics, arms, bias_free, src="gen"):
    argv = [a.ncu, "--profile-from-start", "off", "--replay-mode", "kernel", "--cache-control", "all",
            "--clock-control", "none", "--nvtx", "--kernel-name", KERNEL_FILTER, "--metrics", ",".join(metrics)]
    for s in NCU_SECTIONS:                                                  # [fia3] registers per thread
        argv += ["--section", s]
    argv += ["--target-processes", "application-only", "-f", "-o", rep,
             sys.executable, os.path.abspath(__file__), "--mode", "ncu-cell", "--out", child_out, "--config", a.config,
             "--biattn", a.biattn, "--cells", cell_tag(b, rpr, src), "--seed", str(a.seed), "--arms", ",".join(arms)]
    if "S5" in arms:
        argv += ["--sharedkv", a.sharedkv]
    if "V1" in arms:
        argv += ["--verify-step", getattr(a, "verify_step", None)]
    if any(is_fv_arm(x) for x in arms):                                      # [fia4] [fia5] a swept arm too
        argv += ["--fv-kernel", getattr(a, "fv_kernel", None)]
    if src == "s1":
        argv += ["--s1-tables", getattr(a, "s1_tables", None)]
    if bias_free:
        argv += ["--bias-free"]
    return argv


def run_logged(argv, path, timeout):
    """argv with stdout + stderr to path, in its own process group, killed at the timeout. The exit or 'timeout'."""
    with open(path, "w") as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            return "timeout"


def ncu_metrics(a, out):
    """ncu --query-metrics --chip ga100 (saved): the required names present, the optional kept when present."""
    path = os.path.join(out, "ncu_metrics_query.txt")
    rc = run_logged([a.ncu, "--query-metrics", "--chip", "ga100"], path, 120)
    text = open(path, errors="replace").read() if os.path.exists(path) else ""
    names = set(re.findall(r"^\s*([a-z0-9_]+__[a-z0-9_]+)", text, flags=re.M))
    base = lambda m: m.split(".", 1)[0]          # noqa: E731  [fia3] ".avg.pct_of_peak..." has two dots
    missing = [m for m in NCU_REQUIRED if base(m) not in names]
    kept = [m for m in NCU_OPTIONAL if base(m) in names]
    return rc, missing, list(NCU_REQUIRED) + kept


# ------------------------------------------------------------------ the cell (main mode)
FP_MODES = dict(FP0=("none", "rows"), FP8=("fixed", "rows"), FPU=("none", "off"))


def build_arms(fi, a, geom, case, p3, skv, jit, arms, device, jit_fp=None, vmod=None, fvmod=None):
    out = {}
    rmax = case["R"] + len(SHORT_COUNTS)
    for name in arms:
        if name == "P":
            out[name] = P3Arm(p3, geom, rmax, case["P"], device)
        elif name == "S5":
            if skv is not None and case["rpr"] > 1:
                out[name] = S5Arm(skv, p3, geom, rmax, case["P"], device)
        elif name in FP_ARMS:                                   # [fia3] only when the FP variant built
            if jit_fp is not None:
                split, mask = FP_MODES[name]
                out[name] = FPArm(fi, name, split, mask, geom, case["S"], case["rpr"], jit_fp, device)
        elif name == "V1":                                      # [fia3] only when V1 imports
            if vmod is not None:
                out[name] = V1Arm(vmod, geom, case["rpr"], case["S"])
        elif is_fv_arm(name):                                   # [fia4] only when the fork imports ([fia5] FVN, swept)
            if fvmod is not None and case["rpr"] <= FV_ROWS:
                out[name] = make_fv_arm(fvmod, name, geom, case["S"], case["rpr"], device)
        else:
            split = dict(F8="fixed", F0="none", FH="heuristic")[name]
            out[name] = FIArm(fi, name, split, geom, case["S"], jit, device)
    return out


def make_cell_case(geom, b, rpr, src, seed, s1_groups, with_data=True):
    """[fia3] the cell's case: the generator or the recorded S1 lists; K / V / bias on the GPU for B > 64."""
    dd = "cuda" if (with_data and b > CPU_DATA_MAX_B) else "cpu"
    if src == "s1":
        return make_case_s1(geom, b, rpr, seed, s1_groups, with_data=with_data, data_device=dd)
    return make_case(geom, b, rpr, seed, with_data=with_data, data_device=dd)


def run_cell(fi, a, geom, p3, skv, jit, b, rpr, gates, src="gen", jit_fp=None, vmod=None, s1_groups=None, fvmod=None):
    """All gates and reports of one cell; appends to gates, returns the cell's report dict. [fia4] fvmod: the fork."""
    tag = cell_tag(b, rpr, src)
    custom = jit is not None
    t0 = time.time()
    case = case_to(make_cell_case(geom, b, rpr, src, a.seed, s1_groups), "cuda")
    if not custom:                                                     # the bias-free REPORT (C0 failed)
        case["bias"] = torch.zeros_like(case["bias"])
    bf, ba, bh = batch_of(case, "full"), batch_of(case, "alt"), batch_of(case, "hazard")
    r_ = case["R"]
    short_idx = list(range(r_, r_ + len(SHORT_COUNTS)))
    rows = spread(r_, a.sample)
    rev = [n * rpr + u for n in reversed(range(b)) for u in range(rpr)]
    rnd = torch.randperm(r_, generator=torch.Generator().manual_seed(case["seed"] + 1)).tolist()
    perms = [("reversed_requests", rev), ("random", rnd)]
    poisoned, pstats = poison_caches(case, bf)
    ref = reference_fp64(bf, rows, geom)
    ref0 = reference_fp64(bf, rows, geom, zero_bias=True) if custom else None
    rep = dict(cell=tag, R=r_, rows_sampled=rows, poison=pstats, ideal_bytes=ideal_bytes(bf, geom),
               union_bytes=union_bytes(bf, geom, case["N"], case["S"]), arms={}, src=src,
               data_device=case.get("data_device"), s1_groups=case.get("s1_groups"), s1_calls=case.get("s1_calls"))
    arms = build_arms(fi, a, geom, case, p3, skv, jit, a.arms, "cuda", jit_fp=jit_fp, vmod=vmod, fvmod=fvmod)
    outs = {}
    for name in ("P", "F8", "F0", "S5"):
        if name not in arms:
            continue
        must = custom and name in MUST_ARMS
        try:
            g, full = invariance_suite(name, tag, arms[name], bf, ba, bh, rows, perms, poisoned, short_idx,
                                       hazard_pre=(name == "F8"), must=must)
            gates.extend(g)
            outs[name] = full
            gates.append(correctness_gate(name, tag, full[torch.as_tensor(rows, device=full.device)], ref, TOL_C,
                                          must))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("ARM_%s_%s" % (name, tag), False, must, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                            str(e)[:300])))
    if "FH" in arms:
        try:
            g, _ = negative_control(tag, arms["FH"], bf, rows, custom)
            gates.append(g)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("N_heuristic_negative_%s" % tag, False, custom, "EXCEPTION %s: %s"
                              % (type(e).__name__, str(e)[:300])))
    if custom and "F8" in arms:
        try:
            gates.append(bias_witness(tag, arms["F8"], bf, rows, ref, ref0, TOL_C, True))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("CW_bias_witness_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                             str(e)[:300])))
    if "P" in outs:
        for name in ("F8", "F0", "S5"):
            if name in outs:
                gates.append(bits_vs(name, tag, outs[name], outs["P"]))
    for name in ("F8", "F0"):
        if name in arms and name in outs:
            try:
                gates.append(plan_reuse_check(name, tag, arms[name], bf, ba))
            except Exception as e:
                traceback.print_exc()
                gates.append(gate("R_REUSE_%s_%s" % (name, tag), False, False, "EXCEPTION %s: %s"
                                  % (type(e).__name__, str(e)[:300])))
    run_fp_v1(fi, a, geom, case, arms, outs, bf, rows, ref, ref0, poisoned, rpr, tag, custom, jit_fp, gates, rep)
    run_fv(geom, case, arms, outs, bf, rows, ref, ref0, poisoned, rpr, tag, custom, fvmod, gates, rep)    # [fia4]
    tp = None
    sweep = [x for x in a.arms if fv_sweep_launch(x) is not None and x in arms]          # [fia5] (first call ran)
    for name in ("P", "F8", "F0", "FH", "S5", "FP0", "FP8", "FPU", "V1", "FV", "FVT", "FVN") + tuple(sweep):
        if name not in arms or (name == "V1" and not rep.get("v1_pre")):
            continue
        try:
            t = time_arm(name, arms[name], bf, a.warm, a.reps, geom)
            rep["arms"][name] = t
            med = t["call_ms"]["median"]
            if name == "P":
                tp = med
                txt = "P %.4f ms; %s" % (med, band_text(med, PRED["p3_ms"][tag]) if tag in PRED["p3_ms"] else "no band")
            else:
                ratio = med / tp if tp else None
                bands = PRED["ratio_vs_p3"].get(name, {})
                txt = "%s %.4f ms, x%s of P; %s" % (name, med, "%.3f" % ratio if ratio else "n/a",
                                                    band_text(ratio, bands[tag]) if tag in bands else "no band")
                if "launch_host_ms" in t:                                     # [fia4] before the FP branch
                    txt += fv_time_text(t)
                elif "build_ms_per_layer" in t:
                    txt += fp_time_text(t)
                elif "gather_ms" in t:
                    txt += "; union store gather %.4f ms, exp(cis) values %.4f ms (per layer), W %d slots" % (
                        t["gather_ms"]["median"], t["prep_ms"]["median"], t["store_W"])
                elif "pack_ms_per_layer" in t:
                    txt += "; plan/tick host %.4f (%s) device %.4f (%s); pack/layer %.4f (%s); runs only %.4f" % (
                        t["plan_host_indptr_ms_per_tick"]["median"],
                        band_text(t["plan_host_indptr_ms_per_tick"]["median"], PRED["plan_ms_per_tick_host_indptr"]),
                        t["plan_device_indptr_ms_per_tick"]["median"],
                        band_text(t["plan_device_indptr_ms_per_tick"]["median"],
                                  PRED["plan_ms_per_tick_device_indptr"]),
                        t["pack_ms_per_layer"]["median"], band_text(t["pack_ms_per_layer"]["median"],
                                                                    PRED["pack_ms_per_layer"]),
                        t["runs_only_ms"]["median"])
                    if "pack_fixed_ms_per_layer" in t:
                        txt += "; constant counts: pack/layer %.4f (%s), layer with the tick's plan reused %.4f" % (
                            t["pack_fixed_ms_per_layer"]["median"],
                            band_text(t["pack_fixed_ms_per_layer"]["median"], PRED["pack_ms_per_layer"]),
                            t["layer_with_plan_reuse_ms"]["median"])
                    txt += "; float workspace need %d B (FlashInfer's formula), allocated %d B, plan reserves %d B%s" % (
                        t["float_ws_need_bytes"], t["float_ws_alloc_bytes"], t["float_ws_used_bytes"],
                        "" if t["float_ws_need_covers_use"] else " (NEED BELOW USE: formula wrong)")
            txt += "; workspace %d B, peak delta %d B" % (t["workspace_bytes"], t["peak_alloc_delta_bytes"])
            gates.append(gate("R_T_%s_%s" % (name, tag), True, False, txt))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("R_T_%s_%s" % (name, tag), False, False, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                             str(e)[:300])))
    fv_build_report(rep, tag, gates)                                                            # [fia4]
    if sweep or "FVN" in arms:                                                                  # [fia5]
        fv_sweep_report(rep, tag, gates, sweep)
    if custom and a.strided and "F8" in outs:
        try:
            sa = FIStridedArm(fi, geom, case["S"], jit, "cuda")
            o = sa(bf)
            same = sum(1 for r in range(r_) if bits_equal(o[r], outs["F8"][r]))
            err = float((o[torch.as_tensor(rows, device=o.device)].double() - ref).abs().max())
            t = time_arm("STRIDED", sa, bf, a.warm, a.reps, geom)
            rep["arms"]["STRIDED"] = t
            gates.append(gate("R_STRIDED_probe_%s" % tag, True, False,
                              "the overlapping as_strided view RAN: %d of %d rows bitwise equal to F8 (option A); max "
                              "|err| %.3e; %.4f ms per layer (one call)" % (same, r_, err, t["call_ms"]["median"])))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("R_STRIDED_probe_%s" % tag, False, False, "REFUSED or failed: %s: %s"
                              % (type(e).__name__, str(e)[:300])))
    rep["seconds"] = round(time.time() - t0, 1)
    del arms, outs, case, bf, ba, bh, poisoned
    torch.cuda.empty_cache()
    return rep


def fp_time_text(t):
    """[fia3] the FP arms' part of the R_T line."""
    g = t.get("build_graph_ms_per_layer")
    return ("; runs only %.4f; build (union + CSR + mask) per layer %.4f ms events, %.4f ms host, graph %s; plan per "
            "tick host-indptr %.4f device-indptr %.4f ms; layer with the tick's plan %s; tick overhead (%d layers) "
            "eager %.4f host %.4f graph %s ms (rule <= %.1f); union %s; mask %d B; FlashInfer custom-mask mode %s; "
            "float workspace need %d B, allocated %d B"
            % (t["runs_only_ms"]["median"], t["build_ms_per_layer"]["median"], t["build_host_ms_per_layer"]["median"],
               "%.4f" % g["median"] if g else "n/a (%s)" % t.get("build_graph_error"),
               t["plan_host_indptr_ms_per_tick"]["median"], t["plan_device_indptr_ms_per_tick"]["median"],
               "%.4f" % t["layer_with_plan_reuse_ms"]["median"] if "layer_with_plan_reuse_ms" in t else "n/a (split)",
               LAYERS_PER_TICK, t["tick_overhead_eager_ms"], t["tick_overhead_host_ms"],
               "%.4f" % t["tick_overhead_graph_ms"] if t.get("tick_overhead_graph_ms") is not None else "n/a",
               PRED3["thresholds"]["tick_overhead_ms_max"], t["union"], t["mask_bytes"], t["mask_mode_custom"],
               t["float_ws_need_bytes"], t["float_ws_alloc_bytes"]))


def run_fp_v1(fi, a, geom, case, arms, outs, bf, rows, ref, ref0, poisoned, rpr, tag, custom, jit_fp, gates, rep):
    """[fia3] the gates of the FP arms and of V1 for one cell (appends to gates and outs; rep gets union / v1_pre)."""
    fp_names = [x for x in FP_ARMS if x in arms]
    if not fp_names and "V1" not in arms:
        return
    baq = batch_of(case, "altreq")
    rows_t = torch.as_tensor(rows, device=bf["q"].device)
    u_full = None
    if fp_names:
        try:
            gates.append(block0_gate(tag, [bf, baq], rpr, case["S"], must=True))
            u_full = fp_union(bf, rpr, case["S"])
            rep["union"] = union_stats(u_full, bf)
            pois_pre = poison_precondition(u_full)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("G_block0_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
            pois_pre = (False, "not computed")
        nreq = case["R"] // rpr
        breq = sorted({r // rpr for r in rows})
        rperm = torch.randperm(nreq, generator=torch.Generator().manual_seed(case["seed"] + 2)).tolist()
        rperms = [("reversed_requests", list(reversed(range(nreq)))), ("random", rperm)]
        for name in ("FP0", "FP8"):
            if name not in arms:
                continue
            must = name == "FP0"
            try:
                g, full = request_suite(name, tag, arms[name], bf, baq, rpr, breq, rperms, poisoned, pois_pre, must)
                gates.extend(g)
                outs[name] = full
                gates.append(correctness_gate(name, tag, full[rows_t], ref, TOL_C, must))
            except Exception as e:
                traceback.print_exc()
                gates.append(gate("ARM_%s_%s" % (name, tag), False, must, "EXCEPTION %s: %s"
                                  % (type(e).__name__, str(e)[:300])))
        if "FP0" in outs:
            if custom:
                try:
                    gates.append(bias_witness(tag, arms["FP0"], bf, rows, ref, ref0, TOL_C, True,
                                              gname="CW_FP0_%s" % tag))
                except Exception as e:
                    traceback.print_exc()
                    gates.append(gate("CW_FP0_%s" % tag, False, True, "EXCEPTION %s: %s"
                                      % (type(e).__name__, str(e)[:300])))
            try:
                ones = FPArm(fi, "FP0ones", "none", "ones", geom, case["S"], rpr, jit_fp, "cuda")
                gates.append(mask_witness(tag, ones, bf, rows, ref, u_full, TOL_C, True))
                del ones
            except Exception as e:
                traceback.print_exc()
                gates.append(gate("MW_FP0_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                        str(e)[:300])))
            for fn in (lambda: row_alone_probe("FP0", tag, arms["FP0"], bf, rows, rpr, full=outs["FP0"]),
                       lambda: fp_reuse_check("FP0", tag, arms["FP0"], bf, baq)):
                try:
                    gates.append(fn())
                except Exception as e:
                    traceback.print_exc()
                    gates.append(gate("R_FP0_probe_%s" % tag, False, False, "EXCEPTION %s: %s"
                                      % (type(e).__name__, str(e)[:300])))
        if "FP8" in arms and u_full is not None:
            cells_h, rows_h, chunks = chunk_tile_hazard(u_full)
            gates.append(gate("R_FP8_chunk_tile0_%s" % tag, True, False,
                              "MODEL: %d (request, head, row, chunk) cells of %d (request, head, chunk) chunks have no "
                              "visible token in the chunk's first %d-token KV tile: %d query rows predicted NaN in FP8 "
                              "(a split chunk starts at m = -inf with no guard)"
                              % (cells_h, chunks, KV_TILE_PAGES * PAGE, rows_h)))
        if "P" in outs:
            for name in ("FP0", "FP8"):
                if name in outs:
                    gates.append(bits_vs(name, tag, outs[name], outs["P"]))
    if "V1" in arms:
        try:
            u1 = u_full if u_full is not None else fp_union(bf, rpr, case["S"])
            ok, txt = v1_precondition(u1, bf, rpr)
            rep["v1_pre"] = ok
            gates.append(gate("V1_pre_%s" % tag, ok, False, txt, status=None if ok else "VOID"))
            if ok:
                full = arms["V1"](bf)
                outs["V1"] = full
                gates.append(correctness_gate("V1", tag, full[rows_t], ref, TOL_V1, True))
                if "P" in outs:
                    gates.append(bits_vs("V1", tag, full, outs["P"]))
            else:
                gates.append(gate("C_V1_%s" % tag, False, False, "VOID REPORT: V1's layout does not hold (%s)" % txt,
                                  status="VOID"))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("C_V1_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))


def _fv_try(gates, name, must, fn):
    """[fia4] Append fn()'s gate; an exception becomes a failed gate of that name (MUST as given)."""
    try:
        gates.append(fn())
    except Exception as e:
        traceback.print_exc()
        gates.append(gate(name, False, must, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))


def fv_partial_reach(u):
    """[fia5] Live union entries (KV head, request, entry) that the rows reach only in part (0 < reach < PAGE). FI5
    poisons the tokens past such an entry's reach INSIDE a tile that the fork loads, so FI5 tests the V reach mask
    (lessons L34: a plant must have teeth). One host read."""
    reach = u["ext"].amax(dim=2)
    j = torch.arange(reach.shape[-1], device=reach.device).view(1, 1, -1)
    live = j < u["ucnt"].unsqueeze(-1)
    return int((live & (reach > 0) & (reach < PAGE)).sum())


def fv_suite(name, arm, fvmod, geom, case, bf, baq, rows, ref, ref0, poisoned, rpr, tag, custom, gates, outs, rep,
             u_full, swept=False):
    """[fia4] FV's gates for one cell; [fia5] under the arm's own name, for the default FV and for each swept arm (the
    same gates by construction). The first call (Triton's JIT compile, or its cache) is timed and gives the compile
    witness R_<name>_compile; its failure is ARM_<name> (MUST) for FV, and a REPORT FINDING for a swept arm (which
    then runs no gate). Returns True when the gates ran."""
    rows_t = torch.as_tensor(rows, device=bf["q"].device)
    t0 = time.time()
    try:
        arm(bf)
        if bf["q"].is_cuda:
            torch.cuda.synchronize()
        gates.append(gate("R_%s_first_call_%s" % (name, tag), True, False, "the first %s call (Triton JIT compile of "
                          "the fork, or its cache) took %.1f s" % (name, time.time() - t0)))
    except Exception as e:
        traceback.print_exc()
        if swept:
            gates.append(gate("R_%s_first_call_%s" % (name, tag), False, False, "FINDING: the first call failed (%s: "
                              "%s); this launch configuration runs no gate" % (type(e).__name__, str(e)[:300])))
        else:
            gates.append(gate("ARM_%s_%s" % (name, tag), False, True, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                        str(e)[:300])))
        return False
    g, info = fv_compile_gate(name, tag, fvmod, arm)
    gates.append(g)
    rep.setdefault("fv_compile", {})[name] = info
    nreq = case["R"] // rpr
    breq = sorted({r // rpr for r in rows})
    rperm = torch.randperm(nreq, generator=torch.Generator().manual_seed(case["seed"] + 2)).tolist()
    rperms = [("reversed_requests", list(reversed(range(nreq)))), ("random", rperm)]
    pre5 = (True, "no precondition: V reads 0 past each tile's reach, the K / bias columns past it are -inf in S; "
                  "%d union entries are reached in part (the plant lies inside loaded tiles)"
            % fv_partial_reach(u_full))
    try:
        g, full = request_suite(name, tag, arm, bf, baq, rpr, breq, rperms, poisoned, pre5, True)
        gates.extend(g)
        outs[name] = full
        gates.append(correctness_gate(name, tag, full[rows_t], ref, TOL_C, True))
    except Exception as e:
        traceback.print_exc()
        gates.append(gate("ARM_%s_%s" % (name, tag), False, True, "EXCEPTION %s: %s" % (type(e).__name__,
                                                                                    str(e)[:300])))
    if name not in outs:
        return True
    gn = lambda p, mid="": "%s_%s%s_%s" % (p, name, mid, tag)                  # noqa: E731
    if custom:
        _fv_try(gates, gn("CW"), True,
                lambda: bias_witness(tag, arm, bf, rows, ref, ref0, TOL_C, True, gname=gn("CW")))
    _fv_try(gates, gn("U", "_table"), True,
            lambda: fv_table_gate(tag, arm, [("full", bf), ("altreq", baq)], rpr, case["S"], True,
                                  gname=gn("U", "_table")))
    _fv_try(gates, gn("MW"), True, lambda: mask_witness(
        tag, make_fv_arm(fvmod, name, geom, case["S"], rpr, bf["q"].device, ones=True), bf, rows, ref, u_full, TOL_C,
        True, gname=gn("MW")))
    _fv_try(gates, gn("X_row_alone"), False,
            lambda: fv_row_alone_gate(tag, arm, bf, rows, rpr, outs[name], gname=gn("X_row_alone")))
    _fv_try(gates, gn("R_GRAPH"), False, lambda: fv_graph_probe(tag, arm, bf, gname=gn("R_GRAPH")))
    if "P" in outs:
        gates.append(bits_vs(name, tag, outs[name], outs["P"]))
    if name != "FV" and "FV" in outs:
        o = outs[name]
        same = sum(1 for r in range(o.shape[0]) if bits_equal(o[r], outs["FV"][r]))
        gates.append(gate(gn("R_D_bits_vs_FV"), True, False, "%d of %d rows bitwise equal to the default FV (launch "
                          "%s; a tile of a part of a block or other warps may round differently)"
                          % (same, o.shape[0], arm.launch_cfg)))
    return True


def run_fv(geom, case, arms, outs, bf, rows, ref, ref0, poisoned, rpr, tag, custom, fvmod, gates, rep):
    """[fia4] The gates of the fork for one cell (module docstring [fia4]). Appends to gates and outs. [fia5] the
    swept arms (fv_suite each; a swept arm whose first call fails leaves arms) and FVN (timing only)."""
    if not any(is_fv_arm(x) for x in arms):
        return
    baq = batch_of(case, "altreq")
    rows_t = torch.as_tensor(rows, device=bf["q"].device)
    if not any(g["name"] == "G_block0_%s" % tag for g in gates):            # shared with FP0 (computed once)
        _fv_try(gates, "G_block0_%s" % tag, True, lambda: block0_gate(tag, [bf, baq], rpr, case["S"], must=True))
    u_full = fp_union(bf, rpr, case["S"])
    rep["union"] = rep.get("union") or union_stats(u_full, bf)
    common = (fvmod, geom, case, bf, baq, rows, ref, ref0, poisoned, rpr, tag, custom, gates, outs, rep, u_full)
    if "FV" in arms:
        fv_suite("FV", arms["FV"], *common)
    for x in [n for n in list(arms) if fv_sweep_launch(n) is not None]:      # [fia5] the same gates per swept arm
        if not fv_suite(x, arms[x], *common, swept=True):
            del arms[x]
    if "FVT" in arms:
        try:
            o = arms["FVT"](bf)
            outs["FVT"] = o
            g, info = fv_compile_gate("FVT", tag, fvmod, arms["FVT"])                          # [fia5]
            gates.append(g)
            rep.setdefault("fv_compile", {})["FVT"] = info
            gates.append(correctness_gate("FVT", tag, o[rows_t], ref, TOL_C, False))
            if "FV" in outs:
                same = sum(1 for r in range(o.shape[0]) if bits_equal(o[r], outs["FV"][r]))
                gates.append(gate("R_FVT_equals_FV_%s" % tag, same == o.shape[0], False, "%d of %d rows of FVT (a torch "
                                  "table) bitwise equal to FV (the kernel's table)" % (same, o.shape[0])))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("ARM_FVT_%s" % tag, False, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    if "FVN" in arms:                                                           # [fia5] timing only
        try:
            o = arms["FVN"](bf)
            g, info = fv_compile_gate("FVN", tag, fvmod, arms["FVN"])
            gates.append(g)
            rep.setdefault("fv_compile", {})["FVN"] = info
            d = (o[rows_t].double() - ref.to(o.device)).abs()
            err = float(d.max()) if bool(torch.isfinite(d).all()) else float("inf")
            st = "DETECTED" if err > TOL_C else "BLIND"
            gates.append(gate("R_FVN_not_attention_%s" % tag, st == "DETECTED", False,
                              "%s: FVN (no extents: every row sees every union token) vs the reference %.3e "
                              "(tol %.1e); a timing arm only (FV - FVN = the extents' cost)" % (st, err, TOL_C),
                              status=st))
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("ARM_FVN_%s" % tag, False, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))


def fv_build_report(rep, tag, gates):
    """[fia4] R_FV_build (REPORT): the in-kernel union build per layer and per tick from CUDA events (FV - FVT kernel
    medians; ncu's FV - FVT is in R_DECISION4), next to the torch build of FVT (eager and one CUDA graph)."""
    fv, ft = rep["arms"].get("FV"), rep["arms"].get("FVT")
    if not (fv and ft):
        return
    d = fv["call_ms"]["median"] - ft["call_ms"]["median"]
    fv["build_in_kernel_ms_per_layer_events"] = round(d, 5)
    fv["build_in_kernel_ms_per_tick_events"] = round(LAYERS_PER_TICK * max(d, 0.0), 5)
    gates.append(gate("R_FV_build_%s" % tag, True, False,
                      "in-kernel union build (FV - FVT, events) %.4f ms per layer -> %.4f ms per tick (%d layers; "
                      "rule <= %.1f); the torch build of FVT per tick: eager %s ms, graph %s ms"
                      % (d, fv["build_in_kernel_ms_per_tick_events"], LAYERS_PER_TICK,
                         PRED4["thresholds"]["union_build_ms_per_tick_max"], ft.get("tick_build_eager_ms"),
                         ft.get("tick_build_graph_ms"))))


# ------------------------------------------------------------------ modes
def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("main", "ncu-cell"), default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, help="the model's config.json (NOSA-8B)")
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy)")
    ap.add_argument("--sharedkv", default=None, help="sharedkv5's bi_paged_attention_shared.py (a pinned copy; S5)")
    ap.add_argument("--cells", default=",".join(cell_tag(b, r) for b, r in CELLS))
    ap.add_argument("--arms", default="P,F8,F0,FH,S5")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--ncu", default="none", help="the ncu binary, or 'none'")
    ap.add_argument("--ncu-cap", type=int, default=NCU_CAP_S)
    ap.add_argument("--budget-s", type=int, default=2400)
    ap.add_argument("--no-strided", dest="strided", action="store_false")
    ap.add_argument("--bias-free", action="store_true", help="ncu-cell only: the default module, bias = 0")
    ap.add_argument("--verify-step", default=None, help="[fia3] EQ verify_step.py (a pinned copy; arm V1)")
    ap.add_argument("--s1-tables", default=None, help="[fia3] the recorded S1 skv_tables.pt (a copy; the s1 cells)")
    ap.add_argument("--fv-kernel", default=None, help="[fia4] the fork nosi/nosi/fork_vllm_unified_attention.py (FV, "
                                                      "FVT)")
    ap.add_argument("--fv-sweep", default="", help="[fia5] FV launch configurations, one more arm each: "
                                                   "w<warps>s<stages>[t<tile>][b<16|32>],... (needs the arm FV)")
    a = ap.parse_args(argv)
    a.cells = parse_cells(a.cells)
    a.arms = parse_arms(a.arms)
    a.fv_sweep = parse_fv_sweep(a.fv_sweep)                                       # [fia5]
    if a.fv_sweep and "FV" not in a.arms:
        raise ValueError("--fv-sweep needs the arm FV (the swept arms are FV's launch configurations)")
    a.arms = a.arms + [x for x in a.fv_sweep if x not in a.arms]
    if a.sample < 2 or a.warm < 0 or a.reps < 1 or a.budget_s < 60 or not 1 <= a.ncu_cap <= NCU_CAP_S:
        raise ValueError("--sample >= 2, --warm >= 0, --reps >= 1, --budget-s >= 60, --ncu-cap 1 .. %d" % NCU_CAP_S)
    if "S5" in a.arms and a.mode == "ncu-cell" and not a.sharedkv:
        raise ValueError("ncu-cell with S5 needs --sharedkv")
    if "V1" in a.arms and not a.verify_step:
        raise ValueError("arm V1 needs --verify-step")
    if any(is_fv_arm(x) for x in a.arms) and not a.fv_kernel:                     # [fia4] [fia5] any fork arm
        raise ValueError("arms FV / FVT need --fv-kernel")
    if any(cell_parts(c)[2] == "s1" for c in a.cells) and not a.s1_tables:
        raise ValueError("an s1 cell needs --s1-tables")
    return a


def refuse_common(a):
    """Refusals that need no GPU (before anything is created). None = go."""
    if os.path.exists(a.out):
        return "--out %s exists" % a.out
    for f, what in ((a.config, "--config"), (a.biattn, "--biattn")):
        if not os.path.isfile(f):
            return "%s %s is not a file" % (what, f)
    if a.sharedkv and not os.path.isfile(a.sharedkv):
        return "--sharedkv %s is not a file" % a.sharedkv
    for f, what in ((getattr(a, "verify_step", None), "--verify-step"), (getattr(a, "s1_tables", None), "--s1-tables"),
                    (getattr(a, "fv_kernel", None), "--fv-kernel")):                      # [fia4] --fv-kernel
        if f and not os.path.isfile(f):
            return "%s %s is not a file" % (what, f)
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not os.environ.get("FLASHINFER_WORKSPACE_BASE"):
        return "FLASHINFER_WORKSPACE_BASE is unset (FlashInfer would use HOME: lessons C1)"
    return None


def import_flashinfer():
    import flashinfer
    v = getattr(flashinfer, "__version__", "unknown")
    if not version_ok(v):
        raise RuntimeError("FlashInfer %s is not one of %s" % (v, FI_VERSIONS))
    return flashinfer, v


def build_fp_variant(fi, a, geom, gates, report):
    """[fia3] gates FP_API and C0_FP (MUST when an FP arm is asked): the wrapper's signature, then the JIT build of
    R2FiCisBiasMask (the first wrapper builds, the second loads). Returns jit_args_fp or None."""
    if not any(x in a.arms for x in FP_ARMS):
        return None
    ok, why = check_fp_api(fi)
    gates.append(gate("FP_API", ok, True, "BatchPrefillWithPagedKVCacheWrapper: %s" % why))
    if not ok:
        return None
    jfp = jit_args_fp(geom["D"])
    ws0 = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    try:
        t0 = time.time()
        fi.BatchPrefillWithPagedKVCacheWrapper(ws0, "NHD", backend="fa2", jit_args=jfp)
        build_s = time.time() - t0
        t1 = time.time()
        fi.BatchPrefillWithPagedKVCacheWrapper(ws0, "NHD", backend="fa2", jit_args=jfp)
        report["jit_fp"] = dict(build_s=round(build_s, 1), load_s=round(time.time() - t1, 2))
        gates.append(gate("C0_FP_variant_jit_build", True, True, "R2FiCisBiasMask built and loaded in %.1f s; a second "
                          "wrapper loads in %.2f s" % (build_s, report["jit_fp"]["load_s"])))
        return jfp
    except Exception as e:
        with open(os.path.join(a.out, "jit_fp_build_error.txt"), "w") as f:
            f.write(traceback.format_exc())
        report["findings"].append("the FP variant did not build: %s: %s" % (type(e).__name__, str(e)[:500]))
        gates.append(gate("C0_FP_variant_jit_build", False, True, "FINDING: the FP variant JIT build FAILED (%s: %s; "
                          "jit_fp_build_error.txt); the FP arms do not run" % (type(e).__name__, str(e)[:200])))
        return None
    finally:
        del ws0


def load_v1(a, gates, report):
    """[fia3] V1's module and op, or None with a REPORT FINDING (V1 is the no-new-code route of Task 1.4: a missing
    op is a finding, not a failure of this bench)."""
    if "V1" not in a.arms:
        return None
    vmod, why = import_v1(a.verify_step)
    gates.append(gate("R_V1_import", vmod is not None, False, "verify_step %s + infllm_v2 internal varlen op: %s"
                      % (a.verify_step, "importable" if vmod is not None else "FINDING: " + why)))
    if vmod is None:
        report["findings"].append("V1 is not importable in this environment: %s" % why)
    return vmod


def load_fv(a, gates, report):
    """[fia4] gate C0_FV_import (MUST when FV / FVT is asked): the fork loads by path (torch + triton). Returns the
    module or None (the FV arms do not run; a FINDING)."""
    if not any(is_fv_arm(x) for x in a.arms):                                     # [fia5] any fork arm
        return None
    mod, why = import_fv(a.fv_kernel)
    gates.append(gate("C0_FV_import", mod is not None, True, "the fork %s: %s" % (
        a.fv_kernel, "imported (triton %s)" % getattr(sys.modules.get("triton"), "__version__", "?") if mod is not None
        else "FINDING: " + why)))
    if mod is None:
        report["findings"].append("the fork does not import: %s" % why)
    return mod


def load_s1(a, geom, gates):
    """[fia3] the recorded S1 groups per rows-per-request of the s1 cells: {rpr: groups}, or None (a MUST failure)."""
    rprs = sorted({cell_parts(c)[1] for c in a.cells if cell_parts(c)[2] == "s1"})
    if not rprs:
        return {}
    try:
        tables = load_s1_tables(a.s1_tables)
        s1 = {r: s1_groups(tables, r, geom["Hkv"]) for r in rprs}
        gates.append(gate("R_S1_tables", True, False, "%s: %d recorded calls; requests with exactly r rows: %s"
                          % (a.s1_tables, len(tables), {r: len(v) for r, v in s1.items()})))
        return s1
    except Exception as e:
        traceback.print_exc()
        gates.append(gate("S1_tables", False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
        return None


def mode_main(a):
    why = refuse_common(a)
    if why:
        print("REFUSED: %s" % why, file=sys.stderr)
        return 2
    try:
        fi, fiv = import_flashinfer()
    except Exception as e:
        print("REFUSED: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    t_start = time.time()
    deadline = t_start + a.budget_s
    os.makedirs(a.out)
    geom = geometry_from_config(json.load(open(a.config)))
    p3 = load_by_path(a.biattn, "piece3_bi_paged_attention")
    skv = load_by_path(a.sharedkv, "sharedkv5_bi_paged_attention_shared") if a.sharedkv else None
    gates, report = [], dict(cells={}, ncu={}, findings=[])
    jit = jit_args_for(geom["D"])
    ws0 = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    try:
        t0 = time.time()
        fi.BatchDecodeWithPagedKVCacheWrapper(ws0, "NHD", use_tensor_cores=True, backend="fa2", jit_args=jit)
        build_s = time.time() - t0
        t1 = time.time()
        fi.BatchDecodeWithPagedKVCacheWrapper(ws0, "NHD", use_tensor_cores=True, backend="fa2", jit_args=jit)
        load_s = time.time() - t1
        report["jit"] = dict(build_s=round(build_s, 1), load_s=round(load_s, 2))
        gates.append(gate("C0_variant_jit_build", True, True, "R2FiCisBias built and loaded in %.1f s (%s); a second "
                          "wrapper loads in %.2f s" % (build_s, band_text(build_s, PRED["jit_build_s"]), load_s)))
    except Exception as e:
        tb = traceback.format_exc()
        with open(os.path.join(a.out, "jit_build_error.txt"), "w") as f:
            f.write(tb)
        report["findings"].append("the CIS bias variant did not build: %s: %s" % (type(e).__name__, str(e)[:500]))
        gates.append(gate("C0_variant_jit_build", False, True, "FINDING: the variant JIT build FAILED (%s: %s; "
                          "jit_build_error.txt); the bias-free comparison follows as a REPORT"
                          % (type(e).__name__, str(e)[:200])))
        jit = None
    del ws0
    jit_fp = build_fp_variant(fi, a, geom, gates, report)                 # [fia3]
    vmod = load_v1(a, gates, report)
    fvmod = load_fv(a, gates, report)                                      # [fia4]
    s1 = load_s1(a, geom, gates)
    for c in a.cells:
        b, rpr, src = cell_parts(c)
        tag = cell_tag(b, rpr, src)
        left = deadline - time.time()
        if left < CELL_RESERVE_S:
            gates.append(gate("BUDGET_cell_%s" % tag, False, True, "skipped: %d s left < %d s" % (left, CELL_RESERVE_S)))
            continue
        if src == "s1" and not (s1 or {}).get(rpr):
            gates.append(gate("CELL_%s" % tag, False, True, "no recorded request with %d rows (S1 tables)" % rpr))
            continue
        log("cell %s (%d s left)" % (tag, left))
        try:
            report["cells"][tag] = run_cell(fi, a, geom, p3, skv, jit, b, rpr, gates, src=src, jit_fp=jit_fp,
                                            vmod=vmod, s1_groups=(s1 or {}).get(rpr), fvmod=fvmod)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("CELL_%s" % tag, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    gates.extend(completeness_gates(gates, a.cells, a.arms, jit is not None, fp=jit_fp is not None,
                                    v1=vmod is not None, fv=fvmod is not None))
    if a.ncu != "none":
        try:
            run_ncu(a, geom, jit is None, deadline, gates, report, fp_ok=jit_fp is not None, v1_ok=vmod is not None,
                    s1=s1, fv_ok=fvmod is not None)
        except Exception as e:
            traceback.print_exc()
            gates.append(gate("R_ncu", False, False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:300])))
    if jit_fp is not None and "FP0" in a.arms:
        gates.extend(decision_report(report, gates, a.cells))
    if fvmod is not None and "FV" in a.arms:                                 # [fia4]
        gates.extend(decision_report4(report, gates, a.cells))
        gates.extend(decision_report5(report, gates, a.cells))                # [fia5]
    meta = dict(mode="main", cells=[cell_tag(*c) for c in a.cells], arms=a.arms, seed=a.seed, sample=a.sample,
                warm=a.warm, reps=a.reps, budget_s=a.budget_s, seconds=round(time.time() - t_start, 1), geom=geom,
                torch=torch.__version__, flashinfer=fiv, gpu=torch.cuda.get_device_name(0),
                commit=os.environ.get("FIATTN_COMMIT", "unknown"), biattn=a.biattn, sharedkv=a.sharedkv,
                verify_step=a.verify_step, s1_tables=a.s1_tables, fi_versions_accepted=list(FI_VERSIONS),
                fv_kernel=a.fv_kernel, triton=getattr(sys.modules.get("triton"), "__version__", None),   # [fia4]
                fv_sweep={x: fv_sweep_launch(x) for x in a.fv_sweep}, fv_bias_bits=FV_BIAS_BITS,        # [fia5]
                kv_tile_tokens_model=KV_TILE_PAGES * PAGE,
                bias_free=jit is None, tol_c=TOL_C, env={k: v for k, v in os.environ.items()
                                                         if k.startswith(("FLASHINFER_", "TVM_FFI", "TRITON_",
                                                                          "TORCH_", "MAX_JOBS", "TMPDIR"))})
    write_json(os.path.join(a.out, "summary.json"), dict(meta=meta, gates=gates, report=report, pred=PRED,
                                                         pred3=PRED3, pred4=PRED4, pred5=PRED5))      # [fia5] pred5
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                g["detail"][:500]))
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    print("r2_fiattn_bench: %d MUST gate(s) failed" % nfail)
    return min(nfail, 125)


def run_ncu(a, geom, bias_free, deadline, gates, report, fp_ok=False, v1_ok=False, s1=None, fv_ok=False):
    if not os.access(a.ncu, os.X_OK):
        gates.append(gate("R_ncu", False, False, "no executable ncu at %s" % a.ncu))
        return
    rc, missing, metrics = ncu_metrics(a, a.out)
    if rc != 0 or missing:
        gates.append(gate("R_ncu_metrics", False, False, "query exit %s; missing %s" % (rc, missing)))
        return
    for c in sorted(a.cells, key=ncu_key):                                 # [fia3] fia2's order for the old cells
        b, rpr, src = cell_parts(c)
        tag = cell_tag(b, rpr, src)
        if tag not in report.get("cells", {}):
            gates.append(gate("R_ncu_%s" % tag, False, False, "skipped: the cell did not run"))
            continue
        cap = ncu_cap(deadline - time.time(), a.ncu_cap)
        if cap is None:
            gates.append(gate("R_ncu_%s" % tag, False, False, "skipped: budget"))
            continue
        arms = [x for x in ("P", "F8", "F0") if x in a.arms]
        if rpr > 1 and a.sharedkv and "S5" in a.arms:
            arms.append("S5")
        if fp_ok:
            arms += [x for x in FP_ARMS if x in a.arms]
        if v1_ok and "V1" in a.arms and report["cells"][tag].get("v1_pre"):
            arms.append("V1")
        if fv_ok and rpr <= FV_ROWS:                                         # [fia4]
            arms += [x for x in FV_ARMS if x in a.arms]
            best = report["cells"][tag].get("fv_best")                       # [fia5] the cell's best swept arm
            if best and best not in arms:
                arms.append(best)
        child = os.path.join(a.out, "ncu_%s" % tag)
        rep = os.path.join(a.out, "ncu_%s_rep" % tag)
        t1 = time.time()
        rc = run_logged(ncu_argv(a, b, rpr, child, rep, metrics, arms, bias_free, src=src),
                        os.path.join(a.out, "ncu_%s.log" % tag), cap)
        exp = None
        summ = None
        if os.path.exists(rep + ".ncu-rep"):
            csv_path = os.path.join(a.out, "ncu_%s_raw.csv" % tag)
            exp = run_logged([a.ncu, "--import", rep + ".ncu-rep", "--csv", "--page", "raw", "--print-units", "base"],
                             csv_path, 300)
            if exp == 0:
                case = make_cell_case(geom, b, rpr, src, a.seed, (s1 or {}).get(rpr), with_data=False)
                bf = case["full"]
                summ = ncu_summary(ncu_rows(open(csv_path).read(), list(metrics) + list(NCU_LAUNCH)), arms,
                                   ideal_bytes(bf, geom), union_bytes(bf, geom, case["N"], case["S"]), rpr)
        report["ncu"][tag] = dict(child_exit=rc, export_exit=exp, cap_s=cap, seconds=round(time.time() - t1, 1),
                                  summary=summ)
        txt = "child exit %s (cap %d s), export %s" % (rc, cap, exp)
        if summ:
            txt += "; " + "; ".join(ncu_arm_text(x, p) for x, p in summ["arms"].items())
        gates.append(gate("R_ncu_%s" % tag, rc == 0 and exp == 0, False, txt))


def ncu_arm_text(x, p):
    """One arm's ncu numbers ([fia3] + device time, L2 -> SM bytes, registers, local memory, PRED3 bands)."""
    f = lambda v: "%.3f" % v if v is not None else "n/a"                    # noqa: E731
    txt = ("%s %.1f us, read %.3g B = %s of ideal, %s of union, %d kernels (expected %s); %s"
           % (x, p["time_ns"] / 1e3, p["read"], f(p["read_over_ideal"]), f(p["read_over_union"]), p["n_kernels"],
              p["expected_kernels"], p["band"]))
    if p.get("l2sm_bytes") is not None:
        txt += "; L2->SM %.3g B = %s of union, %s of sum" % (p["l2sm_bytes"], f(p["l2sm_over_union"]),
                                                             f(p["l2sm_over_ideal"]))
    if p.get("regs"):
        txt += "; registers %s" % sorted(set(p["regs"]))
    if p.get("local_bytes") is not None:
        txt += "; local memory %.3g B%s" % (p["local_bytes"], " (SPILLS)" if p["local_bytes"] > 0 else "")
    if p.get("occ"):                                                        # [fia4]
        txt += "; achieved occupancy %s %%" % p["occ"]
    if p.get("stall_lsb"):                                                  # [fia5]
        txt += "; long-scoreboard stall %s %% of active warps" % p["stall_lsb"]
    if p.get("smem"):
        txt += "; shared memory per block %s B" % sorted(set(p["smem"]))
    if p.get("limits"):
        txt += "; occupancy limits (registers, shared memory, waves) %s" % list(dict.fromkeys(p["limits"]))[:3]
    if p.get("band3"):
        txt += "; %s" % p["band3"]
    return txt


def mode_ncu_cell(a):
    """Run under ncu: one cell, every arm planned and warmed OUTSIDE the profiler window, then one call per arm
    inside it, each in an NVTX range fia|<arm>. Writes --out/cell.json."""
    why = refuse_common(a)
    if why:
        print("REFUSED: %s" % why, file=sys.stderr)
        return 2
    fi, fiv = import_flashinfer()
    os.makedirs(a.out)
    geom = geometry_from_config(json.load(open(a.config)))
    p3 = load_by_path(a.biattn, "piece3_bi_paged_attention")
    skv = load_by_path(a.sharedkv, "sharedkv5_bi_paged_attention_shared") if a.sharedkv else None
    b, rpr, src = cell_parts(a.cells[0])
    groups = s1_groups(load_s1_tables(a.s1_tables), rpr, geom["Hkv"]) if src == "s1" else None
    case = case_to(make_cell_case(geom, b, rpr, src, a.seed, groups), "cuda")
    jit = None if a.bias_free else jit_args_for(geom["D"])
    if a.bias_free:
        case["bias"] = torch.zeros_like(case["bias"])
    bf = batch_of(case, "full")
    jit_fp = jit_args_fp(geom["D"]) if any(x in a.arms for x in FP_ARMS) else None      # [fia3] built by the parent
    vmod = import_v1(a.verify_step)[0] if "V1" in a.arms else None
    fvmod = import_fv(a.fv_kernel)[0] if any(is_fv_arm(x) for x in a.arms) else None   # [fia4] (the Triton cache)
    arms = build_arms(fi, a, geom, case, p3, skv, jit, a.arms, "cuda", jit_fp=jit_fp, vmod=vmod, fvmod=fvmod)
    if "V1" in arms and not v1_precondition(fp_union(bf, rpr, case["S"]), bf, rpr)[0]:
        del arms["V1"]
    for arm in arms.values():
        arm.prepare(bf)
        arm.launch()
    for arm in arms.values():
        arm.prepare(bf)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, arm in arms.items():
        torch.cuda.nvtx.range_push("fia|%s" % name)
        arm.launch()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    write_json(os.path.join(a.out, "cell.json"), dict(cell=cell_tag(b, rpr, src), arms=list(arms), flashinfer=fiv,
                                                      ideal_bytes=ideal_bytes(bf, geom),
                                                      union_bytes=union_bytes(bf, geom, case["N"], case["S"])))
    return 0


def main(argv=None):
    try:
        a = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:
        return 2 if e.code else 0
    except ValueError as e:
        print("REFUSED: %s" % e, file=sys.stderr)
        return 2
    a.out = os.path.abspath(a.out)
    try:
        return mode_ncu_cell(a) if a.mode == "ncu-cell" else mode_main(a)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
