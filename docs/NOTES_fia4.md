# R2 FIA4: a fork of vLLM's Triton unified attention for the shared-KV verify (plan Task 0.3, option 2)

REGISTERED <TBD-for-lead-stamp>

Builder stamp: 2026-10-27T18:40:16Z (`date -u`). Status: SOURCE ONLY, NOT RUN, NOT REVIEWED.
Branch r2-fia4, worktree /mnt/beegfs/ojerbi/nosi-r2-fia4, from r2-fia3 273e2d8 (the fia3 bench, stamped and run).
Plan: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md (Task 0.3, second option).
Evidence: SCORING_fia3.md, SURVEY_shared_kv_kernel.md (sections 3.2, 7), analysis/evalfast/EVIDENCE_q2.md.

## 1. Scope and the rule (skill reusing-before-building, rung 4)

- The rung-4 row is written (STATUS_LOG.md 18:05Z). Tool tried on our code: FlashInfer 0.5.3
  BatchPrefillWithPagedKVCacheWrapper + a custom mask (record /mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_f1a_2195573).
  Measured blocker: 1.68-1.70x the device time of piece 3; 77-100 ms of plan + mask build per tick; latency-bound at
  2 CTAs per SM (255 registers + 81 KB shared memory, one wave). Upstream code read: vLLM 0.20.2
  triton_unified_attention.py.
- The human's rule: fork the closest existing kernel, keep its structure, change only what our case needs, tag every
  changed line `# [fork]` with the reason. No other new kernel.
- This change adds ONE fork: nosi/nosi/fork_vllm_unified_attention.py. It adds no other kernel. The bench and the step
  launch nothing else that is new. Production, the reference and the tick do not change.

## 2. Files

| file | what |
|---|---|
| /mnt/beegfs/ojerbi/nosi-r2-fia4/nosi/nosi/fork_vllm_unified_attention.py | the fork (1142 lines; 305 lines added or changed, 281 upstream lines removed, all tagged) |
| /mnt/beegfs/ojerbi/nosi-r2-fia4/benchmarks/Efficiency/tests_cpu/fixtures_vllm_0_20_2/triton_unified_attention.py.upstream | pristine upstream copy (cp, byte for byte) |
| /mnt/beegfs/ojerbi/nosi-r2-fia4/benchmarks/Efficiency/tests_cpu/fixtures_vllm_0_20_2/triton_attention_helpers.py.upstream | pristine upstream copy (cp, byte for byte) |
| /mnt/beegfs/ojerbi/nosi-r2-fia4/benchmarks/Efficiency/r2_fiattn_bench.py | the bench: [fia4] arms FV, FVT, their gates, PRED4, R_DECISION4, ncu occupancy fields |
| /mnt/beegfs/ojerbi/nosi-r2-fia4/benchmarks/Efficiency/r2_fia4_step.sh | the GPU step (new; r2_fia3_step.sh does not change) |
| /mnt/beegfs/ojerbi/nosi-r2-fia4/benchmarks/Efficiency/tests_cpu/test_r2_fia4.py | the CPU tests FV1-FV10 (new) |
| /mnt/beegfs/ojerbi/nosi-r2-fia4/docs/NOTES_fia4.md | this file |

Upstream source: /mnt/beegfs/ojerbi/venvs/vllm-port/lib/python3.10/site-packages/vllm/v1/attention/ops/
triton_unified_attention.py and triton_attention_helpers.py, vLLM 0.20.2 (vllm-0.20.2.dist-info), Apache-2.0
(SPDX header). The fork keeps the SPDX, copyright and authors lines (test FV1).

## 3. The diff against upstream (changed regions)

How the fork was made: the unified file's lines 1-31 + the helpers file from line 14 + the unified file from line 32
(head / tail into one file), then edits only. Test FV2 rebuilds this composed text from the two fixtures and runs
difflib on it: every line of a changed hunk carries `[fork]` (blank lines excepted), and every pure deletion has a
`[fork]` marker within 3 lines. Two planted negative controls (an untagged change, an unmarked deletion) prove that the
check catches both.

| region | upstream | fork | why |
|---|---|---|---|
| header | SPDX, copyright, authors | the same, then 20 `# [fork]` lines | the fork's provenance, the rung-4 row, the change list |
| imports | vllm.envs, logger, platforms, triton_utils, the helpers, KVQuantMode | `import enum`, `import triton`, `import triton.language as tl`; KVQuantMode as an IntEnum with upstream's values 0-3 | the image has no vllm |
| helpers | 11 functions in triton_attention_helpers.py | 7 copied verbatim (cdiv_fn, apply_softcap, resolve_seq_and_query_len, find_seq_idx, init_softmax_M, store_segm_reduce_scalars, softmax_step); 4 removed with markers (compute_tile_loop_bounds, compute_kv_seq_mask, apply_alibi_to_score, load_qq_bias_tile) | the removed ones read token positions; the union tile space has none |
| globals | logger, is_batch_invariant from vllm envs, float8_info from the platform | logger removed; is_batch_invariant = False; float8_info = finfo(float8_e4m3fn) | no vllm; the CUDA platform's fp8 type is e4m3fn (platforms/interface.py) |
| new jit helpers | none | fork_build_union, fork_row_lower_bound (inlined into the kernel; never launched alone) | task item 2: the union table on the device |
| kernel arguments | as upstream | + cis_bias_ptr, union_ext_ptr, union_len_ptr, the 5 row-list pointers, the bias and list strides, row_width, slots_per_entry, BUILD_UNION, USE_CIS_BIAS, P_PAD, BUILD_STEPS | items 2-4 |
| kernel prologue | block_table_offset = seq_idx x stride (after the Q load) | union_row = kv_head x num_seqs + seq; the offset moved up; BUILD_UNION: the program builds its table, then a barrier; else it loads the union length; IS_3D tiles = cdiv(union length, segments) | one table per (request, KV head) |
| loop bounds | context_len, ALiBi / QQ set-up, compute_tile_loop_bounds | row_shift; the loop runs over union entries [0, union length) (a segment's slice in 3D) | positions do not exist here |
| loop body | tile_mask = prefix; causal / window mask; S = scale x q.k; ALiBi, QQ bias; window V mask | the entry's packed extents word; row_ext per tile row; reach = the largest byte (scalar arithmetic); tile_mask = t < reach; the CIS bias read through the block like the per-token-head scales; seq_mask = t < row_ext; S = (q.k + bias) x scale; ALiBi / QQ / window removed with markers | items 3, 4 |
| reduce_segments | segments from seq_len tokens | + union_len_ptr, num_queries_per_kv; segments from the union length | the 3D path behind its flag |
| launcher helpers | none | _fork_inputs (metadata checks, the build arguments, stand-in pointers), _fork_launch_options (num_warps / num_stages, empty by default) | no host read |
| launcher | BLOCK_M 16 (BLOCK_Q 1 at GQA 16); tile 16 / 32; use_3d needs max_seqlen_q == 1 | new keyword arguments; refusals (window, chunked, ALiBi, QQ, PrefixLM, quantized KV, no tables); BLOCK_Q = rows_per_q_block (4), BLOCK_M = 4 x 16 = 64; TILE = block size (64); use_3d allows up to BLOCK_Q query rows; block_table_stride = stride(1) | item 1 |

Unchanged: the Q load, init_softmax_M, the online softmax (softmax_step), the accumulator (ONE 64 x 128 fp32 tile; no
per-row accumulator), the K / V address arithmetic and loads, _cast_kv_tile, the epilogue (2D and 3D), the reduce body,
the grid and the q-block search, _get_tile_size (now unused).

## 4. The design in our terms

- Program = (request, KV head). Tile rows = the request's 4 rows (1 draft + 3 verify) x the 16 query heads of the KV
  head = 64. One tl.dot (64 x 128) x (128 x 64) per union block for q.k, one (64 x 64) x (64 x 128) for p.v. Each K / V
  tile is loaded ONCE for all 64 rows.
- Union table per (KV head, request): the physical blocks (storage entry x S + slot) of the distinct (block id, slot)
  pairs of the rows' lists, in ascending block id (the order of fp_union; union entry 0 = block 0).
- Per union entry: one int32 word. Byte u = the tokens of that block that row u sees: 64 for a listed block, last_len
  for the row's last listed block, 0 for a block the row does not list. The kernel applies it to the SCORE tile
  (-inf). K / V / bias loads stop at the largest byte of the word (the tile's reach), so a token that no row of the
  request sees is never loaded (FI5 needs no precondition for FV).
- The CIS bias: logit = (q.k + bias) x scale (the lead's decision: the shipped decode math).
- The union build in the program (fork_build_union). PRECONDITION: every row lists its blocks in strictly ascending
  block id (piece 3's canonical order, build_page_table). Pass 1: per list entry e, a binary search of its value in
  every row gives the row bytes and the FIRST flag (no earlier row lists the value). An inclusive count C of the FIRST
  flags goes to the program's own ext row as scratch; a barrier. Pass 2: rank_e = the FIRST entries below the value in
  every row (C at the row's lower bound - C at the row start); a barrier; then each entry stores its page and word at
  rank_e. Cost per program: 2 passes x 4 rows x 7 steps on 256 entries (no pairwise compare, no sort, no host read, no
  extra launch). The definitions of FIRST and rank are those of our sharedkv3 _union_kernel
  (nosi/nosi/bi_paged_attention_shared.py:505-611), which passed its CPU and GPU equality tests (SCORING_S1f.md).
- FVT reads a table that torch builds before the launch (fp_union + fv_tables_from_union). It is a timing arm: ncu
  FV - FVT = the device cost of the in-kernel build. The torch build is reported too (it is about 20 torch ops).

## 5. Arms and gates

| arm | what | role |
|---|---|---|
| P | piece 3 v2 (7c826c1, blob 39ed5e18) | the reference arm |
| FP0, FPU | the fia3 FlashInfer arms | same-job continuity |
| FV | the fork, union built in the kernel, no split | the candidate |
| FVT | the fork reading a torch-built table | timing only (the in-kernel build's cost) |

| gate | test | note |
|---|---|---|
| C0_FV_import (MUST) | the fork imports (torch + triton) | a failure is a FINDING; FV does not run |
| G_block0 (MUST) | every row lists block 0 | shared with FP0; computed once |
| C_FV (MUST) | max abs(FV - fp64) <= 1e-3 on 16 rows | |
| FI1, FI2, FI3, FI5 _FV (MUST) | request alone, order, other requests changed, poison | FI5 has no precondition for FV |
| CW_FV (MUST) | FV with bias 0 fails C | precondition: the bias moves the reference >= 1e-2 |
| MW_FV (MUST) | every row given the pair's reach fails C | through the prebuilt-table path (the same loop) |
| U_FV_table (MUST) | the tables the kernel wrote == the torch tables (full and altreq batches) | precondition: sorted rows, else VOID |
| X_row_alone_FV (REPORT) | a row alone (4 copies) == in its request, bitwise | EXACT expected (section 7) |
| R_GRAPH_FV (REPORT) | one FV launch in a CUDA graph; the replay == eager, bitwise | no host read in the launch |
| R_FVT_equals_FV, C_FVT (REPORT) | FVT == FV bitwise; FVT vs fp64 | |
| R_FV_first_call, R_T_FV, R_T_FVT, R_FV_build (REPORT) | times: first call (Triton JIT), launch, launch host time, torch build per layer / per tick, in-kernel build (FV - FVT events) | |
| R_D_bits_vs_P_FV (REPORT) | bits vs piece 3 | rounding differences never reject |
| R_ncu (REPORT) | + achieved occupancy, shared memory per block, occupancy limits, waves | for every arm |
| R_DECISION4 (REPORT) | PRED4's rule | the lead scores it |
| COMPLETE (MUST) | + FV's 9 gates per cell | |

## 6. Registered prediction and the decision rule

The lead stamps the marker at the top of this file before the run. The json block is the bench's PRED4 (test FV6
keeps them equal). The band is the lead's (the brief).

<!-- PRED4-BEGIN -->
```json
{
 "registered": "REGISTERED <TBD-for-lead-stamp>",
 "source": "lead's T0.3b brief 2026-10-27 (PLAN_speckv_on_cached_nosa.md Task 0.3 option 2; SCORING_fia3.md)",
 "fv_over_p_device": {"216x4": [0.7, 0.4, 1.1], "216x4s1": [0.7, 0.4, 1.1]},
 "thresholds": {"fv_over_p_max": 0.8, "union_build_ms_per_tick_max": 1.0, "layers_per_tick": 32},
 "decision": "ADOPT FV if, at B 216 x 4, FV device time <= 0.8 x P device time (ncu) AND the union build <= 1 ms per tick (32 layers; FV builds in the kernel: 32 x (FV - FVT) device time) AND the FV MUST gates pass. Else NOT MET, with the ncu split (registers, spills, achieved occupancy, DRAM and L2->SM bytes) written down."
}
```
<!-- PRED4-END -->

How the bench reads the rule (R_DECISION4, a REPORT):
- FV / P: the ncu gpu__time_duration.sum of FV's kernel over P's two kernels, one layer.
- The union build per tick: 32 x (FV - FVT) ncu device time. Without an FVT ncu row: 32 x (FV - FVT) CUDA-event medians.
- The FV MUST gates of the cell (names with `_FV_`, and G_block0).

Builder expectations (MODEL, NOT registered; for the lead's judgement only):
- Bytes: FV DRAM read / union in [1.0, 1.1]; L2 -> SM / union in [1.0, 1.2] (each union block loaded once per program,
  only the reached tokens; the tables are small).
- Floors at 216 x 4 (survey section 5): HBM 1.06 GB / 1.94 TB/s = 0.55 ms; tensor work 69 GFLOP = 0.22 ms. P = 1.28 ms
  (ncu, fia3). So FV / P >= 0.43 even at the HBM peak; about 0.55-0.75 at 70-80 % of the peak.
- Registers 150-230 at Triton's default 4 warps (accumulator 64 regs, Q 32, S 32, P 16 per thread); spills 0 expected
  but NOT sure. Shared memory: Triton's default num_stages (3) keeps 2-3 K / V tiles (32 KB each) in flight: about 64-96
  KB per block, so 1-2 blocks per SM, achieved occupancy about 6-19 %. This is the main risk: FP0 lost by latency at 2
  blocks per SM. FV's tile work is lighter (one bias load per token, not per logit; the mask from 4 bytes per tile, not
  a bit per logit), and it pipelines the K / V loads. The launcher has num_warps / num_stages overrides
  (FVArm.launch_kw) for a later sweep; this run uses the defaults (upstream's launch).
- In-kernel build: about 1-3 us per program, 0.05-0.4 ms per tick. The torch build of FVT: 2-10 ms per tick (eager),
  above the rule; FVT is not the candidate.
- X_row_alone_FV: 0 rows differ.

## 7. EXACT mode

- A tile is one block (TILE_SIZE = BLOCK_SIZE = 64). A row's visible blocks come in ascending block id, alone or in
  its request. A block that the row does not list gives S = -inf for the row: the max stays, p = 0, alpha = exp(0) =
  1, so the step is an exact identity once the row has seen block 0 (G_block0). The bytes of the other rows change only
  the reach of a tile (more tokens loaded), and the row's scores of those tokens are -inf (p = 0).
- So FV is expected to be EXACT across the rows of a request (X_row_alone_FV) and across requests (FI1-FI3), with no
  split. The exceptions are signed zeros (acc exactly -0.0 plus +0.0) as sharedkv documents. The CPU test FV5 (I8)
  checks it under the interpreter; the GPU check is the REPORT X_row_alone_FV.
- The 3D split path (segments) is not batch-invariant across segment counts (an online softmax per segment, then a
  merge). It is not used in this run.

## 8. Findings from the code (read, not run)

1. Triton 3.2.0 is in the image (summary.json of bi_bgcc_2191262); the readable 3.2.0 copy is
   /mnt/beegfs/ojerbi/venvs/vllm-cu12/lib/python3.10/site-packages/triton. From it: a constexpr on the left of a compare
   with a tensor works (code_generator.py:536-543 reverse methods; core.py:917 `__rlt__`); the interpreter maps
   static_range to range (interpreter.py:963-988), cumsum to numpy (:849-850, :892-893), the barrier to a no-op (:654),
   tl.dot to np.matmul with the accumulator dtype (:545-552).
2. vLLM's softmax_step sets the running max to 0 when a whole tile row is -inf (triton_attention_helpers.py:404-406).
   So a row whose first tiles are masked gives no NaN (FlashInfer's FP8 hazard does not exist here), but its bits then
   differ from the row alone. G_block0 keeps the first tile visible for every row.
3. With BLOCK_Q 4 and 4 rows per request, half of vLLM's grid exits at once (total_num_q_blocks = R / 4 + B, kept from
   upstream).
4. The rows of a request must share one storage entry and fit one q block (max_seqlen_q <= 4): the launcher checks
   max_seqlen_q; FVArm checks the entry on the host in prepare (as FPArm).
5. The interpreter shares CPU memory with the tensors (interpreter.py:1052-1077), so the in-kernel table stores are seen
   by the loop's loads in the CPU test; on the GPU, the barrier gives the same order inside the program.

## 9. CPU tests for the lead (a compute node, inside the image; git HEAD printed before and after, lessons L37)

    cd /mnt/beegfs/ojerbi/nosi-r2-fia4
    git rev-parse HEAD
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fia4.py benchmarks/Efficiency/tests_cpu/test_r2_fia3.py \
        benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py
    git rev-parse HEAD

- test_r2_fia4.py: FV1 the license header and revision (and the fixtures = the venv files when readable); FV2 every
  changed line tagged, with two plants; FV3 one fork, two launched kernels (upstream's), the helpers called only from
  the kernel, no kernel in the bench; FV4 the torch tables = a set model, the width, the sort precondition, and the
  recorded S1 lists sorted (it prints the fact; a failure here means the s1 cells cannot use FV's build); FV5 the fork
  under TRITON_INTERPRET=1 (a fresh process; 13 checks: fp64 at GQA 4 and at the NOSA-8B geometry, the kernel tables =
  the torch tables = the set model, FVT == FV, the planted mask error DETECTED and local, both witnesses, FI1-FI5, a row
  alone bitwise, the block-edge batch, requests of 1 / 3 / 4 rows, the 3D path, the VOID sort precondition, 7 launcher
  refusals); FV6 completeness and the PRED4 block; FV7 the decision report; FV8 the ncu fields, argv and arguments;
  FV9 / FV10 the step (static shape, DRYRUN refusals and pins; the full DRYRUN runs only on a clean checkout of this
  worktree).
- test_r2_fia3.py and test_r2_fiattn.py must still pass (the bench changed only by additions).

## 10. The step and the GPU time

    bash benchmarks/Efficiency/r2_fia4_step.sh --nosi <full sha of r2-fia4 HEAD> <holder job> <run id>
    DRYRUN=1 bash benchmarks/Efficiency/r2_fia4_step.sh --nosi <sha> <holder job> <run id>

- srun into the existing holder: 1 task, 16 CPUs, 64G, gpu:a100:1, --time 45; host timeout 45 min + 60 s.
- Arms P, FP0, FPU, FV, FVT; cells 32x4, 216x4, 32x4s1, 216x4s1; ncu on (cap 600 s per cell).
- Pins: piece 3 (7c826c1, blob 39ed5e18); the fork = this tree's file (HEAD and clean-tree checks; its blob is checked
  again just before the launch and recorded in command.txt); the S1 tables by path and size.
- Output /mnt/beegfs/ojerbi/r2-runs/fiattn/fia4_<run>_<holder>/ and .step/, both frozen. Claim fiattn_active (shared
  with the fia1-fia3 steps). Caches under /mnt/beegfs/ojerbi/nosa-env/jit/fia4_<run>_<holder>/.

| part | estimate (MODEL; fia3 ran in 133.5 s of bench time) |
|---|---|
| FlashInfer JIT (two variants, cold cache) | 30-40 s (fia3: 32.1 s) |
| Triton compile of the fork (FV and FVT, a few integer specializations) | 30-90 s |
| 4 cells (gates and timing) | 40-90 s |
| 4 ncu children (5 arms each) | 60-120 s |
| total | about 3-6 min; the budget is 2400 s inside STEP_MIN 45 |

GPU memory at 216x4: the store and its poison copy (7.2 GB) as in fia3, FlashInfer workspaces, the union tables (2 x
216 x 256 x 4 B x 2 = 0.9 MB). Below 20 GB.

## 11. Open items and risks

1. GPU compile: the interpreter does not check Triton's GPU lowering (L36). A compile error is caught (ARM_FV, MUST)
   and recorded.
2. Occupancy (section 6): if FV is latency-bound at 1 block per SM, the next step is a num_stages / num_warps sweep
   through FVArm.launch_kw (a launch option, no kernel edit), with the lead's GO.
3. The sort precondition of the in-kernel build: the generator lists are sorted; FV4 checks the recorded S1 lists.
   The engine's lists are canonical (piece 3 needs the same order).
4. 3D (split) path: kept behind its flag and tested on the CPU only; not in this run. At B 32 (64 programs on 108 SMs)
   a split may help, as FP8 helped FP0 there.
5. No CUDA-graph capture of a whole tick here (R_GRAPH_FV captures one launch).

## 12. Shell commands the builder ran outside cat / grep / head / tail / ls / sort / cut / diff / git / wc

- `date -u +%FT%TZ` (the stamp above).
- `mkdir -p` (the fixtures directory, the scratchpad), `cp` (the two pristine upstream copies), one `{ head ...; tail
  ...; tail ...; } >` redirect (the composed starting text of the fork, and once more into the scratchpad as the diff
  reference).
- `tr -d '\r'` once, by mistake, in a read-only pipe of an ncu CSV header (output read only). This breaks the "no tr"
  rule.
- Shell variable assignments only otherwise (three of them were named sed_free, sed_unused and sed_x by mistake: no
  sed program ran).
