# R2 FIA3: the shared-KV verify microbenchmark (plan Task 0.3, survey option 1)

REGISTERED 2026-10-27T16:35:45Z (lead stamp, before any run)

Builder stamp: 2026-10-27T16:09:58Z (`date -u`). Status: SOURCE ONLY, NOT RUN, NOT REVIEWED.
Branch r2-fia3, worktree /mnt/beegfs/ojerbi/nosi-r2-fia3, from r2-fiattn 3948fab.
Plan: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md (Task 0.3).
Survey: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/SURVEY_shared_kv_kernel.md (sections 3.1, 6, 7).

## 1. Scope and the rule

- NO NEW KERNEL. The FP arms use the existing FlashInfer paged prefill kernel. The glue is Python. The CIS bias and
  the mask read live in a FlashInfer JIT variant, which is FlashInfer's own customization hook. The F8 arm already
  uses that hook (R2FiCisBias). No FlashInfer, NOSA or InfLLM-v2 source file changes.
- V1 uses NOSA's own prefill varlen op through the EQ verify module (pinned blob). No change to either.
- Production and the tick do not change. The bench is standalone.

## 2. Files

| file | what |
|---|---|
| /mnt/beegfs/ojerbi/nosi-r2-fia3/benchmarks/Efficiency/r2_fiattn_bench.py | the bench: [fia3] arms FP0, FP8, FPU, V1; s1 cells; B up to 256; new gates and reports |
| /mnt/beegfs/ojerbi/nosi-r2-fia3/benchmarks/Efficiency/r2_fia3_step.sh | the GPU step (new; r2_fiattn_step.sh stays as it was for fia1 / fia2) |
| /mnt/beegfs/ojerbi/nosi-r2-fia3/benchmarks/Efficiency/tests_cpu/test_r2_fia3.py | the CPU tests FC1-FC21 (new) |
| /mnt/beegfs/ojerbi/nosi-r2-fia3/benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py | the old tests FA1-FA16 (unchanged; they must still pass) |
| /mnt/beegfs/ojerbi/nosi-r2-fia3/docs/NOTES_fia3.md | this file |

## 3. How the FP arms map the problem onto FlashInfer

- Wrapper: `BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2", jit_args=R2FiCisBiasMask)`. Option A, as F8:
  one wrapper per KV head, two calls per layer, K / V views `[:, :, h:h+1, :]` of the store.
- One FlashInfer request = one (request, KV head) pair. qo_len = 4 (1 draft row + 3 verify rows).
- GQA: the bench uses the wrapper's OWN GQA support. num_qo_heads = 16, num_kv_heads = 1 per call. FlashInfer packs
  qo_len x group = 4 x 16 = 64 query rows into one CTA tile (packed_qo_len, 0.6.8 scheduler.cuh:505-510;
  CTA_TILE_Q 64, utils.cuh:384-403; the packed row index is qo_idx x 16 + head, prefill.cuh:767). So the "16 heads x 4
  tokens" ARE the 64 rows of the tile. Manual packing (num_qo_heads 1, qo_len 64) is not needed.
- KV pages: the union of the 4 rows' pages, in ascending (block id, slot) order. Page id = entry x S + slot.
- Mask: FlashInfer's own packed layout (0.6.8 variants.cuh:57-63, :78-87). Request (h, n) owns a byte-aligned
  segment. Row u sees token t iff bit u x kv_len + t of the segment is set (little bit order). The bits expand from
  the block bits: union page j of row u gives a 64-token word with its first ext[u, j] bits set (64 for a listed page,
  last_len for the row's last listed page, 0 for a page the row does not list). kv_len = (union pages - 1) x 64 + the
  largest extent in the last union page.
- The variant R2FiCisBiasMask = R2FiCisBias (the same LogitsTransform text) + a LogitsMask with DefaultAttention's
  formula. Its extra tensors are `r2_mask` (uint8) and `r2_mask_indptr` (int32, ABSOLUTE byte offsets, one row per
  KV head).
- plan(): `packed_custom_mask` = our mask buffer (it selects the kCustom kernels), `disable_split_kv=True` (FP0) or
  `fixed_split_size=8` (FP8). The plan HOLDS our device buffers (indptr, indices, last_page_len, mask).
- One plan per tick (FP0): with no split, the plan does not read the union sizes (scheduler.cuh:565-593: one chunk per
  request). Later layers write their union, CSR and mask IN PLACE into the held buffers (`relayer`). REPORT
  R_REUSE_FP0 checks this bitwise against a fresh plan, on a layer with other union sizes.
- The build (union + CSR + mask) has no host read. The bench also times it inside one CUDA graph.

## 4. Arms

| arm | what | role |
|---|---|---|
| P | piece 3 v2 (7c826c1, blob 39ed5e18) | the reference arm |
| F8 | FlashInfer decode (fia1 / fia2 arm), fixed split 8 | continuity |
| FP0 | FlashInfer paged prefill over the union, packed mask, CIS variant, no split | the candidate (survey option 1) |
| FP8 | the same with fixed_split_size 8 (pages of the union) | FAST reference; C is a REPORT (section 8.3) |
| FPU | FP0's call with no mask (plan without packed_custom_mask) | timing only: FP0 - FPU = the mask's cost |
| V1 | NOSA's prefill varlen op (EQ verify_step.varlen_attention) over a contiguous union store | the no-new-code route of Task 1.4 |

V1 layout: union page j at slot j, the union's last page at slot W - 1, W = the largest union. seqused_k = (W - 1) x
64 + the last-page extent. The kernel's causal rule (rows < seqused_k - 4 + u + 1, verify_step.py:37-42) then cuts row
u at its own extent. Precondition V1_pre: per request, one shared tail page, the same extent on both KV heads, and
consecutive extents. The bias is pre-scaled: exp(bias x D^-0.5) (section 8.5).

## 5. Gates (MUST unless marked)

| gate | test | precondition / note |
|---|---|---|
| FP_API | the wrapper takes jit_args, backend, packed_custom_mask, disable_split_kv, fixed_split_size | the 0.5.3 Python is inside the sif; this is read at run time |
| C0_FP_variant_jit_build | R2FiCisBiasMask builds | a failure is a FINDING; the FP arms do not run |
| G_block0 | every listed (row, KV head) lists block 0 (full and altreq batches) | the first KV tile then has a visible token for every row (section 8.3) |
| C_FP0 | max abs(FP0 - fp64 reference) <= 1e-3 on 16 rows | C_FP8 is a REPORT |
| FI1_request_alone_FP0 | each sampled request alone (its own plan) == in the batch, bitwise | the unit is the request |
| FI2_order_FP0 | requests reversed and in random order == unpermuted, bitwise | rows keep their order inside a request |
| FI3_other_requests_FP0 | even requests of "altreq" (odd requests: new q, new lists) == full batch | altreq changes exactly the odd requests, else VOID |
| FI5_poison_FP0 | unread store rows NaN: output finite and bitwise equal | MUST only when every middle union page is fully reached; else a VOID REPORT (a loaded but unread V row gives p 0 x NaN) |
| CW_FP0 | FP0 with bias 0 is far (> 1e-3) from the biased reference | the bias moves the reference >= 1e-2, else VOID |
| MW_FP0 | FP0 with every row given the whole union FAILS C | some row misses a union token, else VOID |
| V1_pre (REPORT) | V1's layout holds | VOID: V1 skipped for the cell |
| C_V1 | max abs(V1 - reference) <= 4e-3 (TOL_V1) | MUST when V1 imports and V1_pre holds. TOL_V1 is a builder choice for the lead to accept: NOSA's route rounds two bf16 outputs and divides in bf16 (verify_step.py:200), about 3 x 2^-9 relative, which passes 1e-3 where abs(out) > 0.2. The gate prints the error, so the lead can also score it against 1e-3 |
| COMPLETE | every expected gate of the cell is present | as fia2, plus the FP0 and V1 gates |

REPORTs: R_X_row_alone_FP0 (section 7), R_REUSE_FP0, R_FP8_chunk_tile0 (section 8.3), R_T (times, section 9),
R_D_bits_vs_P, R_V1_import, R_S1_tables, R_ncu (section 10), R_DECISION (section 6).

## 6. Registered predictions and the decision rule

The lead stamps the marker at the top of this file before the run. The json block is the bench's PRED3 table (test
FC15 keeps them equal). The FP0 / P band comes from the plan. The byte bands come from the survey (section 7); the
survey gives intervals only, so their central value is the interval midpoint (a builder choice).

<!-- PRED3-BEGIN -->
```json
{
 "registered": "REGISTERED 2026-10-27T16:35:45Z (lead stamp, before any run)",
 "source": "PLAN_speckv_on_cached_nosa.md Task 0.3; SURVEY_shared_kv_kernel.md section 7",
 "fp0_over_p_device": {"216x4": [0.75, 0.45, 1.3], "216x4s1": [0.75, 0.45, 1.3]},
 "fp0_dram_read_over_union": [1.05, 1.0, 1.1],
 "fp0_l2sm_read_over_union": [1.15, 1.0, 1.3],
 "p_l2sm_read_over_sum": [0.975, 0.9, 1.05],
 "p_device_ms_per_layer_216x4_central": 1.37,
 "thresholds": {"fp0_over_p_max": 0.8, "tick_overhead_ms_max": 1.0, "layers_per_tick": 32, "hbm_bound_l2sm_drop": 3.0},
 "decision": "ADOPT option 1 (FP0) if, at B 216 x 4, FP0 device time <= 0.8 x P device time AND plan + mask build <= 1 ms per tick amortized (one plan per tick, 32 layer builds) AND the FP0 MUST gates pass. Else read the ncu split: mask-bound -> a compact mask (FPc); plan-bound -> survey option 2 (a fork of vLLM's Triton unified attention) with this measured blocker written down as the skill's rung-4 row. If L2->SM bytes fall 3x (P / FP0) but the time does not move (FP0 / P > 0.8): HBM-bound, stop at the 15 % byte bound."
}
```
<!-- PRED3-END -->

How the bench reads the rule (R_DECISION, a REPORT; the lead scores it):
- Device times: the ncu `gpu__time_duration.sum` of FP0's kernels over P's kernels, per call of one layer.
- Tick overhead: plan (device indptr, both heads) once + 32 x the build per layer. The bench prints it twice: with
  the eager build (CUDA events) and with the build replayed from one CUDA graph. "ADOPT" needs either one <= 1 ms.
- Mask cost: (FP0 - FPU) / FP0 from ncu. HBM-bound flag: P's L2 -> SM bytes / FP0's >= 3 and FP0 / P > 0.8.
- Builder expectations (NOT registered, for the lead's judgement only): the build per layer is launch-bound in eager
  mode (about 35 torch ops), so the eager tick overhead may exceed 1 ms while the graph one stays below it; FP8 shows
  NaN rows (section 8.3).

## 7. EXACT mode

- A100, CTA_TILE_Q 64, D 128, bf16: the dispatch picks NUM_MMA_KV = min(8, (83968 - 16384) / 8192 = 8) = 8. The KV
  tile is 8 x 1 x 16 = 128 tokens = 2 pages (0.6.8 prefill.cuh:2584-2601; bench `kv_tile_tokens`).
- A tile of two pages makes a row's bits depend on which pages its siblings add to the union (the online softmax
  rescales per tile).
- The dispatch reads only the device's shared memory size. There is no public option, no plan argument and no JIT
  argument that pins NUM_MMA_KV. So the one-line pin is NOT added. FP0 is FAST-only.
- Across REQUESTS the arm is still batch-invariant by construction (no split, one CTA per pair, CTA_TILE_Q fixed at
  64). FI1-FI3 test this (MUST).
- R_X_row_alone_FP0 measures the within-request effect: each sampled row alone (4 copies of itself) vs in its
  request. Expectation: some rows differ.

## 8. Findings from the code (read, not run)

1. The sif holds FlashInfer 0.5.3, not 0.6.8.post1. Evidence: every JIT cache under /mnt/beegfs/ojerbi/nosa-env/jit/
   is `flashinfer/.cache/flashinfer/0.5.3` (also the newest, 2026-10-27, e.g. sg5_y2_2195574_evalacc216all); fia2's
   summary records 0.5.3 (SCORING_fia2.md:19); the sif file dates from 2026-09-03. The readable 0.6.8.post1 copy is
   /mnt/beegfs/ojerbi/venvs/vllm-port/lib/python3.10/site-packages/flashinfer/. The bench now accepts both versions
   (FI_VERSIONS) and records the one it finds. The 0.5.3 prefill API is checked at run time (gate FP_API).
2. A custom JIT variant ignores the custom mask unless the variant reads it. FlashInfer applies the mask inside
   DefaultAttention's LogitsMask (variants.cuh:57-63, :78-87). The generated PagedParams of a custom module have no
   maybe_custom_mask field (/mnt/beegfs/ojerbi/nosa-env/jit/fiattn_fia2_2191262/flashinfer/.cache/flashinfer/0.5.3/80/
   generated/r2_fiattn_cisbias_bf16_d128/batch_prefill_config.inc). The kernel calls LogitsMask on EVERY KV tile only
   in mode kCustom (prefill.cuh:2307-2316); run() picks kCustom when plan() got a mask (prefill.py:2325-2331). The
   survey's "0 kernel lines" holds; its "optional LogitsMask" is in fact required.
3. NaN hazards. The online softmax starts at m = -inf, d = 1 and has no -inf guard (prefill.cuh:415-421, :878-907):
   a fully masked FIRST tile gives exp2(-inf - -inf) = NaN. FP0: G_block0 removes it (block 0 is union page 0 and
   every row lists it). FP8: every split chunk starts again at -inf, so a row with no visible token in a chunk's first
   128-token tile gets NaN, and the merge keeps it (state.cuh:53-62). R_FP8_chunk_tile0 counts these cells (MODEL).
   On the generator lists (each row misses about 12 of about 76 union pages) such cells are likely.
4. FlashInfer's own packed-mask path: with `packed_custom_mask`, plan() keeps mask_indptr in BITS
   (`_compute_page_mask_indptr`, prefill.py:1348-1368, :1847-1860), while DefaultAttention adds it to a byte pointer.
   Our variant does not use these fields (it reads r2_mask_indptr in bytes), so the FP arms do not depend on this.
   INFERENCE, not checked on a GPU.
5. CIS scale: piece 3 = FA2-NOSA decode uses logit (q.k + bias) x scale (bi_paged_attention.py:28-30). NOSA's prefill
   and the EQ Path 1 use exp(kv_bias) as a value factor with NO scale (nosa_llama.py:301 in nosi-eqmem;
   verify_step.py:167). If both citations hold, Path 1 and the decode path disagree on the CIS scale. V1 here
   pre-scales the bias to match the bench reference. Open item for the Task 1.4 owner.
6. EQ's `build_varlen_args` refuses requests with different seqused_k (verify_step.py:158-163). V1 builds its own
   VarlenArgs (the same 12 fields) and calls `varlen_attention` unchanged.

## 9. Host cost (R_T for FP0 / FP8)

- call: CUDA events around both head calls + the output interleave (20 warm-ups, 200 timed calls, median).
- runs only: the two FlashInfer runs without the interleave.
- build per layer: union + CSR + packed mask, written in place: CUDA events, host wall time, and one CUDA-graph replay.
- plan per tick: host wall time of plan() for both heads, (a) host indptr, (b) device indptr (the tick's form).
- layer with the tick's plan (FP0): relayer + launch, events.
- tick overhead: FP0 = plan + 32 x build; FP8 = 32 x (plan + build) (its plan depends on the union sizes).
- Also: the union size (mean, max, union / sum), the mask bytes, FlashInfer's mask mode per wrapper, the float
  workspace need (FlashInfer's formula with CTA_TILE_Q 64).

## 10. The ncu cell (REPORT; as fia2's cell)

- One ncu process per cell. Order: 216x4, 32x4, 216x4s1, 32x4s1. Cap min(600 s, the budget left - 120 s).
- Arms in the window: P, F8, FP0, FP8, FPU, V1 (V1 only when V1_pre holds). Each arm planned and warmed outside the
  window, then one call in an NVTX range fia|<arm>. --cache-control all, --clock-control none.
- Kernel filter: the old names + `flash_fwd` and `kernel_topk_to_uint64` (InfLLM-v2, V1).
- Metrics: gpu__time_duration, dram__bytes_read / write (required); lts__t_sectors_srcunit_tex_op_read (L2 -> SM,
  x 32 B), lts__t_sectors_srcunit_tex, L2 / DRAM / SM throughput %, l1tex local-memory load / store bytes (spills)
  when the chip lists them; `--section LaunchStats` for launch__registers_per_thread.
- Per arm: device time, DRAM read / union, L2 -> SM bytes / union and / sum, registers, local-memory bytes, and the
  PRED3 bands for FP0 and P.

## 11. GPU time estimate (MODEL; fia2 ran 4 cells + ncu in 71.4 s at B <= 32)

| part | estimate |
|---|---|
| JIT: R2FiCisBias (decode module) + R2FiCisBiasMask (prefill module), cold cache | 30-60 s (fia2: 14.5 s for one) |
| cell 32x4 (gates, timing) | 20-40 s |
| cell 216x4 (data on the GPU, 3.6 GB store + 3.6 GB poison copy; gates; timing) | 60-120 s |
| cell 32x4s1 | 20-40 s |
| cell 216x4s1 | 60-120 s |
| ncu, per cell (process start + case + about 20 kernels x replay passes) | 32x4: 60-120 s; 216x4: 120-300 s |
| total | about 10-20 min; the budget is 2400 s inside STEP_MIN 45 |

GPU memory at 216x4: store 3.6 GB, poison copy 3.6 GB, F8 float workspace about 0.9 GB, FP8 about 1.1 GB, V1 union
store about 2.5 GB (estimates). Below 20 GB on the 80 GB A100.

## 12. CPU tests for the lead (a compute node; git HEAD printed before and after, lessons L37)

    cd /mnt/beegfs/ojerbi/nosi-r2-fia3
    git rev-parse HEAD
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fia3.py benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py
    git rev-parse HEAD

- test_r2_fia3.py (FC1-FC21): cells and refusals; altreq and bitwise-unchanged fia2 data (against the 3948fab bench
  from git); the union against a set model; the packed mask against a per-bit reference (rows at odd bit phases, a
  request crossing a block edge); the FP twin through FlashInfer's flat addresses = the fp64 reference (L28); FPArm
  end to end with a CPU fake of the wrapper (it holds the plan buffers and reads them at run time), plan reuse and its
  negative control (fresh buffers); the request suite with planted dependencies; G_block0 with a negative case; the
  KV tile model, the CTA tile, the FP8 chunk hazard, the workspace formula; V1's layout, its twin = the reference, its
  precondition; the variant source (DefaultAttention's mask formula, the transform verbatim); FP_API; COMPLETE; the
  ncu summary and argv (a fake ncu query); PRED3 = section 6; R_DECISION; the S1 tables (a synthetic file and the
  recorded one: it prints G_block0, the union size and V1_pre for the recorded lists); the step's static shape and
  DRYRUN; no host read in the build.
- The recorded-table test prints facts the GPU run needs. Read its output before the run: if G_block0 fails on the
  recorded lists, the s1 cells will fail their MUST gate G_block0.

## 13. The step

    bash benchmarks/Efficiency/r2_fia3_step.sh --nosi <full sha of r2-fia3 HEAD> <holder job> <run id>
    DRYRUN=1 bash benchmarks/Efficiency/r2_fia3_step.sh --nosi <sha> <holder job> <run id>

- srun into the existing holder: 1 task, 16 CPUs, 64G, gpu:a100:1, --time 45; host timeout 45 min + 60 s.
- Pins: piece 3 (7c826c1, blob 39ed5e18), EQ verify_step (dde91d4, blob 769047f6), the S1 tables by path and size
  (439458 B; the copy's git blob id is recorded in command.txt).
- Output /mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_<run>_<holder>/ and .step/, both frozen. Claim fiattn_active (shared
  with r2_fiattn_step.sh). Caches under /mnt/beegfs/ojerbi/nosa-env/jit/fia3_<run>_<holder>/.

## 14. Open items and risks

1. The 0.5.3 prefill wrapper API is not readable here. FP_API checks it at run time. If 0.5.3 lacks one of the
   parameters, the FP arms do not run (a FINDING) and the decision needs 0.6.8 (not in the sif).
2. 0.5.3's run() for a JIT module may pass the extra arguments differently from 0.6.8. Our tensor names do not start
   with "maybe", so both forms give the same order (0.6.8 utils.py:1263-1288). If 0.5.3 differs, C_FP0 and MW_FP0 fail.
3. G_block0 on the recorded S1 lists is unknown until the CPU test prints it (section 12).
4. V1 imports infllm_v2 inside the sif (nosa_llama.py imports it at module level, so it exists). If the internal op
   name differs in the installed version, R_V1_import reports a FINDING.
5. FP8 will likely show NaN rows (section 8.3). Its C gate is a REPORT for that reason.
6. CUDA-graph capture of the build is attempted once per FP arm per cell; a capture error is a REPORT field.
7. Not built: the single-call option B for FP0 (it needs q and output copies in (request, head, row) order), the
   compact 1-bit-per-page mask (FPc), CUDA-graph capture of run().

## 15. Shell commands the builder ran outside cat / grep / head / tail / ls / sort / cut / diff / git / wc

- `date -u +%FT%TZ` (the stamp above).
- `sed -e ... | head -0` once, by mistake, inside a read-only listing loop (output discarded). This breaks the
  "no sed" rule.
- Shell builtins: `for`, variable assignment, `echo` in loops.
