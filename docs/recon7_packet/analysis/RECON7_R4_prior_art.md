# RECON7 R4: prior art for the gaps of our K-step NOSA design

Workstream R4 of revision 7 (the consolidated reconciliation; request:
`/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md`).
Author: the R4 agent. Stamp: 2026-10-30T10:01Z (date -u). Status: research only. Nothing was built, run or committed.

## Bottom line (details and file:line evidence in sections 1-7)
1. Multi-query verification: every system batches the R rows of a request into one varlen / prefill attention call.
   No system supports per-row block selection + per-query bias over host-offloaded 64-token pages. Upstream vLLM at
   our GQA group 16 has BLOCK_Q 1, so it does NOT share KV reads across rows. Applies: PARTLY.
2. Shared-KV attention: GQA + row packing into one CTA tile with one page load per tile is standard (FlashInfer,
   FA3 PackGQA). Our fork already has it. The floor is U block reads per (request, KV head); no source reads fewer.
   The model's "U - I extra" credit holds only when the draft row and the verify rows share the pass. Applies: PARTLY.
3. Sparse selection + offload: 8 systems read in code. None shares a selected block across several decode rows of one
   request. ShadowKV and InfLLM hide the fetch behind same-layer work on a second stream; NOSI upstream fetches in
   line (exposed). Applies: PARTLY.
4. Graph-safe execution: host work once per step, fixed-address buffers, batch buckets. Per-layer data-dependent
   page lists defeat FlashInfer plan / run (one sync per layer). vLLM's device-only Triton metadata is the model; our
   fork follows it. Applies: PARTLY.
5. Overlap: exact static-index systems overlap with copy-engine DMA from pinned staging; data-dependent systems
   either speculate the selection (InfiniGen) or accept a per-layer host sync (RetroInfer). No source overlaps an
   exact per-layer selection inside a graph. FA3 TMA / warp specialisation is Hopper-only. Applies: PARTLY.
6. Numerical contract: no source claims bit-equality of a spec / multi-row path with plain decode. vLLM batch
   invariance excludes A100. A masked union tile is an exact no-op per row after the row's first valid tile, so the
   union kernel can be bitwise equal to the same kernel run per row, not to the shipped FA2-NOSA decode (section 6).
7. A faster dense or prefill attention benchmark (FA2, FA3, FlashInfer, vLLM, SGLang, MagicDec, TriForce) is NOT a
   compatible NOSA solution by itself: none of them carries the per-row sparse block set and the CIS bias.

## 0. Method, labels and our shapes

### 0.1 Method
- Channel 1: public repos, shallow-cloned (`git clone --depth 1`) into
  `/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_r4_src/`. We read them with `grep -n` and the Read tool only. We ran
  no code from them. Each "code read" row names the repo, the commit and the file:line ranges that we read.
- Channel 2: WebFetch / WebSearch for papers and official docs. WebFetch gives the page through a summarizer. A row
  labelled "full text read" means that the full page went through that summarizer, not that we read every line.
- Labels: "code read" / "full text read" / "abstract or snippet only". All web reads are dated 2026-10-30.
- Stop rule per topic: two implementations read in code, or three independent sources that agree.
- The topic sections came from four parallel research forks of this agent (topic 1; topic 2; topic 3; topics 4 + 5)
  plus the lead R4 agent (section 1.6 TriForce / MagicDec, section 6 numerical contract). The lead merged them.

### 0.2 Repo revisions read (HEAD of the shallow clone, `git rev-parse`, commit date)

| Repo | Commit | Date | Tag |
|---|---|---|---|
| vllm-project/vllm | bc150f502991 | 2026-05-07 | v0.20.2 (the base of our fork) |
| flashinfer-ai/flashinfer | 0e68a2febc58 | 2025-11-20 | v0.5.3 (the version tried on our code) |
| Dao-AILab/flash-attention | 819ef26aae39 | 2026-10-29 | main |
| sgl-project/sglang | e3718813ec5f | 2026-10-30 | main |
| SafeAILab/EAGLE | cb7e0841fe0c | 2026-02-19 | main |
| FasterDecoding/Medusa | e2a5d20c048a | 2024-04-18 | main |
| mit-han-lab/Quest | 01c1623bf939 | 2025-07-10 | main |
| fla-org/native-sparse-attention | bd67af59b90a | 2025-03-19 | main |
| MoonshotAI/MoBA | b5d58363311d | 2025-04-03 | main |
| thunlp/InfLLM | 12b70798f56e | 2024-04-20 | main |
| OpenBMB/infllmv2_cuda_impl | 93cf2ec28e5a | 2026-02-11 | main |
| thunlp/NOSA | 90045b65e662 | 2026-09-28 | main |
| bytedance/ShadowKV | e51904cdeab7 | 2025-05-01 | main |
| microsoft/RetrievalAttention (RetroInfer) | 03f912c6e917 | 2026-09-14 | main |
| FMInference/FlexLLMGen | 004ffef82b46 | 2024-10-27 | main |
| snu-comparch/InfiniGen | 3754f52d131c | 2024-07-09 | main |
| LMCache/LMCache | 1f3d4f0c4e01 | 2026-10-30 | main |
| Infini-AI-Lab/TriForce | 164c8c0131cf | 2024-08-31 | main |
| Infini-AI-Lab/MagicDec | 09cd671c01bc | 2024-11-28 | main |

### 0.3 Our shapes (read from our files, not from memory)
- Model: `/mnt/beegfs/ojerbi/models/NOSA-8B/config.json`: 32 layers, hidden 4096, `num_attention_heads` 32,
  `num_key_value_heads` 2, `head_dim` 128, bf16. So the GQA group is 16 query heads per KV head.
- NOSA selection: block 64 tokens, `topk` 64 (`nosi/nosi/cache_engine_gpu.py:36`), `init_blocks` 1, `window_blocks` 16,
  pooling stride 16 (`docs/R2_ENGINE_FINDINGS.md:17-18` in `/mnt/beegfs/ojerbi/nosi-r2-i4dc`).
- Joint pass: R = 4 rows per request (1 deferred draft row + 3 verify rows). Our kernel:
  `/mnt/beegfs/ojerbi/nosi-r2-i4dc/benchmarks/Efficiency/fork_vllm_unified_attention.py` (tree commit 2e9e497), a fork
  of vLLM 0.20.2 `triton_unified_attention.py`. Header lines 14-35: BLOCK_Q = 4 rows x 16 heads = a 64-row tile; the
  block table of a program is the UNION of its rows' selected blocks for (request, KV head), ascending; an 8-bit
  per-row extent per union entry masks the score tile to -inf; the CIS bias enters as (q.k + bias) x scale; 2D path
  (no split-KV) by default.
- Its stated reason (header lines 14-18): FlashInfer 0.5.3 `BatchPrefillWithPagedKVCacheWrapper` + custom mask was tried
  on our code (record `/mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_f1a_2195573`): 1.68-1.70x the device time of piece 3, and
  77-100 ms of plan + mask build per tick.
- Transport: a separate commit stream, CUDA events per layer, a C++ packer into pinned staging, the copy engine
  (`nosi/nosi/spec_prefetch.py:8-30`; `nosi/nosi/r2_tick.py:432`).

## 1. Multi-query speculative verification

### 1.1 vLLM v0.20.2 spec decode (code read; vllm bc150f502991)
- Layout. Each request sends its draft tokens + 1 bonus position as ordinary query tokens in one varlen batch.
  `query_start_loc` is a cumulative sum (`vllm/v1/worker/gpu_model_runner.py:1890-1896`). `_calc_spec_decode_metadata`
  (`:2585-2661`) builds the target and bonus logits indices. ONE forward pass and ONE attention call per layer take
  all verify tokens. The mask is causal on absolute position. There is no tree.
- Rejection sampler: `vllm/v1/sample/rejection_sampler.py`, class `RejectionSampler` (`:37`), `rejection_sample`
  (`:379`), `rejection_greedy_sample_kernel` (`:695-744`), `rejection_random_sample_kernel` (`:749`). One Triton
  program per request runs a sequential accept loop. It writes the bonus token only when all drafts are accepted.
- Tile shape (`vllm/v1/attention/ops/triton_unified_attention.py:578-581`, re-read by the lead):
  `BLOCK_M = 16 if num_queries_per_kv <= 16 else next_pow2(num_queries_per_kv)`; `BLOCK_Q = BLOCK_M // num_queries_per_kv`.
  For our group 16: BLOCK_M 16, BLOCK_Q 1. So upstream runs one program per (query token, KV head). The 4 rows of a
  request are 4 programs that each read the same pages. Sharing occurs only by chance in L2. Upstream shares a tile
  across tokens only when the group is below 16 (group 4 gives BLOCK_Q 4).
- 2D vs 3D (`:611-625`): the split-KV 3D path needs `max_seqlen_q <= 1`. A verify batch always takes the 2D path.
- Numerical contract (full text read): `docs/features/batch_invariance.md:18-23` says batch invariance is beta and
  "requires NVIDIA GPUs with compute capability 9.0 or higher" (A100 is 8.0). `docs/features/speculative_decoding/README.md:131-161`:
  spec decode is "theoretically lossless up to the precision limits of hardware numerics"; batch-size changes "may
  cause variations in logprobs". vLLM does NOT claim bit-equality of spec vs non-spec decode.
- Applies to us: PARTLY. The batching and the rejection logic transfer. The attention tile does not: at group 16 it
  gives no shared KV reads across rows. It has no per-row sparse table, no per-query bias, no host pages.

### 1.2 SGLang EAGLE (code read; sglang e3718813ec5f)
- `EagleVerifyInput` (`python/sglang/srt/speculative/eagle_info.py:17-50`) carries `custom_mask`, positions and retrieve
  indices. `generate_attn_arg_prefill` (`:91-167`): `qo_indptr = arange(0, (bs+1)*draft_token_num, draft_token_num)`;
  the mask is a full boolean (query x KV token) mask over prefix + tree, `seq_lens_sum*N + N^2*bs` entries.
- `build_tree_kernel_efficient` (`speculative/eagle_utils.py:174-253`) builds the tree mask on the GPU (modes FULL_MASK,
  QLEN_ONLY, QLEN_ONLY_BITPACKING).
- Backend: FlashInfer `BatchPrefillWithPagedKVCacheWrapper` (`layers/attention/flashinfer_backend.py:782-835, 1024-1032`).
  SGLang wrote a sync-free `fast_prefill_plan` (`:196-305`) from host-known indptrs and packs the mask on the device.
  It asserts `page_size == 1` for a custom mask (`:280`).
- Applies to us: PARTLY. Same "one prefill-style call over N rows" design. The mask is per (query token, KV token) and
  the SAME for all heads; our selection differs per KV head. Our selection as a token mask = 64 tokens x union blocks
  x 4 rows x 2 KV heads, rebuilt every layer and tick. That is the plan + mask cost we measured (77-100 ms per tick).

### 1.3 EAGLE and Medusa reference code (code read; EAGLE cb7e0841fe0c, Medusa e2a5d20c048a)
- EAGLE `eagle/model/cnets.py:776-779`: `tree_mask = eye(total_tokens+1)`, column 0 true, ancestors added.
  `eagle/model/modeling_llama_kv.py:1010-1043` writes `min()` into the tree corner of the mask; attention is eager
  `matmul(Q, K^T)` (`:363`). Medusa `medusa/model/utils_legacy.py:31-111` (`generate_medusa_buffers`, mask at `:58-70`),
  applied in `modeling_llama_kv.py:794-825`.
- The shared prefix is one dense GEMM operand for all tree rows: read once by the GEMM, dense, not paged.
- Applies to us: NO for the kernel. YES as the reference for tree-mask meaning. Our K 3 chain is the degenerate tree,
  so a causal cut is enough.

### 1.4 SpecInfer (arXiv 2305.09781v4; full text read for characters 0-100000 of 118543 of the HTML; PDF not readable)
- Section 4.2: a "topology-aware causal mask" fuses the tree attention of all tokens "in a single kernel" and "reuses
  the same key-value cache across all token sequences". Section 5.2: one thread block per (request, head) with all
  tree tokens inside it.
- The paper argues against redundant COMPUTE on a shared prefix. It does not state a read-once property in bytes.
- Applies to us: PARTLY. One CTA per (request, head) with all rows inside = the layout of our 64-row tile. No sparse
  selection, no offload.

### 1.5 FlashInfer v0.5.3, several queries per request (code read; flashinfer 0e68a2febc58)
- `BatchPrefillWithPagedKVCacheWrapper.plan` (`flashinfer/prefill.py:1523-1697`): `qo_indptr` gives any query count per
  request; `custom_mask` is flattened `sum(q_len[i]*k_len[i])` or bit-packed, per token pair, shared by all heads. The
  docstring: "plan cannot be used in Cuda Graph or in torch.compile" (`:1675`). `fixed_split_size` gives a
  deterministic, batch-size-invariant merge, but CUDA-graph use is "NOT guaranteed"; `disable_split_kv` gives
  determinism inside a graph (`:1657-1664`).
- GQA packing: `include/flashinfer/attention/scheduler.cuh:505-553` (`packed_qo_len = qo_len x gqa_group_size`);
  `include/flashinfer/utils.cuh:349-368` `FA2DetermineCtaTileQ` returns 64 on sm80 for 16 < packed <= 64 (re-read by the
  lead). So R 4 x 16 = 64 packed rows fit ONE CTA per (request, KV head), and each page is loaded once for them.
- Decode wrapper: `use_tensor_cores` routes to the prefill kernel with qo_len 1 (`flashinfer/decode.py:678-680, 709-716`).
  Multi-query goes through the prefill wrapper.
- No stock additive per-query bias (section 2.4 has the JIT route).
- Applies to us: PARTLY (shared reads: yes; per-row per-KV-head sparse selection: only as a token mask; bias: only by
  JIT; host pages: no). Tried on our code: 1.68-1.70x our device time + 77-100 ms plan and mask per tick.

### 1.6 Self-speculation with a sparse-KV draft: TriForce and MagicDec (lead; code read + abstract)
These two are the closest published designs to ours: the SAME weights draft with a sparse KV view, and the target
verifies the K drafts in one pass.
- TriForce (code read, Infini-AI-Lab/TriForce 164c8c0131cf; abstract or snippet only, arXiv 2404.11912v3, COLM 2024).
  - Draft KV: `models/cache.py:117-189` `RetrievalCache`. `init_graph_cache` (`:146-178`) scores chunks of 8 tokens with
    the chunk-mean key and ONE query per head (`assert 1 == query_states.shape[1]`, `:152`), takes top-k per head with
    chunk 0 forced (`:159-162`), and gathers the chunks into a contiguous GPU buffer (`:166-175`). The draft uses that
    set for all gamma steps; `update_graph_cache_retrieval` refreshes it after the round (`:191-194`).
  - Verify: `utils/decoding.py:84-85` runs `gamma2 + 1` tokens through the target in one call over the FULL KV.
    Offload variant `OffloadingFlashSimpleCache` (`models/cache.py:63-115`): pinned host KV (`:76-77`), and EVERY layer
    copies the WHOLE layer KV into one GPU buffer with `non_blocking=True` (`:106-107`). A `load_stream` exists (`:83`)
    but the copy at `:106-107` is on the current stream.
  - Applies to us: PARTLY. The draft selection is shared by all rows of a round and is per head (our draft row has its
    own selection per step). The verifier is DENSE full attention. NOSA's verifier must use NOSA's own per-row sparse
    selection to stay the NOSA model, so TriForce's "dense verify" does not transfer. Its offload path streams whole
    layers: no selective fetch, no overlap evidence in code.
- MagicDec (code read, Infini-AI-Lab/MagicDec 09cd671c01bc; abstract or snippet only, arXiv 2408.11049v5).
  - The abstract: at large batch and long context the KV read is the bottleneck and "scales with both sequence length
    and batch size"; a sparse-KV draft; up to 2.51x for Llama3.1-8B at batch 32-256. This is the same regime claim as
    our original model (verify rows share the KV read).
  - Verify: `Engine/StreamingLLM/backend.py:120-151`. `pre_verify` calls FlashInfer
    `BatchPrefillWithPagedKVCacheWrapper.plan(qo_indptr = arange * dec_len, causal=True)` OUTSIDE the compiled region
    (`:138-151`). `run` is a `torch.library` custom op (`:65-76`) inside `torch.compile(mode="max-autotune",
    fullgraph=True)` (`:107-116`). The wrapper uses `use_cuda_graph=True` with fixed index buffers (`:55-64, 86-90`).
  - The draft uses a SEPARATE compressed page set (`draft_paged_kv_indices`, `:78-90`; StreamingLLM budget).
  - Applies to us: PARTLY. It confirms the "R rows of a request in one paged prefill call, GQA-packed" verify, with
    plan outside the graph. The verifier is dense over the full KV (GPU-resident). No per-row sparse selection, no bias,
    no offload.

### Topic 1 verdict
- Applies to us: PARTLY. Every production system batches the R rows of a request into one varlen / prefill call.
  FlashInfer and SpecInfer put all rows x the GQA group into one CTA, so a page is read once per (request, KV head).
  Upstream vLLM at group 16 does NOT (BLOCK_Q 1).
- No system we read supports per-row block selection + per-query bias over host-offloaded 64-token pages in its verify
  call. No system promises bit-equality of the spec path with plain decode; vLLM's batch invariance excludes A100.
- Gap addressed: SHARED KV READS ACROSS ROWS (the kernel-level form). Not addressed: exposed fetch, per-layer host
  work, dense rows.
- What it needs: no stock library call fits. The closest (FlashInfer prefill + custom mask) was measured slower on our
  code. Our fork (BLOCK_Q = R x group, union table, per-row extents, bias) is the matching shape, so the
  new-kernel route is justified for this one piece. A faster dense spec-verify benchmark (vLLM, SGLang, MagicDec,
  TriForce) is NOT a compatible NOSA solution by itself: it reads every KV page, and its verifier is not the NOSA
  sparse model.

## 2. Paged / shared-KV attention for several queries of one sequence

### 2.1 FlashAttention-2 (code read; flash-attention 819ef26aae39)
- Grid (m_block, batch, q_head) (`csrc/flash_attn/src/flash_fwd_launch_template.h:64`; split grid `:107`). A CTA maps its
  query head to a KV head with `bidh / h_h_k_ratio` (`flash_fwd_kernel.h:155, 161, 596-600`). So 16 CTAs read the same
  KV page, once per query head.
- Exception, "seqlenq_ngroups_swapped": when seqlen_q == 1 and there is no ALiBi / window, the API folds the group into
  M (`csrc/flash_attn/flash_api.cpp:1318-1326`; also `:432, :623`). With R = 4 (seqlen_q 4) it is OFF.
- Flash-Decoding split: `num_splits_heuristic` (`:282-316`), at most 128 splits (`:334-344`).
- Paged KV: `page_block_size % 256 == 0` (`:614, :1306`, re-read by the lead). Bias: ALiBi slopes and softcap only.
- Applies to us: NO (page 64 < 256; group not packed at seqlen_q 4; no per-row mask; no bias).

### 2.2 FlashAttention-3 (code read; same commit, `hopper/`)
- PackGQA folds seqlen_q x (h / h_k) into the M tile (`hopper/heuristics.h:9-16`; `hopper/flash_api.cpp:426-438`). It is
  forced on for arch < 90, non-TMA paged KV and split (`:377, :429`). An sm80 build path exists
  (`hopper/static_switch.h:138-146`; `hopper/setup.py:67-76, 206`). README: "optimized for Hopper", "Requirements:
  H100 / H800" (`README.md:31, 45`).
- One page table per sequence: all rows see the same page list. Causal, local window, softcap. No custom mask, no
  additive per-element bias.
- Applies to us: PARTLY. PackGQA with R 4 = our 64-row tile. No per-row selection, no bias. The sm80 path is not the
  documented target and we did not verify it.

### 2.3 Flash-Decoding (full text read: https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Split KV into chunks; run FlashAttention per chunk with its log-sum-exp; reduce. It helps when batch x heads leaves SMs
  idle. It does not discuss several query rows per request.
- Applies to us: PARTLY. A parallelism tool, not a read-sharing tool. It changes the reduction order (section 6).

### 2.4 FlashInfer v0.5.3 (code read; flashinfer 0e68a2febc58)
- Plan / run: `plan()` builds the schedule once; `run()` is reused across layers (`flashinfer/decode.py:583-647`). The
  prefill `plan` does host work: `paged_kv_indptr.to("cpu")`, `seq_lens.cpu()`, `max(...).item()`
  (`flashinfer/prefill.py:1724-1735`) and `segment_packbits` of the mask (`:1694-1704`).
- CUDA graph: fixed buffers, fixed batch size (`decode.py:673-696`; `prefill.py:1355-1358`); the row count cannot pass the
  count of the first plan (`:1737-1749`); a custom mask needs `mask_indptr_buf` (`:1776-1786`).
- GQA in the query tile: packed row = qo_idx x group + head (`include/flashinfer/attention/prefill.cuh:445, 2131-2151`).
  The paged kernel loads each KV tile into shared memory once for all packed rows with cp.async double buffering
  (`page_produce_kv`, `prefill.cuh:2204-2207, 2340-2355`). Same structure as our fork.
- Masks and bias: the mask bit sits at qo_idx x kv_len + kv_idx (`variants.cuh:78-87`): NO head index. Our per-KV-head
  selection needs one "request" per (request, KV head) with num_kv_heads 1, or a JIT variant.
  `REGISTER_LOGITS_TRANSFORM` gets (qo_idx, kv_idx, qo_head_idx, kv_head_idx) (`variants.cuh:67-76`) and JIT modules take
  extra tensors (`jit/attention/modules.py:470-532`). So a per-query additive bias is possible by a custom JIT variant,
  not by a stock flag.
- Cascade: `MultiLevelCascadeAttentionWrapper` (`flashinfer/cascade.py:216-260`) and
  `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` (`:548`) compute shared pages once and merge levels with
  `merge_state` on the LSE (`:33-86`).
- Paper (abstract or snippet only, arXiv 2501.01005): block-sparse KV format, JIT templates, load-balanced scheduling
  compatible with CUDA graphs.
- Applies to us: PARTLY. "One page load per (request, KV head) for R x 16 rows" exists here; our fork has it too. Our
  selection changes per LAYER, so "plan once, run for all layers" does not hold: a plan per layer = per-layer host
  work + syncs. Cascade maps onto our sets (intersection I ~ 60 = shared level, private blocks = unique level), but
  the LSE merge changes the reduction order and the reads stay at U per (request, KV head).

### 2.5 vLLM v0.20.2 (code read; vllm bc150f502991)
- PagedAttention v1 / v2: grid x = query head, y = sequence, z = partition
  (`csrc/attention/attention_kernels.cuh:106-148, 573-576`). One query per sequence; each query head reads its KV head
  on its own. v2 partition 512 (`paged_attention_v2.cu:45`); block sizes 8 / 16 / 32 (`:153-159`). `IS_BLOCK_SPARSE` is
  a fixed vertical-stride + local pattern (`attention_kernels.cuh:207-235, 382-388`), not a per-query selection.
  Applies to us: NO.
- Unified Triton attention: section 1.1 (BLOCK_Q 1 at group 16). Our fork's BLOCK_Q 4 is the ONLY change that makes
  the read shared across rows. `VLLM_BATCH_INVARIANT` forces the 2D kernel (`triton_unified_attention.py:34, 624`).

### Topic 2 verdict
- Applies to us: PARTLY. Packing the GQA group and the query rows into one tile, with each page loaded once per CTA,
  is standard (FlashInfer `prefill.cuh`, FA3 PackGQA). Our fork already does it. No stock call adds per-row,
  per-KV-head selection + a per-query additive bias + page 64 + sm80 batch invariance.
- Gap addressed: SHARED KV READS ACROSS ROWS. The ceiling these sources confirm: one read per UNION block per
  (request, KV head), i.e. U. No source reads fewer than U blocks in the verify call. The original model's credit for
  I ~ 60 blocks "already ready from the draft" therefore needs the draft row and the verify rows to share one pass
  (or residency, so the bytes are already in HBM / L2). The tile alone gives at most U reads per (request, KV head),
  not U - I. Also: per-layer host work is a known cost of plan / run when the page list changes each layer.
- What it needs: no new library for the shared read (our fork has it). A FlashInfer route needs a JIT variant (bias +
  per-head mask), one "request" per (request, KV head), and a device-side plan; measured at 1.7x + 77-100 ms per tick.
  A cascade route needs a new merge and breaks bit-exactness. A faster dense / prefill path (FA2, FA3, FlashInfer) is
  NOT a compatible NOSA solution by itself: none carries the per-row sparse block set and the CIS bias.

## 3. Fused sparse selection and residency (offloaded KV)

### 3.1 Quest (code read; mit-han-lab/Quest 01c1623bf939; paper arXiv 2406.10774 not read)
- Per-page channel max / min of K as metadata (`kernels/include/decode/decode_page.cuh:416-445`). Three launches per
  decode step: `decode_estimate` (page upper bound), `decode_topk`, `decode_sparse_attn` over the selected pages
  (`quest/models/QuestAttention.py:135-157`; wrappers `quest/utils/__init__.py:171-264`). Selection per head
  ([N, page_budget-1], `:262`). Batch 1, q_len 1 (`QuestAttention.py:94`). GPU-resident; no offload, no prefetch.
- Applies to us: NO (not fused, no offload, no multi-row path).

### 3.2 NSA (code read; fla-org/native-sparse-attention bd67af59b90a; paper arXiv 2502.11089 not read)
- `parallel_nsa_kernel_topk` (`native_sparse_attention/ops/parallel.py:338-468`): compressed-key scores summed over the
  G heads of a GQA group (`:438`), in-kernel top-n by bitonic merge (`:441-454`). Selection is fused with scoring, not
  with attention.
- `parallel_nsa_fwd_kernel` (`:471-555`): one program per (token, KV head, V tile); the G heads form a [G, BK] Q tile
  (`:515-518`); the kernel loops over the token's selected blocks and loads each block once for the G heads
  (`:528-549`). Block list per (token, KV head): `block_indices[B, T, H, S]` (`:1026`). Different tokens = different
  programs: R rows read a shared block R times. GPU-resident.
- Applies to us: PARTLY. "GQA group = one tile, one list per KV head" = our 16-head group. Our fork goes further
  (4 rows x 16 heads over the union). No cross-row sharing, no offload.

### 3.3 MoBA (code read; MoonshotAI/MoBA b5d58363311d; paper arXiv 2502.13189 not read)
- Chunk key = chunk mean (`moba/moba_efficient.py:335-340`); fp32 gate einsum (`:345-347`); `torch.topk` per (head,
  query) (`:365`). KV-chunk-major schedule: per (chunk, head), collect every query that chose that chunk (`:375-386`),
  gather the chunk KV with `index_select` (`:330, 406-414`), one `flash_attn_varlen` over (chunk, query-set) pairs
  (`:105`), LSE merge with `index_add_` (`:146-165`).
- Each chunk is read once for all queries that chose it: the only system read that inverts the loop to share KV
  across query rows. Limits: prefill / training only (no decode path); float `index_add_` in scatter order (not
  bit-exact, not batch-invariant); an extra gather copy.
- Applies to us: PARTLY, as a design reference. Our union tile does the same inside one CTA with no gather copy and no
  atomics.

### 3.4 InfLLM (code read; thunlp/InfLLM 12b70798f56e; paper arXiv 2402.04617 not read)
- Representative keys per block (`inf_llm/attention/context_manager.py:268-278`). Selection per request, SHARED by all
  heads and all query tokens of a chunk (the query is averaged over len_q, `:373-378`). Top-k goes to the host with
  `.cpu().tolist()` (`:163`): a host sync per layer.
- GPU block cache with LRU / score eviction (`CudaCache`, `:6-27`; `remove_lru_blocks` `:246`; `:495-515`); blocks on
  pinned CPU memory (`:40-45`). `MemoryUnit.load`: `non_blocking` copies + an event (`:63-96`). A Python loop places the
  blocks into a contiguous buffer and skips blocks already in place (`:398-419`).
- Overlap: selection + load run on a side stream (`GLOBAL_STREAM`, `:492-520`) while the local-window attention runs on
  the main stream (`:486-489`); the main stream joins (`:522-523`). Then dense attention over the assembled buffer
  (`:526-531`).
- Applies to us: PARTLY. The side-stream overlap of the fetch with the local-window attention of the SAME layer is a
  pattern for EXPOSED FETCH (our window is resident). Its per-layer host sync and Python loop are the PER-LAYER HOST
  WORK problem; do not copy them. Its "one selection for all tokens of a chunk" is a semantic change, not exact.

### 3.5 InfLLMv2 CUDA implementation (code read, README only; OpenBMB/infllmv2_cuda_impl 93cf2ec28e5a)
- Stage 1 scores against compressed keys and reduces over the query group; top-k is outside the kernel; stage 2 is
  block-sparse attention on the selected blocks (`README.md:8-58`). Block 64, 64 blocks (`:186-187`).
- Applies to us: we use it already. No offload, no multi-row sharing.

### 3.6 NOSA / NOSI upstream (code read; thunlp/NOSA 90045b65e662; arXiv 2510.13602v2 full text read for characters 0-100000 of 135359)
- Decode layer `nosi/nosi/nosa_llama.py:520-607`: fused QKV + eviction head (`:533-535`); stage 1
  `infllmv2_attn_stage1_fast` (`:569-578`); pooling + top-k inside one captured CUDA graph `after_pooling_graph`
  (capture `:445-476`, replay `:589`), selection `[H_kv, B, topk]`.
- `decode_update_kv` (`nosi/nosi/cache_engine.py:125-170`): `diff_offload` (`flash_cache_engine/diff_offload_kernel.cu:7-100`)
  compares old and new block maps ON THE GPU and gives a load mask and slot reuse; `flash_h2d_from_mask`
  (`flash_cache_engine/flash_h2d_mask.py:10-86`, re-read by the lead: Triton kernel, grid (B, H, M), `tl.load` from the
  host pointer) moves only the missed 64-token blocks with SM loads from pinned host memory (UVA).
- `flash_attn_nosa_with_kvcache` with the per-token bias over a compact slot buffer (`nosa_llama.py:600-606`;
  `topk*block_size` slots per request, `cache_engine.py:88-93`).
- The fetch is in line on the same stream in each layer (the lead's grep finds no `stream` in `cache_engine.py` or
  `nosa_llama.py`): EXPOSED, SM loads, not the copy engine. The tail write-back is outside the graph (`cache_engine.py:158`).
- Paper: locality bound k_e / k (Theorem 2). No speculative decoding, no multi-query rows, no statement on transfer /
  compute overlap.
- Applies to us: YES (our base). It confirms that the reference design has no cross-row sharing and no fetch / compute
  overlap. Our tree adds the commit stream and the copy engine (section 0.3).

### 3.7 ShadowKV (code read; bytedance/ShadowKV e51904cdeab7; paper arXiv 2410.21465 not read)
- Selection: `batch_gemm_softmax` of Q against chunk landmarks, max over the GQA group, `topk` per KV head
  (`models/kv_cache.py:616-645`). `reorder_keys_and_compute_offsets` splits hits and misses against GPU-cached chunks
  (`:643`; `kernels/gather_copy.cu:187-235`).
- V fetch: `gather_copy_with_offsets` moves hits on device and reads misses directly from pinned host memory
  (`kv_cache.py:647-653`; `gather_copy.cu:237-260`) on `copy_stream`. K is rebuilt on the GPU from low-rank factors +
  RoPE (`kv_cache.py:655-666`) on the main stream, overlapped with the V fetch (`models/base.py:150-161`). Then dense
  `flash_attn_with_kvcache` over the gathered buffer (`:164`). Decode q_len 1 (`kv_cache.py:619`).
- Applies to us: PARTLY. Pattern: overlap the miss fetch with independent compute of the same layer on a second
  stream. The hit-aware gather = our diff + load mask. The low-rank K changes the numbers: not compatible with our
  exact contract.

### 3.8 RetroInfer (code read; microsoft/RetrievalAttention 03f912c6e917; paper arXiv 2505.02922 not read, abstract via search snippet)
- Selection: `batch_gemm_softmax` against cluster centroids, sum over the GQA group, `topk`
  (`cache_hub/retroinfer_cache.py:765-769`). Cluster ids go to pinned CPU memory with a blocking `copy_` (`:770`; graph
  path `:844`). A C++ thread pool runs `batch_access` (GPU-block LRU, hit / miss lists in pinned memory;
  `retroinfer_kernels/src/wave_buffer_cpu.cpp:44-189, 705-792`).
- `gather_copy_and_concat` builds an execution buffer from the steady zone, GPU hits and CPU misses with SM-initiated
  reads (`retroinfer_cache.py:797-804`); `weighted_flash_decoding` with an LSE merge of the estimation zone (`:807-815`);
  admit with `gather_copy_and_scatter` (`:819-826`).
- CUDA graphs: the layer is split into 4 graphs (topk, estimation, attention, update; `:831-858`, capture `:863-954`);
  host work runs between replays. Its separate copy stream serves the prefill offload (`:326-333, 517-520`), not the
  decode fetch.
- Applies to us: PARTLY. It shows "graph pieces + a host LRU between them" with a host sync per layer. That is the
  per-layer host work gap, not a fix for it.

### Topic 3 verdict
- Shared KV reads across rows: none of the 8 systems shares a selected block across several decode query rows of one
  request in one attention call. They share only across the GQA heads of a KV head (NSA, InfLLMv2, ShadowKV,
  RetroInfer, NOSI), or merge the selection across heads and tokens (InfLLM: a semantic change). MoBA's chunk-major
  schedule shares across queries, but only in prefill, with gather copies and atomic merges. Our union tile has no
  direct prior art in these systems.
- Exposed fetch: NOSI upstream fetches in line (exposed). ShadowKV and InfLLM overlap the fetch with independent work
  of the same layer on a second stream.
- Per-layer host work: InfLLM and RetroInfer sync to the host per layer. NOSI keeps the diff on the GPU and captures
  pooling + top-k in a graph. NOSI is the model to follow.
- Dense rows: not touched by any system here.
- Applies to us: PARTLY. Gaps addressed: EXPOSED FETCH (side-stream overlap with same-layer work) and PER-LAYER HOST
  WORK (keep the GPU diff + captured selection). Not addressed: shared reads across spec rows, dense rows.
- What it needs: no new kernel for the overlap. It needs stream / event / graph changes and a split of the attention
  into a window part (runs while the selected blocks arrive) and a selected part, joined by an LSE merge. The merge
  changes the reduction order, so it needs a bit-exact (or declared-tolerance) gate against our reference path
  (section 6). A faster kernel from these papers (MoBA, NSA training kernels) is NOT by itself a compatible NOSA
  decode solution.

## 4. Graph-safe execution

### 4.1 vLLM v0.20.2 (code read; vllm bc150f502991)
- Modes NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE (default). PIECEWISE keeps "cudagraph incompatible
  ops (i.e. some attention ops) outside the cudagraph"; FULL_AND_PIECEWISE captures a full graph for decode batches
  and a piecewise graph for prefill / mixed batches (`vllm/config/compilation.py:580-616`, lead re-read `:589-604`).
  With `use_inductor_graph_partition`, every op tagged `cudagraph_unsafe` splits the graph (`:643-661`).
- Buckets: default capture sizes `[1,2,4] + range(8,256,8) + range(256,max+1,16)`, `max = min(2*max_num_seqs, 512)`
  (`:666-681`). The dispatcher rounds a batch up to the next captured size (`vllm/v1/cudagraph_dispatcher.py:71-90`). With
  spec decode a "uniform decode" batch has query length 1 + K (`:37-41`); a padded descriptor has
  `num_reqs = padded_tokens // (1+K)` (`:131-152`).
- Backend capability `AttentionCGSupport` (`vllm/v1/attention/backend.py:499-513`): ALWAYS / UNIFORM_BATCH (the
  spec-verify case) / UNIFORM_SINGLE_TOKEN_DECODE / NEVER. Triton unified attention is ALWAYS
  (`vllm/v1/attention/backends/triton_attn.py:127`, lead re-read). Its `build()` only packs device tensors that exist
  (query_start_loc, seq_lens, block_table, slot_mapping); no host planning (`:211-263`). The 3D split buffers are
  allocated once at a threshold that snaps to a capture size, so each graph keeps one path (`:157-199`).
- Host bookkeeping outside the graph: persistent `CpuGpuBuffer` pairs (pinned CPU + GPU), `copy_to_gpu` = a
  non_blocking copy into fixed addresses (`vllm/v1/utils.py:108-136`; `gpu_model_runner.py:679-723`). `CUDAGraphWrapper`
  asserts at replay that input addresses equal the captured ones (`vllm/compilation/cuda_graph.py:279-282, 343-355`).

### 4.2 SGLang (code read; sglang e3718813ec5f)
- One graph per batch-size bucket (`capture_bs`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:394-400`);
  static width `decode_num_tokens_per_req(num_draft_tokens)`; capture mode TARGET_VERIFY under spec decode
  (`:362-375`). Replay pads to the bucket (`_pad_to_bucket`, `:1657-1666`), fills registered static buffers
  (`:1675-1683`) and re-initialises attention metadata ON THE HOST before replay (`_init_replay_attn_metadata`,
  `:1718-1726`).
- FlashInfer verify wrapper with `use_cuda_graph=True`; index, indptr and custom-mask buffers fixed per bucket
  (`layers/attention/flashinfer_backend.py:1181-1210`); mask buffer `max_num_tokens*max_context_len` uint8 (`:1140-1147`).
- EAGLE draft graph runner: `speculative/eagle_draft_cuda_graph_runner.py` listed, not read.

### 4.3 FlashInfer v0.5.3 plan / run (code read; flashinfer 0e68a2febc58)
- `plan()` runs on the host and is not captured: `.to("cpu")` of indptrs, `.item()` of max lengths
  (`flashinfer/prefill.py:1713-1735`), the C++ host scheduler (`:1883-1908`). In graph mode it copies new indices and the
  mask into the user buffers; batch size fixed; rows <= the captured count (`:1737-1786`; docstring `:1355-1391`). The
  scheduler pads the CTA count to an upper bound (`include/flashinfer/attention/scheduler.cuh:530-542, 602-606`).
- Only `run()` is captured. A new page list needs one host `plan()` before the replay. SGLang and MagicDec (1.6) do that
  ONCE PER STEP for all layers, because the page list is the same in every layer.

### 4.4 PyTorch documentation (full text read through the summarizer)
- CUDA Graph Trees (https://docs.pytorch.org/docs/2.5/torch.compiler_cudagraph_trees.html): graphs are skipped on
  eager-input mutation, CPU ops, several devices, free unbacked symbols (dynamic shapes) and incompatible ops
  (`aten._local_scalar_dense`). Trees "re-record CUDAGraph for every unique shape"; pad to a few shapes. Eager inputs
  are copied to static addresses; parameters and earlier graph outputs are not. One shared memory pool; overwritten
  outputs need `mark_step_begin()` or a clone.
- CUDA semantics (https://docs.pytorch.org/docs/2.5/notes/cuda.html): no CPU sync (`.item()`), no control flow "based on
  CPU or GPU data", no dynamic shapes, same addresses on replay. Several streams may be captured if they fork from and
  rejoin the capture stream. `make_graphed_callables` graphs only the safe parts.
- MagicDec (1.6) is a worked example: `torch.compile(mode="max-autotune", fullgraph=True)` with the FlashInfer `run()` as a
  `torch.library` custom op inside and `plan()` outside (`Engine/StreamingLLM/backend.py:65-76, 107-151`).

### Topic 4 verdict
- Pattern in vLLM, SGLang, FlashInfer and MagicDec: all host work (scheduling, page tables, plan, mask packing) runs
  ONCE PER STEP before the replay; kernels read fixed-address buffers; batches pad to a few buckets.
- Our case differs: the selection of layer l depends on device data of layer l. A host plan per layer (FlashInfer)
  would need a sync per layer. That matches our fia3 blocker (77-100 ms of plan + mask per tick). RetroInfer (3.8)
  accepts this and splits its graphs into 4 per layer.
- The model to follow is vLLM's Triton unified backend: all metadata on the device, grid fixed by bucket, captured as
  ALWAYS. Our fork inherits it, and its BUILD_UNION builds the union table inside the program
  (`fork_vllm_unified_attention.py:506-516`: "no host plan, no extra launch").
- Applies to us: PARTLY. Gap: PER-LAYER HOST WORK. Needs: no new kernel. Keep selection, union build and residency
  bookkeeping on the device; pad B to buckets; keep every host decision (pin, evict, pack, plan) at the step
  boundary, outside the captured region. FlashInfer plan / run cannot carry per-layer, data-dependent page lists
  without one sync per layer.

## 5. Producer / copy / consumer overlap

### 5.1 FlexGen / FlexLLMGen (code read; FlexLLMGen 004ffef82b46)
- Three streams: load_weight, load_cache, store_cache (`flexllmgen/flex_opt.py:618-620`); loads under
  `torch.cuda.stream(...)` (`:653-666, 680-717`). Loop per layer j: issue `load_weight(j+1)`, `load_cache(j+1)`, run
  `compute_layer(j)`, `store_cache(j-1)`, then `sync()` (`:1011-1032`) = `torch.cuda.synchronize()` (`:793-795`, lead
  re-read `:795`): a full host sync per layer.
- Layer j+1's dense KV is known in advance; no data-dependent selection.
- Applies to us: PARTLY (one-layer-ahead pipeline yes; dense prefetch and per-layer sync no).

### 5.2 InfiniGen (code read; InfiniGen 3754f52d131c)
- While layer i runs, a `speculation_stream` computes a partial query of layer i+1 from layer i's hidden state with
  partial Q weights and a partial key cache, and takes a top-k of tokens
  (`speedup/flexgen/infinigen/pytorch_backend.py:399-403`; `speedup/infinigen/infinigen/kv_selection_controller.py:28-63`;
  `.item()` host sync at `:58`). `select_kv` gathers the rows with `F.embedding` (`:5-25`); a `prefetch_cache_stream`
  copies them to the GPU for the next layer (`speedup/flexgen/infinigen/flex_opt.py:430-465, 812-822`; path confirmed by the
  lead's grep). A miss is not corrected.
- Applies to us: PARTLY. Early selection of layer i+1 addresses exposed fetch, but the selection is a speculation, not
  NOSA's exact selection. Compatible with our contract only as a PREFETCH HINT that NOSA's exact selection then checks
  (misses fetched on the critical path).

### 5.3 ShadowKV (code read; ShadowKV e51904cdeab7)
- `get_value_stream.wait_stream(curr)`; the V gather from host-resident V runs on the copy stream while the main stream
  rebuilds K from low-rank factors + RoPE; then `curr.wait_stream(copy)` and flash attention (`models/base.py:150-164`).
  The gather is an SM-driven CUDA kernel that takes the CPU tensor (`models/kv_cache.py:647-653`;
  `kernels/gather_copy.cu:237-260`); a hit pass splits hits (device copy from the previous step) and misses (host copy)
  (`gather_copy.cu:187-235`). That the kernel reads mapped pinned memory is the fork's inference; allocation flags not
  read.
- Applies to us: YES for residency and hit reuse; PARTLY for overlap. Our own earlier kappa measurement (memory note of
  the RetroInfer evaluation, 2026-09-20, not re-verified in this workstream): an SM-driven Triton gather on a side
  stream did NOT overlap (kappa 0.91-0.95); the copy engine from contiguous pinned staging did (0.006-0.05).

### 5.4 RetroInfer wave buffer (code read; RetrievalAttention 03f912c6e917)
- Per layer (`cache_hub/retroinfer_cache.py:754-828`): GPU top-k over centroids; `cluster_ids.copy_(cI)` to pinned CPU
  (`:770`, a blocking copy: host sync per layer); `wave_buffer.batch_access()` on a CPU thread pool (`:794`; pool
  `:160-166`); `gather_copy_and_concat` from steady zone + GPU hits + host misses (`:797-804`); attention;
  `wave_buffer.sync()` and `gather_copy_and_scatter` to admit pages (`:818-826`). Graph variant: 4 graphs per layer
  with host work between replays (`:831-860`; capture `:863-928`). A copy stream and per-device events serve KV offload
  (`:327-333, 515-528, 605-641`).
- Applies to us: YES as a direct analogue. A published offload-sparse system accepts per-layer host work and splits
  its graphs around it. It hides the cost with CPU threads; it does not remove it.

### 5.5 LMCache (code read; LMCache 1f3d4f0c4e01)
- `VLLMPagedMemLayerwiseGPUConnector.batched_to_gpu` yields once per layer. Per layer: the current stream waits
  `load_stream` for the previous layer; on `load_stream`, a `non_blocking` copy from pinned CPU objects into a contiguous
  GPU staging buffer (copy engine), then a scatter kernel into the paged KV (`single_layer_kv_transfer`)
  (`lmcache/inprocess/v1/gpu_connector/gpu_connectors.py:1203-1322`; streams `:1114-1115`). Store mirrors it with
  `store_stream.wait_stream(current)` (`:1400-1435`).
- The token set is known before the step (prefix reuse); no data-dependent selection.
- Applies to us: PARTLY. It is the "contiguous pinned staging -> copy engine -> device scatter" layout that our kappa
  note found to overlap. Its index set is static.

### 5.6 vLLM v0.20.2 KV offloading worker (code read)
- `vllm/v1/kv_offload/worker/cpu_gpu.py:255-290` builds per-block src / dst / size arrays on the host, takes a pooled
  stream and timing events, orders jobs with `wait_event`; it waits on the compute stream only for GPU -> CPU
  (`:292-316`); it polls completion with `event.query()` without a host block (`:332-344`).
- `swap_blocks_batch` submits all copies in one `cuMemcpyBatchAsync` (CUDA >= 12.8) or falls back to one
  `cudaMemcpyAsync` per block (`csrc/cache_kernels.cu:78-150`, lead re-read `:100-117`): copy-engine DMA.
- Step granularity, whole blocks, outside graphs. Applies to us: PARTLY (a library pattern for batched DMA with
  event completion).

### 5.7 FlashAttention-3 producer / consumer (code read; flash-attention 819ef26aae39; abstract read, arXiv 2407.08608)
- Warp specialisation with TMA, interleaved matmul and softmax, FP8 (abstract). sm90 kernel: register split between
  producer and MMA warpgroups (`hopper/flash_fwd_kernel_sm90.h:82-83, 309, 361`). README: "optimized for Hopper",
  H100 / H800 (`README.md:30-45`).
- sm80 path in `hopper/`: a `kStages` cp.async multistage pipeline, comment "we don't use TMA for Sm8x"
  (`hopper/mainloop_fwd_sm80.hpp:31, 51, 418-428, 479`).
- Applies to us: NO. A100 has no TMA / warpgroup MMA. The sm80 analogue is cp.async multistage pipelining (Triton
  `num_stages`), which our fork gets from Triton's pipeliner (fork header lines 29-33, fia5).

### 5.8 CUDA programming guide (abstract or snippet only; https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- Host buffers for async copies "must be pinned and page-locked"; otherwise the copy "will revert to a synchronous
  behavior which will not overlap". The copy-engine count (asyncEngineCount) passage: not found within the fetched
  pages.

### Topic 5 verdict
- Two families overlap a data-dependent sparse fetch: (1) speculate the next selection early (InfiniGen;
  approximate); (2) accept a per-layer sync and hide it with CPU threads + hit reuse (RetroInfer, ShadowKV). Exact,
  static-index systems (FlexGen, LMCache, vLLM offload) overlap one layer (or one step) ahead with copy-engine DMA from
  pinned memory + a device scatter.
- No source read here overlaps an EXACT, per-layer, data-dependent selection inside a CUDA graph.
- Applies to us: PARTLY. Gaps: EXPOSED FETCH and PER-LAYER HOST WORK.
- What it needs: no new attention kernel. A prefetch HINT (the draft row's or the previous step's selection; NOSA
  locality), packed into contiguous pinned staging, moved by the copy engine (`cudaMemcpyAsync` /
  `cuMemcpyBatchAsync`), then a device-side check that fills only the misses before attention. Our tree already has
  this structure (`spec_prefetch.py:15-30`: lag-1 snapshot, one descriptor job, commit stream, per-layer events). Open:
  whether such copies can sit in a captured graph when their addresses depend on data (fixed staging lanes make the
  copy node addresses fixed; the host packing then must run a step ahead).

## 6. Numerical contract across the sources (lead)

- Batch invariance, prior art:
  - vLLM v0.20.2: beta, compute capability >= 9.0 only (`docs/features/batch_invariance.md:18-23`); it forces the 2D
    unified kernel (`triton_unified_attention.py:616-625`). Spec decode: "lossless up to the precision limits of
    hardware numerics" (`docs/features/speculative_decoding/README.md:131-161`). No bit-equality claim.
  - FlashInfer v0.5.3: `fixed_split_size` (deterministic, batch-invariant, graph use not guaranteed) or
    `disable_split_kv` (`flashinfer/prefill.py:1657-1664`).
  - He / Thinking Machines, "Defeating Nondeterminism in LLM Inference" (full text read, through the summarizer:
    https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): split-KV reductions and a different
    grouping of cached vs new KV break batch invariance; the fix is a FIXED SPLIT SIZE (not a fixed split count) and the
    same KV layout for every query count.
  - Three independent sources agree: invariance = no split-KV or a fixed split size + one tile order for every row.
    None of them claims bit-equality between a union / multi-row kernel and a per-row reference kernel with a
    different tile order.
- What upstream online softmax implies for our union tile (lead analysis of upstream code, not a source claim):
  `softmax_step` (`vllm/vllm/v1/attention/ops/triton_attention_helpers.py:361-383`, v0.20.2) sets
  `m_j = max(M, rowmax(S))`, forces `m_j = 0` when it is -inf (`:374`), `P = exp(S - m_j)`, `alpha = exp(M - m_j)`.
  - For a row whose running max M is finite, a tile that is fully masked for that row gives m_j = M, alpha = 1, P = 0:
    L and acc do not change. So the union entries that a row does not select are exact no-ops for that row.
  - If a row's FIRST tiles are fully masked (M = -inf), m_j becomes 0 and the row's later max starts from 0, not from
    its true first max. The result is mathematically equal but the rounding can differ. Our rows all list block 0
    (the init block; plan gate `G_block0`, `PLAN_speckv_on_cached_nosa.md:100`), and the union is ascending, so the
    first union tile is valid for every row. That gate is what keeps the no-op property.
  - Consequence: the union kernel can be BITWISE equal to a per-row run of the SAME kernel (same 64-token tile, same
    ascending block order, same bias path). It cannot be bitwise equal to the shipped FA2-NOSA decode, which uses
    other tile pairs and the window slot order (our own note `nosi/nosi/verify_rows.py:39-40`). The bit-exact
    reference must therefore be named per gate: "union kernel vs the same kernel per row" can be exact; "vs the
    shipped decode" needs a declared tolerance.
- Any split-KV (Flash-Decoding) or cascade / LSE merge (FlashInfer cascade, InfLLM, RetroInfer, MoBA) adds a second
  reduction order. With a fixed split size it can stay batch-invariant (the blog), but it is not bitwise equal to the
  2D path.

## 7. Synthesis: gap -> prior art -> what it would need

| Gap in our design | Closest prior art (section) | Does it close the gap for our shapes? | Needs |
|---|---|---|---|
| Shared KV reads across the R rows of a request | FlashInfer GQA packing (1.5, 2.4); FA3 PackGQA (2.2); SpecInfer CTA layout (1.4); MoBA chunk-major (3.3) | Partly. The mechanism exists; our fork already has it (64-row tile over the union). Floor = U reads per (request, KV head). None reads fewer than U. The model's U - I credit is valid only if the draft row and the verify rows read the I blocks in the SAME pass (our joint pass has the draft row in the tile), or if bytes are counted as already in HBM, not as zero HBM reads. | No new library. Measure the union kernel's DRAM bytes against U x 64 x 2 x D x 2 B per (request, KV head, layer) to confirm read-once. |
| Exposed fetch (H2D of selected misses) | InfLLM side stream (3.4); ShadowKV copy_stream (3.7); topic 5 sources | Partly. Same-layer overlap needs independent work in the same layer: the resident-window part of attention or the next rows' stage 1. Cross-layer prefetch needs the next layer's selection, which depends on this layer's output (InfiniGen-style speculation: see topic 5). | Stream / event changes (our commit stream exists) + a window / selected split with an LSE merge (a reduction-order change) or a prediction of the next selection. |
| Per-layer host work | NOSI GPU diff + captured selection (3.6); FlashInfer plan cost (2.4); SGLang sync-free plan (1.2); MagicDec plan outside the compiled region (1.6); topic 4 | Partly. The prior art moves planning to the device (our BUILD_UNION does, fork `:506-516`) or does it once per step outside the graph. Per-layer host decisions (InfLLM, RetroInfer) are the anti-pattern. | Keep tables device-built; keep host work per step, not per layer; see topic 4. |
| Dense rows (verify rows charged as full model rows in the GEMMs) | All spec systems run the verify tokens as ordinary rows of one batched GEMM (vLLM 1.1, SGLang 1.2, MagicDec 1.6) | The prior art does not reduce dense row work; it shares the WEIGHT read by batching rows into one GEMM (our `verify_rows.py:9, 16`: one GEMM call per weight over E = B x R rows). The original model charges 3 GEMM(B) + GEMM(B x K) per cohort. Our joint pass has E = B x R = B x 4 rows (3 verify + 1 deferred draft); the row and pass count against that formula belongs to the dense-work reconciliation, not to this survey. | Nothing new. Measure that one GEMM per weight is issued over E rows (ncu weight bytes per tick, plan Task 1.4). |

## 8. Source list with labels (all accessed 2026-10-30)

### Code read (shallow clones, commits in section 0.2; file:line ranges in the topic sections)
- vLLM v0.20.2: `vllm/v1/attention/ops/triton_unified_attention.py` (140-355, 560-639), `triton_attention_helpers.py`
  (361-383), `vllm/v1/worker/gpu_model_runner.py` (1890-1896, 2585-2661, 679-723), `vllm/v1/sample/rejection_sampler.py`
  (37, 379, 695-749), `vllm/v1/attention/backend.py` (499-513), `vllm/v1/attention/backends/triton_attn.py` (127,
  140-264), `vllm/config/compilation.py` (575-684), `vllm/v1/cudagraph_dispatcher.py` (30-152), `vllm/v1/utils.py`
  (108-136), `vllm/compilation/cuda_graph.py` (279-282, 343-355), `vllm/v1/kv_offload/worker/cpu_gpu.py` (255-344),
  `csrc/cache_kernels.cu` (78-150), `csrc/attention/attention_kernels.cuh` (106-148, 207-235, 382-388, 573-576),
  `csrc/attention/paged_attention_v2.cu` (45, 100-124, 153-159).
- FlashInfer v0.5.3: `flashinfer/prefill.py` (1350-1399, 1523-1786, 1806-1915), `flashinfer/decode.py` (583-742),
  `flashinfer/cascade.py` (33-86, 216-260, 548), `flashinfer/jit/attention/modules.py` (470-532),
  `include/flashinfer/attention/prefill.cuh` (445, 1609-1672, 2100-2169, 2204-2207, 2290-2359),
  `include/flashinfer/attention/scheduler.cuh` (500-609), `include/flashinfer/attention/variants.cuh` (40-97),
  `include/flashinfer/utils.cuh` (349-368).
- FlashAttention: `csrc/flash_attn/flash_api.cpp` (282-344, 432, 614, 623, 737, 1296-1335),
  `csrc/flash_attn/src/flash_fwd_launch_template.h` (64, 107), `flash_fwd_kernel.h` (155, 161, 596-600);
  `hopper/flash_api.cpp` (377, 426-438, 1033), `hopper/heuristics.h` (9-16), `hopper/static_switch.h` (138-146),
  `hopper/setup.py` (67-76, 206), `hopper/mainloop_fwd_sm80.hpp` (31, 51, 418-428, 479),
  `hopper/flash_fwd_kernel_sm90.h` (82-83, 309, 361), `README.md` (30-45).
- SGLang: `python/sglang/srt/speculative/eagle_info.py` (1-170), `speculative/eagle_utils.py` (174-253),
  `layers/attention/flashinfer_backend.py` (196-305, 758-1309), `model_executor/runner/decode_cuda_graph_runner.py`
  (350-409, 1640-1729).
- EAGLE: `eagle/model/cnets.py` (776-779), `cnets1.py` (771-779), `eagle/model/modeling_llama_kv.py` (363, 1010-1043).
- Medusa: `medusa/model/utils_legacy.py` (31-111), `medusa/model/modeling_llama_kv.py` (794-825).
- TriForce: `models/cache.py` (63-197), `utils/decoding.py` (41-163).
- MagicDec: `Engine/StreamingLLM/backend.py` (30-175, 253-290).
- Quest: `quest/models/QuestAttention.py` (90-179), `quest/utils/__init__.py` (171-264),
  `kernels/include/decode/decode_page.cuh` (416-445).
- native-sparse-attention: `native_sparse_attention/ops/parallel.py` (338-555, 1026), `modeling_nsa.py` (103-125).
- MoBA: `moba/moba_efficient.py` (105, 146-165, 330-414).
- InfLLM: `inf_llm/attention/context_manager.py` (6-96, 163, 246, 268-278, 313, 373-378, 398-419, 486-531).
- infllmv2_cuda_impl: `README.md` (8-58, 186-187) only.
- NOSA: `nosi/nosi/nosa_llama.py` (445-476, 520-619), `nosi/nosi/cache_engine.py` (23-172),
  `nosi/nosi/flash_cache_engine/diff_offload_kernel.cu` (7-100), `flash_h2d_mask.py` (10-86).
- ShadowKV: `models/kv_cache.py` (612-691), `models/base.py` (135-169), `kernels/gather_copy.cu` (187-260).
- RetrievalAttention (RetroInfer): `cache_hub/retroinfer_cache.py` (160-166, 326-333, 515-528, 605-641, 754-954),
  `retroinfer_kernels/src/wave_buffer_cpu.cpp` (44-189, 705-792).
- FlexLLMGen: `flexllmgen/flex_opt.py` (618-620, 653-717, 790-797, 1011-1050).
- InfiniGen: `speedup/flexgen/infinigen/pytorch_backend.py` (395-406), `speedup/flexgen/infinigen/flex_opt.py`
  (430-469, 812-826), `speedup/infinigen/infinigen/kv_selection_controller.py` (1-64).
- LMCache: `lmcache/inprocess/v1/gpu_connector/gpu_connectors.py` (1114-1115, 1203-1322, 1400-1435).

### Full text read (through the WebFetch summarizer)
- vLLM docs in the v0.20.2 tree: `docs/features/batch_invariance.md` (whole file),
  `docs/features/speculative_decoding/README.md` (131-161) (read locally).
- SpecInfer, arXiv 2305.09781v4 HTML, characters 0-100000 of 118543 (https://arxiv.org/html/2305.09781v4).
- NOSA, arXiv 2510.13602v2 HTML, characters 0-100000 of 135359 (https://arxiv.org/html/2510.13602v2).
- Flash-Decoding blog, https://crfm.stanford.edu/2023/10/12/flashdecoding.html (2023-10-12).
- PyTorch 2.5 CUDA Graph Trees, https://docs.pytorch.org/docs/2.5/torch.compiler_cudagraph_trees.html.
- PyTorch 2.5 CUDA semantics, https://docs.pytorch.org/docs/2.5/notes/cuda.html (sections used).
- H. He / Thinking Machines, "Defeating Nondeterminism in LLM Inference",
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/.

### Abstract or snippet only
- SpecInfer abstract, https://arxiv.org/abs/2305.09781 (v4).
- FlashInfer paper, https://arxiv.org/abs/2501.01005.
- FlashAttention-3, https://arxiv.org/abs/2407.08608.
- TriForce, https://arxiv.org/abs/2404.11912 (v3, COLM 2024).
- MagicDec, https://arxiv.org/abs/2408.11049 (v5, 2025-04-02).
- RetroInfer, search snippets for arXiv 2505.02922 and PVLDB 19(5) (https://www.vldb.org/pvldb/vol19/p1016-lu.pdf, not
  opened).
- NOSA search snippet (arXiv 2510.13602; github.com/thunlp/NOSA).
- CUDA programming guide, asynchronous execution,
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (sections 2.5.1, 2.5.2.3).

### Not read (implementation evidence is code only)
- Papers: Quest 2406.10774, NSA 2502.11089, MoBA 2502.13189, InfLLM 2402.04617, ShadowKV 2410.21465, RetroInfer
  2505.02922 full text, FlexGen, InfiniGen, LMCache papers.
- Not found within the fetched pages: the CUDA guide's copy-engine count (asyncEngineCount) passage; the PyTorch
  "stable" CUDA Graph Trees page (redirect / empty / 404 on the stable and raw-GitHub URLs; the 2.5 page was used).
- Not read: SGLang `speculative/eagle_draft_cuda_graph_runner.py`; the FA3 sm80 path was not verified to build or run.

### Local records cited (not online sources)
- `/mnt/beegfs/ojerbi/nosi-r2-i4dc/benchmarks/Efficiency/fork_vllm_unified_attention.py` (header 10-35; 495-516).
- `/mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_f1a_2195573` (named by the fork header; this workstream listed it, did not
  re-analyse it).
- `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md:92-104` (Task 0.3, G_block0).
- `/mnt/beegfs/ojerbi/nosi-r2-i4dc/nosi/nosi/verify_rows.py:1-40`, `spec_prefetch.py:1-50`, `r2_tick.py:432`.
- Kappa values in 5.3 / 5.5: from the agent memory note of the RetroInfer evaluation (2026-09-20). Not re-verified
  here; the lead must cite the exported record before use.

## 9. Rule slips (reported, not hidden)
- Topic 2 fork: one `sed -n '0p'` used as a filter on grep output (`grep -n ... flash_api.cpp | sed -n '0p'`). sed is
  forbidden. It printed nothing, wrote nothing and ran no downloaded code. It is a breach.
- Topic 4/5 fork: a stray token `| awk_placeholder 2>/dev/null` at the end of one grep pipeline (the `gpu_connectors.py`
  grep). No such command exists; it failed. No awk or other interpreter ran. Reported because the token names a
  forbidden program family.
- Topic 1 fork: a WebFetch of the SpecInfer PDF saved a binary copy into the harness tool-results directory (automatic,
  not chosen). The Read tool could not open it (no pdftoppm); nothing ran.
- Lead: `git clone` created the new directory `/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_r4_src/` (git creates
  the target path; no mkdir was run). Two repos (TriForce, MagicDec) and three (RetrievalAttention, NOSA,
  infllmv2_cuda_impl) were cloned after the first batch; all into the allowed directory. No code from any clone ran.
- No GPU job, no sbatch, no queue directory touched, no code tree edited, no commit. The only file written is this
  report.

## 10. Commands and web fetches

### Lead (R4 agent)
Shell (login node; cat / grep / head / tail / ls / wc / date / git only):
1. `cat HUMAN_RECON_REQUEST_2026-10-30.md; ls nosi/nosi`
2. `cd /mnt/beegfs/ojerbi/nosi-r2-i4dc; git log --oneline -1; ls benchmarks/Efficiency | head; grep -rn ...` (head dims, block size)
3. `grep -rln / grep -rn` for config json and the fork kernel structure
4. `ls docs; grep -n ... docs/NOTES_fia4.md` (file absent)
5. `grep -rhn ...` in `docs/R2_ENGINE_FINDINGS.md`, `docs/R2_BASE.md`, `docs/NOTES_verify1.md`, `cache_engine_gpu.py`
6. `grep -n "" docs/R2_BASE.md | head -40; grep -rn '"num_key_value_heads"' /mnt/beegfs/ojerbi --include=config.json -l`
7. `cat /mnt/beegfs/ojerbi/models/NOSA-8B/config.json`
8. `ls -la .../tmp_sc/`; `ls .../recon7_r4_src`
9. `git clone -q --depth 1 --branch v0.20.2 vllm`; `--branch v0.5.3 flashinfer`; `flash-attention`
10. `git clone -q --depth 1` sglang, EAGLE, Medusa, Quest, ShadowKV, InfLLM, MoBA, FlexLLMGen, InfiniGen, LMCache,
    native-sparse-attention
11. `git clone -q --depth 1` RetrievalAttention, NOSA, infllmv2_cuda_impl; `git rev-parse / log -1 / describe` per repo
12. `grep -n "stream\|non_blocking\|pin_memory"` in `r2_tick.py`, `spec_prefetch.py`
13. `ls -la` of the task-output directory; `date -u`
14. `ls /mnt/beegfs/ojerbi/r2-runs/fiattn/`; `ls .../analysis/plan | grep`
15. `ls fia3_f1a_2195573; grep -rhn "Task 0.3" -A12 PLAN_speckv_on_cached_nosa.md`
16. `grep -n "BUILD_UNION\|..." fork_vllm_unified_attention.py; grep -n "window_blocks\|topk\|..." nosa_llama.py`
17. `grep -n` softmax / -inf in vLLM `triton_unified_attention.py`; `grep -rn "def softmax_step" -A22` in helpers
18. `git clone -q --depth 1` TriForce, MagicDec; `git rev-parse`; `ls`
19. `ls -R models utils; grep -n "def \|class " models/cache.py` (TriForce)
20. `grep -n` verify / gamma / graph in TriForce `utils/decoding.py`, `models/modeling_llama.py`
21. `ls -R Engine; grep -rn` FlashInfer / compile / verify in MagicDec
22. `grep -n "graph\|CUDAGraph\|capture"` MagicDec backend; `grep -rn` in `Engine/utils.py`
23. `grep -n -i "streaming\|draft_paged_kv_indices\|def pre_spec"` MagicDec backend
24. `date -u` (twice)
25. `grep -n "bitwise\|never be\|..." docs/NOTES_verify1.md`
26. `grep -n "page_block_size % 256\|..."` FA2 `flash_api.cpp`
27. `grep -n "FA2DetermineCtaTileQ" -A20` FlashInfer `utils.cuh`
28. `ls flash_cache_engine; grep -n` NOSA `flash_h2d_mask.py`; `grep -n stream` NOSA `cache_engine.py nosa_llama.py`
29. `grep -n AttentionCGSupport ...; grep -n FULL_AND_PIECEWISE ...; grep -n cuMemcpyBatchAsync ...; grep -n torch.cuda.synchronize` FlexLLMGen
30. `grep -rln prefetch_cache_stream; grep -rn speculation_stream` (InfiniGen)
31. `grep -n "TOPIC_45_FOLLOWS"; wc -l; grep -c ""; tail -5; grep -n "^## "` on this report
Read tool: the request file; `fork_vllm_unified_attention.py` 1-60, 495-524; `spec_prefetch.py` 1-50;
`verify_rows.py` 1-40; vLLM `triton_unified_attention.py` 565-629; TriForce `models/cache.py` 63-197; MagicDec
`Engine/StreamingLLM/backend.py` 30-164.
Web: WebSearch "RetroInfer vector-storage approach ... github"; WebSearch "NOSA native offloadable sparse attention
github thunlp arXiv"; WebFetch https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/;
WebFetch https://arxiv.org/abs/2408.11049; WebFetch https://arxiv.org/abs/2404.11912.

### Topic 1 fork
Shell: `grep -n` on vLLM `triton_unified_attention.py`, `gpu_model_runner.py`, `rejection_sampler.py`; `ls docs/features`
+ `grep` on `batch_invariance.md` and the spec-decode docs; `ls` SGLang `speculative` + `grep` on `eagle_utils.py`,
`eagle_info.py`, `layers/attention`, `flashinfer_backend.py`; `grep -rn` on Medusa and EAGLE model files; `grep -n` on
FlashInfer `prefill.py` (twice), `prefill.cuh`, `scheduler.cuh`, `decode.py`. Read tool on the ranges in 1.1-1.5.
Web: WebFetch https://arxiv.org/abs/2305.09781; https://arxiv.org/pdf/2305.09781v4 (unreadable);
https://arxiv.org/html/2305.09781v4.

### Topic 2 fork
Shell (17 commands): `grep -n` on FA2 `flash_api.cpp` (one with the sed slip), `hopper/flash_api.cpp`,
`hopper/heuristics.h`, `hopper/static_switch.h`, `hopper/setup.py`, `README.md`, `hopper/*.h`,
`flash_fwd_launch_template.h`, `flash_fwd_kernel.h`; FlashInfer `decode.py`, `prefill.py`, `prefill.cuh`,
`variants.cuh`, `jit/attention/modules.py`, `cascade.py`, `utils.cuh`; `ls vllm/csrc/attention` + `grep` on
`attention_kernels.cuh`, `paged_attention_v1.cu`, `paged_attention_v2.cu`, `triton_unified_attention.py`; `grep -rln
VLLM_BATCH_INVARIANT`; `wc -l` + `grep` on `batch_invariance.md`. Read tool on the ranges in 2.1-2.5.
Web: WebFetch https://crfm.stanford.edu/2023/10/12/flashdecoding.html; https://arxiv.org/abs/2501.01005.

### Topic 3 fork
Shell: read-only `ls`, `grep -n`, `grep -rn`, `head`, `tail`, `wc -l` on Quest, native-sparse-attention, MoBA, InfLLM,
NOSA, ShadowKV, RetrievalAttention, infllmv2_cuda_impl (files in 3.1-3.8); some commands began with `cd` or `D=...`
(shell built-ins). Read tool: `QuestAttention.py` 90-179, NOSA `cache_engine.py` 23-172, `nosa_llama.py` 520-619, ShadowKV
`kv_cache.py` 612-691, RetroInfer `retroinfer_cache.py` 754-833.
Web: WebFetch https://arxiv.org/html/2510.13602v2.

### Topics 4-5 fork
Shell (32 commands, `grep` / `ls` / `cut` / `head` / `tail` / `wc`; command 28 had the stray-token slip): vLLM
`compilation.py`, `backend.py`, `triton_attn.py`, `flashinfer.py`, `flash_attn.py`, `cudagraph_dispatcher.py`,
`utils.py`, `gpu_model_runner.py`, `cuda_graph.py`, `kv_offload/worker`, `csrc` (`swap_blocks_batch`); SGLang
`model_executor`, runner, speculative, `decode_cuda_graph_runner.py`, `base_cuda_graph_runner.py`,
`flashinfer_backend.py`; FlashInfer `prefill.py`, `scheduler.cuh`; FlexLLMGen `flex_opt.py`; InfiniGen
`pytorch_backend.py`, `flex_opt.py`, `kv_selection_controller.py`; ShadowKV models / kernels, `gather_copy.cu`,
`copy.cuh`; RetrievalAttention `retroinfer_cache.py`; LMCache `cache_engine.py`, `gpu_connector/__init__.py`,
`gpu_connectors.py`; flash-attention `README.md`, `hopper/setup.py`, `hopper/mainloop_fwd_sm80.hpp`,
`hopper/flash_fwd_kernel_sm90.h`. Read tool on the ranges in 4.1-5.7.
Web: WebFetch https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html (redirect);
https://docs.pytorch.org/docs/2.15/torch.compiler_cudagraph_trees.html (empty);
https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/torch.compiler_cudagraph_trees.md (404);
https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/torch.compiler_cudagraph_trees.rst (404);
https://pytorch.org/docs/2.5/torch.compiler_cudagraph_trees.html (301);
https://docs.pytorch.org/docs/2.5/torch.compiler_cudagraph_trees.html; https://docs.pytorch.org/docs/2.5/notes/cuda.html;
https://arxiv.org/abs/2407.08608; https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html (contents only);
https://docs.nvidia.com/cuda/cuda-c-programming-guide/02-basics/asynchronous-execution.html (404);
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html.
