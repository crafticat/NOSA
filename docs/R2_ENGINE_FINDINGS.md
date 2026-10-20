# R2 engine findings: the shipped decode path (F1-F4)

Pin: NOSI `6e6f80a` (the engine that the main repository pins).
Branch: `feature/nosi-r2-m2`. Status: SOURCE ONLY. No part of this file
comes from a GPU run.

Notation.

* `N/` is `nosi/nosi/` in this tree. `FCE/` is `nosi/nosi/flash_cache_engine/`.
  `IV2/` is `dependencies/infllmv2_cuda_impl/`.
* `S` is the prompt length. All requests in a shipped batch have the same `S`
  (`N/max_pooling_fused.py:78`).
* Decode step `i = 1, 2, ...`. Step 1 is the warm-up step. Step `i` writes the
  token at position `p = S + i - 1`.
* Shipped constants: block 64, topk 64 (`N/nosa_llama.py:680-681`), so the
  tail slot is slot 63 and `(topk - 1) * 64 = 4032`. Compression window 32,
  stride 16 (`N/nosa_llama.py:677-678`). `init_blocks = 1`,
  `window_blocks = 16` (`N/nosa_llama.py:674-675`; the benchmark keeps these
  defaults, `benchmarks/Efficiency/test_nosa_pg19.py:21-28`).

The CPU tests in `tests/test_r2_engine_findings.py` check every citation
in this file (`test_citations_hold`) and run the CPU reproducers named below.

## Summary

| ID | Condition | First step | Consequence | Loud or silent | In the shipped run |
|----|-----------|-----------|-------------|----------------|--------------------|
| F1 | the tail block id passes the warm-up pooling grid | `i = 65 - S % 64` | the gather overwrites the live tail slot | silent | LIVE after 64 - S % 64 steps |
| F2 | `tail_cis_len` reaches 32 | `i = 16 - S % 16` (`i = 1` when `S % 16 = 15`) | the compressed-cis table never grows | silent | LATENT (no effect inside the F1 window) |
| F3 | decode reaches `p = S + 1024` | `i = 1025` | (a) silent no-op of the cis write, (b) RuntimeError in the tail-bias write; (c) bias gather bound is `S + 8192` | (a) silent, (b) loud, (c) silent | (b) stops the run; (c) LATENT |
| F4 | `p % 64 == 63` | every 64th step | attention reads rows `[0, 4032)` only: it skips the whole tail block, the current token included | silent | LIVE, every 64th step |

F4 was not in the design's list. I found it while I mirrored the shipped
update order for the CPU twin (see the F4 section).

## F1: the decode is valid only for `64 - S % 64` steps

### Code path

1. Prefill puts the tail block in slot 63 and records its id `S // 64`:
   `N/cache_engine.py:309` (`_tail_block_len_on_gpu = S % 64`), `:312`,
   `:321`, `:324` (`_block_map[..., 63] = S // 64`), `:327`
   (`_cache_lens = 4032 + S % 64`).
2. The warm-up step sizes the pooling buffers once:
   `N/nosa_llama.py:800` (`warmup = not self.has_buffers`), `:817`
   (`total_len = S + 1`), `:818` (`out_len = ceil(total_len / 64)`, which is
   `S // 64 + 1`), `:819` (`pooling_buf_all`, last dim `out_len`), `:828`
   (`mask_buf`, last dim `out_len`).
3. The warm-up step captures one CUDA graph: `N/nosa_llama.py:459` to `:490`.
   It contains `nosa_pooling` (`:462-472`, with `max_seqlen_comp` passed as a
   Python int at `:465`) and the two `torch.topk` calls (`:476-478`,
   `:488-490`). Every later step only replays it (`:612`).
4. The pooling grid takes `M` from the output buffer:
   `N/max_pooling_fused.py:83` (`M = q_score_pooled.shape[-1]`) and `:86`
   (`grid = (H, B, M)`). So `M = out_len` for the whole run. `torch.topk` over
   the last dim of an `out_len`-wide buffer returns indices in
   `[0, out_len)`. THE SELECTION CAN ONLY NAME BLOCKS `0 .. S // 64`.
5. The tail rename: step `i` writes position `p` into slot 63, row `p % 64`
   (`N/cache_engine.py:349-351`). When the tail block is full (`:357-359`),
   section D (`:401-409`) writes the block to the host and renames the tail
   slot to the next block: `:407` (`_block_map[..., 63] += 1`).

### Condition (arithmetic)

At the start of step `i` the tail slot holds block `(S + i - 1) // 64`.
That id is equal to `out_len = S // 64 + 1` first when
`S + i - 1 >= 64 * (S // 64 + 1)`, that is when

    i = 65 - S % 64.

So steps `1 .. 64 - S % 64` are valid. Step `65 - S % 64` is the first bad
step. For the contexts in use, `16128` and `65280`, `S % 64 = 0`, so the
window is 64 steps. `HANDOFF-CODEX.md` (main repository, "Known engine
limits", item 2) states this limit as "one 64-token block past the prompt";
this section gives the exact count and the mechanism.

The grid can emit block ids `0 .. out_len - 1 = 0 .. S // 64` only. The
forced local window is the last 17 ids of that range
(`N/max_pooling_fused.py:61`, `local_blocks = window_blocks + 1` at `:105`).

### Consequence (the first bad step)

1. `N/cache_engine.py:349-351` writes the new token into slot 63, row 0.
2. `N/cache_engine.py:363` calls `diff_offload(old_map, topk_idx, ...)`.
   Slot 63 holds id `out_len`. No entry of `topk_idx` is equal to it (step 4
   above). In `FCE/diff_offload_kernel.cu:36-42` the old-hit loop finds no
   match, so `sm_old_hit[63] = 0`: THE KERNEL TREATS SLOT 63 AS FREE. There
   is no tail protection in `diff_kernel` (`:35-98`).
3. Block `out_len - 1` (the block that step `i - 1` completed) is in
   `topk_idx`, because the forced local window gives it `+inf`. It is no
   longer in `old_map` (it was renamed). So `new_hit = false` (`:51-61`) and
   it needs a free slot.
4. Both id lists have 64 entries and no duplicates (`torch.topk` returns
   distinct indices; `diff` keeps the map duplicate-free). So the number of
   free slots is equal to the number of missing ids. The k-th missing id
   goes to the k-th free slot (`:66-90`). EVERY FREE SLOT GETS A LOAD, slot 63
   included: `load_map[63] >= 0` (`:92-98`).
5. `N/cache_engine.py:381-382`: the two gathers copy 64 host rows into slot
   63 (`FCE/flash_h2d_mask.py:61-62`, `FCE/flash_h2d_mask_bias.py:58-59`,
   `:80-81`). They overwrite the token that item 1 wrote.
6. Attention (`N/nosa_llama.py:634-641`) reads `_cache_lens = 4033` rows. Row
   0 of slot 63 now holds a row of another block, not the current token.
7. Later steps write rows 1, 2, ... of slot 63 while `_block_map[..., 63]`
   names the fetched block (`N/cache_engine.py:377`). At the next block
   completion, section D writes slot 63 to the host as the new block
   (`:404-405`). THE HOST COPY IS THEN WRONG TOO.

No assert fires. The output is finite and wrong.

### CPU reproducer

* `test_f1_window_arithmetic`: a mirror of the prefill and decode tail
  bookkeeping (`N/cache_engine.py:309-327`, `:345-359`, `:401-409`) and of
  `out_len` (`N/nosa_llama.py:817-818`) for several `S`. It asserts that the
  first step with a tail id outside the grid is `65 - S % 64`.
* `test_f1_diff_mirror_frees_the_live_tail`: a line-by-line Python mirror of
  `diff_kernel` (`FCE/diff_offload_kernel.cu:35-98`). After the rename, slot
  63 gets a load. Control: before the rename, no slot gets a load and the map
  does not change.
* `test_f1_diff_mirror_random_selections`: 200 random selections after the
  rename. Slot 63 gets a load every time.

### What only a GPU run can confirm

The mirror is a re-implementation. Only a GPU run shows that the compiled
`diff_kernel` and the two Triton gathers overwrite slot 63, and how much the
logits move. The benchmark default is 4 new tokens
(`benchmarks/Efficiency/test_nosa_pg19.py:32`). I did not check which
recorded runs, if any, decoded past the window.

### Two related limits (already known, both loud or inactive)

* The stage-1 score is `round_multiple(max_seqlen_comp, 128)` wide
  (`IV2/csrc/flash_attn/flash_api.cpp:929`, `:936`; the fast variant returns
  it without a slice, `IV2/infllm_v2/infllmv2_sparse_attention.py:729`).
  `N/nosa_llama.py:607` copies it into the warm-up `score_buf`. When the
  number of compressed keys passes a multiple of 128, the shapes differ and
  `copy_` raises. This is item 3 of the HANDOFF limits. For `S = 16128` that
  is the 18th compression after prefill, after the F1 window.
* The pooling `max_seq_len` is the warm-up value (`N/nosa_llama.py:465`,
  used at `N/max_pooling_fused.py:47`). Compressed keys added after the
  warm-up step are masked. Inside the F1 window they belong to blocks of the
  forced local window, so this has no effect on the selection.

## F2: the compressed-cis table never grows

### Code path

* Prefill: `N/cache_engine.py:722-729`. `M` is the number of 32-wide windows
  at stride 16 (`:725-726`). `comp_cis_len = M` (`:727`).
  `compressed_cis` is `(H, total_bsz, M)` (`:728`).
* Decode: `:752-753` writes the cis into `tail_cis`. When `tail_cis_len`
  reaches 32 (`:754`), `:757` computes `tail_comp` with shape `(H, B, 1)`,
  and `:759` runs
  `compressed_cis[:, :, comp_cis_len:comp_cis_len+1].copy_(tail_comp)`.
* `comp_cis_len` is assigned only at `:727`. Nothing increments it.
  `compressed_cis` has exactly `comp_cis_len` columns, so the slice at `:759`
  is `(H, B, 0)`. `copy_` broadcasts the size-1 source dim to 0 and copies
  nothing. No error. `:763` returns the unchanged tensor.
* The upstream intent is the commented-out `torch.cat` at `:758`.
* The GPU-resident control has the same lines:
  `N/cache_engine_gpu.py:370` and `:402`.

### Condition

`tail_cis_len` starts at `16 + S % 16` (`N/cache_engine.py:731`) and grows by
1 per step. It reaches 32 first at step `i = 16 - S % 16` (at the warm-up step
when `S % 16 = 15`), then every 16 steps.

### Consequence

`compressed_cis` stays `M` wide. Each cis summary after the prompt is computed
and thrown away. In the shipped engine this has NO EFFECT on the selection
inside the F1 window: the pooling reads cis entries below the warm-up
`max_seq_len` only (`N/max_pooling_fused.py:47`; F1 related limits), which is
`M`, and the missing summaries cover positions `>= S - 31`. Those positions
are in blocks of the forced local window, which get `+inf` anyway. F2 is
LATENT in the shipped engine. It becomes LIVE in any engine that grows the
pooling grid (R2): the cis score of a block past the warm-up grid would read
memory that no step wrote.

### CPU reproducer

* `test_f2_empty_slice_copy_is_a_silent_noop`: pure torch, the shipped shapes
  (`H = 2`, `total_bsz = 4`, `M = 1007` for `S = 16128`). The `copy_` at
  `:759` raises nothing, the slice has 0 elements, and the table is unchanged.
* `test_f2_mirror_decode_never_grows`: a mirror of the decode branch
  (`:750-763`) over 64 steps. The summary is computed 4 times; the width
  stays `M`.
* `test_f2_comp_cis_len_is_assigned_once`: an AST scan of
  `N/cache_engine.py`. `self.comp_cis_len` has one assignment (`:727`) and no
  augmented assignment.

### What only a GPU run can confirm

Nothing about the mechanism: the slice and the broadcast rule are the same on
CUDA. Only its downstream effect in a grown-grid engine needs a GPU.

## F3: `total_cis` has `S + 1024` rows; the bias gather assumes `S + 8192`

### Code path

* `N/cache_engine.py:768`: `total_cis` is `(total_bsz, S + 1024, H)`.
  The decode write is `:780` (row `cis_len`), then `:781`
  (`cis_len += 1`); `:782` returns `total_cis`.
* `N/nosa_llama.py:584` passes that return value as `ucis`, and `:618` passes
  `ucis` as `kv_bias` to `decode_update_kv` (`N/cache_engine.py:700-702`).
* The host K/V buffers are `S + max_gen_len = S + 8192` rows
  (`N/cache_engine.py:195-196`, `:132`).
* `N/cache_engine.py:382` calls
  `flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, ...)`.
  In the kernel, `S_CPU = cpu_data.shape[1]` (`FCE/flash_h2d_mask_bias.py:93`),
  that is `S + 8192`. The bias load (`:46-51`, `:58`) uses the mask `mask_src`
  (`:43`), which bounds the row by `S_CPU`, not by `kv_bias.shape[1]`
  (`S + 1024`).

### Condition and consequences, in the shipped order

At step `i = 1025` (`p = S + 1024`):

* (a) `N/cache_engine.py:780`: the slice `total_cis[:, S+1024:S+1025]` is
  `(B, 0, H)`. The decode cis is `(B, 1, H)` (`N/nosa_linear.py:105`, `:128`).
  `copy_` copies nothing. SILENT.
* (b) `N/cache_engine.py:354`: the tail-bias write takes
  `kv_bias[:, seq_length-1:seq_length]`, which is now `(B, 0, H)`. `copy_`
  into a `(B, 1, H)` slice raises a RuntimeError (size 0 cannot broadcast to
  size 1). LOUD. The shipped decode cannot pass step 1024.
* (c) The bound mismatch: if a block `c` with `64 c + 63 >= S + 1024` is
  fetched, the bias load reads rows `r` in `[S + 1024, min(64 c + 64, S + 8192))`.
  With the contiguous `total_cis`, row `r` of request `b` is at the same flat
  offset as row `r - (S + 1024)` of request `b + 1`. For the last request the
  offset is past the allocation. SILENT (wrong bias) or a fault.

In the shipped engine (c) cannot happen. The grid of F1 names blocks up to
`S // 64` only. A block comes from the host only after it is complete, and a
block with rows past `S + 1024` is complete only after step 1024, where (b)
stops the run. (c) is LATENT. It becomes LIVE in an engine that removes (b)
(for example, one that writes the tail bias from the step's own cis) and keeps
`S + 1024` rows.

### CPU reproducer

* `test_f3_total_cis_write_is_a_silent_noop` (a) and
  `test_f3_tail_bias_write_raises` (b): pure torch, the shipped shapes.
* `test_f3_bias_rows_alias_the_next_request` (c): the offset arithmetic of
  `FCE/flash_h2d_mask_bias.py:46-51` with the strides of a contiguous
  `(B, S + 1024, H)` tensor.
* `test_citations_hold` checks that `:382` passes `self._v_cpu` and that `:43`
  bounds by `S_CPU`.

### What only a GPU run can confirm

(c): what the masked Triton load reads past the allocation, and whether it
faults. (a) and (b) are torch semantics and are the same on CUDA.

## F4: on a block-completing step, attention skips the tail block

Not in the design's list. I record it because the CPU twin reproduces it
(the twin's `cache_lens` formula `4032 + (pos + 1) % 64` is this behaviour).

### Code path

`N/cache_engine.py:346` increments `_cache_lens`. When the tail block fills,
`:409` sets `_cache_lens` to `4032 + 0` IN PLACE, and `:412` returns that same
tensor. `N/nosa_llama.py:618` receives it and `:639` passes it to attention as
`cache_seqlens`. The GPU-resident control does the same
(`N/cache_engine_gpu.py:183`, `:233`).

### Condition and consequence

At every step with `p % 64 == 63`, attention reads rows `[0, 4032)`: slots
0..62. Slot 63 holds the tail block, the 64 most recent tokens, the current
token included. Attention does not read it on that step. The next step
selects that block again from the host (it is in the forced local window), so
the effect lasts one step in 64. SILENT and deterministic.

### CPU reproducer and GPU part

`test_f4_cache_lens_drop_on_block_completion` mirrors `:327`, `:346`, `:409`.
Only a GPU run shows whether the step moves the logits. An R2 engine that
"fixes" F4 is not bit-identical to the shipped engine on those steps. Decide
the parity target before you change it.

## What the R2 twin takes from these findings

`N/r2_positional_twin.py` (step 2 of the plan) keeps per-request counters and
preallocated tables, and it refuses (raises) instead of writing into an empty
slice: a full compressed-key table, a full `total_cis`, a rollback longer
than the key ring allows, or a stale snapshot. It grows the compressed-cis
table (the F2 intent). It reproduces F4 on purpose, because the parity target
is the shipped engine. It does not model F1 or F3 (c): those are about the
pooling grid and the gather kernels, which the twin does not contain.
