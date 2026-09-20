# Paired tick (stage E2, resident): design notes, engine assumptions, what E4 needs

Spec: retroinfer-eval `docs/superpowers/specs/2026-09-20-paired-tick.md` (sections 1-3), audit
`2026-09-20-schedule-audit.md` (D, B.6, E), reuse ledger `2026-09-20-reuse-ledger.md` section 3
(decision (ii-a)), cost model `docs/evidence/cost_model/DERIVATION.md`. Branch `feature/nosi-paired-tick`
from `feature/nosi-verify-rows` (004f279). Line numbers below are of THIS worktree.

## 1. What runs, in one sentence per file

- `core.py` (pure torch): row layout `[V | S]` adjacent (row 2r = V, 2r+1 = S), the per-row visible-row
  rules, the rows bias with the EXACT extent, the provisional write / NaN poison, the S == 1 tail-write
  twin, the draft state machine (accept / reject / restart), the ledger (misses, divergence, bytes).
- `tick.py` (GPU, lazy kernel binding): `layer_body` = one qkv GEMM over all rows, per-position scoring
  through the shipped kernels + the captured pooling graph, the V row's engine update = the shipped
  decode body CALLED (`cache.decode_update_kv`), the S row's provisional write, ONE rows-attention call
  (the shipped decode kernel via `cache_batch_idx` + per-row bias), wo / FFN over all rows. Drivers:
  `paired_tick`, `s_rows_forward` (restart / seed), `restart_after`, `setup`.
- `twin.py`: `twin_step` (V rows only through the same body) and `InSituTwin` (every term recomputed
  at M = B from the same inputs; `diagnose` names the first differing term).
- `benchmarks/Efficiency/paired_pilot.py`: arms `shipped`, `twin`, `equiv`, `resident`, `compare`.
- `tests/test_paired_core.py`: 36 CPU tests (see section 5).

## 2. Assumptions about the engine, each with the line that carries it

| # | assumption | where it is used | evidence (file:line, this worktree) |
|---|---|---|---|
| A1 | The V row's engine update IS the shipped S == 1 body: `seq_length += 1`, `_cache_lens += 1`, the tail write at `_tail_block_idx_on_gpu*bs + _tail_block_len_on_gpu`, `diff_offload`, `_block_map.copy_`, the two blocking Triton gathers, and on a fill the host write-back + `_block_map[..., topk-1] += 1` + `tail_len = 0` + `_cache_lens = (topk-1)*bs`. | `tick.layer_body` calls `cache.decode_update_kv(k[:, 0], v[:, 0], ucis, l, sel_v)`; nothing in the body is copied | `cache_engine.py:640-716` (`decode_update_has_kv_bias`), bound under `NOSI_VERIFY_ROUND_SLOTS > 0` through `decode_update_has_kv_bias_verify_slots` `:434-446` (the same body, then the 64-slot views) |
| A2 | Its return values (the 64-slot views) can be ignored: the rows call reads the whole allocation `_k_gpu/_v_gpu/_kv_bias_gpu` and the split partition then comes from `W*bs`, the same for the tick and the twin (explicit `num_splits`). | `tick.attend_rows(eng._k_gpu, eng._v_gpu, ...)` | `cache_engine.py:349-398` (section 3c: the partition is a function of the allocated length); `verify/rows_attention.py:25-30` |
| A3 | `_load_mask[h, b, m] >= 0` iff slot m was fetched from the host this step (residual miss). | `v_miss = (eng._load_mask >= 0).sum(-1)` | `flash_cache_engine/flash_h2d_mask.py:30-35` (`if cpu_block_id < 0: return`); `cache_engine.py:666` (diff writes it) |
| A4 | After the update, the shipped decode attends rows `< (topk-1)*bs + tail_len_after`: all window slots in full, the tail slot up to `tail_len_after`, and 0 tail rows at a fill step (the just-completed block is on the host until the next diff fetches it). | `core.v_visible_rows`; the exact extent in `core.paired_bias` | `cache_engine.py:649` (`_cache_lens += 1`), `:711-713` (`= (topk-1)*bs + 0` after a fill); `nosa_llama.py:640-647` (`cache_seqlens=_cache_lens`) |
| A5 | A slot outside the decode's window exists only under `NOSI_VERIFY_ROUND_SLOTS = R > 0`: `W = topk + R + 2`; slots `>= topk` are zero-filled at prefill; the decode never reads them. The paired tick uses slot `topk` (= round slot 0) as the PROVISIONAL slot, slots `topk+1 .. W-3` as E4's ring, and leaves the mirrors `W-2, W-1` (Path 1's) unused. | `core.paired_layout`, `core.check_layout_against_engine` | `cache_engine.py:274-297` (`_gpu_slots`), `:332-347` (3b': zero-fill), `verify/union_store.py:63-71` |
| A6 | Every row below a row's `cache_seqlens` is LOADED by the rows kernel (masked columns contribute exact zeros only if their K/V are finite). Slot 63's rows above the prefill's tail length are `torch.empty` garbage until written; the shipped decode's extent never covers them, the S row's extent (`topk*bs + 1`) does. | `tick.setup` zero-fills `slot 63 rows [tail_len, 64)` once after the warm-up step (never read by the decode) | `verify/rows_attention.py:76-83` (PRECONDITION); `cache_engine.py:295-297` (`torch.empty`), `:402-410` (prefill writes only `tail_len` rows) |
| A7 | The scoring of one position = the decode's chain: `update_no_compress_k_decode`, `update_compress_k_decode`, `update_uncompressed_cis`, `update_cis`, `infllmv2_attn_stage1_fast`, `score_buf.copy_`, `compressed_cis_buf.copy_`, `after_pooling_graph.replay()`, `topk_idx_buf`; the graph and its buffers exist after the warm-up step. The S position's four table updates are bracketed by `spec_loop.LayerJournal.take/restore` (per layer, per call). | `tick.score_position` | `nosa_llama.py:579-619` (decode), `:714-739` (verify, per position); `spec_loop.py:692-760` (journal) |
| A8 | `compressed_cis_buf` is NOT a scratch buffer: the warm-up binds it to `update_cis`'s return value, which is the layer's PERSISTENT compressed-cis table; the decode's `compressed_cis_buf.copy_(compressed_cis)` is a self-copy. A subset (restart) is therefore scored over the WHOLE batch with zero-padded q / key / cis rows (the decode's exact lines) and its selection read off `topk_idx_buf[:, req_idx]`; a slice write into `[:, :n]` corrupts requests 0..n-1 (job 2175550, section 7). `setup` asserts the alias. | `tick.score_position` | `nosa_llama.py:451` (`self.compressed_cis_buf = compressed_cis`), `cache_engine.py:1081` (`return self.compressed_cis`); `nosa_llama.py:613-614` |
| A9 | The stage-1 score is `(Hkv, total_q, max_seqlen_k)` and `score_buf` is fixed-shape (aliased into the captured graph): a 16-token COMPRESS EVENT changes `max_seqlen_comp` and would fail at `score_buf.copy_`. | `tick.score_position` refuses on a compress event; the pilot sizes N from `no_compress_k_len` after the warm-up step (`_budget_or_die`) | `dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py:677`; `nosa_llama.py:465-497`, `:613-618`; `cache_engine.py:984-995` (`update_no_compress_k_decode` shifts at `kernel_size`) |
| A10 | `KV_BIAS_SCALE` must be applied to the provisional bias row exactly as the engine's write sites apply it. | `G.bias_rows = cache_engine._bias_rows` | `cache_engine.py:97-100`, the four write sites `:657`, `:762`, `:863`, `:411` |
| A11 | The FAN host check wants `bias.size(0) == kcache.size(0)`; the kernel addresses query rows through `bias.stride(0)`. A storage of `>= max(R, B)` rows with the narrowed view `bias[:B]` satisfies the check for R = 2B (tick), R = B (twin) and R = n < B (restart). | `tick.attend_rows` | `dependencies/flash-attention-nosa/csrc/flash_attn_nosa/flash_api.cpp:366`; `verify/rows_attention.py:31-43` |
| A12 | `apply_rope_with_cos_sin_cache_inplace` takes flat per-row positions, so `[tau, tau+1]` per request is one call over 2B rows. | `tick.layer_body` | `nosa_llama.py:701-704` (verify at U positions) |
| A13 | Between a fill and the next V write, slot 63 physically holds the completed block T while `_block_map` says T+1. An S row (tick or restart) may read those 64 exact rows by id T. Right after a block-aligned PREFILL (`tail_len == 0`, no fill) the slot holds nothing. | `core.tail_view` (`slot_complete` carried per layer in `Scratch.slot_complete`) | `cache_engine.py:705-713` (rename without clearing the rows); `:402-410` (prefill) |

## 3. Gates and how each is exercised

| gate | CPU test (tests/test_paired_core.py) | GPU (paired_pilot.py) |
|---|---|---|
| G1 provisional KV isolated | `test_g5_...` (the exact-store digest is unchanged by the write and the poison), `test_g7_...` (the poisoned row surfaces only when an extent reaches it) | `equiv`: `NOSI_PAIRED_POISON=1`, every tick's logits must be finite (`finite[B,arm]`) |
| G2 canonical order | `test_g2_...` (deterministic, ascending block id, neighbour-independent) | the twin and the tick build the V rows from the same rule (`bias_equal` in the in-situ record) |
| G3 tick V == twin | - (needs the kernels) | `G3[B,arm vs twin]`: torch.equal per row over ticks WARM..N-1; on failure the equiv arm's in-situ diagnosis names the term |
| G2-tol twin vs shipped | - | `G2-tol[B]`: max dlogit <= 4 x tau_ctrl (`NOSI_VERIFY_TAU_CTRL` required) |
| G4 not-ready excluded | `test_g4_...` (a selected ring slot with `ready=False` is excluded; positive control) | E2: the ring is never ready (no prefetch) |
| G5 reject leaves the store | `test_g5_...` | `equiv`/`resident`: the tick's V rows keep equalling the twin's (G3) after rejects |
| G6 tail rollover | `test_g6_...` at tail_len 61/62/63 -> 62/63/0 through `tail_write._write_one_token` / `_write_back_full_tail` (the S == 1 twin), plus what V and S see at the fill step | NOT exercisable at 16K within the fixed-shape budget: the fill and the S row's compress event land on the same step for every L (section 4) |
| G7 finite mask, poison control | `test_paired_bias_...` (finite -3e4), `test_reference_twin_...` (masked columns contribute exact 0), `test_g7_...` | `finite[B,arm]` |
| G8 lockstep commit | `test_state_machine_sequence`, `lockstep_check` | `G8[B,arm]`: ticks x B committed tokens |
| G9 staging fits | - (E4) | peak GB per arm reported |

## 4. What blocks a faithful [V | S] tick in the present engine

1. **The captured pooling graph is fixed-shape** (A9). A compress event every 16 tokens changes the
   stage-1 score width; the graph's `score_buf` cannot take it. The pilot's window is therefore
   `no_compress_k_len + (N - WARM) < 32`; at L = 16128 that is N <= 15 with the S row one append ahead.
   Production needs either a re-warm at every compress event (a graph recapture, ~the spec loop pilot's
   `needs_rewarm`) or a scoring kernel that takes the current width.
2. **The fill and the compress coincide.** `no_compress_k_len` after prefill is `16 + (L mod 16)` and the
   tail length is `L mod 64`, so the S row's compress fires at step `15 - (L mod 16)` and the fill at
   `63 - (L mod 64)`; with `L mod 64 = 48 + (L mod 16)` these are the same step. G6 is CPU-proved only.
3. **`flash_api.cpp:366`** (A11): the narrowed view is the microbench way in; the shipped fix is the
   one-line relax + an extension rebuild on a compute node.
4. **The 16-token compress in a SUBSET restart** would run the batch-wide `torch.cat` with zero rows for
   the non-restarting requests; refused (A9) rather than journaled.
5. **cuBLAS at M = 2B vs M = B** may pick another kernel and change every row's accumulation order
   (verify_pilot's B2 caveat). G3 can fail for that reason alone; `InSituTwin` isolates it from the
   attention and the bias terms so the verdict says which. Nothing in the engine can be changed to make
   a 2B-row GEMM bit-equal to a B-row GEMM. The OPTIONAL `twin2b` arm (`twin.twin2b_step`,
   `NOSI_PAIRED_MODE=twin2b`) is the U = 1 twin with every GEMM padded to M = 2B by B zero rows
   (`tick.linear_padded`): when G3 against the U = 1 twin fails at a GEMM term, compare gates the tick
   against twin2b (`G3-2b`), which must be torch.equal if the kernel choice was the only difference.
   The spec's twin stays U = 1 at M = B; the author's greedy gate (`G3-greedy` / `G3-tol` /
   `G3-commit` against the SHIPPED rows) is reported beside it, separately.

## 5. Stage E4 (per-layer prefetch into a staging ring): what it needs from the engine

- A side stream + an event per (layer, issue): `spec_loop.PrefetchQueue` and `cache_engine.spec_draft_update`
  steps C already do this for the K-draft round (`cache_engine.py:573-581`); the tick would issue, at
  layer l, the gathers of `sel_S(l) \ resident \ in-flight` into ring slots `topk+1 .. W-3` chosen by
  `spec_loop._assign` (FREE first, then LRU), ordered after a main-stream event recorded AFTER the rows
  attention of layer l (the last reader of an evicted slot).
- Readiness = `event.query()` observed on the host before the NEXT tick's layer l bias is built
  (`PrefetchQueue.done_gen` -> `Store.mark_arrived`): `core.ready_mask(ring_ready=...)` takes exactly
  that per-slot flag; not-ready slots are excluded from the S mask by construction (G4 test).
- The V row's diff must see the ring: today `diff_offload` maps `topk_idx` onto the 64 window slots only
  (`cache_engine.py:666`); a block resident in the ring is still fetched into the window from the host.
  E4 needs either (i) a ring -> window device copy for V's hits (bytes saved, an extra copy) or (ii) the
  V row's mask to name ring slots directly (the rows kernel can: any slot below the extent) and the
  window map to be the union (a diff that knows the ring: `spec_loop.plan_round` is that plan).
- Bias rows for prefetched blocks: `flash_h2d_from_mask_bias` reads them off `total_cis` (GPU resident),
  so no extra wire bytes; the ring's bias rows are written by the same gather.
- A per-layer `slot_complete` / ring occupancy map must survive across ticks (`Scratch` already keeps
  per-layer state); the poison must extend to ring slots evicted before use (they are read by nobody).
- The restart forward (n_rej rows) should not evict ring slots the next tick's V will want.

## 6. Open questions for the GPU run (registered, not assumed)

- G3 at B = 64 and 128: does cuBLAS keep the per-row accumulation order between M = B and M = 2B for
  wqkv / wo / gate_up / down / lm_head? (Prediction: the attention term is torch.equal; the GEMM term is
  the only one that can differ.)
- The twin vs the shipped decode: max dlogit within 4 x 0.4844 = 1.94 at 16K (the same class as the
  Path-1 rows: different split seams over W = 128 vs 64 slots).
- Acceptance a on the fused schedule (teacher-forced and production-style) vs the K = 1 pilot's 0.92-0.98.
- The divergence statistic per layer: prefetch precision / recall of S's selection for V's next position
  (the replay assumed 1).
- The sequential restart's cost at n_rej rows vs `rest(B = 16) = 20.2 ms`; how often n_rej > 0 at B = 128.
- The resident tick's brackets vs the twin's and the shipped step's: the added scoring pass and the
  2B-row attention (registered band for the attention: [13, 51] ms at B = 128 in the reuse ledger).

## 7. Found by the GPU (E2, job 2175550) and fixed

1. **V rows wrong from the tick after the first restart** (resident arm: argmax vs shipped 686/704, max
   dlogit 7.5). The data: tick 4 torch.equal to twin2b on all 64 requests; tick 5 differs on exactly requests
   0..27 with n_restart(4) = 28; tick 6 on 0..32 with n_restart(5) = 33. Cause: the compacted restart scoring
   wrote the gathered compressed-cis rows of the restarted requests into `compressed_cis_buf[:, :n]`, which
   aliases the layer's persistent table (A8), un-journaled. Fix: no compaction of the scoring (full-batch
   stage-1 with zero-padded rows, the decode's own two copy lines), `setup` asserts the alias, a CPU
   harness with the live alias as positive control (`test_subset_scoring_leaves_the_tables_equal...`).
   The same corruption explains the fetch residual (+69 ms/tick at B = 64 = 1.9 GB at 25 GB/s: corrupted
   selections churned the window at 14 blocks per stream-step vs 3.8).
2. **equiv arms crashed** at `twin.py:103`: `position_ids[:, 0].flatten()` is a strided view; the flashinfer
   rope checks contiguity (`rope.cu:122`). Fix: `tick.rope_positions` at every rope call; the in-situ
   comparator records exceptions per layer instead of raising.
3. **Host syncs in every layer's fetch bracket**: `core.paired_bias` read `cache_batch_idx.min/max` and two
   `any()` flags on the device (4 syncs x 32 layers per call, twin and tick alike). Now `check_values` is
   explicit (CPU tests True, the GPU body False: the values come from its own bounded rules).
4. **Open (not fixed, to be profiled)**: the restart's `rest` bracket is 175 ms at n_rej >= 27 rows and 56 ms
   at n_rej = 26 (a threshold, not a slope); candidates: cuBLAS kernel selection at small M, the allocator
   for varying n. The timeline job (audit C.6) should split it.
5. **Diagnostics added**: `InSituTwin` terms bias / qkv (pre-rope) / rope / score / selection / attention /
   wo-FFN / norm / lm_head, `diagnose_layers` per layer of the first differing tick (printed by compare);
   `NOSI_PAIRED_S_OFF=1` (S rows in the GEMMs and the attention call only, no seed / restart).
