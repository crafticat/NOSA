# R2 FIATTN: FlashInfer 0.5.3 paged decode against piece 3 v2 (a bounded comparison)

REGISTERED <stamp-by-lead>

Status: SOURCE ONLY, NOT RUN. Branch r2-fiattn, worktree /mnt/beegfs/ojerbi/nosi-r2-fiattn, from integ6 23b8a38.
Design source: /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/attn-reuse-assessment.md (section 5, item 10, section 6).

## 1. Scope

- The human authorized ONE bounded comparison of an EXISTING kernel against piece 3, before any switch.
- Production and the current reference do not change. Nothing is wired into the tick.
- FlashInfer stays at the installed 0.5.3. The bench refuses another version.
- Files:
  - benchmarks/Efficiency/r2_fiattn_bench.py: the standalone bench (modes main and ncu-cell).
  - benchmarks/Efficiency/r2_fiattn_step.sh: the GPU step (pins, uuid file, claim fiattn_active with an owner file,
    DRYRUN, STEP_MIN 45, timeout, record freeze).
  - benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py: the CPU tests FA1-FA13.

## 2. Arms

| arm | what | role |
|---|---|---|
| P | piece 3 v2: bi_paged_decode (7c826c1, blob 39ed5e18), loaded by path from a pinned copy | reference arm |
| F8 | FlashInfer BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend "fa2"), plan(fixed_split_size=8) | candidate |
| F0 | the same, plan(disable_split_kv=True) | candidate (the CUDA-graph-safe form) |
| FH | the same with the default split heuristic | NEGATIVE CONTROL of the invariance gates |
| S5 | sharedkv5 (5eb7514, blob 5c2e7b25), tile "reload", mode "split", 4-row cells only | optional, REPORT |
| STRIDED | FlashInfer option B (one call through an overlapping as_strided view) | REPORT probe only |

Mapping (option A, the safe form): a FlashInfer request is one row. Each layer makes TWO calls, one per KV head.
- K / V view for head h: the store (N, S*64, Hkv, D) viewed as (N*S, 64, Hkv, D), then [:, :, h:h+1, :].
- Page id: entry * S + slot, in piece 3's canonical order (ascending block id).
- q for head h: the view q[:, 16h:16h+16, :] (FlashInfer reads q_stride_n / q_stride_h).
- The two (R, 16, D) outputs are interleaved into (R, 32, D) (one torch.cat, timed with the call).
- One wrapper per KV head; the two wrappers share one float workspace (the head calls run in stream order).

Option B (REPORT probe): request = (row, head), page id = (entry * S + slot) * 64 * Hkv + h, view strides
(D, Hkv*D, D, 1). The runtime acceptance of the overlapping view is an open item of the assessment.

CIS bias in FlashInfer: the JIT custom variant R2FiCisBias (the ALiBi-branch pattern of FlashInfer's default variant).
- logit = (q.k + bias[page * page_mul + entry * entry_stride + head]) * sm_scale; sm_scale_log2 = log2(e).
- page = indices[indptr[request] + kv_idx / 64], entry = kv_idx % 64. The read happens only for kv_idx < kv_len.
- Option A: page_mul = 64 * Hkv, entry_stride = Hkv, head = h. Option B: 1, Hkv, 0.
- A trait guards the paged-only member: FlashInfer also instantiates the ragged kernels with the same variant.
- If the JIT build fails, gate C0 fails (a FINDING) and the run continues BIAS-FREE (bias = 0 in every arm,
  FlashInfer's default module); every other gate is then a REPORT.

## 3. Cases (fixed, seeded: make_case, seed 20261021 x cell)

- Geometry from /mnt/beegfs/ojerbi/models/NOSA-8B/config.json: Hq 32, Hkv 2, D 128 (group 16).
- 16K context. Every row's tail block is 251 (positions 16064 .. 16127). S = 256 slots per storage entry, with a
  seeded block-to-slot map per entry (the slots are not the block ids).
- Per (row, KV head) 64 pages: block 0, the 15 recent blocks 236 .. 250, 47 scattered blocks drawn with weight
  1 / sqrt(251 - b) (locality), and the tail block cut at last_len.
- Rows of one request sit at consecutive positions. Row u > 0 swaps 4 scattered blocks of the request's base list.
- bf16 q / K / V from N(0, 1). The bias is uniform in [-8, 8], so it moves the logits by about +-0.7 (gate C can see
  a missing bias: witness CW).
- Cells: B in {8, 32} x rows per request in {1, 4}. Order in fia1: 8x1 (the inspected miniature), 8x4, 32x1, 32x4;
  from fia2: 32x4, 32x1, 8x4, 8x1 (section 9).
- Extra batches: "alt" (each odd row gets a new q and new lists), "hazard" (the full batch plus 3 short rows of 5, 8
  and 9 pages on an extra storage entry).

## 4. Gates (MUST unless marked; P, F8, F0 each)

| gate | test | precondition |
|---|---|---|
| C0 | the variant JIT build (time charged separately) | - |
| C | max abs(arm - fp64 reference) <= 1.0e-3 on 16 rows per cell; the reference is the bench's own float64 code | - |
| CW | F8 with bias = 0 is FAR (> 1.0e-3) from the biased reference | the bias moves the reference by >= 1.0e-2, else VOID (= fail) |
| I1 | each sampled row alone (its own plan) == the row in the full batch, bitwise | - |
| I2 | reversed request order and a random row order == unpermuted, bitwise per row | - |
| I3 | the even rows of "alt" == the even rows of the full batch | alt changes exactly the odd rows, else VOID |
| I4 | each short row alone == in the hazard batch; every full row in the hazard batch == in the full batch | F8: the 5- and 8-page rows plan split_kv false alone and true batched, on every head, else VOID |
| I5 | unread rows NaN (unlisted (slot, head) pages, tail rows past the largest extent): output finite and bitwise equal | - |
| N | FH: some sampled row alone != batched | the heuristic's chunk size (or split flag) alone differs from batched, read from the plans, else VOID; pass only when DETECTED |
| D | REPORT: bits vs piece 3 (rows differing, max difference); rounding differences never reject | - |
| T | REPORT: times, plan, packing, JIT, memory | - |
| NCU | REPORT: DRAM bytes per arm | - |

"Bitwise" compares int16 views of the bf16 outputs (a sign of zero counts). "Alone" for FlashInfer is a new plan()
with a batch of one row. Each of I1-I5 also runs on S5 as a REPORT.

## 5. Metrics

- Attention time: CUDA events around one call, 20 warm-ups, 200 timed calls, median and IQR.
  - FlashInfer: both head calls plus the output interleave ("call"); also the two runs alone ("runs only").
  - P: the split kernel and the combine kernel.
- plan(): host wall time for one tick (both heads), (a) indptr / last_page_len on the host, (b) on the device (plan's
  own D2H read). 50 repetitions, a synchronize before each.
- Packing: CUDA events around the device packing of both heads (cumsum and scatter, no host read).
- JIT: the first wrapper construction with the variant (build + load), then the second (load only).
- Memory: the workspace bytes (P: wacc + wml for R + 3 rows; FlashInfer: the 256 MiB float buffer, used bytes from
  the plan, the 8 MiB int buffer and the 8 MiB pinned host buffer per wrapper) and the peak allocated delta of one
  call.
- ncu (REPORT): one process per cell in the order 32x4, 32x1, 8x4, 8x1.
  - Every arm planned and warmed outside the profiler window, then one call per arm in an NVTX range fia|<arm>.
  - --kernel-name regex: the FlashInfer prefill and merge kernels and piece 3's / sharedkv5's kernels only.
  - --cache-control all (cold L2 per kernel), --clock-control none, metrics gpu__time_duration.sum,
    dram__bytes_read.sum, dram__bytes_write.sum (+ lts__t_sector_hit_rate.pct when the chip lists it).
  - The profiler window holds one call, so --launch-skip / --launch-count are not needed.
  - Hard cap per process: min(600 s, the budget left - 120 s). A cell without that time is skipped (REPORT).
  - Compared with the per-row ideal bytes (listed token rows x (K + V + bias)) and the request-union bytes.
  - Label: every arm reuses pages through the HARDWARE cache only. S5 adds a forced temporal locality (one program
    reloads each tile per row). No arm has explicit shared loads.

## 6. Registered predictions

Times are per attention call of one layer (CUDA events, median). Bands are [central, low, high]. The json block is
the bench's PRED table (test FA8 keeps them equal).

Reasoning (MODEL, not measured):
- P: piece 3 v2 measured 0.269 ms at R 64 and 0.446 ms at R 128 (BIA, 1 row per sequence). R 8 is launch- and
  occupancy-bound. The 4-row cells share most pages in L2.
- F8: per head call R x 8 CTAs of 16 query rows; two calls plus two merge kernels plus the interleave. At small R
  the extra launches dominate; at R 128 the bytes dominate and the kernels are comparable.
- F0: one CTA per (row, head) walks 64 pages: few CTAs at small R, so it is slow there.
- Accuracy: P and F0 round once at the end. F8 also rounds its split partials to bf16 (FlashInfer stores DTypeO
  partials), so its error is larger.
- Invariance: the source supports it for F8 / F0 (assessment 2.3). The open hazard is the batch-wide split_kv flag
  (I4).
- FH: the heuristic picks 2 pages per chunk for a batch of one row and 3 / 11 / 64 pages for R 8 / 32 / 128 (the
  binary search in scheduler.cuh:101-130 with 216 CTAs), so the precondition should hold in every cell.

<!-- PRED-BEGIN -->
```json
{
 "registered": "REGISTERED <stamp-by-lead>",
 "tol_c_abs": 0.001,
 "err_vs_fp64": {"P": [0.00012, 0.00005, 0.0004], "F0": [0.00015, 0.00005, 0.0005], "F8": [0.0003, 0.0001, 0.001]},
 "p3_ms": {"8x1": [0.06, 0.03, 0.15], "8x4": [0.15, 0.07, 0.3], "32x1": [0.15, 0.07, 0.3], "32x4": [0.4, 0.2, 0.8]},
 "ratio_vs_p3": {
  "F8": {"8x1": [1.3, 0.7, 2.5], "8x4": [1.1, 0.6, 2.0], "32x1": [1.1, 0.6, 2.0], "32x4": [0.95, 0.6, 1.6]},
  "F0": {"8x1": [3.0, 1.5, 8.0], "8x4": [2.5, 1.2, 6.0], "32x1": [2.5, 1.2, 6.0], "32x4": [1.8, 1.0, 4.0]},
  "S5": {"8x4": [0.9, 0.6, 1.3], "32x4": [0.85, 0.5, 1.3]}
 },
 "plan_ms_per_tick_host_indptr": [0.2, 0.05, 1.0],
 "plan_ms_per_tick_device_indptr": [0.3, 0.08, 1.5],
 "pack_ms_per_layer": [0.05, 0.01, 0.3],
 "jit_build_s": [240, 60, 900],
 "invariance": {"P": "PASS", "F8": "PASS (p 0.7; the split_kv switch I4 is the main risk, p 0.25)",
                "F0": "PASS (p 0.85)", "FH": "DETECTED with the precondition met (p 0.9)"},
 "bits_vs_p3": {"F8": "most rows differ (> 0.9 of rows), max |d| in [1e-4, 1e-3]",
                "F0": "most rows differ (> 0.5 of rows), max |d| <= 5e-4"},
 "dram_read_over_ideal": {
  "1": {"P": [1.0, 0.9, 1.2], "F8": [1.0, 0.9, 1.25], "F0": [1.0, 0.9, 1.25]},
  "4": {"P": [0.45, 0.28, 0.9], "F8": [0.45, 0.28, 0.9], "F0": [0.45, 0.28, 0.9], "S5": [0.33, 0.28, 0.7]}
 }
}
```
<!-- PRED-END -->

What would decide adoption (assessment 5.6, unchanged): F8 (or F0) passes C and I1-I5 with N detected, AND its
attention time is <= P's in the cells that matter (32x4), AND the per-tick plan + packing stays small against the
tick (143.68 ms captured tick at B 8, registry 172). A win only in eager mode (see 8.3) is a constraint, not a pass.
A pass makes FlashInfer a candidate NEW reference after review; bitwise equality with P is not required.

## 7. The step

    bash benchmarks/Efficiency/r2_fiattn_step.sh --nosi <full sha of r2-fiattn HEAD> <holder job> <run id>
    DRYRUN=1 bash benchmarks/Efficiency/r2_fiattn_step.sh --nosi <sha> <holder job> <run id>

- srun into the existing holder: 1 task, 16 CPUs, 64G, gpu:a100:1, --time 45; host timeout 45 min + 60 s.
- Bench budget 2400 s for the whole process (JIT, 4 cells, ncu). A cell starts only with >= 300 s left, else gate
  BUDGET fails.
- Caches under /mnt/beegfs/ojerbi/nosa-env/jit/fiattn_<run>_<holder>/ (FLASHINFER_WORKSPACE_BASE, TVM_FFI_CACHE_DIR,
  TRITON_CACHE_DIR, TORCH_EXTENSIONS_DIR, TMPDIR); FLASHINFER_CUDA_ARCH_LIST 8.0, MAX_JOBS 16. The bench refuses an
  unset FLASHINFER_WORKSPACE_BASE (lessons C1).
- Output /mnt/beegfs/ojerbi/r2-runs/fiattn/fiattn_<run>_<holder>/ and the record .step/, both frozen.
- Claim /mnt/beegfs/ojerbi/r2-runs/fiattn/fiattn_active with an owner file. It stays after a client timeout (the writer
  may still run; lessons C10 / L40).

Preflight (the standing rule): the CPU tests FA1-FA13 at this commit, on a compute node, with git HEAD printed before
and after (lessons L37); the 8x1 cell runs first inside the step and is the inspected miniature.

## 8. Known hazards and doubts (written before the run)

1. plan() rewrites its pinned host buffer and then queues an H2D copy. A second plan() before that copy runs would
   overwrite the data in flight. The bench synchronizes before every plan(). A tick would need the same order (it
   already has the D2H read of indptr inside plan()).
2. plan() reads indptr / last_page_len on the HOST, so every tick has one D2H synchronization per plan, and plan()
   cannot be captured in a CUDA graph. One plan per KV head per tick if every layer has equal page counts; per layer
   otherwise (NOT CHECKED for NOSA).
3. CUDA graph: with fixed_split_size the planner can refuse a batch larger than the padded size (assessment 2.2).
   This bench runs eager only.
4. F8 stores split partials in bf16 (DTypeO); P keeps fp32 partials. Gate C has a common tolerance; D reports the
   bit differences.
5. The variant reads the bias with one 2-byte load per score, inside the inner loop (uncoalesced). If F8 is slow, a
   logical bias buffer is the fallback (not in this bench).
6. ncu may write its own configuration under HOME (not checked); nsight-compute is not a JIT cache, so the bench does
   not refuse it.
7. q per head is a strided view. The kernel reads q_stride_n / q_stride_h (prefill.cuh:448); gate C would show a
   wrong read.

## 9. Fix round after fia1 (job 2191262, commit 693da86; the predictions in section 6 are UNCHANGED)

What the fia1 record shows (/mnt/beegfs/ojerbi/r2-runs/fiattn/fiattn_fia1_2191262.step/stdout.txt):
- All four cells ran. The record holds 81 GATE lines, all PASS, including every MUST gate of 32x1 and 32x4.
  "0 MUST gate(s) failed" and exit 0 were therefore correct for that run.
- The traceback in stderr.txt came from the REPORT-only STRIDED probe at 32x4 (bench line 706,
  FIStridedArm.prepare). Option B makes 256 requests x 8 chunks. Its split partials need 256 MiB (tmp_v) + 2 MiB
  (tmp_s), above the fixed 256 MiB float workspace. The F8 / FH arms need half of that per head, so they fit.
- pack/layer 0.31 ms was the same at R 8 and R 128: launch-bound. About 12 small torch ops per head, two heads.

Changes (source only, not run):
1. The float workspace is sized per plan from FlashInfer's own formula (fi_float_need: G x padded x cta_tile_q x D
   x 4 B + G x padded x cta_tile_q x 4 B, padded = sum of chunks; heuristic bound max(requests, 2 x SMs)). It starts
   at 1 MiB and grows. T reports the need, the allocation and what the plan reserves (and flags a need below use).
2. Gate COMPLETE (MUST) per requested cell: every expected MUST gate of the cell must be present. A cell or an arm
   that never reports now fails the run instead of passing by absence. (Before: a cell exception or a budget skip
   already failed a MUST gate; a partial arm only failed through its exception gate.)
3. Packing: pack_all_heads makes both heads in one pass (the ragged path). With constant counts (every row and head
   lists 64 pages: NOSA's fixed top-k) pack_fixed is one elementwise op that writes in place into the indices buffer
   the plan already holds. Plan depends only on indptr / last_page_len / counts, and the kernel reads the indices
   at run time, so ONE plan per tick (per KV head) can serve all 32 layers when the counts are equal. REPORT
   R_REUSE checks it bitwise (plan on one layer, write another layer's lists in place, compare with a fresh plan);
   T times "pack/layer (constant counts)" and "layer with the tick's plan reused".
4. The step runs the cells in the order 32x4, 32x1, 8x4, 8x1.
