# NOTES: Task 1.4 route 1, the verify rows in the same forward pass (`verify_rows.py`, cell vr1)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`, Task 1.4, route 1.
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-verify1`, branch `r2-verify1`, from `0bdc045` (Task 1.3 tip, `--ragged`).
- Status: SOURCE ONLY. I did not run a test or a job. The lead runs the commands below.
- Style: ASD-STE100. Labels: CODE = read at file:line; INFERRED = my reading; MODEL = hand arithmetic.

## 1. What I made

| file | what |
|---|---|
| `nosi/nosi/verify_rows.py` (new) | the joint pass: `VerifyOps`, `VerifyState`, `attach`, `joint_pass`, `read_stats`, `phase_ms`; modes EXACT and FAST |
| `benchmarks/Efficiency/r2_verify_cell.py` (new) | the GPU cell: SH (mb1), then the arms VRX / VRC / VRF; gates, REGISTERED block `PRED_VERIFY` (TBD) |
| `benchmarks/Efficiency/r2_verify_step.sh` (new) | a copy of `r2_base_ragged_step.sh`: cell `vr1`, ROOT `/mnt/beegfs/ojerbi/nosi-r2-verify1`, output `/mnt/beegfs/ojerbi/r2-runs/verify1/vr_*`, claim `vr_active` |
| `benchmarks/Efficiency/tests_cpu/verify_world.py` (new) | the CPU world of the tests (not a test file), on top of `ragged_world.py` |
| `benchmarks/Efficiency/tests_cpu/test_verify_rows.py` (new) | the CPU tests |
| `benchmarks/Efficiency/tests_cpu/test_ragged_engine.py` | ONE change: the gate-0 `git diff` of Task 1.3 also excludes `nosi/nosi/verify_rows.py` (a new file in `nosi/nosi` would fail it otherwise) |

NO NEW KERNEL. NO KERNEL EDIT. `ragged_engine.py`, `cache_engine.py`, `nosa_llama.py`, `r2_base_cell.py`, the ragged
step script and `dependencies/` are the `0bdc045` bytes. The joint pass uses these kernels as they are:
`stage1_seqused` (the installed `infllm_v2.C` op), `nosa_pooling_per_request`, `torch.topk`, `diff_offload`,
`pool_update`, `flash_pool_swap`, `flash_h2d_from_mask(_bias)`, `r2_writeback.masked_writeback`,
`flash_attn_nosa_with_kvcache`. Everything else is torch.

## 2. One joint pass

Each request runs R = K + 1 rows at the positions p .. p + K (K 3: rows 0-2 verify, row 3 is the draft row; in this
route every row attends its full selection). Entry e = b x R + r. Per layer:

1. rmsnorm, wqkv, nosa_linear, rope over all E = B x R rows: ONE GEMM call per weight (gemm `natural` or `exact`).
2. The twin advances R times (`step_unchecked`, B rows per call: the serial shapes). Row r records its own n_used
   (n_comp after its step) and M_b (pos_before // 64 + 1).
3. ONE batched selection: ONE `stage1_seqused` call with `cu_seqlens_q = arange(E + 1)`,
   `cu_seqlens_k[e] = (e // R) x Lk_max`, `seqused_k[e]` = row e's n_used; the compressed cis copied to the R entries of
   its request (one copy); ONE `nosa_pooling_per_request` call over E entries; ONE top-k pair. No loop over positions.
   Why this is exact per row (CODE): stage 1 never splits and its grid is per entry (`stage1_seqused.py:3-7`); the
   pooling grid is (H, E, M) (`max_pooling_per_request.py:159`); the twin tables only append, so a row reads below its
   own n_used the values the serial step read (`r2_positional_twin.py:85-93`).
4. Residency and attention: by mode (section 3).
5. wo, rmsnorm, gate_up, SiLU, down: ONE GEMM call per weight. Final rmsnorm and lm_head: one call.

Causal inside the group: row r reads the tail rows of the positions p .. p + r and no later row. EXACT: row r's tail
row is written by its own `decode_fixed`, after row r - 1's attention and before row r's. FAST: all R tail rows are
written first; the cut is `cache_seqlens` in the tail slot and the bias mask in the next-block slot.

## 3. The route chosen, and why (the task's question)

I chose the FLATTENED SINGLE-ROW `flash_attn_nosa` ENTRIES that share one window through `cache_batch_idx`. Both
candidate routes need no kernel change. The reasons for this one:

- It is the shipped decode kernel. It computes the target math, (q.k + bias) x scale, with the bias inside the kernel
  (`flash_fwd_kernel.h:950-971`, `flash_api.cpp:117-120`). No pre-scaled bias is needed.
- NOSA's prefill varlen kernel has no bias input (`infllmv2_sparse_attention.py`, SURVEY section 2.1). The route
  through it needs V x exp(bias x scale), a second call for the denominator and a quotient (EQ `verify_step.py`; the
  pilot's copies + call 2 + quotient cost 13.9-74.9 ms, P0_PILOT_AUTOPSY.md E7), plus a contiguous per-request union
  store (a K / V copy). Its bits also differ from the decode kernel's.
- Neither route shares K / V loads between the rows of a request: both tile one token x 16 heads
  (SURVEY section 2.1, 2.2). Load sharing is route 2 (FlashInfer, after Task 0.3).

THE BIAS ROWS PER ENTRY, without a kernel change (CODE). `flash_api.cpp:340` sets `batch_size_c = kcache.size(0)` and
`:366` checks `bcache` against `(batch_size_c, seqlen_k, num_heads_k)`. But the kernel reads the bias with the QUERY
entry `bidb` (`flash_fwd_kernel.h:968-970`), and K / V with `cache_batch_idx[bidb]` (`:649-659`). With B cache rows and
E = B x R entries, a (B, L, H) bias cannot hold E rows. So `verify_rows` builds a contiguous (E, L, H) bias and passes
its first-B-rows view (shape (B, L, H), batch stride L x H): the shape check passes and the kernel reads row e for
entry e. This relies on the kernel's indexing. The GPU gate VRF-IDX checks it (section 6). The CPU stand-in models
both facts (`verify_world.fake_attention_idx`, test `test_fake_attention_idx_matches_fake`).
`cache_batch_idx` forces the split-KV kernel (`flash_api.cpp:536`), the same kernel as the shipped decode at
`NOSI_ATTN_SPLITS=4`.

### Why EXACT cannot use the shared union window, with or without splits

FA2 sums the online softmax tile by tile in slot order; for head dim 128 a key tile is 128 columns = 2 slots
(`flash_api.cpp:223`). A union window puts row r's blocks in other slots and pairs than the serial window `map_r`, and
masked columns sit between them, so the per-tile max, rescale and P x V sums run in another order. The bits differ even
with one split. A union layout equals the serial layout only when every row has the same window. So:

- EXACT (the proof form): the serial residency per row. For r = 0 .. K: `decode_fixed` (tail row p + r, diff + victim
  pool + gathers, write-back, tail id, attention length), then the shipped attention of row r (B entries) on the
  attended window. The window then holds exactly row r's selection, in the serial slot order. The engine ends in the
  serial state. The GEMMs: `rowgroup` (R calls of the shipped shape: bitwise the shipped GEMM; R weight reads) or
  `exact` (one call of a batch-invariant linear: one weight read).
- FAST (the timing form, route 1): ONE residency for row 0's selection (the shipped diff, victim pool and gathers),
  then the EXTRAS of rows 1 .. K (the union's blocks that the window does not hold): a pool hit is copied device to
  device into a small extras store (B, X x 64, H, D) with torch indexing; a miss is fetched from the host with the
  shipped gathers into the same store. Up to X - 1 extras per (head, request); more are COUNTED (overflow), never
  silent. The last extras slot holds the next block's rows when the group crosses a block edge. Attention: call A on
  the window (per-entry bias: the slots of row r's selection; the tail slot through `cache_seqlens`, findings F4 kept),
  call B on the extras store, merged by log-sum-exp (`return_softmax_lse`). End: write-back of a filled block, the
  next block's rows into the tail slot, tail id, attention length.

MASK_BIAS = -3e4, finite, as `avail_policy.py:113` (a split can be all masked; -inf would give NaN there).

## 4. Tests for the lead

Run on a CPU node, inside a job:

    cd /mnt/beegfs/ojerbi/nosi-r2-verify1
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_verify_rows.py

Regression of the touched test file and the ragged world (Task 1.3):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_ragged_engine.py

| test | claim | negative control (precondition, L34) |
|---|---|---|
| `test_flag_off_is_shipped` | `nosi/nosi` (verify_rows.py excepted), `dependencies/`, `r2_base_cell.py`, the ragged step script = `0bdc045`; no flag-off file names `verify_rows`; a VerifyState attached (FAST and EXACT) changes no engine / twin table, and the ragged step stays bitwise the shipped step (B 1, B 8) | `touch` (the attach writes one tail row; precondition: the row is written and finite) |
| `test_verify_equals_serial_shipped[B1, B8]` | EXACT, gemm rowgroup, attach after the shipped prefill (pool P 8): every row of 3 joint passes BITWISE the shipped decode step at its position; the same loads and pool decisions; then every engine table and the twin. Preconditions asserted: the two caches equal before step 0; a block edge (F4 fill) at row 1 of group 0 | `shared_row_lengths` (every row uses row 0's n_used / M_b; precondition: row 2's M_b is row 0's + 1) |
| `test_verify_equals_serial_ragged[B1, B8]` | EXACT, gemm exact (one GEMM call), ragged lengths (S8 of Task 1.3): every row BITWISE the ragged serial step of its request; then every table and twin. Preconditions: in group 0 some requests fill a block (at rows 0 and 2) and others do not (Review Focus) | - |
| `test_every_verify_row_lists_block0` | EXACT and FAST, prompts of 75-77 blocks (SF8): every row selection lists block 0 | `no_init_force` (the pooling forces no init block; precondition: then some row drops block 0 on these data) |
| `test_one_weight_pass` | one embedding call (E ids) and 4 L + 1 GEMM calls per pass, each over E = B x R rows (FAST natural, EXACT exact) | gemm `rowgroup`: R x (4 L + 1) calls of B rows (precondition R > 1) |
| `test_union_holds_every_row_selection` | FAST: every block of every row's selection is present for that row (tail slot, next-block slot, window slot or extras slot), and a window / extras slot holds the host rows and the cis rows of its block; overflow 0. EXACT: when row r attends, the window holds row r's selection. With one extras slot (X 2): the missing blocks equal the COUNTED overflow | a SILENT drop (an extras id cleared after `_extras`; precondition: extras > 0): missing != overflow |
| `test_no_per_position_selection_loop` | static: no stage-1 / pooling / top-k / `_select_rows` call inside a loop of its function in `verify_rows.py`; dynamic: one stage-1 call and one pooling call per layer, each over E rows | `verify_world.looped_select_rows` (the pilot's per-position loop): flagged by the scan; R calls per layer of B rows (precondition: the same bits) |
| `test_fast_rows_match_exact[B1, B8]` | FAST rows within 1e-5 (relative) of the EXACT rows, the same argmax; layer 0's selections bitwise; at most 2 later-layer near-tie moves (counted, skipped); the VRF-IDX path passes on the stand-ins | `no_extras`, `no_row_mask` (precondition: extras > 0), `causal_leak` (precondition: a request has R rows in its tail block and does not fill it) |
| `test_fake_attention_idx_matches_fake` | the extended stand-in = `ragged_world.fake_attention` per entry, bitwise; its log-sum-exp = a float64 recomputation; the bcache shape check | the bias row of the cache index gives other bits |
| `test_refusals`, `test_verify_cell_arguments_and_verdicts`, `test_step_script_vr1` | argument refusals, the pure cell parts, the TBD verdicts, the gates, DRYRUN of vr1 | - |

HOW THE CPU WORLD IS BUILT: `ragged_world.py` (Task 1.3 NOTES section 4) plus `verify_rows.py` through the same
synthetic package with the same stand-ins. `fake_attention_idx` delegates to `ragged_world.fake_attention` when there
is no `cache_batch_idx` and no log-sum-exp, so the EXACT path uses the shipped path's stand-in.

## 5. Files and commands for the GPU cell vr1

    bash benchmarks/Efficiency/r2_verify_step.sh --nosi <sha> <holder> <run id> vr1
    DRYRUN=1 bash benchmarks/Efficiency/r2_verify_step.sh --nosi <sha> <holder> <run id> vr1

- Arms, one process, each with its own shipped prefill: SH = mb1 (`--pool 65`, B 216, 512 steps, `NOSI_ATTN_SPLITS=4
  NOSI_POOL_BLOCKS=65`, plus the logits fingerprint, two reductions per step, declared as in rg1); VRX (EXACT,
  rowgroup); VRC (EXACT, natural: one weight read); VRF (FAST, natural). K 3, R 4, 128 joint passes per arm,
  teacher-forced on SH's tokens. Twin torch, body fused, X 16, a probe pass every 16 passes.
- Per arm: the joint pass median (pass 0 and the probe passes excluded), joint / SH step, committed tok/s
  (B x R / joint pass), per-row argmax agreement with SH, fingerprint equality with SH; on probe passes the phase times
  (pre, twin, select, probe, rows / resid, attn, end, post, head) with `select` = R rows and `probe` = the one-row
  selection of row 0 (B entries); the union per (head, request) mean / max; FAST: extras and the counted overflow.
- MUST gates: SH-RUN; per arm `-RUN`, `-ERR` (no body refusal bit, no twin overflow flag), `-SEL0` (on every probe the
  batched selection's row 0 = the one-row selection as a set); VRF-IDX (pass 0, layer 0: 4 sampled entries of call A
  bitwise equal a plain one-entry call with the entry's own bias row). REPORT: `-SEL0-ORDER`, VRF-OVF.
- STEP_MIN 100, `--mem 240G` (four prefills of about 11 min each; mb1 took 00:14:31 with a 657.7 s prefill).

REGISTERED BLOCK, PREDICTIONS TBD FOR THE LEAD. Fill `PRED_VERIFY` in `r2_verify_cell.py` and its `stamp` (from
`date -u`, L33) in a commit BEFORE the run. Until then each prediction prints REPORT.

| name | quantity | registered value |
|---|---|---|
| VRX-BIT | every VRX fingerprint equal to SH's | TBD |
| VRC-AGREE | VRC teacher-forced argmax agreement >= | TBD |
| VRF-AGREE | VRF teacher-forced argmax agreement >= | TBD |
| VRX-STEP / VRC-STEP / VRF-STEP | joint pass / SH step, (low, high) | TBD |
| VR-SEL | VRF selection of R rows / one row, (low, high) | TBD |
| VR-UNION | VRF union per (head, request), mean, (low, high) | TBD |

INPUTS FOR THE LEAD'S BANDS (MODEL from MAP_cached_nosa.md 2.1, not measured; per shipped step at C 128 B 216:
selection 8.5 ms, residency 14.8, attention 19.2, GEMMs 19.9, rest 2.4 = 64.8):
- VRX: R x (selection + residency + attention + GEMMs) = about 4 x 62 = about 250 ms, ratio about 3.9 (3.4-4.6). Its
  value is the bit test, not the time.
- VRC: the GEMMs at M 864 are compute-bound: 2 x 7.5e9 x 864 = 13 TFLOP, about 45-55 ms against 4 x 19.9 = 80 ms
  as row groups. About 215-230 ms, ratio about 3.5 (3.0-4.2).
- VRF: GEMMs about 50, selection about 4 x 8.5 = 34 (less if the compressed keys stay in L2), residency 14.8 + extras
  5-10, attention call A about 4 x 19.2 = 77 plus masks 3 and call B about 5, twin and copies about 10: about 200 ms,
  ratio about 3.1 (2.5-3.8). Below K + 1 = 4 shipped steps, but not far below: route 1 attends every row separately.
- VR-SEL: about 3.5 (2.5-4.2). VR-UNION: 64 plus about 3.5 changed blocks per stream per step (43,806 pool swaps +
  5,256 loads over 13,824 streams) over 3 row transitions, minus the blocks that come back: about 70 (66-76).
- AGREE: FAST and VRC differ from SH by rounding only (GEMM at M 864, attention sum order): about 0.99 or more.
- VRX-BIT True needs on the GPU: row-invariant stage 1 (per entry, no split), pooling (per program), `torch.topk`
  with sorted=False in the single-block algorithm for both 432 and 1,728 slices of width 261 (PyTorch picks it below
  slice width 800), nosa_linear (one program per row, no autotune: `nosa_linear.py:110`), flashinfer rmsnorm / rope.

## 6. Risks and open points

- I did not run anything. A failing PRECONDITION (for example extras = 0 on SF8, or block 0 never dropped) means the
  CPU data do not exercise that path; tell me and I change the data, not the claim.
- The bias-row view relies on the kernel reading the bias by query entry. VRF-IDX gates it on the GPU; if it fails,
  FAST is invalid and VRF's numbers must not be used.
- FAST is not bitwise (section 3). Near-tie selection moves in later layers can follow from its rounding; the CPU test
  counts them (at most 2 allowed) and the GPU cell reports the agreement.
- FAST's end state differs from the serial one (row 0's residency only; the old tail block is not kept in a window
  slot after a block edge: the next step fetches it). EXACT ends in the serial state.
- FAST copies the X - 1 extras slots of every (head, request) through a `torch.where` even where no pool hit exists
  (about 0.2 GB per layer at B 216, X 16). A compact copy is a later optimisation, not a kernel.
- The torch twin advances R times per layer (about 72 small kernels each, R2 M-P3): about 9,000 launches per pass. The
  joint pass is device-bound by the model above, but the probe phases show the twin share. `--verify-twin fused`
  exists (the fixed-order mean: a declared change of bits; not for VRX-BIT).
- At long context (M_alloc >= 800) `torch.topk` on 1,728 slices can switch to the multi-block algorithm and change the
  order of equal-set results: VRX-BIT can then fail while VR-SEL0 (set) passes. 16K is below that width.
- Memory: VRF adds about 0.25 GB (the extras store at X 16) and the E-row activations (logits 864 x V float, about
  0.44 GB). mb1's peak is 82.0 GB (allocator); OOM in VRF is possible. A smaller X or K is the fallback.
- The joint pass refuses a request that is not active (a per-request accept is Task 1.5). The draft row runs as a full
  row here; Task 1.1's resident-only mask for row K is an integration item (rung L4).
- The CPU stand-in of attention adds the bias after the scale (`ragged_world.fake_attention`); the kernel adds it
  before. This does not change a CPU claim (every compared path uses the same stand-in), but the stand-in is not the
  kernel's math.

## 6b. FIX 1 (CPU run a at ee3f9ee: 40 passed, 3 skipped, 5 failed)

- Record: `/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/vr_ee3f9ee_a.txt`. Failed: `test_verify_equals_serial_shipped[B8]`
  (group 1 row 0), `test_verify_equals_serial_ragged[B8]` (group 0 row 1), `test_every_verify_row_lists_block0`
  (group 1 row 3), and `test_one_weight_pass` / `test_no_per_position_selection_loop` (B 4: every row's logits
  bitwise equal, then the tables `_kv_bias_gpu` and `twin.total_cis` of both layers and `twin.ccis` of layer 1).
- NOT undefined memory (hypothesis a): both drivers assert "the two worlds are bitwise equal before step 0", and that
  check covers `twin.total_cis`, `twin.ccis` and `_kv_bias_gpu` (all fully defined: zeros, NaN poison or prefill).
  No run raised it.
- WHERE (from the tables): only the cis path differs. Keys (`ring`, `comp_k`), counters, K / V, maps and host rows are
  equal. Layer 0's cis is a function of the token only, so its VALUE can differ only through an op that is not
  row-invariant. Every failing test compares an E-row computation with a B-row one; every passing test at B 8
  compares E rows with E rows (FAST against EXACT, union) or B rows with B rows (flag off).
- ROOT CAUSE: the CPU stand-in of nosa_linear (`ragged_world.fake_nosa_linear`) computed `F.softplus` over the whole
  contiguous (rows, 1, H) tensor. PyTorch's CPU elementwise loop (ATen `cpu/Loops.h`, `vectorized_loop`) runs the SIMD
  (Sleef) exp / log1p on full chunks of 2 x the vector width and the scalar libm path on the rest, so an element's
  bits depend on the tensor size. With AVX-512 (chunk 32 floats): B 1 = 2 against 8 elements, both scalar (pass);
  B 4 = 8 (scalar) against 32 (SIMD); B 8 = 16 (scalar) against 64 (SIMD) (fail). With AVX2 (chunk 16) B 8 would
  pass, so the run's host used AVX-512. The KERNEL computes cis in one program per (row, head)
  (`nosa_linear.py:84-93`, grid `:110`): it has no such dependence. So the stand-in broke lesson L28 (a CPU twin must
  model the kernel), not the joint pass. A 1-ULP change of one bias column among about 4,000 attended columns is
  below float32 resolution, which is why the B 4 logits stayed equal while the tables did not. Task 1.3's
  ragged-vs-solo test at B 8 (16 against 2 elements, both scalar under AVX-512) passed by this margin only.
- The first differing element: I could not compute it (no interpreter on the login node). By the code it is an
  element `twin.total_cis[b, p, h]` with p among the group's positions, where Sleef and libm softplus disagree. The
  drivers now print the first differing index of every table and of the logits (`first_diff`).
- FIX: `ragged_world.fake_nosa_linear` computes the cis row by row (`A * softplus` of one (1, H) row at a time), as
  the kernel's per-row program. Rows of B 1 keep their former bits (2 elements were already scalar). No engine code,
  no behaviour assertion and no tolerance changed. New test `test_cis_stand_in_is_row_invariant` (B-row slices of an
  E-row call: same bits; negative control: the former batch-wide softplus, skipped when it is a no-op on the host).
- Other stand-ins on the path, checked: rmsnorm runs per row; silu acts on a non-contiguous slice (its inner loop is
  one row); rope, pooling, top-k and the residual adds are exact; stage 1, attention and the twin means run per entry
  or with the same shapes in both worlds.

## 7. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- `sed`: ONCE, read-only, by mistake, in a pipeline (`grep ... flash_api.cpp | sed -n '1,0p'`): it printed nothing and
  changed nothing. This BROKE the "no sed" rule.
- `cd` (shell built-in) in front of read commands.
- No interpreter, no awk, no heredoc, no `cat >>`, no tr / paste / uniq / xargs, no hashing, no `find`, no srun /
  sbatch / pytest. Files were written with the Write and Edit tools; the commit message with the Write tool.
