# NOTES J1: the compiled joint pooling (AREA 1, fix F1)

Builder notes, 2026-10-31 (UTC from `date -u`). Style: ASD-STE100. Design brief:
/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/DESIGN_JOINT_EFF_2026-10-31.md (fix F1; the ranked list F1-F5).
Tree /mnt/beegfs/ojerbi/nosi-r2-j1 (branch r2-j1) on RECON7 cd0f0c5. SOURCE ONLY: no CPU or GPU job ran for this rung.

## 1. What and why (the evidence)

- MEASURED (KS/cpu-logs/recon7_r2_s2.txt: the I3 joint path = the I4DC arms' selection, one profiled pass; KS =
  /mnt/beegfs/ojerbi/kstep-cpu-runs): `nosa_pool_kernel_per_request`, grid (2, 864, 261), block 128: 12.28 ms per joint
  pass at C 128 B 216 (line 21; 383.88 us per layer); grid (2, 1272, 261): 18.10 ms at C 73 B 318 (line 615). The
  one-row passes: 3.14 ms (line 87) and 4.71 ms (line 678) per pass.
- CODE: nosi/nosi/max_pooling_per_request.py:94-138: ONE program per OUTPUT element (program ids (h, b, m)), 8 lane
  loads, `tl.max`, 2 `tl.where`, one store per output. Bytes per joint layer at B 216: about 7 MB (scores 4.0 MB, cis
  0.9 MB, outputs 1.8 MB; MODEL widths). The arithmetic is a max: any correct implementation gives the same bits.
- CODE: nosi/nosi/verify_rows.py:413-432 (`_select_rows`): stage 1 (:420), the cis copy to R rows (:422), the pooling
  (:424-426), top-k (:427), mask zero / scatter / masked_fill (:428-430), top-k (:431). The joint pass calls it at
  nosi/nosi/spec_tentative.py:360, outside every captured section of the I4DC arms (the I4G section graph starts after
  the residency).

## 2. The change (benchmarks/Efficiency/r2_j1_select.py)

- `pool_fn`: the kernel's arithmetic as torch ops (one advanced-index gather per input, `where`, `amax`, the edge and
  validity `where`s), compiled with torch.compile (Inductor, fullgraph, dynamic=False). It reads the compressed cis as
  `ccis[h, e // R, n]` (no (H, B, R, Lk) copy). It writes vs.pq / vs.pc in place.
- `_J1Mix._layer`: spec_tentative.py:318-369 COPIED, with 3 lines marked `# J1` (R 1 delegates to the I4DC path; the
  joint selection calls `_j1_select`). tests_cpu/test_j1.py::test_j1_layer_copy pins the copy line for line.
- `_j1_select`: the shipped stage 1; `pool_fn`; the shipped top-k / mask ops through verify_rows' torch binding.
- J1-PEQ (check rounds only): the shipped copy + kernel on the same score / cis into scratch; integer-view comparison.
- The runner re-classes the I4DC state (`T.__class__`); every I4DC lever, graph and compile helper stays as it is.
- Untouched: the one-row (draft) selection (fix F2 is the next rung), every kernel, nosi/.

## 2b. F2 (second commit): the draft passes' pooling, and the same fix on the baseline (SHJ1)

- Draft: the I4H draft graphs (benchmarks/Efficiency/r2_i4h_graph.py:304-371) capture A(l) with the layer's
  selection inline (`_Inline`, :155-165, swapped in at :326-328, restored at :340-342). The J1 state overrides
  `_dbegin_a`: (1) before the capture, ONE eager warm call compiles pool_fn for the B-entry shape on scratch outputs;
  (2) after the capture begins, `rs.graphs[l]` = `_InlineJ1` (the compiled pooling + ragged_engine._pool_topk's
  other ops on the same tensors, ragged_engine.py:333-343). Counts (rs.replays) stay the shipped ones: RE._select
  counts after `replay()` (:369-370). Check-key graphs also run the shipped kernel into scratch: device counters
  dpeq_bad / dpeq_vals / dpeq_layers (gate J1-DRAFT).
- MEASURED basis: 3.14 ms per one-row pass at B 216 (recon7_r2_s2.txt:87), 4.71 ms at B 318 (:678).
- Baseline: SH's shipped step calls `nosa_llama.nosa_pooling` (the scalar kernel, max_pooling_fused.py:7-107: the
  same one-program-per-output grid) at nosa_llama.py:447 (eager) and :462 (inside the after-pooling graph). The cell
  runs ONE extra shipped run (SHJ1) right after SH with that global replaced by `ShPool` (pool_fn compiled with dynamic
  shapes behind the shipped signature; the eager warm-up calls compared with the shipped kernel), then restores it.
  Gate J1-SHEQ: SHJ1's tokens = SH's bit for bit. Verdicts J1-SH-SAVING, J1-VS-SHJ1, J1-STRONGEST-MATCHED (MODEL: the
  whole-step graph's 3,407.2 tok/s with SHJ1's per-step saving; the whole-step graph itself lives in the tree
  nosi-r2-graphstep, not in this lineage: a measured matched strongest needs its port, not done).

## 2c. F3 (third commit): the residency inside the captured graphs (arm L4PTFDGUDCDJ3)

- Blocker in the I4G / I4H / I4S rungs: pool_update takes the host int stamp (spec_tentative.py:461-466,
  r2_i4a_lean.py:206-210, r2_i4u_hoist.py:188-192), so the residency stayed eager (r2_i4g_graph.py:12-15,
  r2_i4h_graph.py:15-17, r2_i4s_stack.py:28-31) and the rungs called a device stamp "a kernel edit".
- Reuse: the strongest baseline already removed it without a kernel edit: constant stamp STAMP_K = 64 << 31 and every
  real age (>= 0) moved down by topk before each call (nosi-r2-graphstep 0159e82 nosi/nosi/graph_step.py:17-19, :59,
  :146-154, :243-244; gs1 bitwise vs the eager step, SCORING_gs1.md). StampFrame applies it per call and layer as the
  arm's pool_update (shift_in at the layer's first, eager call; shift_out + the host stamp at export after the run).
- Capture: the joint section graph now holds the residency (r2_i4s_stack.py:132-176 copied, 6 lines removed, 4 marked
  "# J3"; expected host counts + calls["resid"] 1); the draft graphs hold A(l) + residency + B(l) as one graph per
  layer (r2_i4h_graph.py:332-391 overridden: _dstub, _dend_b, _dreplay); the draft key adds meter.on (the drain's
  target-row passes count host blocks).
- Expected effect (quantitative explanation, MODEL): the passes are device-bound. L4PTFD profile i4c e4a (C 128): joint
  busy 195.7-198.9 of 207.5 ms elapsed, draft busy 137.7-138.5 of 144.0 ms (RECON7_R2_joint.md:187-189). Host launch
  work of the residency: about 15 launches x 32 layers x about 5 us = 2.4 ms per pass, issued while the GPU is busy
  with earlier layers; it is exposed only where the queue runs dry (pass start). Bound: joint 0 .. 6.3 ms (BEST's
  non-busy label, R2:300), draft 0 .. 2 x 3 ms; central about 3 ms per round at C 128, 0 .. 3 at C 73. A result inside
  the run spread is the expected outcome, not a defect; J1-RESID and EQF decide correctness.
- Keep arm (C 73), READ-ONLY ANSWER (2026-10-31): capturable with ONE copied method; not built.
  The keep arm's joint residency is r2_i4d_keep.py:381-425 `_resid_k` (kwin), reached through r2_i4k_combo.py:181-185
  `_resid_joint` and stubbed by name in the I4S wrapper (r2_i4s_stack.py:426-431, stub "_resid_joint"). Line by line:
  diff, masked_fill, pool_update (host stamp: StampFrame removes it), pool_swap, the map commit, the lookup of row 0's
  loads in the persistent map kid (:404-410: a broadcast compare, device only), two host gathers and two store gathers
  (:411-416, shipped kernels, persistent sources vs.ext_k / ext_v / ext_bias), the check-only _win_bytes (:438-463: fresh
  scratch tensors and the unmetered gathers, device only; under a capture its scratch comes from the graph pool), the
  skip counters. ONE host-dependent op: :417 `self._krow(self.kpass)[6].add_(found.sum())` indexes the counter table by
  the HOST pass index, so a capture would freeze the row (the I4K rung's own "pass_row" control, r2_i4k_combo.py:56-58).
  The fix is the I4K pattern: add into `self._crow()` (kcur, flushed into the pass's row eagerly after layer L - 1 by
  _kpost / _kflush, :171-179), field 6; the order constraint (the lookup reads kid before step 5 of the same layer writes
  it, r2_i4k_combo.py:34-35) holds inside the captured section because the residency precedes _keep_fill there. The
  keep arm's draft residency is the lean one (r2_i4d_keep.py defines no _resid_lean): F3's draft part applies as is.
  Expected effect at C 73: about 0 (the C 73 passes are device-bound: joint busy 379.3-383.9 of 383.2 ms elapsed,
  RECON7_R2_joint.md:203). RECOMMENDATION: build the keep J3 arm only if j1a shows J3-JOINT-SAVING + J3-DRAFT-SAVING
  above the run spread at C 128.

## 2d. F6 (fourth commit): the joint section's locate without the map sorts (J1 and J3 arms; runner keyword j1f6)

- MEASURED (r7pa, R/recon7_pa1_2195631_r7pa summary.json, the BEST arm's labelled joint window, round 112): the
  section's SELECT class 4.30 ms, of which radixSortKVInPlace<..., 4> 2.46 ms (96 kernels = 3 per layer: the window
  map, the pool map and the host key) and <..., 32> 1.34 ms (32 kernels: the row-list key).
- CODE: r2_i4cc_compile.py:319-325 sort the window and pool maps only to find a row id's slot with searchsorted
  (loc1_fn, :108-124). F6 (r2_j1_select.loc_fn) builds per (head, request) an inverse table block id -> slot by ONE
  scatter of the slot index and reads it with ONE gather; the host-key and row-list sorts stay (the fork needs the
  ascending lists). Exact: valid ids are unique in a map; wslot / pslot are read only under in_win / in_pool (loc2_fn,
  :144). `_cc_locate` is copied with 3 lines marked "# J1" (tests_cpu/test_j1.py::test_cc_locate_copy).
- Expected: about 2/3 of the 2.46 ms short sorts (2 of 3 per layer) = 1.6 ms per joint pass, minus the scatter / gather
  (about 32 x 10 us = 0.3 ms): about 1.3 ms per round at C 128 (MODEL), C 73 similar per row count (x1.47: about 1.9).
- Gates: J1-COMPILE (one "f6" compile, not in a capture, F6 calls = the compiled sections), the I4CC COMPILE gate with
  loc1 left out of the J1 arms' helper list (the cell), EQF.

## 2e. j1f (fifth commit): the fork microbench, matched (benchmarks/Efficiency/r2_j1_forkbench.py)

- Why: the ledger's 0.885 vs 1.53 TB/s compares the fork (4 rows, union pages, per-row masks, no split) with the
  draft's one-row flash call A (1 row, own window, split-KV 4). That pair is UNMATCHED in rows and bytes.
- What: synthetic pages (seeded, x 0.05 bf16) of the MEASURED shapes (B 216 at C 128, B 318 at C 73). Each request has
  4 rows of 64 pages, shifted by (0, 3, 5, 8) pages: union 72 per (KV head, request) (MEASURED 72.2 / 72.24). One pass
  = 32 calls (one per layer). Variants (existing ops only: the fork blob f0dc0c1, flash_attn_nosa, torch):
  FORK (shipped: FORK_LAUNCH w4 s3 t64, bf16 CIS bias, union built in the kernel); FORKT (union tables built before);
  FORK3D2 / 4 / 8 (the fork's existing 3D path: seq_threshold_3D = B, n segments, reduce_segments); FORK1R (the fork,
  1 row per request, BLOCK_Q 1); NS1R1 (flash call, 1 row, own 64 pages, split 4); NS1x4 (flash call, 4 rows, each
  its own 64-page window); FLASHU1 (flash call, 1 row over the 72 union pages); NSRU (flash call, 4 rows over ONE union
  window per request through cache_batch_idx and per-entry masking bias rows: verify_rows route 1).
- Matched ratios: FORK / FLASHU1 (equal logical bytes); FORK1R / NS1R1 (equal rows and bytes); FORK / NS1x4 (equal
  rows); FORKT / FORK (the in-kernel union build); FORK3Dn / FORK; NSRU / FORK. Per variant: median ms per pass and per
  call, logical bytes per call (pages x 32,896 B x KV heads x requests or entries) and logical TB/s.
- Gates (MUST unless noted): <shape>-RUN; <shape>-UNION (FORK union_len = 72, FORK1R = 64 everywhere);
  <shape>-<v>-NUM (max |out - float64 reference| <= 1e-3 on 8 sampled requests, every row and head; FORK3Dn REPORT);
  <shape>-FORKT-EQ (FORKT = FORK bit for bit); <shape>-<v>-EXACT (output after the last timed pass = the first output,
  bit for bit; FORK3Dn REPORT); <shape>-FORK3Dn-PATH (REPORT: the NaN-filled segment maxima were written, so the 3D
  launch ran); <shape>-BUDGET (300 s); RUN-WALL (900 s). A ratio with a member that fails NUM, EXACT or PATH is
  INVALID. Order rotation per pass (lesson L62), 2 warm-up passes per variant, 10 timed passes.
- Decision rules (the lead's): (1) FORK3Dn best ratio <= 0.95 with PATH, NUM and EXACT: the 3D split is a candidate
  (a launch option, not a kernel edit; it changes the float order, so it needs its own EQF check in the joint arm).
  (2) FORK1R / NS1R1 > 1.2: the fork kernel has a per-byte gap at equal work (structural; F5 of the brief). (3) FORK /
  FLASHU1: the cost of the 4-row tile at equal bytes; FORK - FLASHU1 x 32 is the most a rate fix could save per pass.
  (4) FORK / NS1x4 < 1: the shared read already beats the baseline's 4 calls.
- CPU: tests_cpu/test_j1.py::test_forkbench_pure (work, row pages, rotation, summarize, INVALID marking, default deny,
  argument checks) and test_step_script_j1 (the j1f case and its arguments).

## 2f. The exported factories (sixth commit; no behavior change)

- `j1_wrap(STF, SB, RB, arm)` and `j3_wrap(STF, S1, R1, arm)` (r2_j1_select.py) return the (state, runner) pair of
  F1 + F2 + F6, and of F3 on a J1 pair, for ANY base classes. They are the bodies of the old ARMS_J1 / ARMS_J3 loops,
  moved into the local factories `wrap_j1` / `wrap_j3`; the arms of j1_classes are built by them and cached under the
  same key, so `j1_wrap(STF, *DCL[base], arm) == j1_classes(STF)[arm]`. Reason: the integration rung builds J1 on the
  BF1 keep classes (KS/analysis/plan/INTEGRATION_NOTES_2026-10-31.md section 3.3). CPU: test_j1.py::test_j1_wrap_export.

## 3. The reuse line (the lead's rule: upstream basis, incompatibility or cost, minimal adaptation, validation)

- Upstream basis: NOSA's `nosa_pool_kernel` (nosi/nosi/max_pooling_fused.py:7-75) as forked per request in
  max_pooling_per_request.py (G4 / G6 gates of benchmarks/Efficiency/r2_kernels_gate.py:131-169); torch 2.6.0
  `torch.compile` (Inductor), already the I4CC / I4DC rung-1 tool. Measured cost: the per-element launch grid (12.28 /
  18.10 ms per joint pass). Minimal adaptation: the same arithmetic as compiled torch ops (no kernel source, no
  extension rebuild). Validation: J1-PEQ (bitwise on live inputs, every check-round layer), EQF (= L4PTFD bitwise),
  the CPU tests (pool_fn = the kernel's flat-load stand-in and pool_reference, eager and Inductor).

## 4. FlashInfer / SGLang routes for the fork (source account, lead's request 2026-10-31; no new experiment)

Source: KS/analysis/plan/RECON7_R4_prior_art.md (R4) sections 1.2, 1.5, 2.4 (flashinfer 0e68a2f, sglang e371881).
- JIT logits transform (`REGISTER_LOGITS_TRANSFORM`, variants.cuh:67-76, R4:232; extra tensors, jit/attention/
  modules.py:470-532): it receives (qo_idx, kv_idx, qo_head_idx, kv_head_idx), so the CIS bias (a per-token tensor read
  by the physical page) and the per-(row, KV head) selection (a lookup of our 8-bit extents) FIT as a JIT variant, with
  one FlashInfer "request" per (request, KV head) and num_kv_heads 1 (R4:230-231). It is CUDA template code compiled by
  FlashInfer's JIT: the smallest fork of FlashInfer, not a stock call.
- Planning: `plan()` reads indptr / seq_lens to the host (prefill.py:1724-1735, R4:223-224) and "cannot be used in Cuda
  Graph" (:1675). SGLang's sync-free `fast_prefill_plan` needs host-known indptrs and asserts page_size == 1 for a
  custom mask (flashinfer_backend.py:196-305, :280; R4:113-114). Our union lengths are device values that change per
  LAYER, so neither plan form fits without a FIXED padded page list per (request, KV head) (pad entries masked by the
  transform), plan once per shape, `disable_split_kv` for determinism in a graph (prefill.py:1657-1664).
- Rate evidence: the stock prefill wrapper with a custom mask MEASURED 1.68-1.70x our fork's device time on our tables
  (record /mnt/beegfs/ojerbi/r2-runs/fiattn/fia3_f1a_2195573; fork header lines 14-18): 2 CTAs per SM (255 registers,
  81 KB shared memory), one wave, latency bound. A JIT transform removes the mask bytes, not that occupancy limit; a
  padded fixed list adds tiles. So the route has no evidenced rate gain over FV6 on sm80.
- Bit-exactness: any FlashInfer route changes the attention's float order vs the fork (FAST class): committed tokens
  are not bitwise the BEST arm's; it needs the near-tie accuracy gate, not EQF.
- Decision: NOT pursued now. The fork's rate gap is the structure of a 64-row fp32 tile on sm80 (F5 of the brief); the
  evidenced, bit-exact, larger gap in AREA 1 is the pooling grid (this rung).

## 5. GPU cell protocol (the lead runs it; grants and registration are the lead's)

Step script: benchmarks/Efficiency/r2_j1_step.sh (a copy of r2_recon7_step.sh: no hash on the login node; the image
and workload sha256 on the GPU node). Grants: `GRANT_j1m / GRANT_j1a / GRANT_j1b` in r2_j1_cell.py (default deny).
Registration: `PRED_J1_BY_B` + `PRED_STAMP_J1` (the builder's MODEL is the comment block above them).

| cell | config | arms (SH first) | STEP_MIN / --mem | command |
|---|---|---|---|---|
| j1m | C 73 pool 10, B 8, 72 steps, check every round, stress | L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1,L4PTFDGUDCDJ3 | 35 / 100G | `bash benchmarks/Efficiency/r2_j1_step.sh --nosi <sha> <holder> <run> j1m` |
| j1a | C 128 pool 65, B 216, 512 steps | L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1,L4PTFDGUDCDJ3 | 125 / 240G | `... j1a` |
| j1b | C 73 pool 10, B 318, 512 steps | L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1 | 170 / 340G | `... j1b` |
| j1f | no model, no workload; shapes B 216 / B 318, union 72, segments 2,4,8, 32 calls x 10 passes | (section 2e variants; grant GRANT_j1f in r2_j1_forkbench.py) | 25 / 64G | `... j1f` |

MUST gates per J1 arm: every gate of its I4DC arm under the J1 name (RUN, ERR, COMMIT-N, D1, D3, SPAN, RESID, ATTN1,
FILL1, LEAN, MEM, GRAPH, COUNTS, DGRAPH, HOIST, the keep gates, COMPILE, EQF = L4PTFD bitwise), J1-PEQ (0 differing
values; checked layers > 0), J1-COMPILE (at most one compile event, none in a capture, no fallback, calls = joint passes
x 32), J1-DRAFT (every captured one-row layer runs the compiled pooling, one warm call, draft PEQ 0 of > 0 values);
once per job J1-SHEQ (SHJ1's tokens = SH's bitwise; eager pooling PEQ 0; pooling inside the captures). Since the I4DC
arm in the same job also passes EQF, the J1 arm is bitwise its I4DC arm. Decision rule for F2 (j1a): J1-DRAFT-SAVING
in [4.5, 6.2] ms and J1-SH-SAVING reported; the verdict against the baseline uses J1-VS-SHJ1 (matched), not J1-RATIO.
Decision rule (j1a): ADOPT F1 when every MUST gate passes and J1-JOINT-SAVING (REF joint median - J1 joint median) is
in [9.5, 13.5] ms. Falsifier: J1-JOINT-SAVING < 5 ms (the compiled gather is far from its byte floor), or J1-PEQ > 0.
Expected (MODEL; per round = per joint pass, the draft unchanged):

| term (original formula) | C 128 B 216 | C 73 B 318 |
|---|---|---|
| joint selection, pooling part (in H's verify rows; the model charges 0) | 12.28 -> 0.5-1.0 ms: -11.6 [9.5, 13.5] | 18.10 -> 0.7-1.5: -17.1 [14.0, 18.5] |
| round (BEST 324.88 / 524.66 ms MEASURED) | about 313.3 ms; tok/s x1.030-1.043 | about 507.6 ms; x1.027-1.037 |
| draft passes (n td), F2 | 2 x (3.14 -> 0.2-0.4) = -5.6 [4.5, 6.2] per round | 2 x (4.71 -> 0.3-0.5) = -8.6 [7.0, 9.2] |
| round with F1 + F2 | about 307.7 ms; x1.045-1.065 vs REF | about 499 ms; x1.042-1.056 |
| matched baseline SHJ1 (per SH step) | -2.8 [2.0, 3.2] ms | -4.4 [3.4, 4.7] ms |

## 6. One row per formula term owned by AREA 1 (LEDGER_rev7 section-1 format)

| term | original prediction + assumptions | measurement / work volume | gap | evidenced cause vs hypothesis | fix | measured effect |
|---|---|---|---|---|---|---|
| D = n G(B) + G(BK): shapes and calls | 3 G(B) + G(3B) per cohort, two call sets; shared weights | 2 G(B) + G(4B) per round, ONE call per weight over 4B rows (129 launches per pass); 64.80 / 79.88 ms joint | +1.35 / -10.09 vs the two-call charge | MEASURED: shared weight reads hold; reference above M 768 extrapolated | none (lead owns dense) | - |
| extra = td (1 - f)(U - I)/63: union U, ready I | U 69.16, I 60.19 (AUTHOR, historical, four requests, teacher-forced) | LOGICAL union 71.18 + 1.06 tail per (layer, KV head, request), C 73 check arm only; I live: MISSING (no live ready-set counter) | x0.99 logical (C 73) | MEASURED logical; C 128 and live I MISSING | X3: VerifyOps(stats) in a C 128 miniature (j1m can carry it later) | - |
| physical HBM vs logical union bytes | one read per union page | fork DRAM 1.017-1.021 x union (ncu, fia6 microbench on real tables); live pass MISSING | about +2 % | MEASURED (microbench) | X2 ncu on one live fork launch | - |
| attention tiling / masks / per-request independence | KV time hidden by max(H, D) | fork 37.1 / 52.2 ms; 64-row tile, 2 programs per SM (255 regs, 59 KB smem), 8.1 us per tile, 2 waves x 75.8 tiles; X1: A and F do NOT overlap on two streams (97.8 vs 100.2 ms serial) | +11.6 / +14.7 above the model's extra; 0.885 vs 1.527 TB/s logical | MEASURED (fia5 TTGIR + ncu): occupancy and synchronous per-tile loads; 11 launch configs swept | none without a kernel edit (BLOCKER); 3D split = tail balance only (<= 10 %); j1f measures it MATCHED (FORK / FLASHU1, FORK1R / NS1R1, FORK3Dn / FORK) | PENDING j1f |
| selection (verify rows) | 0 (B draft rows inside td) | stage 1 9.82 / 13.54, POOLING 12.28 / 18.10, top-k 2.91 / 4.12, small 2.30 ms per joint pass | +17.83 / +24.43 (3B rows) | MEASURED: pooling = one program per output element (grid-bound, about 98 % of its time) | F1 (this rung): compiled gather + max, bitwise | PENDING j1a / j1b |
| gather / materialization | none | cis copy to R rows (H, B, R, Lk) per layer; chain images (AREA 3); H2D gathers (AREA 2) | small (cis copy < 1 ms) | CODE | F1 removes the cis copy | PENDING |
| launch overhead | td includes its own launches | BEST joint non-busy about 6.3 / about 0 ms (MODEL labels); 2L draft graphs + L eager residencies; passes device-bound (busy 0.94-0.96 of elapsed, L4PTFD profile) | +6.3 / 0 | MODEL (exposed host time only at pass start) | F3 (arm L4PTFDGUDCDJ3): residency inside the joint and draft graphs via the shifted-age frame | PENDING j1a (expected 0-9, central 3 ms per round) |
| draft passes' pooling (n td) | inside td | 3.14 / 4.71 ms per one-row pass (one program per output) | +2 x 2.8 / 2 x 4.4 per round | MEASURED kernel | F2 (in the J1 arms) | PENDING j1a / j1b |
| matched baseline | strongest 3,407.2-3,426.2 tok/s with the shipped pooling | SHJ1: SH with the compiled pooling, same job | per SH step -2.8 / -4.4 (MODEL) | the same kernel in SH | F2 on SH | PENDING (J1-SH-SAVING, J1-VS-SHJ1) |

## 7. Commands (this rung, login node)

The login-node rules held: no interpreter, no sed, no heredoc, no hashing, no `find /`. Files written with the Write /
Edit tools. Commands: `git -C /mnt/beegfs/ojerbi/nosi-r2-recon7 worktree add /mnt/beegfs/ojerbi/nosi-r2-j1 -b r2-j1
cd0f0c5...`, `git -C ... log / status / diff / show`, `ls`, `wc`, `grep`, `head`, `tail`, `cat`, `date -u` (the full
list is in the builder's report to the lead).
