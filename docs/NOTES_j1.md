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
| j1m | C 73 pool 10, B 8, 72 steps, check every round, stress | L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1 | 30 / 100G | `bash benchmarks/Efficiency/r2_j1_step.sh --nosi <sha> <holder> <run> j1m` |
| j1a | C 128 pool 65, B 216, 512 steps | L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1 | 110 / 240G | `... j1a` |
| j1b | C 73 pool 10, B 318, 512 steps | L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1 | 170 / 340G | `... j1b` |

MUST gates per J1 arm: every gate of its I4DC arm under the J1 name (RUN, ERR, COMMIT-N, D1, D3, SPAN, RESID, ATTN1,
FILL1, LEAN, MEM, GRAPH, COUNTS, DGRAPH, HOIST, the keep gates, COMPILE, EQF = L4PTFD bitwise), J1-PEQ (0 differing
values; checked layers > 0), J1-COMPILE (at most one compile event, none in a capture, no fallback, calls = joint passes
x 32). Since the I4DC arm in the same job also passes EQF, the J1 arm is bitwise its I4DC arm.
Decision rule (j1a): ADOPT F1 when every MUST gate passes and J1-JOINT-SAVING (REF joint median - J1 joint median) is
in [9.5, 13.5] ms. Falsifier: J1-JOINT-SAVING < 5 ms (the compiled gather is far from its byte floor), or J1-PEQ > 0.
Expected (MODEL; per round = per joint pass, the draft unchanged):

| term (original formula) | C 128 B 216 | C 73 B 318 |
|---|---|---|
| joint selection, pooling part (in H's verify rows; the model charges 0) | 12.28 -> 0.5-1.0 ms: -11.6 [9.5, 13.5] | 18.10 -> 0.7-1.5: -17.1 [14.0, 18.5] |
| round (BEST 324.88 / 524.66 ms MEASURED) | about 313.3 ms; tok/s x1.030-1.043 | about 507.6 ms; x1.027-1.037 |
| draft passes (n td) | 0 (F2 next: -5.7 / -8.6 per round) | 0 |

## 6. One row per formula term owned by AREA 1 (LEDGER_rev7 section-1 format)

| term | original prediction + assumptions | measurement / work volume | gap | evidenced cause vs hypothesis | fix | measured effect |
|---|---|---|---|---|---|---|
| D = n G(B) + G(BK): shapes and calls | 3 G(B) + G(3B) per cohort, two call sets; shared weights | 2 G(B) + G(4B) per round, ONE call per weight over 4B rows (129 launches per pass); 64.80 / 79.88 ms joint | +1.35 / -10.09 vs the two-call charge | MEASURED: shared weight reads hold; reference above M 768 extrapolated | none (lead owns dense) | - |
| extra = td (1 - f)(U - I)/63: union U, ready I | U 69.16, I 60.19 (AUTHOR, historical, four requests, teacher-forced) | LOGICAL union 71.18 + 1.06 tail per (layer, KV head, request), C 73 check arm only; I live: MISSING (no live ready-set counter) | x0.99 logical (C 73) | MEASURED logical; C 128 and live I MISSING | X3: VerifyOps(stats) in a C 128 miniature (j1m can carry it later) | - |
| physical HBM vs logical union bytes | one read per union page | fork DRAM 1.017-1.021 x union (ncu, fia6 microbench on real tables); live pass MISSING | about +2 % | MEASURED (microbench) | X2 ncu on one live fork launch | - |
| attention tiling / masks / per-request independence | KV time hidden by max(H, D) | fork 37.1 / 52.2 ms; 64-row tile, 2 programs per SM (255 regs, 59 KB smem), 8.1 us per tile, 2 waves x 75.8 tiles; X1: A and F do NOT overlap on two streams (97.8 vs 100.2 ms serial) | +11.6 / +14.7 above the model's extra; 0.885 vs 1.527 TB/s logical | MEASURED (fia5 TTGIR + ncu): occupancy and synchronous per-tile loads; 11 launch configs swept | none without a kernel edit (BLOCKER); 3D split = tail balance only (<= 10 %) | - |
| selection (verify rows) | 0 (B draft rows inside td) | stage 1 9.82 / 13.54, POOLING 12.28 / 18.10, top-k 2.91 / 4.12, small 2.30 ms per joint pass | +17.83 / +24.43 (3B rows) | MEASURED: pooling = one program per output element (grid-bound, about 98 % of its time) | F1 (this rung): compiled gather + max, bitwise | PENDING j1a / j1b |
| gather / materialization | none | cis copy to R rows (H, B, R, Lk) per layer; chain images (AREA 3); H2D gathers (AREA 2) | small (cis copy < 1 ms) | CODE | F1 removes the cis copy | PENDING |
| launch overhead | td includes its own launches | BEST joint non-busy about 6.3 / about 0 ms (MODEL labels); 2L draft graphs + L eager residencies | +6.3 / 0 | MODEL | F3: whole-layer graphs with the shifted-age LRU frame of the strongest baseline | not built |

## 7. Commands (this rung, login node)

The login-node rules held: no interpreter, no sed, no heredoc, no hashing, no `find /`. Files written with the Write /
Edit tools. Commands: `git -C /mnt/beegfs/ojerbi/nosi-r2-recon7 worktree add /mnt/beegfs/ojerbi/nosi-r2-j1 -b r2-j1
cd0f0c5...`, `git -C ... log / status / diff / show`, `ls`, `wc`, `grep`, `head`, `tail`, `cat`, `date -u` (the full
list is in the builder's report to the lead).
