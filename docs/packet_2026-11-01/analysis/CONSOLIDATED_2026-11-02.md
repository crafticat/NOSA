# CONSOLIDATED RESULT 2026-11-02: the two tracks at C81 / K3 / B287 / L2, integration, re-optimization

Writer: the lead. Style: ASD-STE100. Labels: MEASURED (a record field or receipt), DERIVED (arithmetic on MEASURED),
MODEL (a stated assumption), UNKNOWN. NO NEW KERNELS were built. No GPU was allocated (human instruction 2026-11-02:
"dont allocate anything yet"). So every "measured effect" of a track fix is PENDING: the external dependency is a GPU
allocation, nothing else.

Detail: INTEGRATION_NOTES_2026-10-31.md sections 16 (track 1, with the lead corrections in 16.4) and 17 (track 2);
REOPT_2026-11-02.md (re-optimization); LEDGER_rev7.md section 17 (schedule and memory mapping); STATUS_LOG.md.
Reference: crafticat/speculative-kv-handoff @ad409e2 docs/ORIGINAL_CALCULATIONS.md.

## 1. Conclusion

This tested implementation misses the original forecast. The best feasible configuration under the 84 GB cap is still
the measured C128 B216 INT arm: 1,910.0 committed tok/s (mean basis; 2,096.1 median basis) = 0.582 x the strongest
cached NOSA baseline (SHJ1 at the same cell, 3,282.5 mean, MEASURED j1a). Neither track changes the choice on the
MODEL: track 2 adds +0.5 % central (+2.1 % at best); track 1 helps only at C81 / C73 and its best configuration
(C81 B311) stays 7 % below C128 INT (central).

The structural reason (MEASURED, C128 B216): one draft step costs 62.635 ms = 1.002 x the SHJ1 decode step. The
round has two standalone draft passes (125.3 ms) plus the joint pass (169.5 ms). To tie SHJ1 the round must take
<= 173.9 ms (DERIVED from Y_committed 2.642130). The two drafts alone take 125.3 ms. So no fix that keeps two
standalone draft passes at the cost of a baseline step can reach the baseline at C128. At C81 / C73 the draft step is
cheaper (0.787 / 0.673 x SH), but the gather wait (90.5 / 115.0 ms per round) and the residency (51.4 / 64.4 ms)
cancel that advantage.

## 2. Exact source and configuration

| item | value |
|---|---|
| fixed case | C81 / K3 / B287 / L2, cap 84.0 GB (usable 83.75, fit margin 0.6: lesson L60) |
| model | /mnt/beegfs/ojerbi/models/NOSA-8B (32 layers, 32 heads, 2 KV heads, head_dim 128) |
| integration branch | r2-int2-dev @ 4695b61 (worktree /mnt/beegfs/ojerbi/nosi-r2-int2-dev) = r2-int-dev a39aaa9 + merge r2-j1-dev 55de6f9 (6adf3a3) + merge r2-bf1-dev 4692273 (9824df6) + post-merge fixes f0762ce + comment fix 86636fe + test fix 4695b61 |
| track 1 source | r2-bf1-dev @ 4692273 (on a347f7c): arm S = L4PTFDGUDKWCDS, the (a)(b)(c) ledger, cell bf1c |
| track 2 source | r2-j1-dev @ 55de6f9 (on fd91420): FORKA1-3 ahead levels, rows sweep, cell j1n |
| measured anchor | intc: R/int_c1_2195631_intc, arm L4PTFDGUDCDI, C81 B287, 1,854.8 committed tok/s (median basis) |
| best measured | inta: R/int_a1_2195630_inta, C128 B216, 2,096.1 (median) / 1,910.0 (mean) |
| baseline | SHJ1 j1a C128 B216 3,282.5 mean / 3,456.6 median; SH 3,169.1-3,217.0 mean over 4 jobs |

R = /mnt/beegfs/ojerbi/r2-runs/spec.

## 3. Correctness checks

- Contract unchanged: bit-exact EQF / PEQ (the optimized arm commits exactly the plain speculative arm's tokens).
  No numerical relaxation. All track edits are bit-preserving; S never reads stale bytes (the GPU waits a late block).
- New MUST gates (track 1): ledger closure (demand = hit_cache + hit_retained + miss_verifier_only +
  miss_req_incomplete + miss_lost) and the consumer-address check (addr_bad = 0: every retained hit is read at its
  own slot). Planted controls: late fetch, eviction before the read, verifier-only demand, address redirect,
  retention pressure. The address plant fails both -BF and EQF, as required.
- Track 2: only FORK and, if GPU-bit-equal, FORKA1-3 keep the contract; the GPU bit gate (FORKAn-EQ) is in j1n.
- NOT a target-correctness claim: the plain speculative arm's tokens equal SH's on 0.388-0.404 of positions. That
  depends on the FAST tie rule (human decision).
- CPU receipts at the integration tip (all pytest exit 0, clean tree, amd thunder09; files in
  /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/<tag>.txt + .receipt):

| tag | tip | scope | result |
|---|---|---|---|
| int2_f0762ce_g0 | f0762ce | Gate 0, 19 files | 41 passed |
| int2_f0762ce_int2 | f0762ce | INT2 tests | 4 passed |
| int2_f0762ce_int | f0762ce | test_int.py | 20 passed, 1 skipped |
| int2_f0762ce_j1 | f0762ce | test_j1.py | 28 passed, 1 skipped |
| int2_f0762ce_bf1a / bf1b | f0762ce | test_bf1.py parts 1-2 | 17 / 22 passed |
| int2_86636fe_bf1c | 86636fe | test_bf1.py part 3 | 17 passed |
| int2_86636fe_st1a | 86636fe | test_st1.py part 1 | 20 passed |
| int2_4695b61_st1b | 4695b61 | test_st1.py part 2 (SLURM 2251504) | 25 passed, 1 skipped |
| int2_4695b61_st1g0 | 4695b61 | test_st1.py Gate 0 | 1 passed |

  Skips are the full-DRYRUN tests (they need the run tree's own checkout). The tips differ only in a comment
  (86636fe) and a test expectation (4695b61). Owner receipts: bf1_4692273_t 56 passed; j1_55de6f9_t 28 passed
  1 skipped, j1_55de6f9_g0 37 passed.
- Fault found and fixed (test-only, 4695b61): test_st4_placement expected BF1's override on every keep INT arm; the
  keep arm without BF1 (L4PTFDGUDKWCDIN, cell intd) has none. The fault existed at a39aaa9.

## 4. Track 1, KV movement (BF owner)

| step | content |
|---|---|
| prediction (original) | W = B m (n/K) / r = 287 x 6.46583 / 10.72418 = 173.0381 ms, hidden under G = 232.48 ms; covered retained replay: ZERO ADDITIONAL VERIFIER FETCH |
| matched measurement | intc (BF off): J-probe gather span 90.464 ms per round for 2,022.975 MB of NEW host blocks (MEASURED; scope: probe rounds, non-additive, LEDGER 16.8). Live draft-selected share of the joint's NEW blocks c_req = 0.47-0.51 (MEASURED r7pa / r7pb at C128 / C73) |
| cause | ESTABLISHED (source trace producer -> transfer -> retained slot -> attention read): (1) "defer" dropped a job that landed after its layer; (2) nothing protected a retained slot from an earlier layer's miss before its own read; (3) the release point is already the draft layer (nothing earlier exists). INFERENCE from c_req: about half of the verifier's NEW blocks are never requested by a draft, so a perfect schedule still leaves 44-48 of the 90.5 ms (MODEL). UNKNOWN: why the original replay covered the verifier union while the live selection covers about half (definition of NEW vs the replay's union, or selection drift). The C++ pack / submit blocking call is still not established |
| fix | arm S: late "complete" (no job is dropped), retention with overflow fallback (keeps the keep's capacity proof), ready-only snapshots; the ledger splits (a) requested / completed / completed before deadline, (b) hit_cache + hit_retained, (c) miss_verifier_only / miss_req_incomplete / miss_lost; residual verifier transfer = (c) x 32,768 B, not added to W |
| measured effect | PENDING (GPU). Cell bf1c: arms X / B / R / S at C81 B287, 129 slots; STOPS until PRED_STAMP_BF1S is registered. Proposed bands (MODEL): S/X 0.985-1.05 (central x1.016), joint saving 5-45 ms, covered share 0.40-0.90, residual 900-1,600 MB per pass; REPORT miss_verifier_only / demand (MODEL 0.45-0.55) |
| memory at the anchor | MODEL, allocated basis: intc 76.82 GB already holds the X 129 store (record store_GB 2.4358); + BF parts 1.39 GB (2 KV heads) = about 78.2 GB: FITS. X 257: + 2.417 GB = about 80.6 GB: FITS (store size = human decision). Original equation: 83.7655 GB on its device-used basis |
| integration limit | the C81 arm "INT + keep + S" needs new mechanism: ST3's compiled keep fill (r2_st1_commit.py:582-613) runs before BF1's keep hook and has no block-mask input, so S's retention, ledger and address check would not run. Option without new mechanism (human decision): an INT keep arm without ST3 plus S (gives up ST3's 3-9 ms per round, MODEL). Side effect: in L4PTFDGUDKWCDI the merged ledger reports zero demand because ST3 hides it |

## 5. Track 2, shared KV reads (J owner)

| step | content |
|---|---|
| prediction (original) | U = 69.162946, I = 60.193917 (I/U = shared-read intersection, not a fetched share); E = td (1 - f)(U - I) / 63 = 6.9446 ms (td 68.9864, f 0.29290): incremental HBM service only. g(3B) 58.3740, D 126.3444, H 213.9039, G 232.4806 kept distinct |
| matched measurement | matched analogue of E: 3.52 ms (FLASHU1 - NS1R1 interpolated, 0.3927 ms per page x 8.969) = 0.51 x E (DERIVED, synthetic j1f shapes). The fork's 4-row tile costs 17.115 ms per pass, a term absent from the model. Physical HBM bytes of a live joint pass: UNKNOWN (not measured) |
| cause | PARTLY ESTABLISHED: the fork kernel (64-row tile, 255 registers, 2 programs per SM) is latency- and occupancy-bound, not byte-bound (microbench DRAM 1.02 x the union). Half of the E gap: the model prices a page at td (1 - f) / 63, about twice its attention-read cost (INFERENCE). The other half: UNKNOWN |
| fix | bit-preserving only: FORKA1-3 load-ahead levels (-6.9 to +4.6 ms per pass, MODEL; GPU decides) and the union build outside the kernel (<= -1.2 ms). Every library alternative changes the bits or does not fit our 64-token pages |
| measured effect | PENDING (GPU). Cell j1n at C81 B287 READY: rows sweep RSM1-4 bit-gated vs FORKT, RSQ1/2/4, ncu per arm (DRAM bytes, L2 sectors, occupancy) |
| overlap test | the X1 screen (MEASURED) found no overlap of the tested fixed-shape dense + fork stream combinations; the copy engine overlaps (0.8-3.4 %). If the 4-row tile is fully serial, G 232.48 -> 251.08 ms (MODEL); with E replaced by 3.52, G 228.76 (MODEL). Times are not added where the model overlaps them |

## 6. Re-optimization with the updated costs (REOPT_2026-11-02.md; script s3_reopt.py, output s3_reopt_v2.out)

Method: the original batch sweep (memory first on our allocated basis, discard over the strict edge 83.15 GB, then
T and Q_committed = 1,000 B Y / T, ratio-of-sums, one population).

| method | best feasible | memory GB | Q_committed mean, tok/s | x SHJ1 |
|---|---|---|---|---|
| INT | C128 B216 | 83.909 (ran) | 1,910.0 MEASURED | 0.582 |
| INT + S | C81 B311 | 83.13 | 1,774.6 [1,703.8, 1,832.5] MODEL | 0.541 |
| INT + J | C128 B216 | 83.909 | 1,918.8 [1,910.0, 1,950.4] MODEL | 0.585 |
| baseline SHJ1 | C128 B216 | SH 82.042; SHJ1 not recorded (very probably fits) | 3,282.5 MEASURED | 1.000 |

Flip thresholds (s3_reopt_v2.out section 7, DERIVED): C81 + S passes C128 INT only if S saves >= 40.2 ms per round at
B311 (BF MODEL best 22.6). C128 INT ties SHJ1 only at T <= 173.9 ms (measured 298.8). The largest fitting B wins at every
C. Open candidate: no spec-arm data exists at C96 / C113.

## 7. Human decisions and external dependencies

- GPU allocation (the only external dependency for every measured effect above): DRYRUN + bf1m, then bf1c; j1n; an
  INT cell at C96 or C113.
- Store size beyond 129 slots (X 257 fits on the MODEL at the anchor).
- C81 INT keep arm without ST3 plus S (no new mechanism; loses ST3's saving), or a bit-preserving block-mask input to
  ST3's compiled keep fill.
- The FAST tie rule (target correctness); the verify-K-1 restart; bit-changing kernel routes; the NUM tolerance
  (unchanged, 1e-3).
- An algorithm change that lets the verifier rows use the drafts' selections (raises c_req above 0.5); it changes the
  bits and is not built.
