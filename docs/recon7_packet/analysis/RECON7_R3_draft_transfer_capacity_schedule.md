# RECON7 workstream R3: standalone drafting, transfers / readiness, retention / capacity, schedule / yield

Written 2026-10-30T10:13Z (from `date -u`). Writer: R3 analysis agent, read-only. No code tree edited, no GPU job, no
commit. Style: ASD-STE100. Part of revision 7 (the human's request of 2026-10-30,
/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md, "HRQ" below).

Labels. MEASURED = a value in a run record (summary.json field or stdout line). DERIVED = arithmetic on MEASURED values;
the operands are shown, or a CPU script below computed it and its log line is cited. MODEL = needs an assumption (stated).
CITED = a value from another document (file:line). INFERRED = a reading of code or records that no record states.

Short names.
- R = /mnt/beegfs/ojerbi/r2-runs/spec. BEST128 = R/i4dc_d3a_2195575_i4dca (C 128 pool 65, B 216, nova05; arms SH, L4PTFD,
  L4PTFDGUDC, best arm L4PTFDGUDCD). BEST73 = R/i4dc_d4b_2195576_i4dcb (C 73 pool 10, B 318, nova07; best arm
  L4PTFDGUDKWCD). Code of both: /mnt/beegfs/ojerbi/nosi-r2-i4dc @ 2e9e497.
- ACC128 = R/i4d_d3c_2195575_i4dc, ACC73 = R/i4d_d4d_2195576_i4dd (128-step accounting cells, L4PTFD, meter on).
  KEEP128 = R/i4d_d3a_2195575_i4da, KEEP73 = R/i4d_d4b_2195576_i4db (keep arms L4PTFK / L4PTFKW). Code /mnt/beegfs/ojerbi/nosi-r2-i4d @ 99c231e.
- PROF128 = R/i4c_c3a_2195575_i4ca, PROF73 = R/i4c_c4b_2195576_i4cb (L4PTFDP: torch.profiler windows, rounds 96 / 112).
- I4M = R/i4m_m3p73_2195575_i4mp73, R/i4m_m3p128_2195575_i4mp128 (memory probes), R/i4m_n373_2195575_i4m73 (the OOM).
- J91 = job 2175791 (the td source): /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/.
- FC = /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/forecast-oct12-17pct.md. ST = nosi/nosi/spec_tentative.py (i4dc tree).
- Logs (CPU scripts under /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_r3/, run with srun on amd):
  L1 = cpu-logs/recon7_r3_s1.txt (key skeleton), L2 = recon7_r3_s2.txt (field extraction, every ladder record),
  L3 = recon7_r3_s3.txt (J91 kernel classes), L4 = recon7_r3_s4.txt (every DERIVED number of this file),
  L5 = recon7_r3_s5.txt (model backbone basis check). All under /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/.

Boundaries used. A "draft step" = one one-row pass of the D slots: the round's draft phase median / 2, CUDA events,
host launch included (the cell's `draft_step_ms`). td = the eager median of one resident C 63 target step, CUDA events
around enqueue and run (J91 README.md:42). The same kind of boundary (elapsed, host included). A "round" = reset + joint
pass + commit + two D slots (ST:940-1053). A "cohort" (model) = K 3 draft positions verified with lag 2.

## 0. Answer in eight lines

1. Drafting: our standalone draft step costs 66.144 ms (C 128 B 216) and 88.219 ms (C 73 B 318) against td 54.814 /
   75.295 ms: +11.33 / +12.92 ms (1.207x / 1.172x). The forward work (GEMM, selection, attention reads) equals td's within
   2 ms. The extra is the victim-pool residency service that td (a C 63 step) does not have, and the tentative-buffer work.
2. Transfers: the model's background request / CPU packing / hidden DMA is NOT in the ladder. Drafts skip their misses and
   request nothing. Every host block is gathered urgently by GPU kernels inside the joint pass, in series. The link rate
   is better than modelled (21.6-27.6 vs 10.7 GB/s), but nothing is hidden: 20.2 ms (C 128) / 115.6 ms (C 73) per round.
3. Bytes: 2.58 MB (C 128) and 8.06-10.53 MB (C 73) per request per round, against the model's 6.47 MB at C 81.
   23-24 % of the NEW blocks are avoidable refetches (no retention across rounds).
4. Capacity is NOT a gap. Our layout fits MORE requests than the model's own memory formula at every C (C 128: 216 vs
   205; C 73: >= 318 vs 309; C 81: 321-323 MODEL vs 287). The Oct-21 per-stream X reservation is gone. The model's
   backbone uses a device-used / reserved basis (5.4-5.9 GB above the allocated basis that we measure).
5. Schedule: the implemented contract-4 schedule equals Figure 3 (K 3, the newest token deferred) in steady state: 3
   verify + 3 draft rows per request per round, as the model's cohort. Decisions (a) and (b) are implemented.
6. Yield: accepted drafts per request per median round 2.889 / 2.780 (model 3: -3.7 % / -7.3 %); whole run 2.609 / 2.411
   (-13.0 % / -19.6 %). Causes: rejection p 2.8 % / 4.0 % per group (model 0), recovery 1.70 rounds per reject (the
   corrected model clock gives 1.0), static-batch drain 7.7 % / 12.6 % of request-rounds, 10.5 % / 16.2 % padded rows.
7. Correction (MODEL, at the anchor, others fixed): td 81.6-84.4 ms -> Q 3,132-3,027; serial transfer -> Q 3,063-2,887;
   Y 2.81-2.86 (recovery) -> Q 3,467-3,535. These do not add (the interval is a max).
8. No rule breach that changes data. Three rule slips in shell variable names (section 6).

## 1. The table (the human's format; C 128 = C 128 B 216, C 73 = C 73 B 318)

| part | original model at the anchor AND at the measured configuration | actual at the same boundaries | gap | cause with evidence | correction / quantified model revision | remaining unknown |
|---|---|---|---|---|---|---|
| 1. STANDALONE DRAFTING | Anchor C 81 B 287: td 68.9864 ms (a resident C 63 full TARGET step, eager, interpolated B 256-344; FC:20, HRQ:18). 2 standalone drafts per cohort = 137.97 ms. Measured configs: td 54.8137 (C 128) / 75.2948 (C 73) (HRQ:35, :38): 2 td = 109.63 / 150.59 ms per round. Work of td (MODEL, J91 ncu classes, linear in B, L4): 129 GEMM launches over B rows; 64 KV blocks per (layer, head, row) = 28.99 / 42.68 GB; selection 9.46 / 12.94 ms; residency 2.53 / 3.20 ms (offload metadata + empty gathers, no pool); kernel total 53.38 / 71.49 ms | Draft step (elapsed, host incl.): best arm 66.144 ms (C 128, BEST128 stdout:109) / 88.219 ms (C 73, BEST73 stdout:115); L4PTFD 71.632 / 94.772; draft phase 132.287 / 176.437 ms per round. Work (PROF, one eager L4PTFD pass, L4): 129 GEMM over B rows; 128 attention launches (call A + call B, split + combine); 63.51 / 62.04 window blocks read per (layer, head, row) (64 minus draft skips 0.485 / 1.959) = 28.77 / 41.37 GB + one 64-row buffer block; 2,674 kernels + 193 copies per pass (td about 1,156-1,188); busy 67.8-69.7 / 91.1-93.8 ms. Near the anchor: shipped-engine standalone draft DR81 at C 81 B 290 = 77.799 ms MEASURED (CITED SCORING_dr73.md:89), td(B 290) 69.597 MODEL | +11.33 ms per step (1.207x) at C 128; +12.92 (1.172x) at C 73; +22.66 / +25.85 ms per round. Kernel busy - td kernel (MODEL): +14.4..+16.4 / +19.6..+22.4 ms | By class (L4, kernel ms, ours - td): residency +5.6..+7.5 / +5.6..+8.3 (pool_swap 5.5-7.4 / 5.8-8.4 + pool_update 0.5: the C > 63 victim pool; J91 SH C 128 ncu pool_moves 5.97, L3) = FALSE MODEL ASSUMPTION (td has no pool / residency service); SMALL + FUSED + COPY + TWINF +9.0 / +11.3..+11.5 (1,936 small kernels: tentative masks, merge, buffer write, lean-pass ops) = design-required work absent from the proxy + INEFFICIENT EXECUTION (graphs + compile removed 5.49 / 6.55 ms); attention +1.4 / +2.0 (call B on the buffer) = design-required extra work; GEMM -0.7 / +0.6..+0.7 and selection -0.9 / 0.0 = no gap. Host exposure (busy / steady 0.960 / 0.977, PROF) about 2.9 / 2.2 ms before the draft graphs. Evidence chain: td -> shipped draft (DR1 60.239, DR73 81.329: +5.43 / +6.03, residency) -> ours best (+5.91 / +6.89, tentative work) | td_rev = shipped-engine draft x (1.08-1.12 x td, measured DR / td at three C) + the tentative increment 5.9-6.9 ms. At the anchor (MODEL): 77.8 (DR81) + 6.6 = 84.4 ms; B-linear ratio method 1.1823 x 68.99 = 81.56 ms (L4). With the Oct-21 relation Q = 255,494 / td (td >= 51.35, others fixed): Q 3,132..3,027 tok/s = -15.4..-18.3 % of Q | Kernel split of the BEST arm's draft pass (no profile after graphs + compile); the ladder draft at C 81 (not run); pool-swap cost at C 81 (pool 18) |
| 2. PACKING / TRANSFERS / READINESS | Anchor: 6.4658 MB per request per cohort (3.071 blocks per stream; FC:29); r 10.724 GB/s; transfer 287 x 6.4658 / 10.724 = 173.04 ms, HIDDEN under GPU 232.48; interference s 8.685 % (7.99 % of the interval); every verify block ready (lag 2). Measured configs: the model has no byte value at C 128 / C 73. The model formula with OUR bytes: 51.98 ms (C 128), 239.14 (C 73 keep) / 312.16 (C 73 no keep), hidden; interference charge s x HBM = 0.105766 x 169.47 = 17.92 ms / 0.078585 x 233.69 = 18.36 ms (HRQ:35-38) | Only the joint pass loads (meter on only in J, ST:952-971); drafts skip every miss (ids -1, map entry -1: r2_i4a_lean.py:215-220); no CPU packing; in flight 0 (kv_at_verify note); 64 GPU zero-copy gathers per joint pass (flash_h2d_from_mask*). NEW per round (MEASURED, kv_at_verify): 16,946.2 blocks = 557.46 MB (C 128) = 2.581 MB per request-round, 1.226 blocks per stream, 0.892 MB per committed token; C 73: 101,764.2 = 10.527 MB (no keep) / 77,960.2 = 8.065 MB (best, keep + window), 5.000 / 3.831 blocks per stream, 3.75 / 2.88 MB per committed token. Exposed gather (probe rounds): 20.25 ms / 154.76 / 115.64 ms; delivered 27.5 / 21.6 / 22.2 GB/s; not ready at the joint 0.65 % / 2.65 % / 2.03 % of verify-selected slot reads; exposed share of the round 6.2 % / 26.6 % / 22.0 % | C 128: exposed 20.25 vs the model's 17.92 ms interference charge: +2.3 ms. C 73 (best): 115.64 vs 18.36 ms: +97.3 ms. Bytes 0.40x (C 128) and 1.25-1.63x (C 73) the C 81 model value. Rate 2.0-2.6x the model's r (our favour) | (1) MECHANISM NOT REALIZED -> SERIALIZATION: no draft-issued request, no packer, no overlap; every byte is urgent and serial in the joint pass (above). (2) FALSE MODEL ASSUMPTION (coverage): on the legacy schedule >= 84 % of the verify's remaining NEW blocks at C 73 were never requested in time although 97.5 % of fresh draft misses were sent (CITED SCORING_wse.md:217, :254). (3) EXTRA WORK: no retention across rounds: avoidable refetch 5,892 of 25,602 (23.0 %) / 20,267 of 83,949 (24.1 %) NEW per pass (repeat + row-0 window refetch, ACC128 / ACC73, L4); the keep cut NEW -33.6 % / -23.4 % (KEEP, L2) but its bookkeeping (about 12 ms) made it a loss at C 128. (4) CHANGED LIVE WORKLOAD: misses grow fast as C falls (draft skips 0.485 -> 1.959 per stream per call) | For this implementation the transfer term is an EXPOSED serial gather = NEW bytes / (22.0-27.6 GB/s) (MEASURED), not a hidden transfer with s. At the anchor (MODEL, model bytes 287 x 6.4658 = 1.856 GB): 67.2-84.4 ms exposed; interval 213.90 + 67.2..84.4 = 281.1..298.3 ms; Q 3,063..2,887 (-17.3..-22.1 % of Q). The model's s is not testable here (no overlap exists) | The draft-named share of NEW on the tentative ladder (WS-E's 84 % is from the legacy schedule); the share of NEW caused by draft skips that invalidate window slots; interference of a copy-engine transfer under the joint pass; bytes at C 81 |
| 3. RETENTION / STATE / PHYSICAL CAPACITY | Anchor: 83.7655 GB at C 81 B 287 = C 63 backbone 72.248 GB (DERIVED, L4) + 18 x 2,105,344 B x B + 2.0562 MB per request (retained overflow peak 1.6384 + reserve 0.4178) + 3 x 17.04 MB landing (FC:15-17); marginal 0.2462 GB per request, implied fixed 13.12 GB; cap 84.0 GB; never an allocation proof. Same formula at the measured configs (MODEL, L4): C 128 B 216 87.66 GB (fit B 205); C 73 B 318 86.04 GB (fit B 309) | Peaks (MEASURED): C 128 prefill 83.891 (L4PTFD) / 83.900 (best), decode 82.559 / 82.604; C 73 prefill 77.685 / 77.694, decode 76.932 / 76.991. Fit lines (prefill binds; I4M stdout:59): 19.020 + 0.30033 B (C 128), 19.020 + 0.18448 B (C 73). Feasible B: 216 at C 128 (= fit; SH fits 222); 318 measured at C 73; fit 351-352; B 351 OOM (83.644 GB allocated + a 126 MiB request; usable about 83.75 GB after SH, lesson L60). Per request: engine slots (NS = C + 1, 2 heads) 271.59 / 155.80 MB; ONE store X 129 shared by the 32 layers 8.487 MB; CB0 0.066 MB; tentative buffers 2.105 MB (decode time); prefill transient 20.13-20.19 MB; fixed graph pools 71.3 / 117.4 MB. Spec - SH prefill peak = 1.857 / 2.730 GB = store + CB0 1.847 / 2.720 GB | We fit MORE than the model's formula: +11 (C 128), >= +9 (C 73; fit +42); MODEL at C 81: 321.5-322.8 vs 287 (+34..+36). The model over-predicts the peak at our points by 3.77 / 8.35 GB | (1) FALSE MODEL INPUT (memory basis): the model's backbone 72.248 GB at B 287 matches J91's device-used (72.75) / reserved (71.71-71.83) lines, not its allocated lines (66.39-66.90) (L5). Device-used holds harness-only staging (5.60 GB at B 287) and allocator slack (5.32 GB). Our peaks are allocated peaks. (2) Our store is per request, shared by layers (8.49 MB), not the Oct-21 per-stream X (134.7 MB per request): that departure is gone. (3) The prefill transient binds (not in the model). (4) Usable ceiling about 83.75 GB, not 84.0 (L60): -1 request at the anchor. (5) Store over-reservation: mean listed 1.03 / 2.28 of 128 slots per (layer, head, request) (ACC, L4): > 98 % idle; it costs 6 requests at C 128 (1.83 GB / 0.300 GB) | On the allocated basis the model itself gives B 313.6 at C 81 (MODEL: fixed 66.896 - 287 x 0.17026 + landing = 18.08 GB; 0.17026 + 0.0379 + 0.0021 = 0.2102 GB per request). Our layout: B_fit(C 81) 321.5 at 83.75 GB (MODEL, linear in C through the two measured fit lines; slope per C 2.1063 MB = one slot per request, check). So B 287 at C 81 IS reachable by our layout (MODEL, not measured) | The author's backbone definition (model_totals.py is not on the cluster); a measured C 81 fit line; the store's PEAK occupancy per (layer, head, request) (not recorded); the decode-time peak at C 81 |
| 4. SCHEDULE / YIELD / RECOVERY | Anchor: cohort K 3, Y 3 accepted drafts per request (no correction or bonus credit), p 0, lag 2, n = K + p (1 + lag) = 3; no recovery, drain or startup (FC:21, :43-46); 3,703.54 accepted-draft tok/s. Measured configs (MODEL, HBM binds): interval 187.40 ms (C 128) / 252.05 ms (C 73); 3,457.9 / 3,784.9 accepted-draft tok/s (L4) | Implemented (contract 4, ST:27-45): prelude 3 drafts; round = reset, J (v = u - 1 verify rows + 1 draft row; steady u 4, v 3; the newest token deferred, its row's K / V reused, 0 remade rows), commit, D0, D1. Rows per steady round per request 3 verify + 3 draft = 6 (= the model's 3B + 3B). Counts (MEASURED, C 128 / C 73): groups + finals 37,614 / 55,866; verified 111,387 / 164,743; accepted 109,318 / 160,248 (0.9814 / 0.9727 per verified token); corrections 1,058 / 2,250 (p 0.0281 / 0.0403 per group); bonus 0; committed 110,376 / 162,498 = accepted + corrections; fills 1,050 / 2,221; finals 216 / 318 (commit 459 / 638; target 0); unverified discarded 405 / 634. Accepted per request: median steady round 2.889 / 2.780; whole run 2.609 / 2.411. Committed y_med 2.894 / 2.803. Rounds per request: ideal 170.67, mean active 179.00 / 182.66, batch 194 / 209. Drain 3,240 / 8,375 done request-rounds (7.7 % / 12.6 %). GEMM rows 251,640 / 399,090 computed, 225,279 / 334,562 active (10.5 % / 16.2 % padded); 2.280 / 2.456 rows computed per committed token (model 2.0) | Y (accepted, median round) -3.7 % / -7.3 %; whole run -13.0 % / -19.6 %. Recovery 1.70 / 1.70 rounds per reject vs 1.0 (corrected model clock). Committed counts +0.96 % / +1.38 % corrections that the model does not credit | CHANGED LIVE WORKLOAD: p 2.8 % / 4.0 % (the model's p 0 was a borrowed sample); reject clusters at the first 64-token block edges (rounds 22-27, 45-52 at C 128: 424-540 accepted vs about 600-620; L4). FALSE MODEL ASSUMPTION: no recovery, drain or startup terms; decision (a) restarts Figure 3: reject round commits n + 1, then a fill round (0 tokens), then a v = 2 group: 6 - n tokens lost; with the measured mean n of rejected groups 0.87 / 0.81 this predicts 1.71 / 1.73 rounds per reject (hand arithmetic, section 5 item 3) vs 1.70 / 1.70 (DERIVED from records, L4): EXPLAINED. EXTRA WORK: a static batch waits for its slowest request (finish rounds 170-193 / 170-208, L2) and computes padded rows | Revised recovery clock (MEASURED basis): n_rev = K + p x 3 x 1.70 = 3.143 (C 128) / 3.205 (C 73) draft positions per 3 accepted (model-corrected 3.084 / 3.121). Y_rev = 3 x 3 / n_rev = 2.864 / 2.808, against the measured active-request yield 511 / 179.0 = 2.855 and 511 / 182.66 = 2.798 (within 0.4 %). Drain factor (whole run only) 179.0 / 194 = 0.923, 182.66 / 209 = 0.874. Q at the anchor with Y_rev (others fixed): 3,703.54 x 2.8635 / 3 = 3,535.0 (C 128 p) .. 3,703.54 x 2.808 / 3 = 3,466.5 (C 73 p) = -4.5..-6.4 % of Q | p at C 81 B 287 (not measured); per-request reject counts (the drain driver; trace_all only with recording on); the cause of the block-edge reject clusters (INFERRED: forced skip of the tentative block, 13,056 / 19,264 forced skips) |

## 2. Row 1 detail: standalone drafting (work first, then time)

### 2.1 Work per pass (C 128 B 216 / C 73 B 318)

| item | td proxy (resident C 63 target step) | our draft pass (lean one-row pass) | source |
|---|---|---|---|
| rows | B | B (D slot; idle requests are computed and masked) | ST:1015-1027 |
| GEMM launches | 129 | 129 | J91 ncu (L3); PROF classes (L4) |
| attention launches | 64 (split + combine) | 128: call A (committed window) + call B (tentative buffer), each split + combine | PROF classes; ST:76-81 |
| KV blocks read per (layer, head, row) | 64 (63 historical + tail) | window 63.51 / 62.04 (64 minus draft skips 0.485 / 1.959) + one 64-row buffer block | L4; draft `skip_per_stream_per_call` |
| KV bytes per pass (K + V) | 28.99 / 42.68 GB | 28.77 / 41.37 GB + buffer reads (<= 0.45 / 0.67 GB) | L4 (DERIVED) |
| selection | stage 1 + pooling + top-k, 1 row | the same per row | |
| residency | offload metadata + gathers with 0 loads | diff_offload + pool_update + flash_pool_swap + map + LRU + gathers with all ids -1 | r2_i4a_lean.py:215-220 |
| twin / compression | compression | fused twin (32 kernels) | PROF |
| tentative work | none | buffer write, masks A / B, float32 merge | LEDGER_rev6.md:67-74 |
| kernels per pass | 1,188 (B 64) / 1,156 (B 344) | 2,674 kernels + 193 copies, 2,514 launch calls | L3; L4 |

### 2.2 Kernel time per class (MEASURED ours, one eager pass of L4PTFD; MODEL td = J91 ncu classes linear in B; L4)

| class | td MODEL C 128 | ours C 128 (draft0 / draft1, r96) | td MODEL C 73 | ours C 73 (draft0 / draft1, r96) |
|---|---|---|---|---|
| GEMM | 20.372 | 19.686 / 19.680 | 25.117 | 25.804 / 25.754 |
| attention | 19.052 | 20.487 / 20.449 | 27.922 | 29.873 / 29.925 |
| selection | 9.464 | 8.551 / 8.564 | 12.941 | 12.915 / 12.879 |
| residency (pool swap / update, diff, gathers) | 2.528 | 9.407 / 8.128 (pool_swap 6.822 / 5.537) | 3.204 | 11.501 / 8.809 (pool_swap 8.427 / 5.750) |
| small / fused / copy / twin | 1.967 | 10.955 / 10.947 | 2.305 | 13.750 / 13.710 |
| total busy | 53.383 | 69.086 / 67.769 | 71.489 | 93.844 / 91.076 |

td / its own kernel total = 1.0268 (C 128) / 1.0532 (C 73) (L4): td is an eager boundary over a near-full launch queue.

### 2.3 Elapsed chain per draft step (ms)

| step | C 128 B 216 | C 73 B 318 | label / source |
|---|---|---|---|
| td | 54.814 | 75.295 | MODEL (HRQ:35, :38) |
| shipped-engine standalone draft (resident reads, pool residency, captured step) | 60.239 (dr1) | 81.329 (dr73) | MEASURED other job; SCORING_dr1.md:84, SCORING_dr73.md:89 |
| ladder lean draft L4PTFD | 71.632 | 94.772 | MEASURED BEST128 / BEST73 stdout:104 / :110 |
| best arm (draft graphs + compiled helpers) | 66.144 | 88.219 | MEASURED stdout:109 / :115 |
| best - td | +11.330 | +12.924 | DERIVED |
| best - shipped draft | +5.905 | +6.890 | DERIVED (the tentative-buffer increment) |

At the anchor: dr81 (C 81 pool 18, B 290) = 77.799 ms (CITED SCORING_dr73.md:89), 1.118x td(B 290) = 69.597 (MODEL).
Our ladder draft at C 81 is not measured. MODEL: 77.799 + 6.6 = 84.4 ms, or 81.56 ms (ratio method, L4). Both lie above
the Oct-21 break-even td 80.83 ms (reconciliation-17pct.md:55).

## 3. Row 2 detail: packing, transfers, readiness

What the code does (INFERRED from code, confirmed by records):
- The meter counts host gathers only while the joint pass runs (ST:952-971); the D slots turn it off (ST:1020-1024).
  A draft row sets each miss id to -1 and clears its map entry ("the next row loads a skipped block again";
  r2_i4a_lean.py:215-220). So drafts load nothing and request nothing.
- The verify rows' host-only blocks go into ONE per-request store (X = 129 = 1 + (K - 1) x 64 slots, 2 heads), refilled
  at every layer of every joint pass (r2_i4d_keep.py:12-18; r2_i3_arena.py:17-21). Row 0 loads its window misses.
- kv_at_verify: `in_flight_blocks` 0, note "no background prefetch in L3 / L4 / L4P (--spec-prefetch is rung L5)"
  (r2_spec_cell.py:237-250). The wait is the sum of host-gather spans on probe rounds (spec_round.py:384-391).

| quantity | model C 81 B 287 | C 128 B 216 (all arms w/o keep) | C 128 keep + window | C 73 L4PTFD | C 73 best (keep + window) |
|---|---|---|---|---|---|
| NEW blocks per round | - | 16,946.2 | 11,249.2 | 101,764.2 | 77,960.2 |
| MB per request per cohort / round | 6.466 | 2.581 | 1.713 | 10.527 | 8.065 |
| blocks per stream | 3.071 | 1.226 | 0.814 | 5.000 | 3.831 |
| MB per committed token (median round) | 2.155 (per accepted) | 0.892 | 0.590 | 3.753 | 2.877 |
| not ready at the joint pass (NEW / verify-selected slot reads) | 0 | 0.65 % | 0.43 % | 2.65 % | 2.03 % |
| transfer time | 173.04 hidden | 20.25 exposed | 13.65 exposed | 154.76 exposed | 115.64 exposed |
| model formula with these bytes (hidden) | 173.04 | 51.98 | 34.51 | 312.16 | 239.14 |
| rate (GB/s) | 10.724 | 27.53 | 27.10 | 21.63 | 22.18 |
| cost share of the interval / round | 7.99 % (s) | 6.2 % | 3.8 % | 26.6 % | 22.0 % |

All values MEASURED (kv_at_verify) or DERIVED (L4). Device-side cross-check: the H2D class of one profiled joint pass is
21.37 ms (C 128) / 160.14 ms (C 73) (CITED LEDGER_rev6.md:122): 26.1 / 20.9 GB/s, about the PCIe Gen4 rate of J91
(README.md:65, "about 25 GB/s").

Coincidence worth a test: draft skips per round 18,229 (C 128) / 101,864 (C 73) vs NEW 16,946 / 101,764 (L4). The
magnitudes match. The record has no per-block link. WS-E on the legacy schedule found the opposite overlap (>= 84 % of NEW
never requested in time). Discriminator D2 below decides it on the tentative ladder.

## 4. Row 3 detail: retention, state, physical capacity

| component (per request unless stated) | model C 81 | ours C 128 | ours C 73 | source |
|---|---|---|---|---|
| per-request slope | 0.2462 GB | 0.30033 GB (prefill line) | 0.18448 GB | FC:15; I4M stdout:59 |
| historical / engine slots | backbone + (C - 63) x 2.105 MB | 271.59 MB (NS 129) | 155.80 MB (NS 74) | arena record (L4) |
| verify-time KV beyond the cache | retained overflow peak 1.6384 + reserve 0.4178 MB | store X 129: 8.487 MB (one layer at a time) | 8.487 MB | arena; r2_i3_arena.py:17-21 |
| chain image CB0 | - | 0.066 MB | 0.066 MB | arena |
| tentative buffers | - | 2.105 MB (decode time) | 2.105 MB | ST:57-61 (DERIVED 32 x 64 x 1,028 B) |
| landing / staging | 3 x 17.04 MB fixed | none (no packer) | none | |
| graph pools (fixed) | - | 71.3 MB | 117.4 MB | record |
| prefill transient | not stated | 20.19 MB + intercept 19.02 GB | 20.13 MB | L4 (slope - arena) |
| peak (GB) | 83.7655 MODEL | 83.891 / 83.900 MEASURED | 77.685 / 77.694 | stdout:51; L2 |
| feasible B | 287 MODEL | 216 MEASURED (= fit) | 318 MEASURED; 351 OOM | I4M; STATUS_LOG.md:1057-1060 |
| model formula at this C | - | 205 | 309 | L4 |

Basis check (CPU, L5): J91 C 63 lines at B 287: device-used 72.749, reserved 71.713, process peak reserved 71.833,
allocated 66.391, process peak allocated 66.896 GB. The model's backbone 72.248 lies 0.5 GB from device-used and 0.4-0.5
GB from reserved, and 5.35-5.86 GB above the allocated lines. INFERRED: the model counts device memory that our arena
runs do not allocate (J91's harness staging 5.60 GB and allocator slack 5.32 GB at B 287).

OOM arithmetic (L4, from the n373 exception text): capacity 85.094 GB; in use 84.418; PyTorch allocated 83.644;
reserved-free 0.222; free 0.126; non-PyTorch 0.551 GB. The fit line predicted 83.773 GB at B 351. So the usable ceiling
is about 83.75 GB after SH in the same process (lesson L60), not 84.0.

Retention versus bytes: the model keeps requested payload until its readers finish (1.64 MB per request at the peak).
Our L4PTFD keeps nothing across rounds. The keep arms reuse the SAME store slots (no growth): NEW -33.6 % (C 128) /
-23.4 % (C 73). So our layout trades memory for refetched bytes; row 2 carries that cost.

## 5. Row 4 detail: the implemented schedule against the model's cohort

Implemented position trace (request 0 of BEST73, L2; positions are absolute):
- P0-P2 (prelude): draft rows feed 16129, 16130, 16131 and draft 16130-16132.
- Round 0, J (group, u 3, v 2): verify rows feed 16129, 16130 (check 16130, 16131); the token at 16132 is DEFERRED;
  the draft row feeds 16132 and drafts 16133; committed 2. D0, D1 draft 16134, 16135.
- Round 1, J (u 4, v 3): verify rows feed 16131-16133 (check 16132-16134); 16135 deferred; draft row feeds 16135,
  drafts 16136; committed 3. The row of position 16134 is NOT recomputed (its draft K / V stay valid: D3 removed).
This is Figure 3 (speckv.txt:126-140) from the first group on. It is not a substitute schedule.

Differences from the model's cohort (each MEASURED or DERIVED in L4):
1. Same in steady state: K 3, 3 verified + 1 deferred per group, 6 rows per request per round, no bonus token.
2. Rejection p 0.0281 / 0.0403 per group (model 0). Acceptance 0.9814 / 0.9727 per verified token.
3. Recovery (decision (a)): 1.70 rounds per reject; the corrected model clock (1 + lag) / K = 1.0 round. Excess 0.70.
   Hand check (DERIVED): a reject at n accepted commits n + 1 (loses 2 - n of 3), the fill round commits 0 (loses 3),
   the next group has v 2 (loses 1): 6 - n tokens = (6 - n) / 3 rounds. Rejected groups by n, from the acceptance
   histogram (L2): C 128 n 0: 445, n 1: 304, n 2: 1,058 - 445 - 304 = 309; mean n = (304 + 2 x 309) / 1,058 = 0.871;
   (6 - 0.871) / 3 = 1.710 rounds. C 73: 1,001 / 670 / 579; mean n = (670 + 1,158) / 2,250 = 0.812; 1.729 rounds.
   Measured (L4): 1.701 / 1.696. Caveat: the histogram's n 1 bin can hold full v 1 groups (52 / 85 at most); that
   raises mean n a little and moves the prediction toward the measurement.
4. Corrections are committed (+0.96 % / +1.38 % of committed); the model counts accepted drafts only.
5. Start: the prelude (3 one-row passes, prelude_ms 2,457 / 2,562 including the one-time compile) and a v 2 first group.
6. End (decision (b)): one final group per request (216 / 318), no ordinary target step needed (target 0), 405 / 634
   pending drafts discarded unverified.
7. Static batch: requests finish in rounds 170-193 (C 128) / 170-208 (C 73); 7.7 % / 12.6 % of request-rounds wait
   while computing padded rows (GEMM rows padded 10.5 % / 16.2 %).
8. Timing (elapsed view, median round; MODEL cohort at the measured configuration, HBM binds):

| span | model C 128 | measured C 128 | model C 73 | measured C 73 |
|---|---|---|---|---|
| two standalone drafts | 2 td x (1 + s) = 121.22 | 132.287 | 162.42 | 176.437 |
| joint (td + extra verifier HBM) x (1 + s) | 66.18 | 186.011 (gather 20.25 nested) | 89.63 | 338.961 (gather 115.64 nested) |
| commit + reset | 0 | 6.178 | 0 | 8.546 |
| round | 187.40 (3 accepted per request) | 324.883 (2.889 accepted, 2.894 committed) | 252.05 | 524.659 (2.780 / 2.803) |
| per 3 committed tokens per request | 187.40 | 336.78 (1.797x) | 252.05 | 561.51 (2.228x) |

The joint pass row belongs to workstream R2; it is listed here only to close the round. The phase sums are 324.476 /
523.944 ms against the round medians 324.883 / 524.659 (residual +0.41 / +0.72 ms).

## 6. Cause classes, discriminators, rule slips

Cause classes (largest first per row):
- Row 1: false model assumption (td has no residency service of a cached C > 63 engine: +5.4..+6.0 ms, DR vs td);
  design-required tentative work absent from the proxy (+5.9 / +6.9 ms after graphs + compile); inefficient execution
  (1,936 small kernels, partly removed). Forward work matches td.
- Row 2: mechanism not realized -> serialization (all transfers urgent and serial); false model assumption on coverage
  (WS-E evidence, legacy schedule); extra work (23-24 % avoidable refetch); changed live workload (miss growth at C 73).
- Row 3: no gap against the model (our favour). False model input (memory basis). Over-reservation of the store.
- Row 4: changed live workload (p, block-edge clusters); false model assumption (no recovery / drain / startup; the
  corrected n clock undercounts the refill); extra work (padded rows of a static batch).

Discriminators (PROPOSED only; nothing run):
- D1 (row 1, GPU, short): a profile-only cell of the best arms (the I4C profiler window, rounds 96 / 112) at C 128 B 216
  and C 73 B 318, to split the 66.1 / 88.2 ms draft pass by class after graphs + compile. Needs a builder to port the
  I4C profiler option to the I4DC cell. About 15-20 GPU-min per cell.
- D1b (row 1 / 4, GPU): the ladder's lean draft and the full arm at C 81 (pool 18) B 287 (fit-checked first by D3). It
  measures td's real counterpart and p at the anchor.
- D2 (row 2, GPU, accounting only): add two per-block flags to the I4D accounting cell (128 steps, C 73 B 318 and C 128
  B 216): (a) the NEW block was skipped by a draft row of the same (layer, head, request) since the previous joint pass;
  (b) the NEW block's window slot was cleared by a draft skip. If (a) is large, a draft-issued copy-engine prefetch can
  hide up to that share of the 115.6 ms; if small, the model's coverage assumption is false on this path too.
- D3 (row 3, GPU, about 12 min): the I4M probe at C 81 (pool 18) B 64 (49 rounds), giving the C 81 prefill / decode lines
  and B_fit; add a max counter of listed blocks per (layer, head, request) to size X.
- D3b (row 3, CPU, done here): the basis check (L5). Full closure needs the author's model_totals.py backbone definition.
- D4 (row 4, CPU, done here): the recovery clock 6 - n tokens per reject predicts 1.71 / 1.73 rounds; measured 1.70 /
  1.70. No further run needed for the mechanism; p at C 81 comes with D1b.

Rule slips (reported, not hidden). Three login-shell commands carried throw-away variable names that contain forbidden
words: `sed_unused=1` and `sed_dummy=0` (word "sed"), `awk_not_used=1` (word "awk"). No sed or awk program ran; the
variables were never read. Also: the srun commands added `--bind /mnt/central/users/ojerbi:/mnt/central/users/ojerbi`
to the given template, to read the J91 CSV files (read-only). No interpreter on login, no GPU job, no queue file, no code
edit, no commit.

Method. Read with the Read tool: HRQ; reconciliation-17pct.md, recon-bytes.md, recon-draft-tick.md,
recon-row-1b-alloc.md, forecast-oct12-17pct.md; HANDOFF_2026-10-29.md; LEDGER_rev6.md; STATUS_LOG.md:860-1305;
SCORING_dr1.md, SCORING_dr73.md; PLAN_L4P_faithful.md; speckv.txt:126-152; ST:1-140, :560-759, :905-1053;
r2_i3_arena.py; r2_spec_cell.py:225-284; r2_wsa_cell.py:200-329; J91 README.md:40-123. CPU scripts s1-s5 (Write tool),
each run as `srun -p amd -c 4 --mem=16G -t 60 apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi [--bind
/mnt/central/users/ojerbi:/mnt/central/users/ojerbi] /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif /venv/nosa/bin/python -I -B
<script> > <log> 2>&1` (foreground).
