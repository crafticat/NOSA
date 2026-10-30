# RECON7 workstream R1: the ORIGINAL MODEL side

Writer: R1 agent, 2026-10-30 (stamps from `date -u`: 09:58:01Z and 10:00:43Z). Static reads, plus six
arithmetic scripts that ran inside `srun -p amd` (the nosa-env image, `python -I -B`). No GPU job. No code edit. No commit.

Request: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md (REQ). Original block REQ:16-32;
retrospective REQ:34-46.

Labels. MEASURED = a GPU record (cited). DERIVED = arithmetic from cited values, done in a logged script (log:line group
named) or shown by hand. MODEL = a value of the October 21 model formulas. CITED = a value copied from a file:line. INFERENCE
= my reading; each one is marked.

Scripts (Write tool): /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_r1/recon7_r1_model.py, recon7_r1_mem.py,
recon7_r1_round.py, recon7_r1_units.py, recon7_r1_yield.py, recon7_r1_products.py.
Logs: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_r1_model.txt, recon7_r1_mem.txt, recon7_r1_round.txt,
recon7_r1_units.txt, recon7_r1_yield.txt, recon7_r1_products.txt. (Each log starts with harmless shell-profile lines of the container, for
example `"sed": executable file not found`; they come from the image's profile inside srun, not from my commands.)

Short names for sources:
- FC = /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/forecast-oct12-17pct.md
- O3 = /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output3_timing_bw.csv (job 2175791)
- O2 = .../hbm_sweep_2175791/scoring/output2_memory.csv; HBMR = .../hbm_sweep_2175791/README.md
- GS = /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/rows_bench_2175525/gemm_mscan.json and gemm_mscan.md (job 2175525)
- LV = /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/cpupack_confirm_2179735/scoring/layout_variants.csv (job 2179735)
- I3S = /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/SCORING_i3.md; R6 = .../analysis/plan/LEDGER_rev6.md
- Code (i4dc tree, commit 2e9e497, clean): ST = /mnt/beegfs/ojerbi/nosi-r2-i4dc/nosi/nosi/spec_tentative.py,
  STF = .../spec_tentative_fork.py, STU = .../spec_tentative_union.py, VR = .../verify_rows.py; cells under
  /mnt/beegfs/ojerbi/nosi-r2-i4dc/benchmarks/Efficiency/. The schedule files are byte-identical in nosi-r2-i3 (c4c64b2)
  and nosi-r2-i4a (acbe433) (diff -q by the mapping sub-agent); those trees only lack the later lever modules.

---------------------------------------------------------------------------------------------------------------------

## 0. Answer in ten lines

1. The author's script (model_totals.py) is NOT on this cluster (FC:3-5). I rebuilt every formula from the archived
   inputs. Every number of the original block reproduces, except two at the 8th significant digit (r, transfer: CSV
   rounding) and the memory total (it needs one back-solved constant, 0.129239 GB).
2. The rebuilt rules: td = linear interpolation of the C63 RESIDENT TARGET step (eager median); GEMM(M) = sum of the five
   linear shapes, graph-timed, linear interpolation, and above M 768 the PROPORTIONAL rule GEMM(768) x M / 768; s and r =
   linear in B between B64 and B336 of the layout variant "cpu8@hm>orig:group/row" (not the base CPU8 arm).
3. U, I and the transfer bytes are C81 trace values of the author. They cannot be re-derived here. The retrospective
   numbers at C128 / B216 and C73 / B318 reproduce only if U - I = 8.969029 and the bytes stay at their C81 values.
4. All twelve retrospective numbers of REQ:35-39 reproduce (section 1.3), and the SH / joint conversions too.
5. In rows and weight passes the IMPLEMENTED round equals the model cohort: 3 weight passes; B, B and 4B rows; 3 draft
   rows + 3 verify rows per request; full accept commits 3; no bonus row in either.
6. The schedules differ in 16 other ways (section 3). The largest for the model's terms: one GEMM over 4B rows (the model
   charges GEMM(B) + GEMM(3B)); per-row selection for all 4B rows (the model charges none for verify rows); synchronous
   verify-row fetches inside the joint pass with no prefetch (the model overlaps all transfer and assumes readiness);
   a reject costs a FILL round (the model has p = 0 and no recovery).
7. The model's cohort max lends the slack of the two draft passes (td >> GEMM(B)) to the joint pass. Per pass, the joint
   pass is DENSE-bound in the model's own terms at all three points (81.03 > 75.93; 63.45 > 59.85; 89.97 > 83.10 ms).
8. MODEL per cohort (= per round of our schedule): C128 / B216 interval 187.40 ms, Q 3,457.9 tok/s; C73 / B318 252.05 ms,
   Q 3,784.9 tok/s (C81 U, I, bytes carried; Y = 3).
9. The original memory formula does NOT fit either measured configuration under 84 GB: 87.66 GB at C128 / B216 (largest B
   205), 86.04 GB at C73 / B318 (largest B 309). Its backbone holds a harness staging snapshot of 4.2-6.2 GB.
10. Rule slips: two uses of sed (one mine, one by my read-only sub-agent). Both printed or discarded output only (section 6).

---------------------------------------------------------------------------------------------------------------------

## 1. The original calculation: inputs, rules, reproduction

### 1.1 Where each input comes from, and the rule that turns it into the model's value

| term | archived source (file:line) | statistic of the calibration | rows that feed it (C, K, lag) | rule found | interpolated / extrapolated |
|---|---|---|---|---|---|
| td | O3:13 (C63 B256 resident, 62.678017 ms) and O3:15 (C63 B344 resident, 80.585732 ms); O3:11 (C63 B128 resident, 37.512192 ms) for B216 | eager median, n 43 steps, CUDA events around enqueue + run (host launch gaps INCLUDED), nova06, teacher-forced | C 63 only; K none (one target row per request); lag none. The model's C (81 / 128 / 73) never enters td | linear in B between the two neighbours | B287, B318: interpolated (256-344). B216: interpolated (128-256). No extrapolation |
| GEMM(M) | GS rows "default / linear", env base, M 128 / 256 / 320 / 384 / 512 / 640 / 672 / 768 | CUDA-graph replay (graph_ms), warm 10, reps 30, L2 flush 167,772,160 B, nova07, cuBLAS default | none (isolated GEMMs; no C, K, lag) | step sum = sum over wqkv, wo, gate_up, down (32 calls each) + lm_head (1 call) of calls x graph_ms; linear between scan points; above 768: GEMM(768) x M / 768 | GEMM(287), GEMM(318): interpolated 256-320. GEMM(216): 128-256. GEMM(648): 640-672. GEMM(861), GEMM(954): EXTRAPOLATED (proportional rule) |
| s | LV:11 (B336, slowdown_pct 7.378858) and LV:21 (B64, 14.626987), arm cpu8@hm>orig:group/row | p50 decode step beside the packer / p50 decode alone - 1, 20 beside vs 40 alone reps, decode steps 24-27, resident decode (finite plan train) | B 64 and 336; no C, K, lag (GPU-resident trace replay) | linear in B | B216, B287, B318: interpolated |
| r | LV:11 (gbps_conc_p50 11.133855) and LV:21 (8.859767), same arm | p50 useful GB/s delivered beside the decode | same | linear in B | interpolated |
| f | 15.768289 GB weights per step (GS md:28 "15.768 GB"; HBMR T4); 0.132637 GB per request | analytic bytes | C 63 window (63 groups) | f = W / (W + B x 0.132637); 0.132637 GB = 63 x 64 x 32,896 B = 132,636,672 B (DERIVED, log recon7_r1_model.txt section 4) = the C63 window slope in O2 | exact, no fit |
| U, I | CITED FC:22 (U 69.1629464286, I 60.1939174107) | author's trace replay: "64 rounds / 8 documents, simulated one-step arrivals" (FC:44) | C 81, K 3, lag 2 (FC:12-13) | U = 30985 / 448, I = 215735 / 3584 exactly (DERIVED, log section 4): averages over 7 x 64 and 56 x 64 samples (INFERENCE: 64 = 32 layers x 2 KV heads) | not re-derivable here; carried unchanged to C128 and C73 by the retrospective |
| bytes | CITED FC:29 (6.4658285714 MB / request / cohort) | author's retention replay | C 81, K 3, lag 2 | 6.4658285714 x 7 = 45.2608 exactly; = 1.0277 groups of 32,768 B per stream-step x 64 streams x 3 steps (DERIVED, log section 4) | not re-derivable. Nearest archived replay: cache_curve cc2 (B64, C81, steps 26-55) 1.0316 H2D groups per stream-step; cc1 (B192) 1.0563: close, not equal |
| memory | O2:7 (C63 B256 device_used 65.727758 GB), O2:8 (B344 83.870220 GB) | steady device_used after the timed steps (includes the harness's light-restore snapshot) | C 63 | backbone = device_used linear in B (slope 0.206164341 GB / request) + a constant 0.129238760 GB; + (C - 63) x B x 2,105,344 B + B x (1.6384 + 0.417792) MB + 3 x 17,039,360 B (FC:15-17) | B287: interpolated; B216: EXTRAPOLATED below 256 on the 256-344 segment. The constant is back-solved (not found in any file) |

The extrapolation rule for GEMM above M 768 is identified by an exact match only. The scan is flat from 672 to 768
(51.675 -> 52.069 ms), so the rule matters. Other rules give GEMM(861) = 52.45 (last two points) to 60.80 ms (least squares
M >= 512) and GEMM(954) = 52.83 to 67.56 ms (log recon7_r1_model.txt section 2). Only "GEMM(768) x M / 768" gives
58.3739739 (10 digits).

### 1.2 Reproduction of the original block (REQ:16-32, FC:19-31)

All values DERIVED in /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_r1_model.txt section 5 unless stated.

| term | original (REQ / FC) | reproduced | OK? | arithmetic |
|---|---|---|---|---|
| td | 68.9864166 | 68.9864166023 | OK | 62.678017 + 31 x (80.585732 - 62.678017) / 88 = 62.678017 + 31 x 0.20349676 |
| GEMM(B 287) | 22.6567976 | 22.6567976038 | OK | 20.0271180 + 31 x (25.4561339 - 20.0271180) / 64 |
| GEMM(BK 861) | 58.3739739 | 58.3739738593 | OK | 52.0687711 x 861 / 768 |
| U | 69.1629464 | 30985 / 448 = 69.16294643 | OK (CITED; rational form DERIVED) | - |
| I | 60.1939174 | 215735 / 3584 = 60.19391741 | OK (CITED; rational form DERIVED) | - |
| f | 0.2928997375 | 0.2928997375 | OK | 15.768289 / (15.768289 + 287 x 0.132637) = 15.768289 / 53.835108 |
| extra verifier HBM | 6.94463564 | 6.9446356413 | OK | 68.9864166 x 0.7071002625 x 8.969029 / 63 = 48.7803133 x 0.1423655 |
| cohort HBM | 213.9038854 | 213.9038854481 | OK | 3 x 68.9864166 + 6.9446356 |
| cohort dense | 126.3443667 | 126.3443666706 | OK | 3 x 22.6567976 + 58.3739739 |
| r | 10.72418445 | 10.7241847353 | OK to 8 significant digits, NOT to the stated 10 | 11.133855 - 49 x (11.133855 - 8.859767) / 272 |
| transfer | 173.0381278 | 173.0381232510 | OK to 8 significant digits | 287 x 6.4658285714 / 10.7241847 (n / K = 1) |
| s | 8.68458712 % | 8.68458712 % | OK | 7.378858 + 49 x (14.626987 - 7.378858) / 272 |
| GPU | 232.4805547 | 232.4805547357 | OK | 213.9038854 x 1.0868458712 |
| interval | 232.4805547 | 232.4805547357 | OK | max(232.48, 173.04) |
| Q | 3703.5355537 | 3703.5355536668 | OK | 1000 x 287 x 3 / 232.4805547 |
| gain | +17.16999 % | +17.169988 % | OK | 3703.53555 / 3160.822684 - 1 |
| Q without s | 4,025.1723 (FC:32) | 4025.1723 | OK | 861,000 / 213.9038854 |
| sharing: none / weights / +half KV / +full KV | 2822.0557 / 3040.9434 / 3339.6922 / 3703.5356 (FC:35) | same four values | OK | extra = td (f + (1 - f) U / 63) / td (1 - f) U / 63 / td (1 - f)(U - I / 2) / 63 / td (1 - f)(U - I) / 63 |
| n correction | n = K + p (1 + lag) (REQ:31) | p 0: n 3 both ways; p 0.109: 3.218 -> 3.327, transfer 191.90 ms | OK | log section 5 |
| memory | 83.7655 GB (REQ:17); B288 84.0117 | 83.765544 / 84.011660 | PARTLY: the backbone slope reproduces from O2; the constant 0.129239 GB is back-solved | log recon7_r1_mem.txt |

Why r and transfer miss the 9th-10th digit (DERIVED): r reproduced - r original = 10.7241847353 - 10.7241844536 =
2.8e-7 GB/s, a relative 2.6e-8. LV stores the B64 / B336 rates with 6 decimals (rounding error up to 5e-7 each), so the
author's script most likely read unrounded values. The transfer ratio (173.0381278 / 173.0381233) is the same 2.6e-8, so
r is the only cause. Neither term binds (transfer < GPU).

Not reproduced, not in REQ's block: the K grid (FC:40). It needs the memory fit at other K. Out of scope here.

### 1.3 The same formulas at the two measured configurations (REQ:35-39)

DERIVED in recon7_r1_model.txt sections 6-8. All terms OK at the stated digits.

| term | C128 / B216 human | reproduced | inputs (rows) | C73 / B318 human | reproduced | inputs (rows) |
|---|---|---|---|---|---|---|
| td | 54.8137 | 54.8136967 | O3:11-13, interp B128-256 | 75.2948 | 75.2948162 | O3:13-15, interp B256-344 |
| f | - | 0.3549987 | - | - | 0.2721164 | - |
| extra verifier HBM | 5.0333 | 5.0333204 | C81's U - I 8.969029 carried | 7.8025 | 7.8024662 | C81's U - I carried |
| joint HBM proxy td + extra | 59.8470 | 59.8470171 | - | 83.0973 | 83.0972824 | - |
| dense GEMM(B) + GEMM(BK) | 63.4452 | 63.4452179 = 18.3286467 + 45.1165712 | GEMM(216) interp 128-256; GEMM(648) interp 640-672 | 89.9657 | 89.9656538 = 25.2864772 + 64.6791766 | GEMM(318) interp 256-320; GEMM(954) EXTRAPOLATED (x 954 / 768) |
| interference s | 10.5766 % | 10.5765620 % | LV:11, :21 interp | 7.8585 % | 7.8585136 % | interp |
| budget max(...)(1 + s) | 70.1555 | 70.1555407 (dense binds) | - | 97.0356 | 97.0356169 (dense binds) | - |
| SH_ms = 1000 B (I3 / SH) / I3 tok/s | 64.7823 | 64.7823 | I3S:116, :124 | 123.486 (REQ omits) | 123.4860 | - |
| joint_ms = (joint / SH) x SH_ms | 207.368 | 207.368 | I3S:119 | 382.066 | 382.066 | - |
| joint / budget | about 3.0x | 2.9558 (gap 137.213 ms) | - | about 3.9x | 3.9374 (gap 285.030 ms) | - |

Which inputs are NOT re-derived at these configurations (open, section 5): U, I and the bytes stay at C81 values; td stays
a C63 resident step; the GEMM curve stays graph-timed nova07 cuBLAS (the later eager gm1 scan is 1.22x at M 287 / 861,
reconciliation-17pct.md row 7). The GEMM(954) value rests on the proportional extrapolation only.

### 1.4 The original memory formula at the measured configurations (MODEL; log recon7_r1_mem.txt and recon7_r1_units.txt)

| point | backbone (C63 device_used line + 0.129239) | (C - 63) x B x 2,105,344 B | overflow peak + reserve, B x 2.056192 MB | landing 3 x 17,039,360 B | total | largest B under 84.0 GB | harness staging snapshot inside the backbone |
|---|---|---|---|---|---|---|---|
| C81 / B287 | 72.248091 | 10.876207 | 0.590127 | 0.051118 | **83.765544** | 287.95 | 5.5963 GB |
| C128 / B216 | 57.610423 (extrapolated below B256) | 29.559030 | 0.444137 | 0.051118 | **87.664708** | 205.38 | 4.2118 GB |
| C73 / B318 | 78.639186 | 6.694994 | 0.653869 | 0.051118 | **86.039167** | 309.11 | 6.2007 GB |

Reading (INFERENCE): the backbone is a harness measurement. It includes the light-restore snapshot of the 2175791 harness
(O2 column staging_light_restore_snapshot_GB). Without it the totals are 78.17 / 83.45 / 79.84 GB. Our runs fit (I3S:122:
peak 83.891 GB at C128 / B216 with the prefill; 76.932 GB decode at C73 / B318, 77.685 with the prefill, I3S:134). So the
original memory term is conservative by the snapshot and does not describe our store (section 3, row 14).

---------------------------------------------------------------------------------------------------------------------

## 2. Each model term in work units, and the model's cohort schedule

### 2.1 The model's cohort schedule (as written; FC:12-13, :19-31, :42-46; REQ:16-32)

- Batch-wide lockstep. B requests. A COHORT is K = 3 consecutive weight passes.
- In each pass every request has ONE draft row (B rows per pass). The draft row feeds the newest token and drafts the next
  one. The draft reads only its READY groups (KV resident when the pass starts; "ready-selected overlap I"). Its cost is
  td: the time of one C63 resident target step (63 selected groups, all resident, 0 loads).
- The VERIFIER of a cohort has K = 3 verify rows per request (B K rows). They check the K drafted tokens: verify row j
  feeds draft j - 1 (or the last committed token for j = 1) and checks draft j. There is NO bonus row.
- The verifier rides the weight read of one draft pass. So the weights are read K = 3 times per cohort, not 4 (FC:8-10;
  "HBM = 3 td + extra" has no weight term in extra).
- The verify rows read the UNION U of their selections once; I of those groups are the same pass's draft-ready reads; only
  U - I groups are extra HBM reads.
- Deferral clock, lag 2 (INFERENCE from the n correction, REQ:30-31): the verdict on a cohort lands lag = 2 passes after its
  newest draft. During the lag the drafter continues, so a reject wastes the rejected draft plus lag drafts past it: 1 + lag
  passes of transfer per reject, n = K + p (1 + lag). The lag also gives the transfer time to land the verifier's misses
  (readiness is ASSUMED: no waits, no late fetches, FC:45-46). The formulas do not say which of the K passes carries the
  verifier; at p = 0 the lag does not change any timing term.
- PRODUCED token = an accepted draft. Y = 3 per request per cohort (p = 0). Correction and bonus tokens are omitted, and so
  are their costs (REQ:30-31; FC:43-44).
- Overlap: the whole cohort is ONE max: max(3 td + extra, 3 GEMM(B) + GEMM(BK)) x (1 + s), then max with the transfer
  service. So dense time of one pass can hide under HBM time of another pass, and all transfer hides under the GPU
  ("ideal aggregate overlap", REQ:28).

### 2.2 Term definitions in work units

| term | rows per pass | passes per cohort | query positions per request per cohort | weight bytes | KV groups / bytes per layer per KV head | H2D bytes | dependencies assumed | calibration statistic |
|---|---|---|---|---|---|---|---|---|
| td (one draft pass) | B (one row per request) | 3 | 1 per pass, 3 in all | 15.768 GB per pass (all five shapes, 32 layers + lm_head) | 63 selected groups + the tail group per request; 63 x 32,896 B x 64 layer-heads = 132.6 MB per request; also the selection (stage-1 scoring over compressed K, pooling, top-k) for that one row | 0 (resident step: "0 loads", HBMR:18) | none inside td; td is a whole-step WALL time (eager, host launch included) used as an HBM PROXY ("not an isolated HBM counter", REQ:28-29) | eager median, n 43, CUDA events around enqueue + run, nova06 |
| f | - | - | - | W = 15.768289 GB | 0.132637 GB per request = 63 groups x 64 x 32,896 B | - | td is split in TIME in proportion to BYTES: td f = weights, td (1 - f) = 63-group KV time | analytic |
| extra verifier HBM = td (1 - f)(U - I) / 63 | the verifier's B K rows ride one draft pass | 1 verifier per cohort | 3 verify positions | 0 (shared with the draft pass) | U = 69.16 union groups read ONCE for 3 rows; I = 60.19 shared with the draft's ready reads; extra U - I = 8.969 groups = 8.969 x 32,896 x 64 B = 18.88 MB per request per cohort (DERIVED, recon7_r1_round.txt) | in "bytes" below | no selection, scoring or attention compute charged for verify rows beyond the extra bytes; reads overlap the draft's | MODEL from the author's selection traces |
| HBM = 3 td + extra | - | 3 | 6 (3 draft + 3 verify) | 3 x 15.768 = 47.30 GB | draft 3 x 63 = 189 + verifier extra 8.97 groups per layer-head per request | - | "not verifier wall time"; a proxy | sum of proxies |
| GEMM(M) | M rows | - | - | 15.768 GB per call set | none (dense only; no attention, norm, rope, selection) | - | isolated, graph replay | graph median, nova07 |
| dense = 3 GEMM(B) + GEMM(B K) | B, B, B (draft) and B K (verify) in a SEPARATE call | 3 draft passes + 1 verify call set | 6 | not used | - | - | the verifier's GEMM is a second call set in its pass (two calls for one weight read: an idealisation) | graph-timed |
| s | - | per cohort | - | - | - | - | multiplicative GPU slowdown from host packing + DMA beside the decode | p50 beside / p50 alone - 1 |
| r | - | - | - | - | - | useful GB/s beside the decode, CPU8 packer, head-major source, orig destination, group packer | - | p50 |
| bytes | - | per cohort | - | - | 6.4658 MB per request per cohort = 197.3 groups of 32,768 B = 3.08 per stream per cohort (C81) | the payload of draft-ready and verifier misses (INFERENCE: the trace's fetched groups) | prefetched during the lag; never on the critical path | author's replay |
| transfer = B bytes (n / K) / r | - | per cohort | - | - | - | B x 6.4658 MB | overlapped: interval = max(GPU, transfer) | - |
| GPU = max(HBM, dense)(1 + s) | - | per cohort | - | - | - | - | cross-pass and HBM / compute overlap | - |
| Q = 1000 B Y / interval | - | - | Y = 3 accepted drafts | - | - | - | static batch, steady state, no startup / drain, no decision / recovery cost | - |
| memory | - | - | - | - | C81 = 63 + 18 slots per stream; overflow peak 1.6384 MB + reserve 0.417792 MB per request; 3 landing buffers | - | per-request (cross-layer) pooling of overflow | O2 device_used + back-solved constant |

---------------------------------------------------------------------------------------------------------------------

## 3. The IMPLEMENTED schedule mapped onto the model's cohort

### 3.1 The implemented round (K 3, newest draft deferred; steady state, full accept)

Code facts (i4dc 2e9e497; identical schedule in i3 c4c64b2 and i4a acbe433):
- Round order: one joint slot J, then K - 1 = 2 draft slots D (ST:27-45). The run starts with K = 3 draft passes, the
  initial fill (ST:904-910). In steady state each request holds u = K + 1 = 4 pending drafts e1..e4 (ST:1040).
- J = one weight pass over R = K + 1 = 4 rows per request (ST:956-960; plan ST:912-938). Rows 0..2 are VERIFY rows: they
  feed t at q, e1 at q + 1, e2 at q + 2 (ids0 = [t, e1 .. e4], ST:933-935) and check e1, e2, e3 (ST:976-982). Row 3 is the
  DRAFT row: it feeds e4 at q + 4 and drafts e5 (ST:928-929). Position q + 3 (e3 as input) has no row: the twin replays it
  from the draft's buffer entry, with no GEMM, residency or attention (ST:24-25, ST:930-931).
- D slots: two one-row passes of B rows; they draft f and g (ST:1012-1027).
- Every projection of the joint pass is ONE call over all 4B rows (`VR._lin`, natural cuBLAS by default; VR:382-388,
  r2_i4dc_cell.py:155). lm_head too (ST:313).
- Selection is PER ROW for all 4B rows: one batched stage-1 / pooling / top-k per layer (VR:413-432, called ST:360).
- Attention: ONE fork call per layer; the kernel builds each (request, KV head) union of the rows' lists and loads each
  union block ONCE for up to 4 rows (VR:123-127; STF:182-185). Each row sees only its own list.
- Residency: once per layer, for row 0's selection only (STF:145-149). Host-only blocks of rows 1..v - 1 come in ONE store
  fill per layer (STF:171-176). Draft rows read only resident KV and never fetch (`skip = draft | tentative`, ST:469-476).
  No prefetch (ST:773-774). The gathers are in-stream on the current stream (mapping sub-agent: ragged_engine.py:94; I did
  not open that file).
- Commit: greedy accept_prefix (ST:980). Full accept: m = v = 3, no bonus. Reject: m = n + 1 (prefix + correction)
  (ST:981-982). Dropped drafts u + 1 - m (ST:992). After a reject the D slots draft d1', d2' from the correction, the next
  J is a FILL round for that request (one draft row, commit 0, the D slots idle), then a group with v = 2 (ST:38-44,
  ST:1040-1042).
- Lag: J of round r verifies e1, e2, e3, drafted 4, 3 and 2 passes earlier (D1 of r - 2, J of r - 1, D0 of r - 1); the
  deferred e4 (1 pass old) and the new e5 are the drafts past the verdict (ST:1004-1011; mapping report).
- Timing: CUDA events at ST:946, :949, :974, :999, :1037; phases reset / joint / commit / draft; median steady round
  (cells r2_spec_cell.py:206-217; graph-capture rounds excluded in the newer cells).

### 3.2 Term-by-term mapping and every difference (one difference per row)

| # | item | model cohort | implemented round | same? |
|---|---|---|---|---|
| 1 | weight passes per cohort / round | 3 | 3 (J + 2 D) | YES |
| 2 | rows per weight pass | B, B, B + B K = 4B | 4B (J), B, B | YES in count; order differs (the model does not fix it) |
| 3 | draft rows per request | 3 (one per pass) | 3 (one in J, one in each D) | YES |
| 4 | verify rows per request, query positions | 3: inputs at q, q + 1, q + 2; check drafts 1-3 | 3: t at q, e1 at q + 1, e2 at q + 2; check e1-e3 (ST:933-935, :976-982) | YES |
| 5 | bonus row | none | none (ST:41, :982) | YES |
| 6 | verdict lag | 2 (deferral clock) | newest verified draft (e3) is 2 passes old; 2 drafts (e4, e5) lie past the verdict | YES under the reading of 2.1 (INFERENCE) |
| 7 | dense work of the verify pass | GEMM(B) + GEMM(3B): two call sets | ONE call set over 4B rows (VR:382-388) | NO. On the model's own curve GEMM(4B) is 3.20 / 4.87 / 3.73 ms CHEAPER than GEMM(B) + GEMM(3B) at C81 / C128 / C73 (recon7_r1_round.txt) |
| 8 | draft-row KV | td = 63 groups, all resident (C63 resident TARGET step) | resident blocks only; misses skipped and not fetched (ST:469-476); skips 0.49 / 1.96 per stream per call (I3S:225, MEASURED) | NO: the proxy reads more than our draft row, and it is a target step, not a draft step |
| 9 | verifier KV reads | union U read once; I shared with the same pass's draft reads | fork: one read per union block per (request, KV head) for the 4 rows, the draft row included (VR:123-127) | YES in mechanism. The union in code also contains blocks the draft row skips |
| 10 | verify-row selection | not charged (only the KV bytes U - I) | per row, all 4B rows, every layer (VR:413-432) | NO: extra work absent from the model |
| 11 | residency / fetch of verify misses | prefetched inside the lag; transfer overlapped (max); readiness assumed | no prefetch (ST:773-774); row 0 residency + one store fill per layer, in-stream, inside the joint pass (STF:145-176) | NO: on the critical path in our code |
| 12 | union list / locate / chain work | none | per layer (union list VR:1105-1148; chain images CB0 / CB1 STF:105-121, :165-170) | NO: extra work absent from the model |
| 13 | twin / deferred-position replay | none | twin replay of q + 3 per joint pass (ST:24-25, :930-931) | NO: small extra work |
| 14 | retention / state | C81 slots + overflow peak 1.6384 MB + reserve 0.4178 MB per request + 3 landing buffers | per-request tentative buffer (ST:20-23), a store of (K - 1) x 64 listed blocks per layer (STF:42-46), the arena; no retained overflow; KW lever keeps store slots across rounds | NO: different state |
| 15 | produced tokens | accepted drafts only, Y = 3 | accepted prefix + correction; full accept 3 | partly: corrections are 0.96 % / 1.38 % of committed (DERIVED, recon7_r1_yield.txt) |
| 16 | rejection / recovery | p = 0; no recovery service; n only scales transfer | drop u + 1 - m drafts; FILL round (commit 0 for that request, D slots idle); then v = 2 group whose row 3 still occupies GEMM rows (ST:38-44, :570, :992) | NO |
| 17 | overlap across passes | one max over the whole cohort | passes run in sequence; each layer runs residency -> fill -> fork -> GEMM in stream order | NO |
| 18 | per-round host / commit / reset work | 0 | commit 6.004 / 8.351 ms, reset 0.110 / 0.115 ms (I3S:150-151, MEASURED) | NO |
| 19 | start / drain | none (steady) | initial K = 3 draft fill (ST:904-910); final groups and target rows (ST:33-37) | NO (excluded from the steady statistic) |
| 20 | timing statistic | eager median (td), graph median (GEMM), p50 bursts (s, r) | CUDA-event median steady round; graphs in the G arms | NO: different statistics |

Rows 1-6 and 9 agree. The implemented round is the model's cohort in rows, passes and positions. The differences are in
the cost of each row (7, 8, 10, 12, 13), in when KV arrives (11), in state (14), in yield and recovery (15, 16), and in
the dependency structure (17).

---------------------------------------------------------------------------------------------------------------------

## 4. The MODEL column of the human's table

Per round of our schedule = per model cohort in rows and passes (section 3.2 rows 1-4). I give two dependency readings:
(a) the model as written (one max over the cohort); (b) the same terms bounded per pass, at our pass boundaries
(2 D passes + J, no cross-pass overlap). Reading (b) is DERIVED from the model's own terms; it is not the author's formula.
C128 and C73 use C81's U, I and bytes (section 1.3). All values ms unless stated; logs recon7_r1_model.txt (M),
recon7_r1_round.txt (R), recon7_r1_units.txt (UN), recon7_r1_yield.txt (Y), recon7_r1_mem.txt (ME).

| component | term (MODEL) | anchor C81 / B287 | C128 / B216 | C73 / B318 | log |
|---|---|---|---|---|---|
| standalone drafting / dense | td per draft pass (HBM proxy, includes its own GEMM(B)) | 68.9864 | 54.8137 | 75.2948 | M |
| | GEMM(B) per draft pass | 22.6568 | 18.3286 | 25.2865 | M |
| | per-pass draft budget max(td, GEMM(B))(1 + s) | 74.9776 | 60.6111 | 81.2119 | M |
| | per round: our 2 D passes (2 x the line above) | 149.9552 | 121.2222 | 162.4237 | P |
| | per cohort: 3 draft passes inside HBM (3 td) | 206.9592 | 164.4411 | 225.8844 | P |
| joint dense | GEMM(B) + GEMM(3B), the model's charge | 22.6568 + 58.3740 = 81.0308 | 18.3286 + 45.1166 = 63.4452 | 25.2865 + 64.6792 = 89.9657 | M |
| | one call over 4B rows, same curve (our call shape) | 77.8320 (extrap) | 58.5774 (extrap) | 86.2389 (extrap) | R |
| joint selection | verify-row selection | 0 (not charged) | 0 | 0 | - |
| joint attention / shared reads | extra = td (1 - f)(U - I) / 63 | 6.9446 | 5.0333 | 7.8025 | M |
| | extra bytes per cohort, B x 8.969 x 64 x 32,896 B | 5.4194 GB | 4.0787 GB | 6.0048 GB | UN |
| | draft KV per pass, B x 63 x 64 x 32,896 B (inside td) | 38.0667 GB | 28.6495 GB | 42.1785 GB | UN |
| | joint HBM proxy td + extra | 75.9311 | 59.8470 | 83.0973 | M |
| | joint budget max(td + extra, GEMM(B) + GEMM(3B))(1 + s) | 88.0680 (dense binds) | 70.1555 (dense binds) | 97.0356 (dense binds) | M |
| | joint budget with GEMM(4B) | max(75.9311, 77.8320) x 1.086846 = 84.5914 (dense binds) | max(59.8470, 58.5774) x 1.105766 = 66.1768 (HBM binds) | max(83.0973, 86.2389) x 1.078585 = 93.0160 (dense binds) | P |
| packing / transfers / readiness | bytes per request per cohort | 6.4658 MB | 6.4658 MB (C81 value carried) | 6.4658 MB (carried) | UN |
| | r (GB/s) | 10.7242 | 10.1306 | 10.9834 | M |
| | transfer = B bytes (n / K) / r, p = 0 | 173.0381 | 137.8617 | 187.2043 | M |
| | interference s | 8.6846 % | 10.5766 % | 7.8585 % | M |
| | readiness / waits / late fetches | 0 (assumed by the lag) | 0 | 0 | - |
| retention / state / physical capacity | memory formula at B | 83.7655 GB | 87.6647 GB | 86.0392 GB | ME |
| | largest B under 84.0 GB (same formula) | 287.95 | 205.38 | 309.11 | UN |
| | overflow peak + reserve per request | 2.0562 MB | 2.0562 MB | 2.0562 MB | FC:16 |
| schedule / yield / recovery | cohort HBM = 3 td + extra | 213.9039 | 169.4744 | 233.6869 | M |
| | cohort dense = 3 GEMM(B) + GEMM(3B) | 126.3444 | 100.1025 | 140.5386 | M |
| | GPU = max(HBM, dense)(1 + s) | 232.4806 | 187.3990 | 252.0512 | M |
| | interval = max(GPU, transfer): reading (a), per cohort | 232.4806 | 187.3990 | 252.0512 | M |
| | per-pass sum, reading (b), per round of our schedule | 238.0232 | 191.3777 | 259.4594 | M |
| | Y (accepted drafts per request per cohort), p, recovery | 3, 0, 0 | 3, 0, 0 | 3, 0, 0 | FC:21 |
| | correction / bonus credit and cost | omitted | omitted | omitted | REQ:30 |
| | Q = 1000 B Y / interval (accepted-draft tok/s) | 3,703.54 | 3,457.86 | 3,784.94 | M |
| | interval per produced token per request (interval / 3) | 77.4935 | 62.4663 | 84.0171 | M |

P = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_r1_products.txt (script recon7_r1_products.py): it re-computes the
"3 td", "2 D passes" and "joint budget with GEMM(4B)" rows from the logged values. (A first hand version of the GEMM(4B)
row in this file was off in the 4th decimal; the script values replace it.)

Model values with MEASURED yields put in Y (a mixed reading, not the model; recon7_r1_yield.txt): C128 Y 2.8926
(median-round yield, I3S:223) -> 3,334.07 tok/s; Y 2.6088 (accepted per request-round, all rounds) -> 3,006.93. C73 Y 2.8091
-> 3,544.10; Y 2.4111 -> 3,041.99. With the measured corrections per request-round (0.02525 / 0.03385) the corrected n is
3.0757 / 3.1016; the transfer grows to 141.34 / 193.54 ms and stays below the GPU term (187.40 / 252.05).

Orientation only (MEASURED, CITED; the other workstreams own the comparison): the I3 round at C128 / B216 is draft 151.363 +
joint 207.377 + commit 6.004 + reset 0.110 = round 365.423 ms (I3S:148-153); at C73 / B318 194.550 + 382.041 + 8.351 + 0.115
= 585.463 ms. NEW host MB per round 557.462 / 3,347.635 = 2.58 / 10.53 MB per request per round (I3S:209; DERIVED UN),
against the model's C81 6.47 MB per request per cohort.

---------------------------------------------------------------------------------------------------------------------

## 5. Open questions (each blocks a fully re-derived model at the measured C)

1. U, I and bytes at C128 and C73. The author's trace replay (retention_average_oct12, FC:3-5) is the only source. At
   C128 more groups are ready (I up, bytes down); at C73 fewer (INFERENCE, direction only). Need: the author's script or
   its selection traces.
2. The memory constant 0.129239 GB and the role of the staging snapshot (4.2-6.2 GB) in the "guarded C63 backbone". Need:
   model_totals.py.
3. The GEMM rule above M 768 (GEMM(768) x M / 768) is found by an exact match, not by a saved script. GEMM(954) depends on
   it (range 52.8-67.6 ms over plausible rules).
4. The deferral clock. The formulas do not fix which pass carries the verifier or what the lag buys. My reading (verdict 2
   passes after the newest draft; lag gives the transfer time) fits n = K + p (1 + lag) and our schedule (section 3.2
   row 6). The author should confirm it.
5. td is a TARGET step with 63 resident groups and its own selection and host launch (eager). The model calls it a draft
   proxy and an HBM proxy. Whether the model meant the draft to read 63 groups, or only its ready groups (fewer at C73),
   changes the draft term. Need: the author's intent.
6. s and r come from the "hm>orig, group packer" layout variant (head-major source). That source layout is not the store
   our runs read from. Whether the model meant this variant on purpose: unknown.
7. The model charges no verify-row selection and no union / chain / twin work (rows 10, 12, 13). Their MODEL value is 0 by
   omission, not by a claim. Any correction needs a work-unit cost per verify row.

---------------------------------------------------------------------------------------------------------------------

## 6. Rule slips (reported, not hidden)

1. MINE. One login-node command contained `sed --version >/dev/null 2>&1` (in the command that also ran `grep -n "tile
   law" report.md`, `ls tmp_sc`, `ls cpu-logs`). The rules forbid sed of any kind. It changed no file; its output went
   to /dev/null. Cause: a slip while composing the command. No further sed use.
2. MY SUB-AGENT (read-only code mapping). It ran `sed -n 300,316p` once on benchmarks/Efficiency/r2_base_cell.py on the
   login node, to print lines. It changed nothing. It reported the slip itself. I had passed it the no-sed rule.
3. Not a slip, noted for completeness: every srun log starts with the container profile's own errors ("sed" / "hostname" /
   "grep" / "cut": executable file not found). They come from the image's shell profile inside the srun on amd.

No interpreter ran on the login node. No GPU job, no sbatch, no scancel, no push, nothing written in a queue directory.
The only files written: the six scripts, the six logs and this file.

---------------------------------------------------------------------------------------------------------------------

## 7. Shell commands I ran (login node unless marked srun), in order

1. `ls -la /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/ | tail -60; ls /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/ | head -100`
2. `cd /mnt/beegfs/ojerbi/kstep-cpu-runs; grep -n "54.8137\|75.2948\|70.1555\|97.0356\|10.5766\|7.8585" analysis/plan/*.md paper-notes/*.md 2>/dev/null | cut -c1-400 | head -40`
3. `cd /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence; ls; grep -rln "2179735" . | head; grep -rln "2175525" . | head`
4. `cd .../evidence; cat cpupack_confirm_2179735/README.md; ls cpupack_confirm_2179735 cpupack_confirm_2179735/main rows_bench_2175525`
5. `cd .../evidence/cpupack_confirm_2179735; ls scoring; grep -rn "10.724\|8.684\|0.0868" scoring | head -20`
6. `cd .../evidence; ls -la cost_model calculator formula_first_principles; grep -rn "10.724\|0.0868\|8.684\|6.4658\|68.986\|22.6567\|58.3739\|69.1629\|60.1939" cost_model calculator formula_first_principles | cut -c1-300 | head -30`
7. `ls /mnt/beegfs/ojerbi/kstep-cpu-runs/; ls /mnt/central/users/ojerbi | head -80`
8. `cd /mnt/central/users/ojerbi; grep -rln "model_totals\|retention_average_oct12\|conditional_throughput" retroinfer-eval/docs retroinfer-eval/scripts paper-prose /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes 2>/dev/null | head -20`
9. `cd .../rows_bench_2175525; cat gemm_mscan.md; head -c 3000 gemm_mscan.json; echo; grep -n "gemm\|GEMM" report.md | head -30`
10. `cd .../evidence; ls fetch_policy_replay kv_union_K16 selection_ids_2173290 acceptance_2175629 2>/dev/null | head -40; head -c 1500 trace_cache_policy.json`
11. `cd /mnt/central/users/ojerbi/retroinfer-eval; grep -rln "69\.16\|60\.19\|6\.4658\|30985\|1\.6384" docs 2>/dev/null | head -20; ls docs | head -50`
12. `cd .../evidence/cache_curve; ls; ls cc2_2179985 cc1_2179948 | head -40; head -80 README.md 2>/dev/null` (exit 1: no README.md)
13. `cd .../cache_curve; cat cc1_2179948/ccurve_table.md | head -80; head -5 cc1_2179948/ccurve_points.csv`
14. `cd .../cache_curve; grep -n "natural plans at C\|from step 1" cc2_2179985/ccurve_table.md | cut -c1-200`
15. `cd .../analysis/plan; grep -n "69\.16\|U = 69\|6\.4658\|10\.724\|0\.0868\|8\.68" LEDGER_model_vs_measured.md STATUS_LOG.md HANDOFF_2026-10-29.md LEDGER_rev6.md | cut -c1-300 | head -40`
16. `cd .../analysis/plan; cat HANDOFF_2026-10-29.md; grep -n "I3\|1709.8\|1525.8\|own-SH\|0.5128\|3.201" STATUS_LOG.md | tail -30 | cut -c1-300`
17. `cd .../evidence; head -1 hbm_sweep_2175791/scoring/output3_timing_bw.csv | cut -c1-600; cut -d, -f1-8 hbm_sweep_2175791/scoring/output3_timing_bw.csv | head -25; cat cpupack_confirm_2179735/scoring/layout_variants.csv | head -40`
18. `cd .../rows_bench_2175525; sed --version >/dev/null 2>&1; grep -n "tile law\|9.92\|extrapol" report.md | head; ls /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/ | head; ls /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs | tail -5` (SLIP: sed, section 6)
19. srun: `srun -p amd -c 4 --mem=16G -t 60 apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi --bind /mnt/central/users/ojerbi:/mnt/central/users/ojerbi /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif /venv/nosa/bin/python -I -B .../tmp_sc/recon7_r1/recon7_r1_model.py > .../cpu-logs/recon7_r1_model.txt 2>&1; echo rc=$?; cat .../cpu-logs/recon7_r1_model.txt`
20. `cd .../hbm_sweep_2175791/scoring; ls; head -1 output2_memory.csv; cut -d, -f1-6 output2_memory.csv | head -20`
21. srun: the same srun line with recon7_r1_mem.py > recon7_r1_mem.txt; then `grep -v "not a valid\|executable file" .../recon7_r1_mem.txt`
22. `cd .../analysis/plan; grep -n "round\b\|rows per\|R = 4\|deferred\|bonus\|yield" SCORING_i3.md | head -40 | cut -c1-260`
23. `for d in nosi-r2-i4dc nosi-r2-i3 nosi-r2-i4a; do git -C /mnt/beegfs/ojerbi/$d rev-parse --short HEAD; git -C /mnt/beegfs/ojerbi/$d status --short | wc -l; done; date -u`
24. `cd .../analysis/plan; grep -n "R6\|rows per kept\|row count\|6 rows\|rows per request" LEDGER_rev6.md | head -30 | cut -c1-250`
25. srun: the same srun line with recon7_r1_round.py > recon7_r1_round.txt; then grep -v as in 21
26. `cd /mnt/beegfs/ojerbi/kstep-cpu-runs; grep -n "Figure 3" paper-notes/section-drafts.md analysis/plan/PLAN_L4P_faithful.md analysis/plan/PLAN_speckv_on_cached_nosa.md | head -12 | cut -c1-300`
27. `cd /mnt/beegfs/ojerbi/nosi-r2-i4dc/nosi/nosi; grep -n "" verify_rows.py | head -130 | tail -15; grep -n "" spec_tentative.py | head -476 | tail -8; grep -n "" spec_tentative.py | head -775 | tail -4; grep -n "" spec_tentative_fork.py | head -149 | tail -6`
28. srun: the same srun line with recon7_r1_units.py > recon7_r1_units.txt; then grep -v
29. srun: the same srun line with recon7_r1_yield.py > recon7_r1_yield.txt; then grep -v; `date -u`
30. `cd .../evidence; grep -n "hm>orig:group/row" cpupack_confirm_2179735/scoring/layout_variants.csv | cut -c1-60; grep -n "^63,128,resident\|^63,256,resident\|^63,344,resident" hbm_sweep_2175791/scoring/output3_timing_bw.csv | cut -c1-40; grep -n "^63,256,full\|^63,344,full" hbm_sweep_2175791/scoring/output2_memory.csv | cut -c1-40; grep -n "weight bytes per step" rows_bench_2175525/gemm_mscan.md`
31. srun: the same srun line with recon7_r1_products.py > recon7_r1_products.txt; then grep -v

File reads with the Read tool (not shell): REQ; FC; reconciliation-17pct.md; recon-rows-2-3.md; recon-rows-5-6.md;
verifier-working-set.md; HBMR; SCORING_i3.md:85-234; LEDGER_rev6.md:76-185; ST:14-53, :900-1044.
The mapping sub-agent's commands are its own (read-only; one sed slip, section 6).
