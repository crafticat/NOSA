# LEDGER revision 6 (2026-10-29): the I4 levers measured, the new round, the remaining gap

Written 2026-10-29T16:55:48Z (from `date -u`). Writer: the rev-6 analysis agent, read-only. Revision 5 stays in
PL/LEDGER_model_vs_measured.md (R5.0-R5.7, :813-1036); this file does not change it. Style: ASD-STE100.

Labels. MEASURED = a value in a record (stdout REPORT / PREDICTION / GATE line or summary.json field). DERIVED = arithmetic
on MEASURED values (operands shown, or computed by the CPU script named). INFERRED = a reading of code or records that no
record states. MODEL = needs an assumption (stated).

Abbreviations. PL = /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan. R = /mnt/beegfs/ojerbi/r2-runs/spec. LOG3 =
/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/ana_rev6_s3.txt (script tmp_sc/ana_rev6/s3_derive.py, srun on amd); LOG2 =
cpu-logs/ana_rev6_s2.txt (s2_profile.py: the I4C profile records); LOG1 = cpu-logs/ana_rev6_s1.txt (s1_keys.py). SL =
PL/STATUS_LOG.md. NG = PL/NOTES_graph_mean_gap.md. T_SH = the same-job SH steady step (ms). Budget = y_med x T_SH, with
y_med = median tok/s x round median / B (rev 5 identity; ratio = Budget / round). "Strongest" = the same-GPU strongest
cached NOSA steady (whole-step graph, C 128 B 216): nova05 3,426.2, nova07 3,410.1 tok/s (PL/CHECKPOINTS.md:9-11).
Jobs 2195575 = nova05, 2195576 = nova07 (SL:871).

Arm names. L4PTF = I3 (fork joint pass). L4PTFD = I3 + lean draft (lever 1). L4PTFDG = L4PTFD + CUDA graphs of the joint
pass (lever 3). L4PTFK / L4PTFKW = L4PTFD + store keep / + row 0's window from the store (lever 4). L4PTFDGKW = graphs +
keep + window (I4K). Every I4 arm is bit-exact to L4PTFD (EQF MUST PASS: tokens, fingerprints, 227 state checksums,
acceptance record), so every I4 arm has the same correctness status as L4PTF: FAST, near-tie class, NOT certified.

## R6.0 Answer in eight lines

1. Best arm per batch (MEASURED, median steady): C 128 B 216 **L4PTFDG 0.5342 x same-job SH = 0.5247 x strongest** (g3a
   nova05; h4a nova07 0.5345 / 0.5243); C 73 B 318 **L4PTFDGKW 0.6307 x SH = 0.4736 x strongest** (k4b nova07, one run).
2. The best feasible config under the 84 GB cap stays **C 128 B 216** (peak 83.891 GB with prefill, every arm). The
   strongest cached NOSA is 1 / 0.5247 = **1.91x faster** (DERIVED). Registered I4M verdict (L4PTFD): MAIN 0.5152.
3. The four levers delivered about one fifth (C 128) to one third (C 73) of the rev-5 gain: I3 0.5128 -> 0.5342 and
   0.5925 -> 0.6307 own SH, against the rev-5 ceiling 0.620 / 0.708 (DERIVED, R6.1).
4. Per committed token the round now costs d + j1 = **113.8 ms = 1.774 SH (C 128)** and **180.9 ms = 1.457 SH (C 73)**
   (rev 5: 119.6 / 192.2). 1.0 x SH needs < 62.0 / 116.0 ms: -45.5 % / -35.9 % more (DERIVED, R6.3).
5. **The joint pass alone exceeds the whole budget**: 198.6 > 186.0 ms (C 128) and 352.6 > 348.1 ms (C 73) (MEASURED /
   DERIVED). Free drafts would give at most 0.909 / 0.964 x own SH with today's joint pass.
6. Largest remaining terms (share of the best round): the drafts 41.0 % / 34.5 % (a draft is a full sparse model step);
   the dense GEMMs over 4B rows 27.5 % / 22.1 % (x3.0 SH's term); PCIe of verify-row misses 5.8 % / 20.9 %; the fork
   11.7 % / 10.2 %. These are STRUCTURAL or need a new kernel (R6.4).
7. Ceiling with existing components, every remaining lever at its upper bound (MODEL, optimistic): **0.589 own SH = 0.578
   x strongest at C 128** (0.627 / 0.616 if an off-the-shelf compiler removes the whole small-kernel class); 0.668 / 0.501
   at C 73. With the new kernels that are not allowed, plus hidden PCIe, plus standalone drafts: 0.680 / 0.668 (C 128).
8. No path in this design reaches 1.0 x the strongest at these batches. The direction is the human's decision (handoff).

## R6.1 Each lever: the rev-5 estimate against the MEASURED outcome

Estimates: rev 5 R5.4 (:941-950) and R5.5 (:964-1012). "Realised" = measured removal / estimate (DERIVED).

| lever | rev-5 estimate, ms per round, C 128 / C 73 (label) | MEASURED outcome, C 128 B 216 | MEASURED outcome, C 73 B 318 | realised | runs (R/...) |
|---|---|---|---|---|---|
| L1 lean draft (I4A) | 28.657 / 31.396 (DERIVED bound) | draft saving 6.198 (nova05) / 6.847 (nova07); L4PTFD / L4PTF 1.0148 / 1.0169; draft step 1.1183 / 1.1182 SH | 5.658 (nova07) / 5.141 (nova05); 1.0087 / 1.0082; 0.7643 / 0.7647 SH | 21.6-23.9 % / 16.4-18.0 % | i4a_q3a / k4a; q4b / k3b (20 / 20 keys CONFIRMED twice, SL:925-931, :982-986) |
| L2 fork launch sweep (I4B) | 17.73 / 22.32-23.24 (MODEL, NS1 byte rate) | **0**: default W4S3T64 1.13063 ms per layer (repeat 1.11602, margin 0.02584) is the best eligible; next W4S2T64 1.24314 (+9.95 %); W8 configs not bitwise (942 unequal elements), T32 configs fail EQ | **0**: default 1.60154 (repeat 1.58885, margin 0.01585); next W4S2T64 1.75339 (+9.48 %); W8 not bitwise (1,255), T32 fail EQ | 0 % | i4b_w3sa / w4sb (GATE SWP-PICK) |
| L3 host exposure (I4C measure) | 11.45-13.75 / 21.09-25.33 (MODEL; C 73 transferred from C 128) | host share 1 - 0.9486 = 0.0514 (nova05), 1 - 0.9511 = 0.0489 (nova07) of the steady joint: 0.0514 x 206.157 = **10.60 ms** | 1 - 0.9958 = 0.0042: **1.61 ms** (device bound) | the C 73 MODEL was 13-16x too high | i4c_c3a, e4a; c4b |
| L3 CUDA graphs (I4G) | (same) | joint saving 6.962 / 7.364 ms; L4PTFDG / L4PTFD 1.0191 / 1.0197 | joint saving **7.726** (REFUTED ABOVE band -2..3); 1.0134: device-side gaps, not host exposure | 50.6-64.3 % / 30.5-36.6 % (other mechanism) | i4g_g3a, h4a; g4b |
| L4 store refetch / keep (I4D) | 3.218 / 15.506 (MEASURED span delta) | K: wait saving 4.008, joint **+4.446** (slower), round +4.987 (360.535 - 355.548); KW: wait 6.492, joint +5.522, round +5.903; ratio 0.9872 / 0.9862: **a LOSS** (bookkeeping 8.5 / 12.0 ms, SL:1021-1023) | K: wait 16.754, joint saving 5.425, round -4.594; KW: wait 39.248, joint 21.972, round **-21.550** (580.986 - 559.436); ratio 1.0072 / **1.0360** (replicate 1.0075 / 1.0373) | -155 % to -183 % / 30 % (K), 139 % (KW) | i4d_d3a; d4b, f3b |
| L3 + L4 combined (I4K) | not estimated | L4PTFDGKW / L4PTFD 1.0123, / L4PTFDG **0.9930**: the keep still loses at B 216 inside graphs | / L4PTFD **1.0564**, / L4PTFKW 1.0170; joint saving 32.063; round 583.220 -> 551.898 (-31.322) | - | i4k_k3a; k4b |
| L5 union + chain + writes | 29.97 / 30.49 ms, removable share UNKNOWN | device time 24.10 of 29.85 ms = **0.807** (nova07 0.829): mostly device work, so graphs cannot remove it; not built | 26.65 of 30.46 = **0.875** | not built | LOG2 l5 (c3a, e4a, c4b) |
| L6 PCIe at C 73 | 153.8 ms spans; removal so far about 0 | - | the window keep cut NEW host blocks 101,764.2 -> 77,960.2 (-23.4 %) and the wait 154.742 -> 115.494 ms; **115.5 ms remain** = 20.9 % of the best round | 25 % of the spans | i4d_d4b; k4b |
| L7 probe - steady gap | 5.31 / 5.79 unattributed | gap 4.648, uncovered 6.773, residual **1.019** (nova07 0.562): attributed | gap 5.098, uncovered 8.623, residual **-12.675** (REFUTED band -2..2): the label medians do not add at B 318 | - | LOG2 l7 |
| L8 drain | 0 on the median | unchanged on the median; the mean and effective rates move by the capture round (L61) and the first-arm penalty (L62), NG 4-6 | same | - | NG |
| L9 matched memory | 0 ms; ratio "likely DOWN" | FIT_B_128 = 216 = the timing point; MAIN **0.5152** (1,765.1 / 3,426.2), DIAG 0.5237; NOT down against the fixed denominator | FIT_B_73 351 -> **OOM** (83.64 GB allocated + 126 MiB; usable ceiling about 83.75 GB after SH, L60); the largest feasible B is in 318-350 (not measured) | - | i4m_n3128; n373 |
| L10 K | +-1 % (MODEL) | no run; re-derived MODEL: K 2 / 3 / 4 = 0.5304 / 0.5350 / 0.5374 (L4PTFDG) | 0.6264 / 0.6313 / 0.6337 (L4PTFDGKW) | - | LOG3:203, :270 |
| L11 natural GEMM | 0 | 0 (already in L4PTF) | 0 | - | rev 5 R5.2 |

Rung totals (DERIVED, cross-job; GPU spread <= 1.4 % in tok/s, SL:916-919): own-SH ratio 0.5128 (I3, r4a) -> 0.5342 (best,
g3a): +0.0214; rev-5 ceiling 0.620: share realised 0.0214 / (0.620 - 0.5128) = 0.20. C 73: 0.5925 (r3b) -> 0.6307 (k4b):
+0.0382; 0.0382 / (0.708 - 0.5925) = 0.33.

Why L1 stopped at its low edge (DERIVED from the labelled draft passes, LOG1 c3a / c4b `probe_draft.labels_median`,
L4PTFDP, MEASURED): one C 128 draft step = an SH-like body of 64.505 ms (resid 12.708 + select 9.660 + attnA 19.243 + post
18.604 + pre 2.800 + head 0.757 + twin 0.716 + embed 0.017) plus **8.288 ms of tentative-row work** (write 2.742 + amask
1.107 + bmask 0.678 + attnB 1.608 + merge 2.153). The body matches SH's own phases (R4.1: dense 22.012 vs post + pre +
head 22.161; attention 19.346 vs attnA 19.243; selection 9.691 vs select 9.660; residency 12.237 vs resid 12.708). At C 73
the body is 85.348 ms and the tentative-row work 10.493 ms (3.582 + 1.356 + 0.694 + 2.304 + 2.557); the draft resid is
13.773 ms because the draft skips its misses (SH's C 73 residency term is about 54 ms, INFERRED R4.1). So the rev-5 excess
(14.3 / 15.7 ms per step) was about 6-7 ms of removable Python ops (now gone) plus 8.3 / 10.5 ms that the design needs.

## R6.2 The measured round of the best arm at each batch, against the same-job SH step

Phase medians: LOG3:200-206 (g3a) and :267-273 (k4b); the joint split is the eager probe split of the same arm (labels
`groups_median`, 12 / 13 probes). The SH counterpart is R4.1 (C 128 p0all, MEASURED elsewhere; C 73 INFERRED split).

C 128 B 216, L4PTFDG, run R/i4g_g3a_2195575_i4ga (nova05). T_SH 64.176 ms; Budget 2.8981 x 64.176 = **185.991 ms**:

| term | ms (MEASURED) | SH steps | % of round | SH's own term (R4.1) | x SH term |
|---|---|---|---|---|---|
| draft phase, 2 steps | 142.913 | 2.2269 | 41.0 | one SH step per token | 1.1134 per step |
| joint pass, steady (graphs) | 198.579 | 3.0943 | 57.0 | - | - |
| - dense + twin + selection (`dense_next`, 31 layers) | 95.611 | 1.4898 | 27.5 | 22.012 + 9.691 = 31.703 | x3.02 |
| - fork attention | 40.817 | 0.6360 | 11.7 | 19.346 | x2.11 |
| - residency + fetch + union | 20.575 + 13.090 + 14.650 = 48.315 | 0.7529 | 13.9 | 12.237 | x3.95 |
| -- of it the host-gather wait (nested; `kv_at_verify.wait_ms_median`) | 20.206 | 0.3149 | 5.8 | fetch 4.995 | x4.05 |
| - write + chain + check | 10.559 + 3.816 + 0.095 = 14.470 | 0.2255 | 4.2 | none (new work) | - |
| - steady joint - probe total (200.064) | -1.485 | - | - | - | graphs replay below the eager probe |
| reset + commit | 6.128 | 0.0955 | 1.8 | none | - |
| round (sum of phases 347.620; round - sum +0.582) | 348.202 | 5.4257 | 100 | Budget 185.991 = 2.8981 SH | ratio 0.5342 |

Replicates (MEASURED, LOG3:215, :245): h4a nova07 round 349.705 (draft 143.791, joint 199.276, ratio 0.5345); k3a nova05
round 349.068 (joint 198.811, ratio 0.5303).

C 73 B 318, L4PTFDGKW, run R/i4k_k4b_2195576_i4kb (nova07). T_SH 124.195 ms; Budget 2.8027 x 124.195 = **348.082 ms**:

| term | ms (MEASURED) | SH steps | % of round | SH's own term (R4.1, INFERRED) | x SH term |
|---|---|---|---|---|---|
| draft phase, 2 steps | 190.208 | 1.5315 | 34.5 | - | 0.7658 per step |
| joint pass, steady (graphs + keep + window) | 352.620 | 2.8392 | 63.9 | - | - |
| - dense_next | 121.840 | 0.9810 | 22.1 | 26.44 + 13.50 = 39.94 | x3.05 |
| - fork attention | 56.345 | 0.4537 | 10.2 | 28.05 | x2.01 |
| - residency + fetch + union | 65.191 + 86.320 + 14.764 = 166.275 | 1.3388 | 30.1 | 53.98 (remainder) | x3.08 |
| -- of it the host-gather wait (nested) | 115.542 | 0.9303 | 20.9 | about 0.49 x 53.98 = 26.5 (DR73, INFERRED) | about x4.4 |
| - write + chain + check | 11.205 + 4.915 + 0.094 = 16.214 | 0.1306 | 2.9 | none | - |
| - steady joint - probe total (361.870) | -9.250 | - | - | - | - |
| reset + commit | 8.554 | 0.0689 | 1.55 | none | - |
| round (sum 551.382; +0.516) | 551.898 | 4.4438 | 100 | Budget 348.082 = 2.8027 SH | ratio 0.6307 |

PCIe rate (DERIVED, `kv_at_verify`): 2,564.58 MB per round / 115.542 ms = 22.2 GB/s (C 73); 557.46 MB / 20.206 ms = 27.6
GB/s (C 128). One run only at C 73 for this arm (replicate 595_i4krep_b_n5 queued, SL:1097).

Device classes of one profiled joint pass (L4PTFD, round 112, eager, MEASURED, LOG2; busy 197.06 / 383.91 ms):

| class | C 128 ms (kernels) | C 73 ms (kernels) | note |
|---|---|---|---|
| GEMM | 64.48 (129) | 79.88 (129) | 3.24x SH's ncu GEMM 19.882 (C 128, R4.1) |
| H2D (host gathers) | 21.37 (128) | **160.14** (128) | PCIe; 41.7 % of the C 73 busy time |
| OTHER (the fork) | 37.40 (64) | 52.66 (64) | - |
| SMALL | 31.35 (**7,437**) | 35.94 (7,437) | union 7.96, write 8.69, chain 3.47, resid 2.66, twin 2.36, select 2.26, attn 1.62, fetch 1.04 (C 128 labels) |
| POOL + STAGE1 + SELECT | 19.62 + 9.83 + 7.26 | 25.06 + 13.54 + 9.28 | selection |
| FUSED + COPY + TWINF + DIFF | 2.94 + 1.13 + 1.02 + 0.64 | 4.25 + 1.14 + 1.31 + 0.71 | - |

Rows per committed token (DERIVED): each round computes 2 draft rows + 4 joint rows = 6 rows per request for y_med
committed tokens: 6 / 2.8981 = **2.07** (C 128) and 6 / 2.8027 = **2.14** (C 73) rows per token; SH computes 1.

## R6.3 The per-token model, re-derived from the new phase medians

round(K) = (K - 1) d + J0 + (K + 1) j1 + c = K (d + j1) + c, with d = draft / 2, j1 = (joint - d) / 3, J0 = d - j1, c =
reset + commit, a = y_med / 3 held for every K; ratio(K) = a K T_SH / (K (d + j1) + c) (rev 5 R5.4 :955-962). All values
DERIVED in LOG3 (line of the arm).

| batch | arm (run, LOG3 line) | d | j1 | J0 | c | **d + j1** ms (SH) | a x T_SH | ratio K 2 / 3 / 4 / limit |
|---|---|---|---|---|---|---|---|---|
| C 128 | L4PTF rev 5 (i3 r4a, :14) | 75.681 | 43.899 | 31.783 | 6.114 | 119.580 (1.8457) | 62.467 | 0.5094 / 0.5136 / 0.5158 / 0.5224 |
| C 128 | L4PTFD (g3a, :196) | 71.359 | 44.727 | 26.632 | 6.066 | 116.087 (1.8089) | 61.937 | 0.5200 / 0.5244 / 0.5267 / 0.5335 |
| C 128 | **L4PTFDG (g3a, :203)** | 71.457 | 42.374 | 29.082 | 6.128 | **113.831 (1.7737)** | 61.997 | 0.5304 / 0.5350 / 0.5374 / 0.5446 |
| C 128 | L4PTFDG (h4a, :218) | 71.895 | 42.460 | 29.435 | 6.109 | 114.356 (1.7710) | 62.307 | 0.5307 / 0.5353 / 0.5377 / 0.5449 |
| C 128 | L4PTFDGKW (k3a, :255) | 71.746 | 43.531 | 28.215 | 6.098 | 115.277 (1.8021) | 61.858 | 0.5228 / 0.5273 / 0.5296 / 0.5366 |
| C 73 | L4PTF rev 5 (i3 r3b, :30) | 97.275 | 94.922 | 2.353 | 8.466 | 192.197 (1.5564) | 115.632 | 0.5887 / 0.5929 / 0.5951 / 0.6016 |
| C 73 | L4PTFD (k4b, :263) | 94.965 | 96.573 | -1.608 | 8.493 | 191.537 (1.5422) | 116.067 | 0.5928 / 0.5972 / 0.5993 / 0.6060 |
| C 73 | L4PTFDG (g4b, :233) | 94.865 | 93.743 | 1.123 | 8.512 | 188.608 (1.5199) | 116.044 | 0.6017 / 0.6061 / 0.6084 / 0.6153 |
| C 73 | L4PTFKW (d4b, :166) | 94.865 | 88.755 | 6.110 | 8.519 | 183.619 (1.4798) | 115.720 | 0.6159 / 0.6206 / 0.6230 / 0.6302 |
| C 73 | **L4PTFDGKW (k4b, :270)** | 95.104 | 85.839 | 9.265 | 8.554 | **180.943 (1.4569)** | 116.027 | 0.6264 / 0.6313 / 0.6337 / 0.6412 |

Change of the marginal cost per committed token (DERIVED): C 128 119.580 -> 113.831 = -5.749 ms (-4.8 %; 1.8457 -> 1.7737
SH); C 73 192.197 -> 180.943 = -11.254 ms (-5.9 %; 1.5564 -> 1.4569 SH). For 1.0 x SH, d + j1 must fall below a x T_SH:
61.997 / 113.831 = 0.5446 -> a further **-45.5 %** (C 128); 116.027 / 180.943 = 0.6412 -> **-35.9 %** (C 73). K is still not
a lever: K 4 adds +0.0024 / +0.0024 (MODEL). Weak points (unchanged from rev 5): J(1) uses a draft row, which skips misses;
at C 73 J0 goes negative for L4PTFD (-1.6 ms), so the linear row model does not hold there (a verify row costs more than a
draft row because of its misses).

## R6.4 The remaining gap, term by term

Gap = round - Budget: C 128 348.202 - 185.991 = **162.211 ms = 46.6 %** of the round; C 73 551.898 - 348.082 = **203.816
ms = 36.9 %** (DERIVED). Classes: EXIST = removable with existing components (bound given); STRUCT = the design needs it
(no kernel removes it); KERNEL = needs a new kernel (not allowed).

| # | term | C 128: ms (% of round) | C 73: ms (% of round) | class and bound | evidence |
|---|---|---|---|---|---|
| 1 | draft SH-like body (2 one-row model steps) | about 2 x 64.5 = 129 (37 %) | about 2 x 85.3 = 171 (31 %) | **STRUCT**: the method drafts with the model itself; one draft = one sparse step (body = SH's phases, R6.1) | LOG1 probe_draft; R4.1 |
| 2 | dense GEMMs over 4B rows in the joint pass | 95.611 (27.5) | 121.840 (22.1) | **STRUCT**: x3.0 SH's term; FLOPs scale with rows; at 100 % bf16 peak 864 rows need 41.6 ms = 1.89x SH's whole dense term (R4.3 MODEL) | LOG3; LOG2 GEMM 64.48 / 79.88 |
| 3 | PCIe of verify-row misses (host-gather wait, nested in 4) | 20.206 (5.8) | 115.542 (20.9) | **STRUCT** with existing components: the verify rows' selections exist only inside the joint pass; WS-E (L4PE / L4P 0.9969) hid nothing. The keep removed the avoidable part at C 73 (23,804 blocks per round against 20,270 counted avoidable, i4dd); at C 128 the keep costs more than it saves | kv_at_verify; SL:1030-1033, :1044-1050 |
| 4 | fork attention | 40.817 (11.7) | 56.345 (10.2) | STRUCT part at NS1's byte rate about 23.2 / 33.2-34.1 (MODEL, rev 5 L2); **KERNEL** part 17.6 / 22.2-23.1 (spills, 12 % active warps; I4B closed launch settings) | I4B; FIA6 via rev 5 |
| 5 | draft tentative-row work (amask + bmask + attnB + merge) | 2 x 5.546 = 11.093 (3.2) | 2 x 6.911 = 13.823 (2.5) | **EXIST** bound (E2: route the draft row through the pinned fork at R 1; untested) or KERNEL (one fused window + buffer call) | LOG3 CEILINGS |
| 6 | draft buffer write | 2 x 2.742 = 5.484 (1.6) | 2 x 3.582 = 7.164 (1.3) | STRUCT (contract D3) | LOG1 |
| 7 | union lists (in 4) | 14.650 (4.2); device 11.928 | 14.764 (2.7); device 12.507 | **EXIST** bound (E3 build once per pass; E4 compiler fusion); device fraction 0.80 / 0.87 | LOG2 l5 |
| 8 | rest of residency + fetch (pool, diff, H2D kernels, small ops) | 48.315 - 20.206 - 14.650 = 13.459 (3.9) | 166.275 - 115.542 - 14.764 = 35.969 (6.5) | mostly STRUCT (one residency call per layer, as SH) | DERIVED |
| 9 | joint write + chain | 14.375 (4.1) | 16.120 (2.9) | STRUCT function (tentative state); fusing it is KERNEL or E4 | LOG2 l5 (device 0.79 / 0.91) |
| 10 | host exposure left | draft 5.688 (1.6) + joint about 3.634 (1.0) | draft 4.318 (0.8) + joint about 0 | **EXIST** bound (E1 draft graph; joint graphs done) | LOG3 CEILINGS; R6.1 L3 |
| 11 | reset + commit | 6.128 (1.8) | 8.554 (1.55) | small; not a lever | LOG3 |

Ranking of the remaining gap by what can move (C 128 / C 73, ms per round):
- STRUCT, cannot move with existing components or any kernel: draft bodies 129 / 171, dense 95.6 / 121.8, PCIe wait
  20.2 / 115.5 (needs an algorithm change, not a kernel), fork at NS1's rate about 23 / 33, buffer writes 5.5 / 7.2.
- KERNEL (not allowed): fork efficiency 17.6 / 22.2-23.1; fused draft-row attention (= item 5); fused small ops.
- EXIST, not yet tried, upper bounds that overlap: item 5 11.1 / 13.8; item 7 11.9 / 12.5; item 10 9.3 / 4.3. Sum 32.3
  / 30.6 ms = **20 % / 15 % of the gap** (DERIVED: 32.343 / 162.211; 30.647 / 203.816).

## R6.5 The updated ceiling (ELAPSED view; Budget held fixed; removals are UPPER BOUNDS and can overlap)

Inputs: LOG3 "CEILINGS" block (:284-306), except the line marked (h), which corrects a double count in the script (the
script replaced the whole measured draft phase after R1 and R4 had already been removed from it). Own = Budget / round;
strongest = own x SH median / same-GPU strongest (0.98234 at C 128 nova05; 0.75086 at C 73 nova07).

| step (cumulative) | C 128 (L4PTFDG g3a) round -> own / strongest | C 73 (L4PTFDGKW k4b) round -> own / strongest | label |
|---|---|---|---|
| measured best arm | 348.202 -> 0.5342 / 0.5247 | 551.898 -> 0.6307 / 0.4736 | MEASURED |
| - R1 draft-step CUDA graph (host share 1 - 0.9602 / 1 - 0.9773 of the draft phase) | -5.688 | -4.318 | MODEL bound |
| - R2 joint host share left after graphs (0.0514 x 206.157 - 6.962) | -3.634 | 0 (1.610 raw; graphs already took more) | DERIVED bound |
| - R3 union lists once per pass (union device ms) | -11.928 | -12.507 | MODEL bound |
| - R4 draft row through the fork at R 1 (2 x tentative-row labels; fork cost at R 1 not measured) | -11.093 | -13.823 | MODEL bound |
| **= existing components, all at upper bounds** | **315.859 -> 0.5888 / 0.5784** | **521.251 -> 0.6678 / 0.5014** | MODEL, optimistic |
| variant: R3 replaced by the whole SMALL class (compiler fusion, not tried) 31.35 / 35.94 | 296.437 -> 0.6274 / 0.6163 | 497.817 -> 0.6992 / 0.5250 | MODEL, very optimistic |
| + fork at NS1's byte rate (NEW KERNEL) -17.73 / -22.32 (from the main line) | 298.129 -> 0.6239 / 0.6128 | 498.931 -> 0.6977 / 0.5238 | MODEL, not allowed |
| + every host-gather wait hidden (no lever exists) -20.206 / -115.542 | 277.923 -> 0.6692 / 0.6574 | 383.389 -> 0.9079 / 0.6817 | MODEL, no mechanism |
| (h) + drafts at the standalone step 0.947 / 0.6606 SH: draft after R1 + R4 is 126.132 / 172.067, standalone 121.549 / 164.086 | 277.923 - 4.583 = 273.340 -> 0.6804 / 0.6684 | 383.389 - 7.981 = 375.408 -> 0.9272 / 0.6962 | MODEL (DR1 / DR73 transfer); hand arithmetic |
| check: drafts FREE, joint + c as measured | 204.707 -> 0.9086 / 0.8925 | 361.174 -> 0.9638 / 0.7236 | DERIVED (structural check) |

Hand checks of the (h) line: 142.913 - 5.688 - 11.093 = 126.132; 126.132 - 121.549 = 4.583; 185.991 / 273.340 = 0.6804;
x 0.98234 = 0.6684. 190.208 - 4.318 - 13.823 = 172.067; 172.067 - 164.086 = 7.981; 348.082 / 375.408 = 0.9272; x 0.75086 =
0.6962.

Reading. (1) The verdict config is C 128 B 216; its ceiling with existing components is about **0.58 x the strongest**
(0.62 in the very optimistic variant). (2) Even with the forbidden kernels, hidden PCIe and standalone drafts it stays at
0.67. (3) At C 73 the own-SH ratio can approach 0.93-0.97, but C 73's own SH is only 0.751 of the strongest, so the
verdict ratio there tops out near 0.70. (4) Rev 5 gave 0.620 / 0.708 own SH for L1-L4; the levers L2 and L4 (C 128)
measured 0 or negative, so rev 6's existing-component ceiling (0.589 / 0.668) is below rev 5's.

## R6.6 What would need a NEW KERNEL (not allowed), and what is left without one

Needs a new kernel (or is not a kernel question):
- N1 fork efficiency at R 4 (register spills, 12 % active warps; launch settings cannot reach NS1's rate, I4B): 17.6 /
  22.2-23.1 ms per round (MODEL).
- N2 one fused draft-row attention over the window + the tentative buffer (no call B, no merge, no mask copy): <= 11.1 /
  13.8 ms (MODEL bound; E2 below tries the existing fork first).
- N3 fused small ops (union list build, tentative writes, chain, masks): 7,437 SMALL kernels per joint pass (about 232 per
  layer), 31.4 / 35.9 ms device (MEASURED class total; removable share UNKNOWN).
- N4 a device-side keep (block map on the GPU) so the keep pays at B 216: at most the 6.5 ms wait saving (MODEL), against
  12.0 ms of host bookkeeping today.
- NOT a kernel question: the dense FLOPs of 4B rows (item 2; physics floor 1.89x SH's dense term at peak) and the PCIe of
  verify-row misses (item 3; needs the verify rows' selections before the joint pass, an algorithm change).

Left without a new kernel (existing components; none tried yet; bounds overlap):
- E1 CUDA graph of the draft step: move the pool LRU stamp from a host int to a device tensor (torch ops), then capture
  as I4G: <= 5.7 / 4.3 ms (MODEL host share; at B 318 the joint graphs also removed device gaps, so the true value may be
  higher).
- E2 the draft row through the pinned fork blob f0dc0c18 at R 1 with the CB0 chain image: <= 11.1 / 13.8 ms; first an
  isolated R 1 timing of the fork (not measured).
- E3 union lists built once per pass where the layer does not change them: <= 11.9 / 12.5 ms device.
- E4 an off-the-shelf compiler (torch.compile / Inductor) over the small-op sections: <= the SMALL class 31.4 / 35.9 ms;
  it generates kernels, so the lead / human decides whether it counts as "no new kernel" (memory rule: try the SOTA tool
  before a custom kernel).
- E5 C 73 at its largest feasible B (318-350 after L60): not verdict-relevant (INFERRED: SH at B 351 is 2,628.6 tok/s =
  0.767 of the strongest; even 0.63 of that is 0.48 < 0.525 at C 128).
- E6 K 2 or K 4: +-0.5 % (MODEL, R6.3). Closed as a lever.
- E7 the keep at C 128: closed (loses with and without graphs).
With E1-E4 at their bounds the C 128 verdict ceiling is 0.578-0.616 x the strongest (R6.5).

## R6.7 Residuals and limits of this revision

1. The joint split is the eager probe split. For the graph arms the steady joint is below the probe total (-1.485 /
   -9.250 ms); no record splits the graph saving by term.
2. The C 73 SH split stays INFERRED (R4.1); the "x SH term" column at C 73 carries its errors.
3. The ceiling removals are upper bounds; R3 and the SMALL variant overlap; R1 and R4 both act on the draft phase.
4. The draft labels come from labelled probe passes (total 73.02 / 96.46 ms per step against steady 71.5 / 94.6).
5. The fork's cost at R 1 is not measured (E2 / R4 bound assumes zero extra cost).
6. The C 73 best arm has one run (k4b). Running at 16:55Z: i4g_h3b (i4gb replicate, nova05) and i4k_r4a (i4ka replicate,
   nova07), both with empty record directories; queued: 595_i4krep_b_n5, 596_i4mrep_n7 (SL:1097, :1116).
7. Correctness is unchanged: all I4 arms are bit-exact to L4PTFD; L4PTF is FAST, near-tie class (i3g: WF GREEDY 42 of
   6,400, largest gap 0.3125; TFC-SEL 9.15 %), NOT certified; no C 128 check cell (PL/SCORING_i3.md:42-47).
8. Memory: the usable ceiling is about 83.75 GB after SH in the same process (L60), not 84.25 GB.
9. L7 at C 73: the label medians do not add (residual -12.675 ms).
10. The standalone draft ratios (DR1 0.947, DR73 0.6606) come from other runs on nova05 (MODEL transfer).
11. The cells print "vs 3407.2" (an older denominator); this file uses the same-GPU 3,426.2 / 3,410.1.
12. The rev-5 to rev-6 comparisons cross jobs and GPUs (nova07 r4a / nova05 g3a); the I3 replicate spread is <= 0.3 % in
   ratio (SL:916-919).

## R6.8 Method and commands

Read with the Read tool: SL :860-1119; PL/CHECKPOINTS.md; NG; rev 5 (:591-710, :812-1036); stdout files of the runs
named (via grep). CPU scripts (written with the Write tool under /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/ana_rev6/):
s1_keys.py, s2_profile.py, s3_derive.py; each run with `srun -p amd -c 4 --mem=16G -t 60 apptainer exec --bind
/mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif /venv/nosa/bin/python -I <script>`, output to
cpu-logs/ana_rev6_s{1,2,3}.txt (s3 ran twice: the first run stopped at the i4m record, which has no SPEC block). Login
shell: ls, wc, grep, cut, date only. No interpreter on the login node, no job, no code edit.
