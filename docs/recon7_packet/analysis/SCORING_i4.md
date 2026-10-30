# SCORING_i4: rung I4 (twelve families, I4A to I4DC), each family against its stamped registration

Written 2026-10-30T09:55:28Z (from `date -u`). Writer: the I4 scoring agent, read-only. Style: ASD-STE100.
This file closes the gap that STATUS_LOG.md:1300-1303 names: since 2026-10-29T07:48Z the lead scored I4 only in the log.

Labels. MEASURED = a value in a record (stdout PREDICTION / GATE / REPORT line, summary.json field, sacct.txt, exit.txt,
command.txt, git log of the tree). DERIVED = arithmetic on MEASURED values (the operands are shown, or the CPU script
computed it: LOG = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/ana_rev7_a2.txt, script
/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/ana_rev7/a2_score.py, srun on amd). CITED = a value from another plan file
(file:line given). INFERRED = a reading that no record states.

Abbreviations. PL = /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan. R = /mnt/beegfs/ojerbi/r2-runs/spec. SL =
PL/STATUS_LOG.md. C = CONFIRMED, F = REFUTED (the cell's word: REFUTED), RP = REPORT (registered None, or no value).
n05 = nova05 (holder 2195575, GPU f31fc723); n07 = nova07 (holder 2195576, GPU 33cfb929). "/SH" = median steady
committed tok/s of the arm / median steady tok/s of the SAME-JOB SH arm. "/strong" = the arm's median / the same-GPU
strongest cached NOSA steady: n05 3,426.2, n07 3,410.1 (PL/CHECKPOINTS.md:9-11). "/recheck" = the arm's median / the
same-day re-check of that strongest: n05 3,421.6 (R/../gstep/gstep_e3r_2195575_gs1r, gs steady_tok_s MEASURED 3,421.6),
n07 3,402.1 (gstep_e4r_2195576_gs1r; the record prints 3,402.2 = 216 / 0.063489 s, DERIVED 3,402.16; the lead's log uses
3,402.1; the difference is 0.003 % and changes no ratio at 4 digits). Both re-checks: 6 / 6 GS keys CONFIRMED (LOG:10-11).

Note on the cells' own "VS-STRONGEST" line: every I4 cell divides by 3,407.2 (an older single denominator). This file does
not use that line. It uses the same-GPU values above (as LEDGER_rev6.md R6.7 item 11).

## 0. Summary

| family (lever) | registration STAMP / commit (UTC) | first cell start | timing runs scored (both batches) | keys C / F / RP | best arm, /SH (replicates) | best arm, /strong (replicates) |
|---|---|---|---|---|---|---|
| I4A (lever 1, lean draft) | 04:25:58Z / b8914f3 04:26:23Z | 06:41:10Z | 4 | 40 / 0 / 0 | L4PTFD 0.5207, 0.5207 / 0.5975, 0.5980 | 0.5103-0.5113 / 0.4490-0.4499 |
| I4B (lever 2, fork launch sweep) | no registration; PICK_RULE in 7bf83d1 07:40:29Z | 08:44:56Z | 2 sweep cells | none registered | none (default launch kept) | - |
| I4C (lever 3, profile) | 08:57:45Z / 3aef060 08:58:05Z | 10:06:01Z | 3 | 34 / 5 / 0 | L4PTFD (REF) 0.5206, 0.5215 / 0.5982 | 0.5123-0.5128 / 0.4499 |
| I4D (lever 4, store keep) | 09:41:32Z / 99c231e 09:41:54Z | 10:48:59Z | 3 timing + 2 accounting | 51 / 21 / 13 | B 216 L4PTFD 0.5212 (keep loses); B 318 L4PTFKW 0.6205, 0.6213 | 0.5137 / 0.4664-0.4668 |
| I4G (joint CUDA graphs) | 12:00:39Z / d7997e6 12:01:00Z | 12:35:25Z | 6 | 45 / 9 / 0 | L4PTFDG 0.5333-0.5345 / 0.6053-0.6058 | 0.5238-0.5247 / 0.4547-0.4554 |
| I4K (graphs + keep) | 14:57:28Z / 70059e5 14:57:45Z | 15:18:44Z | 5 | 46 / 9 / 10 | B 216 GKW 0.5265, 0.5300 (below G); B 318 L4PTFDGKW 0.6306-0.6307 | 0.5189-0.5193 / 0.4736-0.4743 |
| I4M (matched memory) | 12:48:07Z / 30369eb 12:48:24Z | matched 13:18:19Z (probes before the stamp, by design) | 2 C 128 + 1 C 73 FAILED (OOM) | 6 / 0 / 3 | L4PTFD DIAG 0.5237, 0.5240 | MAIN 0.5152, 0.5143 |
| I4U (union hoist) | 18:16:01Z / 832dca5 18:16:15Z | 18:53:15Z | 5 | 35 / 0 / 0 | L4PTFDU 0.5281-0.5296 / 0.6013, 0.6020 | 0.5169-0.5212 / 0.4517, 0.4524 |
| I4H (draft graphs) | 18:47:25Z / fc6c680 18:47:41Z | 19:36:50Z | 6 | 54 / 6 / 0 | L4PTFDGD 0.5379-0.5419 / 0.6100-0.6111 | 0.5295-0.5314 / 0.4584-0.4598 |
| I4S (stack) | 2026-10-29T22:50:04Z / 8f236f8 22:50:26Z | 23:26:04Z | 7 | 84 / 0 / 0 | L4PTFDGUD 0.5427-0.5456 / L4PTFDGUDKW 0.6381-0.6389 | 0.5336-0.5347 / 0.4798-0.4800 |
| I4CC (compiler, joint) | 2026-10-30T04:10:18Z / d75a946 04:10:35Z | 04:22:26Z | 7 | 78 / 2 / 4 | L4PTFDGUDC 0.5592-0.5636 / L4PTFDGUDKWC 0.6523-0.6534 | 0.5508-0.5518 / 0.4903-0.4909 |
| I4DC (compiler, draft) | 07:27:31Z / 2e9e497 07:27:47Z | 07:45:31Z | 2 (2 pending) | 23 / 0 / 1 | L4PTFDGUDCD 0.5700 / L4PTFDGUDKWCD 0.6627 (n = 1 each) | 0.5616 / 0.4982 |
| **all** | every stamp and every registration commit predates the first cell of its family | - | 55 timing / sweep / accounting / matched runs + 2 probes + 21 miniatures = 78 finished (+1 running) | **496 / 52 / 31** | - | - |

All dates are 2026-10-29 unless the cell says 2026-10-30. Key counts are key x run evaluations in timing and accounting
cells (MEASURED: `grep -c "^PREDICTION.*CONFIRMED"` / `REFUTED` per stdout; RP counted by hand from the PREDICTION lines).
The cross-reference REPORT lines (for example "I4G-VS-D REPORT") and the "VS-STRONGEST REPORT" lines are not registered
keys and are not counted.

**Best arm per batch now (MEASURED, median steady, LOG:383-471 and :144-161):**
- C 128 B 216: L4PTFDGUDCD 1,924.1 tok/s = **0.5700 x same-job SH = 0.5616 x strongest (n05) = 0.5623 x the same-day
  re-check** (i4dc d3a, one run; the swapped replicate i4dc r4a on n07 is running). Replicated runner-up: L4PTFDGUDC
  (I4CC) n = 3: /SH 0.5592-0.5636 (spread 0.78 %), /strong 0.5508-0.5518, tok/s 1,879.1-1,890.5.
- C 73 B 318: L4PTFDGUDKWCD 1,699.0 tok/s = **0.6627 x SH = 0.4982 x strongest (n07) = 0.4994 x re-check** (i4dc d4b, one
  run; replicate r3b queued, no record). Replicated runner-up: L4PTFDGUDKWC (I4CC) n = 4: /SH 0.6523-0.6534, /strong
  0.4903-0.4909.
- The best feasible config under the 84 GB cap stays C 128 B 216 (peak with prefill 83.900 GB for the compiled arms).

**Integrity in one list (details in each section and in section 13):**
1. 77 of 78 finished runs: exit 0, sacct COMPLETED 0:0, 0 MUST failures. One run FAILED: i4m n373 (C 73 B 351), exit 2,
   sacct 2195575.24 FAILED 2:0, MUST FAIL L4PTFD-RUN (CUDA OOM, 126 MiB request) and MUST FAIL EDGE (no peak). Its 3
   registered keys have no value (RP). Lesson L60. Not hidden: the lead logged it (SL:1057-1060).
2. Every registration STAMP and every registration commit predates the first cell of its family. The cell start that
   sacct gives (utc_end of exit.txt minus the sacct elapsed time) equals command.txt utc_start to 0 s in 77 runs and 1 s
   in one (i4s p3a). No registration came after a run.
3. The STATUS_LOG key counts are one too high in 11 entries (section 13, item 3). The records are correct; the log is not.
4. Pending (no exit.txt at 09:55Z): i4dc r4a (n07, C 128, started 2026-10-30T09:16:26Z). Not started (no directory):
   i4dc r3b (619_i4dcrep_b_n5) and i4cc y4a (620_i4ccrep4_a_n7). i4cc y3b ended at 09:53:07Z during this work and is scored.

**Correctness (every family):** every I4 arm passes EQF against L4PTFD in the same job: committed tokens 0 unequal of
110,592 (C 128) / 162,816 (C 73), fingerprints max and sum equal, 227 state checksums equal, acceptance record equal
(MEASURED, the EQF GATE lines). EQF is MUST in the I4G, I4K, I4U, I4H, I4S, I4CC and I4DC timing cells; it is REPORT
(PASS) in the I4A and I4D timing cells and MUST in their miniatures. L4PTFD is bit-exact to L4PTF (I4A EQF). So every I4
arm has the I3 correctness class: FAST, near-tie class, NOT certified (PL/SCORING_i3.md; i3g: WF GREEDY 42 of 6,400,
largest gap 0.3125; TFC-SEL 9.15 %; no C 128 check cell). Token agreement with SH (MEASURED, AGREE keys): 0.403411 (C 128),
0.387904 (C 73), identical in every run.

Same-job SH row, all timing runs (MEASURED, LOG SH lines): C 128 B 216 SH median 3,334.3-3,376.8 tok/s (step 63.965-64.782
ms), SH / strongest 0.9778-0.9856; C 73 B 318 SH 2,560.2-2,579.7 tok/s (step 123.270-124.210 ms), SH / strongest
0.7508-0.7529. The /SH ratio therefore overstates the /strong ratio by about 1.5-2.2 % at C 128 and by 33 % at C 73.

## 1. I4A (lever 1: lean one-row draft step; arm L4PTFD)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4a. Registration: benchmarks/Efficiency/r2_i4a_cell.py at b8914f3,
`PRED_STAMP_I4A = "2026-10-29T04:25:58Z"`, block `PRED_I4A_BY_B` (MEASURED, `git show b8914f3`). Run commit acbe433 =
b8914f3 + a test-only change (tests_cpu/test_i4a.py; the cell file is unchanged, `git show --stat acbe433`).

Integrity. Runs (sacct step, GPU, start from sacct): q3m B 8 miniature 2195575.5 n05 06:41:10Z; q3a C 128 2195575.6 n05
06:47:15Z; q4b C 73 2195576.5 n07 06:43:32Z; k3b C 73 2195575.13 n05 09:12:13Z; k4a C 128 2195576.9 n07 09:37:44Z. All
exit 0, COMPLETED 0:0. MUST: q3m 40 / 40, timing cells 21 / 21 each, 0 failed. STAMP 04:25:58Z and commit 04:26:23Z <
06:41:10Z: registration first. Steady set: the capture-round fix (L61) is not in this cell; it has no graph arm, so no
capture round exists.

Registered keys (value per run; band; verdict). C 128 (q3a n05 / k4a n07):

| key | q3a | k4a | band | verdict |
|---|---|---|---|---|
| I3-RATIO (L4PTF / SH) | 0.5131 | 0.5120 | (0.49, 0.54) | C, C |
| I4A-RATIO (L4PTFD / SH) | 0.5207 | 0.5207 | (0.51, 0.55) | C, C |
| I4A-VS-I3 | 1.0148 | 1.0169 | (1.01, 1.06) | C, C |
| I4A-DRAFT-STEP-SH | 1.1183 | 1.1182 | (1.02, 1.13) | C, C |
| I4A-DRAFT-SAVING (ms per round) | 6.198 | 6.847 | (5.0, 18.0) | C, C |
| I4A-JOINT-RATIO | 1.0032 | 1.0024 | (0.98, 1.02) | C, C |
| I4A-PEAK-GB | 83.891 | 83.891 | (83.5, 84.25) | C, C |
| I4A-AGREE | 0.403411 | 0.403411 | >= 0.25 | C, C |
| I4A-EQ-I3 | True | True | True | C, C |
| I4A-ABOVE-I3 | 1,751.9 > 1,726.3 | 1,740.2 > 1,711.2 | True | C, C |

C 73 (q4b n07 / k3b n05): I3-RATIO 0.5929 / 0.5927 (0.57, 0.61) C, C; I4A-RATIO 0.5980 / 0.5975 (0.59, 0.62) C, C;
VS-I3 1.0087 / 1.0082 (1.005, 1.04) C, C; DRAFT-STEP-SH 0.7643 / 0.7647 (0.70, 0.78) C, C; DRAFT-SAVING 5.658 / 5.141
(5.0, 18.0) C, C (k3b sits 0.14 ms above the low edge); JOINT-RATIO 1.0012 / 1.0009 C, C; PEAK 77.685 (77.3, 78.1) C, C;
AGREE 0.387904 C, C; EQ-I3 True C, C; ABOVE-I3 1,534.1 > 1,520.9 / 1,538.4 > 1,525.9 C, C. Total 40 C, 0 F.

Arm L4PTFD (MEASURED median; DERIVED ratios, LOG:386-389):

| batch | run, GPU | same-job SH median | L4PTFD median | /SH | /strong | /recheck |
|---|---|---|---|---|---|---|
| C 128 B 216 | q3a n05 | 3,364.6 | 1,751.9 | 0.5207 | 0.5113 | 0.5120 |
| C 128 B 216 | k4a n07 | 3,341.9 | 1,740.2 | 0.5207 | 0.5103 | 0.5115 |
| C 73 B 318 | q4b n07 | 2,565.4 | 1,534.1 | 0.5980 | 0.4499 | 0.4509 |
| C 73 B 318 | k3b n05 | 2,574.6 | 1,538.4 | 0.5975 | 0.4490 | 0.4496 |

Replicate spread of /SH: 0.01 % (C 128), 0.08 % (C 73) (DERIVED, LOG). Correctness: L4PTFD-EQF REPORT PASS in every timing
cell (tokens, fingerprints, state checksums vs L4PTF), MUST in q3m. Refuted keys: none. Reading: lever 1 delivered its low
edge (+1.5 % / +0.85 %, DERIVED from VS-I3).

## 2. I4B (lever 2: the fork's launch settings; sweep cells, arm L4PTFL)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4b 7bf83d1 (from i3 c4c64b2; fork kernel body unchanged). No registration block: the
lead registered the PICK_RULE in the source commit 7bf83d1 (07:40:29Z) before the first sweep cell (08:44:56Z) (SL:933-941,
CITED; commit time MEASURED, git log). No timing cells ran (by the rule).

Integrity. Runs: w3m B 8 2195575.11 n05 08:44:56Z; w3sa C 128 B 216 sweep 2195575.12 n05 08:54:34Z; w4sb C 73 B 318 sweep
2195576.8 n07 09:14:29Z. All exit 0, COMPLETED 0:0. MUST 53 / 15 / 15 PASS, 0 failed. REPORT FAIL 3 per cell: the T32
candidates W4S3T32, W4S4T32, W8S4T32 fail EQ (C 128: 3,673,701 unequal of 226,492,416 elements, max 0.125; C 73:
5,494,585 of 333,447,168). This is the rule's eligibility screen, not a defect.

Results (MEASURED, GATE SWP-PICK): C 128 B 216 pick = the default W4S3T64, 1.13063 ms per layer (repeat 1.11602, margin
0.02584), saving 0; C 73 B 318 pick = the default, 1.60154 ms per layer (repeat 1.58885, margin 0.01585), saving 0; SWP-BITS
PASS (default and repeat bitwise equal to the pipeline's call). B 8 pick W8S4T64 0.36422 vs 0.44717 ms (small batch only;
not a ladder batch). Miniature arms (B 8, not a verdict): L4PTF /SH 0.1516, L4PTFL 0.1455.

Verdict. Lever 2 closed by measurement at both ladder batches: 0 ms. No arm ratio exists at B 216 / 318 (no timing arm).
Correctness: no new arm reached a timing cell.

## 3. I4C (lever 3 measure: the profiler window; arms L4PTFD = REF, L4PTFDP = profiled)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4c. Registration: r2_i4c_cell.py at 3aef060, `PRED_STAMP_I4C = "2026-10-29T08:57:45Z"`,
`PRED_I4C_BY_B`, with the DECISION RULE "build the CUDA-graph trial only if 1 - I4C_busy_frac > 0.05 at either batch".

Integrity. Runs: c3m B 8 2195575.14 n05 10:06:01Z; c3a C 128 2195575.15 n05 10:12:07Z; c4m B 8 2195576.10 n07 10:14:39Z;
c4b C 73 2195576.11 n07 10:20:54Z; e4a C 128 replicate 2195576.15 n07 13:09:49Z. All exit 0, COMPLETED 0:0; MUST 26 / 26
each, 0 failed. STAMP 08:57:45Z < 10:06:01Z.

Registered keys, C 128 (c3a n05 / e4a n07) and C 73 (c4b n07):

| key | c3a | e4a | band B 216 | c4b | band B 318 | verdicts |
|---|---|---|---|---|---|---|
| I4C-REF-RATIO | 0.5206 | 0.5215 | (0.50, 0.54) | 0.5982 | (0.58, 0.62) | C C / C |
| I4C-BUSY-FRAC | 0.9486 | 0.9511 | (0.90, 0.98) | **0.9958** | (0.90, 0.98) | C C / **F** |
| I4C-DRAFT-BUSY-FRAC | 0.9602 | 0.9594 | (0.80, 0.98) | 0.9773 | (0.80, 0.99) | C C / C |
| I4C-JOINT-GPU-OPS | 9,076.5 | 9,076.5 | (5,000, 11,000) | 9,076.5 | same | C C / C |
| I4C-JOINT-SMALL-MS | 32.778 | 34.194 | (20, 75) | 37.348 | (25, 90) | C C / C |
| I4C-L5-DEVICE-FRAC | 0.8071 | 0.8289 | (0.30, 0.90) | 0.8747 | same | C C / C |
| I4C-L7-GAP-MS | 4.648 | 5.133 | (4.0, 7.5) | 5.098 | (4.0, 8.0) | C C / C |
| I4C-L7-UNCOVERED-MS | 6.773 | 6.816 | (5, 10) | 8.623 | (5, 12) | C C / C |
| I4C-L7-RESIDUAL-MS | 1.019 | 0.562 | (-2, 2) | **-12.675** | (-2, 2) | C C / **F** |
| I4C-DRAFT-ROW-MS | 40.382 | 40.543 | (30, 60) | 52.579 | (25, 75) | C C / C |
| I4C-DRAFT-OUTSIDE-MS | **0.375** | **0.395** | (0.5, 8.0) | **0.450** | (0.5, 8.0) | **F F / F** |
| I4C-PROF-NEUTRAL | 1.0018 | 0.9983 | (0.98, 1.02) | 1.0012 | same | C C / C |
| I4C-PEQ | True | True | True | True | True | C C / C |

Total 34 C, 5 F. Refuted keys: DRAFT-OUTSIDE-MS (all three runs): the draft phase outside the two one-row passes is 0.38-0.45
ms, below the 0.5 ms floor: smaller than modelled. BUSY-FRAC at B 318: the joint pass is device bound (host share 0.4 %).
L7-RESIDUAL at B 318: the label medians do not add at B 318 (-12.7 ms).

Decision rule (MEASURED inputs, DERIVED margin): 1 - busy = 0.0514 (c3a n05) fired the rule; the n07 replicate gives
1 - 0.9511 = 0.0489, which is BELOW 0.05. The rule fired on one of the two C 128 measurements only (the lead's log says
"fired, narrowly", SL:994; the replicate e4a ran after the I4G builder started, SL:1008-1009). The graph trial later won on
its own registered keys (section 5), so the outcome does not depend on this.

Arms (LOG:394-397): L4PTFD (REF) C 128 1,756.8 / 1,746.9 tok/s, /SH 0.5206 / 0.5215, /strong 0.5128 (n05) / 0.5123 (n07),
/recheck 0.5134 / 0.5135; C 73 1,534.1, /SH 0.5982, /strong 0.4499, /recheck 0.4509. L4PTFDP (profiled, not a speed arm)
/SH 0.5207 / 0.5226 / 0.5984. Same-job SH: 3,374.7 (c3a), 3,349.5 (e4a), 2,564.6 (c4b). Correctness: L4PTFDP-PEQ MUST PASS
(profiled arm = REF bit for bit).

## 4. I4D (lever 4: the store keep; arms L4PTFK and L4PTFKW; accounting cells i4dc / i4dd)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4d. Registration: r2_i4d_cell.py at 99c231e, `PRED_STAMP_I4D = "2026-10-29T09:41:32Z"`,
`PRED_I4D_BY_B` (I4D_above_I4A at B 216 = None, RP).

Integrity. Runs: d3m B 8 2195575.16 n05 10:48:59Z; d3c C 128 accounting (128 steps, meter on) 2195575.17 n05 10:55:48Z;
d3a C 128 timing 2195575.18 n05 11:19:39Z; d4m B 8 2195576.12 n07 11:15:59Z; d4d C 73 accounting 2195576.13 n07 11:23:02Z;
d4b C 73 timing 2195576.14 n07 11:58:10Z; f3b C 73 timing replicate 2195575.26 n05 14:08:31Z. All exit 0, COMPLETED 0:0;
MUST 70 / 13 / 34 / 70 / 13 / 34 / 34, 0 failed. STAMP 09:41:32Z < 10:48:59Z. No C 128 timing replicate exists (one run,
d3a).

Accounting cells (registered keys evaluated there):

| key | d3c (B 216) | band | d4d (B 318) | band | verdicts |
|---|---|---|---|---|---|
| I4A-RATIO | 0.4686 | (0.51, 0.55) | 0.5420 | (0.59, 0.62) | F / F (scope: see below) |
| I4D-REPEAT-SHARE | 0.1348 | (0.05, 0.40) | 0.1503 | (0.05, 0.40) | C / C |
| I4D-LISTED-PASS | 14,253.02 | (3,000, 12,000) | 46,437.95 | (6,000, 30,000) | F / F |
| I4D-WIN-PREV-SHARE | 0.3496 | (0.2, 0.7) | 0.3296 | (0.05, 0.25) | C / F |
| I4D-WIN-PREV-PASS | 3,971.02 | (2,000, 6,400) | 13,287.65 | (4,000, 11,000) | C / F |

Timing cells, C 128 (d3a) and C 73 (d4b n07 / f3b n05):

| key | d3a | band B 216 | d4b | f3b | band B 318 | verdicts |
|---|---|---|---|---|---|---|
| I4A-RATIO | 0.5212 | (0.51, 0.55) | 0.5990 | 0.5990 | (0.59, 0.62) | C / C C |
| I4D-HIT-SHARE | 0.2692 | (0.2, 0.6) | 0.1619 | 0.1619 | (0.15, 0.6) | C / C C |
| I4D-RATIO (L4PTFK / SH) | 0.5145 | (0.507, 0.553) | 0.6033 | 0.6035 | (0.590, 0.629) | C / C C |
| I4D-VS-I4A | **0.9872** | (0.995, 1.006) | 1.0072 | 1.0075 | (1.000, 1.014) | **F** / C C |
| I4D-JOINT-SAVING (ms) | **-4.446** | (-2, 2) | 5.425 | 5.626 | (0, 8) | **F** / C C |
| I4D-WAIT-SAVING (ms) | 4.008 | (1, 6) | 16.754 | 16.536 | (3, 18) | C / C C |
| I4D-NEW-SAVING | 0.1519 | (0.05, 0.30) | 0.0879 | 0.0879 | (0.02, 0.12) | C / C C |
| I4D-PEAK-GB | 83.891 | (83.5, 84.25) | 77.685 | 77.685 | (77.3, 78.15) | C / C C |
| I4D-AGREE | 0.403411 | >= 0.25 | 0.387904 | 0.387904 | >= 0.25 | C / C C |
| I4D-EQ-I4A | True | True | True | True | True | C / C C |
| I4D-ABOVE-I4A | 1,737.4 < 1,759.9 | None | 1,546.3 > 1,535.2 | 1,553.2 > 1,541.7 | True | RP / C C |
| I4DW-WIN-HITS-PASS | 3,091.47 | (2,000, 7,000) | **14,830.58** | **14,830.58** | (4,000, 14,000) | C / **F F** |
| I4DW-RATIO (L4PTFKW / SH) | 0.5140 | (0.508, 0.557) | 0.6205 | 0.6213 | (0.595, 0.637) | C / C C |
| I4DW-VS-I4A | **0.9862** | (0.997, 1.012) | **1.0360** | **1.0373** | (1.009, 1.027) | **F** / **F F** |
| I4DW-JOINT-SAVING (ms) | **-5.522** | (-1, 4) | **21.972** | **22.460** | (5, 15) | **F** / **F F** |
| I4DW-WAIT-SAVING (ms) | 6.492 | (3, 9) | **39.248** | **38.879** | (9, 30) | C / **F F** |
| I4DW-NEW-SAVING | 0.3362 | (0.15, 0.45) | **0.2339** | **0.2339** | (0.06, 0.20) | C / **F F** |
| I4DW-PEAK-GB | 83.891 | as I4D | 77.685 | 77.685 | as I4D | C / C C |
| I4DW-AGREE | 0.403411 | >= 0.25 | 0.387904 | 0.387904 | >= 0.25 | C / C C |
| I4DW-EQ-I4A | True | True | True | True | True | C / C C |
| I4DW-ABOVE-I4A | **1,735.7 < 1,759.9** | True | 1,590.4 > 1,535.2 | 1,599.2 > 1,541.7 | True | **F** / C C |

The four accounting keys print "REPORT: no value" in the timing cells (the meter is off there): 4 RP per timing cell.
Totals: 51 C, 21 F, 13 RP.

Refuted keys, one line each:
- I4A-RATIO in d3c / d4d: a SCOPE MISMATCH, not a lever result. The accounting cell runs 128 steps with the meter on; the
  band was set for the 512-step timing cells. The same key is C in d3a, d4b, f3b.
- I4D-LISTED-PASS (both batches): the union lists more store blocks per pass than modelled (14,253 / 46,438).
- I4D-WIN-PREV-SHARE / -PASS at B 318: more of row 0's window was already in the previous store than modelled (0.33).
- I4D-VS-I4A, I4D-JOINT-SAVING, I4DW-VS-I4A, I4DW-JOINT-SAVING, I4DW-ABOVE-I4A at B 216: the keep LOSES; the joint pass gets
  4.4 / 5.5 ms slower (host bookkeeping, about 3x the builder's op cost; SL:1017-1024, CITED).
- I4DW-WIN-HITS-PASS, -VS-I4A, -JOINT-SAVING, -WAIT-SAVING, -NEW-SAVING at B 318 (both runs): ABOVE the band: the window
  keep saves more than modelled.

Arms (LOG:414-419; d3c / d4d excluded from the speed rows):

| batch | run | SH | L4PTFD /SH, /strong | L4PTFK /SH, /strong | L4PTFKW /SH, /strong, /recheck |
|---|---|---|---|---|---|
| C 128 | d3a n05 | 3,376.8 | 0.5212, 0.5137 | 0.5145, 0.5071 | 0.5140, 0.5066, 0.5073 |
| C 73 | d4b n07 | 2,562.9 | 0.5990, 0.4502 | 0.6033, 0.4534 | 0.6205, 0.4664, 0.4675 |
| C 73 | f3b n05 | 2,573.8 | 0.5990, 0.4500 | 0.6035, 0.4533 | 0.6213, 0.4668, 0.4674 |

Correctness: L4PTFK-EQF and L4PTFKW-EQF REPORT PASS in the timing cells (EQ keys C); MUST in d3m / d4m (CAP, KEEP, EQF,
ACCT-EQ). Verdict: lever 4 = +3.6 % at C 73 (KW), -1.4 % at C 128.

## 5. I4G (CUDA graphs over the joint pass's union + fork section; arm L4PTFDG)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4g. Registration: r2_i4g_cell.py at d7997e6, `PRED_STAMP_I4G = "2026-10-29T12:00:39Z"`,
`PRED_I4G_BY_B` (B 318 = a no-regression check).

Integrity. Runs: g3m B 8 2195575.22 n05 12:35:25Z; g3a C 128 2195575.23 n05 12:41:37Z; g4m B 8 2195576.16 n07 13:47:02Z;
g4b C 73 2195576.17 n07 13:53:23Z; h4a C 128 2195576.18 n07 14:48:18Z; h3b C 73 2195575.29 n05 16:13:20Z; x3a C 128
2195575.31 n05 18:17:20Z; x4b C 73 2195576.23 n07 17:58:24Z. All exit 0, COMPLETED 0:0; MUST 44 (B 8) / 26 (timing), 0
failed. STAMP 12:00:39Z < 12:35:25Z. L61: these cells count the capture round (round 1) as steady (steady 181, no
exclusion; LOG). The median is not affected (NOTES_graph_mean_gap.md); the MEAN and EFFECTIVE of L4PTFDG here carry the
6.2-6.6 s capture and are not comparable with L4PTFD's.

Registered keys:

| key | g3a | h4a | x3a | band B 216 | g4b | h3b | x4b | band B 318 |
|---|---|---|---|---|---|---|---|---|
| I4G-JOINT-SAVING (ms) | 6.962 | 7.364 | 7.347 | (3.0, 10.5) | **7.726** | **7.271** | **7.758** | (-2.0, 3.0) |
| I4G-JOINT-RATIO | 0.9661 | 0.9644 | 0.9643 | (0.95, 0.99) | **0.9799** | **0.9810** | **0.9798** | (0.99, 1.01) |
| I4G-VS-D | 1.0191 | 1.0197 | 1.0209 | (1.008, 1.030) | **1.0134** | **1.0144** | **1.0127** | (0.995, 1.010) |
| I4G-RATIO | 0.5342 | 0.5345 | 0.5333 | (0.52, 0.54) | 0.6053 | 0.6058 | 0.6055 | (0.59, 0.61) |
| I4G-DRAFT-RATIO | 1.0014 | 1.0010 | 1.0015 | (0.98, 1.02) | 0.9999 | 1.0016 | 1.0022 | (0.98, 1.02) |
| I4G-PEAK-GB | 83.891 | 83.891 | 83.891 | (83.85, 83.95) | 77.685 | 77.685 | 77.685 | (77.60, 77.80) |
| I4G-POOL-MB | 14.680 | 14.680 | 14.680 | (2, 100) | 25.166 | 25.166 | 25.166 | (3, 150) |
| I4G-CAPTURE-S | 6.240 | 6.182 | 6.235 | (0.5, 15) | 6.568 | 6.625 | 6.588 | (0.5, 15) |
| I4G-EQ | True | True | True | True | True | True | True | True |

Verdicts: B 216 all C (27); B 318 JOINT-SAVING, JOINT-RATIO, VS-D F in all three runs (ABOVE band: the graphs pay at B 318
too; the saving is device-side gaps between kernels, not host exposure), the rest C. Total 45 C, 9 F.

Arm L4PTFDG (LOG:434-437): C 128 tok/s 1,797.8 / 1,788.0 / 1,794.5; /SH 0.5342 / 0.5345 / 0.5333 (spread 0.22 %);
/strong 0.5247 (n05) / 0.5243 (n07) / 0.5238 (n05); /recheck 0.5254 / 0.5256 / 0.5245. C 73 tok/s 1,551.1 / 1,560.3 /
1,550.5; /SH 0.6053 / 0.6058 / 0.6055 (spread 0.08 %); /strong 0.4549 / 0.4554 / 0.4547. Same-job SH: 3,365.7, 3,345.1,
3,364.7 (C 128); 2,562.6, 2,575.8, 2,560.6 (C 73). Correctness: L4PTFDG-EQF MUST PASS in every run.

## 6. I4K (graphs + keep + window; arm L4PTFDGKW)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4k. Registration: r2_i4k_cell.py at 70059e5, `PRED_STAMP_I4K = "2026-10-29T14:57:28Z"`,
`PRED_I4K_BY_B` (I4K_vs_KW None at B 216, I4K_vs_G None at B 318, I4K_above_best None at both: 2 RP per timing cell).

Integrity. Runs: k3m B 8 2195575.27 n05 15:18:44Z; k3a C 128 2195575.28 n05 15:25:35Z; k4m B 8 2195576.19 n07 15:25:15Z;
k4b C 73 2195576.20 n07 15:32:22Z; r4a C 128 2195576.21 n07 16:44:18Z; r3b C 73 2195575.30 n05 17:07:09Z; t3b C 73
2195575.38 n05 22:35:35Z. All exit 0, COMPLETED 0:0; MUST 74 / 41 / 74 / 40 / 41 / 40 / 40, 0 failed. STAMP 14:57:28Z <
15:18:44Z. L61 not fixed in this cell (capture round in steady).

Registered keys:

| key | k3a | r4a | band B 216 | k4b | r3b | t3b | band B 318 |
|---|---|---|---|---|---|---|---|
| I4K-JOINT-SAVING (ms) | 4.228 | 4.840 | (2.0, 9.5) | **32.063** | 30.885 | **31.101** | (17.0, 31.0) |
| I4K-JOINT-RATIO | 0.9795 | 0.9767 | (0.954, 0.990) | **0.9167** | **0.9190** | **0.9185** | (0.919, 0.956) |
| I4K-VS-D | 1.0123 | 1.0161 | (1.006, 1.028) | **1.0564** | 1.0531 | 1.0532 | (1.030, 1.056) |
| I4K-VS-G | 0.9930 | 0.9941 | (0.986, 1.005) | RP | RP | RP | None |
| I4K-VS-KW | RP | RP | None | **1.0170** | **1.0160** | **1.0152** | (0.994, 1.015) |
| I4K-RATIO | 0.5265 | 0.5300 | (0.522, 0.540) | 0.6307 | 0.6306 | 0.6307 | (0.612, 0.638) |
| I4K-WAIT-SAVING (ms) | 7.352 | 7.277 | (4.0, 9.0) | 39.168 | 39.027 | 38.932 | (30.0, 48.0) |
| I4K-DRAFT-RATIO | 1.0018 | 0.9984 | (0.98, 1.02) | 1.0015 | 1.0007 | 1.0032 | (0.98, 1.02) |
| I4K-PEAK-GB | 83.891 | 83.891 | (83.85, 83.95) | 77.685 | 77.685 | 77.685 | (77.60, 77.80) |
| I4K-POOL-MB | 35.652 | 35.652 | (5, 150) | 29.360 | 29.360 | 29.360 | (5, 200) |
| I4K-CAPTURE-S | 6.791 | 6.608 | (0.5, 20) | 7.193 | 7.128 | 7.144 | (0.5, 20) |
| I4K-EQ | True | True | True | True | True | True | True |
| I4K-ABOVE-BEST | RP: 1,777.9 vs G 1,790.5 | RP: 1,771.0 vs G 1,781.6 | None | RP: 1,614.9 > KW 1,587.9 | RP: 1,625.1 > 1,599.5 | RP: 1,623.9 > 1,599.6 | None |

Totals: 46 C, 9 F, 10 RP. Refuted keys (all B 318, all ABOVE the band; the graph and the keep add more than modelled):
JOINT-SAVING (k4b 32.063, t3b 31.101; r3b 30.885 is C, so the verdict FLIPS between replicates: the band edge 31.0 lies
inside the replicate spread 30.9-32.1); JOINT-RATIO (all three; r3b prints 0.9190 against the edge 0.919, so its unrounded
value is below the edge, INFERRED from the verdict); VS-D (k4b only, 1.0564 vs 1.056; again a flip); VS-KW (all three).

Arm L4PTFDGKW (LOG:450-455): C 128 1,777.9 / 1,771.0 tok/s, /SH 0.5265 / 0.5300, /strong 0.5189 (n05) / 0.5193 (n07): BELOW
L4PTFDG in the same jobs (/SH 0.5303 / 0.5331; GKW / G 0.9930 / 0.9941). C 73 1,614.9 / 1,625.1 / 1,623.9, /SH 0.6307 /
0.6306 / 0.6307 (spread 0.01 %), /strong 0.4736 (n07) / 0.4743 (n05) / 0.4740 (n05), /recheck 0.4747 / 0.4750 / 0.4746.
Same-job SH: 3,376.7 (k3a), 3,341.7 (r4a), 2,560.5 (k4b), 2,577.0 (r3b), 2,574.9 (t3b). Correctness: L4PTFDGKW-EQF MUST
PASS; KEEP-EQ, CAP, COUNTS PASS in k3m / k4m.

## 7. I4M (matched-memory verdict of L4PTFD; cells i4m128 / i4m73, probes i4mp73 / i4mp128)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4m. Registration: r2_i4m_cell.py at 30369eb, `PRED_STAMP_I4M = "2026-10-29T12:48:07Z"`,
`FIT_B_73 = 351`, `FIT_B_128 = 216`, `REG_I4M` (main_ratio C 73 (0.44, 0.48), C 128 (0.50, 0.53); diag C 73 (0.58, 0.62),
C 128 (0.51, 0.53); peak C 73 (83.5, 84.0), C 128 (83.85, 84.0); strongest steady n05 3,426.2 / n07 3,410.1; strongest
effective n05 None / n07 2,778.3). Run commit f7fb2c5 = 30369eb + a test-only change.

Integrity. The probes and the miniature ran at 69ac801 BEFORE the stamp, by design (the registration comment says "AFTER
the probes ... and BEFORE i4m73 / i4m128"; the probes measure FIT_B, they carry no registered key): m3m B 8 2195575.19
12:07:04Z; m3p73 2195575.20 12:12:13Z-12:23:34Z; m3p128 2195575.21 12:23:51Z-12:35:08Z; all COMPLETED 0:0, MUST 15 / 17 /
17, 0 failed. Matched cells after the stamp: n373 (C 73 B 351) 2195575.24 13:18:19Z: **exit 2, sacct FAILED 2:0, MUST
FAIL L4PTFD-RUN (OutOfMemoryError: tried to allocate 126.00 MiB; 77.90 GiB allocated by PyTorch, 0.21 GiB reserved
unallocated, on the 79.25 GiB card) and MUST FAIL EDGE (no peak)**; n3128 (C 128 B 216) 2195575.25 n05 13:42:38Z and v4128
2195576.22 n07 17:32:16Z: exit 0, COMPLETED 0:0, MUST 17 / 17, FIT-B, GPU, EDGE PASS (headroom 0.109 GB < 4 x 0.300 GB).

Registered keys: C 128 n3128 / v4128: I4M-PEAK 83.891 / 83.891 (83.85, 84.0) C, C; I4M-MAIN 0.5152 (= 1,765.1 / 3,426.2) /
0.5143 (= 1,753.8 / 3,410.1) (0.50, 0.53) C, C; I4M-DIAG 0.5237 / 0.5240 (0.51, 0.53) C, C. C 73 n373: PEAK, MAIN, DIAG
"REPORT: no value" (OOM). Totals 6 C, 0 F, 3 RP.

Verdict rows (MEASURED, summary.json `verdict`): MAIN-MEAN 0.4692 / 0.4682; SH / strongest 0.9837 / 0.9814; same-job SH
3,370.5 / 3,346.7; vs re-check DERIVED 1,765.1 / 3,421.6 = 0.5159 and 1,753.8 / 3,402.1 = 0.5155. Effective view (kept
apart): n07 MAIN-EFF 0.4908 = 1,363.5 / 2,778.3; n05 strongest effective not registered (None). The C 73 matched point is
unmeasured: the fit line was right, the usable ceiling was not (L60, SL:1057-1060). Correctness: L4PTFD = the I4A arm.

## 8. I4U (layer-invariant union work hoisted once per pass; arm L4PTFDU)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4u. Registration: r2_i4u_cell.py at 832dca5, `PRED_STAMP_I4U = "2026-10-29T18:16:01Z"`,
`PRED_I4U_BY_B`.

Integrity. Runs: u3m B 8 2195575.32 n05 18:53:57Z; u3a C 128 2195575.33 n05 19:00:03Z; u4m B 8 2195576.24 n07 18:53:15Z;
u4b C 73 2195576.25 n07 18:59:29Z; y3b C 73 2195575.36 n05 20:31:08Z; y4a C 128 2195576.28 n07 21:13:09Z; o3a C 128
2195575.41 n05 2026-10-30T00:40:59Z. All exit 0, COMPLETED 0:0; MUST 43 / 25, 0 failed. STAMP 18:16:01Z < 18:53:15Z. No
graph arm, so no capture round.

Registered keys: C 128 (u3a / y4a / o3a): JOINT-SAVING 4.090 / 4.092 / 4.486 ms (2.0, 6.0); JOINT-RATIO 0.9801 / 0.9803 /
0.9782 (0.970, 0.990); VS-D 1.0099 / 1.0126 / 1.0142 (1.005, 1.018); RATIO 0.5295 / 0.5281 / 0.5296 (0.515, 0.535);
DRAFT-RATIO 1.0011 / 1.0001 / 0.9994 (0.98, 1.02); PEAK 83.891 (83.85, 83.95); EQ True: all C (21). C 73 (u4b / y3b):
JOINT-SAVING 4.458 / 4.165 (1.5, 5.5); JOINT-RATIO 0.9884 / 0.9891 (0.985, 0.996); VS-D 1.0077 / 1.0085 (1.002, 1.010);
RATIO 0.6013 / 0.6020 (0.590, 0.610); DRAFT-RATIO 1.0000 / 1.0017; PEAK 77.685 (77.60, 77.80); EQ True: all C (14). Total
35 C, 0 F. No refuted key.

Arm L4PTFDU (LOG:468-471): C 128 tok/s 1,782.0 / 1,762.7 / 1,785.6; /SH 0.5295 / 0.5281 / 0.5296; /strong 0.5201 (n05) /
0.5169 (n07) / 0.5212 (n05); /recheck 0.5208 / 0.5181 / 0.5219. C 73 1,540.4 / 1,549.9; /SH 0.6013 / 0.6020; /strong
0.4517 (n07) / 0.4524 (n05). Same-job SH: 3,365.3, 3,337.7, 3,371.6; 2,561.7, 2,574.7. Correctness: L4PTFDU-EQF MUST PASS.

## 9. I4H (CUDA graphs of the draft one-row passes; arm L4PTFDGD)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4h. Registration: r2_i4h_cell.py at fc6c680, `PRED_STAMP_I4H = "2026-10-29T18:47:25Z"`,
`PRED_I4H_BY_B` (steady sets without the capture rounds: the L61 fix is in this cell).

Integrity. Runs: j3m B 8 2195575.34 n05 19:36:50Z; j3a C 128 2195575.35 n05 19:43:36Z; j4m B 8 2195576.26 n07 19:54:16Z;
j4b C 73 2195576.27 n07 20:01:10Z; z3b C 73 2195575.37 n05 21:24:52Z; z4a C 128 2195576.29 n07 21:50:08Z; o4a C 128
2195576.30 n07 22:38:03Z; o4b C 73 2195576.33 n07 2026-10-30T00:45:25Z. All exit 0, COMPLETED 0:0; MUST 68 / 41, 0 failed.
STAMP 18:47:25Z < 19:36:50Z. Steady 180 (C 128) / 194 (C 73) rounds; rounds 1 and 16 excluded (LOG).

| key | j3a | z4a | o4a | band B 216 | j4b | z3b | o4b | band B 318 |
|---|---|---|---|---|---|---|---|---|
| I4H-DRAFT-SAVING (ms) | 4.353 | 4.081 | 4.618 | (2.0, 6.5) | 4.939 | 4.516 | 4.814 | (1.5, 6.0) |
| I4H-DRAFT-RATIO | 0.9696 | 0.9716 | 0.9680 | (0.955, 0.986) | 0.9740 | 0.9760 | 0.9747 | (0.968, 0.992) |
| I4H-JOINT-RATIO | 1.0022 | 1.0022 | 0.9989 | (0.99, 1.01) | 0.9995 | 0.9995 | 1.0013 | (0.99, 1.01) |
| I4H-VS-G | 1.0108 | 1.0109 | 1.0158 | (1.004, 1.020) | 1.0075 | 1.0078 | 1.0082 | (1.000, 1.012) |
| I4H-VS-D | 1.0310 | 1.0350 | 1.0355 | (1.023, 1.040) | 1.0204 | 1.0200 | 1.0227 | (1.013, 1.026) |
| I4H-RATIO | 0.5379 | 0.5412 | 0.5419 | (0.536, 0.545) | 0.6100 | 0.6107 | 0.6111 | (0.605, 0.612) |
| I4H-PEAK-GB | 83.891 | 83.891 | 83.891 | (83.85, 83.95) | 77.685 | 77.685 | 77.685 | (77.60, 77.80) |
| I4H-POOL-MB | 75.498 | 75.498 | 75.498 | (40, 400) | 113.246 | 113.246 | 113.246 | (60, 600) |
| I4H-CAPTURE-S | **0.128** | **0.127** | **0.132** | (0.5, 30) | **0.131** | **0.125** | **0.134** | (0.5, 30) |
| I4H-EQ | True | True | True | True | True | True | True | True |

Totals 54 C, 6 F. Refuted: I4H-CAPTURE-S in all six runs: the draft-graph capture takes 0.13 s, below the 0.5 s floor
(faster than modelled; no speed effect).

Arm L4PTFDGD (LOG:441-446): C 128 tok/s 1,814.3 / 1,812.1 / 1,810.8; /SH 0.5379 / 0.5412 / 0.5419 (spread 0.73 %); /strong
0.5295 (n05) / 0.5314 (n07) / 0.5310 (n07); /recheck 0.5302 / 0.5326 / 0.5323. C 73 1,563.1 / 1,575.3 / 1,564.6; /SH 0.6100 /
0.6107 / 0.6111; /strong 0.4584 (n07) / 0.4598 (n05) / 0.4588 (n07). Same-job SH: 3,372.8, 3,348.2, 3,341.8; 2,562.6,
2,579.7, 2,560.2. With capture rounds out of steady, the mean ranks as the median (j3a: L4PTFD 1,603.4, L4PTFDG 1,637.3,
L4PTFDGD 1,655.8; LOG). Correctness: L4PTFDGD-EQF MUST PASS; DGRAPH MUST PASS.

## 10. I4S (the stack: L4PTFDGUD at C 128, L4PTFDGUDKW at C 73)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4s. Registration: r2_i4s_cell.py at 8f236f8, `PRED_STAMP_I4S = "2026-10-29T22:50:04Z"`,
`PRED_I4S_BY_B` (above_best True at both batches).

Integrity. Runs: s3m B 8 2195575.39 n05 23:45:50Z; s3a C 128 2195575.40 n05 23:53:22Z; s4m B 8 2195576.31 n07 23:26:04Z;
s4b C 73 2195576.32 n07 23:33:45Z; r3b C 73 2195575.42 n05 2026-10-30T01:17:36Z; r4a C 128 2195576.34 n07 01:57:14Z; q3a C
128 2195575.43 n05 02:28:09Z; q4b C 73 2195576.35 n07 02:45:23Z; p3a C 128 2195575.45 n05 03:40:43Z (sacct start 03:40:44Z,
1 s rounding). All exit 0, COMPLETED 0:0; MUST 96 / 42 (C 128) / 45 (C 73), 0 failed. STAMP 2026-10-29T22:50:04Z <
23:26:04Z. L61 fix in.

| key | s3a | r4a | q3a | p3a | band B 216 | s4b | r3b | q4b | band B 318 |
|---|---|---|---|---|---|---|---|---|---|
| I4S-JOINT-SAVING (ms) | 1.768 | 2.006 | 1.883 | 2.058 | (1.5, 4.5) | 1.805 | 2.250 | 1.688 | (1.5, 4.5) |
| I4S-DRAFT-SAVING (ms) | 4.437 | 4.495 | 4.544 | 4.551 | (3.0, 5.7) | 4.537 | 4.721 | 4.892 | (1.5, 6.0) |
| I4S-JOINT-RATIO | 0.9911 | 0.9899 | 0.9905 | 0.9896 | (0.977, 0.993) | 0.9949 | 0.9936 | 0.9952 | (0.987, 0.996) |
| I4S-DRAFT-RATIO | 0.9691 | 0.9688 | 0.9683 | 0.9682 | (0.960, 0.979) | 0.9761 | 0.9750 | 0.9743 | (0.968, 0.992) |
| I4S-VS-REF | 1.0195 | 1.0203 | 1.0185 | 1.0185 | (1.013, 1.030) | 1.0114 | 1.0121 | 1.0124 | (1.005, 1.020) |
| I4S-VS-D | 1.0385 | 1.0403 | 1.0407 | 1.0389 | (1.030, 1.054) | 1.0666 | 1.0673 | 1.0701 | (1.055, 1.079) |
| I4S-RATIO | 0.5429 | 0.5456 | 0.5429 | 0.5427 | (0.535, 0.552) | 0.6381 | 0.6383 | 0.6389 | (0.632, 0.645) |
| I4S-PEAK-GB | 83.891 | 83.891 | 83.891 | 83.891 | (83.85, 83.95) | 77.685 | 77.685 | 77.685 | (77.60, 77.80) |
| I4S-POOL-MB | 75.498 | 75.498 | 75.498 | 75.498 | (60, 130) | 117.441 | 117.441 | 117.441 | (80, 250) |
| I4S-CAPTURE-S | 6.770 | 6.782 | 6.783 | 7.018 | (5.5, 10) | 7.384 | 7.578 | 7.309 | (5.5, 10) |
| I4S-EQ | True | True | True | True | True | True | True | True | True |
| I4S-ABOVE-BEST | True | True | True | True | True | True | True | True | True |

REF = L4PTFDG at C 128, L4PTFDGKW at C 73. All 84 key x run evaluations C, 0 F. No refuted key.

Arms (LOG:460-465): L4PTFDGUD C 128 tok/s 1,828.1 / 1,823.5 / 1,828.6 / 1,828.3; /SH 0.5429 / 0.5456 / 0.5429 / 0.5427
(spread 0.54 %); /strong 0.5336 (n05) / 0.5347 (n07) / 0.5337 (n05) / 0.5336 (n05); /recheck 0.5343 / 0.5360 / 0.5344 /
0.5343. L4PTFDGUDKW C 73 1,636.0 / 1,644.0 / 1,636.8; /SH 0.6381 / 0.6383 / 0.6389 (spread 0.12 %); /strong 0.4798 (n07) /
0.4798 (n05) / 0.4800 (n07); /recheck 0.4809 / 0.4805 / 0.4811. Same-job SH: 3,367.4, 3,342.3, 3,368.0, 3,369.1 (C 128);
2,563.8, 2,575.4, 2,562.0 (C 73). Correctness: EQF, GRAPH, DGRAPH, HOIST MUST PASS; KEEP-EQ (= L4PTFKW) in s3m / s4m.

## 11. I4CC (off-the-shelf compiler, torch.compile / Inductor, over the joint pass's small integer / mask / index / copy helpers; arms L4PTFDGUDC, L4PTFDGUDKWC)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4cc. Registration: r2_i4cc_cell.py at d75a946, `PRED_STAMP_I4C2 = "2026-10-30T04:10:18Z"`,
`PRED_I4C2_BY_B` (I4C2_above_best None at B 318: 1 RP per C 73 cell).

Integrity. Runs (all 2026-10-30): c3m B 8 2195575.46 n05 04:28:18Z; c3a C 128 2195575.47 n05 04:37:39Z; c4m B 8 2195576.37
n07 04:22:26Z; c4b C 73 2195576.38 n07 04:31:43Z; v3b C 73 2195575.48 n05 05:36:29Z; v4a C 128 2195576.39 n07 05:43:32Z;
w3a C 128 2195575.49 n05 06:46:53Z; w4b C 73 2195576.40 n07 06:42:40Z; y3b C 73 2195575.52 n05 08:42:55Z (ended
09:53:07Z). All exit 0, COMPLETED 0:0; MUST 151 (B 8) / 59 (C 128) / 48 (C 73), 0 failed. STAMP 04:10:18Z < 04:22:26Z.
REPORT FAIL in every C 73 cell: L4PTFDGUDKW-KEEP-EQ "no keep reference (L4PTFKW or L4PTFDGKW) in this job" (the REF arm has
no keep partner in this job; the compiled arm's own KEEP-EQ is MUST PASS). Not a defect. COMPILE MUST PASS (backend
inductor, 0 graph breaks, no fallback).

| key | c3a | v4a | w3a | band B 216 | c4b | v3b | w4b | y3b | band B 318 |
|---|---|---|---|---|---|---|---|---|---|
| I4C2-JOINT-SAVING (ms) | 10.824 | 10.737 | 10.842 | (3.0, 12.0) | 11.639 | 11.063 | 11.834 | 10.848 | (3.0, 13.0) |
| I4C2-JOINT-RATIO | 0.9449 | 0.9457 | 0.9449 | (0.939, 0.985) | 0.9668 | 0.9683 | 0.9662 | 0.9688 | (0.963, 0.991) |
| I4C2-DRAFT-RATIO | 0.9986 | 1.0013 | 0.9996 | (0.99, 1.01) | 0.9998 | 0.9981 | 0.9992 | 0.9990 | (0.99, 1.01) |
| I4C2-VS-REF | 1.0325 | 1.0323 | 1.0319 | (1.008, 1.037) | 1.0234 | 1.0218 | 1.0223 | 1.0224 | (1.005, 1.025) |
| I4C2-VS-D | 1.0742 | 1.0775 | 1.0732 | (1.044, 1.080) | 1.0919 | 1.0907 | 1.0922 | 1.0900 | (1.066, 1.099) |
| I4C2-RATIO | 0.5612 | 0.5636 | 0.5592 | (0.545, 0.565) | 0.6534 | 0.6523 | 0.6527 | 0.6525 | (0.638, 0.657) |
| I4C2-PEAK-GB | 83.900 | 83.900 | 83.900 | (83.85, 83.95) | 77.694 | 77.694 | 77.694 | 77.694 | (77.60, 77.80) |
| I4C2-POOL-MB | 71.303 | 71.303 | 71.303 | (40, 130) | 117.441 | 117.441 | 117.441 | 117.441 | (60, 250) |
| I4C2-CAPTURE-S | 9.370 | 9.581 | 9.651 | (5.5, 10) | **10.388** | 9.716 | **10.498** | 9.816 | (5.5, 10) |
| I4C2-COMPILE-S | 10.217 | 10.244 | 10.289 | (5, 180) | 10.799 | 10.755 | 10.850 | 10.724 | (5, 180) |
| I4C2-EQ | True | True | True | True | True | True | True | True | True |
| I4C2-ABOVE-BEST | True | True | True | True | RP | RP | RP | RP | None |

Totals 78 C, 2 F, 4 RP. Refuted: I4C2-CAPTURE-S at C 73 in c4b and w4b (10.39 / 10.50 s > 10 s); v3b and y3b are C (9.72 /
9.82 s): the verdict FLIPS between replicates; the band edge lies inside the 9.7-10.5 s spread. One-time cost; no effect
on the steady median.

Arms (LOG:404-410; y3b from its stdout REPORT lines): L4PTFDGUDC C 128 tok/s 1,890.5 / 1,879.1 / 1,887.1; /SH 0.5612 /
0.5636 / 0.5592 (spread 0.78 %); /strong 0.5518 (n05) / 0.5510 (n07) / 0.5508 (n05); /recheck 0.5525 / 0.5523 / 0.5515.
L4PTFDGUDKWC C 73 tok/s 1,673.6 / 1,679.7 / 1,672.1 / 1,681.9; /SH 0.6534 / 0.6523 / 0.6527 / 0.6525; /strong 0.4908 (n07) /
0.4903 (n05) / 0.4903 (n07) / 0.4909 (n05; DERIVED 1,681.9 / 3,426.2); /recheck 0.4919 / 0.4909 / 0.4915 / 0.4916 (y3b
DERIVED 1,681.9 / 3,421.6). Same-job SH: 3,368.9, 3,334.3, 3,374.7 (C 128); 2,561.5, 2,575.0, 2,561.9, 2,577.5 (C 73).

Effective view (kept apart, MEASURED, c3a): L4PTFDGUDC effective 1,197.5 tok/s with capture and compile, 1,332.7 without
capture, 1,519.8 without capture and compile; L4PTFD (first arm, L62) 1,365.9. The one-time 9.4 s capture and 10.2 s cold
compile cost more than the run's steady gain in a 512-step run. Correctness: EQF and COMPILE MUST PASS.

## 12. I4DC (compiler extended to the draft passes; arms L4PTFDGUDCD, L4PTFDGUDKWCD)

Tree /mnt/beegfs/ojerbi/nosi-r2-i4dc. Registration: r2_i4dc_cell.py at 2e9e497, `PRED_STAMP_I4DC = "2026-10-30T07:27:31Z"`,
`PRED_I4DC_BY_B` (above_best None at B 318).

Integrity. Runs (2026-10-30): d3m B 8 2195575.50 n05 07:45:31Z; d3a C 128 2195575.51 n05 07:54:51Z; d4m B 8 2195576.41
n07 07:54:33Z; d4b C 73 2195576.42 n07 08:04:18Z. All exit 0, COMPLETED 0:0; MUST 153 / 46 / 153 / 49, 0 failed. STAMP
07:27:31Z < 07:45:31Z. REPORT FAIL in d4b: L4PTFDGUDKWC-KEEP-EQ (no keep reference in the job; the D arm's KEEP-EQ is MUST
PASS). PENDING: r4a (C 128, n07, started 09:16:26Z, no exit.txt at 09:55Z); r3b (C 73, n05) not started.

Registered keys: C 128 (d3a): DRAFT-SAVING 6.409 ms (3.0, 8.0); DRAFT-RATIO 0.9538 (0.942, 0.978); JOINT-RATIO 1.0010
(0.99, 1.01); VS-REF 1.0188 (1.008, 1.026); VS-D 1.0949 (1.079, 1.106); RATIO 0.5700 (0.564, 0.578); PEAK 83.900 (83.85,
83.95); POOL 71.303 (40, 130); CAPTURE 9.575 s (6, 13); COMPILE 2.179 s (1, 30); EQ True; ABOVE-BEST 1,924.1 > {1,757.3,
1,888.6}: all C (12). C 73 (d4b): DRAFT-SAVING 8.573 (4.0, 10.0); DRAFT-RATIO 0.9537 (0.946, 0.978); JOINT-RATIO 1.0013;
VS-REF 1.0151 (1.006, 1.020); VS-D 1.1081 (1.093, 1.119); RATIO 0.6627 (0.655, 0.669); PEAK 77.694; POOL 117.441; CAPTURE
10.110; COMPILE 2.219; EQ True: all C (11); ABOVE-BEST RP (1,699.0 > {1,533.2, 1,673.8}). Totals 23 C, 0 F, 1 RP.

Arms (LOG:426-431): L4PTFDGUDCD C 128 1,924.1 tok/s, /SH 0.5700, /strong 0.5616 (n05), /recheck 0.5623; same-job
L4PTFDGUDC 1,888.6 (/SH 0.5595), L4PTFD 1,757.3 (0.5206); SH 3,375.5. L4PTFDGUDKWCD C 73 1,699.0, /SH 0.6627, /strong
0.4982 (n07), /recheck 0.4994; same-job KWC 1,673.8 (0.6528), L4PTFD 1,533.2 (0.5980); SH 2,563.9. n = 1 per batch: no
replicate spread yet (I4CC's n = 3-4 runs show 0.16-0.78 % in /SH). Correctness: EQF, COMPILE (joint 129 + draft 128
sections in the miniatures, SL:1286-1287) MUST PASS.

## 13. Integrity findings in full

1. One FAILED run (i4m n373, section 7). All other 77 finished runs: exit 0, COMPLETED 0:0, 0 MUST failures.
2. Registration order: for every family the STAMP and the registration commit (UTC, git log) are earlier than the first
   cell start (sacct elapsed back from exit.txt utc_end). Smallest gap: I4CC, commit 04:10:35Z, first cell 04:22:26Z
   (12 min; DERIVED). No run used a tree commit other than the registration commit or a test-only child (acbe433, f7fb2c5).
3. STATUS_LOG key counts are one too high (records: the stdout PREDICTION lines). SL:1017-1018 i4da "16 CONFIRMED, 5
   REFUTED" (record 15 / 5); SL:1048 i4db "17 CONFIRMED, 5 REFUTED" (16 / 5); SL:1089-1090 i4ka "12 / 12" (11 / 11; 2 keys
   None); SL:1112 i4kb "8 CONFIRMED, 4 REFUTED" (7 / 4); SL:1167-1168 i4ha "10 CONFIRMED, 1 REFUTED" (9 / 1); SL:1174-1175
   i4hb (9 / 1); SL:1207 i4s "13 / 13 at EACH batch (26 / 26)" (12 / 12 each, 24 / 24); SL:1257 i4cc "23 CONFIRMED, 1
   REFUTED" (22 / 1); SL:1289 i4dca "13 / 13" (12 / 12); SL:1297 i4dcb "12 / 12" (11 / 11). No verdict changes; only the
   counts.
4. Replicate verdict flips (bands that sit inside the run spread): I4K-JOINT-SAVING and I4K-VS-D at B 318; I4C2-CAPTURE-S
   at B 318. A single run would have given either verdict.
5. Scope mismatch: I4A-RATIO evaluated in the I4D accounting cells (F twice) is not a speed result.
6. Decision rule I4C fired on 1 of 2 C 128 measurements (0.0514 n05; 0.0489 n07).
7. L61 (capture round counted as steady) is present in the I4A, I4C, I4D, I4G, I4K and I4U cells and fixed in I4H, I4S,
   I4CC and I4DC. Medians are unaffected; the I4G / I4K graph arms' means and effective rates are not comparable across
   that line. L62 (first-arm penalty): L4PTFD runs first in every I4C-I4DC job, so its effective rate is depressed there.
8. EQF is REPORT-level (not MUST) in the I4A and I4D timing cells; it passed in every case.
9. The cells' VS-STRONGEST lines use 3,407.2; this file uses the same-GPU denominators and the same-day re-checks.

## 14. Raw record paths used (for the revision-7 reconciliation workstream)

Every run below has these files: R/<run>/summary.json; R/<run>.step/stdout.txt; R/<run>.step/sacct.txt;
R/<run>.step/exit.txt; R/<run>.step/command.txt (R = /mnt/beegfs/ojerbi/r2-runs/spec). Registration constants: the cell
file named in each section, at the registration commit, in the tree /mnt/beegfs/ojerbi/nosi-r2-<family>.

- I4A: i4a_q3m_2195575_i4am (B 8), i4a_q3a_2195575_i4aa, i4a_k4a_2195576_i4aa, i4a_q4b_2195576_i4ab, i4a_k3b_2195575_i4ab.
- I4B: i4b_w3m_2195575_i4bm (B 8), i4b_w3sa_2195575_i4bsa, i4b_w4sb_2195576_i4bsb (sweeps; summary.json SH block has no
  rates).
- I4C: i4c_c3m_2195575_i4cm, i4c_c4m_2195576_i4cm (B 8), i4c_c3a_2195575_i4ca, i4c_e4a_2195576_i4ca, i4c_c4b_2195576_i4cb.
- I4D: i4d_d3m_2195575_i4dm, i4d_d4m_2195576_i4dm (B 8), i4d_d3c_2195575_i4dc, i4d_d4d_2195576_i4dd (accounting),
  i4d_d3a_2195575_i4da, i4d_d4b_2195576_i4db, i4d_f3b_2195575_i4db.
- I4G: i4g_g3m_2195575_i4gm, i4g_g4m_2195576_i4gm (B 8), i4g_g3a_2195575_i4ga, i4g_h4a_2195576_i4ga, i4g_x3a_2195575_i4ga,
  i4g_g4b_2195576_i4gb, i4g_h3b_2195575_i4gb, i4g_x4b_2195576_i4gb.
- I4K: i4k_k3m_2195575_i4km, i4k_k4m_2195576_i4km (B 8), i4k_k3a_2195575_i4ka, i4k_r4a_2195576_i4ka, i4k_k4b_2195576_i4kb,
  i4k_r3b_2195575_i4kb, i4k_t3b_2195575_i4kb.
- I4M: i4m_m3m_2195575_i4mm (B 8), i4m_m3p73_2195575_i4mp73, i4m_m3p128_2195575_i4mp128 (probes), i4m_n373_2195575_i4m73
  (FAILED), i4m_n3128_2195575_i4m128, i4m_v4128_2195576_i4m128 (summary.json keys ARM, SH, fit, gates, meta, predictions,
  probe_check, verdict; no SPEC block).
- I4U: i4u_u3m_2195575_i4um, i4u_u4m_2195576_i4um (B 8), i4u_u3a_2195575_i4ua, i4u_y4a_2195576_i4ua, i4u_o3a_2195575_i4ua,
  i4u_u4b_2195576_i4ub, i4u_y3b_2195575_i4ub.
- I4H: i4h_j3m_2195575_i4hm, i4h_j4m_2195576_i4hm (B 8), i4h_j3a_2195575_i4ha, i4h_z4a_2195576_i4ha, i4h_o4a_2195576_i4ha,
  i4h_j4b_2195576_i4hb, i4h_z3b_2195575_i4hb, i4h_o4b_2195576_i4hb.
- I4S: i4s_s3m_2195575_i4sm, i4s_s4m_2195576_i4sm (B 8), i4s_s3a_2195575_i4sa, i4s_r4a_2195576_i4sa, i4s_q3a_2195575_i4sa,
  i4s_p3a_2195575_i4sa, i4s_s4b_2195576_i4sb, i4s_r3b_2195575_i4sb, i4s_q4b_2195576_i4sb.
- I4CC: i4cc_c3m_2195575_i4cm2, i4cc_c4m_2195576_i4cm2 (B 8), i4cc_c3a_2195575_i4cca, i4cc_v4a_2195576_i4cca,
  i4cc_w3a_2195575_i4cca, i4cc_c4b_2195576_i4ccb, i4cc_v3b_2195575_i4ccb, i4cc_w4b_2195576_i4ccb, i4cc_y3b_2195575_i4ccb
  (y3b: scored from stdout / sacct / exit; not in LOG, which ran before it ended).
- I4DC: i4dc_d3m_2195575_i4dm2, i4dc_d4m_2195576_i4dm2 (B 8), i4dc_d3a_2195575_i4dca, i4dc_d4b_2195576_i4dcb; pending
  i4dc_r4a_2195576_i4dca (no exit.txt, no summary.json).
- Denominator re-checks: /mnt/beegfs/ojerbi/r2-runs/gstep/gstep_e3r_2195575_gs1r/summary.json and
  gstep_e4r_2195576_gs1r/summary.json (`arms.gs.rates.steady_tok_s`).
- Per-run fields this file used from summary.json (SPEC.<arm>): rates.median_tok_s, rates.round_ms_median, rates.mean_tok_s,
  rates.effective_tok_s, effective_wo_capture_tok_s, effective_wo_setup_tok_s, phases_median, draft_step_sh,
  joint_sh_steps, peak_with_prefill_GB, steady_excluded_rounds; SH.tok_s, SH.rates.steady_step_ms. The extraction log
  with every arm row is LOG (1,200 lines).

## 15. Method and rule record

Reads with the Read tool: SL:880-1303, PL/CHECKPOINTS.md, PL/LEDGER_rev6.md, the rev-6 script, LOG. Login shell: cat,
grep, head, tail, ls, cut, sort, wc, date, git (log / show / status). Arithmetic: two CPU scripts under
/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/ana_rev7/ (a1_struct.py, a2_score.py; written with the Write tool), each run with
`srun -p amd -c 4 --mem=16G -t 60 apptainer exec ... /venv/nosa/bin/python -I -B` (outputs cpu-logs/ana_rev7_a1.txt,
ana_rev7_a2.txt; the first a2 run stopped at the I4B sweep record, which has no SH rates, and was rerun after a guard).
Hand arithmetic in this file: the y3b and n3128 / v4128 re-check ratios (operands shown). No interpreter on the login node,
no job submission, no code-tree edit. Per the lead's scope change, this agent did NOT write LEDGER_rev7.md.
