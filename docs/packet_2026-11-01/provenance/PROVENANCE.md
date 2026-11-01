# PROVENANCE of the LEDGER_rev7 numerical rows (review 3, point 7)

Written 2026-10-31 (bundle made 2026-10-31T11:38:20Z-11:38:23Z on thunder09, SLURM job 2243216; stamps from `date -u` in
MANIFEST.txt). Ledger: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_rev7.md. Style: ASD-STE100. No study was
run again; no trace was copied. The analysis scripts were READ, not run.

## Abbreviations (absolute)

- BND = /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_prov/bundle (MANIFEST: BND/MANIFEST.txt; copy script:
  /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_prov/collect.sh; its stdout: .../recon7_prov/collect_log.txt).
- R = /mnt/beegfs/ojerbi/r2-runs/spec. In BND a run record R/x is at BND/r2-runs/spec/x.
- EV = /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence (in BND: BND/_central_users_ojerbi/retroinfer-eval/docs/evidence).
  The path /mnt/beegfs/ojerbi/retroinfer-eval does not exist; the ledger's J91M path (/mnt/central/...) is the real one.
- LOG = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs. SC = /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc.
- Log line numbers count every line (the first nine lines of each log are image shell-profile messages).

## Bundle content

- COPIED: 303 files, 16,272,225 B (+ MANIFEST.txt). Per-file cap 5,000,000 B (decimal MB, the ledger's unit).
- NOT TRANSFERRED, hashed on the node (sha256 in MANIFEST pins the original): 65 files = 18 Kineto traces (trace),
  15 summary.json above 5 MB, 4 i4a summary.json (no covered row; held out for the 30 MB cap), 28 stderr.txt (no script
  reads stderr). MISSING files: 0 of the listed paths.
- DERIVED (hashed in place, not copied): 53 files = the 25 analysis scripts + 28 logs listed in the task.
- The six core run summaries above 5 MB (i3_r4a 5,003,343; i3_r3b 7,341,335; i4dc_d3a 5,046,590; i4dc_d4b 7,387,411;
  i4dc_r4a 5,046,587; i4c_c4b 5,018,852 B) are NOT TRANSFERRED (above 5 MB). Their step records (command.txt, exit.txt,
  sacct.txt, stdout.txt) ARE in BND. Only i4c_e4a's summary.json (3,460,201 B) of the seven core runs is in BND.

## Input inventory (every path the scripts open)

| script (SC/...) | log (LOG/...) | inputs opened | in BND? |
|---|---|---|---|
| recon7_r1/recon7_r1_model.py, recon7_r1_round.py | recon7_r1_model.txt, recon7_r1_round.txt | EV/hbm_sweep_2175791/scoring/output3_timing_bw.csv; EV/rows_bench_2175525/gemm_mscan.json; EV/cpupack_confirm_2179735/scoring/layout_variants.csv; /mnt/beegfs/ojerbi/r2-runs/bi/bi_gma_2190876_gm1/summary.json (model.py only) | all COPIED |
| recon7_r1/recon7_r1_mem.py, recon7_r1_units.py | recon7_r1_mem.txt, recon7_r1_units.txt | EV/hbm_sweep_2175791/scoring/output2_memory.csv | COPIED |
| recon7_r1/recon7_r1_products.py, recon7_r1_yield.py | recon7_r1_products.txt, recon7_r1_yield.txt | none (constants) | - |
| recon7_r2/s1_records.py | recon7_r2_s1.txt | R/{i3_r4a_2195576_i3a, i3_r3b_2195575_i3b, i3_r2cb_2195574_i3cb, i4c_c3a_2195575_i4ca, i4c_e4a_2195576_i4ca, i4c_c4b_2195576_i4cb, i4dc_d3a_2195575_i4dca, i4dc_d4b_2195576_i4dcb, i4cc_c3a_2195575_i4cca, i4cc_c4b_2195576_i4ccb, i4s_s3a_2195575_i4sa, i4s_s4b_2195576_i4sb}/summary.json | COPIED: i3_r2cb, i4c_c3a, i4c_e4a. Others NOT TRANSFERRED (above 5 MB) |
| recon7_r2/s2_traces.py | recon7_r2_s2.txt | R/{i4c_e4a, i4c_c3a, i4c_c4b}/trace_L4PTFDP_g{112,96}_{joint,draft0,draft1}.json.gz | NOT TRANSFERRED (trace), 18 files |
| recon7_r2/s3_derive.py | recon7_r2_s3.txt | none (constants copied from the s1 / s2 logs, REQ:35-40, FC:22-26) | - |
| recon7_r3/s1_skeleton.py | recon7_r3_s1.txt | R/i4dc_d3a_2195575_i4dca/summary.json | NOT TRANSFERRED (above 5 MB) |
| recon7_r3/s2_extract.py | recon7_r3_s2.txt | R/{i3_r4a, i3_r3a, i3_r3b, i3_r4b, i4a_q3a, i4a_k4a, i4a_q4b, i4a_k3b, i4d_d3c, i4d_d4d, i4d_d3a, i4d_d4b, i4d_f3b, i4dc_d3a, i4dc_d4b, i4m_n3128, i4m_v4128, i4m_m3p73, i4m_m3p128}/summary.json | COPIED: i4d_d3c, i4d_d4d, the four i4m. i4a: NOT TRANSFERRED (no covered row). Others NOT TRANSFERRED (above 5 MB) |
| recon7_r3/s3_resident_classes.py | recon7_r3_s3.txt | EV/hbm_sweep_2175791/scoring/output1_hbm_families_long.csv | COPIED |
| recon7_r3/s4_derive.py | recon7_r3_s4.txt | R/{i4dc_d3a, i4dc_d4b, i4c_c3a, i4c_c4b, i4d_d3a, i4d_d4b, i4d_d3c, i4d_d4d}/summary.json; output1_hbm_families_long.csv | COPIED: i4c_c3a, i4d_d3c, i4d_d4d, csv. Others NOT TRANSFERRED (above 5 MB) |
| recon7_r3/s5_backbone.py | recon7_r3_s5.txt | EV/.../output2_memory.csv | COPIED |
| recon7_s/s1_revised_model.py, s2_round_split.py, s3_review_response.py, s4_review2.py | recon7_s1.txt .. recon7_s4.txt | none: constants transcribed from R1-R3, SI, CP, V-logs, the AUTHOR message (no file) and the J91 csv | - (see the rows) |
| recon7_v/v1_records.py | recon7_v_v1.txt | R/{i3_r4a, i3_r3b, i4c_e4a, i4c_c4b, i4c_c3a, i3_r2cb, i4dc_d3a, i4dc_d4b}/summary.json | COPIED: i4c_e4a, i4c_c3a, i3_r2cb. Others NOT TRANSFERRED (above 5 MB) |
| recon7_v/v1b_tfc.py | recon7_v_v1b.txt | R/i3_r2cb_2195574_i3cb/summary.json | COPIED |
| recon7_v/v2_traces.py | recon7_v_v2.txt | R/{i4c_e4a, i4c_c4b}/trace_L4PTFDP_g{112,96}_{joint,draft0,draft1}.json.gz | NOT TRANSFERRED (trace) |
| recon7_v/v3_model.py | recon7_v_v3.txt | EV/.../output3_timing_bw.csv, output2_memory.csv, gemm_mscan.json, layout_variants.csv | COPIED |
| recon7_v/v4_probe_wholerun.py | recon7_v_v4.txt | R/{i4c_c4b, i4c_e4a, i3_r4a, i3_r3b, i4dc_d3a, i4dc_d4b}/summary.json | COPIED: i4c_e4a only |
| recon7_v/v4b_probe_rounds.py | recon7_v_v4b.txt | R/{i4c_c4b, i4c_e4a}/summary.json | COPIED: i4c_e4a only |
| ana_rev7/a1_struct.py | ana_rev7_a1.txt | R/i4dc_d3a_2195575_i4dca/summary.json; /mnt/beegfs/ojerbi/r2-runs/gstep/gstep_e3r_2195575_gs1r/summary.json | gstep COPIED; i4dc_d3a NOT TRANSFERRED |
| ana_rev7/a2_score.py | ana_rev7_a2.txt | R/i4*.step/{command,exit,sacct}.txt (glob, 81 step dirs); R/<run>/summary.json per step; gstep_e3r / gstep_e4r summary.json | step records COPIED (all 81); summaries: as above |

Inputs that NO script opened but the ledger cites (not in BND): the strongest cached NOSA records behind CP:9-12
(3,426.2 / 3,410.1 / 2,778.3 tok/s); the OOM record R/i4m_n373_2195575_i4m73/summary.json (8,368 B; its step records are
in BND through the a2 glob); /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/pcie-feed-evidence.md:11 (25.2 GB/s, a note,
not a run record); the AUTHOR inputs of review 2 (a coordinator message; NO FILE).

## Status rule

FULL = the derivation (script + log line) is named and every raw input is in BND. PARTIAL = derivation named; at least
one raw input is NOT TRANSFERRED (pinned by sha256 in MANIFEST) or reaches the script only as a transcribed constant.
MISSING = at least one input has no retained raw record that we can name (it is marked "MISSING: not retained").

## Table

| # | numerical row (ledger section / row / value) | derived by (script, log line) | raw inputs | status |
|---|---|---|---|---|
| 1 | 1 / row 1 (b): td 54.8137 / 75.2948, GEMM(B) 18.3286 / 25.2865, 2 draft passes 121.22 / 162.42 | s4_review2.py; recon7_s4.txt:17, :21 (td points cross-checked in output3_timing_bw.csv; GS by v3_model.py, recon7_v_v3.txt:11; s by LV, recon7_v_v3.txt:13) | BND/_central_users_ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output3_timing_bw.csv; .../rows_bench_2175525/gemm_mscan.json; .../cpupack_confirm_2179735/scoring/layout_variants.csv | FULL |
| 2 | 1 / row 1 (c): kernel ms vs td classes (GEMM 19.7 vs 20.4 ...), draft step BEST 66.14 / 88.22, +11.33 / +12.92 | s3_resident_classes.py, recon7_r3_s3.txt; s2_extract.py, recon7_r3_s2.txt:788 (66.144); s3_review_response.py, recon7_s3.txt:89 | output1_hbm_families_long.csv (BND); ours from traces R/i4c_e4a, R/i4c_c4b (NOT TRANSFERRED, trace); R/i4dc_d3a, R/i4dc_d4b summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 3 | 1 / row 2 (b): two-call 63.4452 / 89.9657; per cohort 100.10 / 140.54; our shape 95.24 / 136.81 | s4_review2.py; recon7_s4.txt:17, :21, :65-66 | gemm_mscan.json (BND); GEMM(861) extrapolation rule FC:26 (forecast note, CITED) | FULL (the curve); the rule above M 768 is a CITED model rule |
| 4 | 1 / row 2 (c): joint GEMM 64.80 / 79.88; per round 104.37 / 131.29 | s2_traces.py, recon7_r2_s2.txt:16 (GEMM 129 / 64.7996); s4_review2.py, recon7_s4.txt:65-66 | traces R/i4c_e4a, R/i4c_c4b (NOT TRANSFERRED, trace) | PARTIAL |
| 5 | 1 / row 3 (c): selection 27.31 / 38.47; +17.83 / +24.43 | s2_traces.py, recon7_r2_s2.txt; s3_derive.py, recon7_r2_s3.txt (R2:228, :241, :263) | traces (NOT TRANSFERRED, trace) | PARTIAL |
| 6 | 1 / row 4 (b): extra 5.0333 / 7.8025; U - I 8.969 | s4_review2.py; recon7_s4.txt:10, :17, :21 | U, I: AUTHOR (review 2 message) - MISSING: not retained (no file; historical 4-request replay record not in the packet); W, KV per request: FC:23 (note) | MISSING |
| 7 | 1 / row 4 (c): logical pages 72.24 = 71.18 + 1.06; x0.989 / 0.990 | v1b_tfc.py; recon7_v_v1b.txt:26-44 | BND/r2-runs/spec/i3_r2cb_2195574_i3cb/summary.json (+ step records) | FULL (the model side of the ratio uses U, I: row 6) |
| 8 | 1 / row 4 (c): fork 37.1 / 52.2 ms; logical rate 0.885 / 0.927 TB/s; call A 1.527 / 1.543 | s2_traces.py, recon7_r2_s2.txt:50, :110 (fork 37.118; call A 19.0631); v2_traces.py, recon7_v_v2.txt | traces (NOT TRANSFERRED, trace) | PARTIAL |
| 9 | 1 / row 5 (c): tentative 28.9 / 31.1, increment 25.9 / 27.2; BEST 11.8 / 13.7 | s2_traces.py, recon7_r2_s2.txt; s3_derive.py, recon7_r2_s3.txt (R2:230, :243, :297) | traces (NOT TRANSFERRED); BEST lever deltas from R/i4cc_*, R/i4s_* summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 10 | 1 / row 6 (b): traffic 2.9105 / 9.0978 MB; transfer 62.06 / 263.41 ms; s, r | s4_review2.py; recon7_s4.txt:17-24 | traffic: AUTHOR - MISSING: not retained (no file); s, r: layout_variants.csv (BND) | MISSING |
| 11 | 1 / row 6 (c): live loads 2.581 / 10.527 / 8.065 MB; H2D 21.3 / 160.1; exposed 20.25 / 154.76 / 115.64 | s2_traces.py, recon7_r2_s2.txt:15 (H2D 21.3014); s2_extract.py, recon7_r3_s2.txt; s4_review2.py, recon7_s4.txt:40-42 (transcribed) | traces (NOT TRANSFERRED); R/i3_r3b, R/i4dc_d4b, R/i4c_c4b summary.json (NOT TRANSFERRED, above 5 MB); R/i4c_e4a summary.json (BND) | PARTIAL |
| 12 | 1 / row 7 (b): 87.91 / 87.22 GB; largest B 204.1 / 304.2; anchor 83.77 GB, 287.95 | s4_review2.py; recon7_s4.txt:26-36 | backbone and snapshot points: output2_memory.csv (BND; points cross-checked by grep); constants 2,105,344 B and 0.12740196 GB and the rule: AUTHOR - MISSING: not retained (no file) | MISSING |
| 13 | 1 / row 7 (c): peaks 83.891 / 83.900, 77.685 / 77.694 GB; fit lines; basis gap 72.119 - 66.391 | s2_extract.py, recon7_r3_s2.txt; v3_model.py, recon7_v_v3.txt:34-42; s3_review_response.py, recon7_s3.txt:85 | R/i3_*, R/i4dc_* summary.json (NOT TRANSFERRED, above 5 MB); R/i4m_n3128, R/i4m_v4128, R/i4m_m3p73, R/i4m_m3p128 summary.json (BND); output2_memory.csv (BND) | PARTIAL |
| 14 | 1 / row 7 (c): B351 OOM 83.644 GB + 126 MiB, 0.551 GB non-PyTorch | none of the listed scripts (CITED SI:57-59) | R/i4m_n373_2195575_i4m73.step command / exit / sacct (BND, via the a2 glob); R/i4m_n373_2195575_i4m73/summary.json 8,368 B and .step/stdout.txt 2,260 B: NOT in BND (no script read them) | PARTIAL |
| 15 | 1 / row 8 (b): interval 187.40 / 263.41; Q 3,457.9 / 3,621.8 | s4_review2.py; recon7_s4.txt:18, :22 | depends on rows 6, 10 (AUTHOR) - MISSING: not retained | MISSING |
| 16 | 1 / row 8 (c): counts (accepted 109,318 / 160,248; corrections 1,058 / 2,250; groups 37,614 / 55,866); 1.70 rounds per reject | s4_derive.py, recon7_r3_s4.txt:254-267; v1_records.py, recon7_v_v1.txt:247-258 | R/i4dc_d3a, R/i4dc_d4b, R/i3_r4a, R/i3_r3b summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 17 | 1 / row 8 (d): Y 2.918 / 2.884 (fixed lead), 2.863 / 2.808 (accounting), 2.855 / 2.798 (measured) | s4_review2.py, recon7_s4.txt:67-68 (p from transcribed counts) | as row 16 | PARTIAL |
| 18 | 1 / row 9 (c): launches per joint pass; non-busy 8.5 / -1.9; commit + decision 6.0 / 8.4 | s2_traces.py, recon7_r2_s2.txt; s1_records.py, recon7_r2_s1.txt:26, :89 | traces (NOT TRANSFERRED); R/i3_r4a, R/i3_r3b summary.json (NOT TRANSFERRED) | PARTIAL |
| 19 | 1 / row 10 (b): GPU 187.40 / 252.05; per-pass 191.38 / 259.46; overlap term 41.52 / 57.81 | s4_review2.py; recon7_s4.txt:18-23 | extra uses U, I (AUTHOR) - MISSING: not retained | MISSING |
| 20 | 1 / row 10 (c): 1 stream; 0.000 ms concurrency; 0 host syncs | s2_traces.py, recon7_r2_s2.txt:15; v2_traces.py, recon7_v_v2.txt | traces (NOT TRANSFERRED, trace) | PARTIAL |
| 21 | 1 / row 11 (c): steady 1,709.8 / 1,525.8 / 1,924.1 / 1,699.0; whole run 1,396.7 / 1,275.3 / 1,292.4 / 1,242.5; capture 9.594 / 10.140 s; prelude | v4_probe_wholerun.py, recon7_v_v4.txt:19-26; s3_review_response.py, recon7_s3.txt:11-24 | R/i3_r4a, R/i3_r3b, R/i4dc_d3a, R/i4dc_d4b summary.json (NOT TRANSFERRED, above 5 MB); their stdout.txt (BND) | PARTIAL |
| 22 | 1 / row 11: amortized k = 2.99 / 1.50; strongest EFFECTIVE 2,778.3 | s4_review2.py, recon7_s4.txt:53-62; s3_review_response.py, recon7_s3.txt:14, :24 | as row 21; 2,778.3 = CP:12 - MISSING: not retained here (the strongest run record is not in BND; no listed script read it) | MISSING (2,778.3 only); k values PARTIAL |
| 23 | 2.1 / I3 r4a C128: 0.110 / 207.377 / 6.004 / 151.363; round 365.423; residual +0.569 | s1_records.py, recon7_r2_s1.txt:26; v1_records.py, recon7_v_v1.txt:14-15 | R/i3_r4a_2195576_i3a/summary.json (NOT TRANSFERRED, above 5 MB; sha256 in MANIFEST); step records (BND) | PARTIAL |
| 24 | 2.1 / L4PTFD profile i4c e4a C128: phases, round 357.981; busy 195.7-198.9, H2D 18.2-21.3 | s1_records.py, recon7_r2_s1.txt; s2_traces.py / v2_traces.py for busy and H2D | BND/r2-runs/spec/i4c_e4a_2195576_i4ca/summary.json (phases: FULL); traces (NOT TRANSFERRED) for busy / H2D | PARTIAL (phases FULL) |
| 25 | 2.1 / I3 r3b C73: 0.115 / 382.041 / 8.351 / 194.550; round 585.463 | s1_records.py, recon7_r2_s1.txt:89 | R/i3_r3b_2195575_i3b/summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 26 | 2.1 / L4PTFD profile i4c c4b C73: phases, round 581.517; busy, H2D 156.2-160.1 | s1_records.py, recon7_r2_s1.txt; s2_traces.py | R/i4c_c4b summary.json (NOT TRANSFERRED, above 5 MB); traces (NOT TRANSFERRED) | PARTIAL |
| 27 | 2.1 / BEST d3a C128: 6.178 / 186.011 / 132.287; round 324.883 | s1_records.py, recon7_r2_s1.txt:636; s2_extract.py, recon7_r3_s2.txt:788 | R/i4dc_d3a summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 28 | 2.1 / BEST d4b C73: 8.546 / 338.961 / 176.437; round 524.659 | s1_records.py, recon7_r2_s1.txt; s2_extract.py, recon7_r3_s2.txt | R/i4dc_d4b summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 29 | 2.1 / outside round_ms +41.6 / +24.0 / +54.1 / +41.5 | s3_review_response.py, recon7_s3.txt:12, :16, :19, :22 (from V4) | as row 21 | PARTIAL |
| 30 | 2.2 / four rows (round - (a) 178.024 / 322.055 / 137.484 / 261.251 and parts) | s4_review2.py, recon7_s4.txt:104-107 | measured side as rows 23-28 (PARTIAL); model interval, joint budget, lending use U, I and traffic (AUTHOR) - MISSING: not retained | MISSING |
| 31 | 2.2 / per 3 accepted 379.48 / 631.82 / 337.38 / 566.20 ms | s3_review_response.py, recon7_s3.txt:27-30; ratios vs 263.41 by hand | as row 30 | MISSING (model side) |
| 32 | 2.5 / original probe set: 11 rounds, joint median 374.565, L7 residual -12.675 | CITED SI:164; v4b_probe_rounds.py, recon7_v_v4b.txt:10-14 re-reads the list | R/i4c_c4b_2195576_i4cb/summary.json (NOT TRANSFERRED, above 5 MB; sha256 108f9289...) | PARTIAL |
| 33 | 2.5 / reanalysis C73: excluded rounds 176 / 192 / 208 (49 / 288 / 315 finished); drain-free 391.019 / 390.495 | v4b_probe_rounds.py, recon7_v_v4b.txt:21-33 | R/i4c_c4b summary.json (NOT TRANSFERRED, above 5 MB) | PARTIAL |
| 34 | 2.5 / reanalysis C128: rounds 176, 192 (79 / 210 finished); 211.132 vs 210.182 | v4b_probe_rounds.py, recon7_v_v4b.txt:46-57 | BND/r2-runs/spec/i4c_e4a_2195576_i4ca/summary.json | FULL |
| 35 | 2.5 / drain shares 13.3 % / 20.0 % (steady) and 20.0 % / 27.3 % (probe) | s3_review_response.py, recon7_s3.txt:40 (steady rounds and rounds >= 170 from recon7_v_v4.txt:20-22) | R/i3_r4a, R/i3_r3b summary.json (NOT TRANSFERRED); probe lists as rows 33-34 | PARTIAL |
| 36 | 2.5 / SENSITIVITY SCENARIOS +0.95 / +16.45 ms; C73 gap about 301.5 ms | s3_review_response.py, recon7_s3.txt:38-39; 301.5 by hand (285.005 + 16.454) | +0.95: e4a in BND (FULL); +16.45: c4b NOT TRANSFERRED (above 5 MB) | PARTIAL |
| 37 | 2.6 / GEMM scenario 43.67 / 64.29 vs measured 64.80 / 79.88 | s4_review2.py, recon7_s4.txt:45, :48 (the log says "floor"; relabelled scenario in the ledger) | 312 TF: vendor spec (CITED R2:165, external constant); FLOPs 13.624 / 20.057: RV row 19 (derived from shapes, no raw file); measured: traces (NOT TRANSFERRED) | PARTIAL |
| 38 | 2.6 / attention scenario 21.51 / 31.34 vs fork 37.12 / 52.17 | s4_review2.py, recon7_s4.txt:45, :48; call A rate recon7_r2_s2.txt:110 | traces (NOT TRANSFERRED); KV logical page count from i3_r2cb (BND) | PARTIAL |
| 39 | 2.6 / PCIe scenario 22.12 / 132.84 vs measured 21.30 / 160.14 (counter-example at C128) | s4_review2.py, recon7_s4.txt:45, :48 | 25.2 GB/s: /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/pcie-feed-evidence.md:11 (a note) - MISSING: not retained (the bulk-H2D run record behind the note is not named in the packet); H2D MB: R2:136 from summaries (e4a BND; c4b NOT TRANSFERRED); measured: traces (NOT TRANSFERRED) | MISSING |
| 40 | 2.6 / unmodelled stages 60.60 / 75.69; schedule A 147.90 / 304.15; schedule B 104.27 / 208.53 | s4_review2.py, recon7_s4.txt:46-50 | traces (NOT TRANSFERRED); A and B also use row 39's 25.2 GB/s | MISSING (A, B); PARTIAL (60.60 / 75.69) |
| 41 | 2.6 / ISOLATED KERNEL FORECASTS 65.18 / 95.63 (hand) | hand sum of rows 37 + 38 | as rows 37-38 | PARTIAL |
| 42 | 2.6 + conventions / ORIGINAL INFERRED JOINT BUDGET 70.156 / 97.036 | CITED REQ:40; reproduced s4_review2.py, recon7_s4.txt:19, :23; s3_derive.py MODEL table | td, GEMM, s in BND; extra uses U, I (AUTHOR) - MISSING: not retained | MISSING |
| 43 | 5.3 / measured-config R 365.42 / 585.46 / 324.88 / 524.66 and tok/s | as rows 21, 23-28 | as rows 21, 23-28 | PARTIAL |
| 44 | 5.3 / measured-config V1-V3 (e.g. I3 C128 382.12 / 336.21 / 318.96) | s4_review2.py, recon7_s4.txt:98-101 | s (BND); overlap term (AUTHOR U, I) - MISSING: not retained; /strong denominators 3,426.2 / 3,410.1 = CP:9-11 - MISSING: not retained here | MISSING |
| 45 | 5.3 / anchor table (R 470.97 ... V3 311.34; ratios 0.47-0.78) | s4_review2.py, recon7_s4.txt:70-95 | AUTHOR inputs (MISSING: not retained); measured terms transcribed from R2 / R3 / S3 (as rows 4-11); CP strongest (MISSING: not retained here) | MISSING |
| 46 | 5.3 / drain sensitivity 11.74 ms (anchor, scenario) | s4_review2.py, recon7_s4.txt:71 (lin of 0.950 and 16.454) | as row 36 | PARTIAL |
| 47 | 5.4 / GEMM dispatch 100.10 / 140.54 vs 95.24 / 136.81; measured 104.37 / 131.29 | s4_review2.py, recon7_s4.txt:65-66 | gemm_mscan.json (BND); measured: traces (NOT TRANSFERRED) | PARTIAL |
| 48 | 5.4 / draft positions 3.084 / 3.121 (fixed lead) and 3.144 / 3.205 (accounting) | s4_review2.py, recon7_s4.txt:67-68 | as row 16 | PARTIAL |
| 49 | 5.4 / attempts lost 3.129 / 3.188; elapsed rounds 1.70 / 1.70 | s4_review2.py, recon7_s4.txt:67-68 (n_bar 0.871 / 0.812 transcribed from R3:183-184); v1_records.py, recon7_v_v1.txt:258 | as row 16 | PARTIAL |
| 50 | 5.4 / Y 2.918 / 2.884; 2.863 / 2.808; 2.855 / 2.798 | s4_review2.py, recon7_s4.txt:67-68 | as row 16 | PARTIAL |
| 51 | 5.4 / numerators 109,318 / 160,248; 1,058 / 2,250; 110,376 / 162,498 | s4_derive.py, recon7_r3_s4.txt:254-255; v1_records.py, recon7_v_v1.txt:247 | as row 16 | PARTIAL |
| 52 | 8 / replicate i4dc r4a: 1,916.8 tok/s, /SH 0.5750, 12 CONFIRMED, COMPLETED 0:0 | s2_round_split.py, recon7_s2.txt:18-20 (ratios) | BND/r2-runs/spec/i4dc_r4a_2195576_i4dca.step/{stdout.txt (lines 86-98), exit.txt, sacct.txt, command.txt}; summary.json NOT TRANSFERRED (above 5 MB) | FULL (the cited stdout lines and exit / sacct) |

## Counts

52 rows: FULL 5 (rows 1, 3, 7, 34, 52); PARTIAL 34; MISSING 13 (rows 6, 10, 12, 15, 19, 22, 30, 31, 39, 40, 42, 44,
45). Row 22 counts as MISSING (2,778.3 only); row 40 as MISSING; row 24 as PARTIAL.

## What is missing, explicitly

1. AUTHOR inputs of review 2 (U, I, traffic, overflow, td rule, memory rule constants): a coordinator message; NO FILE
   exists. The td and backbone POINTS do appear in output3_timing_bw.csv and output2_memory.csv (BND); the other
   values do not have a raw record in the packet. MISSING: not retained.
2. Strongest cached NOSA (CP:9-12: 3,426.2 / 3,410.1 / 2,778.3 tok/s): no listed script read the raw record; it is not
   in BND. MISSING here (not retained in the packet); CP names the source.
3. 25.2 GB/s bulk pinned H2D: a note line (paper-notes/pcie-feed-evidence.md:11), not a run record. MISSING: not
   retained in the packet.
4. Six of the seven core summary.json files are above 5 MB and are NOT TRANSFERRED; sha256 pins them. Their step
   records are in BND. The lead can lift the 5 MB cap for these six (34.8 MB total) if review needs the files.
5. Traces (18) are NOT TRANSFERRED by rule; sha256 pins them.
6. The OOM record R/i4m_n373_2195575_i4m73/summary.json (8,368 B) and its stdout.txt (2,260 B) are not in BND (no
   listed script read them; the ledger cites SI). Their step command / exit / sacct ARE in BND.
