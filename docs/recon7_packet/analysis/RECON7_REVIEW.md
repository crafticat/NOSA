# RECON7 independent review: R1 (model), R2 (joint), R3 (draft / transfer / capacity / schedule)

Written 2026-10-30T10:29:52Z (from `date -u`). Reviewer: independent agent. Task: try to refute the three files.
Style: ASD-STE100. All paths absolute.

Files under review (all in /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/):
- R1 = RECON7_R1_model.md, R2 = RECON7_R2_joint.md, R3 = RECON7_R3_draft_transfer_capacity_schedule.md.
- REQ = HUMAN_RECON_REQUEST_2026-10-30.md.

My evidence. I did not use the agents' logs for values. I read the raw records and calibration files with my own
scripts. Each script ran in `srun -p amd`, foreground, in the nosa-env image, `python -I -B`.
- Scripts: /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_v/v1_records.py, v1b_tfc.py, v2_traces.py, v3_model.py,
  v4_probe_wholerun.py, v4b_probe_rounds.py.
- Logs (V1 .. V4B below): /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_v_v1.txt, recon7_v_v1b.txt,
  recon7_v_v2.txt, recon7_v_v3.txt, recon7_v_v4.txt, recon7_v_v4b.txt.
- Raw inputs: /mnt/beegfs/ojerbi/r2-runs/spec/{i3_r4a_2195576_i3a, i3_r3b_2195575_i3b, i3_r2cb_2195574_i3cb,
  i4c_e4a_2195576_i4ca, i4c_c4b_2195576_i4cb, i4c_c3a_2195575_i4ca, i4dc_d3a_2195575_i4dca, i4dc_d4b_2195576_i4dcb}/
  summary.json and the trace_L4PTFDP_g{96,112}_{joint,draft0,draft1}.json.gz files;
  /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output2_memory.csv and
  output3_timing_bw.csv; .../rows_bench_2175525/gemm_mscan.json; .../cpupack_confirm_2179735/scoring/layout_variants.csv.
- Code trees: the five joint-path files (spec_tentative, spec_tentative_fork, spec_tentative_union, verify_rows,
  spec_round) are byte-identical in /mnt/beegfs/ojerbi/nosi-r2-i3 (c4c64b2), nosi-r2-i4c (3aef060) and nosi-r2-i4dc
  (2e9e497); all three trees are clean (`diff | wc -l` = 0; `git status --porcelain | wc -l` = 0). R2's claim holds.

Summary. The arithmetic is good: 36 of 39 checked numbers reproduce from raw data (table at the end). The problems
are in labels, boundaries and coverage: one blocking finding (a causal label that the request forbids), seven major
findings, and ten minor findings.

---------------------------------------------------------------------------------------------------------------------

## Findings

### 1. BLOCKING. R2 calls the max() overlap a "false model assumption" without a concurrency test

- Evidence. R2:41-42 ("Time does not match, because the model takes a MAX ... and the pass is ONE serial stream"),
  R2:50-51 ("the max() overlap assumption alone 34.81 / 50.74"), R2:227-228 and R2:240-241 ("FALSE MODEL ASSUMPTION
  (overlap)"), R2:271-272 ("false model assumption, overlap ... 34.807 / 50.737").
- The traces confirm the facts: one stream only, (pid 0, tid 7, stream 7), for all 9,011 joint ops and all 2,867
  draft ops; sum of durations = merged busy (overlap 0.0000 ms); H2D-kernel overlap 0.0000 ms (V2). But R2 itself
  says at R2:321 (D1) that the zero is "by construction, not a measured limit". One stream cannot show overlap. So
  the traces show that the IMPLEMENTATION does not overlap. They do not show that the model's overlap is false.
- REQ:58-59 and REQ:63-64 forbid this step: "no-kernel limitations are not evidence of a limit on the proposed
  design"; "GPU busy percentage ... do not establish an architectural compute limit".
- Also, "because" overstates the share. The max() term is 34.8 of 137.2 ms (25 %) at C128 and 50.7 of 285.0 ms (18 %)
  at C73 (R2's own serial chain, R2:268-283).
- Correction. Relabel the 34.807 / 50.737 ms (and the "overlap" parts of the R2 3.2 rows) as "MECHANISM NOT REALIZED:
  the implementation runs one dependent stream; feasibility of the overlap is UNTESTED". Keep "false model
  assumption" only for work the model omits (verify-row selection, tentative state). Make R2 D1 the named
  discriminator. Remove "because" at R2:41.

### 2. MAJOR. No whole-run window; the best arms are SLOWER than I3 in the whole run

- Evidence (V4). Whole-run effective tok/s: I3 C128 1,396.7, best C128 (L4PTFDGUDCD) 1,292.4; I3 C73 1,275.3, best C73
  (L4PTFDGUDKWCD) 1,242.5. The steady medians go the other way: 1,709.8 -> 1,924.1 and 1,525.8 -> 1,699.0. Cause seen
  in the record: capture_rounds [[1, 9594.118]] (C128) and [[1, 10140.224]] (C73): one 9.6 / 10.1 s capture round.
- Time outside the median round (DERIVED by hand from V4 values): I3 C128 (wall 79.180 - prelude 0.218) / 194 rounds =
  407.0 ms per round vs median 365.4 (+41.6 ms); I3 C73 609.5 vs 585.5 (+24.0); best C128, capture removed,
  (85.569 - 2.457 - 9.594) / 194 = 379.0 vs 324.9 (+54.1); best C73 (131.034 - 2.562 - 10.140) / 209 = 566.2 vs
  524.7 (+41.5).
- R2:192 gives "about 42.7 per round (MODEL upper bound)" for between-round host work. R3 gives a whole-run YIELD
  (R3:46-47), not a whole-run TIME. No file reconciles wall_s. REQ:31-32 asks to keep "whole-run / steady timing
  windows ... distinct".
- Correction. Add one table per arm and C: wall_s = prefill (outside) + prelude + capture + sum over rounds of
  (round_ms + between-round host). Today the record has no per-round round_ms list. Smallest discriminator (CPU
  after a cell change, no new GPU method): export the per-round ev0..ev4 values and the host time between ev4 and the
  next ev0 that the code already records or can record (ST:705-719). Until then, state that the best arms lose to I3
  in whole-run tok/s because of the capture round.

### 3. MAJOR. The capacity verdict ("Capacity is NOT a gap; we fit +11 / +9") depends on the memory basis

- Evidence. R3:41-43 and R3:59 compare OUR allocated peaks with the model formula on its DEVICE-USED basis. The
  arithmetic reproduces (V3): model largest B under 84.0 GB = 205.38 (C128) and 309.11 (C73); 216 - 205.38 = +10.6;
  318 - 309.11 = +8.9.
- On a like-for-like ALLOCATED basis (V3, the model formula with the J91 allocated lines in place of device-used):
  C128 largest B = 213.2 (process-peak line) / 215.7 (steady allocated line); C73 = 340.9 / 342.4. Ours: C128 216
  (fit 216.4 at 84.0 GB), C73 measured 318, fit 351-352. So at C128 the margin is +0.3..+2.8, not +11. At the
  MEASURED C73 batch (318) we are BELOW the model's 341.
- If the J91 light-restore snapshot is a harness-only cost (R1:143-146 and R3:156-157 both say so), remove it too:
  the model gives 227.6-230.0 (C128) and 379.1-380.0 (C73) (V3). Then the model fits MORE requests than our layout,
  by 12-14 (C128) and 27-29 (C73, against our fit).
- Correction. Replace "Capacity is NOT a gap" with "basis-dependent": state the basis of each side. Compare on one
  basis: allocated bytes, prefill and decode peaks separate, with and without the harness snapshot. The smallest
  discriminator is CPU only: a category-by-category table of J91 output2_memory.csv columns against our arena record
  (store, CB0, tentative buffers, graph pools, prefill transient).

### 4. MAJOR. R3's cause of the 5.4-5.9 GB basis difference does not add up; R1 and R3 use different J91 segments

- Evidence. R3:156-157: the model counts "J91's harness staging 5.60 GB and allocator slack 5.32 GB". 5.60 + 5.32 =
  10.92 GB, but the difference is 5.35-5.86 GB. In output2_memory.csv the staging snapshot is a category INSIDE
  allocated_GB (B256: allocated 60.997 = weights 16.371 + window 33.955 + tail 0.539 + compressed 4.967 + staging
  4.992 + small terms). So it cannot be part of the gap ABOVE the allocated line.
- On the model's own segment (B256-344, which R1 shows reproduces the backbone to 0.129 GB, R1:73), V3 gives at B287:
  backbone 72.248 - allocated 66.391 = 5.857 GB = allocator slack 4.609 + outside-allocator 1.119 + back-solved
  constant 0.129. Exact.
- R3 used the B64-344 line (V3: device-used 72.749 there vs 72.119 on B256-344; slack 5.322 there vs 4.609). So R3's
  "matches device-used (72.75) / reserved (71.71-71.83)" (R3:59, :154-156) is a segment artefact. On B256-344 the
  backbone matches device-used + 0.129 and lies 1.25 GB above reserved (71.000).
- Correction. R3 row 3 cause (1): "the backbone = J91 device-used (B256-344 line) + 0.129 GB; it is 5.86 GB above the
  allocated line (slack 4.61 + outside allocator 1.12 + 0.13); the 5.60 GB staging snapshot is a further harness
  term INSIDE allocated". Use the B256-344 segment in both files.

### 5. MAJOR. The three files disagree on whether the K3 / lag-2 schedule is implemented

- Evidence. R2:117-118: "the model's lag-2 alignment of verify rows with draft ticks is NOT implemented". R1:234: "YES
  under the reading of 2.1 (INFERENCE)". R3:44-45, :175: "equals Figure 3 ... It is not a substitute schedule."
- The record supports R1 / R3 on positions. V1 trace_tail, request 0, i4dc_d3a: round 169 J verifies 16632, 16633,
  16634 (drafted in D1 of round 167, J of 168, D0 of 168), defers 16635 (D1 of 168), drafts 16636. The newest verified
  draft is 2 passes old. Round 0 J (u 3, v 2) checks d1, d2, defers d3, drafts d4: Figure 3 of
  /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/speckv/speckv.txt:126-140.
- But Figure 3 also says "Requests from d1 and d2 are packed on the CPU and transferred during subsequent drafting"
  (speckv.txt:133-135) and section 3.3 protects requested blocks until verification reads them. R3 row 2 itself
  says this is NOT realized (R3:58: drafts request nothing; no packer; no retention). So "equals Figure 3" is true for
  rows and positions only. R2's "NOT implemented" has no cited evidence.
- Correction. One sentence in all three files: "Rows, positions and the 2-pass verdict lag match the K3 deferral
  design (record trace, rounds 0 and 169). The lag's purpose (draft-issued background transfer, retention until the
  verify read) is NOT implemented. The author must confirm the lag semantics (R1 open question 4)."

### 6. MAJOR. Two different "one-step body above td" values (15.29 vs 11.33 ms) for the same label

- Evidence. R2:54 and R2:260: "one-step body (draft0 busy vs td) 15.294 / 18.203 ... belongs to R3 / R1". R3:33-35:
  "+11.33 / +12.92 ms". Both reproduce (V1, V2): R2 = L4PTFDP draft0 DEVICE BUSY of profiled round 112 (70.107 /
  93.498) - td; R3 = best arm ELAPSED draft step (66.144 / 88.219, graphs + compile) - td.
- Other values of the same quantity in the records (V1): L4PTFD elapsed 71.632 / 94.772 -> +16.82 / +19.48; the I3 arm
  L4PTF (not lean) draft phase / 2 = 75.68 / 97.28 -> +20.87 / +21.98.
- Not an arithmetic conflict. It is an arm and boundary difference (lean vs full draft; profiled busy vs elapsed; eager
  vs graphs + compile; nova07 profile job vs nova05). But R2 hands its 15.29 to R3, and R3 reports a different
  number. In R2's serial chain, the draft0 row is only a reference pass: any reference gives the same total. So
  15.29 is not a property of the I3 joint pass.
- Correction. Name the arm and boundary in each value. In the handoff, give one row per arm (I3 L4PTF, L4PTFD, best)
  with elapsed and busy. In R2, rename the row "reference pass (L4PTFDP draft0 busy) minus td" and drop "R3 owns it".

### 7. MAJOR. Different model allocations for the same joint pass (70.16 vs 66.18 ms; interference 6.71 vs 17.92 ms)

- Evidence. R2 uses the REQ budget, max(td + extra, GEMM(B) + GEMM(3B))(1 + s) = 70.1555 / 97.0356 (dense binds;
  V3 reproduces). R3:198 uses (td + extra)(1 + s) = 66.18 / 89.63, a slice of the HBM-binding cohort (V3: 2 td(1 + s)
  121.2222 + 66.1768 = 187.399 = cohort GPU). R1:279 gives a third value with GEMM(4B): 66.18 / 93.02. The
  interference allowance is s x joint dense = 6.71 / 7.07 in R2:103-104, but s x cohort HBM = 17.92 / 18.36 in R3:58.
- So the best-arm joint gap is 186.011 - 70.156 = 115.86 (R2:302) or 186.011 - 66.18 = 119.83 (R3:198 implied).
- Correction. State one allocation rule in the handoff (REQ:40 gives the per-pass max). Show the others as
  sensitivity, labelled. Apply the interference allowance at the same boundary in R2 and R3.

### 8. MAJOR. The "steady" window holds drain rounds; R2's probe correction compares two different windows

- Evidence (V1, V4, V4B). steady_rounds 181 of 194 (C128) and 195 of 209 (C73): all rounds except round 0 and the
  probe rounds. Requests finish from round 170 (C128 max 193; C73 max 208). So 24 / 39 rounds of the steady window
  have finished requests. R2:310-315 removes drain rounds from the PROBE median (391.019 ms, reproduced) but compares
  it with the steady median 383.691 that still holds drain rounds. That is the same mixing that R2 criticises in L7.
- The effect on the joint median is a few ms at C73 (probe medians: 391.0 without drain vs 374.6 with it). It moves
  R2's C73 gap 285.0 and the "+7.788 ms probe overhead" by an unknown share.
- Correction. Define "steady" as rounds 1..169 minus probes (all requests active) and state it. Smallest
  discriminator: the per-round export of finding 2 (median over rounds < first finish).

### 9. MINOR. "Dense formula met (+2.1 %)" hides that the one-call GEMM is 10.6 % above the model's own curve at M 864

- Evidence. GEMM device ms verified (V2): 64.800 (C128) = 31.417 + 17.696 + 6.840 + 6.750 + 2.097; 79.879 (C73). 129
  launches; token tiles 7 x 128 = 896 >= 864 and 10 x 128 (or 5 x 256) = 1,280 >= 1,272: one call per weight over 4B
  rows. The model curve for that call shape, GEMM(4B), is 58.577 (C128) and 86.239 (C73) (V3; both EXTRAPOLATED by
  the proportional rule). Measured - GEMM(4B) = +6.22 (C128) and -6.36 (C73).
- R2:226 compares with the two-call charge GEMM(B) + GEMM(3B) = 63.445. R1:235 shows GEMM(4B) is 4.87 ms cheaper
  than that on the curve. The proportional rule above M 768 is unvalidated (R1:75-78, :321).
- Correction. Report both comparisons. Mark the dense row "extrapolated reference, sign unknown". The scan at M 864 /
  1,272 (graph-timed, existing harness) closes it.

### 10. MINOR. 72.24 pages: different basis from the model, and measured only at C73 but used at C128

- Evidence (V1B). 72.2389 = fork_loaded 47,046,615 / union_samples 651,264 (CHECK.TFC of i3_r2cb, 32 passes, C73).
  union_mean = 71.1764. fork_loaded - union_sum = 1.0625 per sample (the tail / buffer block). The model's 63 + U - I
  = 71.969 counts 63 historical groups and no tail; td's own step reads 64 (63 + tail). Like-for-like ratios: union /
  (63 + U - I) = 0.989; fork_loaded / (64 + U - I) = 0.990. R2:39-40 says x1.004.
- R2:132-133 applies the C73 count to C128 (998,631 pages, 32.85 GB). The C128 fork rate 0.885 TB/s and the
  "inefficient fork 15.60 ms" (R2:166-167, :227) depend on that transfer. R2 D3 (R2:323) admits this.
- Correction. Write "x0.99 like-for-like (C73 check arm only)". Mark the C128 fork byte rate MODEL until D3 runs.

### 11. MINOR. The "fork inefficiency" is inferred from a rate comparison that is not like-for-like

- Evidence. R2:166-169 compares the fork's bytes / ms with the one-row flash call A. The fork serves 4 rows x 16
  query heads per KV head with per-row masks over a union; call A serves 1 row. Registers 255 and local_words 12 are
  facts (V1: compile.regs 255, local_words 12). The 15.60 / 20.83 ms "INEFFICIENT EXECUTION" label is a hypothesis.
- Correction. Label it "hypothesis, pending R2 D2 (ncu)".

### 12. MINOR. R2's residual 0.000 is a closure by construction; single-pass device numbers have a few-ms spread

- Evidence. R2:233 and :246 close the gap with "steady non-busy = I3 joint - profiled busy". So the residual is zero by
  definition. The profiled values come from one pass. Round 96 vs round 112 (V2): joint busy 195.690 vs 198.923
  (C128) and 379.285 vs 383.914 (C73); H2D 18.248 vs 21.301 (C128, -14 %) and 156.164 vs 160.136 (C73).
- Correction. Say "exact partition of one profiled pass plus a cross-job steady correction". Give each item as a
  g96..g112 range. Do not quote three decimals.

### 13. MINOR. R1 puts a COMMITTED yield into the model's ACCEPTED-draft Y

- Evidence. R1:302-303: "C128 Y 2.8926 (median-round yield)". V1: 2.8926 = median_tok_s x round_ms_median / B (I3),
  a committed-token value (corrections included) built from two different medians. The accepted median per steady
  round is 2.8889 (C128) and 2.7799 (C73) (V1, same token streams as I3; R3:46). R1's C73 value 2.8091 vs accepted
  2.7799.
- Correction. Use accepted 2.889 / 2.780 in the model's Y, or label R1's row "committed, mixed medians".

### 14. MINOR. R3's "usable ceiling about 83.75 GB" contradicts R3's own C128 run

- Evidence. R3:59 and :159-161: usable ceiling about 83.75 GB after SH (lesson L60). V1: the C128 best arm peaked at
  peak_with_prefill_GB 83.900 (L4PTFD 83.891) after SH in the same process, and it ran. R3:151 says B216 "= fit",
  which needs 84.0 GB: (84.0 - 19.020) / 0.30033 = 216.4; at 83.75 GB it is 215.5.
- Correction. Treat the 83.75 GB value as one OOM observation that depends on non-PyTorch memory (0.551 GB there), not
  as a ceiling. Recompute B_fit(C81) 321.5 with an explicit range.

### 15. MINOR. "Link rate better than modelled (21.6-27.6 vs 10.7 GB/s)" compares different mechanisms

- Evidence. R3:37-38, :58. The model's r is the useful rate of a CPU8 packer + DMA BESIDE the decode, with interference
  s (R1:68-69). Our rate is an SM-driven zero-copy gather (flash_h2d_from_mask kernels, V2) on the compute stream,
  with no concurrent work. Also, MB is a mean over all rounds and the wait is a median over 12-13 probe rounds.
- Correction. Say "the in-stream gather runs near the PCIe rate; it is not comparable with the model's background
  rate".

### 16. MINOR. The interference allowance is placed in different cause classes in R2 3.2 and 3.3

- Evidence. R2:229 nets the 6.710 ms allowance against residency / PCIe. R2:271-272 nets it against "overlap". The
  allowance is the model's charge for transfer interference, which our code replaces with serial H2D.
- Correction. Net it against the serialization class in both chains.

### 17. MINOR. "Recovery EXPLAINED" is slightly overstated

- Evidence. The measured 1.70 / 1.70 reproduces (V1: 1.7013 / 1.6956). It charges ALL extra active rounds to rejects.
  End effects also add rounds: final groups commit 459 / 216 = 2.125 tokens (C128), not 3. The hand prediction
  (R3:181-186) assumes every hist[1] group is a reject (up to 52 / 85 are full v = 1 groups). The two biases can offset.
- Correction. Say "consistent within about 5 %", and separate the end-effect rounds.

### 18. MINOR. Coverage gaps against REQ

- The reference baseline. The model's +17.17 % uses the old ordinary 3,160.822684 tok/s (REQ:17). The measured own SH at
  C128 / B216 is 3,334.0 (I3 record) and vs_strongest 0.5018 implies 3,407 tok/s. No file restates the model gain
  against the measured baseline (REQ:31-32: "keep strongest cached NOSA ... distinct").
- Decision cost. The commit phase (6.0 / 8.35 ms per round) is measured but not split into decision vs commit (R2 D6).
- Online research, preserved failures and lessons (REQ:60-69) are not in R1-R3. RECON7_R4_prior_art.md exists; I did
  not review it.
- No single I3 per-round table. R3:195-201 reconciles the BEST arm's round against the model; R2 reconciles the I3
  joint and uses L4PTFD for the drafts. The I3 round (365.423 / 585.463 ms vs model 187.40 / 252.05 or per-pass
  191.38 / 259.46) has no one table with spans, overlap and residual.
- Covered adequately: query positions (R1:202-205, R3:169-175, verified in the record), rows and passes (R1:229-233,
  verified in the grids), invocation and launch counts (R2:128-143, verified: 8,014 + 416 + 97 + 452 + 32 runtime
  calls; 0 sync calls), bytes (R2 / R3), dependencies (R2:211-215), correction / bonus (R1:243, R3:60).

### 19. Rule slips of the three agents (reported by them; I confirm the reports are present)

- R1:336-347: two `sed` uses on login (one by R1, one by its sub-agent), output only.
- R3:234-237: shell variable names containing "sed" / "awk"; no program ran. The added `--bind /mnt/central/users/ojerbi`
  is in my given template too, so I do not count it.
- R2:331-333: none reported. I found none in its command list.

---------------------------------------------------------------------------------------------------------------------

## Contradictions between files (check 2)

| # | quantity | R1 | R2 | R3 | kind |
|---|---|---|---|---|---|
| a | draft step vs td | - | +15.29 / +18.20 (L4PTFDP draft0 busy, profiled) | +11.33 / +12.92 (best arm elapsed) | arm + boundary (finding 6) |
| b | joint model allocation | 70.1555 / 97.0356; 66.18 / 93.02 with GEMM(4B) | 70.156 / 97.036 | 66.18 / 89.63 | boundary (finding 7) |
| c | interference charge | s x cohort | 6.71 / 7.07 (s x joint dense) | 17.92 / 18.36 (s x cohort HBM) | boundary (finding 7) |
| d | lag-2 schedule | matches (INFERENCE) | NOT implemented | equals Figure 3 | REAL conflict; R2 unsupported (finding 5) |
| e | memory backbone basis | device-used B256-344 + 0.129 | - | device-used / reserved, B64-344 | REAL conflict; R3 segment wrong (finding 4) |
| f | why the model formula is conservative | staging snapshot inside the backbone | - | staging 5.60 + slack 5.32 | REAL conflict; R3 attribution wrong (finding 4) |
| g | Y median round | 2.8926 / 2.8091 (I3, committed) | - | 2.889 / 2.780 (best, accepted) | definition (finding 13) |
| h | joint ms for the gap | 207.368 / 382.066 (REQ rounding) | 207.377 / 382.041 (record) | 186.011 / 338.961 (best) | rounding (0.009 / 0.025 ms) and arm |
| i | NEW MB per request-round | 2.58 / 10.53 (I3) | 2.581 / 10.53 | 2.58 / 8.06-10.53 | arm (keep) |

---------------------------------------------------------------------------------------------------------------------

## Verified numbers (value claimed, value I derived, source, verdict)

| # | claim (file:line) | claimed | derived | my source | verdict |
|---|---|---|---|---|---|
| 1 | I3 joint C128 (REQ:37) | 207.368 | 207.368 (3.201 x 64.7823); record joint median 207.377; joint_sh_steps 3.2009 | V1 | OK |
| 2 | I3 joint C73 (REQ:39) | 382.066 | 382.066 (3.094 x 123.4860); record 382.041 | V1 | OK |
| 3 | budget C128 (REQ:37) | 70.1555 | 70.1555407 (dense 63.4452179 x 1.10576562) | V3 | OK |
| 4 | budget C73 (REQ:38) | 97.0356 | 97.0356169 | V3 | OK |
| 5 | joint / budget (R1:129) | 2.9558 / 3.9374 | 2.9558 / 3.9374 | V3 | OK |
| 6 | R2 joint GEMM device C128 (R2:153) | 64.800 | 64.800 (5 ampere kernel groups) | V2 | OK |
| 7 | R2 joint GEMM device C73 | 79.879 | 79.879 | V2 | OK |
| 8 | one GEMM per weight over 4B rows (R2:128-129) | 129 calls, M 864 / 1,272 | 129 launches; token tiles 7 x 128 and 10 x 128 / 5 x 256; no split-K reduce kernels | V2 | OK |
| 9 | concurrency / H2D overlap (R2:42) | 0.000 / 0.000 | 0.0000 / 0.0000; one stream (0, 7, 7) | V2 | OK (by construction, finding 1) |
| 10 | joint busy (R2:43) | 198.923 / 383.914 | 198.923 / 383.914 | V2 | OK |
| 11 | draft0 busy (R2:162) | 70.107 / 93.498 | 70.107 / 93.498 | V2 | OK |
| 12 | joint H2D device (R2:157) | 21.301 / 160.136 | 21.301 / 160.136 (128 h2d kernels) | V2 | OK |
| 13 | host syncs in pass (R2:143) | 0 | 0 sync-like calls; launches 8,014 / 416 / 97 / 452 / 32 | V2 | OK |
| 14 | round-span residual (R2:191, :206) | +0.445 / +0.547 | +0.445 / +0.547 | V1 | OK |
| 15 | I3 round residual (R2:195, :208) | +0.569 / +0.406 | +0.569 / +0.406 | V1 | OK |
| 16 | pages per (layer, KV head, request) (R2:132) | 72.24 | 72.2389 = 47,046,615 / 651,264 | V1B | OK (basis, finding 10) |
| 17 | pages vs model (R2:39-40) | x1.004 | 1.0038 as stated; 0.990 like-for-like | V1B | NOT OK (basis) |
| 18 | verify-row selected slots (R2:135) | 189.52 / 188.73 | 2,619,984.3 / 13,824 = 189.52; 3,841,019.1 / 20,352 = 188.73 | V1 | OK |
| 19 | GEMM % of 312 TF peak (R2:165) | 67.4 % / 80.5 % | 13.624 TF / 64.80 ms = 67.4 %; 20.057 TF / 79.879 ms = 80.5 % (hand; 7.884e9 linear params) | V2 + hand | OK |
| 20 | serial - budget (R2:253) | 34.807 / 50.737 | 34.808 / 50.741 | V3 | OK (rounding) |
| 21 | drain-free probe median C73 (R2:312) | 391.019 | 391.019; rounds 176 / 192 / 208 have 49 / 288 / 315 of 318 done | V4B | OK (window, finding 8) |
| 22 | R3 draft step (R3:33) | 66.144 / 88.219 | 66.144 / 88.219 (= draft phase / 2) | V1 | OK |
| 23 | R3 draft - td (R3:33-34) | +11.33 / +12.92 | +11.330 / +12.924 | V1 | OK (arm, finding 6) |
| 24 | R3 MB per request-round (R3:39) | 2.58 / 8.06-10.53 | 2.5808 / 8.0647 / 10.5272 | V1 | OK |
| 25 | R3 rates (R3:127) | 27.53 / 21.63 / 22.18 | 27.53 / 21.63 / 22.18 | V1 | OK (comparability, finding 15) |
| 26 | R3 p (R3:47) | 2.81 % / 4.03 % | 1,058 / 37,614 = 2.813 %; 2,250 / 55,866 = 4.027 % | V1 | OK |
| 27 | R3 Y median round (R3:46) | 2.889 / 2.780 | 2.8889 / 2.7799 | V1 | OK |
| 28 | R3 Y whole run (R3:46) | 2.609 / 2.411 | 2.6088 / 2.4111 | V1 | OK |
| 29 | R3 recovery (R3:47) | 1.70 / 1.70 | 1.7013 / 1.6956 | V1 | OK (finding 17) |
| 30 | R3 drain share (R3:48) | 7.7 % / 12.6 % | 3,240 / 41,904 = 7.73 %; 8,375 / 66,462 = 12.60 % | V1 | OK |
| 31 | R3 capacity margin (R3:41-42) | +11 / >= +9 | +10.6 / +8.9 on mixed bases; +0.3..+2.8 / -23 (measured 318 vs 341) on the allocated basis | V3 | NOT OK (finding 3) |
| 32 | R3 basis difference (R3:43) | 5.4-5.9 GB | 5.352 / 5.857 GB | V3 | OK (value); cause NOT OK (finding 4) |
| 33 | R3 usable ceiling (R3:59) | about 83.75 GB | C128 run peaked at 83.900 GB and ran | V1 | NOT OK (finding 14) |
| 34 | R1 GPU at measured configs (R1:290) | 187.40 / 252.05 | 187.3990 / 252.0512 | V3 | OK |
| 35 | R1 Q at measured configs (R1:295) | 3,457.86 / 3,784.94 | 3,457.863 / 3,784.945 | V3 | OK |
| 36 | R1 memory formula (R1:140-141) | 87.6647 / 86.0392 GB; largest B 205.38 / 309.11 | 87.6647 / 86.0391; 205.38 / 309.11 | V3 | OK |
| 37 | R1 GEMM(4B) cheaper (R1:235) | 3.20 / 4.87 / 3.73 | 3.199 / 4.868 / 3.727 | V3 | OK |
| 38 | R1 anchor reproduction (R1:86-101) | td 68.9864166, GPU 232.4805547, Q 3,703.5356, +17.16999 % | identical to 7 digits; transfer 173.0381233 (R1's note on r) | V3 | OK |
| 39 | R1 corrections share (R1:243) | 0.96 % / 1.38 % | 0.959 % / 1.385 % | V1 | OK |

Count: 36 OK, 3 NOT OK (rows 17, 31, 33). Rows with "OK (finding N)" have the right value but a label or boundary
problem described in that finding.

---------------------------------------------------------------------------------------------------------------------

## Reviewer rule disclosure

One slip by me. On the login node I ran `grep -o '"union_[a-z_]*": [0-9.]*' <i3_r2cb summary.json> | sort | uniq -c
2>/dev/null | head -5`. `uniq` is on the forbidden list. It only printed five lines; it wrote no file. Cause: a habit
when composing a pipe. I did not use it again.

No interpreter on login. No sed, awk, heredoc, tr, paste, xargs, timeout, sleep, mkdir, cp, mv, rm, hashing, zcat or
`find /` on login. No GPU job, no sbatch, nothing in a queue directory, no code-tree edit, no edit of any RECON7_R*
file, no commit. The Write tool made the directory /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_v/ when it wrote
the first script. All Python ran in six foreground `srun -p amd -c 8 --mem=32G -t 60` jobs with the given template.
