# RECON7 R2: the measured joint pass at the model's boundaries, and the per-round elapsed reconciliation

Written 2026-10-30T10:11:35Z (from `date -u`). Writer: workstream R2 (CPU only). Style: ASD-STE100.
Request: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md (REQ).

Labels. MEASURED = a value in a record or trace. DERIVED = arithmetic on MEASURED values (operands shown here or
printed by the script on the cited log line). MODEL = needs an assumption (stated). CITED = from another file (file:line).
CODE = read in the source at file:line.

Abbreviations (all absolute paths):
- LOG1 = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_r2_s1.txt (records; script
  /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_r2/s1_records.py).
- LOG2 = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_r2_s2.txt (Kineto traces; script .../recon7_r2/s2_traces.py).
- LOG3 = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_r2_s3.txt (derivations; script .../recon7_r2/s3_derive.py).
- A6 = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/ana_rev6_s1.txt and A62 = .../ana_rev6_s2.txt (earlier dumps, reused).
- ST, STF, STU, VR, SR = /mnt/beegfs/ojerbi/nosi-r2-i3/nosi/nosi/{spec_tentative, spec_tentative_fork,
  spec_tentative_union, verify_rows, spec_round}.py at c4c64b2. These five files are byte-identical in
  /mnt/beegfs/ojerbi/nosi-r2-i4c (3aef060) and /mnt/beegfs/ojerbi/nosi-r2-i4dc (2e9e497) (`diff`: 0 lines; all three
  trees clean). So the I4C profiles show the I3 joint code path.
- C4C = /mnt/beegfs/ojerbi/nosi-r2-i4c/benchmarks/Efficiency/r2_i4c_cell.py (the profiler window).
- R = /mnt/beegfs/ojerbi/r2-runs/spec. FC = /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/forecast-oct12-17pct.md.
  RR23 = .../paper-notes/recon-rows-2-3.md. REC = .../paper-notes/reconciliation-17pct.md. NG =
  /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/NOTES_graph_mean_gap.md. L6 = .../analysis/plan/LEDGER_rev6.md.
  GM1 = /mnt/beegfs/ojerbi/r2-runs/bi/bi_gma_2190876_gm1/summary.json. TFC = R/i3_r2cb_2195574_i3cb/summary.json.

Data used. I3 timing: R/i3_r4a_2195576_i3a (C 128 B 216, nova07), R/i3_r3b_2195575_i3b (C 73 B 318, nova05).
Profiles (L4PTFDP = I3 joint path + lean draft, torch.profiler, one window per pass, rounds 96 and 112, round 112
labelled): R/i4c_e4a_2195576_i4ca (C 128, nova07: the I3 C 128 GPU), R/i4c_c3a_2195575_i4ca (C 128, nova05),
R/i4c_c4b_2195576_i4cb (C 73, nova07). Best arms: R/i4dc_d3a_2195575_i4dca (L4PTFDGUDCD, C 128, nova05),
R/i4dc_d4b_2195576_i4dcb (L4PTFDGUDKWCD, C 73, nova07). Union counters: TFC (C 73, check arm, 4 verify rows).
Same-GPU check: I3 joint 207.377 (LOG1:26) and the profile job's L4PTFD joint 207.457 (LOG1:290), both nova07:
difference 0.080 ms. At C 73 the profile is on nova07 and I3 on nova05 (382.041 vs 383.231: 1.19 ms).

## 0. Answer in ten lines

1. The joint phase is CUDA event ST:949 to ST:974 (section 1). It holds the whole 4-row model pass and the in-pass
   PCIe gathers. It does NOT hold the reset, the decision / commit, the drafts or the between-round host work.
2. Work matches the model where the model counts work. Rows: 4B = B + BK (864 / 1272). Dense GEMM: 64.80 vs 63.45 ms
   (C 128, +2.1 %) and 79.88 vs 89.97 ms (C 73, -11 %). KV pages read by the fork: 72.24 per (layer, KV head,
   request), once for the 4 rows, vs the model's 63 + (U - I) = 71.97 (x1.004).
3. Time does not match, because the model takes a MAX over HBM and dense, and the pass is ONE serial stream: 9,011
   GPU ops on one stream, 0.000 ms of concurrency, 0.000 ms of H2D-compute overlap, 0 host syncs (LOG2:13-15, :603-605).
4. The device is busy for 198.92 ms (C 128) and 383.91 ms (C 73) of the pass (MEASURED). Host exposure is small:
   8.45 ms (C 128) and about 0 (C 73) (DERIVED).
5. C 128 gap 137.22 ms = attention 40.31 + tentative state / union lists 28.91 + selection 27.31 + residency / PCIe
   26.50 (after the 6.71 ms interference allowance) + host exposure 8.45 + small ops 4.39 + dense 1.35. Residual
   -0.0001 ms (LOG3:35-42).
6. C 73 gap 285.01 ms = residency / PCIe 165.17 + attention 56.11 + selection 38.47 + tentative 31.12 + small 6.10
   + dense -10.09 + non-busy -1.87 (cross-GPU). Residual 0.0000 ms (LOG3:54-61).
7. By cause (C 128 / C 73, ms, DERIVED, disjoint chain of 3.3): the max() overlap assumption alone 34.81 / 50.74
   (serial form, MODEL). Work that the model omits (verify-row selection, tentative state, union lists): 43.75 /
   51.61. Residency + PCIe in the compute stream: 20.63 / 158.10 (H2D 19.86 / 158.01 of it). Attention above the
   model: 11.65 / 14.65 (fork per-byte inefficiency 15.60 / 20.83). Launch exposure: 8.45 / -1.87. Small ops 2.74 /
   4.08. Dense -0.10 / -10.50. The one-step body itself (draft step vs td): 15.29 / 18.20. Sum 137.22 / 285.01.
8. The C 73 PCIe term is also a changed live workload: 5.00 new blocks per (layer, KV head, request) vs 1.23 at C 128.
9. Best arms at the same boundaries: gap 115.86 (C 128) and 241.93 (C 73). The compile lever removed 17.14 / 17.45 ms
   of small-op time, graphs 2.13 / 9.61 ms of non-busy time, the keep 16.81 ms of residency (C 73). The structure of
   the gap does not change (section 3.4).
10. The -12.7 ms "L7 residual" at C 73 is a sampling artefact: 3 of the 11 probe rounds are drain rounds (round >= 170,
   NG:39). Without them the probe median is 391.02 ms, not 374.57 (LOG3:87). It is not an elapsed reconciliation.

## 1. BOUNDARIES

### 1.1 The measured joint phase (CODE)

A round records CUDA events ev0..ev4 (ST:946, :949, :974, :999, :1037; SR:749-752 records each on the stream).
Phase names: reset, joint, commit, draft (ST:131). round_ms = ev0 -> ev4 (ST:1188); phases = consecutive differences
(ST:1187). So per round the four phases add exactly; medians do not. The host sync of the round (ST:712), the host
trace / token work (ST:716, :602-654) and the next host plan (ST:710, :552-585) run AFTER ev4 and BEFORE the next
ev0: they are OUTSIDE round_ms, so they are outside the 207 / 382 ms numbers and outside the median tok/s.

The "joint" phase = ev1 (ST:949, after T.reset at :947) -> ev2 (ST:974, after run_pass returns the logits).

| # | work item | in the measured joint phase? | code | in the model's joint budget? |
|---|---|---|---|---|
| 1 | reset of the twin counters | NO (reset phase) | ST:947 | no |
| 2 | q clone, meter snapshot, J-slot plan (ids, positions, masks) | YES: "plan" 0.488 ms median (A6:46) | ST:951-957, :912-938 | no |
| 3 | embedding of 4B rows | YES | ST:308 | inside td (B rows) |
| 4 | input norm, QKV GEMM over E = 4B rows (one call), nosa_linear, RoPE | YES | ST:327-333, VR:382-393 | GEMM: dense; rest: td |
| 5 | R twin steps per layer + the deferred replay | YES | ST:338-352, :378-393 | no (verify-row twin work) |
| 6 | batched selection of E rows: stage 1, pooling, 2 top-k | YES | VR:413-432 | B rows inside td; 3B rows absent |
| 7 | residency of row 0's selection: diff, pool update / swap, HOST GATHERS (H2D) | YES | STF:149, ST:450-480 | bookkeeping: unknown inside td (a resident proxy); H2D: the separate transfer term (interval = max(GPU, transfer), FC:30) + interference s |
| 8 | buffer writes of R rows (K, V, cis, tpos, counters) | YES | STF:152-153, ST:482-513 | no |
| 9 | chain images CB0 / CB1 | YES | STF:156, :105-121 | no |
| 10 | union / locate (sorts, searchsorted, list) | YES | STF:157-170, VR:1105-1177 | no |
| 11 | store fill of host-only blocks (HOST GATHERS, H2D) | YES | STF:171-176 | transfer term + s |
| 12 | row lists + ONE fork attention call for all rows | YES | STF:177-185, VR:1180-1209, :1095-1102 | KV reads: td (1 - f) for 63 groups + extra for U - I |
| 13 | wo GEMM, residual, norm, gate_up GEMM, SiLU x mul, down GEMM | YES | ST:363-369 | GEMM: dense; rest: td |
| 14 | final norm, LM-head GEMM over 4B rows, logits to float | YES | ST:312-313 | GEMM: dense (the scan includes the LM head, REC:59) |
| 15 | host counters, drow | YES (host) | ST:959-963 | no |
| 16 | DECISION: argmax over (B, 4, V), accept_prefix (SR:167-172), commit of 32 layers, set_twins | NO (commit phase ev2 -> ev3) | ST:975-999 | no (REQ:28-29) |
| 17 | the two D-slot draft passes | NO (draft phase ev3 -> ev4) | ST:1015-1037 | 2 td + 2 GEMM(B) (cohort, not joint) |
| 18 | round sync, host trace, next plan | NO (outside round_ms) | ST:705-719 | no |
| 19 | interference with a concurrent transfer | none exists: H2D runs IN the compute stream | LOG2:14-15 | (1 + s): +6.71 / +7.07 ms |

KV meter: meter.on is True in the joint pass only (ST:954, :971) and in drain D-slots with target rows (ST:1020).
So `kv_at_verify` NEW blocks are the joint pass's host loads (SR:425-434 counts ids >= 0 of each host gather).

### 1.2 The model's joint budget (CITED, REQ:35-40; FC:19-31)

budget = max(td + extra, GEMM(B) + GEMM(BK)) x (1 + s). Checks (LOG3:11-18, DERIVED):
- C 128 B 216: f = 15.7683 / (15.7683 + 216 x 0.132637) = 0.35500; extra = td (1 - f)(U - I) / 63 = 5.0333; HBM =
  59.847 < dense 63.4452; budget = 63.4452 x 1.105766 = 70.1555. Interference allowance = 63.4452 x 0.105766 = 6.7103.
- C 73 B 318: f = 0.27212; extra 7.8025; HBM 83.097 < dense 89.9657; budget 97.0356; allowance 7.0700.
- So at both batches the budget is the DENSE term. The HBM side (td + extra) is hidden by the max.
- td = interpolated eager median of the resident C 63 TARGET step of job 2175791 (RR23:105; REC:55). It is a full
  serial step time (GEMMs, attention, selection, norms), apportioned to bytes by f. It is not an HBM counter (REQ:28).
- dense: the old graph-timed cuBLAS scan (GM1:40-49), interpolated; past M 768 the slope that gives GEMM(861) =
  58.374 (FC:26). My reconstruction: GEMM(216) + GEMM(648) = 18.330 + 45.118 = 63.448 (cited 63.445);
  GEMM(318) + GEMM(954) = 25.290 + 64.678 = 89.968 (cited 89.966) (LOG3:12, :16; DERIVED, R1 owns the model).
- The model's KV-read share: td (1 - f) + extra = 35.355 + 5.033 = 40.388 (C 128); 54.806 + 7.803 = 62.608 (C 73).

### 1.3 Boundary mismatch to keep in mind

- The original model is a per-COHORT max (REQ:42-44); the joint budget is a reconstructed slice of it.
- Our schedule puts the 3 verify rows and the next draft row in one 4-row pass per round, then 2 one-row draft passes.
  The model's rows per cohort are the same (3 B draft rows + 3 B verify rows); the model's lag-2 alignment of verify
  rows with draft ticks is NOT implemented (R3 owns the schedule).
- The decision (item 16) is outside the joint phase. It is 5.98 / 8.34 ms per round inside the commit phase.

## 2. WORK BEFORE TIME

### 2.1 Work counts per pass (round 112, MEASURED unless marked)

| item | C 128 joint (B 216) | C 128 draft step | C 73 joint (B 318) | C 73 draft step | model (per joint) |
|---|---|---|---|---|---|
| query rows | 4B = 864; active 864 (g112), 838 (g96) (A62:73, :23) | 216 | 1272; active 1245 / 1224 (A62:346, :297) | 318 | B + BK = 4B |
| GEMM calls | 129 = 4 per layer + LM head, ONE call per weight | 129 | 129 | 129 | 2 per weight (M = B, M = 3B) |
| rows per GEMM call | 864 (grid M tiles 7 x 128; LOG2:19-32, :73) | 216 (2 x 128; LOG2:83-94) | 1272 (10 x 128 or 5 x 256; LOG2:611-621) | 318 (3 x 128) | B and 3B |
| attention calls | 32 fork calls (one per layer, all 4 rows; grid (432, 2, 1)) | 64 split-KV (call A + B) + 64 combine | 32 (grid (636, 2, 1)) | 64 + 64 | inside td |
| KV pages per (layer, KV head, row) | 63.98 visible (TFC:1139-1141, DERIVED 166,682,880 / 2,605,056) | 64 (63 + tail) | 63.98 | 64 | 63 |
| KV pages per (layer, KV head, request) | 72.24 read ONCE for 4 rows (TFC:1085: 47,046,615 / 651,264); union_mean 71.18, max 99 (TFC:1133-1134) | 64 | 72.24 | 64 | 63 + (U - I) = 71.97 (U 69.16, I 60.19) |
| KV pages per pass (DERIVED, LOG3:24, :29) | 998,631 = 32.85 GB | 884,736 = 29.10 GB | 1,470,207 = 48.36 GB | 1,302,528 = 42.85 GB | 994,900 / 1,464,714 (x1.004 measured / model) |
| selection calls | stage 1: 32 (grid (1, 864, 2)); pooling 32; top-k 64 | 32 / 32 / 64 at 216 rows | 32 / 32 / 64 at 1272 rows | at 318 rows | B rows only (td) |
| verify-row selected slots per (layer, head, request) | 189.52 = 3 x 63.18 (LOG3:26) | - | 188.73 = 3 x 62.91 (LOG3:31) | - | 3 x 63 |
| NEW host blocks per round (kv_at_verify, LOG1:28, :91) | 16,946.2 = 1.226 per (layer, KV head, request); 557.46 MB = 2.581 MB per request | loads skipped (meter off) | 101,764.2 = 5.000; 3,347.64 MB = 10.53 MB per request | skipped | transfer 6.4658 MB per request per cohort at C 81 (FC:29), overlapped |
| ready share of selected slots | 0.9935 (DERIVED 2,603,038 / 2,619,984) | - | 0.9735 | - | I / U = 0.8703 (different definition: overlap with the draft's ready groups) |
| H2D gather kernels (zero-copy reads of pinned host memory) | 128 (64 window loads grid (216, 2, 64); 64 store fill grid (216, 2, 129)) | 64 | 128 | 64 | (no kernel; a DMA term) |
| device-to-device copies | 452 memcpy = 156.2 MB + 32 memsets (LOG2:45) | 161 = 46.1 MB + 32 (LOG2:104) | 452 = 212.0 MB + 32 (LOG2:636) | 161 = 67.9 MB + 32 (LOG2:695) | - |
| GPU ops | 9,011 (g112), 9,142 (g96); SMALL class 7,437 | 2,867 | 9,011 | 2,867 | - |
| launch calls | cudaLaunchKernel 8,014 + cuLaunchKernel 416 + ExC 97 + MemcpyAsync 452 + MemsetAsync 32 (LOG2:46) | 2,225 + 160 + 97 + GraphLaunch 32 + 161 + 32 (LOG2:105) | same as C 128 (LOG2:637) | same | - |
| streams; concurrency; H2D overlap | 1 stream (0, 7); 0.000 ms; 0.000 ms (LOG2:13-15) | 1; 0; 0 (LOG2:77-79) | 1; 0; 0 (LOG2:603-605) | 1; 0; 0 | max() assumes full overlap |
| host syncs inside the pass | 0 (no Synchronize / blocking Memcpy call in the range, LOG2:47) | 0 (LOG2:106) | 0 (LOG2:638) | 0 | - ; one per round outside round_ms (ST:712) |

Reading: the joint pass does the work that the model counts, in the amounts that the model counts (rows, GEMM rows,
shared KV pages). It ALSO does work that the model omits: the selection of 3B verify rows, the tentative state and the
union lists, and the PCIe reads, all inside the pass.

### 2.2 Device ms per class (MEASURED; one profiled pass, round 112; LOG2:16, :80, :606, :671)

| class | C 128 joint | C 128 draft0 | C 73 joint | C 73 draft0 |
|---|---|---|---|---|
| GEMM (cuBLAS) | 64.800 | 19.785 | 79.879 | 25.705 |
| OTHER (fork kernel_unified_attention + 1 scan) | 37.603 (fork 37.118) | - | 52.655 (fork 52.165) | - |
| ATTN (flash split-KV + combine) | - | 20.498 | - | 29.892 |
| SMALL (elementwise, index, where, ...) | 32.718 | 9.486 | 35.943 | 11.446 |
| H2D (gather kernels) | 21.301 | 1.446 | 160.136 | 2.124 |
| POOL (pooling, pool swap / update) | 19.615 | 11.074 | 25.062 | 13.209 |
| STAGE1 | 9.816 | 4.514 | 13.540 | 6.932 |
| SELECT (top-k, sorts, searchsorted) | 7.274 | 0.930 | 9.278 | 1.287 |
| FUSED + COPY + TWINF + DIFF | 2.974 + 1.146 + 1.032 + 0.645 | 1.079 + 0.446 + 0.210 + 0.639 | 4.253 + 1.144 + 1.314 + 0.711 | 1.319 + 0.594 + 0.272 + 0.720 |
| busy (sum; no overlap) | 198.923 | 70.107 | 383.914 | 93.498 |

Rates (DERIVED, LOG3:23-31):
- GEMM: 13.62 TFLOP in 64.80 ms = 67.4 % of the A100 bf16 dense peak 312 TF (vendor spec, CITED); C 73 80.5 %.
- Fork: 32.85 GB in 37.12 ms = 0.885 TB/s. The draft's flash call A reads 29.10 GB in 19.06 ms = 1.527 TB/s. At call
  A's rate the fork's pages need 21.52 ms: excess 15.60 ms (C 128). C 73: 0.927 vs 1.543 TB/s, excess 20.83 ms.
  The fork kernel has 255 registers per thread and estimated occupancy 0 % in the trace (LOG2:18, :608); the record
  shows local_words 12 (spills) (LOG1:37).
- H2D: 557.46 MB / 21.30 ms = 26.2 GB/s (C 128); 3,347.6 MB / 160.14 ms = 20.9 GB/s (C 73). Link reference: 25.2
  GB/s bulk pinned H2D on nova05 (CITED, /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/pcie-feed-evidence.md:11).
  MODEL caveat: the block count is a mean over the probe rounds; the ms is one pass. So the rate is approximate. The
  gathers run near the link rate; they are not slow per byte. They are serial.

## 3. ELAPSED RECONCILIATION

### 3.1 The round, span by span (C 128: job i4c_e4a, nova07 = the I3 GPU)

The profiled windows cannot be stitched into one round: each pass has its own profiler session, and the trace export
runs between them (joint end -> draft0 start 5,110 ms; draft0 end -> draft1 start 1,273 ms; LOG2:196). So the round
uses the steady CUDA-event medians of the reference arm L4PTFD (same job, same tokens, bit-exact to the profiled arm),
and the device busy of the profiled passes.

| span | steady median ms (LOG1:290) | device busy, profiled g112 (g96) | non-busy = steady - busy | H2D device in span | H2D-compute overlap | host syncs |
|---|---|---|---|---|---|---|
| reset | 0.115 | no trace | - | 0 | - | 0 |
| joint | 207.457 | 198.923 (195.690) | 8.534 (11.767) | 21.302 (18.248) | 0.000 | 0 |
| commit + decision | 5.977 | no trace | - | 0 | - | 0 |
| draft step 1 + draft step 2 | 143.987 | 70.107 + 68.437 = 138.544 (137.730) | 5.442 (6.257) | 1.446 + 1.448 | 0.000 | 0 |
| sum of phase medians | 357.536 | | | | | |
| round median | 357.981 | | residual +0.445 (median non-additivity; per round the phases add exactly, ST:1187-1188) | | | |
| between rounds (NOT in round_ms) | about 42.7 per round (MODEL upper bound, LOG3:80; NG:93-95: about 40) | | | | | 1 (ST:712) |

Device busy covers 96.0 % of joint + draft (DERIVED (198.923 + 138.544) / (207.457 + 143.987), LOG3:73).
I3 (r4a, nova07): 0.110 + 207.377 + 6.004 + 151.363 = 364.854; round 365.423; residual +0.569. The I3 draft phase is
7.376 ms above L4PTFD's (the lean-draft lever, not in I3) (LOG3:74). The joint spans are equal within 0.080 ms.

C 73 (job i4c_c4b, nova07; LOG1:411; LOG3:76-78):

| span | steady median ms | device busy g112 (g96) | non-busy | H2D in span | overlap | syncs |
|---|---|---|---|---|---|---|
| reset | 0.117 | no trace | - | 0 | - | 0 |
| joint | 383.231 | 383.914 (379.284) | -0.683 (3.947) | 160.136 (156.164) | 0.000 | 0 |
| commit + decision | 8.344 | no trace | - | 0 | - | 0 |
| draft 1 + draft 2 | 189.278 | 93.498 + 91.531 = 185.029 (184.920) | 4.249 (4.358) | 2.124 + 2.143 | 0.000 | 0 |
| sum / round | 580.970 / 581.517 | | residual +0.547 | | | |

Device busy covers 99.4 % of joint + draft. I3 (r3b, nova05): 0.115 + 382.041 + 8.351 + 194.550 = 585.057; round
585.463; residual +0.406.

Inside the joint pass, one layer (layer 5, profiled, eager under the profiler; LOG2:62-72, :653-663), start offsets in
ms: pre 0.000, twin 0.267, select 0.429, resid 1.313, write 1.873, chain 2.211, union 2.330, fetch 3.194, attn 3.655,
post 5.103, next layer 6.94 (C 128). C 73: pre 0, twin 0.282, select 0.457, resid 1.675, write 2.757, chain 3.097,
union 3.248, fetch 3.741, attn 4.909, post 6.607, next 8.92. Every step waits for the one before it. The H2D gathers
(resid, fetch) stop the stream while PCIe moves the bytes.

### 3.2 The joint gap vs the budget, I3, MAX FORM (the human's formula) (DERIVED, LOG3:33-61)

Measured column: device ms of the work item in the profiled pass (round 112) at the same boundaries; the last line
moves the I3 steady joint onto the profiled busy. Gap = measured - model allocation. The items add to the gap.

C 128 B 216: I3 joint 207.377 - budget 70.156 = 137.221 ms.

| part | measured ms | model allocation | gap ms | cause class | evidence |
|---|---|---|---|---|---|
| dense GEMM, 129 calls at M 864 | 64.799 | 63.445 (dense) | +1.354 | NONE: the dense formula is met (+2.1 %) | grids, 67.4 % of peak (2.1, 2.2) |
| attention: fork 37.118 + lists / sorts 3.194 | 40.312 | 0 (KV share 40.388 hidden by the max) | +40.312 | FALSE MODEL ASSUMPTION (overlap): 24.711 = the same pages at call A's byte rate + lists; INEFFICIENT EXECUTION: 15.601 (fork 0.885 vs 1.527 TB/s) | pages x1.004 of the model: no extra KV work; 1 stream, 0 concurrency |
| selection over 4B rows | 27.310 | 0 | +27.310 | FALSE MODEL ASSUMPTION: overlap of the B-row part 9.480 + OMITTED necessary work of the 3B verify rows 17.830 | linear in rows: x2.88 for x4 rows (LOG3:47); each verify row needs its own selection |
| residency + fetch: H2D 21.302 + bookkeeping 11.905 | 33.207 | interference 6.710 | +26.497 | SERIALIZATION: H2D gathers in the compute stream, 14.592 net of the allowance; bookkeeping 11.905 = a one-row step's residency (draft 11.130 without H2D): UNKNOWN whether td holds it (td is a resident proxy) | overlap 0.000 ms; gathers near the link rate |
| tentative state + union lists: union 12.087 + write 9.329 + chain 3.504 + twin 3.989 | 28.908 | 0 | +28.908 | FALSE MODEL ASSUMPTION (omitted work: D1 / D3 buffers, chain images, union lists, R twin steps) + INEFFICIENT EXECUTION (7,437 small kernels; Inductor later removed 17.142 of elapsed, 3.4) | LOG2:51-61 |
| norms / RoPE / SiLU / embed / logits cast | 4.386 | 0 (inside td) | +4.386 | FALSE MODEL ASSUMPTION (overlap) | rows x4 |
| steady non-busy (I3 joint - busy g112; g96 basis 11.687) | 8.454 | 0 | +8.454 | INEFFICIENT EXECUTION: eager launch exposure | CUDA graphs removed 6.96 ms (L6:52) |
| **sum** | | | **137.222** | residual -0.0001 | |

C 73 B 318: I3 joint 382.041 - budget 97.036 = 285.005 ms.

| part | measured ms | model allocation | gap ms | cause class | evidence |
|---|---|---|---|---|---|
| dense GEMM, 129 calls at M 1272 | 79.879 | 89.966 | -10.087 | NONE (better than the model: one call at M 1272 at 80.5 % of peak vs the model's extrapolated curve) | LOG3:28 |
| attention: fork 52.165 + lists 3.941 | 56.106 | 0 (KV share 62.608 hidden) | +56.106 | OVERLAP ASSUMPTION 35.279; INEFFICIENT EXECUTION 20.827 (0.927 vs 1.543 TB/s) | LOG3:29 |
| selection over 4B rows | 38.472 | 0 | +38.472 | overlap of the B-row part 14.038 + OMITTED 3B-row work 24.434 | LOG3:66 |
| residency + fetch: H2D 160.136 + bookkeeping 12.107 | 172.243 | interference 7.070 | +165.173 | SERIALIZATION 153.066 (H2D in the compute stream) + CHANGED LIVE WORKLOAD (5.000 new blocks per (layer, KV head, request) vs 1.226 at C 128: x4.08) + bookkeeping 12.107 (UNKNOWN in td) | LOG3:31 |
| tentative state + union lists | 31.118 | 0 | +31.118 | omitted work + small-kernel inefficiency | LOG2:641-652 |
| norms / RoPE / SiLU / embed / cast | 6.096 | 0 | +6.096 | overlap assumption | |
| steady non-busy (g96 basis 2.757) | -1.873 | 0 | -1.873 | device-bound pass; the sign comes from the cross-GPU transfer (profile nova07, I3 nova05) | |
| **sum** | | | **285.005** | residual 0.0000 | |

### 3.3 The same gap in SERIAL FORM (separates the max() assumption from the rest) (DERIVED, LOG3:43-51, :62-70)

SERIAL FORM (MODEL, this file): the model's own terms charged in sequence, not as a max: td + extra + (dense -
GEMM(B)). It charges one full step (td), the extra KV reads, and the dense work of the 3B verify rows. C 128: 54.814 +
5.033 + 45.115 = 104.962. C 73: 75.295 + 7.803 + 64.675 = 147.773. Serial - budget = (HBM - GEMM(B)) - interference =
41.517 - 6.710 = **34.807** (C 128) and 57.807 - 7.070 = **50.737** (C 73): the part of the gap that comes from the
max() alone.

Measured = one-step device (draft0 busy, the measured analogue of td) + the joint's increment over it + non-busy.

| term | C 128 measured | C 128 model | excess | C 73 measured | C 73 model | excess | cause class |
|---|---|---|---|---|---|---|---|
| one step (draft0 busy vs td) | 70.107 | 54.814 | +15.294 | 93.498 | 75.295 | +18.203 | belongs to R3 (draft) / R1 (td basis) |
| increment: dense GEMM | 45.014 | 45.115 | -0.101 | 54.174 | 64.675 | -10.501 | none |
| increment: attention | 16.683 | 5.033 (extra) | +11.649 | 22.453 | 7.803 | +14.650 | inefficient fork (15.60 / 20.83) minus the removed call-B / merge / masks of the one-row step |
| increment: selection | 17.830 | 0 | +17.830 | 24.434 | 0 | +24.434 | omitted necessary work |
| increment: residency + fetch | 20.631 | 0 | +20.631 | 158.101 | 0 | +158.101 | serial PCIe (+ live workload at C 73) |
| increment: tentative + union | 25.915 | 0 | +25.915 | 27.171 | 0 | +27.171 | omitted work + small-kernel inefficiency |
| increment: small ops | 2.743 | 0 | +2.743 | 4.084 | 0 | +4.084 | rows x4 |
| non-busy | 8.454 | 0 | +8.454 | -1.873 | 0 | -1.873 | launch exposure / cross-GPU |
| total vs serial form | 207.377 | 104.962 | +102.415 | 382.041 | 147.773 | +234.268 | residual 0.000 / 0.001 |

Cause totals (DERIVED from the serial-form chain only, so the classes are disjoint; C 128 / C 73 ms):
- false model assumption, overlap (max instead of a serial stream; includes -6.710 / -7.070 of unused interference
  allowance): 34.807 / 50.737;
- one-step body above td: 15.294 / 18.203 (R3 / R1 own it);
- false model assumption, omitted necessary work: selection 17.830 + tentative state 25.915 = 43.745 / 24.434 +
  27.171 = 51.605 (the tentative row also holds small-kernel inefficiency; 3.4 shows 17.1 / 17.4 ms of it was removable);
- serialization: residency + fetch increment 20.631 / 158.101, of it H2D gathers in the compute stream 19.856 /
  158.012 (21.302 - 1.446; 160.136 - 2.124); at C 73 also a changed live workload (x4.08 the C 128 new blocks per
  (layer, head, request));
- attention above the model 11.649 / 14.650 = inefficient fork 15.601 / 20.827 + other terms -3.951 / -6.177 (the
  extra pages at call A's rate 2.454 / 3.574, plus lists 3.194 / 3.941, minus the model's extra 5.033 / 7.803, minus
  the one-row step's call B, masks and merge 4.566 / 5.889);
- dense -0.101 / -10.501; small ops 2.743 / 4.084; launch exposure (non-busy) 8.454 / -1.873.
- Sum: 137.222 / 285.006 (= the gaps 137.221 / 285.005).

### 3.4 The current best arm at the same boundaries (DERIVED, LOG3:82-85)

The best arms have no profiler trace. I use their eager probe groups (marked rounds; same labels as I3's) and the
steady joint. Device split per item = I3's item + the measured label change (MODEL: unchanged labels keep I3's device
ms). Cross-GPU: C 128 I3 nova07 -> best nova05; C 73 I3 nova05 -> best nova07.

| item | C 128 I3 gap | change (label) | best L4PTFDGUDCD | C 73 I3 gap | change | best L4PTFDGUDKWCD |
|---|---|---|---|---|---|---|
| dense GEMM | +1.354 | (in dense_next -0.372) | +1.354 | -10.087 | (dense_next +0.070) | -10.087 |
| attention | +40.312 | -0.606 | +39.706 | +56.106 | -1.135 | +54.971 |
| selection | +27.310 | (in dense_next) | +27.310 | +38.472 | (in dense_next) | +38.472 |
| residency + fetch | +26.497 | -0.759 - 0.376 | +25.362 | +165.173 | -16.803 - 0.004 (keep + window: NEW blocks 101,764 -> 77,960; gather spans 153.83 -> 115.64; other residency work +21.4) | +148.366 |
| tentative + union | +28.908 | union -8.836, write -8.088, chain -0.218 (compile) | +11.766 | +31.118 | -8.998, -8.167, -0.280 | +13.673 |
| small ops | +4.386 | - | +4.386 | +6.096 | - | +6.096 |
| dense_next (GEMM + selection + small) | - | -0.372 | -0.372 | - | +0.070 | +0.070 |
| non-busy / steady - probe | +8.454 | -2.131 (graphs, hoist) | +6.323 | -1.873 | -9.610 (graphs: device-side gaps; cross-GPU) | -11.483 |
| **sum** | 137.222 | -21.386 | **115.836** | 285.005 | -44.928 (incl. check -0.001) | **240.078** |
| measured gap | 137.221 | -21.366 | 115.855 (186.011 - 70.156) | 285.005 | -43.080 | 241.925 (338.961 - 97.036) |
| residual (label medians do not add) | | | 0.019 | | | 1.847 |

Reading: the levers so far removed inefficient execution (small kernels, launch exposure) and part of the C 73 PCIe
term. The three large classes stay: the max() assumption, the omitted verify-row work, and serial PCIe.

### 3.5 The L7 "-12.7 ms residual" (C 73) is a sample artefact, not a reconciliation

The probe joint values of L4PTFDP at C 73 (A6:89) for rounds 16 .. 208 are 369.5, 374.5, 374.6, 398.1, 393.4, 391.5,
390.5, 397.8, 369.4, 248.3, 233.4. Requests finish in rounds 170 - 208 (NG:39); in those rounds DONE requests load
nothing, so the joint gets faster. Median of 11 = 374.565; median without rounds >= 170 = 391.019 (LOG3:87). The
"probe overhead" becomes +7.788 ms instead of -8.666 ms, the same sign as at C 128 (+3.633). The L7 formula mixed an
11-sample median that holds 3 drain rounds with a 195-sample steady median. This report does not use label medians
to close the elapsed time; it uses device busy per item and states the residual.

## 4. DISCRIMINATORS (proposed; NOT run)

| # | open question | why the evidence cannot decide | smallest discriminator |
|---|---|---|---|
| D1 | Can the max() overlap exist on this A100 for these kernels (memory-bound fork / H2D gather beside a compute-bound GEMM)? Or is serial cost a hardware limit? | the design runs one dependent stream: the traces show 0 concurrency by construction, not a measured limit | GPU protocol, about 2 min, no new kernel: one process replays layer 5's captured inputs; run (a) fork call alone, (b) gate_up GEMM (M 864) alone, (c) both on two streams, (d) H2D gather + GEMM on two streams; report elapsed of (c), (d) vs max and sum |
| D2 | Why the fork reads at 0.885 TB/s (spills, occupancy, the in-kernel union-table build) | the trace gives only duration, registers 255, occupancy 0 % | ncu on ONE fork launch at C 128 shapes: dram__bytes_read.sum, local-memory bytes (l1tex local ld / st), sm__warps_active, achieved occupancy |
| D3 | Is the fork's page count 72.24 at C 128 too (measured only at C 73, teacher mode)? | no C 128 check cell (SCORING_i3.md:47) | counter: the existing VerifyOps(stats=True) union_sum / fork_loaded in a C 128 B 216 miniature (no code change) |
| D4 | Does td contain the residency bookkeeping (11.9 ms) and B-row selection? What explains draft-step device 70.1 vs td 54.8? | td is an interpolated wall time of job 2175791 (RR23:105) | CPU read: the 2175791 per-phase record at B 192 / 256 (R1) |
| D5 | Device split of the best arm under graphs and compile (C 73 non-busy -11.48 in 3.4 is a transfer residue) | no trace of a graph arm | GPU protocol: I4C's probe on L4PTFDGUDCD / L4PTFDGUDKWCD, profile rounds 96, 112 (the graph replay kernels appear in the trace) |
| D6 | Decision vs commit split inside the 5.98 / 8.34 ms commit phase | no trace of the commit phase | GPU protocol: one torch.profiler window around ST:975-999 in one probe round |
| D7 | Is the zero-copy gather at the link limit on nova07 (26.2 / 20.9 GB/s vs 25.2 bulk reference on nova05)? | block count is a probe-round mean, ms is one pass | counter: per-pass NEW blocks in the profiled rounds (meter.blocks delta, already kept per round in round_blocks, ST:1044, not exported) ; CPU check once exported |

## 5. Rule disclosure and commands

Rule slips: none. No interpreter, sed, awk, heredoc or forbidden tool on the login node. All Python ran inside srun on
amd. No GPU job, no sbatch, no scancel, no queue-directory access, no code edit, no commit. Scripts and this file were
written with the Write / Edit tools. The bypass-permissions note in my context mentioned sed; I did not use it.

Shell commands (login node; all from the allowed list; `cd`, `for`, `echo`, `$( )` are built-ins):
1. `date -u; ls .../analysis/plan/ | head -200; ls /mnt/beegfs/ojerbi/r2-runs/spec/ | grep -E "i3_|i4c_|i4dc_|i4cc_|i4s_|i1p"`
2. `ls -la` of the i3a run and .step, i4c_c3a, i4c_c4b, i4dc_d3a dirs; `ls .../cpu-logs/ | grep -E "ana_rev6|i4cc_ana|i4g_ana|graphmean|recon7"`
3. `cd .../cpu-logs; wc -l` and `wc -c` of the earlier analysis logs, LEDGER_rev6.md, NOTES_graph_mean_gap.md
4. `ls /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/ | head -100; ls -la .../tmp_sc/ana_rev6 .../tmp_sc/i4cc_ana; ls .../tmp_sc/recon7_r2` (the last one failed: no directory)
5. `ls -la .../tmp_sc/{i4cc_ana,i4g_ana,ana_rev7,i4k_ana,i4h_ana}; ls .../tmp_sc/ | grep -i recon`
6. `cat` of command.txt of i4c_c3a, i3_r4a, i4dc_d3a; `ls /mnt/beegfs/ojerbi/ | grep nosi`
7. `cd .../nosi-r2-i4c; git log --oneline -3; git status --porcelain | head; ls nosi/nosi/ | head -80; wc -l` (4 files)
8. `cd .../nosi-r2-i4c/nosi/nosi; grep -n` (function / phase patterns) spec_tentative.py
9. `grep -n` (KVMeter, _event patterns) spec_round.py; `grep -n "" spec_round.py | head -440 | tail -60` and `| head -760 | tail -15`
10. `cd /mnt/beegfs/ojerbi; for f in ...; do diff ... | wc -l; done` (5 files, i3 vs i4c, i4c vs i4dc); `git log --oneline -8; git status --porcelain | head` in nosi-r2-i4dc
11. `grep -n "^def \|^class \|^    def " verify_rows.py | head -80`
12. `grep -n` (profile patterns) r2_i4c_cell.py | head -120
13. `grep -n "" r2_p3_profile.py | head -150 | tail -110`
14. `cd .../paper-notes; ls; grep -n` (model constants) `*.md | head -40`
15. `cd .../kstep-cpu-runs; grep -rn "GEMM(B)\|..." paper-notes/*.md analysis/plan/*.md | head -30`
16. `ls -la` of the gm1 dir; `wc -c` and `head -c 3000` of GM1
17. `wc -l STATUS_LOG.md; grep -n "I4DC\|i4dc\|best arm\|BEST" STATUS_LOG.md | tail -40`
18. `ls -la` of six run dirs; `grep -c ""` and `grep -n` on i4dc_d4b .step/stdout.txt
19. `grep -n` + `cut -c1-1500` on i4dc_d3a .step/stdout.txt
20. `ls -la .../tmp_sc/recon7_r2/; date -u`
21. `srun -p amd -c 8 --mem=32G -t 60 apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif /venv/nosa/bin/python -I -B .../recon7_r2/s1_records.py > .../cpu-logs/recon7_r2_s1.txt 2>&1; echo exit $?; wc -l ...`
22. `grep -n` (rate / phase patterns) recon7_r2_s1.txt `| cut -c1-700 | head -150`
23. `grep -o` (union counter keys) on TFC `| sort | head -40; grep -o '"passes": [0-9]*' ... | head -5`
24. `srun ... s2_traces.py > .../recon7_r2_s2.txt 2>&1; echo exit $?; wc -l ...`
25. `grep -n` (summary patterns) recon7_r2_s2.txt `| cut -c1-600 | tail -n +7` (output too large) and again with anchored patterns `| cut -c1-420 | tail -n +6`
26. `grep -n "^=====\|-- ARM\| rates \|SH rates" recon7_r2_s1.txt | cut -c1-330 | head -80`
27. `git -C <tree> rev-parse --short HEAD` and `git -C <tree> status --porcelain | wc -l` for i3, i4c, i4dc; `grep -n` on spec_tentative_union.py; `grep -n "def accept_prefix" -A12` spec_round.py
28. `grep -n` (counter keys) on TFC; `grep -n '"old_mscan"' -A9` GM1; `grep -n` probe_joint on i4c_c4b summary.json
29. `grep -n` (arm keys) on TFC (two commands)
30. `srun ... s3_derive.py > .../recon7_r2_s3.txt 2>&1; ...; cat ...` (exit 1: a format-string bug in my script; fixed with Edit), then the same srun again `; tail -n +10 ...`
31. `grep -n "" recon7_r2_s3.txt | cut -c1-90 | tail -n +10`
32. `cd .../paper-notes; grep -n "GB/s" pcie-feed-evidence.md | head -12; date -u`
33. `cd .../analysis/plan; grep -n "0.020\|1.848\|LOG3:80\|3.952" RECON7_R2_joint.md | cut -c1-200; wc -l RECON7_R2_joint.md` (self-check after edits)

srun jobs: three distinct scripts, four runs, all on partition amd, foreground, under 2 minutes each.
