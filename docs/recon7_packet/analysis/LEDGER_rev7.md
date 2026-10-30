# LEDGER revision 7 (2026-10-30): every component against the ORIGINAL October 21 calculation

Written 2026-10-30T10:24:56Z; revised 10:34:41Z after review 1 (PL/RECON7_REVIEW.md); revised 18:51:04Z (from `date -u`)
after review 2 with model inputs from the model's author (section 12). Writer: the RECON7 synthesis agent, read-only on
code and records. Style: ASD-STE100. Revision 6 stays in PL/LEDGER_rev6.md. This file answers the human's request
PL/HUMAN_RECON_REQUEST_2026-10-30.md (REQ). It does NOT mark the component comparison complete: section 1 gives a status
per row, and every anchor value of the "actual" side is MODEL (no run at C81 / B287 exists).

Labels. MEASURED = a value in a run record or trace. DERIVED = arithmetic on MEASURED values (operands shown, or a logged
CPU script). MODEL = needs an assumption (stated). AUTHOR = a model input supplied by the model's author on 2026-10-30
(review 2). CITED = copied from another file (file:line). CODE = read in source at file:line. INFERENCE = a reading that
no record states. HYPOTHESIS = a cause that a named discriminator must test.

Abbreviations (all absolute):
- PL = /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan. R = /mnt/beegfs/ojerbi/r2-runs/spec.
- REQ = PL/HUMAN_RECON_REQUEST_2026-10-30.md. R1 = PL/RECON7_R1_model.md. R2 = PL/RECON7_R2_joint.md.
  R3 = PL/RECON7_R3_draft_transfer_capacity_schedule.md. R4 = PL/RECON7_R4_prior_art.md. RV = PL/RECON7_REVIEW.md
  (review 1). RV2 = review 2 (coordinator message of 2026-10-30 with the author's inputs; no file). SI = PL/SCORING_i4.md.
  CP = PL/CHECKPOINTS.md. L6 = PL/LEDGER_rev6.md. NG = PL/NOTES_graph_mean_gap.md. SL = PL/STATUS_LOG.md.
  H29 = PL/HANDOFF_2026-10-29.md. FC = /mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/forecast-oct12-17pct.md.
  J91M = /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output2_memory.csv.
- Review-1 logs: V1, V1B, V2, V3, V4, V4B = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_v_v1.txt, _v1b, _v2, _v3,
  _v4, _v4b (.txt).
- This writer's logs (scripts in /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_s/): S1 = cpu-logs/recon7_s1.txt
  (s1_revised_model.py), S2 = recon7_s2.txt (s2_round_split.py), S3 = recon7_s3.txt (s3_review_response.py), S4 =
  recon7_s4.txt (s4_review2.py, the author's inputs). All ran with srun on amd (nosa-env image, `python -I -B`). Log line
  numbers count every line (the first nine lines are the image's shell-profile messages). WHERE S4 AND S1 / S3 DIFFER,
  S4 IS CURRENT.
- Configs. ANCHOR = C81 / K3 / B287. C128 = C128 pool 65 / B216. C73 = C73 pool 10 / B318.
- Arms. I3 = L4PTF (R/i3_r4a_2195576_i3a nova07, R/i3_r3b_2195575_i3b nova05). BEST = L4PTFDGUDCD at C128
  (R/i4dc_d3a_2195575_i4dca, nova05) and L4PTFDGUDKWCD at C73 (R/i4dc_d4b_2195576_i4dcb, nova07). Profiled reference
  L4PTFDP (R/i4c_e4a_2195576_i4ca, R/i4c_c4b_2195576_i4cb, nova07); the five schedule files are byte-identical in the i3,
  i4c and i4dc trees (R2:16-19; RV:21-23).
- Strongest = the same-GPU strongest cached NOSA, steady, whole-step graph, C128 B216: nova05 3,426.2, nova07 3,410.1
  tok/s (CP:9-11); strongest EFFECTIVE (whole run, E1) nova07 2,778.3, nova05 unknown. Old reference 3,160.822684 tok/s
  (REQ:17). I3 same-job SH 3,334.0 (C128, RV:229), 2,575.2 (C73, rounded ratio, S3:76).

Fixed conventions:
- MODEL INPUTS (AUTHOR, RV2): U = 69.16294642857143, I_ready = 60.193917410714285 at all three configs (U - I =
  8.969029 groups); unit 32 layers x 2 KV heads x 32,768 B = 2.097152 MB; U and I are AVERAGE group counts per request
  after summing over layers and heads, not global counts; U = historical verifier selection union, I = aligned
  ready-selected intersection. New draft traffic per request per K: C128 2.910500571 MB, C73 9.097801143 MB, C81
  6.465828571 MB. Retained-overflow peak per request: C128 0, C73 5.332992 MB, C81 1.6384 MB. These are four-request,
  teacher-forced, no-rejection historical inputs, NOT live observations. The residual verifier historical traffic is 0 by
  the replay's immediate-arrival, same-selection retention policy; this ledger does NOT use that zero as a live fact.
- td (AUTHOR): piecewise interpolation in B of the saved C63 resident eager median (64: 27.376640, 128: 37.512192, 256:
  62.678017, 344: 80.585732 ms), assigned across C as an approximation. It is a target-traffic control with no loads; it
  is NOT draft throughput and NOT an isolated HBM duration.
- MEMORY (AUTHOR): base = interp(C63 max-device-used backbone: 64: 29.814030, 128: 41.247703, 256: 65.727758, 344:
  83.870220 GB) + (C - 63) x B x 2,105,344 / 1e9 + 0.12740196; plus B x [overflow_MB x (1 + 128 / 32768) + state_MB +
  optional copied union MB] / 1000 + 3 x 17,039,360 / 1e9; state_MB = [2 (K + 1 + lag) 32,768 + K x 8,192] / 1e6 =
  0.417792. The backbone retains restore snapshots, allocator slack and overhead. Budget 84.0 GB; the old observed
  device total 85.094826 GB. The sampled backbone does not prove transient fit. This ledger evaluates the aliased-retention
  case (no copied union).
- JOINT MODEL ALLOCATION: the per-pass budget of REQ:40, max(td + extra, GEMM(B) + GEMM(3B))(1 + s) = 70.156 / 97.036 ms.
  It is a slice of an AGGREGATE cohort-max model; its overlap is a model assumption, not a dependency-feasible schedule
  (section 2.6). Sensitivity: R3's cohort slice 66.18 / 89.63; R1's GEMM(4B) budget 66.18 / 93.02.
- INTERFERENCE IS CHARGED ONCE. In the gap decomposition against the REQ:40 budget, the budget's own s x joint dense
  (6.710 / 7.070) is netted against serialization. In the revised-model variants, s multiplies the whole round once:
  interval = (R - removed terms) x (1 + s) (the model's own GPU x (1 + s) rule). No other interference term is added.
- ONE-STEP BODY ABOVE td: each value names its arm and boundary (table 1a). The draft-step excess stays in the draft
  span; it is NOT inside the joint-region sum.
- WINDOWS: "steady" = the cell's steady set (round 0, probe and capture rounds out; drain rounds IN: 24 of 181 at C128,
  39 of 195 at C73). "drain-free" = rounds before the first request finishes (round 170). "whole run" = committed / wall_s
  (one-time capture and prelude in). "warmed" = steady with reused graphs. Each comparison names its window.
- YIELD NUMERATORS: ACCEPTED drafts (the model's Y; 2.889 / 2.780 per request in the median steady round); CORRECTIONS
  (1,058 / 2,250 per run); COMMITTED = accepted + corrections (y_med 2.8926 / 2.8091 I3, 2.894 / 2.803 BEST, mixed
  medians, used only to convert our median tok/s). They stay apart.

## 0. Answer in twelve lines

1. With the author's inputs the original block still reproduces at the anchor (Q 3,703.54, +17.17 %; S4:13-16). At the
   measured configs the model changes: C128 Q 3,457.9 (GPU binds; transfer 62.06 ms); C73 Q 3,621.8 (TRANSFER binds:
   263.41 > GPU 252.05 ms) instead of 3,784.9 (S4:17-24).
2. WORK matches where the model counts work: 6 rows per request per round, 3 weight passes, the 2-pass verdict lag, no
   bonus row; dense charge met (64.8 vs 63.4 ms; 79.9 vs 90.0 ms); LOGICAL union pages x0.99 of the model at C73 only.
   Physical HBM reads are NOT measured (section 2.7).
3. TIME does not match. The joint pass runs one dependent stream with no overlap BETWEEN kernels (R2:41-42). This does not
   show the absence of overlap INSIDE kernels, and it does not show recoverable overlap: the 41.5 / 57.8 ms "overlap term"
   is the gap between the model's aggregate max and its own serial form. Overlap feasibility is OPEN; the aggregate max may
   be optimistic (section 2.6).
4. A dependency-feasible resource view (MODEL floors): GEMM + attention alone fit the budget even in series (65.2 / 95.6
   vs 70.2 / 97.0 ms); the measured excess is efficiency (GEMM 67-80 % of peak; fork at 0.58-0.60 of the flash logical
   rate, unmatched), unmodelled stages (60.6 / 75.7 ms) and PCIe (21.3 / 160.1 ms) (S4:44-50).
5. Per round (steady): I3 is 178.0 / 322.1 ms above the model interval, BEST 137.5 / 261.3 (S4:104-107; C73 now against
   the transfer-bound interval).
6. WHOLE RUN reverses the steady order (BEST 1,292.4 vs I3 1,396.7; 1,242.5 vs 1,275.3 tok/s) because of the first-use
   capture (9.6 / 10.1 s). With the capture amortized over k identical runs, BEST passes I3 at k = 3.0 (C128) and 1.5
   (C73) (S4:53-62; cross-GPU pairs).
7. RETRACTED: "about two model rows per kept token makes 1.0x impossible" and the "GPU busy / launch sweep" arguments
   (section 4).
8. Memory with the author's formula (device-used basis): 87.91 GB at C128 B216 and 87.22 GB at C73 B318 (largest B
   204.1 / 304.2); anchor 83.77 GB (287.95). We RAN B216 and B318 under the 84 GB allocated cap; bases differ, so any
   margin stays MODEL (row 7).
9. Schedule: positions and the 2-pass lag are implemented; the lag's purpose (background transfer, retention) is not.
   Our dispatch is 2 GEMM(B) + GEMM(4B) per round vs the model's 3 GEMM(B) + GEMM(3B) (section 3, 5.4).
10. Recovery: fixed-lead model n = K + p (1 + lag) = 3 + 3p gives Y 2.918 / 2.884; the empirical n = 3 + 5.1p gives
    2.863 / 2.808 vs measured 2.855 / 2.798 (S4:67-68). Recovery enters only through Y; no time is charged twice.
11. Revised model at the anchor (MODEL, S4:70-95, interference once): as implemented 0.47-0.52x the strongest (I3 code)
    and 0.51-0.57x (best code); all three modeled mechanisms realized 0.65-0.68x and 0.74-0.78x. Not limits.
12. Open rows: section 1 status column and section 7 (X1-X19).

## 1. The component table (the human's format)

Columns: (a) original modeled work / cost at the ANCHOR; (b) the same formulas at C128 and C73 (author inputs); (c) actual
work / cost at the SAME boundaries, work counts first, then ms (steady window unless stated); (d) gap; (e) cause with
evidence and class (EXTRA / WRONG WORK, INEFFICIENT EXECUTION, SERIALIZATION / CONTENTION, CHANGED LIVE WORKLOAD, FALSE
MODEL ASSUMPTION, MECHANISM NOT REALIZED where feasibility is untested); (f) correction or quantified model revision; (g)
remaining unknown -> discriminator (section 7); (h) status. Values are C128 / C73 unless stated. Single profiled passes
carry a round-96..112 spread (joint busy 195.7-198.9 / 379.3-383.9; H2D 18.2-21.3 / 156.2-160.1 ms; RV:183-186).

| # | component | (a) original at ANCHOR | (b) model at C128 / C73 | (c) actual, same boundaries: work, then ms | (d) gap | (e) cause, evidence, class | (f) correction / revision | (g) unknown -> discriminator | (h) status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | standalone drafting / dense | td 68.9864 ms (AUTHOR rule: C63 resident eager median, a target-traffic control with no loads, ELAPSED with host launch), GEMM(B) 22.6568; per-pass budget 74.98; 2 draft passes per round 149.96 (S4:13; R1:266-270) | td 54.8137 / 75.2948; GEMM(B) 18.3286 / 25.2865; 2 draft passes 121.22 / 162.42 (S4:17-21) | WORK (one eager L4PTFD pass, R3 2.1-2.2): B rows; 129 GEMM launches (td 129); 128 attention launches = call A + call B (td 64); KV blocks per (layer, head, row) 63.51 / 62.04 + one 64-row buffer block (td 64); 2,674 kernels + 193 copies (td about 1,156-1,188). Kernel ms, ours vs td's J91 ncu classes (R3 audit, cpu-logs/recon7_r3_s3.txt): GEMM 19.7 vs 20.4 / 25.8 vs 25.1; attention 20.5 vs 19.1 / 29.9 vs 27.9; selection 8.6 vs 9.5 / 12.9 vs 12.9; residency 9.4 vs 2.5 / 11.5 vs 3.2; small / fused / copy / twin 11.0 vs 2.0 / 13.8 vs 2.3 (R3:81-88). ELAPSED draft step per arm: table 1a | CHOSEN (same boundary as td): BEST +11.33 / +12.92 ms per step; per round vs 2 max(td, GEMM(B))(1 + s): BEST +11.07 / +14.01, I3 +30.14 / +32.13 (S4:104-107). This excess stays in the DRAFT span | Categories audited against td's own kernel classes BEFORE calling them absent (RV2): pool residency (pool_swap 5.5-7.4 / 5.8-8.4 ms) is absent from td's classes (td residency = offload metadata + empty gathers, 2.5 / 3.2 ms): FALSE MODEL ASSUMPTION for a cached C > 63 draft. Tentative-buffer work +5.91 / +6.89 ms: EXTRA (design-required) work. Graphs + compile removed 5.49 / 6.55 ms: INEFFICIENT EXECUTION, partly fixed. GEMM and selection: no gap (R3:57, :101) | td_rev = elapsed draft-step ratio x td: BEST 1.2067 / 1.1716; ANCHOR 81.56 ms (ratio) to 84.40 ms (DR81 77.799 + 6.6) (S4:71; R3:103-105) | BEST draft kernel split -> X5. Ladder draft at C81 -> X9. td contents -> X4 | measured configs CLOSED in ms; best-arm split OPEN; anchor MODEL |
| 2 | joint dense | GEMM(B) + GEMM(3B) = 22.6568 + 58.3740 = 81.0308 (two call sets; GEMM(861) EXTRAPOLATED x 861 / 768); per cohort 3 GEMM(B) + GEMM(3B) = 126.34 (S4:13-14) | two-call charge 63.4452 / 89.9657; per cohort 100.10 / 140.54; our shape per round 2 GEMM(B) + GEMM(4B) = 95.24 / 136.81 on the same curve (S4:65-66) | WORK: rows 4B = 864 / 1272; 129 GEMM launches per joint pass, ONE call per weight over all 4B rows; token tiles 7 x 128 / 10 x 128 (or 5 x 256); no split-K reduce kernels (R2:127-129; RV row 8). Per round: 2 one-row passes (129 launches each, M = B) + 1 joint pass (M = 4B). MS: joint GEMM 64.80 / 79.88 = 67.4 % / 80.5 % of the bf16 peak; per round measured 2 x 19.785 + 64.800 = 104.37 / 2 x 25.705 + 79.879 = 131.29 (S4:65-66) | joint vs two-call charge +1.35 / -10.09; joint vs one-call curve +6.22 (+10.6 %) / -6.36; per round vs our shape on the curve +9.13 / -5.52; per round vs the model's per-cohort charge +4.27 / -9.25 (S4:65-66, hand differences) | NONE established: the dense FORMULA is met. The reference above M 768 is an EXTRAPOLATION found by an exact match (R1:75-78), so the sign of the one-call difference is UNKNOWN | Keep the dense term; mark "extrapolated reference, sign unknown". ANCHOR: measured device curve at M 1148 75.30 - draft GEMM 23.91 = 51.39 vs GEMM(3B) 58.37: -6.98 (S1:44, MODEL) | GEMM scan at M 864 / 1,148 / 1,272 -> X17 | formula CLOSED; reference sign OPEN |
| 3 | joint selection | 0 for verify rows (B draft rows inside td) | 0 / 0 | WORK: batched selection of 4B rows per layer: stage 1 32 calls (grid (1, 864, 2)), pooling 32, top-k 64 (R2:134); verify-row selected slots per (layer, head, request) 189.52 = 3 x 63.18 / 188.73 = 3 x 62.91 (R2:135). MS: selection over 4B rows 27.31 / 38.47, x2.88 for x4 rows (R2:228, :241) | +17.83 / +24.43 in the serial chain (R2:263) | FALSE MODEL ASSUMPTION (omitted necessary work): td's selection classes cover B rows only (R3:85: td 9.5 / 12.9 vs ours 8.6 / 12.9 per one-row pass); each verify row needs its own selection. The B-row part (9.48 / 14.04) is hidden by the aggregate max in the model; whether it can overlap is OPEN | Add 0.02752 / 0.02561 ms per verify row per pass; ANCHOR 22.55 ms (S4:71, MODEL) | fused multi-row selection headroom: no discriminator proposed; X5 | CLOSED in ms; headroom OPEN |
| 4 | joint attention and shared reads | extra = td (1 - f)(U - I) / 63 = 6.9446 ms; U - I = 8.969 groups = 18.81 MB per request per cohort (AUTHOR unit 2.097152 MB; S4:10); the draft's 63-group KV sits inside td; KV time HIDDEN by the aggregate max | extra 5.0333 / 7.8025 (U - I now AUTHOR-confirmed at both configs) | LOGICAL work: 32 fork calls per pass (grid (432, 2, 1) / (636, 2, 1)); per program a union page list per (request, KV head); 4 rows x 16 query heads share one 64-row tile (R4:71-76). Logical pages per (layer, KV head, request), C73 check arm ONLY: fork_loaded 72.24 = union 71.18 + 1.06 tail / buffer; vs the model like-for-like 0.989 / 0.990 (V1B:40-44). C128 not measured. PHYSICAL HBM bytes: NOT measured (no hardware counter). MS: fork 37.1 / 52.2. LOGICAL-byte rate 0.885 (C128, MODEL page count) / 0.927 TB/s; the one-row flash call A 1.527 / 1.543 TB/s: an UNMATCHED comparison (section 2.7) | +11.65 / +14.65 above the model in the serial chain (R2:262) | Logical read-once tile: REALIZED (by code and counter). Physical one-fetch-per-page: NOT SHOWN (L2 hits or tile re-reads unknown). Fork efficiency: HYPOTHESIS pending X2. Overlap of KV time under dense: OPEN (section 2.6). Prior art: logical U floor holds in the code bases read (R4:257-262) | Keep U - I. The 0.58 / 0.60 logical-rate ratio is an observation, not a cause. ANCHOR attention above the model 13.91 ms, of it the fork term 19.38 (S4:71, MODEL) | C128 counts -> X3; dram bytes, occupancy -> X2; grid axis meaning -> X18 (CPU code read) | OPEN |
| 5 | joint tentative state / union lists / chain | none | 0 / 0 | WORK per layer: union / locate, chain images CB0 / CB1, buffer writes of R rows, R twin steps + deferred replay (R2:80-86); SMALL class 7,437 kernels per joint pass. MS (I3): 28.9 / 31.1 (R2:230, :243); serial increment 25.9 / 27.2; BEST (MODEL labels) 11.8 / 13.7 (R2:297) | +25.9 / +27.2 (I3); BEST +11.8 / +13.7 | FALSE MODEL ASSUMPTION (omitted design-required work; td's small / fused class is 2.0 / 2.3 ms for a step without tentative state, R3:87) + INEFFICIENT EXECUTION (7,437 small kernels; Inductor removed 17.1 / 17.4 ms) | Add 0.120 / 0.085 ms per request (I3); ANCHOR 27.53 (I3 code), 9.65 (best code) (S1:45, :71) | floor of the tentative state -> X5 | CLOSED in ms; floor OPEN |
| 6 | packing / transfers / readiness | AUTHOR traffic 6.4658 MB per request per K (new draft traffic; verifier residual 0 by replay policy); r 10.724 GB/s (CPU8 packer + DMA beside the decode); transfer 173.04 ms HIDDEN under GPU 232.48; s 8.685 %; readiness assumed | AUTHOR traffic 2.9105 / 9.0978 MB; r 10.131 / 10.983; transfer 62.06 ms (hidden under GPU 187.40) / 263.41 ms (BINDS: above GPU 252.05); s 10.577 % / 7.859 % (S4:17-24) | WORK: no CPU packer, no draft-issued request, no prefetch (in_flight 0); drafts skip misses (0.485 / 1.959 per stream per call) and request nothing; 128 H2D gather kernels (SM-driven zero-copy) per joint pass in the compute stream (R2:138; R3:58). LIVE verify-pass NEW host loads per request per round: 2.581 MB (all arms) / 10.527 MB (I3) / 8.065 MB (BEST, keep + window) = x0.887 / x1.157 / x0.886 of the author's draft traffic (S4:39-42; different scope: live verify loads vs historical draft traffic). Not ready at the joint 0.65 % / 2.65 % (BEST 2.03 %). MS: H2D device in the joint pass 21.3 / 160.1; exposed wait (probe-round median) 20.25 / 154.76, BEST 115.64; overlap 0.000 | I3, joint boundary, net of the budget's allowance: +14.59 / +153.07; BEST +13.54 / +108.57 (S3:33-34) | MECHANISM NOT REALIZED -> SERIALIZATION: every byte is urgent and serial inside J. CHANGED LIVE WORKLOAD at C73 (x4.08 new blocks per (layer, KV head, request) vs C128). EXTRA WORK: no retention (23-24 % avoidable refetch). Coverage evidence is from the legacy schedule only (CITED by R3:58 from SCORING_wse.md:217, :254). Rate: the in-stream gather runs near the PCIe rate; NOT comparable with the model's background r (RV:205-211) | For this implementation: exposed = NEW bytes / (22.0-27.6 GB/s). ANCHOR with the AUTHOR bytes 6.4658 MB: 67.24-84.35 ms (S4:72, :78). The earlier C-interpolated byte guesses (7.27 / 9.37 MB) are WITHDRAWN | coverage -> X8; per-pass rate -> X7; copy-engine interference -> X1 (d) | OPEN |
| 7 | retention / state / physical capacity | AUTHOR rule, device-used basis: 83.7655 GB at B287 (backbone 72.1189 + 0.1274 + slots 10.8762 + per-request 0.5920 + landing 0.0511); largest B under 84.0 287.95; without the restore snapshot (5.596 GB, MODEL) 78.17 GB, largest B 312.7 (S4:28-30) | AUTHOR rule: C128 B216 87.91 GB (largest B 204.1); C73 B318 87.22 GB (largest B 304.2; overflow 5.33 MB per request adds 1.84 GB). Without the snapshot (MODEL): 83.69 / 81.02 GB; largest B 217.0 / 332.0 (S4:31-36). Copied-union case: +0.216 / +0.318 GB per MB per request (not evaluated; no MB value given) | STATE per request: engine slots 271.59 / 155.80 MB; ONE store X 129 shared by 32 layers 8.487 MB; CB0 0.066; tentative buffers 2.105; prefill transient 20.1-20.2; graph pools 71.3 / 117.4 MB (fixed); no retained overflow (R3:59, :140-152). PEAK ALLOCATED with prefill 83.891 (I3) / 83.900 (BEST) and 77.685 / 77.694 GB; decode 82.559 / 76.932; B216 RAN at 83.900 GB allocated after SH in the same process; B318 RAN; B351 OOM (83.644 GB allocated + 126 MiB request, 0.551 GB non-PyTorch) (SI:57-59; R3:159-161) | Different bases: the model counts max-device-used (snapshots, allocator slack, overhead inside); we report allocated peaks. As written the model says B216 / B318 do NOT fit (largest 204.1 / 304.2), yet both RAN under the allocated cap. With snapshot removal charged on the MODEL side only, the model allows 217.0 / 332.0; our side's own harness terms (other arms in the process, SH graphs) are not separated, so the same removal cannot be charged on our side yet | BASIS-DEPENDENT; no capacity margin is established in either direction. Basis gap at B287 on the B256-344 segment: device-used 72.119 - allocated 66.391 = slack 4.609 + outside allocator 1.119 + 0.000 (V3:34-42); the author's constant 0.1274 sits on top. The 5.60 GB snapshot is a category inside allocated (RV:88-91) | Capacity: report the model on its basis and ours on ours. Extra requests from snapshot removal stay MODEL until a physical fit. ANCHOR: our layout B_fit(C81) 321.5-322.8 (allocated, MODEL, S3:83) vs the model 287.95 (as written) / 312.7 (snapshot removed, MODEL) | one-basis category table (CPU) -> X16; C81 fit line -> X10; transient fit on device-used -> X10 with nvidia-smi sampling | OPEN |
| 8 | schedule / yield / recovery | K3; Y = 3 accepted; p 0; lag 2; corrected fixed-lead n = K + p (1 + lag) = 3; no recovery / drain / startup; corrections and bonus omitted; Q 3,703.54 | interval 187.40 (GPU) / 263.41 (transfer); Q 3,457.9 / 3,621.8 accepted-draft tok/s (S4:18, :22) | COUNTS (R3:60; RV rows 26-30): groups + finals 37,614 / 55,866; verified 111,387 / 164,743; ACCEPTED 109,318 / 160,248; CORRECTIONS 1,058 / 2,250 (p 2.813 % / 4.027 %); bonus 0; COMMITTED 110,376 / 162,498; fills 1,050 / 2,221; finals 216 / 318 (2.125 tokens per final group at C128); unverified discarded 405 / 634; dropped drafts per reject 4 - n = 3.129 / 3.188 (S4:67-68); padded GEMM rows 10.5 % / 16.2 %. YIELD: accepted 2.889 / 2.780 (median steady round); 2.609 / 2.411 (whole run); recovery 1.70 / 1.70 elapsed rounds per reject; drain 7.7 % / 12.6 % of request-rounds | Y accepted -3.7 % / -7.3 % (median round); -13.0 % / -19.6 % (whole run). Fixed-lead model Y 3 / (1 + p) = 2.918 / 2.884 vs empirical 2.863 / 2.808 vs measured active-request 2.855 / 2.798 (S4:67-68) | CHANGED LIVE WORKLOAD (p 2.8 % / 4.0 %; the author's inputs are no-rejection). FALSE MODEL ASSUMPTION: no recovery / drain / startup. Draft ATTEMPTS lost per reject (3.13 / 3.19) are close to the fixed-lead 1 + lag = 3; the larger elapsed loss (1.70 rounds) comes from the FILL round and the v = 2 group of decision (a), which cost whole rounds of the static batch. Prediction 1.71 / 1.73 vs measured 1.70 / 1.70: consistent within about 5 % (RV:219-224) | n_emp = 3 + 5.1 p is EMPIRICAL accounting (rounds lost x 3); the corrected model is n = 3 + 3p. Recovery enters the revised model ONLY through Y (tokens lost); the round time is the batch's steady median, which already holds the fill and v = 2 rows as padded rows, so no time is charged twice. ANCHOR Y_rev 2.808-2.864 accepted (+0.031 corrections for committed) | p at C81 -> X9; end-effect vs reject rounds -> X15 | CLOSED at measured configs (within 5 %); anchor p OPEN |
| 9 | host / launch, decision / commit | td includes its own host launch; decision, commit, reset 0; between-round host 0 | 0 / 0 | WORK per joint pass: 8,014 cudaLaunchKernel + 416 cuLaunchKernel + 97 ExC + 452 MemcpyAsync + 32 MemsetAsync; 0 host syncs (R2:141, :143). MS: joint non-busy 8.5 / -1.9 (cross-GPU sign); busy fraction 0.949-0.951 / 0.996; commit + decision 6.0 / 8.4, reset 0.11 (I3); outside round_ms per round (whole run): +41.6 / +24.0 (I3), +54.1 / +41.5 (BEST, capture removed) (S3:12-22) | +8.5 / -1.9 (joint); +6.1 / +8.5 (commit + reset) | INEFFICIENT EXECUTION (eager launch), mostly removed by graphs (6.96-7.36 / 7.27-7.76 ms, SI:271). The commit / decision phase is real work absent from the model. The out-of-median time mixes host gaps, eager probe rounds, round 0 and faster drain rounds | c = 6.1-8.6 ms per round; ANCHOR 7.75 (I3) / 7.83 (best) (S4:71) | decision vs commit -> X6; best-arm non-busy -> X5; per-round export -> X15 | OPEN |
| 10 | overlap (the aggregate max) | ONE max over the cohort: max(HBM 213.90, dense 126.34)(1 + s) = 232.48, then max with transfer 173.04. Per joint pass dense binds (81.03 > 75.93). Per-pass reading 238.02 (S4:14-15) | GPU 187.40 / 252.05; per-pass 191.38 / 259.46; C73 interval set by TRANSFER 263.41 (S4:18-23). Overlap term (serial form - max form, before s): 41.52 / 57.81; ANCHOR 53.27 | DEPENDENCIES: 1 stream (pid 0, tid 7) for 9,011 joint and 2,867 draft ops; 0.000 ms concurrency BETWEEN kernels; 0.000 ms H2D-compute overlap; 0 host syncs (R2:142; RV:38-39). Inside each layer: select -> resid -> write -> chain -> union -> fetch -> attn -> post, each waiting for the one before (R2:211-215). Overlap INSIDE kernels (HBM traffic under tensor-core work in the GEMM; KV loads under softmax / MMA in the fork) EXISTS and is not measured | the overlap term is NOT a measured recoverable amount (section 2.6) | MECHANISM NOT REALIZED between kernels; FEASIBILITY OPEN. The aggregate max may be OPTIMISTIC: QKV -> selection -> attention -> MLP are causally dependent per layer and per request; a dependency-feasible overlap needs independent work (another request group, or in-kernel fusion) (section 2.6) | For the implemented schedule use the serial form per pass; keep the max as the model's target, labelled "aggregate, may be optimistic" | two-stream overlap test -> X1; two-half cross-request split -> X19 | OPEN |
| 11 | timing window | steady state only | steady only | STEADY (drain rounds in): I3 1,709.8 / 1,525.8; BEST 1,924.1 / 1,699.0 tok/s. WHOLE RUN: I3 1,396.7 (nova07) / 1,275.3 (nova05); BEST 1,292.4 (nova05) / 1,242.5 (nova07); BEST capture 9.594 / 10.140 s (first use), prelude 2.457 / 2.562 s (incl. compile) vs I3 0.218 / 0.285 s (V4:19-25) | whole / steady: I3 0.817 / 0.836; BEST 0.672 / 0.731 | The order of I3 and BEST reverses between windows because of first-use capture. WARMED (reused graphs) = the steady figures. AMORTIZED (capture once per k identical 512-step runs, same shapes, compile cache reused): BEST C128 1,289.9 (k 1), 1,366.5 (k 2), 1,417.0 (k 5), 1,434.7 (k 10); C73 1,240.1 / 1,290.0 / 1,322.0 / 1,333.0; break-even vs I3 at k = 2.99 / 1.50 (capture only) or 2.13 / 1.36 (capture + prelude) (S4:53-62). Cross-GPU pairs | Report warmed (steady), whole-run (first use) and amortized (state k) separately in every verdict | per-round export -> X15; strongest effective on nova05 (none) | OPEN |

Notes to the table.
- R1 and R2 give joint gaps 137.213 / 285.030 (the human's rounded-ratio conversion) and 137.221 / 285.005 ms (the
  record's steady median joint). This ledger uses the record.
- The model's C73 column changed with the author's traffic: transfer 263.41 ms now binds (was 187.20 ms with the carried
  C81 bytes); Q 3,621.8 (was 3,784.9). The C128 Q is unchanged (GPU binds).

### 1a. "One-step body above td" by arm and boundary (S3:89)

| arm | boundary | C128 ms | above td 54.81 | C73 ms | above td 75.29 | used where |
|---|---|---|---|---|---|---|
| I3 L4PTF (full draft) | elapsed, draft phase / 2, CUDA events, host incl. | 75.68 | +20.87 | 97.28 | +21.98 | I3 draft span (section 2.2) |
| L4PTFD (lean draft) | elapsed | 71.63 | +16.82 | 94.77 | +19.48 | - |
| L4PTFDP draft0 | DEVICE BUSY of one profiled pass (round 112) | 70.11 | +15.29 | 93.50 | +18.20 | REFERENCE PASS inside the joint-region partition (section 2.3); any reference gives the same joint total; not a draft-step excess |
| BEST | elapsed (graphs + compile) | 66.14 | +11.33 | 88.22 | +12.92 | row 1; draft span only |

## 2. Per-round elapsed reconciliation (spans, overlap, measured residuals)

Rules. (1) Label-median sums are NOT an elapsed reconciliation. (2) Per round the four CUDA-event phases are DISJOINT and
add exactly: reset = ev0->ev1, joint = ev1->ev2, commit = ev2->ev3, draft = ev3->ev4 (ST:946-1037, :1187-1188; R2:66-70);
the host span ev4 -> next ev0 is outside round_ms. The records keep only phase MEDIANS, not per-round lists (NG:11-15);
X15 exports the per-round values. (3) Every steady median holds drain rounds; section 2.5 bounds the effect.

### 2.1 Spans at both batches (MEASURED, steady window; residual = round median - sum of phase medians)

| arm, run, GPU | reset | joint | commit + decision | draft (2 passes) | phase sum | round median | residual | outside round_ms per round (whole run) |
|---|---|---|---|---|---|---|---|---|
| I3, i3 r4a, nova07, C128 | 0.110 | 207.377 | 6.004 | 151.363 | 364.854 | 365.423 | +0.569 | +41.6 |
| L4PTFD profile job i4c e4a, nova07, C128 | 0.115 | 207.457 (busy 195.7-198.9; H2D 18.2-21.3) | 5.977 | 143.987 (busy 137.7-138.5) | 357.536 | 357.981 | +0.445 | - |
| I3, i3 r3b, nova05, C73 | 0.115 | 382.041 | 8.351 | 194.550 | 585.057 | 585.463 | +0.406 | +24.0 |
| L4PTFD profile job i4c c4b, nova07, C73 | 0.117 | 383.231 (busy 379.3-383.9; H2D 156.2-160.1) | 8.344 | 189.278 (busy 184.9-185.0) | 580.970 | 581.517 | +0.547 | - |
| BEST, i4dc d3a, nova05, C128 | 6.178 (reset + commit) | 186.011 | (in reset) | 132.287 | 324.476 | 324.883 | +0.407 | +54.1 (capture out) |
| BEST, i4dc d4b, nova07, C73 | 8.546 (reset + commit) | 338.961 | (in reset) | 176.437 | 523.944 | 524.659 | +0.715 | +41.5 (capture out) |

Sources: R2:184-209; R3:200-204; CP:93; S3:12-22. Overlap between kernels in every span: 0.000 ms.

### 2.2 The round against the model interval (DERIVED, S4:104-107; steady window)

Model interval (a) = max(GPU cohort, transfer): 187.40 (C128, GPU) and 263.41 (C73, TRANSFER). The split: draft - 2 max(td,
GEMM(B))(1 + s) [draft span; the draft-step excess lives ONLY here]; joint - joint budget [joint region]; commit + reset;
cross-pass lending (per-pass - GPU); transfer binding (GPU - interval, negative at C73); round - phase sum.

| arm, config | round - (a) | draft | joint | commit + reset | lending | transfer binding | residual | per 3 ACCEPTED per request vs (a) |
|---|---|---|---|---|---|---|---|---|
| I3 C128 | 178.024 | +30.141 | +137.221 | +6.114 | +3.979 | 0 | +0.569 | 379.48 ms = 2.025x |
| I3 C73 | 322.055 | +32.126 | +285.005 | +8.466 | +7.408 | -11.356 | +0.406 | 631.82 ms = 2.399x 263.41 |
| BEST C128 | 137.484 | +11.065 | +115.855 | +6.178 | +3.979 | 0 | +0.407 | 337.38 ms = 1.800x |
| BEST C73 | 261.251 | +14.013 | +241.925 | +8.546 | +7.408 | -11.356 | +0.715 | 566.20 ms = 2.150x 263.41 |

(Per-3 values S3:27-30; the C73 ratios use the new interval by hand: 631.82 / 263.41, 566.20 / 263.41.)

### 2.3 Inside the joint region, I3 (DERIVED, R2 3.3; S3:33-34)

An exact partition of ONE profiled pass plus a cross-job steady correction (non-busy). Its residual is zero BY
CONSTRUCTION, not a measured closure. The first line is the joint's own one-step-body proxy, not the draft-step excess
(which is in the draft span of 2.2). Interference: the budget's s x joint dense is netted once, against serialization.

| term | C128 measured | model | excess | C73 measured | model | excess | class |
|---|---|---|---|---|---|---|---|
| reference pass (L4PTFDP draft0 busy) vs td | 70.107 | 54.814 | +15.294 | 93.498 | 75.295 | +18.203 | joint one-step-body proxy |
| dense increment | 45.014 | 45.115 | -0.101 | 54.174 | 64.675 | -10.501 | none |
| attention increment | 16.683 | 5.033 | +11.649 | 22.453 | 7.803 | +14.650 | fork rate HYPOTHESIS (X2) + other terms |
| selection increment (3B rows) | 17.830 | 0 | +17.830 | 24.434 | 0 | +24.434 | omitted necessary work |
| H2D increment - allowance | 19.856 - 6.710 | 0 | +13.146 | 158.012 - 7.070 | 0 | +150.942 | serialization |
| residency bookkeeping | 0.775 | 0 | +0.775 | 0.089 | 0 | +0.089 | small |
| tentative + union increment | 25.915 | 0 | +25.915 | 27.171 | 0 | +27.171 | omitted work + small kernels |
| small ops | 2.743 | 0 | +2.743 | 4.084 | 0 | +4.084 | rows x4 |
| non-busy | 8.454 | 0 | +8.454 | -1.873 | 0 | -1.873 | launch (C73 sign cross-GPU) |
| aggregate-max overlap term (td + extra - GEMM(B)) | | | +41.518 | | | +57.811 | model assumption; NOT a recoverable amount (2.6) |
| total = joint - budget | 207.377 | 70.156 | +137.221 | 382.041 | 97.036 | +285.005 | |

Hand sums: 137.223 / 285.010 (rounding 0.002 / 0.005 ms).

### 2.4 The best arm at the same boundaries (MODEL labels; R2 3.4)

No profiler trace. Built gap 115.836 / 240.078 vs measured 115.855 / 241.925: residual 0.019 / 1.847 ms (stated).

### 2.5 Drain inside the steady window; the residual-correction decision; whole run

ORIGINAL RESULT (kept as registered, SI:164): the I4C C73 probe set = 11 rounds {16, 32, 48, 64, 80, 128, 144, 160, 176,
192, 208} (rounds 96 and 112 were profiler windows); probe joint median 374.565 ms; I4C-L7-RESIDUAL -12.675 ms, REFUTED
against (-2, 2). Registered verdicts do not change.

REANALYSIS OF SAVED RECORDS (no new run, no corrected probe code): exclusion criterion = "at least one request had
finished before the round" (first finish round 170). Excluded: rounds 176, 192, 208 (49, 288, 315 of 318 requests
finished) (V4B:21-33). Drain-free median 391.019 ms (probe_joint list) / 390.495 ms (total list). At C128 the same
criterion excludes rounds 176 and 192 (79 and 210 of 216 finished): 211.132 vs 210.182 ms (V4B:46-57). No result under
corrected probe code exists; the probe code is unchanged.

Bound on the steady window (MODEL): its drain share (13.3 % / 20.0 %) is below the probe set's (20.0 % / 27.3 %), so a
drain-free steady joint median is higher than the reported one by at most about the probe shift, +0.95 / +16.45 ms; the
C73 joint gap 285.0 ms could be up to about 301 ms drain-free. R2's "+7.788 ms probe overhead" mixes windows and is not used.

Whole run (S3:11-24; S4:53-62): see row 11. wall_s = warm step + prelude + round 0 + capture + rounds + host gaps. The
excess per round over the median is a net of host gaps (+), eager probe rounds (+), round 0 (+) and faster drain rounds (-).

### 2.6 Dependency-feasible schedule and resource service (review 2 point 1; MODEL; S4:44-50)

The REQ:40 budget is a slice of the AGGREGATE cohort max. A span cannot overlap work that it depends on. Per layer and per
request the chain is: norm -> QKV GEMM -> selection (needs Q) -> residency / fetch (needs the selection) -> attention
(needs Q and the fetched KV) -> O proj -> MLP -> next layer (needs this output). Across passes, D0 needs J's draft row, D1
needs D0, the next J needs D1. So inside one request group there is no independent work to overlap at kernel level.
Overlap that respects these dependencies can come only from (i) inside a kernel (HBM traffic under tensor-core work: it
already happens and is inside every measured kernel time), (ii) independent request groups (split B into halves and
interleave one half's attention with the other half's GEMM), or (iii) a prediction of the next selection (a hint; misses
then fetched on the critical path).

Resource service per joint pass. References: 312 TF bf16 peak (vendor, CITED R2:165); the one-row flash call A logical
byte rate 1.527 / 1.543 TB/s as an ACHIEVED memory-rate reference (R2:166; unmatched, 2.7); bulk pinned H2D 25.2 GB/s
(CITED R2:171). Work: 13.624 / 20.057 TFLOP (RV row 19); weights 15.768 GB; KV LOGICAL 32.85 / 48.36 GB (C128 from the C73
page count); H2D 557.46 / 3,347.6 MB. Shared work (the draft row's weights and pages) is counted once.

| stage (dependent order) | resource | C128 floor | C128 measured | C73 floor | C73 measured |
|---|---|---|---|---|---|
| GEMM stage (QKV, O, MLP, LM head) | tensor cores (weights 10.3 / 10.2 ms of HBM inside) | 43.67 | 64.80 | 64.29 | 79.88 |
| attention (fork) | HBM, logical bytes | 21.51 | 37.12 | 31.34 | 52.17 |
| PCIe fetch | link | 22.12 | 21.30 | 132.84 | 160.14 |
| selection + tentative + small (unmodelled) | mixed | not modelled | 60.60 | not modelled | 75.69 |
| A: dependent, one group (sum of the above, floors + measured rest) | | 147.90 | 207.38 (joint) | 304.15 | 382.04 |
| B: two-half cross-request split, ideal interleave, weights read twice, no pipeline fill | | 104.27 | - | 208.53 | - |
| the model's terms only, dependent: GEMM floor + attention floor (hand) | | 65.18 | | 95.63 | |
| the model's aggregate budget (REQ:40) | | 70.16 | | 97.04 | |

Reading. (1) With these floors, GEMM + attention fit the budget even in series (65.2 <= 70.2; 95.6 <= 97.0): the gap to
the budget is mostly efficiency (GEMM at 67-80 % of peak; the fork's logical rate) plus unmodelled stages and PCIe, not
only missing overlap. (2) The 41.5 / 57.8 ms "overlap term" is the difference between the model's serial form and its
aggregate max; it is NOT a measured recoverable amount. (3) Schedule B is the only dependency-feasible way, inside one
pass, to put attention under GEMMs; it pays a second weight read, half-M GEMMs and a pipeline fill; no record tests it
(X19). OVERLAP FEASIBILITY IS OPEN, AND THE AGGREGATE MAX MAY BE OPTIMISTIC.

### 2.7 Logical pages vs physical HBM reads (review 2 point 2)

| item | what we have | label |
|---|---|---|
| scope of a program | one union page list per (request, KV head); a 64-row tile = 4 query rows x 16 query heads of that KV head; per-row 8-bit extents mask the tile; 2D path, no split-KV (fork header lines 14-35, R4:71-76) | CODE (via R4) |
| dispatch | 32 fork launches per pass; grid (432, 2, 1) at B216, (636, 2, 1) at B318 = (2 x B, 2, 1) (R2:130) | MEASURED; the meaning of the two grid axes is NOT established in the inputs (X18) |
| lengths | union mean 71.18 pages per (layer, KV head, request), max 99; + 1.06 tail / buffer block; C73 check arm only, teacher mode, 32 passes (V1B:26-44) | MEASURED (C73) |
| payload | one page = 64 tokens x 128 dims x (K + V) x 2 B = 32,768 B per KV head = the author's unit / 64 | DERIVED |
| logical bytes | pages x 32,768 B; 32.85 / 48.36 GB per pass (C128 uses the C73 count) | DERIVED / MODEL (C128) |
| physical HBM bytes | NOT MEASURED. L2 hits can make them lower; a second grid axis that re-reads K or V pages can make them higher (up to about 2x for the re-read part) | unknown -> X2 (ncu dram__bytes_read.sum) |
| "0.99 ratio" | logical counter (live, C73 check arm) vs the author's averaged historical U / I (four requests, teacher-forced): logical vs logical; it neither proves one physical fetch per page nor validates C128 | scope statement |
| TB/s values | LOGICAL bytes / kernel duration; not a hardware-counter traffic rate | label |
| flash call A comparison | call A: 1 query row x 16 heads, window pages only (63.51 per row), no union, the tentative pages in a separate call B, split-KV + combine, window slot order; fork: 4 rows, union table, per-row masks, CIS bias in the tile, no split; same device-duration boundary. Masks, query rows, page layout and split differ | UNMATCHED comparison |

## 3. The schedule: intended K3 / lag 2 / deferral design vs the implemented Figure 3 schedule

One statement for R1, R2 and R3: ROWS, POSITIONS AND THE 2-PASS VERDICT LAG MATCH THE K3 DEFERRAL DESIGN (record trace,
request 0 of i4dc d3a, rounds 0 and 169; RV:106-109). THE LAG'S PURPOSE (draft-issued background transfer during
subsequent drafting, and retention of requested blocks until the verify read; speckv.txt:133-135 and its section 3.3) IS
NOT IMPLEMENTED (R3:58). The author's inputs are a fixed-lead replay with immediate arrival and same-selection retention;
our code has neither. R2:117-118 holds for the transfer purpose only.

Agreements: 3 weight passes per round; 3 draft rows and 3 verify rows per request; verify inputs t at q, e1 at q + 1, e2
at q + 2; checks e1-e3; no bonus row; verdict lag 2 passes; the union of the verify rows is read once LOGICALLY per
(request, KV head).

| # | item | intended (October 21 model) | implemented (Figure 3, contract 4) | effect on the model's terms | evidence |
|---|---|---|---|---|---|
| S1 | which pass carries the verifier | not fixed | J is the first pass of the round: J, D, D | none at p 0 | R1:163-166 |
| S2 | dispatch shapes per round | 3 GEMM(B) + GEMM(3B) (verify as a second call set) | 2 GEMM(B) (D passes) + GEMM(4B) (J: 3 verify + 1 draft row in one call) | curve: -4.87 / -3.73 ms for our shape; measured per round 104.37 / 131.29 vs 100.10 / 140.54 | S4:65-66 |
| S3 | draft-row KV | td: 63 resident groups of a target step, no loads | resident window only, misses skipped; plus call B; plus pool residency | +11.3 / +12.9 ms per step (BEST, elapsed) | R3 row 1 |
| S4 | lag purpose: background transfer | drafts' requests packed on the CPU and moved during drafting (the author's traffic 2.91 / 9.10 / 6.47 MB) | drafts request nothing; no packer | NOT IMPLEMENTED | R3:109-116 |
| S5 | lag purpose: retention | requested blocks kept until the verify read (overflow peak 0 / 5.33 / 1.64 MB) | no retention across rounds (KW keeps store slots at C73 only) | 23-24 % avoidable refetch | R3:58 |
| S6 | verify-row selection | not charged | per row, 4B rows | +17.8 / +24.4 ms | R2 row 3 |
| S7 | verify misses | ready (lag) | synchronous zero-copy gathers in J | serialization +13.1 / +150.9 ms | S3:33-34 |
| S8 | union / chain / write | none | per layer | part of +25.9 / +27.2 ms | R2:230 |
| S9 | deferred-position replay | none | twin replay of q + 3 | small | R1:241 |
| S10 | the deferral | verdict 2 passes after the newest draft | newest token deferred; K / V reused; no recompute row | none in steady state | RV:106-109 |
| S11 | state | C81 slots + overflow + reserve + landing | tentative buffer + one store X 129 + arena | row 7 | R3 row 3 |
| S12 | produced tokens | accepted drafts | accepted + correction | numerators kept apart | R3:60 |
| S13 | rejection / recovery | fixed lead, n = 3 + 3p, no recovery service | decision (a): drop 4 - n drafts, FILL round, v = 2 group; 1.70 elapsed rounds per reject | Y only (no double time charge) | S4:67-68 |
| S14 | overlap | aggregate max | one dependent stream | open (2.6) | R2:211-215 |
| S15 | host / commit / reset | 0 | 6.1 / 8.5 ms per round + between-round host | row 9 | R2:191-192 |
| S16 | start / drain / capture | steady only | prelude, first-use capture, final groups, static-batch drain | row 11 | V4 |
| S17 | numerical path | not stated | FAST, near-tie class, NOT certified | correctness | SI:67-73 |
| S18 | timing statistic | eager median td, graph median GEMM, p50 s / r | CUDA-event median of the steady set (drain in) | different statistics | R1:248 |

## 4. Retractions and corrections of our own earlier claims

| # | earlier claim (where) | status | why (evidence) |
|---|---|---|---|
| C1 | "about 2 model rows per kept token ... makes 1.0x impossible"; "the remaining gap is structural"; "No path in this design reaches 1.0x" (H29:20, :28-31, :78; L6:41, :37, :128-129, :165-166, :179, :225; SL:1123) | RETRACTED | The original dense formula charges these rows (6 per request per cohort for Y = 3) and still predicted +17.17 %. The dense charge is met. The gap is unmodelled necessary work, serial PCIe, execution efficiency (hypotheses pending X2, X17), the draft-step body, and an aggregate overlap whose feasibility is open (2.6). Extra rows do not by themselves establish proportional elapsed cost (REQ:56-59) |
| C2 | "the GPU is busy with arithmetic"; busy fraction and the launch sweep as evidence of a limit (H29:29-30; L6:51, :168, :217) | RETRACTED as evidence of a limit | busy = the single stream had work queued; it includes PCIe-waiting gather kernels (21.3 / 160.1 ms); GEMMs at 67-80 % of peak; a launch sweep cannot establish an architectural limit (REQ:63-64) |
| C3 | the I4C trigger "fired" (SL:994) | CORRECTED | 0.0514 (nova05) fired; 0.0489 (nova07) would not (SI:174-177) |
| C4 | STATUS_LOG key counts | CORRECTED | one too high in 10 itemized entries (SI section 13; SL:1317-1319); SI:63 says 11 |
| C5 | L7 "label medians do not add at B318 (-12.7 ms)" | CORRECTED | drain-round sampling artefact; registered result kept; reanalysis in 2.5 |
| C6 | L61 "mean and effective now rank as the median" (H29:10-11) | CORRECTED (scope) | mean yes (SI:402-403); whole run no: first-use capture (row 11) |
| C7 | L62 cause | CORRECTED (status) | MODEL cause; guard never implemented (SI:522) |
| C8 | verify-row PCIe "STRUCT, needs an algorithm change" (L6:167, :226) | NOT ESTABLISHED | the lag's transfer purpose is not implemented (S4); coverage untested (X8) |
| C9 | "free drafts give only 0.91x SH" (H29:31) | RE-SCOPED | true for this implementation's joint pass only |
| C10 | this ledger, version 10:24Z: "max() = false model assumption"; "capacity NOT a gap"; "83.75 GB ceiling"; "pages x1.004"; fork "inefficient execution" | CORRECTED (review 1) | see section 10 |
| C11 | this ledger, version 10:34Z: "overlap not realized 41.5 / 57.8 ms" read as a recoverable amount; V1 charged a joint-only allowance; C-interpolated bytes 7.27 / 9.37 MB; model C73 Q 3,784.9 and C73 = 1.105-1.110x strongest; capacity margins computed on R1's segment and constant (+0.3..+2.8 etc.); unscoped "no system / library" wording | CORRECTED (review 2) | overlap term = aggregate-max minus serial form, not recoverable (2.6); interference charged once on the round; the author's bytes replace the guesses; C73 model Q 3,621.8 (transfer binds) = 1.057-1.062x strongest, 1.406x C73's own SH (hand: 3,621.76 / 2,575.2); capacity now on the author's rule and labelled basis-dependent; prior-art claims scoped to the 19 repositories read (section 6) |

## 5. Model revision

### 5.1 Survives

| assumption | verdict | evidence |
|---|---|---|
| shared weight reads (3 weight passes per cohort) | SURVIVES | 129 GEMM launches per pass, one per weight (R2:128) |
| logical read-once union at about U per (request, KV head) | SURVIVES logically at C73 (x0.99); physical reads unmeasured; C128 unmeasured | V1B; 2.7 |
| dense charge | SURVIVES (met); reference sign unknown above M 768 | row 2 |
| rows, positions, 2-pass lag | SURVIVE (implemented) | section 3 |

### 5.2 False, not realized, open or unverified

| assumption | verdict | evidence |
|---|---|---|
| aggregate max (memory under compute within the cohort) | OPEN: not realized between kernels; dependency-feasible only with independent request groups or in-kernel fusion; may be optimistic | 2.6 |
| transfer hidden, readiness (the lag's purpose) | NOT REALIZED | S4, S5 |
| td without pool residency / tentative work | FALSE for a cached C > 63 draft (audited against td's kernel classes) | row 1 |
| verify-row selection, tentative state cost 0 | FALSE (omitted necessary work) | rows 3, 5 |
| p 0, no recovery / drain / startup / capture | FALSE for the live run (the author's inputs are no-rejection) | rows 8, 11 |
| decision / commit cost 0 | FALSE | row 9 |
| zero residual verifier traffic | a REPLAY property, not a live fact | conventions |
| memory | a device-used basis; not comparable with our allocated peaks without X16 | row 7 |

### 5.3 The quantified revised model (MODEL; S4:70-101)

Per round of the implemented schedule: interval R = 2 d + J + c.
- d = td x the measured ELAPSED draft-step ratio (I3 1.3807 / 1.2919; best 1.2067 / 1.1716); anchor I3 90.99, best 81.56
  (alt 84.40) ms.
- J = td + extra + GEMM(3B) [the model's terms, serial] + joint one-step-body proxy + dense excess + attention above the
  model + verify-row selection + tentative state + bookkeeping + EXPOSED transfer (AUTHOR bytes / 22.0-27.6 GB/s) + small
  ops + non-busy. Best code adds the measured lever deltas per request (no residency delta: the bytes carry it).
- c = commit + decision + reset.
- Q = 1000 x B x Y / interval; Y accepted = Y_rev 2.808-2.864 (recovery inside Y only); committed adds 0.031.
- Variants (MODEL upper bounds; interference charged ONCE on the round): V1 = (R - transfer)(1 + s); V2 = (R - transfer -
  overlap term)(1 + s); V3 = V2 also without the fork term. V2 / V3 assume the aggregate max is achievable, which 2.6
  leaves OPEN.
- Drain sensitivity: + 11.74 ms per round at the anchor (interpolated probe shift).

Results. Measured configs (steady; committed tok/s; /strong same GPU):

| point | R (MEASURED) | V1 | V2 | V3 |
|---|---|---|---|---|
| C128 I3 (nova07) | 365.42 ms, 1,709.8, 0.501 | 382.12, 1,635.1, 0.480 | 336.21, 1,858.4, 0.545 | 318.96, 1,958.9, 0.574 |
| C73 I3 (nova05) | 585.46, 1,525.8, 0.445 | 461.04, 1,937.6, 0.566 | 398.69, 2,240.6, 0.654 | 376.22, 2,374.4, 0.693 |
| C128 BEST (nova05) | 324.88, 1,924.1, 0.562 | 336.85, 1,855.7, 0.542 | 290.94, 2,148.5, 0.627 | 273.69, 2,284.0, 0.667 |
| C73 BEST (nova07) | 524.66, 1,699.0, 0.498 | 441.16, 2,020.5, 0.593 | 378.81, 2,353.0, 0.690 | 356.34, 2,501.4, 0.734 |

At C128, V1 is BELOW R: the model's own interference (10.6 % of the round) costs more than the 20 ms of exposed transfer it
would hide. This uses the model's s, measured for a CPU8 packer beside the decode, not for a copy-engine path.

Anchor (MODEL; committed tok/s and the ratio vs 3,426.2 / 3,410.1; accepted about 1 % lower; S4:72-95):

| case | R | R + drain | V1 | V2 | V3 |
|---|---|---|---|---|---|
| I3 code, 27.6 GB/s, Y 2.864 | 470.97 ms, 1,764.3, 0.515 / 0.517 | 0.502 / 0.505 | 438.80, 0.553 / 0.555 | 380.90, 0.637 / 0.640 | 359.83, 0.674 / 0.677 |
| I3 code, 22.0 GB/s, Y 2.808 | 488.09, 1,669.5, 0.487 / 0.490 | 0.476 / 0.478 | 438.80, 0.542 / 0.545 | 380.90, 0.624 / 0.627 | 359.83, 0.661 / 0.664 |
| best code, d 81.56, 27.6 GB/s, Y 2.864 | 426.35, 1,948.9, 0.569 / 0.572 | 0.554 / 0.556 | 390.31, 0.621 / 0.624 | 332.41, 0.730 / 0.733 | 311.34, 0.779 / 0.783 |
| best code, d 84.40, 22.0 GB/s, Y 2.808 | 449.14, 1,814.3, 0.530 / 0.532 | 0.516 / 0.519 | 396.47, 0.600 / 0.603 | 338.57, 0.703 / 0.706 | 317.51, 0.749 / 0.753 |
| original model | 232.48, 3,703.54 accepted, 1.081 / 1.086 (vs old 1.172; vs I3 own SH C128 1.111) | - | - | - | - |

RECOMPUTED ANCHOR RANGE (accepted to committed, both denominators): as implemented 0.47-0.52x (I3 code) and 0.51-0.57x
(best code), drain sensitivity included; V1 0.54-0.56x / 0.59-0.62x; V2 0.62-0.64x / 0.69-0.73x; V3 0.65-0.68x /
0.74-0.78x. (After review 1 the V3 range read 0.70-0.72 / 0.79-0.83; the drop comes from charging interference once on
the whole round.) None of these is a limit.

### 5.4 Dispatch shapes and draft accounting (review 2 point 4; S4:64-68)

| item | model | ours | evidence |
|---|---|---|---|
| GEMM dispatch per round | 3 GEMM(B) + GEMM(3B) = 100.10 / 140.54 ms (curve) | 2 GEMM(B) + GEMM(4B) = 95.24 / 136.81 (curve); measured 104.37 / 131.29 | S4:65-66 |
| draft positions per 3 accepted | fixed lead: n = K + p (1 + lag) = 3 + 3p = 3.084 / 3.121 | empirical accounting n = 3 + 3p x 1.70 = 3 + 5.1p = 3.144 / 3.205 | S4:67-68 |
| draft attempts lost per reject | 1 + lag = 3 | 4 - n_bar = 3.129 / 3.188 (drop list of decision (a)) | S4:67-68; R3:183-184 |
| elapsed rounds per reject | 1.0 (fixed lead) | 1.70 / 1.70 (fill round + v = 2 group; whole static-batch rounds) | R3:60 |
| Y accepted | 3 / (1 + p) = 2.918 / 2.884 | empirical 2.863 / 2.808; measured active-request 2.855 / 2.798 | S4:67-68 |
| where recovery is charged | - | ONLY in Y. The round time is the batch's steady median; the fill / v = 2 rows are padded rows inside it; no separate recovery time is added | 5.3 |
| numerators | accepted drafts | accepted 109,318 / 160,248; corrections 1,058 / 2,250; committed 110,376 / 162,498 | R3:60 |

## 6. What would realize the modeled mechanisms (prior art, SCOPED to the code read)

Scope. R4 read 19 repositories at the commits in R4 0.2 (R4:44-64; SL:1332 says 18): vllm v0.20.2 (bc150f5),
flashinfer v0.5.3 (0e68a2f), flash-attention main (819ef26: FA2 csrc + FA3 hopper), sglang (e371881), EAGLE (cb7e084),
Medusa (e2a5d20), Quest (01c1623), native-sparse-attention (bd67af5), MoBA (b5d5836), InfLLM (12b7079), infllmv2_cuda_impl
(93cf2ec, README only), NOSA (90045b6), ShadowKV (e51904c), RetrievalAttention / RetroInfer (03f912c), FlexLLMGen
(004ffef), InfiniGen (3754f52), LMCache (1f3d4f0), TriForce (164c8c0), MagicDec (09cd671). Every statement below holds
ONLY for these versions and backends. It is not a claim about other systems, versions or backends.

### 6.1 Compatibility matrix (from R4 sections 1-5; "-" = not stated in R4; KERNEL capabilities and TRANSFER management kept apart)

| repository (version) | K1 one page load shared by several query rows of a request (GQA 16) | K2 per-row, per-KV-head block selection | K3 per-query additive bias | K4 page 64 tokens | K5 sm80 | K6 graph-safe with per-layer data-dependent page lists | T1 host-offloaded pages | T2 fetch overlapped with compute | T3 fetch driven by the selection |
|---|---|---|---|---|---|---|---|---|---|
| vLLM v0.20.2, Triton unified attention | NO at group 16 (BLOCK_Q 1) | NO (one block table per sequence) | NO | - | yes | yes (device metadata; ALWAYS) | kv_offload worker: yes (step granularity, whole blocks) | copy-engine batched DMA, event completion, outside graphs | NO (known block set) |
| FlashInfer v0.5.3 prefill wrapper | YES (R x 16 rows in one CTA on sm80) | token mask without head index; per KV head needs one "request" per (request, KV head) or a JIT variant | only by a JIT logits transform | - | yes | NO (host plan per new page list: one sync per layer) | - | - | - |
| FA2 (819ef26) | NO at seqlen_q 4 | NO | ALiBi / softcap only | NO (page % 256) | yes | - | - | - | - |
| FA3 hopper/ (819ef26) | YES (PackGQA) | NO (one page table per sequence) | NO | - | sm80 path exists, not the documented target, not verified | - | - | - | - |
| SGLang (e371881), EAGLE verify | via FlashInfer | one mask for all heads; page_size 1 with a custom mask | NO | NO (page 1) | yes | plan once per step (same list in every layer) | - | - | - |
| MagicDec (09cd671) | via FlashInfer | NO (dense verify) | NO | - | yes | plan outside the compiled region, per step | - | - | - |
| TriForce (164c8c0) | dense verify | draft: shared per-head chunk set | NO | chunk 8 | yes | - | YES (whole layers) | NO (copy on the current stream) | NO |
| EAGLE / Medusa | dense eager matmul | NO | tree mask only | - | - | - | - | - | - |
| Quest (01c1623) | NO (q_len 1, batch 1) | per head | NO | page | - | - | NO | - | - |
| NSA (bd67af5) | GQA group only, not across tokens | YES per (token, KV head) | NO | block | - | - | NO | - | - |
| MoBA (b5d5836) | YES, chunk-major, prefill only, not bit-exact | per (head, query) | NO | chunk | - | - | NO | - | - |
| InfLLM (12b7079) | selection shared by heads and tokens (semantic change) | NO (shared) | NO | block | - | host sync per layer | YES | YES (side stream, same layer) | YES (with a host sync) |
| InfLLMv2 (93cf2ec, README) | - | stage-1 + block-sparse | - | 64 | - | - | NO | - | - |
| NOSA upstream (90045b6) | NO (q_len 1) | per KV head | YES (per-token bias) | 64 | yes | pooling + top-k captured | YES (pinned, UVA SM loads) | NO (in line) | YES (GPU diff) |
| ShadowKV (e51904c) | NO (q_len 1) | per KV head | NO | chunk | - | - | YES (V) | YES (copy_stream; SM gather) | YES |
| RetroInfer (03f912c) | NO | per GQA group (clusters) | weighted LSE merge | cluster | - | 4 graphs per layer, host work between | YES | CPU thread pool; host sync per layer | YES |
| FlexLLMGen (004ffef) | dense | NO | NO | - | - | - | YES | YES (streams; full sync per layer) | NO (dense) |
| InfiniGen (3754f52) | dense | speculated top-k | NO | token | - | - | YES | YES (prefetch stream) | speculated (misses not corrected) |
| LMCache (1f3d4f0) | n/a (connector) | n/a | n/a | n/a | - | - | YES | YES (copy engine + scatter, layer-wise) | NO (static set) |
| OUR fork (2e9e497) | YES (4 rows x 16 heads, union) | YES | YES (CIS bias in the tile) | 64 | yes | YES (device-built union, captured) | YES | NO | YES (in line) |

Scoped conclusions. (1) In these 19 repositories at these commits we found no kernel or library call that combines K1-K6
on sm80. (2) The kernel capability (K columns) and transfer management (T columns) are separate questions: our missing
piece is in the T columns (T2: overlapped, selection-driven fetch), where InfLLM, ShadowKV, RetroInfer, InfiniGen,
LMCache and the vLLM offload worker show patterns, each with a stated limit (host sync per layer, speculated selection,
or a static block set). (3) FlashInfer's custom-mask prefill was tried on our code: 1.68-1.70x our device time + 77-100 ms
of plan per tick (R4:77-79). (4) None of this proves that no other system has the combination.

### 6.2 Mechanisms

| mechanism | blocker now | pattern in the code read | needs | status |
|---|---|---|---|---|
| lag purpose: background transfer + retention | no draft request, no packer, no retention | T2 rows above | no new kernel: streams / events / copy engine + a hint; our spec_prefetch.py (rung L5) never ran on this ladder; window / selected split needs an LSE merge + numerical gate | UNTESTED (X8, X14) |
| overlap of memory-bound and compute-bound work | dependencies (2.6) | none in the code read for an exact per-layer selection inside a graph (R4:524) | independent request groups (two-half split) or in-kernel fusion | OPEN (X1, X19) |
| logical shared reads | realized in our fork | K1 rows above | physical traffic check (X2) | logical MEASURED (C73) |
| verify-row selection | omitted by the model | NSA in-kernel top-k (design reference) | a fused multi-row selection would be a new kernel | not a lever yet |
| small kernels / launch | removed in part | vLLM device-only metadata | done (graphs, Inductor) | MEASURED (warmed); first-use capture in row 11 |

NOT established: no architectural limit; no physical-traffic figure; no overlap figure; no prefetch coverage.

## 7. Discriminators (GPU items PROPOSED; each needs protocol review and a human GO)

| X | question | rows | method | cost | GPU? | status |
|---|---|---|---|---|---|---|
| X1 | can the fork / H2D gather overlap a GEMM on this A100 | 4, 6, 10 | replay layer-5 inputs: fork alone, GEMM alone, both on two streams, gather + GEMM on two streams | about 2 min, no new kernel | yes | PROPOSED |
| X2 | physical HBM bytes and the cause of the fork's logical rate | 4 | ncu on one fork launch: dram__bytes_read.sum, local memory, warps active, occupancy | short | yes | PROPOSED |
| X3 | C128 logical page counts | 4 | VerifyOps(stats=True) in a C128 B216 miniature | miniature | yes | PROPOSED |
| X4 | td contents vs our kernel classes (beyond R3's ncu-class audit) | 1, 3 | CPU read of the J91 per-phase record | CPU | no | not run |
| X5 | best-arm device split | 1, 5, 9 | port the I4C profiler option to the I4DC cell | builder + 15-20 min per cell | yes | PROPOSED |
| X6 | decision vs commit | 9 | profiler window around ST:975-999 | short | yes | PROPOSED |
| X7 | per-pass H2D rate | 6 | export meter round_blocks | small change + run | yes | PROPOSED |
| X8 | prefetch coverage | 6 | per-block flags in the I4D accounting cell | change + run | yes | PROPOSED |
| X9 | anchor counterparts (draft step, p) | 1, 6, 8 | C81 (pool 18) B287 timing cell | timing cell | yes | PROPOSED |
| X10 | C81 fit lines; physical transient fit on device-used (nvidia-smi sampling) | 7 | I4M probe at C81 + sampling | about 12 min | yes | PROPOSED |
| X11 | memory basis | 7 | CPU | no | DONE (review 1, V3) |
| X12 | recovery clock | 8 | CPU | no | DONE (within about 5 %) |
| X13 | remaining author items: GEMM rule above M 768; s / r layout variant; the optional copied-union MB | 2, 6, 7 | author | no | PARTLY ANSWERED (U, I, traffic, overflow, td rule, memory rule supplied 2026-10-30) |
| X14 | kappa (side-stream SM gather vs copy engine) | 6 | cite the exported record | CPU | no | OPEN |
| X15 | per-round round_ms (ev0..ev4) + host gap: drain-free medians, wall_s reconciliation, end-effect rounds | 8, 9, 11 | export values the code already computes; rides any next run | small change | rides a run | PROPOSED |
| X16 | memory category table on one basis | 7 | J91M columns vs our arena record | CPU | no | OPEN |
| X17 | GEMM reference sign at M 864 / 1,148 / 1,272 | 2 | graph-timed scan in the existing harness | short | yes | PROPOSED |
| X18 | meaning of the fork grid axes (432 / 636, 2, 1) | 4 | CPU read of fork_vllm_unified_attention.py launch code | CPU | no | OPEN |
| X19 | dependency-feasible overlap: two-half cross-request split of the joint pass | 10 | builder work + one timing cell | new schedule code | yes | PROPOSED (needs a build decision) |

Standing constraints apply: no new allocation, no new GPU experiment, no blind retry, no threshold relaxation.

## 8. Preserved failures and validated lessons

Preserved failures: i4m n373 OOM (C73 B351, sacct 2195575.24 FAILED 2:0; 83.644 GB allocated + 126 MiB request, 0.551 GB
non-PyTorch) kept as ONE observation; the retracted "83.75 GB ceiling" kept in the record of retractions (C10, K3); 52
refuted key x run evaluations in I4 (SI:40); verdict flips (SI:516-517); the I0 point model miss, I2 falsifier, I3 plan-point
misses, I4B T32 EQ failures, FlashInfer 1.68-1.70x (CP:23, :46, :68-69; SI:131-133; R4:77-79); the I4C trigger that did not
replicate; the whole-run loss of the best arms (row 11).

Replicate after the inputs (MEASURED; not yet scored by a scoring agent): i4dc r4a (R/i4dc_r4a_2195576_i4dca, nova07, sacct
2195576.43 COMPLETED 0:0, exit 0 at 2026-10-30T10:04:27Z): L4PTFDGUDCD 1,916.8 tok/s, /SH 0.5750, I4DC-RATIO CONFIRMED, 12
CONFIRMED PREDICTION lines, no MUST FAIL (stdout:87-98); /strong 0.5621; /re-check 0.5634 (S2:18-19); 1.0088x d3a in /SH.

Validated lessons: L61 for the mean (not for the whole run); phases add per round (six spans, +0.41..+0.72 ms).
Corrected / not validated: L60 re-scoped to "fit with a margin for non-PyTorch memory when another arm ran first"; L62
guard missing.
Proposed (numbers by the lead): P1 an aggregate max is a claim about dependency-feasible concurrency; a one-stream trace
can neither confirm nor refute it. P2 busy fraction / launch sweeps are not limits. P3 compare medians over the same
window. P4 replicate a trigger before it fires. P5 state one memory basis per comparison and charge harness removals on
both sides. P6 report warmed, first-use and amortized rates. P7 label logical-byte rates as logical; physical traffic
needs a hardware counter. P8 scope prior-art claims to the versions read.

## 9. Rule record

This writer: no interpreter on the login node; arithmetic in four scripts under
/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_s/ run with `srun -p amd -c 4 --mem=16G -t 30 apptainer exec --bind
/mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif /venv/nosa/bin/python -I -B <script>`. The
Write tool created tmp_sc/recon7_s with the first script. Hand arithmetic is marked "hand" with operands. One login read
of J91M (head, cut, grep) for the snapshot column. No GPU job, no sbatch, no queue directory, no code edit, no commit; no
other plan file edited. Slips reported by others: R1 (sed x2), R3 (variable names), R4 (sed -n '0p', awk_placeholder
token), review 1 (uniq -c). None changed data.

## 10. Review 1 response (PL/RECON7_REVIEW.md)

| finding | response | evidence / change |
|---|---|---|
| 1 max() called false without a concurrency test | ACCEPTED (and extended by review 2) | relabelled; overlap term not recoverable (2.6) |
| 2 no whole-run window | ACCEPTED | row 11, 2.5, amortization |
| 3 capacity basis | ACCEPTED (superseded by the author's rule) | row 7 |
| 4 basis-gap cause / segment | ACCEPTED | row 7 (slack 4.609 + outside 1.119 on B256-344) |
| 5 schedule disagreement | ACCEPTED | section 3 statement |
| 6 one-step body values | ACCEPTED | table 1a |
| 7 joint allocation / interference | ACCEPTED (interference now once on the round, review 2) | conventions |
| 8 drain in steady | ACCEPTED | 2.5 |
| 9 GEMM curve | ACCEPTED | row 2, X17 |
| 10 pages basis | ACCEPTED | row 4, 2.7 |
| 11 fork inefficiency | ACCEPTED | HYPOTHESIS, X2 |
| 12 residual by construction | ACCEPTED | 2.3 |
| 13 committed vs accepted Y | ACCEPTED | conventions, 5.4 |
| 14 83.75 GB ceiling | ACCEPTED | row 7, section 8 |
| 15 link rate comparability | ACCEPTED | row 6 |
| 16 allowance class | ACCEPTED | 2.3 |
| 17 recovery wording | ACCEPTED | row 8 |
| 18 coverage | PARTLY ACCEPTED | baselines added; prior art, failures, lessons were present |
| 19 rule slips | NOTED | section 9 |

### 10a. Review 1's three not-reproduced numbers (full records, RV:266-311)

| RV row | claim (file:line) | claimed | re-derived | source of the re-derivation | why it failed | status here |
|---|---|---|---|---|---|---|
| 17 | pages vs model (R2:39-40) | x1.004 | 1.0038 as stated, but 0.989 (union / (63 + U - I)) and 0.990 (fork_loaded / (64 + U - I)) like-for-like | V1B:40-44 (CHECK.TFC of i3_r2cb: fork_loaded 47,046,615, union_sum 46,354,647, samples 651,264) | the claim divided a count that includes the tail / buffer block (64 + union) by a model count without it (63 + U - I) | corrected (K1); now also labelled LOGICAL (2.7) |
| 31 | capacity margin (R3:41-42) | +11 (C128) / >= +9 (C73) | +10.6 / +8.9 on mixed bases; +0.3..+2.8 / -23 on an allocated basis | V3:43-54 (model formula with J91 allocated lines) | ours were allocated peaks, the model a device-used backbone | corrected (K2); now superseded by the author's rule (row 7: basis-dependent, no margin established) |
| 33 | usable ceiling (R3:59) | about 83.75 GB | C128 ran at peak_with_prefill 83.900 GB after SH in the same process | V1 (summary.json of i4dc d3a) | one OOM observation was generalised into a ceiling | corrected (K3); the OOM record is preserved |

## 11. Corrections list

| K | claim | old | corrected | evidence |
|---|---|---|---|---|
| K1 | pages vs model | x1.004 | x0.99 like-for-like, LOGICAL, C73 only | 10a |
| K2 | capacity margin | +11 / >= +9; "not a gap" | basis-dependent; the author's device-used rule says B216 / B318 do not fit (204.1 / 304.2), yet both ran under the allocated cap; no margin established | row 7; S4:28-37 |
| K3 | usable ceiling | 83.75 GB | not a ceiling | 10a |
| K4 | model C73 transfer / Q | 187.20 ms hidden; Q 3,784.9 | 263.41 ms BINDS; Q 3,621.8 | S4:22 |
| K5 | U, I, bytes at C128 / C73 "not re-derivable; C81 values carried" | carried | AUTHOR: U, I unchanged; traffic 2.9105 / 9.0978 MB | conventions |
| K6 | anchor byte variants | C-interpolated 7.27 / 9.37 MB (our guess) | withdrawn; the author's 6.4658 MB | row 6 |
| K7 | V1 interference | joint-only allowance | once on the round, (R - removed)(1 + s) | 5.3 |
| K8 | model largest B at C128 / C73 | 205.38 / 309.11 (R1, carried overflow, R1 constant) | 204.11 / 304.18 (author's rule: B128-256 segment at B216, C73 overflow 5.33 MB, constant 0.12740196) | S4:31-34 |

## 12. Review 2 response (2026-10-30, with the author's inputs)

| point | response | change in this file |
|---|---|---|
| author's inputs (U, I, traffic, overflow, td rule, memory rule) | ACCEPTED, all model columns recomputed (S4) | conventions; rows 1, 4, 6, 7, 8, 10; 2.2; 5.3; K4-K8. Main change: at C73 the model is TRANSFER-bound (263.41 ms, Q 3,621.8). U - I = 8.969 groups at every config; extra bytes 18.81 MB per request per cohort. Memory: anchor 83.7655 GB (287.95 largest B, reproduces); C128 B216 87.91 GB (204.1); C73 B318 87.22 GB (304.2); snapshot removal is MODEL on the model side only (217.0 / 332.0 / 312.7) |
| 1 aggregate max; one stream does not refute in-kernel overlap; dependency-feasible schedule | ACCEPTED | 2.6: dependency chain, resource-service floors per stage, schedules A (dependent) and B (two-half split); the 41.5 / 57.8 ms term relabelled "model assumption, not a recoverable amount"; "overlap feasibility is OPEN; the aggregate max may be optimistic"; X19 added |
| 2 logical vs physical pages; scope; unmatched flash comparison | ACCEPTED | 2.7 table: program scope, dispatch, lengths, payload, logical vs physical, the 0.99 ratio scoped, TB/s labelled logical, flash comparison labelled UNMATCHED; X18 added |
| 3 per-round disjoint spans; residual-correction exclusion; reanalysis vs corrected code; warmed vs first-use; amortization | ACCEPTED | 2 rules and 2.1 (disjoint phases; medians only in the records, X15); 2.5: the original 11-round result kept (374.565 ms, -12.675 REFUTED), rounds 176 / 192 / 208 named, criterion "a request had finished", labelled reanalysis of saved records (no corrected-code run exists); row 11: warmed, first-use, amortized (k runs; break-even 2.99 / 1.50) |
| 4 dispatch shapes; fixed-lead n; draft attempts; no double charge; numerators; draft excess outside joint; interference once | ACCEPTED | 5.4 table; row 8; 2.2 / 2.3 (draft excess only in the draft span); conventions (interference once); 5.3 variants recomputed |
| 5 scope prior art; compatibility matrix; transfer vs kernel | ACCEPTED | 6.1 matrix (K and T columns apart), scoped conclusions, version list; R4's "18" vs 19 listed repositories noted |
| 3 of 39 not-reproduced numbers | ACCEPTED | 10a with full records |
