# HANDOFF RECON7 (2026-10-30): our implementation against the original October 21 calculation

Revised 2026-10-30T18:51:04Z (`date -u`) after review 1 (/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/RECON7_REVIEW.md)
and review 2 (with model inputs from the model's author). Full ledger: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_rev7.md
(review responses in sections 10 and 12). This page does NOT claim that the comparison is complete; "Still open" names
the open rows.

## Terms

- ORIGINAL CALCULATION: the October 21 model at C81 / K3 / B287. It predicted 3,703.5 accepted tok/s, +17.17 % over the old
  plain reference 3,160.8 tok/s. MEASURED CONFIGS: C128 / B216 and C73 / B318. No run exists at C81 / B287.
- AUTHOR INPUTS: U, I, draft traffic, overflow peaks, the td rule and the memory rule that the model's author supplied
  today. They are historical replay inputs (four requests, teacher-forced, no rejects), not live measurements.
- ROUND: one joint pass (3 verify rows + 1 draft row per request) + two draft passes (1 row per request).
- I3: the ladder rung with the forked attention kernel. BEST: I3 + lean draft + CUDA graphs + union hoist + compiler (+
  store keep at C73).
- STRONGEST: plain cached NOSA, steady, whole-step graph, C128 B216: 3,426.2 (nova05), 3,410.1 (nova07) tok/s.
- WINDOWS: STEADY = median round, reused graphs ("warmed"), end-of-run rounds still inside. WHOLE RUN = all tokens / all
  wall time, including the first-use graph capture.
- AGGREGATE MAX: the model charges the larger of memory time and compute time over a whole cohort, so it assumes that they
  overlap.
- LOGICAL vs PHYSICAL reads: logical = the pages a kernel's page list names; physical = bytes the GPU memory actually
  serves (needs a hardware counter).

## The table in short form (C128 / C73; ms per round, steady, unless stated)

| component | model says (author inputs) | we measure | verdict |
|---|---|---|---|
| draft pass | td 54.8 / 75.3 | 66.1 / 88.2 per step (BEST, elapsed like td) | +11.3 / +12.9: pool residency + tentative-buffer work that td does not have (checked against td's own kernel classes) |
| joint dense | 63.4 / 90.0 (two GEMM calls) | 64.8 / 79.9 (one call over 4B rows) | charge met; the one-call reference is extrapolated, sign unknown |
| joint selection | 0 for 3 verify rows | +17.8 / +24.4 | omitted necessary work |
| joint attention | logical union read once | logical union read once, x0.99 of the model at C73 only | logical sharing works; physical traffic NOT measured; the speed comparison with flash is unmatched |
| tentative state | 0 | +25.9 / +27.2 (I3); about 11.8 / 13.7 (BEST) | omitted work + small kernels |
| transfers | 2.91 / 9.10 MB per request; 62 ms hidden / 263 ms (at C73 the model itself is TRANSFER-bound) | 2.58 / 10.53 MB (BEST 8.07) urgent, serial, 20 / 116 ms exposed | the background-transfer mechanism is not built |
| memory | device-used basis: 87.9 / 87.2 GB, so B216 / B318 "do not fit" | both ran under the 84 GB allocated cap | different bases; no margin established either way |
| yield | 3 accepted per round | 2.889 / 2.780; 1.70 rounds lost per reject | rejects 2.8 % / 4.0 %; consistent within about 5 % |
| host / commit | 0 | about 8 ms launch (mostly removed); commit 6-8.5 ms | small |
| overlap | aggregate max | one stream, no overlap between kernels | feasibility OPEN; the aggregate max may be optimistic |
| window | steady only | steady: BEST > I3; whole run: BEST 1,292 / 1,243 < I3 1,397 / 1,275 tok/s | the first-use capture (9.6 / 10.1 s) reverses the order; BEST passes I3 if the capture is reused for about 3 / 1.5 runs |

Per round (steady): I3 is 178 / 322 ms above the model interval; BEST is 137 / 261 ms above it.

## Main findings

1. With the author's inputs the original 17 % result still reproduces at the anchor. At C73 the model is now limited by
   its own transfer (Q 3,622 instead of 3,785).
2. The work counts match: rows, passes, the 2-pass lag, the dense charge, logical shared pages.
3. The timing gap is mostly work that the model leaves out (verify-row selection, tentative state, commit), slower
   execution than the model's references, and serial PCIe copies. A resource view shows that even in series the GEMM and
   attention work could fit the model's budget at peak rates; so overlap is not the only missing piece.
4. Overlap between memory work and compute work is not built, and we do not know whether it is possible here. Inside one
   group of requests every step depends on the previous one; overlap needs another group of requests or fused kernels.
5. The lag in the schedule exists for background copies and retention; our code has the lag but neither of these.
6. Steady and whole-run results disagree about the best arm; report both, plus the amortization assumption.

## What we retract or correct

- "About two model rows per kept token makes 1.0x impossible" and "the GPU is busy" as proof of a limit.
- From our own earlier drafts: the overlap term as a recoverable amount; "capacity is not a gap"; the 83.75 GB ceiling;
  "pages x1.004"; our guessed byte values (replaced by the author's); the model's C73 result 3,785 (now 3,622);
  interference charged in two places (now once).
- Unscoped prior-art wording: it now covers only the 19 repositories and commits that R4 read, in a compatibility matrix.

## The revised model (MODEL; LEDGER_rev7.md 5.3)

At the anchor, charged as our code runs: 0.47-0.52x the strongest (I3 code), 0.51-0.57x (best code). With background
transfer: 0.54-0.56x / 0.59-0.62x. With transfer hidden, aggregate overlap and attention at the flash rate: 0.65-0.68x /
0.74-0.78x. The original model: 1.08x the strongest. None of these is a limit; the overlap variants assume an overlap whose
feasibility is open.

## Review 2 response (short; full table in LEDGER_rev7.md section 12)

| point | response |
|---|---|
| author inputs | accepted; all model columns recomputed; at C73 transfer binds; memory now on the author's rule |
| 1 aggregate max vs dependency-feasible schedule | accepted; resource-service table and dependency chain added; overlap feasibility open; max may be optimistic |
| 2 logical vs physical pages | accepted; scope, dispatch, payload, logical / physical split; flash comparison labelled unmatched |
| 3 spans, exclusion decision, warmed vs first-use | accepted; original 11-round result kept; rounds 176 / 192 / 208 named with the criterion; amortization stated |
| 4 dispatch shapes, n = 3 + 3p, no double charge | accepted; 2 GEMM(B) + GEMM(4B) vs 3 GEMM(B) + GEMM(3B); recovery only in yield; interference once |
| 5 scoped prior art | accepted; matrix with kernel and transfer columns apart |
| 3 of 39 not-reproduced numbers | listed with full records (ledger 10a) |

## What is fixable without new kernels

- Background transfer and retention: streams, events, copy engine, a prefetch hint; our spec_prefetch.py (rung L5) never
  ran on this ladder. Needs a numerical gate if attention is split.
- First-use capture: capture once at engine start and reuse.
- Already done: graphs, compiler on small helpers, union hoist.
- A cross-request split to overlap attention with GEMMs is new schedule code (X19), not a kernel; whether it pays is unknown.

## What needs a decision from you

1. GPU tests (all PROPOSED, each needs protocol review and your GO): X1 overlap test (about 2 min), X2 ncu on one
   attention launch (physical bytes), X15 per-round export (rides any run), X19 the two-half split (needs a build).
2. Remaining author items (X13): the GEMM rule above M 768, the s / r layout variant, the copied-union size.
3. The FAST tie rule (unchanged).

## Still open

Overlap feasibility (X1, X19); physical traffic and the fork rate cause (X2, X18 CPU); C128 page counts (X3); transfer
coverage and rate (X7, X8); best-arm split (X5); decision vs commit (X6); drain-free and whole-run per-round values (X15);
memory on one basis (X16 CPU; X10 physical fit); GEMM reference sign (X17); anchor counterparts (X9, X10); td contents
(X4 CPU). Closed in ms at the measured configs: dense charge, selection, tentative state, yield / recovery.

New since the inputs: the I4DC replicate r4a ended clean (1,916.8 tok/s, 0.575x SH, 0.562x the strongest); not yet scored.

## Sources

- /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_rev7.md; /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/RECON7_REVIEW.md
- /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md
- /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/RECON7_R1_model.md, /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/RECON7_R2_joint.md,
  /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/RECON7_R3_draft_transfer_capacity_schedule.md, /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/RECON7_R4_prior_art.md
- /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/SCORING_i4.md, /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/CHECKPOINTS.md,
  /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_rev6.md, /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/NOTES_graph_mean_gap.md,
  /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HANDOFF_2026-10-29.md, /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/STATUS_LOG.md
- Arithmetic: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_s1.txt .. recon7_s4.txt (scripts in
  /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_s/); review logs /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7_v_*.txt
- Memory source: /mnt/central/users/ojerbi/retroinfer-eval/docs/evidence/hbm_sweep_2175791/scoring/output2_memory.csv
- Run records under /mnt/beegfs/ojerbi/r2-runs/spec/: i3_r4a_2195576_i3a, i3_r3b_2195575_i3b, i4dc_d3a_2195575_i4dca,
  i4dc_d4b_2195576_i4dcb, i4dc_r4a_2195576_i4dca, i4c_e4a_2195576_i4ca, i4c_c4b_2195576_i4cb
