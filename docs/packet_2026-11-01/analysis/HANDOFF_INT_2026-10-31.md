# HANDOFF INT (2026-10-31): the original formula term by term, the fixes, their measured effects, the corrected model

Written by the lead. Revision 2, after the asynchronous review 4 (stamp 2026-11-02T09:16:22Z from `date -u`).
Style: ASD-STE100. Ledger: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_rev7.md sections 14 (term rows),
15 (corrected model, 15.6 conclusion) and 16 (review 4 corrections). Status log:
/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/STATUS_LOG.md. Run records: R = /mnt/beegfs/ojerbi/r2-runs/spec. CPU
receipts: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/<tag>.txt + .receipt (command, host, SLURM job, HEAD, status,
source sha256, pytest exit).

## 1. The answer in short

1. CONCLUSION: this tested implementation misses the original forecast, with several mechanisms still open.
2. The three owners carried their largest evidenced gaps to scoped, bit-exact changes with registered predictions and
   measured effects (x1.08-1.09 integrated per configuration). Every INT cell passed its MUST gates. "Bit-exact" here
   means SAME-SPECULATIVE-ARM equivalence (EQF / PEQ: the optimized arm commits exactly the plain speculative arm's
   tokens). It does NOT mean target correctness: the plain speculative arm's tokens equal the shipped SH's on only
   0.388-0.404 of the checked columns (ledger 16.3); that depends on the FAST tie rule (human decision).
3. INTEGRATED results (COMMITTED tok/s, median of the per-round rates in the steady window; not Q_accepted):
   - C128 B216 (inta): 2,096.1 = x1.090 over L4PTFDGUDCD; 0.622x SH; 0.5995x the matched optimized baseline SHJ1 of the
     same job (SHJ1 about 3,496 tok/s, DERIVED from a rounded ratio; the exact SHJ1 rate and its memory are NOT
     recorded, so SHJ1's common-cap qualification stays OPEN).
   - C81 B287, the anchor (intc): 1,854.8 = x1.080; round 434.4 ms vs 232.48 predicted; the joint region 269.5 ms vs an
     inferred budget of 88.07 ms. MUST gates passed; THREE baseline-relative bands REFUTED low (INT-RATIO 0.6586,
     INT-VS-SHJ1 0.6369, INT-VS-STRONGEST 0.5444).
   - C73 B318 without BF1 (intd): 1,843.2 = x1.080; 0.716x SH; 0.693x SHJ1. With the batched BF1 (intb): 1,821.0.
4. 3,407.2 tok/s is a FROZEN REFERENCE (an earlier whole-step graph run), not demonstrably the strongest baseline: the
   same-job SHJ1 at C128 B216 exceeds it. inta = 0.615x that reference.
5. What the tests show: in the TESTED schedule, verification and the transfer did not hide under the two maxima of the
   expression. X1: the tested fixed-shape stream combinations of dense GEMMs and the fork do not overlap. The measured
   draft-issued prefetch reached break-even (b3 x1.0049). What they do NOT show: that no joint implementation can do
   better (the one-row attention control does not prove four-row optimality), and that the fetch's host / readiness
   cost is irreducible (ready-only snapshot batching and the transport are unbuilt). The serial reconciliation at the
   anchor (ledger 16.4) is accounting consistency with the measured draft step and round, not causal closure. Its
   transfer term now uses intc's RECORDED J-probe gather wait (90.464 ms median over 12 probe rounds; bytes 2,023 MB
   per round over the earlier steady set; scope and non-additivity in ledger 16.8). The earlier draft-only ceiling is
   conditional (ledger 16.5).
6. Yield (review 5): on the rates' own round set, ACCEPTED per request per round is 2.617 / 2.521 / 2.417 and COMMITTED
   2.642 / 2.555 / 2.451 (inta / intc / intd). The earlier 2.807 (median x median / B) is withdrawn. Counterfactuals use
   one population and ratio-of-sums; phase vectors are not all serialized, so phase-based counterfactuals are
   approximate. The record field accepted_per_round_steady_mean covers one extra round (capture round 1); the ST owner
   fixes the population in the cells.

## 2. Component results (MEASURED; registered bands in the run records)

| owner | change | C128 B216 | C73 B318 | verdict |
|---|---|---|---|---|
| J (area 1) | F1 joint pooling compiled; F2 draft pooling compiled (+ the same fix on SH, SHJ1) | x1.052 (joint -11.1, draft -5.0 ms) | x1.0515 with F6 (joint -17.8, draft -8.0 ms) | CONFIRMED (j1a, j1b) |
| J | F3 residency inside the CUDA graphs | x1.013 (joint -2.1, draft -1.8 ms) | not built (keep arm; about 0 expected) | CONFIRMED (j1a) |
| J | F6 locate without map sorts | in INT | in j1b (model 1.9 ms) | not separated |
| J | fork attention: matched microbench j1f | FORK / NS1R1 per byte 1.13 / 0.98; 4-row tile +57-60 % over one read of the same union pages | | OPEN: source findings (no finite-value check in num_err; the FLASH controls time an output copy_, FORK does not); fix in progress, re-time needed before any per-byte optimality claim |
| ST (area 3) | ST1 commit in one CUDA graph, row moves | commit 6.08 -> 3.45 ms, x1.0067 | 8.41 -> 4.09 ms, x1.0083 | C128 REFUTED low, C73 CONFIRMED |
| ST | ST2 twin advance per layer | joint -2.57 ms, x1.0084 | joint -2.24 ms, x1.0051 | CONFIRMED |
| ST | ST3 keep-fill compiled | — (no keep fill) | joint -5.45 ms, x1.0109 (intb) | CONFIRMED |
| ST | ST4 draft residency compiled | in inta | in intd (draft -9.67 ms with F2) | CONFIRMED inside INT |
| ST | recovery / yield (rows n, Y) | -2.2 % | -3.0 % | explained by decision (a); option below |
| BF (area 2) | draft-issued prefetch into the kept store | diagnostic only | b1 barrier x1.0085 (3 MUST failures: exit 3, sacct FAILED 3:0); b2 batched 0.953; b3 pipelined x1.0049 | break-even; OPEN: the actual blocking call of the worker, ready-only snapshot batching, transport; in progress |
| LEAD | X1 concurrency screen (repaired after review 3) | the tested fixed-shape stream combinations of dense and fork do not overlap; the copy engine does (0.8-3.4 %) | | MEASURED (scope: the tested combinations) |
| LEAD | INT composition | x1.090 (inta) | x1.080 (intd); x1.067 with BF1 (intb) | CONFIRMED; INT-ADDITIVITY not tested (REPORT, parts {}, product None) |
| LEAD | the anchor C81 B287 | | | intc: x1.080; 3 baseline-relative bands REFUTED |

## 3. Open items and human decisions

- Human decisions: the FAST tie rule (target correctness, ledger 16.3); the verify-K-1 restart (+1.9 / +2.8 %
  committed tokens per round, MODEL; reverses decision (a) and needs a weaker correctness gate); sizing the kept store to
  its measured peak (+11 requests at C81, MODEL; needs an overflow fallback).
- In progress (source + CPU; GPU validation needs an existing holder, none is live): BF owner: the fetch worker's actual
  blocking call and the largest avoidable readiness / host cost, with useful-arrival deadline and draft-contention
  counters. J owner: fail-closed finite checks with planted NaN / Inf controls; kernel vs output-copy timing in j1f.
- Not measured: physical HBM bytes of a live pass (ncu, X2); SHJ1 memory; the PHASE-SUM gate of the profiler cells
  failed on one pass in each of r7pa / r7pb (diagnosis open; the marks were not used).
- Review 3 (X1 repair milestone): DISPOSITION_REVIEW3_2026-10-31.md; code 2d3e57f .. 1b74406 on r2-recon7.
- Review 4 (2026-11-02): ledger section 16; the inta / intc / intd summaries are in the packet as hash-linked chunks.
