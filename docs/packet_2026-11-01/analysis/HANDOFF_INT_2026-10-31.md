# HANDOFF INT (2026-10-31): the original formula term by term, the fixes, their measured effects, the corrected model

Written by the lead (FINAL, stamp 2026-11-01T00:12:44Z from `date -u`). Style: ASD-STE100. Ledger:
/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/LEDGER_rev7.md sections 14 (term rows), 15 (corrected model) and 15.6
(result). Status log: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/STATUS_LOG.md. Run records: R =
/mnt/beegfs/ojerbi/r2-runs/spec. CPU receipts: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/<tag>.txt + .receipt (command,
host, SLURM job, HEAD, status, source sha256, pytest exit).

## 1. The answer in short

1. All three owners carried their largest evidenced gap to a scoped, bit-exact change with a registered prediction and
   a measured effect, or to a quantitative explanation. Every GPU cell below passed its MUST gates (committed tokens
   bit-equal to the plain arm), except where the table says otherwise.
2. INTEGRATED results (committed tok/s, median of the steady rounds; vs the same-job shipped decode SH):
   - C128 B216 (inta): 2,096.1 = x1.090 over the morning's best arm (1,923.0, same job); 0.622x SH.
   - C81 B287, the anchor (intc): 1,854.8 = x1.080; 0.659x SH (2,816.3); round 434.4 ms vs the model's 232.48.
   - C73 B318 without BF1 (intd): 1,843.2 = x1.080; 0.716x SH (2,573.9). With the batched BF1 (intb): 1,821.0.
3. VERDICT: the best demonstrated specKV under the 84 GB cap is 2,096.1 tok/s = 0.615x the strongest cached NOSA
   (3,407.2, C128 whole-step graph). specKV does NOT beat the strongest cached NOSA with these kernels.
4. WHY, term by term: the original +17 % rests on two premises (ledger 15.1). (P1) A draft step is much cheaper than a
   baseline step: it holds at small caches (C73 0.67-0.71, the anchor 0.79) and fails at C128 (0.97). (P2) The
   verification work and the transfer are hidden by the two maxima: NOT available. Dense GEMMs and the attention do not
   overlap as streams (X1). The 4-row verification costs compute beyond the bytes (+57-60 %, j1f). Draft-issued
   prefetch at the measured coverage reaches only break-even (bf1b b3 x1.0049). The serial corrected model (15.4)
   predicts the measured anchor round within 1-6 %.

## 2. Component results (MEASURED; registered bands in the run records)

| owner | change | C128 B216 | C73 B318 | verdict |
|---|---|---|---|---|
| J (area 1) | F1 joint pooling compiled; F2 draft pooling compiled (+ the same fix on SH, SHJ1) | x1.052 (joint -11.1, draft -5.0 ms) | x1.0515 with F6 (joint -17.8, draft -8.0 ms) | CONFIRMED (j1a, j1b) |
| J | F3 residency inside the CUDA graphs | x1.013 (joint -2.1, draft -1.8 ms) | not built (keep arm; about 0 expected) | CONFIRMED (j1a) |
| J | F6 locate without map sorts | in INT | in j1b (model 1.9 ms) | not separated |
| J | fork attention: matched microbench j1f | the fork equals the baseline per byte (1.13 / 0.98); the 4-row tile costs +57-60 % over one read of the same union pages; 3D split slower | | quantitative explanation; no change warranted |
| ST (area 3) | ST1 commit in one CUDA graph, row moves | commit 6.08 -> 3.45 ms, x1.0067 | 8.41 -> 4.09 ms, x1.0083 | C128 REFUTED low (model missed the PCIe write-back and node floor), C73 CONFIRMED |
| ST | ST2 twin advance per layer | joint -2.57 ms, x1.0084 | joint -2.24 ms, x1.0051 | CONFIRMED |
| ST | ST3 keep-fill compiled | — (no keep fill) | joint -5.45 ms, x1.0109 (intb) | CONFIRMED |
| ST | ST4 draft residency compiled | in inta | in intd (draft -9.67 ms with F2) | CONFIRMED inside INT |
| ST | recovery / yield (rows n, Y) | -2.2 % Y | -3.0 % Y | quantitative explanation (decision (a)); option below |
| BF (area 2) | draft-issued prefetch into the kept store | diagnostic only | barrier x1.0085; batched 0.953; pipelined x1.0049 | REFUTED / break-even; limit = host worker time per epoch (311 ms vs 185 ms window); not adopted |
| LEAD | X1 concurrency screen (repaired after review 3) | dense and fork do not overlap as streams; the copy engine does (0.8-3.4 %) | | MEASURED |
| LEAD | INT composition | x1.090 (inta) | x1.080 (intd); x1.067 with BF1 (intb) | CONFIRMED (intb REFUTED low by BF1) |
| LEAD | the anchor C81 B287 | | | intc: x1.080, 0.659x SH, 0.50x Q_accepted |

## 3. Open items and human decisions

- Human decisions: the FAST tie rule (unchanged); the verify-K-1 restart (+1.9 / +2.8 % committed tokens per round,
  MODEL; reverses decision (a) and needs a weaker correctness gate); sizing the kept store to its measured peak (+11
  requests at C81, MODEL; needs an overflow fallback); an optional transport change in gilfree_pack.cpp (BF1 worker time).
- Not measured: physical HBM bytes of a live pass (ncu, X2); the PHASE-SUM gate of the profiler cells failed on one
  pass in each full-size run r7pa / r7pb (diagnosis open; the marks were not used).
- Routes outside scope: fused in-kernel overlap of verification and drafts (new kernels).
- Review 3 (X1 repair milestone): DISPOSITION_REVIEW3_2026-10-31.md; code 2d3e57f .. 1b74406 on r2-recon7; receipts
  recovered (RECON7_RECEIPTS_RECOVERED.md); ledger claims relabelled (section 13); provenance bundle
  (tmp_sc/recon7_prov).
