# The human's reconciliation request (received 2026-10-30, about 09:50Z), recorded verbatim in substance

## Ask
- Share the calculations. Research comparable implementations online (primary papers, author code, official docs).
  Debug EVERY component against the ORIGINAL calculation. Improvements over our preceding implementation and agreement
  with our newer optimization predictions do NOT answer the question.
- Target: realize the modeled mechanisms where feasible. If a model assumption is wrong or an omitted cost is necessary,
  quantify it with evidence and correct the model. Do not tune gates, discard failures, or force agreement.
- We cannot yet certify that the other components match either.
- Fold this into revision 7 / the consolidated handoff. Do not mark the component comparison complete based only on the
  incremental rung predictions.
- This message does NOT authorize another allocation, GPU experiment, blind retry or threshold relaxation. Routine
  allocated-CPU diagnosis stays with the lead. Existing resource / execution limits and GPU-protocol review stay in force.

## ORIGINAL CALCULATION (sent October 21; numerical reference)
16K context, C81 / K3 / B287, assumed lag 2, retained proposal yield Y3 and historical cohort rejection p0. Modeled
physical peak 83.7655 GB under 84 GB; never an allocation proof. Old ordinary reference 3160.822684 tok/s at C128 / B216.
- Resident draft proxy td = 68.9864166 ms. GEMM(B) = 22.6567976 ms; GEMM(B*K) = 58.3739739 ms.
- Verifier union U = 69.1629464 groups; ready-selected overlap I = 60.1939174;
  weight fraction f = 15.768289 / (15.768289 + B * 0.132637) = 0.2928997375.
- extra verifier HBM proxy = td * (1 - f) * (U - I) / 63 = 6.94463564 ms. This is NOT verifier wall time.
- Cohort HBM proxy = 3 * td + extra = 213.9038854 ms.
- Cohort dense service = 3 * GEMM(B) + GEMM(B*K) = 126.3443667 ms.
- Useful transfer rate 10.72418445 GB/s; transfer bytes 6.46582857 decimal MB / request / cohort; transfer service
  173.0381278 ms. Transport interference 8.68458712 %.
- GPU proxy = max(HBM, dense) * (1 + interference) = 232.4805547 ms; interval = max(GPU, transfer) = 232.4805547 ms;
  throughput = 1000 * 287 * 3 / interval = 3703.5355537 accepted-draft tok/s, +17.16999 % over the old reference.
- Assumes actual shared weight / ready-KV reads and ideal aggregate overlap. The resident proxy is not an isolated HBM
  counter; selection, decision / recovery, real deadlines and changed live selections need explicit accounting.
  Correction / bonus tokens and their costs were omitted. For nonzero rejection the old n = K + p * lag must be corrected
  to K + p * (1 + lag) under that deferral clock. Keep strongest cached NOSA, whole-run / steady timing windows and
  committed-token accounting distinct.

## RETROSPECTIVE JOINT COMPARISON (root + independent arithmetic reviewer; historical trace / calibration rows)
- C128 / B216: td 54.8137 ms; extra verifier HBM 5.0333; joint HBM proxy 59.8470; draft + verifier dense 63.4452;
  original interference 10.5766 %; inferred ideal joint-service budget 70.1555 ms. Our I3: 1709.8 tok/s, 0.5128 own-SH,
  3.201 SH joint -> SH 64.7823 ms, joint 207.368 ms: about 3.0x that budget.
- C73 / B318: td 75.2948; extra 7.8025; joint HBM proxy 83.0973; dense 89.9657; interference 7.8585 %; budget 97.0356 ms.
  Our 1525.8 tok/s, 0.5925 own-SH, 3.094 SH joint -> 382.066 ms: about 3.9x (UI separately 382 ms).
- Budget formula: max(td + extra_verifier_HBM, GEMM(B) + GEMM(B*K)) * (1 + interference).
  Conversion: SH_ms = 1000 * B * (I3/SH) / I3_tok_s; joint_ms = (joint/SH) * SH_ms.
- SCOPE: the original model uses a max over the whole cohort and did not register a standalone joint wall-time
  prediction. These are reconstructed ideal budgets, not kernel-efficiency percentages, proven lower bounds or promised
  speedups. The measured joint region may include exposed fetch / selection / residency / synchronization absent or
  hidden in that budget. Confirm its boundaries and use exact raw values / current pins. The gaps of about 137 / 285 ms
  are quantities to explain, not proof of a bug or of a failed design.

## REQUESTED DIAGNOSIS FOR EVERY PART
One table: original modeled work / cost at the original anchor AND at the measured configuration | actual work / cost at
the same boundaries | gap | cause with evidence | correction or quantified model revision | remaining unknown.
Cover: standalone drafting / dense work; joint dense / selection / attention and actual shared reads; packing /
transfers / readiness; retention / state / physical capacity; schedule / yield / recovery. Reconcile query positions,
invocation counts, bytes, dispatch and dependencies BEFORE interpreting milliseconds. Do not silently substitute the
implemented schedule for the intended K3 / deferral design. Explain the total with per-round spans and overlap; the
-12.7 ms residual from adding label medians is not a valid elapsed-time reconciliation.
- The claim "about two model rows per kept token makes 1.0x impossible" is NOT established by a busy GPU: the original
  dense formula already charges the verifier rows. Quantify actual dense time, reuse, row / pass counts and dependencies
  against that formula. Extra rows do not by themselves establish proportional elapsed cost; no-kernel limitations are
  not evidence of a limit on the proposed design.
- Online: multi-query speculative verification, paged / shared-KV attention (FlashAttention / Flash-Decoding,
  FlashInfer, vLLM), fused sparse selection / residency, graph-safe execution, producer / copy / consumer overlap.
  Compare compatible shapes, GQA, sparse masks, per-query bias, page layout and numerical contract; link mechanisms;
  assess whether they apply. A faster dense / prefill benchmark is not automatically a compatible NOSA solution. GPU busy
  percentage and a launch-settings sweep do not establish an architectural compute limit.
- Start from existing raw traces, receipts and source. For each unexplained gap distinguish: extra / wrong work,
  inefficient execution, serialization / contention, changed live workload, false model assumption. Fix scoped
  implementation defects with meaningful guards; if evidence cannot distinguish causes, specify the smallest concrete
  discriminator. Return one coherent source / evidence handoff with this reconciliation, measured residual, preserved
  failures and validated lessons.

## Prior reconciliation material (October 21, pre-R2 stage tick; context only)
/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/reconciliation-17pct.md, forecast-oct12-17pct.md, recon-rows-2-3.md,
recon-rows-5-6.md, verifier-working-set.md, recon-bytes.md, recon-draft-tick.md, recon-row-1b-alloc.md.
