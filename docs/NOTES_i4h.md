# I4H: the draft-step graph, per-layer CUDA graphs over the one-row passes (arm L4PTFDGD)

Written 2026-10-29T17:37:03Z (`date -u`). Style: ASD-STE100. Labels: MEASURED (in a record), DERIVED (hand arithmetic
on MEASURED values, operands shown), CODE (read in the source at file:line), INFERENCE (my estimate; not measured),
MODEL (needs an assumption; stated).

Tree `/mnt/beegfs/ojerbi/nosi-r2-i4h`, branch `r2-i4h` = I4G `d7997e6` + this rung. Abbreviations: ST =
`nosi/nosi/spec_tentative.py`, RE = `nosi/nosi/ragged_engine.py`, LN = `benchmarks/Efficiency/r2_i4a_lean.py`, GR =
`benchmarks/Efficiency/r2_i4g_graph.py`, DG = `benchmarks/Efficiency/r2_i4h_graph.py`, CELL =
`benchmarks/Efficiency/r2_i4h_cell.py`, KS = `/mnt/beegfs/ojerbi/kstep-cpu-runs`. Basis: KS/analysis/plan/LEDGER_rev6.md
(R6.4 item 10, R6.6 E1), KS/analysis/plan/NOTES_graph_mean_gap.md (L61), docs/NOTES_i4g.md.

Files: DG, CELL, `benchmarks/Efficiency/r2_i4h_step.sh`, `benchmarks/Efficiency/tests_cpu/i4h_world.py`,
`tests_cpu/test_i4h.py`, this file; one declaration block in `test_wsa.py`, `test_wsb.py`, `test_wsc.py`, `test_i2.py`,
`test_i3.py`, `test_i4a.py`, `test_i4g.py`; in `test_i4g.py` also ONE line of `test_i4g_gate0_files` (`I4G_LATER`: its
name scan skips CELL and DG, which name the I4G modules), as I4G did in `test_i4a.py`.

NO NEW KERNEL and no kernel edit: `torch.cuda.CUDAGraph` (capture_begin / capture_end on a side stream, the steps of
`torch.cuda.graph`), the shipped kernels and torch ops. DG imports torch and time only. No file of `nosi/` and no I4A or
I4G file changes (Gate 0).

## 0. Answer in six lines

1. ESTIMATE FIRST (section 1): the capturable device-gap + host time of the draft phase is **2.7-5.2 ms per round at C 128
   B 216** and **1.9-4.4 ms at C 73 B 318** (busy basis), **4.8-5.1 / 5.3 ms** (I4G per-op transfer basis). C 128 is
   above the 2 ms stop line on both bases, so I built the arm.
2. BUILT: L4PTFDGD = L4PTFDG + two CUDA graphs per layer for every one-row pass (A: up to the residency; B: after it). The
   residency stays eager (spec_tentative.py:461-466: the host stamp). 64 graphs per key, one shared pool with the joint
   graphs.
3. CPU: L4PTFDGD = L4PTFD bit for bit in three worlds (timing B 8, check B 8, timing B 2 rowgroup); the replayed one-row
   pass has 99 eager aten launches against L4PTFD's 2,702 (CPU world, L 2; tests_cpu/test_i4h.py).
4. L61 FIXED in CELL: the capture rounds of every graph arm are reported and excluded from EVERY spec arm's steady set.
5. MODEL (section 5): draft saving C 128 about 4 ms (2.0-6.5), C 73 about 3.5 ms (1.5-6.0); L4PTFDGD / L4PTFDG 1.004-1.020
   (C 128), 1.000-1.012 (C 73); ratio to SH 0.536-0.545 / 0.605-0.612.
6. GPU only: capture legality of the draft ops (cuBLAS GEMMs on the side stream, flash split-KV with lse, stage 1, the
   fused twin, the inlined selection), the replay's bits (EQF on i4hm), the pool size and the timing.

## 1. The capturable estimate (before the build)

Sources: `/mnt/beegfs/ojerbi/r2-runs/spec/i4c_c3a_2195575_i4ca/` and `/mnt/beegfs/ojerbi/r2-runs/spec/i4c_c4b_2195576_i4cb/`
(summary.json `profile.draft`, `steady_draft_ref`; KS/cpu-logs/ana_rev6_s2.txt lines 12-13, 195, 286-287, 469), the draft
traces `trace_L4PTFDP_g96_draft0/1.json.gz` (unlabelled, the clean ones) and `g112` (labelled), my CPU split of the traces
by kernel name (script KS/tmp_sc/i4h_ana/draftgap.py, srun on amd; log KS/cpu-logs/i4h_ana_draftgap.txt). The split: the
residency span of a layer starts at `diff_kernel` and ends 5 ops after `flash_h2d_from_mask_bias_kernel` (LN:196-224:
masked_fill_, ~tent, &, sum; the `+=` falls to "rest": 32 ops per pass, INFERENCE: negligible); ops launched by a
cudaGraphLaunch are "graphed" (the shipped selection graph, RE:346-370); the rest is capturable. The gap before an op is
charged to that op (as KS/tmp_sc/i4g_ana/gapsplit.py).

| item | C 128 B 216 (i4ca) | C 73 B 318 (i4cb) | label |
|---|---|---|---|
| steady draft phase (2 one-row passes, L4PTFDP REF) | 143.03 ms | 189.278 ms | MEASURED |
| profiled busy (rounds 96 / 112) | 136.855 / 137.819 ms | 184.920 / 185.029 ms | MEASURED |
| busy / steady (median) | 0.9602 | 0.9773 | MEASURED |
| not busy per round | 143.03 x 0.0398 = 5.69 ms | 189.278 x 0.0227 = 4.30 ms | DERIVED |
| ops per pass: graphed / residency / rest | 192 / 640 / 2,035 (of 2,867) | 192 / 640 / 2,035 | MEASURED (split) |
| profiled g96 draft0: idle, host-exposed, device-side | 14.240, 10.796, 3.444 ms | 5.969, 2.907, 3.062 ms | MEASURED (split) |
| g96 draft0 "rest": gap (host-exposed) | 10.441 (7.898) ms | 4.672 (2.388) ms | MEASURED (split) |
| g96 draft0 residency: gap (host-exposed) | 3.697 (2.884) ms | 1.208 (0.520) ms | MEASURED (split) |
| capturable share of the host-exposed idle (draft0 / draft1) | 7.898 / 10.796 = 0.732; 8.587 / 11.707 = 0.733 | 2.388 / 2.907 = 0.821; 2.454 / 3.000 = 0.818 | DERIVED |
| device-gap term of an unprofiled eager pass (I4G section 1: 0.350 us per op, M-P3 SH) | 0.350 x 2,867 x 2 = 2.01 ms | 2.01 ms | DERIVED (INFERENCE: transfer) |
| host exposure per round | 5.69 - 2.01 = 3.68 ms | 4.30 - 2.01 = 2.29 ms | DERIVED |
| BASIS A capturable host exposure | 3.68 x 0.733 = 2.70 ms | 2.29 x 0.82 = 1.88 ms | DERIVED |
| BASIS A device-side gap a graph may remove (I4G's 0.350 + 0.270 us per op, capturable ops) | 0.620 x 2,035 x 2 = 0-2.52 ms | 0-2.52 ms | DERIVED (upper) |
| **BASIS A total** | **2.70-5.22 ms** | **1.88-4.40 ms** | DERIVED |
| I4G measured joint saving / captured ops per joint pass (5,896) | 6.962 / 7.364 ms: 1.181 / 1.249 us per op | 7.726 ms: 1.310 us per op | MEASURED / DERIVED |
| **BASIS B (I4G per-op transfer) x 2 x 2,035 ops** | **4.81 / 5.08 ms** | **5.33 ms** | MODEL (transfer) |

Reading. At C 73 the joint pass was 0.996 busy and the joint graphs still removed 7.7 ms (LEDGER_rev6.md R6.1 L3), so the
busy basis is a floor. C 128 is above 2 ms on both bases; C 73 is 1.9 ms on the floor and 5.3 ms on the transfer. Not
"under about 2 ms at both batches": I built the arm.

## 2. The design (CODE)

What is captured, per layer l and one-row pass (DG:1-62, the module docstring):
- A(l) (DG:323-330 begin, DG:332-344 end): layer 0 only the embedding (DG:312, = ST:308-309); then ST:318-362 up to the
  residency: the input norm, the QKV GEMM, nosa_linear, rope, the twin step and the lengths (ST:338-352), stage 1 and the
  selection (ST:353-358), and LN:145-159 (layer 0: the hoisted tensors LN:105-127; hasq, the injection, ti).
- EAGER (DG:345-348, DG:388): the residency LN:196-224 on A(l)'s outputs (its host count, the shipped kernels, the host
  stamp, the gathers, the meter). Its skip mask is copied into ONE static mask (DG:346-348).
- B(l) (DG:354 begin, DG:357-371 end): LN:163-194 (the buffer write, layer 0: tpos, the device counters and call B's masks;
  call A, call B, the merge) and ST:363-369 (the output projection, the post norm, the MLP, the residual adds).
- The final norm and the LM head stay eager (DG:296-302: one large GEMM).

THE SPLIT INSIDE ONE CALL (DG:304-321): the capture of A(l) begins before TentState._layer(l) runs (DG:314); an instance
attribute `_resid_lean` = DG:332 `_dstub` replaces the residency: it ends A(l), replays it, runs the REAL residency
eagerly, copies the mask and begins B(l); B(l) ends when the layer returns (DG:315). So the captured Python is the shipped
L4PTFD code, line for line. The capture executes nothing; every graph is replayed right after its capture.

THE SELECTION GRAPH (DG:155-167, DG:326-328): RE._select replays the shipped pooling / top-k graph (RE:360-369). No graph
launch is captured: during A(l)'s capture rs.graphs[l] is an `_Inline` whose replay() calls RE._pool_topk on the tensors of
RE:355-356. The same kernels on the same inputs.

THE KEY (DG:223-224): (T.checking, plan.snap): the Python branches inside the captured code (LN:248-250 the check-only
counters; ST:350-351 the counter snapshot). Per key (DG:266-302): the first pass is the WARM pass, eager on the static
inputs, on the capture side stream (DG:279-284: Triton compiles outside a capture, cuBLAS gets the side stream's
workspace); the second pass CAPTURES 2 L graphs; every later pass replays (DG:382-392). EAGER passes (DG:205-221):
T.hooks, T.rec, plan.replay_rows, plan.deferred, plan.lin is not rs.ops.linear, a lean control, no capable backend.
DEAGER_OK = () (DG:65): a cell accepts none.

STATIC INPUTS (DG:226-247): plan.ids (B, 1), plan.pos (B, 1), the act / verify / draft rows; zeros at allocation; copy_
before every pass. Every other tensor that a graph reads is persistent or made by an earlier graph of the same pass.

HOST COUNTS (DG:69-84, DG:249-263): the capture measures the counter dicts (T.calls, T.lean, rs.replays / captures,
vs.calls, vs.stat_n) before and after each captured section; every replay adds that delta. `expected_delta` (DG:69-73):
A(l) adds lean["layers"] 1 (LN:100) and rs.replays 1 (RE:370, with rs.ops.graph); B(l) adds nothing. A capture with
another delta goes to `delta_bad` (DG:366-368) and fails L4PTFDGD-DGRAPH.

HELD TENSORS (in the pool for the arm): per layer ti and inj (the eager residency's inputs at a replay), the pass's
hoisted tensors (A(0)), the last layer's output, the snapshot (snap keys). Every other tensor of a graph dies after the
capture of the graph that last reads it; the pool reuses it for later captures. This is safe because the replays run in
the capture order and nothing outside the pass reads it (INFERENCE from the torch private-pool rule; GPU only: EQF).

THE BACKEND (DG:86-152, `make_backend(GR)`): a subclass of GR.CudaGraphs, so the joint graphs' capture() and the draft
graphs share ONE pool. begin(): synchronize, the side stream, `CUDAGraph.capture_begin(pool=...)`; end():
`capture_end()`, synchronize. No gc.collect / empty_cache per capture (torch.cuda.graph does both; I4G's 32 captures took
6.2-6.8 s with them, NG section 2).

L61 RECORD (DG:291, DG:394-399): the round (T.round) of every draft capture and of every joint capture.

Negative controls (tests only, DG:55-58, DG:64): stale_ids, no_counts, no_key, live_skip.

## 3. Gates (CELL)

MUST: C0_FORK; SH-RUN; per arm -RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN, -RESID, -ATTN1, -FILL1, -LEAN; i4hm also -PRESENT,
-VIS, -VB, -TAIL, -CHAIN, -IDX, -RA, -STRESS, -FSTRESS; for L4PTFDG and L4PTFDGD: -EQF (CELL:128-131, against L4PTFD:
committed tokens, check-round fingerprints, acceptance record, state checksums, bit for bit), -COUNTS
(r2_i4g_cell.counts_gate), -GRAPH (the joint graphs: r2_i4g_cell.graph_gate), -MEM (peak allocated with the prefill <=
84.0 GB); for L4PTFDGD **-DGRAPH** (CELL:134-153: 2 L graphs per captured key, one capture pass per key, a warm pass per
key seen, no `delta_bad`, eager reasons only in DEAGER_OK, warm + eager + capture + replay passes = the lean passes, 2 L
replays per capture or replay pass, at least one replay pass). REPORT: the phases and rates with and without the capture
rounds, the effective rate without the capture seconds (CELL:208-213), the joint and draft graph reports, the pool MB.

L61 FIX (CELL:156-205, CELL:445-447): `capture_rounds_x` reads each graph arm's capture rounds (L4PTFDGD: the T.round
record; L4PTFDG: its joint capture pass = the round when no one-row J slot ran, MEASURED j1_passes 0 in i4ga / i4gb);
`excluded_rounds` takes their union; `restrict` recomputes the rates, vs_sh, vs_strongest, phases_median, draft_step_ms
/ _sh and joint_sh_steps of EVERY spec arm (L4PTFD too) on r2_spec_cell.steady_rounds minus those rounds, and keeps the
old values as `rates_incl_capture` / `phases_median_incl_capture`. The same rounds leave every arm (the arms run the same
schedule, bit for bit). The older trees' files are not edited.

## 4. Cells (r2_i4h_step.sh)

| cell | C (pool) | B | steps | arms | check | STEP_MIN | mem | role |
|---|---|---|---|---|---|---|---|---|
| i4hm | 73 (10) | 8 | 72 | SH, L4PTFD, L4PTFDG, L4PTFDGD | every round | 40 | 100G | the miniature: every gate MUST; run first |
| i4ha | 128 (65) | 216 | 512 | SH, L4PTFD, L4PTFDG, L4PTFDGD | probe rounds | 110 | 240G | timing: the target |
| i4hb | 73 (10) | 318 | 512 | SH, L4PTFD, L4PTFDG, L4PTFDGD | probe rounds | 160 | 340G | timing: the device-bound batch |

STEP_MIN basis. MEASURED (sacct, KS/../r2-runs/spec/i4g_*.step/sacct.txt): i4gm (SH + 2 arms, B 8) 5:55 and 6:05; i4ga
(SH + L4PTFD + L4PTFDG) 36:25 and 36:45; i4gb 54:38 and 53:38. MODEL: total = s + n a (s: model load + SH; a: one spec arm
with its prefill). With s about 4 min (C 128) / 6 min (C 73) (INFERENCE): a = (36.4 - 4) / 2 = 16.2 min and (54.1 - 6) / 2
= 24.1 min. Three arms: 4 + 3 x 16.2 = 52.6 min (i4ha), 6 + 3 x 24.1 = 78.3 min (i4hb); the draft captures add seconds
(section 5). STEP_MIN 110 / 160 = 2.1x / 2.0x (I4G used 2.2x / 2.0x). i4hm: about 8-9 min against 40. mem as I4G (MaxRSS
3.0-3.1 GB; the arms free the host rows between them). i4ha / i4hb STOP while `PRED_STAMP_I4H` is None.

## 5. Registration keys (PRED_I4H_BY_B[216 | 318], PRED_STAMP_I4H) and the builder's MODEL

Every key is (low, high) except `I4H_eq` (bool); every phase and rate on the steady set without the capture rounds. None
gives REPORT. The MODEL is an input for the lead, not a registration. Inputs (MEASURED, LEDGER_rev6.md R6.2-R6.3):
L4PTFDG g3a draft 142.913, round 348.202, own-SH ratio 0.5342 (h4a 0.5345), L4PTFDG / L4PTFD 1.0191 (h4a 1.0197); g4b
draft 2 x 94.865 = 189.73, round 2 x 94.865 + (1.123 + 4 x 93.743) + 8.512 = 574.34 (DERIVED), L4PTFDG / L4PTFD 1.0134,
own-SH ratio 0.605.

| key | C 128 B 216 | C 73 B 318 | derivation |
|---|---|---|---|
| I4H_draft_saving (ms per round) | central 4.0; band (2.0, 6.5) | central 3.5; band (1.5, 6.0) | section 1: basis A 2.70-5.22 / 1.88-4.40, basis B 4.81-5.08 / 5.33; minus the added eager work: one mask copy per layer and 5 static copies per pass, 2 x 37 small copies per round x about 2.5 us = 0.2 ms (INFERENCE); the graph-to-residency boundaries: the host issues about 20 residency launches (about 0.1 ms) while the device runs A(l) (0.5-1.5 ms per layer): no new gap (INFERENCE) |
| I4H_draft_ratio | (0.955, 0.986) | (0.968, 0.992) | (142.913 - 6.5 .. 2.0) / 142.913; (189.73 - 6.0 .. 1.5) / 189.73 |
| I4H_joint_ratio | (0.99, 1.01) | (0.99, 1.01) | the joint pass is L4PTFDG's code; the shared pool only adds memory |
| I4H_vs_G | (1.004, 1.020) | (1.000, 1.012) | 348.202 / (348.202 - 2.0 .. 6.5) = 1.0058-1.0190; 574.34 / (574.34 - 1.5 .. 6.0) = 1.0026-1.0106; widened by the replicate spread (I4G 0.06 %) |
| I4H_vs_D | (1.023, 1.040) | (1.013, 1.026) | 1.0191 x (1.004 .. 1.020); 1.0134 x (1.000 .. 1.012) |
| I4H_ratio | (0.536, 0.545) | (0.605, 0.612) | 0.5342 x (1.004 .. 1.020); 0.605 x (1.000 .. 1.012) |
| I4H_eq | True | True | CPU: L4PTFDGD = L4PTFD bit for bit (test_l4ptfdgd_equals_l4ptfd); a replay runs the same kernels on the same inputs; GPU risk: section 9 |
| I4H_peak_gb | (83.85, 83.95) | (77.60, 77.80) | the prefill peak 83.891 / 77.685 GB (MEASURED i4ga / i4gb, L4PTFDG-MEM) is above the decode peak 82.567 / 76.944 GB + the draft pool (below) |
| I4H_pool_mb | (40, 400) | (60, 600) | the shared pool after the run: I4G's joint pool 14.68 / 25.17 MB (MEASURED) + per draft key held ti + inj 32 x (8 x B x 64 x 9 B) = 31.9 / 46.9 MB, the last output 1.8 / 2.6 MB, + the largest live set of one capture (gate_up 12.4 MB, silu 6.2 MB, the split-KV accumulators about 14 MB, the merge in float32 about 18 MB at B 216; x 1.47 at B 318), reused across captures; two keys in the timing cells (MEASURED i4ga / i4gb: mixed_steps 0, j1_passes 0: keys (F, F) and the probe key (T, F)) (INFERENCE) |
| I4H_capture_s | (0.5, 30) | (0.5, 30) | 2 keys x 64 captures, each two syncs + about 40 recorded launches, 5-100 ms (INFERENCE; no gc.collect / empty_cache) |

The capture rounds in the timing cells (CODE + MEASURED call counts): the first key captures in the prelude (round -1,
outside the steady set already), the probe key in round 16 (a probe round, outside already), the joint graphs in round 1
(I4G, MEASURED `capture rounds (round, joint ms) [(1, 6367.092)]`): the L61 fix removes round 1 from every arm.

## 6. Memory

C 128 B 216 has 84.25 - 83.891 = 0.36 GB (0.359) under the usable ceiling with the prefill peak (MEASURED i4ga), and the
decode peak is 82.567 GB (MEASURED L4PTFDG). The draft graphs live in the decode only (made in the prelude after the
arm's prefill); their pool (section 5: 40-400 MB) plus the decode peak stays below the prefill peak (82.567 + 0.40 =
82.97 < 83.891), so the MEM value stays the prefill peak (INFERENCE; MEM is MUST at 84.0 GB with the prefill). The side
stream of the warm passes caches its freed blocks for that stream (reserved, not allocated; about one pass of
temporaries, INFERENCE). The arm is the last of the job; detach() drops its graphs and the pool handle.

## 7. What the CPU establishes and what only the GPU can establish

CPU (FakeGraphsD, `tests_cpu/i4h_world.py`): the split capture executes nothing, a replay re-executes the recorded ops on
the captured tensor objects, the stand-ins that read lengths on the host (the fake attention, the fake stage 1) are
result nodes, and a host read outside a node raises. With it: L4PTFDGD = L4PTFD bit for bit in a timing world (two keys,
the probe key between replays), a check world (every check inside the graphs) and B 2 rowgroup (three keys incl. the
snap key of one-row J slots); the counts; the gates; the plants; the eager launch count; no host read in DG.

GPU only (i4hm first): that every op of A(l) and B(l) is legal under stream capture on this stack (torch 2.6.0+cu124:
cuBLAS GEMMs on the side stream, nosa_linear, rope, the fused twin (Triton), stage 1, pooling / top-k inlined, flash
split-KV with return_softmax_lse, the fused body kernels), that a replay gives the eager bits on the device (cuBLAS picks
its algorithm with the same workspace size on every stream: INFERENCE), the pool's size, the decode peak, the capture
time and the timing. FakeGraphsD does not model streams, the capture's allocator or kernel launch legality.

## 8. CPU tests

`benchmarks/Efficiency/tests_cpu/test_i4h.py` (its own srun): Gate 0; the stand-in's split semantics with controls (a
result node, a host read, abort); L4PTFDGD = L4PTFD (timing, check, B 2 rowgroup) with the cell's EQF / COUNTS / GRAPH /
DGRAPH / LEAN gates on the CPU evidence; the plants (stale_ids -> EQF; no_counts -> COUNTS and LEAN; no_key -> COUNTS;
live_skip -> EQF); the eager launch count; no host read in DG (SyncTrap); the cell's arguments, pure parts (incl. the
L61 fix on a synthetic record), registration state, step script (DRYRUN) and these notes. Also run: test_i4g.py,
test_i4a.py, test_i3.py, test_i2.py, test_wsa.py, test_wsb.py, test_wsc.py (their declaration blocks).

## 9. Risks

1. A capture may fail on the GPU (an op that synchronizes or allocates illegally under capture, e.g. a lazy init on the
   side stream that the warm pass did not trigger): L4PTFDGD-RUN fails; the arm is the last of the job, so the other
   arms keep their records. i4hm first.
2. EQF on the GPU: a cuBLAS or flash heuristic that picks another algorithm under capture or on the side stream would
   change bits (the CPU cannot show this).
3. The pool reuse rule (section 2, HELD TENSORS) relies on the capture order = the replay order; a later edit that reads
   a graph's tensor outside its pass breaks it (EQF catches it).
4. The busy basis of section 1 transfers SH's 0.350 us device gap (M-P3, B 64); the per-op basis transfers the joint
   pass's per-op saving to smaller draft kernels. The true saving can sit outside both (the bands are wide).
5. The probe rounds keep the check key's graphs (two key sets): more pool memory, no timing effect (probe rounds are
   outside the steady set).
