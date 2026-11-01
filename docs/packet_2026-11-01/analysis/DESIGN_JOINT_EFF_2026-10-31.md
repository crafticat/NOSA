# DESIGN: the avoidable implementation costs of the specKV joint verify pass (and its draft step)

Written 2026-10-31T11:42:21Z (`date -u`; first read 11:32:38Z). Writer: the J1 design / build agent (AREA 1 "joint
execution, attention and memory reads" after the lead's scope extension of 2026-10-31). Read-only brief: no code
change, no job, no commit FOR THIS FILE. Style: ASD-STE100.

Labels. MEASURED = a value in a run record or a trace log. DERIVED = arithmetic on MEASURED values (operands shown).
MODEL = needs an assumption (stated). CODE = read in source at file:line. CITED = from another file (file:line).

Abbreviations (all absolute):
- PL = /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan. LG = /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs.
- L7 = PL/LEDGER_rev7.md. R2 = PL/RECON7_R2_joint.md. F5 = PL/SCORING_fia5.md. F6 = PL/SCORING_fia6.md.
  NG = PL/NOTES_graph_mean_gap.md. TR = LG/recon7_r2_s2.txt (R2's Kineto per-kernel tables, I3 joint path, one
  profiled pass per window).
- T = /mnt/beegfs/ojerbi/nosi-r2-recon7 (cd0f0c5). N = T/nosi/nosi. E = T/benchmarks/Efficiency.
  ST = N/spec_tentative.py, STF = N/spec_tentative_fork.py, VR = N/verify_rows.py, RE = N/ragged_engine.py,
  MP = N/max_pooling_per_request.py, S1 = N/stage1_seqused.py, TF = N/r2_twin_fused.py, FK =
  E/fork_vllm_unified_attention.py, IG = E/r2_i4g_graph.py, IH = E/r2_i4h_graph.py, DCC = E/r2_i4dc_cell.py,
  CD = E/r2_i4dc_compile.py. GS = /mnt/beegfs/ojerbi/nosi-r2-graphstep @ 0159e82 nosi/nosi/graph_step.py (the
  strongest baseline's whole-step graph).
- Configs: C128 = C128 pool 65 / B216; C73 = C73 pool 10 / B318. BEST = L4PTFDGUDCD (C128) / L4PTFDGUDKWCD (C73).
  SH = the plain cached NOSA (one-row decode). STRONG = the whole-step-graph SH, 3,426.2 tok/s at C128 (L7:34) = 63.04 ms
  per step (DERIVED 1000 x 216 / 3,426.2).

## 0. Answer (ranked fix list, largest saving per hour first)

| # | fix (existing ops only) | component / area | expected saving, ms per ROUND, C128 / C73 | label | correctness gate | cost |
|---|---|---|---|---|---|---|
| F1 | NOSA's block pooling of the JOINT pass's 4B selection rows as ONE torch.compile'd gather + max (MP:35-60's arithmetic, already the torch twin of the kernel), reading the per-request compressed cis by request index (no (H, B, R, Lk) copy). The shipped Triton kernel launches ONE 128-thread program PER OUTPUT ELEMENT: grid (2, 864, 261) = 451,008 programs per layer | selection / AREA 1 | joint 12.28 -> about 0.6 = **-11.6** / 18.10 -> about 1.0 = **-17.1** | kernel ms MEASURED (TR:21, :615); compiled time MODEL (8 MB per layer, 20-30 us) | max is exact: pq / pc bitwise = the shipped kernel (probe gate on live inputs, every check round, every layer) + committed tokens bitwise = L4PTFD (EQF) and = the BEST arm | 0.5 day |
| F2 | the same compiled pooling in the ONE-ROW (draft) passes (inside the draft graphs A(l) of IH) | draft step / AREA 1 | 2 x (3.14 - about 0.3) = **-5.7** / 2 x (4.71 - about 0.4) = **-8.6** | MEASURED kernel (TR:87, :678); MODEL compiled | as F1, plus the DGRAPH / COMPILE gates (compile before the first capture) | 0.5 day |
| F3 | one CUDA graph per joint layer (pre-section + residency + section + MLP) and one per draft pass, with the strongest baseline's SHIFTED-AGE LRU frame (GS:17-19, :146-154: constant stamp, ages - topk per step by a torch op) in place of the host stamp that IG:12-15 / IH:15-17 call a kernel edit | launch / AREA 1 | joint non-busy about 6.3 / about 0 (BEST, MODEL labels, R2:300); draft non-busy unknown (BEST has no trace) | MODEL | integer ages: decisions bitwise; EQF + a stamp-frame equality test (GS's shift_in / shift_out round trip) | 1-2 days |
| F4 | fork attention, 3D split path (FK:1028-1070, an EXISTING launcher option: seq_threshold_3D + segment buffers + reduce_segments) for wave balance only | attention / AREA 1 | 0 .. 3.7 / 0 .. 5.2 (at most 10 % of 37.1 / 52.2) | MODEL; MEASURE FIRST (timing sweep) | float order changes: fp64 tolerance gate + committed-token equality REPORT (near-tie class), not bitwise | 0.5 day + a sweep |
| F5 | joint selection small ops: mask zero + scatter + masked_fill into the compiled function of F1 (the two top-k stay eager) | selection / AREA 1 | about 1 / 1.5 (SMALL 2.30 ms in the select span, TR:58) | MODEL | bitwise (bool / inf fills) | with F1 |
| A3 | (AREA 3, named only) twin: R step launches -> TF:242 advance_fused (one launch, per-row counters); chain image CB1: alias the tentative buffer into the arena (no copy) | tentative state | twin 3.99 -> about 1; CB1 about 1.5 of chain 3.29 (C128, MODEL labels R2:297) | MODEL | bitwise (TF is a declared bitwise twin) | AREA 3 |
| A2 | (AREA 2, named only) H2D gathers out of the compute stream | transfer | 21.3 / 160.1 serial ms | MEASURED | - | AREA 2 |

Selection and the fork are NECESSARY work in the algorithm (each verify row needs its own selection; each row needs its
attention). Inside them the avoidable part is the POOLING KERNEL'S LAUNCH GRID (F1, F2: about 17 / 26 ms per round) and
small launch / copy costs. The fork's rate gap is mostly STRUCTURAL for this kernel design (section 4): the launch
configurations were swept in fia5 (11 configurations + the bias dtype; the current FV6 is the best); the remaining
existing option (3D) only balances waves. FAIRNESS: SH runs the same pooling kernel once per step (3.14 ms at B216,
TR:87). F2's fix also speeds STRONG (MODEL: 63.04 -> about 60.2 ms per step); the ledger must report the speculative
arm against an SH that takes the same fix (matched-memory rule).

LEAD INPUT (2026-10-31, X1, run /mnt/beegfs/ojerbi/r2-runs/spec/recon7_x1r3_2195630_r7s, all MUST gates pass): dense
GEMMs and the fork do NOT overlap on two streams (C128 A|F 97.8 vs serial 100.2 ms; C73 132.7 vs 136.0). So every ms
saved in selection, the fork or small kernels is a direct round saving, and the request-group GEMM split is not
bit-equal (cuBLAS at M 2B vs 4B). The fork alone on X1's synthetic harness: 37.5 / 57.3 ms per pass (= production
37.1 / 52.2), so F4's sweep can run on that harness.

FIRST GPU EXPERIMENT (section 5): the J1 cell at C128 B216, arms L4PTFD, L4PTFDGUDC, L4PTFDGUDCD, L4PTFDGUDCDJ1 (BEST +
F1). About 15-20 min. Decision: J1 joint median - BEST joint median in [-13.5, -9.5] ms; PEQ (pooling bits) 0 differing
values; EQF bitwise.

## 1. Per component

### 1.1 Verify-row SELECTION (gap +17.83 / +24.43 ms per joint pass above the model, R2:263)

What the code does per joint pass (CODE; per layer, 32 layers):
- ST:360 calls VR:413-432 `_select_rows` for E = 4B rows (R = 4). It is ALREADY the batched form of the baseline's op:
  one stage-1 call (S1:69, `flash_fwd_splitkv_stage1_kernel`, grid (1, E, 2)), one copy of the compressed cis to the
  R rows (VR:422, (H, B, R, Lk)), one pooling call (MP:149, grid (H, E, M) = (2, 864, 261)), top-k, mask zero, scatter,
  masked_fill, top-k (VR:427-431). 7-9 launches per layer, eager in BEST (IG:11-25 captures only the section after the
  residency).
- SH does the same ops for B rows per step (RE:409-415, RE:333-343), captured per layer (RE:346-370) or in the whole-step
  graph (GS).

MEASURED device ms per pass (TR:16-58 C128, TR:611-633 C73; I3 joint path = BEST's selection, R2:295 "unchanged"):

| op | joint C128 (E 864) | one-row C128 (B 216) | joint C73 (E 1272) | one-row C73 (B 318) | per-row ratio joint / one-row |
|---|---|---|---|---|---|
| stage 1 | 9.82 | 4.51 | 13.54 | 6.93 | 0.54 / 0.49 (DERIVED: 9.82 / 4 / 4.51; 13.54 / 4 / 6.93): L2 serves the shared compressed keys in part |
| pooling `nosa_pool_kernel_per_request` | **12.28** (384 us per layer) | **3.14** (98 us) | **18.10** (566 us) | **4.71** (147 us) | 0.98 / 0.96: linear in programs |
| top-k (gatherTopK) | 2.91 | 0.93 | 4.12 | 1.29 | 0.78 / 0.80 |
| small ops in the span | 2.30 | 0.77 | - | - | - |
| total select span | 27.31 | 9.48 | 38.47 | 14.04 | 0.72 / 0.68 |

NECESSARY (the model omitted it): one selection per verify row (3B extra rows). The batched form already costs 0.72 /
0.68 of one-row per row (L2 hits in stage 1; larger top-k grids).
AVOIDABLE (implementation):
1. The pooling launch grid. One program per OUTPUT element (MP:94-97: program ids (h, b, m)), 128 threads each, 8
   loads and 1 store per program. Bytes per layer at E 864: scores 2 x 864 x 1152 x 2 B = 4.0 MB + cis about 0.9 MB +
   outputs 2 x 2 x 864 x 261 x 2 B = 1.8 MB (MODEL widths: N = round128(Lk) = 1152 from M 261) = about 7 MB; at
   1.5 TB/s about 5 us vs 384 us MEASURED. So about 98 % of the pooling time is grid overhead (MODEL). Saving: F1 / F2.
2. The cis copy to R rows (VR:422): an (H, B, R, Lk) write per layer (about 3.6 MB at C128, MODEL) that only feeds the
   pooling. F1 reads twin.ccis by request index inside the compiled function: no copy.
3. The mask zero + scatter + masked_fill: 3 launches over (H, E, M) per layer. F5.
NOT avoidable without changing bits: one stage-1 call per (request, rows) group with seqlen_q = R. The op returns
SOFTMAX-normalised scores (S1:102 return_softmax; normalisation over each row's own seqused_k); a request-level
seqused_k changes the normalisation of the rows with a smaller n_used (and the group sum over 16 heads does not keep
the order). REJECTED for the bitwise gate.
Baseline equivalent: SH's one-row selection is the same code; F2 would speed SH by the same 2.8 ms per step.

### 1.2 TENTATIVE STATE (+25.9 / +27.2 in I3; about 11.8 / 13.7 in BEST, MODEL labels R2:297) - AREA 3 owns the fixes

Code per joint pass, per layer (CODE): R twin steps + the deferred replay (ST:338-352, :378-393: 4-5 launches of the
fused twin, + clone / copy of pos, n_used, M_b per row), the buffer writes of R rows (ST:482-513, compiled in BEST:
CD:76-82 for the one-row pass, CC for the joint), the chain images CB0 (3 torch.where) and CB1 (3 copy_) (STF:105-121),
union / locate and row lists (VR:1105-1209). 7,437 small kernels per I3 joint pass (TR:16).
SH equivalent: one twin step and one tail-slot write per layer; no buffer, no chain, no union.
NECESSARY: the twin steps of 3 verify rows; the R K / V / cis rows into a tentative buffer (paper 3.3; about 0.9 MB per
layer at B216); the row lists of the fork.
AVOIDABLE: the R twin launches (TF:242-263 advance_fused already does T rows in ONE launch and returns per-row counters;
the per-row clones then read the counter tensor); the CB1 copy (the buffer itself can be arena blocks that the fork lists
in place; FK reads any arena block id); in BEST the remaining small kernels of union / locate (3.25 ms) and the C73
radix sorts (TR:622, :624: 2.35 + 1.98 ms, label SELECT; their owner op must be found in the trace of a BEST pass, X5).
Gates: bitwise (integer / bit-copy ops; TF is a declared bitwise twin).

### 1.3 FORK ATTENTION (37.1 / 52.2 ms; +11.6 / +14.7 above the model; logical rate 0.885 / 0.927 TB/s)

Code (CODE): ONE `kernel_unified_attention` launch per layer (STF:182-184 -> VR:1095-1102 -> FK:904-1073), FV6 launch
(VR:154: 4 warps, 3 stages, tile 64, bf16 bias). Grid (432, 2, 1) at B216: axis 0 = total_num_q_blocks = q.shape[0] //
BLOCK_Q + num_seqs = 864 / 4 + 216 = 432 (FK:1006), so HALF of the axis-0 programs exit at once (FK:498-499); axis 1 =
KV head (FK:485). This answers L7 X18 (CODE). Each working program = one (request, KV head): it builds its union in the
kernel (FK:506-514) and loops over the union tiles serially (FK:570), 64 tile rows = 4 rows x 16 query heads.
SH equivalent: flash split-KV call A (window) + call B (buffer) + combine, B one-row entries (TR:80-86): 20.50 / 29.89 ms
per one-row pass.
NECESSARY: every row's attention over its own selection; the union read once per (request, KV head): DRAM 1.017-1.021 x
the union bytes, MEASURED by ncu on the microbench tables (F6:279, :283, :302). The live joint pass has no hardware
counter (X2 stays open for the LIVE pass; the microbench closes the "reads once physically" question for FV6).
AVOIDABLE vs STRUCTURAL: section 4. Result: structural for this kernel (registers cap 8 warps per SM; synchronous loads
per tile; K / V depth 1); the only untried EXISTING option is the 3D split (F4).
Rows that need no union: there are none in a group pass (every row of a request shares its pages); the one-row passes
already use SH's flash call.

### 1.4 COMMIT / DECISION (6.0 / 8.4 ms per round, I3; reset + commit 6.18 / 8.55 BEST, L7:156-157)

Code (CODE, ST:975-999): argmax over (B, R, V) float logits (ST:976; the joint pass casts the LM-head output to float at
ST:313: 4B x V x 4 B), accept_prefix, the counter gather, then T.commit (ST:247-286): per layer 3 where + the write-back
kernel + 3 where + 2 copies = 9 launches x 32 = 288 launches, eager; set_twins (one foreach copy). Fingerprints are
check-only (ST:545-550).
SH equivalent: none (the step writes the tail slot directly; argmax of B x V).
NECESSARY: the argmax of 4B rows, the accept, the commit of the accepted rows into the tail slot (and the write-back of
a filled block).
AVOIDABLE: the eager launch chain of the commit (about 290 launches; F3's frame does not block it: no pool op here): one
CUDA graph per round shape, AREA 3 owns the commit. The float cast of the logits (ST:313) is avoidable for the argmax
(argmax of bf16 = argmax of its exact float copy; ties take the first index in both), but the cast is inside the joint
phase (about 0.3-0.5 ms of HBM at C128, MODEL); the lead owns the LM-head / dense accounting.

### 1.5 DRAFT STEP BODY (+11.33 / +12.92 ms per step above td; BEST 66.14 / 88.22 ms elapsed, L7:139)

Code (CODE): IH:13-28: per layer graph A(l) (norm, QKV GEMM, nosa_linear, rope, twin, stage 1, pooling, top-k, the
compiled dpre), the EAGER residency (diff, pool_update with the host stamp, pool_swap, the gathers: IH:15-17), graph
B(l) (compiled dwrite / dmask, flash call A, flash call B, the float merge, the MLP).
SH equivalent: STRONG = 63.04 ms per step at C128 (one graph per step bucket; the same pooling, residency and call A;
no call B, no merge; it loads its misses; the draft skips them).
So BEST's draft step is +3.10 ms above STRONG's step (DERIVED 66.14 - 63.04), not +11.33: td (54.81) has no pool
residency (L7 row 1: pool_swap 5.5-7.4 ms, a FALSE MODEL ASSUMPTION for C > 63).
NECESSARY: call B + merge + buffer write (the tentative row; +5.91 / +6.89 eager, L7 row 1).
AVOIDABLE: the pooling grid (F2: 2.8 / 4.4 ms per step, the SAME saving is open to SH); the L eager residencies between
the 2L graphs (F3).

## 2. Which reads are shared and which repeated (AREA 1 question)

| read | per joint pass | shared or repeated | evidence | fix |
|---|---|---|---|---|
| weights (129 GEMMs) | one call per weight over all 4B rows | SHARED (once) | R2:128 MEASURED | - |
| KV union pages (fork) | once per (request, KV head) for 4 rows | SHARED: logical 72.24 pages = 0.99 x the model (C73, L7 row 4); physical 1.017-1.021 x U on the microbench (F6:279) | MEASURED | - |
| compressed keys (stage 1) | one stage-1 entry per ROW (cu_k[e] = (e // R) x Lk, VR:289) | REPEATED x4 per request in the launch; partly served by L2 (per-row cost 0.54 / 0.49 of one-row, DERIVED) | MEASURED times | none bitwise (1.1) |
| compressed cis (pooling) | copied to R rows (VR:422), then read per row | REPEATED copy | CODE | F1 reads it by request index |
| stage-1 scores (pooling input) | 5 loads per output element, one program per element | the grid, not the bytes, costs | MEASURED 384 us vs about 5 us of bytes (MODEL) | F1 |
| pool residency (diff, update, swap) | ONCE per layer (row 0's selection) | SHARED | CODE STF:146-149 | - |
| host-only blocks (H2D) | per pass, serial, zero-copy | (AREA 2) | MEASURED 21.3 / 160.1 ms | AREA 2 |
| chain images CB0 / CB1 | 2 blocks per request per layer written then read by the fork | REPEATED (a copy of the buffer) | CODE STF:105-121 | A3 |

## 3. Correctness gates (all fixes)

- Bit-exact fixes (F1, F2, F3, F5, A3): committed tokens bitwise = L4PTFD (S4.eq_gate, the EQF of every I4 arm) AND =
  the BEST arm of the same job (a new J1-EQB gate: tokens and the check-round logits fingerprints). F1 / F2 add J1-PEQ:
  in every check (probe) round, every layer, the shipped kernel runs on the same inputs and its pq / pc must equal the
  compiled ones bit for bit (int16 views). MUST 0 differing values.
- Float-order fixes (F4): the fp64 tolerance gate of fia5 (max |arm - fp64| <= 1e-3 on 16 rows) + the committed-token
  agreement as REPORT (FAST near-tie class, L7 S17), never a MUST equality.

## 4. The fork's rate: launch configuration or structure?

Knobs that exist (no kernel edit): num_warps, num_stages, tile_size (FK:947-949), the bias dtype, BLOCK_Q (rows per
program; FK:861 allows 1 / 2 / 4, but FK:861 also requires max_seqlen_q <= BLOCK_Q, so BLOCK_Q 2 needs each request
split into two pseudo-sequences: input plumbing, not a kernel edit), the 3D split (FK:1028-1070).
Already MEASURED (F5:97-127, B216): 11 configurations (w4 / w8 x s2 / s3 / s4 at tile 64; w4 / w8 x s3 / s4 at tile
32; bf16 bias). The current FV6 = W4S3T64B16 is the fastest (1.2534 ms per layer, events). Every other one loses:
s2 +9 %, s4 +40 % (93 KB shared memory: 1 program per SM), w8 +21-31 %, tile 32 +15-90 %; the only 16-warp-per-SM
configuration (W8S3T32) is +43 %.
STRUCTURAL limits (MEASURED, F5:146-212): 255 registers x 128 threads and 59,392 B of shared memory allow 2 programs =
8 warps per SM (12.4 % warps active); each warp issues once per 7.3 cycles ("wait" 33 %, long scoreboard 27 %); three
synchronous loads per tile (page ids in the MMA layout, ext_rows, the bias; TTGIR lines 1626, 1631, 1641) and K / V
depth 1 (the next tile's copies are waited for in the same iteration). Not bytes (1.02 x the union), not HBM (46-49 % of
peak), not the spills (B16 has the same spills and is faster). Per tile per program 8.1 us; 2 waves x 75.8 tiles x 8.1
us = 1.23 ms = the measured call (DERIVED).
So the gap to flash call A (0.885 vs 1.527 TB/s logical) is a KERNEL-STRUCTURE issue: a 64-row fp32 accumulator per
program (occupancy) and the Triton 3.2 pipeliner (synchronous per-tile loads). Removing it needs a kernel edit (async
copies of the bias / extents, a 2-deep K / V buffer at less shared memory) or another kernel family: BLOCKER REPORTED
(no new kernel).
Untried EXISTING options, with what they can and cannot do (MODEL):
- 3D split (F4): more programs, not more warps per SM (still 2 programs per SM). It removes only the tail imbalance of
  432 programs with union lengths 71 mean / 99 max (L7 row 4). Bound: at most about 10 % of the call. It adds
  reduce_segments and changes the float order.
- BLOCK_Q 2 (two pseudo-sequences per request): half the accumulator (registers down), but shared memory stays about
  59 KB (2-3 programs per SM) and each request's union is read twice (the second from L2 if both programs run close
  in time). fia5's 16-warp tile-32 arm lost 43 %; a gain is not likely.
Cheapest measurement: a timing sweep in the fia6 microbench harness on the live-shaped tables (cells 216x4, 318x4):
FV6 2D vs 3D with 2 / 4 / 8 segments, events median of 200 calls + one ncu pass (gpu__time_duration, dram__bytes_read,
sm__warps_active) per arm. About 5 GPU-minutes. Only if 3D saves >= 5 % does it enter a cell (F4).

## 5. The smallest first GPU experiment (the largest avoidable cost: the pooling grid, F1)

Cell (built in the J1 tree; section "J1 protocol" of its NOTES): one process, C128 pool 65, B216, 512 steps, K 3, probe
16, SH first, then L4PTFD, L4PTFDGUDC, L4PTFDGUDCD, L4PTFDGUDCDJ1 (= BEST + F1). About 15-20 min on one A100.
MUST gates: every inherited I4DC gate for the J1 arm under its own name (RUN, ERR, COMMIT-N, D1, D3, SPAN, RESID,
ATTN1, FILL1, LEAN, MEM, GRAPH, COUNTS, DGRAPH, HOIST, COMPILE (the D gate), EQF = L4PTFD bitwise), J1-EQB (= BEST
bitwise, tokens), J1-PEQ (pooling bits = the shipped kernel, every probe-round layer, 0 differing values), J1-COMPILE
(one compile of the pooling helper, outside any capture, no graph break, no recompile).
Decision rule: ADOPT F1 if all MUST gates pass and (BEST joint median - J1 joint median) is in [9.5, 13.5] ms at C128.
Falsifier: the saving < 5 ms (the compiled gather is not near the byte floor) or J1-PEQ > 0. Then C73 B318 with the
keep arms (L4PTFDGUDKWCD vs L4PTFDGUDKWCDJ1): expected joint saving 17.1 ms [14, 18.5].

## 6. Shell commands run for this brief (login node; no interpreter, no sed, no heredoc, no hashing)

1. `date -u; ls -la PL | tail -40; wc -l` of L7, R2, RECON7_R1_model.md, RECON7_R4_prior_art.md, NG
2. `git -C T log --oneline -8; git -C T status --porcelain | head; ls N E | head -150`
3. `ls N; cd T; wc -l` of 18 source files
4. `cd N; grep -n "^def \|^class \|^    def "` over 9 modules
5. `cd E; grep -n "def \|\[fork\]" FK | head -260`
6. `cd PL; grep -n "FVW\|num_warps\|stages\|occupancy\|n_regs\|spill\|shared\|3D\|segment\|TB/s" SCORING_fia6.md | head -80`
7. `cd PL; grep -n "FVW[0-9]" SCORING_fia5.md SCORING_fia4.md | head -60; grep -n "3D\|split\|segment" SCORING_fia5.md | head -30`
8. `cd T; grep -rn "_pool_stamp_base" N/*.py E/*.py | head -40`
9. `cd N; grep -n "" RE | head -420 | tail -110; grep -n "" cache_engine.py | head -515 | tail -25`
10. `cd T; grep -n "stamp" N/r2_tick_capture.py N/r2_tick_fixed.py N/r2_capture.py N/r2_residency_capture.py E/r2_i4h_graph.py E/r2_i4k_combo.py | head -40`
11. `cd T; grep -rln "whole-step\|whole step" N E/*.py | head; grep -n "pool" N/r2_tick_capture.py | head -30`
12. `cd T; grep -n "whole-step\|..." E/r2_wsa_cell.py | head -30; grep -n "CudaGraph\|graph" N/nosa_llama.py | head -40`
13. `cd PL; grep -n "whole-step\|..." SCORING_mbstrong.md | head -30; ls /mnt/beegfs/ojerbi/ | grep -i "nosi\|nosa"`
14. `cd PL; grep -n "stamp\|pool_update\|whole-step\|how\|graph" SCORING_gs1.md | head -40`
15. `git -C /mnt/beegfs/ojerbi/nosi-r2-graphstep grep -n "stamp" 0159e82 -- '*.py' | head -20`
16. `git -C .../nosi-r2-graphstep grep -n "shift_out\|..." 0159e82 -- nosi/nosi/r2_graph_step.py | head -40; git -C ... ls-tree --name-only 0159e82 nosi/nosi/ | grep -i graph`
17. `git -C .../nosi-r2-graphstep grep -n "shift_out\|def shift\|stamp\|_pool_age" 0159e82 -- nosi/nosi/graph_step.py | head -50`
18. `cd LG; wc -l recon7_r2_s2.txt; grep -n "pool\|POOL\|topk\|stage1\|..." recon7_r2_s2.txt | cut -c1-260 | head -80`
19. `cd E; grep -n "def _layer\|_select_rows\|..." r2_i4*.py r2_recon7*.py | head -30; grep -n "^def \|^class \|..." r2_i4u_hoist.py | head -40`
20. `cd T; grep -rn "ccis_rows\|vs\.pq\|..." N/*.py E/*.py | grep -v "..." | head -20; grep -n "QK_SELECT\|M_alloc" RE | head`
21. `cd E/tests_cpu; ls | head -150; wc -l test_recon7.py test_i4dc.py i4dc_world.py; ls T/tests | head`
22. `cd E/tests_cpu; grep -n "^def \|^class \|Gate 0\|..." test_recon7.py | head -80`
23. `cd T; git diff 2e9e497 -- tests_cpu/test_i4dc.py tests_cpu/test_wsa.py | head -80`
24. `cd LG; grep -n "1272, 261\|318, 261\|..." recon7_r2_s2.txt | cut -c1-200 | head; grep -n "kernel_unified_attention" recon7_r2_s2.txt | cut -c1-200 | head -4`
25. `cd E; grep -n "total_num_q_blocks\|grid\|num_segments\|..." FK | head -40`
26. `date -u; grep -n "pool_reference\|G6\|G4" E/r2_kernels_gate.py | head -20`
Plus the Read tool on the files named above. No srun, no sbatch, no GPU or CPU job.
