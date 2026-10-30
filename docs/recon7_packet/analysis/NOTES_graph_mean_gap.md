# NOTES: why the CUDA-graph arms lose on MEAN and EFFECTIVE but win on MEDIAN (2026-10-29)

Analysis only. No tree edit, no GPU job. Scripts: /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/ana_graphmean/a1_keys.py,
a2_decomp.py, a3_classes.py. Logs (srun on amd): /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/ana_graphmean_a1.txt,
ana_graphmean_a2.txt, ana_graphmean_a3.txt.

Labels: MEASURED = a field of a summary.json; DERIVED = arithmetic on MEASURED fields; MODEL = needs an assumption.

## 0. Data limit

summary.json has NO per-round time list. The arm record keeps rates, phases_median, wall_s, prelude_ms,
capture_rounds (round, joint ms), joint_phases_fork (probe medians), paper.accepted_per_round (194 values) and the
trace_tail (last 12 entries per request). The runner computes round_ms per round (spec_tentative.py:1188) but
WA.summarize_w (r2_wsa_cell.py:249-291) keeps only aggregates. So the per-class times below are DERIVED totals plus
MODEL splits.

## 1. How the cell computes the three rates (code)

- Steady set S = rounds 1 .. G-1 minus g % 16 == 0 (r2_spec_cell.py:159-161). Its docstring says "Round 0 (the
  first launches, the graph captures) ... excluded". Round 0 is excluded. Round 1 is NOT excluded.
- median = median over S of round tokens / round ms; mean = sum tokens(S) / sum ms(S); effective = all committed
  tokens / wall_s (r2_spec_cell.py:206-217).
- wall_s = perf_counter from before the warm step to after the last round (spec_tentative.py:682-722). It includes
  the warm step, the prelude, every round, and the host time between rounds (_ps_plan, the sync, _ps_after).

## 2. The mechanism (code)

- r2_i4g_graph.py:249-252: the FIRST eligible joint pass of a layer runs EAGER on the static inputs (warm).
  r2_i4g_graph.py:256-276: the SECOND eligible pass CAPTURES (CudaGraphs.capture, :81-91, two syncs per layer) and
  replays. Round 0 is eligible (probe rounds are g > 0 and g % 16 == 0, spec_tentative.py:708), so round 0 = warm and
  round 1 = capture. MEASURED: capture_passes [1] for all 32 layers; capture_rounds [(1, 6547.764 ms)] in i4ka.
- So the 32 captures (capture_s 6.18-6.79 s, MEASURED) sit in round 1, which is IN the steady set and IN wall_s.
- The frozen key (r2_i4g_graph.py:178-179) never changes in these runs: checking is True only in probe rounds
  (spec_tentative.py:866-872 with check_every = probe_every = 16, r2_i4k_cell.py:320), and probe rounds are eager
  through "marks" first (:166-167); meter.on is True in every joint pass (spec_tentative.py:954); deferred = q + P.v,
  never None (spec_tentative.py:937); R = vs.R = K + 1 = 4 in every pass (spec_tentative.py:917; MEASURED joint_rows
  776 = 4 x 194, j1_passes 0). MEASURED: eager_why = {'marks': 384} = 12 probes x 32 layers; no "key" eager call.
  Result: no re-capture, no eager fallback, one capture pass.
- The drain: requests finish in rounds 170-193 (B 216) and 170-208 (B 318) (MEASURED, trace_tail 'final' entries).
  A DONE request only clears masks (spec_tentative.py:552-585, :912-938). The joint pass keeps the (R, B) shapes in
  BOTH arms (STF:124-244: every tensor is (B, R) or (H, B, R); no nonzero / no host-sized list). So in the drain,
  the eager arm and the graph arm launch the same full-B work. The drain is not a differential term.

## 3. Class table (i4ka, B 216, C 128, 512 steps)

| arm | class | n | total ms | mean ms | label |
|---|---|---|---|---|---|
| L4PTFD | steady pre-drain | 160 | 57,081 | 356.75 | DERIVED total, MODEL split |
| L4PTFD | steady drain | 21 | 7,492 | 356.75 | MODEL (no per-round time) |
| L4PTFD | capture | 0 | 0 | - | MEASURED |
| L4PTFD | probe (eager) | 12 | 4,212 | 351.0 | MODEL (probe joint MEASURED 201.7) |
| L4PTFD | round 0 + warm + host gaps | 1 + 1 + 194 | 12,138 | - | DERIVED |
| L4PTFD | prelude | 1 | 238.1 | - | MEASURED |
| L4PTFDG | steady pre-drain (excl rd 1) | 159 | 55,465 | 348.84 | DERIVED total, MODEL split |
| L4PTFDG | steady drain | 21 | 7,326 | 348.84 | MODEL |
| L4PTFDG | capture (round 1, in S) | 1 | 6,697 | 6,697 | MODEL (joint MEASURED 6,547.8) |
| L4PTFDG | probe (eager) | 12 | 4,207 | 350.5 | MODEL (probe joint MEASURED 201.0) |
| L4PTFDG | round 0 (warm) + warm step + host gaps | 1 + 1 + 194 | 8,161 | - | DERIVED |
| L4PTFDG | prelude | 1 | 205.9 | - | MEASURED |
| L4PTFDGKW | steady excl rd 1 | 180 | 63,348 | 351.93 | DERIVED |
| L4PTFDGKW | capture (round 1) | 1 | 7,073 | 7,073 | MODEL (joint MEASURED 6,923.3) |
| L4PTFDGKW | probe | 12 | 4,332 | 361.0 | MODEL |
| L4PTFDGKW | round 0 + warm + gaps | - | 8,613 | - | DERIVED |

Steady sum: L4PTFD 64,572.5 ms, L4PTFDG 69,488.2 ms, L4PTFDGKW 70,421.1 ms (DERIVED). wall_s 81.161 / 82.062 /
83.573 s (MEASURED).

## 4. Decomposition, L4PTFDG minus L4PTFD (same job)

MEAN (steady sum gap +4,915.7 ms, DERIVED; mean -113.2 tok/s, -7.07 %, MEASURED):
- capture round excess +6,348.5 ms = 129 % of the gap (MODEL: L4PTFD's round 1 is a typical round)
- replay saving over the other 180 steady rounds -1,424.9 ms = -29 % (DERIVED from the totals)
- drain: 0 (same rounds, same tokens, same shapes; DERIVED from code + EQF gate)
- mean without round 1: 1,636.2 tok/s = +2.25 % vs L4PTFD (MODEL). Same in i4ga g3a (+2.03 %) and h4a (+2.25 %),
  B 318 g4b (+1.37 %).

EFFECTIVE (wall gap +900.7 ms, DERIVED; effective -14.9 tok/s, -1.09 %, MEASURED):
- capture excess +6,348.5 ms; replay saving -1,424.9 ms; prelude -32.2 ms (MEASURED);
  rest (warm step, round 0, probes, host gaps) -3,982.9 ms.
- The -3,983 ms "rest" credit is NOT a graph effect. It is a FIRST-ARM penalty that L4PTFD pays because it runs
  first in every i4g / i4k job. Evidence (DERIVED, a2/a3 logs): rest of the first arm = 16,221-16,466 ms at B 216
  for ANY arm kind (L4PTF in i4aa, L4PTFD in i4ka / i4ga / i4da); rest at position 1-2 = 12,211-12,945 ms for ANY
  arm kind (L4PTFD in i4aa, L4PTFDG, L4PTFK). Excess 4,006 ms (B 216), 3,668-3,950 ms (B 318). The steady mean of
  L4PTFD does not move with position (1,597-1,607 at position 0; 1,588-1,599 at position 1). Prelude is +33 ms at
  position 0. MODEL for the cause: first launches / JIT compile of the spec kernels (fresh per-job TRITON_CACHE_DIR,
  command.txt) in the warm step or round 0; the record cannot split these.
- Position-matched (L4PTFDG at position 1 vs L4PTFD at position 1 in i4aa jobs, same B, inputs deterministic):
  effective -4.7 % to -6.1 % at B 216, -3.8 % to -4.1 % at B 318 (DERIVED, cross-job). The same-job -1 % hides
  most of the capture cost.
- Graph-arm rest vs position-1 L4PTFD rest: -280 to +129 ms (B 216), -27 / +4 ms (B 318): the warm round 0 adds
  no measurable cost (DERIVED).

Host gaps (MODEL): rest at position >= 1 minus probes ~= 8.0-8.3 s at B 216 for 194 rounds -> about 40 ms per
round of host-only time (_ps_plan + _ps_after Python loops over B). Same in all arms. This is why effective is far
below mean in every arm.

## 5. Fixes without a new kernel (MODEL effects)

1. Re-capture per R / eager fallback in the drain: no effect. R and the key never change; the drain is the same
   full-shape work in both arms.
2. Count the capture round as a warm-up round (cell: steady_rounds also excludes capture_rounds; r2_spec_cell.py:
   159-161 + r2_i4g_cell.py:193-198). Mean L4PTFDG -> ~1,636 (+2.25 % vs L4PTFD). Effective unchanged.
3. Move the capture into round 0 (hook: at r2_i4g_graph.py:249-252 capture right after the warm call and do not
   replay in that pass; subtract the capture's host-count delta because the warm call counted). Round 0 is already
   outside S. Same mean effect as fix 2; effective unchanged (round 0 is inside wall_s).
4. Report effective without the one-time capture (wall_s - capture excess, or wall_s - capture_s): L4PTFDG
   1,460.7 tok/s at B 216 (+1.8 % / +2.7 % vs position-1 L4PTFD 1,435.0 / 1,422.0); B 318 1,303.0 (+0.8 % / +1.1 %
   vs 1,292.2 / 1,288.7). vLLM-style serving captures at engine start, outside the served span.
5. Remove the first-arm penalty for the reference arm (an untimed warm-up pass, or rotate the arm order): L4PTFD
   effective 1,362.6 -> ~1,422-1,435 (+4.4 % to +5.3 %). Without this fix, any same-job effective comparison
   favours every arm that is not first by ~4 s of wall.
With fixes 4 + 5, the effective ordering follows the median ordering (graph about +2 % at B 216, +1 % at B 318).

## 6. The verdict metric (median steady tok/s at matched memory)

- The capture round is 1 of 181 (B 216) / 195 (B 318) samples in S. It moves the median by at most one order
  statistic. DERIVED check: L4PTFDG round_ms_median 349.07 ms vs the mean of the other 180 steady rounds 348.84 ms.
- Probe rounds (eager by design) and round 0 are outside S. The drain rounds (21 / 35 in S) are the same rounds with
  the same tokens in both arms.
- Memory: MEASURED peak with the prefill 83.891 GB in all three i4ka arms; graph pool 14.7 / 35.7 MB, static 8.0 MB.
- So the median verdict is not affected (at most one rank). The mean and the effective rate ARE affected, by the
  capture-in-round-1 placement (both) and by the first-arm penalty (effective only).

## 7. Runs not present

- i4k_k4b_2195576_i4kb (B 318, i4k arms) and i4g_h3b_2195575_i4gb: still running at 2026-10-29T16:21Z, no
  summary.json. Not analysed.
