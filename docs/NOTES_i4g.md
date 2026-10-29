# I4G: lever 3, a CUDA graph trial over the joint pass's union + fork section (arm L4PTFDG)

Written 2026-10-29T11:43:59Z (`date -u`). Style: ASD-STE100. Labels: MEASURED (in a record), DERIVED (hand arithmetic
on MEASURED values, operands shown), CODE (read in the source at file:line), INFERENCE (my estimate; not measured).

Tree `/mnt/beegfs/ojerbi/nosi-r2-i4g`, branch `r2-i4g` = I4A `acbe433` + this rung. Abbreviations: ST =
`nosi/nosi/spec_tentative.py`, STU = `nosi/nosi/spec_tentative_union.py`, STF = `nosi/nosi/spec_tentative_fork.py`,
VR = `nosi/nosi/verify_rows.py`, RE = `nosi/nosi/ragged_engine.py`, GR = `benchmarks/Efficiency/r2_i4g_graph.py`,
CELL = `benchmarks/Efficiency/r2_i4g_cell.py`. The basis: `/mnt/beegfs/ojerbi/nosi-r2-i4c/docs/NOTES_i4c.md` section 7.

Files: GR, CELL, `benchmarks/Efficiency/r2_i4g_step.sh`, `benchmarks/Efficiency/tests_cpu/i4g_world.py`,
`tests_cpu/test_i4g.py`, this file; one declaration block in `test_wsa.py`, `test_wsb.py`, `test_wsc.py`, `test_i2.py`,
`test_i3.py`, `test_i4a.py`; in `test_i4a.py` also ONE line of `test_i4a_gate0_files` (`I4A_LATER`: its name scan skips
CELL and GR, which name the I4A modules), as I4C did.

NO NEW KERNEL and no kernel edit: `torch.cuda.CUDAGraph` (the machinery of RE:346-370 and
`nosi/nosi/r2_tick_capture.py:73-76`), the shipped kernels and torch ops. GR imports torch and time only. No file of
`nosi/` and no I4A file changes (Gate 0).

## 0. Answer in five lines

1. MEASURED (i4ca): the joint pass is busy 195.548 ms of the 206.157 ms steady median: 10.609 ms not busy (DERIVED).
2. DERIVED (section 1): about 3.2 ms of it is the device's own gap between queued kernels (0.350 us x 9,076 ops); the
   host exposure is about 7.4 ms; 5.8-7.0 ms of it sits in STF:150-186 (the capturable section). This is above the 3 ms
   stop line, so I built the trial.
3. MEASURED (i4cb, C 73 B 318): busy / steady 0.9958 (local 0.9825): the host share is 0.4-1.8 %; after the gap term it
   is about 0-3.5 ms. B 318 is a no-regression check only.
4. BUILT: L4PTFDG replays STF._rows_union (STF:124-244) minus the residency from one CUDA graph per layer (32 graphs, one
   shared pool). The residency stays eager (spec_tentative.py:461-466: the host stamp). CPU: L4PTFDG = L4PTFD bit for bit
   (FakeGraphs); 201 fewer eager launches per layer in a replayed pass (an independent counter).
5. MODEL (section 5): C 128 B 216 joint -5.0 to -9.9 ms per round (central -6.5), tok/s x1.014-1.029; C 73 B 318 x1.00.

## 1. The capturable share of the 10.6 ms (before the build)

Sources: `/mnt/beegfs/ojerbi/r2-runs/spec/i4c_c3a_2195575_i4ca.step/stdout.txt` (lines 66, 89-92, 124-137), the traces
`trace_L4PTFDP_g96_joint.json.gz` / `g112` beside its summary.json, my CPU split of the traces (the script
`/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4g_ana/gapsplit.py`, log
`/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/i4g_ana_gapsplit2.txt`), and the R2G / R2E / SH periods of M-P3
(`/mnt/beegfs/ojerbi/r2-runs/p3/p3_mp3_2190876.step/stdout.txt` lines 49-51).

| item | value | label |
|---|---|---|
| steady joint (REF) / profiled busy | 206.157 / 195.548 ms: 10.609 ms not busy | MEASURED / DERIVED |
| GPU ops per joint pass | 9,076.5 (9,142 / 9,011) | MEASURED |
| profiled pass g112: idle in the device window | 42.885 ms = 30.880 host-exposed (the device was free before the launch call returned) + 12.005 device-side | MEASURED (split) |
| gap before a queued op (profiled) | median 1.152 us, mean 1.177 us (7,411 ops) | MEASURED (split) |
| SH, eager, host far ahead: period - profiled device sum | 45.906 - 45.462 = 0.444 ms / 1,269 ops = 0.350 us per op | DERIVED (M-P3) |
| R2G, one CUDA graph: period - profiled device sum | 51.658 - 52.965 = -1.307 ms / 4,841 ops = -0.270 us per op | DERIVED (M-P3) |
| the device-side gap of an unprofiled eager joint pass | 9,076 x 0.350 us = 3.18 ms | DERIVED (INFERENCE: SH's gap transfers) |
| host exposure of the steady joint pass | 10.61 - 3.18 = 7.43 ms | DERIVED |
| share of the host-exposed idle in the section's labels (write, chain, union, fetch, attn) | 28.898 / 30.880 = 0.936 (the idle before ops launched in those labels) | MEASURED (split, round 112) |
| section ops | 5,896 of 9,011 (0.654) | MEASURED (split) |
| section host exposure, basis A | 0.936 x 7.43 = 6.95 ms | DERIVED |
| section host exposure, basis B (marked rounds, elapsed - device - 0.350 us x ops) | union 14.973 - 11.928 - 1.098 = 1.947; write 11.050 - 8.694 - 0.574 = 1.782; chain 3.831 - 3.473 - 0.090 = 0.268; fetch 13.731 - 12.253 - 0.101 = 1.377; attn 40.728 - 40.083 - 0.202 = 0.443; sum 5.82 ms | DERIVED |
| device-side gap a graph may remove in the section | (0.350 + 0.270) us x 5,896 = 3.66 ms at most | DERIVED, INFERENCE (M-P3 kernels, B 64) |

The capturable share is 5.8-7.0 ms of host exposure per joint pass (above 3 ms), plus 0-3.7 ms of device-side gap. Outside
the section: twin 1.35 ms and pre / select 0.63 ms of host-exposed idle (profiled); the residency is eager (section 2).

C 73 B 318 (i4cb, `/mnt/beegfs/ojerbi/r2-runs/spec/i4c_c4b_2195576_i4cb.step/stdout.txt`): busy 381.599 ms, busy /
steady 0.9958, busy / local 0.9825 (MEASURED): 1.6-6.7 ms not busy, minus the 3.2 ms gap term: 0-3.5 ms of host exposure
(DERIVED). The joint pass is device bound there.

## 2. The design (CODE)

What is captured (GR:237-280, `TentStateFDG._rows_union`): the whole `TentStateF._rows_union` (STF:124-244, unchanged) on
STATIC inputs, with its residency call (STF:149) answered by a stub (GR:259) that returns the static skip mask. So a
graph holds STF:127-148 (qb, P, past, hasq, the injection: a recompute of values that the residency does not change) and
STF:150-244: the buffer writes (ST:482-513), the chain images (STF:105-121), the locate (VR:1105-1177), the store fill
(the shipped host gathers, through the KVMeter wrapper SR:425-434 with `on` True: its `blocks +=` is a device op), the
row lists (VR:1180-1209), the ONE fork launch (VR:1095-1102; FK:856-891 checks metadata only) and, with ops.stats /
ops.check, their device counters.

What stays EAGER per layer and pass (GR:218-227, `_resid_eager`): STF:135-148's residency inputs (row 0's hasq, the
injection, ti) and the residency itself (STU:100-102 -> ST:450-480). HARD BREAK, spec_tentative.py:461-466:
`pool_update(..., eng._pool_stamp_base, ...)` passes a host int by value, then `eng._pool_stamp_base += eng.topk`. A
replay would keep the captured stamp; a device stamp needs an edit of `pool_update_kernel.cu` (not allowed). So the
graph starts after the residency.

The order per layer (GR:237-280): (1) a layer runs EAGERLY, STF's code on the real tensors, when GR:164-176 gives a
reason: probe marks (`marks`), the meter's timing (`timing`), hooks, the test records, the "F" / "RA" probe of layer 0
(`probe_F`, a host sync), or a frozen key that differs (`key`). (2) Else the static inputs are filled (GR:181-216: per
layer q4, k4, v4 and pbs by copy_ and one stack; per pass the plan's act / verify / draft / deferred). (3) The first such
call runs STF's code EAGERLY on the static inputs (warm: every Triton kernel compiles with the capture's pointers, as
RE:362). (4) The second CAPTURES (GR:254-275: the eager residency first, the stub during the capture, the capture
executes nothing) and then replays. (5) Every later call: the eager residency, the copies, ONE replay.

The frozen key (GR:178-179): T.checking (ST:507-509), the meter's `on` (SR:426), plan.deferred is None (ST:504), R.
Fixed per runner: tneg, fneg, ops.check / stats / neg, rs.pool.

The host counts that a replay skips (CODE: STF:121 `calls["chain"]`, STF:175-176 `gather_host` 2 and `fills` 1, STF:185
`attn` 1; with ops.stats VR:1175 `stat_n["union"] += H x B`): the capture measures the counter dicts before and after
the captured Python (GR:258-269) and every replay adds that delta (GR:229-235, GR:278). GR.SKIPPED_T / SKIPPED_VS state
the expected delta; a capture whose delta differs is listed (`delta_bad`) and fails L4PTFDG-GRAPH. The residency's count
(STU:101) and STU._rows' counts (STU:109-119) run eagerly.

Addresses: VR:1163 and VR:1174 REASSIGN `vs.stat["extras_max"]` / `["union_max"]` to a new tensor
(`torch.maximum(...)`). Under a graph the dict would point at a graph-pool tensor. The runner makes vs.stat an
`InPlaceDict` (GR:60-70, GR:301): an assignment onto a tensor of the same shape / dtype / device copies into it. Same
bits; the addresses stay.

The arm's end (GR:282-286, GR:322-326): `detach()` drops the graphs, the static inputs and the pool handle.

## 3. Gates (CELL)

MUST: C0_FORK; SH-RUN; per arm -RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN, -RESID, -ATTN1, -FILL1, -LEAN; i4gm also
-PRESENT, -VIS, -VB, -TAIL, -CHAIN, -IDX, -RA, -STRESS, -FSTRESS; **L4PTFDG-EQF** (r2_i4a_cell.eq_gate against L4PTFD:
committed tokens, check-round fingerprints, acceptance record, state checksums, bit for bit); **L4PTFDG-COUNTS** (read()'s
tent counters and host counts, the union stats and vs.calls, round blocks, draft skips, lean, token accounting = L4PTFD's);
**L4PTFDG-GRAPH** (32 graphs, 32 warm calls, no `delta_bad`, eager reasons only in EAGER_OK = marks / timing / probe_F,
probe_F at most once, eager + warm + replays = L x joint passes); **L4PTFDG-MEM** (peak allocated with the prefill <=
84.0 GB). REPORT: SH / L4PTFD MEM, the graph pool MB (torch.cuda.memory_snapshot's segment_pool_id), the static MB, the
capture seconds, the capture rounds' joint phase.

## 4. Cells (r2_i4g_step.sh)

| cell | C (pool) | B | steps | check | STEP_MIN | mem | role |
|---|---|---|---|---|---|---|---|
| i4gm | 73 (10) | 8 | 72 | every round (checks captured into the graphs) | 40 | 100G | the miniature: EQF / COUNTS / GRAPH / MEM MUST |
| i4ga | 128 (65) | 216 | 512 | probe rounds | 80 | 240G | timing: the target |
| i4gb | 73 (10) | 318 | 512 | probe rounds | 110 | 340G | timing: no-regression check only (device bound) |

STEP_MIN basis (MEASURED, sacct): i4am 5:54 (SH + 2 arms, B 8); i4aa 36:24 / 36:44 (SH + L4PTF + L4PTFD, C 128 B 216);
i4ab 54:38 / 53:31 (C 73 B 318); i4ca 36:35. Here SH + L4PTFD + L4PTFDG: L4PTFDG is not slower than the L4PTF arm it
replaces; its 32 captures add seconds (INFERENCE, section 5 I4G_capture_s). The margin stays about 2x. i4ga / i4gb STOP
while `PRED_STAMP_I4G` is None.

## 5. Registration keys (PRED_I4G_BY_B[216 | 318], PRED_STAMP_I4G) and the builder's MODEL

Every key is (low, high) except `I4G_eq` (bool). None gives REPORT. The MODEL is an input for the lead, not a
registration.

| key | C 128 B 216 | C 73 B 318 | derivation |
|---|---|---|---|
| I4G_joint_saving (ms) | 5.0-9.9, central 6.5; band (3.0, 10.5) | -1 to 2; band (-2.0, 3.0) | section host exposure 5.8-7.0 (section 1) + device gap 0-3.7 - added device work 0.8 (128 copies + about 9 recomputed ops per layer inside the graph, about 1.5 us each with gaps); B 318: host exposure 0-3.5 ms, mostly outside the section |
| I4G_joint_ratio | (0.95, 0.99) | (0.99, 1.01) | (206.157 - saving) / 206.157 = 0.952-0.976; B 318: 383.231 +- 3 ms |
| I4G_vs_D | (1.008, 1.030) | (0.995, 1.010) | round 355.851 ms / (355.851 - saving) = 1.014-1.029; B 318 round 581.517 ms |
| I4G_ratio | (0.52, 0.54) | (0.59, 0.61) | L4PTFD / SH 0.5206 (i4aa, i4ca) x I4G_vs_D; B 318: 0.5982 (i4cb) x 0.995-1.010 |
| I4G_draft_ratio | (0.98, 1.02) | (0.98, 1.02) | the draft passes are not captured (CODE: GR replaces only _rows_union of R > 1 passes) |
| I4G_eq | True | True | CPU: L4PTFDG = L4PTFD bit for bit (test_l4ptfdg_equals_l4ptfd); a replay runs the same kernels on the same inputs |
| I4G_peak_gb | (83.85, 83.95) | (77.60, 77.80) | the prefill peak 83.891 / 77.685 GB (MEASURED i4aa / i4ab) is above the decode peak 82.559 / 76.932 GB + about 0.05 GB (static inputs and the live capture set) |
| I4G_pool_mb | (2, 100) | (3, 150) | the per-layer working set of STF:150-186 at B 216: about 15 int64 (H, B, R x 64) tensors of 0.885 MB and the sorts' buffers; one shared pool; allocator blocks of 2 / 20 MB (INFERENCE) |
| I4G_capture_s | (0.5, 15) | (0.5, 15) | 32 captures, each `torch.cuda.graph`'s synchronize + gc.collect + empty_cache + about 190 recorded ops (INFERENCE) |

Not registered by the cell (REPORT): the effective tok/s (the capture round is in the run's wall time; round 0 is not
steady, round 1 holds the 32 captures: one sample of about 180 steady rounds).

## 6. Memory

C 128 B 216 has 0.359 GB under the usable 84.25 GB (REF prefill peak 83.891 GB, MEASURED i4aa). The graphs are made in
the decode, after the arm's prefill; the prefill of the L4PTFDG arm is L4PTFD's. Added in the decode: the static inputs
(q 7.08 MB, k / v 0.44 MB each, the masks: about 8 MB, GR:181-216) and the graph pool (one shared pool: no tensor of a
graph lives after its capture; the section returns a view of the persistent vs.fout). The MEM gate is MUST at 84.0 GB
with the prefill; the cell reports the pool's segments and the decode reserved peak. The arm is the last of the job and
detach() drops its graphs.

## 7. What the CPU establishes and what only the GPU can establish

CPU (FakeGraphs, `tests_cpu/i4g_world.py`): capture executes nothing, a replay re-executes the recorded ops on the
captured tensor objects (so an input that is not copied in is stale), kernels are nodes, a host read or a data-dependent
shape outside a node raises. With it: L4PTFDG = L4PTFD bit for bit in a timing world (probe passes eager between
replays), a check world (checks and stats inside the graphs, the "F" probe eager) and B 2 rowgroup; the counts; the
gates; the plants.

GPU only (i4gm first): that every op of STF:127-244 is legal under stream capture on this stack (torch 2.6.0+cu124, the
Triton fork and gathers with host-pinned UVA reads, sorts, index_put, searchsorted), that a replay gives the eager bits
on the device (Triton specializations on the static pointers), the pool's size, the decode peak, the capture time, and
the timing. FakeGraphs does not model streams, the capture's allocator or kernel launch legality.

## 8. CPU tests

`benchmarks/Efficiency/tests_cpu/test_i4g.py` (its own srun): Gate 0; the stand-in's semantics with controls (a stale
input, a copying reshape, a host read); L4PTFDG = L4PTFD (timing, check, B 2 rowgroup) with the cell's EQF / COUNTS /
GRAPH / LEAN gates on the CPU evidence; the plants (stale_input -> EQF; no_counts -> COUNTS and ATTN1 / FILL1; no_key ->
COUNTS; plain_stats -> the stats lose their addresses); the eager launch count (201 fewer aten launches per layer in a
replayed pass; the kernel stand-ins count 1 each); no host read in GR (SyncTrap); the cell's arguments, pure parts,
registration state, step script (DRYRUN) and these notes. Also run: test_i4a.py, test_i3.py, test_i2.py, test_wsa.py,
test_wsb.py, test_wsc.py (their declaration blocks).

## 9. Risks

1. A capture may fail on the GPU (an op that synchronizes or allocates illegally under capture): the arm then raises
   (L4PTFDG-RUN fails), the job's other arms keep their records. i4gm first.
2. torch.cuda.graph calls gc.collect and empty_cache at each capture: the capture round is slow and the allocator
   re-maps memory afterwards (REPORT: the capture rounds' joint phase).
3. The 0.350 us gap term comes from SH at B 64 (M-P3); if the unprofiled gap is larger, the host exposure and the saving
   are smaller (the floor of the band).
4. The probe rounds stay eager in both arms: the steady medians exclude them.
