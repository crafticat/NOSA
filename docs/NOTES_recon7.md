# RECON7: the measurement cells of the reconciliation's discriminators

Written 2026-10-30 (`date -u` at the commit). Style: ASD-STE100. Labels: MEASURED (a record), DERIVED (arithmetic on
MEASURED values, operands shown), MODEL (needs an assumption, stated), CODE (read in the source at file:line).

Request: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md`. Discriminators:
`RECON7_R2_joint.md` section 4 (D1 - D7) and `RECON7_R3_draft_transfer_capacity_schedule.md` section 6 (D1, D1b, D2,
D3), same folder. Tree `/mnt/beegfs/ojerbi/nosi-r2-recon7`, branch `r2-recon7` = I4DC `2e9e497` + these files.

MEASUREMENT ONLY. NO NEW KERNEL: torch.profiler, CUDA events, record_function ranges, CUDA streams and events, the
shipped kernels, cuBLAS through torch, torch copies. No file of `nosi/` changes. No arm's computation changes (a PEQ
gate proves it for the probed arms).

## 1. Files

| file | what |
|---|---|
| `benchmarks/Efficiency/r2_recon7_protocol.py` | `PROTOCOL_STAMP_RECON7 = None`: every cell refuses (exit 20) and the step script STOPS every cell while it is None |
| `benchmarks/Efficiency/r2_recon7_probe.py` | cell (a)'s probe `ProbeR7`, the counters (GatherTap, SkipRecord, UnionReader), the drain-free probe statistic, the pure records and gates |
| `benchmarks/Efficiency/r2_recon7_cell.py` | cell (a): the profiler-window cell of the best arms |
| `benchmarks/Efficiency/r2_recon7_streams.py` | cell (b): the two-stream concurrency harness (R2 D1) |
| `benchmarks/Efficiency/r2_recon7_mem.py` | cell (c): the C 81 memory probe (R3 D3) |
| `benchmarks/Efficiency/r2_recon7_step.sh` | ONE step script, cells r7pm, r7pa, r7pb, r7s, r7mem |
| `benchmarks/Efficiency/r2_i4c_cell.py` | BYTE COPY of nosi-r2-i4c `3aef060` (I4C's profiler window; imported, not edited) |
| `benchmarks/Efficiency/r2_i4m_cell.py`, `r2_wsm_cell.py` | BYTE COPIES of nosi-r2-i4m `f7fb2c5` (the I4M / WS-M memory method; imported, not edited) |
| `benchmarks/Efficiency/tests_cpu/test_recon7.py` | the CPU tests; one RECON7 declaration block in each older Gate-0 test, one line of test_i4dc's name scan |

## 2. What each discriminator needs, and which cell gives it

| discriminator | question | cell | value printed (line prefix) |
|---|---|---|---|
| R2 D1 | can max(HBM, dense) overlap exist on this A100 for these kernels? | r7s | `R7 D1` (medians, eff, DECISION) |
| R2 D3 | union pages per (layer, KV head, request) at C 128 (72.24 MEASURED only at C 73) | r7pa, r7pb | `R7 D2/D3/D7` (union pages per stream) |
| R2 D5 | the device split of the best arm under graphs and compile | r7pa, r7pb | `R7 D5/D7` (busy, classes, GEMM / fork / H2D counts) |
| R2 D6 | decision vs commit inside the 5.98 / 8.34 ms commit phase | r7pa, r7pb | `R7 D6` (labelled commit window) |
| R2 D7 | the NEW blocks and the H2D ms of the SAME pass (the link rate) | r7pa, r7pb | `R7 D5/D7` (NEW blocks, bytes, GB/s) |
| R3 D1 | the draft pass split after graphs + compile | r7pa, r7pb | profiled draft windows; `R7 R3-D1` (marked replay labels) |
| R3 D2 | is a NEW block one that a draft skipped / a slot that a draft skip cleared? | r7pa, r7pb | `R7 D2` (flagA, flagB shares) |
| R3 D3 | B_fit at C 81 (MEASURED), the store's PEAK occupancy | r7mem | `R7 D3`, `R7 D3 FIT` |
| R2 3.5 | the probe statistic defect (drain rounds in the probe medians) | r7pa, r7pb (and the CPU test) | `R7 PROBE-STAT` |

Not built here: R2 D2 (ncu on one fork launch: a tool run, not a cell), R2 D4 (a CPU read of job 2175791), R3 D1b (the
ladder at C 81 B 287: an arm run after r7mem shows the fit).

## 3. THE GPU PROTOCOL (for review; nothing runs before the human sets PROTOCOL_STAMP_RECON7)

Each subsection states what the run decides, its expected duration (STEP_MIN and its basis), its gates and its
falsifier. Order: r7pm first (its MUST gates decide whether r7pa / r7pb run); r7s and r7mem are independent of (a) and of each
other. One holder, the shared claim `OUTROOT/sp_active_<holder>` (one step at a time). MODEL total: about 3.4 h of one
A100 (10.5 + 60.5 + 91 + 5 + 35 min); the sum of the STEP_MIN limits is 390 min.

Command form: `bash benchmarks/Efficiency/r2_recon7_step.sh --nosi <sha> <holder job> <run id> <cell>` from the tree at
`<sha>`. DRYRUN=1 prints the pins and the command and then STOPS on the stamp like a real launch.

### 3.1 Cell (a): r7pm, r7pa, r7pb (`r2_recon7_cell.py`)

ARMS (one process, one model, SH first): L4PTFD, the C arm, the D arm (the best arm), the PROBED D arm (`...P` = the D
arm with `ProbeR7`). r7pa: C 128 (pool 65) B 216, L4PTFD, L4PTFDGUDC, L4PTFDGUDCD, L4PTFDGUDCDP. r7pb: C 73 (pool 10)
B 318, L4PTFD, L4PTFDGUDKWC, L4PTFDGUDKWCD, L4PTFDGUDKWCDP. r7pm: C 73 B 8, 72 steps, probe 4, both families (7 arms).
Probe rounds (r7pa / r7pb, probe every 16): profiled 96, 112 (labelled 112); counted 48, 64, 144 (recording rounds 47,
63, 143); marked: the other probe rounds (16, 32, 80, 128, 160, 176, ...).

What the probe does in each kind of round (CODE: `r2_recon7_probe.py` module docstring):
- PROFILED: `vs.marks` None, `T.checking` False, the meter's timing False = the STEADY keys of the joint graphs
  (`r2_i4g_graph.py` TentStateFDG._key) and of the draft graphs (`r2_i4h_graph.py` _dkey). A window opens only when
  that key's graphs exist for every layer (no warm, no capture inside a window). Windows: `i4c.joint`, `i4c.commit`
  (split: from the joint pass's return to the next pass's start), `i4c.draft0`, `i4c.draft1`. After each window: the
  path check (the joint replayed L graphs; a one-row pass was one replay pass). The meter's block count before and after
  each window (outside it): the NEW host blocks of that pass (D7).
- LABELLED (a profiled round): record_function ranges at the host calls that a replay still crosses: joint `pre`,
  `twin`, `select`, (`kpre`), `static`, `resid`, `section` (the replay), `post`; one-row `static`, per layer `A`,
  `resid`, `B`, then `head`; commit `decide`, `accept`, `filter`, `plant`, `fp`, `ctr`, `commit`, `twins`, `tokens`,
  `dprep`. A graph's kernels take the range of their `cudaGraphLaunch` only if CUPTI correlates them (GPU only: the
  record reports `unranged_ops`).
- COUNTED: the joint pass runs as every probe round's (eager: marks and timing on). GatherTap wraps
  `rs.ops.kernels["gather_k"]`: every host gather that the meter counts, by kind (window load / store fill / other), its
  NEW blocks, flagA (a draft row skipped this block id of this (layer, KV head, request) since the previous joint pass),
  flagB (window loads: a draft skip cleared this slot and no load refilled it). UnionReader reads `vs.fork_len` (the
  union pages per (KV head, request)) and `vs.npg` after each layer.
- RECORDING (counted round - 1, a steady round): each D-slot replay pass runs with a wrapper on `T._resid_lean`
  (the eager residency between A(l) and B(l)): SkipRecord marks the skipped block ids and the cleared slots. THESE
  ROUNDS ARE EXCLUDED FROM EVERY ARM'S STEADY SET (with the L61 capture rounds): the record adds device ops there.
- MARKED: I4C's joint marks (the eager probe-round pass) and the one-row replay marks, as CUDA events.

What the run DECIDES (rules stated before the run; the values are REPORT, not gates):
- D5: the profiled joint pass of the best arm: device busy / the plain arm's steady joint median, the class ms (GEMM,
  OTHER = the fork, SMALL, H2D, POOL, STAGE1, SELECT, COPY), the GPU ops, GEMM calls, fork calls (32 expected), H2D
  kernels, graph launches. It replaces R2 section 3.4's MODEL rows ("unchanged labels keep I3's device ms") by
  MEASURED rows at the same boundaries.
- D6: commit window: decision = decide + accept, commit = commit + twins; `fp` (the check-round fingerprints: present in
  a probe round, absent in a steady round) and `tokens` / `dprep` (after ev3) are separate labels.
- D7: GB/s = NEW blocks x 32,896 B / the H2D class ms of the SAME pass. Rule: >= 0.8 x 25.2 GB/s (nova05 bulk pinned
  H2D, CITED pcie-feed-evidence.md:11) = link-bound gathers; below = the gathers are slow per byte.
- D3: union pages per (layer, KV head, request), per counted pass. Rule: C 128 within +-5 % of 72.24 (C 73 MEASURED)
  = the count transfers; else the model's U - I is batch-dependent (report both).
- R3 D2: flagA share = flagA / NEW. Rule: >= 0.5: a draft-issued copy-engine prefetch can hide up to that share of the
  exposed gather (115.6 ms per round at C 73, MEASURED); < 0.2: the coverage assumption is false on this path too (as
  WS-E found on the legacy schedule); between: report.
- R3 D1: the profiled draft windows' class split (66.1 / 88.2 ms per draft step after graphs + compile, MEASURED steady).
- R2 3.5: every probe median without the drain rounds, with the all-rounds value and the bias beside it.

MUST GATES (CPU-tested on the stand-ins): SH-RUN, C0_FORK; per arm the I4DC cell's gates (RUN, ERR, COMMIT-N, D1, D3,
SPAN, RESID, ATTN1, FILL1, LEAN, MEM, GRAPH, COUNTS, DGRAPH, HOIST, the keep gates, COMPILE, EQF); for a probed arm the
plain arm's gates under its name with COUNTS against the PLAIN arm (TentState's check-only counters forced, injected,
edge_rows, fill_rows declared: a profiled pass runs with checking False) and -PEQ (committed tokens, check-round
fingerprints, acceptance record and state checksums = the plain arm's, bit for bit), -PROF-PATH, -MARKS-NEUTRAL,
-MARKS-SHAPE, -PHASE-SUM, -COUNT. REPORT: -PROF, -PROF-NEUTRAL.

STEP_MIN / mem (basis MEASURED, sacct of the step records under `/mnt/beegfs/ojerbi/r2-runs/spec/`): i4dca (SH + 3 arms,
C 128 B 216) 47:32; i4dcb (C 73 B 318) 71:41; per arm about 11 min (C 128) and 17 min (C 73), SH 12.7 / 18.5 min
(stdout timestamps); i4ca (SH + L4PTFD + the profiled L4PTFDP) 36:35 (the profile windows add about 1 min); i4dm2
(SH + 6 arms, B 8) 9:08 / 9:18. MODEL: r7pa = 47.5 + 11.5 + 1.5 = about 60.5 min -> STEP_MIN 110 (margin 1.8), --mem
240G; r7pb = 71.7 + 17 + 2 = about 91 min -> STEP_MIN 160 (margin 1.75), --mem 340G; r7pm = 9.1 + 1.3 = about 10.5 min
-> STEP_MIN 30, --mem 100G. Memory (sacct MaxRSS 3.1 GB for the I4DC cells; the pinned host KV is not in it). Device
memory: the probe adds at most L x H x B x (M_alloc + 1 + 64) bools (about 5.6 MB at B 318) during the decode, below the
prefill-bound peak (83.900 GB with the prefill at C 128, MEASURED i4dca).

FALSIFIERS: -PEQ fails: the probe changed the computation, no number of the run is used. -PROF-PATH fails: a window is
not the steady replay path; D5 / D7 of that pass are not valid. `unranged_ops` close to the window's GPU ops: CUPTI did
not correlate graph kernels with their launches; the class split (by kernel name) stays valid, the per-label split under
graphs is not available. -COUNT fails: D2 / D3 have no sample. -MARKS-NEUTRAL fails: a probe or record round leaked into
the steady set.

### 3.2 Cell (b): r7s (`r2_recon7_streams.py`)

Parts (one layer; replayed 32 times = one pass; synthetic inputs of the MEASURED shapes, C 128 B 216 and C 73 B 318):
A = the four dense GEMMs at M = 4 B (cuBLAS: wqkv 4608 x 4096, wo, gate_up 32768 x 4096, down 4096 x 16384); F = ONE
call of the shipped fork with FORK_LAUNCH, rows of 64 pages shifted (0, 3, 5, 8): union 72 per (KV head, request); G =
the shipped zero-copy gathers (window K + V/bias, store K + V/bias) of 1.226 / 5.0 NEW blocks per (layer, KV head,
request) (MEASURED kv_at_verify), half window, half store; M = the same bytes as ONE copy-engine copy from pinned staging.
Configurations: alone A, F, G, M, FG; SER (Aq, G, F, Ar on one stream); concurrent A|F, A|G, A|M, F|M, A|FG, A|F|M
(one stream per part list, one start event, joined on the main stream). 3 warm-up + 15 measured passes each.

DECIDES (R2 D1; rule stated before the run): REACHABLE when T(A|F|M) <= 1.10 x max(A, F, M); SERIAL when T >= 0.90 x
(A + F + M); PARTIAL otherwise (the efficiency eff = (sum - T) / (sum - max) is printed for every concurrent set). The
interference T(A|M) / max(A, M) - 1 is compared with the model's s = 8.685 %. LIMITS: independent streams have no data
dependency, so the answer is an UPPER BOUND of the overlap that the joint pass can get; the real pass needs a pipelined
schedule (layer l's fork needs layer l's QKV GEMM) to approach it. A SERIAL answer is a hardware statement for these
kernels; a REACHABLE answer says the max() is a scheduling question.

MUST GATES: <shape>-RUN, <shape>-<config>-EXACT (every part's output bits after the configuration = its reference run
alone: concurrency changes no value), <shape>-UNION (the fork's union_len = 72 everywhere), <shape>-IDS (the NEW blocks
per layer = the requested count). STEP_MIN 20 (MODEL about 5 min: no model load; one Triton compile of the fork and the
gathers; 2 shapes x 12 configurations x 18 passes of about 0.1 - 0.4 s), --mem 64G. FALSIFIER: an EXACT failure means a
concurrency bug (report it, decide nothing); a UNION failure means the synthetic lists missed the measured union.

### 3.3 Cell (c): r7mem (`r2_recon7_mem.py`)

C 81 (pool 18), B 64 then B 192, 512 steps, 49 probe rounds, L4PTFD (the I4M method: the arena prefill binds; the
graph pools of the best arms add 71 - 117 MB, MEASURED). For each B: SH, then the probe arm (I4M's probe_arm_x line for
line plus the occupancy tap). The prefill line and the decode line are fitted through the two MEASURED points; B_fit at
84.0 GB with margins 0 / 0.10 / 0.25 GB and at the usable 83.75 GB (L60).

DECIDES (R3 D3; rule stated before the run): B_fit at 83.75 GB >= 287: capacity is not a gap at the original anchor
(R3's MODEL 321.5 - 322.8 is checked against the MEASURED value); < 287: the anchor's batch does not fit our layout at
C 81 and the model's capacity premise is false. The store's peak listed blocks per (layer, KV head, request) against
X - 1 = 128: the peak sizes X (R3: the mean is 1.03 / 2.28; > 98 % idle).

MUST GATES (per B): SH-RUN, L4PTFD-RUN, I4M's probe-mode arm gates (ERR, D1, D3, SPAN, RESID, ATTN1, FILL1, LEAN,
COMMIT-P), MEM, PR-ROUNDS, PR-FILL, PR-FLAT, PR-PRE, OCC; FIT. STEP_MIN 70 (MODEL about 35 min: i4mp73 at B 64 took
11:21 MEASURED, of it SH prefill 174 s and the arena prefill 277 s; at B 192 the prefills scale with B: about 24 min),
--mem 240G (the pinned host KV at C 81 B 192 is about 106 GB, DERIVED 192 x 16,896 x 2 x 128 x 2 B x 2 x 32 layers).
FALSIFIER: PR-FLAT fails (the decode peak grows): the decode line is not valid; FIT fails: no line.

## 4. The probe statistic fix (R2 section 3.5)

Defect: the I4C probe medians mixed DRAIN rounds (rounds that begin with a finished request: the joint loads less and
is faster) with the mid-run probe rounds. R2: at C 73 the 11 marked probe joints of L4PTFDP give 374.565 ms; without the
rounds >= 170 (176, 192, 208) 391.019 ms. The L7 "-12.7 ms residual" came from this sample, not from an elapsed effect.

Fix (CODE `r2_recon7_probe.py`): the dispatcher reads the runner's host counts at the first pass of each round
(`h_count`, no device read) and records the finished requests at the round start; `drain_rounds` = rounds with one or
more; `probe_stat` gives the median WITHOUT them (the record's value), the median of ALL rounds and the bias. Every probe
median of cell (a)'s record uses it: the marked joint passes, the marked one-row passes, every arm's own probe mark sum
(`probe_total_nodrain`), L7. The drain rounds come from the probed arm; they hold for every arm whose committed tokens
equal it (EQF / PEQ).

CPU test (`test_probe_statistic_fix`): R2's measured C 73 values planted with their drain rounds: old median 374.6, fixed
391.0, bias -16.4 ms; the record's summaries give the same; the CPU run's drain rounds equal an independent derivation
from `_ps_after` (the host counts after each round).

## 5. What the GPU only can establish

Every millisecond, the device busy, the per-label device split under CUDA graphs (CUPTI correlation of graph kernels),
the profiler's effect on a graph replay, the Triton and Inductor bits inside the captures under the profiler, the H2D
rate, the concurrency answer (D1) and the copy-engine interference, the memory peaks and B_fit (D3), and the flag shares
on the real workload (D2). The CPU tests establish the plumbing, the bit equality of the probed arms with the plain arms
on the CPU backend, the counters' arithmetic on planted tensors, the gates and their plants, the step script's pins and
STOP.

## 6. CPU tests

`benchmarks/Efficiency/tests_cpu/test_recon7.py` (its own srun per part): Gate 0 (files, declarations, byte copies,
nosi, no kernel, nobody else names the modules); the line-for-line copies (`run_arm_r7` = `run_arm_d` + `# R7` lines;
`probe_arm_r7` = `probe_arm_x` + `# R7` lines); the probed runs of both D arms on the CPU world (PEQ, COUNTS, every
inherited gate, the windows on the replay path, the labels, the counted pass's counters close: the tap's NEW blocks =
the meter's); the probe gates and their plants; a probe that computes is CAUGHT by PEQ; the statistic fix; the counters
on planted tensors; the streams harness (lists, ids, the decision rule, the plumbing with an EXACT plant); the memory
cell (the tap, the fit, the arguments); the cell's arguments; the protocol stamp; the step script (refusals, DRYRUN, the
STOP); these notes. The older Gate-0 tests run with `-k gate0`.

## 7. Risks

1. A profiled window of a graph arm: CUPTI may not report a replay's kernels with the launch's correlation id
   (`r2_p3_profile.parse_trace`: "a graph replay's kernels share its launch"); the class split by name stays valid.
2. The commit window of a profiled round is a PROBE round's commit: it holds the check-round fingerprints (`fp`); a
   steady round has none.
3. The profiled passes run with `T.checking` False: their check-only counters are not counted (declared in COUNTS).
4. The recording rounds lose 3 of about 190 steady rounds per arm (all arms the same).
5. r7s replays one layer's buffers 32 times; the 474 MB of weights and the 0.75 GB arena exceed the 40 MB L2, so no
   reuse in L2 helps a later layer (MODEL).
6. r7mem fits a line through two points; I4M measured the prefill line's linearity at C 73 (residual 0.0001 GB at B 8).
