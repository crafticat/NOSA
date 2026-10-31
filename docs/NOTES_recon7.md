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
| `benchmarks/Efficiency/r2_recon7_protocol.py` | the CELL GRANTS, default deny: one `GRANT_<cell>` line per cell (e.g. `GRANT_r7s`) and the granted argument PROFILE of each cell; a cell refuses (exit 20) and the step script STOPS it unless its own line holds a `date -u` stamp (replaces the one-for-all stamp of 8d7dab8 .. 028a5ec) |
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

## X1 protocol (repaired after review 3, 2026-10-31)

The SMALLEST discriminator. X1 = step cell `r7s` = `benchmarks/Efficiency/r2_recon7_streams.py`. No model, no workload,
no arm; one GPU step (hard wall STEP_MIN 20 min, of it the image hash on the GPU node). Grant: `GRANT_r7s` only (default
deny: every other cell stays refused by the wrapper and by its Python main).

Review 3 (2026-10-31) findings and their repair in this section and in the code:

| finding | repair (code) | CPU check (tests_cpu/test_recon7.py) |
|---|---|---|
| 1 one stamp unlocked all five cells | `r2_recon7_protocol.py`: one `GRANT_<cell>` line per cell, default deny; `refusal(cell, module, args)` checks the cell, its module and its granted PROFILE; the wrapper's `granted()` greps one fixed pattern per cell | test_protocol_state; test_step_script_recon7 (planted protocol texts: an r7s grant refuses r7pm / r7pa / r7pb / r7mem; malformed / missing grants refuse); test_recon7_cell_pure, test_mem_pure |
| 2 group bit equality was REPORT only; failures continued to the next shape; the deadline covered the timed passes only; BUDGET was not the true wall; decisions printed under failed MUST gates | `<shape>-GRP-EQ` DECISION GATE governs GRP / GRPL (INELIGIBLE otherwise); `run_shapes()` STOPS THE RUN on free memory, UNION, IDS, EXACT, deadline, exception; the deadline from the shape's start (setup, references, warm-up, passes); BUDGET = the wall from the start to the last gate; RUN-WALL; any failed MUST gate SUPPRESSES every decision | test_streams_run_shapes, test_streams_measure_plumbing, test_streams_pure |
| 3 the buffers do not carry the joint pass's dependencies | mode (b) renamed the ORDERED SYNTHETIC SCHEDULING SCREEN; PF renamed CPY (copy interference: an own buffer, never read); no "attainable", "prefetch hides" or "dependency" claim | test_streams_pure (docstring words present, retracted words absent); test_notes_recon7 |
| 4 A|F|M was judged against the isolated sum while SER ran A + F + G; a 5 % control discrepancy only printed "suspect"; the weight reads were stated as twice / four times 474 MB per layer | a same-ops serial control X-SER for every concurrent set (26 configurations); SERIAL judged against X-SER; isolated maxima an IDEALIZED reference; a control more than 5 % from its independent sum INVALIDATES the conclusion; every timed and warm-up pass kept with n, quartiles, cv; 474 / 948 MB | test_streams_pure (planted medians: consistent, 6 % INVALID, 4 % valid, SERIAL vs the matched control) |
| 5 image sha256 declared, not checked; login-side hashing through PATH; model / workload prerequisites for r7s; output limit an estimate | the wrapper hashes nothing (cmp against `git show`); the CHILD hashes the image (and the workload of the cells that read it) with `/usr/bin/sha256sum` on the GPU node before the image starts (exit 6); r7s has no model / workload check; OUT_MB 60 enforced per file (`ulimit -f`) and in total (`du -sb`, exit 7, evidence kept) | test_step_script_recon7 (no hasher outside CHILD; CHILD order; DRYRUN of every cell) |

### X1.1 What the cell measures

(a) HARDWARE CONCURRENCY CAPABILITY: can independent dense GEMM work (A = Aq + Ar), the fork attention (F), the zero-copy
H2D gathers (G) and a copy-engine H2D (M) run at the same time on this A100 at our shapes? Alone: A, F, G, M, FG, P.
Concurrent (independent streams): A|F, A|G, A|M, F|M, A|FG, A|F|M. MATCHED controls: X-SER = the same ops of X serial
on one stream in the same pass (A|F|M-SER = Aq, Ar, F, M). The isolated maxima are an IDEALIZED reference only.

(b) ORDERED SYNTHETIC SCHEDULING SCREEN: the joint pass's per-layer CALL ORDER Aq, G, F, Ar on synthetic, INDEPENDENT
inputs. F does not read G's destinations or Aq's output; Ar does not read F's output. This is an ordered-call screen and
a copy-interference test. It is NOT an end-to-end dependency validation, NOT a useful-prefetch result, NOT an achieved
bound of the production joint pass. Producer / consumer and reader / DMA lifetime validation is a separate test that is
not built.

| schedule | streams (per layer, in this order) | matched control | scope | work counted once? |
|---|---|---|---|---|
| GRP | stream 1: Aq0, G0, F0, Ar0 (requests 0 .. B/2 - 1); stream 2: Aq1, G1, F1, Ar1 (requests B/2 .. B - 1) | GRP-SER (consistency: GRP0 + GRP1) | each group's outputs, lists, tables and destinations are its own; shared read-only: weights, arena, pinned host KV. ELIGIBLE for a claim only when GRP-EQ passes (the groups' outputs = the full batch's, bit for bit: the request-neighbour invariant) | NO: the full batch reads the weights once per layer (nominal 474 MB = 473,956,352 B in bf16), the two halves read them twice (948 MB); the split cost = GRP-SER - SER, MEASURED; the net = SER - GRP |
| GRPL | as GRP, both streams join after every layer | GRP-SER | as GRP | as GRP |
| CPY | stream 1: Aq, G, F, Ar; stream 2: P = a copy-engine copy of the current pass's NEW-block byte count into its own device buffer, overwritten every layer, never read | CPY-SER (consistency: SER + P) | copy interference beside the ordered calls; not a prefetch (nothing consumes P) | yes (P is extra transfer) |
| WINF | window attention beside the selected-block fetch | - | - | NOT MEASURED: the shipped fork's 2D launch writes no log-sum-exp (fork :1060-1063) |

### X1.2 What each mode can and cannot show

- (a) CAN show whether the GPU runs these kernels and the copy engine concurrently at our shapes, against the same ops in
  series, and the copy-engine interference A|M against the model's s = 8.685 %. It CANNOT show overlap INSIDE one
  kernel, and it CANNOT say whether the joint pass can use the overlap (no data dependency between its streams).
- (b) CAN show what the listed ordered-call schedules do on synthetic independent inputs of the measured volumes, with
  their measured costs (the groups' second weight read). It CANNOT validate producer / consumer dependencies, buffer
  lifetimes against a DMA, a useful prefetch, the unreplayed parts (selection, residency, twin, norms, graph launches),
  the real miss pattern, group skew, WINF, or any schedule not listed.
- Neither mode measures the model's max(HBM, dense). A SERIAL result in these schedules does NOT falsify every route to
  it: intra-kernel overlap, fused kernels, other groupings and other launch configurations stay open.

### X1.3 Pins

| pin | value |
|---|---|
| repository | crafticat/NOSA (git remote `fork` of `/mnt/beegfs/ojerbi/nosi-r2-recon7`), branch `r2-recon7` |
| code commit | `2d3e57fa3b52a58dfc86751b5770dddfaf93fc44` (the repaired code; parent `028a5ecf1de346f527f38393c55b1e79ea367569`) |
| run commit (`--nosi`) | the grant commit S; precondition: `git -C /mnt/beegfs/ojerbi/nosi-r2-recon7 diff --name-only <code commit> S -- nosi ':(glob)benchmarks/Efficiency/*.py' ':(glob)benchmarks/Efficiency/*.sh'` lists only `benchmarks/Efficiency/r2_recon7_protocol.py`, and that file's diff is the one line `GRANT_r7s = "<date -u stamp>"` |
| step script | `benchmarks/Efficiency/r2_recon7_step.sh` sha256 `158ee291e959949b2b25347673e5a4d9f2aaa31d2304e43ef688dbaee1a4a89e` (computed on amd node thunder09), git blob `71b13f6d6feb792b060959f3f75ca70d8474af1b` (the script refuses to run unless its bytes are that file at S: cmp against `git show`) |
| cell | `benchmarks/Efficiency/r2_recon7_streams.py` sha256 `11a7c0462f58e0a881f83b35efba318ed2599f32a69988a88a029059405a297b`, blob `2c623283ed378e4a0a00323446ed33d3953faeae`; `r2_recon7_protocol.py` at the code commit sha256 `1ede15d38a724afcf55059e8232538cfc4224e4c9d09e39eb928a11fe0f6f731` (every GRANT None) |
| fork kernel | `benchmarks/Efficiency/fork_vllm_unified_attention.py` blob `f0dc0c183a388585e216505a6bbfc696531e8add` (fia6; sha256 `2fdbd75c4e7a6bf07ca543a3fc6a38c921e7e92ddf41f0147a765250ba80d07c`; the step script checks the blob) |
| gathers | `nosi/nosi/flash_cache_engine/flash_h2d_mask.py` blob `f76b4f61b754adcc7cdc8eaeb22e1e96c0b74b0d`, `flash_h2d_mask_bias.py` blob `1e3630b87b84c658115aa1bd25c69c5afb03aa0e` |
| image | `/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif` (23,079,374,848 B) sha256 `14bfb15e1258821e637b5936f2060d7c7c2be20410fdee2a54ee0ffef754734a` (the record `/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif.sha256`, 2026-09-03); CHECKED inside the step on the GPU node by `/usr/bin/sha256sum` before the image starts (exit 6 on a mismatch) |
| shapes | C 128: B 216, E = 4 B = 864 rows; C 73: B 318, E = 1,272 rows (run in this order); GEMMs wqkv 4608 x 4096, wo 4096 x 4096, gate_up 32768 x 4096, down 4096 x 16384 (bf16; 473,956,352 B of weights per layer); fork: Hq 32, Hkv 2, D 128, 4 rows per request, 64 pages per row, row shifts (0, 3, 5, 8), union 72, last_len 40 + r, FORK_LAUNCH num_warps 4, num_stages 3, tile_size 64; gathers: 1.226 / 5.0 NEW blocks per (layer, KV head, request) (MEASURED kv_at_verify, RECON7_R2_joint.md section 2.1) = 530 / 3,180 per layer (window 265 / 1,590, store 265 / 1,590), 17,434,880 / 104,609,280 B per layer; 16 host blocks per request; groups: requests [0, B/2) and [B/2, B) |
| seeds | `--seed 7`; shape seed = 7 + C (135 for C 128, 80 for C 73); ids seeds shape seed + 1 .. + 4; inputs: torch CPU generator, N(0, 1) x 0.05 in bf16 |
| passes | 32 layers per pass; 26 configurations; 2 warm-up passes of each, then 10 ORDER-ROTATED passes (pass r starts at configuration r mod 26; lesson L62); every pass kept |

### X1.4 Command and resources

```
cd /mnt/beegfs/ojerbi/nosi-r2-recon7
DRYRUN=1 bash benchmarks/Efficiency/r2_recon7_step.sh --nosi <S> 2195630 <run id> r7s     # pins + the command
bash benchmarks/Efficiency/r2_recon7_step.sh --nosi <S> 2195630 <run id> r7s
```

The step expands to `srun --jobid=2195630 --job-name=recon7-r7s-<run id>-2195630 --ntasks=1 --cpus-per-task=16
--mem=64G --gres=gpu:a100:1 --time=20 --export=NONE` (inside `timeout 1260`), the CHILD on the GPU node (one GPU, the
uuid, the image sha256, `ulimit -f 61440`), apptainer with the pinned image, and `/venv/nosa/bin/python
<tree>/benchmarks/Efficiency/r2_recon7_streams.py --out /mnt/beegfs/ojerbi/r2-runs/spec/recon7_<run id>_2195630_r7s
--cell r7s --shapes 128:216,73:318 --new-blocks 128:1.226,73:5.0 --union-pages 72 --win-share 0.5 --layers 32 --reps 10
--warmup 2 --seed 7 --max-shape-s 240 --max-run-s 600`.

- GPU: ONE A100 in the EXISTING holder SLURM job 2195630 (nova05, GPU uuid f31fc723-9dfe-407a-9947-28173556232a from
  `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/2195630.gpu_uuid`). No new allocation, renewal or replacement.
- Holder availability is checked FRESHLY by the wrapper at the launch: `squeue -j` (owner, RUNNING, time left >= STEP_MIN
  x 60 + 600 = 1,800 s), no running step (`squeue -s`, before and after the claim), the atomic claim directory
  `/mnt/beegfs/ojerbi/r2-runs/spec/sp_active_2195630`, new output / record / cache paths. A read at 2026-10-31T11:08:12Z
  (`date -u`): 2195630 and 2195631 RUNNING since 2026-10-30T13:58Z, 14:49:38 left each, no step; 2197383 / 2197384
  PENDING (Dependency). The packet's "idle since 11:04Z" (README of 028a5ec) was stale and wrong for these jobs (they
  started at 13:58Z); this read replaces it.
- Hard wall time: STEP_MIN 20 (srun --time=20; host timeout 1,260 s); each shape stops at 240 s from its start, the run
  at 600 s. MODEL duration: the image hash 1 - 4 min (23 GB from beegfs); per shape about 6 s per rotation of the 26
  configurations at C 73 (the sum of the expected pass times) x 12 + references and compile: below 150 s.
- Device memory (MODEL, `need_bytes`): 13.9 GB (C 128) and 19.8 GB (C 73), one shape at a time. Host: 16 cpus, 64G;
  pinned host buffers below 0.6 GB.

### X1.5 Output limit

OUT_MB 60 for r7s, ENFORCED: per file by `ulimit -f 61440` (KB) on both sides (srun's stdout / stderr written on the
login side; every file the child writes on the GPU node: summary.json, the Triton cache files); in total by `du -sb` of
`/mnt/beegfs/ojerbi/r2-runs/spec/recon7_<run id>_2195630_r7s`, its `.step` record, `/mnt/beegfs/ojerbi/nosa-env/jit/recon7_<run id>_2195630_r7s`
and `/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_jit/recon7_<run id>_2195630_r7s` after the step, written to the
record's exit.txt; above the limit the step exits 7 and keeps every file. MODEL: summary.json below 1 MB (2 shapes x 26
configurations x 12 passes), the record below 1 MB, the cold Triton cache below 50 MB.

### X1.6 Duplicate-work evidence

No earlier run of this harness exists (checked 2026-10-30T18:47Z, `date -u`; re-checked by the wrapper at the launch:
the output, record and cache paths of the run id must not exist): `ls` of every directory under
`/mnt/beegfs/ojerbi/r2-runs/` (41 directories), filtered for recon7 / stream / concur / x1 / r7s: 16 names (8 runs
with their `.step` records) matched, all run ids of other cells (`sg5_x1_2190876_x1` launched `r2_tick_sg5.py`;
`sp_x13b`, `sp_x1m` launched `r2_spec_cell.py`; `sp_x173_*`, `mbs_o2x128_*` launched `r2_base_cell.py`: the
`Efficiency/*.py` name in their `command.txt`; `sg5_bgb_*` and `sg5_bgbr_*` matched on the batch name "x1344"); `grep -l
recon7 */*.step/command.txt` under `/mnt/beegfs/ojerbi/r2-runs`: 0 files (no step record of any run names a RECON7
file). The R2 traces (one
dependent stream: 0.000 ms concurrency) cannot answer either question.

### X1.7 Gates, decision rules, falsifiers, STOP THE RUN (fixed before the run)

MUST GATES (any failure: every decision of the run is printed as SUPPRESSED; the timings stay in the record): `<shape>-UNION` (the
fork's union_len = 72 at every (KV head, request), full batch and both groups); `<shape>-IDS` (the NEW blocks per layer
as above); `<shape>-<config>-EXACT` for all 26 configurations (every part's output bits after the configuration's last
pass = that part's reference run alone); `<shape>-BUDGET` (the TRUE wall from the shape's start, setup included, to its
last gate <= 240 s); `<shape>-RUN`; `RUN-WALL` (the run's wall <= 600 s).
DECISION GATE: `<shape>-GRP-EQ` (the groups' outputs, concatenated by request, = the full batch's, bit for bit, for Aq,
Ar, F, G: the request-neighbour invariant). It governs the GRP / GRPL claim only (INELIGIBLE when it fails).
MATCHED CONTROLS: X-SER for every concurrent set X (the same ops, one stream); GRP-SER for GRP and GRPL; CPY-SER for CPY.
CONSISTENCY: each control within 5 % of its independent sum (X-SER: the parts alone; GRP-SER: GRP0 + GRP1; CPY-SER:
SER + P); otherwise its conclusion is INVALID (not "suspect").
DECISION RULES (the values are REPORT; the rules fixed now):
- (a) REACHABLE when T(A|F|M) <= 1.10 x max(A, F, M) (the isolated maxima: an IDEALIZED reference); SERIAL when
  T(A|F|M) >= 0.90 x T(A|F|M-SER) (the MATCHED control); PARTIAL otherwise; eff_matched = (control - T) / (control - max)
  printed for every concurrent set. Interference A|M = T(A|M) / max(A, M) - 1 against the model's s = 8.685 %.
- (b) GRP / GRPL: SCREEN SAVING when eligible (GRP-EQ), GRP-SER consistent, and the net (SER - T) >= 5 % of SER (the
  weights read once in SER, twice in the groups). CPY: COPY ABSORBED when (SER + P - CPY) / P >= 0.80 and CPY / SER - 1
  <= 5 % (copy interference only; nothing consumes P).
FALSIFIERS: (a) SERIAL at both shapes falsifies "independent streams of these kernels at these launch configurations
overlap on this A100"; it does NOT falsify the model's max(HBM, dense) through intra-kernel overlap, fused kernels, other
groupings or other launch configurations. (a) REACHABLE with (b) GRP NO SCREEN SAVING or INELIGIBLE: stream overlap
exists, but this request-group split of the ordered synthetic calls does not use it (nothing about producer / consumer
dependencies follows). CPY NOT ABSORBED: a copy-engine copy of this byte count is not hidden beside the ordered calls or
slows them by more than 5 %.
STOP THE RUN (no later shape; the record says why): free device memory below the need; a UNION or IDS failure (before
any timing); an EXACT failure; the shape's deadline passed in setup, references, warm-up or timing; an exception. The
step script stops before the launch on any pin mismatch, a busy or ending holder (less than 1,800 s left), an existing
claim or path, or a missing grant; the child stops (exit 6) on an image sha256 mismatch on the GPU node.

### X1.8 What the result can and cannot decide

- CAN decide: whether stream-level concurrency of these kernels and the copy engine exists at our shapes, against the
  same ops in series (R2 D1, first half); the copy-engine interference against the model's 8.685 %; whether the
  request-group split of the ordered synthetic calls saves time and whether its outputs stay bit-equal.
- CANNOT decide: the end-to-end speedup of the joint pass, any producer / consumer or DMA-lifetime property, a useful
  prefetch, any intra-kernel overlap, WINF, the unreplayed parts, anything about the drafts' compute.

## 3. THE GPU PROTOCOL of the other cells (r7pm, r7pa, r7pb, r7mem: NOT GRANTED; each needs its own GRANT line)

Each subsection states what the run decides, its expected duration (STEP_MIN and its basis), its gates and its
falsifier. Order: r7pm first (its MUST gates decide whether r7pa / r7pb run); r7s and r7mem are independent of (a) and of each
other. One holder, the shared claim `OUTROOT/sp_active_<holder>` (one step at a time). MODEL total: about 3.4 h of one
A100 (10.5 + 60.5 + 91 + 5 + 35 min); the sum of the STEP_MIN limits is 390 min.

Command form: `bash benchmarks/Efficiency/r2_recon7_step.sh --nosi <sha> <holder job> <run id> <cell>` from the tree at
`<sha>`. DRYRUN=1 prints the pins and the command and then STOPS on a missing grant like a real launch.

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

SUPERSEDED for the run by section "X1 protocol (about 5 min)" above (the refined cell: capability and attainable
screen, matched controls, order rotation, --reps 10 --warmup 2; repaired after review 3). The text below is the first version's
summary, kept for the record.

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
cell (the tap, the fit, the arguments); the cell's arguments; the cell grants; the step script (refusals, DRYRUN, the
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
