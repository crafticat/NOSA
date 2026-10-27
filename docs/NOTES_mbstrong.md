# NOTES: r2-mbstrong, the strengthened baseline (`--recap-nogc` and the best batch per cache size)

- Inputs: `kstep-cpu-runs/analysis/plan/SCORING_p0all.md` section 5 (the re-capture cost), `SCORING_dr73.md` section 8
  (C 73 / C 81 are not memory-matched), `/mnt/beegfs/ojerbi/nosi-r2-graphstep/NOTES_graphstep.md` sections 2-3 and 12
  (the re-capture inventory, FIX 3), `kstep-cpu-runs/paper-notes/speckv/speckv.txt` section 4.1 ("each system may
  choose its cache capacity and batch size within that budget").
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-mbstrong`, branch `r2-mbstrong`, from `ded9d41` (cached NOSA, cell mb1).
- Status: SOURCE ONLY. No interpreter, no pytest, no job ran (login-node rule). The lead runs the CPU tests and the GPU
  cells. No date stamp: no `date -u` ran in this pass.
- Labels: CODE = read at file:line. MEASURED = a value in a cited record. DERIVED = hand arithmetic on MEASURED values.
  INFERRED = my reading, not measured. NOT VERIFIED = a fact that I could not read on the login node.

## 1. What changed

| file | change |
|---|---|
| `benchmarks/Efficiency/r2_recap_nogc.py` | NEW. `GraphNoRelease` (torch.cuda.graph without the release on entry), `check_contract`, `release`, `capture_swap`, `ABRecorder` (the instrument of the A/B loop), `logit_digest`, `rn_gc_gate`, `compare_ab`, `load_ref`. No kernel. |
| `benchmarks/Efficiency/r2_base_cell.py` | `--recap-nogc`, `--ab-record`, `--ab-ref DIR`, `--fit-probe` (all default off, `--arms SH` only); `run_shipped_ab` + `shipped_loop_ab`; the pure parts `int32_batch_limit`, `fit_batch`, `rewarm_stats`, `effective_excl_first`, `is_oom`, `mbs_verdicts`; `PRED_MBS` (TBD). Flag off: `main` calls `run_shipped` (unchanged) and writes the same summary keys. |
| `benchmarks/Efficiency/r2_mbstrong_step.sh` | NEW. Copied from `r2_base_fused_step.sh`, with the sub-cell sequence of `r2_base_profile_step.sh` (p0prof). Cells mbs0, mbs1, mbx73, mbx81, mbx63, mbx128, `c<C>b<B>n`. One claim per holder. |
| `benchmarks/Efficiency/tests_cpu/test_mbstrong.py` | NEW. CPU tests MBS1-MBS7. |

The engine is NOT edited: `nosi/` keeps the bytes of `ded9d41`. Two reasons. (1) `tests/test_r2_engine_findings.py:91-110`
pins line numbers of `nosa_llama.py` (459, 612, 800, 817-828), and `test_r2_g0.py` W5 pins the end of the module (the
R2 switch must be the last statement, `apply_rotary_pos_emb` at line 917). (2) The step script can then pin `nosi/` to
`ded9d41` (flag off = the shipped engine, byte for byte), as p0prof did.

## 2. Part A: `--recap-nogc`

### 2.1 The cost (MEASURED and CODE)

- One shipped re-warm takes 5.53-5.75 s. gc takes 0.93-0.97 of it. Every re-warm row has gc_count [0, 0, 32]
  (SCORING_p0all.md:139-146).
- `nosa_llama.py:459-460` makes one `torch.cuda.CUDAGraph()` and opens one `with torch.cuda.graph(...)` per layer, in
  `decode_forward_warmup`. So one re-warm opens 32 contexts.
- torch 2.6.0 `torch/cuda/graphs.py:171-183` (read in `/mnt/beegfs/ojerbi/venvs/vllm-cu12/lib/python3.10/site-packages`;
  the image has 2.6.0+cu124): `__enter__` runs `torch.cuda.synchronize()`, `gc.collect()`, `torch.cuda.empty_cache()`,
  then `self.stream_ctx.__enter__()` and `self.cuda_graph.capture_begin(*self.pool, capture_error_mode=...)`.
  `__exit__` runs `capture_end()` and then the stream context exit.

### 2.2 The change

- `GraphNoRelease` is a subclass of the shipped class. Its `__enter__` is the last two statements of the shipped
  `__enter__`. It keeps the shipped `__init__` (the same default side stream: `__init__` creates it on the SHIPPED class
  if it does not exist; no pool; `capture_error_mode` "global") and the shipped `__exit__`. So the capture uses the
  shipped stream context and the raw `CUDAGraph.capture_begin` / `capture_end`, as the graphstep tree does
  (`nosi/nosi/graph_step.py` capture_all).
- `check_contract()` reads the INSTALLED torch with `inspect` + `ast`. It refuses the run (exit 20, before the model
  loads) unless the shipped `__enter__` is exactly the five statements of torch 2.6.0, `GraphNoRelease.__enter__` is
  exactly the last two, `GraphNoRelease` defines no `__exit__`, and `__init__` takes the 2.6.0 parameters.
- `capture_swap()` runs ONE release (the same three calls, in the order of `__enter__`), then sets
  `torch.cuda.graph = GraphNoRelease` for ONE decode call (the re-warm step), and puts the shipped class back in a
  `finally`. It refuses a nested swap.
- `shipped_loop_ab` is the loop of `run_shipped` with three additions, all through `ab`: `with ab.step(t, need):` around
  the decode call, `ab.after(t, need, lg)` after it, `ab.finish(lg)` after the timed loop. On a steady step `ab.step` does
  nothing. On a re-warm step it counts the `gc.collect` calls (a counting wrapper for that call only) and the
  `GraphNoRelease` contexts, and under `--recap-nogc` it runs `capture_swap`. MBS1 checks by AST that the rest is
  `run_shipped`.
- The release is inside the re-warm period (after `clock.record(ev[t])`), so the re-warm period holds the whole
  re-capture cost in both modes.

Why not "torch.cuda.graph with gc disabled around the loop": `gc.disable()` stops the AUTOMATIC collector only. An
explicit `gc.collect()` runs a full collection also when the collector is disabled (Python documentation of
`gc.disable` and `gc.collect`; MBS3 shows it). The 32 collections are explicit calls in `graph.__enter__`, so
`gc.disable()` removes none of them. Other forms that I did not use: a no-op patch of `gc.collect` (it keeps 32
synchronize + empty_cache per re-warm, 47-82 ms, SCORING_p0all.md:139-146, and it hides every other collection);
`gc.freeze()` (the 32 calls stay; the brief asks for one per re-warm).

### 2.3 Why the graphs and the logits stay the same (argument; the tests check it)

- The three release calls enqueue no work on the capture stream and write no tensor that the captured body reads. The
  body records the same kernels with the same arguments, on the same stream, with the same `capture_begin` arguments.
- The capture does not execute. `replay()` (nosa_llama.py:491, :612) runs on the current stream after the context
  closes, so the order of the eager work before it does not change. The shipped synchronize before each capture is
  not needed for that order.
- Addresses inside a graph's private pool can differ (the allocator state differs). INFERRED: the body allocates no
  memory, because its outputs are the buffers of nosa_llama.py:819-828 (`out=` top-k, in-place scatter / fill), and
  torch.topk uses its single-block path for these slice sizes (about 260 columns; NOT VERIFIED in ATen on the image).
- CPU: MBS4 compares the captured bodies (the same code object, the same `capture_begin` arguments, equal scalars,
  equal buffer layouts, equal input values), every logit and the final state, in three worlds (the loop of
  `run_shipped`, plain, `--recap-nogc`).
- GPU: RN-EQ compares every token (B x 513), the logits digest of every re-warm step and the digest of the last step,
  between the two halves of an A/B pair. The digest is the int64 sum of the int32 bit patterns of the fp32 logits of
  each request: one ULP moves it.

### 2.4 Gates (new; the old gates and predictions are unchanged)

| gate | when | MUST | checks |
|---|---|---|---|
| RN-CONTRACT | `--recap-nogc` or `--ab-record` | yes (a failure refuses before the model loads) | the record of `check_contract` |
| RN-GC | the same | yes with `--recap-nogc`, REPORT in a plain run | one row per re-warm; `--recap-nogc`: 1 `gc.collect` call, 32 `GraphNoRelease` contexts, a release time; plain: 32 `gc.collect` calls, 0 contexts |
| RN-EQ | `--ab-ref DIR` | yes | equal meta (B, steps, S, C, pool, splits, workload, rows), different modes, every token, the re-warm steps, a digest at every re-warm step, every digest, the last-step digest |
| SH-RUN | `--fit-probe` and an out-of-memory error | no (REPORT) | `fit` = "oom"; any other error stays a MUST failure |

The summary gains `SH.ab`, `SH.rewarm_stats` (first, later median / mean, sum), `SH.rates.effective_tok_s_excl_first`,
`meta.{recap_nogc, ab_record, ab_ref, fit_probe, fit, graph_contract, mbs_predictions, fit_model}` and `AB`, all only
under the flags. A REPORT line "REPORT RN ..." prints steady median and mean, effective tok/s (with and without step 0),
the re-warm times, the peak and the model.

## 3. Part B: the cells

### 3.1 The batch arithmetic (`r2_base_cell.fit_batch`; MBS6)

- Memory: the prefill-peak model of mbase, `19.0 + B x (0.155 + (C - 63) x 0.002105)` GB (allocated). It holds within
  0.03 % at three measured points outside or at its fit: C 73 B 318 74.962 (model 74.984), C 81 B 290 74.923 (74.938),
  C 128 B 216 82.042 (82.034) (DERIVED from SCORING_dr73.md:182 and mb1).
- Budgets: "matched" = 82.04 GB, the measured peak of mb1 (cached NOSA at C 128 B 216); "edge" = 84.0 GB, the budget.
  C 128 B 224 (model 84.37) went out of memory in 2175791, so the device edge lies between 82.04 and 84.37 GB.
- int32 envelope (cache_engine.py:216-221, asserted when P > 0): B x (S + steps + 64) x 256 < 2^31 gives B <= 502 at
  S 16128, 512 steps; the GPU side, B x (C + 1) x 64 x 256 < 2^31, gives B <= 1016 at C 128. Neither binds here. (At
  128K the same formula gives 63, p0prof's B.) At P = 0 the engine does not assert; the gathers widen their ids
  (flash_h2d_mask.py:23-27).

| C (pool) | paper B (Figure 2) | matched B (peak model GB) | edge B (peak model GB) |
|---|---|---|---|
| 63 (0) | 344 (72.32) | 406 (81.93) | 419 (83.95), not run |
| 73 (10) | 318 (74.98) | 358 (82.03) | 369 (83.96) |
| 81 (18) | 290 (74.94) | 326 (81.88) | 336 (83.81) |
| 128 (65) | 216 (82.03) = mb1 | 216 | 222 (83.79) |

The brief named "+24 and +48" as examples. I used the derived values: matched is the B whose peak equals the peak of the
cached-NOSA reference cell; edge is the 84 GB budget.

### 3.2 Cells (`r2_mbstrong_step.sh`; every sub-cell: mb1's python arguments at its C and B, 512 steps)

| cell | sub-cells (role, mode) | flags | STEP_MIN | --mem | GPU time (estimate) |
|---|---|---|---|---|---|
| mbs0 | c128b16 (first, plain), c128b16n (second, `--recap-nogc`) | pair | 21 + 21 | 240G | 6-8 min |
| mbs1 | mb1 (plain), mb1n (`--recap-nogc`) | pair | 38 + 38 | 240G | 29-31 min |
| mbx73 | c73b318, c73b318n (pair); c73b358n, c73b369n (probes) | pair + 2 probes | 46 + 46 + 49 + 50 = 191 | 340G | 80-84 min (the edge probe about 2 min if it fails at the prefill) |
| mbx81 | c81b290, c81b290n (pair); c81b326n, c81b336n (probes) | pair + 2 probes | 44 + 44 + 47 + 48 = 183 | 340G | 75-79 min |
| mbx63 | c63b344, c63b344n (pair); c63b406n (probe) | pair + 1 probe | 48 + 48 + 53 = 149 | 340G | 62-66 min |
| mbx128 | c128b222n (probe) | NOT in the brief: the edge of C 128, so that every C has its 84 GB point | 38 | 240G | 15 min |
| `c<C>b<B>n` | one probe (a re-run, or another B near an out-of-memory edge) | probe | 20 + B / 12 | by B | about 3.0 s x B + 3 min |

- Pair: both halves get `--ab-record`; the second gets `--ab-ref <first output>` (RN-EQ). An even holder runs the
  plain half first, an odd holder the `--recap-nogc` half first (Task 0.4).
- Probe: `--recap-nogc --fit-probe`, no plain twin. An out-of-memory SH arm is a REPORT (exit 0), and the sequence
  continues. Any non-zero exit or an incomplete record stops the sequence.
- WITH AND WITHOUT the flag: both at mb1 and at the paper B of every C (the pairs). The matched and edge B run only
  with `--recap-nogc` (time). The flag acts only on re-warm steps, so the steady statistics of a probe are the steady
  statistics of the shipped decode; the plain effective tok/s of a probe can be DERIVED from its steady periods and
  the plain re-warm periods of the pair at the same C (label it DERIVED).
- Time basis (DERIVED): the prefill is 2.97-3.04 s per request (dr73 943.8 s / 318, dr81 878.5 s / 290, dr1
  657.4 s / 216); mb1 took 14:43 in p0all (.9); the model load is 1-2 min; the decode is 503 steady steps plus 9
  re-warms (5.5-6 s plain, about 0.3-0.7 s later re-warms with the flag, INFERRED). STEP_MIN = 20 + B / 12 (about 2x).
  MIN_LEFT = the sum of STEP_MIN + 10 min: mbx73 needs 3 h 21 min left on the holder.
- Total of all cells about 4.5 h of A100 time. Suggested order: mbs0 (inspect), mbs1, then mbx73, mbx81, mbx63
  (two holders can run two cells at once: the claim is per holder), mbx128 last.

Each sub-cell reports: steady tok/s (median: `SH.rates.steady_tok_s`; the median and the mean step: `SH.steady`),
effective tok/s (and without step 0), peak allocated / reserved, device memory used at the end, the re-capture times
(`SH.rewarm_ms`, `SH.rewarm_stats`, and per re-warm the release ms under the flag in `SH.ab.rows`), and `meta.fit`.

## 4. REGISTERED predictions: TBD (the lead)

`r2_base_cell.PRED_MBS` holds `None` for every value. Every verdict prints UNREGISTERED. The lead writes the values and
the stamp (`date -u` in the same command, L33) in a commit before the run. MBS5 accepts both states (all None, or a
stamp and all values set).

Builder's suggestions (NOT registered):

| name | suggestion | basis (DERIVED unless marked) |
|---|---|---|
| rn_steady_ratio | (0.99, 1.01) | the flag changes only re-warm steps; the steady statistic excludes them; P0-HEAD same-job 0.9989 (MEASURED) |
| rn_eff_ratio | {"128": (1.8, 2.3), "73": (1.45, 1.85), "81": (1.5, 1.95), "63": (1.3, 1.6)} | remove 31 x 167.6 ms gc + about 40 ms of sync / empty_cache per re-warm from the plain totals: C 128 91.4 -> 44.3 s (2.06x); C 73 122.5 -> 75.4 s (1.62x); C 81 110.8 -> 63.7 s (1.74x); C 63 B 344 152.6 -> 105.5 s (1.45x) |
| rn_rewarm_later_ms | (200, 1000) | later plain re-warm 5,550-5,950 ms minus about 5,240 ms (MEASURED gc share per p0all row) |
| rn_rewarm_first_ms | (4000, 12000) | first plain re-warm 10.3-14.2 s holds one-time Triton / cuBLAS work (a new JIT directory per sub-cell) |
| mx_steady_tok_s | C 128 B 216 (3330, 3420); B 222 (3300, 3480); C 73 B 318 (2540, 2630), B 358 / 369 (2480, 2700); C 81 B 290 (2830, 2920), B 326 / 336 (2780, 2980); C 63 B 344 (1870, 1930), B 406 (1840, 1960); B 16 REPORT | MEASURED at the paper B (mb1 / dr1 / vr1, dr73, dr81, p0all); above B 300 the step grows about linearly with B (fetch and attention per request, GEMM compute-bound), so tok/s stays flat |
| mx_peak_ratio | (0.99, 1.01) | three measured peaks within 0.03 % of the model |
| mx_fits | matched B: True; edge B: no point (the device edge is not measured; that is the purpose of the probes) | C 128 B 224 OOM at model 84.37; reserved minus allocated was 2.3-2.6 GB at C 73 / C 81 against 0.33 GB at C 128 (SCORING_dr73.md:182) |

What the cells decide (INFERRED from the suggestions): if the steady tok/s at C 73 / C 81 stays flat in B, the
memory-matched small-C baseline stays below cached NOSA at C 128 B 216 (about 2.6k / 2.9k against 3.39k tok/s), and
the strengthened cached-NOSA baseline (with the flag) reaches about 2.5k effective tok/s instead of about 1.2k.

## 5. CPU tests for the lead

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_mbstrong.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_mbase.py
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_r2_engine_findings.py benchmarks/Efficiency/tests_cpu/test_r2_g0.py

- MBS1 flag off = shipped (parse_args defaults; the import and the dispatch only under `if ab_on`; `run_shipped_ab` +
  `shipped_loop_ab` = `run_shipped` by AST without the three ab calls; every new summary key under `if mbs_on`; the
  shipped harness functions = the AST of ded9d41; `nosi/` and `r2_ragged_gate.py` = the bytes of ded9d41).
- MBS2 the context contract (the real torch.cuda.graph and GraphNoRelease with logged CUDA primitives: the same calls
  minus sync / gc / empty, the same stream, the same capture_begin arguments; three planted contract failures; the swap
  is undone after an exception and refuses a nested swap).
- MBS3 `gc.disable()` does not stop an explicit `gc.collect()` (why the brief's second option was not used).
- MBS4 [pool 4, 0] the CPU world (the graphstep fakes; the gathers, the pool swap and diff_offload on FLAT ADDRESSES,
  L28): three worlds, 100 steps, S 6128 (first / out_len / width re-warms; the pool hits and parks at P 4). Logits at
  every step, the final state, every captured body; per re-warm `gc.collect` NLAYERS times as shipped and ONCE with the
  flag (counted by a monkeypatched `gc.collect`); RN-GC and RN-EQ pass; the digests equal the digests of the ref logits.
- MBS5 compare_ab / rn_gc_gate with planted mismatches, logit_digest (one ULP), rewarm_stats, is_oom, mbs_verdicts.
- MBS6 the batch arithmetic (the cited lines of cache_engine.py; 502 / 63; the B table above; the three peaks).
- MBS7 the step script (static; malformed cells STOP before any pin; at a clean checkout of the ROOT, DRYRUN of mbx73,
  mbs1 at an odd holder, c128b222n and mbs0, with `MBS_HOLDERS` = a temporary uuid directory; skipped elsewhere, as MB4).
- The last command re-checks that the engine line pins still hold (nosi/ is unchanged).

## 6. GPU commands (after the review and the registration; a holder with its uuid file)

    DRYRUN=1 bash benchmarks/Efficiency/r2_mbstrong_step.sh --nosi <full sha> <holder> <run id> mbs0
    bash benchmarks/Efficiency/r2_mbstrong_step.sh --nosi <full sha> <holder> <run id> mbs0

Then mbs1, mbx73, mbx81, mbx63, mbx128 with the same pattern. Records:
`/mnt/beegfs/ojerbi/r2-runs/mbstrong/mbs_<run>_<holder>_<sub-cell>{,.step}`; claim `.../mbs_active_<holder>`.

## 7. Risks

1. The swap sets a torch attribute (`torch.cuda.graph`) for one decode call per re-warm. Any other user of
   `torch.cuda.graph` inside that call also gets `GraphNoRelease`. CODE: inside the SH decode only nosa_llama.py:460
   uses it (grep of nosi/nosi and benchmarks/Efficiency); the R2 arm (r2_capture.py:185) cannot run with these flags
   (`--arms SH`).
2. Memory during the captures. The shipped path returns the free cached memory to the driver before each of the 32
   captures; the flag does it once per re-warm. A graph's instantiation uses driver memory outside PyTorch's allocator
   (graphstep FIX 3: gs1 went out of memory in `capture_end` with 0.4 GB free, for 95 large graphs). Here there are 32
   graphs of about 6 nodes each. The old graph of a layer is freed when the layer makes its new one (nosa_llama.py:459),
   as in the shipped path, but its private pool goes back to the driver only at the next release (the next re-warm).
   INFERRED: those pools are empty or small (section 2.3). The risk is highest at the edge B; those are REPORT probes.
3. The automatic collector can now run inside a capture window (the shipped path resets its counters 32 times per
   re-warm). A collection frees Python objects; PyTorch defers the events of blocks freed during a capture (NOT
   VERIFIED on the image). The shipped path has the same window between its own captures.
4. RN-EQ compares two PROCESSES. p0all T-EQ proved token equality between two arms of ONE process only. If RN-EQ fails
   while RN-GC passes, run a plain / plain pair (two plain sub-cells under a new run id) before blaming the flag.
5. The peak model is the PREFILL peak at S 16128. Reserved memory was 2.3-2.6 GB above allocated at C 73 / C 81 (dr73,
   dr81) and 0.33 GB at C 128 (dr1). The edge probes can go out of memory below the 84 GB model value. They are REPORT.
6. Effective tok/s includes the first re-warm (a new Triton cache per sub-cell). The record also gives effective tok/s
   without step 0.
7. `check_contract` refuses any torch whose `graph.__enter__` differs from 2.6.0. That is on purpose.
8. Holder time: a sequence needs the sum of its STEP_MIN + 10 min left. Split the cells (or run single probes) on a
   shorter holder.
9. The digest needs an int64 copy of 32 logits rows at a time (about 19 MB) at each re-warm step and after the loop.
   Plain mb1 without the flags does not run it; the A/B halves both run it.
10. Nothing here ran. The CPU tests are written blind. The CPU world reuses the graphstep fakes (GS4 passed at
    0159e82); the flat-address versions of the gathers, the pool swap and diff_offload are new.

## 8. Shell commands of this pass outside cat / grep / head / tail / ls / sort / cut / diff / git / wc

- `sed -n '1,0p' /dev/null` FOUR times, each in front of a grep in the reading phase (by mistake). It prints zero
  lines of /dev/null. This BREAKS the "no sed" rule. sed read no repository file and edited nothing.
- `git` (worktree add, status, diff, rev-parse, log, show, ls-files, branch --list, add, commit -F) and `wc` are in the
  list. Shell builtin: `cd`. Redirections `2>&1`, `2>/dev/null`.
- No interpreter, awk, heredoc, `cat >>`, tr, paste, uniq, xargs, hashing, find, srun, sbatch, pytest or sleep. Every
  file was written with the Write / Edit tools.
