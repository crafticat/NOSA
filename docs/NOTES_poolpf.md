# NOTES r2-poolpf: Task 1.2, pool prefetch with CPU packing and a soft pin (SOURCE ONLY, not built, not run)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`, Task 1.2 (paper 3.1, 3.3,
  4.4; MAP_cached_nosa.md hooks (b) and (d)).
- Worktree: `/mnt/beegfs/ojerbi/nosi-r2-poolpf`, branch `r2-poolpf`, from `ded9d41` (cached NOSA, mb1).
- No new kernel. No change to `cache_engine.py`, `nosa_llama.py`, `flash_cache_engine/`, `r2_base_cell.py`,
  `r2_base_fused_step.sh` or the old `r2_transport_gilfree.py` (the test `test_flag_off_is_shipped` checks this with
  `git diff --quiet ded9d41`).

## 1. Files

| file | what |
|---|---|
| `nosi/nosi/pool_prefetch.py` | NEW. The opt-in module: `PoolPrefetch` (attach / detach, begin_step, prefetch, report), `derive_cpupack`. |
| `nosi/nosi/r2_transport_gilfree_v3.py` | BYTE COPY of `nosi-r2-f3ldx` `3769b64:nosi/nosi/r2_transport_gilfree.py` (copied with `git show`, compared with `diff`: identical). |
| `benchmarks/Efficiency/gilfree_pack.cpp`, `gilfree_pack.py` | BYTE COPIES of the same commit (packer v3: C++ worker, `issue_desc`, DESC 2 flat stores). |
| `benchmarks/Efficiency/r2_poolpf_cell.py` | NEW. The pf1 cell: G1 PF-HIT, G2 PF-EQ, T timing; the flag `--pool-prefetch`, `--fetch`. |
| `benchmarks/Efficiency/r2_poolpf_step.sh` | NEW. A copy of `r2_base_fused_step.sh` with the cell `pf1` and a STOP while the predictions are TBD. |
| `benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py` | NEW. The CPU tests (section 3). |
| `benchmarks/Efficiency/tests_cpu/pool_prefetch_twins.py` | NEW. CPU twins of diff, pool_update, pool_swap and the two h2d kernels (flat addresses, L28). Not a test file. |

Why a new file name for the transport: this tree already has an OLDER `r2_transport_gilfree.py` (v2 era, per-layer
sources) with its own test (`tests_cpu/test_r2_transport_gilfree.py`, pinned to v2 blobs). The f3t version changed
the geometry (one source per (layer, request)) and the stats. Its own test needs f3t's `r2_residency_sc.py` (the
generation board), which this tree does not have. So we copy it next to the old one and change no tested file.

## 2. The design in short

1. `prefetch(items)`: items are (layer, KV head, request, host block id). Duplicates removed; one staging LANE per item
   and KV head; ONE `issue_desc` job (worker "cpp", the flat-store option `xpool=_Lanes`).
2. The C++ worker packs host rows into pinned staging (OpenMP, no GIL). The copy engine moves each chunk to the landing
   ring. The transport's `index_copy_` writes the rows into the GPU STAGING buffer: lane j of layer l, rows
   `(j * 64 + r) * H + h`.
3. `begin_step()` of the next step COMMITS the job on a side stream (the commit stream), per layer: wait the arrival
   (the job event); wait the LAST READER of the layer's pool slots; drop items already in the window or the pool;
   choose the least recently used UNPINNED pool slot per item; copy the rows with the SHIPPED scatter kernels
   (`flash_h2d_from_mask` for K, `flash_h2d_from_mask_bias` for V and the bias; source = the staging view with batch
   stride 0); write `_pool_map = X` and `_pool_age = PIN_BASE + counter` (2**60: the soft pin); record the commit
   event. The main stream waits it before the layer's next update.
4. The next `pool_update` finds X in the pool: an ordinary pool hit. The shipped swap moves X into the window. The
   hit rewrites the age to a real stamp, so the pin ends.
5. The bias does not cross PCIe. The commit gathers the bias rows from the GPU-resident `total_cis` into a small bias
   staging buffer (same lanes), and the shipped bias kernel copies them with V.

Decisions that the lead must know:

- EVENT PER SLOT. The only readers of pool slots are the layer's `pool_update` and `flash_pool_swap`, ONE launch per
  layer and step, and the host cannot know which slots they read without a sync. So every slot of layer l has the
  same last reader: `reader_ev[l]`, recorded on the main stream after the layer's shipped update. A per-slot table
  would hold this same event in every entry. The commit stream waits it before the scatter and the metadata write.
- NOT A HARD PIN. When every unclaimed slot of a row is pinned, `pool_update` evicts the oldest pinned slot. We count
  it (`pin_evicted`). A commit never overwrites a pinned block: an item without an unpinned slot is dropped and
  counted (`commit_overflow`). Conservation while attached: `committed == pin_hits + pin_evicted + pins_live`.
- CAPTURE STEPS. A re-warm step captures the pooling graph in the default "global" capture mode. A CUDA call of
  another thread (the C++ worker: event queries, event synchronize) can break that capture. So
  `begin_step(capture=True)` drains the transport, sends no job in that step, and `prefetch()` refuses a call in it.
  The cpupack path waits each job's SUBMISSION in the same layer, so the worker is idle before the next layer's
  capture.
- ONE EXTRA D2D PASS. The data goes landing ring -> staging (`index_copy_`, inside the copied transport) -> pool
  (`flash_h2d_from_mask`). The packer could write the pool rows directly (its store = `_k_gpu`, slot `topk + q`), but
  the task asks for the shipped scatter, and the slot choice and the bias need the device state at commit time. The
  extra pass is about 2 x the bytes of the job at HBM speed. Not measured.
- `--fetch cpupack` METHOD. `derive_cpupack` takes the SOURCE of the shipped `decode_update_has_kv_bias_pool` and
  replaces its two gather lines by `self._pf_fetch(kv_bias)`. It refuses when the two lines do not occur exactly once.
  The test compares the two sources line by line. The cpupack path needs one D2H of `_load_mask` and a host wait per
  layer: the CPU path cannot start before the miss list exists. That is the cost the A/B measures.
- `bg_commit "none"` (the pf1 BG arms): the background load ends in the staging buffer. It measures pack + copy engine +
  landing scatter, and it does not fill the pool with pins of random blocks (that would change the baseline's hit
  rate). `bg_commit "pool"` (G2's BGP64) runs the whole path; G2 checks it is bitwise exact.
- STAGING MEMORY at B 216: 2 x 32 layers x 128 lanes x 64 rows x 2 heads x 128 x 2 B = 268 MB (+ 1 MB bias staging,
  + 17 MB landing ring, + 17 MB pinned host). mb1's peak is 82.04 GB (A100 80 GB). The cell reports the peak.

## 3. CPU tests for the lead

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py

Run it in the image (`/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif`): it needs triton, transformers, a C++ compiler and the
CUDA headers (the packer build), as the f3t transport tests do. Not run by the builder.

FIX 1 (CPU job pf_bccd3b1_a, 3 failures at bccd3b1). Root cause: a TEST HARNESS defect, not an engine defect.
`prefill_update` allocates the window and the host rows with `torch.empty` and does not write all of them (the window
slots before their first fill, the tail slot past the tail length, the host rows past S). Two worlds built from one
seed held different leftover bytes there, and the tests compare whole tensors bitwise. Evidence: `_k_cpu` / `_v_cpu`
differed at step 0 although no path writes host rows before the first write-back; the failing print of test 2 ended
in bit-pattern values (3.6141e+03, 3.0794e-41) at the tail slot's unwritten last row; test 1 failed at step 0 on the
outputs (views of the whole window). The engine never reads those rows. Fix: `r2_poolpf_cell.define_undefined` gives
them one fixed value (4096.0) after the prefill, in the CPU worlds AND in the GPU gate PF-HIT (the CUDA caching
allocator reuses blocks too). New preconditions: two worlds of one seed are bitwise equal before step 0 (tests 1, 2,
5 and PF-HIT). No assertion was weakened; the cpupack plant must now change `_k_gpu` (before, leftover bytes alone
could satisfy "some state differs").

FIX 2 (CPU job pf_f832a3c_a: the three new preconditions failed on `_load_mask` only). `_load_mask` is PER-STEP
SCRATCH: `torch.empty_like` (cache_engine.py:230); `diff_offload` writes every slot of it in a well-formed step (old
hits -1, diff_offload_kernel.cu:44-48; the k-th new block to the k-th free slot, :77-98; #free == #new) before every
reader of the step (pool_update, the gathers, the cpupack fetch); nothing reads it between steps. The precondition
before step 0 now skips the list `SCRATCH = ("_load_mask",)` (the reason is written at its definition); every comparison
AFTER a step still compares it. We do not fill it with a value: that would hide a diff that does not write every slot
at step 0. The GPU gate PF-HIT's precondition never compared it.

| test | what it proves |
|---|---|
| `test_flag_off_is_shipped` | Gate 0. The shipped files are unchanged against ded9d41. The binding before attach and after detach is the shipped bound method. Attached with no request: 8 steps (one block edge, one write-back) leave every engine tensor and every output bitwise equal to a never-attached twin. The cpupack source differs in the fetch line and the name only. Refusals. |
| `test_prefetched_block_is_a_pool_hit` | X lands pinned with its host bytes; the next step finds it as a pool hit (SWAP / MOVE_IN at X's window slot), the host gather does not fetch it, the window holds its host bytes. Control: without the prefetch the gather fetches X. The attended windows of both runs are bitwise equal. |
| `test_soft_pin_overflow_is_counted` | P + 1 prefetches into one row: P pinned, 1 `commit_overflow`, no pinned block overwritten. Then 1 hit + 3 misses: `pool_update` evicts the 3 other pins; counters = an independent count from the action plan; conservation holds. Negative control `count=False`: conservation fails on the same plan. |
| `test_scatter_waits_last_reader` | A main-stream reader of pool slot q (a MOVE_IN of Y) that has not run: the commit waits it, the window gets Y. PLANT `skip_reader_wait`: the window gets X. Preconditions (L34): same slot q in both runs, the reader ran after the scatter in the plant run, X and Y bytes differ. |
| `test_cpupack_equals_gpugather_bytes` | 8 steps (block edge, write-back), cpupack with 4 lanes (chunks) vs the shipped gather: every engine tensor bitwise equal after every step; the shipped gathers never ran in the cpupack run; the packer took exactly the misses; step 0's chunk count. PLANT: the packer's lane shift (`set_plant_slot_shift`) changes the window (preconditions: two different misses in one chunk; the plant job had no error). |
| `test_cell_arms_args_and_verdicts` | The cell's arms, the opt-in flag rules, the cores, the per-arm summary, the verdicts (TBD band = REPORT). |
| `test_step_script_stops` | A wrong cell / argument count STOPS; at a clean checkout of the ROOT, DRYRUN pf1 STOPS while `PRED_STAMP` is None. |
| `test_h2d_twin_matches_triton_interpreter` | The REAL `flash_h2d_from_mask` / `_bias` under `TRITON_INTERPRET=1` (fresh process) with the production shapes (pool rows as a narrowed view, staging with batch stride 0) equal the twin. Skipped without triton. A failure here can also be an interpreter limit with stride-0 views; read its stderr. |

## 4. The C++ extension build

- `gilfree_pack.load(build_dir)` builds `gilfree_pack.cpp` with `torch.utils.cpp_extension.load(with_cuda=True)`,
  `-O3 -fopenmp`, no `.cu` file, no nvcc call. `TORCH_CUDA_ARCH_LIST` defaults to 8.0.
- The build directory must be ABSOLUTE and NOT under HOME (`check_build_dir`).
- In the GPU cell: `gilfree_pack.build_dir_from_env()` = `$TORCH_EXTENSIONS_DIR/gilfree_pack_v3`. The step script sets
  `TORCH_EXTENSIONS_DIR=/mnt/beegfs/ojerbi/nosa-env/jit/poolpf_<run>_<holder>_pf1/torch_ext` (a fresh directory per
  run: the build runs in every step, about 1-2 minutes; gate PF-BUILD).
- In the CPU tests: a pytest temporary directory (under `$TMPDIR`; it must not be under HOME).

## 5. The GPU cell pf1 (REGISTERED, predictions TBD for the lead)

    bash benchmarks/Efficiency/r2_poolpf_step.sh --nosi <sha> <holder> <run id> pf1      # DRYRUN=1 first

- G1 `PF-HIT`, `PF-HIT-CTRL` (MUST): the real kernels on two small engine sets (2 layers, B 2).
- G2 `PF-EQ-BGP`, `PF-EQ-CP`, `PF-PIN-CONS` (MUST): B 8, 64 steps, logits of every step bitwise equal to SH.
- `PF-BUILD`, `T-RUN` (MUST). With a failed MUST gate in G1 / G2 the cell skips T (`--stop-on-gate 1`).
- T: B 216, C 128, one prefill (about 11 min), 10 segments x 64 steps = 640 steps:
  `SH,PF0,BG64,BG192,CP,CP,BG192,BG64,PF0,SH`. Every arm against SH of the same job.
- Reported per arm: steady step and ratio to SH; cores (rusage) and the delta to SH; BG: bytes per step, delivery ms
  (median, p90, late count), backlog per step and growth per segment, skipped jobs; CP: fetch ms per layer and the host
  split; the counters; the worker's stats. Delivery instrument: an issue event on an idle stream, an arrival event
  after the job's completion event; a probe set after the landing is flagged `late` (an upper bound).
- PREDICTIONS: `r2_poolpf_cell.py` `PRED_PF` (9 bands, all None) and `PRED_STAMP` (None). The lead writes them, with
  the stamp from `date -u` (L33), before the run. The step script STOPS while `PRED_STAMP` is None.
- STEP_MIN 40, `--mem 240G`, `--cpus-per-task 16`. Estimate: load 2 min + build 2 min + G1 / G2 3 min + T 14 min.
- Evidence for the bands (EVIDENCE_q3_q5_q6.md Q3(a)): copy engine kappa 0.006-0.051; native packer +0.96 % (64 MiB,
  1 thread) to +3.43 % (192 MiB, 8 threads) at B 64 in the sustained harness; Triton gather on a side stream +72 to
  +453 %. Those are harness numbers at B 64, not this engine at B 216.

## 6. Risks and open points

- Nothing here was built or run (no interpreter on the login node). Every claim above is from code reading.
- The hooks add one event record per layer and step (PF0 measures it) and, for pinned layers, about six small kernels.
- The commit's dedup and slot choice run on the device with no host sync; the dummy column P takes the dropped items.
- `report()` and `detach()` synchronize the device (segment ends only).
- Not done here (later tasks): unpin at verify (Task 1.5), the draft miss list (Task 1.1), the per-request lengths.
