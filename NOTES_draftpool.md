# NOTES draftpool (branch r2-draftpool, from r2-mbase ded9d41)

Task 1.1 of `kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`: the draft step on the resident KV, as an
opt-in flag on the cached-NOSA engine (hook (a) of `MAP_cached_nosa.md` section 3). SOURCE ONLY. No test and no job
ran. No new kernel.

## Commits
- One commit on ded9d41: the engine flag, the DR arm of the cell, the step script, the CPU tests, this file.
  See `git log -1`.

## What changed
- `nosi/nosi/cache_engine.py`
  - A block at the END of the file (every line above keeps its number; `tests/test_r2_engine_findings.py` pins
    lines up to 782). It adds `DraftBlockEdge`, the draft step `CacheEngine.decode_update_draft_resident`,
    `CacheEngine.enable_draft_resident / set_draft_resident / draft_misses` and
    `InfLLMv2Cache.enable_draft_resident / draft_resident_allowed / set_draft_resident`.
  - The draft step runs the statements of the pooled decode up to the map commit (tail write, diff_offload,
    pool_update, pool swap, map commit), without the trace hooks. Then it does NOT run the two host gathers. It:
    1. computes `_draft_skip = _load_mask >= 0` (after pool_update) and copies `_load_mask` into `_draft_miss_ids`;
    2. does `_block_map.masked_fill_(_draft_skip, -1)` after the map commit;
    3. copies the live bias view into a scratch view (same shape and strides) and writes `avail_policy.MASK_BIAS`
       (-3e4) on the 64 rows of each skipped (slot, head). It never writes the live bias.
  - Miss list for Task 1.2: `engine.draft_misses()` returns `(slots (H, B, 64) bool, ids (H, B, 64) int64)`.
    `ids` is contiguous (the gather kernels read ids flat, L28) and holds -1 at every slot that was not skipped.
  - Refusals (each raises with the reason): P = 0; a transfer trace or an availability policy in the process; a
    draft before the first shipped step after the prefill (never-loaded slots hold `torch.empty` bytes, and
    0 x NaN = NaN under the mask); a draft step that fills the tail block (`DraftBlockEdge`: its write-back would
    copy draft rows to the host; Task 1.5 owns that). The cache-level `set_draft_resident(True)` checks every layer
    BEFORE the step starts, binds the shipped step on a refusal, then raises. The engine-level check inside the
    draft step is a second guard only: the layer tables of that layer moved already when it raises.
  - Flag off: nothing calls `enable_draft_resident`, so `decode_update` stays the bound method that `__init__`
    makes. An enabled engine with `set_draft_resident(False)` calls that SAME bound-method object.
  - One bias scratch and one skip buffer for all layers (about 7 MB at B 216): correct on ONE stream only.
  - The refusal of NOSI_AVAIL + pool at `__init__` (lines 141-158): the CODE is unchanged. The comment (still 11
    lines) now says why the draft flag may run with the pool and NOSI_AVAIL may not. Reason: the flag does not read
    NOSI_AVAIL, and its map edit removes the duplicate id. NOSI_AVAIL with the pool stays unsafe: the pooled
    decode never calls `on_diff` (the policy would be a silent no-op), and mech=stale keeps Y in the map while the
    pool holds Y.
  - Not changed: `avail_policy.py` (the plan listed "allow the pool": not needed, the flag does not use the
    policy) and `nosa_llama.py:627` (not needed: the engine returns the scratch bias as its third value, and
    decode_forward hands it to attention). The name of the pooled method still occurs exactly twice in the file
    (the guarded bind and the def), as the retroinfer-eval pool test requires.
- `benchmarks/Efficiency/r2_base_cell.py`: `--draft-resident` (needs `--pool > 0`, `--arms SH`), `--dr-k`
  (default 3), `--dr-gate0` (default 4), `--dr-bands DLO:DHI:RLO:RHI | TBD`. After SH, the DR arm: a new cache,
  the same prefill, the flag ON, teacher-forced on SH's tokens. Kinds per step (`dr_kind`): rewarm, g0 (t < 4),
  edge, ship (every 4th step after the prefix), draft. Gates: DR-RUN (MUST), DR-G0 (MUST: steps 0-3 give SH's
  tokens), DR-FIRST (REPORT). Predictions: DR-STEP, DR-RATIO (bands), DR-SHIP, DR-SKIP, DR-AGREE, DR-MEM (REPORT).
  Without `--draft-resident` the cell is unchanged.
- `benchmarks/Efficiency/r2_draftpool_step.sh` (new; the pattern of `r2_base_fused_step.sh`; ROOT
  `/mnt/beegfs/ojerbi/nosi-r2-draftpool`; OUTROOT `/mnt/beegfs/ojerbi/r2-runs/draftpool`; claim `drp_active`):
  - `dr1`: mb1 (C 128, B 216, 512 steps, `--mbase`) + the DR arm; STEP_MIN 60, 240G. It STOPS outside DRYRUN while
    `DR1_BANDS=TBD`.
  - `dr1m`: the miniature (B 8, 80 steps: one edge at t 63, one re-warm at t 64; no bands); STEP_MIN 20, 100G.
  - The REGISTERED block in the header: NOT STAMPED, every PREDICTION "TBD by lead". The lead fills the
    predictions, writes `DR1_BANDS` and stamps the block from `date -u` (L33).

## Pytest files the lead must run (compute node)
- `benchmarks/Efficiency/tests_cpu/test_draft_pool.py` (new):
  `env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_draft_pool.py`
  - Engine (B 1 and B 5): `test_flag_off_is_shipped` (static: the AST of ded9d41 is a prefix, the appended block
    adds only new names, `git diff --quiet ded9d41` on nosa_llama.py / avail_policy.py / transfer_trace.py /
    flash_cache_engine; behaviour: ded9d41 vs flag off vs enabled-all-shipped, bit-equal for 40 steps across a block
    edge; plant: one draft step differs), `test_draft_body_is_shipped_prefix`, `test_all_hit_draft_equals_shipped`,
    `test_pool_has_no_duplicate_ids` (also P 65 at T0 160; plants: keep Y in the map -> duplicate id, keep X ->
    the map lies), `test_skipped_blocks_listed` (the miss list = the shipped step's gather programs; plant: diff's
    raw mask), `test_bias_mask_is_scratch` (plant: mask the live bias in place), `test_refuses_block_edge` (plant:
    a draft before the first shipped step gives NaN), `test_enable_refusals`.
  - Harness and script: `test_dr_kind_schedule`, `test_parse_args_draft_resident`, `test_dr_summary_and_verdicts`,
    `test_step_script_dr1` (DRYRUN of dr1 and dr1m needs a clean checkout of this ROOT; else it skips).
  - The engine tests import the REAL `cache_engine.py` (HEAD, and ded9d41 from `git show`) with CPU twins of
    diff_offload, pool_update, the pool swap and the two gathers. The twins read index tensors flat and assert
    contiguity (L28). They need torch >= 2.1 (`torch.utils.deterministic`), git, and bash. They import
    `transformers.cache_utils` if it exists (else a stub). Expect about 1-2 min (pure-Python twins).
- Existing files that read the changed sources: `tests/test_r2_engine_findings.py` (line citations in
  cache_engine.py; every cited line was checked by hand to keep its number), `benchmarks/Efficiency/tests_cpu/
  test_r2_mbase.py` (r2_base_cell parse_args / verdicts; the defaults did not change).

## Risks
- Nothing ran. The CPU twins are my reading of the kernels. A twin error can hide an engine error. The pool-update
  twin follows `pool_update_kernel.cu` phase by phase; its atomicCAS winner is the lowest slot (no contest when ids
  are distinct).
- DR-G0 compares TOKENS of steps 0-3, not logits (B 216 logits are 63 MB per step; the GPU is near its limit).
  If DR-G0 fails, read DR-FIRST first: a prefill difference between two runs in one process is not the flag.
- No rollback (Task 1.5): a draft step writes its token's K, V and bias into the live tail slot and into the layer
  tables, and teacher forcing does not undo them. So the DR arm's later steps run on a state that drifts from SH.
  DR-AGREE and DR-SHIP measure that drifting state, not a speculative round.
- The draft step still runs pool_update and the pool swap (5.4 ms of the 64.8 ms ncu step). Only the gathers
  (7.6 ms) and their launches go. The draft adds per layer: one compare, one id copy, two masked fills, the skip
  copy and a 3.5 MB bias copy. The harness adds one stack + sum per draft step (the skip count) inside the period.
- One shared scratch for all layers: a Task 1.2 consumer on a side stream must not read it.
- The tail slot is never skipped only because NOSA always selects the tail block (diff takes the old-hit branch at
  slot 63). The shipped step has the same precondition.
- Memory: the DR arm adds about 7 MB (scratch) + 8 MB (miss lists) + 28 KB (skip buffer) at B 216. It runs after
  `free()`, so its peak is its own. mb1's peak was 82.04 GB on an 85.1 GB device.
- Time: dr1 runs two prefills (657.7 s each in 2190876) and two 512-step decodes: about 28 min. STEP_MIN 60.
- The step script is hash-pinned (`git hash-object` of itself against the commit). Run it only from this tree at
  the commit that the lead stamps.
- RULE BREACH by this builder (read-only, no file changed by it): `sed -n` was run 7 times and `od -c` once to
  READ files (see the final report). No interpreter, heredoc, hashing, `find /`, srun, sbatch or pytest ran.
