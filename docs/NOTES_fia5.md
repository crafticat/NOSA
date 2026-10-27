# R2 FIA5: let Triton pipeline K / V inside the vLLM fork (plan Task 0.3b, the fia4 follow-up)

REGISTERED 2026-10-27T21:11:01Z (lead stamp, before any run)

Builder stamp: 2026-10-27T20:41:30Z (`date -u`). Status: SOURCE ONLY, NOT RUN, NOT REVIEWED.
Branch r2-fia5, worktree /mnt/beegfs/ojerbi/nosi-r2-fia5, from r2-fia4 344f7d0 (the fia4 fork, stamped and run).
Evidence: /mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/SCORING_fia4.md (record fia4_k1a_2195573), docs/NOTES_fia4.md.

## 1. Scope and the rule

- The human's rule stays: ONE fork (nosi/nosi/fork_vllm_unified_attention.py), upstream's structure, every changed
  line tagged `# [fork]` with its reason. No other kernel. This change edits the fork only. It adds no kernel and no
  @triton.jit function (test FV3 of test_r2_fia4.py checks the set of jit functions and the two launched kernels).
- The goal (the lead's brief): remove the synchronous loads from the fork's tile loop, so that Triton's pipeliner can
  prefetch K / V as it does for upstream; prefetch the CIS bias too; cut registers and spills where the math allows;
  add a launch sweep and a no-mask timing arm.
- Exactness stays: the MUST gates of FV (C, FI1-FI3, FI5, CW, MW, U_FV_table, G_block0) and the REPORT
  X_row_alone_FV keep their meaning. Each swept configuration runs the same gates under its own name.

## 2. What the fia4 TTGIR says (MEASURED: the record; INFERENCE: my reading of Triton 3.2)

File: /mnt/beegfs/ojerbi/nosa-env/jit/fia4_k1a_2195573/triton/lgsUvvbF0cKuP0kKu1oYrYnjzkTBgQYr9DF9YN-JGUo/
kernel_unified_attention.ttgir (num_warps 4, num_stages 3, shared 59392 B).

- MEASURED: lines 1524-1527 allocate ONE buffer each for K (1x128x64), V (1x64x128) and the two page-id copies
  (1x64xi32). The page ids of tile j+2 are async copies (lines 1748-1763). The K / V copies of tile j+1 start after
  both dots of tile j (lines 1734, 1743), and the same iteration waits for them (line 1766, num = 2). All 8 compiled
  variants of the record have the same 1-deep loop buffers (8 x 1x128x64xbf16, 8 x 1x64x128xbf16, 16 x 1x64xi32).
- MEASURED: three loads in the loop are synchronous `tt.load`: the page ids again in the MMA layout (line 1625), the
  bf16 CIS bias (line 1634; it feeds the QK dot's accumulator), and the scalar extents word of tile j+1 (line 1682),
  whose value masks the K / V copies (lines 1695-1696, 1729, 1738).
- INFERENCE (Triton 3.2 MatmulLoopPipeline, from memory of its source; the C++ source is not on this disk): the
  pipeliner gives a load the stage (max indirection level - its level) x ceil((num_stages - 2) / (max level + 1)), the
  dots num_stages - 1, and every load the buffer count max(use stage - load stage). With the page-id -> K / V
  indirection (level 1) and num_stages 3: page ids stage 0, K / V stage 1, dots stage 2, so ONE buffer. This fits the
  record exactly (all four allocations are 1-deep). Upstream vLLM has the same page-id -> K / V indirection, so at its
  default num_stages 3 it also gets one K / V buffer; upstream hides latency with many small programs per SM.
- INFERENCE: the pipeliner skips a load whose lanes are narrower than 32 bits (cp.async needs 4, 8 or 16 B). The bf16
  bias at token stride Hkv has 16-bit lanes, so it stays synchronous, and so does its MMA-layout page-id load (level 1
  under a skipped load). A scalar load (the extents word) is never pipelined.
- So, per tile, fia4 waits for three dependent global round trips besides the K / V copy: page ids -> bias before the
  QK dot, and the extents word before the K / V copies of the next tile. fia5 removes all three.
- num_stages 4 gives K / V 2 buffers by the same rule (dots at stage 3, K / V at stage 1). At tile 64 that is about
  90 KB of shared memory: 1 program per SM. At tile 32 it is about 52 KB: 2 or more programs per SM. The sweep tests
  both; the bench reports the buffer depth of every configuration (R_<arm>_compile).

## 3. Files

| file | what |
|---|---|
| /mnt/beegfs/ojerbi/nosi-r2-fia5/nosi/nosi/fork_vllm_unified_attention.py | the fork (1184 lines; 60 lines added, 23 removed vs fia4, all tagged) |
| /mnt/beegfs/ojerbi/nosi-r2-fia5/benchmarks/Efficiency/r2_fiattn_bench.py | [fia5]: FVN, --fv-sweep, the swept arms' gates (fv_suite), R_<arm>_compile, R_FV_sweep, PRED5, R_DECISION5, the FV bias as float32 |
| /mnt/beegfs/ojerbi/nosi-r2-fia5/benchmarks/Efficiency/r2_fia5_step.sh | the GPU step (new; a copy of r2_fia4_step.sh with the fia5 cells, arms and FV_SWEEP) |
| /mnt/beegfs/ojerbi/nosi-r2-fia5/benchmarks/Efficiency/tests_cpu/test_r2_fia5.py | the CPU tests F5_1-F5_9 (new) |
| /mnt/beegfs/ojerbi/nosi-r2-fia5/docs/NOTES_fia5.md | this file |

Unchanged and still run on this tree: tests_cpu/test_r2_fia4.py (FV1-FV10: FV2 = the [fork] tag check of the whole
fork, FV5 = the interpreter test with the fp64 reference, the planted mask error and the I15 canary region),
test_r2_fia3.py, test_r2_fiattn.py, r2_fia4_step.sh, docs/NOTES_fia4.md, PRED4.

## 4. The changed regions of the fork vs fia4 (line numbers of the fia5 file)

| region | fia4 | fia5 | why |
|---|---|---|---|
| header (:24-34) | change list | item 3 reworded; 5 lines on fia5 | the record of the change |
| globals (:245-248) | none | FORK_LAST_LAUNCH = {"kernel": None} | the launcher keeps Triton's CompiledKernel (the compile witness) |
| kernel argument (:469) | none | USE_EXTENTS: tl.constexpr | the FVN timing arm (False) |
| 3D segments (:519) | tiles = entries | segments over union ENTRIES; the loop bounds times TILES_PER_BLOCK (:561-567) | a tile can be a part of a block |
| tile loop top (:572-596) | scalar `tl.load(union_ext_ptr + block_table_offset + j)` (fork:561), scalar reach, tile_mask = offs_t < reach | tok (the token rows in the block); USE_EXTENTS: ext_t = a VECTOR load with physical_block_idx's index `seq_offset // BLOCK_SIZE` (:583), reach lane by lane, tile_mask = tok < reach (:590), ext_rows = a VECTOR load along the tile rows with the index `(j * TILE_SIZE + offs_m % TILE_SIZE) // BLOCK_SIZE` (:591), row_ext; else tile_mask all true (:596) | brief item 1: Triton pipelines vector loads (2 tiles ahead for ext_t, a level-1 load like the page ids; 1 tile ahead for ext_rows, with K / V); no scalar load gates a copy |
| K copy (:613) | mask = dim & reach | mask = dim only (the whole tile) | brief item 1: the score mask cuts the columns no row sees |
| V copy (:620) | upstream text, tile_mask = reach | upstream text, tile_mask = reach from the vector | FI5 (see below) |
| CIS bias (:653) | masked, bf16, synchronous | no mask; any float; float32 storage is pipelined | brief item 2 |
| score mask (:657-660) | offs_t < row_ext | tok < row_ext; FVN: all true | the tile can be a part of a block |
| _fork_inputs (:873) | shape check | + the bias is a float | the float32 bias |
| launcher (:949-950, :1018-1021, :1073, :1155) | num_warps / num_stages | + tile_size (None = the block), use_extents; tile asserts (16 .. block, divides it, a power of 2); the handle kept; USE_EXTENTS passed | brief items 3, 4 |

Unchanged: the union build (fork_build_union, fork_row_lower_bound), the Q load, the online softmax, ONE accumulator,
the K / V address arithmetic, the 2D and 3D epilogues, reduce_segments, the grid, the refusals.

Deviation from the brief, on purpose: the brief says "apply the extents ONLY to the score tile, not as a mask on the
K / V loads". K and the bias follow that. V does NOT: FI5 (MUST) poisons every store row past the request's reach
with NaN, inside loaded tiles (fv_partial_reach counts such entries), and p = 0 times a NaN V is NaN in the PV dot.
So V keeps upstream's mask line verbatim, with tile_mask = the reach. The reach now comes from a pipelined vector
(level 1, like the page ids), so the V mask adds no synchronous load. The interpreter test F5_6 J10 proves the line
carries FI5: a copy of the fork without it gives a non-finite poisoned output. J11 does the same for the score mask
(the K / bias columns past the reach ARE loaded now). FVN measures the cost of all extents work together.

The bias as float32: a 16-bit bias at token stride Hkv cannot be an async copy in Triton 3.2. A token-contiguous
bf16 layout does not help either: the page id is per-lane data, so the axis analysis sees contiguity 1. Upstream's
per-token-head scale caches are float32 [num_blocks, block_size, num_kv_heads], read the same way; fia5 takes that
dtype. The FV arms pass bias.float() (made in prepare, outside the timed launch). bf16 -> fp32 is exact, so the math
does not change (F5_6 J5: bitwise equal). Cost: +2 B per token and KV head (+0.4 % of the K + V + bias bytes). For
production this is a storage decision for the lead; the swept arm FVW4S3T64B16 measures the bf16 form at the default
launch.

The token tile (tile_size): upstream's TILE_SIZE choices are 16 / 32. A 32-token tile halves the S, P, K and V
register tiles and the K / V shared buffers, with one accumulator and the same rows per program. It changes the
grouping of the online softmax (rounding only). EXACT mode stays: a tile that a row does not see is an identity step
once the row saw block 0 (G_block0: block 0 is never a row's last page, so both halves are seen); F5_6 J3 and J14
check a row alone at tile 32, bitwise. Splitting the 64 rows (BLOCK_Q 2) is not done: two programs of one request
would race on the in-kernel union build, so it needs the torch build (FVT: 6.7-8.6 ms per tick, above the rule).

## 5. Arms, gates and the sweep

| arm | what | role |
|---|---|---|
| P | piece 3 v2 (7c826c1, blob 39ed5e18) | the reference arm |
| FV | the fork, default launch (Triton's num_warps 4 / num_stages 3, tile 64), float32 bias, in-kernel build | the candidate |
| FVT | FV's launch reading a torch-built table | timing: ncu FV - FVT = the in-kernel build |
| FVN | FV without extents (use_extents=False) | timing only: ncu FV - FVN = the extents' cost; R_FVN_not_attention (REPORT, DETECTED expected) |
| FVW<w>S<s>T<t>B<b> | one per --fv-sweep item: num_warps w, num_stages s, tile t, bias b bits | the same gates as FV under its own name |

- --fv-sweep grammar: `w<warps>s<stages>[t<tile>][b<16|32>]`, comma-separated; tile default 64, bias default 32;
  warps 2 / 4 / 8 / 16, stages 1-6, tiles 16 / 32 / 64. It needs the arm FV. The swept names go after --arms.
- Gates per swept arm (fv_suite, the same code as FV): C, FI1, FI2, FI3, FI5, CW, MW (its own ones-twin with the same
  launch), U_<arm>_table (MUST); X_row_alone, R_GRAPH, R_D vs P, R_D vs FV, R_<arm>_compile, R_<arm>_first_call
  (REPORT). A first call that fails (compile, resources) is a REPORT FINDING; that arm then runs no gate and leaves
  the cell. COMPLETE expects each swept arm's first-call marker, and its gates when the call ran.
- R_<arm>_compile (REPORT): num_warps, num_stages, registers and local memory (4-byte words) per thread from ptxas,
  shared memory, the loop's multi-buffered allocations from the TTGIR (`DxAxBxdtype`, D = the buffer depth), and a
  MODEL of programs per SM. On the fia4 TTGIR the parser gives depth 1 and 2 programs per SM (F5_5), as measured.
- R_FV_sweep_<cell> (REPORT): per configuration the event median, x of P, MUST, registers, local words, shared memory,
  depth; the best = the lowest event median among the swept arms whose MUST gates all pass.
- ncu per cell: P, FV, FVT, FVN and the cell's best swept arm (5 arms, 6 kernels). The ncu children also collect
  the WarpStateStats section (warp stall reasons) into the raw CSV (ncu_<cell>_raw.csv), because SCORING_fia4.md
  section 8 lists their absence. The R_ncu line prints one of them per arm: the long-scoreboard stall share of
  active warps (smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct, an optional metric; it is in the ga100
  metric list of the fia4 record). It is the direct witness of the latency claim: it should fall from fia4 to fia5.
- R_DECISION5 (REPORT): PRED5's rule for FV and the best swept arm (below). R_DECISION4 (PRED4) still prints.

## 6. Registered prediction and the decision rule

The lead stamps the marker at the top of this file before the run. The json block is the bench's PRED5 (test F5_7
keeps them equal). The bands are the lead's suggestion in the brief; the rule is PRED4's.

<!-- PRED5-BEGIN -->
```json
{
 "registered": "REGISTERED 2026-10-27T21:11:01Z (lead stamp, before any run)",
 "source": "lead's fia5 brief 2026-10-27 (SCORING_fia4.md sections 5-6: let Triton pipeline K / V inside the fork)",
 "fv_over_p_device": {"216x4": [0.85, 0.6, 1.05], "216x4s1": [0.85, 0.6, 1.05]},
 "best_swept_over_p_device": {"216x4": [0.85, 0.6, 1.05], "216x4s1": [0.85, 0.6, 1.05]},
 "thresholds": {"fv_over_p_max": 0.8, "union_build_ms_per_tick_max": 1.0, "layers_per_tick": 32},
 "decision": "ADOPT a launch configuration C of the fork (the default FV, or the cell's best swept configuration: the lowest CUDA-event median among the swept arms whose MUST gates pass) if, at B 216 x 4, C's device time <= 0.8 x P device time (ncu) AND the union build <= 1 ms per tick (32 layers; 32 x (FV - FVT) device time at the default launch) AND C's MUST gates pass. When both qualify, the lower device time. Else NOT MET, with the ncu split and the compile witness (registers, spills, shared memory, K / V buffer depth) written down."
}
```
<!-- PRED5-END -->

How the bench reads the rule (R_DECISION5): X / P = the ncu gpu__time_duration.sum of X's kernel over P's two
kernels; the build = 32 x (FV - FVT) ncu (else the event difference); X's MUST gates = the gates with `_X_` in the
name and G_block0 of the cell. Note: the best swept arm is chosen on CUDA events (median of 200) in the same run,
then scored on ncu; the choice can favour a lucky configuration by the event noise (about 1-2 %).

Builder expectations (MODEL, NOT registered; for the lead's judgement only):
- Witnesses (TTGIR, R_<arm>_compile): default FV: K / V depth still 1 (the indirection rule, section 2), but no scalar
  `tt.load` of the extents in the loop, and the bias and its page ids become async copies (4 B lanes). num_stages 4:
  depth 2. Tile 32: about half the registers of S / P / K / V; spills near 0 at 4 warps; shared memory about 52 KB at
  num_stages 4.
- Default FV / P at 216x4: central 0.7, about [0.5, 0.95]. Reason: fia4 spent about 8.4 us per tile per program with
  about four dependent global round trips; fia5 leaves one (the K / V copy). If the rest is about 3 us, FV falls by
  about 0.6x. The HBM floor is 0.436 (SCORING_fia4.md section 5).
- Best swept / P at 216x4: central 0.6, about [0.45, 0.9]; most likely w4 or w8 with s4 and tile 32 (2 K / V buffers
  at 2 or more programs per SM).
- FVN / FV: 0.85-1.0 (the extents work is small once nothing waits on it). FVW4S3T64B16 / FV: 1.0-1.3 (the bias is
  synchronous again).
- The build per tick: as fia4 (<= 0.36 ms; the build code does not change).
- X_row_alone: 0 rows differ for every configuration.

## 7. EXACT mode (unchanged in principle)

- A tile is one block (default) or a part of one (tile_size). A row's visible tokens come in ascending block id and
  token order, alone or in its request. A tile with no visible token for a row is an exact identity step once the row
  saw block 0. The other rows change only which tokens are loaded; the row's scores of those tokens are -inf (p = 0).
- New in fia5: K and bias columns past the reach are loaded (whole tiles) and cut by the score mask; V rows past the
  reach read 0. A row alone and in its request see the same non-identity steps, so EXACT holds (the signed-zero
  exception of NOTES_fia4 section 7 stays).
- num_warps changes the layouts of the row reductions; the bits of a swept arm can differ from FV's (R_D vs FV is a
  REPORT). Exactness inside one configuration does not depend on it.

## 8. CPU tests for the lead (a compute node, inside the image; git HEAD printed before and after, lessons L37)

    cd /mnt/beegfs/ojerbi/nosi-r2-fia5
    git rev-parse HEAD
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \
        benchmarks/Efficiency/tests_cpu/test_r2_fia5.py benchmarks/Efficiency/tests_cpu/test_r2_fia4.py \
        benchmarks/Efficiency/tests_cpu/test_r2_fia3.py benchmarks/Efficiency/tests_cpu/test_r2_fiattn.py
    git rev-parse HEAD

- test_r2_fia5.py: F5_1 the fia5 tile loop (static: vector extents loads, no scalar one, K mask dim only, V mask
  upstream's, no bias mask, the new keywords, the handle, tags); F5_2 the sweep flag (names, defaults, refusals,
  parse_args, the ncu child's arms and argv, make_fv_arm); F5_3 every swept configuration runs the same gates
  (run_fv with a pure-torch fake of the fork; a planted missing gate fails COMPLETE; a failing first call is a REPORT
  FINDING); F5_4 R_FV_sweep and R_DECISION5; F5_5 the compile witness on the fia4 TTGIR lines and the ncu
  long-scoreboard stall field; F5_6 the fork under
  TRITON_INTERPRET=1 (J1-J14: tile 32 and 16, the bf16 bias, FVN, the 3D path at tile 32, ragged requests, the two
  mutation controls, refusals, run_fv on swept arms with the real kernel); F5_7 PRED5 = this file's block; F5_8 / F5_9
  the step (static shape; DRYRUN, the full DRYRUN only on a clean checkout of this worktree).
- test_r2_fia4.py runs on THIS tree's fork: FV2 (every changed line tagged, two plants) and FV5 (the interpreter test:
  fp64, the planted mask error, FI1-FI5, the row alone, the I15 canary region) must stay green. FV9 / FV10 test
  r2_fia4_step.sh (unchanged; its full DRYRUN skips outside the fia4 worktree).
- test_r2_fia3.py and test_r2_fiattn.py must still pass (the bench changed by additions; parse_arms still refuses XX).

## 9. The step and the GPU time

    bash benchmarks/Efficiency/r2_fia5_step.sh --nosi <full sha of r2-fia5 HEAD> <holder job> <run id>
    DRYRUN=1 bash benchmarks/Efficiency/r2_fia5_step.sh --nosi <sha> <holder job> <run id>

- srun into the existing holder: 1 task, 16 CPUs, 64G, gpu:a100:1, --time 45; host timeout 45 min + 60 s.
- Cells 216x4, 216x4s1 (the rule's cells), then 32x4 (continuity with fia4). Arms P, FV, FVT, FVN.
- FV_SWEEP=w4s2,w4s3,w4s4,w8s2,w8s3,w8s4,w4s3t32,w4s4t32,w8s3t32,w8s4t32,w4s3b16: the brief's grid (num_warps 4 / 8 x
  num_stages 2 / 3 / 4, tile 64), tile 32 at num_warps 4 / 8 x num_stages 3 / 4, and the bf16 bias at the default
  launch. FVW4S3T64B32 repeats FV's launch: a same-run noise check.
- Pins as fia4: piece 3 (7c826c1, blob 39ed5e18); the fork = this tree's file (HEAD and clean-tree checks; its blob
  checked again before the launch); the S1 tables by path and size. Output /mnt/beegfs/ojerbi/r2-runs/fiattn/
  fia5_<run>_<holder>/ and .step/, both frozen. Claim fiattn_active. Caches under /mnt/beegfs/ojerbi/nosa-env/jit/
  fia5_<run>_<holder>/ (the Triton cache keeps every configuration's TTGIR).

| part | estimate (MODEL; fia4 ran in 149 s of bench time) |
|---|---|
| FlashInfer JIT (the decode variant only; no FP arm) | 15 s |
| Triton compile: 14 FV-family launches x about 5-7 variants (B 216 / 32 / 1, gen / s1 list width, build / table) x about 7 s | 8-12 min |
| 3 cells (P + 15 FV-family arms: gates and timing) | 2-5 min |
| 3 ncu children (5 arms each) | 1-2 min |
| total | about 12-20 min; the budget is 2400 s inside STEP_MIN 45; a cell starts only with 300 s left |

## 10. Open items and risks

1. GPU compile: the interpreter does not check the GPU lowering (L36). A failure of FV is ARM_FV (MUST); a failure of
   a swept configuration is a REPORT FINDING.
2. The pipeliner reading (section 2) is INFERENCE from memory of Triton 3.2's source, fitted to the record; the run's
   TTGIR (R_<arm>_compile and the Triton cache) is the witness. If ext_rows does not pipeline, its load is still at
   the top of the iteration and gates no copy (only the score mask after the QK dot).
3. num_stages 4 at tile 64 needs about 90 KB of shared memory: 1 program per SM. num_warps 8 may need more than 128
   registers: 1 program per SM. The sweep measures both.
4. The float32 bias is a storage change for production (+0.4 % bytes); a decision for the lead.
5. The V mask deviates from the brief (section 4); FVN measures it with the rest of the extents work.
6. The best swept arm is chosen on events, then scored on ncu (section 6).
7. JIT time is the largest part of the step (section 9). If the budget runs short, 32x4 is the cell that is skipped
   (it runs last). A skipped cell is the MUST gate BUDGET_cell_<tag> (as in fia4), so the step then exits non-zero;
   the rule's cells (216x4, 216x4s1) run first and keep their gates.

## 11. Shell commands the builder ran outside cat / grep / head / tail / ls / sort / cut / diff / git / wc

- `date -u +%FT%TZ` (the stamp above).
- Shell builtins and syntax: cd, echo, `;`, `&&`, variable assignments (T=..., F=...). Four assignments carried the
  names awk_free, sed_unused, sed_never and sed_unused_never by mistake, as environment prefixes of a tail / head
  command or alone: no awk or sed program ran.
- `uniq -c` once, BY MISTAKE, at the end of a read-only pipeline over the fia4 record's TTGIR files (it counted the
  loop buffer shapes of the 8 compiled variants: 8 x 1x128x64xbf16, 8 x 1x64x128xbf16, 16 x 1x64xi32, so every fia4
  variant has one K / V buffer). This breaks the "no uniq" rule. The output was read only.
- `printf` and a `for` loop over line numbers with head / tail (to print cited TTGIR lines).
- No interpreter, no sed, no heredoc, no `cat >>`, no tr / paste / xargs, no hashing, no `find /`, no srun / sbatch /
  pytest, no sleep. Files were written with the Write and Edit tools; the commit message with the Write tool.
