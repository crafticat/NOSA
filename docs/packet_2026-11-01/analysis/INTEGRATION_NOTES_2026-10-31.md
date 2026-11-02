# INTEGRATION NOTES 2026-10-31: J1 (F1 / F2 / F3 / F6) with ST1 / ST2 and BF1 in ONE arm

Author: the AREA 1 builder. READ-ONLY note (no code changed, no job run). Lead's request 2026-10-31 (after j1m PASSED).

Sources (the commits that this note cites; file:line at these commits):
- J1: branch r2-j1-int 9eb59ef, /mnt/beegfs/ojerbi/nosi-r2-j1-int/benchmarks/Efficiency/r2_j1_select.py
- ST1 / ST2: branch r2-st1 2bc0bca, /mnt/beegfs/ojerbi/nosi-r2-st1/benchmarks/Efficiency/r2_st1_commit.py, r2_st1_cell.py
- BF1: branch r2-bf1 9efd728, /mnt/beegfs/ojerbi/nosi-r2-bf1/benchmarks/Efficiency/r2_bf1_fetch.py, r2_bf1_cell.py
- Shared base: cd0f0c5 (RECON7 X1 notes); I4DC classes: r2_i4dc_compile.py (same blob in all three trees).

## 1. Summary

| hook | where it acts | ST1 (commit graph) | ST2 (twin advance) | BF1 (kept-store fetch) |
|---|---|---|---|---|
| F1 joint pooling | state `_layer` copy, joint branch (r2_j1_select.py:518-571; the select at :562) | COMPOSES | COLLIDES: both override `_layer` with a FULL copy (section 3.1) | COMPOSES after a class change (section 3.3) |
| F2 draft pooling | `_dbegin_a` + `_InlineJ1` in draft graph A(l) (:387-407, :216-224) | COMPOSES | COMPOSES (ST2 falls back to the base `_layer` for R = 1: r2_st1_commit.py:355-358) | COMPOSES (BF1's draft hook is `_resid_lean`, eager between A(l) and B(l)) |
| F6 locate | `_cc_locate` copy, in the joint section (:303-349) | COMPOSES | COMPOSES (other method) | COMPOSES (F6 reads `_block_map` / `_pool_map`; BF1 writes `kid` / `kage` / the store only) |
| F3 residency captured | `_J3Mix` `_gsection` copy, `_dstub` / `_dend_b` / `_dreplay`, StampFrame (:631-855) | COMPOSES (ST1 uses its own private pool) | COMPOSES only with the merged `_layer` (section 3.1) | DOES NOT COMPOSE (section 3.4); also no keep J3 arm exists (ARMS_J3 :628) |

Result: two integrated arms are possible, one per best arm.
- C 128 (non-keep, L4PTFDGUDCD base): F1 + F2 + F6 + F3 + ST1 + ST2. No BF1 (BF1 needs the keep store).
- C 73 (keep, L4PTFDGUDKWCD base): F1 + F2 + F6 + ST1 + ST2 + BF1. No F3 (no keep J3 arm, and BF1 needs an eager
  draft residency).

## 2. How each rung builds its classes (the facts the collisions come from)

- J1: `j1_classes(STF)` (r2_j1_select.py:237-242) builds `S = type(..., (_J1Mix, SB))` on the I4DC state SB and a
  runner `R_(RB)` whose `_make_state` re-classes the state the I4DC runner built (`T.__class__ = S`, :606-610).
  `_J1Mix` and the runner are LOCAL to `_make` (:252, :589-618): nothing exports them. J3 does the same on the J1
  classes (`_J3Mix`, :703, :815-854).
- ST1 / ST2: `st1_runner(R, mode, sneg, twin, tneg2)` (r2_st1_commit.py:407-444) wraps ANY runner R. In `__init__`,
  after `super().__init__`, it re-classes the state to `state_class(type(T), twin)` = `(TwinST2,) + (CommitST1, S)`
  (:398-404, :422). `CommitST1.commit` finds the inherited commit by MRO (:125-131). `TwinST2._st2_setup` finds the
  inherited `_layer` by MRO (:263-269) and uses it only for R = 1 or a record (:355-358).
- BF1: `bf_classes(STF)` (r2_bf1_fetch.py:686-691) builds `TentStateBF(SKD)` and `TentRunnerBF(RKD)` on FIXED bases:
  the I4DC keep arm's classes (:696). `TentRunnerBF._make_state` raises SpecError unless `type(T) is SKD` (:743-744),
  then re-classes to TentStateBF (:745). Hooks: `_resid_lean` (:704-708), `_resid_joint` (:710-714), `run_pass`
  (:716-722). `__init__` builds BgFetch on `self.T` and wraps `T.backend` with the capture lock (:728-735).
- I4DC: each runner's `_make_state` builds a NEW state object of its own class (r2_i4dc_compile.py:291-296, :317-323);
  `_wrap_dcapture` wraps `T.backend` begin / end / capture (:329-349).

## 3. Collisions and the fix for each

### 3.1 ST2 x F1: two full copies of TentState._layer (MUST fix; otherwise one hook is OFF without an error)

- J1's `_J1Mix._layer` (r2_j1_select.py:518-571) = spec_tentative.TentState._layer with the select line replaced by
  `self._j1_select(...)` (:562). The twin loop stays the shipped one (:540-554).
- ST2's `TwinST2._layer` (r2_st1_commit.py:352-388) = the same method with the twin loop replaced by `_st2_twin`
  (:378). The select stays the shipped `VR._select_rows` (:379).
- Neither calls the other for R > 1. With `state_class(J1 S, twin=True)` the MRO is (TwinST2, CommitST1, J1 S, ...):
  TwinST2's copy runs, F1 is OFF in every joint pass. If J1's mixin comes first, ST2 is OFF.
- Detection (already in the gates): J1-COMPILE needs `calls == joint passes x L` (r2_j1_select.py:891-893) and J1-PEQ
  needs checked layers > 0 (:888); the ST2 TWIN gate needs `layers == joint passes x L` and no fallback
  (r2_st1_cell.py:274-280). So a wrong order FAILS a MUST gate; it does not pass silently.
- FIX (integration module, no change to either rung): ONE merged copy, `_IntLayer._layer` = ST2's `_layer` with line
  :379 replaced by `self._j1_select(plan.vs, layer, q, twin, l)`; R = 1 or a record -> `self._st2_base(...)` (as ST2).
  Mark the 1 changed line "# INT" and test it like test_j1.py's layer copy test (the dedented body equals ST2's except
  the marked line). Put it FIRST in the MRO: re-class after RunnerST1's `__init__` to `(_IntLayer, type(T))`.
  `_st2_base` then is `_J1Mix._layer` (the next `_layer` after TwinST2), which is correct for R = 1 and records.
- Data check: J1's pooling reads `vs.n_used.view(E)`, `vs.M_b.view(E)` and `twin.ccis` after all rows stepped. ST2
  writes `plan.vs.n_used` and `plan.vs.M_b` as whole (B, R) copies (r2_st1_commit.py:343-344) and claims the twin
  state bitwise. So J1-PEQ and EQF stay the gates of the merged arm.

### 3.2 ST1 x BF1: the commit capture is not under BF1's capture lock (MUST fix for the C 73 arm)

- BF1's worker thread makes CUDA calls (event synchronize, r2_bf1_fetch.py:492-493, :503-504, :537-538; the gilfree
  C++ worker). A global-mode capture fails when another thread calls CUDA. BF1 therefore wraps `T.backend`
  (:306-337). ST1 captures with its OWN backend `CudaGraphsPrivate` (r2_st1_commit.py:65-83), in `T._st1["backend"]`,
  which BF1 does not wrap.
- Timing: the commit's capture is the 2nd commit (:157-168), after a joint pass; at that time the worker can still run
  late jobs of the closed epoch (each starts with an event synchronize under the lock, :492-500).
- FIX (integration runner, after both setups): `self.bf.wrap_backend(self.T._st1["backend"])`. CudaGraphsPrivate has
  `capture(fn, T, vs)`, the signature that wrap_backend wraps (:331-337); the drain of the gilfree transport then also
  runs before the commit capture.
- Memory: disjoint. ST1 writes the engine slots (arena blocks below `store_base`); BF1 writes the store (blocks at and
  above `store_base`): r2_kv_arena.py:59-68. BF1's docstring rule "the round's commit does not touch the store and the
  map" (r2_bf1_fetch.py:48-49) holds for ST1's row commit too (r2_st1_commit.py:206-229 writes `_k_gpu`, `_v_gpu`,
  `_kv_bias_gpu`, `_block_map[..., tail]`, `_cache_lens` only).

### 3.3 BF1 x J1: BF1's fixed base and its exact type check (MUST fix for the C 73 arm)

- `TentRunnerBF` subclasses the I4DC keep runner (r2_bf1_fetch.py:696, :724), not the J1 runner; `_make_state`
  requires `type(T) is SKD` (:743-744).
- Works WITHOUT a BF1 change: build the J1 runner ON TOP of TentRunnerBF (RB = TentRunnerBF, SB = TentStateBF). The
  call chain J1 `_make_state` -> BF1 `_make_state` -> RKD `_make_state` checks the type BEFORE J1 re-classes, so the
  check passes; BF1's `__init__` then builds BgFetch on the J1-re-classed state (it needs only `kid`, `kwin`, `arena`,
  :187-188, which J1 keeps).
- Needs a SMALL J1 change (mine, AREA 1): export the class factory. Today `_J1Mix` and the runner are local to
  `_make` (r2_j1_select.py:252, :589-618). Proposed: `j1_wrap(SB, RB, arm)` -> (S, R) = the body of the loop at
  :585-619, moved, no behavior change (the existing arms call it; test_j1 checks the same classes and gates).
  Same for `_J3Mix` (`j3_wrap`), for the C 128 arm.
- `_arm_name` is defined by J1, J3 and BF1 runners; the outermost wins. The integration runner defines its own.
- `read()` / `detach()` chain by super() in all three (r2_bf1_fetch.py:748-757, r2_st1_commit.py:431-440,
  r2_j1_select.py:612-616): no change.

### 3.4 BF1 x F3: does not compose (no fix in this round)

- F3 calls the REAL `_resid_lean` INSIDE the draft capture (`_dstub`: `del self._resid_lean`, then the call,
  r2_j1_select.py:773-777). With BF1, `TentStateBF._resid_lean` -> `bf.on_draft` would then run ONCE at capture:
  the snapshot copy and the D2H into pinned memory become graph nodes that replay into ONE fixed snapshot slot; the
  host part (slot ring, job queue, r2_bf1_fetch.py:428-443) runs only at capture. The worker gets one job per layer per
  capture and none per replay: the fetch is silently lost (EQF still holds; BF1's use counters would show it).
- BF1's own rule: "Nothing here enters a graph: every op is eager, between graph launches" (r2_bf1_fetch.py:67-69).
- Also F3 exists only for the non-keep arm (ARMS_J3, r2_j1_select.py:628) and BF1 only for the keep arm.
- A later design that would compose: the per-layer candidate mask written by a captured copy into a static
  (L, H, B, topk) buffer, and ONE eager snapshot (D2H + event + job) after the draft pass's graph replay. That is a
  BF1 and J3 change; not proposed now.

### 3.5 Pairs that need no change

- ST1 x F1 / F2 / F6 / F3: `st1_runner(J1 runner)` or `st1_runner(J3 runner)` works as is: ST1 re-classes in
  `__init__` after J1 / J3 re-classed in `_make_state`; CommitST1 finds the I4DC commit by MRO; J1 / J3 define no
  `commit`. ST1's graph is in a private pool (r2_st1_commit.py:18-20, :77-79), so J3's captures in the shared pool do
  not reuse its blocks. The commit does not call `pool_update`, so StampFrame's call count (r2_j1_select.py:651-660)
  is unchanged.
- ST2 x F3 / F6: ST2's `_layer` calls `self._rows(...)` (r2_st1_commit.py:381), which reaches J3's `_gsection` and
  F6's `_cc_locate`.
- BF1 x F1 / F2 / F6: different methods; BF1's barrier is in `run_pass` (r2_bf1_fetch.py:716-722), outside every J1
  method. J1 compiles on the main thread, outside any capture (J1-COMPILE checks "no compile in a capture"); the BF1
  worker never calls compiled code.

## 4. The two integrated arms (recipe for an integration module, e.g. r2_int1_arms.py)

C 128, arm name e.g. L4PTFDGUDCDJ3MT:
1. `S3, R3 = j1_classes(STF)["L4PTFDGUDCDJ3"]` (J1 F1 + F2 + F6 + F3).
2. `RS = st1_runner(R3, mode="rows", twin=True)`.
3. `class RI(RS)`: `__init__` -> super(), then `self.T.__class__ = type(name, (_IntLayer, type(self.T)), {})`;
   `_arm_name` -> name.
C 73, arm name e.g. L4PTFDGUDKWCDJ1MTB:
1. `SBF, RBF = bf_classes(STF)`; `SJ, RJ = j1_wrap(SBF, RBF, name)` (the J1 export of section 3.3).
2. `RS = st1_runner(RJ, mode="rows", twin=True)`.
3. `class RI(RS)`: `__init__` -> super(), re-class with `_IntLayer` (3.1), `self.bf.wrap_backend(self.T._st1["backend"])`
   (3.2); `_arm_name` -> name.
Runner keywords pass through unchanged: J1's (j1neg, j1draft, j1f6), BF1's (bf=dict(...)), ST1's (st1_backend).

MUST gates of an integrated arm = the union: the I4DC arm gates and EQF (= L4PTFD bitwise); J1-PEQ, J1-COMPILE,
J1-DRAFT; J3-RESID (C 128); ST1's (commit OVF, TAIL-EQ, captures) and ST2 TWIN; BF1's (EQF with the fetch on, the
keep byte checks in the miniature). Bands: each rung's own saving term, measured against the SAME job's single-rung arms
(matched), plus the integrated arm vs the best arm (I4DC) and vs SHJ1 / SH.

## 5. Repository and test consequences (Gate 0)

- All three rungs branch from cd0f0c5 and each Gate-0 test requires `changed <= own NEW + own DECL` against cd0f0c5
  (test_j1.py:117-121; test_st1.py:111; test_bf1.py:106). In ONE tree each fails on the other rungs' files.
- Each Gate-0 test also forbids any other module to name its files: test_j1.py:142-145 ("r2_j1"), test_st1.py:131-134
  ("r2_st1"), test_bf1.py:127-130 ("r2_bf1"). An integration module that imports all three fails all three scans.
- The 15 older Gate-0 tests each got an appended declaration block from EACH rung at the same place: a 3-way merge
  gives text conflicts there; the resolution keeps all blocks (each test checks only added lines, its own block).
- So the integration rung needs (RECON7 pattern): one declaration block in test_j1 / test_st1 / test_bf1 (the other
  rungs' files as declared changes, the integration module as a LATER name allowed to name them), and one block in
  each older Gate-0 test for the integration files.

## 6. Order of work (proposal)

1. Wait for j1a (C 128) and the ST1 / BF1 single-rung results. Integrate only the hooks whose own MUST gates passed and
   whose saving is inside its band.
2. J1 export (`j1_wrap`, `j3_wrap`): AREA 1, source only, no behavior change, test_j1 green before and after.
3. Integration module + merged `_layer` + the BF1 backend wrap + the Gate-0 blocks; CPU tests: the layer copy test,
   class MRO asserts (the `_layer` owner is `_IntLayer`; the `commit` owner is CommitST1; BF1's hooks present), a CPU
   fake run where J1 calls = joint passes x L AND ST2 layers = joint passes x L in the SAME run, BF1 inline worker.
4. GPU: a miniature (C 73 B 8, check every round, as j1m) of both integrated arms with their single-rung arms in the
   same job; then the timing cells.

## 7. Open risks (not checked here)

- Memory at C 73 B 318: BF1 adds the landing ring (torch packer: 2 x ring x n_cap x 64 x D x 2 B = 268 MB at ring 8,
  n_cap 1024, D 128; gilfree: x H), ST1 a private pool, J1 Inductor buffers. Lesson L60: usable ceiling about
  83.75 GB, fit margin >= 0.6 GB. The MEM gate of the miniature must show the peak.
- Host threads: BF1's worker plus J1's compile on the first pass (compile happens before the first capture). The
  compile is single-threaded; no compiled function runs on the worker.
- Additivity: F1 (pooling, device), ST2 (twin, mostly host), F3 (host path only) and ST1 (commit, device + host) are
  different labelled components; the joint pass is device-bound (busy 0.94-0.96), so host-only savings add only where
  host time is exposed. Do not predict the sum; predict per term and measure the integrated arm.

## 8. Commands (login node; read-only)

`git -C /mnt/beegfs/ojerbi/nosi-r2-j1-dev merge --ff-only r2-j1-int` (the lead's instruction), `git -C <tree> log
--oneline / branch --show-current` for nosi-r2-st1, -st1-dev, -bf1, -bf1-dev, -j1-int; `ls`, `wc -l`, `grep -n` on
named files; `date -u`. Files read with the Read tool. No interpreter, no sed, no heredoc, no hashing, no job.

## 9. ST3 on the INT line: the keep fill in two compiled helpers (AREA 3; stamp 2026-10-31T14:30:01Z)

Tree /mnt/beegfs/ojerbi/nosi-r2-st3-dev, branch r2-st3-dev, commit 639d4ec5e590b477ccd6c39cf53e08949255608d on INT
76e38d0. Files: benchmarks/Efficiency/r2_st1_commit.py, r2_st1_cell.py, tests_cpu/test_st1.py (no new file; nosi,
r2_int_*, r2_j1_*, r2_bf1_* unchanged; the step scripts unchanged). Not run by the builder (source only).

### 9.1 Where the 2,785 small ops at C 73 come from

MEASURED: /mnt/beegfs/ojerbi/r2-runs/spec/recon7_pb1_2195630_r7pb/summary.json, L4PTFDGUDKWCDP round 112, joint window,
label "section": SMALL 2,785 ops / 16.32 ms = 87 per layer; SELECT 224 ops / 5.65 ms. The C 128 arm without the keep
(r7pa): 1,473 / 6.94 ms = 46 per layer, SELECT 192. Named small kernels of the section (top names): reduce
MaxOps<unsigned char> 32 ops 2.34 ms; where 108 ops 2.49 ms; radixSortKVInPlace 96 ops 3.26 ms. In the whole joint window
CompareEq<long> 96 ops 2.41 ms, of which "resid" 64 / 0.95 ms, so 32 / 1.46 ms in the section.

| source (code) | per layer | per pass (x 32) | evidence |
|---|---|---|---|
| keep fill: r2_i4k_combo.TentStateFKg._keep_fill + keep_assign_d | about 38 SMALL + 1 sort | about 1,216-1,312 SMALL; the E compare 1.46 ms + its max 2.34 ms + its any (not in top names) | 87 - 46 = 41 per layer; the uchar max and the section's CompareEq are its own ops (1 per layer each) |
| chain images (_chain_h: 3 where + 3 copy) | 6 | 192 | where 96 of the 108 ops (about 2.2 ms) |
| compiled helpers (pre, write, loc2, kphys, lists), sort index fills, counters | the rest of 46 | about 1,280 | the C 128 section (same code without the keep) |

The keep fill is the largest avoidable part (the difference to C 128 in ops and most of it in time), and nothing in it
needs eager execution. The chain images come second (about 2.2 ms where + copies) and stay eager (their tensors are
views of one arena storage: the I4CC note).

### 9.2 The hook (INT-compatible)

- r2_st1_commit.KeepFillST3 overrides `_keep_fill` (owner on the INT keep MRO: KeepFillST3; `_layer` stays IntLayer,
  `commit` CommitST1, BF1's classes present). kfa_fn (key, hits, protection, scores, misses, ranks) and kfb_fn (slots,
  kid / kage scatters, hx, the counter row, the remap) through torch.compile (Inductor, fullgraph, dynamic=False); the
  stable sort between them stays eager; the two shipped gathers after. Integer / bool ops only. The layer index and the
  pass step are 0-d device tensors: one compile per helper.
- Check passes run the helpers too; keep_assign_d's check counters and _keep_bytes run on their outputs (the miniature
  checks ST3's map every round). Fallback (counted): acct, controls (kneg, cneg), kslots, no compiled state.
- The helper calls are counted in cc["calls"]; the compile EVENTS go into read()["st3"] only (fix 2 below): ST3
  compiles in the first joint pass, after round 0's draft captures (outside any capture, as J1's F6), and the D root's
  -COMPILE gate (compile_gate_cd) counts every cc event against the first capture of ANY kind. -KFILL checks them:
  at most one per helper, before the first joint capture, none inside a capture.
- FIX 2 (commit after 639d4ec, see 9.5): 639d4ec put the ST3 events into cc["events"] without after_captures, so the
  MTK arm's -COMPILE (and test_st3_cell) would FAIL. Test 639d4ec's test_st3_cell result with this in mind.
- Composition: `STC.st3_runner(R)` for any keep runner R whose state has the I4DC keep section; it re-classes the state
  AFTER R's __init__ (after ST1's, INT's and BF1's own re-classes). For the INT keep arm:
  `STC.st3_runner(INTA.int_runner(STF, "L4PTFDGUDKWCDI", "rows"))`; test_st1.test_st3_int runs it on the CPU world
  through test_int's builder + `STC.st3_attach` (the same re-class).
- Cell: r2_st1_cell arm L4PTFDGUDKWCDMTK = L4PTFDGUDKWCDMT + ST3 (dict ST3, NOT in ST1: r2_int_cell builds its ST arms
  from STCELL.ST1 and would otherwise run the MTK arm without ST3). Gates: the MT arm's set against the MT arm (PEQ,
  DCOUNTS, TAIL-EQ, CGRAPH, TWIN) + -KFILL (every keep fill through the helpers = the compile record's sections, no
  fallback, helper calls = fills, compile events before the first capture). PRED_ST3_BY_B / PRED_STAMP_ST3: None.
- INT cell: NOT changed (owner: INT). To carry ST3 in the integrated arm: one runner_class_int branch for a new arm
  (e.g. L4PTFDGUDKWCDIK -> `STC.st3_runner(INTA.int_runner(...))`), its D_OF / KEEP_NEW entries, and
  `STCELL.kfill_gate(arm, rd.get("st3"), rd.get("compile"))` in int_arm_gates.

### 9.3 MODEL (not registered; the builder's input to the lead)

Keep fill at C 73 B 318, profiled: 2.34 (max) + 1.46 (compare) + 1.5-2.5 (any, MODEL) + 1,120 other launches x 3.0-4.0 us
(the section's other small ops average 3.8 us) = 8.7-10.8 ms; x 0.85 (profiler factor, r7pa) = 7.4-9.2 ms. After:
7-12 Inductor kernels per layer at 3-6 us = 0.7-2.3 ms; the sort stays. Saving 5.1-8.5 ms per round. BAND proposal:
ST3_joint_saving (3.0, 9.0) ms, ST3_vs_mt (1.005, 1.018) at round about 515 ms (run-to-run spread 0.6 %).

### 9.4 CPU tests and GPU protocol (for the lead)

- CPU (amd, srun, by the lead): `env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs
  benchmarks/Efficiency/tests_cpu/test_st1.py -k "st3 or copies or gate0 or pass_plan"`, then the full test_st1.py and
  test_int.py (test_int loads test_st1: the MUT registration of kfa / kfb is module-level there).
- GPU (after review; no step cell exists yet): a miniature C 73 B 8 72 steps --check-every 1 --require-stress 1, arms
  L4PTFD,L4PTFKW,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK (every gate MUST: EQF, PEQ vs
  MT, KEEP gates with the check counters, -KFILL, -COMPILE); then timing C 73 B 318 512 steps, arms
  L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK after PRED_STAMP_ST3 is set. Either
  as two new cells of r2_st1_step.sh (st3m, st3b; ROOT would need this tree) or as INT cells; the lead's choice.

### 9.5 ST3 fix (stamp 2026-10-31T14:43:48Z)

Commit d87b071191fffe3be50b90536ed1b3c74da6b2c4 (r2-st3-dev, on 639d4ec): the ST3 compile events no longer go into
cc["events"]. r2_i4dc_compile.compile_gate_cd (the D root's -COMPILE, which the MTK arm and the INT keep arm take)
counts every cc event as late unless after_captures == 0; ST3 compiles in the first joint pass, after round 0's draft
captures. 639d4ec would fail the MTK arm's -COMPILE and test_st3_cell; the INT fold-in needs d87b071.

## 10. ST4: the draft pass's bit-exact small ops (AREA 3, ledger row td; stamp 2026-10-31T14:43:48Z)

Tree /mnt/beegfs/ojerbi/nosi-r2-st4-dev, branch r2-st4-dev, commit da73561089ee9f2c4d405d6cbf7b61fe3b1bf7ee on
d87b071 (ST3 + fix). Files: r2_st1_commit.py, r2_st1_cell.py, tests_cpu/test_st1.py (no new file). Not run by the
builder (source only).

### 10.1 Attribution of the draft pass's small ops (per one-row pass; 2 passes per round)

MEASURED: r7pa (C 128 B 216, L4PTFDGUDCDP) / r7pb (C 73 B 318, L4PTFDGUDKWCDP), draft0 round 112, SMALL 1,616 kernels
5.40 / 6.22 ms = label A 465 ops 1.12 / 1.24 ms + resid 448 ops 1.65 / 1.79 ms (+ COPY 64 ops 0.15 / 0.16) + B 702 ops
2.54 / 3.06 ms + head 1. Per op names: the eager L4PTFDP sequence (KS/cpu-logs/i4cc_ana_s1.txt:442-638, C 128, per
layer) and the code.

| part (code) | label | per layer | C 128 / C 73 ms per pass | bit-exact? | status |
|---|---|---|---|---|---|
| draft residency ops after pool_swap (r2_i4a_lean._resid_lean: where, 2 copies, gt, ge, or, and, where, masked_fill, not, and, sum, sum copy, add_) | resid | 14 launches, 47.5 us (C 128) | about 1.52 / 1.64 | yes (int / bool / copy) | ST4: one compiled helper (2-4 kernels) |
| ~am + load-mask fill (read by pool_update) | resid | 2, 6.0 us | 0.19 / 0.21 | yes | STAYS: the shipped pool_update reads it between them; one compiled call costs more host time than the 2 launches |
| float32 log-sum-exp merge (maximum, 2 sub, 2 exp, 3 casts, 2 mul, 2 add, div) | B | 13, 55 us | about 1.76 / 2.1 (MODEL x 1.2 at B 318) | NO (float) | STAYS (excluded; the J worker found Inductor not safely bit-exact here) |
| residual adds (bf16) | B | 2, 9.2 us | 0.30 / 0.36 | NO (float) | STAYS |
| dwrite / dmask (I4DC, compiled) + layer-0 hoists and counters (_lean_pass, _lean_mask, tpos) | B | about 5 + 1.1 | about 0.48 / 0.6 | yes | STAYS: already compiled; merging dwrite + dmask for l >= 1 would save about 2 launches per layer (0.1-0.15 ms per pass) and needs a copy of I4DC's _row_lean |
| selection small ops (score copy, 3 fills, zero, scatter, masked_fill, mul, top-k helpers) | A | about 9, 23.5 us | about 0.75 / 0.85 | yes (except the inf fill: a constant) | NOT MINE: J1 F2's region (_InlineJ1 runs ragged_engine._pool_topk's ops); proposal for the J worker: compile them with F2's pool_fn |
| twin pos clone, pos_before copy, lengths (copy, floor_divide, add, copy) | A | 4 SMALL + 2 COPY, 12.5 us | about 0.39 / 0.43 | yes | STAYS: TentState._layer's R = 1 branch and ragged_engine._lengths (nosi); a hook needs a third full copy of _layer (collision 3.1) or a nosi edit; the INT owner could fold it into IntLayer's R = 1 path |
| dpre (compiled) | A | 1-2 | about 0.1 | yes | done (I4DC) |
| J1 F3's StampFrame age shift (C 128 INT only: ge, long, mul, sub_, n add) | resid | 5 | not measured | yes | NOT MINE (J1 F3) |

### 10.2 ST4 (the hook)

- ResidST4._resid_lean = r2_i4a_lean's _resid_lean with the ops after pool_swap in rpost_fn (block-map update, tent,
  skip, ids, the block-map mask moved before the gathers: they read ids only; the skipped / forced / injected counters
  by a one-hot where over (L,)). Every shipped kernel, the host stamp and the host count unchanged.
- PLACEMENT: st4_state inserts ResidST4 DIRECTLY above the class that defines the original (a C3 merge: type(name,
  (S, Bottom)), Bottom = (ResidST4, TentStateFD)). So BF1's TentStateBF._resid_lean (super(), then on_draft), I4H's
  _dstub and J1 F3's captured call all still run and reach it. test_st4_placement checks the MRO in the MT, MTK and both
  INT arms (C 73: TentStateBF is the only override above it).
- INT: `STC.st4_runner(R)` on any runner (MT, MTK, INT). C 128 INT (J3): the residency runs INSIDE the draft graph
  (F3), so rpost's kernels are captured; C 73 INT (BF1): the residency stays eager (BF1's hook), rpost launches 2-4
  kernels in place of 14 + 2 copies. For the INT cell (owner: INT): wrap the INT runner with st4_runner (after
  st3_runner in the keep arm) and add `STCELL.dresid_gate(arm, rd.get("st4"), rd.get("dgraph"), L, rd.get("j3") is not
  None)`.
- Cell arms: L4PTFDGUDCDMTR (C 128: MT + ST4), L4PTFDGUDKWCDMTKR (C 73: MTK + ST4), not in ST1 (r2_int_cell unchanged).
  Gate -DRESID: helper calls = (warm + capture passes + eager + replay passes (not with F3)) x L of the draft-graph
  report, no fallback, no control, compile events none in a capture, at most one.

### 10.3 MODEL (not registered) and what stays, per round

ST4 saving: 34.5-39.5 us per layer = 1.10-1.26 / 1.19-1.36 ms per pass, x 2 passes, x 0.85-1.0 (profiled vs unprofiled):
C 128 1.9-2.5 ms, C 73 2.0-2.7 ms per round. BAND proposal: ST4_draft_saving (1.5, 3.0) / (1.6, 3.3) ms; ST4_vs_ref
(1.004, 1.010) / (1.003, 1.007) (the C 73 band lies inside the 0.6 % run-to-run spread).
STAYS per round (C 128 / C 73, from 10.1 x 2): float merge 3.5 / 4.2 + residual adds 0.6 / 0.7 (float: excluded);
compiled dwrite / dmask / hoists 1.0 / 1.2; selection small ops 1.5 / 1.7 (J1 F2); twin lengths 0.8 / 0.9 (needs a
_layer copy or nosi); residency remainder (2 eager ops + rpost) 0.9-1.3 / 1.0-1.4. Of the ledger's tentative-buffer
work (+5.91 / +6.89 ms per step), the bit-exact part left after I4DC + ST4 is the compiled write / masks (about 0.5 /
0.6 ms per step); the rest is the float merge and call B (design-required: the row's own block has no window slot).

### 10.4 CPU tests and GPU protocol (for the lead)

- CPU: `pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_st1.py -k "st4 or st3 or copies or
  gate0 or pass_plan"`, then the full test_st1.py and test_int.py.
- GPU miniature (C 73 B 8 72 steps, --check-every 1 --require-stress 1): arms L4PTFD,L4PTFKW,L4PTFDGUDC,
  L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDM,L4PTFDGUDKWCDM,L4PTFDGUDCDMT,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,
  L4PTFDGUDCDMTR,L4PTFDGUDKWCDMTKR. Timing: C 128 B 216 L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDM,L4PTFDGUDCDMT,
  L4PTFDGUDCDMTR; C 73 B 318 L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,
  L4PTFDGUDKWCDMTKR (after PRED_STAMP_ST3 / ST4). No step cell exists for them yet.
- LEAD DECISION (2026-10-31): ST4 is measured only inside INT (no separate timing arms; its MODEL is inside the run
  spread). The cell arms MTR / MTKR stay for the CPU tests and a miniature.

## 11. AREA 3 final rows (state, schedule and memory efficiency; stamp 2026-10-31T14:46:34Z)

Values: C 128 B 216 / C 73 B 318, ms per round unless named. STATUS words: MEASURED (a GPU record), MODEL (the
builder's arithmetic from measured inputs), REGISTERED (a band in a commit before the run), PENDING (a run not read
yet), OPTION (quantified, not built). Commits: ST1 a7fc752 (+ bands b3a7bd1, step script 2bc0bca, registration
f79f7c4), ST2 4133836, ST3 639d4ec + d87b071, ST4 da73561.

| row | original model | measured before | change (hook) | expected (band) | status |
|---|---|---|---|---|---|
| commit / discard | 0 (no commit phase) | commit phase median 6.058 / 8.416 (i4dc_d3a / d4b); labelled TentState.commit 6.475 / 9.41 profiled (r7pa / r7pb): 6 where over the whole tail slot + write-back + 2 copies per layer, about 300 eager launches | ST1: the commit in ONE private-pool CUDA graph, J = K + 2 rows moved by index (mode rows) | REGISTERED 2026-10-31T13:02:28Z: ST1_commit_saving (3.0, 5.5) / (4.0, 7.0); ST1_vs_ref (1.008, 1.018) / (1.007, 1.014); peak delta (-0.01, 0.05) GB | correctness MEASURED: st1m and st1a PASSED every MUST gate; C 128 MEASURED (st1a, 11.3): saving 2.629 ms REFUTED low, vs_ref 1.0067 REFUTED low, joint / draft ratios and peak CONFIRMED; diagnosis 11.4; C 73 inside intb |
| joint twin section | 0 (no twin) | twin label per joint pass: 1,063 ops, 4.07 ms device, 46.0 ms host (r7pa); 1,194 ops, 3.85 ms (r7pm) | ST2: every request's step sequence in ONE advance per layer (the shipped fused kernel's advance) | REGISTERED: ST2_joint_saving (1.0, 2.8) / (1.0, 3.0); ST2_vs_m (1.003, 1.009) / (1.002, 1.006) | correctness MEASURED (st1m, st1a MT arms, every MUST gate); C 128 MEASURED (st1a): joint saving 2.573 ms CONFIRMED, vs_m 1.0084 CONFIRMED; C 73 inside intb |
| tentative state, joint section (chain images, buffer writes, lists) | 0 (td holds no tentative work; ledger row 5) | I3 28.9 / 31.1 per joint pass; after I4CC (writes, locate, lists compiled) the section SMALL 6.94 / 16.32 + 3 sorts 4.30 / 5.65 (r7pa / r7pb); chain images 3.58 / 4.69 (eager L4PTFDP labels, cpu-logs/i4cc_ana_s1.txt:12, :641, x 32 / 31) | none for the chain images (their tensors are views of ONE arena storage: Inductor builds a synthetic base of the whole arena; I4CC note); tentative storage = a 64-row buffer for at most K + 2 = 5 used rows, 2.105 MB per request (design) | OPTION, not built: CB1's image of K + 2 rows only (the check reads all 64: a check change); MODEL upper bound the copy half, 1.8 ms at C 73 | MEASURED (inputs); OPTION |
| keep fill (C 73 keep arm) | 0 | the section's keep part: 87 - 46 = 41 small ops per layer (r7pb vs r7pa), incl. the map lookup's max 2.34 ms and compare 1.46 ms per joint pass; the keep fill MODEL 7.4-9.2 ms unprofiled | ST3: the keep fill in two compiled helpers (kfa / kfb), the sort eager; check passes through the helpers too | proposed (noted by the lead for registration): ST3_joint_saving (3.0, 9.0) ms, ST3_vs_mt (1.005, 1.018) | CPU under test (639d4ec + fix d87b071); GPU PENDING inside INT (intm / intb) |
| draft residency (both arms) | td's residency = 2.5 / 3.2 ms per step, no pool (ledger row 1: FALSE MODEL ASSUMPTION for C > 63) | resid SMALL 1.65 / 1.79 + COPY 0.15 / 0.16 ms per one-row pass (r7pa / r7pb); 14 + 2 eager launches per layer | ST4: the ops after pool_swap in one compiled helper (rpost), inserted directly above the original _resid_lean (BF1 / F3 above it) | MODEL 1.9-2.5 / 2.0-2.7 ms per round; band proposal (1.5, 3.0) / (1.6, 3.3); inside the run spread | MODEL; measured only inside INT (lead decision) |
| recovery / yield | fixed lead n = K + p (1 + lag) = 3 + 3p; Y = 3 / (1 + p) = 2.918 / 2.884 | p 2.813 % / 4.027 % per group; 1.70 elapsed rounds per reject; Y (active request) 2.855 / 2.798; gap -2.2 % / -3.0 % (ledger row 8; 3 + 5.1p is accounting consistency only) | none (cause: decision (a): after a reject the next joint slot is a fill round with no verify row, about 2 committed tokens per reject) | OPTION "verify K - 1 restart": +1.9 % / +2.8 % committed tokens per round, no joint time (fits the 4-row pass); reverses decision (a); FAST tokens depend on the residency history, so it needs the human's correctness and tie rules | CONSISTENT (accounting); OPTION awaits a human decision, not built |
| memory fit | the author's device-used rule: largest B 204.1 / 304.2 as written (217.0 / 332.0 with the snapshot removed, MODEL) | allocated peaks 83.900 / 77.694 GB with the prefill (B 216 / B 318 RAN; B 351 OOM at C 73); the prefill transient is a FIXED 2.565 GB (intercept difference at both C; 2.566 at C 81, r7mem), not 20.19 MB per request; decode adds 5.7 MB per request (tentative buffers 2.105 + about 3.6 workspaces); the kept store holds at most 25-26 of 128 listed slots per (layer, KV head, request) | none (lead: the MLP-chunk prefill hook is not built); ST arms add only their pools (registered peak delta -0.01 .. 0.05 GB) | OPTION 1 (prefill transient removed): B_fit 216 -> 220 (C 128, about +1.1 % tok/s), 352 -> 355 (C 73, not binding at B 318), 322.8 -> 326.2 (C 81), MODEL. OPTION 2 (store sized to the measured peak with an overflow fallback): about 6.3 MB per request, 1.8 GB at B 287, B_fit(C 81) 322 -> about 333, MODEL; a design decision (129 slots is the correctness bound) | MEASURED (peaks, transient); OPTIONS not built; physical fit on one basis OPEN (ledger row 10) |

### 11.1 What stays after AREA 3, and why (ms per round, C 128 / C 73)

| part | ms per round | why it stays |
|---|---|---|
| draft float32 log-sum-exp merge (13 ops per layer, 2 passes) | 3.5 / 4.2 (C 73 MODEL x 1.2) | FLOAT arithmetic: the J worker found the merge not safely bit-exact under Inductor (lead, 2026-10-31); EQF needs the bits. Only a bit-exact fused merge (a hand kernel) could remove its launches: NO NEW KERNELS |
| draft call B (flash split-kv + combine on the 64-row buffer, 2 passes) | 2.9 / about 4.3 (C 73 MODEL, scaled by B) | DESIGN-REQUIRED: a draft row past the committed edge sees its own block's chain rows, which have no window slot; one flash call over the window cannot show them without a page table (the fork), and the merge with call A is the price (r2_i4a_lean docstring, KEPT, REQUIRED) |
| draft residual adds (bf16) | 0.6 / 0.7 | float; part of the shipped layer |
| joint chain images (3 where + 3 copy per layer) | 3.6 / 4.7 (eager labels; unchanged by the compiled arms) | bit-exact but views of one arena storage (Inductor's synthetic base); the row-limited copy is an OPTION that changes the chain check |
| joint sorts (window, pool, host key, row-list key; keep score) | 4.30 / 5.65 per joint pass (r7pa / r7pb section SELECT) | aten.sort ties: Inductor may order ties differently; eager keeps the bits (I4CC rule) |
| draft selection small ops | 1.5 / 1.7 | J1 F2's region (bit-exact; proposed to the J worker) |
| draft twin pos / lengths | 0.8 / 0.9 | inside TentState._layer's one-row branch and ragged_engine._lengths: a third full _layer copy (collision 3.1) or a nosi edit |
| commit after ST1 (the graph's 12 nodes per layer + 2 copies) | MODEL 0.6-3.1 / 1.4-4.4 | the J = K + 2 row moves and the shipped write-back are the minimum bit-exact work |
| compiled remainders (dwrite / dmask, kfa / kfb, rpost kernels; draft layer-0 hoists) | about 1.0 / 1.2 (draft) + 0.7-2.3 (keep, C 73) | already one compiled call each; merging dwrite + dmask would save about 0.2-0.3 ms per round |
| recovery loss (fill round after a reject) | -1.9 % / -2.8 % committed tokens per round | decision (a); the verify K - 1 restart is a human decision |

Of the ledger's draft-step excess (+11.3 / +12.9 ms per step vs td; tentative-buffer work +5.91 / +6.89), the bit-exact
work AREA 3 can still remove without a design or correctness-rule change is the compiled remainders only (about
0.5 / 0.6 ms per step); the float merge (1.76 / 2.1 per step) and call B (1.45 / about 2.1 per step) are the
design-required tentative-buffer cost.

### 11.3 Scoring: st1a (C 128 B 216; stamp 2026-10-31T15:51:40Z)

Record /mnt/beegfs/ojerbi/r2-runs/spec/st1_a1_2195631_st1a (nosi b7ed141), stdout lines 88-146; 0 MUST failures;
steady set 180 rounds (L61: rounds 1 and 16 excluded). Arms (median committed tok/s; round median; phases median ms):

| arm | tok/s | round | joint | commit | draft |
|---|---|---|---|---|---|
| L4PTFDGUDCD (REF) | 1,914.6 | 326.394 | 186.865 | 6.075 | 133.08 |
| L4PTFDGUDCDM (+ ST1) | 1,927.5 | 324.058 | 187.135 | 3.446 | 132.97 |
| L4PTFDGUDCDMT (+ ST2) | 1,943.7 | 321.690 | 184.562 | 3.434 | 133.12 |

| key (registered 13:02:28Z) | value | band | verdict |
|---|---|---|---|
| ST1_commit_saving | 2.629 ms | (3.0, 5.5) | REFUTED (low) |
| ST1_joint_ratio | 1.0014 | (0.99, 1.01) | CONFIRMED |
| ST1_draft_ratio | 0.9992 | (0.99, 1.01) | CONFIRMED |
| ST1_vs_ref | 1.0067 | (1.008, 1.018) | REFUTED (low; follows from the saving: round ratio 326.394 / 324.058 = 1.0072, tok/s ratio 1.0067) |
| ST1_peak_delta_gb | +0.009 GB | (-0.01, 0.05) | CONFIRMED |
| ST2_joint_saving | 2.573 ms | (1.0, 2.8) | CONFIRMED |
| ST2_vs_m | 1.0084 | (1.003, 1.009) | CONFIRMED |

CGRAPH (MEASURED): warm 1, captures 1 (round 1), replays 193, overflow 0, capture 0.317 s, private pool 23.07 MB.

### 11.4 Diagnosis of the ST1 shortfall (C 128; no new GPU run)

THE RECORD HAS NO PER-ROUND PHASE LIST: summary.json holds the phase medians (phases_median, with and without the
capture rounds: 3.446 / 3.453 ms) and rollback_per_round (reset + commit: median 3.609, mean 5.59 ms over 181
rounds); stdout prints the same medians. So the split below is MODEL, from the r7pa labelled commit windows of the same
D arm (recon7_pa1_2195631_r7pa stdout:124, rounds 96 / 112: decide 0.2025, accept 0.030, ctr 0.0234, fp 0.7345,
commit 6.475, twins 0.0072 ms device; phase device 7.4726, busy 7.1121, wall 10.3709) and the code
(spec_tentative.py:974-999: the commit phase = from the joint pass's end event to the event after set_twins).

1. THE EAGER PART OF THE PHASE IS SMALL. The fingerprints (fp 0.73 ms) run on CHECK rounds only (TentRunner
   _fp_check_only = True, spec_tentative.py:533, :545-550); r7pa profiled two check rounds (96, 112 = probe rounds), so
   fp is not in the steady median. The rest (argmax decide, accept, ctr, twins, the filter / plant / dropped ops)
   is 0.26 ms device profiled, about 0.24 ms unprofiled (x 0.90 = 6.075 / (7.4726 - 0.7345), the D arm's steady phase
   over its profiled check-round phase without fp). The 4 static-input copies before the replay: about 0.01 ms.
2. NO HOST GAP. The phase is measured by CUDA events on the stream; the round has no host sync and the host runs
   ahead of the device in the joint pass (host 245 ms vs device 348 ms per joint pass at C 73, r7pb; the same order at
   C 128), so the commit-phase work is queued before the device reaches it; one cudaGraphLaunch replaces about 300
   eager launches. The measured phase is device time.
3. SO THE GRAPH REPLAY IS ABOUT 3.2 ms (3.446 - 0.24 - 0.01), not the MODEL's 0.6-1.4. The model's error: it treated
   all 12 nodes per layer as launch-bound tiny nodes (1.5-3.5 us). Two parts are not:
   a. THE WRITE-BACK MOVES BYTES OVER PCIe. Per round about 8.9 requests fill a block (crossed 1,728 / 194 rounds);
      each writes 64 rows x 2 KV heads x 128 x 2 B x (K, V) = 65.5 kB per layer to pinned host memory: 18.68 MB per
      round (the record's bytes_per_round.writeback, from the counted fills). At the measured H2D rate 22.2-22.7 GB/s
      (r7pa stdout:118, 121) that is 0.82 ms; kernel stores through the host pointer may run slower (MODEL 10-22
      GB/s): 0.85-1.9 ms. The original commit pays the same write-back (it is inside its 6.475 ms).
   b. THE NODE FLOOR. 32 x 12 = 384 nodes + about 20 prelude ops = 404 (r7pa's commit window also had 404 GPU ops in
      the eager D arm: the graph holds about as many nodes as the eager commit had launches). At 2-3 us per dependent
      node in a replay: 0.8-1.2 ms.
   c. THE ROW BYTES: 3 gathers of (B, 2J) rows and 6 index_put of (B, J) rows per layer, about 9 MB per layer, 0.28 GB
      per round: 0.2-0.3 ms.
   a + b + c = 1.85-3.4 ms against the measured about 3.2 ms (the upper half: the write-back stores run nearer 10 GB/s
   than 22, or index_put nodes cost more than the 3 us floor; only a profiled M window separates them).
   The eager D commit is then about 6.075 - 0.24 = 5.84 ms (6.475 profiled x 0.90): the six where()s and the
   launches were about 3.9-5.0 ms of it; ST1 removed 2.63 ms of those, the write-back and the node floor stay.
4. CHEAP FIX? None that a timing run can see on its own:
   - Compile the per-layer row moves (Inductor on integer views, as I4DC's dwrite: gather + index_put of K and V fused
     into one kernel per phase): 384 -> about 200 nodes, MODEL saving 0.4-0.6 ms per round (+0.12-0.18 % tok/s), below
     the 0.3 % run-to-run spread at C 128. Cheap (no kernel, bit copies), could ride in INT; not worth its own arm.
   - Move the write-back off the critical path (a side stream, an event that the next pass's first host gather waits
     on): up to 0.85-1.9 ms per round, but the draft residency's H2D gathers share PCIe (1.47 ms per pass) and the next
     pass may load the just-written block from the host rows (an ordering hazard that needs its own gate). Not cheap:
     OPTION, a schedule change.
   - Merging the 32 per-layer block-map / cache-length copies (64 tiny nodes, about 0.15 ms) needs a multi-tensor op
     over 32 engines: no shipped one (torch has no foreach index / copy into distinct tensors with distinct sources
     that would replace them bit for bit); not worth a kernel.
   The ST1 band's error is a MODEL error (the write-back counted as launch-bound; the per-node floor 1.5 us too low),
   not a code fault: CGRAPH, PEQ, TAIL-EQ, EQF all pass and the commit fell 43 % (6.075 -> 3.446).
   What would confirm the split: one profiled commit window of the M arm (the r7 PROF tool), a GPU run: the lead's
   call.

### 11.4b Scoring: st1b (C 73 B 318; stamp 2026-10-31T17:37:51Z)

Record /mnt/beegfs/ojerbi/r2-runs/spec/st1_b1_2195630_st1b (nosi b7ed141), stdout lines 117-158; 0 MUST failures;
steady set 194 rounds (rounds 1 and 16 excluded). CGRAPH: warm 1, captures 1, replays 208, overflow 0, pool 23.07 MB.

| arm | tok/s | round | joint | commit | draft |
|---|---|---|---|---|---|
| L4PTFDGUDKWCD (REF) | 1,707.1 | 522.138 | 337.082 | 8.405 | 175.876 |
| L4PTFDGUDKWCDM (+ ST1) | 1,721.3 | 517.555 | 336.972 | 4.087 | 175.827 |
| L4PTFDGUDKWCDMT (+ ST2) | 1,730.0 | 515.187 | 334.728 | 4.080 | 175.787 |

| key (registered 13:02:28Z) | value | band | verdict |
|---|---|---|---|
| ST1_commit_saving | 4.318 ms | (4.0, 7.0) | CONFIRMED (low end) |
| ST1_joint_ratio | 0.9997 | (0.99, 1.01) | CONFIRMED |
| ST1_draft_ratio | 0.9997 | (0.99, 1.01) | CONFIRMED |
| ST1_vs_ref | 1.0083 | (1.007, 1.014) | CONFIRMED |
| ST1_peak_delta_gb | +0.008 GB | (-0.01, 0.05) | CONFIRMED |
| ST2_joint_saving | 2.244 ms | (1.0, 3.0) | CONFIRMED |
| ST2_vs_m | 1.0051 | (1.002, 1.006) | CONFIRMED |

ST TOTAL (MEASURED, same job): C 128 1,914.6 -> 1,943.7 tok/s (+1.52 %), C 73 1,707.1 -> 1,730.0 (+1.34 %).

C 128 vs C 73 (ST1 saving 2.63 vs 4.32 ms) AGAINST THE 11.4 DIAGNOSIS. The eager D commit grows with B (6.075 -> 8.405
ms, x 1.38 for B x 1.47: the six where()s run over the whole tail slot of every request, MODEL bytes 2.73 -> 4.0 GB per
round, bytes_per_round.tail_where_physical_MODEL); the graph commit grows little (3.446 -> 4.087 ms phase; minus the
eager decide / accept part, about 0.24 -> 0.35 ms scaled by B: graph about 3.2 -> 3.7 ms). That is the diagnosis's
shape: a B-flat node floor plus B-proportional bytes. A TWO-POINT FIT (MODEL; graph = F + W / r + rows / R with the
record's write-back bytes W 18.68 -> 25.53 MB per round (crossed 1,728 / 194 -> 2,544 / 209 fills: 8.9 -> 12.2 per
round), row bytes 0.28 -> 0.41 GB at R about 1.2 TB/s): the step 0.5-0.6 ms needs r about 14-16 GB/s for the
write-back stores, so the write-back is 1.2-1.3 ms at C 128 and 1.6-1.8 ms at C 73, and the node floor F is about
1.6-1.7 ms (about 4 us per node over 404 nodes: twice the 2 us the 11.4 range took at its low end). So: (a) the
write-back is real PCIe work, as diagnosed, at about 15 GB/s for kernel stores into pinned host memory (H2D gathers:
22 GB/s); (b) the node floor is larger than modelled and B-flat, so it eats a larger share of the saving at small B:
C 128's band (built on a 0.6-1.4 ms graph) missed, C 73's (larger eager where()s) held. Two points and an assumed R:
a profiled M commit window would test it. The cheap fix of 11.4 (compile the row moves, about 384 -> 200 nodes) is
then worth about 0.7-0.8 ms per round (+0.2 % / +0.15 % tok/s): still at or below the run spread; it can ride in INT.

### 11.5 Scoring still open (stand-by)

st1a is scored in 11.3. When intm and intb are readable: score ST1 (C 73) and ST2 against PRED_ST1_BY_B (registered 13:02:28Z)
on the cell's steady medians (L61 capture rounds excluded), the MUST gates, and ST3 / ST4 against their proposed bands
inside INT (the ST components are not separable inside INT: per-phase deltas vs the D arm, commit / joint / draft).

## 12. AREA 2 final row: BF1, the draft-issued background fetch (C 73 B 318; written 2026-11-01; BUILDING STOPPED)

Values per round unless named; X = L4PTFDGUDKWCDX (same code, fetch off, store 257), B = L4PTFDGUDKWCDB, same job.
Records: /mnt/beegfs/ojerbi/r2-runs/spec/bf1_b1_2195630_bf1b (nosi f2eba43: barrier + drop, 1 job per transport job,
256-group chunks), bf1_b2_2195631_bf1b (c10bddf: defer + batch 8, one release per batch, 2048-group chunks),
bf1_b3_2195631_bf1b (b051b42: b2 + the PIPELINE). CORRECTED (section 13): b1 had 3 MUST failures
(L4PTFDGUDKWCDX-COMPILE, L4PTFDGUDKWCDX-COUNTS, L4PTFDGUDKWCDB-COUNTS: the gate-spec defects; bf1_b1_2195630_bf1b.step/
stdout.txt "r2_bf1: 3 MUST gate(s) failed"; exit.txt exit=3; sacct FAILED 3:0). b2 and b3: 0 MUST failures (exit 0,
sacct COMPLETED 0:0). EQF (bit-exact against L4PTFD) passed in all three runs, for B and X.

| run | B / X tok/s | round B / X ms | NEW ratio | wait saving | joint B - X | draft B - X | worker per epoch | jobs late / deferred | host wait in J |
|---|---|---|---|---|---|---|---|---|---|
| b1 | x1.0085 | (1,725.7 / 1,711.2 tok/s) | 0.869 | 11.8 ms | -9.0 ms | +7.0 ms (182.5 / 175.5) | 249 ms (3,666 of 13,408 jobs issued) | 72 % late (dropped) | 0 |
| b2 | x0.9534 | 544.8 / 523.2 | 0.603 | 36.0 ms | +11.3 ms (348.3 / 337.0) | +9.8 ms (186.8 / 177.0) | 375 ms (snap 83, select 52, pack 93 + submit 60 C++, issue 52) | 0 late; 59 % deferred | 28.5 ms |
| b3 | x1.0049 | 520.4 / 523.1 | 0.607 | 35.6 ms | -14.3 ms (323.0 / 337.3) | +11.4 ms (188.1 / 176.7) | 311 ms (snap 116, select 52, issue 49, C++ exposed wait 64; C++ span 315) | 0 late; 54 % deferred (7,261 of 13,344) | 4.9 ms |

Bytes (b2 / b3): sent 1.45 GB per round (44.4k head-blocks of 32 KB); useful (NEW 68,985 -> about 41.6k blocks) 0.90 /
0.89 GB, f = 0.62 (the tag count: 0.51); no room before packing 1.34 M items per run; link per round B 1.37 + 1.45 =
2.82 GB vs X 2.27 GB (115 vs 93 ms at 24.5 GB/s: the link is not saturated over the round; what costs is WHEN the bytes
move).

DECOMPOSITION (MODEL calibrated on b2, then checked on b3): gain per round = 40 f S - (6.2 + 2.5 S_d) - W_host - 22 S_J
(ms; GB). 40 ms/GB = the exposed urgent-gather wait per GB (36.0 ms for 0.90 GB: the 24.5 GB/s link). 6.2 ms = the
draft's fixed cost of the worker's host activity (GIL and launches beside the eager draft residency; b1 +7.0 ms at
0.34 GB, b2 +9.8 ms at 1.45 GB: 2.5 ms per GB copied in the draft). 22 ms/GB = the cost of BF1's bytes that cross the
link INSIDE the joint pass (they compete with the remaining urgent gathers; about half of their 41 ms/GB of link time
shows) plus the deferred commits on the compute stream. W_host = the joint pass's host wait for late jobs.
- b2: S_J = 0.59 x 1.45 = 0.86 GB: joint = -36.0 + 28.5 + 18.9 = +11.4 (measured +11.3); round +21.6 (measured +21.6).
- b3: S_J = 0.54 x 1.45 = 0.79 GB: joint = -35.6 + 4.9 + 17.4 = -13.3 (measured -14.3); gain 35.6 - 11.4 - 4.9 - 17.4 =
  +1.9 ms (measured +2.7 ms, x1.0049). The model holds on both runs.

BREAK-EVEN (MODEL): at S = 1.45 GB and f = 0.62 the fetch gains only while S_J and W_host stay small: gain >= 0 needs
22 S_J + W_host <= 24 ms (b3 is just above break-even); the registered x1.04 needs S_J <= about 0.15 GB (deferred share
<= about 10 %) with W_host about 0. If S_J = 0 and W_host = 0, any f >= (6.2 + 2.5 S) / (40 S) = about 0.17-0.22 gains.
f does not decide at C 73; WHEN the bytes move decides. Narrower selection does not help: flagB covers 0.275 of NEW
against flagA's 0.469 (r7pb), BF1 already gets about 85 % of the flagA ceiling (27.4k of 32.4k blocks per round),
and even f = 1 at the same useful bytes (S = 0.90 GB) leaves about 0 ms at the b2 worker time.

WHY THE PIPELINE CUT THE HOST WAIT (28.5 -> 4.9 ms) BUT NOT S_J (0.86 -> 0.79 GB). The pipeline engaged: the b3 record's
B arm has pipelined 2,362 of 2,589 batches and overlapped 1,393 (its previous C++ job still unsubmitted at the select's
start) (summary.json lines 417159 and 420302; the X arm's counters are 0 / 0, fetch off). So the "2 batches" reading is
not in the record, and the counters show NO pipelining defect. The limit is a property of the load: the Python worker
runs snapshot wait -> select -> issue IN SERIES (snap 116 + select 52 + issue 49 + exposed C++ wait 64 = 281 of its 311
ms per epoch), and the snapshot wait is GPU-gated (the host queues each job ahead of the GPU; a batch of up to 8 queued
jobs waits for its newest snapshot). The pipeline overlaps only the C++ job with the next select. With 311 ms of serial
worker time per epoch against a 188 ms draft, the tail of D1's jobs (about half of all bytes) still lands after the
joint pass starts. The model's x1.040 assumed 200 ms per epoch and counted the snapshot wait as overlappable; it is
not.

WHAT WOULD MOVE IT (OPTIONS, not built; the lead stopped BF1 building this round):
- Take into a batch only jobs whose snapshot already completed (event query), so the worker never idles on the newest
  snapshot while older jobs wait: removes most of the 116 ms snapshot wait from the serial chain (MODEL S_J 0.79 ->
  about 0.4 GB, gain about +10 ms, x1.02).
- In gilfree_pack.cpp (transport owner): RETRACTED (section 13.1, row C3). The text said "the copy runs synchronously
  inside the chunk loop". The source does not show that: the staging ring is pinned (r2_transport_gilfree_v3.py:192;
  checked at gilfree_pack.cpp:280), the copy is non_blocking on the copy stream (gilfree_pack.cpp:636), and submit_ns
  (gilfree_pack.cpp:619-653) also times the views, an event wait, two index_copy_ launches and two event records. The
  elapsed time alone does not identify the blocking call (b2 2.17 vs b3 2.73 ms per chunk at the same 51.8 MB chunk).
  This stays with the packer inspection.
- Fewer bytes per round (S): the per-byte stages (pack, H2D, link in J) scale with S; only a selection with f clearly
  above 0.62 at the same useful bytes would help, and none is available from the counters (above).

STATUS: MEASURED (b1, b2, b3); MODEL calibrated on b2 and confirmed on b3; BF1 at C 73 is at break-even (x1.0049, inside
the spread); C 128: no lever (cell docstring decision, MODEL 0.987-1.006). Dev commits: 4981010 (gates, batch, one
release), 64e307f (tests), a329dc6 (pipeline), b451d91 (drop mode without the pipeline); CPU suites passed at
a329dc6 except test_bf1_late[drop] (fixed in b451d91).

## 13. BF1 readiness: diagnosis, the ready-only fix, the arrival counters (AREA 2; written 2026-11-02T09:36:26Z)

Owner: BF (area 2). Source and CPU only; no GPU job ran for this section. Records: R = /mnt/beegfs/ojerbi/r2-runs/spec.
b3 = R/bf1_b3_2195631_bf1b (nosi b051b42), b2 = R/bf1_b2_2195631_bf1b (c10bddf). Per-epoch values: report key
worker_ms_per_epoch (210 epochs, 209 rounds); C++ values: report key transport (sums over the run, divided by 210).
Line numbers: r2_bf1_fetch.py at b051b42 (the b3 code; the same file as b451d91), gilfree_pack.cpp and
nosi/nosi/r2_transport_gilfree_v3.py at the same commit. Dev tree: /mnt/beegfs/ojerbi/nosi-r2-bf1-dev.

### 13.1 Diagnosis: which call blocks, per stage

Python worker thread (bf1-worker). One batch = up to 8 queued jobs (b3: 13,408 jobs in 2,589 batches = 5.18 jobs per
batch, 12.3 batches per epoch).

| row | stage | the call that blocks (file:line) | counter | b3 ms / epoch | b2 | avoidable? |
|---|---|---|---|---|---|---|
| P1 | snapshot wait | `self.snap_ev[s].synchronize()` under cap_lock, r2_bf1_fetch.py:814 (GIL released inside torch) | snap_s | 115.7 | 83.4 | PARTLY: see below |
| P2 | select | numpy on the host under the GIL, :823-880; `ep.ev_map.synchronize()` :830 once per epoch (complete by then: recorded before the first snapshot in stream order) | select_s | 52.1 | 52.4 | no (work), 0.81 ms per job |
| P3 | entry | `self.cv.wait(0.05)` while a deferred job holds the landing entry :898; `self.h2d_done[e].synchronize()` :902 | entry_s | 5.2 | 6.2 | small |
| P4 | C++ wait | `jev.wait_submitted()` :944 -> NativePipe::wait_submitted, a condition-variable wait with the GIL released, gilfree_pack.cpp:772-778 | cxx_wait_s | 63.8 | (serial) | it exposes C3 / C4 |
| P5 | issue | under cap_lock and the epoch lock :948; about 90 launches of the side commit `_commit` :978; GIL held | issue_s | 49.1 | 51.6 | no (work), 4.0 ms per batch |
| P6 | glue | lexsort, item table, `h.tolist()` and NativePipe.submit :1012 (the submit itself only queues: gilfree_pack.cpp:355-388) | busy - sum | 25.3 | - | no |
| C1 | C++ staging ring | `ev_h2d_[s].synchronize()` gilfree_pack.cpp:603 | transport.wait_ns | 1.26 | 0.24 | not a blocker |
| C2 | C++ pack | OpenMP memcpy into the pinned slot, gilfree_pack.cpp:607-608 | transport.pack_ns | 94.5 | 93.2 | packer owner |
| C3 | C++ submit | gilfree_pack.cpp:619-653: views, `ev_sc_[s].block` (stream wait), `ld.copy_(st, non_blocking)` :636 from the PINNED slot (allocated pinned at r2_transport_gilfree_v3.py:192, checked at gilfree_pack.cpp:280), two `index_copy_` launches :642-643, two event records | transport.submit_ns | 76.1 (2.73 ms per 1,581-group chunk) | 60.4 (2.17 ms per 1,586-group chunk) | NOT ESTABLISHED |
| M1 | main thread, joint pass | `self.cv.wait(self.jwait_left)` in on_joint_layer :558 (a deferred job of a layer <= l not finished) | late_wait_s | 4.9 per round | 28.5 | partly (P1 feeds it) |
| M2 | GPU, joint start | `current_stream().wait_event(last)` :503 (the barrier) | barrier_ms_median | 0.074 | 0.066 | no |

C3, the blocking call is NOT ESTABLISHED. The source makes the copy asynchronous by contract (pinned source, non_blocking,
own stream). The same chunk bytes took 2.17 ms (b2) and 2.73 ms (b3), so the time is not a fixed per-byte DMA time.
Candidates that the records cannot separate: (a) the launch calls block (a full launch queue or the driver lock, while
the main thread launches; b1 already showed about 1 ms per 256-group chunk, 8.4 MB, a per-call cost), (b) the copy call
blocks. The counters that separate them: a host timer around the `copy_` call and around the two `index_copy_` calls
inside release_impl, plus a timed event pair around the H2D on the copy stream. That is the packer inspection's work
(not repeated here). Section 12 line 527-528 is retracted.

P1, the LARGEST AVOIDABLE readiness / host cost on the worker. Mechanism (source): the host queues job (pass, layer)
when it runs that draft layer's eager residency, ahead of the GPU; `_loop` takes every queued job of the epoch up to 8
(:701-710) and `_prep` waits for EACH snapshot in order (:810-816), so the batch waits for its NEWEST snapshot. The
PIPELINE then calls `_done(prev)` (:724) only after `_prep` (:716), so the previous batch's issue (its landing and
commit) also waits behind that wait. Size: 9.4 ms per batch (115.7 / 12.3); the GPU makes one job per 2.94 ms
(draft 188.1 ms / 64 jobs), so the newest-oldest spread of a 5.18-job batch is up to 12.3 ms. The records do not split
the wait for the batch's OLDEST snapshot (not avoidable: no data yet) from the wait for the NEWER ones while the oldest
was ready (avoidable). The split counters now exist: snap_first_s and snap_rest_s (13.2).

What the records do NOT show (the round-level costs). B vs X in b3: draft +11.4 ms, joint +21.3 ms beyond the 35.6 ms
wait saving (joint B - X = -14.3). The records cannot assign these to (i) the deferred commits on the compute stream in
J, (ii) BF1 bytes that cross the link inside J beside the urgent gathers, (iii) the worker's GIL and launches beside
the main thread, or (iv) M1. Also, "deferred" (54 % of jobs in b3) is a HOST-clock label: in the steady epochs
inflight_at_close = 31 (the per_epoch records), i.e. at the host's close the GPU still runs D1, and the GPU barrier wait
is 0.074 ms. So section 12's S_J = deferred share x S (0.79 GB) is a host proxy, not a measured GPU quantity. The new
counters measure (i) consume.ms_per_joint_pass, (ii) bytes_sent_in_joint / bytes_useful_in_joint, (iii) host.draft /
host.joint active ms, against the GPU deadlines. [SUPERSEDED in part by 13.6: (i) is the landing wait PLUS the commit
(deferred_wait_and_commit); (ii) is a completion classification, not link bytes in J (*_arrived_after_jstart);
(iii) is host time outside waits, not a GIL hold (host_active_ms).]

### 13.2 The fix (scoped): snap = "ready" (dev commit dd95b87 on b051b42; branch r2-bf1-dev, not pushed)

r2_bf1_fetch.py `_loop_ready` (the b3 loop stays as `_loop_newest` for the same-job A/B):
- The worker drains the queue into a local list (queue order). A batch takes only the leading jobs of one epoch whose
  snapshot event is COMPLETE (`event.query()`, no wait; up to `batch`). Snapshots complete in stream order, so these are
  a prefix.
- No ready job: if the previous batch's C++ job was submitted, it issues that batch first. If that C++ job still runs,
  it polls (1 ms, GIL released while asleep) for the first of the two (oldest snapshot, C++ submission). With nothing
  else to do, it blocks on the OLDEST snapshot only.
- The pipeline order (batch k's select and C++ release before batch k-1's issue), DROP (no pipeline), the carried jobs
  and the stop are the b3 loop's.
Lifetimes (unchanged rules, now in two named methods): `_read_snap` waits the slot's event, reads, then releases the
slot to the main thread; `_stage_acquire` waits until no deferred job holds the entry and the entry's last H2D event
completed, before the worker writes the pinned staging. Exactness: the fetch changes only WHEN bytes arrive; the item
selection, the room rule, the commit and the store are the same code. EQF against L4PTFD in the CPU world for both loops.
Reuse note: no new op; torch events (query / synchronize), the existing transport (NativePipe.submitted).
Cell: new arm L4PTFDGUDKWCDR (= B with snap "ready"); B keeps `--bf-snap newest` (b3's loop, now with the counters).
Step script: new cell bf1r (bf1b + R; STOPS while PRED_STAMP_BF1R is None), and bf1m now has R too.

ARRIVAL counters (record key bf.arrival; cheap: two timing events per batch, one per joint layer, one index_put_ per
commit, one scatter_add_ per joint pass, 32 KB of device counters per epoch + 5.2 MB pbid at B 318 X 257):
- per batch: arrival event (direct: after its side commit; deferred: after its landing) and its items per layer.
- per joint pass: a deadline event at each layer's residency start (on_joint_layer, compute stream); the USED tags
  counted per (batch, layer) on the device.
- report: items_sent_early / _late / _nodeadline, items_useful_early / _late (useful = the joint pass used the slot),
  useful_early_share, bytes_*_in_joint (arrived after J's layer 0 started), margin_ms_useful / margin_ms_sent (p05, p50,
  p95, min, max), late_ms_useful_mean, consume (deferred commits: GPU ms on the compute stream per joint pass), host
  (the worker's busy / wait / active ms per epoch inside the host draft window and the host joint window).
- worker counters: snap_first_s, snap_rest_s, ready_wait_s, polls, snap_unready_taken.
- -BF gate (MUST) gains: useful_counted = used_store + used_win, items_sent_counted = items_sent, the useful split
  closes; snap "ready": snap_unready_taken = 0.
[SUPERSEDED by 13.6 (a347f7c): the useful counts are STORE-ONLY (window uses not covered; the closure is
useful_store_counted = used_store); the key names changed (13.6 label table); the per-epoch counters (32,776 B each)
are now in device_bytes and they GROW per epoch; "active" is host_active_ms, not a GIL hold.]

### 13.3 CPU tests and receipts

Receipt (the whole BF suite at the commit, clean tree): /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/bf1_dd95b87_t.txt and
.txt.receipt: head dd95b870, status clean, SLURM 2250122 on thunder09 (amd), 2026-11-02T10:29:49Z -> 11:09:22Z,
pytest_exit 0, 42 passed (the 28 older tests: no regression; 14 new or extended). A partial pre-commit run:
bf1_dev1.txt (13 passed: gate 0, copy, pure, cell, step, report in every mode).
Evidence lines (from the log):
- (a) test_bf1_ready_only: ready: 120 jobs, unready taken 0 (oracle 0, record 0), polls 1,614, snap_rest_s 0.0;
  CONTROL newest: unready taken 120 (oracle 120, record 120), snap_rest_s 113.6 ms per epoch. Note: with snapshots
  0.3 s apart the ready loop made 120 batches of 1 job (issue_s 137.5 vs 0.07 ms per epoch): the per-batch cost risk
  of 13.4 row A is real when the snapshots arrive slowly; the GPU run reports batches per epoch.
- (b) test_bf1_lifetimes: ready and newest: 240 snapshot accesses, 190-240 staging writes, 0 violations, EQF and gates
  pass. PLANTS caught: no_stage_wait -> 50 "staging reused before its H2D"; no_snap_wait -> 59 "snap read before its
  event".
- (c) test_bf1_arrival_modes: off: disabled, all 0. ce (drop / defer x newest / ready, gilfree once): 62 batches,
  useful counted 299 = used store + window, sent counted = items_sent, early 461 / late 0 (the CPU world issues
  before the joint pass), EQF and gates (with the new -BF closures) pass. [SUPERSEDED by 13.6: those fixtures ran
  acct=True; the 299 mixed store and window uses; the metric is now store-only.]
- (d) test_bf1_arrival_late: stall 200 ms per job, defer: sent early 227 / late 234 = the independent count from the
  raw clocks (227 / 234); useful early 151 / late 148; margin min -947 ms. CONTROL inline: early 461, late 0.
- test_bf1_pipeline (newest, ready): pipelined 48, overlapped 47, EQF and gates pass in both loops.
Not covered on the CPU: the CUDA event clocks, the stream order of the deadline events, the GPU cost of the counters.

### 13.4 Options (MODEL from b3's counters; the lead decides the order)

| option | removes | adds | expected gain per round (MODEL) | CPU-testable plumbing |
|---|---|---|---|---|
| A. ready-only snapshots (BUILT, 13.2) | the wait for newer snapshots (P1 rest) and the issue's wait behind it | smaller batches (more per-batch issue cost: 4.0 ms per batch) and 1 ms polls | arrivals earlier by about 9-14 ms per batch; at the end of the draft about 7.7 GB/s arrive (1.45 GB / 188 ms), so 0.07-0.11 GB leave J; at section 12's 22 ms/GB that is 1.5-2.4 ms, plus M1 (4.9 ms) partly: +2 .. +6 ms of 520: x1.004 .. x1.012 vs B | done (13.3) |
| B. GPU-issued prefetch with the shipped zero-copy gather (lead's candidate; NOT built) | the whole host path: P1-P6 (311 ms per epoch) and C1-C3 (172 ms per epoch), the pinned staging (0.54 GB), the GIL contention (section 12: draft 6.2 ms + 2.5 ms per GB) | (1) SM time beside the drafts: the shipped op `flash_h2d_from_mask{,_bias}` has grid (B, H, M), one CTA per (b, h, block) (nosi/nosi/flash_cache_engine/flash_h2d_mask.py:79); each CTA holds its SM slot while its 16 KB cross PCIe. The only measured overlap of a side-stream SM gather with compute: 0.91-0.95 exposed (2026-09). 1.45 GB at 22-27.6 GB/s = 53-66 ms of link time: 48-63 ms exposed in the draft. (2) No CTA cap in the shipped op: a launch with k valid ids still schedules the whole grid (ids < 0 return at :34-35); into the store the grid is (B, H, X) = 654k programs per launch (as the keep fill, benchmarks/Efficiency/r2_i4d_keep.py:478-479); a cap of k = 32 valid ids per launch needs about 22 launches per draft layer per op, 2.8k per round, about 56-112 ms of Python launch time unless they are captured into the draft graphs; a persistent kernel with a grid cap is a NEW kernel (not allowed). (3) the select on the device without a host sync (dedup, room rule, rank): MODEL 6-13 ms of GPU time per round. (4) J must wait the side stream at its start (the map has one writer stream between joint passes): no defer. | at the measured exposure: +35.6 - (48..63) - (6..13) - (0..10) = -47 .. -18 ms: x0.91 .. x0.97 vs X. IF a k-capped launch overlaps like the copy engine (exposure 0.05): +13 .. +27 ms: x1.025 .. x1.05 vs X (launch cost hidden only in a graph). | device select = host select (store_room, group_rank) on random maps; the CPU gather fakes of the world (W.KERNELS gather_k / gather_v_bias) into the store: EQF, the byte checks, the plants. The stream order and the barrier are GPU-only. |
| C. C++ submit (C3) | 76 ms per epoch of C++ time, if the launch calls block | - | packer inspection owns it | - |

RANKING: A first (built; low risk; its run also measures the J-side terms that B must beat). B ranks BELOW A on the
only measured exposure; it has the higher ceiling. Gate for B, one GPU microbenchmark with the SHIPPED op only (no new
kernel): `flash_h2d_from_mask` on a side stream beside a draft-graph replay at C 73 B 318, k valid ids per launch, k in
{16, 64, 256, all}: the exposed fraction and the achieved GB/s. Build B only if the exposure is <= 0.2 at >= 15 GB/s.
Ordering and lifetimes for B (source check): the slot choice and the gathers run in one side-stream order, which is the
map's only writer between joint passes (as now: the drafts never read the store or the map, r2_bf1_fetch.py module
docstring); J(r + 1) waits the side stream's last event before keep_assign, so the eviction never takes a slot with a
pending gather; the side stream waits an event recorded at the draft layer, after the write-back of the host rows in
stream order; the host rows are pinned and live for the run.

### 13.5 Proposed GPU protocol and bands (for the lead; NOT registered)

0. The run tree /mnt/beegfs/ojerbi/nosi-r2-bf1 (lead only) is at b051b42; dd95b87 is its child (a fast-forward).
1. DRYRUN=1 bash benchmarks/Efficiency/r2_bf1_step.sh --nosi <commit> <holder> <run> bf1m, then bf1m (the miniature, B 8,
   every gate MUST; R and B with the arrival counters, --check-every 1: the keep's byte checks cover R's blocks).
   Inspect: R-EQF, R-BF (snap_unready_taken 0, the closures), arrival block present for B and R.
2. Register PRED_BF1R_BY_B[318] and PRED_STAMP_BF1R (`date -u`) in a commit; then bf1r (7 parts; STEP_MIN 230 from
   b3's 01:46:29 for 6 parts = 17.7 min per part: 124 min, margin 1.85; --mem 340G).
Bands (MODEL; A's row above; b3: B / X 1.0049, joint saving 14.3 ms, draft ratio 1.0645, snap 115.7 ms):
- BF1R_vs_B (0.998, 1.015); BF1R_vs_X (1.000, 1.022).
- BF1R_joint_saving (12, 24) ms (b3 B: 14.3; + 2 .. 6).
- BF1R_draft_ratio (1.03, 1.08) (unchanged work in the draft).
- BF1R_useful_early (0.80, 1.00) [SUPERSEDED by 13.6: key BF1R_useful_early_store, STORE uses only]: the GPU
  deadlines are generous (D1 layer l: snapshot at 94 + 2.94 l ms, deadline at
  196 + 10.1 l ms after the draft start; slack 102 + 7.2 l ms vs a worker latency of 40-60 ms).
- BF1R_snap_rest_ms (40, 110) ms per epoch, measured on B (0.35-0.95 of b3's 115.7).
REPORT-only: R snap_rest_s < 1 ms per epoch; R batches per epoch 12 .. 25 (smaller batches); B and R
bytes_sent_in_joint; consume ms per joint pass; host.joint.active ms. DECISION after bf1r: if R / B < 1.005, ready-only
is not the lever; the arrival block then names the J-side term to attack ((i) consume, (ii) in-J bytes, (iii) host).
[SUPERSEDED in part by 13.6: the REPORT keys are renamed; the counters do not attribute causes (they classify
completions against deadlines); the small-batch overhead is an UNRESOLVED RISK and a protocol check.]

### 13.6 Review 5 (2026-11-02T12:02:58Z): failure propagation, store-only arrival, labels (dev commit a347f7c)

Reviewer findings on dd95b87, all accepted. The earlier text stays; the statements it changes are marked [SUPERSEDED].

1. FAILURE PROPAGATION (r2_bf1_fetch.py). At dd95b87 the readiness `event.query()` (_ready_count), the poll (_poll) and
   the oldest-snapshot `synchronize()` (_wait_oldest) ran outside the worker's _prep / _finish exception boundaries: an
   exception there ended the daemon thread with self.error unset, and the compute thread could wait on a busy snapshot
   slot. Fix (a347f7c): `_loop` has an outer boundary `_fatal`. Every worker exception ends there (also the queue and
   hold calls). It keeps the FIRST exception (error = text and traceback; error_exc = the object), releases the
   snapshot slots of the jobs taken and not read and of every queued job, closes their flight counters, notifies the
   cv, and the worker STOPS (no retry). An error recorded inside _prep / _finish (_record_error) also stops the loop at
   its next iteration. The compute thread raises `BFError(...) from error_exc` at its next hook and also right after
   its own queue put (`_raise_error`, which first drains the queue and releases those slots). Side fix: _prep updates
   the caller's idss list in place, so a failed batch releases only the snapshots it did not yet read (before, a read
   and released slot could be released twice, possibly after the main thread took it again). Not changed: a batch that
   was already prepared (prev) is not issued after a failure; its snapshots were released at its read.
   Control test: test_bf1_worker_failure[query, sync] (a snapshot event whose query() or synchronize() raises a planted
   exception, snap ready, thread worker): the compute thread's BFError.__cause__ is the planted object, error_exc is
   the planted object, the worker thread ended, snap_busy all False, inflight 0, queue empty.

2. USEFUL-ARRIVAL POPULATION. The timing cells run ACCT=0 with arrival on (r2_bf1_step.sh), and on_window returns at
   once without acct (r2_bf1_fetch.py on_window), so window (kwin) uses are never seen there. My dd95b87 fixtures ran
   acct=True; the CPU world had 141 store + 158 window uses. Choice (b): the metric is STORE-ONLY in every mode. The
   per-(batch, layer) device counts now take the store uses only (`used`, not `used | window`); keys say so; the
   -BF closure is useful_store_counted = used_store (window uses on neither side). Window uses are NOT covered;
   with acct=True their total is reported apart (window_uses_acct = used_win), not per batch and not timed. Reason
   for (b), not (a): counting window uses under acct=False needs on_window's compare of every row-0 load against every
   tagged slot ((H, B, topk, X - 1) booleans, about 4 launches per joint layer, 32 layers per round) on the compute
   stream inside J: that changes the joint pass that the run measures. Test: test_bf1_arrival_store_only (the same
   deterministic inline world with acct=True and acct=False): acct=True used_win > 0 (window-only consumers exist);
   acct=False used_win = 0 and unused_F = unused_T + used_win_T; useful_store_counted = used_store in both, the same
   useful_early_share_store in both. I do not predict the GPU store / window mix.

3. LABELS AND MEMORY.

| dd95b87 key | a347f7c key | meaning (exact) |
|---|---|---|
| items_useful_{early,late,nodeadline,unattributed} | items_useful_store_{early,late,nodeadline,unattributed} | store uses only |
| useful_counted | useful_store_counted | store uses only; = used_store (-BF closure) |
| useful_early_share | useful_early_share_store | store uses only; window not covered |
| margin_ms_useful | margin_ms_useful_store | margins weighted by store uses |
| late_ms_useful_mean | late_ms_useful_store_mean | same |
| (new) | useful_scope = "store_only", window_uses_covered = False, window_uses_acct | the scope, stated in the record |
| items_sent_in_joint, items_useful_in_joint, bytes_*_in_joint | items_sent_arrived_after_jstart, items_useful_store_arrived_after_jstart, bytes_*_arrived_after_jstart | the batch's usable-arrival event completed after J's layer-0 start; NOT the DMA interval intersected with J, not bounded by J's end, not link bytes during J, not interference |
| consume = {commits, ms_total, ms_per_joint_pass} | deferred_wait_and_commit = {jobs, ms_total, ms_per_joint_pass} | compute-stream span of a deferred job: its landing wait PLUS its commit |
| host.{draft,joint}.{busy, wait, active} | host.{draft,joint}.{busy_ms, wait_ms, host_active_ms}; gil_hold_measured = False | host time outside waits; NOT a measured GIL hold |
| verdict BF1R-USEFUL-EARLY, key BF1R_useful_early | BF1R-USEFUL-EARLY-STORE, BF1R_useful_early_store | store uses only |
| device_bytes (without the epoch counters) | device_bytes + arrival.memory {pbid_bytes, counter_bytes_per_epoch, epochs_kept, counter_bytes_total} | one (128 x 32 + 1) x 8 = 32,776 B counter per kept epoch (defaults); it grows by that per epoch until the report, no other bound (210 epochs at bf1b: 6.9 MB); pbid fixed (H x B x X x 8 = 5.2 MB at B 318, X 257) |

4. UNRESOLVED RISK (not a speedup claim). In the CPU fixture with snapshots 0.3 s apart, the ready loop raised the
   worker time (453 vs 330 ms per epoch; 120 batches of 1 job; issue_s 137.5 vs 0.07 ms per epoch): ready-only can
   make small batches whose per-batch cost (b3: 4.0 ms per batch issue) exceeds the wait it removes. 13.4 row A's
   MODEL gain stays a MODEL. PROTOCOL CHECK in bf1r: R's batches per epoch, R's worker_busy_s and issue_s per epoch
   against B's; if R's worker_busy_s per epoch exceeds B's by more than the snap_rest_s it removes, record the risk as
   realised.

5. RECEIPTS (amd, clean tree at a347f7c, branch r2-bf1-dev, not pushed):
- The whole BF suite: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/bf1_a347f7c_t.txt (+ .receipt): head a347f7c3, status
  clean, SLURM 2250509 on thunder09, 2026-11-02T13:05:36Z -> 13:46:38Z, pytest_exit 0, 45 passed (the 42 of dd95b87
  + test_bf1_worker_failure[query], [sync] + test_bf1_arrival_store_only).
- The review-5 subset first: bf1_a347f7c_q.txt (+ .receipt): 13 passed.
Evidence lines (from bf1_a347f7c_q.txt):
- worker_failure[query]: BFError from RuntimeError('planted query failure'), raised in _ready_count via _loop_ready;
  [sync]: from RuntimeError('planted sync failure') in _wait_oldest; both: worker alive False, slots owned 0,
  inflight 0.
- arrival_store_only: acct True: used store 141, window-only 158, unused 162, useful store 141 (share 1.0),
  window_uses_acct 158; acct False: used store 141, window 0, unused 320 (= 162 + 158), useful store 141 (share 1.0),
  window_uses_acct None.
- arrival_late (re-run): sent early 286 / late 175 = the independent count; store-useful early 68 / late 73;
  deferred_wait_and_commit jobs > 0. (The thread worker's timing differs from the dd95b87 run: 227 / 234 there.)

## 14. j1f source follow-up and the fork edit plan (AREA 1, J owner; written 2026-11-02T10:24:03Z)

Dev tree /mnt/beegfs/ojerbi/nosi-r2-j1-dev, branch r2-j1-dev. Commits (not pushed): 48da64d (14.1), 8951bc3 (14.3).
The run tree /mnt/beegfs/ojerbi/nosi-r2-j1 is not touched. No grant, no band.

### 14.1 j1f source follow-up (reviewer finding on r2_j1_forkbench.py)

- NUM: `num_check` applies torch.isfinite to the output, to the float64 reference and to their difference BEFORE
  the max. A row with a non-finite value does not enter the max. `num_verdict` FAILS with the count and the first index
  (request, row, head, dim); a variant with no checked row also FAILS. The variant becomes INVALID in the ratios.
- EXACT: `repeat_check` needs bit equality AND only finite values in both outputs (identical NaN bits now FAIL).
- Timing: the four flash controls (NS1R1, NS1x4, FLASHU1, NSRU) timed the wrapper plus an output copy_ (FORK writes its
  preallocated output directly). Each control now has three variants. X = the library op
  flash_attn_2_cuda_nosa.fwd_kvcache (the op that flash_attn_with_kvcache calls; the wrapper passes None at its out_
  argument) with out_ = a view of the preallocated output in the op's swapped (B, HQ/HKV, HKV, HD) layout: option (a),
  kernel-only [SUPERSEDED by 14.4: an END-TO-END library call into a preallocated output, NOT kernel-only]. X_CP =
  the j1f form (wrapper + copy_): total. X_CO = the copy_ alone. New MUST gate <shape>-X-OUTEQ:
  X = X_CP bit for bit. The record adds `flash_timing` (kernel, copy, total, rest) [SUPERSEDED by 14.4: keys
  direct_e2e_ms_pass and residual_ms_pass] per control and the j1f-form ratios
  (FORK1R/NS1R1_CP etc.). The timing loop moved to `time_variants` (timer injected; `cuda_timer` on the GPU).
- I do not re-score the old numbers. j1f's per-byte ratios (FORK1R / NS1R1 1.134 / 0.975) need the rerun below.
  [SUPERSEDED by 14.4 wording: these are time ratios at matched LOGICAL work; they do not give a physical per-byte
  ceiling or per-byte optimality.]
- CPU tests (tests_cpu/test_j1.py): test_forkbench_nonfinite (NaN, +Inf, -Inf planted in the output, the reference and
  both, 3 variants: every plant FAILS the new NUM; the j1f source loaded from git at cd0e286 PASSED 15 of the 27
  plants, the identical-NaN plant included: the control is not a no-op; identical non-finite bits pass the old EXACT
  and FAIL `repeat_check`), test_forkbench_out_view (the view = the op's own return transform; the call's 21 positional
  arguments = the installed wrapper's, out_ at position 13), test_forkbench_timing_plumbing (fake timer).
- Receipts (/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/): j1_48da64d_t (test_j1.py: 22 passed, 1 skipped = the DRYRUN,
  which needs a clean run-tree checkout), j1_48da64d_g0 (Gate 0 of the 15 older files: 37 passed).
- OBSERVATION (INFERENCE, not changed): NUM's TOL 1e-3 is about the size of the outputs (x 0.05 inputs: about 8e-4
  per element). j1f's errors were 8.3e-6 .. 1.0e-5. A mask that adds or drops 3 pages of about 64 changes outputs by
  about 1e-4 .. 7e-4 (MODEL), so NUM may not see it. A TOL of 1e-4 keeps 10x margin over the measured errors.

### 14.2 GPU rerun protocol (lead; same shapes as j1f)

1. Bring r2-j1-dev at 8951bc3 into the run tree (lead's merge), set GRANT_j1f in r2_j1_forkbench.py (date -u stamp).
   [SUPERSEDED by 14.4 reconciliation plan: merge at fd91420, not a fast-forward; re-register before any run.]
2. `bash benchmarks/Efficiency/r2_j1_step.sh --nosi <sha> <holder> <run id> j1f` (DRYRUN=1 first). The step script
   passes the j1f arguments unchanged: --shapes 128:216,73:318 --segments 2,4,8 --layers 32 --reps 10 --warmup 2
   --seed 7 --sample 8 --max-shape-s 300 --max-run-s 900. Variants per shape: 22 (j1f: 10). MODEL wall: about 2.3x
   j1f's 42.7 / 11.1 s per shape, inside 300 s; STEP_MIN 25 unchanged.
3. Read, in this order: every MUST gate (the new NUM non-finite text, EXACT, OUTEQ, FORKAn-EQ); then FORK1R/NS1R1_CP
   and FORK/FLASHU1_CP (they must reproduce j1f's 1.134 / 0.975 and FORK - FLASHU1 within the run spread: if not, the
   setup moved and nothing is re-scored); then FORK1R/NS1R1 and FORK/FLASHU1 (direct write) = the re-scored per-byte
   numbers; the FLASHT lines (kernel-only, copy-only, total, rest per control); FORKAn/FORK (14.3).
   [SUPERSEDED by 14.4: "the re-scored time ratios at matched logical work (end-to-end direct output)"; FLASHT =
   direct (end-to-end), copy-only, total, residual; the 3D saving line is SUPPRESSED when FORK is invalid.]

### 14.3 The fork edit plan (lead's request; the 4-row tile is latency / occupancy-bound, not compute-bound)

MEASURED inputs (SCORING_fia5.md sections 2-5): 255 registers, 59392 B shared memory, 2 programs per SM (both limits),
8 warps per SM; in the tile loop three synchronous loads before the first dot (page ids in the MMA layout, ext_rows,
the bias); the K / V copies of tile j + 1 start after both dots and the same iteration waits for them; 8.1 us per tile
(bf16 bias); stalls: wait 33 %, long scoreboard 27 %. Bits: num_stages and the bias width do not change the bits;
num_warps 8 at tile 64 does (851 / 854 / 127 rows equal); tile 32 does (0 rows equal).
The lead's arithmetic checks (DERIVED): 2 x 64 x 64 x 128 x 2 = 2.1 MFLOP per tile = 0.73 us at 312 TF / 108 SM;
K + V 32 KB = 1.7 us at the per-SM share of HBM; 8.1 us / 2 programs = about 4 us per tile slot.

| edit | target | same reduction order and bits? | status |
|---|---|---|---|
| A1 ahead=1: page id + extents word of tile j + 2 read in iteration j, as scalars | 2 of the 3 synchronous loads | YES: the same values reach the same ops; dots, softmax, stores are the fork's lines | BUILT (flag) |
| A2 ahead=2: A1 + the bias of tile j + 1 read in iteration j | the third synchronous load | YES (as A1) | BUILT (flag) |
| A3 ahead=3: the fork's page-id / ext_t vector loads kept (for Triton's K / V staging), ext_rows + bias ahead | as A2, keep the async K / V copies [SUPERSEDED by 14.4: the compiled loop loses them at level 3 too] | YES (as A1) | BUILT (flag) |
| B1 skip acc x alpha when every alpha = 1 | 64 FMUL per thread per tile | YES (x 1.0 is exact) | NOT BUILT: not on the measured chain |
| B2 reload Q per tile | registers | YES (same values) | NOT BUILT: Q is live at the QK dot anyway (no peak relief); adds a conversion per tile |
| B3 num_stages / bf16 bias | staging | YES (fia5 MEASURED) | DONE by fia5 (bf16 shipped; s4 = 1 program per SM, slower) |
| C1 num_warps 8 at the 64-row tile | occupancy | NO (fia5 MEASURED: rows differ) | NEEDS A HUMAN DECISION (also +21-31 % slower at B 216) |
| C2 tile 32 | registers | NO (fia5 MEASURED: 0 rows equal) | NEEDS A HUMAN DECISION (slower) |
| C3 split the 64 rows over 2 programs (BLOCK_Q 2, 32 rows) | registers -> up to 4 programs per SM | UNPROVEN: the MMA warp layout for M 32 changes the row-sum tree unless num_warps 2; each tile read twice; extents byte offset per q block; a shared union build | NEEDS A HUMAN DECISION |
| C4 fold the scale into Q (S = Q'K + bias x scale) | one 64x64 multiply per tile | NO (other rounding) | NEEDS A HUMAN DECISION |
| C5 exp2 with log2(e) folded | MUFU | NO | NEEDS A HUMAN DECISION |
| C6 3D split | waves | NO (loses EXACT; j1f MEASURED 1.23-1.46x slower) | rejected (j1f) |
Occupancy: 3 programs per SM need <= 168 registers and <= 54 KB shared. Every source route found to that (C1-C3)
changes the MMA layout or the tile, so it changes bits or is unproven. Within the bit constraint the lever is the
per-tile chain of dependent loads (A1-A3).

Implementation: /mnt/beegfs/ojerbi/nosi-r2-j1-dev/benchmarks/Efficiency/fork_vllm_unified_attention_ahead.py = the
fia6 fork (blob f0dc0c18, NOT changed) + lines tagged [ahead]. A sibling file keeps the fia6 pins of 15 step scripts
and the Gate-0 tests valid (an in-place edit changes the blob they pin). Launch flag `ahead` (constexpr AHEAD), default
0. Declared in the J1 line of the 15 older Gate-0 tests and in r2_j1_step.sh's tracked-file list. r2_j1_forkbench.py:
variants FORKA1 / FORKA2 / FORKA3 (production mode, union built in the kernel) with the MUST gate <shape>-FORKAn-EQ
(bit-equal to FORK) and the ratios FORKAn/FORK.

CPU results (receipts j1_ah_k6, j1_8951bc3_t):
- test_ahead_copy: the file = the fork + 67 tagged lines; 8 fork lines only re-indented (into the `else` branches).
- test_ahead_interp (TRITON_INTERPRET=1; heads 32 / 2, dim 128, block 64; 2 requests x 4 rows, shifted pages, union 8,
  partial last pages, a NaN page behind the tables' tail): ahead 0, 1, 2, 3 = the fork BIT FOR BIT in 32 of 32 cases
  (union built and read, bias on / off, tile 64 / 32); fork vs float64 7.9e-8; PLANT stale page (tile j reads tile
  j + 1's page): 6144 values differ (DETECTED); swapped tile order: 2200-2344 values differ at 5e-8 (the bits see the
  order). LIMIT: the interpreter runs numpy. It shows the same values in the same operations, not the GPU code
  generator's reduction order.
- test_ahead_compile_witness (NEW METHOD, no GPU: Triton 3.2 compiles both files for sm_80 through JIT warmup with a
  stand-in driver; cuobjdump reads the cubin; production launch w4 s3 tile 64, B 216). It reproduces fia5's FV
  resources (255 registers, 59392 B shared):

| arm (bf16 bias; f32 the same) | registers | stack B | shared B | loop: sync tt.load / async copies / async_wait |
|---|---|---|---|---|
| fork (fia6) | 255 | 48 | 59392 | 3 / 5 / 4 |
| ahead 0 | 255 | 48 | 59392 | 3 / 5 / 4 |
| ahead 1 | 251 | 0 | 49152 | 5 / 0 / 0 |
| ahead 2 | 255 | 0 | 49152 | 5 / 0 / 0 |
| ahead 3 | 255 | 0 | 49152 | 8 / 0 / 0 |
| fork, tables read | 255 | 24 | 59392 | 3 / 5 / 4 |

  FINDINGS: (1) any load carried across iterations makes Triton 3.2's pipeliner drop the K / V async copies (all three
  levels, A3 included): K / V become plain loads. The spills go (stack 48 -> 0) and shared memory falls to 48 KB, but
  registers stay at 251-255, so the SM still holds 2 programs. The fork's own copies did not overlap the dots either
  (fia5 TTGIR), so the GPU timing decides. (2) ahead 0 has the fork's TTGIR (MUST, PASS). Its bf16 PTX and SASS differ
  from the fork's (register numbering) with line info and are IDENTICAL without line info (TRITON_DISABLE_LINE_INFO=1);
  a byte copy of the fork at another path (same line numbers) is identical. So shifting source lines alone changes
  the bf16 machine code. The bits are therefore a GPU question for every arm, ahead 0 included: FORKAn-EQ.
- Suites at 8951bc3: j1_8951bc3_t (test_j1.py: 25 passed, 1 skipped = the DRYRUN), j1_8951bc3_g0 (37 passed).

NEXT (lead, GPU): the 14.2 run also times FORKA1-3. Decision rule (proposal): keep a level only if FORKAn-EQ passes at
both shapes AND FORKAn/FORK <= 0.97 at both (the run spread is about 1 %). Then a production plumbing step (verify_rows
passes `ahead`) and a j1a / j1b EQF run against L4PTFD; folding the [ahead] lines into the fork (a new blob) needs the
human's re-pin decision. [SUPERSEDED by 14.4: routine pinning of a bit-preserving correction is lead-owned; only
numerical (bit-changing) edits need the human.]

### 14.4 Review 5 (written 2026-11-02T12:11:25Z; dev commit fd91420 on 8951bc3)

Statements above that this review changes carry "[SUPERSEDED by 14.4]"; nothing is deleted.

COMPARISON VALIDITY (r2_j1_forkbench.py `summarize`, `decision_lines`):
- best_3d now follows the ratios' rule. FORK invalid (or did not run) -> best_3d None, best_3d_note "INVALID: FORK ...";
  no valid 3D member -> "INVALID: no valid 3D variant"; the decision line prints "best 3D SUPPRESSED (...)" and no number.
  FORK - FLASHU1 prints "SUPPRESSED" when either member is invalid. An invalid RATIO line keeps its value and ends in
  INVALID (unchanged rule).
- PLANT (test_forkbench_best3d_validity): FORK invalid -> the saving is suppressed; the j1f source at cd0e286 still
  reported a 4.0 ms saving and FORK - FLASHU1 12.0 ms on the same input (the control is not a no-op). The fastest 3D
  member invalid -> the next valid one is chosen.

LABELS:
- The direct form X is an END-TO-END library call into a preallocated output, NOT kernel-only: the pinned C++ op still
  allocates softmax_lse and the split-KV scratch. Record key `direct_e2e_ms_pass` (was kernel_ms_pass); FLASHT line
  "direct (end-to-end, preallocated output)"; OUTEQ text "end-to-end direct output = X_CP".
- X_CP - X - X_CO = `residual_ms_pass` (was rest_ms_pass): a residual of separately timed forms, not an isolated
  measurement of the wrapper's allocations.
- FORK1R / NS1R1 is "fork time / flash end-to-end time at matched logical work; not a physical per-byte ceiling". The
  docstring says the logical-byte ratios do not give a physical per-byte ceiling or optimality (that needs DRAM
  counters). No decision line says "per-byte efficiency" or "optimal" (test check).

COMPILE SCOPE: the ahead file header, the forkbench docstring and the FORKA3 pair text now state the compile witness:
every level 1-3, A3 included, loses Triton's async K / V copies (K / V become plain loads); registers stay 251-255, so
the SM holds 2 programs, as the fork; the spill removal (stack 48 -> 0 B) and 48 KB shared memory imply NO GPU
speed-up and NO bit equality. The GPU run decides both (FORKAn/FORK, FORKAn-EQ).

PIN SCOPE AND RECONCILIATION PLAN (for the future run version):
- Facts (git, read-only): the GPU-run pin 89e324e (branch r2-j1, run tree /mnt/beegfs/ojerbi/nosi-r2-j1) and the dev
  head fd91420 (r2-j1-dev) diverge at 9eb59ef. r2-j1 holds 2 commits after it: 7ebb84f (GRANT_j1f =
  "2026-10-31T13:22:30Z") and 89e324e (318 bands re-registered F6-inclusive, stamp 2026-10-31T15:47:52Z). r2-j1-dev
  holds the older 318 bands (joint saving 14.0-18.5 etc.) because it never took those two commits. I do NOT edit the
  old bands in either branch.
- A trial merge (`git merge-tree 9eb59ef 89e324e fd91420`, writes no tree) has 0 conflicts: the dev side did not touch
  the registration lines after 9eb59ef, so a merge keeps 89e324e's 318 bands, PRED_STAMP_J1 and GRANT_j1f lines as
  registered.
- Step 1 (lead): MERGE r2-j1-dev at fd91420 into r2-j1 at 89e324e (a merge commit; a fast-forward is not possible).
  The run tree is then fast-forwarded to that merge commit.
- What changes at the merge commit (blob ids by `git rev-parse <commit>:<path>`):
  - fork blob: benchmarks/Efficiency/fork_vllm_unified_attention.py = f0dc0c18 (UNCHANGED; FORK_BLOB pins stay).
  - new tracked file: benchmarks/Efficiency/fork_vllm_unified_attention_ahead.py (blob dec2ff66 at fd91420); the
    step script now requires it at NSHA.
  - wrapper blob: benchmarks/Efficiency/r2_j1_step.sh 4638d912 (89e324e) -> 6d2ca572 (fd91420): comments and the
    tracked-file list only; the lead's job file / dispatcher pin of the wrapper blob changes.
  - cell blob: benchmarks/Efficiency/r2_j1_forkbench.py d3f88341 (89e324e) -> the merged blob (fd91420's code plus
    89e324e's GRANT_j1f line). The old GRANT_j1f stamp named the j1f source; it must not carry over to new code.
  - r2_j1_select.py: the dev side adds the export j1_wrap / j3_wrap (b4bd542, "no behavior change",
    test_j1_wrap_export); r2_j1_cell.py: no dev change.
- Step 2 (lead, before ANY run on the merged commit): re-register with a NEW `date -u` stamp in one lead commit:
  GRANT_j1f (new stamp; the cell code changed), and PRED_STAMP_J1 / PRED_J1_BY_B re-stamped for the merged commit if a
  j1a / j1b run follows (values unchanged unless the lead decides otherwise; the old stamps and bands stay in git
  history). Then DRYRUN the step script at that commit.
- Ownership: pinning a bit-preserving correction (the forkbench labels and gates; the ahead file as a timing arm) is
  lead-owned routine work. Numerical (bit-changing) edits (C1-C5 in 14.3) stay a human decision. Folding an `ahead`
  level into the shipped fork (a new fork blob and every FORK_BLOB pin) is routine ONLY if FORKAn-EQ passes on the GPU
  at both shapes and the j1a / j1b EQF holds; else it is not done.

CPU receipts (/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/): j1_fd91420_t (test_j1.py: 26 passed, 1 skipped = the
DRYRUN), j1_fd91420_g0 (Gate 0 of the 15 older files: 37 passed); targeted runs j1_r5_k1 / k2 (test fixes) and k3
(6 passed).

## 15. Yield / window population repair (ST; review 5; written 2026-11-02T13:04:43Z)

Dev commit a39aaa9 on b9031c9 (tree /mnt/beegfs/ojerbi/nosi-r2-int-dev, branch r2-int-dev, not pushed).

DEFECT (verified by the lead): summarize (r2_spec_cell.py:343-380; r2_wsa_cell.summarize_w for the L4PT arms) puts
paper.accepted_per_round_steady_mean and kv_at_verify on SC.steady_rounds(G, probe_every). restrict
(r2_i4h_cell.py) then removes the capture rounds X from the rates and phases only. inta / intc / intd: the yield on
181 / 187 / 195 rounds, the rates on 180 / 186 / 194.

FIX: one shared helper r2_i4h_cell.population_repair, called by restrict. Every cell that calls HC.restrict gets it:
i4h, i4s, i4cc, i4dc, recon7, st1, bf1, and the INT cells (r2_int_cell's _HCProxy forwards restrict). New record fields:
- yield_final (the FINAL set = the rates' set): rounds, accepted / committed sums, accepted_per_round_mean,
  accepted_per_request_per_round, committed_per_request_per_round (ratios of sums over the final set);
- paper.accepted_per_round_steady_mean on the final set; the old value in accepted_per_round_steady_mean_incl_capture;
  paper also gets accepted_per_request_per_round and committed_per_request_per_round;
- kv_at_verify: the measurement does NOT change. wait_population = the probe rounds (non-None probe_wait_ms) and their
  count; the spans are the J / verify slot's host gathers only (meter.timing; spec_tentative.py:954); D-slot loads are
  counted but not timed (spec_tentative.py:1020). bytes_population = SC.steady_rounds WITH the capture rounds, its count
  and the capture rounds in it. round_blocks is in read(), so new_blocks_per_round_final / new_MB_per_round_final /
  final_rounds are on the final set (None + final_note when round_blocks is absent);
- rec["new"] (BF1 arms): new_blocks_per_round_final, final_rounds;
- steady_rounds_list (the final set) and populations (the round set of every summary field: final, incl_capture,
  probe, all rounds, wall). rollback_per_round stays on the incl_capture set (recorded in populations, not changed).

TESTS (test_int.py, section R5): test_r5_final_set_fields (synthetic rd, G 40, probe 16, B 4, X = [1, 16]; final 36
rounds, old 37; every field = the hand value; *_incl_capture = the old computation; the old and new values differ: the
precondition), test_r5_missing_vectors_and_controls (no round_accepted / round_blocks: None with notes; X = []: the old
values; a second call keeps the first old value), test_r5_every_cell_reaches_the_one_restrict.

RECEIPTS (/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/, one srun job 2250428 on amd, tree at b9031c9 + the edits =
a39aaa9's bytes): st_r5_a_r5 (3 passed), st_r5_a_g0 (Gate 0 of the 19 tests: 41 passed), st_r5_a_int (test_int.py:
18 passed, 1 skipped = the DRYRUN), st_r5_a_st1int (test_st1.py -k int: 6 passed), st_r5_a_i4h (test_i4h.py: 12
passed, 1 skipped = the DRYRUN), st_r5_a_bf1 (test_bf1.py: 27 passed).

PINS (for the lead):
- r2_i4h_cell.py blob aa8ad290 (b9031c9) -> 141d6ddf (a39aaa9). No step script pins this blob: r2_i4s / i4cc / i4dc /
  recon7 / j1 / st1 / bf1 / int _step.sh only require it tracked at NSHA. FORK_BLOB is unchanged. The run trees need
  the new NSHA; the lead decides if the grants of cells that run this code (GRANT_int*) need a new stamp.
- Gate 0: one "# R5" block of added lines in test_i4s.py (r2_i4h_cell.py also leaves the I4H lever's tip-bytes check),
  test_i4cc.py, test_i4dc.py, test_recon7.py, test_j1.py, test_st1.py, test_bf1.py; test_int.py R5_FIX in its allowed
  set. The older Gate-0 tests already list r2_i4h_cell.py.
- Saved records (inta / intc / intd) keep the old fields; the lead's recomputation from the saved vectors stands
  (final-set accepted per request per round 2.617130 / 2.521056 / 2.417250; committed 2.642130 / 2.555017 /
  2.451340).

## 16. TRACK 1, KV movement: the completed-fetch / retention schedule and its ledger (AREA 2, BF; written 2026-11-02T14:15:26Z)

Owner: BF. Human priority of 2026-11-02. Source and CPU only (no GPU allocated). Base: a347f7c (review 5). Dev tree
/mnt/beegfs/ojerbi/nosi-r2-bf1-dev, branch r2-bf1-dev, not pushed. Reference: the original equations (copy of
crafticat/speculative-kv-handoff @ad409e2 docs/ORIGINAL_CALCULATIONS.md, in the session scratchpad). Anchor: C81 / K3 /
B287 / L 2, 84 GB cap. Original: W = B m (n / K) / r = 287 x 6.46583 MB / 10.72418 GB/s = 173.0381 ms, charged inside
max(G, W) (G = 232.48 ms); the covered retained replay assumes ZERO ADDITIONAL VERIFIER FETCH (not zero transfer).
intc's best arm (L4PTFDGUDCDI) had BF off: its 90.464 ms J-probe gather span (LEDGER_rev7 16.8) is the no-prefetch
case; it does not test the intended schedule.

### 16.1 What the intended schedule needs, and where the code stood (source)

Trace of one fetched block (producer -> transfer -> retained destination -> consumer read), file:line at the commit:
1. PRODUCER: the draft residency of layer l (TentStateFD._resid_lean -> TentStateBF._resid_lean -> BgFetch.on_draft):
   the draft's selection is known there; the candidates (skipped loads of complete host blocks) are snapshotted and
   QUEUED AT THAT LAYER. This is the earliest release point the source allows (the selection does not exist earlier);
   what follows is the worker's latency (section 13: snapshot wait, select, C++ pack and submit, issue).
2. TRANSFER: the worker packs the rows into pinned staging (gilfree C++ team), the copy stream lands them in the
   landing ring, the side stream (direct) or the compute stream at the item's joint layer (deferred) COMMITS them into
   the kept store slot (BgFetch._commit): key = layer x 2^32 + block in kid, age kage = g L + layer.
3. RETAINED DESTINATION: the kept store (the arena's B x X slots per head, one store for all layers; X = 129 at the
   anchor). The ONLY evictor between the commit and the read is the joint pass's keep (r2_i4d_keep.keep_assign at each
   layer: dead slots first, then alive slots with the furthest predicted next use). A committed slot is "alive" with its
   next use predicted at its layer, so under capacity pressure an earlier layer's miss CAN evict a retained slot before
   its layer reads it (LOST RETENTION). Nothing protected it before this commit.
4. CONSUMER READ: at the item's joint layer, keep_assign finds the key (a hit: protected, not refilled), and the fork's
   page id of that listed block becomes store_b + slot (r2_i4d_keep._keep_fill, the return value `lphys`): the
   attention reads the retained slot. The keep step is CAPTURED in the graph arms (r2_i4k_combo.py THE SPLIT), so the
   new code there is graph-safe (device step scalar kpass_d, a persistent accumulator flushed eagerly per pass).
Completion: in "defer" a job that lands after its layer ran is dropped ("passed"; its blocks are gathered
synchronously); "drop" abandons every job not issued at the joint pass's start. Neither guarantees completion.

### 16.2 What was built (dev commit 4692273 on a347f7c)

r2_bf1_fetch.py:
- late = "complete" (new LATES value): a late job commits at its layer; the host waits with NO budget, so no job is
  passed; the compute stream waits the job's landing event (the GPU waits a late block; it never reads stale bytes).
- retain = True (RETENTION): before each joint layer's keep, the unread retained slots of LATER layers that still hold
  their keys join keep_assign's `block` mask for that one step, up to the spare capacity of (head, request): spare =
  (X - 1) - this layer's listed entries - test-blocked slots. The keep's capacity proof (r2_i4d_keep (iii)) still holds
  (#misses <= (X - 1) - #hits - |protected|; protected slots hold other layers' keys, so they are never this layer's
  hits). The rest is the OVERFLOW FALLBACK (retain_overflow: those slots stay evictable). The side commit keeps its own
  fixed copy of the test-only mask (kblock_d), so it never sees the temporary mask.
- THE LEDGER (ledger = True, every ce arm): bitmaps req (bool) and com (uint8) of shape (L, H, B, nblk + 1) per epoch
  (requested at on_draft by the drafts; completed at every commit); per joint layer, after the keep, a per-slot
  classification (BgFetch.pre_keep / post_keep through TentStateBF._keep_fill), counters per joint pass (=
  per round), and the CONSUMER-ADDRESS CHECK: the fork's page ids of the listed blocks are marked per slot; every
  retained hit must be read at its slot (addr_ok), else addr_bad. A retained hit is, by definition, not refilled
  (host_x < 0 at its slot), so its read is the retained bytes, not a fresh gather.
- Negative controls (tests only, lneg): "evict_retained" (an eviction before the read), "addr_redirect" (the fork reads
  slot 0 instead of a retained hit's slot).
r2_bf1_cell.py: arm L4PTFDGUDKWCDS (S) = ready snapshots + late "complete" + retain; --bf-ledger; the -BF gate adds
the ledger closure and addr_bad = 0 (MUST), and no passed job in "complete"; REPORT verdicts BF1S-*
(PRED_STAMP_BF1S = None). r2_bf1_step.sh: cell bf1c (C81 pool 18, B287, 512 steps, --bf-store 0 = 129 slots, arms
L4PTFD, L4PTFDGUDKWC, L4PTFDGUDKWCD, X, B, R, S; 8 parts, STEP_MIN 260; STOPS while PRED_STAMP_BF1S is None); bf1m gains S.
No new kernel; no edit outside the BF files.
REPAIRS MAPPED TO THE EVIDENCE (requirement 3): readiness: the snapshot wait (section 13, P1, 115.7 ms per epoch in b3)
-> ready-only snapshots (kept; S uses them). Late issue: "defer" drops a job that lands after its layer -> "complete"
(S). Lifetime: unread retained slots were evictable by earlier layers' misses -> retention with the overflow fallback
(S). Release point: already the draft layer (16.1 item 1); nothing earlier exists in the source. Packing / queueing
(C++ pack 94.5 + submit 76.1 ms per epoch, section 13 C2 / C3): the packer inspection's work, not repeated here; the
blocking call of C3 is still not established. Store size beyond 129 slots: a human decision (16.4).

### 16.3 THE ACCOUNTING (definitions; record bf.arrival.ledger, per joint pass in per_pass, totals)

(a) FETCH side (unique (layer, head, request, block) keys of the epoch): requested (a draft's candidate), completed
    (committed or revived into the store), completed_before_deadline (arrival block: a batch's items whose usable
    arrival precedes the residency start of their joint layer).
(b) DEMAND side, covered: demand = the verifier's unique HOST demand of each joint layer (the keep's listed entries:
    blocks of the joint pass's four rows that neither the window nor the pool holds). hit_retained = found in the store
    in a slot this epoch's fetch committed (still holding its key); hit_cache = found in the store otherwise (kept from
    earlier rounds).
(c) DEMAND side, NOT covered (each one is a synchronous gather now): miss_verifier_only (no draft requested it),
    miss_req_incomplete (requested, not completed this epoch: late / dropped / passed / no room), miss_lost (requested
    AND completed, but not in the map at the read: lost retention).
NO DOUBLE COUNT: (a) counts fetches, (b) and (c) count demand (different populations); within the demand,
demand = hit_cache + hit_retained + miss_verifier_only + miss_req_incomplete + miss_lost exactly (a MUST: closure_ok).
Independent cross-checks (CPU tests): the demand equals the keep's own listed count, the (c) sum its fetched count,
the (b) sum its hits (r2_i4d_keep keep_read). Extra counters: lost_tagged (retained slots of a layer that lost their key
before that layer's keep, demanded or not), retain_protected / retain_overflow (slot-steps), addr_ok / addr_bad.
RESIDUAL VERIFIER TRANSFER = the (c) bytes: residual_MB_per_pass = (c) items x 32,768 B (K + V of one head-block); in
the original's notation B x verifier_union_MB x (1 - a) / r with a = (hit_cache + hit_retained) / demand. The bytes in W
(the fetch during drafting) are NOT added again: they are (a), not (c).

### 16.4 Memory at the anchor (the original capacity equation vs ours; MODEL with the stated inputs)

Original (bytes): M = guarded_C63_backbone(B) + (C - 63) B x 2,105,344 + B x [peak_retained_overflow x (1 + 128/32768)
+ tentative_state] + 3 x 17,039,360 <= 84e9; tentative_state = 2 (K + 1 + L) 32768 + K 8192 = 2 x 6 x 32768 + 3 x 8192
= 417,792 B per request (119.9 MB at B287); extra cache 18 x 287 x 2,105,344 = 10.876 GB; landing 51.1 MB; total
83.7655 GB at B287 (the document). The retained term is the per-request PEAK.
Ours at C81 B287 (intc, no BF, INT arm): allocated peak 76.82 GB MEASURED. It already holds the kept store: one store
for all layers, B x X slots x H 8 x 32,896 B = 263,168 B per slot per request; X 129: 33.9 MB per request, 9.74 GB at
B287 (our "retained" term: a FIXED 128-slot store, not a peak) and the tentative buffers (2.105 MB + about 3.6 MB of
workspaces per request, section 11). BF adds (MODEL, defaults of bf1c):
| part | bytes at B287 | original counterpart |
|---|---|---|
| landing ring (ring 6 x n_cap 6144 x 32,768 B) | 1.208 GB | 3 x 17,039,360 = 51.1 MB |
| transport landing (2 x cap 2048 x 33,296 B) | 0.136 GB | (in the 51.1 MB) |
| snapshot ring (4 L x H x B x topk 64 x 4 B) | 0.075 GB | none |
| ledger bitmaps (2 x L x H x B x (nblk 261 + 1) B) | 0.038 GB | none |
| arrival counters (32,776 B x 210 epochs) + pbid + tags | 0.015 GB | none |
| sum | 1.47 GB | |
So S / R / B at the anchor: 76.82 + 1.47 = 78.3 GB (MODEL; the KW arm's own peak at C81 is not measured: the INT arm's
is used) against the usable 83.75 GB (L60): FITS, about 5.4 GB spare. --bf-store 257 adds 128 x 263,168 x 287 = 9.67 GB:
88.0 GB: DOES NOT FIT. A store sized to a measured peak with an overflow fallback is a HUMAN DESIGN DECISION (not
built): each extra slot costs 75.5 MB at B287, so the spare 5.4 GB allows about 71 more slots (X <= about 200). The
landing ring can also shrink toward the original's 51 MB (smaller n_cap = more batches: the section 13 per-batch cost).

LEAD CORRECTION to 16.4 (2026-11-02T15:57:33Z, `date -u`; checked against records, supersedes the store figures above):
- H in the store is the KV-head count, 2 (config.json num_key_value_heads 2), not 8. One slot per request is
  2 x 32,896 = 65,792 B. Record evidence: intd (R/int_d1_2195630_intd, C73 B318, X 129): store_GB 2.6989 =
  318 x 129 x 65,792 B; LEDGER_rev7 row 7: "ONE store X 129 shared by 32 layers 8.487 MB" per request.
- RE-CORRECTED (2026-11-02, re-optimization analyst; lead re-grep): intc DOES hold the kept store: its record has
  store_GB 2.4358 (= 287 x 129 x 65,792 B). The lead's first grep used the wrong quote form and missed the field (rule
  slip, L70 repeated). The BF owner was right that 76.82 GB holds the store; only its size per slot was wrong.
- Corrected anchor fit (MODEL, allocated basis): 76.82 (intc, measured, store included) + BF parts 1.47 GB (themselves
  overstated by about 0.085 GB by the same H 8 error) = about 78.2-78.3 GB. FITS under the usable 83.75 GB (L60) with
  about 5.4 GB spare.
- X 257: + 128 x 65,792 x 287 = 2.417 GB -> about 80.7 GB: FITS with about 3 GB spare. Each extra slot costs 18.9 MB at
  B287 (not 75.5). The 257-slot store at the anchor is FEASIBLE ON THE MODEL; it stays a human decision (overflow
  fallback, physical fit of the KW arm at C81 not measured).
- The KW arm's own C81 peak is not measured: 80.74 GB uses the INT arm's peak plus the store, MODEL.

### 16.5 MODEL prediction for the anchor (inputs labelled; prediction -> measurement plan)

Inputs: intc J-probe gather span 90.464 ms for 2,022.975 MB of NEW host blocks per round (MEASURED; scope caveat of
LEDGER_rev7 16.8); draft-selected coverage of the joint's NEW blocks c_req = 0.47-0.51 (r7pa / r7pb, MEASURED at C128 /
C73); b3 (C73): sent 1.45 GB per round, draft +11.4 ms, joint overhead beyond the saving 21.3 ms (MEASURED), MODEL
scaled to C81 by the NEW bytes (2,023 / 2,565 = 0.79).
- THE ORIGINAL'S ASSUMPTION vs THE LIVE SELECTION: zero additional verifier fetch needs every verifier block requested
  by a draft and retained. Live, only c_req = 0.47-0.51 of the joint's NEW blocks were draft-selected, so even a
  PERFECT completed + retained schedule leaves miss_verifier_only of about 0.49-0.53 of the NEW bytes: a residual of
  44-48 ms of the 90.5 ms (MODEL). The ledger measures this split directly (miss_verifier_only / demand).
- S (completed + retained): a = c_req x q, q = completion x retention at 129 slots, q = 0.6-0.95 (no measurement at 129
  slots; b3 at 257 slots used 0.62 of the sent items, which includes selection waste): a = 0.28-0.48. Saving in J:
  90.46 a = 25-44 ms. Costs: draft 9-11 ms (6.2 + 2.5 x 1.15 GB, section 12's coefficients), J-side 14-20 ms
  (b3's 21.3 scaled by 0.79, plus the ledger / retention ops about 2-4 ms of 30-50 launches per layer).
  S - X = -6 .. +21 ms per round, central +7 (a 0.38): round 434-470 ms -> S / X = x0.987 .. x1.049, central x1.016.
- R (ready-only, defer, no retention) and B (b3's loop) at C81: as section 13.4 row A and b3: B / X x0.995-1.010,
  R / B x0.998-1.015.
- Proposed bands (for the lead to register; MODEL above): BF1S_vs_X (0.985, 1.05); BF1S_vs_B (0.995, 1.04); BF1S_vs_R
  (0.99, 1.03); BF1S_joint_saving (5, 45) ms; BF1S_covered_share (0.40, 0.90) (hit_cache is inside the demand at
  the anchor and is not modelled separately: wide); BF1S_residual_MB (900, 1,600) MB per joint pass
  (2,023 x (1 - a) with a 0.28-0.48, plus the cache part). REPORT-only (not banded): miss_verifier_only / demand
  (MODEL 0.45-0.55), miss_lost and lost_tagged (retention at 129 slots), retain_overflow, completed / requested,
  completed_before_deadline / completed, late_passed (S: MUST 0), addr_bad (MUST 0).
- What decides the next lever: if miss_verifier_only / demand is about 0.5, the schedule cannot reach the original's
  zero additional fetch with today's selection, whatever the transport; the lever is then the selection overlap (the
  verifier's rows using the drafts' selections, an algorithm change for the human) or the store size. If miss_lost
  is large, the store size (human decision) or the retention order is the lever. If miss_req_incomplete is large, the
  worker path (section 13) still is.

### 16.6 GPU protocol (proposed; the lead runs it; no GPU allocated now)

1. DRYRUN=1 bash benchmarks/Efficiency/r2_bf1_step.sh --nosi <commit> <holder> <run> bf1m, then bf1m (B 8, 72 steps,
   --check-every 1: the keep's overflow / missing / byte checks cover S's retention; every gate MUST). Inspect per ce
   arm: -BF (closure_ok, addr_bad 0; S: late_passed 0), the ledger block, keep check overflow 0.
2. Register PRED_BF1S_BY_B[287] and PRED_STAMP_BF1S (`date -u`) in a commit (and R / B bands for C81 if wanted).
3. bf1c at the anchor (8 parts, STEP_MIN 260, --mem 340G): arms X (fetch off), B (current, b3 loop), R (ready-only),
   S (completed schedule); every ce arm reports the ledger, so the (a) (b) (c) split is measured for B, R and S in one
   job. Score: S / X, S / B, S / R, joint saving, covered share, residual MB per pass; then compare the measured
   residual to 90.46 x (1 - a) with the measured a, and miss_verifier_only / demand to 1 - c_req.

### 16.7 CPU tests and receipts

Pre-commit run of the track-1 selection: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/bf1_dev3.txt (+ .receipt), 23
passed (the new tests, gate 0, cell, step, report in every mode, the older plants). Evidence (CPU world, B 8, 48
tokens; small numbers, plumbing only):
- test_bf1_ledger (B inline / drop, B thread / defer, S thread / complete / ready / retain): demand 846 = the keep's
  listed count; hit_cache 645 + hit_retained 141 = its hits; miss_verifier_only 60 = its fetched count; miss_lost 0,
  miss_req_incomplete 0; addr_ok 141 = hit_retained, addr_bad 0; requested 1,385, completed 461 (all before the
  deadline); S: retain_protected 227, overflow 0, late_passed 0.
- test_bf1_ledger_check_rounds (S, check scenario): keep check overflow 0, dup 0, missing 0, bytes_bad 0 of 846 checked:
  the retention mask keeps the capacity proof and every read byte is the host block's.
- test_bf1_retention_pressure (kslots = the plain arm's most listed blocks): retain off: lost_tagged 12, miss_lost 6;
  retain on: lost_tagged 9, miss_lost 4, protected 197, overflow fallback 9; both: overflow 0, bytes_bad 0.
- PLANT eviction before the read (evict_retained): lost_tagged 310, miss_lost 89 (requested, completed, refetched);
  bytes_bad 0, EQF holds: lost retention is counted and no stale byte is read.
- PLANT late fetch: complete: passed 0, deferred committed 62, sent late 234, completed_before_deadline 227 of 461
  (late but completed, read retained); defer with a 0 ms budget: passed 27, miss_req_incomplete 125 (counted (c)).
- PLANT verifier-only (no draft request): miss_verifier_only 339 = the keep's fetched count, hit_retained 0,
  requested 0.
- PLANT consumer address (addr_redirect): addr_bad 131 = hit_retained, addr_ok 0; -BF FAILS and EQF FAILS (the wrong
  read is caught by both).
FULL SUITE at the commit: /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/bf1_4692273_t.txt (+ .receipt): head 46922736,
status clean, SLURM 2251043 on thunder09 (amd), 2026-11-02T14:56:45Z -> 15:49:53Z, pytest_exit 0, 56 passed (the 45
of a347f7c + 11 new: test_bf1_ledger x3, ledger_check_rounds, retention_pressure x2, lost_retention_plant,
late_fetch_plant x2, verifier_only_plant, consumer_address_plant). The suite now takes 53 min of the 60 min srun
limit: the next additions should split it.
NOT covered on the CPU: the GPU clocks, the stream order of the deferred commits against the copy stream, the cost of
the ledger / retention ops inside the captured graphs (MODEL 2-4 ms per round), the store mix at 129 slots and B287.

## 17. TRACK 2, shared KV reads in the joint pass (AREA 1, J owner; written 2026-11-02T14:22:26Z; dev 55de6f9)

Dev tree /mnt/beegfs/ojerbi/nosi-r2-j1-dev (r2-j1-dev; on fd91420, 14.4). Source and CPU only; no GPU job. The
arithmetic below is printed by /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/j1_ahead/track2_numbers.py (run on amd; output
track2_numbers.out beside it). Labels: MEASURED (a record), DERIVED (arithmetic on MEASURED), MODEL (an assumption,
stated), INFERENCE, UNKNOWN.

### 17.1 What "one physical read" needs

A logical union, shared addresses or a completed CPU transfer do not prove one HBM read. The fork's structure reads each
union tile ONCE per (request, KV head) for its 64 rows (1 draft + 3 verify rows x 16 heads: fork header lines 20-23).
Evidence so far: the fia6 ncu MICROBENCH, fork DRAM bytes 1.017-1.021x the union (MEASURED). The LIVE joint pass: no
ncu (UNKNOWN). 17.3 (cell j1n) and 17.4 (live protocol) are the tests.

### 17.2 Eligible implementations (matched query work = the same 4 rows per request and the same union / per-row
selections; the output contract = the fork's bits, which EQF against L4PTFD requires: L4PTF's joint attention is the
fork, spec_tentative_fork.py:386 joint_attn="fork")

| route | matched work | numerical topology (per output) | contract vs the fork | eligibility | evidence |
|---|---|---|---|---|---|
| FORK (fia6, FV6) | yes: one program per (request, KV head), 64-row tile, union tables, per-row extents | online softmax over the union tiles in ascending block order; one MMA per tile (w4, tile 64) | the contract | ELIGIBLE (shipped) | verify_rows.py:154; fork :20-28 |
| FORKA1-3 (ahead, 14.3) | yes | the same ops in the same order; the GPU code differs (compile witness) | GPU gate FORKAn-EQ | ELIGIBLE AS-IS only if FORKAn-EQ passes (CPU interpreter: 32 / 32 bitwise) | 14.3, 14.4 |
| FORK3D (the fork's split) | yes | segments + reduce_segments: another summation | loses EXACT | NUMERICAL CHANGE (human); MEASURED 1.23-1.46x slower | j1f record; fork :1028-1037 |
| FORK, num_warps 8 / tile 32 | yes | another MMA layout / softmax grouping | bits differ (MEASURED: 851 / 854 / 127 rows equal at w8; 0 at tile 32) | NUMERICAL CHANGE (human); MEASURED slower | SCORING_fia5.md:79-81, :103-112 |
| NS1x4 (flash_attn_nosa, 4 one-row calls) | rows yes; reads each row's 64 pages (no union) | FA2 split-KV 4 + combine | another kernel | NOT ELIGIBLE as a shared read (4 x 64 / 72 = 3.6x the union's logical bytes; MEASURED 2.2x FORK) | j1f; verify_rows.py:46 |
| FLASHU1 type (1 row over the union) | NO (1 row, not 4) | FA2 | another kernel | NOT ELIGIBLE (a control only) | j1f |
| NSRU / route 1 "fast" (flash_attn_nosa + cache_batch_idx over one union window, per-entry bias) | yes (sharing only through L2) | FA2 split-KV; window order | "FAST is NOT bitwise the serial step" | NUMERICAL CHANGE (human); the route exists (route 1) | verify_rows.py:26-40, :48-52 |
| FlashAttention kvcache + block_table (paged) | one row per batch entry; no cross-row read | FA2 | another kernel | NOT ELIGIBLE as-is: page_block_size must be a multiple of 256 (ours: 64) | flash_api.cpp:337 |
| FlashInfer 0.5.3 prefill + custom mask (fia3) | yes (a query tile over the pages); the CIS bias needs a JIT variant | another kernel and order | bits differ | NUMERICAL CHANGE (human); MEASURED 1.68-1.70x of piece 3 (2 CTAs per SM, latency-bound) + 77-100 ms plan / mask per tick | fork :14-17; record fia3_f1a_2195573 |
| FlashInfer cascade (MultiLevelCascadeAttentionWrapper; Batch{Decode,Prefill}WithSharedPrefixPagedKVCacheWrapper; merge_state / merge_states) | partly: the shared level must be COMMON to all rows of a request; our rows share a union with per-row starts (ROW_SHIFT), so the shared level = the intersection (I), the rest per row | two partial softmaxes merged by log-sum-exp | bits differ | NUMERICAL CHANGE (human); bias via JIT variant (not tried) | flashinfer/cascade.py:34, :160, :216, :548, :788 (image, 0.5.3) |
| InfLLM-v2 varlen (NOSA prefill) | no bias input | two calls + copies | another kernel | NOT ELIGIBLE (no bias; copies) | verify_rows.py:42-46 |

### 17.3 Smallest useful evidence: the rows sweep and the ncu cell j1n (READY; the lead runs it)

r2_j1_forkbench.py (55de6f9), on FORKT's union tables (the same pages, extents and V reach; built once by FORK):
- RSMn, n = 1..4: n query rows per request at BLOCK_Q 4 = FORKT's kernel and launch (the same MMA work); MUST gate
  RSMn-EQ (rows bit-equal to FORKT). Masking rows by zero extents was NOT used: a row that sees no token stores 0 / 0
  (softmax_step: L becomes 0), so the query length masks them.
- RSQn, n = 1, 2, 4: BLOCK_Q n (tile rows 16 n); REPORT gate RSQn-EQ (another tile layout).
- Summary `rows_sweep`: masked slope (ms per extra active row at fixed MMA work), tile steps (16 -> 32 -> 64 rows),
  e_fork = RSQ1 - FORK1R (one row over the 72-page union minus one row over its own 64 pages, IN THE FORK: the E
  analogue in the shipped kernel), tile4_fork = FORKT - RSQ1 (the 4-row tile at fixed bytes); SUPPRESSED when a member
  is invalid.
- What separates what: repeated physical reads = ncu DRAM read / logical bytes per arm (expect FORK about 1.0-1.03;
  NS1x4 about 3.6); extra query arithmetic = the RSQ steps (+ ncu occupancy / registers per RSQ); masking / softmax per
  row = the RSM slope; selection / state = excluded (FORKT form; FORK - FORKT = the in-kernel union build); dispatch =
  one launch per call in every form (the flash forms' end-to-end / copy split: FLASHT).
- ncu: --ncu runs one child per arm (FORK, FORKT, FORK1R, FLASHU1, NS1R1, NS1x4, RSM1, RSM4, RSQ1, RSQ2, RSQ4): warm-up
  outside the profiler window, ONE call (one layer) inside, every kernel replayed with the caches flushed
  (--cache-control all). Metrics: gpu__time_duration.sum, dram__bytes_read.sum, lts__t_sectors_srcunit_tex_op_read.sum
  (required); achieved occupancy (sm__warps_active.avg.pct_of_peak_sustained_active), warps per cycle, registers, the
  occupancy limits, L2 hit rate (when the metric query names them). Per arm: DRAM / logical and L2 -> SM / logical.
- Shape: C 81 B 287 with the synthetic per-row 64 pages and union 72 (the C128 / C73 records measured unions of 72.2 /
  72.24; the shape is the same at both, so the C81 "interpolation" is that shape at B 287). The recon7 records' real
  tables are NOT used (NOT CHECKED whether they hold per-layer union tables).
- Cell: r2_j1_step.sh j1n (POOL 18 = C 81, 6 passes, 3D segments 2, ncu = the fia6 pin
  /opt/nvidia/nsight-compute/2025.1.0/ncu, --ncu-cap 240 s per child, STEP_MIN 50 MODEL, OUT_MB 400). GRANT_j1n = None:
  the lead grants it (date -u) in the reconciled run tree (14.4), DRYRUN first:
  `DRYRUN=1 bash benchmarks/Efficiency/r2_j1_step.sh --nosi <sha> <holder> <run id> j1n`.
- CPU tests: test_forkbench_rows_sweep_pure (work, rows mapping, planted value / NaN detected, slopes, suppression),
  test_forkbench_ncu_plumbing (argv, metric filter, the fia6 record's real raw CSV parsed read-only, the summary
  arithmetic, the refusal of a missing ncu), test_ahead_interp (TRITON_INTERPRET=1: RSM rows n = 1..4 and RSQ rows
  n = 1, 2 = the 4-row table read's rows, bitwise, 24 / 24), test_step_script_j1 (the j1n arguments parse).

### 17.4 Live-pass ncu protocol (one joint layer of the INT arm) — PROTOCOL ONLY

Not built: it needs the INT owner's step script (nosi-r2-int: r2_int_step.sh) and a dump hook. Steps:
1. Arm: an INT arm (r2_int_arms.py:32-36; at C73 B318 L4PTFDGUDKWCDIN now; at C81 B287 when an INT cell exists).
2. Wrap the cell's python in ncu: `--kernel-name regex:kernel_unified_attention --launch-skip S --launch-count 1`, S =
   (warm-up joint passes x 32) + l (one steady joint pass, layer l, e.g. 16). Two runs: (a) --cache-control all, the
   required metrics (the kernel's own physical reads); (b) --cache-control none with dram__bytes_read.sum only (one
   replay pass: the live L2 state left by the preceding kernels, e.g. the draft's).
3. Logical bytes of that launch: the union_len of that layer x 32,896 B (a hook in spec_tentative_fork._fork_state that
   saves the layer's union_len after the call: NOT built).
4. Decision: (a) DRAM / logical <= 1.05: one physical read per union page in the live pass; > 1.3: repeated reads
   (then L2 -> SM / logical locates them). (a) - (b) = the live L2 reuse across the draft / verifier kernels.
5. Cost: one profiled kernel; the rest of the cell runs under ncu's interception (MODEL: wall x 1.1-2).

### 17.5 The chain at the anchor (C81 / K3 / B287 / L2)

E (incremental verifier HBM service; NOT the verification budget):
- PREDICTION (original, Oct 12): E = td (1 - f)(U - I) / 63 = 48.7803 / 63 x 8.969029 = 0.77429 ms per group x 8.969
  = 6.9446 ms (DERIVED check: reproduced).
- MATCHED MEASUREMENT (DERIVED from j1f + MODEL interpolation linear in B): the union-bytes increment of one row in the
  flash call, FLASHU1 - NS1R1 = 2.32 / 3.50 ms per pass (B216 / B318) -> 3.141 ms at B287 for 8 pages = 0.3927 ms per
  page; x (U - I) = 3.522 ms = 0.507 x E. Caveats: logical bytes; the j1f flash forms carried the wrapper + copy_ in
  both members (it cancels to first order); the synthetic U - I is 8 pages vs the model's 8.97 groups.
- IN THE FORK: e_fork = RSQ1 - FORK1R (j1n: PENDING the GPU).
- CAUSE of the factor 0.51 (DERIVED + INFERENCE): NS1R1 per page = 0.4027 ms = 0.520 x the model's per-group rate.
  The model prices a group at the draft's whole KV share of its HBM-service proxy (td (1 - f) / 63); the attention
  read of a page costs about half of that. The other half of td (1 - f) is draft KV traffic outside the attention
  read (UNKNOWN: not decomposed here).
- PHYSICAL SHARING: the draft row and the verify rows read ONE union per (request, KV head) in the fork (structure;
  microbench DRAM 1.017-1.021x the union, MEASURED). Live: UNKNOWN (17.4).
- FIX: none needed (E is smaller than predicted). TEST: j1n (FORK DRAM / logical) and 17.4.
- MODEL EFFECT (what-if: the model's E replaced by 3.522): H 213.9039 -> 210.4812; G 232.4806 -> 228.7606 (x0.98400);
  Q x1.01626.

THE 4-ROW TILE (absent from the model):
- PREDICTION: 0. D charges the dense rows g(3B) = 58.3740 (GEMMs), E charges bytes; no term holds the attention
  arithmetic / latency of 4 rows.
- MATCHED MEASUREMENT (j1f, equal logical bytes): FORK - FLASHU1 = 12.50 / 19.13 ms per pass -> 17.115 ms at B287
  (MODEL interpolation). FORK / NS1x4 = 0.4543 at B287 (the shared read saves 55.3 ms vs four one-row calls).
- CAUSE (MEASURED fia5 + DERIVED): the tile loop is latency / occupancy-bound: 255 registers + 59 KB shared -> 2
  programs per SM; 3 synchronous loads per tile; 8.1 us per tile vs 0.73 us of MMA and 1.7 us of HBM share per tile.
  The j1n rows sweep splits it: RSQ steps (tile rows: MMA, registers, occupancy) vs the RSM slope (active rows).
- FIX (bit-preserving): the ahead levels (built; CPU compile witness: K / V lose the async copies, registers 251-255):
  MODEL -15 % .. +10 % of FORK (46.028 ms at B287) = -6.904 .. +4.603 ms per pass; the GPU decides (FORKAn/FORK,
  FORKAn-EQ). F7 (union build outside the kernel): ceiling FORKT - FORK = -1.204 ms per pass (MODEL interpolation),
  before the outside build's own launches. Grid order (KV head fastest): MODEL about 0 (the two heads read disjoint
  bytes; the fork is not bandwidth-bound: 46 % of DRAM peak, fia5). Extents loads: inside the ahead levels.
- MODEL EFFECT on the anchor: the model has no term. Bounding what-ifs (NOT predictions; do not add overlapping times):
  fully serial with H: H 213.9039 -> 231.0189, G 232.4806 -> 251.0820 (x1.0800), Q x0.9259; the tile minus E (if E were
  already counting it): G -> 243.5342 (x1.0475). The overlap assumption max(H, D) is UNTESTED here: in one CUDA stream
  the fork kernel and the GEMMs run one after the other (INFERENCE), so the measured joint pass, not the model's
  maximum, decides. 17.4 measures the live attention kernel.

### 17.6 Bit-changing routes (the human's list; MODEL gain at B287)

| route | MODEL gain | basis |
|---|---|---|
| cascade + merge_state (FlashInfer) | <= 0 | bytes at best equal to the fork's (the union once); FlashInfer's kernel MEASURED 1.68-1.70x of piece 3 (fia3) |
| num_warps 8 at tile 64 | negative | MEASURED +21-31 % slower at B216 (fia5) |
| tile 32 | negative | MEASURED +15-90 % (fia5) |
| row split (BLOCK_Q 2, two programs per request) | UNKNOWN until j1n: gain = FORKT - (the time of two RSQ2 programs at the occupancy they reach) | RSQ2 and its ncu occupancy in j1n; bits: RSQ2-EQ (REPORT) |
| 3D split | negative | MEASURED 1.23-1.46x slower (j1f) |

### 17.7 Commits and receipts

Commit 55de6f9 (on fd91420; not pushed). Receipts (/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/): j1_t2_k1 (targeted:
8 passed, 1 skipped), j1_55de6f9_t (test_j1.py: 28 passed, 1 skipped = the DRYRUN), j1_55de6f9_g0 (Gate 0 of the 15
older files: 37 passed).
