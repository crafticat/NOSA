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
bf1_b3_2195631_bf1b (b051b42: b2 + the PIPELINE). Every run: 0 MUST failures (EQF bit-exact against L4PTFD).

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
- In gilfree_pack.cpp (transport owner): overlap the next chunk's pack with the current chunk's H2D (b3 submit 2.73 ms per
  1,581-group chunk = the copy itself: the copy runs synchronously inside the chunk loop): C++ stage 170 -> about 94 ms.
- Fewer bytes per round (S): the per-byte stages (pack, H2D, link in J) scale with S; only a selection with f clearly
  above 0.62 at the same useful bytes would help, and none is available from the counters (above).

STATUS: MEASURED (b1, b2, b3); MODEL calibrated on b2 and confirmed on b3; BF1 at C 73 is at break-even (x1.0049, inside
the spread); C 128: no lever (cell docstring decision, MODEL 0.987-1.006). Dev commits: 4981010 (gates, batch, one
release), 64e307f (tests), a329dc6 (pipeline), b451d91 (drop mode without the pipeline); CPU suites passed at
a329dc6 except test_bf1_late[drop] (fixed in b451d91).
