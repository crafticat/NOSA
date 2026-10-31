# NOTES INT: J1 + ST1 / ST2 + BF1 in ONE arm (the integration rung)

Builder: the AREA 1 builder (code); the lead owns the decisions, the registration, the grants and every run.
Design: KS/analysis/plan/INTEGRATION_NOTES_2026-10-31.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs).
Tree: /mnt/beegfs/ojerbi/nosi-r2-int-dev, branch r2-int-dev = r2-j1-int 9eb59ef + r2-j1-dev (the J1 factory export
b4bd542, the test fix d376c92) + r2-st1 2bc0bca + r2-bf1 73e9cb4 (BF1 dev 3: at the merge r2-bf1 had moved there from
9efd728) + the INT files. The run tree (lead-only) is to be /mnt/beegfs/ojerbi/nosi-r2-int (the step script's ROOT).

## 1. The arms

| arm | base (C, B) | hooks | classes |
|---|---|---|---|
| L4PTFDGUDCDI | L4PTFDGUDCD (C 128, B 216) | F1 F2 F6 F3 ST1 ST2 | r2_j1_select's J3 runner -> st1_runner(twin) -> IntLayer |
| L4PTFDGUDKWCDI | L4PTFDGUDKWCD (C 73, B 318) | F1 F2 F6 ST1 ST2 BF1 | bf_classes -> j1_wrap -> st1_runner(twin) -> IntLayer + lock |

F3 is not in the BF1 arm (INTEGRATION_NOTES 3.4: F3 runs the draft residency inside the capture; BF1's draft hook must
run eagerly; there is no keep J3 arm).

ST3 (AREA 3, r2-st3-dev 639d4ec, merged fast-forward; KS/analysis/plan/INTEGRATION_NOTES_2026-10-31.md section 9):
the C 73 INT arm L4PTFDGUDKWCDI now also carries ST3 (the keep fill in two compiled helpers). The cell builds its runner
as `r2_st1_commit.st3_runner(r2_int_arms.int_runner(...))` (INT_ST3 in r2_int_cell.py; r2_int_arms.int_runner stays
without ST3: test_st1.test_st3_int wraps it itself) and adds `-KFILL` (r2_st1_cell.kfill_gate). The ST3 control arm
L4PTFDGUDKWCDMTK (= MT + ST3, its PEQ reference MT, which needs M) runs in intm and intb. Registration keys (None):
INT_st3_joint_saving (MT joint - MTK joint), INT_st3_vs_mt (MTK / MT); the ST1 cell's ST3-JOINT-SAVING / ST3-VS-MT
verdicts are emitted too (r2_st1_cell.verdicts).

ST4 (AREA 3, r2-st4-dev da73561 on d87b071 = the ST3 fix: ST3's compile events in its own record, so the root -COMPILE
gate no longer sees them; INTEGRATION_NOTES section 10): BOTH INT arms carry ST4 (the draft residency's integer ops
in one compiled helper): runner = st4_runner(st3_runner(int_runner)) at C 73, st4_runner(int_runner) at C 128 (INT_ST4);
gate -DRESID (r2_st1_cell.dresid_gate; F3 = the J3 record present). No ST4 timing control in inta / intb (GPU time;
its effect is inside the INT arm, MODEL about 2 ms per round); the controls L4PTFDGUDCDMTR (MT + ST4) and
L4PTFDGUDKWCDMTKR (MTK + ST4) run in intm only (gates, correctness; their PEQ chain M, MT is in intm). No new
registration key. CPU: test_int.py::test_int_equal evaluates the cell's gates on the INT run + ST3 + ST4
(`_int_full`: st3_attach / st4_attach in the cell's order) and checks that run = the INT run, every bit.

BF1 64e307f (merged on the run commit 5418a3a; the lead's registration and grants kept): batched jobs (--bf-batch 8),
--bf-cap-groups 2048, ring / n_cap per cell (intm, inta 8 x 2048; intb 6 x 6144 = 1.21 GB landing), the passed
items, BF1's COUNTS gate for a store size other than REF's (r2_bf1_cell.counts_gate_bf: union.cap / store_slots
declared and checked against X) and the store-X warm (TentStateBF.bf_warm_joint: loc2 compiles before the first
capture). INT: the C 73 arm's COUNTS = counts_gate_bf; its warm runs on the J1 section's locate (F6) path
(r2_int_arms.bf_state_int: pre, _j1_loc, loc2, the same inputs as the section's; J1's F6 call count restored), since
the INT arm never calls the I4CC loc1. Memory at intb: MODEL about 81.7 GB (BF1's B arm about 81.6 GB from bf1b's
MEASURED 80.80 GB + 0.67 landing + 0.12 transport staging, + ST1 pool + helpers), about 2.0 GB under 83.75 GB.
PRED_INT_BY_B untouched (the lead re-registers 318).

FIXES after the CPU run of 00bfb51 (cpu-logs int_00bfb51_int.txt):
- Gate 0 (f): r2_st1_cell.py's comment named the INT cell file: reworded ("the INT cell"); r2_st1_commit.py's ST3 /
  ST4 docstrings name the INT classes' module: test_int declares both files (NAMED_BY: text allowed, no import of an
  r2_int module).
- -DRESID (C 128): rpost compiled twice (draft pass 1 and pass 12: `chk` was a Python bool, so the first check pass
  compiled a second graph). r2_st1_commit.rpost_fn now takes chk as a 0-d bool tensor (`on = sel & chk`; off adds 0:
  the same counter values) and ResidST4 passes `_st4_chk[checking]`: one graph. CPU: test_st1.test_st4_one_graph.
- C 73: Inductor's CPU backend failed to compile kfa_fn (`valid & ~(hv != 0)`: VecMask<float> & VecMask<uint8_t>,
  /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_inductor/5j/c5jgj5faa2jqbfdmxgyfo4l3h7z7gjdkj4hgj4ugqxxuwt2bgp5d.cpp
  line 92). OPEN: the ST worker fixes kfa_fn in r2-st4-dev (owner of r2_st1_commit.py); INT merges that commit.

BF1 68028b1 (BF1 dev 4-7, merged on b095d84: the DEFER mechanism, 14 packer threads, 1024-group chunks, the 250 ms
host budget, the room check, the lead's BF1 registration). The defer hook is in TentStateBF._resid_joint (eager in the
C 73 INT arm: no F3 there), so it does not touch IntLayer. The INT cell's BF1 options = r2_bf1_cell.bf_opt's keywords
(gf cap_groups, late, late_wait_ms); the step script passes r2_bf1_step.sh's: --bf-ring 16 --bf-threads 14
--bf-cap-groups 1024 --bf-late defer --bf-late-wait-ms 250 (test_int checks both scripts). CPU: test_int_defer (thread
worker, defer) = the D arm's run, INT gates pass.
MEMORY, C 73 INT arm at B 318 (MODEL): BF1's own B arm 81.0 GB (the BF1 writer's MODEL: L4PTFDGUDKWCD 77.69 GB MEASURED
in i4ccb + the store at X 257 + the landing ring 1.07 GB at ring 16) + ST1's private pool (< 0.05 GB; the ST1 band
ST1_peak_delta_gb (-0.01, 0.05)) + the compiled helpers' buffers (J1 pooling / F6 tables / ST2 / ST3 / ST4: about
0.03 GB at B 318, MODEL from their shapes) = about 81.1 GB; the usable ceiling 83.75 GB (L60) leaves about 2.6 GB
(margin rule >= 0.6 GB). The MEM gate of intb measures it.

## 2. The two collisions (benchmarks/Efficiency/r2_int_arms.py)

1. ST2 x F1: both rungs override TentState._layer with a full copy. `IntLayer._layer` = `TwinST2._layer` with 2 lines
   marked "# INT": ST2's `VR = self._st2_vr` becomes `VR = None` (the J1 selection needs no VR), and `VR._select_rows(plan.vs, layer, q, twin)` becomes `self._j1_select(plan.vs, layer, q, twin, l)`.
   The runner re-classes the state to (IntLayer, the ST state) after st1_runner's re-class; ST2's fallback (R 1, a
   record) stays the next `_layer` after TwinST2 (the J1 state's). CPU: test_int.py::test_int_layer_copy.
2. ST1 x BF1: `bf.wrap_backend(T._st1["backend"])` after both setups (the commit capture holds BF1's capture lock and
   drains the gilfree transport first). CPU: test_int.py::test_int_lock (exactly one more locked capture).
ST2's VR lookup (fix after the CPU run of b0d68a4: test_int_stacked raised KeyError 'VR' at r2_st1_commit.py:359,
`base.__globals__["VR"]` with a J1 base, whose module binds no VR): `_st2_setup` now finds the shipped verify_rows
module once, as the first inherited `_layer` whose module binds VR (spec_tentative.py:123), and `_layer` reads
`self._st2_vr`. The module keeps its torch / time imports (test_st1 Gate 0 (e)); the ST arms' behavior is unchanged.
Negative controls (tests only, `ineg`): "stacked" (no IntLayer; the run is the same, so EQF cannot see it; J1-COMPILE,
J1-PEQ, INT-CLASSES, INT-LAYER catch it) and "nolock" (INT-LOCK catches it).
Also: `r2_j1_select.j1_wrap / j3_wrap` (AREA 1 commit b4bd542, no behavior change) build J1 on any base classes.

## 3. The cell (benchmarks/Efficiency/r2_int_cell.py) and its gates

The J1 cell's pattern: the I4DC cell loaded a second time ("r2_int_dcc"), its tables extended, its run / runner /
gate / verdict functions wrapped; SHJ1 after SH (r2_j1_cell's _BCProxy); the ST1 commit capture round joins the L61
exclusion (_HCProxy). Every arm's read() carries ST1's tail checksum (the TAIL-EQ reference). The single-rung arms may
run in the same job with their own cells' gates (J1 / J3, ST M / MT, BF1 X / B).
MUST gates of an INT arm: the D arm's I4DC gates under the INT name (COMPILE without loc1; EQF = L4PTFD bit for bit;
the C 73 arm without COUNTS / KEEP-EQ / KEEP-REF and with BF1's COUNTS (no round_blocks) / LISTED / BF against the D
arm); J1-PEQ, J1-COMPILE, J1-DRAFT; J1-RESID (C 128); PEQ (= the D arm), DCOUNTS (C 128), TAIL-EQ, CGRAPH, TWIN;
INT-CLASSES, INT-LAYER, INT-LOCK (C 73); J1-SHEQ once per job.
Verdicts (PRED_INT_BY_B, all None, PRED_STAMP_INT None): INT-VS-REF, INT-RATIO, INT-VS-SHJ1, INT-VS-STRONGEST,
INT-JOINT/DRAFT/COMMIT-SAVING, INT-PEAK-GB, INT-EQ; REPORT lines INT-ADDITIVITY (the single-rung arms' ratios in the
same job and their product vs X / REF) and INT-ARM. The builder's MODEL is the comment block above PRED_STAMP_INT.

## 4. CPU selectors (the lead runs them on amd)

`pytest -q benchmarks/Efficiency/tests_cpu/test_int.py` (Gate 0, copy, classes, both arms x timing / check = the D arm
and L4PTFD bit for bit with every component counted, the controls, the gate plants, the cell, the registration, the
step script) and the Gate-0 tests of the rungs and of the 15 older rungs: test_j1.py, test_st1.py, test_bf1.py,
test_wsa, wsb, wsc, i2, i3, i4a, i4g, i4d, i4k, i4u, i4h, i4s, i4cc, i4dc, recon7 (`-k gate0` is the short form; the
full files are the merge check).
Gate-0 blocks: one "# INT" block in each of the 15 older tests; in test_j1 / test_st1 / test_bf1 one block (the other
rungs' files + the INT files; `*_LATER = ("r2_int_arms.py", "r2_int_cell.py")`) and their name scan's skip line marked
"# INT".

## 5. GPU protocol (the lead runs it; grants and registration are the lead's)

Step script benchmarks/Efficiency/r2_int_step.sh (a copy of r2_j1_step.sh: no hash on login, child pins, default-deny
grants GRANT_intm / GRANT_inta / GRANT_intb in r2_int_cell.py, OUT_MB 2048, the timing cells stop while PRED_STAMP_INT
is None). Every cell: --st1-commit rows, BF1 --bf-ring 8 --bf-worker thread --bf-close-wait-ms 0 --bf-packer gilfree
--bf-threads 8 (TORCH_EXTENSIONS_DIR = the step's JIT directory).

| cell | config | arms (SH, SHJ1 first) | STEP_MIN / --mem | command |
|---|---|---|---|---|
| intm | C 73 pool 10, B 8, 72 steps, probe 4, check every round, stress, --bf-acct 1, --bf-store 0 | L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDM,L4PTFDGUDKWCDM,L4PTFDGUDCDMT,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDCDMTR,L4PTFDGUDKWCDMTKR,L4PTFDGUDCDI,L4PTFDGUDKWCDI | 55 / 100G | `bash benchmarks/Efficiency/r2_int_step.sh --nosi <sha> <holder> <run> intm` |
| inta | C 128 pool 65, B 216, 512 steps | L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDI | 125 / 240G | `... inta` |
| intb | C 73 pool 10, B 318, 512 steps, --bf-store 257 | L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDKWCDI | 270 / 340G | `... intb` |

intb with the ST3 control costs 3 more B 318 arm parts (M, MT, MTK: about 54 min MODEL). Without the control, intb is 4
arms and STEP_MIN 170 (the earlier form); the lead decides (the control needs M and MT: MTK's PEQ reference chain).

Order: intm first (every gate MUST). Decision (the lead's): adopt an INT arm when every MUST gate passes and INT-VS-REF
is inside its registered band; INT-ADDITIVITY reports whether the components' savings add (a job with the single-rung
arms too is the lead's choice: the cell accepts them; each adds one arm part of time).

## 6. Open risks

- Memory at C 73 B 318: BF1's landing ring + store 257 (bf1b's setting) + ST1's private pool on the D arm's 77.7 GB
  peak; the MEM gate shows it (lesson L60: usable about 83.75 GB).
- F3 + ST2 + F1 in one C 128 arm: each was gated alone; the CPU world runs them together (test_int_equal C128).
- BF1 dev commits after 73e9cb4 merge in on the lead's word.

## 7. Commands (login node)

`git -C /mnt/beegfs/ojerbi/nosi-r2-j1 worktree add /mnt/beegfs/ojerbi/nosi-r2-int-dev -b r2-int-dev r2-j1-int` (the
lead's command), `git -C /mnt/beegfs/ojerbi/nosi-r2-int-dev merge --ff-only r2-j1-dev`, `merge --no-ff --no-commit
r2-st1`, `merge --no-ff --no-commit r2-bf1` (conflicts resolved with the Edit tool: every block kept), `add`, `commit -F`,
`log`, `status`, `diff`, `rev-parse`; `ls`, `wc`, `grep`, `head`, `date -u`. No interpreter, no sed, no heredoc, no hash.
