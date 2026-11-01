# RECON7 CPU test runs: recovered execution receipts

Written at 2026-10-31T11:17:59Z (from `date -u`). Read-only recovery. No code change, no new job, no commit.

Scope: the 12 files `/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7b_*.txt` (the reviewer said 11; the 12 are
wip_a, wip_a2, wip_b, wip_b2, wip_g0, x1wip_a, 8d7dab8_{a,b,g0}, e83eec9_{a,b,g0}).
This file supersedes the "pytest summary line only" table
`/mnt/beegfs/ojerbi/nosi-r2-recon7/docs/recon7_packet/CPU_TEST_RECEIPTS.txt` as the receipt for these runs.

## 1. Method

1. Log size and mtime: `stat -c '%y %s %n'`. The mtime is UTC+2 on the file system. The table gives UTC.
2. Commands: the builder subagent transcript
   `/mnt/central/users/ojerbi/.claude/projects/-mnt-central-users-ojerbi/66f85c1d-e0d1-4cff-b94d-d3c2572c3f0c/subagents/agent-a080373ac4f503fdc.jsonl`
   ("Build discriminator instrumentation (CPU)"). I matched each `> .../recon7b_X.txt 2>&1` redirect to its Bash
   `tool_use` id, its timestamp and its `tool_result` (which holds the printed `exit $?` or the bash job-control status).
   Two other transcripts mention `recon7b_`: the lead transcript
   (`/mnt/central/users/ojerbi/.claude/projects/-mnt-central-users-ojerbi/66f85c1d-e0d1-4cff-b94d-d3c2572c3f0c.jsonl`, only
   in the lead's prompt to the builder at 2026-10-30T10:17:56Z and in this task's prompt) and the ledger agent
   `agent-a0024040620fbded4.jsonl` (1 mention, no command). The lead did not launch these runs.
3. Jobs: `env TZ=UTC sacct -u ojerbi -S 2026-10-30T10:50 -E 2026-10-30T19:10 -X -P` (partition amd), and then
   `sacct -j ... -o Submit,WorkDir,SubmitLine`. SLURM keeps the full `SubmitLine`. It gives the selector of each job
   independently of the transcript. The match rule: job End = log mtime (to 1 s) AND SubmitLine selector = transcript
   selector AND Submit = transcript tool_use time. All 12 logs match one job each by all three. No log matches two jobs.
4. Source identity: commit times (`git log --format='%H %cI %s'`), plus the `git status --porcelain` output that the
   builder recorded in the transcript right after each commit, plus the order of Edit calls in the transcript. The logs
   themselves record NO commit id, NO tree hash and NO `git status`.
5. Background runs (wip_b, wip_g0) printed `exit $?` into Claude task output files on the login node:
   `/tmp/claude-59005/-mnt-central-users-ojerbi/66f85c1d-e0d1-4cff-b94d-d3c2572c3f0c/tasks/bz9fc17rb.output` ("exit 1") and
   `.../tasks/b7mvm5xbq.output` ("exit 0"). These files still exist (mtime 13:02 and 13:10 local).

Common launcher (all 12 runs; from the transcript and identical in every SubmitLine):

```
srun -p amd -c 16 --mem=64G -t 240 apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi \
  --env TMPDIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc,TORCH_EXTENSIONS_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_ext,TORCHINDUCTOR_CACHE_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_inductor,NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-recon7 \
  /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif bash -c "cd /mnt/beegfs/ojerbi/nosi-r2-recon7 && env PYTHONDONTWRITEBYTECODE=1 \
  /venv/nosa/bin/python -m pytest -q -p no:cacheprovider -rfEs <SELECTOR>" > /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7b_<X>.txt 2>&1
```

The exit of `srun` = the exit of `apptainer exec` = the exit of `bash -c` = the exit of the last command, pytest.
So the SLURM ExitCode IS the pytest exit code. `-rfEs` makes pytest list every failed, errored and SKIPPED test.

Selectors (`<SELECTOR>`; file = `benchmarks/Efficiency/tests_cpu/test_recon7.py` unless stated):

- SA1 (wip_a): `test_recon7.py -k 'statistic_fix_pure or skip_flags or streams or mem_pure or cell_pure or copies or protocol'`
- SA2 (wip_a2, x1wip_a; x1wip_a lists the same words in another order): `test_recon7.py -k 'skip_flags or streams or mem_pure or cell_pure or copies or protocol'`.
  SA1 and SA2 select the same 7 tests (`statistic_fix_pure` matches no test; both runs show 8 deselected).
- SB1 (wip_b): `-x test_recon7.py -k 'probe_equal or probe_gates or statistic_fix'` (4 selected; `-x` stops at the 1st failure)
- SB2 (wip_b2, 8d7dab8_b, e83eec9_b): `test_recon7.py -k 'probe_equal or probe_gates or statistic_fix or probe_plant'` (5 selected)
- SA3 (8d7dab8_a, e83eec9_a): `test_recon7.py -k 'not probe_equal and not probe_gates and not probe_plant and not statistic_fix'` (10 selected; includes `test_step_script_recon7` and `test_notes_recon7`)
- SG1 (wip_g0): `test_wsa.py test_wsb.py test_wsc.py test_i2.py test_i3.py test_i4a.py test_i4g.py test_i4d.py test_i4k.py test_i4u.py test_i4h.py test_i4s.py test_i4cc.py test_i4dc.py test_recon7.py -k 'gate0 or notes_recon7'` (all under `benchmarks/Efficiency/tests_cpu/`; 38 selected)
- SG2 (8d7dab8_g0, e83eec9_g0): the same 14 files without `test_recon7.py`, `-k gate0` (36 selected)

The 8d7dab8 and e83eec9 sets ran as one Bash call that started the three `srun` in the background (`&`) and then
`wait`. The transcript result prints `[1] Done`, `[2]- Done`, `[3]+ Done` for all three. Bash prints `Done` only for
exit 0 (else `Exit N`).

## 2. One row per log

Times are UTC. "tx" = the builder transcript above; the id is the Bash `tool_use` id.

| log (size B; mtime UTC) | command (recovered? source) | selector | job | state/exit | start-end UTC (2026-10-30) | node | source identity | pytest summary | pytest exit 0? |
|---|---|---|---|---|---|---|---|---|---|
| recon7b_wip_a.txt (12085; 10:57:47) | RECOVERED: tx toolu_01K8LFZ4odMC2ArjJVPejFND (10:57:39Z), result "exit 1"; SubmitLine of 2207856 | SA1 | 2207856 | FAILED 1:0 | 10:57:40-10:57:47 | thunder09 | Uncommitted working tree on base 2e9e497 (worktree made 10:18Z), before the tap fix (Edits 10:57:55Z, 10:57:57Z). Runtime source identity NOT RETAINED (no hash, no diff) | 2 failed, 5 passed, 8 deselected, 2 warnings in 4.16s | NO (1; job 1:0, tx "exit 1", log "srun: error: thunder09: task 0: Exited with exit code 1") |
| recon7b_wip_a2.txt (1622; 10:58:19) | RECOVERED: tx toolu_01PozeThVBvjGQ1rKPjxumQN (10:58:11Z), result "exit 0"; SubmitLine of 2207857 | SA2 | 2207857 | COMPLETED 0:0 | 10:58:11-10:58:19 | thunder09 | Uncommitted working tree, after the tap fix. Runtime source identity NOT RETAINED | 7 passed, 8 deselected, 2 warnings in 4.22s | YES (0; job 0:0, tx "exit 0") |
| recon7b_wip_b.txt (3014; 11:02:34) | RECOVERED: tx toolu_01FqmYe8RmwhDySFzKY3FQTD (10:58:29Z, background bz9fc17rb), output file "exit 1"; SubmitLine of 2207858 | SB1 | 2207858 | FAILED 1:0 | 10:58:29-11:02:34 | thunder09 | Uncommitted working tree, with the tap fix, before the test relaxation (Edit 11:10:14Z). Runtime source identity NOT RETAINED | 1 failed, 11 deselected in 238.77s (0:03:58) | NO (1; job 1:0, task output "exit 1", log "srun: error ... exit code 1") |
| recon7b_wip_g0.txt (647; 11:10:00) | RECOVERED: tx toolu_017fV4aHDM1WMQhwLdRK9GE2 (11:01:29Z, background b7mvm5xbq), output file "exit 0"; SubmitLine of 2207889 (WorkDir /mnt/beegfs/ojerbi/nosi-r2-recon7, because the call began with `cd`) | SG1 | 2207889 | COMPLETED 0:0 | 11:01:29-11:10:00 | thunder09 | Uncommitted working tree. Runtime source identity NOT RETAINED | 38 passed, 277 deselected in 507.52s (0:08:27) | YES (0) |
| recon7b_wip_b2.txt (4571; 11:19:45) | RECOVERED: tx toolu_01DtqDoTBuvCR41w6rb2S1SL (11:10:19Z), result "exit 0"; SubmitLine of 2207891 | SB2 | 2207891 | COMPLETED 0:0 | 11:10:19-11:19:45 | thunder09 | Uncommitted tree. Runtime source identity NOT RETAINED. Inference only: the builder made its last Edit at 11:10:14Z and no Edit until commit 8d7dab8 (11:20:01Z), which left `git status --porcelain` empty; so the builder's files equal 8d7dab8. Edits by other sessions are not excluded by this evidence | 5 passed, 10 deselected, 6 warnings in 559.88s (0:09:19) | YES (0) |
| recon7b_8d7dab8_a.txt (1978; 11:20:21) | RECOVERED: tx toolu_01F7kH4pcK3FdXb2TXT2e5uv (11:20:10Z; `LOG=.../recon7b_8d7dab8; $S "$P ..." > ${LOG}_a.txt 2>&1 &`), result "[1] Done"; SubmitLine of 2207894 | SA3 | 2207894 | COMPLETED 0:0 | 11:20:10-11:20:21 | thunder00 | 8d7dab8 (commit 11:20:01Z). Known by: tx toolu_01WDzMdrW7EcwddqqJhuN7ng result at 11:20:02Z shows HEAD 8d7dab8 and porcelain count 0; no builder tool call between 11:20:02Z and launch 11:20:10Z. Runtime check inside the run: `test_step_script_recon7` reached its full DRYRUN branch (it skips unless `_clean_head()` finds a clean tracked tree; no skip shown). The HEAD value itself is not in the log | 10 passed, 5 deselected, 3 warnings in 7.87s | YES (0; job 0:0, bash "Done") |
| recon7b_8d7dab8_b.txt (4571; 11:29:24) | RECOVERED: same call, `> ${LOG}_b.txt`, result "[2]- Done"; SubmitLine of 2207892 | SB2 | 2207892 | COMPLETED 0:0 | 11:20:10-11:29:25 | thunder09 | 8d7dab8, same evidence as row above (same launch). No in-run check of the tree | 5 passed, 10 deselected, 6 warnings in 549.64s (0:09:09) | YES (0) |
| recon7b_8d7dab8_g0.txt (647; 11:28:48) | RECOVERED: same call, `> ${LOG}_g0.txt`, result "[3]+ Done"; SubmitLine of 2207893 | SG2 | 2207893 | COMPLETED 0:0 | 11:20:10-11:28:48 | thunder09 | 8d7dab8, same evidence (same launch) | 36 passed, 264 deselected in 514.85s (0:08:34) | YES (0) |
| recon7b_x1wip_a.txt (1559; 18:51:45) | RECOVERED: tx toolu_01Kmy4hRNPf1Vs7LaQcjjWZG (18:51:38Z), result "exit 0"; SubmitLine of 2217874 | SA2 | 2217874 | COMPLETED 0:0 | 18:51:38-18:51:45 | thunder09 | Uncommitted X1 tree on 8d7dab8. Runtime source identity NOT RETAINED. Inference only: last builder Edit 18:51:23Z; next builder action was commit 3432876 (18:52:06Z), porcelain before it = 3 modified files, after it = 0. So the builder's files equal 3432876. The test file changed again in cbafb0a | 7 passed, 8 deselected, 2 warnings in 4.28s | YES (0) |
| recon7b_e83eec9_a.txt (1915; 18:55:42) | RECOVERED: tx toolu_01WZ66A3kcRmUbKJPQJBAhPD (18:55:30Z; `LOG=.../recon7b_e83eec9`), result "[1] Done"; SubmitLine of 2218005 | SA3 | 2218005 | COMPLETED 0:0 | 18:55:31-18:55:42 | thunder09 | e83eec9 (commit 18:55:21Z). Known by: tx toolu_01Dt4CwspR5QUVsogEyxoqs2 result at 18:55:22Z shows HEAD e83eec9, porcelain count 0, and 0 code paths changed vs 3432876; no builder Edit between 18:55:22Z and launch. In-run check: `test_step_script_recon7` full DRYRUN branch ran (clean tree) | 10 passed, 5 deselected, 3 warnings in 8.00s | YES (0) |
| recon7b_e83eec9_b.txt (4571; 19:04:58) | RECOVERED: same call, `> ${LOG}_b.txt`, "[2]- Done"; SubmitLine of 2218004 | SB2 | 2218004 | COMPLETED 0:0 | 18:55:31-19:04:58 | thunder09 | e83eec9, same evidence (same launch) | 5 passed, 10 deselected, 6 warnings in 563.01s (0:09:23) | YES (0) |
| recon7b_e83eec9_g0.txt (647; 19:02:40) | RECOVERED: same call, `> ${LOG}_g0.txt`, "[3]+ Done"; SubmitLine of 2218006 | SG2 | 2218006 | COMPLETED 0:0 | 18:55:31-19:02:40 | thunder00 | e83eec9, same evidence (same launch) | 36 passed, 264 deselected in 425.58s (0:07:05) | YES (0) |

Container identity (all runs): `/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif`, sha256
`14bfb15e1258821e637b5936f2060d7c7c2be20410fdee2a54ee0ffef754734a`. Job 2217923 (amd, 18:52:11-18:53:08Z, COMPLETED 0:0)
computed this hash; it equals `/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif.sha256`. The image mtime is 2026-09-03, so the same
image served every run (mtime inference). The same job 2217923 recorded sha256 = git blob = HEAD blob for
`r2_recon7_step.sh`, `r2_recon7_streams.py`, `r2_recon7_protocol.py` at HEAD 3432876 (after x1wip_a, before e83eec9).

Other amd jobs in the window are not test runs (by SubmitLine): 2207890 (a 1-CPU `until grep` wait loop over the wip_b and
wip_g0 logs), 2217857 (`s4_review2.py`), 2218327 and 2218344 (`assemble.sh`, the packet).

## 3. Setup diagnostics

Every one of the 12 logs starts with the same 10 lines:

```
WARNING: Environment variable TMPDIR already has value [/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc], will not forward new value [/tmp] from parent process environment
"sed": executable file not found in $PATH
"hostname": executable file not found in $PATH
"sed": executable file not found in $PATH
"grep": executable file not found in $PATH
"cut": executable file not found in $PATH
15:4: not a valid test operator:
15:4: not a valid test operator:
21:4: not a valid test operator: (
21:4: not a valid test operator:
```

D1. TMPDIR WARNING. Producer: `apptainer exec` (the srun command). Cause: the host shell has `TMPDIR=/tmp`; the command
also passes `--env TMPDIR=.../tmp_sc`. Apptainer keeps the `--env` value and does not forward the host value. Effect: none.
The intended TMPDIR was used (pytest's basetemp in wip_b is `/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/pytest-of-ojerbi/...`).

D2. `"sed"/"hostname"/"grep"/"cut": executable file not found in $PATH` and D3. `15:4` / `21:4: not a valid test operator`.
Producer: the container start of `apptainer exec`, BEFORE `bash -c` and pytest start. Evidence:
(a) the lines come first in every log, before pytest's first line;
(b) they occur in the g0 logs, which never call the step script or any RECON7 helper;
(c) the test helper `_step()` (test_recon7.py:761-768) runs the step script with `capture_output=True`, so the step
    script's stderr cannot reach the log;
(d) the same lines occur in 600 of the 702 logs in `/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/`, and the memory note of
    2026-10-01 already records them as "nosa-env.sif startup" lines;
(e) the message formats are those of a Go program (`executable file not found in $PATH` is Go's exec error;
    `line:col: not a valid test operator` is the form of an embedded Go shell interpreter). Apptainer evaluates the
    image's environment scripts with such an interpreter. This point is a format match, not a direct observation.
Diagnosis: a startup environment script in the image calls `sed`, `hostname`, `grep` and `cut` in command
substitutions. The startup interpreter does not find them. The substitutions give empty strings. The `[ ... ]` tests at
line 15 col 4 and line 21 col 4 of that script then get an empty or broken operand (the empty operator and the `(`
operator in D3). So D3 is a consequence of D2. NOT DETERMINED: which script inside the image this is, and why these tools
are not on the startup PATH. Reading the image needs a tool that is not allowed on the login node.
Effect on the tests: none found. The later `bash -c` process finds these tools: `test_step_script_recon7` (in 8d7dab8_a and
e83eec9_a) ran the step script's DRYRUN, which uses `git`, `grep`, `stat` and `sha256sum` before it prints the pins. All 5
cells printed their step command ("RECON7 DRYRUN: 5 cells printed their step command; protocol approved False"), and the
test did not skip. The variables that the failed startup lines compute are NOT RETAINED (no environment dump).

D4. The `_step()` PATH edit. `_step()` removes `kstep-cpu-runs/bin-compute` from PATH when `shutil.which("srun")` is None.
Inside the container `srun` is absent, so this branch runs. Purpose: the bin-compute `sha256sum` wrapper calls srun (L55);
without the edit the DRYRUN would fail in the container. Effect: the DRYRUN reached the workload sha256 check and the
protocol-stamp stop for all 5 cells (no skip, no "pin outside the cell stage"). Whether bin-compute was on the container
PATH at all: NOT RETAINED (PATH not logged).

D5. Skips. No log has a `SKIPPED` line, although `-rfEs` lists skips. No selected test skipped. The words "skipped []" in
the wip_b2/8d7dab8_b/e83eec9_b Evidence lines are probe counters, not pytest skips.

D6. wip_b captured stderr (shown only because the test failed):
`No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'` and
`ERROR:2026-10-30 13:01:34 1618036:1618036 DeviceProperties.cpp:47] gpuGetDeviceCount failed with code 35`.
Producer: the torch stack on a CPU node. The first line is torch's CUDA_HOME notice when no CUDA runtime is present. The
second comes from the torch profiler's start: `r2_recon7_probe.py:881-883` creates `torch.profiler.profile` for the
profiled pass, and the profiler's device query fails with code 35 (no usable CUDA driver). Which module triggers the first
line: NOT DETERMINED. Effect: the profiler records CPU activity only. So the REPORT gate `P-PROF` is False on CPU in every
probe run ("joint windows 0 (busy median None ms)"). `test_probe_gates` asserts only that P-PROF is a REPORT gate
(test_recon7.py:313) and asserts P-PROF-PATH, P-MARKS-NEUTRAL, P-MARKS-SHAPE, P-COUNT True (lines 311-312). So no verdict
changed. The passing b runs captured the same stderr, but pytest does not show captured output for passed tests: NOT RETAINED.

D7. wip_a argparse `error:` lines and the `REFUSED: r2_recon7_mem: PROTOCOL_STAMP_RECON7 is None` line. Producer:
`test_mem_pure` itself (5 `pytest.raises(SystemExit)` negative cases and the planned refusal `main(...) == 20`). They are
expected outputs. They show only because the test failed later (captured stderr).

D8. Evidence text "P-PHASE-SUM: (True, '... out of bound')". Not a diagnostic. The source text is
"out of bound %s" (`r2_i4c_cell.py:597`), and the Evidence line cuts each message at 140 characters
(test_recon7.py:310), so the list is cut off.

## 4. Preserved failures and their corrections

F1. recon7b_wip_a.txt (job 2207856, FAILED 1:0): 2 failed.

```
FAILED benchmarks/Efficiency/tests_cpu/test_recon7.py::test_skip_flags_synthetic
FAILED benchmarks/Efficiency/tests_cpu/test_recon7.py::test_mem_pure - Runtim...
E           RuntimeError: GatherTap: gather_k is not the tap (another wrapper was installed over it)   (r2_recon7_probe.py:442)
E           RuntimeError: OccTap: gather_k is not the tap                                             (r2_recon7_mem.py:138)
2 failed, 5 passed, 8 deselected, 2 warnings in 4.16s
```

Cause: a bound-method identity bug in both taps. `attach()` installed `self._gk` and `detach()` tested
`self.kern["gather_k"] is not self._gk`. Each read of `self._gk` makes a NEW bound-method object, so the identity test was
always true and `detach()` always raised. Fix (builder Edits at 10:57:55Z and 10:57:57Z): store one bound method,
`self._fn = self._gk`, install `self._fn`, and test `is not self._fn`. The fix is in 8d7dab8
(`r2_recon7_probe.py:442`, `r2_recon7_mem.py:138`, checked with `git show 8d7dab8:<file> | grep`).
Correction runs: recon7b_wip_a2.txt (job 2207857, same 7 tests, 7 passed, exit 0); then recon7b_8d7dab8_a.txt and
recon7b_e83eec9_a.txt (10 tests, exit 0).

F2. recon7b_wip_b.txt (job 2207858, FAILED 1:0): 1 failed, then `-x` stopped the run.

```
FAILED benchmarks/Efficiency/tests_cpu/test_recon7.py::test_probe_equal[GUDCD]
>       assert pv["profile"]["skipped"] == [] and pv["profile"]["done"] == prof
E       AssertionError: assert ([(4, 'draft0'...rue, False)')] == []
E         Left contains 2 more items, first extra item: (4, 'draft0', 'marks: no draft graphs of key (True, False)')
1 failed, 11 deselected in 238.77s (0:03:58)
```

Cause: NOT the bound-method bug. wip_b started at 10:58:29Z, after the tap fix. The failure was a test assertion that was
too strict. In the marks mode the probe skips the draft passes of round 4 that have no draft graphs, and it records them in
`profile.skipped` with the reason "marks: no draft graphs of key (True, False)". The test required an empty list. The
builder's note (tx, 11:10Z): "the only failure is in my test's assertion, since the skipped list also includes marked draft
passes from round 4 as expected." The probed-equals-plain claim (`assert bad == []`, earlier in the same test) had passed.
Correction (builder Edit 11:10:14Z, a TEST change, not a code change): the assertion became
`assert not [s for s in sk if not s[2].startswith("marks: ")] and pv["profile"]["done"] == prof, sk`
(test_recon7.py:250 at 8d7dab8). Reviewers note: this relaxes the test to accept "marks: " skips.
Because of `-x`, 3 selected tests (the GUDKCD case, probe_gates, statistic_fix) did not run in wip_b.
Correction runs: recon7b_wip_b2.txt (job 2207891; selector adds probe_plant; 5 passed, exit 0); then recon7b_8d7dab8_b.txt
and recon7b_e83eec9_b.txt (5 passed, exit 0).

## 5. Not retained (summary)

1. Runtime source identity of wip_a, wip_a2, wip_b, wip_g0, wip_b2, x1wip_a: no commit, tree hash, diff or `git status`
   was recorded at run time. (wip_b2 ~ 8d7dab8 and x1wip_a ~ 3432876 are inferences from the builder's Edit order only.)
2. For the 6 commit-named logs: the HEAD value inside the run is not in the log. It comes from the builder's `git status`
   2-9 s before launch. Edits by sessions other than the builder during the run windows are not excluded.
3. Per-file hashes of the tested tree at 8d7dab8 and e83eec9 at run time (only 3 files at 3432876 were hashed, by job 2217923).
4. The environment inside the container (PATH, the variables the failed startup lines compute).
5. The image script that prints D2/D3 and why the tools are missing from its PATH.
6. Captured stderr of passing tests (pytest hides it), e.g. the D6 profiler lines in the passing b runs.
7. Separate per-job exit lines for the parallel 8d7dab8/e83eec9 jobs. Bash printed only "Done"; the SLURM 0:0 rows are
   the per-job receipts.
8. The exact login-side shell exit of the wait call (`wait` with no argument returns 0). It does not matter because
   SLURM records each job's exit.
