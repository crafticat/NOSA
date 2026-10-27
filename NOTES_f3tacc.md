# NOTES f3tacc (branch r2-f3tacc, from r2-f3ldx 3769b64)

## Commits
- One commit on 3769b64: the cell `evalacc216f3t`, its tests, this file. See `git log -1`.

## What changed
- `benchmarks/Efficiency/r2_tick_sg5_step.sh`:
  - New cell `evalacc216f3t`. It is `evalacc216f3lia` with the `evalfast2f3t` flag set:
    `--worker cpp --b2-issue desc --no-transfer-mirror` in place of `--inflight-array`. `GF3=1` (packer v3 of --nosi, DESC 2).
    `--link-record` = the same job's `evalfast2f3t` summary.json.
  - Case line, usage line (appended at the end), header doc.
  - `NOSI_EVAL` = `/mnt/beegfs/ojerbi/nosi-r2-f3tacc` (own worktree, as every earlier tree did).
  - New REGISTERED block "evalacc216f3t" with the line "ACC3T: same gates as ACC3LIA". NOT STAMPED. It has no
    PREDICTION line and no TBD, so no static test reads it as an unstamped block. The lead stamps it.
- Tests: the roots that follow `NOSI_EVAL` now name nosi-r2-f3tacc (hostfix, evalsplit, flean, admitfill, acc216,
  f3lia, f3ld, f3ldx). The three usage-line asserts (hostfix HF6a, f3lia F3G, f3ld TO5) include `|evalacc216f3t`.
  No behaviour assertion changed.
- New `tests_cpu/test_r2_f3tacc.py`: TA1 (static cell, flag set = F3FLAGS of evalfast2f3t, control unchanged,
  REGISTERED ACC3T), TA2 (DRYRUN argv parses with the f3t and acc216 flags), TA3 (stops outside DRYRUN).

## Conflicts
- None (no merge).

## Pytest files the lead must run (compute node, one Triton-interpreter file per job)
- `tests_cpu/test_r2_f3tacc.py` (new; static and DRYRUN, no Triton).
- Static / DRYRUN files with edits: `test_r2_hostfix.py -k HF6`, `test_r2_f3lia.py -k "F3G or F3H"`,
  `test_r2_f3ld.py -k TO5`, `test_r2_f3ldx.py -k "XD5 or XD6 or XD7"`, `test_r2_acc216.py -k AC9`,
  `test_r2_flean.py -k FL7`, `test_r2_admitfill.py -k IA7`, `test_r2_evalsplit.py`.
- No source file changed, so the behaviour files (TO0-TO4, XD0-XD4, AC1-AC8) keep their 3769b64 results.

## Risks
- The serial reference (`serial_reference`, r2_evalacc RealEnv.subset / frc) does not use the residency path, so the
  f3t flags do not change ETD's reference. BI and AE compare the f3t FAST point against the f3t EXACT point.
- `--no-transfer-mirror` needs `--lean-time` and `--arrival-lag >= 1` (parse_args). The cell has both.
- The f3t flags have not run with `--modes fast,bi_c_strict` on GPU before: the EXACT point (bi_c_strict) runs a second
  round on the same OwnResidency config with the desc issue and the table poll. If ACC3T-BI or ACC3T-ETD fails while
  ACC3LIA passes in the same job, the cause is in the f3t path, not in the gates.
- `--link-record` points at `evalfast2f3t` of the same run id and holder: run evalfast2f3t first in the job.
