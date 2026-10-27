# NOTES f3tall (branch r2-f3tall = r2-f3all 3a7f51a + merge of r2-f3ldx 3769b64)

## Commits
- One merge commit: parent 1 = 3a7f51a (f3all), parent 2 = 3769b64 (f3ldx, range 89c9169..3769b64). It holds the
  conflict resolution, the joining line, the three tall cells, the REGISTERED TALL block, the test fixes, the new
  test file and this file. See `git log -1`.

## Conflicts and how I resolved them (both feature sets kept in every case)
- `nosi/nosi/r2_own2.py` OwnResidency signature: kept `issue`, `no_transfer_mirror` (f3ldx) AND `end_batch` (f3all).
- `benchmarks/Efficiency/r2_tick_sg5.py`: make_engine passes `issue`, `no_transfer_mirror` and `end_batch`;
  main's meta keeps `end_batch` / `apply_batched` and `worker` / `b2_issue` / `no_transfer_mirror`.
- `r2_tick_sg5_step.sh`: header docs of both sides; REGISTERED F3LD block (f3ldx) added next to the F3ALL block;
  NOSI_EVAL = nosi-r2-f3tall; usage line = f3all cells, then f3ld/f3ldx cells, then tall cells; case lines and cell
  blocks of both sides.
- Tests: the roots that follow NOSI_EVAL name nosi-r2-f3tall (hostfix, evalsplit, flean, admitfill, acc216, f3lia,
  f3ld, f3ldx, f3all, applyev, endbatch). Usage-line asserts: HF6a has the full list; f3lia F3G, f3all FA7 and f3ld
  TO5 check their own run of cells and "] [gilfree|gather]" (cells follow).
- `test_r2_acc216.py` AC2 (line compare of the timed round): PORT_SHA = 89c9169 (the merge base; its r2_tick_sg5.py
  is the same blob as f3all's 5cdfc75). The delta lists are the union of both sides (one_point: f3ldx's 3 REPORT
  lines + applyev's 5 lines; make_engine: issue_clock, 2 engine assignments, the merged keyword-argument run). The
  "each line writes one REPORT key" check still holds for the f3ldx lines. Every other AC2 assertion of both sides kept.
  AC9a keeps the stricter f3ldx form.

## Semantic merge (the one line of joining code)
- b2-desc added `_hm[b] = -1` (drop the ended request's write-back marks) to `OwnResidency.end_request`.
  endbatch's `OwnResidency.end_requests` was written before b2-desc and did not have it. A text merge would leave
  `--end-batch --b2-issue desc` with stale marks in the ended rows (state no longer equal to the per-request ends).
  I added the same drop to `end_requests` (a loop over the ended slots; host tensor; no kernel).
- `test_r2_endbatch.py` EB4a compared end_request to endbatch's base text; it now removes exactly the two b2-desc lines
  (asserted to occur once) before the compare. No other assertion changed.
- T21 gate lists in `test_r2_f3ld.py` (TO4) and `test_r2_f3ldx.py` (XD5) now include `apply_batched_gates` (f3all's
  own lists already had it).

## New cells
- `evalfast2tall` = evalfast2f3t + `--end-batch --apply-batched` (TALL_BANDS=TBD, refused outside DRYRUN).
- `evalfast2c3tall` = evalfast2c3t + the two flags (CHECK).
- `evalacc216tall` = acc216 proof on the tall flags, link = evalfast2tall.
- Controls unchanged: evalfast2f3t, evalfast2all, evalfast2f3lia (and evalfast2c3t, evalfast2c3all, evalacc216all).
- REGISTERED TALL block: NOT STAMPED, every PREDICTION "TBD by lead".
- "desc acted" has NO harness gate (I did not add gate code). C3TALL registers it as a MUST that the lead scores
  from the record: residency.report.b2 issue "desc", desc_jobs > 0, desc_takes > 0, transfer_mirror false,
  no_transfer_mirror.jobs > 0, info.issue desc_calls > 0. `_AB_acted` is a real harness MUST gate.

## Pytest files the lead must run (compute node; one Triton-interpreter file per job)
- NEW `tests_cpu/test_r2_f3tall.py` (builds packer v3; no Triton): TL1 (end-batch == per-request ends on the desc path,
  stream / pool, mirror / no-mirror, every tick, plus _hm), TL1z (preconditions), TL2 (negative control: the joining
  line removed is seen), TL3-TL5 (harness, cells, DRYRUN).
- Triton-interpreter files (one per job): `test_r2_f3all.py` (FA child), `test_r2_f3lia.py`, `test_r2_f3ld.py`
  (TO1f; it also builds packer v3).
- Packer v3 files: `test_r2_b2_desc.py`, `test_r2_f3ldx.py`.
- Others with edits: `test_r2_endbatch.py`, `test_r2_applyev.py`, `test_r2_acc216.py`, `test_r2_hostfix.py -k HF6`,
  `test_r2_flean.py -k FL7`, `test_r2_admitfill.py -k IA7`, `test_r2_evalsplit.py`.

## Risks
- End path: YES, end-batch and the desc issue touch the same per-request bookkeeping (`_hm`, the T4 write-back marks).
  Fixed by the joining line; TL1 / TL2 prove it on CPU. If TL1z says "no ended slot held a write-back mark", the CPU
  world does not exercise the line, and only the static TL3 check covers it.
- `--no-transfer-mirror` keeps the in-flight jobs in `_nm_jobs` / `_nm_keys` (keyed by issue tick, not by request).
  end_request does not touch them and neither does end_requests; the table entries of the ended slots are written by
  O.end_owners exactly as by O.end_owner (endbatch's own proof), so the table poll sees the same owners. TL1 (nm
  cases) checks this per tick on CPU.
- Apply path: TickEngine._apply_batched runs after backend.decisions and writes only the twins' draft banks and the
  draft ring rows; the desc jobs write store lanes in the history arena [0, A); the ring rows are outside it. No CPU
  test runs desc + apply-batched together (apply-batched needs the Triton child); C3TALL on GPU is the first joint
  proof. Run evalfast2c3tall first in the job.
- TL1 does not prove 2+ ends in ONE call on the desc path (the forced round may end requests in separate ticks);
  the batching itself is proven by test_r2_endbatch.py / test_r2_f3all.py on the object path. TL1z prints the
  coverage.
- No harness gate for "desc acted" (see above).
