# NOTES: Phase 2, rung L4P: the paper's schedule (`--spec-arms ... L4P`)

- Plan: `/mnt/beegfs/ojerbi/kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md`, Phase 2.
- Paper: `/mnt/beegfs/ojerbi/kstep-cpu-runs/paper-notes/speckv/speckv.txt`, sections 3, 3.1-3.4, Figure 3.
- Tree: `/mnt/beegfs/ojerbi/nosi-r2-specp`, branch `r2-specp`, from `a6eb5dd` (r2-spec), then fast-forward merges of
  r2-spec `20202f8` (host mirror clone), `00f8964` (truncate: no host write in an open group) and `6de1cfd` (the lead's
  PRED_SPEC stamp), as the coordinator asked.
- Status: SOURCE ONLY. I did not run a test, an interpreter or a job. The lead runs the commands in sections 5 and 6.
- Style: ASD-STE100. Labels: CODE = read at file:line; INFERRED = my reading; MODEL = hand arithmetic.

## 1. What I made

| file | what |
|---|---|
| `nosi/nosi/spec_round.py` | level `"L4P"` (`SpecRunner._round_l4p`), `DraftResident` mode `"rows"` (`_rows_call`), `RequestPin`, the L4P negative controls, `read()` keys for L4P. L3 and L4 keep their code paths. |
| `benchmarks/Efficiency/r2_spec_cell.py` | arm `L4P` (EXACT, gemm rowgroup), `--spec-pin off/on`, the L4P report line, PRED_SPEC keys `L4P_*` and `stamp_L4P` (TBD) |
| `benchmarks/Efficiency/r2_spec_step.sh` | ROOT = this tree; arms `L3,L4X,L4F,L4P` in sp3m / sp3a / sp3b; STEP_MIN 30 / 100 / 140; `--spec-pin off` |
| `benchmarks/Efficiency/tests_cpu/test_spec_paper.py` (new) | the L4P CPU tests (section 6) |
| `benchmarks/Efficiency/tests_cpu/spec_world.py` | `pin=` pass-through, `pinned_slots`, the committed-length host check (`ev["host_c"]`, L4P), the L4P branch of `unkept_fills` |
| `benchmarks/Efficiency/tests_cpu/test_spec_round.py` | two existing tests follow the cell and the script: the verdicts include the L4P keys; the step script has five arms, the new STEP_MIN, `--spec-pin off` and ROOT `nosi-r2-specp` |

NO NEW KERNEL. NO KERNEL EDIT. No change to `ragged_rollback.py`, `verify_rows.py`, `ragged_engine.py` or any shipped
file. The joint pass is `verify_rows.joint_pass` (EXACT) with `VerifyOps(K = K + 1)` (R = K + 2 rows). Everything new is
torch on the existing hooks.

## 2. The schedule (paper section 3, Figure 3; section 3.4)

Request b has its committed position q, its last committed token t (not fed yet) and u TENTATIVE drafts e_1 .. e_u. The
drafts never stop. Every K draft steps, ONE joint pass verifies every tentative draft BUT THE NEWEST (v = u - 1 verify
rows), and drafts the next token. The newest draft is DEFERRED to the next verification group (in the paper its KV
request is one draft step old; rung L5 fetches it during the next drafts).

One round, R = K + 2 rows at q .. q + K + 1 (row r feeds x_r: x_0 = t, x_r = e_r):

| rows | role | residency |
|---|---|---|
| 0 .. v - 1 | VERIFY: row r gives the target's token, which checks e_{r+1} | the target's full selection, the serial residency per row (EXACT) |
| v | DEFERRED row: feeds e_v, makes the draft K / V of its position again; its token is not used | resident-only (DraftResident mode "rows") |
| u = v + 1 | DRAFT row: feeds e_u, gives the next draft e_new | resident-only |
| > u | PAD rows (only when u < K + 1); truncate undoes them | resident-only |

Then: accept (greedy, n = the accepted prefix; a full accept commits v tokens and NO bonus token; a reject commits n
accepted drafts and the target's correction), truncate (full accept: keep rows 0 .. u; reject: keep m = n + 1 rows),
K - 1 drafts (full accept: from e_new; reject: from the correction), commit(m). The next drafts: full accept
[e_u, e_new, y_1 .. y_{K-1}] (u = K + 1); reject [y_1 .. y_{K-1}] (u = K - 1). The group holds 2 K + 1 tentative steps.

FIGURE 3, K = 3 (positions from p = the committed position of round 0):

| round | joint-pass rows | verified | deferred | drafted |
|---|---|---|---|---|
| prelude | 3 draft steps (undone): d1, d2, d3 | - | - | d1, d2, d3 |
| 0 | t@p (checks d1), d1@p+1 (checks d2), d2@p+2 (deferred row), d3@p+3 (draft row), pad@p+4 | d1, d2 | d3 | d4 (draft row), d5, d6 |
| 1, after a full accept (t = d2 at p+2) | d2@p+2 (checks d3), d3@p+3 (d4), d4@p+4 (d5), d5@p+5 (deferred row), d6@p+6 (draft row) | d3, d4, d5 | d6 | d7, d8, d9 |
| after a reject (correction c at q) | c@q (checks c1), c1@q+1 (deferred row), c2@q+2 (draft row), 2 pads | c1 | c2 | c3, c4, c5 |

Round 0 is the paper's sentence: "the joint pass drafts d4 and verifies d1, d2, while d3 is deferred". Round 1 is the
figure's "d3 is deferred to the next verification group: d3, d4, d5".

COMMITTED TOKENS PER ROUND AND REQUEST: at most K - 1 in round 0, K after a full accept, K - 2 after a reject. K >= 3:
with K = 2 a request that rejects has no verify row in its next joint pass (the cell refuses `--spec 2` with L4P).

## 3. Decisions and differences, for the lead

1. THE DEFERRED ROW. Figure 3 shows one draft row. Here the joint pass has one more resident-only row per request. The
   reason: the verify rows must start from the committed engine state (the serial residency of the shipped steps: the
   condition of the bitwise proof), so the round undoes every tentative step before the next joint pass. The draft row
   at q + u reads the K / V of position q + u - 1, and only a draft step made them. The deferred row makes them again in
   the same pass. Cost: one more row of GEMM and one draft residency per request (no host fetch). Its K / V use the
   verify rows' target K / V in the tail (the paper's tentative buffer would hold the earlier draft K / V).
2. THE GROUP SIZE IN THE STEADY STATE IS K, NOT K - 1. The brief says "verifies the OLDEST K-1 tentative tokens" and
   "committed per joint pass <= K-1 + corrections". That holds for the FIRST group only. The figure's next group is
   "d3, d4, d5" (K tokens), and with one joint pass per K draft steps a group of K - 1 would leave one more unverified
   draft every round. I built "verify every tentative draft but the newest". The count test asserts the exact rule:
   per pass m = n + (1 if a reject), n <= v, m <= v <= K, and v = K - 1 in round 0 (so round 0 commits at most K - 1).
3. EXACT ONLY. A FAST L4P needs a per-(request, row) verify mask in `verify_rows._extras` (the union must leave out the
   deferred, draft and pad rows). Not built; `SpecOps(level="L4P", mode="fast")` is refused with that reason.
4. NO HOST WRITE IN AN OPEN GROUP. The schedule uses only `truncate` and `commit`. Since `00f8964` the truncate keeps a
   kept fill staged, and the commit writes it only if it keeps the step. A full accept keeps the deferred row and the
   draft row through the truncate; the commit keeps m = v rows, so a fill of those two rows is dropped. I first wrote an
   opt-in `n_fill` argument for truncate for this case; `00f8964` makes it unnecessary, so I removed it before any
   commit (no rollback change in this tree).
5. THE REQUESTED-KV PIN (paper 3.3), flag `pin` (SpecOps) / `--spec-pin` (cell), OFF by default and in the cells.
   Every draft call records its selection (hits and misses; in the joint pass the draft row only: the deferred row
   remakes the position of the previous round's last draft, which recorded already). Before the joint pass (outside a
   group) each pool slot that holds a recorded block gets Task 1.2's soft pin (`_pool_age = PIN_BASE + n`,
   `PIN_BASE = 1 << 60` as in `pool_prefetch.py`). After the round's commit each slot that is still pinned gets its old
   age back. In this rung nothing is fetched, so the pin keeps the draft's pool HITS until the verify reads them. The
   tokens do not change (a window slot gets the same bytes from the pool or from the host). L5 pins its fetched blocks
   with the same rule. The pins stay through the K - 1 drafts after the joint pass (the journal of those steps holds
   pinned ages, so an earlier release would come back through an undo).
6. PER REQUEST (ragged). Requests have different u and v in one pass. `DraftResident` mode "rows" takes an (R, B)
   mask: in call r of each layer, a request with mask[r, b] gets the draft residency (its host load ids become -1, the
   staged block stays with redirect, its skipped slots are masked in the scratch bias), the other requests get the full
   residency (their load ids go to the gathers unchanged; they read the scratch bias, which has the live values for
   them). One joint pass per K draft steps for the whole batch.
7. THE L5 SEAM. `ops.prefetch.on_draft(miss_ids)` after each draft step and after the joint pass (`miss_ids` = the
   draft row's misses per request). `on_round` as before.

## 4. Negative controls (SpecOps.neg, tests only)

| control | plant | caught by |
|---|---|---|
| `verify_newest` | no deferral: the joint pass verifies the newest draft too | test_paper_sched_defers_newest |
| `keep_joint_draft` | a reject keeps the deferred row and the draft row and drafts on from e_new | test_paper_reject_discards_joint_draft |
| `commit_draft_rows` | the commit keeps the deferred row and the draft row of a full accept (their draft K / V and fill become committed) | test_paper_no_draft_row_reaches_host |
| `commit_deferred` | a full accept also commits the deferred row's token | test_paper_commit_counts |
| `pin_no_release` | the pins stay after the round | test_paper_pin_flag |
| `commit_draft_at_reject`, `miscount` | as for L3 / L4 | tokens test, counts test |

## 5. The GPU cells (PRED_SPEC L4P keys TBD for the lead)

    DRYRUN=1 bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3m
    bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3m    # the miniature first
    bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3a
    bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder> <run id> sp3b

| cell | C (pool) | B | steps | arms | STEP_MIN | mem |
|---|---|---|---|---|---|---|
| sp3m | 128 (65) | 8 | 64 | SH + L3, L4X, L4F, L4P | 30 | 100G |
| sp3a | 128 (65) | 216 | 512 | SH + L3, L4X, L4F, L4P | 100 | 240G |
| sp3b | 73 (10) | 318 | 512 | SH + L3, L4X, L4F, L4P | 140 | 340G |

The step script's ROOT is this tree. Its claim is the same as r2-spec's (`/mnt/beegfs/ojerbi/r2-runs/spec/sp_active_<holder>`):
the two copies cannot run at the same time on one holder (intended). The lead can also run L4P alone through the cell
(`--spec-arms L4P`); the step script has fixed arms.

Per L4P arm the cell reports: committed tok/s (median over steady rounds, mean, effective) and each against SH's of
the same job; acceptance (accepted / verified drafts, the n histogram, full fraction) and the accepted drafts per round;
phases per round (joint, truncate, draft, commit; ms); KV at verify (NEW blocks and MB per round from the joint pass's
host gathers: the verify rows' loads, and for a draft row only the staged block of its group (the redirect); READY;
IN FLIGHT 0; the gather WAIT ms on probe
rounds); the verify rows per pass (histogram); skipped slots per joint-pass row; token and fingerprint checks against
SH (bitwise); the pin counters (None when off). MUST gates: L4P-RUN, L4P-ERR, L4P-COMMIT.

REGISTERED, TBD (keys added to `PRED_SPEC`, with their own stamp `stamp_L4P`; the existing stamp covers the L3 / L4
keys only):

| name | quantity | value |
|---|---|---|
| L4P-TOK | committed tokens bitwise SH's | TBD |
| L4P-RATIO | median steady committed tok/s / SH steady tok/s, (low, high) | TBD |
| L4P-ACCEPT | accepted / verified drafts, (low, high) | TBD |
| L4P-VS-L4X | L4P / L4X median steady tok/s, (low, high) | TBD |

INPUTS FOR THE BANDS (MODEL, not measured), sp3a: SH step 63.6 ms (dr1); draft step 60.2 ms (dr1); the vr1 MEASURED
EXACT rowgroup joint pass of 4 rows (VRX, bitwise, the form of L4X and L4P) = 4.563 SH steps
(`analysis/plan/SCORING_vr1.md` lines 15-20; VRC 4.144 and VRF 6.300 are the other arms), so about 72.5 ms per row.
The L4P joint pass has 5 rows: about 363 ms (two rows load nothing from the host, so at most that). Round = 363 +
truncate 30 + 2 drafts 120 + commit 45 = about 558 ms for at most 3 tokens; SH 191 ms for 3 tokens: L4P-RATIO about
0.34 (0.25-0.45). L4X with the same terms: 290 + 30 + 120 + 60 = about 500 ms per round, so L4P / L4X about 0.9 before
the token difference; L4P commits K - 2 after a reject where L4X commits up to K, so with per-token acceptance about
0.99 the factor is about 0.97: L4P-VS-L4X about 0.87 (0.7-1.05). L4P-ACCEPT: as dr1's draft agreement, 0.97-1.0. This
rung has no fetch: the deferral is a cost here; its benefit needs rung L5.

## 6. CPU tests for the lead

Run on a CPU node, inside a job:

    cd /mnt/beegfs/ojerbi/nosi-r2-specp
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_spec_paper.py

Regressions (the spec suite and the hooks, as in NOTES_spec.md section 5):

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
        benchmarks/Efficiency/tests_cpu/test_spec_round.py benchmarks/Efficiency/tests_cpu/test_rollback.py \
        benchmarks/Efficiency/tests_cpu/test_verify_rows.py benchmarks/Efficiency/tests_cpu/test_ragged_engine.py \
        benchmarks/Efficiency/tests_cpu/test_draft_pool.py benchmarks/Efficiency/tests_cpu/test_pool_prefetch.py \
        benchmarks/Efficiency/tests_cpu/test_r2_mbase.py benchmarks/Efficiency/tests_cpu/test_r2_tick.py \
        tests/test_r2_base.py tests/test_r2_capture_step.py tests/test_r2_p3.py

test_step_script_sp3 (test_spec_round.py) needs a clean checkout of THIS tree at HEAD (its STEP_ROOT is now
`/mnt/beegfs/ojerbi/nosi-r2-specp`); otherwise it skips.

| test | claim | negative control (precondition, L34) |
|---|---|---|
| `test_paper_sched_tokens_equal_shipped_greedy[B1, B8]` | committed tokens = the SHIPPED greedy decode; committed rows' logits bitwise; residency, host rows (twin position and committed length) after every round; no refusal, no overflow; pin off: no soft pin | `commit_draft_at_reject` CAUGHT (a reject of a valid draft). Preconditions: rejects and full accepts, v in {K - 2, K - 1, K}, a block edge, draft skips, pool moves, (B 8) ragged |
| `test_paper_sched_defers_newest[B1, B8]` | round 0: u = K, v = K - 1, the compared drafts are exactly d_1 .. d_{K-1}; every pass: the resident-only rows are rows >= v; after a full accept the next group starts with the deferred token, verifies it with row 0 (a target row) and has e_new second; row 0 never skips (always a verify row); the draft rows skipped host loads | `verify_newest` CAUGHT (d_K exists) |
| `test_paper_reject_discards_joint_draft[B1, B8]` | a reject commits n drafts + the target's correction; after the truncate the twin is at q + m; the drafts start from the correction; the next group is K - 1 new drafts; a full accept keeps rows 0 .. u, drafts from e_new, next group [e_u, e_new, ...] | `keep_joint_draft` CAUGHT (a reject in the plant run) |
| `test_paper_no_draft_row_reaches_host[B1, B8]` | S 4924: round 0's deferred row fills block 76 in a full accept with a pad row; after every round no tentative or draft row in the host rows (both checks) | `commit_draft_rows` and the rollback's `no_defer` CAUGHT (the fill exists in the plant run) |
| `test_paper_commit_counts` (B 8) | per pass m = n + correction, n <= v, m <= v <= K, v = K - 1 in round 0; device counters, n and v histograms, accepted and committed per round = a host recount; prints the accepted drafts per round | `miscount` (m != n somewhere), `commit_deferred` (a full accept) CAUGHT |
| `test_paper_pin_flag` (B 8) | off by default, no soft pin ever; on: recorded, applied > 0, released every round, tokens and invariants unchanged; PIN_BASE = pool_prefetch's | `pin_no_release` CAUGHT (pins applied) |
| `test_paper_rows_mode_per_request[B1, B8]` | mode "rows": a draft request = mode "all" (logits, state rows, miss_ids), a verify request = the target step, bit for bit; counters; bad arguments refused; views restored | preconditions: a draft request skipped a load, a verify request loaded one |
| `test_paper_attached_runner_is_shipped[B1, B8]` | an attached L4P runner (pin off / on), no round: 16 plain steps bitwise the plain world; detach restores every callable | precondition: a block fill in 16 steps |
| `test_paper_ops_refusals` | K < 3, K > 30, fast, L3 / L4 plants, pin outside L4P, gemm exact without linear: refused | - |
| `test_paper_no_host_sync_in_round[pin-off, pin-on]` | only run() and the redirect flag read host values; the rollback none | the trap sees run()'s sync |
| `test_paper_prefetch_wiring` | on_draft K times in the prelude and K times per round (joint pass + K - 1 drafts); on_round begin / end | - |
| `test_paper_cell_arguments_and_report` | L4P arguments and refusals; the cell's summarize of a CPU L4P run: tokens and fingerprints equal the shipped run, the L4P report fields, MUST gates pass | - |

If a precondition fails, the data do not exercise the path: tell me and I change the data, not the claim.

## 7. Risks and open points

- Nothing ran. The CPU tests are my reading of the stand-ins. The most data-dependent preconditions: the draft rows'
  skips (`row_skipped[K - 2:]`), v = K - 2 after a reject before the last round, a target request's host load in the
  rows-mode test, pins applied > 0 (a recorded block in a pool slot at the round start).
- The cell test compares float64 logit sums of a (B, K, V) slice with those of the reference's (B, N, V) tensor. The
  GPU cell relies on the same per-row sum; on the CPU a different reduction order would show here first.
- The deferred row adds one row per request to every joint pass (MODEL: about 72 ms at sp3a B 216, from VRX).
- After a reject, the next pass verifies K - 2 tokens (paper rule "all but the newest"); L4X commits up to K there.
- MEMORY: the journal holds 2 K + 1 = 7 steps (about 2.3 GB at 256 slots, B 216: MODEL) and the joint pass has 5 B
  rows. If L4P stops on memory, lower `--spec-journal-blocks`.
- The pin is SOFT (Task 1.2): when every free slot of a (head, request) row is pinned, pool_update still evicts the
  oldest pin. The release puts back the ages from before apply(); a slot that a kept verify row hit has a real stamp
  already. RequestPin counts applied pins and pins released without a hit; it does NOT count evicted pins separately
  (a hit and an eviction both change the slot's block; Task 1.2's PoolPrefetch counts pin_evicted for its own pins).
- `K_MAX["L4P"] = 30` assumes the twin's ring and one block edge per group as for L4 (2 K + 2 positions per group);
  the cells use K = 3.
- The shared claim name with r2-spec's copy of the step script blocks parallel runs of both copies on one holder.

## 8. Shell commands outside cat / grep / head / tail / ls / sort / cut / diff / git / wc (this task)

- `git worktree add` (as instructed), `git merge --no-edit r2-spec` twice (fast-forward of this new branch, as the
  coordinator asked), `git checkout -- nosi/nosi/ragged_rollback.py` (dropped my own uncommitted edit before the second
  merge; HEAD did not move), `git add`, `git commit -F` (message written with the Write tool): git.
- `date -u +%FT%TZ` about 15 times (a clock read while the read-only reviewer agent ran).
- Shell built-ins only otherwise (`cd`, `echo`, `;`, `&&`, `|`).
- TWO BREACHES (sed, read-only, by mistake): `sed -n 1p /dev/null` inside a grep command (it read nothing), and later
  `grep ... spec_round.py | sed -n 1p` (it printed the first grep line). No other sed, no interpreter, no awk, no
  heredoc, no `cat >>`, no tr / paste / uniq / xargs, no hashing, no `find`, no srun / sbatch / pytest, no sleep. Files
  written with the Write and Edit tools.
