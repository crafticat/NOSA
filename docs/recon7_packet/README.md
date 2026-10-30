# RECON7 review packet (specKV on cached NOSA vs the original October 21 calculation)

Repository: github.com/crafticat/NOSA (git remote `fork`), branch `r2-recon7`. The packet commit is the commit that adds
this directory; its full sha and parent are stated in the delivery message and in `git log`. Code under review (source
only, never run on a GPU): r2-recon7 at e83eec99a1c296963ea802bd9bb8055666526148 (parent
cbafb0a347766ee3436248688cd964be676f1056; earlier 3432876b13fd995a7a56c6e22d3346fb2271ba66 and
8d7dab89d0df8d5298805626275e52ef0530c1a9, from the best arm's tree at 2e9e497de3dd2104341bada4769363dce820c11b).
Every file in this directory is listed with sha256 and size in MANIFEST.txt (computed on an amd compute node).

Status words: MEASURED (in a run record), DERIVED (arithmetic on measured values, script + log given), MODEL (needs an
assumption), CITED (copied from another file). This is an OWNER-REPORTED reconciliation. It is not an independent
acceptance of any numerical row, and FAST correctness stays OPEN (the near-tie contract is the human's decision).

## Where each requested item is

| requested item | location |
|---|---|
| consolidated handoff | analysis/HANDOFF_RECON7.md |
| ledger revision 7 (the one-table reconciliation, per-round spans, schedule mapping, retractions, revised model, discriminators X1-X19, preserved failures, lessons) | analysis/LEDGER_rev7.md |
| review 1: 18 findings and their dispositions | analysis/RECON7_REVIEW.md (findings); analysis/LEDGER_rev7.md sections 10-11 (dispositions, corrections K1-K8) |
| records of the 3 of 39 numbers not reproduced | analysis/LEDGER_rev7.md section 10a; reviewer logs logs/recon7_v_*.txt; reviewer scripts scripts/recon7_v/ |
| review 2: the five points and the supplied model inputs | analysis/LEDGER_rev7.md section 12 (point 1: section 2.6 dependency-feasible schedule; point 2: section 2.7 logical vs physical pages; point 3: per-round spans, the eleven probe rounds, rounds 176 / 192 / 208 and the criterion, warmed vs first-use vs amortized; point 4: dispatch shapes, n = 3 + 3p vs 3 + 5.1p, draft attempts per reject, numerators; point 5: prior-art compatibility matrix); recomputation log logs/recon7_s4.txt |
| workstream inputs | analysis/RECON7_R1_model.md (model), RECON7_R2_joint.md (joint pass), RECON7_R3_draft_transfer_capacity_schedule.md, RECON7_R4_prior_art.md (scoped to the repositories and commits listed in its section 0.2) |
| the original request and calculation, restated | analysis/HUMAN_RECON_REQUEST_2026-10-30.md |
| rung scoring and integrity (I4 families, registration-before-run checks, the preserved OOM) | analysis/SCORING_i4.md, analysis/CHECKPOINTS.md |
| decision log with UTC stamps | analysis/STATUS_LOG.md (entries from 2026-10-30T09:54Z cover this work) |
| test sources and their dependencies | benchmarks/Efficiency/tests_cpu/test_recon7.py; the cells r2_recon7_{protocol,probe,cell,streams,mem}.py and r2_recon7_step.sh; byte copies r2_i4c_cell.py (3aef060), r2_i4m_cell.py and r2_wsm_cell.py (f7fb2c5); the rest of the tree at e83eec9 (nosi/ unchanged from 2e9e497) |
| executed CPU command / exit / log receipts | CPU_TEST_RECEIPTS.txt (pytest summary line per log; pytest exits 0 only when the line has "passed" and no "failed" / "error"); logs/ (all recon7 CPU logs); the srun template is in docs/NOTES_recon7.md and in each workstream file's command list |
| the concrete smallest discriminator protocol (X1, about 5 min) | docs/NOTES_recon7.md, section "X1 protocol (about 5 min)", lines 45-170 at e83eec9 |
| analysis scripts behind DERIVED numbers | scripts/ (recon7_r1, recon7_r2, recon7_r3, recon7_s, recon7_v, ana_rev7) |

## Preserved failures in the CPU receipts

recon7b_wip_a.txt (2 failed) and recon7b_wip_b.txt (1 failed) are pre-commit runs that found a real defect: the
bound-method identity check in the gather and occupancy taps never matched. It was fixed before 8d7dab8. The committed
code's logs are recon7b_8d7dab8_* and recon7b_e83eec9_* (all passed). Other preserved failures (the i4m n373 OOM at C73
B351, refuted registered keys) are in analysis/SCORING_i4.md and analysis/LEDGER_rev7.md section 8.

## GPU state

No GPU run belongs to this packet. PROTOCOL_STAMP_RECON7 is None, so every recon7 cell refuses to run. Current holders:
SLURM 2195630 (nova05, GPU f31fc723) and 2195631 (nova07, GPU 33cfb929), both idle since 2026-10-30T11:04Z; one further
link per chain is queued (2197383 / 2197384); no new allocation or renewal was made. The X1 protocol names 2195630.
