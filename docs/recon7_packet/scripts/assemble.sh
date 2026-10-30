#!/bin/bash
# Assemble the RECON7 review packet inside the recon7 worktree. Runs on an amd compute node (srun), never on login.
set -euo pipefail
T=/mnt/beegfs/ojerbi/nosi-r2-recon7
K=/mnt/beegfs/ojerbi/kstep-cpu-runs
D=$T/docs/recon7_packet
mkdir -p $D/analysis $D/scripts $D/logs
for f in HANDOFF_RECON7.md LEDGER_rev7.md RECON7_R1_model.md RECON7_R2_joint.md \
         RECON7_R3_draft_transfer_capacity_schedule.md RECON7_R4_prior_art.md RECON7_REVIEW.md \
         HUMAN_RECON_REQUEST_2026-10-30.md SCORING_i4.md CHECKPOINTS.md STATUS_LOG.md \
         HANDOFF_2026-10-29.md LEDGER_rev6.md NOTES_graph_mean_gap.md; do
    cp -p $K/analysis/plan/$f $D/analysis/
done
for s in recon7_r1 recon7_r2 recon7_r3 recon7_s recon7_v ana_rev7; do
    mkdir -p $D/scripts/$s
    cp -p $K/tmp_sc/$s/*.py $D/scripts/$s/ 2>/dev/null || true
done
cp -p $K/tmp_sc/recon7_pkt/assemble.sh $D/scripts/
cp -p $K/cpu-logs/recon7_*.txt $K/cpu-logs/recon7b_*.txt $K/cpu-logs/ana_rev7_*.txt $D/logs/
# Pytest summary line per CPU log (pytest exits 0 only when the summary has passed and no failed / error).
{
  echo "log | pytest summary line"
  for f in $D/logs/recon7b_*.txt; do
      s=$(grep " in [0-9.]*s" "$f" | tail -n 1 || true)
      echo "$(basename $f) | ${s:-no pytest summary (analysis log)}"
  done
} > $D/CPU_TEST_RECEIPTS.txt
# Hashes and sizes of every packet file and of the tree's recon7 code, tests and protocol.
cd $T
{
  echo "# sha256  size_bytes  path (relative to the repository root); computed on $(hostname) at $(date -u +%FT%TZ)"
  for p in $(ls benchmarks/Efficiency/r2_recon7_* benchmarks/Efficiency/tests_cpu/test_recon7.py docs/NOTES_recon7.md; \
             cd $T && find docs/recon7_packet -type f ! -name MANIFEST.txt | sort); do
      echo "$(sha256sum $p | cut -d' ' -f1)  $(stat -c %s $p)  $p"
  done
} > $D/MANIFEST.txt
wc -l $D/MANIFEST.txt
du -sh $D
