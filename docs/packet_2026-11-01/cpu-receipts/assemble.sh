#!/bin/bash
# Assemble the consolidated handoff packet (review 3 + the formula-term milestone) in the packet worktree.
# Runs on an amd compute node inside srun, never on login:
#   srun -p amd -c 2 --mem=4G -t 30 bash /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/packet_1101/assemble.sh
set -euo pipefail
T=/mnt/beegfs/ojerbi/nosi-r2-packet
K=/mnt/beegfs/ojerbi/kstep-cpu-runs
S=/mnt/beegfs/ojerbi/r2-runs/spec
D=$T/docs/packet_2026-11-01
mkdir -p $D/analysis $D/provenance $D/cpu-receipts $D/gpu-records
for f in HANDOFF_INT_2026-10-31.md DISPOSITION_REVIEW3_2026-10-31.md LEDGER_rev7.md STATUS_LOG.md \
         INTEGRATION_NOTES_2026-10-31.md RECON7_RECEIPTS_RECOVERED.md DESIGN_BGFETCH_2026-10-31.md \
         DESIGN_JOINT_EFF_2026-10-31.md HANDOFF_RECON7.md HUMAN_RECON_REQUEST_2026-10-30.md; do
    cp -p $K/analysis/plan/$f $D/analysis/
done
cp -p $K/tmp_sc/recon7_prov/PROVENANCE.md $D/provenance/
cp -rp $K/tmp_sc/recon7_prov/bundle $D/provenance/
cp -p $K/tmp_sc/recon7_fix/run_receipt.sh $K/tmp_sc/recon7_fix/run_receipt_tree.sh $K/tmp_sc/packet_1101/assemble.sh $D/cpu-receipts/
for p in recon7f_ j1_ st1_ st3_ bf1_ int_; do
    for f in $K/cpu-logs/${p}*; do
        case $f in *.txt|*.receipt) cp -p "$f" $D/cpu-receipts/ ;; esac
    done
done
RUNS="recon7_x1r3_2195630_r7s recon7_pm1_2195631_r7pm recon7_mem1_2195630_r7mem recon7_pa1_2195631_r7pa
recon7_pb1_2195630_r7pb j1_m1_2195631_j1m j1_f1_2195630_j1f j1_a1_2195631_j1a j1_b1_2195631_j1b st1_m1_2195630_st1m
st1_a1_2195631_st1a st1_b1_2195630_st1b bf1_m1_2195631_bf1m bf1_m2_2195630_bf1m bf1_b1_2195630_bf1b
bf1_m3_2195631_bf1m bf1_b2_2195631_bf1b bf1_m4_2195631_bf1m bf1_b3_2195631_bf1b int_m1_2195630_intm
int_a1_2195630_inta int_m2_2195630_intm int_b1_2195630_intb int_c1_2195631_intc int_d1_2195630_intd"
{
  echo "# GPU run records: the step record (command, exit, sacct, stdout, stderr) is copied; summary.json is copied"
  echo "# when below 5 MB, otherwise listed with its sha256 and size (traces are never copied). Computed on $(hostname) at $(date -u +%FT%TZ)"
  for r in $RUNS; do
      [ -d $S/$r.step ] || { echo "MISSING $r"; continue; }
      mkdir -p $D/gpu-records/$r
      cp -p $S/$r.step/command.txt $S/$r.step/exit.txt $S/$r.step/sacct.txt $S/$r.step/stdout.txt $S/$r.step/stderr.txt \
          $D/gpu-records/$r/ 2>/dev/null || true
      sj=$S/$r/summary.json
      if [ -f $sj ]; then
          sz=$(stat -c %s $sj)
          if [ "$sz" -lt 5000000 ]; then cp -p $sj $D/gpu-records/$r/; echo "COPIED $r/summary.json $sz";
          else echo "NOT COPIED $r/summary.json $sz $(sha256sum $sj | cut -d' ' -f1)"; fi
      fi
  done
} > $D/gpu-records/INDEX.txt
cd $T
{
  echo "# sha256  size_bytes  path (relative to the repository root); computed on $(hostname) at $(date -u +%FT%TZ)"
  find docs/packet_2026-11-01 -type f ! -name MANIFEST.txt | sort | while read -r p; do
      echo "$(sha256sum "$p" | cut -d' ' -f1)  $(stat -c %s "$p")  $p"
  done
} > $D/MANIFEST.txt
wc -l $D/MANIFEST.txt
du -sh $D
