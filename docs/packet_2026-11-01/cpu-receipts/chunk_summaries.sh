#!/bin/bash
# Review 4: put the immutable inta / intc / intd summary.json files into the packet as hash-linked chunks.
# Runs on an amd compute node inside srun, never on login:
#   srun -p amd -c 2 --mem=4G -t 30 bash /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/packet_1102/chunk_summaries.sh
# Then run /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/packet_1101/assemble.sh again (it rebuilds INDEX and MANIFEST).
set -euo pipefail
S=/mnt/beegfs/ojerbi/r2-runs/spec
D=/mnt/beegfs/ojerbi/nosi-r2-packet/docs/packet_2026-11-01/gpu-records
OUT=$D/SUMMARY_CHUNKS.txt
{
  echo "# Hash-linked chunks of summary.json files above 5 MB. Computed on $(hostname) at $(date -u +%FT%TZ)."
  echo "# Rebuild: cat <run>/summary.json.part_* (in name order) > summary.json; its sha256 must equal the ORIGINAL line."
  for r in int_a1_2195630_inta int_c1_2195631_intc int_d1_2195630_intd; do
      src=$S/$r/summary.json
      mkdir -p $D/$r
      split -b 4000000 -d -a 2 $src $D/$r/summary.json.part_
      echo "ORIGINAL $(sha256sum $src | cut -d' ' -f1) $(stat -c %s $src) $src"
      for p in $D/$r/summary.json.part_*; do
          echo "PART $(sha256sum $p | cut -d' ' -f1) $(stat -c %s $p) $r/$(basename $p)"
      done
      re=$(cat $D/$r/summary.json.part_* | sha256sum | cut -d' ' -f1)
      orig=$(sha256sum $src | cut -d' ' -f1)
      if [ "$re" = "$orig" ]; then echo "REBUILD_OK $r $re"; else echo "REBUILD_MISMATCH $r $re vs $orig"; exit 9; fi
  done
} > $OUT
cat $OUT
