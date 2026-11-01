#!/bin/bash
# RECON7 review-3 repair: ONE pytest selection of the recon7 tree, with a RECEIPT (review 3 finding 6).
# Runs on an amd compute node inside srun (never on login):
#   srun -p amd -c 16 --mem=64G -t 60 bash /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_fix/run_receipt.sh <tag> <pytest args...>
# Writes /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/recon7f_<tag>.txt (the pytest output) and .receipt (the command,
# the host, the SLURM job and step, the UTC start / end, the tree HEAD and its status, the sha256 of the recon7 sources,
# the pytest exit). The script exits with the pytest exit.
set -u
TREE=/mnt/beegfs/ojerbi/nosi-r2-recon7
LOGS=/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs
TAG=$1
shift
LOG=$LOGS/recon7f_$TAG.txt
REC=$LOG.receipt
[ ! -e "$LOG" ] && [ ! -e "$REC" ] || { echo "run_receipt: $LOG or $REC exists (a new tag)" >&2; exit 90; }
{
    echo "tag=$TAG"
    echo "utc_start=$(date -u +%FT%TZ) host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} slurm_step=${SLURM_STEP_ID:-none}"
    echo "runner=$0"
    echo "runner_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
    printf 'pytest_args='
    printf '%q ' "$@"
    echo
    echo "head=$(git -C "$TREE" rev-parse HEAD)"
    echo "status_begin (tracked and untracked under benchmarks docs nosi)"
    git -C "$TREE" status --porcelain --untracked-files=all -- benchmarks docs nosi
    echo "status_end"
    for f in benchmarks/Efficiency/r2_recon7_protocol.py benchmarks/Efficiency/r2_recon7_probe.py \
             benchmarks/Efficiency/r2_recon7_cell.py benchmarks/Efficiency/r2_recon7_streams.py \
             benchmarks/Efficiency/r2_recon7_mem.py benchmarks/Efficiency/r2_recon7_step.sh \
             benchmarks/Efficiency/tests_cpu/test_recon7.py docs/NOTES_recon7.md; do
        echo "sha256 $(sha256sum "$TREE/$f" | cut -d' ' -f1) $f"
    done
} > "$REC"
apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi \
    --env TMPDIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc,TORCH_EXTENSIONS_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_ext,TORCHINDUCTOR_CACHE_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_inductor,NOSI_REPO=$TREE \
    /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif bash -c 'cd "$0" && env PYTHONDONTWRITEBYTECODE=1 /venv/nosa/bin/python -m pytest -q -p no:cacheprovider -rfEs "$@"' \
    "$TREE" "$@" > "$LOG" 2>&1
rc=$?
echo "pytest_exit=$rc utc_end=$(date -u +%FT%TZ)" >> "$REC"
exit $rc
