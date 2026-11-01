#!/bin/bash
# ONE pytest selection of a NOSI worktree, with a RECEIPT. Runs on an amd compute node inside srun (never on login):
#   srun -p amd -c 16 --mem=64G -t 60 bash /mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_fix/run_receipt_tree.sh <tree> <tag> <pytest args...>
# Writes /mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs/<tag>.txt (the pytest output) and <tag>.txt.receipt (the command,
# the host, the SLURM job and step, the UTC start / end, the tree HEAD and its status, the sha256 of every file that
# differs from cd0f0c5 (the recon7 base) plus the test files named in the arguments, the pytest exit).
set -u
TREE=$1
TAG=$2
shift 2
LOGS=/mnt/beegfs/ojerbi/kstep-cpu-runs/cpu-logs
BASE=cd0f0c59c1f1cac8a3ef05f6c3abcc0a2079d926
LOG=$LOGS/$TAG.txt
REC=$LOG.receipt
case $TREE in /mnt/beegfs/ojerbi/nosi-*) ;; *) echo "run_receipt_tree: tree $TREE" >&2; exit 90 ;; esac
[ ! -e "$LOG" ] && [ ! -e "$REC" ] || { echo "run_receipt_tree: $LOG or $REC exists (a new tag)" >&2; exit 90; }
{
    echo "tag=$TAG tree=$TREE"
    echo "utc_start=$(date -u +%FT%TZ) host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} slurm_step=${SLURM_STEP_ID:-none}"
    echo "runner=$0 runner_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
    printf 'pytest_args='
    printf '%q ' "$@"
    echo
    echo "head=$(git -C "$TREE" rev-parse HEAD) branch=$(git -C "$TREE" rev-parse --abbrev-ref HEAD)"
    echo "status_begin (tracked and untracked under benchmarks docs nosi)"
    git -C "$TREE" status --porcelain --untracked-files=all -- benchmarks docs nosi
    echo "status_end"
    for f in $(git -C "$TREE" diff --name-only "$BASE" -- benchmarks nosi docs); do
        [ -f "$TREE/$f" ] && echo "sha256 $(sha256sum "$TREE/$f" | cut -d' ' -f1) $f"
    done
} > "$REC"
apptainer exec --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi \
    --env TMPDIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc,TORCH_EXTENSIONS_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_ext,TORCHINDUCTOR_CACHE_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_inductor,NOSI_REPO=$TREE \
    /mnt/beegfs/ojerbi/nosa-env/nosa-env.sif bash -c 'cd "$0" && env PYTHONDONTWRITEBYTECODE=1 /venv/nosa/bin/python -m pytest -q -p no:cacheprovider -rfEs "$@"' \
    "$TREE" "$@" > "$LOG" 2>&1
rc=$?
echo "pytest_exit=$rc utc_end=$(date -u +%FT%TZ)" >> "$REC"
exit $rc
