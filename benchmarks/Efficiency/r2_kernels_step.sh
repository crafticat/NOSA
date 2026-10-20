#!/bin/bash
# R2 MILESTONE 2, STEP 4 (KERNELS): ONE bounded GPU step in an EXISTING holder that runs the eager kernel gate
# benchmarks/Efficiency/r2_kernels_gate.py (the per-request stage-1 wrapper nosi/nosi/stage1_seqused.py and the
# per-request pooling kernel nosi/nosi/max_pooling_per_request.py against the shipped calls; the topk order question).
# The pattern of benchmarks/Efficiency/spec_loop_r2_step.sh (pins, claim, frozen record, sacct row, a main() body).
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_kernels_step.sh --nosi <sha> <holder job> <run id>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_kernels_step.sh --nosi <sha> <holder job> <run id>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#
# OUTPUT: OUTROOT/r2m2k_<run>_<holder>/summary.json (every gate: passed, must, detail; the topk order report) and the
# step record OUTROOT/r2m2k_<run>_<holder>.step/ (command, stdout with one line per gate, stderr, exit + accounting,
# sacct row), both frozen. EXIT: the gate's exit = the number of failed MUST gates (G1-G7; G3-tail and G7-order are
# REPORT only); 2 the gate refused (--out exists); 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout;
# 1 a STOP before the launch; 5 an exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): no SLURM_ / SRUN_ / SALLOC_ / SBATCH_ variable in the login shell;
# the checkout at --nosi <sha>, no tracked change, no untracked file under nosi/; this file is the blob of <sha>; the
# image exists; the holder RUNNING, ours, at least MIN_LEFT seconds left, no step besides batch / extern. The claim:
# OUTROOT/r2m2k_active (mkdir, atomic); a stop after the claim leaves it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-kernels                                                # ROOT
GPU_UUID=f31fc723-9dfe-407a-9947-28173556232a                                            # PIN (holder 2188818)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/m2k                                                   # ROOT
SELF=benchmarks/Efficiency/r2_kernels_step.sh
STEP_MIN=20
MIN_LEFT=3600
main() {
stop() { echo "R2 kernels step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_kernels_step.sh --nosi <sha> <holder> <run id>" >&2; exit 2; }
[ $# = 4 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
[ -f "$SIF" ] || stop "no image $SIF"
OUT=$OUTROOT/r2m2k_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/r2m2k_${RUN}_$JOB
NAME=r2m2k-${RUN}-$JOB
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 kernels child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 kernels child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" r2k-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi
      --env PYTHONDONTWRITEBYTECODE=1 --env TORCH_CUDA_ARCH_LIST=8.0 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
      --env TRITON_CACHE_DIR="$JIT/triton" --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_kernels_gate.py" --out "$OUT")
if [ "$DRY" = 1 ]; then
    echo "R2 kernels step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob"
    printf '%q ' "${ARGV[@]}"
    echo
    exit 0
fi
line=$(timeout 60 squeue -j "$JOB" -h -o '%u|%T|%L') || stop "squeue failed"
su=${line%%|*} rest=${line#*|}
ss=${rest%%|*} sl=${rest#*|}
[ "$su|$ss" = "ojerbi|RUNNING" ] || stop "holder $JOB is '$line'"
left=0
case $sl in
    *-*:*:*) left=$(( ${sl%%-*} * 86400 + 10#$(echo "${sl#*-}" | cut -d: -f1) * 3600 + 10#$(echo "$sl" | cut -d: -f2) * 60 )) ;;
    *:*:*) left=$(( 10#$(echo "$sl" | cut -d: -f1) * 3600 + 10#$(echo "$sl" | cut -d: -f2) * 60 )) ;;
esac
[ "$left" -ge "$MIN_LEFT" ] || stop "holder $JOB has $sl left (< $MIN_LEFT s)"
nosteps() {
    steps=$(timeout 60 squeue -s -j "$JOB" -h -o '%i') || return 1
    for s in $steps; do case $s in *.batch|*.extern) ;; *) echo "$s"; return 1 ;; esac; done
}
busy=$(nosteps) || stop "a step is running in $JOB (or squeue -s failed): '$busy'"
mkdir -p "$OUTROOT" || stop "cannot create $OUTROOT"
mkdir "$OUTROOT/r2m2k_active" 2>/dev/null || stop "$OUTROOT/r2m2k_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (r2m2k_active stays)"
mkdir "$REC" && mkdir -p "$JIT" || stop "cannot create $REC / $JIT (r2m2k_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (r2m2k_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (r2m2k_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s holder=%s left=%s run=%s utc_start=%s\n' "$NSHA" "$blob" "$GPU_UUID" \
    "$JOB" "$sl" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" || stop "cannot write $REC (r2m2k_active stays)"
"${ARGV[@]}" < /dev/null > "$REC/stdout.txt" 2> "$REC/stderr.txt"
rc=$?
ok=1
echo "exit=$rc utc_end=$(date -u +%FT%TZ)" > "$REC/exit.txt" || ok=0
sleep 10
timeout 60 env TZ=UTC sacct -j "$JOB" -n -P -o JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS < /dev/null |
    grep -F "|$NAME|" > "$REC/sacct.txt" || ok=0
acct="no single sacct row"
if [ "$(wc -l < "$REC/sacct.txt")" = 1 ]; then
    IFS='|' read -r _ _ sstate scode _ < "$REC/sacct.txt"
    case "$rc:$sstate:$scode" in
        "0:COMPLETED:0:0") acct=consistent ;;
        0:*) acct="inconsistent: exit 0, sacct $sstate $scode" ;;
        *) acct="exit $rc, sacct $sstate $scode" ;;
    esac
fi
echo "accounting=$acct" >> "$REC/exit.txt" || ok=0
[ "$rc" != 0 ] || [ "$acct" = consistent ] || ok=0
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/r2m2k_active" || ok=0
fi
echo "R2 kernels step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
