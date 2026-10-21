#!/bin/bash
# R2 STEP 7 (B2), B2TG: ONE short bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_own2_b2t.py
# on CUDA tensors (B2-T on the device; docs/R2_OWN2_B2_DESIGN.md rev 2 section 5.1): the same world, records and ops as
# the CPU run (job 2191023: one thread, 164.364 ms per round tick, the host falsifier FIRED), in both modes (checked:
# every predicate, two host reads per apply; unchecked: the verification-only components skipped), with the component
# timings (each synchronized, plus CUDA events), the pipelined whole-tick timing, one profiled tick per mode (device
# busy time, launches, synchronizations, per component) and the same node's host as a reference. REPORT, scored
# against the block below. No model, no workload, no compile.
# REGISTERED B2TG BEGIN
# frozen 2026-10-21 before the first GPU run (rev 2 section 5.1; r2_own2_b2t.py REGISTERED_DEVICE)
# D1 checked pipelined median per round tick: PREDICTION 9.0 ms, band 5.0-18.0 ms; FALSIFIER > 18.0 ms
# D1u unchecked pipelined median per round tick: PREDICTION 5.5 ms, band 3.0-11.0 ms; FALSIFIER > 11.0 ms
# D2 checked synced component sum: PREDICTION 10.0 ms, band 5.0-20.0 ms; FALSIFIER > 20.0 ms
# D3 profiled device busy time: PREDICTION 3.0 ms, band 1.5-6.0 ms; FALSIFIER > 6.0 ms
# D4 kernel launches per tick: band 500-1200 (point 800); D5 host synchronizations: band 35-120 (point 60)
# D6 every component with wall >= 0.2 ms has profiled device time <= 0.5 x its wall (host-bound)
# REGISTERED B2TG END
# The pattern of benchmarks/Efficiency/r2_gfw_step.sh (E, feature/nosi-r2-gfw2 8fbabe2): the GPU uuid from
# HOLDERS/<holder job id>.gpu_uuid, checked inside the step; --export=NONE; the record and the output frozen.
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_own2_b2tg_step.sh --nosi <sha> <holder job> <run id>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_own2_b2tg_step.sh --nosi <sha> <holder job> <run id>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, B2TG_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses B2TG_HOLDERS.
#
# OUTPUT: OUTROOT/b2tg_<run>_<holder>/b2tg.json, and the step record OUTROOT/b2tg_<run>_<holder>.step/ (command with
# the uuid, its file and the pins, stdout, stderr, exit + accounting, sacct row), both frozen.
# EXIT: the bench's exit = the number of failed coherence checks (counts, predicates, equal final tables); 2 a refusal;
# 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed
# uuid file included); 5 an exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-own2                                                  # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/b2tg                                                  # ROOT
SELF=benchmarks/Efficiency/r2_own2_b2tg_step.sh
STEP_MIN=10
MIN_LEFT=$((STEP_MIN * 60 + 600))
BENCH_ARGS=(--device cuda --mode both --threads 1 --iters 20 --warmup 5 --pipelined-iters 20 --profile --cpu-reference)
main() {
stop() { echo "R2 B2TG step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_own2_b2tg_step.sh --nosi <sha> <holder> <run id>" >&2; exit 2; }
[ $# = 4 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${B2TG_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "B2TG_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$B2TG_HOLDERS
fi
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] || stop "no GPU uuid file $UFILE (the lead writes it for each holder)"
GPU_UUID=$(head -n 1 "$UFILE")
[ "$(wc -l < "$UFILE")" -le 1 ] || stop "$UFILE has more than one line"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold a uuid (no GPU- prefix): '$GPU_UUID'"
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_own2_b2t.py nosi/nosi/r2_own2_ops.py nosi/nosi/r2_residency_sparse.py \
         nosi/nosi/r2_residency_sc.py nosi/nosi/r2_residency_sb.py nosi/nosi/r2_residency.py \
         nosi/nosi/kstep/ownership.py nosi/nosi/kstep/errors.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
BENCH_BLOB=$(git -C "$NOSI" rev-parse "$NSHA:benchmarks/Efficiency/r2_own2_b2t.py")
OPS_BLOB=$(git -C "$NOSI" rev-parse "$NSHA:nosi/nosi/r2_own2_ops.py")
[ -f "$SIF" ] || stop "no image $SIF"
OUT=$OUTROOT/b2tg_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/b2tg_${RUN}_$JOB
NAME=b2tg-${RUN}-$JOB
ENVS=(--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
      --env TRITON_CACHE_DIR="$JIT/triton" --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env TMPDIR="$JIT/tmp"
      --env B2TG_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 B2TG child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 B2TG child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=8 --mem=32G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" b2tg-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_own2_b2t.py" --out "$OUT/b2tg.json"
      "${BENCH_ARGS[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 B2TG step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob gpu_uuid $GPU_UUID ($UFILE); bench blob $BENCH_BLOB ops blob $OPS_BLOB"
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
mkdir "$OUTROOT/b2tg_active" 2>/dev/null || stop "$OUTROOT/b2tg_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (b2tg_active stays)"
mkdir "$OUT" && mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $OUT / $REC / $JIT (b2tg_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (b2tg_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (b2tg_active stays)"
printf '\nnosi=%s wrapper_blob=%s bench_blob=%s ops_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$BENCH_BLOB" "$OPS_BLOB" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$(date -u +%FT%TZ)" \
    >> "$REC/command.txt" || stop "cannot write $REC (b2tg_active stays)"
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
[ "$rc" != 0 ] || [ -s "$OUT/b2tg.json" ] || ok=0
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/b2tg_active" || ok=0
fi
echo "R2 B2TG step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
