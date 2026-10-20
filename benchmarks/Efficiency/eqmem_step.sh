#!/bin/bash
# EQUAL-MEMORY HEADLINE (paper 4.1): ONE bounded GPU step per invocation in an EXISTING holder (the pattern of
# spec_loop_r2_step.sh). One step = one cell = one process of benchmarks/Efficiency/eqmem_bench.py (eqmem_core.py has
# the rules). SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/eqmem_step.sh --nosi <sha> <holder job> <run id> <budget GB> <cell> [<rep>]
#   DRYRUN=1 bash benchmarks/Efficiency/eqmem_step.sh ...   (login node: the pin checks and the printed step command;
#                                                          no squeue, claim, srun or write)
# CELLS (L = 16128, N = 512 generated tokens per request, margin 1 GB, NOSI_ATTN_SPLITS = 0 (the shipped heuristic) and
# PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True in EVERY cell; the batch is eqmem_core.plan_cell's for the budget):
#   original     NOSI_POOL_BLOCKS 0, NOSI_VERIFY_ROUND_SLOTS 0
#   cached:<P>   NOSI_POOL_BLOCKS P (1 .. 160), round slots 0
#   spec:<K>     NOSI_VERIFY_ROUND_SLOTS 62 (W 128), pool 0, K draft steps per round (1 .. 8), LOCKSTEP commit
#   report       eqmem_bench.py --mode report over the run directory (no engine)
# SUGGESTED ORDER for one budget (the lead runs one invocation at a time): original, cached:33, cached:50, spec:2,
# spec:3, report; repetitions with <rep> 1, 2 after the first report.
# OUTPUTS: OUTROOT/eqmem_<run>_<holder>_b<budget>/<cell name>_rep<rep>/ (result.json, tokens.pt, fit_row.json,
# SHA256SUMS; frozen after the step) and <cell name>_rep<rep>.<n>.step/ (command, stdout, stderr, exit with the
# accounting line, sacct row with MaxRSS; frozen); the report: report_<n>/ (report.json, report.md).
# Checks before the claim (each STOPS with exit 1, nothing written, nothing launched): no SLURM_ / SRUN_ / SALLOC_ /
# SBATCH_ variable in the login shell; the NOSI checkout at --nosi <sha>, no tracked change, no untracked file under
# nosi/ or benchmarks/Efficiency/; this file is the blob of <sha>; the image, the model (size pinned) and the workload
# (sha256 of the token file and the manifest pinned) exist; the holder RUNNING, ours, at least MIN_LEFT seconds left, no
# step besides batch / extern. The claim: OUTROOT/eqmem_active (mkdir, atomic: one step at a time); a stop after the
# claim leaves it (the lead decides). Exit: the step's exit (eqmem_bench: 0 VALID, 1 error or OVER_BUDGET, 2 out
# exists, 3 another GPU, 4 refused, 6 module origin; 124 / 137 the host timeout); 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-eqmem                                                       # ROOT
GPU_UUID=f31fc723-9dfe-407a-9947-28173556232a                                            # PIN (holder 2188818)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                             # ROOT
TOK_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                 # PIN (SHA256SUMS)
MAN_SHA=d1c072f032714b5adf3283de34c0b56f431fa406d2e671dcdcf96e4a4d0febd2                 # PIN (SHA256SUMS)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/eqmem-runs                                                    # ROOT
SELF=benchmarks/Efficiency/eqmem_step.sh
N=512 MARGIN=1.0 SPLITS=0 R_SLOTS=62
MIN_LEFT=7200
main() {
stop() { echo "eqmem step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: eqmem_step.sh --nosi <sha> <holder> <run id> <budget GB> <original|cached:P|spec:K|report> [<rep>]" >&2
          exit 2; }
{ [ $# = 6 ] || [ $# = 7 ]; } && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 BUDGET=$5 CELL=$6 REP=${7:-0}
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
[[ $BUDGET =~ ^[0-9]{2}(\.[0-9])?$ ]] || stop "budget '$BUDGET' (GB, e.g. 75 or 72.5)"
case $REP in [0-9]) ;; *) stop "rep '$REP' (0-9)" ;; esac
P=0 R=0 MODE=arm
case $CELL in
    report) MODE=report STEP_MIN=15 NAME_CELL=report ;;
    original) STEP_MIN=50 NAME_CELL=original ;;
    cached:*) P=${CELL#cached:}
              case $P in *[!0-9]*|"") stop "pool '$P'" ;; esac
              [ "$P" -ge 1 ] && [ "$P" -le 160 ] || stop "pool $P (1 .. 160)"
              STEP_MIN=45 NAME_CELL=cached_p$P ;;
    spec:*) K=${CELL#spec:}
            case $K in [1-8]) ;; *) stop "K '$K' (1 .. 8)" ;; esac
            R=$R_SLOTS STEP_MIN=60 NAME_CELL=spec_k${K}_r$R ;;
    *) stop "cell '$CELL'" ;;
esac
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi benchmarks/Efficiency)" ] \
    || stop "$NOSI has untracked files in nosi/ or benchmarks/Efficiency/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
[ "$(sha256sum < "$WL/c16k_tokens.bin" | cut -d' ' -f1)" = "$TOK_SHA" ] || stop "the token file is not $TOK_SHA"
[ "$(sha256sum < "$WL/c16k_manifest.json" | cut -d' ' -f1)" = "$MAN_SHA" ] || stop "the manifest is not $MAN_SHA"
RUNDIR=$OUTROOT/eqmem_${RUN}_${JOB}_b$BUDGET
JIT=$JITBASE/eqmem_${RUN}_$JOB
NAME=eqmem-${NAME_CELL}-r$REP-b$BUDGET-${RUN}-$JOB
if [ "$MODE" = report ]; then
    STEM=report
    DRV=(--mode report --results "$RUNDIR")
else
    STEM=${NAME_CELL}_rep$REP
    DRV=(--mode arm --cell "$CELL" --budget-gb "$BUDGET" --n-new "$N" --margin-gb "$MARGIN" --rep "$REP"
         --model "$MODEL" --tokens "$WL/c16k_tokens.bin" --manifest "$WL/c16k_manifest.json" --gpu-uuid "$GPU_UUID")
fi
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env NOSI_REPO="$NOSI" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
      --env TRITON_CACHE_DIR="$JIT/triton" --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext"
      --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer" --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS
      --env NOSI_POOL_BLOCKS=$P --env NOSI_VERIFY_ROUND_SLOTS=$R)
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "eqmem child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "eqmem child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
argv_for() {      # $1 = the driver's --out
    ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
          --ntasks=1 --cpus-per-task=16 --mem=440G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
          /bin/sh -c "$CHILD" eqmem-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
          "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/eqmem_bench.py" "${DRV[@]}" --out "$1")
}
if [ "$DRY" = 1 ]; then
    argv_for "$RUNDIR/$STEM"
    echo "eqmem step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob"
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
mkdir "$OUTROOT/eqmem_active" 2>/dev/null || stop "$OUTROOT/eqmem_active exists (another step, or an uncertain one)"
mkdir -p "$RUNDIR" || stop "cannot create $RUNDIR (eqmem_active stays)"
n=1
until mkdir "$RUNDIR/$STEM.$n.step" 2>/dev/null; do
    n=$((n + 1))
    [ "$n" -le 100 ] || stop "no free attempt number below 100 (eqmem_active stays)"
done
REC=$RUNDIR/$STEM.$n.step
OUT=$RUNDIR/$STEM
[ "$MODE" = report ] && OUT=$RUNDIR/report_$n
[ ! -e "$OUT" ] || stop "$OUT exists: one output per cell, rep and run (eqmem_active stays)"
mkdir -p "$JIT" || stop "cannot create $JIT (eqmem_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (eqmem_active stays)"
argv_for "$OUT"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (eqmem_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s holder=%s left=%s cell=%s rep=%s budget=%s run=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$JOB" "$sl" "$CELL" "$REP" "$BUDGET" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (eqmem_active stays)"
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
if [ -d "$OUT" ]; then
    find "$OUT" -type f -exec chmod 0444 {} + && find "$OUT" -type d -exec chmod 0555 {} + || ok=0
fi
find "$REC" -type f -exec chmod 0444 {} + && chmod 0555 "$REC" || ok=0
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/eqmem_active" || ok=0
fi
echo "eqmem step: cell $CELL rep $REP budget $BUDGET exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
