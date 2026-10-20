#!/bin/bash
# R2 VERIFY BRANCH (2026-10-20): the GPU gate of the EXACT verify (nosi/nosi/spec_loop.py section EXACT VERIFY: every
# verified position is the shipped decode step; driver benchmarks/Efficiency/spec_exact_gate.py; CPU tests
# benchmarks/Efficiency/tests_cpu/test_spec_exact.py). ONE bounded GPU step per invocation in an EXISTING holder (the
# pattern of spec_loop_r2_step.sh: pins, a claim, a frozen record). SOURCE ONLY until reviewed: it never submits,
# renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/spec_exact_r2_step.sh --nosi <sha> <holder job> <run id> <cell>
#   DRYRUN=1 bash benchmarks/Efficiency/spec_exact_r2_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks and the printed step command only; no squeue, claim, srun or write)
#
# CELLS (pool OFF, NOSI_VERIFY_ROUND_SLOTS 62 = W 128 slots, NOSI_ATTN_SPLITS 4 PINNED; PG-19 test parquet; greedy):
#   smoke          B 4, 2 documents, L 8192, 24 tokens, K 2, arms tf round early: the placement / shape / import
#                  diagnostic, minutes (lesson L25: a small GPU run before the real-model cell).
#   gate:<B>:<D>   L 16128, 112 tokens, K 1 2 3, arms tf round early, B 64 or 128, D 2 or B. In one process: the
#                  shipped decode (the reference tokens and every step's logits), then per K and arm the exact loop
#                  from the same prompt state. GATES (spec_exact_<tag>.md): tf-logits (every computed position's fp32
#                  logits torch.equal the shipped step's, the rolled-back positions included), tf-control (a
#                  comparison with the next step differs), tf-coverage (rollbacks, rejected fills), round-seq /
#                  early-seq (the committed sequences torch.equal the shipped one), round-logits / early-logits,
#                  round-syncs (1 decision sync per round) / early-syncs (<= U - 1). TIMING: draft ms per step, verify
#                  ms per position and per round, rollback ms, ms per committed token, tok/s, served fraction, wire.
# REGISTERED ORDER: smoke, then gate:64:2. A FAIL in smoke stops the order. gate:128:2 only on the lead's decision.
# OUTPUTS: OUTROOT/r2v_<run>_<holder>/spec_exact_<tag>.pt / .md / .json (frozen after the step); per step
# OUTROOT/r2v_<run>_<holder>.<tag>.<n>.step/ (command, stdout, stderr, exit with the accounting line, sacct row incl.
# MaxRSS, copies of the .md / .json), frozen and read back.
# Checks before the claim (each STOPS with exit 1, nothing written, nothing launched): no SLURM_ / SRUN_ / SALLOC_ /
# SBATCH_ variable in the login shell; the NOSI checkout at --nosi <sha> with no tracked change and no untracked file
# under nosi/ or benchmarks/; this file is the blob benchmarks/Efficiency/spec_exact_r2_step.sh of <sha>; the image,
# the model (size pinned) and the parquet exist; the holder RUNNING, ours, at least MIN_LEFT seconds left, no step
# besides batch / extern. The claim: OUTROOT/r2v_active (mkdir, atomic: one step at a time); a stop after the claim
# leaves it (the lead decides). Exit: the step's exit (0 = every gate PASS; n = failed gates; 2 a driver refusal; 3
# another GPU; 4 not exactly one GPU; 124 / 137 the host timeout); 1 a STOP before the launch; 5 an exit-0 step whose
# record is incomplete (no completion token, sacct not COMPLETED 0:0, a failed copy or freeze).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-verify                                                   # ROOT
GPU_UUID=f31fc723-9dfe-407a-9947-28173556232a                                            # PIN (holder 2188818)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
PQ=/mnt/beegfs/ojerbi/datasets/pg19/test-00000-of-00001-29a571947c0b5ccc.parquet         # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/verify                                                # ROOT
SELF=benchmarks/Efficiency/spec_exact_r2_step.sh
R_SLOTS=62 SPLITS=4 ARMS="tf round early"
MIN_LEFT=5400
main() {
stop() { echo "R2V step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: spec_exact_r2_step.sh --nosi <sha> <holder> <run id> <smoke|gate:B:D>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
case $CELL in
    smoke) B=4 D=2 LL=8192 TOT=24 KL=2 STEP_MIN=25 TAG=smoke ;;
    gate:*:*)
        rest=${CELL#gate:}
        B=${rest%%:*} D=${rest#*:}
        case $B in 64|128) ;; *) stop "batch '$B' (64 or 128)" ;; esac
        [ "$D" = 2 ] || [ "$D" = "$B" ] || stop "distinct '$D' (2 or the batch $B)"
        LL=16128 TOT=112 KL="1 2 3" TAG=gateb${B}d${D}
        STEP_MIN=60
        [ "$B" = 128 ] && STEP_MIN=100 ;;
    *) stop "cell '$CELL'" ;;
esac
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi benchmarks)" ] || stop "$NOSI has untracked files in nosi/ or benchmarks/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
[ -f "$NOSI/benchmarks/Efficiency/spec_exact_gate.py" ] || stop "no gate driver in $NOSI"
[ -f "$SIF" ] || stop "no image $SIF"
[ -f "$PQ" ] || stop "no parquet $PQ"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
OUT=$OUTROOT/r2v_${RUN}_$JOB
JIT=$JITBASE/r2v_${RUN}_$JOB
NAME=r2v-${TAG}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
      --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
      --env TRITON_CACHE_DIR="$JIT/triton" --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext"
      --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer" --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_VERIFY_ROUND_SLOTS=$R_SLOTS --env NOSI_MODEL_PATH="$MODEL" --env NOSI_PG19_PARQUET="$PQ"
      --env NOSI_SPEC_OUT="$OUT" --env NOSI_SPEC_TAG=$TAG --env NOSI_SPEC_L=$LL --env NOSI_SPEC_DOCS=$B
      --env NOSI_SPEC_DISTINCT=$D --env NOSI_SPEC_TOTAL_TOKENS=$TOT --env NOSI_SPEC_K_LIST="$KL"
      --env NOSI_EXACT_ARMS="$ARMS")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2V child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2V child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=240G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" r2v-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/spec_exact_gate.py")
if [ "$DRY" = 1 ]; then
    echo "R2V step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob"
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
mkdir "$OUTROOT/r2v_active" 2>/dev/null || stop "$OUTROOT/r2v_active exists (another R2V step, or an uncertain one)"
n=1
until mkdir "$OUT.$TAG.$n.step" 2>/dev/null; do
    n=$((n + 1))
    [ "$n" -le 100 ] || stop "no free attempt number below 100 (r2v_active stays)"
done
REC=$OUT.$TAG.$n.step
mkdir -p "$OUT" "$JIT" || stop "cannot create $OUT / $JIT (r2v_active stays)"
for f in "$OUT/spec_exact_$TAG.pt" "$OUT/spec_exact_$TAG.md" "$OUT/spec_exact_$TAG.json"; do
    [ ! -e "$f" ] || stop "$f exists: one payload per cell and run (r2v_active stays)"
done
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (r2v_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (r2v_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s holder=%s left=%s cell=%s run=%s utc_start=%s\n' "$NSHA" "$blob" \
    "$GPU_UUID" "$JOB" "$sl" "$CELL" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (r2v_active stays)"
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
last=$(tail -n 1 "$REC/stdout.txt" 2>/dev/null) || last=
case $last in
    "SPEC_EXACT_GATE_DONE failed="*) echo "completion=$last" >> "$REC/exit.txt" || ok=0 ;;
    *) echo "completion=absent" >> "$REC/exit.txt" || ok=0
       [ "$rc" != 0 ] || ok=0 ;;
esac
for f in spec_exact_$TAG.md spec_exact_$TAG.json; do
    [ ! -f "$OUT/$f" ] || cp -p "$OUT/$f" "$REC/" || ok=0
done
for f in spec_exact_$TAG.pt spec_exact_$TAG.md spec_exact_$TAG.json; do
    [ ! -f "$OUT/$f" ] || chmod 0444 "$OUT/$f" || ok=0
done
find "$REC" -type f -exec chmod 0444 {} + && chmod 0555 "$REC" || ok=0
[ "$(stat -c %a "$REC" 2>/dev/null)" = 555 ] || ok=0
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/r2v_active" || ok=0
fi
echo "R2V step: cell $CELL exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
