#!/bin/bash
# ONE bounded GPU step of the sustained decode + feeder harness (sustained_feed.py) in an EXISTING holder job. It never
# submits, renews, cancels or replaces a holder, never retries, and runs exactly one srun step per invocation.
#
#   bash benchmarks/Efficiency/sustained_step.sh <nosi commit (40 hex)> <holder job> <tag> [driver args ...]
#
# Checks before the launch (each STOPS with exit 1, nothing launched): no SLURM_ / SRUN_ / SALLOC_ / SBATCH_ variable
# in the environment; this worktree's HEAD is <nosi commit> with no tracked change and no untracked file under
# benchmarks/Efficiency and nosi/nosi; the output directory results/sustained/<tag> and its record <tag>.step do not
# exist. The step: srun --jobid=<holder> --job-name=sustained-<tag> --ntasks=1 --cpus-per-task=16 --mem=240G
# --gres=gpu:a100:1 --time=$STEP_MIN --export=NONE, a /bin/sh child that requires CUDA_VISIBLE_DEVICES to name exactly
# one GPU and passes it into the image (apptainer exec --nv --cleanenv) with fresh JIT roots, PYTHONPATH = this
# worktree's nosi/ and NOSI_REPO = this worktree, the driver with --gpu-uuid, under a host timeout of STEP_MIN minutes +
# 60 s (KILL 15 s later). Record: <tag>.step/ (command.txt, stdout.txt, stderr.txt, exit.txt). Exit: the step's exit
# (sustained_feed.py: 0 PASS, 1 FAIL, 2 out exists, 3 another GPU, 4 refused config, 6 module origin; 124 / 137 the
# host timeout; srun's own nonzero exit when the step's --time ends it); 1 a STOP before the launch.
set -u
main() {
HERE=$(cd "$(dirname "$0")" && pwd)
NOSI=$(cd "$HERE/../.." && pwd)                                                          # the worktree root
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                             # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
GPU_UUID=f31fc723-9dfe-407a-9947-28173556232a                                            # PIN (holder 2188818)
STEP_MIN=${STEP_MIN:-60}
stop() { echo "sustained step: STOP: $*" >&2; exit 1; }
[ $# -ge 3 ] || { echo "usage: sustained_step.sh <nosi commit> <holder job> <tag> [driver args ...]" >&2; exit 2; }
SHA=$1 JOB=$2 TAG=$3
shift 3
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || stop "commit '$SHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $TAG =~ ^[A-Za-z0-9_.-]+$ ]] || stop "tag '$TAG'"
case $STEP_MIN in *[!0-9]*|"") stop "STEP_MIN '$STEP_MIN'" ;; esac
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$SHA" ] || stop "$NOSI HEAD is not $SHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- benchmarks/Efficiency nosi/nosi)" ] \
    || stop "$NOSI has untracked files under benchmarks/Efficiency or nosi/nosi"
OUTROOT=$NOSI/results/sustained
OUT=$OUTROOT/$TAG
REC=$OUTROOT/$TAG.step
[ ! -e "$OUT" ] && [ ! -e "$REC" ] || stop "$OUT or $REC exists"
mkdir -p "$OUTROOT" && mkdir "$REC" || stop "cannot create $REC"
JIT=$JITBASE/sustained_$TAG
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "sustained child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
CMD=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60))
     srun --jobid="$JOB" --job-name="sustained-$TAG" --ntasks=1 --cpus-per-task=16 --mem=240G --gres=gpu:a100:1
     --time="$STEP_MIN" --export=NONE /bin/sh -c "$CHILD" sustained-child "$APPTAINER"
     --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi
     --env PYTHONPATH="$NOSI/nosi" --env NOSI_REPO="$NOSI" --env PYTHONDONTWRITEBYTECODE=1
     --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
     --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
     --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
     --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi"
     "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/sustained_feed.py"
     --model "$MODEL" --tokens "$WL/c16k_tokens.bin" --manifest "$WL/c16k_manifest.json" --out "$OUT"
     --gpu-uuid "$GPU_UUID" "$@")
printf '%q ' "${CMD[@]}" > "$REC/command.txt"
printf '\nnosi=%s holder=%s utc_start=%s\n' "$SHA" "$JOB" "$(date -u +%FT%TZ)" >> "$REC/command.txt"
"${CMD[@]}" > "$REC/stdout.txt" 2> "$REC/stderr.txt"
RC=$?
printf '%s\nutc_end=%s\n' "$RC" "$(date -u +%FT%TZ)" > "$REC/exit.txt"
exit $RC
}
main "$@"
