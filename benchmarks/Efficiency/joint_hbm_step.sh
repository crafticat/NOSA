#!/bin/bash
# PAPER SECTION 4.5 ("Does joint execution reduce GPU-memory traffic?"): ONE bounded GPU step in an EXISTING holder that
# runs benchmarks/Efficiency/joint_hbm.py --mode orchestrate for one batch size: the unprofiled cell U (gates, latency,
# launch / idle overhead, exact accounting), the Nsight Compute cells P_none (--cache-control none, an L2 scrub before
# every window) and P_all (--cache-control all, the cold-L2 control), the CSV exports and the summary. The pattern of
# benchmarks/Efficiency/spec_loop_r2_step.sh (pins, claim, frozen record, sacct row, a main() body). SOURCE ONLY until
# reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/joint_hbm_step.sh --nosi <sha> <holder job> <run id> <b16|b32|b64>
#   DRYRUN=1 bash benchmarks/Efficiency/joint_hbm_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#
# CELLS: L = 16128, K = 3, W = 20, the first B documents of the c16k workload (sha256 pinned), NOSI_ATTN_SPLITS = 4,
# NOSI_POOL_BLOCKS = 0, all four forms (sep_private, joint_private, sep_shared, joint_shared), 10 timed repetitions.
# The private windows cost B (K + 1) x 4.2 MB x 32 layers (34 GB at B = 64); B > 64 is not offered here.
# OUTPUT: OUTROOT/jh_<run>_<holder>_<cell>/ (config.json, U/ result.json + capture.pt + traces, P_none/ P_all/
# windows.json, rep_*.ncu-rep, *_raw.csv, *_raw_nvtx.csv, U.log, P_*.log, summary.json, summary.md) and the step record
# OUTROOT/jh_<run>_<holder>_<cell>.step/ (command, stdout, stderr, exit + accounting, sacct row), both frozen.
# EXIT: the orchestrator's exit = the number of failed MUST gates (G1-G5 of U, O_U_ran, O_P_<cc>_ran,
# O_P_<cc>_windows, O_P_none_single_pass); 2 a refusal (--out exists, bad arguments, a refused workload or engine
# knob); 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an exit-0
# step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): no SLURM_ / SRUN_ / SALLOC_ / SBATCH_ variable in the login shell;
# the checkout at --nosi <sha>, no tracked change, no untracked file under nosi/; this file is the blob of <sha> and the
# harness files are tracked at <sha>; the image, the model (size pinned) and the workload token file (sha256 pinned) exist; the holder
# RUNNING, ours, at least MIN_LEFT seconds left, no step besides batch / extern. The claim: OUTROOT/jh_active (mkdir,
# atomic); a stop after the claim leaves it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-joint-hbm                                                   # ROOT
GPU_UUID=f31fc723-9dfe-407a-9947-28173556232a                                            # PIN (holder 2188818)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
NCU=/opt/nvidia/nsight-compute/2025.1.0/ncu                                              # PIN (inside the image; job 2175791)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/jhbm                                                  # ROOT
SELF=benchmarks/Efficiency/joint_hbm_step.sh
SPLITS=4
main() {
stop() { echo "joint-HBM step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: joint_hbm_step.sh --nosi <sha> <holder> <run id> <b16|b32|b64>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
case $CELL in
    b16) B=16 STEP_MIN=50 CHILD_S=1500 ;;
    b32) B=32 STEP_MIN=60 CHILD_S=1800 ;;
    b64) B=64 STEP_MIN=80 CHILD_S=2100 ;;
    *) stop "cell '$CELL' (b16, b32 or b64)" ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in joint_hbm.py joint_hbm_plan.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:benchmarks/Efficiency/$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
OUT=$OUTROOT/jh_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/jh_${RUN}_${JOB}_$CELL
NAME=jh-${CELL}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_VERIFY_ROUND_SLOTS=0 --env NOSI_MODEL_PATH="$MODEL" --env JH_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "joint-HBM child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "joint-HBM child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=240G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" jh-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/joint_hbm.py" --mode orchestrate --out "$OUT"
      --model "$MODEL" --workload "$WL" --batch "$B" --L 16128 --W 20 --K 3 --reps 10 --warm 2
      --ncu "$NCU" --variants none,all --child-timeout "$CHILD_S")
if [ "$DRY" = 1 ]; then
    echo "joint-HBM step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA"
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
mkdir "$OUTROOT/jh_active" 2>/dev/null || stop "$OUTROOT/jh_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (jh_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (jh_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (jh_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (jh_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s holder=%s left=%s run=%s cell=%s workload=%s utc_start=%s\n' "$NSHA" "$blob" \
    "$GPU_UUID" "$JOB" "$sl" "$RUN" "$CELL" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (jh_active stays)"
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
    rmdir "$OUTROOT/jh_active" || ok=0
fi
echo "joint-HBM step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
