#!/bin/bash
# Task 1.4 route 1 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE bounded
# GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_verify_cell.py at the cached-NOSA cell (C 128 =
# pool 65, B 216, S 16128, N 512, the PG19 94-document workload). A COPY of r2_base_ragged_step.sh (0bdc045): the same
# pins, checks, claim, record and freeze; only the ROOT, the output root, the claim, the names, the cell and the
# python script change. ONE JOB, FOUR ARMS, each with its own shipped prefill:
#   SH   the shipped decode = cell mb1 (the same-job baseline: r2_base_cell.run_shipped, plus the logits fingerprint,
#        two reductions per step, declared, as rg1)
#   VRX  the joint pass (nosi/nosi/verify_rows.py), EXACT mode, gemm rowgroup (R shipped-shape GEMM calls per weight)
#   VRC  EXACT mode, gemm natural (one GEMM call per weight over the B x R rows)
#   VRF  FAST mode, gemm natural (one residency + the extras store + two masked flash_attn_nosa calls, LSE merge)
# Each verify arm is teacher-forced on SH's tokens: K = 3, R = 4 rows per request per pass, 128 passes.
# The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_verify_step.sh --nosi <sha> <holder job> <run id> vr1
#   DRYRUN=1 bash benchmarks/Efficiency/r2_verify_step.sh --nosi <sha> <holder job> <run id> vr1
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# CELL vr1 (C 128 = NOSI_POOL_BLOCKS 65, NOSI_ATTN_SPLITS 4, B 216, 512 steps): --verify-k 3 --verify-arms
#   VRX,VRC,VRF --verify-extra 16 --verify-twin torch --verify-body fused --probe-every 16; STEP_MIN 100, --mem 240G.
# Basis of STEP_MIN: mb1 took 00:14:31 (prefill 657.7 s, decode 92.1 s); rg1 has three prefills in 75 min; vr1 has
# four prefills (about 44 min) plus 3 x 128 joint passes (about 0.2-0.4 s each, about 2.5 min) and the model load.
#
# REGISTERED (TBD BY THE LEAD: fill r2_verify_cell.PRED_VERIFY and its stamp from `date -u` in a commit BEFORE the
# run; until then every prediction is REPORT):
#   VRX-BIT    every VRX logits fingerprint equal to SH's (bitwise): TBD
#   VRC-AGREE  VRC teacher-forced argmax agreement with SH >= TBD
#   VRF-AGREE  VRF teacher-forced argmax agreement with SH >= TBD
#   VRX-STEP, VRC-STEP, VRF-STEP   steady joint pass / SH steady step inside (TBD, TBD)
#   VR-SEL     VRF selection of R rows / one-row selection (ms per pass) inside (TBD, TBD)
#   VR-UNION   VRF union of the R rows' selections per (head, request), mean, inside (TBD, TBD)
#   MUST gates (not predictions): SH-RUN; <arm>-RUN, <arm>-ERR, <arm>-SEL0 per arm; VRF-IDX.
#
# OUTPUT: OUTROOT/vr_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/vr_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): as r2_base_step.sh. The claim: OUTROOT/vr_active (mkdir,
# atomic); a stop after the claim leaves it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-verify1                                                  # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/verify1                                               # ROOT
SELF=benchmarks/Efficiency/r2_verify_step.sh
SPLITS=4
main() {
stop() { echo "R2 verify step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_verify_step.sh --nosi <sha> <holder> <run id> <vr1>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
POOL=65 BATCH=216 STEPS=512
case $CELL in
    vr1) K=3 ARMS=VRX,VRC,VRF EXTRA=16 TWIN=torch STEP_MIN=100 MEM=240G ;;
    *) stop "cell '$CELL' (vr1)" ;;
esac
VER=(--verify-k "$K" --verify-arms "$ARMS" --verify-extra "$EXTRA" --verify-twin "$TWIN" --verify-body fused
     --probe-every 16)
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_verify_cell.py benchmarks/Efficiency/r2_base_cell.py \
         benchmarks/Efficiency/r2_ragged_gate.py nosi/nosi/verify_rows.py nosi/nosi/ragged_engine.py \
         nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py \
         nosi/nosi/r2_writeback.py nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py nosi/nosi/r2_bi_decode.py \
         nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py \
         nosi/nosi/cache_engine.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
OUT=$OUTROOT/vr_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/vr_${RUN}_${JOB}_$CELL
NAME=vr-${CELL}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 verify child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 verify child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" vr-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_verify_cell.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --pool "$POOL" "${VER[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 verify step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 verify step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH arms SH+$ARMS K $K steps $STEPS extra $EXTRA twin $TWIN STEP_MIN $STEP_MIN mem $MEM"
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
    *:*) left=$(( 10#${sl%%:*} * 60 )) ;;                       # squeue prints MM:SS below one hour
esac
[ "$left" -ge "$MIN_LEFT" ] || stop "holder $JOB has $sl left (< $MIN_LEFT s)"
nosteps() {
    steps=$(timeout 60 squeue -s -j "$JOB" -h -o '%i') || return 1
    for s in $steps; do case $s in *.batch|*.extern) ;; *) echo "$s"; return 1 ;; esac; done
}
busy=$(nosteps) || stop "a step is running in $JOB (or squeue -s failed): '$busy'"
mkdir -p "$OUTROOT" || stop "cannot create $OUTROOT"
mkdir "$OUTROOT/vr_active" 2>/dev/null || stop "$OUTROOT/vr_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (vr_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (vr_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (vr_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (vr_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s pool=%s batch=%s arms=%s k=%s steps=%s extra=%s twin=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" "$K" "$STEPS" "$EXTRA" \
    "$TWIN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (vr_active stays)"
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
    rmdir "$OUTROOT/vr_active" || ok=0
fi
echo "R2 verify step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
