#!/bin/bash
# R2 "r2base-fused": ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_base_cell.py with
# --r2 fused at one B (docs/R2_FUSED.md section 8): the r2base cell of cd77de2 (the shipped decode, re-warmed where it
# must be, against the CAPTURED R2 decode step teacher-forced on its tokens, same process; S 16128, N 512, the PG19
# 94-document workload) with the FUSED twin step and the LEAN body in the R2 arm; the predictions P1F, P2, P3F, P4F,
# P5. The cell gt64 also runs the fused twin's GPU gate first (r2_fused_cell.part_gt: GT-EQ, GT-COVER, GT-NEG,
# GT-NEG-ORDER as MUST gates), then B 64. The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from
# the lead-owned file HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or
# replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_base_fused_step.sh --nosi <sha> <holder job> <run id> <gt64|b64|b128|b219>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_base_fused_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# CELLS (as r2_base_step.sh; host rows S + N + 64 per request on both engines):
#   gt64  GT, then B 64; STEP_MIN 30, --mem 140G
#   b64   B  64, STEP_MIN 25, --mem 140G
#   b128  B 128, STEP_MIN 35, --mem 220G
#   b219  B 219, STEP_MIN 50, --mem 220G
# MBASE CELLS (r2-mbase: the matched baseline of the +17 % model, paper-notes/recon-rows-2-3.md row 2; r2_base_cell
# --pool P --mbase; C = 63 + P = NOSI_POOL_BLOCKS + 63; B = r2_base_cell.mbase_batch(C) except C 63; 512 steps; the
# SH steady step excludes every re-warm / re-capture). P > 0: the SH arm only (the R2 body has no victim pool).
#   mb1     C 128 (P 65) B 216, SH;                 STEP_MIN 35, --mem 240G   (2175791's reference cell)
#   mb2c113 C 113 (P 50) B 240, SH;                 STEP_MIN 35, --mem 240G
#   mb2c96  C  96 (P 33) B 280, SH;                 STEP_MIN 40, --mem 340G
#   mb2c79  C  79 (P 16) B 328, SH;                 STEP_MIN 45, --mem 340G
#   mb2c63  C  63 (P  0) B 344, SH + R2 fused;      STEP_MIN 70, --mem 340G   (2175791's C 63 maximum, not the GPU's)
#   fit<C>b<B>  a FIT PROBE (C in 63 79 96 113 128, B a multiple of 8 in 8 .. 400): SH only, 16 steps, no --mbase;
#           STEP_MIN 30, --mem 240G (B <= 248) or 340G. Exit 0 = B fits (prefill + 16 steps); SH-RUN FAIL = it did not.
# Host memory: the pinned host rows are rounded up to a power of two per (layer, K / V) block (r2_base_cell.free):
# 2 GiB blocks (128 GiB) up to B 251, 4 GiB (256 GiB) above; hence 240G / 340G.
# OUTPUT: OUTROOT/basef_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/basef_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): as r2_base_step.sh. The claim: OUTROOT/basef_active (mkdir,
# atomic); a stop after the claim leaves it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-mbase                                                    # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/basef                                                 # ROOT
SELF=benchmarks/Efficiency/r2_base_fused_step.sh
SPLITS=4
main() {
stop() { echo "R2 base-fused step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_base_fused_step.sh --nosi <sha> <holder> <run id> <gt64|b64|b128|b219|mb1|mb2c<C>|fit<C>b<B>>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
GT=()
POOL=0 STEPS=512 ARMS=SH,R2 MB=()
case $CELL in
    gt64) BATCH=64 STEP_MIN=30 MEM=140G GT=(--gt) ;;
    b64) BATCH=64 STEP_MIN=25 MEM=140G ;;
    b128) BATCH=128 STEP_MIN=35 MEM=220G ;;
    b219) BATCH=219 STEP_MIN=50 MEM=220G ;;
    mb1) POOL=65 BATCH=216 ARMS=SH STEP_MIN=35 MEM=240G MB=(--mbase) ;;
    mb2c113) POOL=50 BATCH=240 ARMS=SH STEP_MIN=35 MEM=240G MB=(--mbase) ;;
    mb2c96) POOL=33 BATCH=280 ARMS=SH STEP_MIN=40 MEM=340G MB=(--mbase) ;;
    mb2c79) POOL=16 BATCH=328 ARMS=SH STEP_MIN=45 MEM=340G MB=(--mbase) ;;
    mb2c63) POOL=0 BATCH=344 STEP_MIN=70 MEM=340G MB=(--mbase) ;;
    fit*b*)
        fc=${CELL#fit} fc=${fc%%b*} fb=${CELL##*b}
        case $fc in 63|79|96|113|128) ;; *) stop "cell '$CELL': C '$fc' (63, 79, 96, 113 or 128)" ;; esac
        [[ $fb =~ ^[1-9][0-9]{0,2}$ ]] && [ $((fb % 8)) = 0 ] && [ "$fb" -le 400 ] ||
            stop "cell '$CELL': B '$fb' (a multiple of 8 in 8 .. 400)"
        [ "$CELL" = "fit${fc}b$fb" ] || stop "cell '$CELL' (fit<C>b<B>)"
        POOL=$((fc - 63)) BATCH=$fb ARMS=SH STEPS=16 STEP_MIN=30 MEM=240G
        [ "$fb" -le 248 ] || MEM=340G ;;
    *) stop "cell '$CELL' (gt64, b64, b128, b219, mb1, mb2c113, mb2c96, mb2c79, mb2c63 or fit<C>b<B>)" ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_base_cell.py benchmarks/Efficiency/r2_fused_cell.py \
         benchmarks/Efficiency/r2_p3_profile.py benchmarks/Efficiency/r2_ragged_gate.py \
         benchmarks/Efficiency/r2_g0_compare.py nosi/nosi/r2_capture.py nosi/nosi/r2_decode_path.py \
         nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py nosi/nosi/r2_writeback.py \
         nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py nosi/nosi/r2_bi_decode.py \
         nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py nosi/nosi/cache_engine.py \
         nosi/nosi/flash_cache_engine/pool_update.cpp nosi/nosi/flash_cache_engine/pool_update_kernel.cu \
         nosi/nosi/flash_cache_engine/flash_pool_swap.py; do
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
OUT=$OUTROOT/basef_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/basef_${RUN}_${JOB}_$CELL
NAME=basef-${CELL}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 base-fused child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 base-fused child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" basef-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_base_cell.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --warmup 2 --r2 fused --arms "$ARMS" --pool "$POOL" "${GT[@]}"
      "${MB[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 base-fused step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 base-fused step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH arms $ARMS steps $STEPS STEP_MIN $STEP_MIN mem $MEM"
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
mkdir "$OUTROOT/basef_active" 2>/dev/null || stop "$OUTROOT/basef_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (basef_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (basef_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (basef_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (basef_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s pool=%s batch=%s arms=%s steps=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" "$STEPS" "$WL_SHA" \
    "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (basef_active stays)"
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
    rmdir "$OUTROOT/basef_active" || ok=0
fi
echo "R2 base-fused step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
