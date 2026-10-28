#!/bin/bash
# Phase 2, rung L5 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE
# bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_spec5_cell.py (the background prefetch of
# the draft misses, nosi/nosi/spec_prefetch.py, opt-in --spec-prefetch on). A COPY of r2_spec_step.sh (a6c9d10): the
# same pins, checks, record and freeze; the ROOT (this tree, nosi-r2-spec5), the cells, the arms and the cell file
# change. The claim is the SAME as r2_spec_step.sh's (OUTROOT/sp_active_<holder>): a step of either script blocks the
# other on one holder (intended). ONE JOB, FIVE ARMS, SH FIRST, each with its own shipped prefill:
#   SH    the shipped decode at the cell's C and B: the same-job baseline
#   L3    K = 3 resident-only drafts (undone), 3 serial verify steps, accept / commit (rung L3)
#   L3P   L3 + the background prefetch of the drafts' skipped blocks (C++ packer, copy engine, the shipped scatter into
#         soft-pinned pool slots, committed outside every rollback group before the verify group)
#   L4P   the paper's schedule (rung L4P), the requested-KV pin off
#   L4PP  L4P + the prefetch + the requested-KV pin (paper 3.3)
# The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_spec5_step.sh --nosi <sha> <holder job> <run id> <sp5m|sp5a|sp5b>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_spec5_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# CELLS (--spec 3 --spec-arms L3,L3P,L4P,L4PP --spec-prefetch on --spec-twin fused --spec-body fused --spec-graph on
# --spec-redirect on --spec-pf-threads 8 --probe-every 16):
#   sp5m  the MINIATURE: C 73 (pool 10), B 8, 64 steps, journal 8 slots per layer and step, 8 staging lanes,
#         --require-stress 1: the MUST gate <arm>-STRESS needs journal overflow and overflow_dead > 0 and a prefetch
#         overflow path (lane refusal, commit overflow or pin overflow), so the miniature reaches the stressed regime of
#         sp5b. Run it first. STEP_MIN 40, --mem 100G.
#   sp5a  C 128 (pool 65), B 216 (cell mb1), journal 256, 128 lanes (268 MB staging: L4P peaked at 83.19 GB of 85.1 in
#         sp3a). STEP_MIN 110, --mem 240G. STOPS while r2_spec5_cell.PRED_STAMP5 is None at <sha>.
#   sp5b  C 73 (pool 10), B 318, journal 256, 1024 lanes (2.15 GB staging: L4P peaked at 76.82 GB in sp3b; the drafts
#         skip about 2.2 blocks per stream per call there). STEP_MIN 160, --mem 340G. STOPS while PRED_STAMP5 is None.
# Basis of STEP_MIN: sp3b (SH + 4 arms at B 318) took 01:31:30, sp3a (B 216) 01:02:24, sp3m 00:07:06; plus the packer
# build (about 1-2 min, PF-BUILD) and the prefetch arms' host work (MODEL: up to +20 % per decode).
#
# REGISTERED: r2_spec5_cell.PRED_SPEC5 and PRED_STAMP5, TBD BY THE LEAD (`date -u`, in a commit BEFORE sp5a / sp5b).
# MUST gates (not predictions): SH-RUN, PF-BUILD, <arm>-RUN, -ERR, -COMMIT, -TOK, -FP for every arm (all EXACT);
# <arm>-PF-ACT, -PF-CONS, -PF-PIN, -PF-NEW for L3P / L4PP; sp5m also <arm>-STRESS.
#
# OUTPUT: OUTROOT/sp_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/sp_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-spec5                                                    # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/spec                                                  # ROOT
SELF=benchmarks/Efficiency/r2_spec5_step.sh
CELLPY=benchmarks/Efficiency/r2_spec5_cell.py
SPLITS=4
main() {
stop() { echo "R2 spec5 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_spec5_step.sh --nosi <sha> <holder> <run id> <sp5m|sp5a|sp5b>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
K=3 ARMS=L3,L3P,L4P,L4PP STRESS=0 NEEDPRED=1
case $CELL in
    sp5m) POOL=10 BATCH=8 STEPS=64 JOURNAL=8 LANES=8 STRESS=1 NEEDPRED=0 STEP_MIN=40 MEM=100G ;;
    sp5a) POOL=65 BATCH=216 STEPS=512 JOURNAL=256 LANES=128 STEP_MIN=110 MEM=240G ;;
    sp5b) POOL=10 BATCH=318 STEPS=512 JOURNAL=256 LANES=1024 STEP_MIN=160 MEM=340G ;;
    *) stop "cell '$CELL' (sp5m, sp5a or sp5b)" ;;
esac
DESC="arms SH+$ARMS steps $STEPS K $K journal $JOURNAL lanes $LANES stress $STRESS"
SPK=(--spec "$K" --spec-arms "$ARMS" --spec-journal-blocks "$JOURNAL" --spec-twin fused --spec-body fused
     --spec-graph on --spec-redirect on --spec-prefetch on --spec-pf-lanes "$LANES" --spec-pf-threads 8
     --spec-pf-cap-groups 256 --probe-every 16 --require-stress "$STRESS")
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in "$CELLPY" benchmarks/Efficiency/r2_spec_cell.py benchmarks/Efficiency/r2_base_cell.py \
         benchmarks/Efficiency/r2_ragged_gate.py benchmarks/Efficiency/gilfree_pack.py \
         benchmarks/Efficiency/gilfree_pack.cpp nosi/nosi/spec_prefetch.py nosi/nosi/pool_prefetch.py \
         nosi/nosi/r2_transport_gilfree_v3.py nosi/nosi/kstep/backend.py nosi/nosi/kstep/transfer.py \
         nosi/nosi/spec_round.py nosi/nosi/ragged_engine.py \
         nosi/nosi/ragged_rollback.py nosi/nosi/verify_rows.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py \
         nosi/nosi/r2_positional_twin.py nosi/nosi/r2_writeback.py nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py \
         nosi/nosi/max_pooling_per_request.py nosi/nosi/cache_engine.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py \
         nosi/nosi/flash_cache_engine/flash_h2d_mask.py nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
if [ "$NEEDPRED" = 1 ] && git -C "$NOSI" show "$NSHA:$CELLPY" | grep -q '^PRED_STAMP5 = None$'; then
    stop "r2_spec5_cell.PRED_STAMP5 is None at $NSHA: register PRED_SPEC5 and its stamp (date -u) before $CELL"
fi
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
OUT=$OUTROOT/sp_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/sp_${RUN}_${JOB}_$CELL
NAME=sp-${CELL}-${RUN}-$JOB
CLAIM=$OUTROOT/sp_active_$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 spec5 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 spec5 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" sp5-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/$CELLPY" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --pool "$POOL" "${SPK[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 spec5 step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 spec5 step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH $DESC STEP_MIN $STEP_MIN mem $MEM claim $CLAIM"
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
mkdir "$CLAIM" 2>/dev/null || stop "$CLAIM exists (another step of this holder, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id ($CLAIM stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT ($CLAIM stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIM stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIM stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s kind=spec5 pool=%s batch=%s arms=%s steps=%s k=%s journal=%s lanes=%s stress=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" "$STEPS" "$K" "$JOURNAL" \
    "$LANES" "$STRESS" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIM stays)"
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
    rmdir "$CLAIM" || ok=0
fi
echo "R2 spec5 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
