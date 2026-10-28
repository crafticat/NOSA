#!/bin/bash
# Rung I2 of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE bounded GPU step in an
# EXISTING holder that runs benchmarks/Efficiency/r2_i2_cell.py (arm L4PTU = WS-A's tentative buffers + WS-B's union
# residency, nosi/nosi/spec_tentative_union.py, opt-in --joint-residency union). A COPY of r2_wsa_step.sh (83dac96): the
# same pins, checks, record and freeze; the ROOT (this tree, nosi-r2-i2), the cells, the arms and the cell file change.
# The claim is the SAME as the spec / WS-A step scripts' (OUTROOT/sp_active_<holder>): a step of any of them blocks the
# others on one holder (intended). SH FIRST, each arm with its own shipped prefill, on the inputs of sp3a / sp3b and
# wa1a / wa1b (the same workload, prefill, rows):
#   SH     the shipped decode at the cell's C and B: the same-job baseline
#   L4PT   I1 (WS-A), r2_wsa_cell.run_arm unchanged: the previous rung, re-measured in the same job
#   L4PTU  I2 (+ WS-B): one residency per layer for the union of the joint pass's rows; gemm natural
#   SHR    (check cells) the serial ragged step teacher-forced on SH's tokens (the reference selections)
#   TUC    (check cells) L4PTU's joint pass teacher-forced on SH's tokens, check on, against SHR
# The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <holder job> <run id> <i2m|i2a|i2b|i2ca|i2cb>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_i2_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# CELLS (--spec 3 --joint-residency union --tu-gemm natural --spec-twin fused --spec-body fused --spec-graph on
# --spec-redirect on --probe-every 16):
#   i2m   the MINIATURE: C 73 (pool 10), B 8, 64 steps, arms L4PT,L4PTU,SHR,TUC, X 4 (C 64 store fills: the spill runs;
#         slow by design), 15 teacher-forced passes, --check-every 1 (every round a check round; L4PTU's union checks
#         and accounting on), --require-stress 1: the MUST gates L4PT-STRESS / L4PTU-STRESS (commits across a block edge,
#         rows past an edge, F4 fill rows, dropped entries, tentative-block skips, draft skips, corrections, refills,
#         final groups), L4PTU-USTRESS and TUC-STRESS (the spill served entries), L4PTU-PRESENT / -VIS / -IDX, so the
#         miniature reaches the stressed regime of the big cells. Run it first. STEP_MIN 45, --mem 100G.
#   i2a   timing, C 128 (pool 65), B 216 (sp3a's, wa1a's), arms L4PT,L4PTU, X 48. STEP_MIN 70, --mem 240G. STOPS while
#         r2_i2_cell.PRED_STAMP_I2 is None at <sha>.
#   i2b   timing, C 73 (pool 10), B 318 (sp3b's, wa1b's), arms L4PT,L4PTU, X 48. STEP_MIN 95, --mem 340G. STOPS while
#         PRED_STAMP_I2 is None.
#   i2ca  check, C 128 (pool 65), B 216, 129 steps, arms SHR,TUC, X 48, 32 teacher-forced passes. STEP_MIN 70, --mem 240G.
#   i2cb  check, C 73 (pool 10), B 318, 129 steps, arms SHR,TUC, X 48, 32 passes. STEP_MIN 95, --mem 340G.
#   i2g   WS-G's check (r2_wsg_check.py of b1ae579, copied; r2_i2_check.ContractCapture), wgcm's shape: C 73 (pool 10),
#         B 100, 64 steps, arms WJ (L4PJ, the EXACT positive control: WJ-CHK MUST), WT (L4PT), WU (L4PTU) (REPORT),
#         WUP (L4PTU with drop_spill at X 4: the planted control, WUP-PLANT-CHK-PRE / -CAUGHT MUST); each arm, then
#         REF (SH's computation teacher-forced on the arm's committed tokens); slot check every 4 rounds. STEP_MIN 90
#         (wgcm: SH + 2 x (arm + REF) took 26:47; here SH + 4 x (arm + REF)), --mem 200G.
# Basis of STEP_MIN: wa1a (SH + 3 arms at B 216) is registered at 75, wa1b (B 318) at 110, wb1m at 40 (its prefills
# 557-960 s are most of a cell); here SH + 2 arms (i2a / i2b), SH + 2 arms (i2ca / i2cb), SH + 4 arms at B 8 (i2m).
#
# REGISTERED: r2_i2_cell.PRED_I2 / PRED_I2_BY_B and PRED_STAMP_I2, TBD BY THE LEAD (`date -u`, in a commit BEFORE i2a /
# i2b). Inferred bands: docs/NOTES_i2.md. MUST gates (not predictions): SH-RUN, <arm>-RUN, -ERR, -COMMIT-N, -D1, -D3, -SPAN;
# L4PTU-RESID; i2m also <arm>-STRESS, L4PTU-USTRESS, -PRESENT, -VIS, -IDX; check cells SHR-FP / -TOK, TUC -SEL, -PRESENT,
# -VIS, -RESID, -IDX, -IDXT.
#
# OUTPUT: OUTROOT/i2_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/i2_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-i2                                                       # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/spec                                                  # ROOT
SELF=benchmarks/Efficiency/r2_i2_step.sh
CELLPY=benchmarks/Efficiency/r2_i2_cell.py
SPLITS=4
main() {
stop() { echo "R2 i2 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_i2_step.sh --nosi <sha> <holder> <run id> <i2m|i2a|i2b|i2ca|i2cb|i2g>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
K=3 X=48 PASSES=32 STRESS=0 CHECK=0 NEEDPRED=0 SLOT=32
case $CELL in
    i2g)  POOL=10 BATCH=100 STEPS=64 ARMS=WJ,WT,WU,WUP SLOT=4 STEP_MIN=90 MEM=200G ;;
    i2m)  POOL=10 BATCH=8 STEPS=64 ARMS=L4PT,L4PTU,SHR,TUC X=4 PASSES=15 STRESS=1 CHECK=1 STEP_MIN=45 MEM=100G ;;
    i2a)  POOL=65 BATCH=216 STEPS=512 ARMS=L4PT,L4PTU NEEDPRED=1 STEP_MIN=70 MEM=240G ;;
    i2b)  POOL=10 BATCH=318 STEPS=512 ARMS=L4PT,L4PTU NEEDPRED=1 STEP_MIN=95 MEM=340G ;;
    i2ca) POOL=65 BATCH=216 STEPS=129 ARMS=SHR,TUC STEP_MIN=70 MEM=240G ;;
    i2cb) POOL=10 BATCH=318 STEPS=129 ARMS=SHR,TUC STEP_MIN=95 MEM=340G ;;
    *) stop "cell '$CELL' (i2m, i2a, i2b, i2ca, i2cb or i2g)" ;;
esac
DESC="arms SH+$ARMS steps $STEPS K $K X $X passes $PASSES check $CHECK stress $STRESS slot $SLOT"
SPK=(--spec "$K" --arms "$ARMS" --joint-residency union --joint-extra "$X" --tu-gemm natural --check-passes "$PASSES"
     --spec-journal-blocks 256 --spec-twin fused --spec-body fused --spec-graph on --spec-redirect on --probe-every 16
     --check-every "$CHECK" --require-stress "$STRESS" --chk-slot-every "$SLOT" --chk-plant-extra 4)
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in "$CELLPY" benchmarks/Efficiency/r2_wsa_cell.py benchmarks/Efficiency/r2_wsb_cell.py \
         benchmarks/Efficiency/r2_wsg_check.py benchmarks/Efficiency/r2_i2_check.py \
         benchmarks/Efficiency/r2_spec_cell.py benchmarks/Efficiency/r2_base_cell.py \
         benchmarks/Efficiency/r2_ragged_gate.py nosi/nosi/spec_tentative_union.py nosi/nosi/spec_tentative.py \
         nosi/nosi/spec_round.py nosi/nosi/ragged_engine.py \
         nosi/nosi/ragged_rollback.py nosi/nosi/verify_rows.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py \
         nosi/nosi/r2_positional_twin.py nosi/nosi/r2_writeback.py nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py \
         nosi/nosi/max_pooling_per_request.py nosi/nosi/cache_engine.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py \
         nosi/nosi/flash_cache_engine/flash_h2d_mask.py nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
if [ "$NEEDPRED" = 1 ] && git -C "$NOSI" show "$NSHA:$CELLPY" | grep -q '^PRED_STAMP_I2 = None$'; then
    stop "r2_i2_cell.PRED_STAMP_I2 is None at $NSHA: register PRED_I2_BY_B and its stamp (date -u) before $CELL"
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
OUT=$OUTROOT/i2_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/i2_${RUN}_${JOB}_$CELL
NAME=i2-${CELL}-${RUN}-$JOB
CLAIM=$OUTROOT/sp_active_$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 i2 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 i2 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" i2-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/$CELLPY" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --pool "$POOL" "${SPK[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 i2 step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 i2 step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH $DESC STEP_MIN $STEP_MIN mem $MEM claim $CLAIM"
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
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s kind=i2 pool=%s batch=%s arms=%s steps=%s k=%s x=%s passes=%s check=%s stress=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" "$STEPS" "$K" "$X" \
    "$PASSES" "$CHECK" "$STRESS" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
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
echo "R2 i2 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
