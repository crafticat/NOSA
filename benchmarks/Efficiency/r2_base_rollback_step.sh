#!/bin/bash
# Task 1.5 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE bounded GPU
# step in an EXISTING holder that runs benchmarks/Efficiency/r2_base_cell.py with --rollback at the cached-NOSA cell
# (C 128 = pool 65, B 216, S 16128, N 512, the PG19 94-document workload). A COPY of r2_base_ragged_step.sh (0bdc045):
# the same pins, checks, claim, record and freeze; only the ROOT, the output root, the claim, the names and the cells
# change. ONE JOB, THREE ARMS, each with its own shipped prefill:
#   SH   the shipped decode = cell mb1 (the same r2_base_cell arguments; the same-job baseline)
#   RBR  the REFERENCE: nosi/nosi/ragged_engine.py, plain steps, teacher-forced on SH's tokens; group g has K = 3 steps,
#        and in step u only the requests with u < n_acc[g, b] are active (it runs ONLY the kept steps)
#   RBS  the ROLLBACK: the same engine with nosi/nosi/ragged_rollback.py; per group begin(), K = 3 tentative steps with
#        every request active, then commit(n_acc[g]) (CUDA events: the commit / rollback cost per group)
# n_acc[g, b] is uniform in 0 .. 3, seeded (--rb-seed 0), 128 groups. RB-BIT: every kept row (u < n_acc) of RBS has
# the logits fingerprint of RBR, bitwise, in every group.
# The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_base_rollback_step.sh --nosi <sha> <holder job> <run id> <rb1|rb1x>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_base_rollback_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# CELLS (C 128 = NOSI_POOL_BLOCKS 65, B 216, 512 steps, --arms SH --pool 65 --mbase --rollback --rb-k 3
# --rb-groups 128 --rb-seed 0 --rb-journal-blocks 256 --ragged-twin torch --ragged-body fused --ragged-graph on):
#   rb1   --ragged-gemm natural (the shipped GEMM: at one B the rows do not depend on each other; the check compares
#         runs at the same B); STEP_MIN 75, --mem 240G
#   rb1x  --ragged-gemm exact (r2_spec_core.PaddedLinear: the EXACT mode of the plan's proofs); STEP_MIN 75, --mem 240G
# Basis of STEP_MIN: mb1 (basef_mba_2190876_mb1) took 00:14:31, with prefill 657.7 s and the decode 92.1 s; three
# prefills, 512 + 384 + 384 steps and 128 commits are about 40 minutes.
# MEMORY (INFERRED): RBS adds the journal (about 10 MB per layer and step at 256 slots, x 32 x 3 = 1.0 GB) and the
# staging blocks (0.45 GB) to the engine of rg1; mb1's allocator peak was 82.04 GB. If RBS stops on memory, the lead
# lowers --rb-journal-blocks (a slot past it comes back from the host; RB-OVF counts them).
#
# REGISTERED (TBD BY THE LEAD: fill r2_base_cell.PRED_ROLLBACK and its stamp from `date -u` in a commit BEFORE the run;
# until then every RB prediction is REPORT):
#   RB-BIT     every kept row of RBS bitwise RBR's (max value, float64 sum and argmax of the logits): TBD
#   RB-STEP    RBS median tentative step / RBR median step inside (TBD, TBD)
#   RB-COMMIT  RBS median commit per group (ms) inside (TBD, TBD)
#   RB-OVF     no journal slot past --rb-journal-blocks: TBD
#   MUST gates (not predictions): SH-RUN, RBR-RUN, RBR-ERR, RBR-CAP, RBS-RUN, RBS-ERR, RBS-CAP.
#
# OUTPUT: OUTROOT/rb_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/rb_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): as r2_base_step.sh. The claim: OUTROOT/rb_active (mkdir,
# atomic); a stop after the claim leaves it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-rollback                                                 # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/rollback                                              # ROOT
SELF=benchmarks/Efficiency/r2_base_rollback_step.sh
SPLITS=4
main() {
stop() { echo "R2 base-rollback step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_base_rollback_step.sh --nosi <sha> <holder> <run id> <rb1|rb1x>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
POOL=65 BATCH=216 STEPS=512 ARMS=SH MB=(--mbase)
case $CELL in
    rb1) GEMM=natural STEP_MIN=75 MEM=240G ;;
    rb1x) GEMM=exact STEP_MIN=75 MEM=240G ;;
    *) stop "cell '$CELL' (rb1 or rb1x)" ;;
esac
RBK=(--rollback --rb-k 3 --rb-groups 128 --rb-seed 0 --rb-journal-blocks 256 --ragged-twin torch --ragged-body fused
     --ragged-gemm "$GEMM" --ragged-graph on)
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_base_cell.py benchmarks/Efficiency/r2_ragged_gate.py nosi/nosi/ragged_engine.py \
         nosi/nosi/ragged_rollback.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py \
         nosi/nosi/r2_positional_twin.py nosi/nosi/r2_writeback.py nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py \
         nosi/nosi/max_pooling_per_request.py nosi/nosi/cache_engine.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py \
         nosi/nosi/flash_cache_engine/flash_h2d_mask.py nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py; do
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
OUT=$OUTROOT/rb_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/rb_${RUN}_${JOB}_$CELL
NAME=rb-${CELL}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 base-rollback child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 base-rollback child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" rb-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_base_cell.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --warmup 2 --r2 fused --arms "$ARMS" --pool "$POOL"
      "${MB[@]}" "${RBK[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 base-rollback step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 base-rollback step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH arms $ARMS+RBR+RBS steps $STEPS gemm $GEMM K 3 groups 128 STEP_MIN $STEP_MIN mem $MEM"
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
mkdir "$OUTROOT/rb_active" 2>/dev/null || stop "$OUTROOT/rb_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (rb_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (rb_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (rb_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (rb_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s pool=%s batch=%s arms=%s steps=%s gemm=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" "$STEPS" "$GEMM" "$WL_SHA" \
    "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (rb_active stays)"
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
    rmdir "$OUTROOT/rb_active" || ok=0
fi
echo "R2 base-rollback step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
