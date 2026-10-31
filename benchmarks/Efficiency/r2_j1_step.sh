#!/bin/bash
# Rung J1 (AREA 1 of the RECON7 follow-up; KS/analysis/plan/DESIGN_JOINT_EFF_2026-10-31.md fix F1; KS =
# /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/
# r2_j1_cell.py (arms L4PTFDGUDCDJ1 / L4PTFDGUDKWCDJ1 = the I4DC best arms with the joint pooling compiled; what changes:
# benchmarks/Efficiency/r2_j1_select.py). A COPY of r2_recon7_step.sh (cd0f0c5): the same pins, the same HASHING
# BOUNDARY (this wrapper hashes nothing; the image and workload sha256 are checked on the GPU node by
# /usr/bin/sha256sum), the same output limit, record and freeze; the ROOT (this tree, nosi-r2-j1), the cells, the cell
# file, the files list, the grants (in r2_j1_cell.py) and the registration stop change. The claim is the SAME as the
# other spec step scripts' (OUTROOT/sp_active_<holder>).
# THE CELL GRANTS (default deny): a cell runs only when the line `GRANT_<cell> = "<date -u stamp>"` of
# benchmarks/Efficiency/r2_j1_cell.py at <sha> matches granted() below. The timing cells j1a / j1b also STOP while
# r2_j1_cell.PRED_STAMP_J1 is None at <sha> (the lead registers PRED_J1_BY_B first).
# THE COMPILE CACHES (lesson C1): TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR and TRITON_HOME in a NEW directory per step
# under CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/j1_jit (cold). It never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_j1_step.sh --nosi <sha> <holder job> <run id> <j1m|j1a|j1b>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_j1_step.sh --nosi <sha> <holder job> <run id> <cell>
#
# CELLS (--spec 3 --fork-store 0 --prefill-hook nocopy --tu-gemm natural --spec-twin fused --spec-body fused --spec-graph
# on --spec-redirect on --probe-every 16; docs/NOTES_j1.md holds the protocol):
#   j1m  the MINIATURE: C 73 (pool 10), B 8, 72 steps, arms L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,
#        L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1,L4PTFDGUDCDJ3, --check-every 1 (J1-PEQ in every round),
#        --require-stress 1; every gate MUST. Run it first. STEP_MIN 35, --mem 100G.
#   j1a  timing, C 128 (pool 65), B 216, 512 steps, arms L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1,L4PTFDGUDCDJ3.
#        STEP_MIN 125, --mem 240G. STOPS while PRED_STAMP_J1 is None.
#   j1f  the FORK MICROBENCH (benchmarks/Efficiency/r2_j1_forkbench.py; grant GRANT_j1f in THAT file): no model, no
#        workload; synthetic pages of the measured shapes (B 216, B 318; union 72); variants FORK, FORKT, FORK3D2/4/8,
#        FORK1R, NS1R1, NS1x4, FLASHU1, NSRU; 32 calls per pass, 10 order-rotated passes. STEP_MIN 25 (the image hash of
#        23 GB inside it), --mem 64G, OUT_MB 60.
#   j1b  timing, C 73 (pool 10), B 318, 512 steps, arms L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1. STEP_MIN
#        170, --mem 340G. STOPS while PRED_STAMP_J1 is None.
# Every cell also runs SHJ1 right after SH (the shipped run with the compiled pooling: the matched baseline of F2;
# about one more SH run, 1-4 min; inside the margins below).
# Basis of STEP_MIN (MEASURED, sacct, CITED by r2_recon7_step.sh:42-46): i4dca (SH + 3 arms, C 128 B 216) 00:47:32;
# i4dcb 01:11:41; i4dm2 (SH + 6 arms, B 8) 00:09:08 / 00:09:18. MODEL: j1a = i4dca + one D-arm part (11.5 min) + one
# compile (<= 1 min) = about 60 min, + the J3 part (11.5) + SHJ1 (2) = about 74 min (margin 1.7); j1b = i4dcb + 17 +
# 1 + SHJ1 (4) = about 94 min (margin 1.8); j1m = i4dm2 + 3 parts (3 min) = about 12.5 min (margin 2.8). Memory: the
# I4DC cells' --mem.
#
# OUTPUT: OUTROOT/j1_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/j1_<run>_<holder>_<cell>.step/, both
# frozen. EXIT: as r2_recon7_step.sh (the cell's exit 0-19 / 20 / 30; 1 STOP; 2 usage; 3 another GPU; 4 not one GPU;
# 5 incomplete record; 6 a sha256 mismatch on the GPU node; 7 above OUT_MB; 124 / 137 the host timeout).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-j1                                                       # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
SIF_SHA=14bfb15e1258821e637b5936f2060d7c7c2be20410fdee2a54ee0ffef754734a                  # PIN (nosa-env.sif.sha256)
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/j1_jit                                    # ROOT (the compile caches)
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/spec                                                  # ROOT
SELF=benchmarks/Efficiency/r2_j1_step.sh
CELLPY=benchmarks/Efficiency/r2_j1_cell.py
FORKPY=benchmarks/Efficiency/fork_vllm_unified_attention.py
FORK_BLOB=f0dc0c183a388585e216505a6bbfc696531e8add                                       # PIN (fia6 94d5bb0's fork)
SPLITS=4
OUT_MB=2048
# granted <cell> <cell file>: true only when the file holds the cell's grant line with a `date -u` stamp.
granted() {
    grep -Eq "^GRANT_$1 = \"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\"\$" "$2"
}
main() {
stop() { echo "R2 j1 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_j1_step.sh --nosi <sha> <holder> <run id> <j1m|j1a|j1b>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
K=3 STRESS=0 CHECK=0 NEEDPRED=0 NEEDWL=1
case $CELL in
    j1f) POOL=65 BATCH=0 STEPS=0 STEP_MIN=25 MEM=64G NEEDWL=0 OUT_MB=60 ARMS=-
         CELLPY=benchmarks/Efficiency/r2_j1_forkbench.py ;;
    j1m) POOL=10 BATCH=8 STEPS=72 CHECK=1 STRESS=1 STEP_MIN=35 MEM=100G
         ARMS=L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDJ1,L4PTFDGUDKWCDJ1,L4PTFDGUDCDJ3 ;;
    j1a) POOL=65 BATCH=216 STEPS=512 NEEDPRED=1 STEP_MIN=125 MEM=240G
         ARMS=L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDJ1,L4PTFDGUDCDJ3 ;;
    j1b) POOL=10 BATCH=318 STEPS=512 NEEDPRED=1 STEP_MIN=170 MEM=340G
         ARMS=L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDJ1 ;;
    *) stop "cell '$CELL' (j1m, j1a, j1b or j1f)" ;;
esac
DESC="arms SH+$ARMS steps $STEPS K $K fork-store 0 check $CHECK stress $STRESS"
CARGS=(--batch "$BATCH" --steps "$STEPS" --pool "$POOL" --spec "$K" --arms "$ARMS" --fork-store 0 --prefill-hook nocopy
       --tu-gemm natural --spec-journal-blocks 256 --spec-twin fused --spec-body fused --spec-graph on --spec-redirect on
       --probe-every 16 --check-every "$CHECK" --require-stress "$STRESS")
if [ "$CELL" = j1f ]; then                    # the fork microbench: no model, no workload (synthetic inputs)
    DESC="fork microbench: shapes C 128 B 216, C 73 B 318; union 72; 3D segments 2,4,8; 32 calls per pass"
    CARGS=(--cell j1f --shapes 128:216,73:318 --segments 2,4,8 --layers 32 --reps 10 --warmup 2 --seed 7 --sample 8
           --max-shape-s 300 --max-run-s 900)
fi
FILE_KB=$((OUT_MB * 1024))
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
cmp -s -- "$0" <(git -C "$NOSI" show "$NSHA:$SELF") || stop "$0 is not the bytes of $SELF at $NSHA (blob $blob)"
for f in "$CELLPY" "$FORKPY" benchmarks/Efficiency/r2_j1_select.py benchmarks/Efficiency/r2_i4dc_cell.py \
         benchmarks/Efficiency/r2_i4dc_compile.py benchmarks/Efficiency/r2_i4cc_cell.py \
         benchmarks/Efficiency/r2_i4cc_compile.py benchmarks/Efficiency/r2_i4s_cell.py \
         benchmarks/Efficiency/r2_i4s_stack.py benchmarks/Efficiency/r2_i4h_cell.py \
         benchmarks/Efficiency/r2_i4h_graph.py benchmarks/Efficiency/r2_i4k_cell.py \
         benchmarks/Efficiency/r2_i4k_combo.py benchmarks/Efficiency/r2_i4d_cell.py benchmarks/Efficiency/r2_i4d_keep.py \
         benchmarks/Efficiency/r2_i4u_cell.py benchmarks/Efficiency/r2_i4u_hoist.py benchmarks/Efficiency/r2_i4g_cell.py \
         benchmarks/Efficiency/r2_i4g_graph.py benchmarks/Efficiency/r2_i4a_cell.py \
         benchmarks/Efficiency/r2_i4a_lean.py benchmarks/Efficiency/r2_i3_cell.py \
         benchmarks/Efficiency/r2_i3_arena.py benchmarks/Efficiency/r2_kv_arena.py \
         benchmarks/Efficiency/r2_i2_cell.py benchmarks/Efficiency/r2_wsa_cell.py benchmarks/Efficiency/r2_wsb_cell.py \
         benchmarks/Efficiency/r2_wsg_check.py benchmarks/Efficiency/r2_i2_check.py \
         benchmarks/Efficiency/r2_spec_cell.py benchmarks/Efficiency/r2_base_cell.py \
         benchmarks/Efficiency/r2_ragged_gate.py nosi/nosi/spec_tentative_fork.py nosi/nosi/spec_tentative_union.py \
         nosi/nosi/spec_tentative.py nosi/nosi/spec_round.py nosi/nosi/ragged_engine.py \
         nosi/nosi/ragged_rollback.py nosi/nosi/verify_rows.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py \
         nosi/nosi/r2_positional_twin.py nosi/nosi/r2_writeback.py nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py \
         nosi/nosi/max_pooling_per_request.py nosi/nosi/cache_engine.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py \
         nosi/nosi/flash_cache_engine/flash_h2d_mask.py nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ "$(git -C "$NOSI" rev-parse "$NSHA:$FORKPY")" = "$FORK_BLOB" ] || stop "$FORKPY at $NSHA is not the fia6 blob $FORK_BLOB"
APPROVED=0
granted "$CELL" <(git -C "$NOSI" show "$NSHA:$CELLPY") && APPROVED=1
REGISTERED=1
git -C "$NOSI" show "$NSHA:$CELLPY" | grep -q '^PRED_STAMP_J1 = None$' && REGISTERED=0
[ -f "$SIF" ] || stop "no image $SIF (its sha256 is checked on the GPU node)"
WLF=- WLS=-
WARGS=()
if [ "$NEEDWL" = 1 ]; then                    # the model and the workload: only the cells that load them
    [ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
    [ -f "$WL/c16k_tokens.bin" ] || stop "no $WL/c16k_tokens.bin (its sha256 is checked on the GPU node)"
    WLF=$WL/c16k_tokens.bin WLS=$WL_SHA
    WARGS=(--model "$MODEL" --workload "$WL")
fi
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
OUT=$OUTROOT/j1_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/j1_${RUN}_${JOB}_$CELL
CCJIT=$CCROOT/j1_${RUN}_${JOB}_$CELL
NAME=j1-${CELL}-${RUN}-$JOB
CLAIM=$OUTROOT/sp_active_$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$CCJIT/triton"
      --env TRITON_HOME="$CCJIT/triton_home" --env TORCHINDUCTOR_CACHE_DIR="$CCJIT/inductor"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
# THE CHILD (on the GPU node, inside the step): one GPU, the uuid, the image and workload sha256 (absolute
# /usr/bin/sha256sum), the per-file output limit, then the image.
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 j1 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1 IMG=$2 IMGSHA=$3 WLF=$4 WLSHA=$5 KB=$6
shift 6
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 j1 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
echo "R2 j1 child: host $(hostname) gpu $seen; hashing the image at $(date -u +%FT%TZ)" >&2
h=$(/usr/bin/sha256sum -- "$IMG") || { echo "R2 j1 child: cannot hash $IMG" >&2; exit 6; }
[ "${h%% *}" = "$IMGSHA" ] || { echo "R2 j1 child: image sha256 ${h%% *} is not $IMGSHA" >&2; exit 6; }
echo "R2 j1 child: image sha256 $IMGSHA OK at $(date -u +%FT%TZ)" >&2
if [ "$WLF" != - ]; then
    h=$(/usr/bin/sha256sum -- "$WLF") || { echo "R2 j1 child: cannot hash $WLF" >&2; exit 6; }
    [ "${h%% *}" = "$WLSHA" ] || { echo "R2 j1 child: workload sha256 ${h%% *} is not $WLSHA" >&2; exit 6; }
    echo "R2 j1 child: workload sha256 $WLSHA OK" >&2
fi
ulimit -f "$KB" || { echo "R2 j1 child: ulimit -f $KB failed" >&2; exit 6; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/bash -c "$CHILD" j1-child "$GPU_UUID" "$SIF" "$SIF_SHA" "$WLF" "$WLS" "$FILE_KB" "$APPTAINER"
      --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/$CELLPY" --out "$OUT" "${WARGS[@]}" "${CARGS[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 j1 step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob fork $FORK_BLOB image sha256 $SIF_SHA (checked on the GPU node) workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE); cell granted $APPROVED; registered $REGISTERED"
    echo "R2 j1 step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH $DESC STEP_MIN $STEP_MIN mem $MEM out_mb $OUT_MB claim $CLAIM compile caches $CCJIT"
    printf '%q ' "${ARGV[@]}"
    echo
fi
[ "$APPROVED" = 1 ] || stop "cell $CELL is not granted at $NSHA ($CELLPY: GRANT_$CELL; default deny)"
[ "$NEEDPRED" = 0 ] || [ "$REGISTERED" = 1 ] ||
    stop "r2_j1_cell.PRED_STAMP_J1 is None at $NSHA: register PRED_J1_BY_B and its stamp (date -u) before $CELL"
[ "$DRY" = 1 ] && exit 0
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
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] && [ ! -e "$CCJIT" ] ||
    stop "$OUT, $REC, $JIT or $CCJIT exists: a new run id ($CLAIM stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" "$CCJIT/inductor" "$CCJIT/triton" "$CCJIT/triton_home" ||
    stop "cannot create $REC / $JIT / $CCJIT ($CLAIM stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIM stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIM stays)"
printf '\nnosi=%s wrapper_blob=%s fork_blob=%s image_sha256=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s kind=j1 pool=%s batch=%s arms=%s steps=%s k=%s check=%s stress=%s workload=%s out_mb=%s compile_caches=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$FORK_BLOB" "$SIF_SHA" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" \
    "$STEPS" "$K" "$CHECK" "$STRESS" "$WL_SHA" "$OUT_MB" "$CCJIT" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIM stays)"
( ulimit -f "$FILE_KB" && exec "${ARGV[@]}" ) < /dev/null > "$REC/stdout.txt" 2> "$REC/stderr.txt"
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
total=0
for t in "$OUT" "$REC" "$JIT" "$CCJIT"; do
    [ -e "$t" ] || continue
    b=$(du -sb -- "$t" | cut -f1) || { ok=0; continue; }
    echo "bytes $t $b" >> "$REC/exit.txt" || ok=0
    total=$((total + b))
done
over=0
[ "$total" -le $((OUT_MB * 1024 * 1024)) ] || over=1
echo "output_bytes=$total limit_bytes=$((OUT_MB * 1024 * 1024)) over=$over" >> "$REC/exit.txt" || ok=0
[ "$rc" != 0 ] || [ "$acct" = consistent ] || ok=0
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rmdir "$CLAIM" || ok=0
fi
echo "R2 j1 step: exit $rc; record $REC; output $OUT; output bytes $total (limit $OUT_MB MB, over $over); record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
[ "$rc" = 0 ] && [ "$over" = 1 ] && exit 7
exit "$rc"
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi
