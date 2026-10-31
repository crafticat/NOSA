#!/bin/bash
# Rung INT (KS/analysis/plan/INTEGRATION_NOTES_2026-10-31.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE bounded GPU
# step in an EXISTING holder that runs benchmarks/Efficiency/r2_int_cell.py (arms L4PTFDGUDCDI = L4PTFDGUDCD + F1 + F2 +
# F6 + F3 + ST1 + ST2, L4PTFDGUDKWCDI = L4PTFDGUDKWCD + F1 + F2 + F6 + ST1 + ST2 + BF1; the classes:
# benchmarks/Efficiency/r2_int_arms.py). A COPY of r2_j1_step.sh (the repaired pattern of r2_recon7_step.sh): the same
# pins, the same HASHING BOUNDARY (this wrapper hashes nothing; the image and workload sha256 are checked on the GPU node
# by /usr/bin/sha256sum), the same output limit, record and freeze; the ROOT (the run tree nosi-r2-int, lead-owned),
# the cells, the cell file, the files list, the grants (in r2_int_cell.py) and the registration stop change. The claim
# is the SAME as the other spec step scripts' (OUTROOT/sp_active_<holder>).
# THE CELL GRANTS (default deny): a cell runs only when the line `GRANT_<cell> = "<date -u stamp>"` of
# benchmarks/Efficiency/r2_int_cell.py at <sha> matches granted() below. The timing cells inta / intb also STOP while
# r2_int_cell.PRED_STAMP_INT is None at <sha> (the lead registers PRED_INT_BY_B first).
# THE COMPILE CACHES (lesson C1): TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR and TRITON_HOME in a NEW directory per step
# under CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/int_jit (cold); TORCH_EXTENSIONS_DIR (the BF1 gilfree packer's
# build, about 1-2 min at the first BF1 arm) in the step's JIT directory. It never submits, renews, cancels or replaces
# an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_int_step.sh --nosi <sha> <holder job> <run id> <intm|inta|intb>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_int_step.sh --nosi <sha> <holder job> <run id> <cell>
#
# CELLS (--spec 3 --fork-store 0 --prefill-hook nocopy --tu-gemm natural --spec-twin fused --spec-body fused --spec-graph
# on --spec-redirect on; --st1-commit rows; BF1 as r2_bf1_step.sh at 64e307f: --bf-ring / --bf-ncap per cell,
# --bf-worker thread --bf-close-wait-ms 0 --bf-packer gilfree --bf-threads 14 --bf-cap-groups 2048 --bf-batch 8
# --bf-late defer --bf-late-wait-ms 250;
# docs/NOTES_int.md holds the protocol):
#   intm the MINIATURE: C 73 (pool 10), B 8, 72 steps, probe 4, arms L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,
#        L4PTFDGUDKWCD,L4PTFDGUDCDM,L4PTFDGUDKWCDM,L4PTFDGUDCDMT,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK (the ST3 control),
#        L4PTFDGUDCDMTR,L4PTFDGUDKWCDMTKR (the ST4 controls: gates and correctness only),L4PTFDGUDCDI (+ ST4),
#        L4PTFDGUDKWCDI (+ ST3 + ST4), --check-every 1 (J1-PEQ, ST3's map checks and the keep's fresh-fetch byte checks
#        in every round), --require-stress 1, --bf-acct 1, --bf-store 0, --bf-ncap 2048; every gate MUST. Run it first.
#        STEP_MIN 55, --mem 100G.
#   inta timing, C 128 (pool 65), B 216, 512 steps, probe 16, arms L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDI. STEP_MIN
#        125, --mem 240G. STOPS while PRED_STAMP_INT is None.
#   intc THE ORIGINAL ANCHOR: timing, C 81 (pool 18), B 287, 512 steps, probe 16, arms L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,
#        L4PTFDGUDCDI (the C 128 INT composition: F1 + F2 + F6 + F3 + ST1 + ST2 + ST4; no BF1, no keep). STEP_MIN 170
#        (MEASURED inta 70 min at B 216 x 287 / 216 = 93 min, C 81's smaller attention set ignored: an upper side; margin
#        1.8), --mem 300G. Memory: L4PTFD at C 81 B 287 peaks at 76.80 GB (MEASURED, r7mem); INT adds about 0.1 GB.
#        STOPS while PRED_STAMP_INT is None; the B 287 keys of PRED_INT_BY_B are None (REPORT) until the lead registers
#        them (with a new stamp) before intc.
#   intb timing, C 73 (pool 10), B 318, 512 steps, probe 16, --bf-store 257 (as bf1b), --bf-ncap 2048, arms L4PTFD,
#        L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDKWCDI (MTK = the ST3
#        control; its PEQ reference MT needs M). STEP_MIN 270, --mem 340G. STOPS while PRED_STAMP_INT is None.
# Every cell also runs SHJ1 right after SH (the shipped run with the compiled pooling: INT-VS-SHJ1's matched baseline).
# Basis of STEP_MIN (MEASURED, sacct, CITED by r2_j1_step.sh:35-39): i4dca (SH + 3 arms, C 128 B 216) 00:47:32; i4dcb
# 01:11:41; i4dm2 (SH + 6 arms, B 8) 00:09:18. MODEL: inta = i4dca + one D-arm part (11.9 min) + SHJ1 (2) + compile
# (<= 1) = about 62 min (margin 2.0); intb = i4dcb + 4 parts x 17.9 + SHJ1 (4) + the gilfree build (2) = about 149 min
# (margin 1.8); intm = i4dm2 + 9 parts (about 1.3 min each) + SHJ1 + the build = about 24 min (margin 2.3). Memory: the
# I4DC cells' --mem; BF1's landing ring per cell as r2_bf1_step.sh at 64e307f (intm / inta: ring 8 x n_cap 2048;
# intb: ring 6 x n_cap 6144 = 1.21 GB); device peak of the C 73 INT arm at intb: MODEL about 81.7 GB (BF1's own B arm
# about 81.6 GB, the BF1 writer's MODEL from bf1b's MEASURED 80.80 GB + the landing and the transport staging; + ST1's
# private pool < 0.05 GB + the compiled helpers' buffers about 0.03 GB), under the usable 83.75 GB (L60) by about 2.0 GB.
#
# OUTPUT: OUTROOT/int_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/int_<run>_<holder>_<cell>.step/,
# both frozen. EXIT: as r2_j1_step.sh (the cell's exit 0-19 / 20 / 30 / 2; 1 STOP; 2 usage; 3 another GPU; 4 not one GPU;
# 5 incomplete record; 6 a sha256 mismatch on the GPU node; 7 above OUT_MB; 124 / 137 the host timeout).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-int                                                      # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
SIF_SHA=14bfb15e1258821e637b5936f2060d7c7c2be20410fdee2a54ee0ffef754734a                  # PIN (nosa-env.sif.sha256)
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/int_jit                                   # ROOT (the compile caches)
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/spec                                                  # ROOT
SELF=benchmarks/Efficiency/r2_int_step.sh
CELLPY=benchmarks/Efficiency/r2_int_cell.py
FORKPY=benchmarks/Efficiency/fork_vllm_unified_attention.py
FORK_BLOB=f0dc0c183a388585e216505a6bbfc696531e8add                                       # PIN (fia6 94d5bb0's fork)
SPLITS=4
OUT_MB=2048
# granted <cell> <cell file>: true only when the file holds the cell's grant line with a `date -u` stamp.
granted() {
    grep -Eq "^GRANT_$1 = \"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\"\$" "$2"
}
main() {
stop() { echo "R2 int step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_int_step.sh --nosi <sha> <holder> <run id> <intm|inta|intb|intc>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
K=3 STRESS=0 CHECK=0 NEEDPRED=0 PE=16 ACCT=0 BFX=0 NCAP=2048 RING=8
case $CELL in
    intm) POOL=10 BATCH=8 STEPS=72 CHECK=1 STRESS=1 PE=4 ACCT=1 STEP_MIN=55 MEM=100G
          ARMS=L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDM,L4PTFDGUDKWCDM,L4PTFDGUDCDMT,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDCDMTR,L4PTFDGUDKWCDMTKR,L4PTFDGUDCDI,L4PTFDGUDKWCDI ;;
    inta) POOL=65 BATCH=216 STEPS=512 NEEDPRED=1 STEP_MIN=125 MEM=240G
          ARMS=L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDI ;;
    intc) POOL=18 BATCH=287 STEPS=512 NEEDPRED=1 STEP_MIN=170 MEM=300G
          ARMS=L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDI ;;
    intb) POOL=10 BATCH=318 STEPS=512 NEEDPRED=1 BFX=257 NCAP=6144 RING=6 STEP_MIN=270 MEM=340G
          ARMS=L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDM,L4PTFDGUDKWCDMT,L4PTFDGUDKWCDMTK,L4PTFDGUDKWCDI ;;
    *) stop "cell '$CELL' (intm, inta, intb or intc)" ;;
esac
DESC="arms SH+$ARMS steps $STEPS K $K probe $PE check $CHECK stress $STRESS bf-store $BFX ncap $NCAP ring $RING acct $ACCT"
CARGS=(--batch "$BATCH" --steps "$STEPS" --pool "$POOL" --spec "$K" --arms "$ARMS" --fork-store 0 --prefill-hook nocopy
       --tu-gemm natural --spec-journal-blocks 256 --spec-twin fused --spec-body fused --spec-graph on --spec-redirect on
       --probe-every "$PE" --check-every "$CHECK" --require-stress "$STRESS" --st1-commit rows --bf-store "$BFX"
       --bf-ncap "$NCAP" --bf-ring "$RING" --bf-worker thread --bf-acct "$ACCT" --bf-close-wait-ms 0 --bf-packer gilfree
       --bf-threads 14 --bf-cap-groups 2048 --bf-batch 8 --bf-late defer --bf-late-wait-ms 250)
FILE_KB=$((OUT_MB * 1024))
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
cmp -s -- "$0" <(git -C "$NOSI" show "$NSHA:$SELF") || stop "$0 is not the bytes of $SELF at $NSHA (blob $blob)"
for f in "$CELLPY" "$FORKPY" benchmarks/Efficiency/r2_int_arms.py benchmarks/Efficiency/r2_j1_select.py \
         benchmarks/Efficiency/r2_j1_cell.py benchmarks/Efficiency/r2_st1_commit.py benchmarks/Efficiency/r2_st1_cell.py \
         benchmarks/Efficiency/r2_bf1_fetch.py benchmarks/Efficiency/r2_bf1_cell.py benchmarks/Efficiency/gilfree_pack.py \
         benchmarks/Efficiency/gilfree_pack.cpp nosi/nosi/r2_transport_gilfree_v3.py nosi/nosi/kstep/backend.py \
         nosi/nosi/kstep/transfer.py benchmarks/Efficiency/r2_i4dc_cell.py \
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
git -C "$NOSI" show "$NSHA:$CELLPY" | grep -q '^PRED_STAMP_INT = None$' && REGISTERED=0
[ -f "$SIF" ] || stop "no image $SIF (its sha256 is checked on the GPU node)"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
[ -f "$WL/c16k_tokens.bin" ] || stop "no $WL/c16k_tokens.bin (its sha256 is checked on the GPU node)"
WLF=$WL/c16k_tokens.bin
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
OUT=$OUTROOT/int_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/int_${RUN}_${JOB}_$CELL
CCJIT=$CCROOT/int_${RUN}_${JOB}_$CELL
NAME=int-${CELL}-${RUN}-$JOB
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
    ""|*,*) echo "R2 int child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1 IMG=$2 IMGSHA=$3 WLF=$4 WLSHA=$5 KB=$6
shift 6
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 int child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
echo "R2 int child: host $(hostname) gpu $seen; hashing the image at $(date -u +%FT%TZ)" >&2
h=$(/usr/bin/sha256sum -- "$IMG") || { echo "R2 int child: cannot hash $IMG" >&2; exit 6; }
[ "${h%% *}" = "$IMGSHA" ] || { echo "R2 int child: image sha256 ${h%% *} is not $IMGSHA" >&2; exit 6; }
echo "R2 int child: image sha256 $IMGSHA OK at $(date -u +%FT%TZ)" >&2
h=$(/usr/bin/sha256sum -- "$WLF") || { echo "R2 int child: cannot hash $WLF" >&2; exit 6; }
[ "${h%% *}" = "$WLSHA" ] || { echo "R2 int child: workload sha256 ${h%% *} is not $WLSHA" >&2; exit 6; }
echo "R2 int child: workload sha256 $WLSHA OK" >&2
ulimit -f "$KB" || { echo "R2 int child: ulimit -f $KB failed" >&2; exit 6; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/bash -c "$CHILD" int-child "$GPU_UUID" "$SIF" "$SIF_SHA" "$WLF" "$WL_SHA" "$FILE_KB" "$APPTAINER"
      --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/$CELLPY" --out "$OUT" --model "$MODEL" --workload "$WL" "${CARGS[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 int step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob fork $FORK_BLOB image sha256 $SIF_SHA (checked on the GPU node) workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE); cell granted $APPROVED; registered $REGISTERED"
    echo "R2 int step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH $DESC STEP_MIN $STEP_MIN mem $MEM out_mb $OUT_MB claim $CLAIM compile caches $CCJIT"
    printf '%q ' "${ARGV[@]}"
    echo
fi
[ "$APPROVED" = 1 ] || stop "cell $CELL is not granted at $NSHA ($CELLPY: GRANT_$CELL; default deny)"
[ "$NEEDPRED" = 0 ] || [ "$REGISTERED" = 1 ] ||
    stop "r2_int_cell.PRED_STAMP_INT is None at $NSHA: register PRED_INT_BY_B and its stamp (date -u) before $CELL"
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
printf '\nnosi=%s wrapper_blob=%s fork_blob=%s image_sha256=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s kind=int pool=%s batch=%s arms=%s steps=%s k=%s probe=%s check=%s stress=%s bf_store=%s ncap=%s acct=%s workload=%s out_mb=%s compile_caches=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$FORK_BLOB" "$SIF_SHA" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$ARMS" \
    "$STEPS" "$K" "$PE" "$CHECK" "$STRESS" "$BFX" "$NCAP" "$ACCT" "$WL_SHA" "$OUT_MB" "$CCJIT" "$(date -u +%FT%TZ)" \
    >> "$REC/command.txt" || stop "cannot write $REC ($CLAIM stays)"
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
echo "R2 int step: exit $rc; record $REC; output $OUT; output bytes $total (limit $OUT_MB MB, over $over); record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
[ "$rc" = 0 ] && [ "$over" = 1 ] && exit 7
exit "$rc"
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi
