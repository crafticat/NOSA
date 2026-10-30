#!/bin/bash
# RECON7 (KS/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the MEASUREMENT
# cells of the reconciliation's discriminators. ONE bounded GPU step in an EXISTING holder per call. A COPY of
# r2_i4dc_step.sh (2e9e497): the same pins, checks, record and freeze; the ROOT (this tree, nosi-r2-recon7), the cells,
# the cell files, the files list and the PROTOCOL STOP change. The claim is the SAME as the other spec step scripts'
# (OUTROOT/sp_active_<holder>): a step of any of them blocks the others on one holder (intended).
# THE PROTOCOL STOP: EVERY cell STOPS while benchmarks/Efficiency/r2_recon7_protocol.py reads
# `PROTOCOL_STAMP_RECON7 = None` at <sha> (the human approves the GPU protocol of docs/NOTES_recon7.md section 3 and sets
# the stamp in one commit). DRYRUN prints the pins and the step command first, then STOPS the same way.
# THE COMPILE CACHES (lesson C1: a HOME cache leaks): TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR and TRITON_HOME live in
# a NEW directory per step under CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_jit (cold).
# The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_recon7_step.sh --nosi <sha> <holder job> <run id> <r7pm|r7pa|r7pb|r7s|r7mem>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_recon7_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (the pin checks, the uuid file read, the printed step command, the protocol STOP; no squeue, claim, srun or
#        write)
#
# CELLS (docs/NOTES_recon7.md section 3 holds the protocol of each: what it decides, duration, gates, falsifier):
#   r7pm  cell (a), the MINIATURE: C 73 (pool 10), B 8, 72 steps, probe 4, profile 8,12 (label 12), count 16,20, arms
#         L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDP,L4PTFDGUDKWCDP. Run it first.
#         STEP_MIN 30, --mem 100G.
#   r7pa  cell (a), C 128 (pool 65), B 216, 512 steps, probe 16, profile 96,112 (label 112), count 48,64,144, arms
#         L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDP. STEP_MIN 110, --mem 240G.
#   r7pb  cell (a), C 73 (pool 10), B 318, the same rounds, arms L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDP.
#         STEP_MIN 160, --mem 340G.
#   r7s   cell X1 (b), the concurrency harness: (a) capability, (b) attainable overlap of the legal pairs, matched serial
#         controls, order-rotated; shapes C 128 B 216 and C 73 B 318 (pool 65: the knob only; no model). STEP_MIN 15
#         (hard wall), --mem 64G. Protocol: docs/NOTES_recon7.md section "X1 protocol (about 5 min)".
#   r7mem cell (c), the memory probe: C 81 (pool 18), B 64 then B 192, 512 steps, 49 probe rounds. STEP_MIN 70,
#         --mem 240G.
# Basis of STEP_MIN (MEASURED, sacct, the step records' sacct.txt under OUTROOT): i4dca (SH + L4PTFD, L4PTFDGUDC,
# L4PTFDGUDCD; C 128 B 216) 00:47:32; i4dcb (C 73 B 318, the same parts) 01:11:41; i4dm2 (SH + 6 arms, B 8) 00:09:08 /
# 00:09:18; i4ca (SH + L4PTFD + the profiled L4PTFDP) 00:36:35, i4cb 00:54:48; i4mp73 / i4mp128 (SH + the probe at B 64)
# 00:11:21 / 00:11:17. MODEL: r7pa = i4dca + one more D-arm part (11.5 min) + the probe's windows (about 1.5 min) =
# about 60.5 min (margin 1.8); r7pb = i4dcb + 17 + 2 = about 91 min (margin 1.75); r7pm = i4dm2 + one part + the
# windows = about 10.5 min (margin 2.8); r7s = no model: setup, one Triton compile of the fork and of the gathers, 2 shapes
# x 18 configurations x (2 + 10) passes = about 5 min (margin 3; each shape stops itself at 240 s); r7mem = i4mp73 (B 64) + B 192 (the prefills scale with
# B: 3 x 174 s SH, 3 x 277 s arena) = about 35 min (margin 2). Memory: the I4DC / I4M cells' --mem (sacct MaxRSS 3.1 GB;
# the pinned host KV is not in MaxRSS: 106 GB at C 81 B 192).
#
# OUTPUT: OUTROOT/recon7_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/recon7_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-recon7                                                   # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/recon7_jit                                # ROOT (the compile caches)
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/spec                                                  # ROOT
SELF=benchmarks/Efficiency/r2_recon7_step.sh
PROTPY=benchmarks/Efficiency/r2_recon7_protocol.py
FORKPY=benchmarks/Efficiency/fork_vllm_unified_attention.py
FORK_BLOB=f0dc0c183a388585e216505a6bbfc696531e8add                                       # PIN (fia6 94d5bb0's fork)
SPLITS=4
main() {
stop() { echo "R2 recon7 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_recon7_step.sh --nosi <sha> <holder> <run id> <r7pm|r7pa|r7pb|r7s|r7mem>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
CK=(--spec 3 --fork-store 0 --prefill-hook nocopy --tu-gemm natural --spec-journal-blocks 256 --spec-twin fused
    --spec-body fused --spec-graph on --spec-redirect on)
case $CELL in
    r7pm)  CELLPY=benchmarks/Efficiency/r2_recon7_cell.py POOL=10 BATCH=8 STEP_MIN=30 MEM=100G
           ARMS=L4PTFD,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD,L4PTFDGUDCDP,L4PTFDGUDKWCDP
           CARGS=(--batch "$BATCH" --steps 72 --pool "$POOL" --arms "$ARMS" "${CK[@]}" --probe-every 4 --check-every 0
                  --require-stress 0 --profile-rounds 8,12 --label-rounds 12 --count-rounds 16,20 --draft-marks on)
           DESC="arms SH+$ARMS steps 72 probe 4 profile 8,12 label 12 count 16,20" ;;
    r7pa)  CELLPY=benchmarks/Efficiency/r2_recon7_cell.py POOL=65 BATCH=216 STEP_MIN=110 MEM=240G
           ARMS=L4PTFD,L4PTFDGUDC,L4PTFDGUDCD,L4PTFDGUDCDP
           CARGS=(--batch "$BATCH" --steps 512 --pool "$POOL" --arms "$ARMS" "${CK[@]}" --probe-every 16 --check-every 0
                  --require-stress 0 --profile-rounds 96,112 --label-rounds 112 --count-rounds 48,64,144
                  --draft-marks on)
           DESC="arms SH+$ARMS steps 512 probe 16 profile 96,112 label 112 count 48,64,144" ;;
    r7pb)  CELLPY=benchmarks/Efficiency/r2_recon7_cell.py POOL=10 BATCH=318 STEP_MIN=160 MEM=340G
           ARMS=L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD,L4PTFDGUDKWCDP
           CARGS=(--batch "$BATCH" --steps 512 --pool "$POOL" --arms "$ARMS" "${CK[@]}" --probe-every 16 --check-every 0
                  --require-stress 0 --profile-rounds 96,112 --label-rounds 112 --count-rounds 48,64,144
                  --draft-marks on)
           DESC="arms SH+$ARMS steps 512 probe 16 profile 96,112 label 112 count 48,64,144" ;;
    r7s)   CELLPY=benchmarks/Efficiency/r2_recon7_streams.py POOL=65 BATCH=0 STEP_MIN=15 MEM=64G
           CARGS=(--shapes 128:216,73:318 --new-blocks 128:1.226,73:5.0 --union-pages 72 --win-share 0.5 --layers 32
                  --reps 10 --warmup 2 --seed 7 --max-shape-s 240)
           DESC="X1: shapes C 128 B 216, C 73 B 318; NEW blocks per (layer, KV head, request) 1.226 / 5.0; union 72; seed 7" ;;
    r7mem) CELLPY=benchmarks/Efficiency/r2_recon7_mem.py POOL=18 BATCH=0 STEP_MIN=70 MEM=240G
           CARGS=(--batches 64,192 --steps 512 --pool "$POOL" "${CK[@]}" --probe-every 16 --probe-rounds 49)
           DESC="C 81 B 64 then B 192, 512 steps, 49 probe rounds" ;;
    *) stop "cell '$CELL' (r7pm, r7pa, r7pb, r7s or r7mem)" ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in "$CELLPY" "$PROTPY" "$FORKPY" benchmarks/Efficiency/r2_recon7_cell.py benchmarks/Efficiency/r2_recon7_probe.py \
         benchmarks/Efficiency/r2_recon7_streams.py benchmarks/Efficiency/r2_recon7_mem.py \
         benchmarks/Efficiency/r2_i4c_cell.py benchmarks/Efficiency/r2_i4m_cell.py benchmarks/Efficiency/r2_wsm_cell.py \
         benchmarks/Efficiency/r2_p3_profile.py benchmarks/Efficiency/r2_i4dc_cell.py \
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
APPROVED=1
git -C "$NOSI" show "$NSHA:$PROTPY" | grep -q '^PROTOCOL_STAMP_RECON7 = None$' && APPROVED=0
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
OUT=$OUTROOT/recon7_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/recon7_${RUN}_${JOB}_$CELL
CCJIT=$CCROOT/recon7_${RUN}_${JOB}_$CELL
NAME=recon7-${CELL}-${RUN}-$JOB
CLAIM=$OUTROOT/sp_active_$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$CCJIT/triton"
      --env TRITON_HOME="$CCJIT/triton_home" --env TORCHINDUCTOR_CACHE_DIR="$CCJIT/inductor"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 recon7 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 recon7 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
WARGS=()
case $CELL in r7pm|r7pa|r7pb|r7mem) WARGS=(--model "$MODEL" --workload "$WL") ;; esac
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" recon7-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/$CELLPY" --out "$OUT" "${WARGS[@]}" "${CARGS[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 recon7 step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob fork $FORK_BLOB workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE); protocol approved $APPROVED"
    echo "R2 recon7 step: cell $CELL: C $((63 + POOL)) (pool $POOL) $DESC STEP_MIN $STEP_MIN mem $MEM claim $CLAIM compile caches $CCJIT"
    printf '%q ' "${ARGV[@]}"
    echo
fi
[ "$APPROVED" = 1 ] || stop "PROTOCOL_STAMP_RECON7 is None at $NSHA ($PROTPY): the human approves the GPU protocol of docs/NOTES_recon7.md first"
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
printf '\nnosi=%s wrapper_blob=%s fork_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s kind=recon7 pool=%s batch=%s desc=%s workload=%s compile_caches=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$FORK_BLOB" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$DESC" "$WL_SHA" \
    "$CCJIT" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
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
echo "R2 recon7 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
