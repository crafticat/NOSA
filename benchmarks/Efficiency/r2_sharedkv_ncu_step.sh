#!/bin/bash
# R2 SHARED READY-KV READS, CELL S3 (docs/R2_SHAREDKV2_KEYS_S3.md): ONE bounded GPU step in an EXISTING holder that
# runs benchmarks/Efficiency/r2_sharedkv_ncu.py --mode orchestrate: Nsight Compute (the image's ncu, PIN below; used
# by r2-runs/jhbm, job 2175791) on the eager tick with SharedKVProbe, DRAM bytes / L2 hit rate of piece 3 v2 vs the
# shared kernel on the same keyed calls. A copy of benchmarks/Efficiency/r2_sharedkv_step.sh (its pins, uuid file,
# claim, record and freeze); the changes: the worktree, OUTROOT, the claim skv2_active, the driver, and the ARM:
#   b32-split | b32-whole   --backend resident (the ResidentStore), B 32, M_FIX 128
#   b64-split | b64-whole   --backend b2 (OwnResidency on the SG5 path, the plan's page keys), B 64, M_FIX 256,
#                           C 96, X 32, R_ver 16 (SG5's primary point), the gather transport
#   b32-split-fused | b64-split-fused   [sharedkv3] the same with --union fused (the groups once per tick + one union
#                           kernel per call; docs/R2_SHAREDKV3_UNION.md)
# Order (the plan): b32-split, b64-split, b32-whole, b64-whole; stop at the first MUST failure. [sharedkv3] then
# b32-split-fused, b64-split-fused.
# [sharedkv3] NOSI is the r2-sharedkv3 worktree. [sharedkv4] now r2-sharedkv4; the claim as r2_sharedkv_step.sh
# (existence check first, an owner file, a bounded wait and a kept claim on a client timeout).
# SOURCE ONLY, NOT RUN, NOT REVIEWED: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_sharedkv_ncu_step.sh --nosi <sha> <holder job> <run id> <arm>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_sharedkv_ncu_step.sh --nosi <sha> <holder job> <run id> <arm>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, TICK_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses TICK_HOLDERS.
#
# OUTPUT: OUTROOT/s3_<arm>_<run>_<holder>/ (summary.json, metrics_query.txt, rep_*.ncu-rep, *_raw.csv, P_*/cell.json,
# P_*.log) and the step record OUTROOT/s3_<arm>_<run>_<holder>.step/, both frozen. EXIT: the driver's failed MUST
# count; 2 refusal; 3 / 4 GPU identity; 124 / 137 timeout; 1 STOP before the launch; 5 an exit-0 step with an
# incomplete record or a non-COMPLETED 0:0 sacct row.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-sharedkv4                                                # ROOT (this branch)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (r2_g0_step.sh)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/sharedkv2                                             # ROOT
NCU=/opt/nvidia/nsight-compute/2025.1.0/ncu                                              # PIN (inside the image)
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN (piece 3 v2)
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
SKV_PATH=nosi/nosi/bi_paged_attention_shared.py                                          # tracked at --nosi
SPLITS=4
STEPS=16
WINDOW=32,64
SELF=benchmarks/Efficiency/r2_sharedkv_ncu_step.sh
STEP_MIN=90
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 sharedkv S3 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_sharedkv_ncu_step.sh --nosi <sha> <holder> <run id> b32-split|b32-whole|b64-split|b64-whole|b32-split-fused|b64-split-fused" >&2
          exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 ARM=$5
UNION=torch
case $ARM in
    b32-split) BACKEND=resident BATCH=32 M_FIX=128 MODE=split ;;
    b32-whole) BACKEND=resident BATCH=32 M_FIX=128 MODE=whole ;;
    b64-split) BACKEND=b2 BATCH=64 M_FIX=256 MODE=split ;;
    b64-whole) BACKEND=b2 BATCH=64 M_FIX=256 MODE=whole ;;
    b32-split-fused) BACKEND=resident BATCH=32 M_FIX=128 MODE=split UNION=fused ;;
    b64-split-fused) BACKEND=b2 BATCH=64 M_FIX=256 MODE=split UNION=fused ;;
    *) usage ;;
esac
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${TICK_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "TICK_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$TICK_HOLDERS
fi
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] || stop "no GPU uuid file $UFILE (the lead writes it for each holder)"
GPU_UUID=$(head -n 1 "$UFILE")
[ "$(wc -l < "$UFILE")" -le 1 ] || stop "$UFILE has more than one line"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold a uuid (no GPU- prefix): '$GPU_UUID'"
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_sharedkv_ncu.py benchmarks/Efficiency/r2_tick_sg5.py \
         benchmarks/Efficiency/r2_g0bi_gate.py benchmarks/Efficiency/r2_g0_gate.py nosi/nosi/r2_tick.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py \
         nosi/nosi/r2_positional_twin.py nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py \
         nosi/nosi/max_pooling_per_request.py nosi/nosi/r2_residency.py nosi/nosi/r2_residency_sb.py \
         nosi/nosi/r2_residency_sc.py nosi/nosi/r2_residency_sparse.py nosi/nosi/r2_own2.py nosi/nosi/r2_own2_ops.py \
         "$SKV_PATH"; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3 in this tree ($NSHA:$BIATTN_PATH) is not the pinned blob $BIATTN_BLOB"
SKV_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$SKV_PATH") || stop "$SKV_PATH is not tracked at $NSHA"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
OUT=$OUTROOT/s3_${ARM}_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/s3_${ARM}_${RUN}_$JOB
NAME=skv2-s3-${ARM}-${RUN}-$JOB
ATTN_MOD=$REC/piece3_bi_paged_attention.py
SKV_MOD=$REC/bi_paged_attention_shared.py
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_R2_DECODE=1 --env NOSI_MODEL_PATH="$MODEL" --env TICK_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 sharedkv S3 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 sharedkv S3 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=200G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" skv2-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_sharedkv_ncu.py" --mode orchestrate --out "$OUT"
      --model "$MODEL" --workload "$WL" --biattn "$ATTN_MOD" --sharedkv "$SKV_MOD" --backend "$BACKEND"
      --attn-mode "$MODE" --union "$UNION" --batch "$BATCH" --S 16128 --steps "$STEPS" --K 3 --LAMBDA 7 --m-fix "$M_FIX"
      --C 96 --X 32 --R-ver 16 --window "$WINDOW" --ncu "$NCU" --variants all,none --child-timeout 2400)
if [ "$DRY" = 1 ]; then
    echo "R2 sharedkv S3 step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 sharedkv S3 step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; shared $SKV_BLOB -> $SKV_MOD; arm $ARM backend $BACKEND B $BATCH M_FIX $M_FIX mode $MODE union $UNION window $WINDOW"
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
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] ||                # [sharedkv4 MINOR-7b] BEFORE the claim
    stop "$OUT, $REC or $JIT exists: a new run id (no claim taken)"
mkdir "$OUTROOT/skv2_active" 2>/dev/null || stop "$OUTROOT/skv2_active exists (another step, or an uncertain one; its owner file says which)"
printf 'arm=%s run=%s holder=%s step_name=%s nosi=%s out=%s utc_claim=%s\n' "$ARM" "$RUN" "$JOB" "$NAME" "$NSHA" \
    "$OUT" "$(date -u +%FT%TZ)" > "$OUTROOT/skv2_active/owner" ||           # [sharedkv4 MINOR-7a] the claim's owner
    stop "cannot write $OUTROOT/skv2_active/owner (skv2_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (skv2_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (skv2_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (skv2_active stays)"
git -C "$NOSI" show "$SKV_BLOB" > "$SKV_MOD" && [ "$(git hash-object -- "$SKV_MOD")" = "$SKV_BLOB" ] ||
    stop "cannot copy the shared module's blob $SKV_BLOB to $SKV_MOD (skv2_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (skv2_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (skv2_active stays)"
printf 'biattn=%s:%s blob=%s copy=%s sharedkv=%s blob=%s copy=%s arm=%s backend=%s batch=%s m_fix=%s mode=%s window=%s ncu=%s splits=%s steps=%s\n' \
    "$BIATTN_SHA" "$BIATTN_PATH" "$BIATTN_BLOB" "$ATTN_MOD" "$NSHA:$SKV_PATH" "$SKV_BLOB" "$SKV_MOD" "$ARM" "$BACKEND" \
    "$BATCH" "$M_FIX" "$MODE" "$WINDOW" "$NCU" "$SPLITS" "$STEPS" >> "$REC/command.txt" ||
    stop "cannot write $REC (skv2_active stays)"
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
writer=ended
case $rc in
    124|137)                                                  # [sharedkv4 MINOR-7c, C10] a timeout: the client
        ok=0                                                  # ended, the step may still write. Wait (bounded) until
        writer=running                                        # no step of the holder is listed; the claim STAYS
        for _ in $(seq 1 60); do
            if nosteps > /dev/null; then writer=gone_from_squeue; break; fi
            sleep 10
        done
        echo "timeout: writer=$writer after the wait; skv2_active stays (L40 cleanup after sacct)" >> "$REC/exit.txt" ;;
esac
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    [ "$t" = "$OUT" ] && [ "$writer" = running ] && continue       # never touch an output that may still be written
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rm -f "$OUTROOT/skv2_active/owner" && rmdir "$OUTROOT/skv2_active" || ok=0
fi
echo "R2 sharedkv S3 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
