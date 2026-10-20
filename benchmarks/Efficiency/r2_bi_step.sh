#!/bin/bash
# R2 MILESTONE 2, STEP 7 (2): ONE bounded GPU step in an EXISTING holder that runs the batch-invariance microgates
# benchmarks/Efficiency/r2_bi_microgate.py (B1 rmsnorm / nosa_linear / rope, B2 stage 1 with the shared-table alias,
# B3 GEMMs: cuBLAS natural M / cuBLAS padded M (G-b, r2_spec_core.PaddedLinear) / the BI matmul of the bigemm
# branch when its pins below are set, with the cost table, B4 piece 3's BI paged attention with its cost against
# FA2-NOSA, B5 the deterministic selection). Random bf16 inputs with NOSA-8B shapes: no model, no workload.
# The pattern of benchmarks/Efficiency/r2_g0_step.sh, with two changes:
#   - the GPU uuid is NOT in this file. It is read from the lead-owned file HOLDERS/<holder job id>.gpu_uuid (one
#     line, a uuid without "GPU-");
#   - modules of OTHER branches (piece 3; the BI GEMM later) are pinned by commit, path and blob id, copied from the
#     shared object store into the step record (git show <blob>), checked by git hash-object, and loaded by path.
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_bi_step.sh --nosi <sha> <holder job> <run id>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_bi_step.sh --nosi <sha> <holder job> <run id>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, BI_HOLDERS=<dir> replaces HOLDERS (the CPU test S13 uses it); a real run refuses BI_HOLDERS.
#
# OUTPUT: OUTROOT/bi_<run>_<holder>/summary.json and the step record OUTROOT/bi_<run>_<holder>.step/ (command with the
# uuid and its file, stdout with one GATE / REPORT line per gate, stderr, exit + accounting, sacct row), both frozen.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed uuid file included); 5 an exit-0 step
# whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-spec                                                  # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/bi                                                    # ROOT
# piece 3 (nosi-r2-m2-biattn, accepted: CPU 24/24 at 4c1243f): the BI paged attention, loaded by path
BIATTN_SHA=4c1243f3f0441725943e8dcd515dfaac8c27446c                                      # PIN
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=7477ddd01b7b47585d5c45557c7d41c687906b2d                                     # PIN
# the BI GEMM (nosi-r2-m2-bigemm, option G-a): EMPTY until its interface is forwarded; empty = B3 has no BI column
BIGEMM_SHA=                                                                              # PIN (unset)
BIGEMM_PATH=                                                                             # PIN (unset)
BIGEMM_BLOB=                                                                             # PIN (unset)
BIGEMM_FN=                                                                               # PIN (unset)
SELF=benchmarks/Efficiency/r2_bi_step.sh
STEP_MIN=30
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 BI step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_bi_step.sh --nosi <sha> <holder> <run id>" >&2; exit 2; }
[ $# = 4 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${BI_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "BI_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$BI_HOLDERS
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
for f in benchmarks/Efficiency/r2_bi_microgate.py nosi/nosi/r2_spec_core.py nosi/nosi/r2_positional_twin.py \
         nosi/nosi/stage1_seqused.py nosi/nosi/nosa_linear.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
GEMM_ARGS=()
if [ -n "$BIGEMM_SHA$BIGEMM_PATH$BIGEMM_BLOB$BIGEMM_FN" ]; then
    [ -n "$BIGEMM_SHA" ] && [ -n "$BIGEMM_PATH" ] && [ -n "$BIGEMM_BLOB" ] && [ -n "$BIGEMM_FN" ] ||
        stop "the BIGEMM pins are partly set"
    [ "$(git -C "$NOSI" rev-parse -q --verify "$BIGEMM_SHA:$BIGEMM_PATH")" = "$BIGEMM_BLOB" ] ||
        stop "BI GEMM: $BIGEMM_PATH at $BIGEMM_SHA is not the blob $BIGEMM_BLOB"
fi
[ -f "$SIF" ] || stop "no image $SIF"
OUT=$OUTROOT/bi_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/bi_${RUN}_$JOB
NAME=bi-${RUN}-$JOB
ATTN_MOD=$REC/piece3_bi_paged_attention.py
[ -z "$BIGEMM_SHA" ] || GEMM_ARGS=(--bigemm "$REC/bigemm_$(basename "$BIGEMM_PATH")" --bigemm-fn "$BIGEMM_FN")
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp" --env BI_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 BI child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 BI child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" bi-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_bi_microgate.py" --out "$OUT" --batches 64,128 --U 3
      --warm 5 --reps 30 --seed 7 --biattn "$ATTN_MOD" ${GEMM_ARGS[@]+"${GEMM_ARGS[@]}"})
if [ "$DRY" = 1 ]; then
    echo "R2 BI step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 BI step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; BI GEMM ${BIGEMM_SHA:-unset}${BIGEMM_SHA:+:$BIGEMM_PATH blob $BIGEMM_BLOB fn $BIGEMM_FN}"
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
mkdir "$OUTROOT/bi_active" 2>/dev/null || stop "$OUTROOT/bi_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (bi_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (bi_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (bi_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (bi_active stays)"
if [ -n "$BIGEMM_SHA" ]; then
    git -C "$NOSI" show "$BIGEMM_BLOB" > "${GEMM_ARGS[1]}" && [ "$(git hash-object -- "${GEMM_ARGS[1]}")" = "$BIGEMM_BLOB" ] ||
        stop "cannot copy the BI GEMM blob $BIGEMM_BLOB to ${GEMM_ARGS[1]} (bi_active stays)"
fi
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (bi_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s utc_start=%s\n' "$NSHA" "$blob" \
    "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" || stop "cannot write $REC (bi_active stays)"
printf 'biattn=%s:%s blob=%s copy=%s bigemm=%s:%s blob=%s fn=%s\n' "$BIATTN_SHA" "$BIATTN_PATH" "$BIATTN_BLOB" "$ATTN_MOD" \
    "${BIGEMM_SHA:-unset}" "${BIGEMM_PATH:-}" "${BIGEMM_BLOB:-}" "${BIGEMM_FN:-}" >> "$REC/command.txt" ||
    stop "cannot write $REC (bi_active stays)"
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
    rmdir "$OUTROOT/bi_active" || ok=0
fi
echo "R2 BI step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
