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
#   bash benchmarks/Efficiency/r2_bi_step.sh --nosi <sha> <holder job> <run id> gm1
#        (r2-mbase cell gm1: r2_bi_microgate.py --gemm-scan, the GEMM cost scan at M 64 .. 1148 and the EXACT tick's
#        pairs (u B, 4 B) for B 287 / 216 / 64; no B1-B5, no piece 3; STEP_MIN 15; output bi_<run>_<holder>_gm1)
#   bash benchmarks/Efficiency/r2_bi_step.sh --nosi <sha> <holder job> <run id> bigc-tune|bigc-gate|bigc-time
#        (r2-bigemmc, G-c = nosi/nosi/bi_linear_c.py; the strict variant is THE CANDIDATE, the bucket variant a
#        RESEARCH ARM only (the relaxed law is NOT approved). Their OWN claim OUTROOT/bigc_active (not bi_active).
#        RUN ORDER: bigc-tune first (freezes the table: table.json + "BIGC_TABLE_SHA <hex>" in its stdout), then set
#        the pins BIGC_TABLE / BIGC_TABLE_SHA below (or commit the table into bi_linear_c.TABLE with "tuned": True),
#        then bigc-gate and bigc-time; both STOP while neither holds.
#          bigc-tune  bigc_tune.py: every candidate launch timed per shape and M, cross-launch bits; STEP_MIN 60
#          bigc-gate  bigc_gate.py --variants strict,bucket: rows across M, windows, CUDA graph, MUST-detect controls
#                     (split-K 2, natural cuBLAS, K reversed); STEP_MIN 30
#          bigc-time  r2_bi_microgate.py --gemm-scan (gm1's arguments) --bigemm-c both: G-c columns, predictions;
#                     STEP_MIN 20)
#
# OUTPUT: OUTROOT/bi_<run>_<holder>/summary.json and the step record OUTROOT/bi_<run>_<holder>.step/ (command with the
# uuid and its file, stdout with one GATE / REPORT line per gate, stderr, exit + accounting, sacct row), both frozen.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed uuid file included); 5 an exit-0 step
# whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-bigemmc                                                  # ROOT
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
# [bigemmc] the frozen G-c table of a bigc-tune run (its table.json under OUTROOT and the BIGC_TABLE_SHA line of its
# stdout): EMPTY until bigc-tune ran; the python cell refuses a file whose canonical digest is not the pin
BIGC_TABLE=                                                                              # PIN (unset)
BIGC_TABLE_SHA=                                                                          # PIN (unset)
SELF=benchmarks/Efficiency/r2_bi_step.sh
STEP_MIN=30
main() {
stop() { echo "R2 BI step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_bi_step.sh --nosi <sha> <holder> <run id> [gm1|bigc-tune|bigc-gate|bigc-time]" >&2; exit 2; }
{ [ $# = 4 ] || [ $# = 5 ]; } && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=${5:-}
CLAIM=bi_active
case $CELL in
    "") ;;
    gm1) STEP_MIN=15 ;;
    bigc-tune) STEP_MIN=60 CLAIM=bigc_active ;;
    bigc-gate) STEP_MIN=30 CLAIM=bigc_active ;;
    bigc-time) STEP_MIN=20 CLAIM=bigc_active ;;
    *) stop "cell '$CELL' (none, gm1, bigc-tune, bigc-gate or bigc-time)" ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${BI_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "BI_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$BI_HOLDERS
fi
if [ -n "${BIGC_TEST_TABLE:-}${BIGC_TEST_TABLE_SHA:-}" ]; then          # the CPU test BC-S only
    [ "$DRY" = 1 ] || stop "BIGC_TEST_TABLE is for DRYRUN (the CPU test) only"
    BIGC_TABLE=${BIGC_TEST_TABLE:-} BIGC_TABLE_SHA=${BIGC_TEST_TABLE_SHA:-}
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
TABLE_ARGS=()
case $CELL in
    bigc-*)
        for f in nosi/nosi/bi_linear_c.py benchmarks/Efficiency/bigc_gate.py benchmarks/Efficiency/bigc_tune.py; do
            git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
        done
        if [ -n "$BIGC_TABLE$BIGC_TABLE_SHA" ]; then
            [ "$CELL" != bigc-tune ] || stop "bigc-tune makes the table: unset BIGC_TABLE / BIGC_TABLE_SHA"
            [ -n "$BIGC_TABLE" ] && [ -n "$BIGC_TABLE_SHA" ] || stop "the BIGC_TABLE pins are partly set"
            [[ $BIGC_TABLE_SHA =~ ^[0-9a-f]{64}$ ]] || stop "BIGC_TABLE_SHA '$BIGC_TABLE_SHA' is not a sha256"
            case $BIGC_TABLE in /mnt/beegfs/ojerbi/*) ;; *) stop "BIGC_TABLE $BIGC_TABLE is not under /mnt/beegfs/ojerbi (bound)" ;; esac
            [ -f "$BIGC_TABLE" ] || stop "no table file $BIGC_TABLE"
            TABLE_ARGS=(--table "$BIGC_TABLE" --table-sha "$BIGC_TABLE_SHA")
        elif [ "$CELL" != bigc-tune ]; then
            git -C "$NOSI" show "$NSHA:nosi/nosi/bi_linear_c.py" | grep -q '^    "tuned": True,$' ||
                stop "$CELL needs the frozen table: run bigc-tune, then set BIGC_TABLE / BIGC_TABLE_SHA (the source TABLE is untuned)"
        fi ;;
esac
[ -f "$SIF" ] || stop "no image $SIF"
OUT=$OUTROOT/bi_${RUN}_$JOB${CELL:+_$CELL}
REC=$OUT.step
JIT=$JITBASE/bi_${RUN}_$JOB${CELL:+_$CELL}
NAME=bi-${RUN}-$JOB${CELL:+-$CELL}
ATTN_MOD=$REC/piece3_bi_paged_attention.py
[ -z "$BIGEMM_SHA" ] || [ -n "$CELL" ] || GEMM_ARGS=(--bigemm "$REC/bigemm_$(basename "$BIGEMM_PATH")" --bigemm-fn "$BIGEMM_FN")
PYFILE=r2_bi_microgate.py
PYARGS=(--batches 64,128 --U 3 --warm 5 --reps 30 --seed 7 --biattn "$ATTN_MOD" ${GEMM_ARGS[@]+"${GEMM_ARGS[@]}"})
SCAN=(--gemm-scan --ms 64,128,256,320,512,768,861,1024,1148 --exact-b 287,216,64 --U 3 --warm 5 --reps 30 --seed 7)
case $CELL in
    gm1) PYARGS=("${SCAN[@]}") ;;
    bigc-tune) PYFILE=bigc_tune.py PYARGS=(--warm 3 --reps 10 --seed 7) ;;
    bigc-gate) PYFILE=bigc_gate.py PYARGS=(--variants strict,bucket --seed 1234 ${TABLE_ARGS[@]+"${TABLE_ARGS[@]}"}) ;;
    bigc-time)
        PYARGS=("${SCAN[@]}" --bigemm-c both)
        [ ${#TABLE_ARGS[@]} = 0 ] || PYARGS+=(--bigc-table "$BIGC_TABLE" --bigc-table-sha "$BIGC_TABLE_SHA") ;;
esac
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
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/$PYFILE" --out "$OUT" "${PYARGS[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 BI step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 BI step: cell ${CELL:-default} STEP_MIN $STEP_MIN claim $CLAIM"
    echo "R2 BI step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; BI GEMM ${BIGEMM_SHA:-unset}${BIGEMM_SHA:+:$BIGEMM_PATH blob $BIGEMM_BLOB fn $BIGEMM_FN}"
    echo "R2 BI step: G-c table ${BIGC_TABLE:-source}${BIGC_TABLE_SHA:+ sha $BIGC_TABLE_SHA}"
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
mkdir "$OUTROOT/$CLAIM" 2>/dev/null || stop "$OUTROOT/$CLAIM exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id ($CLAIM stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT ($CLAIM stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIM stays)"
if [ -z "$CELL" ]; then                                  # gm1 and the bigc cells load neither piece 3 nor the BI GEMM
    git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
        stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD ($CLAIM stays)"
fi
if [ -n "$BIGEMM_SHA" ] && [ -z "$CELL" ]; then
    git -C "$NOSI" show "$BIGEMM_BLOB" > "${GEMM_ARGS[1]}" && [ "$(git hash-object -- "${GEMM_ARGS[1]}")" = "$BIGEMM_BLOB" ] ||
        stop "cannot copy the BI GEMM blob $BIGEMM_BLOB to ${GEMM_ARGS[1]} ($CLAIM stays)"
fi
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIM stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s claim=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "${CELL:-default}" "$CLAIM" "$(date -u +%FT%TZ)" \
    >> "$REC/command.txt" || stop "cannot write $REC ($CLAIM stays)"
printf 'biattn=%s:%s blob=%s copy=%s bigemm=%s:%s blob=%s fn=%s bigc_table=%s sha=%s\n' "$BIATTN_SHA" "$BIATTN_PATH" \
    "$BIATTN_BLOB" "$ATTN_MOD" "${BIGEMM_SHA:-unset}" "${BIGEMM_PATH:-}" "${BIGEMM_BLOB:-}" "${BIGEMM_FN:-}" \
    "${BIGC_TABLE:-source}" "${BIGC_TABLE_SHA:-}" >> "$REC/command.txt" || stop "cannot write $REC ($CLAIM stays)"
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
    rmdir "$OUTROOT/$CLAIM" || ok=0
fi
echo "R2 BI step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
