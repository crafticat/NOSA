#!/bin/bash
# R2 FIA3 (docs/NOTES_fia3.md; plan kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md Task 0.3): ONE bounded GPU
# step in an EXISTING holder that runs benchmarks/Efficiency/r2_fiattn_bench.py --mode main with the fia3 arms:
# piece 3 v2 (P), FlashInfer decode (F8), FlashInfer paged PREFILL over the request union with a packed custom mask
# (FP0 no split, FP8 fixed split 8, FPU the same call without the mask: timing only), NOSA's own prefill varlen route
# over a contiguous union store (V1: EQ verify_step.py, pinned), cells 32x4 and 216x4 from the seeded generator and
# from the recorded S1 tick tables (32x4s1, 216x4s1), then the ncu passes (REPORT) inside the same budget.
# The pattern of benchmarks/Efficiency/r2_fiattn_step.sh (fia1 / fia2; unchanged): the GPU uuid from the lead-owned
# file HOLDERS/<holder>.gpu_uuid; modules of OTHER branches (piece 3 v2, EQ verify_step) pinned by commit, path and blob
# id, copied from the shared object store into the step record (git show <blob>), checked by git hash-object and loaded
# by path; the S1 tables pinned by path and size, copied into the record (their git blob id is RECORDED, not pinned);
# the claim fiattn_active (shared with r2_fiattn_step.sh: one fiattn-family step at a time; existence check first, an
# owner file, a kept claim on a client timeout); STEP_MIN 45; the record frozen.
# SOURCE ONLY, NOT RUN, NOT REVIEWED: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_fia3_step.sh --nosi <sha> <holder job> <run id>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_fia3_step.sh --nosi <sha> <holder job> <run id>
#        (the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, FIATTN_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses FIATTN_HOLDERS.
#
# OUTPUT: OUTROOT/fia3_<run>_<holder>/ (summary.json, jit_*_error.txt if any, ncu_* files) and the step record
# OUTROOT/fia3_<run>_<holder>.step/ (command with the uuid and its file, the pinned module copies, the tables copy,
# stdout with one GATE / REPORT line per gate, stderr, exit + accounting, sacct row), both frozen. JIT caches
# (FlashInfer, tvm-ffi, Triton, torch extensions, TMPDIR) under JITBASE/fia3_<run>_<holder>/ (lessons C1: no HOME cache).
# EXIT: the bench's failed MUST count; 2 refusal; 3 / 4 GPU identity; 124 / 137 the host timeout; 1 a STOP before the
# launch; 5 an exit-0 step with an incomplete record or a non-COMPLETED 0:0 sacct row.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-fia3                                                     # ROOT (this branch)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT (FlashInfer as installed:
                                                                                         # every JIT cache says 0.5.3)
APPTAINER=/usr/bin/apptainer                                                             # ROOT
CONFIG=/mnt/beegfs/ojerbi/models/NOSA-8B/config.json                                     # ROOT (the geometry)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/fiattn                                                # ROOT
NCU=/opt/nvidia/nsight-compute/2025.1.0/ncu                                              # PIN (inside the image)
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN (piece 3 v2)
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
VSTEP_SHA=dde91d4cfe345810c57450047e81e0be1524f38d                                       # PIN (EQ nosi-eqmem, V1)
VSTEP_PATH=nosi/nosi/verify/verify_step.py                                               # PIN
VSTEP_BLOB=769047f602e81c4f2792888210ac4a65f8607792                                      # PIN
S1_TABLES=/mnt/beegfs/ojerbi/r2-runs/sharedkv/skv_probe_s1a_2191262/skv_tables.pt         # PIN (path; frozen record)
S1_BYTES=439458                                                                          # PIN (size)
CELLS=32x4,216x4,32x4s1,216x4s1                                                          # 32x4 first: fia2 continuity
ARMS=P,F8,FP0,FP8,FPU,V1
SEED=20261021
BUDGET_S=2400
SELF=benchmarks/Efficiency/r2_fia3_step.sh
STEP_MIN=45
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 fia3 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_fia3_step.sh --nosi <sha> <holder> <run id>" >&2; exit 2; }
[ $# = 4 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${FIATTN_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "FIATTN_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$FIATTN_HOLDERS
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
BENCH_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:benchmarks/Efficiency/r2_fiattn_bench.py") ||
    stop "benchmarks/Efficiency/r2_fiattn_bench.py is not tracked at $NSHA"
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$VSTEP_SHA:$VSTEP_PATH")" = "$VSTEP_BLOB" ] ||
    stop "V1: $VSTEP_PATH at $VSTEP_SHA is not the blob $VSTEP_BLOB"
[ -f "$S1_TABLES" ] || stop "no S1 tables $S1_TABLES"
[ "$(wc -c < "$S1_TABLES")" = "$S1_BYTES" ] || stop "$S1_TABLES is not $S1_BYTES bytes"
[ -f "$SIF" ] || stop "no image $SIF"
[ -f "$CONFIG" ] || stop "no model config $CONFIG"
OUT=$OUTROOT/fia3_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/fia3_${RUN}_$JOB
NAME=fia3-${RUN}-$JOB
ATTN_MOD=$REC/piece3_bi_paged_attention.py
VSTEP_MOD=$REC/eq_verify_step.py
S1_COPY=$REC/skv_tables.pt
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env FLASHINFER_CUDA_ARCH_LIST=8.0 --env MAX_JOBS=16
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp" --env FIATTN_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 fia3 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 fia3 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" fia3-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_fiattn_bench.py" --mode main --out "$OUT"
      --config "$CONFIG" --biattn "$ATTN_MOD" --verify-step "$VSTEP_MOD" --s1-tables "$S1_COPY" --cells "$CELLS"
      --arms "$ARMS" --no-strided --seed "$SEED" --warm 20 --reps 200 --ncu "$NCU" --ncu-cap 600
      --budget-s "$BUDGET_S")
if [ "$DRY" = 1 ]; then
    echo "R2 fia3 step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob bench $BENCH_BLOB gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 fia3 step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; V1 $VSTEP_SHA:$VSTEP_PATH blob $VSTEP_BLOB -> $VSTEP_MOD; S1 tables $S1_TABLES ($S1_BYTES B) -> $S1_COPY; cells $CELLS; arms $ARMS; budget $BUDGET_S s in $STEP_MIN min"
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
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] ||                # BEFORE the claim
    stop "$OUT, $REC or $JIT exists: a new run id (no claim taken)"
mkdir "$OUTROOT/fiattn_active" 2>/dev/null ||
    stop "$OUTROOT/fiattn_active exists (another step, or an uncertain one; its owner file says which)"
printf 'run=%s holder=%s step_name=%s nosi=%s out=%s utc_claim=%s\n' "$RUN" "$JOB" "$NAME" "$NSHA" "$OUT" \
    "$(date -u +%FT%TZ)" > "$OUTROOT/fiattn_active/owner" ||                  # the claim's owner
    stop "cannot write $OUTROOT/fiattn_active/owner (fiattn_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (fiattn_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (fiattn_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (fiattn_active stays)"
git -C "$NOSI" show "$VSTEP_BLOB" > "$VSTEP_MOD" && [ "$(git hash-object -- "$VSTEP_MOD")" = "$VSTEP_BLOB" ] ||
    stop "cannot copy V1's blob $VSTEP_BLOB to $VSTEP_MOD (fiattn_active stays)"
cp -- "$S1_TABLES" "$S1_COPY" && [ "$(wc -c < "$S1_COPY")" = "$S1_BYTES" ] && cmp -s -- "$S1_TABLES" "$S1_COPY" ||
    stop "cannot copy the S1 tables to $S1_COPY (fiattn_active stays)"
S1_BLOB=$(git hash-object -- "$S1_COPY") || stop "cannot record the S1 tables' blob id (fiattn_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (fiattn_active stays)"
printf '\nnosi=%s wrapper_blob=%s bench_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$BENCH_BLOB" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (fiattn_active stays)"
printf 'biattn=%s:%s blob=%s copy=%s verify_step=%s:%s blob=%s copy=%s s1_tables=%s bytes=%s blob_recorded=%s copy=%s cells=%s arms=%s seed=%s budget_s=%s ncu=%s\n' \
    "$BIATTN_SHA" "$BIATTN_PATH" "$BIATTN_BLOB" "$ATTN_MOD" "$VSTEP_SHA" "$VSTEP_PATH" "$VSTEP_BLOB" "$VSTEP_MOD" \
    "$S1_TABLES" "$S1_BYTES" "$S1_BLOB" "$S1_COPY" "$CELLS" "$ARMS" "$SEED" "$BUDGET_S" "$NCU" >> "$REC/command.txt" ||
    stop "cannot write $REC (fiattn_active stays)"
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
    124|137)                                                  # a timeout (C10): the client ended, the step may
        ok=0                                                  # still write. Wait (bounded) until no step of the
        writer=running                                        # holder is listed; the claim STAYS
        for _ in $(seq 1 60); do
            if nosteps > /dev/null; then writer=gone_from_squeue; break; fi
            sleep 10
        done
        echo "timeout: writer=$writer after the wait; fiattn_active stays (L40 cleanup after sacct)" >> "$REC/exit.txt" ;;
esac
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    [ "$t" = "$OUT" ] && [ "$writer" = running ] && continue       # never touch an output that may still be written
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rm -f "$OUTROOT/fiattn_active/owner" && rmdir "$OUTROOT/fiattn_active" || ok=0
fi
echo "R2 fia3 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
