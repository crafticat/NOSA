#!/bin/bash
# R2 MILESTONE 2, STEP 7, G0-BI: ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/
# r2_g0bi_gate.py: the batch-invariant serial decode (nosi/nosi/r2_bi_decode.py, a mode of the R2 path; G-b
# PaddedLinear, piece 3 v2's paged attention, the deterministic selection, the fixed-order twin means). MUST: E2
# determinism, I1 request 0 alone, I2 the batch reversed (torch.equal at every one of 256 steps), the prefill / first
# token controls. REPORT: the quality against the shipped decode (part C: 256 continuous steps against the R2
# shipped-order path; part W: 4 windows x 64 steps against the shipped decode itself), the free-running first
# divergence, the natural-GEMM negative control NI, ms per step. B = 8 documents of pg19-test-c16k, S = 16128.
# The pattern of benchmarks/Efficiency/r2_bi_step.sh: the GPU uuid comes from the lead-owned file
# HOLDERS/<holder job id>.gpu_uuid; modules of other branches (piece 3; the BI GEMM when pinned) are pinned by commit,
# path and blob, copied into the step record and loaded by path.
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_g0bi_step.sh --nosi <sha> <holder job> <run id>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_g0bi_step.sh --nosi <sha> <holder job> <run id>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, G0BI_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses G0BI_HOLDERS.
#
# OUTPUT: OUTROOT/g0bi_<run>_<holder>/summary.json + differences.json, and the step record
# OUTROOT/g0bi_<run>_<holder>.step/ (command with the uuid, its file and the pins, the piece-3 copy, stdout with one
# GATE / REPORT line per gate, stderr, exit + accounting, sacct row), both frozen.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed uuid file included); 5 an exit-0 step
# whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-g0bi                                                  # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (r2_g0_step.sh)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/g0bi                                                  # ROOT
# piece 3 v2 (nosi-r2-m2-biattn 7c826c1: fixed 8-page split-KV, GQA tile; GPU gate PASS, 2188818): the BI paged
# attention, loaded by path
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
# the BI GEMM (nosi-r2-m2-bigemm 26c45bd, option G-a; GPU gate PASS) is NOT used: G-b (PaddedLinear(M_FIX), M_FIX =
# 4B) is cheaper for every linear at the tick's row mix (docs/R2_STEP7_SPEC_ROUND.md 3.3). Empty pins = G-b.
BIGEMM_SHA=                                                                              # PIN (unset)
BIGEMM_PATH=                                                                             # PIN (unset)
BIGEMM_BLOB=                                                                             # PIN (unset)
BIGEMM_FN=                                                                               # PIN (unset)
M_FIX=32
SPLITS=4
SELF=benchmarks/Efficiency/r2_g0bi_step.sh
STEP_MIN=60
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 G0-BI step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_g0bi_step.sh --nosi <sha> <holder> <run id>" >&2; exit 2; }
[ $# = 4 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${G0BI_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "G0BI_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$G0BI_HOLDERS
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
for f in benchmarks/Efficiency/r2_g0bi_gate.py benchmarks/Efficiency/r2_g0_gate.py benchmarks/Efficiency/r2_g0_compare.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py \
         nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py; do
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
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
OUT=$OUTROOT/g0bi_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/g0bi_${RUN}_$JOB
NAME=g0bi-${RUN}-$JOB
ATTN_MOD=$REC/piece3_bi_paged_attention.py
GEMM=exact
[ -z "$BIGEMM_SHA" ] || { GEMM=bigemm; GEMM_ARGS=(--bigemm "$REC/bigemm_$(basename "$BIGEMM_PATH")" --bigemm-fn "$BIGEMM_FN"); }
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_R2_DECODE=1 --env NOSI_MODEL_PATH="$MODEL" --env G0BI_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 G0-BI child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 G0-BI child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=120G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" g0bi-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_g0bi_gate.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --biattn "$ATTN_MOD" --gemm "$GEMM" --batch 8 --S 16128 --steps-c 256 --windows 4 --steps-w 64
      --ni-steps 8 --m-fix "$M_FIX" --parts C,NI,W ${GEMM_ARGS[@]+"${GEMM_ARGS[@]}"})
if [ "$DRY" = 1 ]; then
    echo "R2 G0-BI step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 G0-BI step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; gemm $GEMM; BI GEMM ${BIGEMM_SHA:-unset}${BIGEMM_SHA:+:$BIGEMM_PATH blob $BIGEMM_BLOB fn $BIGEMM_FN}; M_FIX $M_FIX"
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
mkdir "$OUTROOT/g0bi_active" 2>/dev/null || stop "$OUTROOT/g0bi_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (g0bi_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (g0bi_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (g0bi_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (g0bi_active stays)"
if [ -n "$BIGEMM_SHA" ]; then
    git -C "$NOSI" show "$BIGEMM_BLOB" > "${GEMM_ARGS[1]}" && [ "$(git hash-object -- "${GEMM_ARGS[1]}")" = "$BIGEMM_BLOB" ] ||
        stop "cannot copy the BI GEMM blob $BIGEMM_BLOB to ${GEMM_ARGS[1]} (g0bi_active stays)"
fi
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (g0bi_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (g0bi_active stays)"
printf 'biattn=%s:%s blob=%s copy=%s gemm=%s bigemm=%s:%s blob=%s fn=%s m_fix=%s splits=%s\n' "$BIATTN_SHA" \
    "$BIATTN_PATH" "$BIATTN_BLOB" "$ATTN_MOD" "$GEMM" "${BIGEMM_SHA:-unset}" "${BIGEMM_PATH:-}" "${BIGEMM_BLOB:-}" \
    "${BIGEMM_FN:-}" "$M_FIX" "$SPLITS" >> "$REC/command.txt" || stop "cannot write $REC (g0bi_active stays)"
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
    rmdir "$OUTROOT/g0bi_active" || ok=0
fi
echo "R2 G0-BI step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
