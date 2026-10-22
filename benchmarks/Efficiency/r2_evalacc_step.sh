#!/bin/bash
# [evalsplit] EVAL CELL (b): ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_evalacc.py:
# the ACCURACY EQUIVALENCE of FAST (natural cuBLAS) and EXACT (G-c strict, bi_c_strict) in the R2 path (human decision
# part 3, 2026-10-22; paper-notes/fast-eval-cells.md section 2). Prompts: pg19-test-c16k; gold continuation:
# pg19-test-ctx65280 (the same 55 documents' next tokens). Teacher-forced task metric (top-1 accuracy, NLL), the
# speculative round's acceptance and committed tokens, batch invariance of EXACT, and the planted negative control.
# The pattern of benchmarks/Efficiency/r2_tick_step.sh: the GPU uuid from HOLDERS/<holder job id>.gpu_uuid; piece 3
# pinned by commit, path and blob, copied into the step record and loaded by path. SOURCE ONLY until reviewed: it never
# submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_evalacc_step.sh --nosi <sha> <holder job> <run id> <evalacc|evalacc-mini>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_evalacc_step.sh --nosi <sha> <holder job> <run id> <evalacc|evalacc-mini>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, EACC_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses EACC_HOLDERS.
#   In DRYRUN only, EACC_CLAIM_PROBE=<dir>: after the uuid file checks, mkdir <dir>/<the claim's name> (the real
#        claim's mkdir, under <dir> and not OUTROOT), then exit 0 when it took the claim, 1 when the claim is held; no
#        other check or write (the CPU test uses it); a real run refuses EACC_CLAIM_PROBE.
#
# CLAIM ([sg5par], 2026-10-22): ONE CLAIM PER HOLDER, OUTROOT/eacc_active_<holder job id> (was OUTROOT/eacc_active).
#   Two steps in the same holder refuse each other; steps in different holders run at the same time. Every path that a
#   step writes holds the run id AND the holder job (OUT, REC, JIT with its caches and TMPDIR, the job name, the sacct
#   row); the step reads only the pinned workloads, the model and the image (read-only). Eval cell (a) reads the
#   summary.json of the SAME holder only.
#
# CELLS:
#   evalacc-mini  the inspected miniature (preflight rule, 2026-09-20): B 8, alt 4, 32 gold tokens, 16 SPEC tokens,
#                 the three plant doses; STEP_MIN 45. Run it first and read its summary before evalacc.
#   evalacc       B 48 (of the 55 matched documents), alt 8 (6 chunks), 256 gold tokens (255 x 48 = 12,240 scored
#                 positions), 64 SPEC tokens, K 3, LAMBDA 7, draft drop 0.5 seed 7, doses 2^-8, 2^-4, 2^-2;
#                 STEP_MIN 120. Eval cell (a) (r2_tick_sg5_step.sh evalfast, same run id and holder) reads its
#                 summary.json and refuses without a passing one.
# REGISTERED: r2_evalacc.py's docstring (2026-10-22T14:21:48Z): EA-P0, EA-E, EA-TF-AGREE [0.970, 0.998], EA-TF-MAXD
#   p50 [0.005, 0.30] p99 [0.05, 2.0], EA-TF-ACC E acc [0.35, 0.65] NLL [1.6, 3.0], EA-TF-D |d acc|, |d NLL| <= 0.003,
#   EA-NI, EA-FREP, EA-SPEC, EA-NEG (2^-2 AFFECTS with |d NLL| > 0.05; 2^-8 EQUIVALENT), EA-VERDICT FAST EQUIVALENT.
#   The band of "does not affect accuracy": |d| <= max(2 x FAST's own spread, 3 x se) + floor and |d| <= cap (acc and
#   NLL floor 0.001, cap 0.01; acceptance and gold match floor 0.002, cap 0.02). Wall (MODEL): evalacc 25-60 min.
# OUTPUT: OUTROOT/eacc_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/eacc_<run>_<holder>_<cell>.step/
# (command with the uuid, its file and the pins, the piece-3 copy, stdout with one GATE / REPORT line per gate,
# stderr, exit + accounting, sacct row), both frozen.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch; 5 an exit-0 step whose record is incomplete or whose sacct
# row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-sg5par                                                   # ROOT ([sg5par]: the per-holder claim)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (r2_g0_step.sh)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
GOLD=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280                        # ROOT
GOLD_SHA=f2fb36f441ec12b7a79239d5e86263c6454b7713436f53f81086ef2bedcc34b3                 # PIN (ctx65280_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/evalacc                                               # ROOT
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN (piece 3 v2)
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
SPLITS=4
SELF=benchmarks/Efficiency/r2_evalacc_step.sh
main() {
stop() { echo "R2 evalacc step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_evalacc_step.sh --nosi <sha> <holder> <run id> <evalacc|evalacc-mini>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
case $CELL in
    evalacc) BATCH=48 ALT=8 NGOLD=256 SPEC=64 STEP_MIN=120 SUFFIX=_evalacc ;;
    evalacc-mini) BATCH=8 ALT=4 NGOLD=32 SPEC=16 STEP_MIN=45 SUFFIX=_evalaccmini ;;
    *) usage ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${EACC_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "EACC_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$EACC_HOLDERS
fi
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] || stop "no GPU uuid file $UFILE (the lead writes it for each holder)"
GPU_UUID=$(head -n 1 "$UFILE")
[ "$(wc -l < "$UFILE")" -le 1 ] || stop "$UFILE has more than one line"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold a uuid (no GPU- prefix): '$GPU_UUID'"
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
CLAIMNAME=eacc_active_$JOB                                  # [sg5par] one claim per holder (header: CLAIM)
CLAIM=$OUTROOT/$CLAIMNAME
if [ -n "${EACC_CLAIM_PROBE:-}" ]; then                     # [sg5par] the CPU test of the claim's name and its mkdir
    [ "$DRY" = 1 ] || stop "EACC_CLAIM_PROBE is for DRYRUN (the CPU test) only"
    if mkdir "$EACC_CLAIM_PROBE/$CLAIMNAME" 2>/dev/null; then
        echo "R2 evalacc step: CLAIM_PROBE took $CLAIMNAME"
        exit 0
    fi
    echo "R2 evalacc step: CLAIM_PROBE $CLAIMNAME is held" >&2
    exit 1
fi
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_evalacc.py benchmarks/Efficiency/r2_g0bi_gate.py benchmarks/Efficiency/r2_g0_gate.py \
         nosi/nosi/r2_tick.py nosi/nosi/r2_bi_decode.py nosi/nosi/bi_linear_c.py nosi/nosi/r2_decode_path.py \
         nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py nosi/nosi/r2_spec_core.py \
         nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
gl=$(sha256sum "$GOLD/ctx65280_tokens.bin" 2>/dev/null) || stop "no $GOLD/ctx65280_tokens.bin"
[ "${gl%% *}" = "$GOLD_SHA" ] || stop "ctx65280_tokens.bin sha256 is not $GOLD_SHA"
OUT=$OUTROOT/eacc_${RUN}_$JOB$SUFFIX
REC=$OUT.step
JIT=$JITBASE/eacc_${RUN}_$JOB$SUFFIX
NAME=eacc-${RUN}-$JOB${SUFFIX//_/-}
ATTN_MOD=$REC/piece3_bi_paged_attention.py
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_R2_DECODE=1 --env NOSI_MODEL_PATH="$MODEL" --env EVAL_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 evalacc child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 evalacc child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=160G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" eacc-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_evalacc.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --gold-workload "$GOLD" --biattn "$ATTN_MOD" --exact-gemm bi_c_strict --batch "$BATCH"
      --alt-batch "$ALT" --S 16128 --n-gold "$NGOLD" --spec-steps "$SPEC" --K 3 --LAMBDA 7 --draft-drop 0.5 --seed 7
      --parts TF,SPEC,NEG)
if [ "$DRY" = 1 ]; then
    echo "R2 evalacc step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA gold $GOLD_SHA gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 evalacc step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; cell $CELL: batch $BATCH alt $ALT n_gold $NGOLD spec $SPEC; exact bi_c_strict; STEP_MIN $STEP_MIN; MIN_LEFT $MIN_LEFT s"
    echo "R2 evalacc step: claim $CLAIM (one per holder)"                                             # [sg5par]
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
mkdir "$CLAIM" 2>/dev/null || stop "$CLAIM exists (another step in holder $JOB, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id ($CLAIMNAME stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT ($CLAIMNAME stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIMNAME stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD ($CLAIMNAME stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIMNAME stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s gold=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$GOLD_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIMNAME stays)"
printf 'biattn=%s:%s blob=%s copy=%s cell=%s batch=%s alt=%s n_gold=%s spec=%s splits=%s claim=%s\n' "$BIATTN_SHA" \
    "$BIATTN_PATH" "$BIATTN_BLOB" "$ATTN_MOD" "$CELL" "$BATCH" "$ALT" "$NGOLD" "$SPEC" "$SPLITS" "$CLAIM" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIMNAME stays)"
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
echo "R2 evalacc step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
