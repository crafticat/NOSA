#!/bin/bash
# R2 MILESTONE 1 (route R2: the paper's stage 3 inside NOSA's batched engine; the K-step slice is the exact reference):
# the FIRST batched GPU numbers of the existing speculative-loop pilot (benchmarks/Efficiency/spec_loop_pilot.py,
# schedule round:K-draft-then-verify), EXPLICITLY LOCKSTEP: every round commits c = 1 + min_b n_acc(b) for the whole
# batch (spec_loop.accept_commit); the per-request n_acc is recorded per round and the ragged yield is a PROJECTION.
# ONE bounded GPU step per invocation in an EXISTING holder (the lead's `srun --jobid=<holder>` pattern of
# retroinfer-kstep scripts/kstep_k3x_step.sh; the environment of retroinfer-eval env/slurm/nosi_pool.sbatch). SOURCE
# ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/spec_loop_r2_step.sh --nosi <sha> <holder job> <run id> <cell>
#   DRYRUN=1 bash benchmarks/Efficiency/spec_loop_r2_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node, CPU only: the pin checks and the printed step command; no squeue, claim, srun or write)
#
# CELLS (L = 16128, PG-19 test parquet as test_nosa_pg19.py; greedy; pool OFF: NOSI_POOL_BLOCKS = 0 in every cell,
# the engine refuses round slots and the pool together, cache_engine.py:223; NOSI_ATTN_SPLITS = 4 PINNED in every cell:
# the split count of the decode attention, so every arm has the same fp32 combine order):
#   ref:<B>:<D>    NOSI_SPEC_MODE decode_ref, NOSI_VERIFY_ROUND_SLOTS 0: NOSA's SHIPPED decode at batch B (the arm
#                  "shipped_ref"), 112 tokens; its sequence is gated against the spec process's shipped arm.
#   spec:<B>:<D>   NOSI_SPEC_MODE spec, NOSI_VERIFY_ROUND_SLOTS 62 (W = 128 slots: the registered verifier setting,
#                  jobs 2175382 / 2175441): the in-process shipped arm (64-slot views, gate T0) and, per K in K_LIST,
#                  the speculative loop until every request committed 112 tokens (re-warm at each 64-token block
#                  boundary inside the clock, both arms).
#   probe:<B>:<D>  spec with NOSI_SPEC_PROBE=1 (round 3) and K 2 only: THE ROLLBACK PROBE of the pilot's docstring
#                  (the post-round engine state torch.equal to a fresh replay of the c committed tokens from the
#                  round-start snapshot, with the positive control: a replay of another length differs).
#   report         NOSI_SPEC_MODE report over the run's .pt files (exit = failed gates): the SEQUENCE IDENTITY gate
#                  (every spec cell and every shipped_ref arm torch.equal to the shipped arm of the same (L, batch,
#                  distinct, documents) over the compared tokens; a mismatch is classified by the shipped top-2 margin
#                  against 4 x tau_ctrl, tau_ctrl = 0.4844 at 16128, job 2175374), the rollback probe, overflow 0.
#   B is 64 or 128. D is 2 ((a) two documents repeated over the batch) or B ((b) B distinct documents). The PG-19 test
#   parquet has 100 books, so (b) at B = 128 cycles over the qualifying books (at most 100 distinct); the payload
#   records distinct_docs and the report keys on it.
# REGISTERED ORDER (the lead runs the cells one invocation at a time; a STOP rule is binding):
#   gates   ref:64:2, spec:64:2, probe:64:2, report  -> every gate PASS, else milestone 1 STOPS (no timing is reported)
#   timing  ref:64:64, spec:64:64, ref:128:2, spec:128:2, ref:128:128, spec:128:128, report
# OUTPUTS (per run: OUTROOT/r2m1_<run>_<holder>/): decode_ref_<tag>.pt / spec_<tag>.pt (frozen after their step): the
# arms' committed sequences, per-round records (rounds[r].n_acc = the per-request ACCEPTED LENGTHS of round r, c, the
# re-warm rounds, the timing brackets), the committed tok/s (after warm-up and over the window after 64 tokens), the
# round count (agg.spec_rounds), the peak HBM (peak_gb / reserved_gb) and host pinned GB; spec_loop.md / .json (the
# report; copied into the report step's record). Per step: OUTROOT/r2m1_<run>_<holder>.<cell>.<n>.step/ (command,
# stdout, stderr, exit with the accounting line, sacct row incl. MaxRSS), frozen.
# Checks before the claim (each STOPS with exit 1, nothing written, nothing launched): no SLURM_ / SRUN_ / SALLOC_ /
# SBATCH_ variable in the login shell; the NOSI checkout at --nosi <sha> with no tracked change and no untracked file
# under nosi/; this file is the blob benchmarks/Efficiency/spec_loop_r2_step.sh of <sha>; the model and parquet exist
# (model size pinned); the holder RUNNING, ours, at least MIN_LEFT seconds left, no step besides batch / extern. The
# claim: OUTROOT/r2m1_active (mkdir, atomic: one R2 step at a time); a stop after the claim leaves it (the lead
# decides). Exit: the step's exit (0; the report's failed-gate count; 2 a pilot refusal; 3 another GPU; 4 not exactly
# one GPU; 124 / 137 the host timeout); 1 a STOP before the launch; 5 an exit-0 step whose record is incomplete or whose
# sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2                                                          # ROOT
GPU_UUID=f31fc723-9dfe-407a-9947-28173556232a                                            # PIN (holder 2188818)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
PQ=/mnt/beegfs/ojerbi/datasets/pg19/test-00000-of-00001-29a571947c0b5ccc.parquet         # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/m1                                                    # ROOT
SELF=benchmarks/Efficiency/spec_loop_r2_step.sh
L=16128 TOTAL=112 GATE=64 R_SLOTS=62 SPLITS=4 TAU=0.4844 K_LIST="1 2 3"
MIN_LEFT=7200
main() {
stop() { echo "R2 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: spec_loop_r2_step.sh --nosi <sha> <holder> <run id> <ref:B:D|spec:B:D|probe:B:D|report>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
MODE= B=0 D=0 R=0 PROBE=0 KL=$K_LIST
case $CELL in
    report) MODE=report STEP_MIN=15 TAG=report ;;
    ref:*:*|spec:*:*|probe:*:*)
        kind=${CELL%%:*} rest=${CELL#*:}
        B=${rest%%:*} D=${rest#*:}
        case $B in 64|128) ;; *) stop "batch '$B' (64 or 128)" ;; esac
        [ "$D" = 2 ] || [ "$D" = "$B" ] || stop "distinct '$D' (2 or the batch $B)"
        TAG=${kind}b${B}d${D}
        case $kind in
            ref) MODE=decode_ref R=0 STEP_MIN=40 ;;
            spec) MODE=spec R=$R_SLOTS STEP_MIN=90 ;;
            probe) MODE=spec R=$R_SLOTS PROBE=1 KL=2 STEP_MIN=60 ;;
        esac ;;
    *) stop "cell '$CELL'" ;;
esac
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
[ -f "$NOSI/benchmarks/Efficiency/spec_loop_pilot.py" ] || stop "no pilot in $NOSI"
[ -f "$SIF" ] || stop "no image $SIF"
[ -f "$PQ" ] || stop "no parquet $PQ"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
OUT=$OUTROOT/r2m1_${RUN}_$JOB
JIT=$JITBASE/r2m1_${RUN}_$JOB
NAME=r2m1-${TAG}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
      --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
      --env TRITON_CACHE_DIR="$JIT/triton" --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext"
      --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer" --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_VERIFY_ROUND_SLOTS=$R --env NOSI_MODEL_PATH="$MODEL" --env NOSI_PG19_PARQUET="$PQ"
      --env NOSI_SPEC_OUT="$OUT" --env NOSI_SPEC_MODE=$MODE --env NOSI_SPEC_TAG=$TAG --env NOSI_SPEC_L=$L
      --env NOSI_SPEC_DOCS=$B --env NOSI_SPEC_DISTINCT=$D --env NOSI_SPEC_GATE_TOKENS=$GATE
      --env NOSI_SPEC_TOTAL_TOKENS=$TOTAL --env NOSI_SPEC_K_LIST="$KL" --env NOSI_SPEC_PROBE=$PROBE
      --env NOSI_SPEC_PROBE_ROUND=3 --env NOSI_SPEC_TAU_CTRL=$TAU)
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=240G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" r2-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/spec_loop_pilot.py")
if [ "$DRY" = 1 ]; then
    echo "R2 step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob"
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
mkdir "$OUTROOT/r2m1_active" 2>/dev/null || stop "$OUTROOT/r2m1_active exists (another R2 step, or an uncertain one)"
n=1
until mkdir "$OUT.$TAG.$n.step" 2>/dev/null; do
    n=$((n + 1))
    [ "$n" -le 100 ] || stop "no free attempt number below 100 (r2m1_active stays)"
done
REC=$OUT.$TAG.$n.step
mkdir -p "$OUT" "$JIT" || stop "cannot create $OUT / $JIT (r2m1_active stays)"
if [ "$MODE" != report ]; then
    for f in "$OUT/${MODE}_$TAG.pt"; do [ ! -e "$f" ] || stop "$f exists: one payload per cell and run (r2m1_active stays)"; done
fi
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (r2m1_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (r2m1_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s holder=%s left=%s cell=%s run=%s utc_start=%s\n' "$NSHA" "$blob" \
    "$GPU_UUID" "$JOB" "$sl" "$CELL" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (r2m1_active stays)"
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
if [ "$MODE" = report ]; then
    for f in spec_loop.md spec_loop.json; do [ ! -f "$OUT/$f" ] || cp -p "$OUT/$f" "$REC/" || ok=0; done
elif [ -f "$OUT/${MODE}_$TAG.pt" ]; then
    chmod 0444 "$OUT/${MODE}_$TAG.pt" || ok=0
fi
find "$REC" -type f -exec chmod 0444 {} + && chmod 0555 "$REC" || ok=0
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/r2m1_active" || ok=0
fi
echo "R2 step: cell $CELL exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
