#!/bin/bash
# CONTEXT SWEEP (paper 4.2): ONE bounded GPU step per invocation in an EXISTING holder (the pattern of eqmem_step.sh).
# One step = one cell = one process of benchmarks/Efficiency/ctxsweep_bench.py (ctxsweep_core.py has the rules;
# eqmem_bench.run_arm runs the arm). SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/ctxsweep_step.sh --nosi <sha> <holder> <run id> <budget GB> <ctx> <cell> <batch|max> <mem GiB> [<rep>]
#   bash benchmarks/Efficiency/ctxsweep_step.sh --nosi <sha> <holder> <run id> <budget GB> report
#   DRYRUN=1 bash ...   (login node: the argument, GPU-uuid-file and pin checks and the printed step command; no squeue,
#                       claim, srun or write)
# CELLS (N = 512 generated tokens per request, margin 1 GB, NOSI_ATTN_SPLITS = 0 and PYTORCH_CUDA_ALLOC_CONF =
# expandable_segments:True in EVERY cell; ctxsweep_cells.txt is the chain's list):
#   ctx     16128 | 65280 | 130816 (the workload files of that context, sha256 pinned below)
#   cell    original | cached:<P> (P 1 .. 160) | spec:<K> (K 1 .. 8; NOSI_VERIFY_ROUND_SLOTS 62, pool 0, LOCKSTEP)
#   batch   an explicit B (1 .. 4096, at most the int32 envelope of the context) or max (the largest B that
#           ctxsweep_core.plan_ctx_cell admits; the arm refits from the fit_row.json files already in the run directory
#           and, at budget 75, in the earlier runs PRIOR_RUNS: run cs1)
#   mem     the srun --mem in GiB, 64 .. 440 (the holder has 480G; the pinned host KV is rounded up to powers of two)
#   report  ctxsweep_bench.py --mode report over the run directory (and, at budget 75, PRIOR_RUNS and the 4.1 run
#           directory EQ41)
# GPU PIN: GPU_UUID is read from the lead-owned file HOLDERS/<holder>.gpu_uuid (one line, the uuid without "GPU-").
# A missing, unreadable or malformed file STOPS (exit 1) before any claim. The value and the file's path go into
# command.txt; the child checks the visible GPU against it (exit 3). DRYRUN only: CTXSWEEP_DRYRUN_HOLDERS=<dir> reads the
# file from <dir> (the CPU test of the refusals, test_ctxsweep.py CS9).
# OUTPUTS: OUTROOT/ctxsweep_<run>_<holder>_b<budget>/s<ctx>_<cell name>_b<batch>_rep<rep>/ (result.json, tokens.pt,
# fit_row.json, SHA256SUMS; frozen after the step) and <same>.<n>.step/ (command, stdout, stderr, exit with the
# accounting line, sacct row with MaxRSS; frozen); the report: report_<n>/ (report.json, report.md).
# Checks before the claim (each STOPS with exit 1, nothing written, nothing launched): the arguments; no SLURM_ / SRUN_ /
# SALLOC_ / SBATCH_ variable in the login shell; the GPU uuid file; the NOSI checkout at --nosi <sha>, no tracked
# change, no untracked file under nosi/ or benchmarks/Efficiency/; this file is the blob of <sha>; the image, the model
# (size pinned) and the context's workload (sha256 of the token file and the manifest pinned) exist; the holder RUNNING,
# ours, at least MIN_LEFT seconds left (the step's limit + 15 min), no step besides batch / extern. The step's limit:
# 8 min + 1.5 x the prefill at 0.21 ms per prompt token (of the explicit B, or the int32 bound for max) + 8 min of decode
# (25 for spec). The claim: OUTROOT/ctxsweep_active (mkdir, atomic: one step at a time); a stop after the claim leaves it
# (the lead decides). Exit: the step's exit (ctxsweep_bench: 0 VALID, 1 error or OVER_BUDGET, 2 out exists, 3 another
# GPU, 4 refused, 6 module origin; 124 / 137 the host timeout); 1 a STOP before the launch; 2 a usage error; 5 an exit-0
# step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-ctxsweep-v2                                                 # ROOT (feature/nosi-ctxsweep-v2)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned, mode 444)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WLROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload                                        # ROOT
TOK_SHA_16128=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce           # PIN (eqmem_step.sh; 94 x 16128)
MAN_SHA_16128=d1c072f032714b5adf3283de34c0b56f431fa406d2e671dcdcf96e4a4d0febd2           # PIN
TOK_SHA_65280=f2fb36f441ec12b7a79239d5e86263c6454b7713436f53f81086ef2bedcc34b3           # PIN (lead 2026-10-20; 55 x 65280)
MAN_SHA_65280=d60b7aaeab1bd4506a6706930fa481e946d71fbb5a245628f939895474b913c2           # PIN
TOK_SHA_130816=ff5df555025599d59d0c5c586c39ec7ed112cede78175b565387c0dde185a5d2          # PIN (lead 2026-10-20; 24 x 130816)
MAN_SHA_130816=6f64ce9aa9d003ef05c8491cd6753ae98d84baeba971090ab766885a926a7bee          # PIN
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/ctxsweep-runs                                                 # ROOT
EQ41=/mnt/beegfs/ojerbi/eqmem-runs/eqmem_h41_2188818_b75                                 # ROOT (4.1: the 16K max-batch cells)
PRIOR_RUNS="/mnt/beegfs/ojerbi/ctxsweep-runs/ctxsweep_cs1_2188818_b75"                   # ROOT (4.2 runs at budget 75 before this one)
SELF=benchmarks/Efficiency/ctxsweep_step.sh
N=512 MARGIN=1.0 SPLITS=0 R_SLOTS=62 MEM_MAX=440
main() {
stop() { echo "ctxsweep step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: ctxsweep_step.sh --nosi <sha> <holder> <run id> <budget GB> (<ctx> <cell> <batch|max> <mem GiB> [<rep>] | report)" >&2
          exit 2; }
[ $# -ge 2 ] && [ "$1" = --nosi ] || usage
NSHA=$2
shift 2
case $# in
    4) [ "$4" = report ] || usage
       JOB=$1 RUN=$2 BUDGET=$3 MODE=report ;;
    7|8) JOB=$1 RUN=$2 BUDGET=$3 CTX=$4 CELL=$5 BATCH=$6 MEM=$7 REP=${8:-0} MODE=arm ;;
    *) usage ;;
esac
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
[[ $BUDGET =~ ^[0-9]{2}(\.[0-9])?$ ]] || stop "budget '$BUDGET' (GB, e.g. 75 or 72.5)"
P=0 R=0 K=0 FAM=report
if [ "$MODE" = arm ]; then
    case $CTX in
        16128) WL=$WLROOT/pg19-test-c16k STEMF=c16k TOK_SHA=$TOK_SHA_16128 MAN_SHA=$MAN_SHA_16128 ;;
        65280) WL=$WLROOT/pg19-test-ctx65280 STEMF=ctx65280 TOK_SHA=$TOK_SHA_65280 MAN_SHA=$MAN_SHA_65280 ;;
        130816) WL=$WLROOT/pg19-test-ctx130816 STEMF=ctx130816 TOK_SHA=$TOK_SHA_130816 MAN_SHA=$MAN_SHA_130816 ;;
        *) stop "ctx '$CTX' (16128, 65280 or 130816)" ;;
    esac
    case $CELL in
        original) FAM=original NAME_CELL=original ;;
        cached:*) P=${CELL#cached:}
                  case $P in *[!0-9]*|""|0*) stop "pool '$P'" ;; esac
                  [ "$P" -ge 1 ] && [ "$P" -le 160 ] || stop "pool $P (1 .. 160)"
                  FAM=cached NAME_CELL=cached_p$P ;;
        spec:*) K=${CELL#spec:}
                case $K in [1-8]) ;; *) stop "K '$K' (1 .. 8)" ;; esac
                R=$R_SLOTS FAM=spec NAME_CELL=spec_k${K}_r$R ;;
        *) stop "cell '$CELL'" ;;
    esac
    MG=$(( (N + K + 64 + 63) / 64 * 64 ))
    BINT=$(( 2147483647 / ((CTX + MG) * 256) ))
    case $BATCH in
        max) BUP=$BINT ;;
        *[!0-9]*|""|0*) stop "batch '$BATCH' (1 .. 4096 or max)" ;;
        *) [ "$BATCH" -le 4096 ] && [ "$BATCH" -le "$BINT" ] \
               || stop "batch $BATCH (1 .. 4096, at most the int32 envelope $BINT at ctx $CTX)"
           BUP=$BATCH ;;
    esac
    case $MEM in *[!0-9]*|""|0*) stop "mem '$MEM' (GiB)" ;; esac
    [ "$MEM" -ge 64 ] && [ "$MEM" -le "$MEM_MAX" ] || stop "mem ${MEM}G (64 .. $MEM_MAX GiB; the holder has 480G)"
    case $REP in [0-9]) ;; *) stop "rep '$REP' (0-9)" ;; esac
    DEC=8
    [ "$FAM" = spec ] && DEC=25
    STEP_MIN=$(( 8 + BUP * CTX * 21 / 4000000 + DEC ))
    STEP_MEM=${MEM}G
else
    STEP_MIN=15 STEP_MEM=64G NAME_CELL=report
fi
MIN_LEFT=$(( STEP_MIN * 60 + 900 ))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
HDIR=$HOLDERS
if [ "$DRY" = 1 ] && [ -n "${CTXSWEEP_DRYRUN_HOLDERS:-}" ]; then HDIR=$CTXSWEEP_DRYRUN_HOLDERS; fi
UUID_FILE=$HDIR/$JOB.gpu_uuid
[ -f "$UUID_FILE" ] || stop "no GPU uuid file $UUID_FILE (the lead writes <holder>.gpu_uuid after an nvidia-smi check in the holder)"
[ -r "$UUID_FILE" ] || stop "the GPU uuid file $UUID_FILE is not readable"
GPU_UUID=$(cat -- "$UUID_FILE" 2>/dev/null) || stop "cannot read the GPU uuid file $UUID_FILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
    || stop "the GPU uuid file $UUID_FILE does not hold exactly one uuid line without GPU- (8-4-4-4-12 lowercase hex)"
[ "$(git -C "$NOSI" rev-parse HEAD 2>/dev/null)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi benchmarks/Efficiency)" ] \
    || stop "$NOSI has untracked files in nosi/ or benchmarks/Efficiency/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
if [ "$MODE" = arm ]; then
    [ "$(sha256sum < "$WL/${STEMF}_tokens.bin" | cut -d' ' -f1)" = "$TOK_SHA" ] || stop "the $CTX token file is not $TOK_SHA"
    [ "$(sha256sum < "$WL/${STEMF}_manifest.json" | cut -d' ' -f1)" = "$MAN_SHA" ] || stop "the $CTX manifest is not $MAN_SHA"
fi
RUNDIR=$OUTROOT/ctxsweep_${RUN}_${JOB}_b$BUDGET
JIT=$JITBASE/ctxsweep_${RUN}_$JOB
PRIOR=()                  # budget 75: the earlier 4.2 runs (their measured rows refit this run's plans; the report reads them)
if [ "$BUDGET" = 75 ]; then
    for d in $PRIOR_RUNS; do [ -d "$d" ] && [ "$d" != "$RUNDIR" ] && PRIOR+=("$d"); done
fi
if [ "$MODE" = report ]; then
    STEM=report
    NAME=ctxs-report-b$BUDGET-$RUN-$JOB
    DRV=(--mode report --results "$RUNDIR")
    for d in "${PRIOR[@]}"; do DRV+=(--results "$d"); done
    [ "$BUDGET" = 75 ] && [ -d "$EQ41" ] && DRV+=(--results "$EQ41")
else
    STEM=s${CTX}_${NAME_CELL}_b${BATCH}_rep$REP
    NAME=ctxs-s$CTX-$NAME_CELL-b$BATCH-r$REP-b$BUDGET-$RUN-$JOB
    DRV=(--mode arm --ctx "$CTX" --cell "$CELL" --batch "$BATCH" --mem-gib "$MEM" --budget-gb "$BUDGET" --n-new "$N"
         --margin-gb "$MARGIN" --rep "$REP" --fit-rows-dir "$RUNDIR" --model "$MODEL"
         --tokens "$WL/${STEMF}_tokens.bin" --manifest "$WL/${STEMF}_manifest.json" --gpu-uuid "$GPU_UUID")
    for d in "${PRIOR[@]}"; do DRV+=(--fit-rows-dir "$d"); done
fi
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env NOSI_REPO="$NOSI" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
      --env TRITON_CACHE_DIR="$JIT/triton" --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext"
      --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer" --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS
      --env NOSI_POOL_BLOCKS=$P --env NOSI_VERIFY_ROUND_SLOTS=$R)
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "ctxsweep child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "ctxsweep child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
argv_for() {      # $1 = the driver's --out
    ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
          --ntasks=1 --cpus-per-task=16 --mem="$STEP_MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
          /bin/sh -c "$CHILD" ctxsweep-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi
          "${ENVS[@]}" "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/ctxsweep_bench.py" "${DRV[@]}" --out "$1")
}
if [ "$DRY" = 1 ]; then
    argv_for "$RUNDIR/$STEM"
    echo "ctxsweep step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob;" \
         "gpu_uuid $GPU_UUID from $UUID_FILE; step limit $STEP_MIN min, --mem $STEP_MEM, MIN_LEFT $MIN_LEFT s"
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
mkdir "$OUTROOT/ctxsweep_active" 2>/dev/null || stop "$OUTROOT/ctxsweep_active exists (another step, or an uncertain one)"
mkdir -p "$RUNDIR" || stop "cannot create $RUNDIR (ctxsweep_active stays)"
n=1
until mkdir "$RUNDIR/$STEM.$n.step" 2>/dev/null; do
    n=$((n + 1))
    [ "$n" -le 100 ] || stop "no free attempt number below 100 (ctxsweep_active stays)"
done
REC=$RUNDIR/$STEM.$n.step
OUT=$RUNDIR/$STEM
[ "$MODE" = report ] && OUT=$RUNDIR/report_$n
[ ! -e "$OUT" ] || stop "$OUT exists: one output per cell, batch, rep and run (ctxsweep_active stays)"
mkdir -p "$JIT" || stop "cannot create $JIT (ctxsweep_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (ctxsweep_active stays)"
argv_for "$OUT"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (ctxsweep_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s ctx=%s cell=%s batch=%s mem=%s rep=%s budget=%s run=%s step_min=%s prior_runs=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UUID_FILE" "$JOB" "$sl" "${CTX:-}" "${CELL:-report}" "${BATCH:-}" "$STEP_MEM" \
    "${REP:-}" "$BUDGET" "$RUN" "$STEP_MIN" "${PRIOR[*]:-none}" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (ctxsweep_active stays)"
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
if [ -d "$OUT" ]; then
    find "$OUT" -type f -exec chmod 0444 {} + && find "$OUT" -type d -exec chmod 0555 {} + || ok=0
fi
find "$REC" -type f -exec chmod 0444 {} + && chmod 0555 "$REC" || ok=0
if [ "$ok" = 1 ]; then
    rmdir "$OUTROOT/ctxsweep_active" || ok=0
fi
echo "ctxsweep step: ${CELL:-report} ctx ${CTX:-} batch ${BATCH:-} rep ${REP:-} budget $BUDGET exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
