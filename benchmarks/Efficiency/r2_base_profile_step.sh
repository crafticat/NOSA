#!/bin/bash
# Task 0.1 + Task 0.4 "p0prof" (kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md): the profile of the
# cached-NOSA decode step and the ceiling for speculation, with the same-job baseline. One srun step per cell, in an
# EXISTING holder. The pattern of r2_base_fused_step.sh (ded9d41): the GPU uuid comes from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: the script never submits, renews, cancels or replaces an
# allocation, and it never retries.
#
#   bash benchmarks/Efficiency/r2_base_profile_step.sh --nosi <sha> <holder job> <run id> <cell>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_base_profile_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step commands; no squeue, claim, srun or write)
#   In DRYRUN only, P0_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses P0_HOLDERS.
#
# CELLS (NOSI_ATTN_SPLITS 4; C = 63 + P; 512 steps; the arm H steady step excludes every re-warm):
#   mb1        THE BASELINE (cached NOSA): r2_base_cell.py with the python arguments of basef 2190876 mb1 (C 128, P 65,
#              B 216, S 16128, SH, --mbase). STEP_MIN 35, --mem 240G.
#   p0c128     C 128 (P 65) B 216 S 16128: r2_base_profile_cell.py, arm H (no timer) and then arm T (--stage-timers).
#              STEP_MIN 45, --mem 240G.
#   p0c63b344  C  63 (P  0) B 344 S 16128, as p0c128. STEP_MIN 60, --mem 340G.
#   p0c63b216  C  63 (P  0) B 216 S 16128, as p0c128. STEP_MIN 45, --mem 240G.
#   p0k128b63  C 128 (P 65) B 63 S 130816 (the 24-document 128K workload), as p0c128 without --mbase. B 63 is the
#              largest B at 128K: the pooled engine's int32 gather envelope B x (S + 576) x 256 < 2^31
#              (cache_engine.py:216) allows 63 and refuses 64; C 113 B 63 used 62.2 GB at 128K (ctxsweep cs2 2190876
#              fit_row.json), C 128 adds about 2 GB. STEP_MIN 100, --mem 440G (the ctxsweep 128K cells).
#   p0nsys     C 128 (P 65) B 216 S 16128, 40 steps: `nsys profile -t cuda,nvtx` with the capture range of
#              cudaProfilerStart / Stop = the steps 8 .. 27 (arm N only; no arm H, no timers). STEP_MIN 30, --mem 240G.
#              The report: <record>/nsys_p0nsys.nsys-rep. NSYS below is the binary of the image (jobs 2175759, 2178106).
#   p0all      THE SEQUENCE (Task 0.4): mb1 FIRST, then p0c128 p0c63b344 p0c63b216 p0k128b63 (an even holder id) or
#              p0k128b63 p0c63b216 p0c63b344 p0c128 (an odd holder id). A failed mb1 stops the sequence.
# THE SAME-JOB BASELINE RULE (Task 0.4): every p0 cell gets --baseline OUTROOT/p0prof_<run>_<holder>_mb1/summary.json and
# prints "variant / baseline" for the median, the mean and the effective tok/s. A single p0 cell STOPS unless that mb1
# record is complete (exit=0, accounting=consistent, summary.json present). There is no --skip-shipped: arm H always runs.
# FLAG OFF = BASELINE: the script STOPS unless nosi/, r2_base_cell.py and r2_ragged_gate.py are the bytes of BASE_SHA.
# OUTPUT: OUTROOT/p0prof_<run>_<holder>_<cell>/summary.json and the step record ..._<cell>.step/ (command + pins,
# stdout, stderr, exit + accounting, sacct row), all frozen (0444 / 0555).
# EXIT: one cell: the cell's exit (the number of failed MUST gates 0-19; 20 refused; 30 raised); 2 a usage error; 3
# another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an exit-0 step whose
# record is incomplete or whose sacct row is not COMPLETED 0:0. p0all: the first non-zero cell exit (5 if every cell
# exited 0 and a record is incomplete).
# Checks before the claim (each STOPS with exit 1): as r2_base_fused_step.sh, plus the BASE_SHA bytes and the 128K
# workload pin. The claim: OUTROOT/p0prof_active (mkdir, atomic) for the whole sequence; a stop after the claim leaves
# it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-p0prof                                                   # ROOT
BASE_SHA=ded9d412d81aef76e2c1b6983673e6eaa7010adc                                        # PIN (the mb1 code)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
NSYS=/usr/local/cuda-12.8/NsightSystems-cli-2024.6.2/target-linux-x64/nsys              # IN THE IMAGE
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
WLK=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx130816                        # ROOT
WLK_SHA=ff5df555025599d59d0c5c586c39ec7ed112cede78175b565387c0dde185a5d2                  # PIN (its SHA256SUMS)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/p0prof                                                # ROOT
SELF=benchmarks/Efficiency/r2_base_profile_step.sh
SPLITS=4
NSYS_WINDOW=8
main() {
stop() { echo "R2 p0prof step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_base_profile_step.sh --nosi <sha> <holder> <run id> <mb1|p0c128|p0c63b344|p0c63b216|p0k128b63|p0nsys|p0all>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
[[ $JOB =~ ^[1-9][0-9]{0,11}$ ]] || stop "holder job id '$JOB'"
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
case $CELL in
    mb1|p0c128|p0c63b344|p0c63b216|p0k128b63|p0nsys) CELLS=("$CELL") ;;
    p0all)
        if [ $((JOB % 2)) = 0 ]; then
            CELLS=(mb1 p0c128 p0c63b344 p0c63b216 p0k128b63)
        else
            CELLS=(mb1 p0k128b63 p0c63b216 p0c63b344 p0c128)
        fi ;;
    *) stop "cell '$CELL' (mb1, p0c128, p0c63b344, p0c63b216, p0k128b63, p0nsys or p0all)" ;;
esac
if [ -n "${P0_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "P0_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$P0_HOLDERS
fi
cfg() {   # $1 = a cell: POOL BATCH STEPS STEP_MIN MEM CTX WLX PYF PYX (the python file and its cell arguments)
    PYX=()
    CTX=16128 WLX=$WL PYF=r2_base_profile_cell.py
    case $1 in
        mb1) POOL=65 BATCH=216 STEPS=512 STEP_MIN=35 MEM=240G PYF=r2_base_cell.py
             PYX=(--warmup 2 --r2 fused --arms SH --pool 65 --mbase) ;;
        p0c128) POOL=65 BATCH=216 STEPS=512 STEP_MIN=45 MEM=240G PYX=(--pool 65 --mbase --stage-timers) ;;
        p0c63b344) POOL=0 BATCH=344 STEPS=512 STEP_MIN=60 MEM=340G PYX=(--pool 0 --mbase --stage-timers) ;;
        p0c63b216) POOL=0 BATCH=216 STEPS=512 STEP_MIN=45 MEM=240G PYX=(--pool 0 --mbase --stage-timers) ;;
        p0k128b63) POOL=65 BATCH=63 STEPS=512 STEP_MIN=100 MEM=440G CTX=130816 WLX=$WLK
                   PYX=(--context 130816 --pool 65 --stage-timers) ;;
        p0nsys) POOL=65 BATCH=216 STEPS=40 STEP_MIN=30 MEM=240G PYX=(--pool 65 --nsys-window "$NSYS_WINDOW") ;;
        *) stop "cfg: no cell '$1'" ;;
    esac
}
TOTAL_MIN=0
for c in "${CELLS[@]}"; do
    cfg "$c"
    TOTAL_MIN=$((TOTAL_MIN + STEP_MIN))
done
MIN_LEFT=$((TOTAL_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_base_cell.py benchmarks/Efficiency/r2_base_profile_cell.py \
         benchmarks/Efficiency/r2_p3_profile.py benchmarks/Efficiency/r2_ragged_gate.py \
         benchmarks/Efficiency/r2_g0_compare.py nosi/nosi/nosa_llama.py nosi/nosi/cache_engine.py \
         nosi/nosi/max_pooling_fused.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
git -C "$NOSI" diff --quiet "$BASE_SHA" "$NSHA" -- nosi benchmarks/Efficiency/r2_base_cell.py \
    benchmarks/Efficiency/r2_ragged_gate.py benchmarks/Efficiency/r2_g0_compare.py ||
    stop "nosi/, r2_base_cell.py, r2_ragged_gate.py or r2_g0_compare.py differ from $BASE_SHA (flags off = baseline)"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
wlk=$(sha256sum "$WLK/ctx130816_tokens.bin" 2>/dev/null) || stop "no $WLK/ctx130816_tokens.bin"
[ "${wlk%% *}" = "$WLK_SHA" ] || stop "ctx130816_tokens.bin sha256 is not $WLK_SHA"
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
BASEREC=$OUTROOT/p0prof_${RUN}_${JOB}_mb1
base_ok() {
    [ -f "$BASEREC/summary.json" ] && [ -f "$BASEREC.step/exit.txt" ] &&
        grep -q '^exit=0 ' "$BASEREC.step/exit.txt" && grep -qx 'accounting=consistent' "$BASEREC.step/exit.txt"
}
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 p0prof child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 p0prof child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
build() {   # $1 = a cell: OUT REC JIT NAME ARGV
    cfg "$1"
    OUT=$OUTROOT/p0prof_${RUN}_${JOB}_$1
    REC=$OUT.step
    JIT=$JITBASE/p0prof_${RUN}_${JOB}_$1
    NAME=p0prof-$1-$RUN-$JOB
    BX=()
    if [ "$1" != mb1 ]; then
        BX=(--baseline "$BASEREC/summary.json")
    fi
    PRE=()
    if [ "$1" = p0nsys ]; then
        PRE=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
             --capture-range=cudaProfilerApi --capture-range-end=stop -o "$REC/nsys_$1")
    fi
    ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
          --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
          --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
          --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
          --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
          --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS
          --env NOSI_POOL_BLOCKS=$POOL --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
    ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
          --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
          /bin/sh -c "$CHILD" p0prof-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi
          "${ENVS[@]}" "$SIF" "${PRE[@]}" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/$PYF" --out "$OUT"
          --model "$MODEL" --workload "$WLX" --batch "$BATCH" --steps "$STEPS" "${PYX[@]}" "${BX[@]}")
}
if [ "$DRY" = 1 ]; then
    if base_ok; then bstate=complete; else bstate=absent; fi
    echo "R2 p0prof step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob base $BASE_SHA workload $WL_SHA 128K workload $WLK_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 p0prof step: cells ${CELLS[*]}; STEP_MIN total $TOTAL_MIN; baseline record $BASEREC ($bstate)"
    for c in "${CELLS[@]}"; do
        build "$c"
        echo "R2 p0prof step: cell $c: C $((63 + POOL)) (pool $POOL) B $BATCH S $CTX steps $STEPS STEP_MIN $STEP_MIN mem $MEM python $PYF"
        printf '%q ' "${ARGV[@]}"
        echo
    done
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
[ "$left" -ge "$MIN_LEFT" ] || stop "holder $JOB has $sl left (< $MIN_LEFT s for ${CELLS[*]})"
nosteps() {
    steps=$(timeout 60 squeue -s -j "$JOB" -h -o '%i') || return 1
    for s in $steps; do case $s in *.batch|*.extern) ;; *) echo "$s"; return 1 ;; esac; done
}
busy=$(nosteps) || stop "a step is running in $JOB (or squeue -s failed): '$busy'"
if [ "${CELLS[0]}" != mb1 ]; then
    base_ok || stop "no complete mb1 record $BASEREC(.step) for run $RUN in holder $JOB (Task 0.4: run mb1 or p0all first)"
fi
mkdir -p "$OUTROOT" || stop "cannot create $OUTROOT"
mkdir "$OUTROOT/p0prof_active" 2>/dev/null || stop "$OUTROOT/p0prof_active exists (another step, or an uncertain one)"
for c in "${CELLS[@]}"; do
    build "$c"
    [ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (p0prof_active stays)"
done
run_cell() {   # $1 = a cell: RC (its exit) and OK (1 = the record is complete and frozen)
    build "$1"
    RC=1 OK=0
    busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (p0prof_active stays)"
    mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (p0prof_active stays)"
    printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (p0prof_active stays)"
    printf '\nnosi=%s wrapper_blob=%s base=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s sequence=%s pool=%s batch=%s context=%s steps=%s workload=%s utc_start=%s\n' \
        "$NSHA" "$blob" "$BASE_SHA" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$1" "${CELLS[*]}" "$POOL" "$BATCH" \
        "$CTX" "$STEPS" "$WLX" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
        stop "cannot write $REC (p0prof_active stays)"
    "${ARGV[@]}" < /dev/null > "$REC/stdout.txt" 2> "$REC/stderr.txt"
    RC=$?
    OK=1
    echo "exit=$RC utc_end=$(date -u +%FT%TZ)" > "$REC/exit.txt" || OK=0
    sleep 10
    timeout 60 env TZ=UTC sacct -j "$JOB" -n -P -o JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS < /dev/null |
        grep -F "|$NAME|" > "$REC/sacct.txt" || OK=0
    acct="no single sacct row"
    if [ "$(wc -l < "$REC/sacct.txt")" = 1 ]; then
        IFS='|' read -r _ _ sstate scode _ < "$REC/sacct.txt"
        case "$RC:$sstate:$scode" in
            "0:COMPLETED:0:0") acct=consistent ;;
            0:*) acct="inconsistent: exit 0, sacct $sstate $scode" ;;
            *) acct="exit $RC, sacct $sstate $scode" ;;
        esac
    fi
    echo "accounting=$acct" >> "$REC/exit.txt" || OK=0
    [ "$RC" != 0 ] || [ "$acct" = consistent ] || OK=0
    for t in "$OUT" "$REC"; do
        [ -d "$t" ] || continue
        find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || OK=0
    done
    echo "R2 p0prof step: cell $1: exit $RC; record $REC; output $OUT; record complete and frozen: $OK"
}
first=0 allok=1
for c in "${CELLS[@]}"; do
    run_cell "$c"
    [ "$OK" = 1 ] || allok=0
    [ "$first" != 0 ] || first=$RC
    if [ "$c" = mb1 ] && [ "${#CELLS[@]}" -gt 1 ]; then
        if [ "$RC" != 0 ] || [ "$OK" != 1 ] || ! base_ok; then
            echo "R2 p0prof step: mb1 failed (exit $RC, record complete $OK): the sequence stops before the p0 cells" >&2
            break
        fi
    fi
done
if [ "$allok" = 1 ]; then
    rmdir "$OUTROOT/p0prof_active" || allok=0
fi
echo "R2 p0prof step: cells ${CELLS[*]}: first non-zero exit $first; every record complete and frozen: $allok"
[ "$first" = 0 ] && [ "$allok" = 0 ] && exit 5
exit "$first"
}
main "$@"; exit $?
