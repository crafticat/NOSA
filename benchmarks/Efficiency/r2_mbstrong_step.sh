#!/bin/bash
# r2-mbstrong: the strengthened baseline (paper section 4.1: "each system may choose its cache capacity and batch size
# within that budget"). Copied from r2_base_fused_step.sh (ded9d41): the pins, the uuid file, the record and the exit
# codes are the same; the sequence of srun steps is the pattern of r2_base_profile_step.sh (nosi-r2-p0prof 3351726).
# One srun step per SUB-CELL, in an EXISTING holder. Every sub-cell runs benchmarks/Efficiency/r2_base_cell.py with the
# python arguments of mb1 (--warmup 2 --r2 fused --arms SH --pool P --mbase; S 16128, 512 steps, the PG19 94-document
# workload) at its C and B, plus the r2-mbstrong flags of its role. SOURCE ONLY until reviewed: the script never
# submits, renews, cancels or replaces an allocation, and it never retries.
#
#   bash benchmarks/Efficiency/r2_mbstrong_step.sh --nosi <sha> <holder job> <run id> <cell>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_mbstrong_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (the pin checks, the uuid file read and the printed step commands; no squeue, claim, srun or write)
#   In DRYRUN only, MBS_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses MBS_HOLDERS.
#
# SUB-CELLS: mb1 = C 128 B 216 plain; mb1n = the same with --recap-nogc; c<C>b<B> = C, B plain; c<C>b<B>n = with
# --recap-nogc. C in 63 73 81 96 113 128 (P = C - 63, NOSI_POOL_BLOCKS = P); B in 1 .. 502 (the int32 envelope of
# cache_engine.py:216 at S 16128, 512 steps: r2_base_cell.int32_batch_limit). STEP_MIN = 20 + B / 12 (the prefill is
# about 3.0 s per request: dr1 657.4 s at B 216, dr73 943.8 s at B 318, dr81 878.5 s at B 290); --mem 240G up to
# B 248, else 340G (the pinned host rows: r2_base_fused_step.sh).
# ROLES: an A/B PAIR (one plain, one --recap-nogc sub-cell of one C and B): both get --ab-record; the SECOND also gets
# --ab-ref <the first's output> (MUST gate RN-EQ). The pair order is plain first for an even holder id and --recap-nogc
# first for an odd one (Task 0.4). A PROBE (--recap-nogc, no twin) gets --fit-probe: an out-of-memory SH arm is a
# REPORT (exit 0). Every --recap-nogc sub-cell has the MUST gate RN-GC.
# CELLS (the B values: r2_base_cell.fit_batch; "matched" = the peak model at most 82.04 GB, the measured peak of mb1;
# "edge" = at most 84.0 GB, the budget):
#   mbs0    pair c128b16 / c128b16n: the preflight miniature (every gate and the --ab-ref path in a few minutes)
#   mbs1    pair mb1 / mb1n                                               (C 128 B 216: mb1 with and without the flag)
#   mbx73   pair c73b318 / c73b318n, probes c73b358n (matched), c73b369n (edge)
#   mbx81   pair c81b290 / c81b290n, probes c81b326n (matched), c81b336n (edge)
#   mbx63   pair c63b344 / c63b344n, probe c63b406n (matched)
#   mbx128  probe c128b222n (edge; NOT in the brief: the same 84 GB budget for C 128, for symmetry)
#   c<C>b<B>n  one probe (a re-run, or another B near an out-of-memory edge)
# A sub-cell that exits non-zero, or whose record is incomplete, stops the sequence (an out-of-memory probe exits 0).
# OUTPUT: OUTROOT/mbs_<run>_<holder>_<sub-cell>/summary.json (+ ab_record.pt in a pair) and the step record
# ..._<sub-cell>.step/ (command + pins, stdout, stderr, exit + accounting, sacct row), all frozen (0444 / 0555).
# EXIT: the first non-zero sub-cell exit (the number of failed MUST gates 0-19; 20 refused; 30 raised); 2 a usage error;
# 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before a launch; 5 every sub-cell exited 0
# and a record is incomplete or its sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): as r2_base_fused_step.sh, plus nosi/ and r2_ragged_gate.py are the
# bytes of BASE_SHA (the engine is not edited: flag off = the shipped engine). The claim: OUTROOT/mbs_active_<holder>
# (mkdir, atomic; one per holder, so two holders can run two cells); a stop after the claim leaves it (the lead
# decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-mbstrong                                                 # ROOT
BASE_SHA=ded9d412d81aef76e2c1b6983673e6eaa7010adc                                        # PIN (the mb1 code)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/mbstrong                                              # ROOT
SELF=benchmarks/Efficiency/r2_mbstrong_step.sh
SPLITS=4
STEPS=512
main() {
stop() { echo "R2 mbstrong step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_mbstrong_step.sh --nosi <sha> <holder> <run id> <mbs0|mbs1|mbx73|mbx81|mbx63|mbx128|c<C>b<B>n>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
[[ $JOB =~ ^[1-9][0-9]{0,11}$ ]] || stop "holder job id '$JOB'"
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
cfg() {   # $1 = a sub-cell: C POOL BATCH MODE STEP_MIN MEM
    local s=$1
    case $s in
        mb1) C=128 BATCH=216 MODE=plain ;;
        mb1n) C=128 BATCH=216 MODE=nogc ;;
        c*b*)
            MODE=plain
            s=${s#c}
            case $s in *n) MODE=nogc s=${s%n} ;; esac
            C=${s%%b*} BATCH=${s#*b}
            [ "$1" = "c${C}b${BATCH}" ] || [ "$1" = "c${C}b${BATCH}n" ] || stop "sub-cell '$1' (c<C>b<B> or c<C>b<B>n)" ;;
        *) stop "sub-cell '$1' (mb1, mb1n, c<C>b<B> or c<C>b<B>n)" ;;
    esac
    case $C in 63|73|81|96|113|128) ;; *) stop "sub-cell '$1': C '$C' (63, 73, 81, 96, 113 or 128)" ;; esac
    [[ $BATCH =~ ^[1-9][0-9]{0,2}$ ]] && [ "$BATCH" -le 502 ] ||
        stop "sub-cell '$1': B '$BATCH' (1 .. 502: the int32 envelope at S 16128, $STEPS steps)"
    POOL=$((C - 63))
    STEP_MIN=$((20 + BATCH / 12))
    MEM=240G
    [ "$BATCH" -le 248 ] || MEM=340G
}
PAIR=() PROBES=()
case $CELL in
    mbs0) PAIR=(c128b16 c128b16n) ;;
    mbs1) PAIR=(mb1 mb1n) ;;
    mbx73) PAIR=(c73b318 c73b318n) PROBES=(c73b358n c73b369n) ;;
    mbx81) PAIR=(c81b290 c81b290n) PROBES=(c81b326n c81b336n) ;;
    mbx63) PAIR=(c63b344 c63b344n) PROBES=(c63b406n) ;;
    mbx128) PROBES=(c128b222n) ;;
    c*b*n) PROBES=("$CELL") ;;
    *) stop "cell '$CELL' (mbs0, mbs1, mbx73, mbx81, mbx63, mbx128 or c<C>b<B>n)" ;;
esac
if [ "${#PAIR[@]}" = 2 ] && [ $((JOB % 2)) = 1 ]; then
    PAIR=("${PAIR[1]}" "${PAIR[0]}")                     # Task 0.4: the pair order swaps with the holder parity
fi
CELLS=("${PAIR[@]}" "${PROBES[@]}")
role() {   # $1 = a sub-cell: ROLE first | second | probe
    ROLE=probe
    if [ "${#PAIR[@]}" = 2 ]; then
        if [ "$1" = "${PAIR[0]}" ]; then ROLE=first; fi
        if [ "$1" = "${PAIR[1]}" ]; then ROLE=second; fi
    fi
}
TOTAL_MIN=0
for c in "${CELLS[@]}"; do
    cfg "$c"
    role "$c"
    if [ "$ROLE" = probe ] && [ "$MODE" != nogc ]; then
        stop "sub-cell '$c': a probe runs with --recap-nogc (name it c<C>b<B>n)"
    fi
    TOTAL_MIN=$((TOTAL_MIN + STEP_MIN))
done
MIN_LEFT=$((TOTAL_MIN * 60 + 600))
if [ -n "${MBS_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "MBS_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$MBS_HOLDERS
fi
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_base_cell.py benchmarks/Efficiency/r2_recap_nogc.py \
         benchmarks/Efficiency/r2_ragged_gate.py benchmarks/Efficiency/r2_fused_cell.py \
         nosi/nosi/nosa_llama.py nosi/nosi/cache_engine.py nosi/nosi/max_pooling_fused.py \
         nosi/nosi/flash_cache_engine/pool_update.cpp nosi/nosi/flash_cache_engine/pool_update_kernel.cu \
         nosi/nosi/flash_cache_engine/flash_pool_swap.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
git -C "$NOSI" diff --quiet "$BASE_SHA" "$NSHA" -- nosi benchmarks/Efficiency/r2_ragged_gate.py ||
    stop "nosi/ or r2_ragged_gate.py differ from $BASE_SHA (the engine must be the shipped bytes)"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
CLAIM=$OUTROOT/mbs_active_$JOB
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 mbstrong child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 mbstrong child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
build() {   # $1 = a sub-cell: OUT REC JIT NAME RX ARGV (and C POOL BATCH MODE STEP_MIN MEM ROLE)
    cfg "$1"
    role "$1"
    OUT=$OUTROOT/mbs_${RUN}_${JOB}_$1
    REC=$OUT.step
    JIT=$JITBASE/mbs_${RUN}_${JOB}_$1
    NAME=mbs-$1-$RUN-$JOB
    RX=()
    if [ "$MODE" = nogc ]; then
        RX+=(--recap-nogc)
    fi
    case $ROLE in
        first) RX+=(--ab-record) ;;
        second) RX+=(--ab-record --ab-ref "$OUTROOT/mbs_${RUN}_${JOB}_${PAIR[0]}") ;;
        probe) RX+=(--fit-probe) ;;
    esac
    ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
          --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
          --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
          --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
          --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
          --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS
          --env NOSI_POOL_BLOCKS=$POOL --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
    ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
          --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
          /bin/sh -c "$CHILD" mbs-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi
          "${ENVS[@]}" "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_base_cell.py" --out "$OUT"
          --model "$MODEL" --workload "$WL" --batch "$BATCH" --steps "$STEPS" --warmup 2 --r2 fused --arms SH
          --pool "$POOL" --mbase "${RX[@]}")
}
if [ "$DRY" = 1 ]; then
    echo "R2 mbstrong step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob base $BASE_SHA workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 mbstrong step: cells ${CELLS[*]}; STEP_MIN total $TOTAL_MIN; claim $CLAIM"
    for c in "${CELLS[@]}"; do
        build "$c"
        echo "R2 mbstrong step: sub-cell $c: C $C (pool $POOL) B $BATCH mode $MODE role $ROLE steps $STEPS STEP_MIN $STEP_MIN mem $MEM"
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
mkdir -p "$OUTROOT" || stop "cannot create $OUTROOT"
mkdir "$CLAIM" 2>/dev/null || stop "$CLAIM exists (another step in this holder, or an uncertain one)"
for c in "${CELLS[@]}"; do
    build "$c"
    [ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id ($CLAIM stays)"
done
run_cell() {   # $1 = a sub-cell: RC (its exit) and OK (1 = the record is complete and frozen)
    build "$1"
    RC=1 OK=0
    busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIM stays)"
    mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT ($CLAIM stays)"
    printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIM stays)"
    printf '\nnosi=%s wrapper_blob=%s base=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s sub_cell=%s role=%s mode=%s sequence=%s pool=%s batch=%s steps=%s workload=%s utc_start=%s\n' \
        "$NSHA" "$blob" "$BASE_SHA" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$1" "$ROLE" "$MODE" \
        "${CELLS[*]}" "$POOL" "$BATCH" "$STEPS" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
        stop "cannot write $REC ($CLAIM stays)"
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
    if [ "$ROLE" = first ] && [ "$RC" = 0 ] && [ ! -f "$OUT/ab_record.pt" ]; then
        OK=0                                             # the second half of the pair needs it
    fi
    for t in "$OUT" "$REC"; do
        [ -d "$t" ] || continue
        find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || OK=0
    done
    echo "R2 mbstrong step: sub-cell $1 ($ROLE, $MODE): exit $RC; record $REC; output $OUT; record complete and frozen: $OK"
}
first=0 allok=1
for c in "${CELLS[@]}"; do
    run_cell "$c"
    [ "$OK" = 1 ] || allok=0
    [ "$first" != 0 ] || first=$RC
    if [ "$RC" != 0 ] || [ "$OK" != 1 ]; then
        echo "R2 mbstrong step: sub-cell $c failed (exit $RC, record complete $OK): the sequence stops" >&2
        break
    fi
done
if [ "$allok" = 1 ]; then
    rmdir "$CLAIM" || allok=0
fi
echo "R2 mbstrong step: cells ${CELLS[*]}: first non-zero exit $first; every record complete and frozen: $allok"
[ "$first" = 0 ] && [ "$allok" = 0 ] && exit 5
exit "$first"
}
main "$@"; exit $?
