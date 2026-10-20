#!/bin/bash
# R2 MILESTONE 2, STEP 7, PIECE 4, SG5: ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/
# r2_tick_sg5.py: the R-PERF budget of the flattened tick (nosi/nosi/r2_tick.py) on piece 5's Residency (r2_residency*.py,
# C's 2ec08da: backends sc / sb / v1; the gilfree transport over agent E's packer v2), in EXACT and FAST GEMM modes,
# at the primary point C 80 / X 32 / R_ver 16, then C's sweep C {64, 80, 96} x X {16, 32, 64} (R_ver = X / 2).
# B = 64 documents of pg19-test-c16k, S = 16128, 48 committed tokens per request (24 in the sweep). T_shipped (and the
# equal-memory shipped decode, NOSI_POOL_BLOCKS = C + X + 4 - 64) in child processes. REPORT; MUST = the runs complete.
# The pattern of benchmarks/Efficiency/r2_tick_step.sh: the GPU uuid from HOLDERS/<holder job id>.gpu_uuid; piece 3
# and the packer (gilfree_pack.py / .cpp of f2b981a) pinned by commit, path and blob, copied into the step record and
# loaded by path (the packer is built into the step's JIT directory).
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_tick_sg5_step.sh --nosi <sha> <holder job> <run id> [gilfree|gather]
#   DRYRUN=1 bash benchmarks/Efficiency/r2_tick_sg5_step.sh --nosi <sha> <holder job> <run id> [gilfree|gather]
#        (the transport: gilfree, the default; gather = NOSI's Triton gather on a side stream, r2_tick_sg5.py)
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, SG5_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses SG5_HOLDERS.
#
# OUTPUT: OUTROOT/sg5_<run>_<holder>/summary.json (+ the children's shipped_*.json), and the step record
# OUTROOT/sg5_<run>_<holder>.step/ (command with the uuid, its file and the pins, the copies, stdout with one GATE /
# REPORT line per gate, stderr, exit + accounting, sacct row), both frozen.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed uuid file included); 5 an exit-0 step
# whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-sg5                                                   # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (r2_g0_step.sh)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/sg5                                                   # ROOT
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN (piece 3 v2)
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
GF_SHA=f2b981ac4ea450a1fc9d993a2c5d2c9a10a7db43                                          # PIN (E's packer v2)
GF_PY=benchmarks/Efficiency/gilfree_pack.py                                              # PIN
GF_PY_BLOB=45c56f679b017436810bea8e16fa2719d642ba54                                      # PIN
GF_CPP=benchmarks/Efficiency/gilfree_pack.cpp                                            # PIN
GF_CPP_BLOB=df3e3c8026fb0a48cc2fe2597013c3f20ab1338b                                     # PIN
SWEEP=64:16,64:32,64:64,80:16,80:32,80:64,96:16,96:32,96:64
SPLITS=4
THREADS=8
SELF=benchmarks/Efficiency/r2_tick_sg5_step.sh
STEP_MIN=120
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 SG5 step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_tick_sg5_step.sh --nosi <sha> <holder> <run id> [gilfree|gather]" >&2; exit 2; }
{ [ $# = 4 ] || [ $# = 5 ]; } && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 TRANSPORT=${5:-gilfree}
case $TRANSPORT in gilfree|gather) ;; *) usage ;; esac
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${SG5_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "SG5_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$SG5_HOLDERS
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
for f in benchmarks/Efficiency/r2_tick_sg5.py benchmarks/Efficiency/r2_g0bi_gate.py benchmarks/Efficiency/r2_g0_gate.py \
         benchmarks/Efficiency/r2_g0_compare.py nosi/nosi/r2_tick.py nosi/nosi/r2_bi_decode.py \
         nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py \
         nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py \
         nosi/nosi/r2_residency.py nosi/nosi/r2_residency_sb.py nosi/nosi/r2_residency_sc.py \
         nosi/nosi/r2_transport_gilfree.py nosi/nosi/kstep/ownership.py nosi/nosi/kstep/transfer.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$GF_SHA:$GF_PY")" = "$GF_PY_BLOB" ] ||
    stop "packer: $GF_PY at $GF_SHA is not the blob $GF_PY_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$GF_SHA:$GF_CPP")" = "$GF_CPP_BLOB" ] ||
    stop "packer: $GF_CPP at $GF_SHA is not the blob $GF_CPP_BLOB"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
OUT=$OUTROOT/sg5_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/sg5_${RUN}_$JOB
NAME=sg5-${RUN}-$JOB
ATTN_MOD=$REC/piece3_bi_paged_attention.py
GF_DIR=$REC/gilfree
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_R2_DECODE=1 --env NOSI_MODEL_PATH="$MODEL" --env TICK_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 SG5 child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 SG5 child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=120G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" sg5-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_tick_sg5.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --biattn "$ATTN_MOD" --gilfree "$GF_DIR" --gilfree-build "$JIT/gilfree_build"
      --threads "$THREADS" --transport "$TRANSPORT" --batch 64 --S 16128 --steps 48 --K 3 --LAMBDA 7 --modes exact,fast --backends sc,sb,v1
      --C 80 --X 32 --R-ver 16 --sweep "$SWEEP" --sweep-steps 24 --draft-ticks 12 --shipped-steps 24 --pool-baseline)
if [ "$DRY" = 1 ]; then
    echo "R2 SG5 step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 SG5 step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; packer $GF_SHA:$GF_PY blob $GF_PY_BLOB, $GF_CPP blob $GF_CPP_BLOB -> $GF_DIR; transport $TRANSPORT"
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
mkdir "$OUTROOT/sg5_active" 2>/dev/null || stop "$OUTROOT/sg5_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (sg5_active stays)"
mkdir "$REC" && mkdir "$GF_DIR" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (sg5_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (sg5_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (sg5_active stays)"
git -C "$NOSI" show "$GF_PY_BLOB" > "$GF_DIR/gilfree_pack.py" && [ "$(git hash-object -- "$GF_DIR/gilfree_pack.py")" = "$GF_PY_BLOB" ] ||
    stop "cannot copy the packer's blob $GF_PY_BLOB (sg5_active stays)"
git -C "$NOSI" show "$GF_CPP_BLOB" > "$GF_DIR/gilfree_pack.cpp" && [ "$(git hash-object -- "$GF_DIR/gilfree_pack.cpp")" = "$GF_CPP_BLOB" ] ||
    stop "cannot copy the packer's blob $GF_CPP_BLOB (sg5_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (sg5_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (sg5_active stays)"
printf 'biattn=%s:%s blob=%s packer=%s py=%s cpp=%s threads=%s transport=%s sweep=%s splits=%s\n' "$BIATTN_SHA" \
    "$BIATTN_PATH" "$BIATTN_BLOB" "$GF_SHA" "$GF_PY_BLOB" "$GF_CPP_BLOB" "$THREADS" "$TRANSPORT" "$SWEEP" "$SPLITS" \
    >> "$REC/command.txt" ||
    stop "cannot write $REC (sg5_active stays)"
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
    rmdir "$OUTROOT/sg5_active" || ok=0
fi
echo "R2 SG5 step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
