#!/bin/bash
# R2 MILESTONE 2, STEP 7, PIECE 5, GFW: ONE short bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/
# r2_gfw_gate.py: the gilfree C++ WORKER (packer v3 of this commit, Q-C2) on the A100 before the captured tick depends on
# it. MUST: GFW_ready (v3 and the pinned v2 built), GFW1 (S-C's exact wait WITHOUT its submission wait: a T4 event behind
# a GPU sleep, 1500 launches behind the spin; err 0, a real spin > 10 ms, every launch ran, rows the host truth) and
# GFW1n (the worker's publication planted off: the spin ends by its bound with err 1). REPORT: GFW_T64 / GFW_T192, the
# worker's per-release service time beside packer v2's in the same step (A-B-A) and v2's record (b64_s512_gilfree2,
# 8 threads: 4.59 ms at 64 MiB, 12.36 ms at 192 MiB; 15.86 ms with 4 threads).
# REGISTERED GFW BEGIN
# frozen 2026-10-21 before gfw1; per size; ratio = median(worker total) / median(v2 total, both v2 blocks)
# PREDICTION |ratio - 1| <= 0.15 at 64 and at 192 MiB
# FALSIFIER ratio >= 1.30 (the worker slower by 30 % or more) at either size; otherwise GREY
# REGISTERED GFW END
# The pattern of benchmarks/Efficiency/r2_fetch_fg_step.sh (feature/nosi-r2-m2-fg): the GPU uuid from
# HOLDERS/<holder job id>.gpu_uuid; packer v2 (gilfree_pack.py / .cpp of f2b981a) pinned by commit, path and blob;
# packer v3 = the blobs of benchmarks/Efficiency/gilfree_pack.{py,cpp} at --nosi; both copied into the step record and
# loaded by path (built into the step's JIT directory). No model, no workload.
# SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_gfw_step.sh --nosi <sha> <holder job> <run id>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_gfw_step.sh --nosi <sha> <holder job> <run id>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, GFW_HOLDERS=<dir> replaces HOLDERS (the CPU test uses it); a real run refuses GFW_HOLDERS.
#
# OUTPUT: OUTROOT/gfw_<run>_<holder>/summary.json, and the step record OUTROOT/gfw_<run>_<holder>.step/ (command with the
# uuid, its file and the pins, the copies, stdout with one GATE / REPORT line per gate, stderr, exit + accounting,
# sacct row), both frozen.
# EXIT: the gate's exit = the number of failed MUST gates; 2 a refusal; 3 another GPU; 4 not exactly one GPU;
# 124 / 137 the host timeout; 1 a STOP before the launch (a missing or malformed uuid file included); 5 an exit-0 step
# whose record is incomplete or whose sacct row is not COMPLETED 0:0.
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-gilfree-worker                                           # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/gfw                                                   # ROOT
GF_SHA=f2b981ac4ea450a1fc9d993a2c5d2c9a10a7db43                                          # PIN (E's packer v2)
GF_PY=benchmarks/Efficiency/gilfree_pack.py                                              # PIN
GF_PY_BLOB=45c56f679b017436810bea8e16fa2719d642ba54                                      # PIN
GF_CPP=benchmarks/Efficiency/gilfree_pack.cpp                                            # PIN
GF_CPP_BLOB=df3e3c8026fb0a48cc2fe2597013c3f20ab1338b                                     # PIN
THREADS=8
SIZES=64,192
SELF=benchmarks/Efficiency/r2_gfw_step.sh
STEP_MIN=12
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 GFW step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_gfw_step.sh --nosi <sha> <holder> <run id>" >&2; exit 2; }
[ $# = 4 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${GFW_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "GFW_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$GFW_HOLDERS
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
for f in benchmarks/Efficiency/r2_gfw_gate.py nosi/nosi/r2_transport_gilfree.py nosi/nosi/r2_residency_sc.py \
         nosi/nosi/r2_residency_sb.py nosi/nosi/r2_residency.py nosi/nosi/kstep/backend.py \
         nosi/nosi/kstep/ownership.py nosi/nosi/kstep/transfer.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
V3_PY_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$GF_PY") || stop "packer v3: $GF_PY is not tracked at $NSHA"
V3_CPP_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$GF_CPP") || stop "packer v3: $GF_CPP is not tracked at $NSHA"
[ "$V3_CPP_BLOB" != "$GF_CPP_BLOB" ] || stop "packer v3: $GF_CPP at $NSHA is v2's blob (no worker)"
git -C "$NOSI" show "$V3_CPP_BLOB" | grep -q 'mod.attr("VERSION") = 3;' || stop "packer v3: $GF_CPP at $NSHA is not VERSION 3"
[ "$(git -C "$NOSI" rev-parse -q --verify "$GF_SHA:$GF_PY")" = "$GF_PY_BLOB" ] ||
    stop "packer v2: $GF_PY at $GF_SHA is not the blob $GF_PY_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$GF_SHA:$GF_CPP")" = "$GF_CPP_BLOB" ] ||
    stop "packer v2: $GF_CPP at $GF_SHA is not the blob $GF_CPP_BLOB"
[ -f "$SIF" ] || stop "no image $SIF"
OUT=$OUTROOT/gfw_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/gfw_${RUN}_$JOB
NAME=gfw-${RUN}-$JOB
V3_DIR=$REC/packer_v3
V2_DIR=$REC/packer_v2
ENVS=(--env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env TMPDIR="$JIT/tmp" --env GFW_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 GFW child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 GFW child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=32G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" gfw-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_gfw_gate.py" --out "$OUT" --nosi-pkg "$NOSI/nosi/nosi"
      --v3 "$V3_DIR" --v3-build "$JIT/gfw_v3_build" --v2 "$V2_DIR" --v2-build "$JIT/gfw_v2_build"
      --threads "$THREADS" --releases 20 --warmup 3 --sizes "$SIZES" --launches 1500 --spin-cycles 200000000)
if [ "$DRY" = 1 ]; then
    echo "R2 GFW step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 GFW step: packer v3 $NSHA:$GF_PY blob $V3_PY_BLOB, $GF_CPP blob $V3_CPP_BLOB -> $V3_DIR; packer v2 $GF_SHA:$GF_PY blob $GF_PY_BLOB, $GF_CPP blob $GF_CPP_BLOB -> $V2_DIR"
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
mkdir "$OUTROOT/gfw_active" 2>/dev/null || stop "$OUTROOT/gfw_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (gfw_active stays)"
mkdir "$REC" && mkdir "$V3_DIR" && mkdir "$V2_DIR" && mkdir -p "$JIT/tmp" ||
    stop "cannot create $REC / $JIT (gfw_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (gfw_active stays)"
git -C "$NOSI" show "$V3_PY_BLOB" > "$V3_DIR/gilfree_pack.py" && [ "$(git hash-object -- "$V3_DIR/gilfree_pack.py")" = "$V3_PY_BLOB" ] ||
    stop "cannot copy packer v3's blob $V3_PY_BLOB (gfw_active stays)"
git -C "$NOSI" show "$V3_CPP_BLOB" > "$V3_DIR/gilfree_pack.cpp" && [ "$(git hash-object -- "$V3_DIR/gilfree_pack.cpp")" = "$V3_CPP_BLOB" ] ||
    stop "cannot copy packer v3's blob $V3_CPP_BLOB (gfw_active stays)"
git -C "$NOSI" show "$GF_PY_BLOB" > "$V2_DIR/gilfree_pack.py" && [ "$(git hash-object -- "$V2_DIR/gilfree_pack.py")" = "$GF_PY_BLOB" ] ||
    stop "cannot copy packer v2's blob $GF_PY_BLOB (gfw_active stays)"
git -C "$NOSI" show "$GF_CPP_BLOB" > "$V2_DIR/gilfree_pack.cpp" && [ "$(git hash-object -- "$V2_DIR/gilfree_pack.cpp")" = "$GF_CPP_BLOB" ] ||
    stop "cannot copy packer v2's blob $GF_CPP_BLOB (gfw_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (gfw_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (gfw_active stays)"
printf 'packer_v3=%s py=%s cpp=%s packer_v2=%s py=%s cpp=%s threads=%s sizes=%s\n' "$NSHA" "$V3_PY_BLOB" "$V3_CPP_BLOB" \
    "$GF_SHA" "$GF_PY_BLOB" "$GF_CPP_BLOB" "$THREADS" "$SIZES" >> "$REC/command.txt" ||
    stop "cannot write $REC (gfw_active stays)"
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
    rmdir "$OUTROOT/gfw_active" || ok=0
fi
echo "R2 GFW step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
