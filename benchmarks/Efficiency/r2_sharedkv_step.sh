#!/bin/bash
# R2 SHARED READY-KV READS (docs/R2_SHAREDKV_PLAN.md): ONE bounded GPU step in an EXISTING holder that runs
# benchmarks/Efficiency/r2_tick_gate.py with the attention selector --attn <arm> (nosi/nosi/bi_paged_attention_shared.py
# over piece 3 v2). A copy of benchmarks/Efficiency/r2_tick_step.sh (its pins, uuid file, claim, record and freeze);
# the changes: the worktree, OUTROOT, B = 32 at 16K (B 64 does not fit this gate: the ResidentStore base + one clone
# = 69 GB + the model; plan section 3), M_FIX = 4B = 128, the 32 deltas, and the ARM argument:
#   probe | probe-whole | probe-plant   --parts SG1 --draft-drop 0.0  (the in-tick A/B: piece 3's output drives the
#                                        tick; both kernels on every keyed call; bitwise, tiles, time; the plant = the
#                                        negative control)
#   shared | shared-whole               --parts SG0,SG1,SG2,SG3 --draft-drop 0.5  (the shared kernel's output drives
#                                        the tick; SG1-SG3 gate it bitwise against the BI serial REF)
#   per_row                             --parts SG1 --draft-drop 0.0 --attn per_row (piece 3 only: the B 32 miniature
#                                        that checks memory and SG1 coverage at this batch first; review MINOR-11)
# ORDER (review MINOR-9, docs/R2_SHAREDKV_PLAN.md section 6): per_row, probe-plant, probe, probe-whole, shared,
# shared-whole; stop at the first MUST failure. A probe result without a passed probe-plant is VOID (F2).
# The shared module is frozen into the record (git show <sha>:<path>, hash checked) and the gate loads that copy
# (review MINOR-7). [sharedkv2] NOSI is the r2-sharedkv2 worktree (the review fixes live there).
# [sharedkv3] NOSI is now the r2-sharedkv3 worktree; more arms with --union fused (docs/R2_SHAREDKV3_UNION.md):
#   probe-fused | probe-plant-fused     --attn probe / probe-plant --union fused, SG1, drop 0.0
#   shared-fused                        --attn shared --union fused, SG0-SG3, drop 0.5
#   order after the torch arms: probe-plant-fused, probe-fused, shared-fused.
# [sharedkv4] NOSI is the r2-sharedkv4 worktree. Claim (review MINOR-7): the OUT / REC / JIT existence check runs
# BEFORE the claim; the claim holds an owner file (arm, run, holder, step name, nosi, out, time); on a client timeout
# (124 / 137) the wrapper waits up to 10 min for the holder's steps to leave squeue, never freezes an output whose
# writer may run, and KEEPS the claim. Use a NEW run id per chain (the fused arms: not s1a).
# SOURCE ONLY, NOT RUN, NOT REVIEWED: it never submits, renews, cancels or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_sharedkv_step.sh --nosi <sha> <holder job> <run id> <arm>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_sharedkv_step.sh --nosi <sha> <holder job> <run id> <arm>
#        (login node: the pin checks and the printed step command; no squeue, claim, srun or write)
#   In DRYRUN only, TICK_HOLDERS=<dir> replaces HOLDERS, and TICK_PIN_BIATTN_SHA / TICK_PIN_BIATTN_BLOB replace the
#   piece-3 pins (the CPU test test_step_dryrun_refusals_and_pins uses them); a real run refuses all three.
#
# OUTPUT: OUTROOT/skv_<arm>_<run>_<holder>/summary.json and the step record OUTROOT/skv_<arm>_<run>_<holder>.step/,
# both frozen. EXIT: as r2_tick_step.sh (the gate's failed MUST count; 2 refusal; 3 / 4 GPU identity; 124 / 137
# timeout; 1 STOP before the launch; 5 an exit-0 step with an incomplete record or a non-COMPLETED 0:0 sacct row).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-sharedkv4                                                # ROOT (this branch)
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                        # ROOT (lead-owned uuid files)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (r2_g0_step.sh)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/sharedkv                                              # ROOT
BIATTN_SHA=7c826c1bc7235ee9153e3af82c99d85836915416                                      # PIN (piece 3 v2)
BIATTN_PATH=nosi/nosi/bi_paged_attention.py                                              # PIN
BIATTN_BLOB=39ed5e1894ca1662f92e384a10e01a4eaab99042                                     # PIN
SKV_PATH=nosi/nosi/bi_paged_attention_shared.py                                          # tracked at --nosi
GEMM=exact
BATCH=32
M_FIX=128
DELTAS=0,1,17,63,64,65,777,4097,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4,6,7,9,10,11,12,14
SPLITS=4
STEPS=96
SELF=benchmarks/Efficiency/r2_sharedkv_step.sh
STEP_MIN=90
MIN_LEFT=$((STEP_MIN * 60 + 600))
main() {
stop() { echo "R2 sharedkv step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_sharedkv_step.sh --nosi <sha> <holder> <run id> per_row|probe|probe-whole|probe-plant|shared|shared-whole|probe-fused|probe-plant-fused|shared-fused" >&2
          exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 ARM=$5
ATTN=$ARM UNION=torch
case $ARM in
    per_row|probe|probe-whole|probe-plant) PARTS=SG1 DROP=0.0 ;;
    shared|shared-whole) PARTS=SG0,SG1,SG2,SG3 DROP=0.5 ;;
    probe-fused|probe-plant-fused) PARTS=SG1 DROP=0.0 ATTN=${ARM%-fused} UNION=fused ;;      # [sharedkv3]
    shared-fused) PARTS=SG0,SG1,SG2,SG3 DROP=0.5 ATTN=shared UNION=fused ;;                  # [sharedkv3]
    *) usage ;;
esac
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
if [ -n "${TICK_HOLDERS:-}" ]; then
    [ "$DRY" = 1 ] || stop "TICK_HOLDERS is for DRYRUN (the CPU test) only"
    HOLDERS=$TICK_HOLDERS
fi
if [ -n "${TICK_PIN_BIATTN_SHA:-}${TICK_PIN_BIATTN_BLOB:-}" ]; then                     # the pin controls (CPU test)
    [ "$DRY" = 1 ] || stop "TICK_PIN_BIATTN_SHA / TICK_PIN_BIATTN_BLOB are for DRYRUN (the CPU test) only"
    BIATTN_SHA=${TICK_PIN_BIATTN_SHA:-$BIATTN_SHA}
    BIATTN_BLOB=${TICK_PIN_BIATTN_BLOB:-$BIATTN_BLOB}
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
for f in benchmarks/Efficiency/r2_tick_gate.py benchmarks/Efficiency/r2_g0bi_gate.py benchmarks/Efficiency/r2_g0_gate.py \
         benchmarks/Efficiency/r2_g0_compare.py nosi/nosi/r2_tick.py nosi/nosi/r2_bi_decode.py \
         nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py nosi/nosi/r2_positional_twin.py \
         nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py nosi/nosi/max_pooling_per_request.py "$SKV_PATH"; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ "$(git -C "$NOSI" rev-parse -q --verify "$BIATTN_SHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3: $BIATTN_PATH at $BIATTN_SHA is not the blob $BIATTN_BLOB"
[ "$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$BIATTN_PATH")" = "$BIATTN_BLOB" ] ||
    stop "piece 3 in this tree ($NSHA:$BIATTN_PATH) is not the pinned blob $BIATTN_BLOB"
SKV_BLOB=$(git -C "$NOSI" rev-parse -q --verify "$NSHA:$SKV_PATH") || stop "$SKV_PATH is not tracked at $NSHA"
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
OUT=$OUTROOT/skv_${ARM}_${RUN}_$JOB
REC=$OUT.step
JIT=$JITBASE/skv_${ARM}_${RUN}_$JOB
NAME=skv-${ARM}-${RUN}-$JOB
ATTN_MOD=$REC/piece3_bi_paged_attention.py
SKV_MOD=$REC/bi_paged_attention_shared.py
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=0
      --env NOSI_R2_DECODE=1 --env NOSI_MODEL_PATH="$MODEL" --env TICK_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 sharedkv child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 sharedkv child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem=120G --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" skv-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_tick_gate.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --biattn "$ATTN_MOD" --gemm "$GEMM" --batch "$BATCH" --S 16128 --deltas "$DELTAS"
      --steps "$STEPS" --sg0-steps 64 --K 3 --LAMBDA 7 --draft-drop "$DROP" --seed 7 --m-fix "$M_FIX"
      --sg4-steps 32 --phase 3 --parts "$PARTS" --attn "$ATTN" --union "$UNION" --sharedkv "$SKV_MOD")
if [ "$DRY" = 1 ]; then
    echo "R2 sharedkv step: DRYRUN (nothing claimed, written or launched); pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID ($UFILE)"
    echo "R2 sharedkv step: piece 3 $BIATTN_SHA:$BIATTN_PATH blob $BIATTN_BLOB -> $ATTN_MOD; shared $SKV_BLOB -> $SKV_MOD; arm $ARM parts $PARTS drop $DROP attn $ATTN union $UNION; B $BATCH M_FIX $M_FIX"
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
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] ||                # [sharedkv4 MINOR-7b] BEFORE the claim
    stop "$OUT, $REC or $JIT exists: a new run id (no claim taken)"
mkdir "$OUTROOT/skv_active" 2>/dev/null || stop "$OUTROOT/skv_active exists (another step, or an uncertain one; its owner file says which)"
printf 'arm=%s run=%s holder=%s step_name=%s nosi=%s out=%s utc_claim=%s\n' "$ARM" "$RUN" "$JOB" "$NAME" "$NSHA" \
    "$OUT" "$(date -u +%FT%TZ)" > "$OUTROOT/skv_active/owner" ||            # [sharedkv4 MINOR-7a] the claim's owner
    stop "cannot write $OUTROOT/skv_active/owner (skv_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (skv_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (skv_active stays)"
git -C "$NOSI" show "$BIATTN_BLOB" > "$ATTN_MOD" && [ "$(git hash-object -- "$ATTN_MOD")" = "$BIATTN_BLOB" ] ||
    stop "cannot copy piece 3's blob $BIATTN_BLOB to $ATTN_MOD (skv_active stays)"
git -C "$NOSI" show "$SKV_BLOB" > "$SKV_MOD" && [ "$(git hash-object -- "$SKV_MOD")" = "$SKV_BLOB" ] ||
    stop "cannot copy the shared module's blob $SKV_BLOB to $SKV_MOD (skv_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (skv_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s workload=%s utc_start=%s\n' "$NSHA" \
    "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$WL_SHA" "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (skv_active stays)"
printf 'biattn=%s:%s blob=%s copy=%s sharedkv=%s blob=%s copy=%s arm=%s parts=%s gemm=%s batch=%s m_fix=%s splits=%s steps=%s draft_drop=%s\n' \
    "$BIATTN_SHA" "$BIATTN_PATH" "$BIATTN_BLOB" "$ATTN_MOD" "$NSHA:$SKV_PATH" "$SKV_BLOB" "$SKV_MOD" "$ARM" "$PARTS" "$GEMM" "$BATCH" "$M_FIX" \
    "$SPLITS" "$STEPS" "$DROP" >> "$REC/command.txt" || stop "cannot write $REC (skv_active stays)"
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
    124|137)                                                  # [sharedkv4 MINOR-7c, C10] a timeout: the client
        ok=0                                                  # ended, the step may still write. Wait (bounded) until
        writer=running                                        # no step of the holder is listed; the claim STAYS
        for _ in $(seq 1 60); do                              # either way (a timeout proves nothing about the writer)
            if nosteps > /dev/null; then writer=gone_from_squeue; break; fi
            sleep 10
        done
        echo "timeout: writer=$writer after the wait; skv_active stays (L40 cleanup after sacct)" >> "$REC/exit.txt" ;;
esac
for t in "$OUT" "$REC"; do
    [ -d "$t" ] || continue
    [ "$t" = "$OUT" ] && [ "$writer" = running ] && continue       # never touch an output that may still be written
    find "$t" -type f -exec chmod 0444 {} + && find "$t" -type d -exec chmod 0555 {} + || ok=0
done
if [ "$ok" = 1 ]; then
    rm -f "$OUTROOT/skv_active/owner" && rmdir "$OUTROOT/skv_active" || ok=0
fi
echo "R2 sharedkv step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
