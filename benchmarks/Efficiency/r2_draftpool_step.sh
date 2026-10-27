#!/bin/bash
# r2-draftpool: ONE bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_base_cell.py with
# --draft-resident (Task 1.1 of kstep-cpu-runs/analysis/plan/PLAN_speckv_on_cached_nosa.md): the cached-NOSA baseline
# (cell mb1 of r2_base_fused_step.sh: C 128 = 63 + pool 65, S 16128, the PG19 94-document workload) as the SH arm, then
# the DR arm in the SAME process: the same engine with the draft-resident flag ON (cache_engine.py, the r2-draftpool
# block), teacher-forced on SH's tokens. The pattern of benchmarks/Efficiency/r2_base_fused_step.sh: the GPU uuid READ
# from the lead-owned file HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels
# or replaces an allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_draftpool_step.sh --nosi <sha> <holder job> <run id> <dr1|dr1m>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_draftpool_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# CELLS (host rows S + N + 64 per request; arms SH then DR; --dr-k 3 --dr-gate0 4):
#   dr1   C 128 (P 65) B 216, 512 steps, --mbase, --dr-bands "$DR1_BANDS"; STEP_MIN 60, --mem 240G.
#         It STOPS outside DRYRUN while DR1_BANDS is TBD (the lead fills and stamps the REGISTERED block first).
#   dr1m  the miniature: C 128 (P 65) B 8, 80 steps (one edge step, t 63, and one re-warm, t 64), no bands, no --mbase;
#         STEP_MIN 20, --mem 100G. Run it first: it checks the wiring (DR-RUN, DR-G0) at a small cost.
# OUTPUT: OUTROOT/drp_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/drp_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): as r2_base_fused_step.sh. The claim: OUTROOT/drp_active (mkdir,
# atomic); a stop after the claim leaves it (the lead decides).
#
# REGISTERED for dr1 / dr1m ([draftpool] Task 1.1, STAMP 2026-10-27T16:18:30Z by the lead, from `date -u`, before any
# run, L33; DR1_BANDS = DLO:DHI:RLO:RHI). WHAT:
# SH = mb1 (the same-job baseline). DR = a new cache of the same engine, the flag ON after the prefill, teacher-forced
# on SH's tokens; kinds: rewarm and t < 4 (g0) shipped, edge (the tail block fills) shipped, then 3 draft steps and 1
# shipped step (ship). A draft step skips the two host gathers and masks the slots that still need the host.
# BASIS for the lead (MAP_cached_nosa.md section 2.1: ncu step 62 of job 2175791, C 128 B 216, 64.790 ms of kernels):
# gather 7.585 ms, pool swap 5.447 ms, resolve 1.178 ms, engine copies 0.586 ms; mb1 SH steady 63.761 ms (2190876).
# The plan's estimate (Task 1.1): the draft step = shipped - gather - part of the swap, about 52-57 ms. The draft step
# still runs the swap; it adds 4 small launches per layer (compare, two masked fills, the bias copy of 3.5 MB).
#   DR-G0     MUST: the DR steps 0 .. 3 (flag on, shipped) give SH's next tokens, 216 x 4 of 216 x 4.
#             PREDICTION: PASS
#   DR-RUN    MUST: 512 steps, at least one draft step. PREDICTION: PASS
#   DR-STEP   the draft step median (ms) inside DLO:DHI. PREDICTION: 57 ms (63.8 - gather 7.6 + about 1.3 of the
#             4 extra launches per layer), band [50, 62]
#   DR-RATIO  draft median / SH steady median inside RLO:RHI. PREDICTION: 0.89, band [0.78, 0.97]. FALSIFIER: a
#             ratio above 0.97 means the gather is not the cost we think it is (read the stage timers of Task 0.1).
#   DR-SHIP   REPORT: the ship step median (after 3 drafts) / SH steady median. PREDICTION: 1.2 (it loads the misses
#             of 3 drafts), range [1.0, 1.5]
#   DR-SKIP   REPORT: skipped slots per (layer, KV head, request) per draft step. PREDICTION: 1-4, growing over the
#             3 drafts of a group
#   DR-AGREE  REPORT: teacher-forced agreement of the draft steps with SH's tokens. PREDICTION: 0.93-0.98 (R2 ev1
#             acceptance 0.9586)
#   DR-MEM    REPORT: DR peak allocated - SH peak allocated (GB). PREDICTION: below 0.1 GB (one 3.5 MB bias scratch)
#   MB-STEP   REPORT (--mbase, as mb1): the SH steady step. PREDICTION: 63-66 ms (mb1 measured 63.761); the old band
#             (64.92, 70.39) is kept for reference only
#   dr1m      DR-RUN and DR-G0 PASS; kinds include edge at t 63 and rewarm at t 64. PREDICTION: PASS
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-draftpool                                                # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/draftpool                                             # ROOT
SELF=benchmarks/Efficiency/r2_draftpool_step.sh
SPLITS=4
DR1_BANDS=50:62:0.78:0.97         # REGISTERED DR-STEP / DR-RATIO (STAMP 2026-10-27T16:18:30Z)
# REGISTERED for dr73 / dr81 (STAMP 2026-10-27T17:58:39Z by the lead, from `date -u`, before any run; L33). WHY: dr1
# (C 128 B 216) measured a draft saving of only 5.3 % (SCORING_dr1.md): the fetch is small there. The paper's lever is
# a smaller cache with a larger batch (paper Figure 2: B 318 at C 73, B 290 at C 81; fetch / rest about 1.36 at C 63 and
# 0.35 at C 81). Same flow as dr1 (SH = the shipped engine at the SAME C and B, then DR), no --mbase band.
#   DR73-STEP / DR73-RATIO  PREDICTION: draft / SH about 0.80, band [0.60, 0.95]; draft step band [70, 170] ms
#   DR81-STEP / DR81-RATIO  PREDICTION: draft / SH about 0.87, band [0.70, 0.97]; draft step band [60, 140] ms
#   FALSIFIER (both): a ratio above 0.97 means a smaller cache does not make the draft pay more than at C 128.
DR73_BANDS=70:170:0.60:0.95
DR81_BANDS=60:140:0.70:0.97
main() {
stop() { echo "R2 draftpool step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_draftpool_step.sh --nosi <sha> <holder> <run id> <dr1|dr1m|dr73|dr81>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
POOL=65 MB=() BANDARG=()
case $CELL in
    dr1) BATCH=216 STEPS=512 STEP_MIN=60 MEM=240G MB=(--mbase) BANDS=$DR1_BANDS
        [ "$DRY" = 1 ] || [ "$BANDS" != TBD ] ||
            stop "cell $CELL: --dr-bands TBD (the lead fills and stamps the REGISTERED dr1 block first)"
        BANDARG=(--dr-bands "$BANDS") ;;
    dr1m) BATCH=8 STEPS=80 STEP_MIN=20 MEM=100G BANDS=none ;;
    dr73) POOL=10 BATCH=318 STEPS=512 STEP_MIN=90 MEM=340G BANDS=$DR73_BANDS                  # [lead] paper Fig. 2
        BANDARG=(--dr-bands "$BANDS") ;;
    dr81) POOL=18 BATCH=290 STEPS=512 STEP_MIN=90 MEM=340G BANDS=$DR81_BANDS                  # [lead] paper Fig. 2
        BANDARG=(--dr-bands "$BANDS") ;;
    *) stop "cell '$CELL' (dr1, dr1m, dr73 or dr81)" ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_base_cell.py benchmarks/Efficiency/r2_ragged_gate.py \
         benchmarks/Efficiency/r2_g0_compare.py nosi/nosi/cache_engine.py nosi/nosi/nosa_llama.py \
         nosi/nosi/avail_policy.py nosi/nosi/transfer_trace.py \
         nosi/nosi/flash_cache_engine/pool_update.cpp nosi/nosi/flash_cache_engine/pool_update_kernel.cu \
         nosi/nosi/flash_cache_engine/flash_pool_swap.py nosi/nosi/flash_cache_engine/flash_h2d_mask.py \
         nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py; do
    git -C "$NOSI" rev-parse -q --verify "$NSHA:$f" > /dev/null || stop "$f is not tracked at $NSHA"
done
[ -f "$SIF" ] || stop "no image $SIF"
[ "$(stat -c %s "$MODEL/model.safetensors" 2>/dev/null)" = "$MODEL_SIZE" ] || stop "model size is not $MODEL_SIZE"
wl=$(sha256sum "$WL/c16k_tokens.bin" 2>/dev/null) || stop "no $WL/c16k_tokens.bin"
[ "${wl%% *}" = "$WL_SHA" ] || stop "c16k_tokens.bin sha256 is not $WL_SHA"
UFILE=$HOLDERS/$JOB.gpu_uuid
[ -f "$UFILE" ] && [ -r "$UFILE" ] || stop "no readable $UFILE (the lead writes it for each holder)"
GPU_UUID=$(cat -- "$UFILE") || stop "cannot read $UFILE"
[[ $GPU_UUID =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    stop "$UFILE does not hold exactly one uuid (got '$GPU_UUID')"
OUT=$OUTROOT/drp_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/drp_${RUN}_${JOB}_$CELL
NAME=drp-${CELL}-${RUN}-$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 draftpool child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 draftpool child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" drp-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/r2_base_cell.py" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --arms SH --pool "$POOL" --draft-resident --dr-k 3
      --dr-gate0 4 "${BANDARG[@]}" "${MB[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 draftpool step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 draftpool step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH arms SH,DR steps $STEPS STEP_MIN $STEP_MIN mem $MEM bands $BANDS"
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
    *:*) left=$(( 10#${sl%%:*} * 60 )) ;;                       # squeue prints MM:SS below one hour
esac
[ "$left" -ge "$MIN_LEFT" ] || stop "holder $JOB has $sl left (< $MIN_LEFT s)"
nosteps() {
    steps=$(timeout 60 squeue -s -j "$JOB" -h -o '%i') || return 1
    for s in $steps; do case $s in *.batch|*.extern) ;; *) echo "$s"; return 1 ;; esac; done
}
busy=$(nosteps) || stop "a step is running in $JOB (or squeue -s failed): '$busy'"
mkdir -p "$OUTROOT" || stop "cannot create $OUTROOT"
mkdir "$OUTROOT/drp_active" 2>/dev/null || stop "$OUTROOT/drp_active exists (another step, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id (drp_active stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT (drp_active stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' (drp_active stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt (drp_active stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s pool=%s batch=%s steps=%s bands=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$POOL" "$BATCH" "$STEPS" "$BANDS" "$WL_SHA" \
    "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC (drp_active stays)"
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
    rmdir "$OUTROOT/drp_active" || ok=0
fi
echo "R2 draftpool step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
