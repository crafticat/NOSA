#!/bin/bash
# Phase 2, rungs L3 and L4 (KS/analysis/plan/PLAN_speckv_on_cached_nosa.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): ONE
# bounded GPU step in an EXISTING holder that runs benchmarks/Efficiency/r2_spec_cell.py (the speculative round runner,
# nosi/nosi/spec_round.py, opt-in --spec K). A COPY of r2_base_rollback_step.sh (962cd1b): the same pins, checks,
# record and freeze; the ROOT, the output root, the names, the cells and the claim change. The claim is PER HOLDER
# (OUTROOT/sp_active_<holder>): two holders can run cells of this script at the same time (a global claim blocked
# the second holder for r2_draftpool_step.sh). ONE JOB, FOUR ARMS, each with its own shipped prefill:
#   SH   the shipped decode at the cell's C and B (cell mb1's flow at C 128 B 216): the same-job baseline
#   L3   per round: K = 3 resident-only drafts (undone), 3 serial verify steps (the shipped ragged step), accept / correct
#        / commit per request (greedy, no bonus token)
#   L4X  per round: ONE joint pass (3 verify rows + the draft row; EXACT, gemm rowgroup), truncate, 2 drafts, commit:
#        the token proof of L4
#   L4F  the same with the FAST joint pass (gemm natural): the timing of L4
# The pattern of benchmarks/Efficiency/r2_base_step.sh: the GPU uuid READ from the lead-owned file
# HOLDERS/<holder job>.gpu_uuid. SOURCE ONLY until reviewed: it never submits, renews, cancels or replaces an
# allocation, never retries.
#
#   bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder job> <run id> <sp3m|sp3a|sp3b|rg73|rb73>
#   DRYRUN=1 bash benchmarks/Efficiency/r2_spec_step.sh --nosi <sha> <holder job> <run id> <cell>
#        (login node: the pin checks, the uuid file read and the printed step command; no squeue, claim, srun or write)
#
# TWO DISCRIMINATING CELLS for the sp3b divergence (SCORING_sp3b.md section 8; docs/NOTES_spec.md section 11). They run
# benchmarks/Efficiency/r2_base_cell.py (not edited) at sp3b's C 73 (pool 10) and B 318, 512 steps, --warmup 2, with the
# FUSED twin and the fused body (--ragged-twin fused --ragged-body fused --ragged-gemm natural --ragged-graph on):
#   rg73  --ragged --ragged-spread 64: arms SH, RG, RGS. The PLAIN per-request step, no speculation, no rollback; every
#         RG / RGS logits fingerprint against SH's (prediction RG-BIT / RGS-BIT). Bitwise excludes "the plain ragged step
#         is not SH at pool 10". STEP_MIN 90, --mem 340G.
#   rb73  --rollback --rb-k 3 --rb-groups 128 --rb-seed 0 --rb-journal-blocks 256: arms SH, RBR, RBS. The ROLLBACK
#         only (random n_acc in 0 .. 3, no draft step): every kept row of RBS against RBR (prediction RB-BIT), with a
#         saturated journal (RB-OVF reports the overflow). STEP_MIN 90, --mem 340G.
#   Their registered predictions are r2_base_cell.PRED_RAGGED / PRED_ROLLBACK (stamped for rg1 / rb1 at C 128 B 216):
#   the bitwise ones apply here; the step-ratio and commit bands were set for C 128 B 216 (the lead decides).
#   Builder's expectation (INFERRED from the root cause in NOTES_spec.md section 11, not a registration): rg73 and rb73
#   are both bitwise. A MOVE_IN needs an empty window slot, and outside a draft only the first step after the prefill
#   has empty window slots (with an empty pool), so neither cell runs the undo of a MOVE_IN.
# Basis of STEP_MIN 90: dr73 (SH + DR at B 318) had STEP_MIN 90; sp3b (four prefills at B 318) took 01:13:19, about 18
# minutes per arm; three arms and 512 + 512 + 575 (rg73) or 512 + 384 + 384 (rb73) steps.
#
# CELLS (512 steps unless stated; --spec 3 --spec-arms L3,L4X,L4F --spec-journal-blocks 256 --spec-twin fused
# --spec-body fused --spec-graph on --spec-redirect on --spec-prefetch off --probe-every 16). The FUSED twin in every
# cell (SCORING_rg1.md: rg1f, the fused twin, +0.6 % / +1.7 % against cached NOSA with the logits bitwise SH's; rg1, the
# torch twin, +13-14 %, about 8 ms per step of small launches). The torch twin stays in the CPU tests' EXACT proofs.
#   sp3m  the MINIATURE: C 128 (pool 65), B 8, 64 steps. Run it first: it checks the wiring (every MUST gate, the token
#         checks) at a small cost. STEP_MIN 25, --mem 100G.
#   sp3a  C 128 (pool 65), B 216 (cell mb1). STEP_MIN 80, --mem 240G.
#   sp3b  C 73 (pool 10), B 318 (the paper's smaller cache with a larger batch; dr73's cell). STEP_MIN 115, --mem 340G.
# Basis of STEP_MIN: mb1 took 00:14:31 with a prefill of 657.7 s at B 216; four prefills (about 11 min each at B 216,
# about 16 min at B 318 by B), SH's decode (1.5 / 2 min) and three speculative decodes of ceil(512 / 3) = 171 rounds
# (about 0.4-0.5 s per round at B 216 by the dr1 / vr1 models: 1.5 min each).
# MEMORY (INFERRED): L3 adds the rollback journal of 3 steps per layer (about 10 MB per layer and step at 256 slots: 1.0
# GB) and the staging blocks (0.45 GB at B 216); L4 doubles the journal (6 steps: 2.0 GB) and adds the joint pass's
# E = 4 B rows (logits about 0.4 GB) and, FAST, the extras store (0.25 GB). mb1's allocator peak was 82.04 GB. If an arm
# stops on memory, the lead lowers --spec-journal-blocks (a lost slot past it comes back from the host: exact, slower;
# the journal overflow counter reports it).
#
# REGISTERED (TBD BY THE LEAD: fill r2_spec_cell.PRED_SPEC and its stamp from `date -u` in a commit BEFORE the run;
# until then every SP prediction is REPORT):
#   L3-TOK      L3's committed tokens bitwise SH's (and the committed rows' logits fingerprints): TBD
#   L4X-TOK     L4X's committed tokens bitwise SH's: TBD
#   L4F-AGREE   L4F's committed tokens equal to SH's, fraction >= TBD
#   L3-RATIO / L4X-RATIO / L4F-RATIO   median steady committed tok/s / SH steady tok/s inside (TBD, TBD)
#   SP-ACCEPT   L3's accepted / valid drafts inside (TBD, TBD)
#   L4F-VS-L3   L4F / L3 median steady committed tok/s inside (TBD, TBD) (the plan's L4 gate: faster than L3)
#   MUST gates (not predictions): SH-RUN, <arm>-RUN, <arm>-ERR, <arm>-COMMIT for L3, L4X, L4F; <arm>-TOK (committed
#   tokens bitwise SH's) and <arm>-FP (their logits fingerprints bitwise SH's) for the EXACT arms L3, L4X.
#   rg73 / rb73: the MUST gates of r2_base_cell.py (SH-RUN, RG / RGS or RBR / RBS -RUN, -ERR, -CAP).
#
# OUTPUT: OUTROOT/sp_<run>_<holder>_<cell>/summary.json and the step record OUTROOT/sp_<run>_<holder>_<cell>.step/
# (command + pins incl. the uuid and its file, stdout, stderr, exit + accounting, sacct row), both frozen. EXIT: the
# cell's exit = the number of failed MUST gates (0-19); 20 the cell refused; 30 the cell raised; 2 a usage error of
# this script; 3 another GPU; 4 not exactly one GPU; 124 / 137 the host timeout; 1 a STOP before the launch; 5 an
# exit-0 step whose record is incomplete or whose sacct row is not COMPLETED 0:0.
# Checks before the claim (each STOPS with exit 1): as r2_base_step.sh. The claim: OUTROOT/sp_active_<holder> (mkdir,
# atomic); a stop after the claim leaves it (the lead decides).
set -u
NOSI=/mnt/beegfs/ojerbi/nosi-r2-spec                                                     # ROOT
HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders                                         # ROOT (lead-owned)
SIF=/mnt/beegfs/ojerbi/nosa-env/nosa-env.sif                                             # ROOT
APPTAINER=/usr/bin/apptainer                                                             # ROOT
MODEL=/mnt/beegfs/ojerbi/models/NOSA-8B                                                  # ROOT
MODEL_SIZE=16370581512                                                                   # PIN (nosi_pool.sbatch)
WL=/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-c16k                              # ROOT
WL_SHA=a25d1e16592c72d40550a3a05a2ac7673df0b57ce7fb1e9293f1451de4c8a0ce                   # PIN (c16k_tokens.bin)
JITBASE=/mnt/beegfs/ojerbi/nosa-env/jit                                                  # ROOT
OUTROOT=/mnt/beegfs/ojerbi/r2-runs/spec                                                  # ROOT
SELF=benchmarks/Efficiency/r2_spec_step.sh
SPLITS=4
main() {
stop() { echo "R2 spec step: STOP: $*" >&2; exit 1; }
usage() { echo "usage: r2_spec_step.sh --nosi <sha> <holder> <run id> <sp3m|sp3a|sp3b|rg73|rb73>" >&2; exit 2; }
[ $# = 5 ] && [ "$1" = --nosi ] || usage
NSHA=$2 JOB=$3 RUN=$4 CELL=$5
DRY=${DRYRUN:-0}
[[ $NSHA =~ ^[0-9a-f]{40}$ ]] || stop "--nosi '$NSHA' is not a full commit id"
case $JOB in *[!0-9]*|"") stop "holder job id '$JOB'" ;; esac
[[ $RUN =~ ^[a-z0-9]{1,16}$ ]] || stop "run id '$RUN' (1-16 of a-z 0-9)"
K=3 ARMS=L3,L4X,L4F KIND=spec
case $CELL in
    sp3m) POOL=65 BATCH=8 STEPS=64 STEP_MIN=25 MEM=100G ;;
    sp3a) POOL=65 BATCH=216 STEPS=512 STEP_MIN=80 MEM=240G ;;
    sp3b) POOL=10 BATCH=318 STEPS=512 STEP_MIN=115 MEM=340G ;;
    rg73) POOL=10 BATCH=318 STEPS=512 STEP_MIN=90 MEM=340G KIND=ragged ARMS=RG,RGS ;;
    rb73) POOL=10 BATCH=318 STEPS=512 STEP_MIN=90 MEM=340G KIND=rollback ARMS=RBR,RBS ;;
    *) stop "cell '$CELL' (sp3m, sp3a, sp3b, rg73 or rb73)" ;;
esac
ENG=(--ragged-twin fused --ragged-body fused --ragged-gemm natural --ragged-graph on)
case $KIND in
    spec) PY=r2_spec_cell.py DESC="arms SH+$ARMS steps $STEPS K $K"
          SPK=(--spec "$K" --spec-arms "$ARMS" --spec-journal-blocks 256 --spec-twin fused --spec-body fused
               --spec-graph on --spec-redirect on --spec-prefetch off --probe-every 16) ;;
    ragged) PY=r2_base_cell.py DESC="arms SH+RG+RGS steps $STEPS twin fused (no speculation)"
          SPK=(--warmup 2 --r2 fused --arms SH --ragged --ragged-spread 64 "${ENG[@]}") ;;
    rollback) PY=r2_base_cell.py DESC="arms SH+RBR+RBS steps $STEPS K $K groups 128 journal 256 twin fused (no draft)"
          SPK=(--warmup 2 --r2 fused --arms SH --rollback --rb-k "$K" --rb-groups 128 --rb-seed 0 --rb-journal-blocks 256
               "${ENG[@]}") ;;
esac
MIN_LEFT=$((STEP_MIN * 60 + 600))
for v in $(compgen -e); do case $v in SLURM_*|SRUN_*|SALLOC_*|SBATCH_*) stop "the environment exports $v" ;; esac; done
[ "$(git -C "$NOSI" rev-parse HEAD)" = "$NSHA" ] || stop "$NOSI HEAD is not $NSHA"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=no)" ] || stop "$NOSI has tracked changes"
[ -z "$(git -C "$NOSI" status --porcelain --untracked-files=all -- nosi)" ] || stop "$NOSI has untracked files in nosi/"
blob=$(git -C "$NOSI" rev-parse "$NSHA:$SELF") || stop "$SELF is not tracked at $NSHA"
[ "$(git hash-object -- "$0")" = "$blob" ] || stop "$0 is not the blob $SELF of $NSHA ($blob)"
for f in benchmarks/Efficiency/r2_spec_cell.py benchmarks/Efficiency/r2_base_cell.py \
         benchmarks/Efficiency/r2_ragged_gate.py nosi/nosi/spec_round.py nosi/nosi/ragged_engine.py \
         nosi/nosi/ragged_rollback.py nosi/nosi/verify_rows.py nosi/nosi/r2_decode_path.py nosi/nosi/r2_decode_body.py \
         nosi/nosi/r2_positional_twin.py nosi/nosi/r2_writeback.py nosi/nosi/r2_twin_fused.py nosi/nosi/r2_body_fused.py \
         nosi/nosi/r2_bi_decode.py nosi/nosi/r2_spec_core.py nosi/nosi/stage1_seqused.py \
         nosi/nosi/max_pooling_per_request.py nosi/nosi/cache_engine.py nosi/nosi/flash_cache_engine/pool_update.cpp \
         nosi/nosi/flash_cache_engine/pool_update_kernel.cu nosi/nosi/flash_cache_engine/flash_pool_swap.py \
         nosi/nosi/flash_cache_engine/flash_h2d_mask.py nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py; do
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
OUT=$OUTROOT/sp_${RUN}_${JOB}_$CELL
REC=$OUT.step
JIT=$JITBASE/sp_${RUN}_${JOB}_$CELL
NAME=sp-${CELL}-${RUN}-$JOB
CLAIM=$OUTROOT/sp_active_$JOB
ENVS=(--env PYTHONPATH="$NOSI/nosi" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1
      --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TORCH_CUDA_ARCH_LIST=8.0
      --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR="$JIT/triton"
      --env TORCH_EXTENSIONS_DIR="$JIT/torch_ext" --env FLASHINFER_WORKSPACE_BASE="$JIT/flashinfer"
      --env TVM_FFI_CACHE_DIR="$JIT/tvm_ffi" --env TMPDIR="$JIT/tmp"
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env NOSI_ATTN_SPLITS=$SPLITS --env NOSI_POOL_BLOCKS=$POOL
      --env NOSI_R2_DECODE=0 --env NOSI_MODEL_PATH="$MODEL" --env BASE_COMMIT="$NSHA")
CHILD='case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|*,*) echo "R2 spec child: CUDA_VISIBLE_DEVICES must name exactly one GPU (is \"${CUDA_VISIBLE_DEVICES:-}\")" >&2
            exit 4 ;;
esac
U=$1
shift
seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "$seen" = "GPU-$U" ] || { echo "R2 spec child: the visible GPU is \"$seen\", not GPU-$U" >&2; exit 3; }
A=$1
shift
exec "$A" exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"'
ARGV=(timeout --verbose --signal=TERM --kill-after=15 $((STEP_MIN * 60 + 60)) srun --jobid="$JOB" --job-name="$NAME"
      --ntasks=1 --cpus-per-task=16 --mem="$MEM" --gres=gpu:a100:1 --time="$STEP_MIN" --export=NONE
      /bin/sh -c "$CHILD" sp-child "$GPU_UUID" "$APPTAINER" --bind /mnt/beegfs/ojerbi:/mnt/beegfs/ojerbi "${ENVS[@]}"
      "$SIF" /venv/nosa/bin/python "$NOSI/benchmarks/Efficiency/$PY" --out "$OUT" --model "$MODEL"
      --workload "$WL" --batch "$BATCH" --steps "$STEPS" --pool "$POOL" "${SPK[@]}")
if [ "$DRY" = 1 ]; then
    echo "R2 spec step: DRYRUN (nothing claimed, written or launched); cell $CELL; pins OK: nosi $NSHA blob $blob workload $WL_SHA gpu_uuid $GPU_UUID (from $UFILE)"
    echo "R2 spec step: cell $CELL: C $((63 + POOL)) (pool $POOL) B $BATCH $DESC STEP_MIN $STEP_MIN mem $MEM claim $CLAIM"
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
mkdir "$CLAIM" 2>/dev/null || stop "$CLAIM exists (another step of this holder, or an uncertain one)"
[ ! -e "$OUT" ] && [ ! -e "$REC" ] && [ ! -e "$JIT" ] || stop "$OUT, $REC or $JIT exists: a new run id ($CLAIM stays)"
mkdir "$REC" && mkdir -p "$JIT/tmp" || stop "cannot create $REC / $JIT ($CLAIM stays)"
busy=$(nosteps) || stop "a step is running in $JOB under the claim: '$busy' ($CLAIM stays)"
printf '%q ' "${ARGV[@]}" > "$REC/command.txt" || stop "cannot write $REC/command.txt ($CLAIM stays)"
printf '\nnosi=%s wrapper_blob=%s gpu_uuid=%s gpu_uuid_file=%s holder=%s left=%s run=%s cell=%s kind=%s pool=%s batch=%s arms=%s steps=%s k=%s workload=%s utc_start=%s\n' \
    "$NSHA" "$blob" "$GPU_UUID" "$UFILE" "$JOB" "$sl" "$RUN" "$CELL" "$KIND" "$POOL" "$BATCH" "$ARMS" "$STEPS" "$K" "$WL_SHA" \
    "$(date -u +%FT%TZ)" >> "$REC/command.txt" ||
    stop "cannot write $REC ($CLAIM stays)"
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
echo "R2 spec step: exit $rc; record $REC; output $OUT; record complete and frozen: $ok"
[ "$rc" = 0 ] && [ "$ok" = 0 ] && exit 5
exit "$rc"
}
main "$@"; exit $?
