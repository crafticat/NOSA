"""CPU-PACKING TRANSPORT beside NOSI's resident decode (authorized 2026-09-29, the user via Codex; ledger 'CPU-PACKING
TRANSPORT EXPERIMENT AUTHORIZED'). Measurement harness only: no production-baseline change.

CONFIRMATION (feature/nosi-cpupack-c; Codex's review of job 2179683, relayed by the user 2026-10-01; cpupack_golden.py has
the mechanism and its proof):
  * NO DESTRUCTIVE OPERATION IN A MEASUREMENT PROCESS. The logits negative control (a second decode without a restore,
    2179683's cpupack_transport.py:961) is gone from correct_stage; it and the other destructive controls (window row, tail
    row, block map, compression state, counters, the guard itself) run in the separate CONTROLS process (CP_MODE=controls,
    run_controls), first; its failure stops the job. Every decode goes through Runner.decode -> DecodeGuard.check: inside
    a gated step only the flushed reference and restored steps are legal (UnsanctionedDecode otherwise, before the model
    runs). tail_write_real puts back the host rows it writes.
  * GOLDEN GATE (CP_CONFIRM=1, run_confirm). One process: prefill -> PostPrefillSnapshot (hosted in pageable memory) ->
    [CAPTURE: the 63 natural steps, verified against the SAVED 2179683 plans (CP_SAVED_PLANS_NPZ, CP_SAVED_PLANS_SHA256)]
    -> restart -> GOLDEN pass (the same gated sequence with no transports and no controls; exported) -> restart ->
    MEASURED pass. Every restart must reproduce the post-prefill start digest, and every pass's warm natural steps the
    first pass's logits and full state digests (exit 20 otherwise = fallback (ii): CP_STAGES=GOLDEN in one process,
    CP_GOLDEN_JSON in the next). At every gated step BEFORE any timing: reference logits sha256, post-step maps, tail,
    counters, compression state and window == golden, and the pre-timing resident check (restore -> decode: 0 loads,
    logits torch.equal the reference); after the advance: advance logits sha256 and the post-advance state == golden.
    A mismatch is GATE_FAIL: no timing at that step (pre) / its rows excluded (post), counted as a failure.
  * C0 runs under inference mode (2179683 defect 2); TABLE keeps only ok rows of gate-passing steps, streams the
    per-request CSV per group through gzip and reports every exclusion with its denominator (defect 3).

QUESTION. GPU produces the missing-KV list -> CPU packs scattered KV into contiguous pinned staging -> bulk DMA -> GPU places
the data in the required slots. Does it keep useful transfer speed while reducing decode interference, and where is the cost
(list delivery, descriptor preparation, packing, DMA, placement)?

LABELS. 'transport/placement-ready, NOT live-LRU-ready' (no last-reader / version / publication mechanism exists at
dc818e4: natural plans are replayed into a SEPARATE scratch with the cache's physical geometry; safe overwrite and
publication are NOT measured). 'GPU-resident trace replay' (all 32 layer plans become ready at the gate; not paced by the
decode). Prepacked / contiguous / scatter-only arms are DIAGNOSTIC CEILINGS.

STAGES (CP_STAGES; one process per batch; B336 primary, B64 comparison, B320 registered fallback = the sbatch reruns ALL arms):
  CAPTURE   natural chronological plans: prefill -> PostPrefillSnapshot -> CP_NCAP (63 = the whole no-rollover budget at
            L % 64 == 0) natural decode steps under an int16 AloneTrace (verify_alone.py:368 + transfer_trace.record_mask
            :201-209), ONE harvest at the end, a per-step cross-check archive == engines' _load_mask / _block_map, invariants
            I1-I9, canonical hashes, npz export; GPU-resident int32 plan store; restore the snapshot. CP_PLANS_NPZ loads an
            export instead (two-process fallback / the timeline process); either way the warm natural steps 0..WARM-1 of the
            replay must reproduce the captured logits and masks (the hygiene gate, exit 20 otherwise).
  CORRECT   at the first gated step: plan store == capture; descriptors == nosi_items(i256) (orig layout); every layer exact
            (direct shipped gather flash_h2d_from_mask, W8, CPU pack) against logical references with a NaN-poisoned scratch
            and exact canary rows; the step-0 full list; synthetic empty / one-full-stream / duplicate-source / random
            per-head plans; negative controls (skipped scatter, poisoned staged row, missing staging wait, missing landing
            wait, missing scatter-on-its-H2D wait, logits without restore) must be DETECTED; positive lifetime controls
            must pass. The lifetime controls run on the step's layer with the most groups, with the chunk cap lowered until
            there are more chunks than ring slots (n_chunks recorded; UNTESTABLE is a failure). The warm-up rep of every arm
            is checked and counted like the timed reps. A failure skips MAIN, LAYOUT and TIMELINE (exit 21).
  MAIN      untraced, gated steps CP_STEPS (plans[it] = the natural plan of the same step index), per arm REPS transfer-alone
            and REPS beside the resident decode (worker_sweep.py:491-514 light restore; 0 loads and logits torch.equal every
            rep), decode-alone reps before and after. Arms: w8 (flash_h2d_persistent n_ctas=8, num_warps=4,
            bypass_cache=False, K and V), cpu1/2/4/8 (pack on N physical cores), cpu8_c16 (16 MiB chunks), cpu8_lite (events
            lite), cpu8_L / cpu8_LP / cpu8_LPD (cumulative ablations), cpu8_prepacked, ceil_contig, scatter_only (ceilings).
            Then SWEEP: transfer-alone w8 and cpu8 over every captured step (steady 1..62 and the step-0 full list).
  LAYOUT    (layout_ablation; runs AFTER the original arms' JSON is flushed and the main-done marker is written; isolated:
            any exception is a layout failure, never an exit 20..24) the 2x2 of {orig, head-major} CPU SOURCE x {orig,
            head-major} GPU SCRATCH, same natural plans (CP_LAYOUT_PLAN_STEPS), W8 and cpu8 variants (row / group packer x
            row / group placer), each alone and beside the decode, with CP_LAYOUT_REPS decode-alone reps before and after the
            variants at EVERY (cell, plan step) gated decode step (the same-step slowdown denominator). The host cache is
            reordered IN PLACE (no second host cache, no extra pinned memory; cpupack_core.convert_inplace) between the
            orig-source and head-major-source cells, with a bit-exact per-request logical check and a flushed-reference
            logits check before == after; the conversion is PREDICTED first (cpupack_core.probe_convert + C0) and guarded per
            layer against the deadline. The resident decode's own GPU window layout is untouched. Plus the measured
            tail-write cost per layout (small buffer, and the real cache's never-read rows [S_cpu-64, S_cpu)) and the
            one-time conversion cost (reorder and verification separately). A rollover tripwire guards every gated step.
  TIMELINE  (separate process under nsys --capture-range=cudaProfilerApi) alone, alone_late (negative control), w8, cpu4,
            cpu8 beside the decode: untraced, traced, untraced reps with NVTX 'lt|b<B>|<arm>|step<it>|<phase>|rep<k>'.
MODES (CP_MODE): run | calib (node-local CPU pack calibration, no model) | table.
EXIT: 0 ok; 1..19 failure count; 20 hygiene gate (two-process fallback); 21 correctness failed; 22 memory fallback
(OOM, pinned allocation failure, free HBM below need, peak reserved above CP_PEAK_LIMIT_GB) -- only BEFORE the main-done
marker; 23 an uncaught exception (a crash, never a failure count); 24 CPU placement refused (before the model is loaded).
MAIN-DONE MARKER: <CP_OUT>/<CP_TAG>.main_done is written once MAIN + SWEEP are flushed and passed the memory gate, before
LAYOUT. From then on nothing returns 20..22: a LAYOUT exception only adds to layout_fails, and the sbatch keeps a batch whose
marker exists (no B320 rerun) whatever the exit code (a timeout inside LAYOUT included).
DEADLINE: CP_STAGE_DEADLINE (epoch s, from the sbatch: the stage's timeout bound - 3 min) bounds SWEEP and LAYOUT; LAYOUT
also stops at CP_LAYOUT_BUDGET_S, skips a step that its observed step time does not fit, and skips the head-major cells when
the PREDICTED in-place conversion (probe of 3 requests + C0's measurement, x CP_CONV_SAFETY) plus one step does not fit.
"""
import csv
import gc
import gzip
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import traceback

import cpupack_cpu as CC

MODE = os.environ.get("CP_MODE", "run")   # run | calib | table | controls
EARLY = None
if __name__ == "__main__" and MODE in ("run", "calib"):
    _gn = os.environ.get("CP_GPU_NUMA_NODE", "").strip()
    EARLY = CC.early_placement(int(os.environ.get("CP_TEAM_MAX", "8")), int(_gn) if _gn.isdigit() else None)

os.environ.setdefault("NOSI_ALONE_MODE", "points")
for _src, _dst in (("CP_B", "NOSI_ALONE_B"), ("CP_L", "NOSI_ALONE_L"), ("CP_NCAP", "NOSI_ALONE_N"), ("CP_WARM", "NOSI_ALONE_WARM"),
                   ("CP_OUT", "NOSI_ALONE_OUT")):
    if _src in os.environ:
        os.environ[_dst] = os.environ[_src]
os.environ.setdefault("NOSI_ALONE_N", "63")
os.environ.setdefault("NOSI_ALONE_DOCS", "1")
os.environ.setdefault("NOSI_ALONE_D", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cpupack_core as K  # noqa: E402
import cpupack_golden as G  # noqa: E402
import cpupack_plans as PL  # noqa: E402
import cpupack_timing as TM  # noqa: E402
import numa_maps as NM  # noqa: E402
import verify_alone as VA  # noqa: E402

STAGES = tuple(os.environ.get("CP_STAGES", "CAPTURE CORRECT MAIN LAYOUT").split())
STEPS = tuple(int(x) for x in os.environ.get("CP_STEPS", "4 5 6 7 8 9 10 11").split())
REPS = int(os.environ.get("CP_REPS", "5"))
CORES = tuple(sorted(int(x) for x in os.environ.get("CP_CORES", "1 2 4 8").split()))
CHUNK_KB = int(os.environ.get("CP_CHUNK_KB", "8192"))
CHUNK_KB_ALT = int(os.environ.get("CP_CHUNK_KB_ALT", "16384"))
RING = int(os.environ.get("CP_RING", "3"))
SLEEP_MS = float(os.environ.get("CP_SLEEP_MS", "50"))
ARMS_ONLY = tuple(os.environ.get("CP_ARMS", "").split())
SWEEP = os.environ.get("CP_SWEEP", "1") == "1"
LAYOUT_PLAN_STEPS = tuple(int(x) for x in os.environ.get("CP_LAYOUT_PLAN_STEPS", "4 5 6 7").split())
LAYOUT_REPS = int(os.environ.get("CP_LAYOUT_REPS", "3"))
LAYOUT_BUDGET_S = float(os.environ.get("CP_LAYOUT_BUDGET_S", "900"))
STAGE_DEADLINE = float(os.environ.get("CP_STAGE_DEADLINE", "0") or 0) or None   # epoch s: the sbatch's timeout bound - 3 min
FINAL_RESERVE_S = float(os.environ.get("CP_FINAL_RESERVE_S", "60"))      # left for the last inventory + JSON flush
LAYOUT_STEP_EST_S = float(os.environ.get("CP_LAYOUT_STEP_EST_S", "90"))  # first LAYOUT step estimate (then the observed max)
SWEEP_STEP_EST_S = float(os.environ.get("CP_SWEEP_STEP_EST_S", "10"))
CALIB_JSON = os.environ.get("CP_CALIB_JSON", "")                           # C0's calib_*.json (conversion-time predictor)
CONV_SAFETY = float(os.environ.get("CP_CONV_SAFETY", "1.5"))
LAYOUT_BACK = os.environ.get("CP_LAYOUT_BACK", "0") == "1"
LAYOUT_CHECK_LAYERS = tuple(int(x) for x in os.environ.get("CP_LAYOUT_CHECK_LAYERS", "0 15 16 31").split())
PLANS_NPZ = os.environ.get("CP_PLANS_NPZ", "")
NO_SNAPSHOT = os.environ.get("CP_NO_SNAPSHOT", "0") == "1"
T_ARMS = tuple(os.environ.get("CP_T_ARMS", "alone alone_late w8 cpu4 cpu8").split())
T_STEPS = tuple(int(x) for x in os.environ.get("CP_T_STEPS", "4 5").split())
T_REPS = int(os.environ.get("CP_T_REPS", "3"))
T_UNTRACED = int(os.environ.get("CP_T_UNTRACED", "3"))
T_LATE_MS = float(os.environ.get("CP_T_LATE_MS", "2.0"))
MEM_MARGIN_GB = float(os.environ.get("CP_MEM_MARGIN_GB", "1.0"))
PEAK_LIMIT_GB = float(os.environ.get("CP_PEAK_LIMIT_GB", "84.1"))
CONTIG_BYTES = int(os.environ.get("CP_CONTIG_MB", "256")) << 20
LIFETIME_CAP = int(os.environ.get("CP_LIFETIME_CAP_GROUPS", "16"))
LIFETIME_DELAY_MS = float(os.environ.get("CP_LIFETIME_DELAY_MS", "20"))
# ---- the confirmation (module docstring 'CONFIRMATION')
CONFIRM = os.environ.get("CP_CONFIRM", "0") == "1"
SAVED_PLANS_NPZ = os.environ.get("CP_SAVED_PLANS_NPZ", "")                 # the 2179683 plans (read-only)
SAVED_PLANS_SHA256 = os.environ.get("CP_SAVED_PLANS_SHA256", "")           # its expected file sha256 ('' = not checked)
GOLDEN_JSON = os.environ.get("CP_GOLDEN_JSON", "")                         # fallback (ii): gate against this export
XGOLDEN_JSON = os.environ.get("CP_XGOLDEN_JSON", "")                       # another process's golden: a determinism record
ARM_ORDER = os.environ.get("CP_ARM_ORDER", "fixed")                        # fixed | alternate (reversed on odd steps)
LAYOUT_PLACERS = tuple(os.environ.get("CP_LAYOUT_PLACERS", "row group").split())
CONTROL_STEPS = tuple(int(x) for x in os.environ.get("CP_CONTROL_STEPS", "4 5").split())
GOLDEN_FOR = tuple(os.environ.get("CP_GOLDEN_FOR", "CORRECT MAIN LAYOUT").split())   # a GOLDEN-only process: whose schedule
OUT = VA.OUT
TAG = os.environ.get("CP_TAG", "cp_b%d" % VA.BATCH)
RC_HYGIENE, RC_CORRECT, RC_MEMGATE, RC_CRASH, RC_PLACEMENT = 20, 21, 22, 23, 24
W8 = dict(n_ctas=8, num_warps=4, bypass_cache=False)
CHUNK_FIELDS = ("g0", "g1", "slot", "pk", "h2d0", "h2d1", "sc0", "sc1", "sub", "bp_ms", "pack_ms", "pack_coord_cpu_ms", "api_ms", "bytes",
                "useful")
FIELD_NOTES = dict(
    pack_coord_cpu_ms="thread_time of the COORDINATOR thread only during the pack call; it EXCLUDES the OpenMP helper threads, so "
                      "it is NOT the total pack CPU time (the helpers' ticks are in thread_use census deltas)",
    desc_coord_cpu_ms="thread_time of the coordinator thread during descriptor preparation (same caveat)",
    pack_ms="host wall time of the pack call (all team threads)")


class MemGate(Exception):
    """A registered B320-fallback trigger (the sbatch reruns ALL arms at the fallback batch)."""


class PlacementRefused(Exception):
    """The CPU placement cannot give the launch core + the full team on the GPU's node (exit RC_PLACEMENT)."""


def placement_problem(early):
    """None when the early placement is usable for team size max(CORES), else the refusal text."""
    e = early or {}
    if e.get("omp_env_problems"):
        return "refused: %s" % e["omp_env_problems"]
    team = e.get("team") or []
    if not e.get("ok") or len(team) < max(CORES):
        return "placement gives %d physical team cores < %d on the GPU node (%s)" % (len(team), max(CORES), e.get("notes"))
    return None


# ---------------------------------------------------------------------------------------------------------------- arms
def arm(name, kind, **kw):
    d = dict(name=name, kind=kind, cores=kw.pop("cores", None), mode=kw.pop("mode", "full"), lite=kw.pop("lite", False),
             chunk_kb=kw.pop("chunk_kb", CHUNK_KB), packer=kw.pop("packer", "row"), placer=kw.pop("placer", "row"),
             ceiling=kw.pop("ceiling", False), label=kw.pop("label", K.LABEL))
    if kw:
        raise ValueError(kw)
    return d


def main_arms(cores=CORES, alt_kb=CHUNK_KB_ALT, only=ARMS_ONLY):
    """The MAIN stage arms in the ORIGINAL layout (orig source, orig scratch, row packer, row placer)."""
    top = max(cores)
    a = [arm("w8", "w8")] + [arm("cpu%d" % n, "cpu", cores=n) for n in cores]
    if alt_kb:
        a.append(arm("cpu%d_c%d" % (top, alt_kb // 1024), "cpu", cores=top, chunk_kb=alt_kb))
    a.append(arm("cpu%d_lite" % top, "cpu", cores=top, lite=True))
    for m in ("L", "LP", "LPD"):
        a.append(arm("cpu%d_%s" % (top, m), "cpu", cores=top, mode=m))
    a.append(arm("cpu%d_prepacked" % top, "cpu", cores=top, mode="prepacked", ceiling=True, label=K.CEILING_LABEL + "; stale payload"))
    a.append(arm("ceil_contig", "contig", ceiling=True, label=K.CEILING_LABEL + "; one contiguous H2D per layer, no list/pack/scatter"))
    a.append(arm("scatter_only", "scatter", ceiling=True, label=K.CEILING_LABEL + "; placement only from device memory"))
    return [x for x in a if not only or x["name"] in only]


LAYOUT_CELLS_PRE = (("orig", "orig"), ("orig", "hm"))
LAYOUT_CELLS_POST = (("hm", "hm"), ("hm", "orig"))


def layout_variants(src, dst, top=8, placers_hm=None):
    """layout_ablation: W8 plus the cpu<top> packer x placer variants of one (source, destination) cell. The packer effect
    is row vs group at a fixed placer (head-major source only); the placer effect is row vs group at a fixed packer
    (head-major destination only). placers_hm (default CP_LAYOUT_PLACERS) limits the head-major destination's placers:
    the confirmation registers ('row',) = row/row in every cell plus the group packer on head-major source cells."""
    out = [arm("w8@%s>%s" % (src, dst), "w8")]
    packers = ("row", "group") if src == "hm" else ("row",)
    ph = tuple(p for p in (LAYOUT_PLACERS if placers_hm is None else placers_hm) if p in ("row", "group")) or ("row",)
    placers = ph if dst == "hm" else ("row",)
    for p in packers:
        for q in placers:
            out.append(arm("cpu%d@%s>%s:%s/%s" % (top, src, dst, p, q), "cpu", cores=top, packer=p, placer=q))
    return out


def timeline_arms(names=T_ARMS):
    top = {a["name"]: a for a in main_arms(only=())}
    out = []
    for n in names:
        if n in ("alone", "alone_late"):
            out.append(dict(name=n, kind="alone"))
        elif n in top:
            out.append(top[n])
        else:
            raise ValueError("unknown timeline arm %r" % n)
    return out


def pow2(n: int) -> int:
    """torch 2.6 CachingHostAllocator rounds every pinned block up to a power of two (ATen/core/CachingHostAllocator.h:132)."""
    return 1 << (int(n) - 1).bit_length() if n > 0 else 0


def mem_need_gb(B, S_dst, H, D, ring, slot, contig, extra_gb=0.35) -> float:
    """HBM the replay needs after prefill + capture (checked against mem_get_info before any arm; registered trigger (b))."""
    scratch = 2 * B * S_dst * H * D * 2
    return (scratch + 2 * ring * slot + contig) / 1e9 + extra_gb


class Job:
    def __init__(self, fn):
        self.fn, self.done, self.error, self.result, self.exc = fn, threading.Event(), None, None, None


class Coordinator(threading.Thread):
    """ONE transport thread. Its mask = the arm's N team cores and torch.set_num_threads(N) (per-thread OpenMP ICV); its
    OpenMP helpers are re-pinned to team[:N] after every configure (their tids are the threads that appeared during
    configure calls). Everything it enqueues on the GPU goes to the copy / scatter streams and the aux marker stream."""

    def __init__(self, team):
        super().__init__(daemon=True, name="cp-coordinator")
        self.q = queue.Queue()
        self.team = list(team)
        self.n = None
        self.helpers = set()
        self.configs = []

    def run(self):
        self.tid = threading.get_native_id()
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                with torch.inference_mode():                            # thread-local: the main thread's tensors are inference
                    job.result = job.fn(self)                           # tensors, written in place here (landing, scratch)
            except BaseException as e:
                job.error = traceback.format_exc()
                job.exc = e
            finally:
                job.done.set()

    def submit(self, fn) -> Job:
        j = Job(fn)
        self.q.put(j)
        return j

    def call(self, fn):
        """Run fn on the coordinator and wait. A registered memory trigger (MemGate, a CUDA OOM) keeps its own type, so the
        exit-code rule sees it exactly; every other error becomes RuntimeError('coordinator: <traceback>'), which is never
        read as a memory trigger (its traceback text may contain any word, e.g. 'pinned')."""
        j = self.submit(fn)
        j.done.wait()
        if j.error:
            if isinstance(j.exc, (MemGate, torch.cuda.OutOfMemoryError)):
                raise j.exc
            raise RuntimeError("coordinator: " + j.error)
        return j.result

    def configure(self, n):
        def f(c):
            before = {r["tid"] for r in CC.census()}
            mask = CC.set_mask(c.team[:n])
            torch.set_num_threads(n)
            x = torch.zeros((4096 * max(n, 1), 128), dtype=torch.bfloat16)
            torch.index_select(x, 0, torch.arange(x.shape[0] - 1, -1, -1), out=torch.empty_like(x))   # creates the team
            after = CC.census()
            c.helpers |= {r["tid"] for r in after if r["tid"] not in before and r["tid"] != c.tid}
            pinned = []
            for tid in sorted(c.helpers):
                try:
                    os.sched_setaffinity(tid, set(c.team[:n]))
                    pinned.append(tid)
                except OSError:
                    pass
            c.n = n
            rec = dict(n=n, mask=mask, threads=torch.get_num_threads(), helpers=sorted(c.helpers), helpers_pinned=pinned,
                       helper_masks={r["tid"]: r["cpus_allowed"] for r in CC.census() if r["tid"] in c.helpers})
            c.configs.append(rec)
            return rec
        return self.call(f)


# ----------------------------------------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, model, ids, docs, distinct, flush):
        self.model, self.ids, self.docs, self.distinct, self.flush_fn = model, ids, docs, distinct, flush
        self.fails = 0
        self.layout_fails = 0
        self.memgate = []
        self.nonce = 0
        self.rows, self.sweep, self.timeline, self.layout_rows = [], [], [], []
        self.correct = {}
        self.layout = dict(cells=[], conversion=None, tail_write=None, tail_write_real={}, notes=[], partial=False, memgate=[])
        self.main_done = False
        self.sweep_notes = []
        self.t_start = time.time()
        self.payload_extra = {}
        self.inv_snaps = {}
        self.thread_use = []
        self.dev = "cuda"                                                # the CPU tests drive transport / checks / conversion with "cpu"
        # ---- the confirmation (cpupack_golden.py)
        self.guard = G.DecodeGuard("controls" if MODE == "controls" else "measure")
        self.golden_mode = None                                          # None (original flow) | "record" | "check"
        self.golden = dict(steps={})
        self.golden_fail = []                                            # golden-pass steps whose own resident check failed
        self.gate_log, self.gate_fail_steps, self.restarts, self.warm_log = [], [], [], []
        self.schedule = None
        self.inject = None                                               # CONTROLS only: perturbation hooks
        self.in_layout = False
        self.arm_orders = []
        self.plans_vs_saved = None
        self.golden_cross = None

    def snap_inventory(self, label):
        """One memory / NUMA inventory at a named point (a /proc/self/numa_maps read walks ~268 GB of pinned pages at B336, so it
        is taken at a few points only, never per step)."""
        try:
            self.inv_snaps[label] = self.inventory()
        except Exception as e:                                          # never lose a run to the inventory
            self.inv_snaps[label] = dict(error=repr(e))

    # ------------------------------------------------------------------------------------------------ setup
    def log(self, msg):
        print("[cpupack] " + msg, flush=True)

    def setup_early(self):
        """Before prefill: the coordinator (team cores), pinned staging + plan list + contiguous-ceiling source (allocated by
        the coordinator so first-touch lands on the team's node), device landing ring, streams, events."""
        B = int(self.ids.shape[0])
        self.B = B
        self.H, self.D, self.R, self.M = K.H_DEF, K.D_DEF, K.R_DEF, K.M_DEF
        self.HB = self.H * B
        self.W = 1 + self.HB + self.HB * self.M
        self.cap_main = CHUNK_KB * 1024 // K.useful_bytes(1)
        self.cap_alt = (CHUNK_KB_ALT * 1024 // K.useful_bytes(1)) if CHUNK_KB_ALT else 0
        self.cap_max = max(self.cap_main, self.cap_alt, LIFETIME_CAP)
        self.slot = K.slot_bytes(self.cap_max)
        team = (EARLY or {}).get("team") or []
        why = placement_problem(EARLY)
        if why:
            raise PlacementRefused(why)
        self.coord = Coordinator(team)
        self.coord.start()
        for n in CORES:                                      # nested helper creation: 1, 2, 4, 8
            self.coord.configure(n)
        W, slot = self.W, self.slot

        def alloc(c):
            try:
                st = torch.zeros((RING, slot), dtype=torch.uint8).pin_memory()
                ph = torch.full((32, W), -7, dtype=torch.int32).pin_memory()
                cs = torch.zeros(CONTIG_BYTES, dtype=torch.uint8).pin_memory()
            except RuntimeError as e:
                raise MemGate("pinned staging allocation failed: %s" % e)
            return st, ph, cs
        self.stage, self.plan_h, self.contig_src = self.coord.call(alloc)
        self.land = torch.zeros((RING, slot), dtype=torch.uint8, device="cuda")
        self.contig_dev = torch.zeros(CONTIG_BYTES, dtype=torch.uint8, device="cuda")
        self.main = torch.cuda.current_stream()
        self.s_plan, self.s_side, self.s_copy, self.s_scatter, self.aux = (torch.cuda.Stream() for _ in range(5))
        self.cbe = K.CudaBackend(dict(copy=self.s_copy, scatter=self.s_scatter), self.aux, pool_size=(40000 if SWEEP else 8192))
        mk = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(32)]
        self.ev_pr, self.ev_list, self.ev_g0, self.ev_g1 = mk(), mk(), mk(), mk()
        self.bev = {k: torch.cuda.Event(enable_timing=True) for k in ("pre", "gate", "t0", "tm", "sub", "lag")}
        self.sleep_cycles = int(SLEEP_MS * 1e-3 * 1.41e9)
        self.pipes = {}

    def setup_min(self):
        """CONTROLS process: the geometry only (no coordinator, no pinned staging, no transport)."""
        B = int(self.ids.shape[0])
        self.B = B
        self.H, self.D, self.R, self.M = K.H_DEF, K.D_DEF, K.R_DEF, K.M_DEF
        self.HB = self.H * B
        self.W = 1 + self.HB + self.HB * self.M

    def pipe(self, cap):
        if cap not in self.pipes:
            self.pipes[cap] = K.Pipe(self.cbe, self.stage, self.land, cap)
        return self.pipes[cap]

    def setup_model(self):
        from nosi import state_snapshot as ss
        self.ss = ss
        model, cache, logits, position_ids, forced, run_meta = VA._setup(self.model, self.ids)
        self.cache, self.forced, self.run_meta = cache, forced, run_meta
        self.engines = [lay.cache_engine for lay in cache.layers]
        self.NL = len(self.engines)
        self.pos0 = position_ids[:, -1:] + 1
        self.cu = torch.arange(0, self.B + 1, dtype=torch.int, device="cuda")
        e0 = self.engines[0]
        _, self.S_dst, H, D = e0._k_gpu.shape
        self.S_cpu = int(e0._k_cpu.shape[1])
        assert (H, D) == (self.H, self.D) and int(e0.block_size) == self.R
        for e in self.engines:
            assert e._k_cpu.is_pinned() and e._k_cpu.is_contiguous() and e._v_cpu.is_contiguous()
        self.host_layout = "orig"
        self.host_phys = [(e._k_cpu, e._v_cpu) for e in self.engines]
        if hasattr(self, "plan_h"):
            assert self.NL <= self.plan_h.shape[0], "the plan-list buffer holds %d layers < %d" % (self.plan_h.shape[0], self.NL)
        torch.cuda.reset_peak_memory_stats()

    def loaded_now(self):
        return int(torch.stack([(e._load_mask >= 0).sum() for e in self.engines]).sum())

    def decode(self, tok, pos, warmup=False):
        """EVERY decode of this harness goes through here: the guard checks it BEFORE the model runs (cpupack_golden)."""
        self.guard.check("decode")
        return self.model.decode_inference(tok, self.cu, pos, self.cache, warmup=warmup) if warmup else \
            self.model.decode_inference(tok, self.cu, pos, self.cache)

    def sync(self):
        if self.dev == "cuda":
            torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------------------- capture
    def capture(self, snapshot=None, digest_steps=()):
        """snapshot=False: the caller owns the restart (run_confirm's hosted PostPrefillSnapshot). digest_steps: the warm
        steps whose logits sha256 and FULL state digest are recorded (cap['warm']) as the restart proof's reference."""
        from nosi import transfer_trace as _tt
        snap = None
        warm = []
        if (not NO_SNAPSHOT) if snapshot is None else snapshot:
            try:
                snap = self.ss.PostPrefillSnapshot(self.cache).take()
            except torch.cuda.OutOfMemoryError as e:                     # registered rule (e): two processes first
                raise _SnapshotOOM(str(e)[:300])
        AloneTrace, _ = VA.make_trace_classes()
        T16 = PL.make_int16_trace_class(AloneTrace)
        ncap = VA.N
        tr = T16(self.NL, "1", max_steps=ncap)
        _tt.TRACE = tr
        pos = self.pos0.clone()
        sha, loaded, xfail = [], [], 0
        t0 = time.time()
        for it in range(ncap):
            lg = self.decode(self.forced[:, it:it + 1], pos, warmup=(it == 0))
            torch.cuda.synchronize()
            sha.append(VA.sha(lg))
            loaded.append(self.loaded_now())
            for l, e in enumerate(self.engines):
                xfail += int(not torch.equal(tr.mask_archive[it, l].to(torch.int64), e._load_mask))
                xfail += int(not torch.equal(tr.map_archive[it, l].to(torch.int64), e._block_map))
            if it in digest_steps:
                warm.append(dict(step=it, logits_sha=sha[-1], logits_sha256=G.sha_parts([lg]), digest=self.state_digest(G.FAMILIES_PRE)))
            pos = pos + 1
        step_rows, _, _, masks, maps = tr.harvest(timed_steps=())
        _tt.TRACE = None
        ok_trace = tr.dropped_steps == 0 and len(tr.steps) == ncap
        self.store = torch.empty((ncap, self.NL, self.W), dtype=torch.int32, device="cuda")
        for s in range(ncap):
            m32 = tr.mask_archive[s].to(torch.int32)
            self.store[s, :, 1 + self.HB:] = m32.reshape(self.NL, -1)
            self.store[s, :, 1:1 + self.HB] = (m32[..., :K.TAIL_SLOT] >= 0).sum(-1).to(torch.int32).reshape(self.NL, -1)
            self.store[s, :, 0] = 0
        del tr
        self.masks, self.maps = masks.to(torch.int16), maps.to(torch.int16)
        self.logits_sha, self.step_loaded = sha, loaded
        inv = PL.invariants(self.masks, self.maps, L_prompt=VA.L, step_loaded=loaded)
        meta = dict(batch=self.B, L=VA.L, ncap=ncap, docs=self.docs, ids_sha=VA.sha(self.ids.to(torch.float32)), nosi_commit=os.environ.get("NOSI_COMMIT"),
                    attn_splits=os.environ.get("NOSI_ATTN_SPLITS"), pool_blocks=os.environ.get("NOSI_POOL_BLOCKS"),
                    round_slots=os.environ.get("NOSI_VERIFY_ROUND_SLOTS"))
        path = os.path.join(OUT, "%s_plans.npz" % TAG)
        hs = PL.export_npz(path, self.masks, self.maps, meta, loaded, sha)
        cap = dict(source="capture (int16 AloneTrace, one harvest)", ncap=ncap, seconds=time.time() - t0, cross_check_fails=xfail,
                   trace_ok=ok_trace, invariants=inv, accepted=PL.accepted(inv) and xfail == 0 and ok_trace, run_digest=hs["run_digest"],
                   step_digest=hs["step_digest"], export=path, step_loaded=loaded, logits_sha=sha,
                   natural_step_ms=[r.get("step_ms") for r in step_rows], snapshot=snap is not None, warm=warm)
        if snap is not None:
            snap.restore()
            del snap
        gc.collect()
        torch.cuda.empty_cache()
        self.model.has_buffers = False
        return cap

    def load_plans(self, path):
        z = PL.load_npz(path)
        if int(z["meta"]["batch"]) != self.B or int(z["meta"]["L"]) != VA.L:
            raise RuntimeError("plan export %s is for batch %s L %s" % (path, z["meta"]["batch"], z["meta"]["L"]))
        self.masks, self.maps = z["masks"], z["maps"]
        self.logits_sha, self.step_loaded = z["logits_sha"], z["step_loaded"]
        ncap = self.masks.shape[0]
        self.store = torch.empty((ncap, self.NL, self.W), dtype=torch.int32, device="cuda")
        for s in range(ncap):
            m32 = self.masks[s].to(torch.int32)
            row = torch.cat([torch.zeros((self.NL, 1), dtype=torch.int32), (m32[..., :K.TAIL_SLOT] >= 0).sum(-1).to(torch.int32).reshape(self.NL, -1),
                             m32.reshape(self.NL, -1)], dim=1)
            self.store[s].copy_(row)
        return dict(source="export %s (cross-process)" % path, ncap=ncap, run_digest=z["hashes"]["run_digest"], meta=z["meta"],
                    invariants=PL.invariants(self.masks, self.maps, L_prompt=VA.L, step_loaded=self.step_loaded))

    # ---------------------------------------------------------------------------------------------- scratch / cells
    def set_scratch(self, layout):
        if getattr(self, "scr_layout", None) == layout:
            return
        for n in ("scr_k", "scr_v"):
            if hasattr(self, n):
                delattr(self, n)
        gc.collect()
        torch.cuda.empty_cache()
        self.scr_k = K.alloc_phys(layout, self.B, self.S_dst, self.H, self.D, torch.bfloat16, "cuda")
        self.scr_v = K.alloc_phys(layout, self.B, self.S_dst, self.H, self.D, torch.bfloat16, "cuda")
        e0 = self.engines[0]
        if layout == "orig":
            assert self.scr_k.stride() == e0._k_gpu.stride(), "the orig scratch must have the engine window's strides"
        self.scr_layout = layout
        self.chk = {}

    def cell(self, src=None, dst=None):
        src = src or self.host_layout
        dst = dst or self.scr_layout
        if src != self.host_layout or dst != self.scr_layout:
            raise RuntimeError("cell (%s, %s) but host %s scratch %s" % (src, dst, self.host_layout, self.scr_layout))
        return dict(src=src, dst=dst)

    def src_log(self, l, which=0):
        return K.logical(self.host_phys[l][which], self.host_layout)

    def scr_log(self, which=0):
        return K.logical(self.scr_k if which == 0 else self.scr_v, self.scr_layout)

    def dest_mask(self, plan_rows_cpu, layers):
        """Expected written rows of the scratch (physical row order), derived by LOGICAL indexing (independent of the
        address formulas): [B, S_dst, H] logical bool -> the physical order of the scratch layout."""
        m = torch.zeros((self.B, self.S_dst, self.H), dtype=torch.bool)
        for l in layers:
            d = K.build_desc(plan_rows_cpu[l][1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout="orig",
                             s_src=self.S_cpu, s_dst=self.S_dst)
            b, t, h = K.dest_rows_logical(d)
            m[b, t, h] = True
        phys = m if self.scr_layout == "orig" else m.permute(0, 2, 1).contiguous()
        return phys.reshape(-1).to(self.dev)

    def ref_layer(self, plan_rows_cpu, l):
        """(desc, K ref rows, V ref rows) of layer l by LOGICAL indexing of the host source (CPU), uploaded."""
        d = K.build_desc(plan_rows_cpu[l][1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout="orig",
                         s_src=self.S_cpu, s_dst=self.S_dst)
        rk = K.reference_rows(self.src_log(l, 0), d).to(self.dev)
        rv = K.reference_rows(self.src_log(l, 1), d).to(self.dev)
        b, t, h = K.dest_rows_logical(d)
        return dict(desc=d, rk=rk, rv=rv, bth=(b.to(self.dev), t.to(self.dev), h.to(self.dev)), n=d.n)

    def check_scratch(self, expect_mask, ref=None) -> dict:
        ck = K.canary(self.scr_k, expect_mask)
        cv = K.canary(self.scr_v, expect_mask)
        out = dict(canary_ok=ck["ok"] and cv["ok"], extra_rows=ck["extra_rows"] + cv["extra_rows"], missing_rows=ck["missing_rows"] + cv["missing_rows"])
        if ref is not None:
            b, t, h = ref["bth"]
            gk = self.scr_log(0)[b, t, h]
            gv = self.scr_log(1)[b, t, h]
            out["content_ok"] = K.bits_equal(gk, ref["rk"]) and K.bits_equal(gv, ref["rv"])
        return out

    def rows_of(self, ps):
        """(GPU store rows [NL, W], CPU int32 rows [NL, W]) of a captured step."""
        cpu = torch.cat([torch.zeros((self.NL, 1), dtype=torch.int32),
                         (self.masks[ps][..., :K.TAIL_SLOT] >= 0).sum(-1).to(torch.int32).reshape(self.NL, -1),
                         self.masks[ps].to(torch.int32).reshape(self.NL, -1)], dim=1)
        return self.store[ps], cpu

    def prepare(self, ps, arms, rows=None):
        """Per plan step, untimed: expected rows (union over layers), the last layer's logical reference, prebuilt descriptors
        for the prepacked ceiling, device indices for scatter_only, per-layer useful bytes / request groups."""
        key = (ps, self.host_layout, self.scr_layout)
        if self.chk.get("key") == key:
            return self.chk
        gpu, cpu = rows if rows is not None else self.rows_of(ps)
        L = list(range(self.NL))
        c = dict(key=key, gpu=gpu, cpu=cpu, union=self.dest_mask(cpu, L), last=self.ref_layer(cpu, self.NL - 1),
                 useful=[K.useful_bytes(int((cpu[l][1 + self.HB:] >= 0).sum())) for l in L],
                 req=[cpu[l][1 + self.HB:].view(self.H, self.B, self.M)[..., :K.TAIL_SLOT].ge(0).sum(dim=(0, 2)).tolist() for l in L])
        c["prebuilt"] = {}
        for a in arms:
            if a["kind"] == "cpu" and a["mode"] == "prepacked":
                c["prebuilt"][(a["packer"], a["placer"])] = [
                    K.build_desc(cpu[l][1 + self.HB:].view(self.H, self.B, self.M), src_layout=self.host_layout, dst_layout=self.scr_layout,
                                 s_src=self.S_cpu, s_dst=self.S_dst, packer=a["packer"], placer=a["placer"]) for l in L]
            if a["kind"] == "scatter":
                c["so_idx"] = [K.build_desc(cpu[l][1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout=self.scr_layout,
                                            s_src=self.S_cpu, s_dst=self.S_dst).dst_idx.to(self.dev) for l in L]
        self.chk = c
        return c

    # ---------------------------------------------------------------------------------------------- the bracket
    def bracket(self, a, c, with_decode, step_fn=None, layers=None, faults=None, nvtx=None, late_ms=0.0, cap=None):
        """ONE rep of arm `a` over the plan rows c['gpu'] (all 32 layers unless `layers`). GPU sleep -> gate -> the plan
        stream's replay (E_pr[l] = the IDENTICAL plan-ready milestone of every arm) -> [cpu arms: list D2H -> E_list[l]] ->
        pre-enqueued GPU arms / the coordinator handoff -> t0 -> the decode -> tm. The coordinator acts only after
        E_list[l] (nothing plan-dependent is enqueued before the gate)."""
        L = list(range(self.NL)) if layers is None else list(layers)
        kind = a["kind"]
        need_list = kind == "cpu"
        cell = self.cell()
        torch.cuda.synchronize()
        K.poison_(self.scr_k)
        K.poison_(self.scr_v)
        self.plan_h.fill_(-7)
        self.nonce += 1
        nonce = self.nonce
        self.cbe.reset()
        pipe = None
        if kind == "cpu":
            pipe = self.pipe(cap or (a.get("chunk_kb", CHUNK_KB) * 1024 // K.useful_bytes(1)))
            pipe.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.bev["lag"].record(self.aux)
        self.bev["lag"].synchronize()
        lag_us = (time.perf_counter() - t) * 1e6
        ev = self.bev
        if nvtx:
            torch.cuda.nvtx.range_push(nvtx)
        ev["pre"].record(self.main)
        torch.cuda._sleep(self.sleep_cycles)
        ev["gate"].record(self.main)
        h0 = time.perf_counter()
        with torch.cuda.stream(self.s_plan):
            self.s_plan.wait_event(ev["gate"])
            for l in (L if kind != "alone" else ()):
                self.work[l].copy_(c["gpu"][l])
                self.work[l, 0:1].fill_(nonce)
                self.ev_pr[l].record(self.s_plan)
                if need_list:
                    self.plan_h[l].copy_(self.work[l], non_blocking=True)
                    self.ev_list[l].record(self.s_plan)
        job = None
        if kind == "w8":
            from nosi.flash_cache_engine.flash_h2d_persistent import flash_h2d_persistent
            with torch.cuda.stream(self.s_side):
                for l in L:
                    self.s_side.wait_event(self.ev_pr[l])
                    self.ev_g0[l].record(self.s_side)
                    ids = self.work[l, 1 + self.HB:].view(self.H, self.B, self.M)
                    flash_h2d_persistent(self.scr_log(0), self.src_log(l, 0), ids, self.R, **W8)
                    flash_h2d_persistent(self.scr_log(1), self.src_log(l, 1), ids, self.R, **W8)
                    self.ev_g1[l].record(self.s_side)
        elif kind == "contig":
            with torch.cuda.stream(self.s_copy):
                for l in L:
                    self.s_copy.wait_event(self.ev_pr[l])
                    self.ev_g0[l].record(self.s_copy)
                    nb = c["useful"][l]
                    while nb > 0:
                        k = min(nb, CONTIG_BYTES)
                        self.contig_dev[:k].copy_(self.contig_src[:k], non_blocking=True)
                        nb -= k
                    self.ev_g1[l].record(self.s_copy)
        elif kind == "scatter":
            with torch.cuda.stream(self.s_scatter):
                for l in L:
                    self.s_scatter.wait_event(self.ev_pr[l])
                    self.ev_g0[l].record(self.s_scatter)
                    idx = c["so_idx"][l]
                    n = int(idx.numel())
                    cap_rows = CONTIG_BYTES // (self.D * 2)                  # the device source holds cap_rows rows: pieces
                    for a0 in range(0, n, cap_rows):
                        k = min(cap_rows, n - a0)
                        src = self.contig_dev[:k * self.D * 2].view(torch.bfloat16).view(k, self.D)
                        K.rows2d(self.scr_k).index_copy_(0, idx[a0:a0 + k], src)
                        K.rows2d(self.scr_v).index_copy_(0, idx[a0:a0 + k], src)
                    self.ev_g1[l].record(self.s_scatter)
        elif kind == "cpu":
            spec = dict(arm=a, layers=L, pipe=pipe, faults=faults, prebuilt=c.get("prebuilt", {}).get((a["packer"], a["placer"])), cell=cell)
            job = self.coord.submit(lambda co, spec=spec: self.transport(co, spec))
        host_enqueue_ms = 1000 * (time.perf_counter() - h0)
        ev["t0"].record(self.main)
        out, h1, h2, gate_done = None, None, None, None
        if with_decode:
            if late_ms > 0:
                self.main.synchronize()                                  # negative control: wait for the gate, then sleep
                time.sleep(late_ms / 1000.0)
            if nvtx:
                torch.cuda.nvtx.range_push("lt_submit|" + nvtx)
            h1 = time.perf_counter()
            out = step_fn()
            h2 = time.perf_counter()
            if nvtx:
                torch.cuda.nvtx.range_pop()
            gate_done = ev["gate"].query()
            ev["sub"].record(self.aux)
        ev["tm"].record(self.main)
        if job is not None:
            job.done.wait()
        torch.cuda.synchronize()
        if nvtx:
            torch.cuda.nvtx.range_pop()
        g = ev["gate"]
        ms = lambda e: None if e is None else round(g.elapsed_time(e), 4)
        r = dict(arm=a["name"], kind=kind, mode=a.get("mode"), lite=a.get("lite", False), cores=a.get("cores"), packer=a.get("packer"),
                 placer=a.get("placer"), chunk_kb=a.get("chunk_kb"), src_layout=cell["src"], dst_layout=cell["dst"], with_decode=bool(with_decode),
                 nonce=nonce, layers=L, marker_lag_us=round(lag_us, 2), sleep_ms=ev["pre"].elapsed_time(g), host_enqueue_ms=host_enqueue_ms,
                 host_lag_ms=ms(ev["t0"]), t0=ms(ev["t0"]), tm=ms(ev["tm"]),
                 pr=([ms(self.ev_pr[l]) for l in L] if kind != "alone" else []),
                 useful=([c["useful"][l] for l in L] if kind != "alone" else []), ceiling=a.get("ceiling", False), label=a.get("label"))
        if with_decode:
            r.update(main_ms=ev["t0"].elapsed_time(ev["tm"]), host_submit_ms=1000 * (h2 - h1), gate_done_at_submit_end=bool(gate_done),
                     submit_end_ms=ms(ev["sub"]), late_ms=late_ms)
        if kind in ("w8", "contig", "scatter"):
            r.update(g0=[ms(self.ev_g0[l]) for l in L], g1=[ms(self.ev_g1[l]) for l in L])
        if job is not None:
            if job.error:
                r["coord_error"] = job.error
            else:
                r["list"] = [ms(self.ev_list[l]) for l in L]
                lays = []
                for x in job.result:
                    ch = [[x2.get("g0"), x2.get("g1"), x2.get("slot"), ms(x2.get("pk")), ms(x2.get("h2d0")), ms(x2.get("h2d1")), ms(x2.get("sc0")),
                           ms(x2.get("sc1")), ms(x2.get("sub")), x2["bp_ns"] / 1e6, x2["pack_ns"] / 1e6, x2["pack_cpu_ns"] / 1e6, x2["api_ns"] / 1e6,
                           x2["bytes"], K.useful_bytes(x2["g1"] - x2["g0"])] for x2 in x.get("chunks", [])]
                    lays.append([ms(x.get("hk")), ms(x.get("desc_ev")), x["list_wait_ns"] / 1e6, x.get("desc_ns", 0) / 1e6,
                                 x.get("desc_cpu_ns", 0) / 1e6, x.get("n"), ch])
                r["layers_rec"] = lays
                r["h2d_bytes"] = sum(ch[13] for lay in lays for ch in lay[6]) if MODES_DMA(a) else 0
            r["list_bytes"] = 4 * self.W * len(L)
            r["list_padding_bytes"] = 4 * sum(self.HB * self.M - int((c["cpu"][l][1 + self.HB:] >= 0).sum()) for l in L)
        return out, r

    def transport(self, co, spec):
        """The coordinator's rep (runs on the coordinator thread): per layer in order, wait for THIS rep's list (E_list[l]),
        then descriptors, then the pipeline."""
        a, L, pipe, f = spec["arm"], spec["layers"], spec["pipe"], spec["faults"]
        be, lite = self.cbe, a["lite"]
        out = []
        for l in L:
            r = dict(l=l)
            t = time.perf_counter_ns()
            be.host_wait(self.ev_list[l])
            r["list_wait_ns"] = time.perf_counter_ns() - t
            r["hk"] = None if lite else be.marker()
            if a["mode"] == "L":
                out.append(r)
                continue
            t, tc = time.perf_counter_ns(), time.thread_time_ns()
            if spec["prebuilt"] is not None:
                d = spec["prebuilt"][l]
            else:
                d = K.build_desc(self.plan_h[l][1 + self.HB:].view(self.H, self.B, self.M), src_layout=self.host_layout,
                                 dst_layout=self.scr_layout, s_src=self.S_cpu, s_dst=self.S_dst, packer=a["packer"], placer=a["placer"])
            r["desc_ns"], r["desc_cpu_ns"] = time.perf_counter_ns() - t, time.thread_time_ns() - tc
            r["desc_ev"] = None if lite else be.marker()
            r["n"] = d.n
            sk, dk = K.views_for(d, self.host_phys[l][0], self.scr_k)
            sv, dv = K.views_for(d, self.host_phys[l][1], self.scr_v)
            r["chunks"] = pipe.layer(d, sk, sv, dk, dv, mode=a["mode"], faults=f, lite=lite)
            out.append(r)
        return out

    def rep_checks(self, r, c, expect_mask, ref, lg=None, lg_ref=None):
        """Per-rep correctness (outside the bracket): list delivery (nonce, counts, plan == capture), coordinator errors,
        scratch canary (+ last-layer content for content-carrying arms), decode logits and 0 loads."""
        ok = True
        if r["kind"] == "cpu":
            ph = self.plan_h
            good = True
            for l in r["layers"]:
                good &= int(ph[l, 0]) == r["nonce"] and bool(torch.equal(ph[l], torch.cat([torch.tensor([r["nonce"]], dtype=torch.int32), c["cpu"][l][1:]])))
            r["list_ok"] = bool(good)
            ok &= good and "coord_error" not in r
        writes = r["kind"] in ("w8", "scatter") or (r["kind"] == "cpu" and r["mode"] in ("full", "prepacked"))
        content = r["kind"] == "w8" or (r["kind"] == "cpu" and r["mode"] == "full")
        exp = expect_mask if writes else torch.zeros_like(expect_mask)
        ch = self.check_scratch(exp, ref if content else None)
        r.update(ch)
        ok &= ch["canary_ok"] and ch.get("content_ok", True)
        if lg is not None:
            r["logits_equal"] = bool(torch.equal(lg, lg_ref))
            r["loads"] = self.loaded_now()
            ok &= r["logits_equal"] and r["loads"] == 0
        r["ok"] = bool(ok)
        return ok

    # ---------------------------------------------------------------------------------------------- the gated step
    def gated(self, it, pos, work):
        """worker_sweep.py:491-507 (light restore) around `work(ctx)`, then the advance (:584-587). With a golden mode
        (cpupack_golden): the gate BEFORE work (digests after the reference decode + the pre-timing resident check) and
        after the advance; a failing pre-gate skips `work` (no timing at this step)."""
        tok = self.forced[:, it:it + 1]
        tails = [int(e._tail_block_len_on_gpu) for e in self.engines]
        if max(tails) >= self.R - 1:                                     # the next decode would complete the tail block: a
            raise RuntimeError("rollover tripwire at step %d: tail lengths %s reach %d (a tail write-back would follow)"
                               % (it, sorted(set(tails)), self.R - 1))     # write-back into the host cache (section D)
        snap = self.snap
        snap.take()
        for e in self.engines:
            self.mc.flush_map(e)
        self.guard.begin_step(it)
        try:
            lg_ref = self.decode(tok, pos)
            self.sync()
            for i, e in enumerate(self.engines):
                slot = snap.layers[i]["engine"]
                for name in ("_block_map", "_new_block_map_buf"):
                    slot[name].copy_(getattr(e, name))
            maps = getattr(self, "maps", None)
            agree = [PL.selection_agreement(self.engines[l]._block_map.cpu(), maps[it, l]) for l in range(self.NL)] \
                if maps is not None and it < maps.shape[0] else []

            def restore():
                snap.restore()
                self.ss.assert_transients_intact(self.model, self.trans)
                self.guard.on_restore()
            ctx = dict(it=it, tok=tok, pos=pos, lg_ref=lg_ref, restore=restore, step_fn=lambda: self.decode(tok, pos), tail_len_max=max(tails),
                       agreement=dict(same_frac_mean=float(np.mean([x["same_frac"] for x in agree])) if agree else None,
                                      jaccard_mean=float(np.mean([x["jaccard_mean"] for x in agree])) if agree else None,
                                      jaccard_min=float(min(x["jaccard_min"] for x in agree)) if agree else None))
            if self.inject and self.inject.get("after_ref"):             # CONTROLS process only (a destructive perturbation)
                self.inject["after_ref"](ctx)
            pre = self.gate_pre(it, lg_ref, ctx)
            marks = self.row_marks()
            if pre["ok"]:
                work(ctx)
            else:
                self.log("GATE_FAIL step %d BEFORE timing (%s): %s" % (it, self.golden_mode, pre.get("why")))
            restore()
            lg_adv = self.decode(tok, pos)
            self.sync()
            post = self.gate_post(it, lg_adv)
            ok = bool(pre["ok"] and post["ok"])
            self.stamp_rows(marks, it, ok)
            self.gate_log.append(dict(step=it, ok=ok, pre=pre, post=post, layout=self.in_layout, work_ran=bool(pre["ok"])))
            if not ok:
                if self.golden_mode == "check":
                    self.gate_fail_steps.append(it)
                    if self.in_layout:
                        self.layout_fails += 1
                    else:
                        self.fails += 1
                    if post is not None and not post["ok"]:
                        self.log("GATE_FAIL step %d AFTER the advance: %s (the step's rows are excluded)" % (it, post.get("why")))
                elif self.golden_mode == "record":
                    self.golden_fail.append(it)
        finally:
            self.guard.end_step()

    # ---------------------------------------------------------------------------------- the golden gate (cpupack_golden)
    def state_digest(self, families):
        return G.state_digest(self.cache, families, R=self.R)

    def start_digest(self):
        return self.state_digest(G.FAMILIES_START)

    def gate_pre(self, it, lg_ref, ctx):
        """BEFORE any timing: digests after the flushed reference decode, then the pre-timing resident check (restore ->
        decode: 0 loads, logits torch.equal the reference). 'record' stores them as the golden; 'check' compares."""
        rec = dict(step=it, mode=self.golden_mode, ok=True, why=[])
        if self.golden_mode is None:
            return rec
        t = time.time()
        rec["ref_sha"] = G.sha_parts([lg_ref])
        rec["pre"] = self.state_digest(G.FAMILIES_PRE)
        ctx["restore"]()
        lg = ctx["step_fn"]()
        self.sync()
        rec["resident"] = dict(loads=self.loaded_now(), logits_equal=bool(torch.equal(lg, lg_ref)))
        why = [] if (rec["resident"]["loads"] == 0 and rec["resident"]["logits_equal"]) else \
            ["resident(loads=%d, logits_equal=%s)" % (rec["resident"]["loads"], rec["resident"]["logits_equal"])]
        if self.golden_mode == "record":
            self.golden["steps"].setdefault(it, {}).update(ref_sha=rec["ref_sha"], pre=rec["pre"], resident=rec["resident"])
        else:
            g = self.golden["steps"].get(it)
            if g is None:
                why.append("no golden record for step %d" % it)
            else:
                if g.get("ref_sha") != rec["ref_sha"]:
                    why.append("ref_logits")
                why += G.compare(g.get("pre"), rec["pre"])
        rec["why"], rec["ok"] = why[:64], not why
        rec["seconds"] = time.time() - t
        return rec

    def gate_post(self, it, lg_adv):
        rec = dict(step=it, mode=self.golden_mode, ok=True, why=[])
        if self.golden_mode is None:
            return rec
        t = time.time()
        rec["adv_sha"] = G.sha_parts([lg_adv])
        rec["post"] = self.state_digest(G.FAMILIES_POST)
        why = []
        if self.golden_mode == "record":
            self.golden["steps"].setdefault(it, {}).update(adv_sha=rec["adv_sha"], post=rec["post"])
        else:
            g = self.golden["steps"].get(it)
            if g is None:
                why.append("no golden record for step %d" % it)
            else:
                if g.get("adv_sha") != rec["adv_sha"]:
                    why.append("adv_logits")
                why += G.compare(g.get("post"), rec["post"])
        rec["why"], rec["ok"] = why[:64], not why
        rec["seconds"] = time.time() - t
        return rec

    def row_marks(self):
        return (len(self.rows), len(self.layout_rows), len(self.timeline), id(self.correct))

    def stamp_rows(self, marks, it, ok):
        """Every row a gated step produced carries its gate verdict (TABLE keeps only gate_ok rows)."""
        for lst, m in ((self.rows, marks[0]), (self.layout_rows, marks[1]), (self.timeline, marks[2])):
            for r in lst[m:]:
                r["gate_ok"] = bool(ok)
                r["gate_step"] = it
        if self.correct and id(self.correct) != marks[3]:
            self.correct["gate_ok"] = bool(ok)
            self.correct["gate_step"] = it

    def resident(self, ctx, label=None):
        ctx["restore"]()
        lg, r = self.bracket(dict(name="alone", kind="alone"), None, True, ctx["step_fn"], nvtx=label)
        r["logits_equal"] = bool(torch.equal(lg, ctx["lg_ref"]))
        r["loads"] = self.loaded_now()
        r["ok"] = r["logits_equal"] and r["loads"] == 0
        return r

    def arm_reps(self, a, ctx, c, reps, rows, tag):
        """warm-up (untimed, recorded), REPS transfer-alone, REPS beside the decode."""
        if a.get("cores"):
            self.coord.configure(a["cores"])
        th0 = CC.census()
        _, w = self.bracket(a, c, False)
        w.update(step=ctx["it"], plan_step=c["key"][0], phase="warmup", rep=0, stage=tag)
        nf = int(not self.rep_checks(w, c, c["union"], c["last"]))      # the warm-up is untimed but its correctness counts
        rows.append(w)
        for mode in ("alone", "conc"):
            for rep in range(reps):
                if mode == "conc":
                    ctx["restore"]()
                lg, r = self.bracket(a, c, mode == "conc", ctx["step_fn"])
                r.update(step=ctx["it"], plan_step=c["key"][0], phase=mode, rep=rep, stage=tag)
                ok = self.rep_checks(r, c, c["union"], c["last"], lg, ctx["lg_ref"] if mode == "conc" else None)
                nf += int(not ok)
                rows.append(r)
        self.thread_use.append(dict(stage=tag, step=ctx["it"], arm=a["name"], team=(self.coord.team[:a["cores"]] if a.get("cores") else None),
                                    delta=CC.census_delta(th0, CC.census())))
        return nf

    # ---------------------------------------------------------------------------------------------- CORRECT
    def exact_layer(self, method, c, l, cap=None, faults=None):
        """Single-layer run of `method` ('direct' | an arm) on plan rows c; poisoned scratch; the layer's exact rows and
        content by logical indexing."""
        ref = self.ref_layer(c["cpu"], l)
        exp = self.dest_mask(c["cpu"], [l])
        if method == "direct":
            from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
            torch.cuda.synchronize()
            K.poison_(self.scr_k)
            K.poison_(self.scr_v)
            ids = c["gpu"][l][1 + self.HB:].view(self.H, self.B, self.M)
            flash_h2d_from_mask(self.scr_log(0), self.src_log(l, 0), ids, self.R)
            flash_h2d_from_mask(self.scr_log(1), self.src_log(l, 1), ids, self.R)
            torch.cuda.synchronize()
            r = dict(arm="direct")
        else:
            if method.get("cores"):
                self.coord.configure(method["cores"])
            _, r = self.bracket(method, c, False, layers=[l], faults=faults, cap=cap)
            if "coord_error" in r:
                return dict(ok=False, error=r["coord_error"][-400:], layer=l, arm=r["arm"])
        ch = self.check_scratch(exp, ref)
        return dict(ok=ch["canary_ok"] and ch["content_ok"], layer=l, arm=r["arm"], **ch)

    def lifetime_control_layer(self, cpu_rows):
        """The lifetime controls need MORE chunks than ring slots (every slot reused). The layer with the most natural groups
        of the correctness step, with the chunk cap lowered from CP_LIFETIME_CAP_GROUPS until n_chunks >= RING + 2 when the
        layer is thin; 'testable' is False (a loud failure, verdict UNTESTABLE) only if even cap 1 gives <= RING chunks."""
        per = []
        for l in range(self.NL):
            p = cpu_rows[l][1 + self.HB:].view(self.H, self.B, self.M)
            per.append(p[..., :K.TAIL_SLOT].ge(0).sum(dim=(0, 2)).tolist())
        tot = [sum(x) for x in per]
        l = max(range(self.NL), key=lambda i: (tot[i], -i))
        cap = LIFETIME_CAP
        n = len(K.chunk_requests(per[l], cap))
        while n < RING + 2 and cap > 1:
            cap = max(1, cap // 2)
            n = len(K.chunk_requests(per[l], cap))
        return dict(layer=l, groups=tot[l], cap=cap, n_chunks=n, ring=RING, testable=n > RING)

    def synthetic_rows(self, kind, seed=0):
        """Layer-0 synthetic plans (labelled SYNTHETIC): 'empty', 'one_full' (stream (0, 0) loads 63 blocks), 'dup_src' (two
        slots of stream (0, 0) load the same host block), 'perm' (independent random per-head plans)."""
        g = torch.Generator().manual_seed(seed)
        plan = torch.full((self.H, self.B, self.M), -1, dtype=torch.int32)
        nb = VA.L // self.R
        if kind == "one_full":
            plan[0, 0, :K.TAIL_SLOT] = torch.randperm(nb, generator=g)[:K.TAIL_SLOT].to(torch.int32)
        elif kind == "dup_src":
            plan[0, 0, 3] = 17
            plan[0, 0, 40] = 17
            plan[1, self.B - 1, 5] = 17
        elif kind == "perm":
            for h in range(self.H):
                for b in range(self.B):
                    k = int(torch.randint(0, 8, (1,), generator=g))
                    if k:
                        plan[h, b, torch.randperm(K.TAIL_SLOT, generator=g)[:k]] = torch.randperm(nb, generator=g)[:k].to(torch.int32)
        elif kind != "empty":
            raise ValueError(kind)
        cpu = torch.zeros((self.NL, self.W), dtype=torch.int32)
        cpu[:, 1 + self.HB:] = -1
        cpu[0, 1:1 + self.HB] = (plan[..., :K.TAIL_SLOT] >= 0).sum(-1).reshape(-1)
        cpu[0, 1 + self.HB:] = plan.reshape(-1)
        return cpu.to(self.dev), cpu

    def correct_stage(self, ctx):
        """The CORRECTNESS gate (module docstring). Returns the failure count; the receipts go into self.correct."""
        ps = ctx["it"]
        res = dict(plan_step=ps, label=K.LABEL)
        nf = 0
        gpu, cpu = self.rows_of(ps)
        res["store_equals_capture"] = bool(torch.equal(gpu.cpu(), cpu))
        nf += int(not res["store_equals_capture"])
        # descriptors vs nosi_items (orig layout; the sglang_hicache plan.py address formulas are orig-only)
        from nosi.flash_cache_engine.sglang_hicache import plan as HCP
        bad = 0
        for l in range(self.NL):
            p = cpu[l][1 + self.HB:].view(self.H, self.B, self.M)
            d = K.build_desc(p, src_layout="orig", dst_layout="orig", s_src=self.S_cpu, s_dst=self.S_dst)
            K.check_unique_dst(d)
            si, di = HCP.nosi_items(p, s_cpu=self.S_cpu, s_dst=self.S_dst, block_rows=self.R, item="i256")
            mine = set(zip(d.src_idx.tolist(), d.dst_idx.tolist()))
            bad += int(mine != set(zip(si.tolist(), di.tolist())))
            ov = K.overfetch_report(d)
            bad += int(not ov["ok"])
        res["desc_vs_nosi_items_bad_layers"] = bad
        nf += int(bad > 0)
        c = dict(gpu=gpu, cpu=cpu, useful=[K.useful_bytes(int((cpu[l][1 + self.HB:] >= 0).sum())) for l in range(self.NL)])
        cpu8 = arm("cpu%d" % max(CORES), "cpu", cores=max(CORES))
        per = []
        for l in range(self.NL):
            for m in ("direct", arm("w8", "w8"), cpu8):
                x = self.exact_layer(m, c, l)
                per.append(x)
                nf += int(not x["ok"])
        res["per_layer"] = per
        res["per_layer_all_ok"] = all(x["ok"] for x in per)
        g0, c0 = self.rows_of(0)
        full = dict(gpu=g0, cpu=c0, useful=[K.useful_bytes(int((c0[l][1 + self.HB:] >= 0).sum())) for l in range(self.NL)])
        res["step0_full"] = [self.exact_layer(m, full, l) for l in (0, self.NL - 1) for m in (arm("w8", "w8"), cpu8)]
        nf += sum(int(not x["ok"]) for x in res["step0_full"])
        syn = []
        for kind in ("empty", "one_full", "dup_src", "perm"):
            sg, sc = self.synthetic_rows(kind)
            sc_ = dict(gpu=sg, cpu=sc, useful=[K.useful_bytes(int((sc[l][1 + self.HB:] >= 0).sum())) for l in range(self.NL)])
            for m in ("direct", arm("w8", "w8"), cpu8):
                x = self.exact_layer(m, sc_, 0)
                x["synthetic"] = kind
                syn.append(x)
                nf += int(not x["ok"])
        res["synthetic"] = syn
        # negative controls: each must be DETECTED (the check fails); positive lifetime controls must pass
        lc = self.lifetime_control_layer(cpu)
        res["lifetime_control"] = lc
        negs = []
        for name, fl, expect_fail, cap in (
                ("skip_scatter", K.Faults(skip_scatter_chunk=0), True, None),
                ("poison_stage", K.Faults(poison_stage=True), True, None),
                ("no_slot_wait+delay_copy", K.Faults(no_slot_wait=True, delay_copy_ms=LIFETIME_DELAY_MS), True, lc["cap"]),
                ("slot_wait+delay_copy (positive)", K.Faults(delay_copy_ms=LIFETIME_DELAY_MS), False, lc["cap"]),
                ("no_landing_wait+delay_scatter", K.Faults(no_landing_wait=True, delay_scatter_ms=LIFETIME_DELAY_MS), True, lc["cap"]),
                ("landing_wait+delay_scatter (positive)", K.Faults(delay_scatter_ms=LIFETIME_DELAY_MS), False, lc["cap"]),
                ("no_h2d_wait+delay_copy", K.Faults(no_h2d_wait=True, delay_copy_ms=LIFETIME_DELAY_MS), True, lc["cap"])):
            lifetime = cap is not None
            if lifetime and not lc["testable"]:
                x = dict(ok=False, layer=lc["layer"], arm=cpu8["name"], error="UNTESTABLE: %d chunks <= ring %d" % (lc["n_chunks"], RING))
            else:
                x = self.exact_layer(cpu8, c, lc["layer"] if lifetime else 0, cap=cap, faults=fl)
            x.update(control=name, expect_fail=expect_fail, cap=cap, n_chunks=(lc["n_chunks"] if lifetime else None))
            if lifetime and not lc["testable"]:
                x["verdict"] = "UNTESTABLE"
            else:
                x["verdict"] = ("DETECTED" if not x["ok"] else "NOT_DETECTED") if expect_fail else ("PASS" if x["ok"] else "FAIL")
            nf += int(x["verdict"] in ("NOT_DETECTED", "FAIL", "UNTESTABLE"))
            negs.append(x)
        res["controls"] = negs
        # decode: the zero-load resident step (restore -> decode: legal, the guard is armed by the restore). The DESTRUCTIVE
        # logits negative control of job 2179683 (a second decode WITHOUT a restore; its gathers overwrite live window KV
        # and tail state that the CounterSnapshot does not restore, and gated() then advanced from it) is NOT run in a
        # measurement process: it runs in the separate CONTROLS process (run_controls), first; a failure there stops the job.
        ctx["restore"]()
        lg1 = ctx["step_fn"]()
        self.sync()
        res["resident_zero_load"] = self.loaded_now() == 0 and bool(torch.equal(lg1, ctx["lg_ref"]))
        res["logits_negative_control"] = "MOVED_TO_CONTROLS_PROCESS"
        ctx["restore"]()
        nf += int(not res["resident_zero_load"])
        res["fails"] = nf
        self.chk = {}
        return nf, res

    # ---------------------------------------------------------------------------------------------- MAIN / SWEEP
    def main_step(self, ctx, arms):
        ps = ctx["it"]
        c = self.prepare(ps, arms)
        nf = 0
        rows = []
        for rep in range(REPS):
            r = self.resident(ctx)
            r.update(step=ps, plan_step=ps, phase="decode_alone_pre", rep=rep, stage="MAIN", agreement=ctx["agreement"])
            nf += int(not r["ok"])
            rows.append(r)
        order = list(arms)
        if ARM_ORDER == "alternate" and ps % 2:                          # balance the arms' position inside the step
            order = order[::-1]
        self.arm_orders.append(dict(step=ps, order=[a["name"] for a in order]))
        for a in order:
            nf += self.arm_reps(a, ctx, c, REPS, rows, "MAIN")
        for rep in range(REPS):
            r = self.resident(ctx)
            r.update(step=ps, plan_step=ps, phase="decode_alone_post", rep=rep, stage="MAIN")
            nf += int(not r["ok"])
            rows.append(r)
        self.rows.extend(rows)
        return nf

    def sweep_stage(self):
        """Transfer-alone w8 and cpu8 over every captured step (steady 1.. and the step-0 full list LAST). Bounded by
        CP_STAGE_DEADLINE: a step that its observed step time does not fit is skipped (sweep_notes, partial)."""
        nf = 0
        arms = [arm("w8", "w8"), arm("cpu%d" % max(CORES), "cpu", cores=max(CORES))]
        self.coord.configure(max(CORES))
        order = list(range(1, self.masks.shape[0])) + [0]
        took = []
        for i, ps in enumerate(order):
            est = max(took) if took else SWEEP_STEP_EST_S
            if STAGE_DEADLINE and time.time() + est > STAGE_DEADLINE - FINAL_RESERVE_S:
                self.sweep_notes.append("deadline: SWEEP stopped before plan step %d (%d of %d steps done)" % (ps, i, len(order)))
                break
            t_s = time.time()
            c = self.prepare(ps, arms)
            for a in arms:
                _, r = self.bracket(a, c, False)
                r.update(plan_step=ps, phase="sweep_alone", rep=0, stage="SWEEP", full_list=(ps == 0))
                ok = self.rep_checks(r, c, c["union"], c["last"])
                nf += int(not ok)
                self.sweep.append(r)
            took.append(time.time() - t_s)
        self.chk = {}
        return nf

    # ---------------------------------------------------------------------------------------------- LAYOUT (layout_ablation)
    def tail_write_bench(self, reps=5):
        """The section-D write form of cache_engine.py:707-708 (`host[:, p:p+64].copy_(window[:, t:t+64], non_blocking=True)`)
        into a small pinned buffer of each layout ([B, 128, H, D] orig, [B, H, 128, D] head-major via its logical view):
        host-timed (perf_counter + synchronize) and CUDA-event-timed per tensor. The pitch differs from the real cache's;
        the run structure (B runs of 32 KiB vs B*H runs of 16 KiB) is the real one."""
        e = self.engines[0]
        t_slot = int(e._tail_block_idx_on_gpu)
        src = e._k_gpu[:, t_slot * self.R:(t_slot + 1) * self.R]
        out = {}
        for lay in K.LAYOUTS:
            phys = K.alloc_phys(lay, self.B, 2 * self.R, self.H, self.D, torch.bfloat16, "cpu", pin=True)
            log = K.logical(phys, lay)
            host, gpu = [], []
            for i in range(reps + 1):
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t = time.perf_counter()
                e0.record()
                log[:, self.R:2 * self.R].copy_(src, non_blocking=True)
                e1.record()
                torch.cuda.synchronize()
                if i:
                    host.append(1000 * (time.perf_counter() - t))
                    gpu.append(e0.elapsed_time(e1))
            okc = K.bits_equal(log[:, self.R:2 * self.R].to("cuda"), src)
            med = float(np.median(host))
            out[lay] = dict(host_ms=host, event_ms=gpu, content_ok=okc, pinned_bytes=phys.numel() * 2, pinned_reserved=pow2(phys.numel() * 2),
                            accounting=K.tail_write_accounting(lay, self.B, self.NL, measured_ms_per_tensor=med))
            del phys, log
        return out

    def tail_write_real(self, reps=3):
        """N1 realism: the SAME section-D write form (cache_engine.py:707-708) through the REAL host tensors' logical views
        (e._k_cpu / e._v_cpu: the real per-request pitch and NUMA placement, orig or head-major), every layer's K and V, into
        rows [S_cpu - R, S_cpu), which nothing reads in this process: plans name blocks < L/R + 1 and no rollover happens
        (checked here; skipped with a note otherwise). Per-rollover cost = the SUM of the 2*NL measured per-tensor medians."""
        R, lo = self.R, self.S_cpu - self.R
        max_blk = int(self.masks.max()) if hasattr(self, "masks") else -1
        if lo < VA.L + R * (VA.N // R + 2) or (max_blk + 1) * R > lo:
            return dict(layout=self.host_layout, skipped="rows [%d, %d) not provably unread (L=%d N=%d max planned block %d)"
                        % (lo, self.S_cpu, VA.L, VA.N, max_blk))
        host_ms, ev_ms, ok, restored = [], [], True, True
        for e in self.engines:
            t_slot = int(e._tail_block_idx_on_gpu)
            for win, host in ((e._k_gpu, e._k_cpu), (e._v_gpu, e._v_cpu)):
                src = win[:, t_slot * R:(t_slot + 1) * R]
                dst = host[:, lo:lo + R]
                keep = dst.clone()                                       # the confirmation: live host rows are put back below
                hs, es = [], []
                for i in range(reps + 1):
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t = time.perf_counter()
                    e0.record()
                    dst.copy_(src, non_blocking=True)
                    e1.record()
                    torch.cuda.synchronize()
                    if i:
                        hs.append(1000 * (time.perf_counter() - t))
                        es.append(e0.elapsed_time(e1))
                ok &= K.bits_equal(dst.to("cuda"), src)
                torch.cuda.synchronize()
                dst.copy_(keep)                                          # no write to live engine state survives the bench
                restored &= K.bits_equal(dst, keep)
                del keep
                host_ms.append(float(np.median(hs)))
                ev_ms.append(float(np.median(es)))
        roll = float(sum(host_ms))
        if not restored:
            self.layout_fails += 1
        return dict(layout=self.host_layout, rows=[lo, self.S_cpu], tensors=len(host_ms), reps=reps, host_ms_per_tensor=host_ms,
                    event_ms_per_tensor=ev_ms, content_ok=bool(ok), host_rows_restored=bool(restored),
                    measured_ms_per_rollover=roll, measured_ms_per_token=roll / R,
                    accounting=K.tail_write_accounting(self.host_layout, self.B, self.NL, measured_ms_per_tensor=float(np.median(host_ms))),
                    note="the real cache tensors through the engine's own copy_ form; a torch D2H into a non-contiguous host view "
                         "goes through a contiguous temporary (Copy.cu copy_requires_temporaries) in BOTH layouts")

    def _convert_layer(self, l, frm, to):
        """One layer's K and V host tensors reordered in place; the engine's tensors become the new logical views."""
        e = self.engines[l]
        new, infos = [], []
        for which in (0, 1):
            phys = self.host_phys[l][which]
            np_, info = K.convert_inplace(phys, frm, to, verify=True)
            assert np_.data_ptr() == phys.data_ptr()
            new.append(np_)
            infos.append(info)
        self.host_phys[l] = (new[0], new[1])
        e._k_cpu, e._v_cpu = K.logical(new[0], to), K.logical(new[1], to)
        assert K.layout_of(e._k_cpu) == to and K.layout_of(e._v_cpu) == to
        assert self.dev != "cuda" or (e._k_cpu.is_pinned() and e._v_cpu.is_pinned())
        return infos

    def convert_host(self, to, deadline=None):
        """In-place reorder of every layer's K and V host tensor (cpupack_core.convert_inplace; bit-exact per-request logical
        check); the engines read the new layout through their logical views (the shipped gathers take strides:
        flash_h2d_mask.py:76-77 / :41-56, flash_h2d_mask_bias.py:97-98 / :62-77). DEADLINE GUARD (layer granularity): after
        each layer, if now + (layers left) x (slowest layer so far) passes `deadline`, the conversion stops and takes the
        cheaper way back to ONE layout (revert the converted layers, or finish); 'aborted' records it and the caller skips
        the head-major cells. The time is split into ms_reorder and ms_verify (the verification is not a conversion cost).
        NOT the production path: a head-major host cache would be written at prefill by the strided D2H of
        cache_engine.py:422 into the logical view; this is the ablation's one-time reorder of an orig cache."""
        frm = self.host_layout
        per, bad, ms, ms_re, ms_ve, lay_s = [], 0, [], [], [], []
        if self.dev == "cuda":
            torch.cuda.synchronize()                                     # no gather may be reading the host cache
        t0 = time.time()
        aborted, done = None, 0

        def book(l, infos, direction):
            nonlocal bad
            for info in infos:
                bad += info["bad"]
                ms.append(info["ms"])
                ms_re.append(info["ms_reorder"])
                ms_ve.append(info["ms_verify"])
            per.append(dict(layer=l, direction=direction, ms_k=infos[0]["ms"], ms_v=infos[1]["ms"], bad=infos[0]["bad"] + infos[1]["bad"]))
        for l in range(self.NL):
            tl = time.time()
            book(l, self._convert_layer(l, frm, to), "forward")
            done = l + 1
            lay_s.append(time.time() - tl)
            if deadline is not None and done < self.NL:
                left = (self.NL - done) * max(lay_s)
                if time.time() + left > deadline:
                    spent = time.time() - t0
                    aborted = dict(after_layers=done, spent_s=spent, left_s_est=left, action=("revert" if spent <= left else "finish"))
                    break
        final = to
        if aborted is not None:
            if aborted["action"] == "revert":
                for l in range(done):
                    book(l, self._convert_layer(l, to, frm), "revert")
                final = frm
            else:
                for l in range(done, self.NL):
                    book(l, self._convert_layer(l, frm, to), "forward")
        self.host_layout = final
        self.chk = {}
        fwd = [x for x in per if x["direction"] == "forward"]
        acc = K.conversion_accounting(self.B, self.S_cpu, self.H, self.D, 2, self.NL, [v for x in fwd for v in (x["ms_k"], x["ms_v"])])
        return dict(frm=frm, to=to, final_layout=final, bad_requests=bad, per_layer=per, accounting=acc, extra_pinned_bytes=0,
                    seconds=time.time() - t0, ms_total=float(sum(ms)), ms_reorder_total=float(sum(ms_re)), ms_verify_total=float(sum(ms_ve)),
                    aborted=aborted, deadline_epoch=deadline,
                    note="one-time ablation reorder of an orig cache (NOT the production path: a head-major cache would be written at "
                         "prefill through the strided D2H of cache_engine.py:422)")

    def predict_conversion(self, to):
        """The conversion-time prediction BEFORE any head-major cell: an in-place round trip of 3 requests (K and V of the
        first / middle / last layer, first / middle / last request; cpupack_core.probe_convert, the same per-request code,
        the data left bit-identical) and, when CP_CALIB_JSON is readable, C0's measured per-tensor time scaled by bytes.
        predicted = CP_CONV_SAFETY x max(probe, C0)."""
        if self.dev == "cuda":
            torch.cuda.synchronize()
        picks = sorted({(0, 0), (self.NL // 2, self.B // 2), (self.NL - 1, self.B - 1)})
        probes = []
        for l, b in picks:
            for which in (0, 1):
                pr = K.probe_convert(self.host_phys[l][which], self.host_layout, b)
                pr.update(layer=l, tensor="kv"[which])
                probes.append(pr)
        per_req = max(p["ms_fwd"] for p in probes)
        probe_s = per_req * self.B * 2 * self.NL / 1000.0
        c0_s, c0_note = None, None
        if CALIB_JSON:
            try:
                with open(CALIB_JSON) as f:
                    cj = json.load(f)
                pt = cj["conversion"]["per_tensor"]
                c0_s = float(np.mean([x["ms"] for x in pt])) * (self.B * self.S_cpu) / float(cj["B"] * cj["S"]) * 2 * self.NL / 1000.0
            except Exception as e:                                       # a missing C0 leaves the probe alone
                c0_note = "C0 calibration unusable: %r" % (e,)
        return dict(to=to, probes=probes, probe_bad=sum(1 for p in probes if p["bad"] or not p["restored"]), per_request_ms_max=per_req,
                    probe_s=probe_s, c0_s=c0_s, c0_note=c0_note, calib_json=CALIB_JSON or None, safety=CONV_SAFETY,
                    predicted_s=CONV_SAFETY * max(probe_s, c0_s or 0.0))

    def layout_deadline(self, t0):
        d = t0 + LAYOUT_BUDGET_S
        if STAGE_DEADLINE:
            d = min(d, STAGE_DEADLINE - FINAL_RESERVE_S)
        return d

    def layout_resident(self, ctx, phase, cellk, ps):
        """LAYOUT_REPS decode-alone reps at THIS gated step (the same-step denominator of the cell's slowdown)."""
        for rep in range(LAYOUT_REPS):
            r = self.resident(ctx)
            r.update(step=ctx["it"], plan_step=ps, phase=phase, rep=rep, stage="LAYOUT", cell=cellk)
            self.layout_fails += int(not r["ok"])
            self.layout_rows.append(r)

    def layout_peak_check(self, it):
        """Registered trigger (c) seen inside LAYOUT: MAIN is done, so it is a layout note + failure, never a fallback."""
        if self.dev != "cuda" or self.layout["memgate"]:
            return
        pk = torch.cuda.max_memory_reserved() / 1e9
        if pk > PEAK_LIMIT_GB:
            self.layout["memgate"].append(dict(step=it, peak_reserved_gb=pk, note="LAYOUT peak reserved above %.1f GB: a layout "
                                               "failure, NOT a fallback trigger (MAIN is done)" % PEAK_LIMIT_GB))
            self.layout_fails += 1

    def layout_stage(self, steps_iter, pos_of):
        """layout_ablation: the 2x2 (module docstring). steps_iter yields free decode steps; each (cell, plan step) uses one
        gated decode step: LAYOUT_REPS decode-alone reps, the variants (each: warm-up, LAYOUT_REPS alone, LAYOUT_REPS beside
        the decode), LAYOUT_REPS decode-alone reps. Failures go to self.layout_fails (separate from the original arms').
        Deadline = min(start + CP_LAYOUT_BUDGET_S, CP_STAGE_DEADLINE - CP_FINAL_RESERVE_S); a step is started only if the
        slowest step so far still fits; the head-major cells only if the predicted conversion + one step fit."""
        t0 = time.time()
        deadline = self.layout_deadline(t0)
        L = self.layout
        L.update(deadline_epoch=deadline, stage_deadline_epoch=STAGE_DEADLINE, budget_s=LAYOUT_BUDGET_S, step_s=[])
        dev = self.dev
        est = lambda: max(L["step_s"]) if L["step_s"] else LAYOUT_STEP_EST_S

        def stop(msg):
            L["partial"] = True
            L["notes"].append(msg)
            self.log("LAYOUT: " + msg)
        if time.time() + est() > deadline:
            stop("not started: %.0f s left < one step estimate %.0f s" % (deadline - time.time(), est()))
            return
        L["tail_write"] = self.tail_write_bench()
        L["tail_write_real"][self.host_layout] = self.tail_write_real()
        orig_refs = {}                                                   # references computed BEFORE any conversion
        self.set_scratch("orig")
        for ps in LAYOUT_PLAN_STEPS:
            _, cpu = self.rows_of(ps)
            for l in sorted(set(LAYOUT_CHECK_LAYERS) | {self.NL - 1}):
                r = self.ref_layer(cpu, l)
                orig_refs[(ps, l)] = dict(rk=r["rk"].cpu(), rv=r["rv"].cpu())
        halt = False
        for src, dst in list(LAYOUT_CELLS_PRE) + list(LAYOUT_CELLS_POST):
            if halt:
                break
            cellk = "%s>%s" % (src, dst)
            conv = None
            if src != self.host_layout:
                pred = self.predict_conversion(src)
                L.setdefault("conversion_predictions", []).append(pred)
                if pred["probe_bad"]:
                    self.layout_fails += 1
                    stop("conversion probe NOT exact (%d probes): head-major cells skipped" % pred["probe_bad"])
                    break
                if time.time() + pred["predicted_s"] + est() > deadline:
                    stop("head-major cells skipped: predicted conversion %.0f s + one step %.0f s > %.0f s left"
                         % (pred["predicted_s"], est(), deadline - time.time()))
                    break
                conv = dict(deadline=deadline - est(), pred=pred)
            elif time.time() + est() > deadline:
                stop("deadline before cell %s" % cellk)
                break
            self.set_scratch(dst)
            cell = dict(src=src, dst=dst, variants=[a["name"] for a in layout_variants(src, dst, max(CORES))], exact=[], steps=[])
            L["cells"].append(cell)
            for j, ps in enumerate(LAYOUT_PLAN_STEPS):
                if j and time.time() + est() > deadline:
                    stop("deadline inside cell %s after %d plan steps" % (cellk, j))
                    halt = True
                    break
                it = next(steps_iter)
                box = dict(conv_s=0.0, aborted=False)

                def work(ctx, ps=ps, j=j, box=box):
                    nonlocal conv
                    if conv is not None:
                        tc = time.time()
                        res = self.convert_host(src, deadline=conv["deadline"])
                        res["prediction"] = conv["pred"]
                        conv = None
                        ctx["restore"]()
                        for e in self.engines:
                            self.mc.flush_map(e)
                        lg2 = self.decode(ctx["tok"], ctx["pos"])
                        if dev == "cuda":
                            torch.cuda.synchronize()
                        res["reference_logits_equal_before_after"] = bool(torch.equal(lg2, ctx["lg_ref"]))
                        res["loads_of_reference_after"] = self.loaded_now()
                        self.layout_fails += int(res["bad_requests"] > 0) + int(not res["reference_logits_equal_before_after"])
                        L["conversion"] = res
                        box["conv_s"] = time.time() - tc
                        self.flush_payload(True)
                        if res["aborted"] is not None:
                            box["aborted"] = True
                            return
                    c = self.prepare(ps, [])
                    # the stored pre-conversion reference replaces the live one: content must equal the ORIGINAL bytes
                    ref = c["last"]
                    o = orig_refs[(ps, self.NL - 1)]
                    ref["rk"], ref["rv"] = o["rk"].to(dev), o["rv"].to(dev)
                    self.layout_resident(ctx, "decode_alone_pre", cellk, ps)
                    for a in layout_variants(src, dst, max(CORES)):
                        if j == 0:
                            for l in LAYOUT_CHECK_LAYERS:
                                x = self.exact_layer(a, c, l)
                                rr = self.ref_layer(c["cpu"], l)
                                o2 = orig_refs[(ps, l)]
                                x["live_ref_equals_original"] = K.bits_equal(rr["rk"].cpu(), o2["rk"]) and K.bits_equal(rr["rv"].cpu(), o2["rv"])
                                x["ok"] = bool(x["ok"] and x["live_ref_equals_original"])
                                cell["exact"].append(dict(x, variant=a["name"]))
                                self.layout_fails += int(not x["ok"])
                        rows = []
                        self.layout_fails += self.arm_reps(a, ctx, c, LAYOUT_REPS, rows, "LAYOUT")
                        for r in rows:
                            r["cell"] = cellk
                        self.layout_rows.extend(rows)
                    self.layout_resident(ctx, "decode_alone_post", cellk, ps)
                    cell["steps"].append(dict(decode_step=ctx["it"], plan_step=ps, agreement=ctx["agreement"], tail_len_max=ctx.get("tail_len_max")))
                ts = time.time()
                self.gated(it, pos_of(it), work)
                L["step_s"].append(time.time() - ts - box["conv_s"])
                self.layout_peak_check(it)
                self.flush_payload(True)
                if box["aborted"]:
                    stop("conversion stopped by its deadline guard (%s): head-major cells skipped" % (L["conversion"]["aborted"],))
                    halt = True
                    break
            if self.host_layout not in L["tail_write_real"]:
                L["tail_write_real"][self.host_layout] = self.tail_write_real()
        if LAYOUT_BACK and self.host_layout != "orig":
            pred = self.predict_conversion("orig")
            if time.time() + pred["predicted_s"] <= deadline:
                L["conversion_back"] = self.convert_host("orig", deadline=deadline)
            else:
                L["notes"].append("conversion back skipped: predicted %.0f s past the deadline" % pred["predicted_s"])
        L["seconds"] = time.time() - t0

    def run_layout(self, steps_iter, pos_of):
        """LAYOUT isolated from the job's control flow: ANY exception (OOM and pinned-allocation failures included) is a
        layout failure with a note and the traceback; it never reaches main()'s exit-code rule (so it can never discard the
        batch's MAIN or start a B320 rerun)."""
        self.in_layout = True                                            # a LAYOUT GATE_FAIL is a layout failure
        try:
            self.layout_stage(steps_iter, pos_of)
        except StopIteration:
            self.layout["notes"].append("ran out of no-rollover decode steps")
            self.layout["partial"] = True
        except Exception as e:
            self.layout_fails += 1
            self.layout["partial"] = True
            self.layout["error"] = traceback.format_exc()[-4000:]
            self.layout["notes"].append("LAYOUT aborted by %s: %s" % (type(e).__name__, str(e)[:300]))
            self.log("LAYOUT aborted by %s (MAIN results are kept): %s" % (type(e).__name__, str(e)[:300]))
        self.in_layout = False
        self.snap_inventory("after_layout")
        try:
            self.flush_payload(True)
        except Exception as e:                                           # the final flush in run() retries
            self.log("flush after LAYOUT failed: %r" % (e,))

    def mark_main_done(self, path=None):
        """The main-done marker (module docstring): written once MAIN + SWEEP are flushed and passed the memory gate."""
        path = path or os.path.join(OUT, "%s.main_done" % TAG)
        rec = dict(tag=TAG, batch=self.B, fails=self.fails, epoch=time.time(), seconds=time.time() - self.t_start, sweep_notes=self.sweep_notes)
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        self.main_done = True
        return path

    # ---------------------------------------------------------------------------------------------- TIMELINE
    def timeline_step(self, ctx):
        nf = 0
        c = self.prepare(ctx["it"], timeline_arms())
        for a in timeline_arms():
            if a.get("cores"):
                self.coord.configure(a["cores"])
            late = T_LATE_MS if a["name"] == "alone_late" else 0.0
            for phase, n in (("untraced_pre", T_UNTRACED), ("traced", 1 + T_REPS), ("untraced_post", T_UNTRACED)):
                if phase == "traced":
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                for rep in range(n):
                    ctx["restore"]()
                    label = "lt|b%d|%s|step%d|%s|rep%d" % (self.B, a["name"], ctx["it"], phase, rep)
                    lg, r = self.bracket(a, c, True, ctx["step_fn"], nvtx=label, late_ms=late)
                    r.update(batch=self.B, step=ctx["it"], plan_step=ctx["it"], phase=phase, rep=rep, label=label,
                             warmup=(phase == "traced" and rep == 0), stage="TIMELINE")
                    if a["kind"] == "alone":
                        r["logits_equal"] = bool(torch.equal(lg, ctx["lg_ref"]))
                        r["loads"] = self.loaded_now()
                        r["ok"] = r["logits_equal"] and r["loads"] == 0
                    else:
                        self.rep_checks(r, c, c["union"], c["last"], lg, ctx["lg_ref"])
                    r["transfer_equal"] = r.get("content_ok", True) and r.get("canary_ok", True)
                    nf += int(not r["ok"])
                    self.timeline.append(r)
                if phase == "traced":
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
        return nf

    # ---------------------------------------------------------------------------------------------- inventory / payload
    def inventory(self):
        e0 = self.engines[0] if hasattr(self, "engines") else None
        host_bytes = sum(t.numel() * t.element_size() for pair in getattr(self, "host_phys", []) for t in pair)
        inv = dict(
            hbm=dict(peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                     free_gb=torch.cuda.mem_get_info()[0] / 1e9, total_gb=torch.cuda.mem_get_info()[1] / 1e9,
                     scratch_bytes=(2 * self.scr_k.numel() * 2 if hasattr(self, "scr_k") else 0), landing_bytes=self.land.numel(),
                     contig_dev_bytes=self.contig_dev.numel(), plan_store_bytes=(self.store.numel() * 4 if hasattr(self, "store") else 0)),
            pinned=dict(host_cache_bytes=host_bytes,
                        host_cache_reserved_pow2=sum(pow2(t.numel() * t.element_size()) for pair in getattr(self, "host_phys", []) for t in pair),
                        staging_bytes=self.stage.numel(), staging_reserved_pow2=pow2(self.stage.numel()), plan_list_bytes=self.plan_h.numel() * 4,
                        contig_src_bytes=self.contig_src.numel(), note="torch 2.6 rounds each pinned block to a power of two "
                        "(ATen/core/CachingHostAllocator.h:132); reserved_pow2 is that virtual size, *_bytes the logical size"),
            meminfo=CC.meminfo(), node_meminfo=CC.node_meminfo())
        try:
            vmas = NM.parse()
            inv["numa"] = dict(staging=NM.range_pages(vmas, self.stage.data_ptr(), self.stage.numel()),
                               plan_list=NM.range_pages(vmas, self.plan_h.data_ptr(), self.plan_h.numel() * 4),
                               contig_src=NM.range_pages(vmas, self.contig_src.data_ptr(), self.contig_src.numel()))
            if e0 is not None:
                per = []
                for l, (k, v) in enumerate(self.host_phys):
                    kk = NM.range_pages(vmas, k.data_ptr(), k.numel() * 2)
                    vv = NM.range_pages(vmas, v.data_ptr(), v.numel() * 2)
                    per.append(dict(layer=l, k_node=NM.node_of(kk), v_node=NM.node_of(vv), k=(kk or {}).get("pages_per_node"), v=(vv or {}).get("pages_per_node")))
                inv["numa"]["host_cache"] = per
                inv["numa"]["k_nodes"] = "".join(p["k_node"] for p in per)
                inv["numa"]["v_nodes"] = "".join(p["v_node"] for p in per)
        except OSError as e:
            inv["numa"] = dict(error=str(e))
        return inv

    def payload(self, partial):
        return dict(meta=getattr(self, "run_meta", {}), tag=TAG, batch=self.B, label=K.LABEL, replay_label=K.REPLAY_LABEL,
                    ceiling_label=K.CEILING_LABEL, partial=partial, fails=self.fails, layout_fails=self.layout_fails, memgate=self.memgate,
                    config=dict(stages=STAGES, steps=STEPS, reps=REPS, cores=CORES, chunk_kb=CHUNK_KB, chunk_kb_alt=CHUNK_KB_ALT, ring=RING,
                                sleep_ms=SLEEP_MS, w8=W8, layout_plan_steps=LAYOUT_PLAN_STEPS, layout_reps=LAYOUT_REPS, layout_budget_s=LAYOUT_BUDGET_S,
                                t_arms=T_ARMS, t_steps=T_STEPS, t_reps=T_REPS, t_untraced=T_UNTRACED, lifetime_cap=LIFETIME_CAP,
                                lifetime_delay_ms=LIFETIME_DELAY_MS, peak_limit_gb=PEAK_LIMIT_GB, chunk_fields=CHUNK_FIELDS,
                                layer_fields=("hk", "desc", "list_wait_ms", "desc_ms", "desc_coord_cpu_ms", "n", "chunks"), field_notes=FIELD_NOTES,
                                stage_deadline=STAGE_DEADLINE, final_reserve_s=FINAL_RESERVE_S, calib_json=CALIB_JSON or None, conv_safety=CONV_SAFETY,
                                env={k: v for k, v in os.environ.items() if k.startswith(("CP_", "NOSI_", "OMP_", "CUDA_", "TORCH_", "TRITON_"))}),
                    provenance=dict(nosi_commit=os.environ.get("NOSI_COMMIT"), repo_commit=os.environ.get("REPO_COMMIT"), torch=torch.__version__,
                                    host=os.uname().nodename, pid=os.getpid()),
                    cpu=dict(placement=EARLY, model=CC.cpu_model(), smt_active=CC.smt_active(), configs=getattr(getattr(self, "coord", None), "configs", None)),
                    capture=getattr(self, "capture_rec", None), hygiene=getattr(self, "hygiene", None), correct=self.correct,
                    main_done=self.main_done, sweep_notes=self.sweep_notes, crash=getattr(self, "crash", None),
                    rows=self.rows, sweep=self.sweep, layout=dict(self.layout, rows=self.layout_rows), timeline=self.timeline,
                    inventory=self.inv_snaps, thread_use=self.thread_use, seconds=time.time() - self.t_start, docs=self.docs,
                    distinct_books=self.distinct,
                    confirm=dict(enabled=CONFIRM, mode=MODE, golden_mode=self.golden_mode, schedule=self.schedule,
                                 gate=self.gate_log, gate_fail_steps=self.gate_fail_steps, golden_fail=self.golden_fail,
                                 golden_steps=sorted(self.golden.get("steps", {})), golden_source=self.golden.get("source"),
                                 restarts=self.restarts, warm=self.warm_log, plans_vs_saved=self.plans_vs_saved,
                                 golden_cross_process=self.golden_cross, guard=dict(role=self.guard.role, decodes=self.guard.decodes,
                                                                                   log=self.guard.log[-200:]),
                                 arm_orders=self.arm_orders, arm_order=ARM_ORDER, layout_placers=LAYOUT_PLACERS,
                                 saved_plans=SAVED_PLANS_NPZ or None, golden_json=GOLDEN_JSON or None),
                    gated_labels=([r.get("label") for r in self.timeline if row_keep(r)[0]] if (self.timeline and self.golden_mode) else None),
                    **self.payload_extra)

    def flush_payload(self, partial):
        self.flush_fn(self.payload(partial))

    # ---------------------------------------------------------------------------------------------- run
    def alloc_measure(self):
        """The plan work rows, the registered memory gate (trigger (b)), the scratch; peak statistics restart here."""
        self.work = torch.zeros((self.NL, self.W), dtype=torch.int32, device="cuda")
        need = mem_need_gb(self.B, self.S_dst, self.H, self.D, RING, self.slot, CONTIG_BYTES)
        free = torch.cuda.mem_get_info()[0] / 1e9
        self.payload_extra["memory_gate"] = dict(free_gb=free, need_gb=need, margin_gb=MEM_MARGIN_GB)
        if free < need + MEM_MARGIN_GB:
            raise MemGate("free HBM %.2f GB < need %.2f + margin %.2f GB" % (free, need, MEM_MARGIN_GB))
        self.set_scratch("orig")
        torch.cuda.reset_peak_memory_stats()
        self.snap_inventory("after_setup")

    @torch.inference_mode()
    def run(self):
        from nosi.verify import miss_control as mc
        self.mc = mc
        self.setup_early()
        self.setup_model()
        if "CAPTURE" in STAGES and not PLANS_NPZ:
            try:
                self.capture_rec = self.capture()
            except _SnapshotOOM as e:
                self.capture_rec = dict(snapshot_oom=str(e))
                self.log("PostPrefillSnapshot OOM -> the two-process fallback (exit %d)" % RC_HYGIENE)
                self.flush_payload(False)
                return RC_HYGIENE
            if not self.capture_rec["accepted"]:
                self.log("capture NOT accepted: %s" % json.dumps({k: self.capture_rec[k] for k in ("invariants", "cross_check_fails", "trace_ok")}))
                self.fails += 1
                self.flush_payload(False)
                return RC_CORRECT
            if STAGES == ("CAPTURE",):
                self.flush_payload(False)
                return 0
        elif PLANS_NPZ:
            self.capture_rec = self.load_plans(PLANS_NPZ)
            self.model.has_buffers = False
        else:
            raise SystemExit("no plans: CAPTURE not in CP_STAGES and no CP_PLANS_NPZ")
        self.alloc_measure()
        # replay: natural warm steps reproduce the capture (the hygiene gate)
        pos = self.pos0.clone()
        hyg = []
        for it in range(VA.WARM):
            lg = self.decode(self.forced[:, it:it + 1], pos, warmup=(it == 0))
            torch.cuda.synchronize()
            ok_l = VA.sha(lg) == self.logits_sha[it]
            ok_m = all(bool(torch.equal(e._load_mask, self.store[it, l, 1 + self.HB:].view(self.H, self.B, self.M).to(torch.int64)))
                       for l, e in enumerate(self.engines))
            hyg.append(dict(step=it, logits_equal=bool(ok_l), masks_equal=bool(ok_m)))
            pos = pos + 1
        self.hygiene = dict(steps=hyg, ok=all(h["logits_equal"] and h["masks_equal"] for h in hyg))
        if not self.hygiene["ok"]:
            self.log("HYGIENE GATE FAILED: %s" % hyg)
            self.fails += 1
            self.flush_payload(False)
            return RC_HYGIENE
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        return self.measured(pos)

    def plan_schedule(self, stages=None):
        """(order, layout_steps): the gated steps of CORRECT / MAIN / TIMELINE in order, then the LAYOUT steps (consecutive
        after the last one; one per (cell, plan step)). The golden pass runs EXACTLY these steps. A GOLDEN-only process
        (fallback (ii)) records the schedule of the stages it serves (CP_GOLDEN_FOR)."""
        stages = STAGES if stages is None else tuple(stages)
        main_steps = [s for s in STEPS if "MAIN" in stages]
        t_steps = [s for s in T_STEPS if "TIMELINE" in stages]
        first = min(main_steps + t_steps) if (main_steps or t_steps) else VA.WARM
        order = sorted(set(main_steps + t_steps + ([first] if "CORRECT" in stages else [])))
        lay = []
        if "LAYOUT" in stages:
            start = (max(order) + 1) if order else VA.WARM
            n = len(LAYOUT_CELLS_PRE + LAYOUT_CELLS_POST) * len(LAYOUT_PLAN_STEPS)
            lay = list(range(start, min(start + n, VA.N)))
        return order, lay

    def measured(self, pos):
        """The gated steps (CORRECT at the first, MAIN, TIMELINE), SWEEP, the main-done marker, LAYOUT. With a schedule
        (the confirmation) LAYOUT takes exactly the scheduled steps the golden pass recorded."""
        if EARLY and EARLY.get("launch") is not None:
            CC.set_mask([EARLY["launch"]])                               # the launch thread alone on its core from here on
        max_step = self.masks.shape[0] - 1
        main_steps = [s for s in STEPS if "MAIN" in STAGES]
        t_steps = [s for s in T_STEPS if "TIMELINE" in STAGES]
        arms = main_arms()
        correct_done = "CORRECT" not in STAGES
        order, lay_sched = self.schedule if self.schedule is not None else (self.plan_schedule()[0], None)
        cur = VA.WARM

        def advance_to(target, pos):
            nonlocal cur
            while cur < target:                                          # natural steps between gated steps
                self.decode(self.forced[:, cur:cur + 1], pos)
                self.sync()
                pos = pos + 1
                cur += 1
            return pos
        for it in order:
            if it > max_step or it >= VA.N:
                raise RuntimeError("step %d beyond the captured / no-rollover budget" % it)
            pos = advance_to(it, pos)

            def work(ctx, it=it):
                nonlocal correct_done
                if not correct_done:
                    nf, res = self.correct_stage(ctx)
                    self.correct = res
                    correct_done = True
                    if nf:
                        self.fails += nf
                        raise _CorrectFail()
                if it in main_steps:
                    self.fails += self.main_step(ctx, arms)
                if it in t_steps:
                    self.fails += self.timeline_step(ctx)
                pk = torch.cuda.max_memory_reserved() / 1e9
                if pk > PEAK_LIMIT_GB:
                    self.memgate.append(dict(step=it, peak_reserved_gb=pk, trigger="(c) peak reserved above %.1f GB" % PEAK_LIMIT_GB))
            try:
                self.gated(it, pos, work)
            except _CorrectFail:
                self.log("CORRECTNESS FAILED (%d): MAIN / LAYOUT / TIMELINE skipped" % self.fails)
                self.flush_payload(False)
                return RC_CORRECT
            pos = pos + 1
            cur = it + 1
            self.log("step %d done (%.0fs, fails %d)" % (it, time.time() - self.t_start, self.fails))
            self.flush_payload(True)
            if self.memgate:                                            # registered (c): the step's results are kept, then exit
                self.flush_payload(False)
                raise MemGate("; ".join(m["trigger"] for m in self.memgate))
        if "MAIN" in STAGES and SWEEP:
            self.fails += self.sweep_stage()
            pk = torch.cuda.max_memory_reserved() / 1e9
            if pk > PEAK_LIMIT_GB:
                self.memgate.append(dict(step="SWEEP", peak_reserved_gb=pk, trigger="(c) peak reserved above %.1f GB (SWEEP)" % PEAK_LIMIT_GB))
            self.flush_payload(True)
        self.snap_inventory("after_main")
        if self.memgate:
            self.flush_payload(False)
            raise MemGate("; ".join(m["trigger"] for m in self.memgate))
        if "MAIN" in STAGES:
            self.flush_payload(True)                                    # the original arms are on disk before the add-on
            self.mark_main_done()
            self.log("MAIN done: marker written; nothing after this point can trigger a fallback")
        if "LAYOUT" in STAGES:
            free_steps = iter(range(cur, VA.N) if lay_sched is None else [s for s in lay_sched if s >= cur])
            pos_box = dict(pos=pos, cur=cur)

            def pos_of(target):
                nonlocal cur
                p = advance_to(target, pos_box["pos"])
                cur = target + 1                                        # the gated step consumes `target`
                pos_box["pos"] = p + 1
                return p
            self.run_layout(free_steps, pos_of)
        self.flush_payload(False)
        return min(self.fails + self.layout_fails, 19)

    # ---------------------------------------------------------------------------------------------- the confirmation
    def golden_meta(self):
        return dict(batch=self.B, L=VA.L, warm=VA.WARM, schedule=self.schedule, tag=TAG, mode=MODE, stages=STAGES,
                    nosi_commit=os.environ.get("NOSI_COMMIT"), ids_sha=VA.sha(self.ids.to(torch.float32)),
                    families_pre=G.FAMILIES_PRE, families_post=G.FAMILIES_POST, families_start=G.FAMILIES_START)

    def warm_pass(self, label, ref=None):
        """The natural warm steps 0..WARM-1 from a (re)started post-prefill state. Checks: load masks and logits (VA.sha)
        against the plans (when there are plans), and logits sha256 + the FULL state digest (window included) against
        `ref` (this process's first pass, or the golden export). Returns (pos, records, ok); ok False = the restart proof
        failed (the caller raises _ProofFail, exit 20)."""
        pos = self.pos0.clone()
        recs, ok = [], True
        store = getattr(self, "store", None)
        for it in range(VA.WARM):
            lg = self.decode(self.forced[:, it:it + 1], pos, warmup=(it == 0))
            self.sync()
            r = dict(step=it, logits_sha=VA.sha(lg), logits_sha256=G.sha_parts([lg]), digest=self.state_digest(G.FAMILIES_PRE))
            if store is not None:
                r["masks_equal_plans"] = all(bool(torch.equal(e._load_mask, store[it, l, 1 + self.HB:].view(self.H, self.B, self.M).to(torch.int64)))
                                             for l, e in enumerate(self.engines))
                r["logits_equal_plans"] = r["logits_sha"] == self.logits_sha[it]
            if ref is not None:
                rr = next((x for x in ref if int(x["step"]) == it), None)
                r["mismatch"] = ["<no reference>"] if rr is None else \
                    (([] if rr.get("logits_sha256") == r["logits_sha256"] else ["logits"]) + G.compare(rr.get("digest"), r["digest"]))
            r["ok"] = bool(r.get("masks_equal_plans", True) and r.get("logits_equal_plans", True) and not r.get("mismatch"))
            ok &= r["ok"]
            recs.append(r)
            pos = pos + 1
        self.warm_log.append(dict(label=label, ok=bool(ok), reference=("given" if ref is not None else None),
                                  steps=[{k: v for k, v in x.items() if k != "digest"} for x in recs]))
        self.hygiene = dict(steps=[dict(step=x["step"], logits_equal=x.get("logits_equal_plans"), masks_equal=x.get("masks_equal_plans"),
                                        mismatch=x.get("mismatch")) for x in recs], ok=bool(ok))
        return pos, recs, bool(ok)

    def gated_sequence(self, pos, work=None):
        """The scheduled gated steps (natural steps in between), each through gated(); work=None = the golden's no-op."""
        order, lay = self.schedule
        cur = VA.WARM
        for it in list(order) + list(lay):
            if it >= VA.N:
                raise RuntimeError("step %d beyond the no-rollover budget" % it)
            while cur < it:
                self.decode(self.forced[:, cur:cur + 1], pos)
                self.sync()
                pos = pos + 1
                cur += 1
            self.gated(it, pos, work or (lambda ctx: None))
            pos = pos + 1
            cur = it + 1
        return pos

    def restart(self, pps, label):
        """Back to the post-prefill state (the hosted PostPrefillSnapshot) and PROVE it: the start digest must equal the
        one taken right after prefill (cpupack_golden 'RESTART PROOF'); the model re-warms at the next decode."""
        self.snap = None                                                 # the CounterSnapshot (~6.4 GB at B336) goes first
        gc.collect()
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        t = time.time()
        G.snapshot_restore_hosted(pps)
        bad = G.compare(self.d0, self.start_digest())
        self.model.has_buffers = False
        rec = dict(label=label, ok=not bad, mismatched=bad[:64], seconds=time.time() - t)
        self.restarts.append(rec)
        self.log("restart '%s': start digest %s (%.1fs)" % (label, "IDENTICAL" if not bad else "DIFFERS %s" % bad[:6], rec["seconds"]))
        if bad:
            raise _ProofFail("restart '%s': the post-prefill restore is not bit-identical (%s)" % (label, bad[:8]))
        return rec

    @staticmethod
    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(8 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    def verify_saved(self, path, fresh=True):
        """The SAVED 2179683 plans against this process's fresh capture: file sha256 (CP_SAVED_PLANS_SHA256), the
        per-step plan digests (masks + maps of every layer; the run digest also hashes the meta, whose commit differs),
        the captured logits (VA.sha), the per-step load counts and the input ids. Any mismatch is a failure."""
        r = dict(path=path, file_sha256=self.file_sha256(path), expected_sha256=SAVED_PLANS_SHA256 or None, fresh_capture=bool(fresh))
        r["file_sha_ok"] = (not SAVED_PLANS_SHA256) or r["file_sha256"] == SAVED_PLANS_SHA256
        z = PL.load_npz(path)                                            # re-verifies the sidecar's run digest
        r["saved_meta"] = {k: z["meta"].get(k) for k in ("batch", "L", "ncap", "ids_sha", "nosi_commit")}
        r["ids_sha_fresh"] = VA.sha(self.ids.to(torch.float32))
        r["ids_equal"] = r["ids_sha_fresh"] == z["meta"].get("ids_sha")
        ok = r["file_sha_ok"] and r["ids_equal"]
        if fresh:
            mine = PL.plan_hashes(self.masks, self.maps)["step_digest"]
            saved = z["hashes"]["step_digest"]
            r["steps_fresh"], r["steps_saved"] = len(mine), len(saved)
            r["step_digest_mismatch"] = [i for i, (a, b) in enumerate(zip(mine, saved)) if a != b][:16]
            r["logits_sha_equal"] = [str(x) for x in self.logits_sha] == [str(x) for x in z["logits_sha"]]
            r["step_loaded_equal"] = [int(x) for x in self.step_loaded] == [int(x) for x in z["step_loaded"]]
            ok = ok and len(mine) == len(saved) and not r["step_digest_mismatch"] and r["logits_sha_equal"] and r["step_loaded_equal"]
        r["ok"] = bool(ok)
        return r

    @torch.inference_mode()
    def run_confirm(self):
        """The confirmation process (module docstring 'CONFIRMATION'; cpupack_golden.py): plans -> GOLDEN -> MEASURED."""
        if getattr(self, "mc", None) is None:                         # the CPU tests inject the file-loaded module
            from nosi.verify import miss_control as mc
            self.mc = mc
        self.setup_early()
        self.setup_model()
        self.schedule = self.plan_schedule(GOLDEN_FOR if STAGES == ("GOLDEN",) else STAGES)
        use_snap = not NO_SNAPSHOT and not GOLDEN_JSON
        pps, ref_warm = None, None
        if use_snap:
            try:
                pps = self.ss.PostPrefillSnapshot(self.cache).take()
            except torch.cuda.OutOfMemoryError as e:
                self.restarts.append(dict(label="take", ok=False, error=str(e)[:300]))
                self.log("PostPrefillSnapshot OOM -> fallback (ii) (exit %d)" % RC_HYGIENE)
                self.flush_payload(False)
                return RC_HYGIENE
            self.d0 = self.start_digest()
            self.payload_extra["restart_snapshot"] = dict(hosted_bytes=G.snapshot_offload(pps), start_digest_families=G.FAMILIES_START)
            gc.collect()
            torch.cuda.empty_cache()
        # 1. the plans
        if "CAPTURE" in STAGES:
            if pps is None:
                raise SystemExit("CAPTURE in a confirmation process needs the restart snapshot (CP_NO_SNAPSHOT=0, no CP_GOLDEN_JSON)")
            cap = self.capture(snapshot=False, digest_steps=range(VA.WARM))
            ref_warm = cap["warm"]
            self.capture_rec = dict(cap, warm=[{k: v for k, v in x.items() if k != "digest"} for x in cap["warm"]])
            if not cap["accepted"]:
                self.log("capture NOT accepted: %s" % json.dumps({k: cap[k] for k in ("invariants", "cross_check_fails", "trace_ok")}))
                self.fails += 1
                self.flush_payload(False)
                return RC_CORRECT
            if SAVED_PLANS_NPZ:
                self.plans_vs_saved = self.verify_saved(SAVED_PLANS_NPZ, fresh=True)
                self.log("fresh capture vs SAVED plans %s: %s" % (SAVED_PLANS_NPZ, "IDENTICAL" if self.plans_vs_saved["ok"] else
                                                                  json.dumps(self.plans_vs_saved)[:600]))
                if not self.plans_vs_saved["ok"]:
                    self.fails += 1
                    self.flush_payload(False)
                    return RC_CORRECT
                self.capture_rec["plans_used"] = self.load_plans(SAVED_PLANS_NPZ)
            self.restart(pps, "after capture")
        else:
            path = SAVED_PLANS_NPZ or PLANS_NPZ
            if not path:
                raise SystemExit("no plans: CAPTURE not in CP_STAGES and neither CP_SAVED_PLANS_NPZ nor CP_PLANS_NPZ")
            if SAVED_PLANS_NPZ:
                self.plans_vs_saved = self.verify_saved(SAVED_PLANS_NPZ, fresh=False)
                if not self.plans_vs_saved["ok"]:
                    self.log("SAVED plans refused: %s" % json.dumps(self.plans_vs_saved)[:600])
                    self.fails += 1
                    self.flush_payload(False)
                    return RC_CORRECT
            self.capture_rec = self.load_plans(path)
        self.flush_payload(True)
        # 2. the golden trajectory
        if GOLDEN_JSON:
            self.golden = G.golden_load(GOLDEN_JSON)
            self.golden["source"] = "cross-process export %s (fallback (ii))" % GOLDEN_JSON
            ref_warm = self.golden.get("warm")
        else:
            self.golden = dict(source="in-process golden pass (option (i))", meta=self.golden_meta(), warm=None, steps={})
            self.golden_mode = "record"
            pos, warm, ok = self.warm_pass("golden", ref=ref_warm)
            if not ok:
                raise _ProofFail("the golden pass's warm steps differ from the %s" % ("capture" if ref_warm is not None else "plans"))
            self.golden["warm"] = warm
            ref_warm = ref_warm if ref_warm is not None else warm
            self.snap = self.ss.CounterSnapshot(self.cache)
            self.trans = self.ss.transient_ids(self.model)
            self.gated_sequence(pos)
            gpath = os.path.join(OUT, "%s_golden.json" % TAG)
            G.golden_export(gpath, self.golden)
            self.payload_extra["golden_export"] = gpath
            self.log("golden pass: %d gated steps recorded -> %s" % (len(self.golden["steps"]), gpath))
            if self.golden_fail:
                self.log("the golden pass's own resident check FAILED at steps %s" % self.golden_fail)
                self.fails += len(self.golden_fail)
                self.flush_payload(False)
                return RC_CORRECT
            if STAGES == ("GOLDEN",):
                self.flush_payload(False)
                return 0
            if pps is None:
                raise SystemExit("a measured pass after an in-process golden pass needs the restart snapshot")
            self.restart(pps, "after golden")
        if pps is not None:
            del pps
            gc.collect()
            torch.cuda.empty_cache()
        if XGOLDEN_JSON:
            try:
                self.golden_cross = dict(G.golden_cross(self.golden, G.golden_load(XGOLDEN_JSON)), other=XGOLDEN_JSON)
            except Exception as e:                                       # a determinism record, never a gate
                self.golden_cross = dict(error=repr(e)[:300], other=XGOLDEN_JSON)
        # 3. the measured pass
        self.golden_mode = "check"
        self.alloc_measure()
        pos, _, ok = self.warm_pass("measured", ref=ref_warm)
        if not ok:
            raise _ProofFail("the measured pass's warm steps differ from the reference pass")
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        return self.measured(pos)

    # ------------------------------------------------------------------------------------------- the CONTROLS process
    def control_specs(self):
        """The destructive controls (each in its own pass from a proven restart). expect = the required verdict."""
        NL, s0 = self.NL, CONTROL_STEPS[0]
        out = [dict(name="clean", kind="clean", expect="PASS", step=s0),
               dict(name="logits_without_restore", kind="defect2179683", expect="DETECTED", step=s0)]
        for name, fam, l in (("window_row", "win", 7), ("tail_row", "tail", 11), ("block_map", "map", 3),
                             ("compression_state", "comp", 20), ("counters", "cnt", 9)):
            out.append(dict(name=name, kind="perturb", family=fam, layer=l % NL, expect="DETECTED", step=s0))
        out.append(dict(name="guard_unsanctioned_decode", kind="guard", expect="RAISED_AND_CLEAN", step=s0))
        return out

    def perturb(self, what, l):
        """One destructive write to LIVE engine state (CONTROLS only; called inside guard.destructive)."""
        e, lay = self.engines[l], self.cache.layers[l]
        t0, tl = int(e._tail_block_idx_on_gpu) * self.R, int(e._tail_block_len_on_gpu)
        b, h = 1 % self.B, 1 % self.H
        if what == "window_row":
            e._k_gpu.view(torch.int16)[0, 5, 0, :8].bitwise_xor_(0x0101)
            return "layer %d _k_gpu[0, 5, 0, :8] ^= 0x0101 (non-tail slot 0, row 5)" % l
        if what == "tail_row":
            e._v_gpu.view(torch.int16)[b, t0 + tl - 1, h, :4].bitwise_xor_(0x0101)
            return "layer %d _v_gpu[%d, %d, %d, :4] ^= 0x0101 (the valid tail row just written)" % (l, b, t0 + tl - 1, h)
        if what == "block_map":
            m = e._block_map
            x = m[h, b, 0].clone()
            m[h, b, 0] = m[h, b, 1]
            m[h, b, 1] = x
            return "layer %d _block_map[%d, %d, 0] <-> [.., 1]" % (l, h, b)
        if what == "compression_state":
            lay.no_compress_k_cache.view(torch.int16)[0, 0, 0, :4].bitwise_xor_(0x0101)
            return "layer %d no_compress_k_cache[0, 0, 0, :4] ^= 0x0101" % l
        if what == "counters":
            e._cache_lens[0] += 1
            return "layer %d _cache_lens[0] += 1" % l
        raise ValueError(what)

    def control_pass(self, spec, warm):
        """One control from a proven restart: warm steps (== the golden pass's), then the scheduled gated steps in CHECK
        mode with the control injected at spec['step']; the gate log decides the verdict."""
        name, s0 = spec["name"], spec["step"]
        self.golden_mode = "check"
        pos, _, ok = self.warm_pass(name, ref=warm)
        if not ok:
            raise _ProofFail("CONTROLS pass %s: the warm steps differ from the golden pass" % name)
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        extra, work, role = {}, None, self.guard.role
        if spec["kind"] == "perturb":
            def after_ref(ctx):
                if ctx["it"] == s0:
                    with self.guard.destructive(name):
                        extra["perturbed"] = self.perturb(name, spec["layer"])
            self.inject = dict(after_ref=after_ref)
        elif spec["kind"] == "defect2179683":
            def work(ctx):
                if ctx["it"] != s0:
                    return
                ctx["restore"]()
                lg1 = ctx["step_fn"]()                                   # the legal resident step (2179683's lg1)
                self.sync()
                extra["resident_before"] = dict(loads=self.loaded_now(), logits_equal=bool(torch.equal(lg1, ctx["lg_ref"])))
                with self.guard.destructive(name):
                    lg2 = self.decode(ctx["tok"], ctx["pos"])          # 2179683 :961: a second decode WITHOUT a restore
                self.sync()
                extra["original_control"] = "DETECTED" if not torch.equal(lg2, ctx["lg_ref"]) else "NOT_DETECTED"
                ctx["restore"]()                                         # 2179683 :964: the CounterSnapshot restore only
                lg3 = ctx["step_fn"]()                                   # what 2179683's later step-4 rows saw
                self.sync()
                extra["after_control"] = dict(loads=self.loaded_now(), logits_equal=bool(torch.equal(lg3, ctx["lg_ref"])))
        elif spec["kind"] == "guard":
            self.guard.role = "measure"                                  # exactly a measurement process's guard

            def work(ctx):
                if ctx["it"] != s0:
                    return
                ctx["restore"]()
                ctx["step_fn"]()
                self.sync()
                n0 = self.guard.decodes
                try:
                    ctx["step_fn"]()                                     # unsanctioned: no restore since the last decode
                    extra["raised"] = False
                except G.UnsanctionedDecode as e:
                    extra["raised"] = True
                    extra["message"] = str(e)[:300]
                extra["decode_ran"] = self.guard.decodes != n0
                try:
                    with self.guard.destructive("probe"):
                        pass
                    extra["destructive_refused"] = False
                except G.UnsanctionedDecode:
                    extra["destructive_refused"] = True
        start = len(self.gate_log)
        try:
            self.gated_sequence(pos, work)
        finally:
            self.inject = None
            self.guard.role = role
        log = self.gate_log[start:]
        compact = [dict(step=g["step"], ok=g["ok"], work_ran=g["work_ran"], pre_ok=g["pre"]["ok"], pre_why=g["pre"].get("why"),
                        post_ok=g["post"]["ok"], post_why=g["post"].get("why")) for g in log]
        return dict(name=name, step=s0, expect=spec["expect"], family=spec.get("family"), layer=spec.get("layer"),
                    verdict=self.control_verdict(spec, compact, extra), extra=extra, gate=compact)

    def control_verdict(self, spec, log, extra):
        by = {g["step"]: g for g in log}
        s0 = spec["step"]
        steps = list(self.schedule[0]) + list(self.schedule[1])
        if spec["kind"] == "clean":
            return "PASS" if len(log) == len(steps) and all(g["ok"] for g in log) else "FAIL"
        if spec["kind"] == "perturb":
            g = by.get(s0)
            if g is None:
                return "UNTESTABLE"
            key = "%s[%d]" % (spec["family"], spec["layer"])
            return "DETECTED" if (not g["pre_ok"] and not g["work_ran"] and key in (g["pre_why"] or [])) else "NOT_DETECTED"
        if spec["kind"] == "defect2179683":
            if extra.get("original_control") != "DETECTED":
                return "NOT_DETECTED(original control)"
            ac = extra.get("after_control") or {}
            if ac.get("loads", 0) == 0 and ac.get("logits_equal", True):
                return "NO_CORRUPTION"                                   # nothing to detect: the control is vacuous
            nxt = [s for s in steps if s > s0]
            g0, g1 = by.get(s0), (by.get(nxt[0]) if nxt else None)
            caught = (g0 is not None and not g0["post_ok"]) or (g1 is not None and not g1["pre_ok"])
            return "DETECTED" if caught else "NOT_DETECTED"
        if spec["kind"] == "guard":
            good = extra.get("raised") and not extra.get("decode_ran") and extra.get("destructive_refused") and \
                len(log) == len(steps) and all(g["ok"] for g in log)
            return "RAISED_AND_CLEAN" if good else "FAIL"
        return "UNKNOWN"

    @torch.inference_mode()
    def run_controls(self):
        """The CONTROLS process (run FIRST by the job; any failure stops the job before measurement): its own setup at a
        small batch, a golden pass, then every destructive control of control_specs() in its own pass from a PROVEN restart.
        Nothing here is timed or reported as a measurement."""
        if getattr(self, "mc", None) is None:                         # the CPU tests inject the file-loaded module
            from nosi.verify import miss_control as mc
            self.mc = mc
        self.setup_min()
        self.setup_model()
        self.schedule = (list(CONTROL_STEPS), [])
        pps = self.ss.PostPrefillSnapshot(self.cache).take()
        self.d0 = self.start_digest()
        self.payload_extra["restart_snapshot"] = dict(hosted_bytes=G.snapshot_offload(pps), start_digest_families=G.FAMILIES_START)
        self.golden = dict(source="CONTROLS in-process golden pass", meta=self.golden_meta(), warm=None, steps={})
        self.golden_mode = "record"
        pos, warm, _ = self.warm_pass("golden")
        self.golden["warm"] = warm
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        self.gated_sequence(pos)
        gpath = os.path.join(OUT, "%s_golden.json" % TAG)
        G.golden_export(gpath, self.golden)
        self.payload_extra["golden_export"] = gpath
        nf = len(self.golden_fail)
        self.controls_results = []
        self.payload_extra["controls"] = self.controls_results          # partial results reach every flush
        for spec in self.control_specs():
            self.restart(pps, spec["name"])                              # a _ProofFail propagates: exit 20
            try:
                r = self.control_pass(spec, warm)
            except _ProofFail:
                raise
            except Exception as e:
                r = dict(name=spec["name"], expect=spec["expect"], verdict="ERROR", error="%s: %s" % (type(e).__name__, str(e)[:300]),
                         traceback=traceback.format_exc()[-3000:])
            r["pass"] = r.get("verdict") == spec["expect"]
            nf += int(not r["pass"])
            self.controls_results.append(r)
            self.log("CONTROL %-26s expect %-16s got %s" % (spec["name"], spec["expect"], r.get("verdict")))
            self.flush_payload(True)
        self.fails += nf
        self.flush_payload(False)
        return RC_CORRECT if nf else 0


class _CorrectFail(Exception):
    pass


class _ProofFail(Exception):
    """The restart proof failed (a restore is not bit-identical, or warm steps differ): exit 20 = fallback (ii)."""


class _SnapshotOOM(Exception):
    pass


def MODES_DMA(a) -> bool:
    return K.MODES[a["mode"]][1] if a.get("kind") == "cpu" else False


# ------------------------------------------------------------------------------------------------------------ calib
@torch.inference_mode()     # job 2179683 defect 2: the staging tensor is made under the coordinator's inference_mode, so
def calib():                # Pipe.reset()'s zero_ on it (cpupack_core.py:547) must run under inference mode too
    """Node-local CPU calibration (stage C0 of the job, no model): SYNTHETIC Poisson(CP_CALIB_MEAN) plans at the batch
    geometry, the coordinator on 1/2/4/8 team cores, the row packer from the original layout, then the host tensor reordered
    in place to head-major (the one-time conversion, timed: reorder and verification separately; PAGEABLE tensors, so
    LAYOUT's conversion predictor takes max(C0, its own in-place probe of the real pinned cache)) and the row and group
    packers from it. Pack only (mode LP), 8 MiB chunks, pinned staging. Label: 'synthetic plan, node-local pack
    calibration'. main() refuses (exit 24) before this runs when the placement has no full team on the GPU node."""
    B = int(os.environ.get("CP_B", "336"))
    S = VA.L + 8192
    mean = float(os.environ.get("CP_CALIB_MEAN", "3.7"))
    reps = int(os.environ.get("CP_CALIB_REPS", "5"))
    H, D, R, M = K.H_DEF, K.D_DEF, K.R_DEF, K.M_DEF
    g = torch.Generator().manual_seed(0)
    plan = torch.full((H, B, M), -1, dtype=torch.int32)
    cnt = torch.poisson(torch.full((H, B), mean), generator=g).clamp_(0, 63).to(torch.int64)
    for h in range(H):
        for b in range(B):
            k = int(cnt[h, b])
            if k:
                plan[h, b, torch.randperm(63, generator=g)[:k]] = torch.randperm(VA.L // R, generator=g)[:k].to(torch.int32)
    team = (EARLY or {}).get("team") or []
    co = Coordinator(team)
    co.start()
    for n in CORES:
        co.configure(n)
    cap = CHUNK_KB * 1024 // K.useful_bytes(1)
    pin = torch.cuda.is_available()                                     # the job: pinned staging; a CPU-only test node cannot pin
    stage = co.call(lambda c: (lambda t: t.pin_memory() if pin else t)(torch.zeros((RING, K.slot_bytes(cap)), dtype=torch.uint8)))
    land = torch.zeros_like(stage)
    phys_k = K.alloc_phys("orig", B, S, H, D, torch.bfloat16)
    phys_v = K.alloc_phys("orig", B, S, H, D, torch.bfloat16)
    phys_k.view(torch.int16).random_(-3000, 3000, generator=g)
    phys_v.view(torch.int16).random_(-3000, 3000, generator=g)
    dst = K.alloc_phys("orig", B, 4096, H, D, torch.bfloat16)
    res = dict(label="synthetic plan (Poisson %.2f per stream), node-local pack calibration; pack only (LP)" % mean, B=B, S=S,
               groups=int((plan[..., :63] >= 0).sum()), placement=EARLY, cpu_model=CC.cpu_model(), rows=[], conversion=None,
               pinned_staging=bool(pin and stage.is_pinned()), cores=list(CORES), reps_per_row=reps)

    def run(layout, packer, n):
        co.configure(n)
        pipe = K.Pipe(K.CpuBackend(), stage, land, cap)

        def f(c):
            ts = []
            for i in range(reps + 1):
                d0 = time.perf_counter_ns()
                d = K.build_desc(plan, src_layout=layout, dst_layout="orig", s_src=S, s_dst=4096, packer=packer, placer="row")
                d1 = time.perf_counter_ns()
                sk, dk = K.views_for(d, phys_k, dst)
                sv, dv = K.views_for(d, phys_v, dst)
                pipe.reset()
                t = time.perf_counter_ns()
                recs = pipe.layer(d, sk, sv, dk, dv, mode="LP")
                t2 = time.perf_counter_ns()
                if i:
                    ts.append(dict(desc_ms=(d1 - d0) / 1e6, pack_ms=(t2 - t) / 1e6, chunks=len(recs), bytes=K.useful_bytes(d.n)))
            return ts
        ts = co.call(f)
        med = float(np.median([x["pack_ms"] for x in ts]))
        res["rows"].append(dict(layout=layout, packer=packer, cores=n, reps=ts, pack_ms_median=med,
                                gbps_median=ts[0]["bytes"] / (med * 1e6), desc_ms_median=float(np.median([x["desc_ms"] for x in ts]))))
        print("[calib] %s %s cores=%d pack %.2f ms/layer (%.1f GB/s)" % (layout, packer, n, med, ts[0]["bytes"] / (med * 1e6)), flush=True)
    for n in CORES:
        run("orig", "row", n)
    conv = []
    for name, t in (("k", phys_k), ("v", phys_v)):
        new, info = K.convert_inplace(t, "orig", "hm", verify=True)
        conv.append(dict(tensor=name, **info))
        if name == "k":
            phys_k = new
        else:
            phys_v = new
    res["conversion"] = dict(per_tensor=conv, accounting=K.conversion_accounting(B, S, H, D, 2, 32, [x["ms"] for x in conv] * 32))
    for n in CORES:
        run("hm", "row", n)
        run("hm", "group", n)
    co.q.put(None)
    fn = os.path.join(OUT, "calib_%s.json" % TAG)
    json.dump(res, open(fn, "w"), indent=1)
    bad = sum(c["bad"] for c in conv)
    print("[calib] saved %s (conversion bad requests %d)" % (fn, bad), flush=True)
    return int(bad > 0)


# ------------------------------------------------------------------------------------------------------------ table
def request_rows(r, req_by_layer):
    """Per-request samples of one rep (cpupack_timing), from the compact record."""
    out = []
    if r["kind"] == "w8":
        for i, l in enumerate(r["layers"]):
            out += TM.request_samples_w8(l, r["pr"][i], r["g0"][i], r["g1"][i], req_by_layer[l])
        return out
    if r["kind"] != "cpu" or r.get("lite") or r.get("mode") != "full" or "layers_rec" not in r:
        return out
    for i, l in enumerate(r["layers"]):
        hk, desc, _, _, _, n, ch = r["layers_rec"][i]
        chunks = [dict(zip(CHUNK_FIELDS, c)) for c in ch]
        lay = dict(pr=r["pr"][i], list=r["list"][i], hk=hk, desc=desc, chunks=chunks)
        out += TM.request_samples_cpu(l, lay, req_by_layer[l])
    return out


def rep_view(r):
    """The cpupack_timing.rep_aggregate input of one compact record."""
    v = dict(kind=r["kind"], pr=r["pr"], t0=r.get("t0"), tm=r.get("tm"), with_decode=r.get("with_decode"))
    if r["kind"] in ("w8", "contig", "scatter"):
        v["kind"] = "w8"
        v["g1"] = r.get("g1")
    elif "layers_rec" in r:
        v["layers"] = [dict(chunks=[dict(zip(CHUNK_FIELDS, c)) for c in lay[6]]) for lay in r["layers_rec"]]
    return v


TABLE_CSV_FIELDS = ("stage", "cell", "arm", "step", "plan_step", "rep", "mode", "layer", "b", "groups", "chunk", "e2e") + \
    tuple(TM.CPU_STAGES) + tuple(TM.W8_STAGES)
PAYLOAD_RE = re.compile(r"^(cp|t|ctl)_b\d+\.json$")


def row_keep(r):
    """(keep, reason) of one row: a row enters a statistic only when its own checks passed (ok is True) AND its gated step
    passed the golden gate (gate_ok is not False). A row without gate_ok comes from a payload without the gate (job
    2179683, or a row outside any gated step): kept on ok alone, counted 'ungated'."""
    if r.get("ok") is not True:
        return False, "not_ok"
    if r.get("gate_ok") is False:
        return False, "gate_fail"
    return True, ("ungated" if "gate_ok" not in r else "gated")


class CsvSink:
    """The per-request CSV of one batch, streamed through gzip group by group (never held in memory)."""

    def __init__(self, path):
        self.path, self.n = path, 0
        self.f = gzip.open(path, "wt", compresslevel=1, newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(TABLE_CSV_FIELDS)

    def write(self, samples):
        for s in samples:
            self.w.writerow([s.get(k, "") for k in TABLE_CSV_FIELDS])
        self.n += len(samples)

    def close(self):
        self.f.close()


def exclusion_counts(rows, default_cell="orig>orig"):
    """{(stage, cell, arm, phase): dict(total, kept, not_ok, gate_fail, ungated)} and the kept rows; decode-alone phases
    pool as 'decode_alone'."""
    excl, kept = {}, []
    for r in rows:
        ph = r.get("phase") or ""
        key = (str(r.get("stage")), r.get("cell", default_cell), str(r.get("arm")), "decode_alone" if ph.startswith("decode_alone") else ph)
        c = excl.setdefault(key, dict(total=0, kept=0, not_ok=0, gate_fail=0, ungated=0))
        c["total"] += 1
        k, why = row_keep(r)
        if k:
            c["kept"] += 1
            c["ungated"] += int(why == "ungated")
            kept.append(r)
        else:
            c[why] += 1
    return excl, kept


def gate_summary(p):
    conf = p.get("confirm") or {}
    gate = conf.get("gate") or []
    restarts = conf.get("restarts") or []
    warm = conf.get("warm") or []
    pvs = conf.get("plans_vs_saved")
    xg = conf.get("golden_cross_process")
    pre_bad = [g["step"] for g in gate if not (g.get("pre") or {}).get("ok", True)]
    post_bad = [g["step"] for g in gate if not (g.get("post") or {}).get("ok", True)]
    return dict(enabled=bool(conf.get("enabled") or gate), golden_mode=conf.get("golden_mode"), source=conf.get("golden_source"),
                steps=len(gate), pre_fail=pre_bad, post_fail=post_bad, bad=sorted(set(pre_bad) | set(post_bad)),
                restarts_ok=sum(1 for r in restarts if r.get("ok")), restarts=len(restarts),
                warm_ok=sum(1 for w in warm if w.get("ok")), warm=len(warm),
                plans_vs_saved=(None if pvs is None else bool(pvs.get("ok"))), cross=(None if not xg else xg.get("equal")))


def table(out_dir, csv_requests=None):
    """Markdown + CSV summaries of every cp_b<B>.json under out_dir (medians / p95 over individual samples), the TIMELINE
    payloads under ../timeline and the CONTROLS payloads under ../controls (gate and exclusion summaries).
    ROW FILTER (row_keep): only ok rows of gate-passing steps enter ANY statistic -- decode-alone pools, groups, the drift
    line, SWEEP cells; every exclusion is reported with its denominator (the 'Exclusions' table and exclusions_b<B>.csv).
    Slowdown = median decode beside / median decode alone AT THE SAME decode step(s) - 1: MAIN and LAYOUT each record their
    own decode-alone reps at every gated step (a group without them prints nan, never a cross-step ratio). Request e2e
    p50 / p95 are given for the transfer-alone AND the beside-decode reps; the per-request samples are STREAMED per group
    into requests_b<B>.csv.gz (CP_TABLE_CSV=0 skips the file, not the statistics). Also: the registered low / mid / high
    tercile cells of the captured per-(step, layer) totals over steps 1.. (SWEEP request e2e by cell; step 0 = the full
    list) with the zero-miss stream denominator; the LAYOUT (orig, orig) drift against MAIN on the same plan steps; the
    conversion (reorder vs verification) and the tail-write accounting (small buffer + real cache)."""
    import glob
    csv_requests = (os.environ.get("CP_TABLE_CSV", "1") == "1") if csv_requests is None else csv_requests
    t_start = time.time()
    L = ["# CPU-packing transport (%s; %s)" % (K.LABEL, K.REPLAY_LABEL), "",
         "Rows enter a statistic only when ok AND their gated step passed the golden gate (row_keep); every exclusion is counted "
         "with its denominator.", ""]
    ok_all = True
    files = [fn for fn in sorted(glob.glob(os.path.join(out_dir, "cp_*.json"))) if PAYLOAD_RE.match(os.path.basename(fn))]
    for fn in files:
        with open(fn) as f:
            p = json.load(f)
        B = p.get("batch")
        lay = p.get("layout") or {}
        gs = gate_summary(p)
        ok = (p.get("fails", 1) == 0 and not p.get("partial") and not p.get("layout_fails") and not lay.get("partial")
              and not p.get("crash") and not gs["bad"])
        ok_all &= ok
        L.append("## B=%s (%s): fails %s, layout fails %s%s%s%s%s" % (
            B, os.path.basename(fn), p.get("fails"), p.get("layout_fails"), " PARTIAL" if p.get("partial") else "",
            " LAYOUT-PARTIAL" if lay.get("partial") else "", " CRASH(%s)" % p["crash"].get("kind") if p.get("crash") else "",
            "" if p.get("main_done") in (None, True) else " (MAIN not done)"))
        if gs["enabled"]:
            L.append("- golden gate (%s; %s): %d gated steps; GATE_FAIL before timing at %s, after the advance at %s; restarts "
                     "%d/%d bit-identical; warm passes %d/%d; fresh capture == saved 2179683 plans: %s; golden cross-process equal: %s" % (
                         gs["golden_mode"], gs["source"], gs["steps"], gs["pre_fail"] or "none", gs["post_fail"] or "none",
                         gs["restarts_ok"], gs["restarts"], gs["warm_ok"], gs["warm"], gs["plans_vs_saved"], gs["cross"]))
        for n in (p.get("sweep_notes") or []) + ["LAYOUT: " + x for x in lay.get("notes") or []]:
            L.append("- note: %s" % n)
        cap = p.get("capture") or {}
        masks = None
        npz = cap.get("export") or (cap.get("source", "").split(" ")[1] if cap.get("source", "").startswith("export") else None)
        used = (cap.get("plans_used") or {}).get("source", "")
        if used.startswith("export"):
            npz = used.split(" ")[1]
        if npz and os.path.exists(npz):
            masks = PL.load_npz(npz)["masks"]
        req = {}

        def req_of(ps):
            if ps not in req and masks is not None and ps is not None and 0 <= ps < masks.shape[0]:
                req[ps] = [(masks[ps, l][..., :63] >= 0).sum(dim=(0, 2)).tolist() for l in range(masks.shape[1])]
            return req.get(ps)
        excl, kept = exclusion_counts(p.get("rows", []) + lay.get("rows", []))
        groups, alone = {}, {}
        for r in kept:
            ph = r.get("phase", "")
            if ph.startswith("decode_alone"):
                alone.setdefault(r.get("step"), []).append(r["main_ms"])
                continue
            if ph == "warmup":
                continue
            groups.setdefault((r.get("stage"), r.get("cell", "orig>orig"), r["arm"]), []).append(r)
        L += ["", "| stage | cell | arm | kept reps alone/conc | useful GB/s alone p50 | useful GB/s conc p50 | e2e alone p50 / p95 ms | "
              "e2e conc p50 / p95 ms (requests) | list / wake / desc / pack / submit / dma / scq / scatter p50 ms (conc) | "
              "decode alone -> conc ms p50 (same steps; kept decode-alone n) | slowdown | late ms p50 | overlap p50 | excluded alone+conc (not ok / gate) of total |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        sink = CsvSink(os.path.join(out_dir, "requests_b%s.csv.gz" % B)) if csv_requests else None
        summ = {}
        try:
            for (stage, cellk, a), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
                al = [x for x in rs if not x.get("with_decode")]
                co = [x for x in rs if x.get("with_decode")]
                agg_a = [TM.rep_aggregate(rep_view(x), x["useful"]) for x in al]
                agg_c = [TM.rep_aggregate(rep_view(x), x["useful"]) for x in co]
                e2e_a, conc_vals = [], {k: [] for k in ("e2e",) + tuple(TM.CPU_STAGES)}
                for mode, xs in (("alone", al), ("conc", co)):
                    for x in xs:
                        rq = req_of(x.get("plan_step"))
                        if rq is None:
                            continue
                        smp = request_rows(x, rq)
                        for s_ in smp:
                            s_.update(stage=stage, cell=cellk, arm=a, step=x.get("step"), plan_step=x.get("plan_step"), rep=x.get("rep"), mode=mode)
                            if mode == "alone":
                                e2e_a.append(s_["e2e"])
                            else:
                                for k, v in conc_vals.items():
                                    if k in s_:
                                        v.append(s_[k])
                        if sink is not None:
                            sink.write(smp)
                sa = dict(e2e=dict(p50=TM.pctl(e2e_a, 50), p95=TM.pctl(e2e_a, 95)))
                sm = {k: dict(p50=TM.pctl(v, 50), p95=TM.pctl(v, 95)) for k, v in conc_vals.items()}
                dec_c = [x["main_ms"] for x in co]
                dec_a = [m for st in sorted({x.get("step") for x in co}, key=str) for m in alone.get(st, [])]
                da, dc = TM.pctl(dec_a, 50), TM.pctl(dec_c, 50)
                slow = 100 * (dc / da - 1) if da == da and da > 0 else float("nan")
                ga, gc_ = TM.pctl([g["useful_gbps"] for g in agg_a], 50), TM.pctl([g["useful_gbps"] for g in agg_c], 50)
                summ[(stage, cellk, a)] = dict(gbps_alone=ga, gbps_conc=gc_, slowdown_pct=slow, plan_steps=sorted({x.get("plan_step") for x in rs}, key=str))
                ex = [excl.get((str(stage), cellk, str(a), ph), {}) for ph in ("alone", "conc")]
                tot = sum(e.get("total", 0) for e in ex)
                L.append("| %s | %s | %s | %d/%d | %.2f | %.2f | %.2f / %.2f | %.2f / %.2f | %s | %.2f -> %.2f (n %d) | %+.1f%% | %.2f | %.2f | %d (%d / %d) of %d |" % (
                    stage, cellk, a, len(al), len(co), ga, gc_, sa["e2e"]["p50"], sa["e2e"]["p95"], sm["e2e"]["p50"], sm["e2e"]["p95"],
                    " / ".join("%.2f" % sm[k]["p50"] for k in TM.CPU_STAGES), da, dc, len(dec_a), slow, TM.pctl([g.get("late_ms") for g in agg_c], 50),
                    TM.pctl([g.get("overlap_frac") for g in agg_c], 50), tot - sum(e.get("kept", 0) for e in ex),
                    sum(e.get("not_ok", 0) for e in ex), sum(e.get("gate_fail", 0) for e in ex), tot))
        finally:
            if sink is not None:
                sink.close()
        if sink is not None:
            L.append("- per-request samples: %d rows streamed to %s" % (sink.n, os.path.basename(sink.path)))
        # N3: the (orig, orig) LAYOUT pair against MAIN on the same plan steps (the cell-order / time drift); kept rows only
        lsteps = set()
        for (stage, cellk, a), v in summ.items():
            if stage == "LAYOUT":
                lsteps |= set(v["plan_steps"])
        for main_arm, lay_arm in (("w8", "w8@orig>orig"), ("cpu8", "cpu8@orig>orig:row/row")):
            lo = summ.get(("LAYOUT", "orig>orig", lay_arm))
            if lo is None:
                continue
            ms_ = [x for x in groups.get(("MAIN", "orig>orig", main_arm), []) if x.get("plan_step") in lsteps and not x.get("with_decode")]
            gm = TM.pctl([TM.rep_aggregate(rep_view(x), x["useful"])["useful_gbps"] for x in ms_], 50)
            L.append("- drift LAYOUT %s vs MAIN %s on plan steps %s: useful GB/s alone %.2f / %.2f = %.3f (the cell-order confound; "
                     "hm cells run after the conversion)" % (lay_arm, main_arm, sorted(lsteps), lo["gbps_alone"], gm,
                                                               lo["gbps_alone"] / gm if gm == gm and gm > 0 else float("nan")))
        # the registered cells: terciles of the captured per-(step, layer) totals over steps 1.., the zero-miss denominator
        if masks is not None and masks.shape[0] > 1:
            steady = list(range(1, masks.shape[0]))
            tot_ = PL.layer_totals(masks, range(masks.shape[0]))
            cuts = PL.tercile_cells(tot_, steady)
            cnt = PL.counts_of(masks)[1:]
            zero, streams = int((cnt == 0).sum()), int(cnt.numel())
            L.append("")
            L.append("registered cells: per-(step, layer) load totals over steps 1..%d: low <= %.1f < mid <= %.1f < high (n %d, min %d, max %d); "
                     "zero-miss (layer, head, request) streams %d of %d (%.1f%%) = the separate denominator; step 0 = the full list" % (
                         steady[-1], cuts["lo"], cuts["hi"], cuts["n"], cuts["min"], cuts["max"], zero, streams, 100.0 * zero / max(streams, 1)))
            sw_excl, sw_kept = exclusion_counts(p.get("sweep", []))
            by = {}
            for r in sw_kept:
                ps = r.get("plan_step")
                rq = req_of(ps)
                if rq is None:
                    continue
                for s_ in request_rows(r, rq):
                    cn = "full" if ps == 0 else PL.cell_of(tot_[(ps, s_["layer"])], cuts)
                    by.setdefault((r["arm"], cn), []).append(s_["e2e"])
            excl.update({k: v for k, v in sw_excl.items()})
            if by:
                L += ["", "| SWEEP arm | cell | requests | e2e alone p50 / p95 ms |", "|---|---|---|---|"]
                order = {"low": 0, "mid": 1, "high": 2, "full": 3}
                for (a, cn), xs in sorted(by.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
                    L.append("| %s | %s | %d | %.2f / %.2f |" % (a, cn, len(xs), TM.pctl(xs, 50), TM.pctl(xs, 95)))
        # every exclusion with its denominator
        L += ["", "### Exclusions B=%s (rows: total / kept / excluded not ok / excluded GATE_FAIL / kept without a gate)" % B, "",
              "| stage / cell / arm | phase | total | kept | not ok | GATE_FAIL | kept ungated |", "|---|---|---|---|---|---|---|"]
        with open(os.path.join(out_dir, "exclusions_b%s.csv" % B), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("batch", "stage", "cell", "arm", "phase", "total", "kept", "not_ok", "gate_fail", "ungated"))
            for k in sorted(excl, key=lambda k: tuple(str(x) for x in k)):
                c = excl[k]
                w.writerow((B,) + k + (c["total"], c["kept"], c["not_ok"], c["gate_fail"], c["ungated"]))
                L.append("| %s / %s / %s | %s | %d | %d | %d | %d | %d |" % (k + (c["total"], c["kept"], c["not_ok"], c["gate_fail"], c["ungated"])))
        n_tot = sum(c["total"] for c in excl.values())
        n_kept = sum(c["kept"] for c in excl.values())
        L.append("- B=%s: %d of %d rows kept; %d excluded (not ok %d, GATE_FAIL %d)" % (
            B, n_kept, n_tot, n_tot - n_kept, sum(c["not_ok"] for c in excl.values()), sum(c["gate_fail"] for c in excl.values())))
        if lay.get("conversion"):
            cv = lay["conversion"]
            L.append("")
            L.append("layout_ablation conversion %s->%s (final %s): bad requests %s, reference logits equal before/after %s, reorder %.1f s + "
                     "verification %.1f s (in place, extra pinned 0; predicted %s s; aborted %s)" % (
                         cv["frm"], cv["to"], cv.get("final_layout"), cv["bad_requests"], cv.get("reference_logits_equal_before_after"),
                         (cv.get("ms_reorder_total") or 0) / 1000, (cv.get("ms_verify_total") or 0) / 1000,
                         "%.0f" % cv["prediction"]["predicted_s"] if cv.get("prediction") else "-", cv.get("aborted")))
        for k, v in (lay.get("tail_write") or {}).items():
            a = v["accounting"]
            L.append("layout_ablation tail write %s (small pinned buffer): %d runs x %d B per tensor; measured %.3f ms/tensor (host), %.3f ms "
                     "per rollover, %.4f ms per token amortized; link bound %.4f ms per token" % (
                         k, a["runs"], a["run_bytes"], float(np.median(v["host_ms"])), a["measured_ms_per_rollover"], a["measured_ms_per_token"],
                         a["link_bound_ms_per_token"]))
        for k, v in (lay.get("tail_write_real") or {}).items():
            if v.get("skipped"):
                L.append("layout_ablation tail write %s (real cache): skipped: %s" % (k, v["skipped"]))
            else:
                L.append("layout_ablation tail write %s (real cache, %d tensors): %.3f ms per rollover (sum of per-tensor medians), %.4f ms per "
                         "token amortized; content ok %s; host rows restored %s" % (k, v["tensors"], v["measured_ms_per_rollover"],
                                                                                v["measured_ms_per_token"], v["content_ok"], v.get("host_rows_restored")))
        L.append("")
    # the TIMELINE and CONTROLS payloads of the same job (gate + exclusions; the launch analysis is launch_timeline.py's)
    root = os.path.dirname(os.path.abspath(out_dir))
    for sub, pat in (("timeline", "t_*.json"), ("controls", "ctl_*.json")):
        for fn in sorted(glob.glob(os.path.join(root, sub, pat))):
            if not PAYLOAD_RE.match(os.path.basename(fn)):
                continue
            with open(fn) as f:
                p = json.load(f)
            gs = gate_summary(p)
            ctl = p.get("controls") or []
            if sub == "controls":                                        # its GATE_FAILs are the intended detections
                okp = p.get("fails", 1) == 0 and not p.get("crash") and bool(ctl) and all(c.get("pass") for c in ctl)
            else:
                okp = p.get("fails", 1) == 0 and not p.get("crash") and not gs["bad"]
            ok_all &= okp
            L.append("## %s %s (B=%s): fails %s%s -> %s" % (sub.upper(), os.path.basename(fn), p.get("batch"), p.get("fails"),
                                                            " CRASH(%s)" % p["crash"].get("kind") if p.get("crash") else "",
                                                            "PASS" if okp else "FAIL"))
            if sub == "controls":
                L.append("- restarts %d/%d bit-identical; warm passes %d/%d (each control runs from a proven restart)" % (
                    gs["restarts_ok"], gs["restarts"], gs["warm_ok"], gs["warm"]))
            elif gs["enabled"]:
                L.append("- golden gate (%s): %d gated steps; GATE_FAIL before timing at %s, after the advance at %s; restarts %d/%d "
                         "bit-identical; warm passes %d/%d; golden cross-process equal: %s" % (
                             gs["source"], gs["steps"], gs["pre_fail"] or "none", gs["post_fail"] or "none", gs["restarts_ok"],
                             gs["restarts"], gs["warm_ok"], gs["warm"], gs["cross"]))
            if sub == "controls":
                for c in ctl:
                    L.append("- control %s: expect %s, got %s -> %s" % (c.get("name"), c.get("expect"), c.get("verdict"),
                                                                      "PASS" if c.get("pass") else "FAIL"))
            else:
                ex, _ = exclusion_counts([dict(r, stage="TIMELINE", cell="-") for r in p.get("timeline", [])])
                for k in sorted(ex, key=lambda k: tuple(str(x) for x in k)):
                    c = ex[k]
                    L.append("- TIMELINE %s %s: kept %d of %d (not ok %d, GATE_FAIL %d)" % (k[2], k[3], c["kept"], c["total"], c["not_ok"], c["gate_fail"]))
            L.append("")
    L.append("table built in %.1f s" % (time.time() - t_start))
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "cpupack_table.md"), "w") as f:
        f.write(text)
    print(text)
    return 0 if ok_all else 1


# ------------------------------------------------------------------------------------------------------------ main
def is_memory_trigger(exc) -> bool:
    """Registered trigger (a): a MemGate, a CUDA OOM, or a RuntimeError from a CUDA / pinned allocation. A coordinator
    error ('coordinator: <traceback>') is never one: its text is a traceback and may contain any word."""
    if isinstance(exc, (MemGate, torch.cuda.OutOfMemoryError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if msg.startswith("coordinator:"):
            return False
        low = msg.lower()
        return "out of memory" in low or "cudahostalloc" in low or "pinned" in low
    return False


def exit_code_for(runner, exc) -> int:
    """The exit code of an exception that left Runner.run. Before the main-done marker: a placement refusal -> 24, a
    registered memory trigger -> 22 (the sbatch's B320 fallback), anything else -> 23 (a crash, never read as a failure
    count). After the marker the batch's MAIN results are final: a failure count 1..19, never 20..24."""
    if getattr(runner, "main_done", False):
        return max(1, min(int(runner.fails) + int(runner.layout_fails) + 1, 19))
    if isinstance(exc, _ProofFail):                                       # the restart proof (cpupack_golden): fallback (ii)
        return RC_HYGIENE
    if isinstance(exc, PlacementRefused):
        return RC_PLACEMENT
    if is_memory_trigger(exc):
        return RC_MEMGATE
    return RC_CRASH


def main():
    if MODE == "table":
        return table(OUT)
    os.makedirs(OUT, exist_ok=True)
    why = placement_problem(EARLY) if MODE != "controls" else None        # CONTROLS packs nothing: no team needed
    if why:                                                              # before the corpus / model load (minutes at B336)
        print("[cpupack] PLACEMENT REFUSED (exit %d): %s; placement %s" % (RC_PLACEMENT, why, json.dumps(EARLY)), flush=True)
        return RC_PLACEMENT
    if MODE == "calib":
        return calib()
    VA.check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    corpus = VA.load_corpus(path, VA.BATCH)
    model = VA.load_model(path)
    ids, docs, distinct = VA.pick_batch(corpus, VA.BATCH, 0)
    print("[cpupack] B=%d L=%d Ncap=%d stages=%s docs %s%s (%d distinct) placement %s" % (VA.BATCH, VA.L, VA.N, STAGES, docs[:6], "..." if len(docs) > 6 else "",
                                                                                   distinct, json.dumps(EARLY)), flush=True)
    fn = os.path.join(OUT, "%s.json" % TAG)

    def flush(p):
        tmp = fn + ".tmp"
        json.dump(p, open(tmp, "w"), default=str)
        os.replace(tmp, fn)
    runner = Runner(model, ids, docs, distinct, flush)
    try:
        rc = runner.run_controls() if MODE == "controls" else (runner.run_confirm() if CONFIRM else runner.run())
    except Exception as e:
        rc = exit_code_for(runner, e)
        kind = {RC_MEMGATE: "MEMORY FALLBACK TRIGGER", RC_PLACEMENT: "PLACEMENT REFUSED", RC_CRASH: "CRASH",
                RC_HYGIENE: "RESTART PROOF FAILED (fallback (ii): a separate golden process)"}.get(
            rc, "FAILURE AFTER MAIN (MAIN results kept, no fallback)")
        if rc == RC_MEMGATE:
            runner.memgate.append(dict(trigger=(str(e) if isinstance(e, MemGate) else "(a) %s" % str(e)[:300])))
        else:
            runner.crash = dict(kind=kind, exit=rc, error=traceback.format_exc()[-4000:])
        runner.log("%s (exit %d): %s: %s" % (kind, rc, type(e).__name__, str(e)[:300]))
        traceback.print_exc()
        try:
            runner.flush_payload(False)
        except Exception:
            pass
        return rc
    print("[cpupack] saved %s (fails %d, layout fails %d, peak reserved %.2f GB)" % (fn, runner.fails, runner.layout_fails,
                                                                               torch.cuda.max_memory_reserved() / 1e9), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
