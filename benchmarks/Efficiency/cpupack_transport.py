"""CPU-PACKING TRANSPORT beside NOSI's resident decode (authorized 2026-09-29, the user via Codex; ledger 'CPU-PACKING
TRANSPORT EXPERIMENT AUTHORIZED'). Measurement harness only: no production-baseline change.

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
            wait, logits without restore) must be DETECTED; positive lifetime controls must pass. A failure skips MAIN,
            LAYOUT and TIMELINE (exit 21).
  MAIN      untraced, gated steps CP_STEPS (plans[it] = the natural plan of the same step index), per arm REPS transfer-alone
            and REPS beside the resident decode (worker_sweep.py:491-514 light restore; 0 loads and logits torch.equal every
            rep), decode-alone reps before and after. Arms: w8 (flash_h2d_persistent n_ctas=8, num_warps=4,
            bypass_cache=False, K and V), cpu1/2/4/8 (pack on N physical cores), cpu8_c16 (16 MiB chunks), cpu8_lite (events
            lite), cpu8_L / cpu8_LP / cpu8_LPD (cumulative ablations), cpu8_prepacked, ceil_contig, scatter_only (ceilings).
            Then SWEEP: transfer-alone w8 and cpu8 over every captured step (steady 1..62 and the step-0 full list).
  LAYOUT    (layout_ablation; runs AFTER the original arms' JSON is flushed, own deadline CP_LAYOUT_BUDGET_S) the 2x2 of
            {orig, head-major} CPU SOURCE x {orig, head-major} GPU SCRATCH, same natural plans (CP_LAYOUT_PLAN_STEPS), W8 and
            cpu8 variants (row / group packer x row / group placer), each alone and beside the decode. The host cache is
            reordered IN PLACE (no second host cache, no extra pinned memory; cpupack_core.convert_inplace) between the
            orig-source and head-major-source cells, with a bit-exact per-request logical check and a flushed-reference
            logits check before == after. The resident decode's own GPU window layout is untouched. Plus the measured
            tail-write cost per layout and the one-time conversion cost.
  TIMELINE  (separate process under nsys --capture-range=cudaProfilerApi) alone, alone_late (negative control), w8, cpu4,
            cpu8 beside the decode: untraced, traced, untraced reps with NVTX 'lt|b<B>|<arm>|step<it>|<phase>|rep<k>'.
MODES (CP_MODE): run | calib (node-local CPU pack calibration, no model) | table.
EXIT: 0 ok; 1..19 failure count; 20 hygiene gate (two-process fallback); 21 correctness failed; 22 memory fallback
(OOM, pinned allocation failure, free HBM below need, peak reserved above CP_PEAK_LIMIT_GB).
"""
import gc
import json
import os
import queue
import sys
import threading
import time
import traceback

import cpupack_cpu as CC

MODE = os.environ.get("CP_MODE", "run")
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
LAYOUT_BUDGET_S = float(os.environ.get("CP_LAYOUT_BUDGET_S", "1200"))
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
OUT = VA.OUT
TAG = os.environ.get("CP_TAG", "cp_b%d" % VA.BATCH)
RC_HYGIENE, RC_CORRECT, RC_MEMGATE = 20, 21, 22
W8 = dict(n_ctas=8, num_warps=4, bypass_cache=False)
CHUNK_FIELDS = ("g0", "g1", "slot", "pk", "h2d0", "h2d1", "sc0", "sc1", "sub", "bp_ms", "pack_ms", "pack_cpu_ms", "api_ms", "bytes", "useful")
LAYER_FIELDS = ("pr", "list", "hk", "desc", "list_wait_ms", "desc_ms", "desc_cpu_ms", "n")


class MemGate(Exception):
    """A registered B320-fallback trigger (the sbatch reruns ALL arms at the fallback batch)."""


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


def layout_variants(src, dst, top=8):
    """layout_ablation: W8 plus the cpu<top> packer x placer variants of one (source, destination) cell. The packer effect
    is row vs group at a fixed placer (head-major source only); the placer effect is row vs group at a fixed packer
    (head-major destination only)."""
    out = [arm("w8@%s>%s" % (src, dst), "w8")]
    packers = ("row", "group") if src == "hm" else ("row",)
    placers = ("row", "group") if dst == "hm" else ("row",)
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
        self.fn, self.done, self.error, self.result = fn, threading.Event(), None, None


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
            except BaseException:
                job.error = traceback.format_exc()
            finally:
                job.done.set()

    def submit(self, fn) -> Job:
        j = Job(fn)
        self.q.put(j)
        return j

    def call(self, fn):
        j = self.submit(fn)
        j.done.wait()
        if j.error:
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
        self.layout = dict(cells=[], conversion=None, tail_write=None, notes=[], partial=False)
        self.t_start = time.time()
        self.payload_extra = {}
        self.inv_snaps = {}
        self.thread_use = []
        self.dev = "cuda"                                                # the CPU tests drive transport / checks / conversion with "cpu"

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
        if (EARLY or {}).get("omp_env_problems"):
            raise RuntimeError("refused: %s" % EARLY["omp_env_problems"])
        if not (EARLY or {}).get("ok") or len(team) < max(CORES):
            raise RuntimeError("placement gives %d physical team cores < %d on the GPU node (%s)" % (len(team), max(CORES), EARLY))
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
        assert self.NL <= self.plan_h.shape[0], "the plan-list buffer holds %d layers < %d" % (self.plan_h.shape[0], self.NL)
        torch.cuda.reset_peak_memory_stats()

    def loaded_now(self):
        return int(torch.stack([(e._load_mask >= 0).sum() for e in self.engines]).sum())

    def decode(self, tok, pos, warmup=False):
        return self.model.decode_inference(tok, self.cu, pos, self.cache, warmup=warmup) if warmup else \
            self.model.decode_inference(tok, self.cu, pos, self.cache)

    # ---------------------------------------------------------------------------------------------- capture
    def capture(self):
        from nosi import transfer_trace as _tt
        snap = None
        if not NO_SNAPSHOT:
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
                   natural_step_ms=[r.get("step_ms") for r in step_rows], snapshot=not NO_SNAPSHOT)
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
                    src = self.contig_dev[:n * self.D * 2].view(torch.bfloat16).view(n, self.D)
                    K.rows2d(self.scr_k).index_copy_(0, idx, src)
                    K.rows2d(self.scr_v).index_copy_(0, idx, src)
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
        """worker_sweep.py:491-507 (light restore) around `work(ctx)`, then the advance (:584-587)."""
        tok = self.forced[:, it:it + 1]
        snap = self.snap
        snap.take()
        for e in self.engines:
            self.mc.flush_map(e)
        lg_ref = self.decode(tok, pos)
        torch.cuda.synchronize()
        for i, e in enumerate(self.engines):
            slot = snap.layers[i]["engine"]
            for name in ("_block_map", "_new_block_map_buf"):
                slot[name].copy_(getattr(e, name))
        agree = [PL.selection_agreement(self.engines[l]._block_map.cpu(), self.maps[it, l]) for l in range(self.NL)] if it < self.maps.shape[0] else []

        def restore():
            snap.restore()
            self.ss.assert_transients_intact(self.model, self.trans)
        ctx = dict(it=it, tok=tok, pos=pos, lg_ref=lg_ref, restore=restore, step_fn=lambda: self.decode(tok, pos),
                   agreement=dict(same_frac_mean=float(np.mean([x["same_frac"] for x in agree])) if agree else None,
                                  jaccard_mean=float(np.mean([x["jaccard_mean"] for x in agree])) if agree else None,
                                  jaccard_min=float(min(x["jaccard_min"] for x in agree)) if agree else None))
        work(ctx)
        restore()
        self.decode(tok, pos)
        torch.cuda.synchronize()

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
        self.rep_checks(w, c, c["union"], c["last"])
        rows.append(w)
        nf = 0
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
        negs = []
        for name, fl, expect_fail, cap in (
                ("skip_scatter", K.Faults(skip_scatter_chunk=0), True, None),
                ("poison_stage", K.Faults(poison_stage=True), True, None),
                ("no_slot_wait+delay_copy", K.Faults(no_slot_wait=True, delay_copy_ms=LIFETIME_DELAY_MS), True, LIFETIME_CAP),
                ("slot_wait+delay_copy (positive)", K.Faults(delay_copy_ms=LIFETIME_DELAY_MS), False, LIFETIME_CAP),
                ("no_landing_wait+delay_scatter", K.Faults(no_landing_wait=True, delay_scatter_ms=LIFETIME_DELAY_MS), True, LIFETIME_CAP),
                ("landing_wait+delay_scatter (positive)", K.Faults(delay_scatter_ms=LIFETIME_DELAY_MS), False, LIFETIME_CAP)):
            x = self.exact_layer(cpu8, c, 0, cap=cap, faults=fl)
            x.update(control=name, expect_fail=expect_fail, n_chunks_hint=cap)
            x["verdict"] = ("DETECTED" if not x["ok"] else "NOT_DETECTED") if expect_fail else ("PASS" if x["ok"] else "FAIL")
            nf += int(x["verdict"] in ("NOT_DETECTED", "FAIL"))
            negs.append(x)
        res["controls"] = negs
        # decode: zero-load resident step, and the logits negative control (a second step without restore)
        ctx["restore"]()
        lg1 = ctx["step_fn"]()
        torch.cuda.synchronize()
        res["resident_zero_load"] = self.loaded_now() == 0 and bool(torch.equal(lg1, ctx["lg_ref"]))
        lg2 = ctx["step_fn"]()
        torch.cuda.synchronize()
        res["logits_negative_control"] = "DETECTED" if not torch.equal(lg2, ctx["lg_ref"]) else "NOT_DETECTED"
        ctx["restore"]()
        nf += int(not res["resident_zero_load"]) + int(res["logits_negative_control"] != "DETECTED")
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
        for a in arms:
            nf += self.arm_reps(a, ctx, c, REPS, rows, "MAIN")
        for rep in range(REPS):
            r = self.resident(ctx)
            r.update(step=ps, plan_step=ps, phase="decode_alone_post", rep=rep, stage="MAIN")
            nf += int(not r["ok"])
            rows.append(r)
        self.rows.extend(rows)
        return nf

    def sweep_stage(self):
        nf = 0
        arms = [arm("w8", "w8"), arm("cpu%d" % max(CORES), "cpu", cores=max(CORES))]
        self.coord.configure(max(CORES))
        for ps in list(range(1, self.masks.shape[0])) + [0]:
            c = self.prepare(ps, arms)
            for a in arms:
                _, r = self.bracket(a, c, False)
                r.update(plan_step=ps, phase="sweep_alone", rep=0, stage="SWEEP", full_list=(ps == 0))
                ok = self.rep_checks(r, c, c["union"], c["last"])
                nf += int(not ok)
                self.sweep.append(r)
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

    def convert_host(self, to):
        """In-place reorder of every layer's K and V host tensor (cpupack_core.convert_inplace; bit-exact per-request logical
        check); the engines read the new layout through their logical views (the shipped gathers take strides:
        flash_h2d_mask.py:76-77 / :41-56, flash_h2d_mask_bias.py:97-98 / :62-77)."""
        frm = self.host_layout
        per, bad, ms = [], 0, []
        if self.dev == "cuda":
            torch.cuda.synchronize()                                     # no gather may be reading the host cache
        for l, e in enumerate(self.engines):
            new = []
            for which in (0, 1):
                phys = self.host_phys[l][which]
                np_, info = K.convert_inplace(phys, frm, to, verify=True)
                assert np_.data_ptr() == phys.data_ptr()
                bad += info["bad"]
                ms.append(info["ms"])
                new.append(np_)
            self.host_phys[l] = (new[0], new[1])
            e._k_cpu, e._v_cpu = K.logical(new[0], to), K.logical(new[1], to)
            assert K.layout_of(e._k_cpu) == to and K.layout_of(e._v_cpu) == to
            assert self.dev != "cuda" or (e._k_cpu.is_pinned() and e._v_cpu.is_pinned())
            per.append(dict(layer=l, ms_k=ms[-2], ms_v=ms[-1]))
        self.host_layout = to
        self.chk = {}
        acc = K.conversion_accounting(self.B, self.S_cpu, self.H, self.D, 2, self.NL, ms)
        return dict(frm=frm, to=to, bad_requests=bad, per_layer=per, accounting=acc, extra_pinned_bytes=0)

    def layout_stage(self, steps_iter, pos_of):
        """layout_ablation: the 2x2 (module docstring). steps_iter yields free decode steps; each (cell, plan step) uses one
        gated decode step. Returns the failure count (kept separate from the original arms')."""
        t0 = time.time()
        nf = 0
        self.layout["tail_write"] = self.tail_write_bench()
        orig_refs = {}                                                   # references computed BEFORE any conversion
        self.set_scratch("orig")
        for ps in LAYOUT_PLAN_STEPS:
            _, cpu = self.rows_of(ps)
            for l in sorted(set(LAYOUT_CHECK_LAYERS) | {self.NL - 1}):
                r = self.ref_layer(cpu, l)
                orig_refs[(ps, l)] = dict(rk=r["rk"].cpu(), rv=r["rv"].cpu())
        cells = list(LAYOUT_CELLS_PRE) + list(LAYOUT_CELLS_POST)
        for ci, (src, dst) in enumerate(cells):
            if time.time() - t0 > LAYOUT_BUDGET_S:
                self.layout["partial"] = True
                self.layout["notes"].append("deadline reached before cell %s>%s" % (src, dst))
                break
            conv_needed = src != self.host_layout
            self.set_scratch(dst)
            cell = dict(src=src, dst=dst, variants=[a["name"] for a in layout_variants(src, dst, max(CORES))], exact=[], steps=[])
            for j, ps in enumerate(LAYOUT_PLAN_STEPS):
                if j and time.time() - t0 > LAYOUT_BUDGET_S:
                    self.layout["partial"] = True
                    self.layout["notes"].append("deadline reached inside cell %s>%s after %d plan steps" % (src, dst, j))
                    break
                it = next(steps_iter)

                def work(ctx, ps=ps, j=j):
                    nonlocal nf, conv_needed
                    if conv_needed:
                        res = self.convert_host(src)
                        ctx["restore"]()
                        for e in self.engines:
                            self.mc.flush_map(e)
                        lg2 = self.decode(ctx["tok"], ctx["pos"])
                        torch.cuda.synchronize()
                        res["reference_logits_equal_before_after"] = bool(torch.equal(lg2, ctx["lg_ref"]))
                        res["loads_of_reference_after"] = self.loaded_now()
                        nf += int(res["bad_requests"] > 0) + int(not res["reference_logits_equal_before_after"])
                        self.layout["conversion"] = res
                        conv_needed = False
                        self.flush_payload(True)
                    c = self.prepare(ps, [])
                    # the stored pre-conversion reference replaces the live one: content must equal the ORIGINAL bytes
                    ref = c["last"]
                    o = orig_refs[(ps, self.NL - 1)]
                    ref["rk"], ref["rv"] = o["rk"].to("cuda"), o["rv"].to("cuda")
                    for a in layout_variants(src, dst, max(CORES)):
                        if j == 0:
                            for l in LAYOUT_CHECK_LAYERS:
                                x = self.exact_layer(a, c, l)
                                rr = self.ref_layer(c["cpu"], l)
                                o2 = orig_refs[(ps, l)]
                                x["live_ref_equals_original"] = K.bits_equal(rr["rk"].cpu(), o2["rk"]) and K.bits_equal(rr["rv"].cpu(), o2["rv"])
                                ok = x["ok"] and x["live_ref_equals_original"]
                                x["ok"] = ok
                                cell["exact"].append(dict(x, variant=a["name"]))
                                nf += int(not ok)
                        rows = []
                        nf += self.arm_reps(a, ctx, c, LAYOUT_REPS, rows, "LAYOUT")
                        for r in rows:
                            r["cell"] = "%s>%s" % (src, dst)
                        self.layout_rows.extend(rows)
                    cell["steps"].append(dict(decode_step=ctx["it"], plan_step=ps, agreement=ctx["agreement"]))
                self.gated(it, pos_of(it), work)
                self.flush_payload(True)
            self.layout["cells"].append(cell)
        if LAYOUT_BACK and self.host_layout != "orig":
            self.layout["conversion_back"] = self.convert_host("orig")
        self.layout["seconds"] = time.time() - t0
        return nf

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
                                layer_fields=("hk", "desc", "list_wait_ms", "desc_ms", "desc_cpu_ms", "n", "chunks"),
                                env={k: v for k, v in os.environ.items() if k.startswith(("CP_", "NOSI_", "OMP_", "CUDA_", "TORCH_", "TRITON_"))}),
                    provenance=dict(nosi_commit=os.environ.get("NOSI_COMMIT"), repo_commit=os.environ.get("REPO_COMMIT"), torch=torch.__version__,
                                    host=os.uname().nodename, pid=os.getpid()),
                    cpu=dict(placement=EARLY, model=CC.cpu_model(), smt_active=CC.smt_active(), configs=getattr(getattr(self, "coord", None), "configs", None)),
                    capture=getattr(self, "capture_rec", None), hygiene=getattr(self, "hygiene", None), correct=self.correct,
                    rows=self.rows, sweep=self.sweep, layout=dict(self.layout, rows=self.layout_rows), timeline=self.timeline,
                    inventory=self.inv_snaps, thread_use=self.thread_use, seconds=time.time() - self.t_start, docs=self.docs,
                    distinct_books=self.distinct, **self.payload_extra)

    def flush_payload(self, partial):
        self.flush_fn(self.payload(partial))

    # ---------------------------------------------------------------------------------------------- run
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
        self.work = torch.zeros((self.NL, self.W), dtype=torch.int32, device="cuda")
        need = mem_need_gb(self.B, self.S_dst, self.H, self.D, RING, self.slot, CONTIG_BYTES)
        free = torch.cuda.mem_get_info()[0] / 1e9
        self.payload_extra["memory_gate"] = dict(free_gb=free, need_gb=need, margin_gb=MEM_MARGIN_GB)
        if free < need + MEM_MARGIN_GB:
            raise MemGate("free HBM %.2f GB < need %.2f + margin %.2f GB" % (free, need, MEM_MARGIN_GB))
        self.set_scratch("orig")
        torch.cuda.reset_peak_memory_stats()
        self.snap_inventory("after_setup")
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
        if EARLY and EARLY.get("launch") is not None:
            CC.set_mask([EARLY["launch"]])                               # the launch thread alone on its core from here on
        max_step = self.masks.shape[0] - 1
        main_steps = [s for s in STEPS if "MAIN" in STAGES]
        t_steps = [s for s in T_STEPS if "TIMELINE" in STAGES]
        first = min(main_steps + t_steps) if (main_steps or t_steps) else VA.WARM
        arms = main_arms()
        correct_done = "CORRECT" not in STAGES
        order = sorted(set(main_steps + t_steps + ([first] if not correct_done else [])))
        cur = VA.WARM

        def advance_to(target, pos):
            nonlocal cur
            while cur < target:                                          # natural steps between gated steps
                self.decode(self.forced[:, cur:cur + 1], pos)
                torch.cuda.synchronize()
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
        if "MAIN" in STAGES and SWEEP:
            self.fails += self.sweep_stage()
            self.flush_payload(True)
        self.snap_inventory("after_main")
        if self.memgate:
            self.flush_payload(False)
            raise MemGate("; ".join(m["trigger"] for m in self.memgate))
        if "LAYOUT" in STAGES:
            self.flush_payload(True)                                    # the original arms are on disk before the add-on
            free_steps = iter(range(cur, VA.N))
            pos_box = dict(pos=pos, cur=cur)

            def pos_of(target):
                nonlocal cur
                p = advance_to(target, pos_box["pos"])
                cur = target + 1                                        # the gated step consumes `target`
                pos_box["pos"] = p + 1
                return p
            try:
                self.layout_fails = self.layout_stage(free_steps, pos_of)
            except StopIteration:
                self.layout["notes"].append("ran out of no-rollover decode steps")
                self.layout["partial"] = True
            self.snap_inventory("after_layout")
            self.flush_payload(True)
        self.flush_payload(False)
        return min(self.fails + self.layout_fails, 19)


class _CorrectFail(Exception):
    pass


class _SnapshotOOM(Exception):
    pass


def MODES_DMA(a) -> bool:
    return K.MODES[a["mode"]][1] if a.get("kind") == "cpu" else False


# ------------------------------------------------------------------------------------------------------------ calib
def calib():
    """Node-local CPU calibration (stage C0 of the job, no model): SYNTHETIC Poisson(CP_CALIB_MEAN) plans at the batch
    geometry, the coordinator on 1/2/4/8 team cores, the row packer from the original layout, then the host tensor reordered
    in place to head-major (the one-time conversion, timed) and the row and group packers from it. Pack only (mode LP),
    8 MiB chunks, pinned staging. Label: 'synthetic plan, node-local pack calibration'."""
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
    stage = co.call(lambda c: torch.zeros((RING, K.slot_bytes(cap)), dtype=torch.uint8).pin_memory())
    land = torch.zeros_like(stage)
    phys_k = K.alloc_phys("orig", B, S, H, D, torch.bfloat16)
    phys_v = K.alloc_phys("orig", B, S, H, D, torch.bfloat16)
    phys_k.view(torch.int16).random_(-3000, 3000, generator=g)
    phys_v.view(torch.int16).random_(-3000, 3000, generator=g)
    dst = K.alloc_phys("orig", B, 4096, H, D, torch.bfloat16)
    res = dict(label="synthetic plan (Poisson %.2f per stream), node-local pack calibration; pack only (LP)" % mean, B=B, S=S,
               groups=int((plan[..., :63] >= 0).sum()), placement=EARLY, cpu_model=CC.cpu_model(), rows=[], conversion=None)

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


def table(out_dir):
    """Markdown + CSV summaries of every cp_*.json under out_dir (medians / p95 over individual samples)."""
    import csv
    import glob
    L = ["# CPU-packing transport (%s; %s)" % (K.LABEL, K.REPLAY_LABEL), ""]
    ok_all = True
    for fn in sorted(glob.glob(os.path.join(out_dir, "cp_*.json"))):
        p = json.load(open(fn))
        B = p.get("batch")
        ok_all &= p.get("fails", 1) == 0 and not p.get("partial")
        L.append("## B=%s (%s): fails %s, layout fails %s%s" % (B, os.path.basename(fn), p.get("fails"), p.get("layout_fails"),
                                                             " PARTIAL" if p.get("partial") else ""))
        cap = p.get("capture") or {}
        masks = None
        npz = cap.get("export") or (cap.get("source", "").split(" ")[1] if cap.get("source", "").startswith("export") else None)
        if npz and os.path.exists(npz):
            masks = PL.load_npz(npz)["masks"]
        req = {}

        def req_of(ps):
            if ps not in req and masks is not None:
                req[ps] = [(masks[ps, l][..., :63] >= 0).sum(dim=(0, 2)).tolist() for l in range(masks.shape[1])]
            return req.get(ps)
        groups = {}
        alone = {}
        for r in p.get("rows", []) + p.get("layout", {}).get("rows", []):
            key = (r.get("stage"), r.get("cell", "orig>orig"), r["arm"])
            if r.get("phase", "").startswith("decode_alone"):
                alone.setdefault(r.get("step"), []).append(r["main_ms"])
                continue
            if r.get("phase") == "warmup":
                continue
            groups.setdefault(key, []).append(r)
        L += ["", "| stage | cell | arm | reps alone/conc | useful GB/s alone p50 | useful GB/s conc p50 | e2e p50 / p95 ms (requests) "
              "| list / wake / desc / pack / submit / dma / scq / scatter p50 ms | decode alone -> conc ms p50 | slowdown | late ms p50 | overlap p50 | ok |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        csv_rows = []
        for (stage, cellk, a), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
            al = [x for x in rs if not x.get("with_decode")]
            co = [x for x in rs if x.get("with_decode")]
            agg_a = [TM.rep_aggregate(rep_view(x), x["useful"]) for x in al]
            agg_c = [TM.rep_aggregate(rep_view(x), x["useful"]) for x in co]
            samples = []
            for x in co:
                rq = req_of(x.get("plan_step"))
                if rq is not None:
                    for s in request_rows(x, rq):
                        s.update(stage=stage, cell=cellk, arm=a, step=x.get("step"), rep=x.get("rep"), mode="conc")
                        samples.append(s)
            csv_rows += samples
            sm = TM.summarize(samples, ("e2e",) + TM.CPU_STAGES)
            dec_c = [x["main_ms"] for x in co]
            dec_a = [m for x in co for m in alone.get(x.get("step"), [])]
            da, dc = TM.pctl(dec_a, 50), TM.pctl(dec_c, 50)
            L.append("| %s | %s | %s | %d/%d | %.2f | %.2f | %.2f / %.2f | %s | %.2f -> %.2f | %+.1f%% | %.2f | %.2f | %d/%d |" % (
                stage, cellk, a, len(al), len(co), TM.pctl([g["useful_gbps"] for g in agg_a], 50), TM.pctl([g["useful_gbps"] for g in agg_c], 50),
                sm["e2e"]["p50"], sm["e2e"]["p95"], " / ".join("%.2f" % sm[k]["p50"] for k in TM.CPU_STAGES), da, dc,
                100 * (dc / da - 1) if da == da and da > 0 else float("nan"), TM.pctl([g.get("late_ms") for g in agg_c], 50),
                TM.pctl([g.get("overlap_frac") for g in agg_c], 50), sum(1 for x in rs if x.get("ok")), len(rs)))
        if csv_rows:
            keys = sorted({k for s in csv_rows for k in s})
            with open(os.path.join(out_dir, "requests_b%s.csv" % B), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for s in csv_rows:
                    w.writerow(s)
        lay = p.get("layout") or {}
        if lay.get("conversion"):
            cv = lay["conversion"]
            L.append("")
            L.append("layout_ablation conversion %s->%s: bad requests %s, reference logits equal before/after %s, total %.1f s (%s)" % (
                cv["frm"], cv["to"], cv["bad_requests"], cv.get("reference_logits_equal_before_after"),
                (cv["accounting"].get("measured_ms_total") or 0) / 1000, "in place, extra pinned 0"))
        if lay.get("tail_write"):
            for k, v in lay["tail_write"].items():
                a = v["accounting"]
                L.append("layout_ablation tail write %s: %d runs x %d B per tensor; measured %.3f ms/tensor (host), %.3f ms per rollover, "
                         "%.4f ms per token amortized; link bound %.4f ms per token" % (k, a["runs"], a["run_bytes"], float(np.median(v["host_ms"])),
                                                                                       a["measured_ms_per_rollover"], a["measured_ms_per_token"],
                                                                                       a["link_bound_ms_per_token"]))
        L.append("")
    text = "\n".join(L) + "\n"
    open(os.path.join(out_dir, "cpupack_table.md"), "w").write(text)
    print(text)
    return 0 if ok_all else 1


# ------------------------------------------------------------------------------------------------------------ main
def main():
    if MODE == "table":
        return table(OUT)
    os.makedirs(OUT, exist_ok=True)
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
        rc = runner.run()
    except MemGate as e:
        runner.memgate.append(dict(trigger=str(e)))
        runner.log("MEMORY FALLBACK TRIGGER: %s" % e)
        try:
            runner.flush_payload(False)
        except Exception:
            pass
        return RC_MEMGATE
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg or "cudaHostAlloc" in msg or "pinned" in msg.lower():
            runner.memgate.append(dict(trigger="(a) %s" % msg[:300]))
            runner.log("MEMORY FALLBACK TRIGGER (OOM / pinned): %s" % msg[:300])
            try:
                runner.flush_payload(False)
            except Exception:
                pass
            return RC_MEMGATE
        raise
    print("[cpupack] saved %s (fails %d, layout fails %d, peak reserved %.2f GB)" % (fn, runner.fails, runner.layout_fails,
                                                                               torch.cuda.max_memory_reserved() / 1e9), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
