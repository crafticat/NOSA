"""CACHE-SIZE INTERFERENCE CURVE beside NOSI's resident decode (authorized 2026-10-01, the user via Codex: 'THE CACHE-SIZE CURVE',
now the priority; retroinfer-eval REPRODUCE.md 'CACHE-CURVE BUILD STARTED'). Measurement harness only: no production-baseline
change (cache_engine.py, nosa_llama.py and state_snapshot.py are NOT edited; the pool state is handled by pool_state.py).
Built on feature/nosi-interference-curve @ 957588e (interference_curve.CurveRunner: the sustained window, the pump, the
receipts, CONTROLS, TABLE; cpupack_transport.Runner: gated(), the golden gate, the guard, the restart proof) on fork branch
feature/nosi-cache-curve.

QUESTION. NOSA-8B / PG-19 / L = 16128 / A100: how do the decode interference and the delivered rate of NOSI's GPU gather (W8)
and the CPU-pack transport (CPU8) change with the cache capacity C in {63, 73, 81, 96, 113, 128} historical 64-token/head
groups per (layer, KV head, request) (C = 63 attended non-tail slots + P = NOSI_POOL_BLOCKS victim slots; the tail slot is one
more 64-token group per stream, always resident, written locally and never fetched), as the natural miss volume falls?

LABELS: 'transport/placement-ready, NOT live-LRU-ready'; 'GPU-resident trace replay'; saturation = 'resource-contention control,
NOT live verifier throughput'; DECODE-PACED (raw key 'paced', CC1's 'arrival-paced') = ccurve_core.DECODE_PACED_LABEL: releases at
each method's OWN decode tick starts, a method-dependent arrival schedule, NOT a matched external load (Codex's CC1 audit);
'RESIDENT decode controls, NOT cache-draft / verifier throughput'; CPU8 / W8 copy rows = 'prebuilt-plan copy microbenchmark, NOT an
integrated LRU getter baseline' (descriptors are built LIVE for CPU8, never the prebuilt cache).

THE HOST-PACK DIAGNOSTIC (CC2, authorized by the user via Codex 2026-10-01: 'a small HOST-PACK diagnostic ... alongside the B64
curve'): at the cells CC_DIAG_CAPS (default '63 128') only, for the arms CC_DIAG_ARMS (default 'cpu8 cpu8_hostpack'):
  * 'cpu8_hostpack' (cc_arm; ccurve_core.HOSTPACK_LABEL) = the cpu8 code path with ONLY the pipe mode 'LP' (pack, no H2D, no
    scatter): fresh list D2H, host wait, live descriptors and CPU pack on 8 cores, all charged as for cpu8. Gates (hostpack_gates):
    the scratch is bit-identical before / after (scratch_gate), no H2D / scatter is issued (counted at the branch), the staging slots
    hold the logical source rows (staging_check), receipts, golden gate. Run in saturation (the same train as cpu8) and ext-paced.
  * EXT-PACED (ext_window; ccurve_core.EXT_LABEL): a prerecorded external release schedule, fixed per (cell, step) right after
    decode-alone pre and before any arm (ext_schedule_for: P = that step's decode-alone tick p50, T0 = CC_EXT_T0_MS), issued by a
    dedicated release thread; cpu8 and cpu8_hostpack see the IDENTICAL schedule (SCHEDULE gate: digest, every release issued by the
    release thread, lateness recorded). Diagnostic numbers per window: rec['diag'] (ccurve_core.diag_metrics).
  Everywhere else (and for w8) nothing changes: w8 + cpu8 saturation + decode-paced at every C. The table adds the diagnostic
  section and ccurve_hostpack.csv (side by side, NO subtraction).
TABLE CAVEATS (printed once, ccurve_core.TABLE_CAVEATS): during GB/s vs covered-tick slowdown are not a cost per byte; every
resident tick includes the decode-state restore (its bytes per tick per cell are reported, note_restore); H1 / H2 not firing is not
evidence of no host contention; no additive GPU share, the GIL is a hypothesis.

ONE PROCESS = ONE BATCH B, a list of capacities CC_CAPACITIES in order (default '128 63 73 81 96 113': the most memory-hungry
cell, C128, FIRST = the fit probe; C63 second = the cross-C reference). The engine is imported with NOSI_POOL_BLOCKS = the
largest P of the list, so the ONE prefill allocates the largest window + pool; every later cell restarts to the post-prefill
state and RE-POOLS the engines (pool_state.repool: same host KV, the GPU window / pool re-initialised exactly as prefill_update
section 2-3c would for that P; certified by the restart proof, pool_is_fresh and the cross-C check). CC_CAPACITIES of length 1
= one process per (B, C) (the fallback when reuse is not wanted).

PER CELL (B, C): [restart + repool] -> CAPTURE (CP_NCAP natural steps at C: post-pool H2D masks, window maps, pool actions,
pool maps; invariants I1..I9 where they hold + pool-aware P1..P5; natural accounting = the TRUE natural misses / victims / pool
hits at C) -> [C63 at B64: the capture equals the SAVED 2179683 plans on its prefix] -> restart -> GOLDEN (warm + gated steps;
pool-aware digests) -> cross-C check against C63 (capture logits, reference / advance logits; a cell that differs is NOT
measured and is MISSING) -> restart -> MEASURED: per gated step, decode-alone pre, SATURATION (transfer-alone + overlap per arm
x CV_REPS), DECODE-PACED (CC_PACED_TICKS releases per arm x CV_REPS), [at a diagnostic cell: host-pack saturation +
EXT-PACED], decode-alone post. Per cell: cc_b<B>_c<C>.json (payload),
.result.json (rc, stop class), _plans.npz (+ pool archives), _golden.json.

GATES AND RECEIPTS with the pool (pool_state.py): the CounterSnapshot is PoolCounterSnapshot (pool map / ages / targets / actions
+ the host stamp base; post-reference state copied by gated()'s post_reference hook); the golden digests carry the 'pool' and
'poolrows' families before timing and after the advance; the gate also requires ZERO duplicate residency (window U pool) and
ZERO pool actions in the pre-timing resident check; every tick's receipt = 0 H2D loads AND 0 pool actions AND logits bit-exact.

EXIT: 0 ok; 1..19 failure count; 20 restart proof; 21 correctness / plan mismatch / uncertified; 22 memory trigger (the stop
class is in the marker: the sbatch's fit-probe fallback reads it); 23 crash or leak (the sbatch resumes the cells without a
marker in a new process); 24 CPU placement refused.
"""
import csv
import gc
import glob
import gzip
import json
import math
import os
import re
import sys
import time
import traceback
import types

import cpupack_cpu as CC

MODE = os.environ.get("CC_MODE", "run")                                # run | controls | table
STEPS_S = os.environ.get("CC_STEPS", os.environ.get("CV_STEPS", "4 5 6 7 8 9"))
PACED = os.environ.get("CC_PACED", "1") == "1"
PACED_TICKS = int(os.environ.get("CC_PACED_TICKS", "24"))
TRAIN_STEPS_ = int(os.environ.get("CC_TRAIN_STEPS", os.environ.get("CV_TRAIN_STEPS", "4")))
_STEPS = tuple(int(x) for x in STEPS_S.split())


def replayed_steps(ncap):
    """The natural steps the windows replay: the first gated step .. the capture's last. A victim pool is EMPTY after prefill and
    fills over the first decode steps (miniature 2179945, B32: C128 H2D 3.73 groups per stream-step at step 1, 0.65 at 17), so a
    summary from step 1 mixes that cold start into the replayed miss volume."""
    return list(range(min(_STEPS), ncap))


os.environ["CV_MODE"] = {"run": "run", "controls": "controls", "table": "table"}.get(MODE, "run")
os.environ["CV_STEPS"] = STEPS_S
os.environ["CV_TRAIN_STEPS"] = str(TRAIN_STEPS_)
_EXT_NEEDS_PACED = os.environ.get("CC_EXT", "1") == "1" and bool(os.environ.get("CC_DIAG_CAPS", "63 128").split())   # ext replays the paced train
os.environ.setdefault("CP_NCAP", str(max(_STEPS) + max(TRAIN_STEPS_, PACED_TICKS if (PACED or _EXT_NEEDS_PACED) else 0) + 1))
os.environ["CV_ARMS"] = os.environ.get("CC_ARMS", "w8 cpu8")
os.environ["CV_TRADE_BATCHES"] = ""
os.environ["CV_CPU_BATCHES"] = ""
os.environ["CV_BURST_BATCHES"] = ""
EARLY = None
if __name__ == "__main__" and MODE == "run":
    _gn = os.environ.get("CP_GPU_NUMA_NODE", "").strip()
    EARLY = CC.early_placement(int(os.environ.get("CP_TEAM_MAX", "8")), int(_gn) if _gn.isdigit() else None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ccurve_core as CCC  # noqa: E402
import cpupack_core as K  # noqa: E402
import cpupack_golden as G  # noqa: E402
import cpupack_plans as PL  # noqa: E402
import interference_curve as IC  # noqa: E402
import pool_state as PS  # noqa: E402

CT, CV, VA = IC.CT, IC.CV, IC.VA
CT.EARLY = EARLY
IC.EARLY = EARLY
BATCH = int(os.environ.get("CC_BATCH", os.environ.get("CP_B", "192")))
CAPS = tuple(int(x) for x in os.environ.get("CC_CAPACITIES", "128 63 73 81 96 113").split())
CONTROLS_C = int(os.environ.get("CC_CONTROLS_C", "128"))
ARMS = tuple(os.environ.get("CC_ARMS", "w8 cpu8").split())
REF_C = 63
CELL_EST_S = float(os.environ.get("CC_CELL_EST_S", "600"))
CELL_EST_FIXED_S = float(os.environ.get("CC_PREFILL_EST_S_PER_REQ", "3.2"))
CROSS_REQUIRED = os.environ.get("CC_CROSS_REQUIRED", "1") == "1"
LEAK_LIMIT_GB = float(os.environ.get("CC_LEAK_LIMIT_GB", "2.0"))
NUMA_INVENTORY = os.environ.get("CC_NUMA_INVENTORY", "first")        # first | every | never
XREF_DIRS = tuple(os.environ.get("CC_XREF_DIRS", "").split())          # other jobs' cell directories (the C63 reference of leftovers)
DIAG_CAPS = tuple(int(x) for x in os.environ.get("CC_DIAG_CAPS", "63 128").split())     # the host-pack diagnostic's cells
DIAG_ARMS = tuple(os.environ.get("CC_DIAG_ARMS", "cpu8 cpu8_hostpack").split())          # its ext-paced arms (+ host-pack saturation)
EXT = os.environ.get("CC_EXT", "1") == "1"                                                # the ext-paced regime at the diagnostic cells
EXT_T0_MS = float(os.environ.get("CC_EXT_T0_MS", "2.0"))                                  # release 0 at the anchor + T0
EXT_LATE_TOL_MS = float(os.environ.get("CC_EXT_LATE_TOL_MS", "1.0"))                      # a release later than this is counted late
OUT = VA.OUT
SAT_PHASES = ("transfer_alone", "overlap")
CAPACITY_CLASSES = CV.GPU_CLASSES + CV.HOST_CLASSES


def saved_for(B):
    return os.environ.get("CC_SAVED_%d" % B, ""), os.environ.get("CC_SAVED_SHA_%d" % B, "")


def cc_arm(name):
    """interference_curve.curve_arm + the HOST-PACK control 'cpu8_hostpack' (ccurve_core.HOSTPACK_LABEL): the cpu8 arm with ONLY
    the pipe mode changed to 'LP' = (pack, no DMA, no scatter, host writes the slot) (cpupack_core.MODES). Same coordinator, cores,
    chunk, packer / placer, list D2H, live descriptors and release handling as cpu8."""
    if name == "cpu8_hostpack":
        if K.MODES["LP"] != (True, False, False, True):
            raise RuntimeError("cpupack_core.MODES['LP'] is %s, not pack-only" % (K.MODES["LP"],))
        a = dict(IC.curve_arm("cpu8"))
        a.update(name=name, mode="LP", hostpack=True, label=CCC.HOSTPACK_LABEL, done=CCC.HOSTPACK_DONE_LABEL)
        return a
    return IC.curve_arm(name)


def snapshot_bytes(snap):
    """Bytes the sanctioned per-tick decode-state restore copies (every tensor of the CounterSnapshot: engine bookkeeping + pool
    tensors + the layer compress / CIS tables); HBM traffic ~2x (read the saved copy + write the live tensor)."""
    tot, n = 0, 0
    for slot in getattr(snap, "layers", None) or []:
        for part in ("engine", "layer"):
            for v in (slot.get(part) or {}).values():
                if torch.is_tensor(v):
                    tot += v.numel() * v.element_size()
                    n += 1
    return dict(bytes=tot, tensors=n, hbm_traffic_bytes_estimate=2 * tot)


def tag_of(B, C):
    return "cc_b%d_c%d" % (B, C)


class SSProxy:
    """The state_snapshot module with CounterSnapshot replaced by PoolCounterSnapshot. Attribute WRITES go to the real module
    (interference_curve.restore_nosync elides the per-restore host synchronize by assigning ss._sync_host_window, which the
    real CacheSnapshot.restore reads from its own module globals)."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_over", dict(CounterSnapshot=PS.make_pool_snapshots(real)[0]))

    def __getattr__(self, n):
        over = object.__getattribute__(self, "_over")
        return over[n] if n in over else getattr(object.__getattribute__(self, "_real"), n)

    def __setattr__(self, n, v):
        setattr(object.__getattribute__(self, "_real"), n, v)


class _Uncertified(Exception):
    pass


# --------------------------------------------------------------------------------------------------------- runner
class CacheCurveRunner(IC.CurveRunner):
    """One batch, several capacities (module docstring). Shared across cells: the model, the prefilled cache (host KV), the
    hosted PostPrefillSnapshot, the coordinator, the pinned staging, the streams and backends. Per cell: everything else."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cell_C = None
        self.cell_P = None
        self.cross = {}                                                  # {C: capture_sha, ref_sha, adv_sha} of this process
        self.cell = {}
        self._pacts = []
        self.skip_post_ref_at = None
        self.diag_arms = []
        self._cpu_out = None
        self.ext_late_ns = None                                          # CPU tests only: a deliberately late release

    # ------------------------------------------------------------------------------------------------ setup
    def install_pool_snapshots(self, ss):
        self.ss = ss if isinstance(ss, SSProxy) else SSProxy(ss)
        return self.ss

    def setup_model(self):
        super().setup_model()                                            # VA._setup: ONE prefill at the import P
        self.install_pool_snapshots(self.ss)
        self.P_prefill = PS.pool_blocks(self.engines[0])
        if self.cell_P is None:                                          # CONTROLS sets it before; a cell keeps its own P
            self.cell_P = self.P_prefill

    def setup_curve(self, minimal=False):
        super().setup_curve(minimal)
        self.rc_swaps = torch.zeros(IC.K_MAX + max(PACED_TICKS, 1) + 1, dtype=torch.int64, device=self.dev)
        if self.rc_loads.numel() < self.rc_swaps.numel():
            self.rc_loads = torch.zeros_like(self.rc_swaps)
            self.rc_eq = torch.zeros(self.rc_swaps.numel(), dtype=torch.bool, device=self.dev)
        if minimal:
            return
        self.arms = [IC.curve_arm(a) for a in ARMS]
        self.diag_arms = [cc_arm(a) for a in DIAG_ARMS] if DIAG_CAPS else []
        n_req = max(IC.TRAIN_STEPS, PACED_TICKS if (PACED or (self.diag_arms and EXT)) else 0) * self.NL
        if self.dev == "cuda":
            self.be_main = K.CudaBackend(dict(main=self.s_main, plan=self.s_plan), self.aux_main, pool_size=4 * (IC.K_MAX + PACED_TICKS) + 64)
            self.be_pump = K.CudaBackend(dict(side=self.s_side), self.aux_pump, pool_size=4 * n_req + 64)
            self.aux_plan = torch.cuda.Stream()
            self.be_plan = K.CudaBackend(dict(plan=self.s_plan), self.aux_plan, pool_size=2 * n_req + 64)
            self.cbe = K.CudaBackend(dict(copy=self.s_copy, scatter=self.s_scatter), self.aux, pool_size=int(os.environ.get("CC_CBE_EVENTS", "60000")))
            self.pipes = {}
            if self.diag_arms and EXT:                                   # the ext-paced RELEASE thread's own stream + backend
                self.s_rel, self.aux_rel = torch.cuda.Stream(), torch.cuda.Stream()
                self.be_rel = K.CudaBackend(dict(release=self.s_rel), self.aux_rel, pool_size=PACED_TICKS + 64)
        if any(a["kind"] == "cpu" for a in list(self.arms) + list(self.diag_arms)):
            W = self.W

            def alloc(c):
                try:
                    t = torch.full((n_req, W), -7, dtype=torch.int32)
                    return t.pin_memory() if self.dev == "cuda" else t
                except RuntimeError as e:
                    raise CT.MemGate("pinned list rows allocation failed: %s" % e)
            self.plan_t = self.coord.call(alloc) if hasattr(self, "coord") else alloc(None)

    def snap_inventory(self, label):
        if NUMA_INVENTORY == "every" or (NUMA_INVENTORY == "first" and not getattr(self, "_numa_done", False)):
            self._numa_done = True
            return super().snap_inventory(label)
        try:
            inv = dict(hbm=dict(peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                                free_gb=torch.cuda.mem_get_info()[0] / 1e9, total_gb=torch.cuda.mem_get_info()[1] / 1e9),
                       pinned=self.pinned_inventory(), numa="skipped (CC_NUMA_INVENTORY=%s)" % NUMA_INVENTORY, meminfo=CC.meminfo())
        except Exception as e:
            inv = dict(error=repr(e))
        self.inv_snaps[label] = inv

    def pinned_inventory(self):
        host = sum(t.numel() * t.element_size() for pair in getattr(self, "host_phys", []) for t in pair)
        st = getattr(self, "stage", None)
        pt = getattr(self, "plan_t", None)
        return dict(host_cache_bytes=host, host_cache_reserved_pow2=sum(CT.pow2(t.numel() * t.element_size()) for pair in getattr(self, "host_phys", []) for t in pair),
                    staging_bytes=(st.numel() if st is not None else 0), plan_rows_bytes=(pt.numel() * 4 if pt is not None else 0))

    # ------------------------------------------------------------------------------------------------ cells
    def begin_cell(self, C, flush):
        """Per-cell bookkeeping reset: the base constructors re-initialise every per-run field (rows, windows, golden, gate log,
        restarts, fails, guard, ...); the shared resources (model, cache, engines, pps, coordinator, staging, streams) are
        attributes the constructors do not touch."""
        dev, arms = self.dev, list(getattr(self, "arms", []) or [])
        IC.CurveRunner.__init__(self, self.model, self.ids, self.docs, self.distinct, flush)
        self.dev = dev
        for n in ("capture_rec", "hygiene", "crash", "accounting", "cross_check", "repool_recs"):
            if hasattr(self, n):
                delattr(self, n)
        self.chk = {}
        self.cell_C, self.cell_P = int(C), PS.pool_of(C)
        self.cell = dict(C=self.cell_C, P=self.cell_P, tail_groups_per_stream=1, regimes=(["saturation", "paced"] if PACED else ["saturation"]),
                         paced_ticks=(PACED_TICKS if PACED else 0), arms=list(ARMS), label_resident=CV.RESIDENT_LABEL, label_paced=CCC.PACED_LABEL,
                         note="C = 63 attended non-tail slots + P pool slots per (layer, KV head, request); the tail slot (1 group) is separate")
        self.curve["ptrains"] = {}
        self.arms = arms if arms else [IC.curve_arm(a) for a in ARMS]
        self.diag_arms = [cc_arm(a) for a in DIAG_ARMS] if DIAG_CAPS else []
        self.mark_diag()

    def mark_diag(self):
        """The cell's record of the diagnostic (what the table reads to emit its rows); idempotent."""
        if self.is_diag() and "diag" not in self.cell:
            self.cell["diag"] = dict(caps=list(DIAG_CAPS), arms=[a["name"] for a in self.diag_arms], ext=EXT, ext_t0_ms=EXT_T0_MS,
                                     late_tol_ms=EXT_LATE_TOL_MS, hostpack_saturation=[a["name"] for a in self.hostpack_arms()],
                                     label_ext=CCC.EXT_LABEL, label_hostpack=CCC.HOSTPACK_LABEL)
            self.cell["regimes"] = list(self.cell.get("regimes") or []) + (["ext-paced"] if EXT else [])

    def is_diag(self):
        return bool(self.diag_arms) and self.cell_C in DIAG_CAPS

    def hostpack_arms(self):
        return [a for a in self.diag_arms if a.get("hostpack")]

    def ext_arms(self):
        return list(self.diag_arms) if EXT else []

    def end_cell(self):
        """Drop the per-cell tensors (scratch, plan store, archives, snapshot, references); the shared ones stay."""
        for n in ("scr_k", "scr_v", "store", "masks", "maps", "acts", "pool_maps", "snap", "work", "_masks"):
            if hasattr(self, n):
                try:
                    delattr(self, n)
                except AttributeError:
                    pass
        self.scr_layout = None
        self.refs, self.hs_plans, self._pacts = {}, {}, []
        gc.collect()
        if self.dev == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def restart(self, pps, label):
        """cpupack_transport.Runner.restart + the pool: PostPrefillSnapshot restore, then repool / reset_pool to this cell's P,
        then the proofs (P-independent start families == the post-prefill digest; pool_is_fresh on every engine)."""
        self.pps = pps
        return self.restart_to(self.cell_P if self.cell_P is not None else PS.pool_blocks(self.engines[0]), label)

    def restart_to(self, P, label):
        self.snap = None
        gc.collect()
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        t = time.time()
        G.snapshot_restore_hosted(self.pps)
        ext = True
        if self.dev == "cuda":
            from nosi import cache_engine as _ce
            ext = _ce._pool is not None and _ce._flash_pool_swap is not None
        reps = []
        if any(PS.pool_blocks(e) != P for e in self.engines):
            for e in self.engines:
                reps.append(PS.repool(e, P, ext_loaded=ext))
            if self.dev == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        else:
            for e in self.engines:
                PS.reset_pool(e)
        bad = G.compare(self.d0, self.start_digest())
        fresh = [PS.pool_is_fresh(e) for e in self.engines]
        fresh_bad = [i for i, f in enumerate(fresh) if not f["ok"]]
        self.model.has_buffers = False
        self.S_dst = int(self.engines[0]._k_gpu.shape[1])
        rec = dict(label=label, ok=not bad and not fresh_bad, mismatched=bad[:64], P=P, repooled=bool(reps), pool_fresh_bad_layers=fresh_bad[:16],
                   rows=self.S_dst, seconds=time.time() - t)
        if reps:
            rec["repool"] = reps[0]
        self.restarts.append(rec)
        self.log("restart '%s' (P=%d%s): start digest %s, pool %s (%.1fs)" % (label, P, ", repooled" if reps else "", "IDENTICAL" if not bad else "DIFFERS %s" % bad[:6],
                                                                         "fresh" if not fresh_bad else "NOT FRESH at %s" % fresh_bad[:6], rec["seconds"]))
        if not rec["ok"]:
            raise CT._ProofFail("restart '%s': not bit-identical (%s; pool not fresh at %s)" % (label, bad[:8], fresh_bad[:8]))
        return rec

    # ------------------------------------------------------------------------------------------------ digests / gates
    def state_digest(self, families):
        d = super().state_digest(families)
        if tuple(families) != tuple(G.FAMILIES_START) and any(PS.pool_active(e) for e in self.engines):
            d.update(PS.pool_digest(self.cache))
        return d

    def _pool_gate(self, rec, resident_actions):
        if self.golden_mode is None or not any(PS.pool_active(e) for e in self.engines):
            return rec
        dup = PS.pool_dup(self.cache)
        rec["pool_dup"] = sum(dup)
        why = list(rec.get("why") or [])
        if resident_actions is not None:
            rec.setdefault("resident", {})["pool_actions"] = resident_actions
            if resident_actions:
                why.append("resident(pool_actions=%d)" % resident_actions)
        if sum(dup):
            why.append("pool_dup%s" % [i for i, x in enumerate(dup) if x][:8])
        rec["why"], rec["ok"] = why[:64], not why
        return rec

    def gate_pre(self, it, lg_ref, ctx):
        rec = super().gate_pre(it, lg_ref, ctx)                          # restore -> resident decode -> loads + logits
        acts = PS.pool_actions_now(self.engines) if any(PS.pool_active(e) for e in self.engines) else None
        rec = self._pool_gate(rec, acts)
        if self.golden_mode == "record" and acts is not None:
            self.golden["steps"].setdefault(it, {})["resident"] = rec.get("resident")
        return rec

    def gate_post(self, it, lg_adv):
        return self._pool_gate(super().gate_post(it, lg_adv), None)

    def gated(self, it, pos, work):
        if not isinstance(self.mc, PS.PooledMissControl):                # miss_control._window refuses the pooled allocation
            self.mc = PS.PooledMissControl(self.mc)
        snap = self.snap
        if snap is not None:
            snap.skip_post_reference = (self.skip_post_ref_at == it)
        try:
            return super().gated(it, pos, work)
        finally:
            if snap is not None:
                snap.skip_post_reference = False

    # ------------------------------------------------------------------------------------------------ receipts / ticks
    def receipt(self, k, lg, ref):
        """interference_curve.receipt + the tick's pool actions (device-side, read after the window)."""
        super().receipt(k, lg, ref)
        if self._pacts:
            self.rc_swaps[k] = PS.actions_receipt(self._pacts)

    def _ticks(self, ctx, K_, box=None):
        """THE DECODE THREAD (interference_curve._ticks) + the decode-paced release: with a ReleaseBox, tick k's start event
        (after the restore, before the decode) is published as release k (ext-paced: box=None, it publishes nothing). No sleep, no synchronize, no host read of a device
        value, no wait for the transfer."""
        be, ref, step = self.be_main, ctx["lg_ref"], ctx["step_fn"]
        hook = self.tick_hook
        t0s, t1s, host, idle = [], [], [], []
        prev = None
        for k in range(K_):
            h0 = time.perf_counter()
            if hook is None or not hook(k, "skip_restore"):
                self.restore_nosync()
            if hook is not None:
                hook(k, "pre_decode")
            idle.append(None if prev is None else CV.ev_done(prev))
            a = be.event_rec("main")
            if box is not None:
                box.publish(a)
            lg = step()
            b = be.event_rec("main")
            host.append(1e3 * (time.perf_counter() - h0))
            self.receipt(k, lg, ref)
            t0s.append(a)
            t1s.append(b)
            prev = b
        return t0s, t1s, host, idle

    def _arm_receipts(self, ticks):
        if ticks:
            self.rc_swaps.zero_()
            self._pacts = [e._pool_action for e in self.engines if PS.pool_active(e)]

    def _finish_receipts(self, rec, K_):
        if not K_ or "ticks" not in rec:
            return
        sw = [int(x) for x in self.rc_swaps[:K_].tolist()]
        rec["tick_pool_actions"] = sw
        if any(sw):
            rec["receipts_ok"] = False
            if rec.get("ok"):
                rec["ok"] = False
                self.fails += 1

    def window(self, ctx, arm, train, K_, phase, rep, ticks=True, transfer=True):
        """interference_curve.window + the pool receipts. At a diagnostic cell, a CPU arm's window also gets the diagnostic
        numbers (rec['diag'], bookkeeping only); the HOST-PACK control's window is judged by its own gates (scratch unchanged,
        no H2D / scatter issued, staging pack content) instead of the delivered-content check it cannot pass."""
        hp = bool(transfer and arm and arm.get("hostpack"))
        cpu_diag = bool(transfer and arm and arm.get("kind") == "cpu" and self.is_diag())
        fp0 = None
        if hp:
            self.sync()
            K.poison_(self.scr_k)                                        # the window poisons again: the same bytes
            K.poison_(self.scr_v)
            fp0 = (G.fp_t(self.scr_k), G.fp_t(self.scr_v))
        self._cpu_out = None
        self._arm_receipts(ticks)
        rec = super().window(ctx, arm, train, K_, phase, rep, ticks, transfer)
        self._finish_receipts(rec, K_ if ticks else 0)
        rec.update(regime=("saturation" if transfer else "alone"), C=self.cell_C, P=self.cell_P)
        if hp or cpu_diag:
            out = self._cpu_out or []
            iv = self.coord_intervals(rec.get("requests") or [], out)
            tk = [tuple(x) for x in rec.get("ticks", [])]
            rec["diag"] = CCC.diag_metrics(rec.get("requests") or [], tk, iv, done_label=arm.get("done", "scattered into the scratch"))
            rec["schedule"] = CCC.SCHEDULES["saturation"]
            if hp:
                self.hostpack_gates(rec, arm, train, out, fp0)
        return rec

    def run_cpu_train(self, arm, train, lists):
        out = super().run_cpu_train(arm, train, lists)
        self._cpu_out = out                                              # read by window() after job.done (bookkeeping only)
        return out

    # ------------------------------------------------------------------------------------------------ host-pack gates
    def coord_intervals(self, reqs, raw):
        """The coordinator's host-work intervals (ms on the gate clock) of a CPU window, from its raw records: each request's
        markers are placed relative to its own hk marker, whose gate time is the request's g0 (no gate event needed)."""
        iv = dict(desc=[], pack=[], api=[], release=[])
        g0 = {q["i"]: q.get("g0") for q in reqs}
        for q in raw:
            base = g0.get(q.get("i"))
            if base is None or q.get("hk") is None:
                continue
            at = lambda ev: (None if ev is None else (lambda d: None if d is None else base + d)(CV.ev_ms(q["hk"], ev)))
            de = at(q.get("desc_ev"))
            if de is not None:
                iv["desc"].append((de - q["desc_ns"] / 1e6, de))
            for c in q.get("chunks") or []:
                pk, sub = at(c.get("pk")), at(c.get("sub"))
                if pk is not None:
                    iv["pack"].append((pk - c["pack_ns"] / 1e6, pk))
                    if sub is not None:
                        iv["api"].append((pk, max(pk, sub)))
            mk = at(q.get("rel_issued"))
            if mk is not None and q.get("rel_issue_ns") is not None:
                iv["release"].append((mk - q["rel_issue_ns"] / 1e6, mk))
        return iv

    def scratch_gate(self, fp0):
        """HOST-PACK SCRATCH gate: the destination scratch (the region the transport would write, and every other row) is
        bit-identical before / after the window (fingerprint of the poisoned scratch) and holds no non-poison row."""
        fk, fv = G.fp_t(self.scr_k), G.fp_t(self.scr_v)
        written = int(K.nonpoison_rows(self.scr_k).sum()) + int(K.nonpoison_rows(self.scr_v).sum())
        ok = fp0 is not None and (fk, fv) == tuple(fp0) and written == 0
        return dict(ok=bool(ok), fp_before=list(fp0) if fp0 else None, fp_after=[fk, fv], written_rows=written)

    def staging_check(self, arm, raw):
        """STAGING PACK-CONTENT check: the last chunk packed into each staging slot (still in place after the window) holds the
        LOGICAL source rows of its request's groups (cpupack_core.reference_rows, independent of the address formulas) and the
        descriptor's destination indices. Requests whose last packed chunk survives are 'verified'; the rest are counted."""
        last = {}
        for q in raw:
            for c in q.get("chunks") or []:
                last[c["slot"]] = (q, c)
        cap = arm["chunk_kb"] * 1024 // K.useful_bytes(1)
        bad, chunks, reqs = [], 0, set()
        layout = getattr(self, "host_layout", None) or "orig"
        for s, (q, c) in sorted(last.items()):
            r = self._req_by_i.get(q["i"])
            if r is None:
                bad.append(dict(slot=s, i=q["i"], why="request not in the train"))
                continue
            d = K.build_desc(self.plan_t[r.i][1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout="orig",
                             s_src=self.S_cpu, s_dst=self.S_dst, packer=arm["packer"], placer=arm["placer"])
            g0, g1 = int(c["g0"]), int(c["g1"])
            kst, vst, ist, _, _ = K.stage_views(self.stage[s], g1 - g0, d.packer, d.placer, cap, R=self.R, D=self.D)
            ek = K.reference_rows(K.logical(self.host_phys[r.layer][0], layout), d)[g0 * self.R:g1 * self.R]
            ev = K.reference_rows(K.logical(self.host_phys[r.layer][1], layout), d)[g0 * self.R:g1 * self.R]
            okk = K.bits_equal(kst.reshape(-1, self.D), ek.reshape(-1, self.D).to(kst.device))
            okv = K.bits_equal(vst.reshape(-1, self.D), ev.reshape(-1, self.D).to(vst.device))
            oki = bool(torch.equal(ist, d.dst_idx[g0 * d.dst_per_group:g1 * d.dst_per_group].to(ist.device)))
            chunks += 1
            reqs.add(r.i)
            if not (okk and okv and oki):
                bad.append(dict(slot=s, i=r.i, layer=r.layer, groups=[g0, g1], k=okk, v=okv, idx=oki))
        return dict(ok=(not bad and chunks > 0) or (not last and not any(r.groups for r in self._req_by_i.values())), slots_checked=chunks,
                    requests_verified=len(reqs), requests_total=len(raw), bad=bad[:8])

    def hostpack_gates(self, rec, arm, train, raw, fp0, base_ok=None, count=True):
        """The host-pack window's own verdict (replaces content_ok, which needs a delivery): no error, every request done and
        charged, every list delivered, the tick receipts (base_ok: the ext window's own checks, which include these), AND the
        scratch gate, NO H2D and NO scatter issued (counted at the branch: the pipe's chunk records carry no H2D / scatter
        event), the staging pack-content check."""
        was = bool(rec.get("ok"))
        self._req_by_i = {r.i: r for r in train}
        sc = self.scratch_gate(fp0)
        h2d = sum(1 for q in raw for c in q.get("chunks") or [] if c.get("h2d1") is not None or c.get("h2d0") is not None)
        scat = sum(1 for q in raw for c in q.get("chunks") or [] if c.get("sc1") is not None or c.get("sc0") is not None)
        pk = self.staging_check(arm, raw)
        rec.update(scratch_gate=sc, h2d_issued=h2d, scatter_issued=scat, pack_check=pk, content_ok=None, done=CCC.HOSTPACK_DONE_LABEL,
                   content_check="n/a for the host-pack control (nothing is delivered): scratch gate + staging pack-content check instead")
        if base_ok is None:
            ok = "error" not in rec and "coord_error" not in rec and bool(rec.get("all_requests_done")) and bool(rec.get("bytes_charged"))
            ok &= bool(rec.get("list_ok"))
            if rec.get("with_decode"):
                ok &= bool(rec.get("receipts_ok"))
        else:
            ok = bool(base_ok)
        ok &= sc["ok"] and pk["ok"] and h2d == 0 and scat == 0
        rec["hostpack_ok"] = bool(ok)
        rec["ok"] = bool(ok)
        if count:
            self.fails += int(not ok) - int(not was)
        return rec

    # ------------------------------------------------------------------------------------------------ the paced regime
    def paced_train_for(self, it):
        if it + PACED_TICKS > self.masks.shape[0]:
            raise RuntimeError("paced train %d..%d beyond the captured %d steps" % (it, it + PACED_TICKS - 1, self.masks.shape[0]))
        tr = CCC.paced_train(self.groups_of, it, PACED_TICKS, self.NL, IC.REGIONS)
        self.curve.setdefault("ptrains", {})[it] = dict(steps=[it, it + PACED_TICKS - 1], releases=PACED_TICKS, digest=CV.train_digest(tr), **CV.train_bytes(tr))
        return tr

    def run_pump_paced(self, arm, train, gate, box, it):
        be = self.be_pump
        be.reset()
        return CCC.pump_paced(be, train, self.launcher(arm), gate, IC.QUEUE, box, it)

    def run_cpu_paced(self, arm, train, gate, box, it, ext=False):
        """cpu8 on the COORDINATOR thread: per release, the plan stream waits for the release event, the 32 list D2Hs follow
        (charged); per request: host-wait its list, descriptors LIVE, the pipe (pack -> H2D -> scatter; the host-pack control:
        pack only, arm['mode']). ext=True (the ext-paced regime) also records, per release, a plan-stream event after the wait
        (the list D2H spans are then consecutive plan-stream events), the host time of the release's issue (plan-stream wait +
        32 D2H enqueues) and an aux marker after it."""
        be, pb = self.cbe, getattr(self, "be_plan", None) or self.cbe
        be.reset()
        if pb is not be:
            pb.reset()
        pipe = self.pipe(arm["chunk_kb"] * 1024 // K.useful_bytes(1))
        pipe.reset()
        mode = arm.get("mode", "full")
        by_rel = {}
        for r in train:
            by_rel.setdefault(CCC.release_index(r, it), []).append(r)
        out, gated = [], False
        for j in sorted(by_rel):
            ev, wns = box.wait(j)
            lists = {}
            if ext:
                t_iss = time.perf_counter_ns()
            with pb.stream("plan"):
                if not gated:
                    pb.stream_wait("plan", gate)
                    gated = True
                pb.stream_wait("plan", ev)
                l0 = pb.event_rec("plan") if ext else None
                for r in by_rel[j]:
                    pb.memcpy(self.plan_t[r.i], self.store[r.step, r.layer], "plan")
                    lists[r.i] = pb.event_rec("plan")
            if ext:
                iss_ns = time.perf_counter_ns() - t_iss
                iss_mk = be.marker()
            first = True
            for r in by_rel[j]:
                t = time.perf_counter_ns()
                be.host_wait(lists[r.i])
                rec = dict(i=r.i, release=j, release_wait_ns=(wns if first else 0), wait_ns=time.perf_counter_ns() - t, hk=be.marker(), list_ev=lists[r.i])
                if ext and first:
                    rec.update(rel_l0=l0, rel_issue_ns=iss_ns, rel_issued=iss_mk)
                first = False
                t = time.perf_counter_ns()
                d = K.build_desc(self.plan_t[r.i][1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout="orig",
                                 s_src=self.S_cpu, s_dst=self.S_dst, packer=arm["packer"], placer=arm["placer"])
                rec["desc_ns"] = time.perf_counter_ns() - t
                rec["desc_ev"] = be.marker()
                rec["n"] = d.n
                sk, dk = K.views_for(d, self.host_phys[r.layer][0], self.scr_k)
                sv, dv = K.views_for(d, self.host_phys[r.layer][1], self.scr_v)
                rec["chunks"] = pipe.layer(d, sk, sv, dk, dv, mode=mode)
                out.append(rec)
        return out

    def paced_window(self, ctx, arm, train, K_p, phase, rep):
        """One DECODE-PACED window (raw key 'paced'; ccurve_core module docstring): K_p ticks, release k at THIS arm's own tick k
        start (a method-dependent schedule, NOT a matched load), the transport on the coordinator. Same checks as the saturation
        window + releases complete."""
        it = ctx["it"]
        kind = arm["kind"]
        rec = dict(stage="PACED", regime="paced", batch=self.B, C=self.cell_C, P=self.cell_P, step=it, plan_step=it, arm=arm["name"], kind=kind,
                   phase=phase, rep=rep, K=K_p, with_decode=True, with_train=True, queue=IC.QUEUE, regions=IC.REGIONS, label=K.LABEL,
                   replay_label=CCC.PACED_LABEL, schedule=CCC.SCHEDULES["paced"], saturated_label=None, copy_label=CV.PREBUILT_LABEL,
                   arm_label=arm.get("label"), W=arm.get("W"), train=self.curve.setdefault("ptrains", {}).get(it))
        self.sync()
        K.poison_(self.scr_k)
        K.poison_(self.scr_v)
        want = arm.get("cores") or 1
        if hasattr(self, "coord") and getattr(self.coord, "n", None) != want:
            self.coord.configure(want)
        if kind == "cpu":
            self.plan_t.fill_(-7)
        self.rc_loads.zero_()
        self.rc_eq.zero_()
        self._masks = [e._load_mask for e in self.engines]
        self._arm_receipts(True)
        retries0 = IC._alloc_retries()
        self.be_main.reset()
        self.sync()
        gate = self.be_main.event_rec("main")
        box = CCC.ReleaseBox()
        if kind == "cpu":
            job = self.coord.submit(lambda co, a=arm, t=train, g=gate, bx=box: self.run_cpu_paced(a, t, g, bx, it))
        else:
            job = self.coord.submit(lambda co, a=arm, t=train, g=gate, bx=box: self.run_pump_paced(a, t, g, bx, it))
        tk, err = None, None
        try:
            tk = self._ticks(ctx, K_p, box)
        except (G.UnsanctionedDecode, torch.cuda.OutOfMemoryError, CT.MemGate):
            box.close()
            job.done.wait()
            self.sync()
            raise
        except Exception:
            err = traceback.format_exc()[-3000:]
        finally:
            box.close()                                                  # a release never published fails the transport, never hangs it
        end = self.be_main.event_rec("main")
        job.done.wait()
        self.sync()
        if isinstance(job.exc, (torch.cuda.OutOfMemoryError, CT.MemGate, MemoryError)):
            raise job.exc
        rec["alloc_retries"] = IC._alloc_retries() - retries0
        if err:
            rec["error"] = err
        if job.error:
            rec["coord_error"] = job.error[-3000:]
        ms = lambda e: CV.ev_ms(gate, e)
        rec["end_ms"] = ms(end)
        rel_ms = [ms(e) for e in box.evs]
        rec["release_ms"] = rel_ms
        reqs = []
        if not job.error:
            if kind == "cpu":
                by = {r.i: r for r in train}
                for q in job.result:
                    r = by[q["i"]]
                    ch = [[ms(c.get("pk")), ms(c.get("h2d0")), ms(c.get("h2d1")), ms(c.get("sc0")), ms(c.get("sc1")), c["bytes"],
                           c["pack_ns"] / 1e6, c["bp_ns"] / 1e6, c["g1"] - c["g0"]] for c in q["chunks"]]
                    g1 = max((c[4] for c in ch if c[4] is not None), default=ms(q["desc_ev"]))
                    reqs.append(dict(i=r.i, step=r.step, layer=r.layer, groups=r.groups, useful=r.useful, wire=sum(c[5] for c in ch), region=r.region,
                                     release=q["release"], release_wait_ms=q["release_wait_ns"] / 1e6, release_wait_ns=q["release_wait_ns"],
                                     issue=ms(q["list_ev"]), g0=ms(q["hk"]), g1=g1, wait_ms=q["wait_ns"] / 1e6, desc_ms=q["desc_ns"] / 1e6,
                                     waited_for=None, inflight_at_issue=None, chunks=ch))
            else:
                for q in job.result:
                    reqs.append(dict(i=q["i"], step=q["step"], layer=q["layer"], groups=q["groups"], useful=q["useful"], wire=q["wire"], region=q["region"],
                                     release=q["release"], release_wait_ms=q["release_wait_ns"] / 1e6, release_wait_ns=q["release_wait_ns"],
                                     issue=ms(q["issue"]), g0=ms(q["g0"]), g1=ms(q["g1"]), wait_ms=q["wait_ns"] / 1e6, api_ms=q["api_ns"] / 1e6,
                                     waited_for=q["waited_for"], inflight_at_issue=q["inflight_at_issue"], dep=q["dep"]))
        rec["requests"] = reqs
        if tk is not None:
            t0s, t1s, host, idle = tk
            rec["ticks"] = [[ms(a), ms(b)] for a, b in zip(t0s, t1s)]
            rec["tick_host_ms"] = host
            rec["main_idle_at_submit"] = idle
            rec["tick_loads"] = [int(x) for x in self.rc_loads[:K_p].tolist()]
            rec["tick_bit_exact"] = [bool(x) for x in self.rc_eq[:K_p].tolist()]
        ticks_ms = [tuple(x) for x in rec.get("ticks", [])]
        m = CV.window_metrics(ticks_ms, reqs, IC.COVER_MIN, IC.SKIP_FIRST) if ticks_ms else CV.transfer_metrics(reqs)
        m["paced"] = CCC.paced_metrics(rel_ms, reqs, ticks_ms)
        rec["m"] = m
        ok = "error" not in rec and "coord_error" not in rec
        ref = self.refs.get(CV.train_digest(train))
        fk, fv = G.fp_t(self.scr_k), G.fp_t(self.scr_v)
        rec["content_ok"] = bool(ref is not None and fk == ref["fp_k"] and fv == ref["fp_v"])
        rec["all_requests_done"] = len(reqs) == len(train)
        rec["bytes_charged"] = sum(q["useful"] for q in reqs) == sum(r.useful for r in train)
        rec["releases_ok"] = len(rel_ms) == K_p and bool(m["paced"]["all_released"])
        ok &= rec["content_ok"] and rec["all_requests_done"] and rec["bytes_charged"] and rec["releases_ok"]
        if kind == "cpu":
            rows = torch.stack([self.store[r.step, r.layer].cpu() for r in train]) if self.dev == "cuda" else torch.stack([self.store[r.step, r.layer] for r in train])
            rec["list_ok"] = bool(torch.equal(self.plan_t[:len(train)], rows))
            ok &= rec["list_ok"]
        else:
            mx = max((q["inflight_at_issue"] for q in reqs), default=0)
            rec["queue_ok"] = mx < IC.QUEUE
            ok &= rec["queue_ok"]
        rec["ticks_complete"] = len(rec.get("ticks", [])) == K_p
        rec["receipts_ok"] = bool(rec["ticks_complete"] and all(x == 0 for x in rec.get("tick_loads", [1])) and all(rec.get("tick_bit_exact", [False])))
        ok &= rec["receipts_ok"]
        rec["ok"] = bool(ok)
        if self.dev == "cuda":
            free, total = torch.cuda.mem_get_info()
            rec["device_used_gb"] = (total - free) / 1e9
            rec["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        self.windows.append(rec)
        self.fails += int(not ok)
        self._finish_receipts(rec, K_p)
        return rec

    # ------------------------------------------------------------------------------------------------ the ext-paced regime
    def ext_schedule_for(self, it, pre, ptrain):
        """The step's PRERECORDED schedule (ccurve_core.ext_schedule), FIXED before any arm of the step runs: P = the steady
        decode-alone tick p50 of this step's decode_alone_pre window (its start-to-start period p50 is recorded beside it),
        T0 = CC_EXT_T0_MS, R = CC_PACED_TICKS releases of the paced train's trace steps. None (recorded) when the pre window
        cannot give P (not ok, no steady tick)."""
        tk = [tuple(x) for x in pre.get("ticks", [])]
        P = CV.pctl(CV.steady_ticks(tk, IC.SKIP_FIRST), 50)
        starts = [a for a, _ in tk]
        period = CV.pctl([starts[k + 1] - starts[k] for k in range(IC.SKIP_FIRST, len(starts) - 1)], 50)
        rec = self.curve.setdefault("ext_sched", {})
        if not pre.get("ok") or not (P == P and P > 0):
            rec[it] = dict(skipped=True, why="decode_alone_pre not ok or no steady tick (P=%s)" % P)
            return None
        steps = list(range(it, it + PACED_TICKS))
        per = [sum(r.useful for r in ptrain if r.step == st) for st in steps]
        sched = CCC.ext_schedule(per, steps, P, EXT_T0_MS, PACED_TICKS)
        sched.update(P_source="decode_alone_pre steady tick p50 (step %d)" % it, alone_period_p50_ms=period, late_tol_ms=EXT_LATE_TOL_MS)
        rec[it] = {k: v for k, v in sched.items() if k != "offsets_ns"}
        return sched

    def release_backend(self):
        be = getattr(self, "be_rel", None)
        if be is None:
            if self.dev == "cuda":                                       # the release thread's OWN stream (never a host-only twin)
                self.s_rel, self.aux_rel = torch.cuda.Stream(), torch.cuda.Stream()
                be = self.be_rel = K.CudaBackend(dict(release=self.s_rel), self.aux_rel, pool_size=PACED_TICKS + 64)
            else:                                                        # CPU tests: an idle-GPU twin
                be = self.be_rel = CV.ImmediateBackend()
        return be

    def ext_window(self, ctx, arm, train, sched, phase, rep):
        """One EXT-PACED window (ccurve_core module docstring): the release thread issues the step's fixed schedule; the decode
        thread runs sched['ticks'] back-to-back resident ticks and publishes nothing; the transport (the coordinator) consumes
        the releases exactly as in decode-paced. Gates: the paced window's (receipts, every request done and charged, lists,
        releases) + the SCHEDULE gate + content (cpu8: the delivered scratch == the reference; the host-pack control: its own
        gates) + the staging pack-content check for every CPU arm."""
        it = ctx["it"]
        kind, hp = arm["kind"], bool(arm.get("hostpack"))
        K_t, R = int(sched["ticks"]), int(sched["R"])
        rec = dict(stage="EXT", regime="ext-paced", batch=self.B, C=self.cell_C, P=self.cell_P, step=it, plan_step=it, arm=arm["name"], kind=kind,
                   phase=phase, rep=rep, K=K_t, with_decode=True, with_train=True, queue=IC.QUEUE, regions=IC.REGIONS, label=K.LABEL,
                   replay_label=CCC.EXT_LABEL, schedule=CCC.SCHEDULES["ext-paced"], saturated_label=None, copy_label=CV.PREBUILT_LABEL,
                   arm_label=arm.get("label"), W=arm.get("W"), hostpack=hp, done=arm.get("done", "scattered into the scratch" if kind == "cpu" else "gathered"),
                   train=self.curve.setdefault("ptrains", {}).get(it),
                   sched={k: sched[k] for k in ("digest", "R", "P_ms", "T0_ms", "ticks", "bytes_total", "offered_gbps", "scheduled_ms")})
        self.sync()
        K.poison_(self.scr_k)
        K.poison_(self.scr_v)
        fp0 = (G.fp_t(self.scr_k), G.fp_t(self.scr_v)) if hp else None
        want = arm.get("cores") or 1
        if hasattr(self, "coord") and getattr(self.coord, "n", None) != want:
            self.coord.configure(want)
        if kind == "cpu":
            self.plan_t.fill_(-7)
        self.rc_loads.zero_()
        self.rc_eq.zero_()
        self._masks = [e._load_mask for e in self.engines]
        self._arm_receipts(True)
        retries0 = IC._alloc_retries()
        self.be_main.reset()
        be_rel = self.release_backend()
        be_rel.reset()
        self.sync()
        gate = self.be_main.event_rec("main")
        anchor = time.perf_counter_ns()                                  # the GPU is idle (synchronized): gate ~ anchor
        box = CCC.ReleaseBox()
        if kind == "cpu":
            job = self.coord.submit(lambda co, a=arm, t=train, g=gate, bx=box: self.run_cpu_paced(a, t, g, bx, it, ext=True))
        else:
            job = self.coord.submit(lambda co, a=arm, t=train, g=gate, bx=box: self.run_pump_paced(a, t, g, bx, it))
        rt = CCC.ReleaseThread(box, be_rel, anchor, sched["offsets_ns"], late_ns=self.ext_late_ns)
        rt.start()
        tk, err = None, None
        try:
            tk = self._ticks(ctx, K_t)                                   # the decode thread: NO box, publishes nothing
        except (G.UnsanctionedDecode, torch.cuda.OutOfMemoryError, CT.MemGate):
            rt.halt()
            rt.join()
            box.close()
            job.done.wait()
            self.sync()
            raise
        except Exception:
            err = traceback.format_exc()[-3000:]
        end = self.be_main.event_rec("main")
        if err:
            rt.halt()
        left = (anchor + sched["offsets_ns"][-1] + sum((self.ext_late_ns or {}).values()) - time.perf_counter_ns()) / 1e9
        rt.join(timeout=max(0.0, left) + 30.0)
        if rt.is_alive():
            rt.halt()
            rt.join()
            err = (err or "") + "release thread still running after its schedule + 30 s"
        box.close()                                                      # a release never published fails the transport, never hangs it
        job.done.wait()
        self.sync()
        if isinstance(job.exc, (torch.cuda.OutOfMemoryError, CT.MemGate, MemoryError)):
            raise job.exc
        rec["alloc_retries"] = IC._alloc_retries() - retries0
        if err:
            rec["error"] = err
        if job.error:
            rec["coord_error"] = job.error[-3000:]
        if rt.error:
            rec["release_error"] = rt.error
        ms = lambda e: CV.ev_ms(gate, e)
        rec["end_ms"] = ms(end)
        rel = [dict(k=r["k"], scheduled_ms=r["scheduled_ns"] / 1e6, issue_ms=r["issue_ns"] / 1e6, publish_ms=r["publish_ns"] / 1e6,
                    gpu_ms=ms(r["ev"]), lateness_ms=r["lateness_ns"] / 1e6, publish_cost_ms=r["publish_cost_ns"] / 1e6,
                    injected_delay_ms=r["injected_delay_ns"] / 1e6, lateness_ns=r["lateness_ns"], publish_cost_ns=r["publish_cost_ns"]) for r in rt.recs]
        rec["releases"] = rel
        rel_ms = [r["gpu_ms"] for r in rel]
        rec["release_ms"] = rel_ms
        rec["release_publishers"] = sorted(set(box.who))
        reqs, raw = [], []
        if not job.error:
            raw = job.result
            if kind == "cpu":
                scat = K.MODES[arm.get("mode", "full")][2]
                by = {r.i: r for r in train}
                prev = {}
                for q in raw:
                    r = by[q["i"]]
                    j = q["release"]
                    ch = [[ms(c.get("pk")), ms(c.get("h2d0")), ms(c.get("h2d1")), ms(c.get("sc0")), ms(c.get("sc1")), c["bytes"],
                           c["pack_ns"] / 1e6, c["bp_ns"] / 1e6, c["g1"] - c["g0"]] for c in q["chunks"]]
                    if scat:
                        g1 = max((c[4] for c in ch if c[4] is not None), default=ms(q["desc_ev"]))
                    else:
                        g1 = max((c[0] for c in ch if c[0] is not None), default=ms(q["desc_ev"]))
                    lm = ms(q["list_ev"])
                    if q.get("rel_l0") is not None:
                        prev[j] = ms(q["rel_l0"])
                    d2h = (lm - prev[j]) if (j in prev and lm is not None and prev[j] is not None) else None
                    prev[j] = lm
                    x = dict(i=r.i, step=r.step, layer=r.layer, groups=r.groups, useful=r.useful, wire=sum(c[5] for c in ch), region=r.region,
                             release=j, release_wait_ms=q["release_wait_ns"] / 1e6, release_wait_ns=q["release_wait_ns"], issue=lm, g0=ms(q["hk"]), g1=g1,
                             wait_ms=q["wait_ns"] / 1e6, desc_ms=q["desc_ns"] / 1e6, d2h_ms=d2h, list_ms=lm, waited_for=None, inflight_at_issue=None, chunks=ch)
                    if q.get("rel_issue_ns") is not None:
                        x.update(rel_issue_ms=q["rel_issue_ns"] / 1e6, rel_l0_ms=ms(q["rel_l0"]), rel_issued_ms=ms(q["rel_issued"]))
                    reqs.append(x)
            else:
                for q in raw:
                    reqs.append(dict(i=q["i"], step=q["step"], layer=q["layer"], groups=q["groups"], useful=q["useful"], wire=q["wire"], region=q["region"],
                                     release=q["release"], release_wait_ms=q["release_wait_ns"] / 1e6, release_wait_ns=q["release_wait_ns"],
                                     issue=ms(q["issue"]), g0=ms(q["g0"]), g1=ms(q["g1"]), wait_ms=q["wait_ns"] / 1e6, api_ms=q["api_ns"] / 1e6,
                                     waited_for=q["waited_for"], inflight_at_issue=q["inflight_at_issue"], dep=q["dep"]))
        rec["requests"] = reqs
        if tk is not None:
            t0s, t1s, host, idle = tk
            rec["ticks"] = [[ms(a), ms(b)] for a, b in zip(t0s, t1s)]
            rec["tick_host_ms"] = host
            rec["main_idle_at_submit"] = idle
            rec["tick_loads"] = [int(x) for x in self.rc_loads[:K_t].tolist()]
            rec["tick_bit_exact"] = [bool(x) for x in self.rc_eq[:K_t].tolist()]
        ticks_ms = [tuple(x) for x in rec.get("ticks", [])]
        m = CV.window_metrics(ticks_ms, reqs, IC.COVER_MIN, IC.SKIP_FIRST) if ticks_ms else CV.transfer_metrics(reqs)
        m["paced"] = CCC.paced_metrics(rel_ms, reqs, ticks_ms)
        iv = self.coord_intervals(reqs, raw) if kind == "cpu" else {}
        m["ext"] = CCC.diag_metrics(reqs, ticks_ms, iv, rel_ms=rel_ms, rel=rel, sched=sched, done_label=rec["done"], late_tol_ms=EXT_LATE_TOL_MS)
        rec["m"] = m
        rec["diag"] = m["ext"]
        ref_sched = self.curve.get("ext_sched", {}).get(it) or {}
        rec["schedule_gate"] = CCC.schedule_gate(sched, ref_sched.get("digest"), rel, len(box.evs), box.who, rt.error)
        ok = "error" not in rec and "coord_error" not in rec and "release_error" not in rec
        rec["all_requests_done"] = len(reqs) == len(train)
        rec["bytes_charged"] = sum(q["useful"] for q in reqs) == sum(r.useful for r in train)
        rec["releases_ok"] = len(rel_ms) == R and all(x is not None for x in rel_ms) and bool(m["paced"]["all_released"])
        ok &= rec["all_requests_done"] and rec["bytes_charged"] and rec["releases_ok"] and rec["schedule_gate"]["ok"]
        if kind == "cpu":
            rows = torch.stack([self.store[r.step, r.layer].cpu() for r in train]) if self.dev == "cuda" else torch.stack([self.store[r.step, r.layer] for r in train])
            rec["list_ok"] = bool(torch.equal(self.plan_t[:len(train)], rows))
            ok &= rec["list_ok"]
        else:
            mx = max((q["inflight_at_issue"] for q in reqs), default=0)
            rec["queue_ok"] = mx < IC.QUEUE
            ok &= rec["queue_ok"]
        rec["ticks_complete"] = len(rec.get("ticks", [])) == K_t
        rec["receipts_ok"] = bool(rec["ticks_complete"] and all(x == 0 for x in rec.get("tick_loads", [1])) and all(rec.get("tick_bit_exact", [False])))
        ok &= rec["receipts_ok"]
        if hp:
            self.hostpack_gates(rec, arm, train, raw, fp0, base_ok=ok, count=False)
        else:
            ref = self.refs.get(CV.train_digest(train))
            fk, fv = G.fp_t(self.scr_k), G.fp_t(self.scr_v)
            rec["content_ok"] = bool(ref is not None and fk == ref["fp_k"] and fv == ref["fp_v"])
            ok &= rec["content_ok"]
            if kind == "cpu":
                self._req_by_i = {r.i: r for r in train}
                rec["pack_check"] = self.staging_check(arm, raw)
                ok &= rec["pack_check"]["ok"]
            rec["ok"] = bool(ok)
        if self.dev == "cuda":
            free, total = torch.cuda.mem_get_info()
            rec["device_used_gb"] = (total - free) / 1e9
            rec["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        self.windows.append(rec)
        self.fails += int(not rec["ok"])
        self._finish_receipts(rec, K_t)
        return rec

    # ------------------------------------------------------------------------------------------------ one gated step
    def curve_step(self, ctx):
        """decode-alone pre -> saturation (every arm; + the host-pack control at a diagnostic cell) -> decode-paced (the base
        arms) -> ext-paced (the diagnostic arms, at a diagnostic cell) -> decode-alone post. The ext schedule is fixed right after
        decode-alone pre, before any arm of the step runs. Outside the diagnostic cells nothing differs from CC1."""
        it = ctx["it"]
        train = self.train_for(it)
        self.train_reference(train)
        if self.curve["K"] is None:
            self.calibrate(ctx, train)
        K_ = self.curve["K"]["K"]
        K_da = max(K_, PACED_TICKS if PACED else 0)
        pre = self.window(ctx, None, train, K_da, "decode_alone_pre", 0, ticks=True, transfer=False)
        self.note_restore(it)
        diag = self.is_diag()
        self.mark_diag()
        ptrain, sched = None, None
        if diag and self.ext_arms():
            ptrain = self.paced_train_for(it)
            sched = self.ext_schedule_for(it, pre, ptrain)               # FIXED here, before any arm of the step
        sat = list(self.arms) + (self.hostpack_arms() if diag else [])
        order = sat if it % 2 == 0 else sat[::-1]
        for rep in range(IC.REPS):
            for a in order:
                self.window(ctx, a, train, K_, "transfer_alone", rep, ticks=False, transfer=True)
                self.window(ctx, a, train, K_, "overlap", rep, ticks=True, transfer=True)
        if PACED:
            ptrain = ptrain or self.paced_train_for(it)
            self.train_reference(ptrain)
            base = list(self.arms) if it % 2 == 0 else list(self.arms)[::-1]
            for rep in range(IC.REPS):
                for a in base:
                    self.paced_window(ctx, a, ptrain, PACED_TICKS, "paced", rep)
        if sched is not None:
            self.train_reference(ptrain)
            ea = self.ext_arms()
            eorder = ea if it % 2 == 0 else ea[::-1]
            for rep in range(IC.REPS):
                for a in eorder:
                    self.ext_window(ctx, a, ptrain, sched, "ext_paced", rep)
        self.window(ctx, None, train, K_da, "decode_alone_post", 0, ticks=True, transfer=False)
        self.curve["steps_done"].append(it)

    def note_restore(self, it):
        """The bytes of the sanctioned per-tick decode-state restore at this step (snapshot tensors; labelled in the table)."""
        try:
            b = snapshot_bytes(self.snap)
        except Exception as e:
            b = dict(error=repr(e)[:200])
        self.cell.setdefault("restore_per_tick", {})[int(it)] = b

    # ------------------------------------------------------------------------------------------------ capture at C
    def capture_pool(self, digest_steps=()):
        """cpupack_transport.Runner.capture at capacity C: the int16 AloneTrace + the pool archives (pool_state natural
        accounting, pool-aware invariants); the GPU-resident plan store holds the POST-pool H2D masks = the natural misses at
        C (what both methods replay)."""
        from nosi import transfer_trace as _tt
        warm = []
        AloneTrace, _ = VA.make_trace_classes()
        T16 = PS.make_pool_trace_class(PL.make_int16_trace_class(AloneTrace), self.engines)
        ncap = VA.N
        tr = T16(self.NL, "1", max_steps=ncap)
        _tt.TRACE = tr
        pos = self.pos0.clone()
        sha, loaded, xfail = [], [], 0
        t0 = time.time()
        try:
            for it in range(ncap):
                lg = self.decode(self.forced[:, it:it + 1], pos, warmup=(it == 0))
                torch.cuda.synchronize()
                sha.append(VA.sha(lg))
                loaded.append(self.loaded_now())
                for l, e in enumerate(self.engines):
                    xfail += int(not torch.equal(tr.mask_archive[it, l].to(torch.int64), e._load_mask))
                    xfail += int(not torch.equal(tr.map_archive[it, l].to(torch.int64), e._block_map))
                    if PS.pool_active(e):
                        xfail += int(not torch.equal(tr.pool_archive[it, l], e._pool_action))
                        xfail += int(not torch.equal(tr.pmap_archive[it, l].to(torch.int64), e._pool_map))
                if it in digest_steps:
                    warm.append(dict(step=it, logits_sha=sha[-1], logits_sha256=G.sha_parts([lg]), digest=self.state_digest(G.FAMILIES_PRE)))
                pos = pos + 1
        finally:
            _tt.TRACE = None
        step_rows, _, _, masks, maps = tr.harvest(timed_steps=())
        ok_trace = tr.dropped_steps == 0 and len(tr.steps) == ncap
        has_pool = PS.pool_active(self.engines[0])
        acts_d = tr.pool_archive[:ncap] if has_pool else None
        pmaps_d = tr.pmap_archive[:ncap] if has_pool else None
        acc = PS.natural_accounting(tr.mask_archive[:ncap], tr.map_archive[:ncap], acts_d, pmaps_d, tail_id=VA.L // self.R)
        self.store = torch.empty((ncap, self.NL, self.W), dtype=torch.int32, device="cuda")
        for s in range(ncap):
            m32 = tr.mask_archive[s].to(torch.int32)
            self.store[s, :, 1 + self.HB:] = m32.reshape(self.NL, -1)
            self.store[s, :, 1:1 + self.HB] = (m32[..., :K.TAIL_SLOT] >= 0).sum(-1).to(torch.int32).reshape(self.NL, -1)
            self.store[s, :, 0] = 0
        self.acts = acts_d.cpu() if has_pool else None
        self.pool_maps = pmaps_d.to(torch.int16).cpu() if has_pool else None
        del tr
        self.masks, self.maps = masks.to(torch.int16), maps.to(torch.int16)
        self.logits_sha, self.step_loaded = sha, loaded
        inv = PL.invariants(self.masks, self.maps, L_prompt=VA.L, step_loaded=loaded)
        if has_pool:                                                    # I5 / I6 do not hold with a pool (a hit changes a slot
            inv = {k: v for k, v in inv.items() if not k.startswith(("I5", "I6"))}   # without a load); P1..P5 replace them
        inv.update(acc["summary"]["invariants"])
        inv["I1p_pool_unwritten"] = PS.unwritten(self.acts, self.pool_maps) if has_pool else 0
        meta = dict(batch=self.B, L=VA.L, ncap=ncap, docs=self.docs, ids_sha=VA.sha(self.ids.to(torch.float32)), nosi_commit=os.environ.get("NOSI_COMMIT"),
                    attn_splits=os.environ.get("NOSI_ATTN_SPLITS"), pool_blocks=str(self.cell_P), capacity=self.cell_C, round_slots=os.environ.get("NOSI_VERIFY_ROUND_SLOTS"))
        path = os.path.join(OUT, "%s_plans.npz" % CT.TAG)
        hs = PL.export_npz(path, self.masks, self.maps, meta, loaded, sha)
        if has_pool:
            np.savez(os.path.join(OUT, "%s_pool.npz" % CT.TAG), acts=self.acts.numpy(), pool_maps=self.pool_maps.numpy())
        self.accounting = dict(summary=acc["summary"], per_step={k: v.sum(dim=1).tolist() for k, v in acc["per"].items()},
                               steady=PS.steady_summary(acc, replayed_steps(ncap)),               # the steps the windows REPLAY
                               all_steps=PS.steady_summary(acc, list(range(1, ncap))))           # incl. the cold-pool start
        cap = dict(source="capture at C=%d (int16 AloneTrace + pool archives, one harvest)" % self.cell_C, ncap=ncap, seconds=time.time() - t0,
                   cross_check_fails=xfail, trace_ok=ok_trace, invariants=inv, accepted=PL.accepted(inv) and xfail == 0 and ok_trace,
                   run_digest=hs["run_digest"], step_digest=hs["step_digest"], export=path, step_loaded=loaded, logits_sha=sha,
                   natural_step_ms=[r.get("step_ms") for r in step_rows], warm=warm)
        gc.collect()
        torch.cuda.empty_cache()
        self.model.has_buffers = False
        return cap

    # ------------------------------------------------------------------------------------------------ cross-C
    def cross_record(self):
        steps = self.golden.get("steps", {})
        return dict(capture_sha=[str(x) for x in (self.logits_sha or [])], ref_sha={int(s): v.get("ref_sha") for s, v in steps.items()},
                    adv_sha={int(s): v.get("adv_sha") for s, v in steps.items()})

    def cross_now(self):
        cells = dict(load_cross(OUT, self.B))
        cells.update(self.cross)
        return CCC.cross_c_check(cells, REF_C)

    # ------------------------------------------------------------------------------------------------ one cell
    @torch.inference_mode()
    def run_cell(self, C, first):
        if getattr(self, "mc", None) is None:
            from nosi.verify import miss_control as mc
            self.mc = mc
        P = PS.pool_of(C)
        if first:
            self.setup_early()
            self.setup_model()
            self.setup_curve()
            try:
                pps = self.ss.PostPrefillSnapshot(self.cache).take()
            except torch.cuda.OutOfMemoryError as e:
                self.stop_class = "GPU_OOM"
                raise CT.MemGate("restart snapshot: CUDA out of memory: %s" % str(e)[:200])
            self.d0 = self.start_digest()
            self.pps = pps
            self.restart_snapshot = dict(hosted_bytes=G.snapshot_offload(pps), start_digest_families=G.FAMILIES_START, P_prefill=PS.pool_blocks(self.engines[0]))
            gc.collect()
            torch.cuda.empty_cache()
        self.payload_extra["restart_snapshot"] = getattr(self, "restart_snapshot", None)
        self.cell_C, self.cell_P = int(C), P
        if first and PS.pool_blocks(self.engines[0]) == P:
            fr = [PS.pool_is_fresh(e) for e in self.engines]
            self.restarts.append(dict(label="post-prefill (no restart)", ok=all(f["ok"] for f in fr), P=P, pool_fresh_bad_layers=[i for i, f in enumerate(fr) if not f["ok"]]))
            if not all(f["ok"] for f in fr):
                raise CT._ProofFail("the prefilled pool is not fresh")
        else:
            self.restart_to(P, "cell C%d start" % C)
        self.S_dst = int(self.engines[0]._k_gpu.shape[1])
        self.flush_payload(True)
        # 1. the natural plans at C
        cap = self.capture_pool(digest_steps=range(VA.WARM))
        ref_warm = cap["warm"]
        self.capture_rec = dict(cap, warm=[{k: v for k, v in x.items() if k != "digest"} for x in cap["warm"]])
        self.cross[C] = self.cross_record()
        if not cap["accepted"]:
            self.fails += 1
            self.flush_payload(False)
            return CT.RC_CORRECT
        sv, sh = saved_for(self.B)
        if sv and C == REF_C:
            self.plans_vs_saved = self.verify_saved_prefix(sv, sh)
            if not self.plans_vs_saved["ok"]:
                self.fails += 1
                self.flush_payload(False)
                return CT.RC_CORRECT
        self.restart(self.pps, "C%d after capture" % C)
        self.flush_payload(True)
        # 2. the golden trajectory at C
        self.schedule = (list(IC.STEPS), [])
        self.golden = dict(source="in-process golden pass (option (i)) at C=%d" % C, meta=self.golden_meta(), warm=None, steps={})
        self.golden_mode = "record"
        pos, warm, ok = self.warm_pass("golden", ref=ref_warm)
        if not ok:
            raise CT._ProofFail("the golden pass's warm steps differ from the capture at C=%d" % C)
        self.golden["warm"] = warm
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        self.gated_sequence(pos)
        gpath = os.path.join(OUT, "%s_golden.json" % CT.TAG)
        G.golden_export(gpath, self.golden)
        self.payload_extra["golden_export"] = gpath
        self.cross[C] = self.cross_record()
        if self.golden_fail:
            self.fails += len(self.golden_fail)
            self.flush_payload(False)
            return CT.RC_CORRECT
        xc = self.cross_now()
        self.cross_check = xc.get(C)
        if C != REF_C and CROSS_REQUIRED and REF_C in xc and not xc[C]["certified"]:
            self.fails += 1
            self.flush_payload(False)
            raise _Uncertified("C=%d is NOT certified against C%d: %s" % (C, REF_C, xc[C]["why"][:4]))
        # 3. the measured pass
        self.restart(self.pps, "C%d after golden" % C)
        self.golden_mode = "check"
        self.alloc_measure()
        pos, _, ok = self.warm_pass("measured", ref=ref_warm)
        if not ok:
            raise CT._ProofFail("the measured pass's warm steps differ from the reference pass at C=%d" % C)
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        return self.measured_curve(pos)

    def alloc_measure(self):
        for n in ("scr_k", "scr_v"):
            if hasattr(self, n):
                delattr(self, n)
        self.scr_layout = None
        super().alloc_measure()

    # ------------------------------------------------------------------------------------------------ CONTROLS (pool)
    def control_specs(self):
        s0 = CT.CONTROL_STEPS[0]
        out = super().control_specs()
        if not any(PS.pool_active(e) for e in self.engines):
            return out
        return out + [
            dict(name="pool_map", kind="perturb", family="pool", layer=5 % self.NL, expect="DETECTED", step=s0),
            dict(name="pool_rows", kind="perturb", family="poolrows", layer=13 % self.NL, expect="DETECTED", step=s0),
            dict(name="pool_stamp", kind="perturb", family="pool", layer=17 % self.NL, expect="DETECTED", step=s0),
            dict(name="pool_post_reference_skipped", kind="pool_noref", family="pool", expect="DETECTED", step=s0),
            dict(name="tick_pool_hit", kind="tick", what="pool_hit", layer=6 % self.NL, tick=1, expect="DETECTED", step=s0)]

    def perturb(self, what, l):
        e = self.engines[l]
        b, h = 1 % self.B, 1 % self.H
        att = int(e.topk) * int(e.block_size)
        if what == "pool_map":
            e._pool_map[h, b, 0] = int(e._pool_map[h, b, 0]) + 7 if int(e._pool_map[h, b, 0]) >= 0 else 9973
            return "layer %d _pool_map[%d, %d, 0] changed" % (l, h, b)
        if what == "pool_rows":
            e._k_gpu.view(torch.int16)[b, att, h, :8].bitwise_xor_(0x0101)
            return "layer %d _k_gpu[%d, %d (pool row 0), %d, :8] ^= 0x0101" % (l, b, att, h)
        if what == "pool_stamp":
            e._pool_stamp_base += 1
            return "layer %d _pool_stamp_base += 1" % l
        return super().perturb(what, l)

    def control_pass(self, spec, warm):
        if spec["kind"] == "pool_noref":
            s0 = spec["step"]
            self.golden_mode = "check"
            pos, _, ok = self.warm_pass(spec["name"], ref=warm)
            if not ok:
                raise CT._ProofFail("CONTROLS pass %s: the warm steps differ from the golden pass" % spec["name"])
            self.snap = self.ss.CounterSnapshot(self.cache)
            self.trans = self.ss.transient_ids(self.model)
            with self.guard.destructive(spec["name"]):                  # refused outside the CONTROLS process
                self.skip_post_ref_at = s0
            start = len(self.gate_log)
            try:
                self.gated_sequence(pos)
            finally:
                self.skip_post_ref_at = None
            log = self.gate_log[start:]
            compact = [dict(step=g["step"], ok=g["ok"], work_ran=g["work_ran"], pre_ok=g["pre"]["ok"], pre_why=g["pre"].get("why"),
                            post_ok=g["post"]["ok"], post_why=g["post"].get("why")) for g in log]
            return dict(name=spec["name"], step=s0, expect=spec["expect"], family="pool", layer=None, verdict=self.control_verdict(spec, compact, {}),
                        extra={}, gate=compact)
        if spec.get("what") != "pool_hit":
            return super().control_pass(spec, warm)
        if not hasattr(self, "rc_loads"):
            self.setup_curve(minimal=True)
        name, s0 = spec["name"], spec["step"]
        self.golden_mode = "check"
        pos, _, ok = self.warm_pass(name, ref=warm)
        if not ok:
            raise CT._ProofFail("CONTROLS pass %s: the warm steps differ from the golden pass" % name)
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        extra = {}

        def hook(k, where):
            if k != spec["tick"] or where != "pre_decode":
                return False
            with self.guard.destructive(name):
                extra["perturbed"] = plant_pool_hit(self.engines[spec["layer"]], 1 % self.H, 1 % self.B, VA.L // self.R)
            return True

        def work(ctx):
            if ctx["it"] != s0:
                return
            self.tick_hook = hook
            try:
                w = self.window(ctx, None, None, IC.CTL_TICKS, "control_" + name, 0, ticks=True, transfer=False)
                extra.update(loads=w.get("tick_loads"), bit_exact=w.get("tick_bit_exact"), pool_actions=w.get("tick_pool_actions"), receipts_ok=w.get("receipts_ok"))
                self.fails -= int(not w["ok"])
                self.curve.setdefault("control_windows", []).append(self.windows.pop())
            finally:
                self.tick_hook = None
        start = len(self.gate_log)
        self.gated_sequence(pos, work)
        log = self.gate_log[start:]
        compact = [dict(step=g["step"], ok=g["ok"], work_ran=g["work_ran"], pre_ok=g["pre"]["ok"], pre_why=g["pre"].get("why"),
                        post_ok=g["post"]["ok"], post_why=g["post"].get("why")) for g in log]
        return dict(name=name, step=s0, expect=spec["expect"], family=None, layer=spec.get("layer"), verdict=self.control_verdict(spec, compact, extra),
                    extra=extra, gate=compact)

    def control_verdict(self, spec, log, extra):
        if spec["kind"] == "pool_noref":
            g = {x["step"]: x for x in log}.get(spec["step"])
            if g is None:
                return "UNTESTABLE"
            why = g["pre_why"] or []
            return "DETECTED" if (not g["pre_ok"] and not g["work_ran"] and any(w.startswith("pool[") or w.startswith("pool_dup") for w in why)) else "NOT_DETECTED"
        if spec.get("what") == "pool_hit":
            loads, eq, sw = extra.get("loads") or [], extra.get("bit_exact") or [], extra.get("pool_actions") or []
            t = spec["tick"]
            if len(loads) != IC.CTL_TICKS or len(sw) != IC.CTL_TICKS:
                return "UNTESTABLE"
            before = all(x == 0 for x in loads[:t]) and all(eq[:t]) and all(x == 0 for x in sw[:t])
            return "DETECTED" if (before and sw[t] > 0 and loads[t] == 0) else "NOT_DETECTED"
        return super().control_verdict(spec, log, extra)

    # ------------------------------------------------------------------------------------------------ payload
    def payload(self, partial):
        p = super().payload(partial)
        p["cell"] = dict(self.cell, C=self.cell_C, P=self.cell_P, accounting=getattr(self, "accounting", None), cross_record=self.cross.get(self.cell_C),
                         cross_check=getattr(self, "cross_check", None), pinned=self.pinned_inventory(), paced=PACED, paced_ticks=PACED_TICKS,
                         diag_arms=list(getattr(self, "diag_arms", []) or []))
        return p


def plant_pool_hit(e, h, b, n_blocks):
    """CONTROLS only: make the next decode of engine e serve the block of attended slot 0 from the POOL, bit-exactly: park its
    exact rows in pool slot 0 (map = that block), and name an unselected, non-resident block in slot 0. diff then refills slot 0
    with the selected block, pool_update finds it in pool slot 0 -> SWAP, the H2D load is cleared (0 loads) and the bytes are
    right (logits bit-exact): only the pool-action receipt can see it."""
    R = int(e.block_size)
    att = int(e.topk) * R
    X = int(e._block_map[h, b, 0])
    res = set(int(v) for v in e._block_map[h, b].tolist()) | set(int(v) for v in e._pool_map[h, b].tolist())
    Y = next(v for v in range(n_blocks) if v not in res and v != X)
    for n in ("_k_gpu", "_v_gpu", "_kv_bias_gpu"):
        t = getattr(e, n)
        t[b, att:att + R, h].copy_(t[b, 0:R, h])
    e._pool_map[h, b, 0] = X
    e._block_map[h, b, 0] = Y
    return "layer pool hit planted: stream (%d, %d) slot 0 block %d parked in pool slot 0, slot 0 renamed to %d" % (h, b, X, Y)


# ------------------------------------------------------------------------------------------------------------ loading
def load_model_pool(path, P_need):
    """VA.load_model without its pool refusal (verify_alone.py:452-453): the engine must be imported with NOSI_POOL_BLOCKS >=
    every cell's P (the prefill allocates that pool; smaller cells re-pool)."""
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    if _ce.VERIFY_ROUND_SLOTS != 0:
        VA.die("NOSI_VERIFY_ROUND_SLOTS must be 0 for the cache-size curve")
    if _ce.POOL_BLOCKS < P_need:
        VA.die("NOSI_POOL_BLOCKS=%d < %d (the largest pool of the cells)" % (_ce.POOL_BLOCKS, P_need))
    if _ce._avail.SPEC not in ("", "0", "off"):
        VA.die("NOSI_AVAIL must be off with the pool (cache_engine.py:192-198)")
    print("[ccurve] engine imported with NOSI_POOL_BLOCKS=%d (extension %s); L=%d N=%d warm=%d" % (
        _ce.POOL_BLOCKS, "loaded" if _ce._pool is not None else "absent", VA.L, VA.N, VA.WARM), flush=True)
    return Llama(model_name=path, device="cuda", offload=True)


def load_cross(out_dir, B, xref=None):
    """{C: cross record} from the cell payloads already on disk (a resumed process compares against an earlier C63; a leftover
    job against CC1's, CC_XREF_DIRS). Records of out_dir win over the cross-reference directories."""
    cells = {}
    for d in list(XREF_DIRS if xref is None else xref) + [out_dir]:
        cells.update(_cross_in(d, B))
    return cells


def _cross_in(out_dir, B):
    cells = {}
    for fn in glob.glob(os.path.join(out_dir, "cc_b%d_c*.json" % B)):
        m = re.match(r"^cc_b\d+_c(\d+)\.json$", os.path.basename(fn))
        if not m:
            continue
        try:
            with open(fn) as f:
                p = json.load(f)
        except Exception:
            continue
        rec = ((p.get("cell") or {}).get("cross_record"))
        if rec:
            cells[int(m.group(1))] = rec
    return cells


# ------------------------------------------------------------------------------------------------------------ table
def _kept(rows):
    return [r for r in rows if CT.row_keep(r)[0]]


def _restore_gb(cell):
    v = [x.get("bytes") for x in (cell.get("restore_per_tick") or {}).values() if isinstance(x, dict) and x.get("bytes") is not None]
    return CV.pctl(v, 50) / 1e9 if v else float("nan")


def _point_specs(cell):
    """(method, regime, phase, covered) of every point of a cell: the base arms x (saturation, decode-paced) exactly as CC1, then
    at a diagnostic cell the host-pack saturation and the ext-paced arms (from the payload's own diag record)."""
    specs = [(a, rg, ph, rg == "saturation") for a in ARMS for rg, ph in (("saturation", "overlap"), ("paced", "paced"))]
    dg = cell.get("diag") or {}
    for a in dg.get("hostpack_saturation") or []:
        specs.append((a, "saturation", "overlap", True))
    if dg.get("ext"):
        for a in dg.get("arms") or []:
            specs.append((a, "ext-paced", "ext_paced", False))
    return specs


def _med(ws, key):
    return CV.pctl([(w.get("diag") or {}).get(key) for w in ws], 50)


def summarize_cell(p, cert):
    """Per (method, regime): the plot points of one cell payload, from kept rows only (row_keep); every exclusion counted. At a
    diagnostic cell also the diagnostic rows (cpu8 vs the host-pack control; ext-paced and saturation; NO subtraction)."""
    B, cell = p.get("batch"), p.get("cell") or {}
    C = cell.get("C")
    rows = p.get("windows") or []
    excl = {}
    for r in rows:
        key = (r.get("arm"), r.get("phase"))
        c = excl.setdefault(key, dict(total=0, kept=0, not_ok=0, gate_fail=0))
        c["total"] += 1
        k, why = CT.row_keep(r)
        c["kept" if k else (why if why in c else "not_ok")] += 1
    kept = _kept(rows)
    alone = {}
    for r in kept:
        if r.get("phase") in ("decode_alone_pre", "decode_alone_post"):
            alone.setdefault(r["step"], []).extend(CV.steady_ticks([tuple(x) for x in r.get("ticks", [])], IC.SKIP_FIRST))
    acc = (cell.get("accounting") or {}).get("steady") or {}
    mem = ((p.get("curve") or {}).get("mem_samples") or [])
    pin = cell.get("pinned") or {}
    rgb = _restore_gb(cell)
    points, per_step_rows, diag_rows = [], [], []
    for a, regime, phase, covered in _point_specs(cell):
        ov = [r for r in kept if r.get("arm") == a and r.get("phase") == phase]
        pt = CCC.blank_point(B, C, a, regime, "OK" if (cert and ov) else "MISSING")
        pt["certified"] = bool(cert)
        conc = {}
        for r in ov:
            tk = [tuple(x) for x in r.get("ticks", [])]
            if covered:
                cov = CV.tick_coverage(tk, [(q["g0"], q["g1"]) for q in r.get("requests", []) if q.get("g0") is not None])
                conc.setdefault(r["step"], []).extend(CV.steady_ticks(tk, IC.SKIP_FIRST, cov, IC.COVER_MIN))
            else:
                conc.setdefault(r["step"], []).extend(CV.steady_ticks(tk, IC.SKIP_FIRST))
        pc = CV.paired_by_step(alone, conc)
        gb = {}
        for r in ov:
            gb.setdefault(r["step"], []).append((r.get("m") or {}).get("during_gbps"))
        gstep = [CV.pctl(v, 50) for s, v in sorted(gb.items())]
        g_mean, g_lo, g_hi = CCC.step_ci(gstep)
        ms = [r.get("m") or {} for r in ov]
        pz = [m.get("paced") or {} for m in ms]
        ta = [r.get("m") or {} for r in kept if r.get("arm") == a and r.get("phase") == "transfer_alone"]
        if regime == "paced":
            offered = CV.pctl([z.get("offered_gbps") for z in pz], 50)                  # per METHOD (its own release intervals)
        elif regime == "ext-paced":
            offered = CV.pctl([(m.get("ext") or {}).get("offered_gbps") for m in ms], 50)  # the schedule's (identical for every arm)
        else:
            offered = float("nan")
        paced_like = regime in ("paced", "ext-paced")
        note = ("" if cert else "NOT certified") + ("" if ov else "; no kept %s rows" % phase)
        if a.endswith("hostpack"):
            note = (note + "; " if note else "") + "useful / window GB/s = " + CCC.HOSTPACK_DONE_LABEL
        pt.update(n_steps=pc["n_steps"], slowdown_pct=pc["extra_pct_mean"], slowdown_ci_lo=pc["extra_pct_ci"][0], slowdown_ci_hi=pc["extra_pct_ci"][1],
                  extra_ms=pc["extra_ms_mean"], extra_ms_ci_lo=pc["extra_ms_ci"][0], extra_ms_ci_hi=pc["extra_ms_ci"][1],
                  decode_alone_p50_ms=pc["alone_p50"], decode_alone_p95_ms=pc["alone_p95"], tick_p50_ms=pc["conc_p50"], tick_p95_ms=pc["conc_p95"],
                  useful_gbps=g_mean, useful_gbps_ci_lo=g_lo, useful_gbps_ci_hi=g_hi, window_gbps=CV.pctl([m.get("window_gbps") for m in ms], 50),
                  alone_gbps=CV.pctl([m.get("alone_gbps") for m in ta], 50), offered_gbps=offered,
                  useful_bytes=CV.pctl([m.get("useful_bytes") for m in ms], 50), wire_bytes=CV.pctl([m.get("wire_bytes") for m in ms], 50),
                  full_ready_ms=CV.pctl([m.get("full_ready_ms") for m in ms], 50), overlap_frac=CV.pctl([m.get("overlap_frac") for m in ms], 50),
                  backlog_slope_bytes_per_release=(CV.pctl([z.get("backlog_slope_bytes_per_release") for z in pz], 50) if paced_like else float("nan")),
                  backlog_end_bytes=(CV.pctl([z.get("backlog_end_bytes") for z in pz], 50) if paced_like else float("nan")),
                  overloaded_windows=sum(1 for z in pz if z.get("overloaded")),
                  drain_ms=(CV.pctl([z.get("drain_ms") for z in pz], 50) if paced_like else float("nan")),
                  ready_p95_ms=(CV.pctl([z.get("ready_p95_ms") for z in pz], 50) if paced_like else float("nan")),
                  h2d_per_stream_step=acc.get("h2d_per_stream_step", float("nan")), pool_hits_per_stream_step=acc.get("hits_per_stream_step", float("nan")),
                  peak_allocated_gb=max([m["peak_allocated_gb"] for m in mem] or [float("nan")]), peak_reserved_gb=max([m["peak_reserved_gb"] for m in mem] or [float("nan")]),
                  device_used_gb=max([m["device_used_gb"] for m in mem] or [float("nan")]),
                  pinned_host_gb=(pin.get("host_cache_reserved_pow2", 0) + pin.get("staging_bytes", 0) + pin.get("plan_rows_bytes", 0)) / 1e9,
                  note=note, restore_gb_per_tick=rgb)
        points.append(pt)
        for x in pc["per_step"]:
            per_step_rows.append(dict(batch=B, C=C, method=a, regime=regime, step=x["step"], alone_p50=x["alone_p50"], conc_p50=x["conc_p50"],
                                      extra_ms=x["extra_ms"], extra_pct=x["extra_pct"], n_alone=x["n_alone"], n_conc=x["n_conc"],
                                      during_gbps_p50=CV.pctl(gb.get(x["step"], []), 50)))
        if cell.get("diag") and a in (cell["diag"].get("arms") or []) and regime in CCC.DIAG_REGIMES:
            dws = [r for r in ov if r.get("diag")]
            sig = {}
            for r in dws:
                if regime == "ext-paced":
                    sig.setdefault(r["step"], set()).add((r.get("sched") or {}).get("digest"))
            same = (all(len(v) == 1 for v in sig.values()) and bool(sig)) if regime == "ext-paced" else None
            diag_rows.append(dict(batch=B, C=C, method=a, regime=regime, status=pt["status"], label=CCC.method_label(a) or "cpu8: " + IC.curve_arm("cpu8")["label"],
                                  done=(CCC.HOSTPACK_DONE_LABEL if a.endswith("hostpack") else "scattered into the scratch"), n_steps=pc["n_steps"],
                                  n_windows=len(dws), slowdown_pct=pt["slowdown_pct"], slowdown_ci_lo=pt["slowdown_ci_lo"], slowdown_ci_hi=pt["slowdown_ci_hi"],
                                  offered_gbps=offered, P_ms=_med(dws, "P_ms"), T0_ms=_med(dws, "T0_ms"), ready_p50_ms=_med(dws, "ready_p50_ms"),
                                  ready_p95_ms=_med(dws, "ready_p95_ms"), full_release_p50_ms=_med(dws, "full_release_p50_ms"),
                                  full_release_p95_ms=_med(dws, "full_release_p95_ms"), full_window_ms=_med(dws, "full_window_ms"),
                                  backlog_end_bytes=_med(dws, "backlog_end_bytes"), backlog_slope_bytes_per_release=_med(dws, "backlog_slope_bytes_per_release"),
                                  overloaded_windows=sum(1 for r in dws if (r.get("diag") or {}).get("overloaded")), drain_ms=_med(dws, "drain_ms"),
                                  lateness_p50_ms=_med(dws, "lateness_p50_ms"), lateness_p95_ms=_med(dws, "lateness_p95_ms"),
                                  lateness_max_ms=CV.pctl([(r.get("diag") or {}).get("lateness_max_ms") for r in dws], 100),
                                  late_releases=(sum(int((r.get("diag") or {}).get("late_releases") or 0) for r in dws) if regime == "ext-paced" else float("nan")),
                                  handoff_p50_ms=_med(dws, "handoff_p50_ms"), handoff_p95_ms=_med(dws, "handoff_p95_ms"), d2h_p50_ms=_med(dws, "d2h_p50_ms"),
                                  d2h_p95_ms=_med(dws, "d2h_p95_ms"), coord_duty=_med(dws, "coord_duty"), pack_duty=_med(dws, "pack_duty"),
                                  decode_duty=_med(dws, "decode_duty"), done_gbps_in_window=_med(dws, "done_gbps_in_window"), schedule_identical=same,
                                  note=pt["note"]))
    return dict(batch=B, C=C, points=points, per_step=per_step_rows, diag=diag_rows, excl=excl, kept=len(kept), total=len(rows), restore_gb=rgb)


def table_ccurve(out_dir, csv_requests=None):
    """ccurve_points.csv (the plot input), ccurve_table.md, ccurve_plans.csv (the natural plans per C), ccurve_per_step.csv,
    ccurve_cross.json, ccurve_windows.csv and the gzip-streamed ticks / requests timelines, from kept rows only."""
    csv_requests = (os.environ.get("CV_TABLE_CSV", "1") == "1") if csv_requests is None else csv_requests
    t0 = time.time()
    root = os.path.dirname(os.path.abspath(out_dir))
    L = ["# Cache-size interference curve (%s; %s)" % (K.LABEL, CV.SUSTAINED_LABEL), "",
         "Saturation = %s. Decode-paced (raw key 'paced'; called 'arrival-paced' in CC1's raw files) = %s. Copy rows: %s (CPU8 "
         "descriptors built live). Decode: %s." % (CV.SATURATED_LABEL, CCC.DECODE_PACED_LABEL, CV.PREBUILT_LABEL, CV.RESIDENT_LABEL),
         "The decode-paced offered GB/s is PER METHOD (each from its own release intervals); the methods' offered loads are NOT matched. "
         "Ext-paced (the diagnostic cells only) = %s. Host-pack rows = %s; their GB/s are %s." % (CCC.EXT_LABEL, CCC.HOSTPACK_LABEL,
                                                                                               CCC.HOSTPACK_DONE_LABEL),
         "CAVEATS: " + " ".join(CCC.TABLE_CAVEATS), "",
         "C = 63 attended non-tail slots + P pool slots per (layer, KV head, request); the tail slot (one more 64-token group per stream) is "
         "always resident and never fetched. Rows enter a statistic only when ok AND their gated step passed the golden gate (row_keep). "
         "Slowdown = per trace step p50(beside) / p50(decode-alone pre + post) - 1, mean over steps, 95%% bootstrap CI over steps; saturation "
         "uses covered ticks (>= %.0f%% inside the transfer), paced uses every steady tick. A cell that is not certified (cross-C, capture, "
         "restarts, gates) is MISSING; C63 plans are never substituted." % (100 * IC.COVER_MIN), ""]
    ok_all = True
    cl, cok = IC.controls_lines(root)
    L += cl
    ok_all &= cok
    files = sorted(f for f in glob.glob(os.path.join(out_dir, "cc_b*_c*.json")) if re.match(r"^cc_b\d+_c\d+\.json$", os.path.basename(f)))
    pays, markers = {}, {}
    for fn in files:
        with open(fn) as f:
            p = json.load(f)
        m = re.match(r"^cc_b(\d+)_c(\d+)\.json$", os.path.basename(fn))
        pays[(int(m.group(1)), int(m.group(2)))] = p
    for fn in glob.glob(os.path.join(out_dir, "cc_b*_c*.result.json")):
        m = re.match(r"^cc_b(\d+)_c(\d+)\.result\.json$", os.path.basename(fn))
        if m:
            with open(fn) as f:
                markers[(int(m.group(1)), int(m.group(2)))] = json.load(f)
    batches = sorted({b for b, _ in list(pays) + list(markers)})
    cross_all, points, per_step, plans_rows, diag_rows = {}, [], [], [], []
    tick_sink = gzip.open(os.path.join(out_dir, "ticks_ccurve.csv.gz"), "wt", compresslevel=1, newline="") if csv_requests else None
    req_sink = gzip.open(os.path.join(out_dir, "requests_ccurve.csv.gz"), "wt", compresslevel=1, newline="") if csv_requests else None
    tw = csv.writer(tick_sink) if tick_sink else None
    rw = csv.writer(req_sink) if req_sink else None
    if tw:
        tw.writerow(("batch", "C", "step", "arm", "regime", "phase", "rep", "tick", "t0_ms", "t1_ms", "tick_ms", "coverage", "loads", "pool_actions", "bit_exact", "kept"))
        rw.writerow(("batch", "C", "step", "arm", "regime", "phase", "rep", "i", "plan_step", "layer", "groups", "useful", "wire", "release", "release_ms",
                     "issue_ms", "g0_ms", "g1_ms", "ready_ms", "release_wait_ms", "kept"))
    n_t = n_r = 0
    excl_rows = []
    try:
        for B in batches:
            cells = {}
            for d in XREF_DIRS:
                cells.update(_cross_in(d, B))
            cells.update({C: (p.get("cell") or {}).get("cross_record") for (b, C), p in pays.items() if b == B and (p.get("cell") or {}).get("cross_record")})
            xc = CCC.cross_c_check(cells, REF_C)
            cross_all[B] = xc
            L.append("## B=%d" % B)
            L.append("- cross-C check against C%d (capture logits per natural step; golden reference / advance logits per gated step): %s" % (
                REF_C, "; ".join("C%d %s (%s)" % (C, "CERTIFIED" if x["certified"] else "NOT certified", x["compared"] if x["certified"] else "; ".join(x["why"][:2]))
                                 for C, x in sorted(xc.items())) or "no records"))
            for C in CCC.CAPACITIES:
                p, res = pays.get((B, C)), markers.get((B, C), {})
                if p is None:
                    why_ = "no payload; marker rc %s class %s" % (res.get("rc"), res.get("class"))
                    for a in ARMS:
                        for rg in CCC.REGIMES:
                            points.append(CCC.blank_point(B, C, a, rg, "MISSING", why_))
                    if C in DIAG_CAPS and DIAG_ARMS:                     # the registered diagnostic cells: MISSING, never dropped
                        for a in DIAG_ARMS:
                            if a.endswith("hostpack"):
                                points.append(CCC.blank_point(B, C, a, "saturation", "MISSING", why_))
                            if EXT:
                                points.append(CCC.blank_point(B, C, a, "ext-paced", "MISSING", why_))
                            for rg in (("saturation", "ext-paced") if EXT else ("saturation",)):
                                d_ = {k: float("nan") for k in CCC.DIAG_FIELDS}
                                d_.update(batch=B, C=C, method=a, regime=rg, status="MISSING", label=CCC.method_label(a), done="", n_steps=0, n_windows=0,
                                          late_releases=None, schedule_identical=None, overloaded_windows=0, note=why_)
                                diag_rows.append(d_)
                    L.append("- C%d: NO PAYLOAD (marker rc %s, class %s): MISSING" % (C, res.get("rc"), res.get("class")))
                    ok_all = False
                    continue
                cell = p.get("cell") or {}
                gs = CT.gate_summary(p)
                cap = p.get("capture") or {}
                restarts = (p.get("confirm") or {}).get("restarts") or []
                own_ok = bool(cap.get("accepted") and all(r.get("ok") for r in restarts) and not gs["bad"] and p.get("fails", 1) == 0 and not p.get("crash")
                              and res.get("rc", 1) == 0)
                cert = bool(xc.get(C, {}).get("certified")) and bool(cap.get("accepted")) and all(r.get("ok") for r in restarts)
                s = summarize_cell(p, cert)
                for pt in s["points"]:
                    if not own_ok:
                        pt["note"] = (pt["note"] + "; cell fails %s rc %s" % (p.get("fails"), res.get("rc"))).strip("; ")
                points += s["points"]
                per_step += s["per_step"]
                diag_rows += s["diag"]
                ok_all &= own_ok and cert
                acc = cell.get("accounting") or {}
                st = acc.get("steady") or {}
                summ = acc.get("summary") or {}
                L.append("- C%d (P=%s): %s; fails %s, rc %s, class %s; kept %d of %d windows; capture accepted %s (invariants %s); restarts %d/%d; "
                         "gate fails %s" % (C, cell.get("P"), "CERTIFIED" if cert else "NOT CERTIFIED -> MISSING", p.get("fails"), res.get("rc"), res.get("class"),
                                             s["kept"], s["total"], cap.get("accepted"), {k: v for k, v in (cap.get("invariants") or {}).items() if v} or "all 0",
                                             sum(1 for r in restarts if r.get("ok")), len(restarts), gs["bad"] or "none"))
                sts = st.get("steps") or [float("nan")]
                L.append("  - natural plans at C%d (replayed steps %s..%s): H2D %.4f groups per stream-step (the PCIe misses BOTH methods replay), pool hits %.4f "
                         "(device-to-device, NOT replayed), evictions %.4f; %s" % (C, sts[0], sts[-1], st.get("h2d_per_stream_step", float("nan")),
                                                                                  st.get("hits_per_stream_step", float("nan")),
                                                                                  st.get("evicted_per_stream_step", float("nan")), summ.get("replayed", "")))
                al = acc.get("all_steps") or {}
                if al:
                    L.append("  - from step 1 (incl. the cold victim pool after prefill): H2D %.4f, pool hits %.4f per stream-step; per-step series in "
                             "ccurve_plans.csv" % (al.get("h2d_per_stream_step", float("nan")), al.get("hits_per_stream_step", float("nan"))))
                cur = p.get("curve") or {}
                pdig = {x.get("digest") for x in (cur.get("ptrains") or {}).values()}
                cov = {"saturation": [], "paced": []}
                for dg, x in (cur.get("refs") or {}).items():
                    if "requests_loading" in x:
                        cov["paced" if dg in pdig else "saturation"].append("%d/%d" % (x["requests_content_verified"], x["requests_loading"]))
                rpt = cell.get("restore_per_tick") or {}
                rb = [x.get("bytes") for x in rpt.values() if isinstance(x, dict) and x.get("bytes") is not None]
                L.append("  - decode-state restore (sanctioned, every resident tick, alone AND beside): %s GB copied per tick from the snapshot "
                         "tensors (HBM traffic ~2x: read + write)%s" % (("%.3f" % (CV.pctl(rb, 50) / 1e9)) if rb else "NOT RECORDED (pre-relabel payload)",
                                                                        "" if not rb else " over %d gated steps" % len(rb)))
                dg = cell.get("diag")
                if dg:
                    es = cur.get("ext_sched") or {}
                    L.append("  - DIAGNOSTIC cell: arms %s; host-pack saturation %s; ext-paced %s. Ext schedules (fixed per step before any arm): %s" % (
                        dg.get("arms"), dg.get("hostpack_saturation"), "ON" if dg.get("ext") else "off",
                        "; ".join("step %s: %s" % (k, ("SKIPPED (%s)" % v.get("why")) if v.get("skipped") else "R %d, P %.3f ms (alone period %.3f ms), T0 %.3f ms, "
                                  "offered %.3f GB/s, %d ticks, digest %s" % (v["R"], v["P_ms"], v.get("alone_period_p50_ms", float("nan")), v["T0_ms"],
                                                                           v["offered_gbps"], v["ticks"], v["digest"]))
                                  for k, v in sorted(es.items(), key=lambda kv: int(kv[0]))) or "none"))
                L.append("  - content-check coverage per train (requests whose rows survive in the final scratch / loading requests): saturation %s; "
                         "decode-paced %s. A fully rewritten request's delivery is evidenced by its launch + events only; plan identity across the two "
                         "methods is structural (the same train and per-C plan store; CPU8 list rows compared per request)" % (
                             ", ".join(cov["saturation"]) or "none", ", ".join(cov["paced"]) or "none"))
                ps = acc.get("all_steps") or st                             # the per-step series from step 1 (incl. the cold pool)
                for i, stp in enumerate(ps.get("steps") or []):
                    plans_rows.append(dict(batch=B, C=C, step=stp, h2d_groups=ps["h2d_groups"][i], h2d_bytes=ps["h2d_bytes"][i], pool_hits=ps["hits"][i],
                                           d2d_moves=ps["d2d_moves"][i], evicted=ps["evicted"][i]))
                for (arm_, ph), c in sorted(s["excl"].items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
                    excl_rows.append((B, C, arm_, ph, c["total"], c["kept"], c["not_ok"], c["gate_fail"]))
                if tw:
                    for r in p.get("windows") or []:
                        kp = int(CT.row_keep(r)[0])
                        tk = [tuple(x) for x in r.get("ticks", [])]
                        rq = r.get("requests") or []
                        cov = CV.tick_coverage(tk, [(q["g0"], q["g1"]) for q in rq if q.get("g0") is not None]) if tk else []
                        for i, (a0, a1) in enumerate(tk):
                            tw.writerow((B, C, r.get("step"), r.get("arm"), r.get("regime"), r.get("phase"), r.get("rep"), i, a0, a1, a1 - a0, cov[i] if cov else "",
                                         (r.get("tick_loads") or [""] * len(tk))[i], (r.get("tick_pool_actions") or [""] * len(tk))[i],
                                         (r.get("tick_bit_exact") or [""] * len(tk))[i], kp))
                            n_t += 1
                        rel = r.get("release_ms") or []
                        for q in rq:
                            j = q.get("release")
                            rms = rel[j] if (j is not None and 0 <= j < len(rel)) else ""
                            rw.writerow((B, C, r.get("step"), r.get("arm"), r.get("regime"), r.get("phase"), r.get("rep"), q.get("i"), q.get("step"), q.get("layer"),
                                         q.get("groups"), q.get("useful"), q.get("wire"), j if j is not None else "", rms, q.get("issue"), q.get("g0"), q.get("g1"),
                                         (q["g1"] - rms) if (rms != "" and q.get("g1") is not None) else "", q.get("release_wait_ms", ""), kp))
                            n_r += 1
            L.append("")
            L += ["| C | method | regime | status | steps | slowdown %% [CI] | extra ms | decode-alone p50 / p95 | beside p50 / p95 | useful GB/s [CI] | "
                  "window GB/s | offered GB/s (per method) | backlog slope MB/release | overloaded | drain ms | H2D / pool hits per stream-step | peak alloc / reserved / used GB |",
                  "|" + "---|" * 17]
            for pt in [x for x in points if x["batch"] == B]:
                L.append("| %d | %s | %s | %s | %d | %+.2f [%.2f, %.2f] | %+.3f | %.3f / %.3f | %.3f / %.3f | %.2f [%.2f, %.2f] | %.2f | %.2f | %.2f | %s | %.1f | "
                         "%.3f / %.3f | %.1f / %.1f / %.1f |" % (
                             pt["C"], pt["method"], CCC.REGIME_NAMES.get(pt["regime"], pt["regime"]), pt["status"], pt["n_steps"], pt["slowdown_pct"], pt["slowdown_ci_lo"], pt["slowdown_ci_hi"],
                             pt["extra_ms"], pt["decode_alone_p50_ms"], pt["decode_alone_p95_ms"], pt["tick_p50_ms"], pt["tick_p95_ms"], pt["useful_gbps"],
                             pt["useful_gbps_ci_lo"], pt["useful_gbps_ci_hi"], pt["window_gbps"], pt["offered_gbps"], pt["backlog_slope_bytes_per_release"] / 1e6,
                             pt["overloaded_windows"], pt["drain_ms"], pt["h2d_per_stream_step"], pt["pool_hits_per_stream_step"], pt["peak_allocated_gb"],
                             pt["peak_reserved_gb"], pt["device_used_gb"]))
            L.append("")
            dr = [x for x in diag_rows if x["batch"] == B]
            if dr:
                L += diag_lines(dr)
    finally:
        if tick_sink:
            tick_sink.close()
            req_sink.close()
    with open(os.path.join(out_dir, "ccurve_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CCC.POINT_FIELDS)
        for pt in points:
            w.writerow([CCC.fmt(pt.get(k)) for k in CCC.POINT_FIELDS])
    if diag_rows:
        with open(os.path.join(out_dir, "ccurve_hostpack.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CCC.DIAG_FIELDS)
            for r in diag_rows:
                w.writerow([CCC.fmt(r.get(k)) for k in CCC.DIAG_FIELDS])
    with open(os.path.join(out_dir, "ccurve_per_step.csv"), "w", newline="") as f:
        cols = ("batch", "C", "method", "regime", "step", "alone_p50", "conc_p50", "extra_ms", "extra_pct", "n_alone", "n_conc", "during_gbps_p50")
        w = csv.writer(f)
        w.writerow(cols)
        for r in per_step:
            w.writerow([CCC.fmt(r.get(k)) for k in cols])
    with open(os.path.join(out_dir, "ccurve_plans.csv"), "w", newline="") as f:
        cols = ("batch", "C", "step", "h2d_groups", "h2d_bytes", "pool_hits", "d2d_moves", "evicted")
        w = csv.writer(f)
        w.writerow(cols)
        for r in plans_rows:
            w.writerow([r[k] for k in cols])
    with open(os.path.join(out_dir, "ccurve_windows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("batch", "C", "arm", "phase", "total", "kept", "not_ok", "gate_fail"))
        for r in excl_rows:
            w.writerow(r)
    with open(os.path.join(out_dir, "ccurve_cross.json"), "w") as f:
        json.dump({str(b): {str(c): v for c, v in x.items()} for b, x in cross_all.items()}, f, indent=1)
    if tw:
        L.append("- per-tick receipts: %d rows -> ticks_ccurve.csv.gz; per-request timeline: %d rows -> requests_ccurve.csv.gz" % (n_t, n_r))
    L.append("table built in %.1f s" % (time.time() - t0))
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "ccurve_table.md"), "w") as f:
        f.write(text)
    print(text)
    return 0 if ok_all else 1


def diag_lines(rows):
    """The diagnostic section of the table: cpu8 vs the host-pack control side by side, NO subtraction, NO 'GPU share'."""
    f = lambda x, d=2: "" if (x is None or (isinstance(x, float) and x != x)) else ("%.*f" % (d, x) if isinstance(x, float) else str(x))
    L = ["### Host-pack diagnostic (cpu8 vs %s)" % CCC.HOSTPACK_LABEL,
         "Side by side; NOTHING is subtracted (no additive 'GPU share'; the GIL is a hypothesis). Ext-paced = %s. Host-pack 'done' = %s. "
         "Saturation: every request released at the gate (ready = from the gate); covered ticks. Ext-paced: every steady tick; "
         "lateness = actual publication - scheduled; handoff = host wait for the list + live descriptor build; D2H = list copy span; "
         "coordinator / pack duty = its host work inside the tick window / the window; decode duty = decode spans / window." % (
             CCC.EXT_LABEL, CCC.HOSTPACK_DONE_LABEL), "",
         "| C | method | regime | status | steps / windows | slowdown %% [CI] | offered GB/s | ready p50 / p95 ms | full release p50 / window ms | "
         "backlog end MB / slope MB per release | lateness p50 / p95 / max ms (late) | handoff p50 / p95 ms | D2H p50 / p95 ms | coord / pack / decode duty | "
         "done GB/s in window | schedule identical |", "|" + "---|" * 16]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s / %s | %s [%s, %s] | %s | %s / %s | %s / %s | %s / %s | %s / %s / %s (%s) | %s / %s | %s / %s | %s / %s / %s | %s | %s |" % (
            r["C"], r["method"], CCC.REGIME_NAMES.get(r["regime"], r["regime"]), r["status"], r["n_steps"], r["n_windows"], f(r["slowdown_pct"]),
            f(r["slowdown_ci_lo"]), f(r["slowdown_ci_hi"]), f(r["offered_gbps"]), f(r["ready_p50_ms"], 1), f(r["ready_p95_ms"], 1),
            f(r["full_release_p50_ms"], 1), f(r["full_window_ms"], 1), f(r["backlog_end_bytes"] / 1e6 if r["backlog_end_bytes"] == r["backlog_end_bytes"] else float("nan"), 1),
            f(r["backlog_slope_bytes_per_release"] / 1e6 if r["backlog_slope_bytes_per_release"] == r["backlog_slope_bytes_per_release"] else float("nan"), 2),
            f(r["lateness_p50_ms"], 3), f(r["lateness_p95_ms"], 3), f(r["lateness_max_ms"], 3), f(r["late_releases"]), f(r["handoff_p50_ms"], 3),
            f(r["handoff_p95_ms"], 3), f(r["d2h_p50_ms"], 3), f(r["d2h_p95_ms"], 3), f(r["coord_duty"], 3), f(r["pack_duty"], 3), f(r["decode_duty"], 3),
            f(r["done_gbps_in_window"]), "" if r["schedule_identical"] is None else ("YES" if r["schedule_identical"] else "NO")))
    L += ["- rows -> ccurve_hostpack.csv", ""]
    return L


# ------------------------------------------------------------------------------------------------------------ main
def write_result(tag, rec):
    path = os.path.join(OUT, "%s.result.json" % tag)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, default=str)
    os.replace(path + ".tmp", path)
    return path


def _cell_class(runner, e, rc):
    cx = CV.classify_exception(e)
    if isinstance(e, _Uncertified):
        return "UNCERTIFIED"
    return runner.stop_class or (cx if (rc == CT.RC_MEMGATE or (rc == CT.RC_CRASH and cx in CV.HOST_CLASSES)) else
                                 {CT.RC_HYGIENE: "RESTART_PROOF", CT.RC_PLACEMENT: "PLACEMENT", CT.RC_CRASH: "CRASH", CT.RC_CORRECT: "CORRECTNESS"}.get(rc, "CRASH"))


def main_controls():
    """The CONTROLS process: cpupack_transport's destructive controls, the tick controls and the POOL controls at B = CC_BATCH
    (small) and C = CC_CONTROLS_C (the engine imported with that pool)."""
    os.makedirs(OUT, exist_ok=True)
    VA.check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    P = PS.pool_of(CONTROLS_C)
    corpus = VA.load_corpus(path, BATCH)
    model = load_model_pool(path, P)
    ids, docs, distinct = VA.pick_batch(corpus, BATCH, 0)
    tag = "ctl_b%d_c%d" % (BATCH, CONTROLS_C)
    CT.TAG = tag
    fn = os.path.join(OUT, "%s.json" % tag)

    def flush(p):
        with open(fn + ".tmp", "w") as f:
            json.dump(p, f, default=str)
        os.replace(fn + ".tmp", fn)
    runner = CacheCurveRunner(model, ids, docs, distinct, flush)
    runner.cell_C, runner.cell_P = CONTROLS_C, P
    try:
        rc = runner.run_controls()
    except Exception as e:
        rc = CT.exit_code_for(runner, e)
        runner.crash = dict(kind=_cell_class(runner, e, rc), exit=rc, error=traceback.format_exc()[-4000:])
        traceback.print_exc()
        try:
            runner.flush_payload(False)
        except Exception:
            pass
    print("[ccurve] CONTROLS b%d c%d rc=%d" % (BATCH, CONTROLS_C, rc), flush=True)
    return rc


def main():
    if MODE == "table":
        return table_ccurve(OUT)
    if MODE == "controls":
        return main_controls()
    os.makedirs(OUT, exist_ok=True)
    why = CT.placement_problem(EARLY)
    if why:
        print("[ccurve] PLACEMENT REFUSED (exit %d): %s; placement %s" % (CT.RC_PLACEMENT, why, json.dumps(EARLY)), flush=True)
        return CT.RC_PLACEMENT
    VA.check_budget()
    B = BATCH
    todo = [C for C in CAPS if not os.path.exists(os.path.join(OUT, "%s.result.json" % tag_of(B, C)))]
    if not todo:
        print("[ccurve] every cell of B=%d %s already has a result marker" % (B, CAPS), flush=True)
        return 0
    path = os.environ["NOSI_MODEL_PATH"]
    corpus = VA.load_corpus(path, B)
    model = load_model_pool(path, max(PS.pool_of(C) for C in todo))
    base_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    ids, docs, distinct = VA.pick_batch(corpus, B, 0)
    print("[ccurve] B=%d L=%d Ncap=%d cells %s (todo %s) arms %s paced %s x %d docs %s%s (%d distinct) placement %s" % (
        B, VA.L, VA.N, CAPS, todo, ARMS, PACED, PACED_TICKS, docs[:6], "..." if len(docs) > 6 else "", distinct, json.dumps(EARLY)), flush=True)
    runner = CacheCurveRunner(model, ids, docs, distinct, lambda p: None)
    total, took, prev_alloc = 0, [], None
    for idx, C in enumerate(todo):
        tag = tag_of(B, C)
        CT.TAG = tag
        fn = os.path.join(OUT, "%s.json" % tag)

        def flush(p, fn=fn):
            with open(fn + ".tmp", "w") as f:
                json.dump(p, f, default=str)
            os.replace(fn + ".tmp", fn)
        est = (max(took) if took else CELL_EST_S) + (CELL_EST_FIXED_S * B if idx == 0 else 0)
        if CT.STAGE_DEADLINE and time.time() + est > CT.STAGE_DEADLINE:
            for C2 in todo[idx:]:
                write_result(tag_of(B, C2), dict(batch=B, C=C2, tag=tag_of(B, C2), rc=1, est_s=est, left_s=CT.STAGE_DEADLINE - time.time(), **{"class": "SKIPPED_DEADLINE"}))
                total += 1
            print("[ccurve] cells %s SKIPPED: estimated %.0f s > %.0f s left" % (todo[idx:], est, CT.STAGE_DEADLINE - time.time()), flush=True)
            break
        runner.begin_cell(C, flush)
        t_c = time.time()
        cls, err, rc = "OK", None, 0
        proc_deadline = CT.STAGE_DEADLINE
        if proc_deadline:                                                # an equal share of what is left per remaining cell: no
            pre = CELL_EST_FIXED_S * B if idx == 0 else 0.0             # capacity is starved by an earlier one (the measured
            share = max(0.0, (proc_deadline - time.time() - pre) / (len(todo) - idx))   # pass stops its gated steps at it)
            CT.STAGE_DEADLINE = min(proc_deadline, time.time() + pre + share)
            runner.payload_extra["cell_deadline"] = dict(process=proc_deadline, cell=CT.STAGE_DEADLINE, share_s=share)
        try:
            rc = runner.run_cell(C, first=(idx == 0))
            if rc:
                cls = "FAILED_CHECKS" if rc < 20 else {CT.RC_CORRECT: "CORRECTNESS"}.get(rc, "RC%d" % rc)
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            rc = CT.RC_CORRECT if isinstance(e, _Uncertified) else CT.exit_code_for(runner, e)
            cls = _cell_class(runner, e, rc)
            err = "%s: %s" % (type(e).__name__, str(e)[:600])
            if rc == CT.RC_MEMGATE:
                runner.memgate.append(dict(trigger=err, cls=cls))
            else:
                runner.crash = dict(kind=cls, exit=rc, error=traceback.format_exc()[-4000:])
            print("[ccurve] B=%d C=%d stopped (exit %d, %s): %s" % (B, C, rc, cls, err), flush=True)
            traceback.print_exc()
            try:
                runner.flush_payload(False)
            except Exception:
                pass
        CT.STAGE_DEADLINE = proc_deadline
        res = dict(batch=B, C=C, P=PS.pool_of(C), tag=tag, rc=rc, error=err, fails=getattr(runner, "fails", None), seconds=time.time() - t_c,
                   partial=(runner.curve or {}).get("partial"), first_in_process=(idx == 0), probe=(idx == 0 and C == max(CAPS)), **{"class": cls})
        try:
            if torch.cuda.is_available():
                res.update(peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
            res["pinned"] = runner.pinned_inventory()
        except Exception:
            pass
        runner.end_cell()
        alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        res["allocated_after_cell_gb"] = alloc / 1e9
        eng = sum(t.numel() * t.element_size() for e in getattr(runner, "engines", []) for t in (e._k_gpu, e._v_gpu, e._kv_bias_gpu)) if hasattr(runner, "engines") else 0
        res["engine_window_pool_gb"] = eng / 1e9
        leak = None if prev_alloc is None else ((alloc - eng) - prev_alloc) / 1e9
        res["leak_vs_previous_cell_gb"] = leak
        prev_alloc = alloc - eng
        write_result(tag, res)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        took.append(time.time() - t_c)
        total += rc if rc < 20 else 1
        print("[ccurve] B=%d C=%d done rc=%d class=%s (%.0f s); allocated %.2f GB after the cell (window + pool %.2f GB)" % (
            B, C, rc, cls, res["seconds"], alloc / 1e9, eng / 1e9), flush=True)
        if rc in (CT.RC_MEMGATE, CT.RC_HYGIENE, CT.RC_CRASH, CT.RC_PLACEMENT) or (leak is not None and leak > LEAK_LIMIT_GB):
            print("[ccurve] STOP after C=%d (rc %d%s): the sbatch decides (fit-probe fallback on a capacity class; otherwise a new process resumes the "
                  "cells without a marker)" % (C, rc, "" if leak is None or leak <= LEAK_LIMIT_GB else ", leak %.2f GB" % leak), flush=True)
            IC.teardown(runner, model, base_alloc)
            return rc if rc >= 20 else CT.RC_CRASH
    td = IC.teardown(runner, model, base_alloc)
    print("[ccurve] process done: teardown %.2f GB allocated" % td["allocated_gb"], flush=True)
    return min(total, 19)


if __name__ == "__main__":
    sys.exit(main())
