"""Rung I4G of KS/analysis/plan/PLAN_L4P_faithful.md (KS = /mnt/beegfs/ojerbi/kstep-cpu-runs), LEVER 3 of the
specKV-on-cached-NOSA ladder: ARM L4PTFDG = L4PTFD (rung I4A: L4PTF with the lean one-row pass, r2_i4a_lean.py) whose
per-layer union + fork section of the JOINT pass is replayed from CUDA graphs. Docs: docs/NOTES_i4g.md. OPT-IN:
benchmarks/Efficiency/r2_i4g_cell.py (arm L4PTFDG); no nosi module and no older cell imports this file. NO NEW KERNEL
and no kernel edit: torch.cuda.CUDAGraph (the machinery of ragged_engine.py:346-370 and r2_tick_capture.py:73-76),
the shipped kernels and torch ops. This file imports torch only.

WHY A CLASS FACTORY (as r2_i4a_lean.lean_classes): graph_classes(STF, base) builds the classes on the GIVEN
spec_tentative_fork module and on the GIVEN lean pair (TentStateFD, TentRunnerFD) of r2_i4a_lean.lean_classes(STF).

WHAT IS CAPTURED (one CUDA graph per layer l, all graphs in ONE shared memory pool): THE WHOLE
spec_tentative_fork.TentStateF._rows_union (STF:124-244, nosi unchanged) EXCEPT its residency call (STF:149). The
residency stays EAGER: pool_update takes the host int eng._pool_stamp_base by value and then adds topk on the host
(spec_tentative.py:461-466); a replay would keep the captured stamp, and moving the stamp to the device is a kernel edit
(pool_update_kernel.cu; not allowed). So per layer and joint pass:
  1. EAGER (TentStateFDG._resid_eager): the residency inputs of STF:135-148 (qb, the head's tail-id injection, ti) and
     the residency itself (TentStateU._resid -> TentState._resid: its host count, the shipped kernels, the host stamp).
  2. the static inputs (one shared set, filled by copy_ before every replay): q4, k4, v4 (the GEMM / nosa_linear / rope
     outputs of the layer: new tensors each layer), the rows' positions pbs (twin.pos.clone() per row: new tensors), and
     per pass the plan's act / verify / draft / deferred masks (new tensors each round).
  3. REPLAY of layer l's graph: STF:127-244 on the static inputs with the residency call answered by a stub that returns
     the static skip mask (the eager residency's result, copied in when the accounting reads it). The captured code
     recomputes STF:135-148 (qb, P, past, hasq, the injection; their values are the eager ones: no input of theirs moves
     in the residency) and runs STF:150-244: the buffer writes, the chain images, the locate, the store fill (the
     shipped host gathers), the row lists, the ONE fork call, and with ops.stats / ops.check their device counters.
WHEN A LAYER IS EAGER (STF's code on the real tensors, as L4PTFD): a probe pass (phase marks, vs.marks), the meter's
timing (KVMeter spans), hooks, the test records (T.rec, ops.record), the "F" / "RA" probe of layer 0 (a host sync), a
pass whose frozen key differs from the captured one. THE ORDER per layer: the first eligible call runs EAGERLY on the
static inputs (Triton compiles every kernel with the capture's pointers outside the capture: ragged_engine.py:362); the
second CAPTURES (capture executes nothing) and replays; every later eligible call replays.
THE FROZEN KEY (Python branches inside the captured code): T.checking (TentState._write's check-only counters), the
meter's `on` (KVMeter._gk counts host blocks), plan.deferred is None (the D3 counter), plan.R. Fixed for a runner: tneg,
fneg, ops.check / stats / neg, rs.pool.
HOST COUNTS that a replay does not run (CODE: STF:121 chain, STF:175-176 gather_host 2 and fills 1, STF:185 attn 1;
VR:1175 stat_n union with ops.stats): the capture measures them (the counter dicts before and after the captured Python)
and every replay adds that delta (SKIPPED: the expected delta; the cell's L4PTFDG-GRAPH gate checks it). The residency's
host count (STU:101) runs eagerly.
ADDRESSES: verify_rows reassigns two accounting entries (VR:1163 extras_max, VR:1174 union_max = torch.maximum(...)): the
runner makes vs.stat an InPlaceDict (an item assignment onto a tensor of the same shape / dtype / device copies into
the old tensor), so the stats keep their addresses; the same bits.
MEMORY: one shared graph pool (r2_tick_capture.py:73-76; replays never overlap; no tensor that a graph allocates is
alive after its capture: the section returns a view of the persistent vs.fout); the static inputs (about 8 MB at B 216).
NEGATIVE CONTROLS (tests only; gneg): "stale_input" (q4 is not copied into the static input: a replay reads an old q),
"no_counts" (the skipped host counts are not added), "plain_stats" (vs.stat stays a plain dict), "no_key" (a graph
replays when the frozen key differs).
HOST SYNCS: none in a replayed pass (the capture synchronizes, once per layer).
"""
import time

import torch

GNEG = (None, "stale_input", "no_counts", "plain_stats", "no_key")
# CODE (STF:121, :175-176, :185): the host counts inside the captured code per layer and pass (ops.stats adds VR:1175's
# stat_n["union"] += H x B). TentState.calls "chain"; VerifyState.calls the rest.
SKIPPED_T = dict(chain=1)
SKIPPED_VS = dict(gather_host=2, fills=1, attn=1)
EAGER_OK = ("marks", "timing", "probe_F")          # the expected reasons for an eager layer in the cells


class InPlaceDict(dict):
    """A dict whose item assignment of a tensor onto a tensor of the same shape, dtype and device COPIES into the old
    tensor (its address stays; module docstring, ADDRESSES). Any other assignment is a plain one."""

    def __setitem__(self, k, v):
        old = dict.get(self, k)
        if (old is not None and old is not v and torch.is_tensor(old) and torch.is_tensor(v)
                and old.shape == v.shape and old.dtype == v.dtype and old.device == v.device):
            old.copy_(v)
            return
        dict.__setitem__(self, k, v)


class CudaGraphs:
    """The capture backend: torch.cuda.CUDAGraph with ONE shared pool (r2_tick_capture.py:73-76). capture(fn, T, vs)
    -> (graph with .replay(), fn's result). The CPU tests pass a stand-in with the same interface."""

    def __init__(self):
        self.pool = None
        self.capture_s = 0.0

    def capture(self, fn, T, vs):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn()
        torch.cuda.synchronize()
        self.capture_s += time.perf_counter() - t0
        return g, out

    def pool_bytes(self):
        """The bytes of the allocator segments of the shared pool (torch.cuda.memory_snapshot's segment_pool_id);
        None when the snapshot does not name pools."""
        if self.pool is None:
            return 0
        try:
            segs = torch.cuda.memory_snapshot()
        except Exception:                                              # the report is evidence, not a gate input
            return None
        tot, named = 0, False
        want = tuple(self.pool) if isinstance(self.pool, (tuple, list)) else (self.pool,)
        for s in segs:
            pid = s.get("segment_pool_id")
            if pid is None:
                continue
            named = True
            if (tuple(pid) if isinstance(pid, (tuple, list)) else (pid,)) == want:
                tot += int(s.get("total_size", 0))
        return tot if named else None

    def release(self):
        self.pool = None


def _counts(T, vs):
    return dict(T=dict(T.calls), vs=dict(vs.calls), n=dict(vs.stat_n))


def _delta(a, b):
    return {g: {k: b[g][k] - a[g].get(k, 0) for k in b[g] if b[g][k] != a[g].get(k, 0)} for g in b}


def expected_delta(stats: bool, H: int, B: int) -> dict:
    """SKIPPED as a _delta: the host counts per layer that a replay must add."""
    return dict(T=dict(SKIPPED_T), vs=dict(SKIPPED_VS), n=(dict(union=H * B) if stats else {}))


_CLASSES = {}


def graph_classes(STF, base):
    """(TentStateFDG, TentRunnerFDG) built on the module STF and the lean pair base = (TentStateFD, TentRunnerFD)."""
    key = (id(STF), id(base[0]), id(base[1]))
    if key not in _CLASSES:
        _CLASSES[key] = (STF, base) + _make(STF, base)
    return _CLASSES[key][2:]


def _make(STF, base):
    SpecError = STF.SpecError
    Plan = STF.ST._Plan
    SD, RD = base

    class TentStateFDG(SD):
        """L4PTFD's state whose joint-pass section runs from per-layer CUDA graphs (module docstring)."""

        def __init__(self, rs, attn, tneg=None, poison: bool = False, fneg=None, dneg=None, gneg=None):
            if gneg not in GNEG:
                raise SpecError("unknown graph control %r (one of %s)" % (gneg, GNEG))
            super().__init__(rs, attn, tneg=tneg, poison=poison, fneg=fneg, dneg=dneg)
            self.gneg = gneg
            self.backend = None                                        # set by the runner
            self.meter = None                                          # set by the runner (KVMeter: on / timing)
            self._g = {}                                               # layer -> dict(key, graph, out, delta)
            self._warm = {}                                            # layer -> the key of its eager warm call
            self._sb = None                                            # the static inputs
            self._splan = self._plan_src = None
            self.graph = dict(captures=0, replays=0, warm=0, eager=0, eager_why={}, delta_bad=[], deltas=[],
                              static_bytes=0, capture_passes=[])

        # ---------------------------------------------------------- the policy
        def _why(self, l, vs, plan):
            """None: the layer may run from its graph; else the reason it runs eagerly."""
            if vs.marks is not None:
                return "marks"
            if self.meter is None or self.meter.timing:
                return "timing"
            if self.hooks is not None:
                return "hooks"
            if self.rec is not None or vs.ops.record:
                return "record"
            if vs.fa_check is not None and l == 0 and "F" not in vs.fa_check:
                return "probe_F"
            return None

        def _key(self, plan):
            return (bool(self.checking), bool(self.meter.on), plan.deferred is None, int(plan.R))

        def _static(self, q4, k4, v4, pbs, plan):
            """The static inputs: per layer q4, k4, v4, pbs; per pass the plan's masks (module docstring, step 2)."""
            sb = self._sb
            R, B = plan.R, self.B
            if sb is None:
                dev = q4.device
                sb = self._sb = dict(
                    q=torch.empty(tuple(q4.shape), dtype=q4.dtype, device=dev),
                    k=torch.empty(tuple(k4.shape), dtype=k4.dtype, device=dev),
                    v=torch.empty(tuple(v4.shape), dtype=v4.dtype, device=dev),
                    pb=torch.empty((R, B), dtype=pbs[0].dtype, device=dev),
                    act=torch.empty((R, B), dtype=torch.bool, device=dev),
                    ver=torch.empty((R, B), dtype=torch.bool, device=dev),
                    drf=torch.empty((R, B), dtype=torch.bool, device=dev),
                    dfr=torch.empty((B,), dtype=torch.int64, device=dev),
                    skip=torch.zeros((self.H, B, self.topk), dtype=torch.bool, device=dev))
                sb["pbl"] = [sb["pb"][r] for r in range(R)]
                self.graph["static_bytes"] = sum(t.numel() * t.element_size() for t in sb.values()
                                                 if torch.is_tensor(t))
            if self.gneg != "stale_input":                             # NEGATIVE CONTROL (tests only): q not copied
                sb["q"].copy_(q4)
            sb["k"].copy_(k4)
            sb["v"].copy_(v4)
            torch.stack(pbs, out=sb["pb"])
            if plan is not self._plan_src:                             # a new pass: its masks
                sb["act"].copy_(plan.act)
                sb["ver"].copy_(plan.verify)
                sb["drf"].copy_(plan.draft)
                if plan.deferred is not None:
                    sb["dfr"].copy_(plan.deferred)
                self._plan_src = plan
                self._splan = Plan(R=plan.R, kind=plan.kind, vs=plan.vs, act=sb["act"], verify=sb["ver"],
                                   draft=sb["drf"], deferred=(sb["dfr"] if plan.deferred is not None else None),
                                   pos=None, ids=None, replay=None, replay_rows=(), lin=None, snap=False)
            return sb["q"], sb["k"], sb["v"], sb["pbl"], self._splan

        # ---------------------------------------------------------- the eager residency (step 1)
        def _resid_eager(self, l, eng, twin, sel, plan):
            """STF:135-149's residency with its inputs: the same values (hasq of row 0 = STF:140's hasq[:, :, 0:1])."""
            B = self.B
            qb = self.qc // self.bs
            qb3 = qb.view(1, B, 1)
            s0 = sel[:, :, 0]
            hasq0 = (s0 == qb3).any(-1, keepdim=True)
            inj = (s0 > qb3) & ~hasq0
            ti = torch.where(inj, qb3.expand_as(s0), s0).contiguous()
            return self._resid(l, eng, twin, ti, inj, plan.act[0], plan.draft[0], qb3)

        def _apply(self, delta, vs):
            if self.gneg == "no_counts":                               # NEGATIVE CONTROL (tests only)
                return
            for grp, dst in (("T", self.calls), ("vs", vs.calls), ("n", vs.stat_n)):
                for k, v in delta[grp].items():
                    dst[k] = dst.get(k, 0) + v

        # ---------------------------------------------------------- the joint pass, one layer
        def _rows_union(self, l, eng, twin, sel, q4, k4, v4, pbs, plan):
            vs = plan.vs
            why = self._why(l, vs, plan)
            key = None if why is not None else self._key(plan)
            g = self._g.get(l)
            if why is None and g is not None and g["key"] != key and self.gneg != "no_key":
                why = "key"
            if why is not None:                                        # EAGER: STF's code on the real tensors
                self.graph["eager"] += 1
                self.graph["eager_why"][why] = self.graph["eager_why"].get(why, 0) + 1
                return super()._rows_union(l, eng, twin, sel, q4, k4, v4, pbs, plan)
            sq, sk, sv, spb, splan = self._static(q4, k4, v4, pbs, plan)
            if g is None and self._warm.get(l) != key:                 # WARM: eager on the static inputs
                self._warm[l] = key
                self.graph["warm"] += 1
                return super()._rows_union(l, eng, twin, sel, sq, sk, sv, spb, splan)
            skip = self._resid_eager(l, eng, twin, sel, plan)          # step 1: EAGER (the host stamp)
            if vs.ops.stats:
                self._sb["skip"].copy_(skip)
            if g is None:                                              # CAPTURE (executes nothing), then replay
                sk0 = self._sb["skip"]
                before = _counts(self, vs)
                self._resid = lambda *a, **k: sk0                       # the stub: the residency ran above
                try:
                    graph, out = self.backend.capture(
                        lambda: super(TentStateFDG, self)._rows_union(l, eng, twin, sel, sq, sk, sv, spb, splan),
                        self, vs)
                finally:
                    del self._resid
                d = _delta(before, _counts(self, vs))
                want = expected_delta(bool(vs.ops.stats), self.H, self.B)
                if d != want:
                    self.graph["delta_bad"].append((l, d, want))
                self.graph["deltas"].append((l, d))
                g = self._g[l] = dict(key=key, graph=graph, out=out, delta=d)
                self.graph["captures"] += 1
                self.graph["capture_passes"].append(int(self.calls["joint_passes"]))
                graph.replay()                                         # the counts of this pass ran at the capture
                self.graph["replays"] += 1
                return out
            g["graph"].replay()
            self._apply(g["delta"], vs)
            self.graph["replays"] += 1
            return g["out"]

        def release(self):
            """Drops every graph and the static inputs (the arm's end)."""
            self._g.clear()
            self._warm.clear()
            self._sb = self._splan = self._plan_src = None

    class TentRunnerFDG(RD):
        """L4PTFD (TentRunnerFD) with the joint-pass section replayed from CUDA graphs (arm L4PTFDG). gneg: a graph
        control (tests only); graph_backend: the capture backend (None: CudaGraphs; the CPU tests: a stand-in)."""

        def __init__(self, *args, gneg=None, graph_backend=None, **kw):
            if gneg not in GNEG:                                       # before the meter attaches
                raise SpecError("unknown graph control %r (one of %s)" % (gneg, GNEG))
            self._gneg = gneg
            super().__init__(*args, **kw)
            T = self.T
            T.meter = self.meter
            T.backend = graph_backend if graph_backend is not None else CudaGraphs()
            if gneg != "plain_stats":
                self.vs.stat = InPlaceDict(self.vs.stat)
            self.meta.update(section_graph=dict(gneg=gneg, backend=type(T.backend).__name__, skipped_T=SKIPPED_T,
                                                skipped_vs=SKIPPED_VS))

        def _make_state(self, rs, attn, tneg, poison):
            return TentStateFDG(rs, attn, tneg=tneg, poison=poison, fneg=self._f["fneg"], dneg=self._dneg,
                                gneg=self._gneg)

        def graph_report(self) -> dict:
            T = self.T
            b = T.backend
            pb = b.pool_bytes() if hasattr(b, "pool_bytes") else None
            return dict(T.graph, layers=sorted(T._g), pool_bytes=pb, capture_s=round(getattr(b, "capture_s", 0.0), 3),
                        eager_why=dict(T.graph["eager_why"]))

        @torch.inference_mode()
        def read(self) -> dict:
            out = super().read()
            out.update(graph=self.graph_report())
            return out

        def detach(self) -> None:
            self.T.release()
            if hasattr(self.T.backend, "release"):
                self.T.backend.release()
            super().detach()

    return TentStateFDG, TentRunnerFDG
