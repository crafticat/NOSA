"""The CPU WORLD of tests_cpu/test_i4h.py (rung I4H, the draft-step graph: the arm L4PTFDGD = L4PTFDG with the one-row
passes replayed from per-layer CUDA graphs, benchmarks/Efficiency/r2_i4h_graph.py; not a test file). Load it by file
path, as the older tests load their worlds. It builds on tests_cpu/i4g_world.py (read its docstring first): the same
synthetic package, stand-ins, arena, lean and graph classes, and its FakeGraphs.

CUDA GRAPHS DO NOT RUN ON THE CPU. FakeGraphsD adds to i4g_world.FakeGraphs the SPLIT capture of r2_i4h_graph's
backend (begin() / end() / side()) with the same three properties (capture executes nothing on the state; a replay
re-executes the recorded ops on the captured tensor objects; kernels are nodes, and outside a node a host read raises),
and one more node kind: a stand-in whose result is a NEW tensor (the fake attention T.attn and the fake stage-1 op
stage1_ext.varlen_fwd_stage1: they read lengths on the host, as a GPU kernel reads them on the device) is recorded as
ONE call and, at a replay, called again with its recorded arguments and its result copied into the tensor it returned at
the capture (later ops hold that tensor). It does NOT establish GPU capture legality, dispatch, timing or memory (the
GPU miniature i4hm and the timing cells do).

THIS FILE ADDS
  DGR, TentStateFDGD, TentRunnerFDGD   r2_i4h_graph.draft_classes(STF, (TentStateFDG, TentRunnerFDG))
  FakeGraphsD, FakeGraphD             the stand-in above
  arm_builder(kind, ...)              "D" = L4PTFD, "G" = L4PTFDG (FakeGraphs), "GD" = L4PTFDGD (FakeGraphsD)
  run_arm(kind, ...)                  one run (no hooks, no test record: the GPU cell's runner arguments)
  cell_inputs(ev, C)                  the cell's EQF / COUNTS inputs from a CPU run
  draft_op_count(runner)              an INDEPENDENT count of the EAGER launches of every one-row pass (a FakeGraphD
                                      replay is not counted: on the GPU it is one graph launch)
"""
from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


XG = _load("i4g_world")
X4, I3, W, WW, SW, SR, BC, VR, STF, KA, RE, RW, Z, WB, LN, GR = (
    XG.X4, XG.I3, XG.W, XG.WW, XG.SW, XG.SR, XG.BC, XG.VR, XG.STF, XG.KA, XG.RE, XG.RW, XG.Z, XG.WB, XG.LN, XG.GR)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4h_graph as DGR        # noqa: E402

TentStateFD, TentRunnerFD = XG.TentStateFD, XG.TentRunnerFD
TentStateFDG, TentRunnerFDG = XG.TentStateFDG, XG.TentRunnerFDG
TentStateFDGD, TentRunnerFDGD = DGR.draft_classes(STF, (TentStateFDG, TentRunnerFDG))
L, BS, TOPK, V = W.L, W.BS, W.TOPK, W.V
K0, N0, S0 = WW.K0, WW.N0, WW.S0
SCEN = XG.SCEN


# ---------------------------------------------------------------------------
# the split-capture stand-in
# ---------------------------------------------------------------------------
class _RecorderD(XG._Recorder):
    def node_out(self, fn):
        """A stand-in whose result is a new tensor: ONE recorded call; a replay copies its new result into the tensor
        that the capture returned (module docstring)."""
        def call(*a, **k):
            self.depth += 1
            try:
                out = fn(*a, **k)
            finally:
                self.depth -= 1
            self.ops.append(("kernel_out", fn, a, k, out))
            return out
        return call


class FakeGraphD(XG.FakeGraph):
    def replay(self):
        with torch.inference_mode():
            for e in self.ops:
                if e[0] == "kernel":
                    e[1](*e[2], **e[3])
                    continue
                if e[0] == "kernel_out":
                    XG._into(e[4], e[1](*e[2], **e[3]))
                    continue
                _, func, args, kwargs, out, kind = e
                if kind == "view":
                    continue
                res = func(*args, **kwargs)
                if kind == "pure":
                    XG._into(out, res)
        self.replays += 1


class FakeGraphsD(XG.FakeGraphs):
    """r2_i4h_graph.make_backend(GR)'s interface on the CPU: capture() (the joint pass, i4g_world's), begin() / end() /
    abort() / side() (the one-row passes). T: the runner's state (set by arm_builder)."""

    def __init__(self):
        super().__init__()
        self.T = None
        self._open = None
        self.dcapture_s = 0.0
        self.dcaptures = 0
        self.sides = 0

    def _nodes(self):
        T = self.T
        out = [(T, "attn")]
        ext = T.rs.ops.kernels.get("stage1_ext")
        if ext is not None:
            out.append((ext, "varlen_fwd_stage1"))
        return out

    def begin(self):
        assert self._open is None, "a capture is open"
        rec = _RecorderD()
        patched = []
        for obj, name in self._nodes():
            orig = getattr(obj, name)
            patched.append((obj, name, orig))
            setattr(obj, name, rec.node_out(orig))
        rec.__enter__()
        self._open = (rec, patched)

    def end(self):
        rec, patched = self._open
        self._open = None
        try:
            rec.__exit__(None, None, None)
        finally:
            for obj, name, orig in reversed(patched):
                setattr(obj, name, orig)
            for t, c in reversed(rec.saved):                          # property 1: the capture executes nothing
                t.copy_(c)
            self.restored += len(rec.saved)
        g = FakeGraphD(rec.ops)
        self.graphs.append(g)
        self.dcaptures += 1
        return g

    def abort(self):
        if self._open is not None:
            try:
                self.end()
            except Exception:
                pass

    def side(self, fn):
        self.sides += 1
        return fn()


# ---------------------------------------------------------------------------
# builders and drivers
# ---------------------------------------------------------------------------
def arm_builder(kind: str, gemm="natural", check=False, stats=False, gneg=None, dgneg=None, phase_marks=False,
                probe_every=0):
    """"D" L4PTFD, "G" L4PTFDG (FakeGraphs), "GD" L4PTFDGD (FakeGraphsD) with the GPU cell's runner arguments."""
    if kind in ("D", "G"):
        return XG.gfd_builder(kind == "G", gemm=gemm, check=check, stats=stats, gneg=gneg, phase_marks=phase_marks,
                              probe_every=probe_every)
    assert kind == "GD", kind
    fa, fk = WB.fake_attention_u, Z.fake_fork

    def build(model, cache, S, horizon, tok0, K=K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, KA.store_slots(TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, record=record, probe_every=probe_every)
        be = FakeGraphsD()
        run = TentRunnerFDGD(model, rs, sops, BC.HostClock(), fork=fk, t0=tok0, hooks=None, check_every=check_every,
                             union_check=check, union_stats=stats, phase_marks=phase_marks, dneg=None, gneg=gneg,
                             dgneg=dgneg, graph_backend=be)
        be.T = run.T
        return run
    return build


def run_arm(kind: str, scen="timing", B=8, N=None, seed=0, gneg=None, dgneg=None, gemm="natural", on_runner=None):
    """One run of arm `kind` in scenario SCEN[scen] (i4g_world's). Returns dict(runner, rd)."""
    s = SCEN[scen]
    N = N or N0
    model, c, S, hz, tok0 = XG.world(B, N, seed)
    run = arm_builder(kind, gemm=gemm, check=s["check"], stats=s["stats"], gneg=gneg, dgneg=dgneg,
                      phase_marks=s["phase_marks"], probe_every=s["probe_every"])(model, c, S, hz, tok0,
                                                                                check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    if on_runner is not None:
        on_runner(run)
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen)


def cell_inputs(ev, C):
    """The cell's EQF / COUNTS inputs (r2_i4h_cell) from a CPU run; C the cell module."""
    r = ev["runner"]
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=C.IA.acct_record(ev["rd"]),
                state=C.IA.state_checksum(r), counts=C.GC.counts_record(ev["rd"]))


_REPLAY_D = FakeGraphD.replay


def restore_replay():
    FakeGraphD.replay = _REPLAY_D
    XG.restore_replay()


def draft_op_count(runner) -> dict:
    """Wraps runner.T.run_pass (an instance attribute): per ONE-ROW pass the EAGER launches: aten compute ops (views
    and empty allocations excluded) plus ONE per kernel stand-in call (diff, pool_update, pool_swap, the two host gathers:
    their internal ops are not counted); a FakeGraphD replay is not counted (box["replays"]: the replays per pass; on the
    GPU each is one graph launch). Call restore_replay() after the run."""
    T, rs = runner.T, runner.rs
    box = dict(passes=[], replays=[], captures=[], names=[])
    run0 = T.run_pass
    cnt_box = {}

    def graphs():
        return getattr(getattr(T, "backend", None), "graphs", [])

    def kernel(fn):
        def call(*a, **k):
            c = cnt_box.get("c")
            if c is not None and not c.off:
                c.n += 1
            if c is not None:
                c.off += 1
            try:
                return fn(*a, **k)
            finally:
                if c is not None:
                    c.off -= 1
        return call
    kern = rs.ops.kernels
    for name in ("diff", "pool_update", "pool_swap"):
        if name in kern:
            kern[name] = kernel(kern[name])
    runner.meter.orig = tuple(kernel(f) for f in runner.meter.orig)

    def run_pass(model, plan):
        if plan.R != 1:
            return run0(model, plan)
        cnt = XG._OpCount()
        cnt_box["c"] = cnt
        n_rep = sum(g.replays for g in graphs())
        n_cap = getattr(getattr(T, "backend", None), "dcaptures", 0)
        try:
            with cnt:
                out = run0(model, plan)
        finally:
            cnt_box.pop("c", None)
        box["passes"].append(cnt.n)
        box["replays"].append(sum(g.replays for g in graphs()) - n_rep)
        box["captures"].append(getattr(getattr(T, "backend", None), "dcaptures", 0) - n_cap)
        box["names"].append(collections.Counter(cnt.names))
        return out
    T.run_pass = run_pass

    def replay(self_):
        c = cnt_box.get("c")
        if c is not None:
            c.off += 1
        try:
            return _REPLAY_D(self_)
        finally:
            if c is not None:
                c.off -= 1
    FakeGraphD.replay = replay
    return box
