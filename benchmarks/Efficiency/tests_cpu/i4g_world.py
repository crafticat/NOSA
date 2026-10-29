"""The CPU WORLD of tests_cpu/test_i4g.py (rung I4G, lever 3: the arm L4PTFDG = L4PTFD with the per-layer union + fork
section of the joint pass replayed from CUDA graphs, benchmarks/Efficiency/r2_i4g_graph.py; not a test file). Load it by
file path, as the older tests load their worlds. It builds on tests_cpu/i4a_world.py (read its docstring first): the same
synthetic package, stand-ins, arena and lean classes.

CUDA GRAPHS DO NOT RUN ON THE CPU. FakeGraphs is a stand-in with r2_i4g_graph.CudaGraphs' interface that keeps the three
properties the hook relies on, so the static-input plumbing, the host-count bookkeeping, the frozen branches, the
address rules and the eager fallback are tested; it does NOT establish GPU capture legality, dispatch, timing or memory
(the GPU miniature i4gm and the timing cells do):
  1. CAPTURE EXECUTES NOTHING ON THE STATE: the captured Python runs once (its host counts move, as on the GPU) under a
     TorchDispatchMode; every tensor that an op writes is saved before its first write and restored after the capture.
  2. A REPLAY RE-EXECUTES THE RECORDED OPS ON THE CAPTURED TENSOR OBJECTS (fixed addresses): a write op runs again on
     its recorded arguments; an out-of-place op runs again and its result is copied into the tensor it returned at the
     capture (later ops hold that tensor); a view op is skipped (its alias stays; an alias-annotated op whose output
     does not share an input's storage, e.g. reshape of an expanded tensor, counts as out-of-place: _aliases). So an
     input that the hook did not copy
     into a static buffer is read from the capture pass (stale), as on the GPU.
  3. KERNELS ARE NODES: the stand-ins of the host gathers (KVMeter.orig) and of the fork (VerifyOps.fork) are recorded
     as one call each and called again at every replay (they read their tensors then, as a kernel reads memory).
     Outside a node a host read or a data-dependent shape (ILLEGAL) raises: such an op cannot be captured on the GPU.

THIS FILE ADDS
  GR, TentStateFDG, TentRunnerFDG   r2_i4g_graph.graph_classes(I3.STF, (TentStateFD, TentRunnerFD))
  FakeGraphs, FakeGraph            the stand-in above
  gfd_builder(...)                 L4PTFD (graph False) or L4PTFDG with FakeGraphs (gneg: a graph control)
  run_arm(...)                     one run (no hooks, no test record: the GPU cell's runner arguments)
  cell_inputs(ev)                  the cell's EQF / COUNTS inputs from a CPU run
  eager_op_count(runner)           an INDEPENDENT count of the EAGER aten ops of every joint pass (a FakeGraph replay
                                   is not counted: on the GPU it is one graph launch)
"""
from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X4 = _load("i4a_world")
I3, W, WW, SW, SR, BC, VR, STF, KA, RE, RW, Z, WB, LN = (X4.I3, X4.W, X4.WW, X4.SW, X4.SR, X4.BC, X4.VR, X4.STF,
                                                         X4.KA, X4.RE, X4.RW, X4.I3.Z, X4.WB, X4.LN)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4g_graph as GR         # noqa: E402

TentStateFD, TentRunnerFD = X4.TentStateFD, X4.TentRunnerFD
TentStateFDG, TentRunnerFDG = GR.graph_classes(STF, (TentStateFD, TentRunnerFD))
L, BS, TOPK, V = W.L, W.BS, W.TOPK, W.V
K0, N0, S0 = WW.K0, WW.N0, WW.S0
ILLEGAL = {"_local_scalar_dense", "item", "is_nonzero", "nonzero", "masked_select", "unique", "_unique2", "unique_dim",
           "unique_consecutive", "equal"}


# ---------------------------------------------------------------------------
# the CUDA graph stand-in
# ---------------------------------------------------------------------------
def _is_view(func) -> bool:
    return any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns)


def _written(func, args, kwargs):
    """The tensors that the op writes (the schema's write alias annotations)."""
    out = []
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is None or not a.alias_info.is_write:
            continue
        v = args[i] if i < len(args) else kwargs.get(a.name)
        if torch.is_tensor(v):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [t for t in v if torch.is_tensor(t)]
    return out


def _aliases(out, args, kwargs) -> bool:
    """The op's tensor outputs share storage with a tensor input (a real view). An alias-annotated composite op (e.g.
    reshape of an expanded tensor) may COPY: then it is replayed as an out-of-place op."""
    ins = set()
    for v in list(args) + list(kwargs.values()):
        for t in (v if isinstance(v, (list, tuple)) else [v]):
            if torch.is_tensor(t):
                ins.add(t.untyped_storage().data_ptr())
    outs = [t for t in (out if isinstance(out, (tuple, list)) else [out]) if torch.is_tensor(t)]
    return bool(outs) and all(o.untyped_storage().data_ptr() in ins for o in outs)


def _into(out, res):
    if torch.is_tensor(out):
        assert tuple(out.shape) == tuple(res.shape) and out.dtype == res.dtype, (out.shape, res.shape)
        out.copy_(res)
    elif isinstance(out, (tuple, list)):
        for o, r in zip(out, res):
            _into(o, r)
    else:
        assert out == res, (out, res)


class _Recorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops, self.saved, self.seen, self.depth = [], [], set(), 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        if self.depth == 0 and name in ILLEGAL:
            raise RuntimeError("FakeGraphs: %s inside the captured section (a host read or a data-dependent shape: not "
                               "capturable)" % name)
        wr = _written(func, args, kwargs)
        for t in wr:                                                   # property 1: saved before the first write
            if id(t) not in self.seen:
                self.seen.add(id(t))
                self.saved.append((t, t.clone()))
        out = func(*args, **kwargs)
        if self.depth == 0:
            kind = "mut" if wr else ("view" if (_is_view(func) and _aliases(out, args, kwargs)) else "pure")
            self.ops.append(("op", func, args, kwargs, out, kind))
        return out

    def node(self, fn):
        """Property 3: fn recorded as ONE kernel call; its ops run (and their writes are restored) but are not recorded."""
        def call(*a, **k):
            self.ops.append(("kernel", fn, a, k))
            self.depth += 1
            try:
                return fn(*a, **k)
            finally:
                self.depth -= 1
        return call


class FakeGraph:
    def __init__(self, ops):
        self.ops = ops
        self.replays = 0

    def replay(self):
        """Property 2: the recorded ops on the captured tensor objects."""
        with torch.inference_mode():
            for e in self.ops:
                if e[0] == "kernel":
                    e[1](*e[2], **e[3])
                    continue
                _, func, args, kwargs, out, kind = e
                if kind == "view":
                    continue
                res = func(*args, **kwargs)
                if kind == "pure":
                    _into(out, res)
        self.replays += 1


class FakeGraphs:
    """r2_i4g_graph.CudaGraphs' interface on the CPU (module docstring)."""

    def __init__(self):
        self.graphs = []
        self.capture_s = 0.0
        self.restored = 0

    def capture(self, fn, T, vs):
        rec = _Recorder()
        meter = T.meter
        orig_m, orig_f = meter.orig, vs.ops.fork
        meter.orig = tuple(rec.node(f) for f in orig_m)
        vs.ops.fork = rec.node(orig_f)
        try:
            with rec:
                out = fn()
        finally:
            meter.orig, vs.ops.fork = orig_m, orig_f
            for t, c in reversed(rec.saved):                          # property 1: the capture executes nothing
                t.copy_(c)
            self.restored += len(rec.saved)
        g = FakeGraph(rec.ops)
        self.graphs.append(g)
        return g, out

    def pool_bytes(self):
        return 0

    def release(self):
        self.graphs = []


# ---------------------------------------------------------------------------
# builders and drivers
# ---------------------------------------------------------------------------
def gfd_builder(graph: bool, gemm="natural", check=False, stats=False, dneg=None, gneg=None, X=None,
                phase_marks=False, probe_every=0, backend=None):
    """L4PTFD (graph False: r2_i4a_lean's TentRunnerFD) or L4PTFDG (TentRunnerFDG with FakeGraphs) with the GPU cell's
    runner arguments (no hooks, no test record; check / stats: union_check / union_stats)."""
    fa, fk = WB.fake_attention_u, Z.fake_fork

    def build(model, cache, S, horizon, tok0, K=K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, X or KA.store_slots(TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, record=record, probe_every=probe_every)
        kw = dict(fork=fk, t0=tok0, hooks=None, check_every=check_every, union_check=check, union_stats=stats,
                  phase_marks=phase_marks, dneg=dneg)
        if graph:
            return TentRunnerFDG(model, rs, sops, BC.HostClock(), gneg=gneg,
                                 graph_backend=backend if backend is not None else FakeGraphs(), **kw)
        return TentRunnerFD(model, rs, sops, BC.HostClock(), **kw)
    return build


def world(B: int, N: int, seed: int = 0):
    """(model, cache, S, horizon, tok0): a new synthetic cache (two calls with one seed give the same world)."""
    S = S0
    hz = SW.horizon_for(N, K0)
    model, _, c = SW.two_caches(B, S, hz + 64, seed)
    tok0 = torch.randint(0, V, (B,), generator=torch.Generator().manual_seed(seed + 2))
    return model, c, S, hz, tok0


SCEN = dict(
    timing=dict(check_every=4, probe_every=4, phase_marks=True, check=False, stats=False, fa=False),   # i4ga / i4gb
    check=dict(check_every=1, probe_every=4, phase_marks=True, check=True, stats=True, fa=True),       # i4gm
    toggle=dict(check_every=3, probe_every=0, phase_marks=False, check=False, stats=False, fa=False),  # no_key's world
)


def run_arm(graph: bool, scen="timing", B=8, N=None, seed=0, gneg=None, gemm="natural", on_runner=None):
    """One run of L4PTFD or L4PTFDG in scenario SCEN[scen]. Returns dict(runner, rd)."""
    s = SCEN[scen]
    N = N or N0
    model, c, S, hz, tok0 = world(B, N, seed)
    run = gfd_builder(graph, gemm=gemm, check=s["check"], stats=s["stats"], gneg=gneg, phase_marks=s["phase_marks"],
                      probe_every=s["probe_every"])(model, c, S, hz, tok0, check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    if on_runner is not None:
        on_runner(run)
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen)


def cell_inputs(ev, C):
    """The cell's EQF / COUNTS inputs (r2_i4g_cell) from a CPU run; C the cell module."""
    r = ev["runner"]
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=C.IA.acct_record(ev["rd"]),
                state=C.IA.state_checksum(r), counts=C.counts_record(ev["rd"]))


class _OpCount(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.names = collections.Counter()
        self.off = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if not self.off and not X4._is_view(func) and name not in X4._NOT_COMPUTE:
            self.n += 1
            self.names[name] += 1
        return func(*args, **(kwargs or {}))


_REPLAY = FakeGraph.replay


def restore_replay():
    FakeGraph.replay = _REPLAY


def eager_op_count(runner) -> dict:
    """Wraps runner.T.run_pass (an instance attribute): per JOINT pass the EAGER launches: aten compute ops (views and
    empty allocations excluded) plus ONE per kernel stand-in call (diff, pool_update, pool_swap, the two host gathers,
    the fork: their internal ops are not counted, as a GPU kernel is one launch); a FakeGraph replay is not counted
    (box["replays"]: the replays per pass; on the GPU each is one graph launch). Call restore_replay() after the run."""
    T, rs, vs = runner.T, runner.rs, runner.vs
    box = dict(joint=[], replays=[])
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
    vs.ops.fork = kernel(vs.ops.fork)

    def run_pass(model, plan):
        if plan.R == 1:
            return run0(model, plan)
        cnt = _OpCount()
        cnt_box["c"] = cnt
        n_rep = sum(g.replays for g in graphs())
        try:
            with cnt:
                out = run0(model, plan)
        finally:
            cnt_box.pop("c", None)
        box["joint"].append(cnt.n)
        box["replays"].append(sum(g.replays for g in graphs()) - n_rep)
        return out
    T.run_pass = run_pass

    def replay(self_):
        c = cnt_box.get("c")
        if c is not None:
            c.off += 1
        try:
            return _REPLAY(self_)
        finally:
            if c is not None:
                c.off -= 1
    FakeGraph.replay = replay
    return box
