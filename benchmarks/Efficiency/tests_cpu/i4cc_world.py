"""The CPU WORLD of tests_cpu/test_i4cc.py (rung I4CC: the compiled arms L4PTFDGUDC and L4PTFDGUDKWC,
benchmarks/Efficiency/r2_i4cc_compile.py; not a test file). Load it by file path, as the older tests load their worlds.
It builds on tests_cpu/i4s_world.py (the stacked arms, FakeGraphsD): read its docstring and i4g_world's first.

CUDA GRAPHS DO NOT RUN ON THE CPU, AND INDUCTOR'S CPU BACKEND IS NOT ITS GPU BACKEND. These tests establish the plumbing
(the compiled helpers in place of the section's eager ops, the host counts, the fallback rules, the compile-event
record), the bit equality of the compiled helpers with the eager section on the CPU backend (C++ kernels generated from
the same graphs: the same graph lowering and fusion decisions of the scheduler, another code generator), and that no
graph breaks. Only the GPU miniature i4cm2 and the timing cells establish Triton's bits, capture legality of the compiled
kernels, timing and memory.

THIS FILE ADDS
  CC                       r2_i4cc_compile; CCL = CC.compile_classes(STF)
  FakeGraphsDC             i4h_world's FakeGraphsD whose joint capture() also makes every compiled helper call ONE
                           kernel node (property 3 of i4g_world: the call runs once at the capture with the recorder
                           left, its written arguments saved and restored after the capture (property 1);
                           a replay calls the compiled helper again on the captured tensor objects and copies its
                           results into the captured outputs)
  c_builder(arm, ...)      the GPU cell's runner arguments for GUDC (L4PTFDGUDC) and GUDKC (L4PTFDGUDKWC)
  run_arm(arm, scen, ...)  one run: the C arms here, every other arm through i4s_world.run_arm
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


XS = _load("i4s_world")
XH, XK, XG = XS.XH, XS.XK, XS.XG
W, WW, SR, BC, STF, KA, RE, RW, Z, WB, I3, SK = (XS.W, XS.WW, XS.SR, XS.BC, XS.STF, XS.KA, XS.RE, XS.RW, XS.Z, XS.WB,
                                                 XS.I3, XS.SK)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4cc_compile as CC      # noqa: E402

CCL = CC.compile_classes(STF)
L, BS, TOPK = W.L, W.BS, W.TOPK
K0, N0 = WW.K0, WW.N0
SCEN = XS.SCEN
world = XS.world
CARMS = dict(GUDC="L4PTFDGUDC", GUDKC="L4PTFDGUDKWC")
KEEP = XS.KEEP + ("GUDKC",)
# the written arguments of each helper (argument positions, r2_i4cc_compile's signatures)
MUT = dict(pre=(), write=(0, 1, 2), loc1=(), loc2=(14, 15), kphys=(), lists=(4, 5, 6))


def _cnode(rec):
    """The node maker of one capture (module docstring, FakeGraphsDC). The recorder is LEFT for the call and entered
    again after it: dynamo does not compile a new frame while a dispatch mode is entered (a popped mode still counts),
    and a compile inside a capture must stay visible to the compile-event record (the control "late")."""
    def wrap(name, f, args):
        rec.__exit__(None, None, None)
        try:
            for i in MUT[name]:
                t = args[i]
                if id(t) not in rec.seen:                            # property 1: saved before the first write
                    rec.seen.add(id(t))
                    rec.saved.append((t, t.clone()))
            out = f(*args)
        finally:
            rec.__enter__()

        def again():
            res = f(*args)
            if out is not None:
                XG._into(out, res)
        rec.ops.append(("kernel", again, (), {}))
        return out
    return wrap


class FakeGraphsDC(XH.FakeGraphsD):
    """FakeGraphsD with the compiled helpers as kernel nodes in the joint capture (module docstring)."""

    def capture(self, fn, T, vs):
        rec = XG._Recorder()
        meter = T.meter
        orig_m, orig_f = meter.orig, vs.ops.fork
        meter.orig = tuple(rec.node(f) for f in orig_m)
        vs.ops.fork = rec.node(orig_f)
        T._cc_wrap = _cnode(rec)
        try:
            with rec:
                out = fn()
        finally:
            T._cc_wrap = None
            meter.orig, vs.ops.fork = orig_m, orig_f
            for t, c in reversed(rec.saved):                          # property 1: the capture executes nothing
                t.copy_(c)
            self.restored += len(rec.saved)
        g = XG.FakeGraph(rec.ops)
        self.graphs.append(g)
        return g, out


def c_builder(arm, gemm="natural", check=False, stats=False, kcheck=False, gneg=None, dgneg=None, hneg=None,
              sneg=None, kneg=None, kslots=None, cneg=None, ccneg=None, cc_backend="inductor", phase_marks=False,
              probe_every=0):
    """The C arm's runner with the GPU cell's runner arguments (no hooks, no test record)."""
    assert arm in CARMS, arm
    fa, fk = WB.fake_attention_u, Z.fake_fork
    Runner = CCL[CARMS[arm]][1]

    def build(model, cache, S, horizon, tok0, K=K0, check_every=0, record=False):
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, KA.store_slots(TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level="L4P", K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, record=record, probe_every=probe_every)
        kw = dict(fork=fk, t0=tok0, hooks=None, check_every=check_every, union_check=check, union_stats=stats,
                  phase_marks=phase_marks, dneg=None, hneg=hneg, gneg=gneg, sneg=sneg, dgneg=dgneg,
                  cc_backend=cc_backend, ccneg=ccneg)
        if arm in KEEP:
            kw.update(keep=True, kwin=True, kcheck=kcheck, kneg=kneg, kslots=kslots, cneg=cneg)
        be = FakeGraphsDC()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, **kw)
        be.T = run.T
        return run
    return build


def run_arm(arm, scen="timing", B=8, N=None, seed=0, gemm="natural", gneg=None, dgneg=None, hneg=None, sneg=None,
            kneg=None, kslots=None, cneg=None, ccneg=None, cc_backend="inductor", on_runner=None):
    """One run of an arm in scenario SCEN[scen]. Returns dict(runner, rd, N, B, scen, arm)."""
    if arm not in CARMS:
        return XS.run_arm(arm, scen, B=B, N=N, seed=seed, gemm=gemm, gneg=gneg, dgneg=dgneg, hneg=hneg, sneg=sneg,
                          kneg=kneg, kslots=kslots, cneg=cneg, on_runner=on_runner)
    CC.configure()
    s = SCEN[scen]
    N = N or N0
    model, c, S, hz, tok0 = world(B, N, seed)
    kc = bool(s["check"]) and arm in KEEP
    run = c_builder(arm, gemm=gemm, check=s["check"], stats=s["stats"], kcheck=kc, gneg=gneg, dgneg=dgneg, hneg=hneg,
                    sneg=sneg, kneg=kneg, kslots=kslots, cneg=cneg, ccneg=ccneg, cc_backend=cc_backend,
                    phase_marks=s["phase_marks"], probe_every=s["probe_every"])(model, c, S, hz, tok0,
                                                                              check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    if on_runner is not None:
        on_runner(run)
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen, arm=arm)


def cell_inputs(ev, C):
    """The cell's gate inputs from a CPU run (C: the cell module; i4s_world's)."""
    return XS.cell_inputs(ev, C)
