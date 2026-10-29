"""The CPU WORLD of tests_cpu/test_i4s.py (rung I4S: the STACKED arms L4PTFDGUD and L4PTFDGUDKW,
benchmarks/Efficiency/r2_i4s_stack.py; not a test file). Load it by file path, as the older tests load their worlds. It
builds on tests_cpu/i4h_world.py (FakeGraphsD: the split capture of the draft graphs, over i4g_world's FakeGraphs) and
tests_cpu/i4k_world.py (the keep arms): read their docstrings first. CUDA GRAPHS DO NOT RUN ON THE CPU: the stand-ins
capture without executing and replay the recorded ops on the captured tensor objects; they establish the plumbing (the
class stack, the hoist inside layer 0's graph, the host counts, the keep inside the graphs, the draft graphs on top), not
GPU capture legality, dispatch, timing or memory (the GPU miniature i4sm and the timing cells do).

THIS FILE ADDS
  SK                       benchmarks/Efficiency/r2_i4s_stack.py; CLS = SK.stack_classes(STF)
  s_builder(arm, ...)      the GPU cell's runner arguments (no hooks, no test record) for the arms
                             D     L4PTFD                  G     L4PTFDG (FakeGraphs)       GD   L4PTFDGD (FakeGraphsD)
                             FKW   L4PTFKW (eager keep)    FDGKW L4PTFDGKW (FakeGraphs)
                             GU    L4PTFDGU (FakeGraphs)   GUD   L4PTFDGUD (FakeGraphsD)
                             FKU   L4PTFKU (eager)         GUK   L4PTFDGUK (FakeGraphs)     GUDK L4PTFDGUDKW (FakeGraphsD)
  run_arm(arm, scen, ...)  one run in i4g_world.SCEN[scen] (check: the union checks, stats, the "F" probe and, for a
                           keep arm, the keep's fresh-fetch byte checks)
  cell_inputs(ev, C)       the r2_i4s_cell gate inputs (EQF, COUNTS, KEEP-EQ) from a CPU run
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


XH = _load("i4h_world")
XK = _load("i4k_world")
XG = XH.XG
X4, I3, W, WW, SW, SR, BC, VR, STF, KA, RE, RW, Z, WB, LN, GR = (
    XG.X4, XG.I3, XG.W, XG.WW, XG.SW, XG.SR, XG.BC, XG.VR, XG.STF, XG.KA, XG.RE, XG.RW, XG.Z, XG.WB, XG.LN, XG.GR)
DGR, KP, CB = XH.DGR, XK.KP, XK.CB
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4u_hoist as HU         # noqa: E402
import r2_i4s_stack as SK         # noqa: E402

CLS = SK.stack_classes(STF)
L, BS, TOPK, V = W.L, W.BS, W.TOPK, W.V
K0, N0, S0 = WW.K0, WW.N0, WW.S0
SCEN = XG.SCEN
world = XG.world
FakeGraphs, FakeGraphsD = XG.FakeGraphs, XH.FakeGraphsD
STACK = dict(GU="L4PTFDGU", GUD="L4PTFDGUD", FKU="L4PTFKU", GUK="L4PTFDGUK", GUDK="L4PTFDGUDKW")
KEEP = ("FKW", "FDGKW", "FKU", "GUK", "GUDK")
DRAFT = ("GD", "GUD", "GUDK")
ARMS = ("D", "G", "GD", "FKW", "FDGKW") + tuple(STACK)


def s_builder(arm, gemm="natural", check=False, stats=False, kcheck=False, gneg=None, dgneg=None, hneg=None, sneg=None,
              kneg=None, kslots=None, cneg=None, phase_marks=False, probe_every=0):
    """The arm's runner with the GPU cell's runner arguments (no hooks, no test record)."""
    assert arm in ARMS, arm
    if arm in ("D", "G", "GD"):
        return XH.arm_builder(arm, gemm=gemm, check=check, stats=stats, gneg=gneg, dgneg=dgneg,
                              phase_marks=phase_marks, probe_every=probe_every)
    if arm in ("FKW", "FDGKW"):
        return XK.k4_builder(arm, gemm=gemm, check=check, stats=stats, kcheck=kcheck, gneg=gneg, kneg=kneg,
                             kslots=kslots, cneg=cneg, phase_marks=phase_marks, probe_every=probe_every)
    fa, fk = WB.fake_attention_u, Z.fake_fork
    Runner = CLS[STACK[arm]][1]

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
                  phase_marks=phase_marks, dneg=None, hneg=hneg)
        if arm in KEEP:
            kw.update(keep=True, kwin=True, kcheck=kcheck, kneg=kneg, kslots=kslots, cneg=cneg)
        if arm != "FKU":
            kw.update(gneg=gneg, sneg=sneg)
        be = None
        if arm in DRAFT:
            be = FakeGraphsD()
            kw.update(dgneg=dgneg, graph_backend=be)
        elif arm != "FKU":
            kw.update(graph_backend=FakeGraphs())
        run = Runner(model, rs, sops, BC.HostClock(), **kw)
        if be is not None:
            be.T = run.T
        return run
    return build


def run_arm(arm, scen="timing", B=8, N=None, seed=0, gemm="natural", gneg=None, dgneg=None, hneg=None, sneg=None,
            kneg=None, kslots=None, cneg=None, on_runner=None):
    """One run of an arm in scenario SCEN[scen]. Returns dict(runner, rd, N, B, scen, arm)."""
    s = SCEN[scen]
    N = N or N0
    model, c, S, hz, tok0 = world(B, N, seed)
    kc = bool(s["check"]) and arm in KEEP
    run = s_builder(arm, gemm=gemm, check=s["check"], stats=s["stats"], kcheck=kc, gneg=gneg, dgneg=dgneg, hneg=hneg,
                    sneg=sneg, kneg=kneg, kslots=kslots, cneg=cneg, phase_marks=s["phase_marks"],
                    probe_every=s["probe_every"])(model, c, S, hz, tok0, check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    if on_runner is not None:
        on_runner(run)
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen, arm=arm)


def cell_inputs(ev, C):
    """The r2_i4s_cell gate inputs from a CPU run; C the cell module."""
    r = ev["runner"]
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=C.IA.acct_record(ev["rd"]),
                state=C.IA.state_checksum(r), counts=C.GC.counts_record(ev["rd"]), keep=ev["rd"].get("keep"))
