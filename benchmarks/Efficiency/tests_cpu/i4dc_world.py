"""The CPU WORLD of tests_cpu/test_i4dc.py (rung I4DC: the arms L4PTFDGUDCD and L4PTFDGUDKWCD with the compiled draft
helpers, benchmarks/Efficiency/r2_i4dc_compile.py; not a test file). Load it by file path, as the older tests load their
worlds. It builds on tests_cpu/i4cc_world.py (the C arms, FakeGraphsDC): read its docstring and i4h_world's first.

CUDA GRAPHS DO NOT RUN ON THE CPU, AND INDUCTOR'S CPU BACKEND IS NOT ITS GPU BACKEND. These tests establish the plumbing
(the compiled draft helpers in place of the lean row's eager ops, the fallback rules, the compile-event record), the bit
equality with L4PTFD on the CPU backend (C++ kernels generated from the same graphs) and that no graph breaks. Only the
GPU miniature i4dm2 and the timing cells establish Triton's bits, capture legality of the compiled kernels inside the
draft graphs, timing and memory.

THIS FILE ADDS
  CD                       r2_i4dc_compile; DCL = CD.dc_classes(STF)
  FakeGraphsDCD            i4cc_world's FakeGraphsDC whose split draft capture (begin() / end()) also makes every
                           compiled helper call ONE kernel node (the same node maker as the joint capture's: the call
                           runs once at the capture with the recorder left, its written arguments saved and restored
                           after the capture; a replay calls the compiled helper again on the captured tensor objects
                           and copies its results into the captured outputs)
  d_builder(arm, ...)      the GPU cell's runner arguments for GUDCD (L4PTFDGUDCD) and GUDKCD (L4PTFDGUDKWCD)
  run_arm(arm, scen, ...)  one run: the D arms here, every other arm through i4cc_world.run_arm
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


XC = _load("i4cc_world")
XS, XH, XG = XC.XS, XC.XH, XC.XG
W, WW, SR, BC, STF, KA, RE, RW, Z, WB, I3, CC = (XC.W, XC.WW, XC.SR, XC.BC, XC.STF, XC.KA, XC.RE, XC.RW, XC.Z, XC.WB,
                                                 XC.I3, XC.CC)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4dc_compile as CD      # noqa: E402

DCL = CD.dc_classes(STF)
L, BS, TOPK = W.L, W.BS, W.TOPK
K0, N0 = WW.K0, WW.N0
SCEN = XC.SCEN
world = XC.world
DARMS = dict(GUDCD="L4PTFDGUDCD", GUDKCD="L4PTFDGUDKWCD")
KEEP = XC.KEEP + ("GUDKCD",)
# the written arguments of each draft helper (argument positions, r2_i4dc_compile's signatures)
XC.MUT.update(dpre=(), dwrite=(0, 1, 2), dmask=(5,))


class FakeGraphsDCD(XC.FakeGraphsDC):
    """FakeGraphsDC with the compiled draft helpers as kernel nodes in the split draft capture (module docstring)."""

    def begin(self):
        super().begin()
        self.T._cc_wrap = XC._cnode(self._open[0])

    def end(self):
        self.T._cc_wrap = None
        return super().end()


def d_builder(arm, gemm="natural", check=False, stats=False, kcheck=False, gneg=None, dgneg=None, hneg=None,
              sneg=None, kneg=None, kslots=None, cneg=None, ccneg=None, cdneg=None, cc_backend="inductor",
              phase_marks=False, probe_every=0):
    """The D arm's runner with the GPU cell's runner arguments (no hooks, no test record)."""
    assert arm in DARMS, arm
    fa, fk = WB.fake_attention_u, Z.fake_fork
    Runner = DCL[DARMS[arm]][1]

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
                  cc_backend=cc_backend, ccneg=ccneg, cdneg=cdneg)
        if arm in KEEP:
            kw.update(keep=True, kwin=True, kcheck=kcheck, kneg=kneg, kslots=kslots, cneg=cneg)
        be = FakeGraphsDCD()
        run = Runner(model, rs, sops, BC.HostClock(), graph_backend=be, **kw)
        be.T = run.T
        return run
    return build


def run_arm(arm, scen="timing", B=8, N=None, seed=0, gemm="natural", gneg=None, dgneg=None, hneg=None, sneg=None,
            kneg=None, kslots=None, cneg=None, ccneg=None, cdneg=None, cc_backend="inductor", on_runner=None):
    """One run of an arm in scenario SCEN[scen]. Returns dict(runner, rd, N, B, scen, arm)."""
    if arm not in DARMS:
        return XC.run_arm(arm, scen, B=B, N=N, seed=seed, gemm=gemm, gneg=gneg, dgneg=dgneg, hneg=hneg, sneg=sneg,
                          kneg=kneg, kslots=kslots, cneg=cneg, ccneg=ccneg, cc_backend=cc_backend,
                          on_runner=on_runner)
    CC.configure()
    s = SCEN[scen]
    N = N or N0
    model, c, S, hz, tok0 = world(B, N, seed)
    kc = bool(s["check"]) and arm in KEEP
    run = d_builder(arm, gemm=gemm, check=s["check"], stats=s["stats"], kcheck=kc, gneg=gneg, dgneg=dgneg, hneg=hneg,
                    sneg=sneg, kneg=kneg, kslots=kslots, cneg=cneg, ccneg=ccneg, cdneg=cdneg, cc_backend=cc_backend,
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
