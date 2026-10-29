"""The CPU WORLD of tests_cpu/test_i4k.py (rung I4K: the combined arm L4PTFDGKW = L4PTFD + the store keep + the window
loads from the kept store + the per-layer CUDA graphs of the joint pass's section, benchmarks/Efficiency/r2_i4k_combo.py;
not a test file). Load it by file path, as the older tests load their worlds. It builds on tests_cpu/i4g_world.py (read
its docstring first: FakeGraphs, the CUDA graph stand-in) and so on i4a_world / i3_world (the synthetic package, the
stand-ins, the arena, the lean classes).

CUDA GRAPHS DO NOT RUN ON THE CPU: FakeGraphs captures without executing (the capture's writes are restored) and replays
the recorded ops on the captured tensor objects; the gathers and the fork are kernel nodes; a host read inside a
capture raises. It establishes the plumbing (the device step, the counter rows, the persistent map, the stub of the
keep's residency, the eager parts), not GPU capture legality, dispatch, timing or memory.

THIS FILE ADDS
  CB, KP                    r2_i4k_combo, r2_i4d_keep
  TentStateFKg / TentRunnerFKg, TentStateFDGK / TentRunnerFDGK = CB.combo_classes(STF)
  TentRunnerFK              KP.keep_classes(STF)[1] (I4D's eager keep arm)
  k4_builder(arm, ...)      the GPU cell's runner arguments (no hooks, no test record) for the arms
                              FD    L4PTFD (r2_i4a_lean)        FDa   L4PTFD with the accounting (r2_i4d_keep)
                              FDG   L4PTFDG (r2_i4g_graph, FakeGraphs)
                              FKW   L4PTFKW (r2_i4d_keep)       FK    L4PTFK (r2_i4d_keep, kwin off)
                              FKWg  L4PTFKW with the graph-safe keep, eager (r2_i4k_combo TentRunnerFKg)
                              FDGKW L4PTFDGKW (r2_i4k_combo, FakeGraphs)   FDGK  the same with kwin off
  run_arm(arm, scen, ...)   one run in i4g_world.SCEN[scen] (check: the keep's byte checks on as well)
  cell_inputs(ev, C)        the r2_i4k_cell gate inputs (EQF, COUNTS, KEEP-EQ) from a CPU run
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


G4 = _load("i4g_world")
X4 = G4.X4
I3, W, WW, SW, SR, BC, VR, STF, KA, RE, RW, Z, WB, LN, GR = (G4.I3, G4.W, G4.WW, G4.SW, G4.SR, G4.BC, G4.VR, G4.STF,
                                                             G4.KA, G4.RE, G4.RW, G4.Z, G4.WB, G4.LN, G4.GR)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4d_keep as KP          # noqa: E402
import r2_i4k_combo as CB         # noqa: E402

TentStateFD, TentRunnerFD = G4.TentStateFD, G4.TentRunnerFD
TentStateFDG, TentRunnerFDG = G4.TentStateFDG, G4.TentRunnerFDG
TentStateFK, TentRunnerFK = KP.keep_classes(STF)
TentStateFKg, TentRunnerFKg, TentStateFDGK, TentRunnerFDGK = CB.combo_classes(STF)
FakeGraphs, FakeGraph = G4.FakeGraphs, G4.FakeGraph
L, BS, TOPK, V = G4.L, G4.BS, G4.TOPK, G4.V
K0, N0, S0 = G4.K0, G4.N0, G4.S0
ARMS = ("FD", "FDa", "FDG", "FKW", "FK", "FKWg", "FDGKW", "FDGK")
SCEN = G4.SCEN
world = G4.world


def k4_builder(arm, gemm="natural", check=False, stats=False, acct=False, kcheck=False, gneg=None, kneg=None,
               kslots=None, cneg=None, phase_marks=False, probe_every=0, backend=None):
    """The arm's runner with the GPU cell's runner arguments (no hooks, no test record)."""
    assert arm in ARMS, arm
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
        kw = dict(fork=fk, t0=tok0, hooks=None, check_every=check_every, union_check=check, union_stats=stats,
                  phase_marks=phase_marks)
        keep = dict(acct=acct, kneg=kneg, kslots=kslots, kcheck=kcheck, kwin=arm in ("FKW", "FKWg", "FDGKW"))
        gb = backend if backend is not None else FakeGraphs()
        if arm == "FD":
            return TentRunnerFD(model, rs, sops, BC.HostClock(), **kw)
        if arm == "FDa":
            return TentRunnerFK(model, rs, sops, BC.HostClock(), keep=False, acct=True, **kw)
        if arm == "FDG":
            return TentRunnerFDG(model, rs, sops, BC.HostClock(), gneg=gneg, graph_backend=gb, **kw)
        if arm in ("FKW", "FK"):
            return TentRunnerFK(model, rs, sops, BC.HostClock(), keep=True, **keep, **kw)
        if arm == "FKWg":
            return TentRunnerFKg(model, rs, sops, BC.HostClock(), cneg=cneg, **keep, **kw)
        return TentRunnerFDGK(model, rs, sops, BC.HostClock(), gneg=gneg, graph_backend=gb, cneg=cneg, **keep, **kw)
    return build


def run_arm(arm, scen="timing", B=8, N=None, seed=0, gemm="natural", acct=False, kslots=None, cneg=None, gneg=None,
            kneg=None, on_runner=None):
    """One run of an arm in scenario SCEN[scen] (check: union checks, stats, the "F" probe and, for a keep arm, the
    keep's fresh-fetch byte checks). Returns dict(runner, rd, N, B, scen, arm)."""
    s = SCEN[scen]
    N = N or N0
    model, c, S, hz, tok0 = world(B, N, seed)
    kc = bool(s["check"]) and arm not in ("FD", "FDa", "FDG")
    run = k4_builder(arm, gemm=gemm, check=s["check"], stats=s["stats"], acct=acct, kcheck=kc, gneg=gneg, kneg=kneg,
                     kslots=kslots, cneg=cneg, phase_marks=s["phase_marks"],
                     probe_every=s["probe_every"])(model, c, S, hz, tok0, check_every=s["check_every"])
    if s["fa"]:
        run.vs.fa_check = {}
    if on_runner is not None:
        on_runner(run)
    run.run(N)
    rd = run.read()
    return dict(runner=run, rd=rd, N=N, B=B, scen=scen, arm=arm)


def cell_inputs(ev, C):
    """The r2_i4k_cell gate inputs from a CPU run; C the cell module."""
    r = ev["runner"]
    return dict(out=r.out.cpu(), fp=(r.fmax.cpu(), r.fsum.cpu()), fp_ok=r.fp_ok.cpu(), acct=C.A4.acct_record(ev["rd"]),
                state=C.A4.state_checksum(r), counts=C.GC.counts_record(ev["rd"]), keep=ev["rd"].get("keep"))
