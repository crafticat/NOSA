"""The CPU WORLD of tests_cpu/test_i4a.py (rung I4A, lever 1: the arm L4PTFD = L4PTF with the lean one-row pass,
benchmarks/Efficiency/r2_i4a_lean.py; not a test file). Load it by file path, as the older tests load their worlds. It
builds on tests_cpu/i3_world.py (read its docstring first): the same synthetic package, stand-ins and arena.

THIS FILE ADDS
  LN                       benchmarks/Efficiency/r2_i4a_lean.py; TentStateFD / TentRunnerFD = LN.lean_classes(I3.STF)
  tentfd_builder(...)      i3_world.tentf_builder with TentRunnerFD (dneg: the lean controls)
  run_pair(...)            the same world run twice: L4PTF (i3_world.tentf_builder) and L4PTFD (or a planted L4PTFD)
  OpCount                  a TorchDispatchMode that counts aten COMPUTE ops (view ops and empty allocations excluded)
  count_one_row_ops(...)   per one-row pass and layer: the aten compute ops inside TentState._rows (and per pass)
  past_rows(...)           a test-side recorder: rows past the committed block's edge, draft skips (one-row passes)
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


I3 = _load("i3_world")
I, Z, WW, WB, SW, W, RW, VW, RE, SR, BC, ST, STU, VR, STF, KA, IA = (
    I3.I, I3.Z, I3.WW, I3.WB, I3.SW, I3.W, I3.RW, I3.VW, I3.RE, I3.SR, I3.BC, I3.ST, I3.STU, I3.VR, I3.STF, I3.KA,
    I3.IA)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4a_lean as LN          # noqa: E402

TentStateFD, TentRunnerFD = LN.lean_classes(STF)
L, BS, TOPK, V = W.L, W.BS, W.TOPK, W.V
K0, N0, S0, TOL = WW.K0, WW.N0, WW.S0, WW.TOL


def tentfd_builder(X=None, gemm="rowgroup", check=True, stats=True, dneg=None, fneg=None, vrecord=False):
    """i3_world.tentf_builder's builder (level L4P) with TentRunnerFD (the lean one-row pass; dneg a lean control)."""
    fa, fk = WB.fake_attention_u, Z.fake_fork

    def build(model, cache, S, horizon, tok0, level="L4P", K=K0, draft_filter=None, neg=None, tneg=None, poison=False,
              hooks=None, check_every=1, after_round=None, record=True):
        assert level == "L4P", "L4PTFD is an L4P arm"
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, X or KA.store_slots(TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level=level, K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, draft_filter=draft_filter, neg=neg, record=record, probe_every=0,
                          after_round=after_round)
        return TentRunnerFD(model, rs, sops, BC.HostClock(), fork=fk, t0=tok0, hooks=hooks, check_every=check_every,
                            poison=poison, tneg=tneg, union_check=check, union_stats=stats, vneg=None, fneg=fneg,
                            vrecord=vrecord, dneg=dneg)
    return build


def run_f(B: int, gemm="rowgroup", filt=None, ref=True, tneg=None, on_runner=None, max_rounds=0):
    """One L4PTF run (i3_world.run_tentf, check and stats on)."""
    return I3.run_tentf(B, build=I3.tentf_builder(gemm=gemm), draft_filter=filt, tneg=tneg, with_reference=ref,
                        max_rounds=max_rounds, on_runner=on_runner)


def run_fd(B: int, gemm="rowgroup", filt=None, ref=True, tneg=None, dneg=None, on_runner=None, max_rounds=0):
    """The same run with L4PTFD (dneg: a planted control)."""
    return I3.run_tentf(B, build=tentfd_builder(gemm=gemm, dneg=dneg), draft_filter=filt, tneg=tneg,
                        with_reference=ref, max_rounds=max_rounds, on_runner=on_runner)


def without_lean(ev: dict) -> dict:
    """The evidence with read()'s "lean" host counts removed (L4PTF's read() has no such key)."""
    out = dict(ev)
    out["rd"] = {k: v for k, v in ev["rd"].items() if k != "lean"}
    return out


# ---------------------------------------------------------------------------
# the aten op counter
# ---------------------------------------------------------------------------
_NOT_COMPUTE = {"_unsafe_view", "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "detach",
                "lift_fresh", "alias"}


def _is_view(func) -> bool:
    """A view op: its result aliases an input without a write (the schema's alias annotation)."""
    rets = func._schema.returns
    return any(r.alias_info is not None and not r.alias_info.is_write for r in rets)


class OpCount(TorchDispatchMode):
    """Counts the aten COMPUTE ops dispatched while active: every op except a view op and an empty allocation
    (_NOT_COMPUTE). names: a Counter by overload packet."""

    def __init__(self):
        super().__init__()
        self.n = 0
        self.names = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if not _is_view(func) and name not in _NOT_COMPUTE:
            self.n += 1
            self.names[name] += 1
        return func(*args, **(kwargs or {}))


def count_one_row_ops(runner) -> dict:
    """Wraps runner.T.run_pass and runner.T._rows (instance attributes): for every ONE-ROW pass, the aten compute ops
    of the whole pass (box["passes"]) and of TentState._rows per layer (box["layers"]: a list per pass); for every
    joint pass the whole pass (box["joint"]). Every count is made by OpCount, independently of the module's counters."""
    T = runner.T
    box = dict(passes=[], layers=[], joint=[], names=collections.Counter())
    run0, rows0 = T.run_pass, T._rows

    def run_pass(model, plan):
        cnt = OpCount()
        if plan.R == 1:
            box["layers"].append([])
        with cnt:
            out = run0(model, plan)
        (box["passes"] if plan.R == 1 else box["joint"]).append(cnt.n)
        return out

    def rows(l, eng, twin, sel, q4, k4, v4, pbs, plan):
        if plan.R != 1:
            return rows0(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        cnt = OpCount()
        with cnt:
            out = rows0(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        box["layers"][-1].append(cnt.n)
        box["names"].update(cnt.names)
        return out
    T.run_pass, T._rows = run_pass, rows
    return box


def past_rows(runner) -> dict:
    """A test-side recorder (host reads; tests only) of the ONE-ROW passes of an L4PTFD runner: the active rows past
    the committed block's edge, the active rows in the committed block, and the draft skips, counted at layer 0."""
    T = runner.T
    box = dict(past=0, inblock=0, skipped=0, passes=0)
    row0 = T._row_lean

    def row(l, eng, twin, sel, q_r, k_r, v_r, p, plan):
        s0 = int(T.skipped[l])
        out = row0(l, eng, twin, sel, q_r, k_r, v_r, p, plan)
        if l == 0:
            act = plan.act[0]
            box["past"] += int((T._lp.past & act).sum())
            box["inblock"] += int((~T._lp.past & act).sum())
            box["skipped"] += int(T.skipped[l]) - s0
            box["passes"] += 1
        return out
    T._row_lean = row
    return box


def small_world(B=2, seed=0):
    """(model, cache, S, horizon, tok0) of a small run (test_i3._small's world, N 8); a new cache on every call (two
    calls with one seed give the same world)."""
    S = S0
    hz = SW.horizon_for(8, K0)
    model, _, c = SW.two_caches(B, S, hz + 64, seed)
    tok0 = torch.randint(0, V, (B,), generator=torch.Generator().manual_seed(seed + 2))
    return model, c, S, hz, tok0
