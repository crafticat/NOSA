"""The CPU WORLD of tests_cpu/test_i4u.py (rung I4U, lever L5: the arm L4PTFDU = L4PTFD with the joint pass's
layer-invariant tensors hoisted, benchmarks/Efficiency/r2_i4u_hoist.py; not a test file). Load it by file path, as the
older tests load their worlds. It builds on tests_cpu/i4a_world.py (read its docstring first): the same synthetic
package, stand-ins, arena, op counter (OpCount) and the L4PTFD builder.

THIS FILE ADDS
  HU                       benchmarks/Efficiency/r2_i4u_hoist.py; TentStateFDU / TentRunnerFDU = HU.hoist_classes(...)
  tentfdu_builder(...)     i4a_world.tentfd_builder with TentRunnerFDU (hneg: the hoist controls)
  run_fd / run_fdu(...)    one run of L4PTFD / L4PTFDU (check and stats as given)
  without_hoist(ev)        the evidence without read()'s "hoist" host counts (L4PTFD's read() has no such key)
  count_joint_ops(...)     per joint pass and layer: the aten compute ops inside TentState._rows (and per pass), and the
                           one-row passes' totals
  joint_facts(...)         a test-side recorder (host reads; tests only) of the joint passes: inactive rows, rows past
                           the committed edge, duplicate positions of a request (an inactive row at an active row's p)
"""
from __future__ import annotations

import collections
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


X4 = _load("i4a_world")
I3, I, W, RW, SW, SR, BC, VR, KA, LN, STF, WB, Z, RE = (X4.I3, X4.I, X4.W, X4.RW, X4.SW, X4.SR, X4.BC, X4.VR, X4.KA,
                                                         X4.LN, X4.STF, X4.WB, X4.Z, X4.RE)
if str(W.EFF) not in sys.path:
    sys.path.insert(0, str(W.EFF))
import r2_i4u_hoist as HU          # noqa: E402

TentStateFD, TentRunnerFD = X4.TentStateFD, X4.TentRunnerFD
TentStateFDU, TentRunnerFDU = HU.hoist_classes(STF, (TentStateFD, TentRunnerFD))
L, BS, TOPK = X4.L, X4.BS, X4.TOPK
K0, N0, TOL = X4.K0, X4.N0, X4.TOL
OpCount = X4.OpCount


def tentfdu_builder(X=None, gemm="rowgroup", check=True, stats=True, hneg=None, dneg=None, fneg=None, vrecord=False):
    """i4a_world.tentfd_builder's builder (level L4P) with TentRunnerFDU (hneg: a hoist control)."""
    fa, fk = WB.fake_attention_u, Z.fake_fork

    def build(model, cache, S, horizon, tok0, level="L4P", K=K0, draft_filter=None, neg=None, tneg=None, poison=False,
              hooks=None, check_every=1, after_round=None, record=True):
        assert level == "L4P", "L4PTFDU is an L4P arm"
        B = int(cache.layers[0].cache_engine._k_gpu.shape[0])
        arena = I3.chain_arena(cache, B, X or KA.store_slots(TOPK, K))
        ops = RE.RaggedOps(W.KERNELS, twin="torch", body="torch", graph=True)
        rs = RE.attach(model, cache, S, horizon, ops)
        KA.attach(rs, arena)
        RW.define_undefined(rs)
        sops = SR.SpecOps(level=level, K=K, mode="exact", gemm=gemm, extra=arena.X, attn=fa, journal=RW.CAP,
                          redirect=True, draft_filter=draft_filter, neg=neg, record=record, probe_every=0,
                          after_round=after_round)
        return TentRunnerFDU(model, rs, sops, BC.HostClock(), fork=fk, t0=tok0, hooks=hooks, check_every=check_every,
                             poison=poison, tneg=tneg, union_check=check, union_stats=stats, vneg=None, fneg=fneg,
                             vrecord=vrecord, dneg=dneg, hneg=hneg)
    return build


def run_fd(B: int, gemm="rowgroup", filt=None, ref=True, tneg=None, check=True, stats=True, on_runner=None,
           fneg=None):
    """One L4PTFD run (i4a_world.tentfd_builder; check / stats as given)."""
    return I3.run_tentf(B, build=X4.tentfd_builder(gemm=gemm, check=check, stats=stats, fneg=fneg), draft_filter=filt,
                        tneg=tneg, with_reference=ref, on_runner=on_runner)


def run_fdu(B: int, gemm="rowgroup", filt=None, ref=True, tneg=None, check=True, stats=True, hneg=None, on_runner=None,
            fneg=None):
    """The same run with L4PTFDU (hneg: a planted control)."""
    return I3.run_tentf(B, build=tentfdu_builder(gemm=gemm, check=check, stats=stats, hneg=hneg, fneg=fneg),
                        draft_filter=filt, tneg=tneg, with_reference=ref, on_runner=on_runner)


def without_hoist(ev: dict) -> dict:
    """The evidence with read()'s "hoist" host counts removed (L4PTFD's read() has no such key)."""
    out = dict(ev)
    out["rd"] = {k: v for k, v in ev["rd"].items() if k != "hoist"}
    return out


def count_joint_ops(runner) -> dict:
    """Wraps runner.T.run_pass and runner.T._rows (instance attributes): for every JOINT pass the aten compute ops of the
    whole pass (box["joint"]) and of TentState._rows per layer (box["layers"]: a list per pass, with the op names per
    layer in box["names"]); for every one-row pass the whole pass (box["one"]). Every count is made by OpCount,
    independently of the module's counters."""
    T = runner.T
    box = dict(joint=[], layers=[], one=[], names=[])
    run0, rows0 = T.run_pass, T._rows

    def run_pass(model, plan):
        cnt = OpCount()
        if plan.R > 1:
            box["layers"].append([])
            box["names"].append([])
        with cnt:
            out = run0(model, plan)
        (box["joint"] if plan.R > 1 else box["one"]).append(cnt.n)
        return out

    def rows(l, eng, twin, sel, q4, k4, v4, pbs, plan):
        if plan.R == 1:
            return rows0(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        cnt = OpCount()
        with cnt:
            out = rows0(l, eng, twin, sel, q4, k4, v4, pbs, plan)
        box["layers"][-1].append(cnt.n)
        box["names"][-1].append(collections.Counter(cnt.names))
        return out
    T.run_pass, T._rows = run_pass, rows
    return box


def joint_facts(runner) -> dict:
    """A test-side recorder (host reads; tests only) of the JOINT passes at layer 0: the passes, the inactive rows, the
    active rows past the committed edge, and the inactive rows whose position equals an active row's of the same request
    (the case that forbids one batched write of the R rows)."""
    T = runner.T
    box = dict(passes=0, inactive=0, past=0, dup=0)
    rows0 = T._rows

    def rows(l, eng, twin, sel, q4, k4, v4, pbs, plan):
        if plan.R > 1 and l == 0:
            import torch
            act = plan.act                                             # (R, B)
            P = torch.stack(pbs, 0)                                    # (R, B)
            qb = T.qc // BS
            box["passes"] += 1
            box["inactive"] += int((~act).sum())
            box["past"] += int((((P // BS) > qb.view(1, -1)) & act).sum())
            for b in range(P.shape[1]):
                pa = {int(P[r, b]) for r in range(P.shape[0]) if bool(act[r, b])}
                box["dup"] += sum(1 for r in range(P.shape[0]) if not bool(act[r, b]) and int(P[r, b]) in pa)
        return rows0(l, eng, twin, sel, q4, k4, v4, pbs, plan)
    T._rows = rows
    return box
