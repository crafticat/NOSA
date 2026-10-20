"""CPU tests U1-U3 of PositionalTwin.step_unchecked / advance_unchecked (nosi/nosi/r2_positional_twin.py): the
sync-free twin step of the R2 step-7 tick. Written by the verify agent (Q3), reviewed and accepted by track C under
conditions (a)-(d) of the Q3 answer (2026-10-21).

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_twin_unchecked.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi/nosi"
PKG = "r2twinu"


def _load(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
        stub = types.ModuleType(PKG + ".nosa_llama")
        stub.QK_SELECT = 33
        sys.modules[PKG + ".nosa_llama"] = stub
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


TW = _load("r2_positional_twin")
BD = _load("r2_bi_decode")
FIELDS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis")


S0 = (230, 233, 250, 190)                     # + 36 rows each: every bank crosses a 64-row block boundary


def _pair(S_list=S0, H=2, D=8, lk=40, nmax=420, seed=0, mean_fn=BD.fixed_order_mean):
    g = torch.Generator().manual_seed(seed)
    twins = []
    for _ in range(2):
        t = TW.PositionalTwin(len(S_list), H, D, lk, nmax)
        t.mean_fn = mean_fn
        twins.append(t)
    for b, S in enumerate(S_list):
        k, c = torch.randn(S, H, D, generator=g), torch.randn(S, H, generator=g)
        for t in twins:
            t.prefill(b, k, c)
    return twins


@pytest.mark.parametrize("mean_fn", [BD.fixed_order_mean, None])
def test_U1_unchecked_equals_checked(mean_fn):
    """U1: advance_unchecked gives bitwise advance()'s tables, counters and ctr over 24 ticks with RAGGED n per bank
    (0 .. 3 rows, a bank idle for several ticks), every bank passing key and cis compression events and a block
    boundary (250 -> 256 and 233 -> 256 ... ), with the mean_fn hook set (fixed_order_mean) and unset (torch's .mean);
    the overflow flag stays False. Negative control: a twin whose unchecked step skips the cis compression differs."""
    a, b = _pair(mean_fn=mean_fn)
    g = torch.Generator().manual_seed(1)
    B, H, D = a.B, a.H, a.D
    ns = [[3, 0, 2, 1], [1, 2, 0, 3], [2, 3, 1, 0], [0, 1, 3, 2]]
    for tick in range(24):
        n = torch.tensor(ns[tick % 4], dtype=torch.int64)
        T = int(n.max())
        keys = torch.randn(B, T, H, D, generator=g)
        cis = torch.randn(B, T, H, generator=g)
        c1 = a.advance(keys, cis, n)
        c2 = b.advance_unchecked(keys, cis, n)
        assert torch.equal(c1, c2), tick
    for f in FIELDS:
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    assert not bool(b.overflow)
    assert all(int(x) // 64 > s // 64 for x, s in zip(a.pos, S0))                           # every bank crossed a block
    assert all(int(n2) > s // 16 - 1 for n2, s in zip(a.n_comp, S0))                        # and compressed keys
    assert all(int(n2) > s // 16 - 1 for n2, s in zip(a.n_ccis, S0))                        # and compressed cis

    class NoCis(TW.PositionalTwin):
        def _cis_due(self, p, n):
            return torch.zeros_like(p, dtype=torch.bool)
    c = NoCis(B, H, D, a.lk_max, a.n_max)
    for f in FIELDS:
        getattr(c, f).copy_(getattr(_pair(mean_fn=mean_fn)[0], f))
    c.mean_fn = mean_fn
    g = torch.Generator().manual_seed(1)
    for tick in range(24):
        n = torch.tensor(ns[tick % 4], dtype=torch.int64)
        T = int(n.max())
        c.advance_unchecked(torch.randn(B, T, H, D, generator=g), torch.randn(B, T, H, generator=g), n)
    assert not torch.equal(c.ccis, a.ccis) or not torch.equal(c.n_ccis, a.n_ccis)


def test_U2_overflow_writes_nothing_and_flags():
    """U2: a bank at N_max - 1 advanced by 2 rows: the checked path refuses (TwinCapacityError); the unchecked path
    writes the first row, then for the second writes NOTHING for that bank (total_cis, ring and counters as after the
    first row), sets the sticky overflow flag, and the other banks advance normally."""
    a, b = _pair(S_list=(410, 200), nmax=411, lk=40)
    with pytest.raises(TW.TwinCapacityError):
        a.advance(torch.randn(2, 2, a.H, a.D), torch.randn(2, 2, a.H), torch.tensor([2, 2]))
    keys, cis = torch.randn(2, 2, b.H, b.D), torch.randn(2, 2, b.H)
    before = b.total_cis.clone()
    b.advance_unchecked(keys, cis, torch.tensor([2, 2], dtype=torch.int64))
    assert bool(b.overflow)
    assert int(b.pos[0]) == 411 and int(b.pos[1]) == 202
    assert torch.equal(b.total_cis[0, 410], cis[0, 0]) and torch.equal(b.total_cis[0, :410], before[0, :410])
    assert torch.equal(b.total_cis[1, 200:202], cis[1])


def test_U3_unchecked_refuses_host_inputs():
    """U3: active / n must be device tensors of the right dtype (a CPU twin: CPU tensors; a list or an int32 n is
    refused), so a pageable host copy cannot slip into the tick."""
    t = _pair()[0]
    with pytest.raises(ValueError):
        t.step_unchecked(torch.zeros(t.B, t.H, t.D), torch.zeros(t.B, t.H), [True] * t.B)
    with pytest.raises(ValueError):
        t.advance_unchecked(torch.zeros(t.B, 1, t.H, t.D), torch.zeros(t.B, 1, t.H), torch.ones(t.B, dtype=torch.int32))
