"""CPU tests of R2 milestone 2 step 6 (RAGGED ISOLATION): the shape-stable summaries of nosi/nosi/r2_positional_twin.py,
the negative-control kinds and the preallocated widths of nosi/nosi/r2_decode_path.py, R2State.set_active, and the
plan helpers of benchmarks/Efficiency/r2_ragged_gate.py. No CUDA: the modules are loaded through a synthetic package
(as tests_cpu/test_r2_g0.py does).

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_ragged.py
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
sys.path.insert(0, str(EFF))
import r2_ragged_gate as RG  # noqa: E402


def _load(name):
    if "r2rag" not in sys.modules:
        pkg = types.ModuleType("r2rag")
        pkg.__path__ = [str(NOSI)]
        sys.modules["r2rag"] = pkg
    full = "r2rag." + name
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


RD = _load("r2_decode_path")
TW = sys.modules["r2rag.r2_positional_twin"]


class RecordingTwin(TW.PositionalTwin):
    """Records the batch size of every key-window call made by step()."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.window_rows = []

    def key_window(self, b, start):
        self.window_rows.append(int(b.numel()))
        return super().key_window(b, start)


def _prefilled(cls, S_list, extra=200, seed=0, H=2, D=8):
    g = torch.Generator().manual_seed(seed)
    B = len(S_list)
    N = max(S_list) + extra
    keys = torch.randn((B, N, H, D), generator=g)
    cis = torch.randn((B, N, H), generator=g)
    t = cls(B, H, D, lk_max=N // 16 + 2, n_max=N + 64)
    for b, S in enumerate(S_list):
        t.prefill(b, keys[b, :S], cis[b, :S])
    return t, keys, cis


def test_R1_summaries_are_computed_for_the_whole_batch():
    """R1: with a ragged activity, every key-window read made by step() covers ALL B requests (the reduction shape
    never depends on how many are due), and the cis summary gathers the whole batch (AST: the window index is
    all_b)."""
    t, keys, cis = _prefilled(RecordingTwin, [200, 233, 263])
    g = torch.Generator().manual_seed(1)
    pos = t.pos.clone()
    for u in range(80):
        act = torch.rand(3, generator=g) < 0.6
        act[0] = True
        t.step(torch.stack([keys[b, int(pos[b])] for b in range(3)]), torch.stack([cis[b, int(pos[b])] for b in range(3)]),
               active=act)
        pos = t.pos.clone()
    assert t.window_rows and set(t.window_rows) == {3}
    step_src = ast.unparse(next(n for n in ast.walk(ast.parse((NOSI / "r2_positional_twin.py").read_text()))
                                if isinstance(n, ast.FunctionDef) and n.name == "step"))
    assert "self.total_cis[all_b[:, None], rows]" in step_src and "self.key_window(all_b, STRIDE * self.n_comp)" in step_src


def test_R2_ragged_batch_equals_independent_runs():
    """R2: a B = 3 batch with different prompt lengths and a random activity equals three B = 1 twins fed the same
    tokens, on every live table and counter after every step (200 steps: many key and cis summaries)."""
    S_list = [200, 233, 263]
    t, keys, cis = _prefilled(TW.PositionalTwin, S_list, extra=260)
    solo = []
    for b, S in enumerate(S_list):
        s = TW.PositionalTwin(1, 2, 8, lk_max=t.lk_max, n_max=t.n_max)
        s.prefill(0, keys[b, :S], cis[b, :S])
        solo.append(s)
    g = torch.Generator().manual_seed(2)
    for u in range(200):
        act = torch.rand(3, generator=g) < 0.7
        act[0] = True
        pos = t.pos.clone()
        t.step(torch.stack([keys[b, int(pos[b])] for b in range(3)]), torch.stack([cis[b, int(pos[b])] for b in range(3)]),
               active=act)
        for b in range(3):
            if bool(act[b]):
                p = int(pos[b])
                solo[b].step(keys[b, p:p + 1], cis[b, p:p + 1])
            s = solo[b]
            assert int(t.pos[b]) == int(s.pos[0]) and int(t.n_comp[b]) == int(s.n_comp[0]) and int(t.n_ccis[b]) == int(s.n_ccis[0])
            n, m, p = int(s.n_comp[0]), int(s.n_ccis[0]), int(s.pos[0])
            assert torch.equal(t.comp_k[b, :n], s.comp_k[0, :n]) and torch.equal(t.ccis[:, b, :m], s.ccis[:, 0, :m])
            assert torch.equal(t.total_cis[b, :p], s.total_cis[0, :p]) and torch.equal(t.live_keys(b), s.live_keys(0))
    t.check_invariants()
    assert int(t.n_comp[0]) > S_list[0] // 16 - 1 + 8


def test_R3_negative_control_kinds():
    """R3: 'shared' gives every request the source request's key length and pooled width from its step on; 'frozen'
    keeps m0 (None -> the natural widths); 'short' is unchanged; an unknown kind is refused."""
    pos = torch.tensor([16128, 16000, 12031])
    n = torch.tensor([1007, 999, 750])
    nat_n, nat_m = RD.pooling_lengths(pos, n, 0)
    assert nat_m.tolist() == [253, 251, 188]
    sh = dict(kind="shared", request=2, step=3)
    assert RD.pooling_lengths(pos, n, 2, sh)[0].tolist() == [1007, 999, 750]
    nu, mb = RD.pooling_lengths(pos, n, 3, sh)
    assert nu.tolist() == [750] * 3 and mb.tolist() == [188] * 3 and nu.is_contiguous() and mb.is_contiguous()
    fr = dict(kind="frozen")
    assert RD.pooling_lengths(pos, n, 0, fr, None)[1].tolist() == [253, 251, 188]
    later = pos + 64
    assert RD.pooling_lengths(later, n, 64, None)[1].tolist() == [254, 252, 189]
    assert RD.pooling_lengths(later, n, 64, fr, nat_m)[1].tolist() == [253, 251, 188]
    assert RD.pooling_lengths(pos, n, 5, dict(step=5, request=0, short=7))[0].tolist() == [1000, 999, 750]
    with pytest.raises(RD.R2Error, match="unknown negative-control kind"):
        RD.pooling_lengths(pos, n, 0, dict(kind="bogus"))


def test_R4_preallocated_widths_cover_the_run():
    """R4: ragged_widths for S_max 16128 and 512 steps: M_alloc 261 >= every pooled width pos // 64 + 1 of the run,
    Lk_max 1041 >= every compressed-key count, the score width round128(1041) = 1152 constant, N_max = n_host 16704 >
    the last position; the same widths for any shorter neighbours (they depend on S_max only)."""
    w = RD.ragged_widths([16128, 16000, 12031], 512)
    assert w == dict(M_alloc=261, lk_max=1041, n_max=16704, n_host=16704, score_width=1152)
    last = 16128 + 512 - 1
    assert last // 64 + 1 <= w["M_alloc"] and (last + 1) // 16 - 1 <= w["lk_max"] and last < w["n_max"]
    assert RD.ragged_widths([16128], 512) == w
    long = RD.ragged_widths([16128, 15000], 1100)
    assert long["n_max"] > 16128 + 1100 and long["n_max"] > 16128 + 1024


def test_R5_set_active():
    """R5: set_active copies the host mask to the host mirror and the device tensor; a wrong shape is refused."""
    st = RD.R2State.__new__(RD.R2State)
    st.B = 3
    st.active = torch.ones(3, dtype=torch.bool)
    st.active_host = torch.ones(3, dtype=torch.bool)
    m = torch.tensor([True, False, True])
    st.set_active(m)
    assert st.active.tolist() == [True, False, True] and st.active_host.tolist() == [True, False, True]
    m[1] = True
    assert st.active_host.tolist() == [True, False, True]                        # a copy, not an alias
    with pytest.raises(RD.R2Error):
        st.set_active(torch.ones(4, dtype=torch.bool))


def test_R6_gate_plan():
    """R6: arms start with E, E2; deltas has B values with d_0 = 0; the first boundary is step 64 at S0 = 16128 (62 at
    16130); the activity keeps request 0 active and is seeded; the arms' plans (R ragged, NS shared from request B - 1,
    NF frozen, B1 alone); the frozen width differs from the natural one first at that boundary."""
    base = ["--out", "x", "--workload", "w"]
    for bad in (["--arms", "E2,E"], ["--arms", "E,R"], ["--deltas", "1,0,0,0,0,0,0,0"], ["--deltas", "0,1"],
                ["--nf-steps", "60"], ["--arms", "E,E2,Q"]):
        with pytest.raises(SystemExit):
            RG.parse_args(base + bad)
    a = RG.parse_args(base)
    assert a.batch == 8 and a.steps == 512 and a.arms == list(RG.ARMS)
    assert RG.first_boundary(16128) == 64 and RG.first_boundary(16130) == 62
    act = RG.activity(50, 8, 0.7, 6, True)
    assert bool(act[:, 0].all()) and not bool(act.all()) and torch.equal(act, RG.activity(50, 8, 0.7, 6, True))
    assert bool(RG.activity(5, 8, 0.7, 6, False).all())
    assert RG.arm_plan("R", a)[0] == [16128 - d for d in a.deltas] and RG.arm_plan("R", a)[1] is True
    assert RG.arm_plan("NS", a)[2] == dict(kind="shared", request=7, step=0) and RG.arm_plan("NF", a)[2] == dict(kind="frozen")
    assert RG.arm_plan("B1", a)[4] == 1 and RG.arm_plan("NF", a)[3] == a.nf_steps
    pos0 = torch.tensor([16128])
    m0 = RD.pooling_lengths(pos0, torch.tensor([1007]), 0)[1]
    first = next(s for s in range(1, 200)
                 if not torch.equal(RD.pooling_lengths(pos0 + s, torch.tensor([1007]), s, dict(kind="frozen"), m0)[1],
                                    RD.pooling_lengths(pos0 + s, torch.tensor([1007]), s)[1]))
    assert first == RG.first_boundary(16128)


def test_R7_the_shipped_order_is_kept_in_step():
    """R7 (AST of PositionalTwin.step): the key summary is written before the raw key (deferred: it reads only older
    positions), the cis summary after the raw cis (immediate), the counter last: the shipped order the G0 gate relies on."""
    src = ast.unparse(next(n for n in ast.walk(ast.parse((NOSI / "r2_positional_twin.py").read_text()))
                           if isinstance(n, ast.FunctionDef) and n.name == "step"))
    i_key = src.index("self.comp_k[bk, nk] = means[bk]")
    i_ring = src.index("self.ring[b, p % RING] = keys[b]")
    i_cis = src.index("self.total_cis[b, p] = cis[b]")
    i_ccis = src.index("self.ccis[:, bc, nc] = means_c[:, bc]")
    i_pos = src.index("self.pos[b] = p + 1")
    assert i_key < i_ring < i_cis < i_ccis < i_pos
