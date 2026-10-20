"""CPU tests of R2 step 7, G0-BI: nosi/nosi/r2_bi_decode.py (the fixed-order mean, the BI ops and attach_bi, the
page-table glue on the body's window, subset_state), the twin's mean hook (r2_positional_twin.py), the BI dispatch in
r2_decode_path.decode_inference, the quality helpers of benchmarks/Efficiency/r2_g0bi_gate.py, and the uuid file and
pins of benchmarks/Efficiency/r2_g0bi_step.sh. No CUDA: the modules load through a synthetic package with a stub
nosa_llama (QK_SELECT only); the CUDA kernels of the BI decode are the GPU gate's. Each check has a negative control.

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_g0bi.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
PKG = "r2g0bi"


def _load(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
        stub = types.ModuleType(PKG + ".nosa_llama")             # what r2_decode_path reads of nosa_llama on CPU
        stub.QK_SELECT = 33
        sys.modules[PKG + ".nosa_llama"] = stub
        pkg.nosa_llama = stub
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        setattr(sys.modules[PKG], mod, m)
    return sys.modules[full]


BD = _load("r2_bi_decode")
RP = sys.modules[PKG + ".r2_decode_path"]
TW = sys.modules[PKG + ".r2_positional_twin"]
DB = _load("r2_decode_body")
sys.path.insert(0, str(EFF))
spec = importlib.util.spec_from_file_location("r2_g0bi_gate_cpu", str(EFF / "r2_g0bi_gate.py"))
GATE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GATE)


# =====================================================================================================================
# G1 the fixed-order mean, G2 the twin's mean hook
# =====================================================================================================================
def test_G1_fixed_order_mean():
    """G1: fixed_order_mean is the mean (1e-6 in float32, within one bf16 ulp in bf16), and a row's result is bitwise
    the same alone and in a batch (along dim 1 of (B, 32, H, D) and dim -1 of (H, B, 32)). Negative control: the same
    adds in the reverse order give other bits somewhere, so the order is a real constraint."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 32, 2, 8, generator=g)
    m = BD.fixed_order_mean(x, 1)
    assert m.shape == (5, 2, 8) and float((m - x.mean(dim=1)).abs().max()) < 1e-6
    for b in range(5):
        assert torch.equal(BD.fixed_order_mean(x[b:b + 1], 1)[0], m[b])
    y = torch.randn(2, 6, 32, generator=g)
    my = BD.fixed_order_mean(y, -1)
    assert my.shape == (2, 6) and float((my - y.mean(dim=-1)).abs().max()) < 1e-6
    for b in range(6):
        assert torch.equal(BD.fixed_order_mean(y[:, b:b + 1], -1)[:, 0], my[:, b])
    xb = x.to(torch.bfloat16)
    mb = BD.fixed_order_mean(xb, 1)
    assert mb.dtype == torch.bfloat16 and float((mb.float() - xb.float().mean(dim=1)).abs().max()) <= 2 ** -7
    rev = BD.fixed_order_mean(x.flip(1), 1)
    assert not torch.equal(rev, m)


def _twin(B, H=2, D=8, S=(200, 233, 217), seed=0, mean_fn=None):
    g = torch.Generator().manual_seed(seed)
    N = max(S) + 120
    keys = torch.randn((3, N, H, D), generator=g)
    cis = torch.randn((3, N, H), generator=g)
    return keys, cis


def _run_twin(rows, keys, cis, S, steps, mean_fn):
    """A twin of len(rows) requests holding requests `rows` of the world (keys, cis), prefilled and stepped."""
    B, H, D = len(rows), keys.shape[2], keys.shape[3]
    t = TW.PositionalTwin(B, H, D, lk_max=max(S) // 16 + steps // 16 + 8, n_max=max(S) + steps + 64)
    t.mean_fn = mean_fn
    for j, r in enumerate(rows):
        t.prefill(j, keys[r, :S[r]], cis[r, :S[r]])
    for s in range(steps):
        k = torch.stack([keys[r, S[r] + s] for r in rows])
        c = torch.stack([cis[r, S[r] + s] for r in rows])
        t.step(k, c)
    return t


def test_G2_twin_mean_hook():
    """G2: the twin's summary mean is a hook. mean_fn None is torch's .mean (bitwise the same tables as an explicit
    .mean hook); the fixed-order mean gives tables within 1e-5 of it with the same counters; and with the fixed-order
    mean, request 1 stepped alone (B = 1) has bitwise the tables it has in a batch of 3 (40 steps: key and cis
    compressions on every 16th step). Negative control: the reverse-order mean is caught by the same comparison."""
    S = (200, 233, 217)
    keys, cis = _twin(3, S=S)
    base = _run_twin([0, 1, 2], keys, cis, S, 40, None)
    expl = _run_twin([0, 1, 2], keys, cis, S, 40, lambda x, d: x.mean(dim=d))
    fixed = _run_twin([0, 1, 2], keys, cis, S, 40, BD.fixed_order_mean)
    for a in (expl, fixed):
        assert torch.equal(a.pos, base.pos) and torch.equal(a.n_comp, base.n_comp) and torch.equal(a.n_ccis, base.n_ccis)
    assert torch.equal(expl.comp_k, base.comp_k) and torch.equal(expl.ccis, base.ccis)
    assert float((fixed.comp_k - base.comp_k).abs().max()) < 1e-5 and float((fixed.ccis - base.ccis).abs().max()) < 1e-5
    alone = _run_twin([1], keys, cis, S, 40, BD.fixed_order_mean)

    def same_row(a, b, ra, rb):
        n, m, p = int(a.n_comp[ra]), int(a.n_ccis[ra]), int(a.pos[ra])
        return (torch.equal(a.comp_k[ra, :n], b.comp_k[rb, :n]) and torch.equal(a.ccis[:, ra, :m], b.ccis[:, rb, :m])
                and torch.equal(a.total_cis[ra, :p], b.total_cis[rb, :p]) and torch.equal(a.ring[ra], b.ring[rb]))
    assert same_row(fixed, alone, 1, 0)
    rev = _run_twin([1], keys, cis, S, 40, lambda x, d: BD.fixed_order_mean(x.flip(d), d))
    assert not same_row(fixed, rev, 1, 0)


# =====================================================================================================================
# G3 the page table on the body's window, G4 BIOps / attach_bi
# =====================================================================================================================
def _stand_in_build_page_table(block_ids, slots):
    """piece 3's contract (bi_paged_attention.build_page_table): slots in ascending block id, -1 after the valid."""
    key = torch.where(block_ids >= 0, block_ids, torch.full_like(block_ids, 1 << 40))
    order = torch.sort(key, dim=-1, stable=True).indices
    sl = torch.gather(slots, -1, order)
    ok = torch.gather(block_ids >= 0, -1, order)
    return torch.where(ok, sl, torch.full_like(sl, -1)).to(torch.int32), (block_ids >= 0).sum(-1).to(torch.int32)


ATTN = types.SimpleNamespace(build_page_table=_stand_in_build_page_table, bi_paged_decode=lambda *a, **k: None)


def _window(H=2, B=3, topk=64, seed=2):
    """A body window after the update of the step that wrote pos_before: per (head, request) 64 distinct block ids,
    slot 63 = the tail. Request 1 completed its block (pos % 64 == 63): the update renamed slot 63 to T + 1."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.tensor([16128 + 5, 16128 + 63, 16128 + 64], dtype=torch.int64)
    bm = torch.empty((H, B, topk), dtype=torch.int64)
    for b in range(B):
        T = int(pos[b]) // 64
        for h in range(H):
            hist = torch.randperm(T, generator=g)[:topk - 1]
            bm[h, b, :topk - 1] = hist
            bm[h, b, topk - 1] = (int(pos[b]) + 1) // 64                 # what the body's update leaves in slot 63
    eng = types.SimpleNamespace(_block_map=bm, _tail_block_idx_on_gpu=topk - 1)
    return eng, pos


def _pages_ok(table, n, last, eng, pos):
    H, B, topk = eng._block_map.shape
    for b in range(B):
        T = int(pos[b]) // 64
        if int(last[b]) != int(pos[b]) % 64 + 1:
            return False
        for h in range(H):
            ids = eng._block_map[h, b].clone()
            ids[topk - 1] = T
            got = [int(ids[s]) for s in table[b, h, :int(n[b, h])].tolist()]
            if got != sorted(got) or got[-1] != T or int(table[b, h, int(n[b, h]) - 1]) != topk - 1 or len(set(got)) != topk:
                return False
    return True


def test_G3_page_table_names_the_own_block_last():
    """G3: page_table lists, per (request, head), the 64 slots in ascending block id with slot 63 LAST and named by the
    row's own block p // 64 (also for the request whose update renamed slot 63 to T + 1 after a full block: the name
    is restored, the order is the same since T + 1 and T are both above every selected block), and last_len =
    p % 64 + 1. Negative controls: a table in SLOT order (the shipped window order) and a last_len one row short both
    fail the same check."""
    eng, pos = _window()
    table, n, last = BD.page_table(ATTN, eng, pos)
    assert table.dtype == torch.int32 and tuple(table.shape) == (3, 2, 64) and last.tolist() == [6, 64, 1]
    assert _pages_ok(table, n, last, eng, pos)
    slot_order = torch.arange(64, dtype=torch.int32).expand(3, 2, 64).contiguous()
    assert not _pages_ok(slot_order, n, last, eng, pos)
    assert not _pages_ok(table, n, last - 1, eng, pos)


def _fake_state(B=3, body="r2", neg=None):
    twins = [TW.PositionalTwin(B, 2, 8, 20, 300) for _ in range(2)]
    return types.SimpleNamespace(body_mode=body, neg=neg, B=B, cu_q=torch.arange(B + 1, dtype=torch.int32),
                                 twins=twins, meta={})


def test_G4_biops_and_attach_bi():
    """G4: BIOps refuses an attention module without bi_paged_decode / build_page_table; the GEMM mode is a run
    parameter: "exact" (the default: G-b PaddedLinear with m_fix), "fast" (natural F.linear, the weaker claim),
    "bigemm" (G-a: the given linear, and only then a linear); attach_bi sets state.bi, seq_idx = arange(B) int32, every
    twin's mean_fn and the meta. Negative controls: an unknown mode, "bigemm" without a linear, a linear with another
    mode, body 'shipped', a negative-control hook and B > m_fix are refused."""
    with pytest.raises(RP.R2Error):
        BD.BIOps(types.SimpleNamespace(bi_paged_decode=lambda: 0), 8)
    ops = BD.BIOps(ATTN, 8)
    assert ops.m_fix == 8 and ops.mode == "exact" and ops.linear is ops.padded and "G-b" in ops.gemm
    fast = BD.BIOps(ATTN, 8, gemm="fast")
    assert fast.mode == "fast" and fast.linear is torch.nn.functional.linear and fast.gemm.startswith("FAST")
    lin = lambda x, w: x @ w.t()                                         # noqa: E731
    assert BD.BIOps(ATTN, 8, gemm="bigemm", linear=lin, linear_name="bi_linear").gemm == "G-a bi_linear"
    for kw in (dict(gemm="padded"), dict(gemm="bigemm"), dict(linear=lin), dict(gemm="fast", linear=lin)):
        with pytest.raises(RP.R2Error):
            BD.BIOps(ATTN, 8, **kw)
    st = BD.attach_bi(_fake_state(), ops)
    assert st.bi is ops and st.seq_idx.tolist() == [0, 1, 2] and st.seq_idx.dtype == torch.int32
    assert all(t.mean_fn is BD.fixed_order_mean for t in st.twins) and st.meta["bi"]["m_fix"] == 8
    for bad in (_fake_state(body="shipped"), _fake_state(neg=dict(step=0, request=0, short=1)), _fake_state(B=9)):
        with pytest.raises(RP.R2Error):
            BD.attach_bi(bad, ops)
    x, w = torch.randn(5, 16), torch.randn(4, 16)
    assert float((ops.linear(x, w) - x @ w.t()).abs().max()) < 1e-5


# =====================================================================================================================
# G5 subset_state, G6 the dispatch
# =====================================================================================================================
def _engine(B, n_host=300, H=2, D=8):
    eng = types.SimpleNamespace(head_num=H, head_dim=D, block_size=64, topk=64, dtype=torch.float32, device="cpu",
                                pool_blocks=0)
    DB.DecodeBody(False).allocate(eng, B, n_host, pin=False)
    return eng


def _source(B=3, L=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    layers, twins = [], []
    for _ in range(L):
        e = _engine(B)
        for name in ("_k_cpu", "_v_cpu", "_k_gpu", "_v_gpu", "_kv_bias_gpu"):
            getattr(e, name).copy_(torch.randn(getattr(e, name).shape, generator=g))
        e._block_map.copy_(torch.randint(0, 200, e._block_map.shape, generator=g))
        e._cache_lens.copy_(torch.randint(4032, 4096, (B,), generator=g).to(torch.int32))
        layers.append(types.SimpleNamespace(cache_engine=e))
        t = TW.PositionalTwin(B, 2, 8, 30, 400)
        for name in ("comp_k", "ring", "total_cis", "ccis"):
            getattr(t, name).copy_(torch.randn(getattr(t, name).shape, generator=g))
        for name in ("pos", "n_comp", "n_ccis", "epoch"):
            getattr(t, name).copy_(torch.randint(1, 300, (B,), generator=g))
        t.prefilled.fill_(True)
        twins.append(t)
    model = types.SimpleNamespace(layers=[types.SimpleNamespace(wqkv=torch.zeros(1), topk_blocks=64)] * L)
    st = RP.R2State.__new__(RP.R2State)
    st.B, st.H, st.S, st.lk_max, st.n_max, st.M_alloc = B, 2, 250, 30, 400, 9
    st.twins = twins
    st._common(model, B, 2, torch.device("cpu"), "r2", types.SimpleNamespace(body=DB.DecodeBody(True)), True, None,
               torch.tensor([250, 251, 252]))
    st.meta["builder"] = "test"
    return model, types.SimpleNamespace(layers=layers), st


def _subset_matches(src_cache, st, new_cache, s2, rows):
    for l, (a, b) in enumerate(zip(src_cache.layers, new_cache.layers)):
        ea, eb = a.cache_engine, b.cache_engine
        for j, r in enumerate(rows):
            for name in ("_k_cpu", "_v_cpu", "_k_gpu", "_v_gpu", "_kv_bias_gpu", "_cache_lens"):
                if not torch.equal(getattr(eb, name)[j], getattr(ea, name)[r]):
                    return False
            if not torch.equal(eb._block_map[:, j], ea._block_map[:, r]):
                return False
            ta, tb = st.twins[l], s2.twins[l]
            for name in TW_FIELDS:
                if not torch.equal(getattr(tb, name)[j], getattr(ta, name)[r]):
                    return False
            if not torch.equal(tb.ccis[:, j], ta.ccis[:, r]):
                return False
    return s2.pos_host.tolist() == [int(st.pos_host[r]) for r in rows]


TW_FIELDS = ("comp_k", "ring", "total_cis", "pos", "n_comp", "n_ccis", "epoch", "prefilled")


def test_G5_subset_state_copies_the_rows():
    """G5: subset_state(rows) gives a cache and state whose every per-request table (host K / V rows, the GPU window,
    bias window, map, attention length; the twin's tables and counters; the host position mirror) equals the source's
    rows in that order, with the source's widths and f4_fix, B = len(rows), body 'r2'. Negative controls: the same
    check refuses the subset taken in the other order; duplicate, out-of-range and empty rows, and a body-'shipped'
    source, are refused."""
    model, cache, st = _source()
    rows = [2, 0]
    new = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=types.SimpleNamespace(
        head_num=2, head_dim=8, block_size=64, topk=64, dtype=torch.float32, device="cpu", pool_blocks=0)) for _ in range(2)])
    c2, s2 = BD.subset_state(model, cache, st, rows, pin=False, new_cache=new,
                             adapter=types.SimpleNamespace(body=DB.DecodeBody(True)))
    assert c2 is new and c2.r2_state is s2 and s2.B == 2 and s2.body_mode == "r2" and s2.f4_fix is True
    assert (s2.lk_max, s2.n_max, s2.M_alloc) == (st.lk_max, st.n_max, st.M_alloc) and s2.meta["rows"] == rows
    assert _subset_matches(cache, st, c2, s2, rows)
    assert not _subset_matches(cache, st, c2, s2, [0, 2])
    for bad in ([], [0, 0], [3], [-1]):
        with pytest.raises(RP.R2Error):
            BD.subset_state(model, cache, st, bad, pin=False, new_cache=new, adapter=types.SimpleNamespace(body=None))
    st.body_mode = "shipped"
    with pytest.raises(RP.R2Error):
        BD.subset_state(model, cache, st, [0], pin=False, new_cache=new, adapter=types.SimpleNamespace(body=None))


def test_G6_dispatch_to_the_bi_decode(monkeypatch):
    """G6: r2_decode_path.decode_inference runs r2_bi_decode.decode_inference when the state carries BI ops and the R2
    body otherwise (negative control: without state.bi a 2-token input reaches the R2 path's own refusal)."""
    seen = []
    monkeypatch.setattr(BD, "decode_inference", lambda *a: seen.append(a) or "bi")
    cache = types.SimpleNamespace(r2_state=types.SimpleNamespace(bi=object()))
    assert RP.decode_inference(None, torch.zeros(2, 1, dtype=torch.int64), None, None, cache) == "bi" and len(seen) == 1
    cache.r2_state.bi = None
    with pytest.raises(RP.R2Error, match="one token"):
        RP.decode_inference(None, torch.zeros(2, 2, dtype=torch.int64), None, None, cache)
    assert len(seen) == 1


# =====================================================================================================================
# G7 the gate's quality helpers, G8 the step script
# =====================================================================================================================
def test_G7_quality_helpers():
    """G7: on planted logits (N = 6 steps, B = 3, V = 10) quality_records / summarize_quality count exactly the planted
    argmax differences, each in its margin class with its regret; agreement, per-document agreement, exact zeros and
    the histograms add up; first_divergence of a free-running sequence and first_disagreement of the teacher-forced
    records agree; eq_verdict names the first differing step. Negative control: an unplanted copy has no difference."""
    g = torch.Generator().manual_seed(7)
    N, B, V = 6, 3, 10
    ref = torch.randn(N, B, V, generator=g)
    ref[:, :, 0] += 5.0                                                   # token 0 leads by a large margin ...
    ref[2, 1, 3] = ref[2, 1, 0] - 0.05                                    # ... except two planted small margins
    ref[4, 2, 7] = ref[4, 2, 0] - 0.3
    bi = ref.clone()
    bi[2, 1, 3] = bi[2, 1, 0] + 0.01                                      # BI flips both
    bi[4, 2, 7] = bi[4, 2, 0] + 0.02
    bi[0, 0, 5] += 1e-4                                                   # a tiny difference, no flip
    tok = torch.cat([torch.zeros(1, B, dtype=torch.int64), ref.argmax(-1)], 0)
    rec = GATE.quality_records(ref, tok, bi)
    assert rec["ref_token_mismatch"] == 0 and int((~rec["agree"]).sum()) == 2
    summ, diffs = GATE.summarize_quality([("t", rec)], "T")
    assert summ["comparisons"] == N * B and summ["differences"] == 2 and abs(summ["agreement"] - 16 / 18) < 1e-9
    got = sorted((d["step"], d["doc"], d["bi_token"], d["margin_class"]) for d in diffs)
    assert got == [(2, 1, 3, "[0.01, 0.1)"), (4, 2, 7, "[0.1, 0.5)")]
    assert all(d["regret"] > 0 for d in diffs)
    assert summ["max_d"]["exact_zero"] == N * B - 3
    assert sum(h["n"] for h in summ["max_d"]["histogram"]) == N * B == sum(h["n"] for h in summ["ref_margin"]["histogram"])
    cls = {c["cls"]: c for c in summ["differences_by_ref_margin"]}
    assert cls["[0.01, 0.1)"]["differences"] == 1 and cls["[0.1, 0.5)"]["differences"] == 1
    fr = tok.clone()
    fr[3, 1] = 3                                                          # the free-running run takes BI's token at step 2
    fr[4:, 1] = 9
    fr[5, 2] = 7
    assert GATE.first_divergence(fr, tok) == [None, 2, 4]
    assert GATE.first_disagreement(rec) == [None, 2, 4]
    assert GATE.eq_verdict([True, True, False, True]) == (False, "1 of 4 steps differ; first at step 2")
    assert GATE.eq_verdict([True] * 3)[0] and not GATE.eq_verdict([])[0]
    s2, d2 = GATE.summarize_quality([("t", GATE.quality_records(ref, tok, ref.clone()))], "T")
    assert s2["differences"] == 0 and d2 == [] and s2["max_d"]["exact_zero"] == N * B


PIECE3_V2 = dict(sha="7c826c1bc7235ee9153e3af82c99d85836915416", path="nosi/nosi/bi_paged_attention.py",
                 blob="39ed5e1894ca1662f92e384a10e01a4eaab99042")

CHILD_V2 = r'''
import importlib.util, os, sys, types
sys.dont_write_bytecode = True
import torch
root, p3path = sys.argv[1], sys.argv[2]
pkg = types.ModuleType("r2k"); pkg.__path__ = [root]; sys.modules["r2k"] = pkg
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
core = load("r2k.r2_spec_core", os.path.join(root, "r2_spec_core.py"))
p3 = load("piece3_v2", p3path)
assert p3.PAGES_PER_SPLIT == 8 and callable(p3.bi_paged_decode) and callable(p3.build_page_table)
g = torch.Generator().manual_seed(5)
PAGE, D, G, Hk, S, R, P = 64, 16, 4, 2, 24, 4, 20
k = torch.randn(2, S * PAGE, Hk, D, generator=g); v = torch.randn(2, S * PAGE, Hk, D, generator=g)
bias = (torch.rand(2, S * PAGE, Hk, generator=g) - 0.5) * 4
q = torch.randn(R, Hk * G, D, generator=g)
req = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
npages = [20, 9, 8, 3]                                   # two rows past one split of 8 pages, one at it, one below
pages = torch.full((R, Hk, P), -1, dtype=torch.int32)
npg = torch.empty((R, Hk), dtype=torch.int32)
for r in range(R):
    for h in range(Hk):
        ids = torch.sort(torch.randperm(S, generator=g)[:npages[r]]).values
        pages[r, h, :npages[r]] = ids.to(torch.int32); npg[r, h] = npages[r]
last = torch.tensor([17, 64, 1, 40], dtype=torch.int32)
scale = D ** -0.5
mine = core.paged_attention_reference(q, k, v, bias, req, pages, npg, last, scale)
twin = p3.bi_paged_decode_reference(q, k, v, bias, req, pages, npg, last, softmax_scale=scale)
orc = p3.attention_oracle(q, k, v, bias, req, pages, npg, last, softmax_scale=scale)
d_twin = float((mine - twin).abs().max()); d_orc = float((mine.double() - orc).abs().max())
assert d_twin < 1e-5, ("mine vs v2 twin", d_twin)
assert d_orc < 1e-5, ("mine vs v2 oracle", d_orc)
after = p3.attention_oracle(q, k, v, bias / scale, req, pages, npg, last, softmax_scale=scale)
assert float((mine.double() - after).abs().max()) > 1e-2, "bias-after-scale is not caught"
print("CHILD OK %.2e %.2e" % (d_twin, d_orc))
'''


def test_G9_contract_agreement_with_piece3_v2(tmp_path):
    """G9: the step's attention pin (piece 3 v2, 7c826c1: a FIXED 8-page split-KV and a fixed-order combine) computes
    the attention of the independent reference r2_spec_core.paged_attention_reference (one page per step, no split)
    within 1e-5, for rows of 20, 9, 8 and 3 pages (the split order differs from the reference's for more than 8 pages,
    so the agreement is to rounding, not bitwise); its interface is the one r2_bi_decode calls. The pinned blob is
    copied from the shared object store and loaded by path in a fresh process. Negative control: the bias-after-scale
    semantics differs by far more than the tolerance."""
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:%s" % (PIECE3_V2["sha"], PIECE3_V2["path"])],
                          capture_output=True, text=True)
    assert head.returncode == 0 and head.stdout.strip() == PIECE3_V2["blob"], (
        "piece 3 v2 is not reachable at its pin: %r %r" % (head.stdout.strip(), head.stderr.strip()[-300:]))
    body = subprocess.run(["git", "-C", str(ROOT), "show", PIECE3_V2["blob"]], capture_output=True)
    assert body.returncode == 0 and body.stdout
    p3 = tmp_path / "piece3_v2.py"
    p3.write_bytes(body.stdout)
    r = subprocess.run([sys.executable, "-c", CHILD_V2, str(NOSI), str(p3)],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), cwd=str(tmp_path), capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0 and (r.stdout.strip().splitlines() or [""])[-1].startswith("CHILD OK"), (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:]))


def test_G8_step_script_uuid_file_and_pins(tmp_path):
    """G8: r2_g0bi_step.sh reads the GPU uuid from <holders>/<job>.gpu_uuid: a missing file, a "GPU-" prefix and two
    lines STOP (exit 1) before any other check; G0BI_HOLDERS outside DRYRUN is refused; a valid file passes the uuid
    stage; on a clean checkout at HEAD the DRYRUN prints the uuid with its path, piece 3's pins, the BI GEMM unset and
    the gate arguments (256 continuous steps, 4 windows x 64, batch 8, m_fix 32, parts C,NI,W)."""
    script = EFF / "r2_g0bi_step.sh"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    sha = "0" * 40

    def run(job, content=None, dry=True):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, G0BI_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"], env=e, capture_output=True, text=True,
                              timeout=120)
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    r = run(211)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(212, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(213, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(214, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(215, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr and "uuid" not in r.stderr.lower()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        pytest.skip("not a clean git checkout: the full DRYRUN needs one")
    sha = head.stdout.strip()
    r = run(216, uid + "\n")
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the uuid stage: %s" % r.stderr.strip()[-200:])
    assert "gpu_uuid %s (%s/216.gpu_uuid)" % (uid, tmp_path) in r.stdout
    assert ("piece 3 %s:%s blob %s" % (PIECE3_V2["sha"], PIECE3_V2["path"], PIECE3_V2["blob"])) in r.stdout
    assert "BI GEMM unset" in r.stdout
    for arg in ("--biattn /mnt/beegfs/ojerbi/r2-runs/g0bi/g0bi_t1_216.step/piece3_bi_paged_attention.py", "--gemm exact",
                "--batch 8",
                "--S 16128", "--steps-c 256", "--windows 4", "--steps-w 64", "--m-fix 32", "--parts C,NI,W",
                "NOSI_R2_DECODE=1", "NOSI_ATTN_SPLITS=4", "NOSI_POOL_BLOCKS=0"):
        assert arg in r.stdout, arg
    assert "--bigemm" not in r.stdout
