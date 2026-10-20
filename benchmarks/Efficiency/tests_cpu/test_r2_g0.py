"""CPU tests of R2 milestone 2 step 5 (WIRING + G0): nosi/nosi/r2_decode_path.py (the bridge from the shipped prefill
state into the positional twin, the pooling lengths and the negative-control hook, the body adapter's refusal), the
switch appended to nosi/nosi/nosa_llama.py (AST), benchmarks/Efficiency/r2_g0_compare.py (the comparator and its
verdicts on planted differences) and the arguments of benchmarks/Efficiency/r2_g0_gate.py. No CUDA: r2_decode_path is
loaded through a synthetic package (nosi/nosi/__init__.py imports CUDA modules; r2_decode_path imports nosa_llama only
on first use).

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_g0.py
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
import r2_g0_compare as C  # noqa: E402
import r2_g0_gate as G  # noqa: E402


def _package():
    if "r2cpu" not in sys.modules:
        pkg = types.ModuleType("r2cpu")
        pkg.__path__ = [str(NOSI)]
        sys.modules["r2cpu"] = pkg
    spec = importlib.util.spec_from_file_location("r2cpu.r2_decode_path", str(NOSI / "r2_decode_path.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r2cpu.r2_decode_path"] = mod
    spec.loader.exec_module(mod)
    return mod


RD = _package()
TW = sys.modules["r2cpu.r2_positional_twin"]


class FakeLayer:
    """The attributes of InfLLMv2CacheLayer after the shipped prefill of S tokens per request, as the shipped code
    builds them: compress_k_cache_varlen (B, n, H, D), cached_compressed_max_seqlen n, no_compress_k_cache (B, 32, H, D)
    holding positions [S - ncl, S) in rows [0, ncl), no_compress_k_len ncl, total_cis (B, S + 1024, H), compressed_cis
    (H, B, n). Built from a twin's own prefill (the twin computes the compressions with the shipped ops)."""

    def __init__(self, keys, cis, S):
        B, _, H, D = keys.shape
        ref = TW.PositionalTwin(B, H, D, lk_max=S // 16 + 8, n_max=S + 64)
        for b in range(B):
            ref.prefill(b, keys[b, :S], cis[b, :S])
        n = S // 16 - 1
        ncl = 16 + S % 16
        self.compress_k_cache_varlen = ref.comp_k[:, :n].clone()
        self.cached_compressed_max_seqlen = n
        self.no_compress_k_cache = torch.zeros((B, 32, H, D))
        self.no_compress_k_cache[:, :ncl] = keys[:, S - ncl:S]
        self.no_compress_k_len = ncl
        self.total_cis = torch.zeros((B, S + 1024, H))
        self.total_cis[:, :S] = cis[:, :S]
        self.compressed_cis = ref.ccis[:, :, :n].clone()
        self.ref = ref


def _world(B=3, S=200, steps=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn((B, S + steps, 2, 8), generator=g)
    cis = torch.randn((B, S + steps, 2), generator=g)
    return keys, cis


@pytest.mark.parametrize("S", [192, 200, 207])
def test_W1_bridge_equals_the_twins_own_prefill_and_steps_alike(S):
    """W1: a twin bridged from the shipped-layout prefill state equals the twin's own prefill on every live table and
    counter (compressed keys [0, n), live raw keys, total_cis [0, S), compressed cis [0, n), pos / n_comp / n_ccis),
    passes check_invariants, and after 40 identical steps (crossing compression events) the two stay equal."""
    keys, cis = _world(S=S)
    lay = FakeLayer(keys, cis, S)
    ref = lay.ref
    tw = TW.PositionalTwin(3, 2, 8, lk_max=ref.lk_max, n_max=ref.n_max)
    RD.bridge_layer(tw, lay, S)

    def same(a, b):
        assert torch.equal(a.pos, b.pos) and torch.equal(a.n_comp, b.n_comp) and torch.equal(a.n_ccis, b.n_ccis)
        n, m, p = int(a.n_comp.max()), int(a.n_ccis.max()), int(a.pos.max())
        assert torch.equal(a.comp_k[:, :n], b.comp_k[:, :n]) and torch.equal(a.ccis[:, :, :m], b.ccis[:, :, :m])
        assert torch.equal(a.total_cis[:, :p], b.total_cis[:, :p])
        for r in range(a.B):
            assert torch.equal(a.live_keys(r), b.live_keys(r))

    same(tw, ref)
    for u in range(40):
        tw.step(keys[:, S + u], cis[:, S + u])
        ref.step(keys[:, S + u], cis[:, S + u])
        same(tw, ref)
    assert int(tw.n_comp[0]) > S // 16 - 1 and int(tw.n_ccis[0]) > S // 16 - 1      # events happened


def test_W2_bridge_refusals():
    """W2: a compressed-key count other than S // 16 - 1, a live raw-key count other than 16 + S % 16, a compressed
    cis width other than n, and a too-small Lk_max are refused before any write."""
    keys, cis = _world(S=200)
    for attr, val in (("cached_compressed_max_seqlen", 10), ("no_compress_k_len", 17)):     # true: 11 and 24
        lay = FakeLayer(keys, cis, 200)
        setattr(lay, attr, val)
        tw = TW.PositionalTwin(3, 2, 8, lk_max=40, n_max=400)
        with pytest.raises(RD.R2Error, match="^bridge: "):
            RD.bridge_layer(tw, lay, 200)
        assert int(tw.pos.max()) == 0
    lay = FakeLayer(keys, cis, 200)
    lay.compressed_cis = lay.compressed_cis[:, :, :-1]
    with pytest.raises(RD.R2Error, match="compressed cis"):
        RD.bridge_layer(TW.PositionalTwin(3, 2, 8, lk_max=40, n_max=400), lay, 200)
    with pytest.raises(RD.R2Error, match="too small"):
        RD.bridge_layer(TW.PositionalTwin(3, 2, 8, lk_max=5, n_max=400), FakeLayer(keys, cis, 200), 200)


def test_W3_pooling_lengths_and_the_negative_hook():
    """W3: M_b = pos_before // 64 + 1 (int32, contiguous); n_used = n_comp; the hook shortens request r by k from
    step s0 on, and only that request."""
    pos = torch.tensor([16128, 16191, 16192])
    n = torch.tensor([1007, 1010, 1011])
    nu, mb = RD.pooling_lengths(pos, n, step=3, neg=None)
    assert nu.dtype == mb.dtype == torch.int32 and nu.is_contiguous() and mb.is_contiguous()
    assert mb.tolist() == [253, 253, 254] and nu.tolist() == [1007, 1010, 1011]
    neg = dict(step=5, request=1, short=200)
    assert RD.pooling_lengths(pos, n, step=4, neg=neg)[0].tolist() == [1007, 1010, 1011]
    assert RD.pooling_lengths(pos, n, step=5, neg=neg)[0].tolist() == [1007, 810, 1011]
    assert n.tolist() == [1007, 1010, 1011]                                          # the counters are not touched


def test_W4_the_r2_body_adapter_calls_DecodeBody_decode_in_its_order():
    """W4: DecodeBody.decode (r2_decode_body.py, e048b3d) has the parameters of the FINAL interface (eng, key_states,
    value_states, kv_bias, topk_idx, pos, active, pos_host, active_host, diff_offload, gather_k, gather_v_bias), and
    R2Body.update passes them in that order: K and V unsqueezed to (B, 1, H, D), the kernels last; the tuple the body
    returns is returned. DecodeBody refuses a non-bool f4_fix."""
    import importlib
    import inspect
    RB = importlib.import_module("r2cpu.r2_decode_body")
    assert list(inspect.signature(RB.DecodeBody.decode).parameters) == [
        "self", "eng", "key_states", "value_states", "kv_bias", "topk_idx", "pos", "active", "pos_host", "active_host",
        "diff_offload", "gather_k", "gather_v_bias"]
    with pytest.raises(TypeError):
        RB.DecodeBody(None)
    rec = []

    class FakeBody:
        def decode(self, *args):
            rec.append(args)
            return "tuple"

    ad = object.__new__(RD.R2Body)
    ad.body, ad.kernels = FakeBody(), ("diff", "gk", "gvb")
    k, v = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    assert ad.update("eng", k, v, "bias", "topk", "pos", "act", "ph", "ah") == "tuple"
    args = rec[0]
    assert args[0] == "eng" and args[1].shape == (3, 1, 2, 4) and torch.equal(args[1][:, 0], k)
    assert args[2].shape == (3, 1, 2, 4) and torch.equal(args[2][:, 0], v)
    assert args[3:] == ("bias", "topk", "pos", "act", "ph", "ah", "diff", "gk", "gvb")


def test_W5_the_switch_is_appended_and_default_off():
    """W5 (AST of nosi/nosi/nosa_llama.py): R2_DECODE = os.environ.get('NOSI_R2_DECODE', '0') == '1'; the shipped
    function is kept as _shipped_decode_inference = Llama.decode_inference; the switch routes on cache_engine.r2_state
    and otherwise calls the shipped function with every argument; the rebinding 'if R2_DECODE: Llama.decode_inference
    = _decode_inference_r2_switch' is the LAST statement; all of it comes after the last function of 6e6f80a
    (apply_rotary_pos_emb), so no cited line moves; Llama.decode_inference itself has no R2 reference."""
    src = (NOSI / "nosa_llama.py").read_text()
    tree = ast.parse(src)
    body = tree.body
    names = [getattr(n, "name", None) for n in body]
    i_rope = names.index("apply_rotary_pos_emb")
    tail = body[i_rope + 1:]
    assert [type(n).__name__ for n in tail] == ["Assign", "Assign", "FunctionDef", "If"]
    assert ast.unparse(tail[0]) == "R2_DECODE = os.environ.get('NOSI_R2_DECODE', '0') == '1'"
    assert ast.unparse(tail[1]) == "_shipped_decode_inference = Llama.decode_inference"
    sw = ast.unparse(tail[2])
    assert "getattr(cache_engine, 'r2_state', None) is not None" in sw
    assert "_r2.decode_inference(self, input_ids, cu_seqlens, position_ids, cache_engine)" in sw
    assert "return _shipped_decode_inference(self, input_ids, cu_seqlens, position_ids, cache_engine, warmup)" in sw
    assert ast.unparse(tail[3]) == "if R2_DECODE:\n    Llama.decode_inference = _decode_inference_r2_switch"
    llama = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == "Llama")
    di = next(n for n in llama.body if isinstance(n, ast.FunctionDef) and n.name == "decode_inference")
    assert "r2" not in ast.unparse(di).lower()
    lines = src.splitlines()
    assert "def apply_rotary_pos_emb" in lines[916] and "assert block_size == 64" in lines[679]


# ------------------------------------------------------------------ the comparator
def _views(B=3):
    g = torch.Generator().manual_seed(1)
    t = torch.randn((2, B, 5), generator=g)
    w = torch.randn((B, 7, 2), generator=g)
    return dict(topk=(t, 1), kwin=(w, 0), cl=(torch.arange(B), 0), counters=((5, 16, 200), None))


def test_W6_compare_views():
    """W6: equal views -> nothing; a change in request 1 of a (H, B, .) table -> [1]; in request 2 of a (B, .) table ->
    [2]; a shape change -> 'shape'; a different counter tuple -> 'whole'; a name in one view only -> 'missing'."""
    a, b = _views(), _views()
    assert C.compare_views(a, b) == {}
    b["topk"][0][1, 1, 3] += 1
    b["kwin"][0][2, 0, 0] += 1e-6
    b["cl"] = (torch.arange(4), 0)
    b["counters"] = ((5, 17, 200), None)
    b["extra"] = (torch.zeros(1), None)
    assert C.compare_views(a, b) == dict(topk=[1], kwin=[2], cl="shape", counters="whole", extra="missing")


def test_W7_tracker_and_verdicts():
    """W7: verdict_equal passes only with no difference; verdict_neg passes when the first difference is exactly at
    the planted step and only in the planted requests; it fails on no difference (the comparator missed it), on an
    earlier or later first step, and on a difference in another request."""
    t = C.Tracker()
    for s in range(5):
        t.update(s, 0, {})
    assert C.verdict_equal(t)[0] and not C.verdict_neg(t, 3, [0])[0]
    t.update(3, 2, dict(topk=[0]))
    t.update(4, None, dict(logits=[0]))
    assert not C.verdict_equal(t)[0]
    assert C.verdict_neg(t, 3, [0])[0] and not C.verdict_neg(t, 2, [0])[0] and not C.verdict_neg(t, 4, [0])[0]
    assert C.verdict_neg(t, 3, None)[0]
    t.update(4, 5, dict(block_map=[1]))
    ok, why = C.verdict_neg(t, 3, [0])
    assert not ok and "outside requests" in why
    s = t.summary()
    assert s["first_step"] == 3 and s["first"]["topk"] == dict(step=3, layer=2, requests=[0])


def test_W8_gate_arguments():
    """W8: phases must start with P and be known; S must be 0 mod 64; steps in 1 .. 64; the neg step inside the
    window; block_step(S) = 63 for S = 0 mod 64 (the step whose written position completes the block)."""
    base = ["--out", "x", "--workload", "w"]
    for bad in (["--phases", "G0T,P"], ["--phases", "P,G9"], ["--S", "16130"], ["--steps", "65"], ["--neg-step", "64"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    a = G.parse_args(base + ["--batches", "1,8", "--phases", "P,G0T,NEGT"])
    assert a.batches == [1, 8] and a.phases == ["P", "G0T", "NEGT"] and a.steps == 64
    assert G.block_step(16128) == 63 and G.block_step(64) == 63
    assert all((16128 + G.block_step(16128) + 1) % 64 == 0 for _ in [0])
