"""CPU tests of R2 milestone 2 step 4 (KERNELS): nosi/nosi/stage1_seqused.py (argument plumbing with a STUB extension,
and the call compared by AST with the shipped wrapper's source) and nosi/nosi/max_pooling_per_request.py (the torch
twin on a hand case; the Triton kernel under TRITON_INTERPRET=1 when triton is installed, else skipped). The GPU gate
is benchmarks/Efficiency/r2_kernels_gate.py. Modules are loaded by path (nosi/nosi/__init__.py imports CUDA kernels).

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_kernels.py
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
S1 = ROOT / "nosi/nosi/stage1_seqused.py"
MPR = ROOT / "nosi/nosi/max_pooling_per_request.py"
MPF = ROOT / "nosi/nosi/max_pooling_fused.py"
SHIPPED = ROOT / "dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = _load(S1, "stage1_seqused_under_test")
mpr = _load(MPR, "max_pooling_per_request_under_test")


class StubExt:
    """The extension's varlen_fwd_stage1, recorded: every call's positional arguments; returns [sentinel]."""

    def __init__(self):
        self.calls = []
        self.sentinel = torch.zeros(1)

    def varlen_fwd_stage1(self, *args, **kw):
        self.calls.append((args, kw))
        return [self.sentinel]


def _inputs(B=3, HQ=4, HKV=2, D=8, n=5):
    g = torch.Generator().manual_seed(7)
    q = torch.randn((B, HQ, D), generator=g)
    k = torch.randn((B * n, HKV, D), generator=g)
    cu_q = torch.arange(B + 1, dtype=torch.int32)
    cu_k = torch.arange(B + 1, dtype=torch.int32) * n
    return q, k, cu_q, cu_k


def test_K1_the_wrapper_passes_seqused_k_at_the_ops_seventh_argument():
    """K1: one op call with 21 positional arguments and no keyword: q reshaped as the shipped wrapper does (view (total_q,
    HKV, G, D), permute (0, 2, 1, 3), reshape (total_q G, HKV, D)), k, v, None (out), cu_seqlens_q x G, cu_seqlens_k,
    SEQUSED_K (the very tensor; None when not given), None (leftpad), block_table None, alibi None, max_seqlen_q x G,
    max_seqlen_k, dropout 0.0, scale D^-0.5, zero_tensors True, causal False, window -1 / -1, softcap 0.0,
    return_softmax True, generator None; the result is the op's first element."""
    q, k, cu_q, cu_k = _inputs()
    seq = torch.tensor([5, 3, 1], dtype=torch.int32)
    for given in (seq, None):
        ext = StubExt()
        r = s1.stage1_seqused(q, k, k, cu_q, cu_k, 1, 5, seqused_k=given, ext=ext)
        assert r is ext.sentinel and len(ext.calls) == 1
        args, kw = ext.calls[0]
        assert kw == {} and len(args) == 21
        want_q = q.view(3, 2, 2, 8).permute(0, 2, 1, 3).reshape(6, 2, 8)
        assert torch.equal(args[0], want_q) and args[1] is k and args[2] is k and args[3] is None
        assert torch.equal(args[4], cu_q * 2) and args[5] is cu_k and args[6] is given
        assert args[7:10] == (None, None, None) and args[10] == 2 and args[11] == 5
        assert args[12:] == (0.0, 8 ** -0.5, True, False, -1, -1, 0.0, True, None)


def _op_call(tree, fn_name):
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "varlen_fwd_stage1"]
    assert len(calls) == 1, (fn_name, len(calls))
    assigns = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            assigns.setdefault(n.targets[0].id, []).append(ast.dump(n.value))
    return calls[0], assigns


def test_K2_without_seqused_k_the_call_is_the_shipped_wrappers_call():
    """K2 (static, against the shipped source dependencies/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py
    infllmv2_attn_stage1_fast): the op call has the same 21 positional argument expressions in the same order except
    the 7th (shipped: the constant None; here: seqused_k), no keyword; the q reshape, the adjusted cu_seqlens_q and
    max_seqlen_q and the default scale are the same expressions."""
    shipped, sa = _op_call(ast.parse(SHIPPED.read_text()), "infllmv2_attn_stage1_fast")
    ours, oa = _op_call(ast.parse(S1.read_text()), "stage1_seqused")
    assert len(shipped.args) == len(ours.args) == 21 and not shipped.keywords and not ours.keywords
    for i, (a, b) in enumerate(zip(shipped.args, ours.args)):
        if i == 6:
            assert isinstance(a, ast.Constant) and a.value is None
            assert isinstance(b, ast.Name) and b.id == "seqused_k"
            continue
        assert ast.dump(a) == ast.dump(b), (i, ast.dump(a), ast.dump(b))
    for name in ("q", "cu_seqlens_q_adjusted", "max_seqlen_q_adjusted", "softmax_scale"):
        assert sa[name] == oa[name], (name, sa[name], oa[name])


@pytest.mark.parametrize("case", ["dtype", "shape", "strided"])
def test_K3_seqused_k_is_checked_before_the_call(case):
    """K3 (negative): an int64 seqused_k (TypeError), a (B + 1,) one, a non-contiguous one (ValueError) are refused
    before the op is called (the op's own checks, flash_api.cpp:901-906, raised in Python first)."""
    q, k, cu_q, cu_k = _inputs()
    bad = dict(dtype=torch.tensor([5, 3, 1], dtype=torch.int64), shape=torch.tensor([5, 3, 1, 1], dtype=torch.int32),
               strided=torch.tensor([5, 0, 3, 0, 1, 0], dtype=torch.int32)[::2])[case]
    ext = StubExt()
    with pytest.raises(TypeError if case == "dtype" else ValueError, match="^stage1_seqused: seqused_k must be"):
        s1.stage1_seqused(q, k, k, cu_q, cu_k, 1, 5, seqused_k=bad, ext=ext)
    assert ext.calls == []


def _direct(score, n_comp, M_b, M, init, local):
    """An independent statement of the pooling (max_pooling_fused.py:38-74, per request): loop form."""
    H, B, N = score.shape
    out = torch.empty((H, B, M), dtype=score.dtype)
    for h in range(H):
        for b in range(B):
            for m in range(M):
                if m >= M_b[b]:
                    out[h, b, m] = float("-inf")
                elif m < init or M_b[b] - m <= local + 1:
                    out[h, b, m] = float("inf")
                else:
                    vals = [float(score[h, b, n]) for n in range(4 * m - 1, 4 * m + 4) if 0 <= n < n_comp[b]]
                    out[h, b, m] = max(vals) if vals else float("-inf")
    return out


def test_K4_the_pooling_twin_on_a_hand_case():
    """K4: pool_reference (H 2, B 3, N 24, M 6; init 1, local 0, so the last own block is forced) equals the loop
    statement exactly: request 0 full (n_comp 24, M_b 6), request 1 short (n_comp 9: block 2 sees windows 7 and 8
    only, M_b 4: blocks 4 and 5 are -inf), request 2 (n_comp 0: every unforced block is -inf, M_b 3)."""
    g = torch.Generator().manual_seed(3)
    q = torch.randn((2, 3, 24), generator=g)
    c = torch.randn((2, 3, 24), generator=g)
    n_comp, M_b = [24, 9, 0], [6, 4, 3]
    pq, pc = mpr.pool_reference(q, c, torch.tensor(n_comp), torch.tensor(M_b), 6, init_blocks=1, local_blocks=0)
    assert torch.equal(pq, _direct(q, n_comp, M_b, 6, 1, 0)) and torch.equal(pc, _direct(c, n_comp, M_b, 6, 1, 0))
    assert float(pq[0, 1, 2]) == max(float(q[0, 1, 7]), float(q[0, 1, 8])) and float(pq[0, 1, 4]) == float("-inf")


def test_K5_the_kernels_under_the_triton_interpreter(monkeypatch):
    """K5 (needs triton; skipped without it): with TRITON_INTERPRET=1, on CPU float32 tensors, the per-request kernel
    with every n_comp[b] = the shipped scalar and every M_b[b] = M equals the SHIPPED kernel (max_pooling_fused.py) on
    both outputs, and with varied n_comp / M_b it equals pool_reference."""
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    new = _load(MPR, "max_pooling_per_request_interpreted")
    old = _load(MPF, "max_pooling_fused_interpreted")
    g = torch.Generator().manual_seed(5)
    H, B, N, M = 2, 3, 40, 10
    q = torch.randn((H, B, N), generator=g)
    c = torch.randn((H, B, N), generator=g)
    po, pc = torch.empty((H, B, M)), torch.empty((H, B, M))
    old.nosa_pooling(q, c, 37, po, pc, stride=16, block_size=64, local_blocks=2, init_blocks=1)
    qo, co = torch.empty((H, B, M)), torch.empty((H, B, M))
    new.nosa_pooling_per_request(q, c, torch.full((B,), 37, dtype=torch.int32), torch.full((B,), M, dtype=torch.int32),
                                 qo, co, stride=16, block_size=64, local_blocks=2, init_blocks=1)
    assert torch.equal(qo, po) and torch.equal(co, pc)
    nc, mb = torch.tensor([40, 21, 5], dtype=torch.int32), torch.tensor([10, 6, 2], dtype=torch.int32)
    new.nosa_pooling_per_request(q, c, nc, mb, qo, co, stride=16, block_size=64, local_blocks=2, init_blocks=1)
    rq, rc = mpr.pool_reference(q, c, nc, mb, M, init_blocks=1, local_blocks=2)
    assert torch.equal(qo, rq) and torch.equal(co, rc)
