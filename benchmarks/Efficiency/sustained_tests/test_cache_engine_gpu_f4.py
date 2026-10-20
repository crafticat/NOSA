"""F4 (the sustained run b64_s960_v1, 2026-10-20): cache_engine_gpu's decode compress update added an arange(B) adder
to the (B + 1)-entry cumulative cu_seqlens and raised "The size of tensor a (65) must match the size of tensor b (64)".
These tests execute the two methods of the layer class, extracted from the source by AST, on a stand-in object (no
CUDA, no extension import):
- F4a after a prefill of B requests of lengths n_b and k decode compress events, cu_seqlens = the prefix sums of
  n_b + k and the max length grows by k;
- F4b (negative control) the pre-fix adder arange(B) raises the size mismatch.

    python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/sustained_tests/test_cache_engine_gpu_f4.py
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SRC = Path(__file__).resolve().parents[3] / "nosi" / "nosi" / "cache_engine_gpu.py"
NAMES = ("update_compress_k_prefill", "update_compress_k_decode")


def _methods(src_text):
    tree = ast.parse(src_text)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    seen = {}
    for f in fns:
        seen.setdefault(f.name, f)                     # the layer class's methods (the first definition of each name)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[seen[n] for n in NAMES], type_ignores=[]), str(SRC), "exec"), ns)
    return ns[NAMES[0]], ns[NAMES[1]]


def _run(prefill, decode, lens, k):
    B, H, D = len(lens), 2, 4
    mx = max(lens)
    self = SimpleNamespace()
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    prefill(self, torch.zeros(B * mx, H, D), cu, mx, 0, B)
    for _ in range(k):
        decode(self, torch.zeros(B, 1, H, D), None)
    return self


def test_F4a_decode_compress_events_keep_the_prefix_sums():
    prefill, decode = _methods(SRC.read_text())
    lens, k = [5, 3, 7, 2], 3
    s = _run(prefill, decode, lens, k)
    want = [0] + torch.tensor([n + k for n in lens]).cumsum(0).tolist()
    assert s.cached_compressed_cu_seqlens.tolist() == want
    assert s.cached_compressed_max_seqlen == max(lens) + k


def test_F4b_the_prefix_adder_raises():
    text = SRC.read_text().replace("torch.arange(total_bsz + 1, dtype=cu_seqlens.dtype",
                                   "torch.arange(total_bsz, dtype=cu_seqlens.dtype")
    assert text != SRC.read_text()
    prefill, decode = _methods(text)
    with pytest.raises(RuntimeError, match="must match the size"):
        _run(prefill, decode, [5, 3, 7, 2], 1)
