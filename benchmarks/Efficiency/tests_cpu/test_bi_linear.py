"""CPU tests of the M-INVARIANT linear (nosi/nosi/bi_linear.py; R2-M2 step 7, design note section 3.3 option G-a).
The module is loaded by path. The torch twin runs here; the Triton kernel runs under TRITON_INTERPRET=1 in a FRESH
process (Triton reads the flag at its first import; job 2189872), skipped when triton is not installed.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_bi_linear.py

(compute node only; PYTHONDONTWRITEBYTECODE=1 keeps the checkout free of untracked __pycache__.)
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
MOD = ROOT / "nosi/nosi/bi_linear.py"
GATE = ROOT / "benchmarks/Efficiency/bigemm_gate.py"
STEP = ROOT / "benchmarks/Efficiency/bigemm_step.sh"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = _load(MOD, "bi_linear_under_test")
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
          107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
          229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
M_SAMPLE = sorted(set(list(range(1, 21)) + PRIMES + [63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 299, 300]))


def make(seed, M=300, N=96, K=256, dtype=torch.float32):
    """K = 256 = FOUR 64-column chunks by default: with three chunks (K = 192) every in-order K-split gives the same
    fold ((c0 + c1) + c2), so a split change is invisible (CPU job 2190757; test_three_chunks_hide_any_split)."""
    g = torch.Generator().manual_seed(500 + seed)
    x = torch.randn(M, K, generator=g).to(dtype)
    w = (torch.randn(N, K, generator=g) * 0.1).to(dtype)
    return x, w


def make_wide(seed, M=300, N=96, K=256):
    """fp32 data with a wide dynamic range (each value scaled by e**u, u uniform in [-4, 4]): partial sums of very
    different sizes, so a different grouping of the chunk sums changes the fp32 rounding."""
    g = torch.Generator().manual_seed(700 + seed)
    x = torch.randn(M, K, generator=g) * torch.exp(torch.rand(M, K, generator=g) * 8 - 4)
    w = torch.randn(N, K, generator=g) * torch.exp(torch.rand(N, K, generator=g) * 8 - 4)
    return x, w


def m_dependent(cfg):
    """NEGATIVE CONTROL: a split-K chosen by M (what a heuristic GEMM does): no split up to 64 rows, 2 above. At
    K = 256 (four chunks) split 1 folds ((c0 + c1) + c2) + c3 and split 2 adds (c0 + c1) + (c2 + c3)."""
    def pick(M):
        return cfg._replace(split_k=1 if M <= 64 else 2)
    return pick


def orders_differ(fn, x, w, cfg):
    """The PRECONDITION of every negative control: split 1 and split 2 give different bits on these rows. Without it
    the control (and the invariance checks on the same data) would be blind."""
    return not torch.equal(fn(x, w, config=cfg._replace(split_k=1)), fn(x, w, config=cfg._replace(split_k=2)))


# ---------------------------------------------------------------------------
# (1) the table, the arithmetic
# ---------------------------------------------------------------------------
def test_config_table_is_a_function_of_the_weight_shape_only():
    assert list(inspect.signature(bl.config_for).parameters) == ["N", "K"]
    want = {"wqkv": (64, 2), "wo": (64, 2), "gate_up": (128, 1), "down": (64, 4), "lm_head": (128, 1)}
    for name, (N, K) in bl.NOSA8B_SHAPES.items():
        c = bl.config_for(N, K)
        assert (c.block_n, c.split_k) == want[name], (name, c)
        assert c.block_m == 64 and c.block_k == 64 and c.num_warps == 4 and c.num_stages == 4
    assert bl.NOSA8B_SHAPES == {"wqkv": (4608, 4096), "wo": (4096, 4096), "gate_up": (32768, 4096),
                                "down": (4096, 16384), "lm_head": (73448, 4096)}
    c = bl.config_for(96, 192)
    assert c.split_k == 2 and bl.k_per_split(192, c) == 128          # split 1 runs past K = 192 (masked chunk)
    assert bl.k_per_split(16384, bl.config_for(4096, 16384)) == 4096
    with pytest.raises(ValueError):
        bl.config_for(0, 5)


# ---------------------------------------------------------------------------
# (2) the twin: semantics, invariance, the negative control
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("shape", [(96, 192), (70, 100), (128, 64)])
def test_twin_equals_the_float64_product(dtype, tol, shape):
    N, K = shape
    x, w = make(1, M=37, N=N, K=K, dtype=dtype)
    y = bl.bi_linear_reference(x, w)
    ref = x.double() @ w.double().t()
    assert y.dtype == dtype and y.shape == (37, N)
    assert float((y.double() - ref).abs().max()) <= tol * (1.0 + float(ref.abs().max()))
    x3 = x.view(37, 1, K)                                            # (..., K) inputs keep their leading shape
    assert torch.equal(bl.bi_linear_reference(x3, w).view(37, N), y)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_twin_rows_do_not_depend_on_m_order_or_neighbours(dtype):
    x, w = make(2, dtype=dtype)
    # the data can show a split change (fp32 partials; the precondition is checked in fp32 on the same values)
    assert orders_differ(bl.bi_linear_reference, x.float(), w.float(), bl.config_for(96, 256))
    full = bl.bi_linear_reference(x, w)
    for M in (1, 2, 7, 64, 65, 193, 299):
        assert torch.equal(bl.bi_linear_reference(x[:M], w), full[:M]), M
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(3))
    assert torch.equal(bl.bi_linear_reference(x[perm], w), full[perm])
    x2 = torch.randn_like(x.float()).to(dtype)
    x2[::7] = x[::7]
    assert torch.equal(bl.bi_linear_reference(x2, w)[::7], full[::7])


def test_three_chunks_hide_any_split():
    """Why the first control was blind (CPU job 2190757): at K = 192 (three chunks) splits 1, 2 and 3 all compute
    ((c0 + c1) + c2) -- 0 + c0 == c0 exactly and the partials are added in split order -- so they are bitwise equal.
    Pinned so that a future control never uses fewer than four chunks."""
    x, w = make_wide(5, M=20, K=192)
    cfg = bl.config_for(96, 192)
    outs = [bl.bi_linear_reference(x, w, config=cfg._replace(split_k=s)) for s in (1, 2, 3)]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def test_negative_m_dependent_split_k_is_caught_in_the_twin():
    x, w = make_wide(4)                                     # fp32, K = 256: four chunks
    cfg = bl.config_for(96, 256)
    assert orders_differ(bl.bi_linear_reference, x[:10], w, cfg), "precondition: split 1 and 2 differ on these rows"
    pick = m_dependent(cfg)
    small = bl.bi_linear_reference(x[:10], w, config=pick(10))
    big = bl.bi_linear_reference(x, w, config=pick(300))[:10]
    assert not torch.equal(small, big)
    # control: the fixed table on the same rows is equal
    assert torch.equal(bl.bi_linear_reference(x[:10], w), bl.bi_linear_reference(x, w)[:10])


# ---------------------------------------------------------------------------
# (3) the Triton kernel under the interpreter (fresh process)
# ---------------------------------------------------------------------------
INTERP_CHILD = r'''
import importlib.util, os, sys
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton imported before the flag was read"
sys.dont_write_bytecode = True
import torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

bl = load(sys.argv[1], "bi_linear_interp")
t = load(sys.argv[2], "bi_linear_tests")
fails = []
# the data can show a split change on the KERNEL (four chunks): else every invariance check below is blind
x, w = t.make(10, N=96, K=256)
if not t.orders_differ(bl.bi_linear, x, w, bl.config_for(96, 256)):
    print("FAILS [('precondition-unmet: the interpreted kernel gives the same bits for split 1 and 2 at K = 256',)]")
    sys.exit(1)
for N, K in ((96, 256), (70, 100)):                    # split 2 over four chunks; odd N and K (a masked chunk)
    x, w = t.make(10, N=N, K=K)
    full = bl.bi_linear(x, w)
    ref = x.double() @ w.double().t()
    twin = bl.bi_linear_reference(x, w)
    # 1e-3: an interpreter may emulate TF32 for an fp32 dot; an indexing error is O(1)
    if float((full.double() - ref).abs().max()) > 1e-3 * (1 + float(ref.abs().max())):
        fails.append(("semantics", N, K, float((full.double() - ref).abs().max())))
    if float((full - twin).abs().max()) > 1e-3 * (1 + float(twin.abs().max())):
        fails.append(("twin", N, K))
    for M in range(1, 301):                            # EVERY M = 1 .. 300 (primes included)
        if not torch.equal(bl.bi_linear(x[:M], w), full[:M]):
            fails.append(("M", N, K, M))
    for seed in (1, 2):
        perm = torch.randperm(300, generator=torch.Generator().manual_seed(seed))
        if not torch.equal(bl.bi_linear(x[perm], w), full[perm]):
            fails.append(("permutation", N, K, seed))
    x2 = torch.randn_like(x)
    x2[::5] = x[::5]
    if not torch.equal(bl.bi_linear(x2, w)[::5], full[::5]):
        fails.append(("neighbours", N, K))
    cfg = bl.config_for(N, K)
    ws = torch.full((cfg.split_k, 400, N), float("nan"))
    if not torch.equal(bl.bi_linear(x[:123], w, workspace=ws), full[:123]):
        fails.append(("workspace", N, K))
# the no-split path (split 1) is invariant too
x, w = t.make(11)
one = bl.config_for(96, 256)._replace(split_k=1)
full1 = bl.bi_linear(x, w, config=one)
for M in t.M_SAMPLE:
    if not torch.equal(bl.bi_linear(x[:M], w, config=one), full1[:M]):
        fails.append(("split1-M", M))
# NEGATIVE CONTROL: a split-K chosen by M must be caught; its precondition first (the two orders differ on the rows)
xw, ww = t.make_wide(12)
cfg = bl.config_for(96, 256)
if not t.orders_differ(bl.bi_linear, xw[:64], ww, cfg):
    fails.append(("negative-precondition-unmet",))
else:
    pick = t.m_dependent(cfg)
    caught = any(not torch.equal(bl.bi_linear(xw[:M], ww, config=pick(M)), bl.bi_linear(xw, ww, config=pick(300))[:M])
                 for M in (5, 33, 64))
    if not caught:
        fails.append(("negative-not-caught",))
if fails:
    print("FAILS", fails[:20])
    sys.exit(1)
print("BIGEMM-INTERP OK")
'''


def test_kernel_under_the_triton_interpreter():
    """The Triton kernel itself on CPU float32 (TRITON_INTERPRET=1, a fresh process): within rounding of the float64
    product and of the twin; for EVERY M = 1 .. 300 the rows torch.equal the M = 300 rows; permutations, new
    neighbours, a preallocated (NaN-filled, larger) workspace, the split-1 path; the M-dependent split-K is caught."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD, str(MOD), str(Path(__file__).resolve())], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["BIGEMM-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


# ---------------------------------------------------------------------------
# (4) source pins
# ---------------------------------------------------------------------------
def test_kernel_source_keeps_the_law_and_the_attribution():
    src = MOD.read_text()
    lines = src.splitlines()
    assert lines[0].startswith("# Adapted from SGLang python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py")
    assert "thinking-machines-lab/batch_invariant_ops" in src and "Apache License, Version 2.0" in src
    assert "triton.autotune" not in src and "triton.heuristics" not in src
    assert src.count('do_not_specialize=["M", "s_ps"]') == 2
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bi_linear")
    code = ast.get_source_segment(src, fn).split('"""')[2]
    code = "\n".join(l.split("#")[0] for l in code.splitlines())
    for bad in (".item(", ".tolist(", ".cpu(", "bool(", "synchronize"):
        assert bad not in code, bad
    assert "cfg = config_for(N, K) if config is None else config" in code
    assert "tl.static_range(1, SPLIT_K)" in src


def test_gate_driver_shape():
    src = GATE.read_text()
    assert src.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())')
    assert 'print("BIGEMM_GATE_DONE failed=%d" % nfail, flush=True)' in src
    assert "MS = (1, 64, 128, 192, 256, 320)" in src and "M_FIX = 320" in src and "F.linear" in src


def test_step_script_reads_the_holder_uuid_and_keeps_its_pins():
    text = STEP.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert "HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders" in text and 'UFILE=$HOLDERS/$JOB.gpu_uuid' in text
    assert "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" in text
    assert 'mkdir "$OUTROOT/bigemm_active"' in text and "STEP_MIN=15" in text and "--export=NONE" in text
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-bigemm" in text and "bigemm_gate.py" in text
    assert text.index('UFILE=$HOLDERS/$JOB.gpu_uuid') < text.index('mkdir "$OUTROOT/bigemm_active"')
    assert "<<" not in text and "ulimit" not in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
