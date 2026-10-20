"""CPU tests of the device-driven masked write-back (nosi/nosi/r2_writeback.py). Design: docs/R2_WRITEBACK.md.

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2-wb python -m pytest -q tests/test_r2_writeback.py

NOSI_REPO defaults to the tree that holds this file. Modules are loaded by file path (nosi/nosi/__init__.py imports
CUDA modules).

W1  the torch reference masked_writeback_torch == the body's torch backend (DecodeBody.writeback), bitwise, over the
    WHOLE NaN-poisoned host buffers (an extra write replaces a NaN, a missing write leaves one).
W2  check_args refuses malformed arguments (no triton needed).
W3  no host sync in the launch path: masked_writeback / _launch / check_args call none of .item / .cpu / .tolist /
    .numpy / .nonzero / synchronize (AST).
W5  the Triton KERNEL under TRITON_INTERPRET=1 in a fresh process (triton reads the flag at its first import):
    == the torch reference on four cases (the fourth checks the row-range guard) and == DecodeBody.writeback on the
    three in-range ones, and the three negative controls (row shift -1, row shift +1, unmasked) are each caught by the
    same comparison. Skipped without triton.
The decode-body integration (DeviceWritebackBody through G-EQ, G-RAG, G-POISON) is in tests/test_r2_decode_body.py.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
WB = NOSI_REPO / "nosi/nosi/r2_writeback.py"
BODY = NOSI_REPO / "nosi/nosi/r2_decode_body.py"

if not WB.is_file() or not BODY.is_file():
    pytest.skip("no r2_writeback.py / r2_decode_body.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wb = _load(WB, "nosi_r2_writeback_under_test")
rb = _load(BODY, "nosi_r2_decode_body_for_wb_test")

B, H, D, BS, TOPK, N = 6, 2, 8, 64, 4, 640
TAIL0 = (TOPK - 1) * BS


def bits(t):
    return t.contiguous().view(torch.int32)


def _tables(g):
    k = torch.randn((B, TOPK * BS, H, D), generator=g)
    v = torch.randn((B, TOPK * BS, H, D), generator=g)
    return k, v, torch.full((B, N, H, D), float("nan")), torch.full((B, N, H, D), float("nan"))


def test_W1_reference_equals_the_body_torch_backend():
    g = torch.Generator().manual_seed(1)
    for trial in range(20):
        k, v, kc, vc = _tables(g)
        full = torch.rand(B, generator=g) < 0.5
        pos = torch.randint(BS, N, (B,), generator=g)
        pos = torch.where(full, (pos // BS) * BS + BS - 1, pos)          # full requests end a block (in range)
        wb.masked_writeback_torch(k, v, kc, vc, full, pos, TAIL0, BS)
        e = types.SimpleNamespace(block_size=BS, _tail_block_idx_on_gpu=TOPK - 1, _k_gpu=k, _v_gpu=v,
                                  _k_cpu=torch.full_like(kc, float("nan")), _v_cpu=torch.full_like(vc, float("nan")))
        rb.DecodeBody(False).writeback(e, full, full.clone(), pos, pos.clone())
        assert torch.equal(bits(e._k_cpu), bits(kc)) and torch.equal(bits(e._v_cpu), bits(vc)), "trial %d" % trial
        for b in range(B):                                              # and the written rows are the tail rows
            if bool(full[b]):
                p = int(pos[b])
                assert torch.equal(kc[b, p - BS + 1:p + 1], k[b, TAIL0:TAIL0 + BS])


def test_W2_check_args_refusals():
    g = torch.Generator().manual_seed(2)
    k, v, kc, vc = _tables(g)
    full = torch.zeros(B, dtype=torch.bool)
    pos = torch.zeros(B, dtype=torch.int64)
    wb.check_args(k, v, kc, vc, full, pos, TAIL0, BS)                  # the good call passes
    bad = [
        (k, v[:, :-1], kc, vc, full, pos, TAIL0, BS),                  # K / V shapes differ
        (k, v, kc[:, :, :1], vc[:, :, :1], full, pos, TAIL0, BS),      # host heads differ
        (k, v, kc.double(), vc.double(), full, pos, TAIL0, BS),        # dtypes differ
        (k, v.transpose(2, 3).contiguous().transpose(2, 3), kc, vc, full, pos, TAIL0, BS),   # V strides differ
        (k, v, kc, vc, full.int(), pos, TAIL0, BS),                    # mask not bool
        (k, v, kc, vc, full, pos.int(), TAIL0, BS),                    # positions not int64
        (k, v, kc, vc, full[:-1], pos, TAIL0, BS),                     # mask shape
        (k, v, kc, vc, full, pos, TAIL0 + 1, BS),                      # tail rows past the GPU rows
        (k, v, kc, vc, full, pos, TAIL0, 48),                          # block size not a power of two
        (k, v, kc, vc, full, torch.zeros(2 * B, dtype=torch.int64)[::2], TAIL0, BS),   # positions not contiguous
    ]
    for i, args in enumerate(bad):
        with pytest.raises(ValueError):
            wb.check_args(*args)
        with pytest.raises(ValueError):
            wb.masked_writeback(*args)                                 # refused before any launch (triton or not)
    k3 = torch.randn((B, TOPK * BS, H, 6), generator=g)               # head_dim not a power of two
    with pytest.raises(ValueError):
        wb.check_args(k3, k3.clone(), torch.zeros((B, N, H, 6)), torch.zeros((B, N, H, 6)), full, pos, TAIL0, BS)


def test_W3_no_host_sync_in_the_launch_path():
    tree = ast.parse(WB.read_text(encoding="utf-8"))
    banned = {"item", "cpu", "tolist", "numpy", "nonzero", "synchronize"}
    seen = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("masked_writeback", "_launch", "check_args"):
            seen[fn.name] = sorted({n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)} & banned)
    assert seen == {"masked_writeback": [], "_launch": [], "check_args": []}, seen


WB_CHILD = r'''
import importlib.util, os, sys, types
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "WB child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "WB child: triton was imported before the interpreter flag was read"
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wb = load(sys.argv[1], "r2_writeback_interpreted")
rb = load(sys.argv[2], "r2_decode_body_interpreted")
assert wb.triton is not None, "WB child: triton did not import"
B, H, D, BS, TOPK, N = 6, 2, 8, 64, 4, 640
TAIL0 = (TOPK - 1) * BS
g = torch.Generator().manual_seed(8)


def bits(t):
    return t.contiguous().view(torch.int32)


def tables():
    k = torch.randn((B, TOPK * BS, H, D), generator=g)
    v = torch.randn((B, TOPK * BS, H, D), generator=g)
    return k, v, torch.full((B, N, H, D), float("nan")), torch.full((B, N, H, D), float("nan"))


cases = [
    ([True, False, True, False, False, True], [127, 300, 575, 64, 0, 639]),   # in range; non-full rows elsewhere
    ([False] * B, [63, 127, 191, 255, 319, 383]),                             # nothing is full (positions end blocks)
    ([True] * B, [63, 127, 191, 255, 319, 639]),                              # everything is full
    ([True, True, False, False, False, False], [30, 700, 5, 5, 5, 5]),        # the guard: rows < 0 and >= N dropped
]
for full_l, pos_l in cases:
    k, v, kc, vc = tables()
    full = torch.tensor(full_l)
    pos = torch.tensor(pos_l, dtype=torch.int64)
    rk, rv = kc.clone(), vc.clone()
    wb.masked_writeback_torch(k, v, rk, rv, full, pos, TAIL0, BS)
    wb.masked_writeback(k, v, kc, vc, full, pos, TAIL0, BS)
    assert torch.equal(bits(kc), bits(rk)) and torch.equal(bits(vc), bits(rv)), \
        "WB child: kernel != torch reference on %s %s" % (full_l, pos_l)
    if all(p - (BS - 1) >= 0 and p < N for f, p in zip(full_l, pos_l) if f):
        e = types.SimpleNamespace(block_size=BS, _tail_block_idx_on_gpu=TOPK - 1, _k_gpu=k, _v_gpu=v,
                                  _k_cpu=torch.full_like(kc, float("nan")), _v_cpu=torch.full_like(vc, float("nan")))
        rb.DecodeBody(False).writeback(e, full, full.clone(), pos, pos.clone())
        assert torch.equal(bits(e._k_cpu), bits(kc)) and torch.equal(bits(e._v_cpu), bits(vc)), \
            "WB child: kernel != DecodeBody.writeback on %s %s" % (full_l, pos_l)
k, v, kc0, vc0 = tables()
full = torch.tensor(cases[0][0])
pos = torch.tensor(cases[0][1], dtype=torch.int64)
rk, rv = kc0.clone(), vc0.clone()
wb.masked_writeback_torch(k, v, rk, rv, full, pos, TAIL0, BS)
caught = []
for shift, masked in ((-1, True), (1, True), (0, False)):
    kc, vc = kc0.clone(), vc0.clone()
    wb.check_args(k, v, kc, vc, full, pos, TAIL0, BS)
    wb._launch(k, v, kc, vc, full, pos, TAIL0, BS, shift, masked)
    caught.append(not (torch.equal(bits(kc), bits(rk)) and torch.equal(bits(vc), bits(rv))))
assert caught == [True, True, True], "WB child: a negative control was not caught: %s" % caught
print("WB OK %d cases, 3 of 3 negative controls caught" % len(cases))
'''


def test_W5_the_kernel_under_the_triton_interpreter():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", WB_CHILD, str(WB), str(BODY)], env=env, cwd=str(NOSI_REPO),
                       capture_output=True, text=True, timeout=900)
    last = r.stdout.strip().splitlines()[-1:]
    assert r.returncode == 0 and last and last[0].startswith("WB OK"), (
        "WB child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
