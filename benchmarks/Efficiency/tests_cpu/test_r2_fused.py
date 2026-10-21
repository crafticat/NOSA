"""CPU tests of the FUSED TWIN STEP (nosi/nosi/r2_twin_fused.py) and the LEAN BODY (nosi/nosi/r2_body_fused.py).
Design: docs/R2_FUSED.md. The kernels run under the Triton interpreter (TRITON_INTERPRET=1) in a fresh process (triton
reads the flag at its first import), as tests/test_r2_writeback.py W5 does.

F1  no triton needed: the fused twin's constants are the twin's (KERNEL, STRIDE, RING); check_twin refuses a wrong
    dtype, a non-contiguous table and a twin without the overflow flag; attach refuses a twin whose mean_fn is not
    fixed_order_mean unless the change is declared, and binds the named methods on the instance only; FusedBody refuses
    a subclass that overrides one of DecodeBody's pieces (the torch body's negative controls); the launch paths call no
    host sync (.item / .cpu / .tolist / .numpy / .nonzero / synchronize).
F2  the twin kernel (interpreter) against PositionalTwin.step_unchecked / advance_unchecked with mean_fn =
    fixed_order_mean, BITWISE (tables, counters, overflow flag, ctr) after every call, float32 and bfloat16 tables:
    30 single steps with ragged activity, then 12 advance ticks of T 3 with ragged n (0 .. 3); every bank passes key and
    cis compression events and three banks a 64-row block boundary (the counts are asserted). An overflow run (an
    unprefilled bank, N_max and Lk_max reached) is bitwise too and sets the flag. NEGATIVE CONTROLS (float32, the same
    schedule): the key summary summed in reverse order, the cis summary read before step 3, and the ring row written for
    an inactive bank are each caught.
F3  the body kernel (interpreter): FusedBody.decode_fixed against DeviceWritebackBody.decode_fixed, BITWISE (every engine
    table, the host rows, err) after every step: B 4, prompts 6016 / 6017 / 6079 / 6100, 150 steps with ragged activity
    (write-backs and inactive requests counted), f4_fix False and True, the same torch stand-ins for the diff kernel,
    the gathers and the write-back (tests/test_r2_decode_body.py); a refused step (a position out of range) sets ERR_POS
    and changes no persistent table in both (the diff's scratch maps are rewritten by both). NEGATIVE CONTROLS: the tail rows written for an inactive request, the tail id not set,
    and an inactive request's load mask kept are each caught.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_fused.py
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi/nosi"
PKG = "r2fusedparent"


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
FU = _load("r2_twin_fused")
FB = _load("r2_body_fused")
RB = _load("r2_decode_body")


def _twin(B=3, H=2, D=8, lk=40, nmax=420, dtype=torch.float32):
    t = TW.PositionalTwin(B, H, D, lk, nmax)
    for name in ("comp_k", "ring", "total_cis", "ccis"):
        setattr(t, name, getattr(t, name).to(dtype).contiguous())
    return t


def test_F1_constants_checks_attach_and_no_sync():
    assert (FU.WIN, FU.STRIDE, FU.RING) == (TW.KERNEL, TW.STRIDE, TW.RING)
    FU.check_twin(_twin())
    FU.check_twin(_twin(dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        FU.check_twin(_twin(dtype=torch.float16))
    t = _twin()
    t.ring = t.ring.transpose(2, 3).contiguous().transpose(2, 3)                       # same shape, not contiguous
    with pytest.raises(ValueError):
        FU.check_twin(t)
    t = _twin()
    del t.overflow
    with pytest.raises(ValueError):
        FU.check_twin(t)
    t = _twin()
    with pytest.raises(ValueError, match="fixed_order_mean"):                         # torch's .mean: not bitwise
        FU.attach(t)
    FU.attach(t, require_fixed_order=False, methods=("step_fixed",))                   # a DECLARED change
    assert t.step_fixed.func is FU.step_fused and t.step_fixed.args == (t,)
    t2 = _twin()
    t2.mean_fn = BD.fixed_order_mean
    FU.attach(t2)
    assert t2.advance_unchecked.func is FU.advance_fused and "step_unchecked" in vars(t2)
    assert "step_unchecked" not in vars(_twin()) and TW.PositionalTwin.step_unchecked is not FU.step_fused
    with pytest.raises(ValueError):
        FU.attach(t2, methods=("commit",))
    # FusedBody refuses an override of a piece its kernel would ignore
    Fused = FB.fused_body_class(RB.DeviceWritebackBody, RB.DecodeBody)
    Fused(False, lambda *a: None)

    class Shifted(Fused):
        def positions(self, pos):
            return pos + 1
    with pytest.raises(TypeError, match="positions"):
        Shifted(False, lambda *a: None)
    assert FB.fused_body_class(RB.DeviceWritebackBody, RB.DecodeBody) is Fused
    # no host sync in the launch paths
    banned = {"item", "cpu", "tolist", "numpy", "nonzero", "synchronize"}
    for mod, fns in ((FU, ("_launch", "step_fused", "advance_fused")), (FB, ("_make_class",))):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in fns:
                calls = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)}
                assert not (calls & banned), (mod.__name__, node.name, calls & banned)


# =====================================================================================================================
# F2 / F3: the kernels under the Triton interpreter, in a fresh process
# =====================================================================================================================
CHILD_HEAD = r'''
import importlib.util, os, sys, types
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton was imported before the interpreter flag was read"
import torch
NOSI, ROOT = sys.argv[1], sys.argv[2]
PKG = "r2fusedchild"


def load_pkg(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [NOSI]
        sys.modules[PKG] = pkg
        stub = types.ModuleType(PKG + ".nosa_llama")
        stub.QK_SELECT = 33
        sys.modules[PKG + ".nosa_llama"] = stub
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, os.path.join(NOSI, mod + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


def bits(t):
    t = t.contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t
'''

TWIN_CHILD = CHILD_HEAD + r'''
TW = load_pkg("r2_positional_twin")
BD = load_pkg("r2_bi_decode")
FU = load_pkg("r2_twin_fused")
assert FU.triton is not None, "F2 child: triton did not import"
FIELDS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis", "overflow")


def diff(a, b):
    return [k for k in FIELDS if not torch.equal(bits(getattr(a, k)), bits(getattr(b, k)))]


def pair(dtype, S_list=(230, 233, 250, 190), H=2, D=8, lk=40, nmax=420, seed=0, unprefilled=0):
    g = torch.Generator().manual_seed(seed)
    B = len(S_list) + unprefilled
    data = [(torch.randn(S, H, D, generator=g), torch.randn(S, H, generator=g)) for S in S_list]
    out = []
    for _ in range(2):
        t = TW.PositionalTwin(B, H, D, lk, nmax)                    # the float32 prefill, then the tables cast
        for b, (k, c) in enumerate(data):
            t.prefill(b, k, c)
        for name in ("comp_k", "ring", "total_cis", "ccis"):
            setattr(t, name, getattr(t, name).to(dtype).contiguous())
        t.mean_fn = BD.fixed_order_mean
        out.append(t)
    return out[0], out[1], g


def run(dtype, plant=0, seed=0, overflow=False):
    if overflow:
        ref, fu, g = pair(dtype, nmax=255, lk=14, seed=seed, unprefilled=1)
    else:
        ref, fu, g = pair(dtype, seed=seed)
    B, H, D = ref.B, ref.H, ref.D
    first, ev = None, dict(key=0, cis=0, cross=0, inactive=0, calls=0)

    def note(where, extra=()):
        nonlocal first
        d = diff(ref, fu) + list(extra)
        if d and first is None:
            first = (where, d)
        if d and plant == 0:
            raise AssertionError("F2 child: %s %s plant 0 differs at %s: %s" % (dtype, "overflow" if overflow else "",
                                                                              where, d))
    for s in range(30):
        act = torch.rand(B, generator=g) < 0.7
        if s % 7 == 0:
            act[:] = True
        keys = torch.randn(B, H, D, generator=g).to(dtype)
        cis = torch.randn(B, H, generator=g).to(dtype)
        p0, k0, c0 = ref.pos.clone(), ref.n_comp.clone(), ref.n_ccis.clone()
        ref.step_unchecked(keys, cis, act)
        FU.step_fused(fu, keys, cis, act, plant=plant)
        ev["key"] += int((ref.n_comp != k0).sum())
        ev["cis"] += int((ref.n_ccis != c0).sum())
        ev["cross"] += int(((p0 // 64) != (ref.pos // 64)).sum())
        ev["inactive"] += int((~act).sum())
        ev["calls"] += 1
        note(("step", s))
    for s in range(12):
        n = torch.randint(0, 4, (B,), generator=g)
        keys = torch.randn(B, 3, H, D, generator=g).to(dtype)
        cis = torch.randn(B, 3, H, generator=g).to(dtype)
        p0, k0, c0 = ref.pos.clone(), ref.n_comp.clone(), ref.n_ccis.clone()
        c1 = ref.advance_unchecked(keys, cis, n)
        c2 = FU.advance_fused(fu, keys, cis, n, plant=plant)
        ev["key"] += int((ref.n_comp != k0).sum())
        ev["cis"] += int((ref.n_ccis != c0).sum())
        ev["cross"] += int(((p0 // 64) != (ref.pos // 64)).sum())
        ev["calls"] += 1
        note(("advance", s), [] if torch.equal(c1, c2) else ["ctr"])
    return first, ev, bool(ref.overflow)


for dtype in (torch.float32, torch.bfloat16):
    first, ev, ovf = run(dtype)
    assert first is None and not ovf, (dtype, first, ovf)
    assert ev["key"] >= 4 and ev["cis"] >= 4 and ev["cross"] >= 3 and ev["inactive"] > 0, (dtype, ev)
    first, ev, ovf = run(dtype, overflow=True)
    assert first is None and ovf, (dtype, "overflow", first, ovf)
caught = {}
for plant in (1, 2, 3):
    first, ev, _ = run(torch.float32, plant=plant)
    caught[plant] = first
assert all(v is not None for v in caught.values()), "F2 child: a negative control was not caught: %s" % caught
print("F2 OK float32 + bfloat16 bitwise (ragged steps, advance T 3, overflow); 3 of 3 negative controls caught: %s"
      % {k: v[0] for k, v in caught.items()})
'''

BODY_CHILD = CHILD_HEAD + r'''
spec = importlib.util.spec_from_file_location("r2_body_tests_for_fused", os.path.join(ROOT, "tests",
                                                                                      "test_r2_decode_body.py"))
TB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TB)
assert TB.wbm is not None, "F3 child: no r2_writeback.py"
FB = load_pkg("r2_body_fused")
assert FB.triton is not None, "F3 child: triton did not import"
Fused = FB.fused_body_class(TB.rb.DeviceWritebackBody, TB.rb.DecodeBody)
WB = TB.wbm.masked_writeback_torch


def run(f4, plant=0, S_list=(6016, 6017, 6079, 6100), steps=150, seed=3):
    B = len(S_list)
    total = max(S_list) + steps + TB.BS
    keys, values, cis = TB._streams(B, total, seed)
    twin = TB.tw.PositionalTwin(B, TB.H_, TB.D_, total // 16, total)
    base = TB._engine(B, total, seed + 1)
    ref_body, fu_body = TB.rb.DeviceWritebackBody(f4, WB), Fused(f4, WB, plant=plant)
    for b, S in enumerate(S_list):
        twin.prefill(b, keys[b, :S], cis[b, :S])
        ref_body.prefill_request(base, b, keys[b, :S], values[b, :S], twin.total_cis[b, :S])
    e1, e2 = TB._clone(base), TB._clone(base)
    err1, err2 = torch.zeros((), dtype=torch.int32), torch.zeros((), dtype=torch.int32)
    g = torch.Generator().manual_seed(seed + 2)
    bi = torch.arange(B)
    first, st = None, dict(full=0, inactive=0)

    def note(where):
        nonlocal first
        d = [k for k in TB.TABLES if not torch.equal(bits(getattr(e1, k)), bits(getattr(e2, k)))]
        d += [] if torch.equal(err1, err2) else ["err"]
        if d and first is None:
            first = (where, d)
        if d and plant == 0:
            raise AssertionError("F3 child: f4 %s plant 0 differs at %s: %s" % (f4, where, d))
    for i in range(steps):
        act = (bi % 2 == 0) if i == 0 else (torch.rand(B, generator=g) < 0.6)
        p = twin.pos.clone()
        st["full"] += int((act & ((p + 1) % TB.BS == 0)).sum())
        st["inactive"] += int((~act).sum())
        tok_k, tok_v, tok_c = keys[bi, p], values[bi, p], cis[bi, p]
        twin.step(tok_k, tok_c, active=act)
        topk = TB._topk(seed, p)
        ref_body.decode_fixed(e1, tok_k.unsqueeze(1), tok_v.unsqueeze(1), twin.total_cis, topk, p, act, err1,
                              *TB.STUBS)
        fu_body.decode_fixed(e2, tok_k.unsqueeze(1), tok_v.unsqueeze(1), twin.total_cis, topk, p, act, err2,
                             *TB.STUBS)
        note(i)
    # a refused step: request 0 at a position past the host rows; nothing changes, ERR_POS in both
    p = twin.pos.clone()
    bad = p.clone()
    bad[0] = total + 5
    act = torch.ones(B, dtype=torch.bool)
    snap = {k: getattr(e1, k).clone() for k in TB.PERSIST}           # the scratch maps (new map, load mask) are
    tok = keys[bi, p.clamp(max=total - 1)]                             # rewritten by the diff in both bodies
    for body, e, err in ((ref_body, e1, err1), (fu_body, e2, err2)):
        err.zero_()
        body.decode_fixed(e, tok.unsqueeze(1), tok.unsqueeze(1), twin.total_cis, TB._topk(seed, p), bad, act, err,
                          *TB.STUBS)
    note("refused")
    if plant == 0:
        assert int(err1) == int(err2) == TB.rb.DecodeBody.ERR_POS
        for k in TB.PERSIST:
            assert torch.equal(bits(getattr(e2, k)), bits(snap[k])), "F3 child: the refused step changed %s" % k
    return first, st


for f4 in (False, True):
    first, st = run(f4)
    assert first is None, (f4, first)
    assert st["full"] >= 4 and st["inactive"] > 0, (f4, st)
caught = {}
for plant in (1, 2, 3):
    caught[plant] = run(False, plant=plant, steps=80)[0]
assert all(v is not None for v in caught.values()), "F3 child: a negative control was not caught: %s" % caught
print("F3 OK FusedBody == decode_fixed bitwise (f4_fix False / True, ragged, write-backs, refusal); 3 of 3 negative "
      "controls caught: %s" % {k: v[0] for k, v in caught.items()})
'''


def _child(script, tag):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1", NOSI_REPO=str(ROOT))
    r = subprocess.run([sys.executable, "-c", script, str(NOSI), str(ROOT)], env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=3000)
    last = r.stdout.strip().splitlines()[-1:]
    assert r.returncode == 0 and last and last[0].startswith(tag + " OK"), (
        "%s child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (tag, r.returncode, r.stdout[-4000:],
                                                                     r.stderr[-6000:]))


def test_F2_the_twin_kernel_under_the_interpreter():
    _child(TWIN_CHILD, "F2")


def test_F3_the_body_kernel_under_the_interpreter():
    _child(BODY_CHILD, "F3")
