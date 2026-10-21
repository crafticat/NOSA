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
F4  the tick profile's pure parts (benchmarks/Efficiency/r2_tick_profile.py): window, cuts, labels (with a refused
    sequence as the negative control), tables, means, verdicts TK2-TK4.
F5  why PLANT 1 is blind on bfloat16 data (M-F f1 GT-NEG): close exponents -> the float32 chain is exact in any order;
    the GT gate's N(0, 1) bfloat16 data: the means almost never change; the ORDER PROBE changes every mean. F2's child
    also runs the probe through the kernels: unplanted == torch, PLANT 1 effective and caught (float32, bfloat16).
F6  the negative-control verdicts of the GPU cell (plant_status, neg_gates) and the fused r2base predictions
    (r2_base_cell.verdicts_fused, --r2 fused / --gt arguments).

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


def run_order(dtype, steps=20):
    """GT-NEG-ORDER on the CPU: three twins of one state with the order probe in their rings (order_probe_ring), fed
    the same inputs (every bank active, the probe's keys at its positions): a (torch), u (unplanted kernel), p
    (PLANT 1). Returns the first step where u != a (must be None), p != u (effective) and p != a (caught)."""
    g = torch.Generator().manual_seed(7)
    S_list, H, D = (230, 233, 250, 190), 2, 8
    data = [(torch.randn(S, H, D, generator=g), torch.randn(S, H, generator=g)) for S in S_list]
    tw3 = []
    for _ in range(3):
        t = TW.PositionalTwin(len(S_list), H, D, 40, 420)
        for b, (k, c) in enumerate(data):
            t.prefill(b, k, c)
        for name in ("comp_k", "ring", "total_cis", "ccis"):
            setattr(t, name, getattr(t, name).to(dtype).contiguous())
        t.mean_fn = BD.fixed_order_mean
        FU.order_probe_ring(t)
        tw3.append(t)
    a, u, p = tw3
    act = torch.ones(len(S_list), dtype=torch.bool)
    broke = eff = got = None
    for s in range(steps):
        keys = FU.order_probe_keys(a.pos, H, D).to(dtype)
        cis = torch.randn(len(S_list), H, generator=g).to(dtype)
        a.step_unchecked(keys, cis, act)
        FU.step_fused(u, keys, cis, act)
        FU.step_fused(p, keys, cis, act, plant=1)
        if broke is None and diff(a, u):
            broke = s
        if eff is None and diff(u, p):
            eff = s
        if got is None and diff(a, p):
            got = s
    return broke, eff, got


order = {}
for dtype in (torch.float32, torch.bfloat16):
    order[str(dtype)] = run_order(dtype)
    broke, eff, got = order[str(dtype)]
    assert broke is None and eff is not None and got is not None, ("F2 child: order probe", dtype, broke, eff, got)
print("F2 OK float32 + bfloat16 bitwise (ragged steps, advance T 3, overflow); 3 of 3 negative controls caught: %s; "
      "order probe (unplanted break, plant 1 effective, caught): %s" % ({k: v[0] for k, v in caught.items()}, order))
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


def _o(name, ts, dur=1.0, rng=None, rid=None, corr=None, cls=None):
    return dict(name=name, cat="kernel", cls=cls or "SMALL", dur_us=float(dur), range=rng, ts=float(ts), rid=rid,
                corr=corr)


def test_F4_tick_profile_pure_parts():
    """F4: the pure parts of the tick profile M-T (benchmarks/Efficiency/r2_tick_profile.py): the window choice, the
    tick cuts (eager region ranges, graph launches), the labels (TF's own; a graph tick's from its reference: equal
    names, or equal without the twin with the fused kernels as t.twin; NEGATIVE CONTROL: another sequence is refused),
    the per-tick tables and their means, and the verdicts TK2-TK4 (CONFIRMED on a table that meets them, REFUTED when
    the fused tick saves nothing)."""
    sys.path.insert(0, str(ROOT / "benchmarks/Efficiency"))
    import r2_tick_profile as TPR
    assert TPR.cap_label((8, 1), 8) == "G_d" and TPR.cap_label((32, 3), 8) == "G_v"
    labels = ["G_d"] * 45 + ["G_v", "G_d", "G_v", "G_d", "G_d", "G_d", "G_v"]
    assert TPR.pick_window(labels, 40, 6) == (42, 48)
    assert TPR.pick_window(["G_d"] * 60, 40, 6) is None
    ops = [_o("p", 1), _o("a", 2), _o("b", 5), _o("c", 6), _o("d", 9), _o("e", 10)]
    segs = TPR.cut(list(reversed(ops)), [9, 2, 6])
    assert [[o["name"] for o in s] for s in segs] == [["a", "b"], ["c"], ["d", "e"]]
    eager = [_o("x", 3, rng="t.tick", rid="r1"), _o("y", 2, rng="t.tick", rid="r1"), _o("z", 8, rng="t.tick", rid="r2"),
             _o("w", 1, rng="t.twin", rid="q"), _o("v", 0)]
    assert TPR.tf_starts(eager) == [2.0, 8.0]
    assert TPR.eager_ranges(eager) == ["-", "-", "-", "t.twin", "out"]
    graph = [_o("a", 5, corr=7), _o("b", 4, corr=7), _o("c", 9, corr=8), _o("u", 1, corr=3)]
    assert TPR.graph_starts(graph, {7, 8}) == [4.0, 9.0]
    ref_names, ref_ranges = ["a", "b", "tw1", "tw2", "c"], ["-", "t.gemm", "t.twin", "t.twin", "out"]
    same = [_o(n, i) for i, n in enumerate(ref_names)]
    assert TPR.transfer(same, ref_names, ref_ranges) == (ref_ranges, "equal")
    fused = [_o("a", 0), _o("b", 1), _o("_twin_kernel", 2, cls="TWINF"), _o("c", 3)]
    assert TPR.transfer(fused, ref_names, ref_ranges) == (["-", "t.gemm", "t.twin", "out"], "equal without the twin")
    lab, how = TPR.transfer([_o("a", 0), _o("x", 1), _o("c", 2)], ref_names, ref_ranges)
    assert lab is None and how.startswith("differs")
    t1 = TPR.tick_table(same, ref_ranges)
    assert t1["total"] == [5, 5.0] and t1["ranges"]["t.twin"] == [2, 2.0] and t1["cross"]["t.gemm"] == {"SMALL": [1, 1.0]}
    t2 = TPR.tick_table(fused, ["-", "t.gemm", "t.twin", "out"])
    m = TPR.mean_tables([t1, t2])
    assert m["ticks"] == 2 and m["total"] == dict(ops=4.5, ms=round(9.0 / 1e3 / 2, 4))
    assert m["ranges"]["t.twin"] == dict(ops=1.5, ms=round(3.0 / 1e3 / 2, 4))
    assert m["classes"]["TWINF"] == dict(ops=0.5, ms=round(1.0 / 1e3 / 2, 4))

    def tab(total_ms, twin_ops, twin_ms):
        return dict(ticks=3, total=dict(ops=0, ms=total_ms), ranges={"t.twin": dict(ops=twin_ops, ms=twin_ms)},
                    classes={}, cross={})
    prof = dict(TG=dict(G_v=tab(136.0, 18500, 30.0), G_d=tab(100.0, 6200, 10.0)),
                TGF=dict(G_v=tab(106.4, 32, 0.4), G_d=tab(90.2, 32, 0.2)))
    v = {n: x for n, x, _ in TPR.verdicts(prof)}
    assert v == {"%s %s" % (k, c): "CONFIRMED" for k in ("TK2", "TK3", "TK4") for c in ("G_v", "G_d")}, v
    prof["TGF"]["G_v"]["total"]["ms"] = 136.0
    v = {n: x for n, x, _ in TPR.verdicts(prof)}
    assert v["TK4 G_v"] == "REFUTED" and v["TK4 G_d"] == "CONFIRMED"
    assert {x for _, x, _ in TPR.verdicts(dict(TG=prof["TG"]))} == {"UNDETERMINED"}


def test_F3_the_body_kernel_under_the_interpreter():
    _child(BODY_CHILD, "F3")


def _chain(x, reverse=False):
    """The kernel's key summary arithmetic over dim 0: acc = row_0; acc = acc + row_w in float32 (rows reversed:
    PLANT 1)."""
    rows = list(x.float().unbind(0))
    if reverse:
        rows = rows[::-1]
    acc = rows[0].clone()
    for r in rows[1:]:
        acc = acc + r
    return acc


def _rev_mean(x, dim):
    return BD.fixed_order_mean(x.flip(dim), dim)


def test_F5_why_plant_1_is_blind_on_bfloat16_and_the_order_probe():
    """F5 (the GT-NEG failure of M-F f1: PLANT 1 'no-op on this data'): (a) 32 bfloat16 values per column whose
    exponents lie within 8 of each other: the float32 chain is EXACT (it equals the float64 sum) in both orders, so the
    reversed chain and the reversed fixed-order mean give the SAME bits; (b) the GT gate's own data, N(0, 1) in
    bfloat16: more than 95 % of the column sums are exact and at most 1e-4 of the bfloat16 means change when the order is
    reversed; (c) NEGATIVE CONTROL of the claim, the order probe (r2_twin_fused.order_probe_keys): every 32-row window
    from a multiple of 16 sums to 0 in window order and to 14 t reversed, and the means differ in every element."""
    g = torch.Generator().manual_seed(5)
    n = 1 << 16
    e = torch.randint(-4, 4, (32, n), generator=g).float()
    sign = torch.where(torch.rand(32, n, generator=g) < 0.5, -1.0, 1.0)
    x = (torch.pow(2.0, e) * (1 + torch.rand(32, n, generator=g)) * sign).to(torch.bfloat16)
    f, r = _chain(x), _chain(x, True)
    assert torch.equal(f.view(torch.int32), r.view(torch.int32))
    assert torch.equal(f.double(), x.double().sum(0))                                  # exact, in any order
    assert torch.equal(BD.fixed_order_mean(x, 0).view(torch.int16), _rev_mean(x, 0).view(torch.int16))
    y = torch.randn(32, 1 << 18, generator=g).to(torch.bfloat16)
    f, r = _chain(y), _chain(y, True)
    exact = float((f.double() == y.double().sum(0)).double().mean())
    fp32_diff = int((f.view(torch.int32) != r.view(torch.int32)).sum())
    bf_diff = int((BD.fixed_order_mean(y, 0).view(torch.int16) != _rev_mean(y, 0).view(torch.int16)).sum())
    assert exact > 0.95 and bf_diff <= 1e-4 * y.shape[1], (exact, fp32_diff, bf_diff)
    q = torch.arange(16 * 40)
    keys = FU.order_probe_keys(q, 2, 3)
    kb = keys.to(torch.bfloat16)
    assert torch.equal(kb.float(), keys)                                               # exact in bfloat16
    for s in range(0, 16 * 38, 16):
        w = kb[s:s + 32]
        f, r = _chain(w), _chain(w, True)
        assert bool((f == 0).all()) and bool((r == 14 * FU.PROBE_T).all()), s
        m_f, m_r = BD.fixed_order_mean(w, 0), _rev_mean(w, 0)
        assert bool((m_f.view(torch.int16) != m_r.view(torch.int16)).all()), s
        assert bool((m_r.float() == 14 * FU.PROBE_T / 32).all()) and bool((m_f.float() == 0).all()), s


def test_F6_negative_control_verdicts_and_the_fused_r2base_predictions():
    """F6: (a) r2_fused_cell.plant_status / neg_gates: a plant that never changes the kernel's output is 'no-op on this
    data' (allowed for plant 1 on the run's data, refused for plants 2 and 3), an effective plant the comparison misses
    fails GT-NEG, an unplanted kernel that differs from torch fails both gates, GT-NEG-ORDER needs plant 1 caught on the
    probe; (b) r2_base_cell.verdicts_fused: CONFIRMED inside the registered bands, REFUTED outside, REPORT at a B without
    a band; --gt only with --r2 fused."""
    sys.path.insert(0, str(ROOT / "benchmarks/Efficiency"))
    import r2_base_cell as BCF
    import r2_fused_cell as FC
    assert FC.plant_status(None, None) == FC.NOOP and FC.plant_status((0, ["ring"]), (0, ["ring"])) == "caught"
    assert FC.plant_status((0, ["ring"]), None) == "MISSED"

    def run(status, broke=None):
        eff = None if status == FC.NOOP else (3, ["comp_k"])
        got = (3, ["comp_k"]) if status == "caught" else None
        return dict(status=status, effective=eff, caught=got, unplanted_differs_at=broke)

    def g(data, order):
        return {x[0]: x[2] for x in FC.neg_gates(data, order)}
    good = {1: run(FC.NOOP), 2: run("caught"), 3: run("caught")}
    assert g(good, run("caught")) == {"GT-NEG": True, "GT-NEG-ORDER": True}
    assert g({**good, 1: run("caught")}, run("caught"))["GT-NEG"]
    assert not g({**good, 1: run("MISSED")}, run("caught"))["GT-NEG"]
    assert not g({**good, 2: run(FC.NOOP)}, run("caught"))["GT-NEG"]
    assert not g({**good, 3: run("caught", broke=5)}, run("caught"))["GT-NEG"]
    assert not g(good, run(FC.NOOP))["GT-NEG-ORDER"] and not g(good, run("caught", broke=0))["GT-NEG-ORDER"]
    assert not g({1: run(FC.NOOP)}, None)["GT-NEG"] and not g(good, None)["GT-NEG-ORDER"]
    exp = BCF.expected_rewarms(BCF.S0, 512)
    sh = dict(rewarms=[list(x) for x in exp], rates=dict(steady_step_ms=100.0, effective_tok_s=100.0),
              gpu_peak_alloc_GB=50.0)
    good = dict(warmup=2, rates=dict(steady_step_ms=97.0, effective_tok_s=290.0), gpu_peak_alloc_GB=55.0)
    ag = dict(n_diverged=0, tf_agreement=1.0, n_requests=64, first_divergence_min=None)
    v = {n: x for n, x, _ in BCF.verdicts_fused(64, sh, good, ag, exp)}
    assert v == dict(P1F="CONFIRMED", P2="CONFIRMED", P3F="CONFIRMED", P4F="CONFIRMED", P5="CONFIRMED"), v
    bad = dict(good, rates=dict(steady_step_ms=100.5, effective_tok_s=250.0))
    v = {n: x for n, x, _ in BCF.verdicts_fused(64, sh, bad, dict(ag, tf_agreement=0.998), exp)}
    assert (v["P1F"], v["P3F"], v["P4F"]) == ("REFUTED", "REFUTED", "REFUTED"), v
    v = {n: x for n, x, _ in BCF.verdicts_fused(100, sh, good, ag, exp)}
    assert (v["P3F"], v["P4F"]) == ("REPORT", "REPORT")
    v = {n: x for n, x, _ in BCF.verdicts_fused(219, sh, dict(good, rates=dict(steady_step_ms=99.0,
                                                                                 effective_tok_s=180.0)), ag, exp)}
    assert (v["P3F"], v["P4F"]) == ("CONFIRMED", "CONFIRMED"), v
    base = ["--out", "x", "--workload", "w", "--batch", "64"]
    assert BCF.parse_args(base).r2 == "built" and BCF.parse_args(base + ["--r2", "fused", "--gt"]).gt
    with pytest.raises(SystemExit):
        BCF.parse_args(base + ["--gt"])
