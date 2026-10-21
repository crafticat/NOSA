"""CPU tests of the FUSED TWIN STEP (nosi/nosi/r2_twin_fused.py, from 586f49a) and of its option in the SG5 tick line:
TickEngine(twin="fused") (nosi/nosi/r2_tick.py, the lines marked [twin-variant]) and r2_tick_sg5.py --twin. The kernels
run under the Triton interpreter (TRITON_INTERPRET=1) in a fresh process (triton reads the flag at its first import).

F1  (586f49a) no triton needed: the constants are the twin's; check_twin refuses a wrong dtype, a non-contiguous table
    and a twin without the overflow flag; attach refuses a mean_fn other than fixed_order_mean unless declared, binds
    on the instance only; the launch paths call no host sync.
F2  (586f49a) the kernel against step_unchecked / advance_unchecked, BITWISE (tables, counters, overflow, ctr) after
    every call, float32 and bfloat16; an overflow run; three planted faults caught (float32); the ORDER PROBE through
    the kernels (unplanted == torch; PLANT 1 effective and caught, float32 and bfloat16).
F5  (586f49a) why PLANT 1 is blind on bfloat16 data, and the order probe.
S1  TickEngine's twin variant: "torch" by default (nothing bound), "fused" binds step_unchecked / advance_unchecked of
    every twin to the kernel, a clone keeps the variant, an unknown variant and a mean_fn other than fixed_order_mean
    are refused (TickError); r2_tick_sg5.py --twin (default torch, fused accepted, others refused) reaches make_engine
    and the meta.
S2  ONE SG5-SHAPED ROUND (the interpreter): the free-running greedy round (r2_tick.run_round, K 3, LAMBDA 7;
    draft_drop 0.5, so the tiny model's round has rejections) on piece 5's Residency (ResidencySC, C 12 / X 24 /
    R_ver 4, K3X's adversarial transport), the tiny model, B 4: the fused
    twin equals the torch twin BITWISE in the tick plan (rows), the decisions (argmax per tick, committed tokens, the
    decision reads / rollbacks / publications) and the twin counters per tick and every twin table at the end; the
    host mirrors agree. NEGATIVE CONTROL: PLANT 2 in the fused path (the cis summary read before the write) changes the
    fused output (effective) and the comparison with the torch twin sees it (caught).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_twin_fused.py
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TT = _load_by_path("test_r2_tick", HERE / "test_r2_tick.py")       # the tick's CPU world (its package stub r2tick)
TK = TT.TK
TW = TT.TW
BD = TT.BD
FU = TT._load("r2_twin_fused")                                       # the module r2_tick's relative import binds
p3 = TT.p3                                                           # pytest fixture, re-exported


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
    t.ring = t.ring.transpose(2, 3).contiguous().transpose(2, 3)
    with pytest.raises(ValueError):
        FU.check_twin(t)
    t = _twin()
    del t.overflow
    with pytest.raises(ValueError):
        FU.check_twin(t)
    t = _twin()
    with pytest.raises(ValueError, match="fixed_order_mean"):
        FU.attach(t)
    FU.attach(t, require_fixed_order=False, methods=("step_fixed",))
    assert t.step_fixed.func is FU.step_fused and t.step_fixed.args == (t,)
    t2 = _twin()
    t2.mean_fn = BD.fixed_order_mean
    FU.attach(t2)
    assert t2.advance_unchecked.func is FU.advance_fused and "step_unchecked" in vars(t2)
    assert "step_unchecked" not in vars(_twin()) and TW.PositionalTwin.step_unchecked is not FU.step_fused
    with pytest.raises(ValueError):
        FU.attach(t2, methods=("commit",))
    banned = {"item", "cpu", "tolist", "numpy", "nonzero", "synchronize"}
    tree = ast.parse(Path(FU.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_launch", "step_fused", "advance_fused"):
            calls = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            assert not (calls & banned), (node.name, calls & banned)


CHILD_HEAD = r'''
import importlib.util, os, sys, types
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton was imported before the interpreter flag was read"
import torch
NOSI, ROOT = sys.argv[1], sys.argv[2]


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
        t = TW.PositionalTwin(B, H, D, lk, nmax)
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
            raise AssertionError("F2 child: %s plant 0 differs at %s: %s" % (dtype, where, d))
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


def run_order(dtype, steps=20):
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
order = {}
for dtype in (torch.float32, torch.bfloat16):
    order[str(dtype)] = run_order(dtype)
    broke, eff, got = order[str(dtype)]
    assert broke is None and eff is not None and got is not None, ("F2 child: order probe", dtype, broke, eff, got)
print("F2 OK float32 + bfloat16 bitwise (ragged steps, advance T 3, overflow); 3 of 3 negative controls caught: %s; "
      "order probe (unplanted break, plant 1 effective, caught): %s" % ({k: v[0] for k, v in caught.items()}, order))
'''

TICK_CHILD = CHILD_HEAD + r'''
import functools
spec = importlib.util.spec_from_file_location("test_r2_tick", os.path.join(ROOT, "benchmarks", "Efficiency",
                                                                           "tests_cpu", "test_r2_tick.py"))
TT = importlib.util.module_from_spec(spec)
sys.modules["test_r2_tick"] = TT
spec.loader.exec_module(TT)
spec = importlib.util.spec_from_file_location("piece3_v2_child", sys.argv[3])
P3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P3)
TK = TT.TK
FU = TT._load("r2_twin_fused")
assert FU.triton is not None, "S2 child: triton did not import"
ops = TT.make_ops(P3)
FIELDS = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis", "overflow")


def run(twin, plant=0):
    """ONE SG5-shaped round: the free-running greedy round on ResidencySC (C 12 / X 24 / R_ver 4, the adversarial
    transport), B 4, the tiny model."""
    eng = TT.make_res_engine(ops, kind="sc", C=12, X=24, R_ver=4, twin=twin, draft_drop=0.5)
    assert eng.twin_variant == twin
    if plant:                                                     # the planted fault, in the fused path only
        for t in eng.twins:
            t.advance_unchecked = functools.partial(FU.advance_fused, t, plant=plant)
    rec = []

    def on_tick(rows, tokens, out):
        ctr = torch.stack([torch.stack((t.pos, t.n_comp, t.n_ccis, t.epoch)) for t in eng.twins])
        rec.append(([(r.b, r.kind, r.pos) for r in rows], out.argmax.clone(), ctr.clone()))
    rnd = TK.run_round(eng, TT.FIRST, [TT.STEPS] * eng.B, K=3, LAMBDA=7, on_tick=on_tick)
    eng.check_overflow()
    tables = [{k: bits(getattr(t, k)).clone() for k in FIELDS} for t in eng.twins]
    st = {k: eng.stats[k] for k in ("ticks", "rows", "decision_reads", "rollbacks", "publications", "void_ticks")}
    return dict(rec=rec, out=[list(r.out) for r in rnd.reqs], tables=tables, stats=st, mirror=eng.mirror_failures())


def first_diff(x, y):
    if len(x["rec"]) != len(y["rec"]):
        return ("ticks", len(x["rec"]), len(y["rec"]))
    for i, (a, b) in enumerate(zip(x["rec"], y["rec"])):
        if a[0] != b[0]:
            return (i, "rows")
        if not torch.equal(a[1], b[1]):
            return (i, "argmax")
        if not torch.equal(a[2], b[2]):
            return (i, "twin counters")
    if x["out"] != y["out"]:
        return ("end", "committed tokens")
    if x["stats"] != y["stats"]:
        return ("end", "decisions %s / %s" % (x["stats"], y["stats"]))
    for l, (a, b) in enumerate(zip(x["tables"], y["tables"])):
        for k in FIELDS:
            if not torch.equal(a[k], b[k]):
                return ("end", "layer %d %s" % (l, k))
    return None


rt = run("torch")
rf = run("fused")
assert rt["mirror"] == [] and rf["mirror"] == [], (rt["mirror"], rf["mirror"])
assert rt["stats"]["ticks"] > 20 and rt["stats"]["rollbacks"] > 0 and rt["stats"]["publications"] > 0, rt["stats"]
assert all(len(o) == TT.STEPS for o in rt["out"]), [len(o) for o in rt["out"]]
d = first_diff(rt, rf)
assert d is None, "S2 child: the fused twin differs from the torch twin: %s" % (d,)
try:
    rp = run("fused", plant=2)
    eff, got = first_diff(rf, rp), first_diff(rt, rp)
except Exception as e:                                            # the fault may void its own round: also a catch
    eff = got = ("raised", "%s: %s" % (type(e).__name__, str(e)[:120]))
assert eff is not None and got is not None, ("S2 child: PLANT 2 not effective / caught", eff, got)
print("S2 OK the SG5-shaped round (ResidencySC, free-running, %d ticks, %d decision reads, %d rollbacks, %d "
      "publications): fused == torch bitwise; PLANT 2 effective at %s, caught at %s"
      % (rt["stats"]["ticks"], rt["stats"]["decision_reads"], rt["stats"]["rollbacks"], rt["stats"]["publications"],
         eff, got))
'''


def _child(script, tag, *extra):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1", NOSI_REPO=str(ROOT))
    r = subprocess.run([sys.executable, "-c", script, str(NOSI), str(ROOT)] + [str(x) for x in extra], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=3000)
    last = r.stdout.strip().splitlines()[-1:]
    assert r.returncode == 0 and last and last[0].startswith(tag + " OK"), (
        "%s child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (tag, r.returncode, r.stdout[-4000:],
                                                                     r.stderr[-6000:]))


def test_F2_the_twin_kernel_under_the_interpreter():
    _child(TWIN_CHILD, "F2")


def _chain(x, reverse=False):
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
    g = torch.Generator().manual_seed(5)
    n = 1 << 16
    e = torch.randint(-4, 4, (32, n), generator=g).float()
    sign = torch.where(torch.rand(32, n, generator=g) < 0.5, -1.0, 1.0)
    x = (torch.pow(2.0, e) * (1 + torch.rand(32, n, generator=g)) * sign).to(torch.bfloat16)
    f, r = _chain(x), _chain(x, True)
    assert torch.equal(f.view(torch.int32), r.view(torch.int32))
    assert torch.equal(f.double(), x.double().sum(0))
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
    assert torch.equal(kb.float(), keys)
    for s in range(0, 16 * 38, 16):
        w = kb[s:s + 32]
        f, r = _chain(w), _chain(w, True)
        assert bool((f == 0).all()) and bool((r == 14 * FU.PROBE_T).all()), s
        m_f, m_r = BD.fixed_order_mean(w, 0), _rev_mean(w, 0)
        assert bool((m_f.view(torch.int16) != m_r.view(torch.int16)).all()), s
        assert bool((m_r.float() == 14 * FU.PROBE_T / 32).all()) and bool((m_f.float() == 0).all()), s


def test_S1_the_twin_variant_of_the_tick_and_of_sg5(p3):
    ops = TT.make_ops(p3)
    e = TT.make_engine(ops)
    assert e.twin_variant == "torch" and all("advance_unchecked" not in vars(t) for t in e.twins)
    f = TT.make_engine(ops, twin="fused")
    assert f.twin_variant == "fused"
    assert all(vars(t)["advance_unchecked"].func is FU.advance_fused and vars(t)["step_unchecked"].func is FU.step_fused
               and vars(t)["advance_unchecked"].args == (t,) for t in f.twins)
    c = f.clone()
    assert c.twin_variant == "fused" and all(vars(t)["advance_unchecked"].func is FU.advance_fused for t in c.twins)
    assert e.clone().twin_variant == "torch" and f.clone(twin="torch").twin_variant == "torch"
    with pytest.raises(TK.TickError, match="twin must be one of"):
        TT.make_engine(ops, twin="triton")
    with pytest.raises(TK.TickError, match="twin 'fused'"):
        TT.make_engine(SimpleNamespace(linear=ops.linear, attn=ops.attn, mean=None), twin="fused")
    assert TK.TWIN_VARIANTS == ("torch", "fused")
    SG5 = _load_by_path("r2_tick_sg5_cpu", EFF / "r2_tick_sg5.py")
    base = ["--out", "x", "--workload", "w", "--biattn", "p", "--gilfree", "g", "--gilfree-build", "gb"]
    assert SG5.parse_args(base).twin == "torch" and SG5.parse_args(base + ["--twin", "fused"]).twin == "fused"
    with pytest.raises(SystemExit):
        SG5.parse_args(base + ["--twin", "triton"])
    src = (EFF / "r2_tick_sg5.py").read_text()
    assert "TickEngine.from_r2(model, cache, st, ops, hist_slots=C + X, twin=a.twin, **ekw)" in src   # [integ2] bigb's ekw
    assert 'meta["twin"] = a.twin' in src
    tick_src = (NOSI / "r2_tick.py").read_text()
    assert tick_src.count("[twin-variant]") >= 10 and "ctr = twin.advance_unchecked(keys, cis_b, n_dev)" in tick_src


def test_S2_one_sg5_shaped_round_fused_equals_torch_and_a_planted_fault(p3):
    _child(TICK_CHILD, "S2", p3.__file__)
