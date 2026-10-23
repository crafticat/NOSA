"""[resdev] CPU tests of --res-dev (OwnResidency(res_lean=True, res_dev=True) -> ResidencySC.enable_res_dev ->
LeanConsts.dev True; nosi/r2_resdev.py; analysis/evalfast/SCORING_resfix.md 2.3). The lean ops (dev False) stay the
reference; each rewritten op must give the same integer values.

  RD1  OP LEVEL on random inputs, edge shapes included (one row, one column, empty, all True, all False, many ties):
       excl_scan_rows == cumsum(-1) - x; group_start == 1-D cummax of where(first, pos, 0); sorted_free == the sort of
       where(mask, ar, NP); sort_key32 == torch.sort(stable=True) (the permutation and the key values; the int64
       fallback above the bound); _compact_lean, _selected_mask_lean with dev == without dev.
  RD2  FUNCTION LEVEL: test_r2_resfix.py's RL1 comparison (the REFERENCE functions vs the lean ones) run with a dev
       LeanConsts on its 60 draws (pressure draws included): layer mirror (the lazy ranks too), verify, draft, record,
       binds, bias fills; and the record's int32 FIRST keys against the lean record with a host key bound.
  RD3  ROUND LEVEL: the B2 round with --res-lean --res-dev equals the REFERENCE resolve at every tick and every layer
       (test_r2_resfix.py's RL2 comparison), per-stream and pool layout, default and pressure, checked and unchecked.
  RD4  NEGATIVE CONTROLS (L34: each plant is checked on the draws where it changes a value): an inclusive scan in place
       of excl_scan_rows (XPool.bind_lean's ranks), a free list that drops its first free column (sorted_free), and a
       spread that sends every masked-out entry to ONE dummy shared with a real element. RD2's comparison sees each.
  RD5  THE FLAG: refusals and wiring (--res-dev needs --res-lean; OwnResidency(res_dev=True) needs res_lean); static: the
       rewritten ops hold no host read and no host-to-device copy.
  RD6  GPU ONLY (skips on the CPU): RD1's op equalities on CUDA (the device kernels: cub scan, int32 radix sort).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_resdev.py
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


RF = _shared("test_r2_resfix_shared_rd", "test_r2_resfix.py")         # RL1's _case, RL2's rounds, the fixtures
HF, TT, OWN2 = RF.HF, RF.TT, RF.OWN2
RSB, RSC, RSP, XPM = RF.RSB, RF.RSC, RF.RSP, RF.XPM
RD = TT._load("r2_resdev")
p3, ref = RF.p3, RF.ref                                               # the fixtures, by name
_same = RF._same


class DevConsts(RSB.LeanConsts):
    """[resdev] the service's LeanConsts with --res-dev on."""
    dev = True


def _gen(seed):
    return torch.Generator().manual_seed(7000 + seed)


# =====================================================================================================================
# RD1 the rewritten ops on random inputs, edge shapes included
# =====================================================================================================================
SCAN_SHAPES = [(1, 1), (1, 9), (2, 1), (2, 347), (5, 13), (3, 0), (0, 4), (7, 64)]


def _scan_cases(dev="cpu"):
    for i, (R, N) in enumerate(SCAN_SHAPES):
        g = _gen(i)
        for lo, hi in ((0, 2), (-5, 6), (-(1 << 40), 1 << 40)):
            yield torch.randint(lo, hi, (R, N), generator=g, dtype=torch.int64).to(dev)


def _first_cases(dev="cpu"):
    for n in (1, 2, 3, 17, 1000):
        g = _gen(n)
        for p in (0.0, 0.1, 0.5, 1.0):
            f = torch.rand(n, generator=g) < p
            f[0] = True                                               # the lean callers' precondition (first[0])
            yield f.to(dev)


def _mask_cases(dev="cpu"):
    for i, (S, NP) in enumerate([(1, 1), (1, 7), (2, 5), (7, 33), (3, 1), (4, 64), (2, 300)]):
        g = _gen(100 + i)
        for p in (0.0, 0.3, 0.9, 1.0):
            yield (torch.rand((S, NP), generator=g) < p).to(dev)


def _key_cases(dev="cpu"):
    for i, n in enumerate((1, 2, 5, 64, 2000)):
        g = _gen(200 + i)
        for hi in (1, 3, 50, 1 << 20):
            yield torch.randint(0, hi + 1, (n,), generator=g, dtype=torch.int64).to(dev), hi


def _op_diffs(dev="cpu"):
    """Every RD1 comparison on `dev`: the list of differences (empty: all equal) and the case count."""
    diffs, n = [], 0
    for x in _scan_cases(dev):
        n += 1
        if not torch.equal(RD.excl_scan_rows(x), x.cumsum(-1) - x):
            diffs.append(("scan", tuple(x.shape)))
    for f in _first_cases(dev):
        n += 1
        pos = torch.arange(f.shape[0], device=dev)
        refv = torch.cummax(torch.where(f, pos, torch.zeros_like(pos)), 0).values
        got = RD.group_start(f, pos)
        if got.dtype != refv.dtype or not torch.equal(got, refv):
            diffs.append(("group_start", f.shape[0], float(f.float().mean())))
    for mk in _mask_cases(dev):
        n += 1
        S, NP = mk.shape
        ar = torch.arange(NP, device=dev).view(1, NP).expand(S, NP)
        refv = torch.sort(torch.where(mk, ar, torch.full_like(ar, NP)), dim=-1).values
        got = RD.sorted_free(mk)
        if got.dtype != refv.dtype or not torch.equal(got, refv):
            diffs.append(("sorted_free", (S, NP), float(mk.float().mean())))
    for key, hi in _key_cases(dev):
        n += 1
        ks0, p0 = torch.sort(key, stable=True)
        ks1, p1 = RD.sort_key32(key, hi)
        if ks1.dtype != torch.int32 or not torch.equal(p0, p1) or not torch.equal(ks0, ks1.to(torch.int64)):
            diffs.append(("sort_key32", key.shape[0], hi))
        ks2, p2 = RD.sort_key32(key, RD.INT32_MAX)                     # the bound does not fit: the int64 sort
        if ks2.dtype != torch.int64 or not torch.equal(p0, p2) or not torch.equal(ks0, ks2):
            diffs.append(("sort_key32 fallback", key.shape[0], hi))
    return diffs, n


def test_RD1a_the_rewritten_ops_equal_the_lean_ops():
    diffs, n = _op_diffs("cpu")
    assert n >= 90, "precondition: the cases ran"
    assert not diffs, diffs


@pytest.mark.parametrize("seed", range(12))
def test_RD1b_compact_and_selected_mask_with_dev_equal_without(seed):
    g = _gen(300 + seed)
    lean, dev = RSB.LeanConsts("cpu"), DevConsts("cpu")
    for N in (1, 2, 9, 200):
        mask = torch.rand(N, generator=g) < 0.4
        fields = [torch.randint(-3, 50, (N,), generator=g) for _ in range(5)]
        for cap in (1, 2, N, N + 3):
            a = RSP._compact_lean(mask, fields, cap, lean)
            b = RSP._compact_lean(mask, fields, cap, dev)
            assert _same(a, b) is None, (seed, N, cap)
    S, H, NBLK = 6, 2, 11
    parts = []
    for R in (1, 4):
        s_ = torch.randint(-2, NBLK + 2, (R, H, 5), generator=g)
        ok = (s_ >= 0) & (s_ < NBLK) & (torch.rand((R, H, 5), generator=g) < 0.7)
        rb = torch.randint(0, S // H, (R,), generator=g)
        sidn3 = (rb.view(R, 1, 1) * H + torch.arange(H).view(1, H, 1)) * NBLK
        parts.append((s_, ok, sidn3))
    a = RSC._selected_mask_lean(S, H, NBLK, parts, lean)
    b = RSC._selected_mask_lean(S, H, NBLK, parts, dev)
    assert a.dtype == b.dtype == torch.bool and torch.equal(a, b), seed


# =====================================================================================================================
# RD2 the device functions (RL1's comparison with a dev LeanConsts)
# =====================================================================================================================
@pytest.mark.parametrize("seed", RF.SEEDS)
def test_RD2a_the_dev_functions_equal_the_reference(seed):
    diffs, _ = RF._case(seed, cst_cls=DevConsts)
    assert not diffs, (seed, diffs)


def _record_case(seed, cst):
    """The lean sparse record of one RL1 draw with the record constants of a service that numbers K keys (the host
    bound of the FIRST keys): returns the record."""
    d = RF._draw(seed)
    cst.new_tick()
    H, k, l = d.H, d.k, seed % 2
    stub = SimpleNamespace(_lean_c=cst, H=H, nblk=d.NBLK, n_hist=d.NPOS, xpool=None, device=torch.device("cpu"))
    rcv = RSC.ResidencySC._lean_rows_c(stub, d.pv, k)
    rcd = RSC.ResidencySC._lean_rows_c(stub, d.pd, k)
    m = d.dm.layer_lean(l, cst)
    sel_v, sel_d = d.sel[:, d.pv[4]], d.sel[:, d.pd[4]]
    sv_d = sel_d.permute(1, 0, 2)
    v0_d = (sv_d >= 0) & (sv_d < d.pd[1].view(-1, 1, 1)) & (sv_d < d.NBLK)
    vl = RSC.device_verify_lean(m, *d.pv[:4], sel_v, d.C, d.X, cst, rcv, others=[(sv_d, v0_d, rcd["sidn3"])])
    dl = RSB.device_draft_lean(m, *d.pd[:4], sel_d, d.C, d.X, d.R_ver, cst, rcd, s=sv_d, valid0=v0_d)
    K = 8
    kid_all = d.ri(1, K + 1, d.R)
    tk = dict(v=d.pv, d=d.pd, extra=torch.cat([kid_all, torch.zeros(d.L * d.B, dtype=torch.int64)]))
    stub2 = SimpleNamespace(_lean_c=cst, H=H, nblk=d.NBLK, _kids={i: i for i in range(K)})   # len 8: kid <= 8
    rcv2 = RSP.ResidencySparse._lean_rec_c(stub2, tk, "v", d.R, rcv)
    rcd2 = RSP.ResidencySparse._lean_rec_c(stub2, tk, "d", d.R, rcd)
    assert rcv2["fk_bound"] == (K * H + H - 1) * (d.NBLK + 1) + d.NBLK
    return RSP.sparse_layer_record_lean(d.dm.code[l], vl, dl, d.B, H, d.NBLK, d.NPOS, d.Rv * H * k, d.Rd * H * k,
                                        d.R, cst, rcv=rcv2, rcd=rcd2)


@pytest.mark.parametrize("seed", range(0, 60, 3))
def test_RD2b_the_int32_first_keys_give_the_lean_record(seed):
    a = _record_case(seed, RSB.LeanConsts("cpu"))
    b = _record_case(seed, DevConsts("cpu"))
    assert _same(a, b, "rec") is None, seed


# =====================================================================================================================
# RD3 the B2 round, reference vs --res-lean --res-dev, every tick and every layer
# =====================================================================================================================
@pytest.mark.parametrize("layout,cfg,forced,seed,checked", RF.RL2_CASES,
                         ids=["-".join(str(x) for x in c) for c in RF.RL2_CASES])
def test_RD3_the_dev_round_equals_the_reference_at_every_layer(p3, ref, layout, cfg, forced, seed, checked):
    toks, _ = ref
    base = RF._round(RF._engine(p3, layout, cfg, seed, False, checked), toks, forced)
    dev = RF._round(RF._engine(p3, layout, cfg, seed, True, checked, svc_kw=dict(res_dev=True)), toks, forced)
    assert base.svc.res_dev is False and dev.svc.res_lean is True and dev.svc.res_dev is True
    assert dev.svc._lean_c.dev is True
    assert base.err is None and len(base.ticks) > 10, ("precondition: the reference round ran", base.err)
    d = RF._rounds_diff(base, dev)
    assert d is None, ("--res-dev differs from the reference", d, dev.err)
    HF._same_end(base, dev)
    if layout == "pool" and cfg == "pressure":                        # the bind / audit / bias-fill rewrites acted
        assert dev.svc.xpool.binds > 0, "precondition: the X pool bound lanes"


# =====================================================================================================================
# RD4 negative controls: a wrong rewritten op is seen
# =====================================================================================================================
def _plant(monkeypatch, name, wrong):
    """Replace RD.<name> by wrong(real, *args); changed counts the calls where the plant changed the output (L34)."""
    real = getattr(RD, name)
    changed = []

    def planted(*a, **k):
        out_r, out_w = real(*a, **k), wrong(real, *a, **k)
        if _same(out_r, out_w) is not None:
            changed.append(1)
        return out_w
    monkeypatch.setattr(RD, name, planted)
    return changed


def _inclusive(real, x):
    return real(x) + x                                                # the inclusive scan: every rank one too high


def _drop_first_free(real, mask, flat=None):
    out = real(mask, flat=flat)                                       # each row's first free column is lost
    return torch.cat([out[:, 1:], torch.full_like(out[:, :1], mask.shape[1])], dim=1)


def _shared_dummy(real, idx, ok, base, ar):
    return torch.where(ok, idx, torch.zeros_like(idx))                # every masked-out entry hits element 0 (a REAL one)


PLANTS = dict(scan=("excl_scan_rows", _inclusive), free=("sorted_free", _drop_first_free),
              spread=("spread", _shared_dummy))


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_RD4_negative_control_a_wrong_dev_op_is_seen(monkeypatch, name):
    """The plant changes the op's output on some draws (L34 precondition, counted per draw); RD2's comparison must
    report differences on at least 5 draws, and only on draws where the plant changed a value (no other cause). A
    changed op output need not reach a decision on every draw (a shifted free list changes nothing without a grant),
    so the claim is the count, not every draw."""
    fn, wrong = PLANTS[name]
    changed = _plant(monkeypatch, fn, wrong)
    hit, seen = [], []
    for seed in RF.SEEDS:
        n0 = len(changed)
        diffs, _ = RF._case(seed, cst_cls=DevConsts)
        if len(changed) > n0:                                         # L34: the plant changed a value on this draw
            hit.append(seed)
        if diffs:
            seen.append(seed)
    assert len(hit) >= 5, ("precondition: draws where the plant changes a value", name, hit)
    assert len(seen) >= 5, ("the planted %s was not seen" % name, hit, seen)
    assert set(seen) <= set(hit), ("differences without the plant acting", name, sorted(set(seen) - set(hit)))
    print("REPORT RD4 %s: plant acted on %d draws, seen on %d" % (name, len(hit), len(seen)))


# =====================================================================================================================
# RD5 the flag
# =====================================================================================================================
def test_RD5a_the_flag_refusals_and_its_wiring(p3):
    ops = TT.make_ops(p3)
    with pytest.raises(ValueError):                                   # --res-dev needs --res-lean
        TT.make_res_engine(ops, kind="b2", svc_kw=dict(res_dev=True), **RF.CFG["default"])
    assert RSC.ResidencySC.res_dev is False and RSB.LeanConsts.dev is False
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["res_dev"].default is False
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).res_dev is False
    on = G.parse_args(base + ["--backends", "b2", "--X", "64", "--res-lean", "--res-dev"])
    assert on.res_lean and on.res_dev
    with pytest.raises(SystemExit):
        G.parse_args(base + ["--backends", "b2", "--X", "64", "--res-dev"])
    assert 'res_dev=bool(getattr(a, "res_dev", False))' in inspect.getsource(G.make_engine)


def test_RD5b_the_rewritten_ops_hold_no_host_transfer():
    toks = RF.TRM.FORBIDDEN + ("_d2h(",) + RF.TRM.H2D
    fns = (RD.excl_scan_rows, RD.group_start, RD.sorted_free, RD.sort_key32, RD.spread, RSC._verify_tail_dev,
           RSC._mirror_ranks, RSC._LazyRanks, RSB.dev_flat)
    for fn in fns:
        src = inspect.getsource(fn)
        assert [t for t in toks if t in src] == [], fn.__qualname__
        body = src.split('"""', 2)[-1]
        assert "] = True" not in body and "] = False" not in body, fn.__qualname__


# =====================================================================================================================
# RD6 GPU only
# =====================================================================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only: the CUDA kernels of the rewritten ops")
def test_RD6_the_rewritten_ops_on_cuda():
    diffs, n = _op_diffs("cuda")
    assert n >= 90 and not diffs, diffs
