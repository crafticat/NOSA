"""[flean] CPU tests of the fused resolver with --res-lean / --res-dev (r2_fusedres.FusedResolver._lean_on: the lean
host checks, XPool.bind_lean and draft_bias_fill_flat_lean in place of _check_rows / _row_sig / _new_plan, XPool.bind
and draft_bias_fill_flat; with --res-dev, LeanConsts.dev selects r2_resdev's ops inside them). The fused kernels do not
change. The check mode's shadow pool keeps the REFERENCE bind. The kernels run under the Triton interpreter
(TRITON_INTERPRET=1) in a FRESH process (L36), kernel v3 (FR.DEFAULT_KERNEL = "v3", FR.KERNELS = ("v3",)).

  FL1  FUNCTION LEVEL, in process: the hook's calls on test_r2_resfix.py's 60 random draws (RF._draw: the pressure
       draws and the C_FREE-heavy draws, seed % 3 == 0, included), with the int64 decision fields that the kernels
       write: XPool.bind_lean with FusedResolver._xbase's base == XPool.bind, then a release and a bind at the other
       layer; draft_bias_fill_flat_lean == draft_bias_fill_flat. LeanConsts and a dev LeanConsts.
  FL2  ROUND LEVEL (child): fused v3 + --res-lean --res-dev == fused v3 alone at EVERY layer (decisions, records,
       caps, plan), state and counters, on the lockstep schedules and fr3's randomized configs (r6, r12b, r20b, r21b,
       r21c, r21d, r21e: the block of test_r2_fusedres_v3.py's FR3_CHILD, used as it is). The check mode with the flags
       compares every layer, 0 diffs.
  FL3  ROUND LEVEL (child): the B2 tick rounds of test_r2_resfix.py's RL2_CASES (per-stream and pool layout, default
       and pressure, forced and free, checked and unchecked): fused v3 + flags == fused v3 at every tick and layer
       (the X pool's map, free lists, flag, owner, handoffs, peak, binds too), the same end; check mode 0 diffs.
       PRECONDITIONS: a pool pressure round binds and hands a lane between requests; a pressure round evicts.
  FL4  THE HOST WORK (child): with the flags, _check_rows and _row_sig run once per tick, _new_plan never, and no
       Python scalar is scattered through an advanced index inside resolve (the launches excluded); the reference
       fused path: once per layer each, scatters > 0 (the counter's precondition).
  FL5  THE FLAG: the refusal is gone for resolve fused / check only: OwnResidency(resolve fused / check, res_lean,
       res_dev) builds, FusedResolver._lean_on() is True; res_dev without res_lean, --res-lean without --backends b2,
       and the fused path's own negative-control refusals stay. Without the flags _lean_on() is False and the
       reference calls stay in the source.
  FL6  NEGATIVE CONTROLS (L34: each plant is checked where it changes a value): (a) a _lean_rows_ok that skips the
       cancelled-key check, with a key cancelled between two layers: the reference raises, the lean path raises the
       same, the planted path does not; (b) an inclusive scan in place of r2_resdev.excl_scan_rows (bind_lean's ranks):
       the fused + flags round differs from the reference, and the CHECK mode reports xpool.* differences (its shadow
       pool binds with the REFERENCE bind, so the lane compare stays a real check); (c) FL1's comparison sees the same
       wrong scan on >= 5 draws.
  FL7  THE CELLS (static, then a DRYRUN on a clean checkout of this ROOT): evalfast2f3p / evalfast2c3l / evalfast2f3l
       arguments; the REGISTERED block's tags ("STAMP_TBD, before any run").

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_flean.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


RF = _shared("test_r2_resfix_shared_fl", "test_r2_resfix.py")         # RL1's draws, RL2's rounds and cases
V3 = _shared("test_r2_fusedres_v3_shared_fl", "test_r2_fusedres_v3.py")   # fr3's child (its RAND block), SCHED
HF, TT = RF.HF, RF.TT
RSB, RSC, XPM = RF.RSB, RF.RSC, RF.XPM
RD = TT._load("r2_resdev")
FR = TT._load("r2_fusedres")                                          # the same package as XPM (r2tick)
p3 = RF.p3                                                            # the fixture, by name
_same = RF._same


class DevConsts(RSB.LeanConsts):
    """[resdev] the service's LeanConsts with --res-dev on."""
    dev = True


def _need_triton():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")


# =====================================================================================================================
# FL1 the hook's calls on random draws (in process; no kernel)
# =====================================================================================================================
def _hook_case(seed, cst_cls):
    """The fused hook's bind / bias calls on RL1's draw `seed`, against the reference calls of FusedResolver._layer
    without the flags. The REFERENCE decisions stand in for the kernels' (both are int64; FR3 / FR3-R prove them
    equal). Returns (the differences, coverage stats)."""
    d = RF._draw(seed)
    cst = cst_cls("cpu")
    cst.new_tick()
    l, H, k = seed % 2, d.H, d.k
    inflight = seed % 3 != 0
    diffs = []

    def cmp(name, a, b):
        s = _same(a, b, name)
        if s:
            diffs.append(s)
    mr = d.dm.layer(l)
    sel_v, sel_d = d.sel[:, d.pv[4]], d.sel[:, d.pd[4]]
    vr = RSC.device_verify(mr, *d.pv[:4], sel_v, d.C, d.X, others=[(d.pd[0], d.pd[1], sel_d)], inflight=inflight)
    dr = RSB.device_draft(mr, *d.pd[:4], sel_d, d.C, d.X, d.R_ver)
    for f in ("granted", "evict", "take_slot"):                        # the kernels' dtype (the lean torch path: bool)
        if vr.t[f].dtype != torch.int64:
            diffs.append("precondition: int64 verify field " + f)
    xr = XPM.XPool(d.L, d.B, H, d.NPOS, mode="pool", C=d.C, X_pool=d.X + seed % 3)
    xl = xr.clone()
    fz = dict(v=d.pv, d=d.pd, Rv=d.Rv, Rd=d.Rd, bh={})
    stub = SimpleNamespace(svc=SimpleNamespace(H=H, device=torch.device("cpu")), xp=xl)
    stub._bh = lambda fz_, k_: FR.FusedResolver._bh(stub, fz_, k_)
    stub._bhd = lambda fz_, k_: FR.FusedResolver._bhd(stub, fz_, k_)
    bh, bhd = stub._bh(fz, k), stub._bhd(fz, k)
    cmp("bh (precondition: FusedResolver._bh == device_verify's)", (vr.bh[0], vr.bh[1]), bh)
    xbv, xbd = FR.FusedResolver._xbase(stub, fz, "v", k), FR.FusedResolver._xbase(stub, fz, "d", k)
    cmp("xbase v", xbv, (bh[0] * H + bh[1]) * xl.n_hist)
    # FusedResolver._layer: the verify bind (recipe 2), then the draft bind and the bias rows (recipe 3)
    xr.bind(l, bh[0], bh[1], vr.t["take_slot"], (vr.t["granted"] | vr.t["evict"]) != 0)
    xl.bind_lean(l, bh[0], bh[1], vr.t["take_slot"], vr.t["granted"] | vr.t["evict"], cst, base=xbv)
    cmp("xp verify bind", RF._xps(xr), RF._xps(xl))
    xr.bind(l, bhd[0], bhd[1], dr.t["take_slot"], dr.t["granted"] != 0)
    xl.bind_lean(l, bhd[0], bhd[1], dr.t["take_slot"], dr.t["granted"], cst, base=xbd)
    cmp("xp draft bind", RF._xps(xr), RF._xps(xl))
    bs = 4
    cis = torch.randn((d.B, (d.NBLK + 1) * bs, H), generator=d.g)
    store = torch.randn((xr.M * bs, H), generator=d.g)
    sr, sl = store.clone(), store.clone()
    XPM.draft_bias_fill_flat(xr, l, sr, cis, dr, bhd[0], bhd[1], bs)
    XPM.draft_bias_fill_flat_lean(xl, l, sl, cis, dr, bhd[0], bhd[1], bs, cst)
    cmp("bias flat", (sr, xr.err), (sl, xl.err))
    code = d.ri(0, 6, (d.L, d.B, H, d.NPOS))                          # the next tick: a release, a bind elsewhere
    xr.release_free(code)
    xl.release_free(code)
    cst.new_tick()
    fz["bh"] = {}
    xr.bind(1 - l, bh[0], bh[1], vr.t["take_slot"], (vr.t["granted"] | vr.t["evict"]) != 0)
    xl.bind_lean(1 - l, bh[0], bh[1], vr.t["take_slot"], vr.t["granted"] | vr.t["evict"], cst,
                 base=FR.FusedResolver._xbase(stub, fz, "v", k))
    cmp("xp bind after release", RF._xps(xr), RF._xps(xl))
    stats = dict(v_grant=int(vr.t["granted"].sum()), d_grant=int(dr.t["granted"].sum()),
                 v_evict=int(vr.t["evict"].sum()), binds_assigned=int((xr.pos_of >= 0).sum()),
                 cfree_heavy=int(seed % 3 == 0 and int(dr.t["granted"].sum()) > 0))
    return diffs, stats


@pytest.mark.parametrize("cst_cls", [RSB.LeanConsts, DevConsts], ids=["lean", "dev"])
@pytest.mark.parametrize("seed", RF.SEEDS)
def test_FL1a_the_hook_calls_equal_the_reference_calls(seed, cst_cls):
    diffs, _ = _hook_case(seed, cst_cls)
    assert not diffs, (seed, diffs)


def test_FL1b_the_draws_reach_every_bind_path():
    """PRECONDITION of FL1a: the draws grant verify and draft takes, evict, bind lanes, and a C_FREE-heavy draw grants
    draft takes (summed over the draws)."""
    tot = {}
    for seed in RF.SEEDS:
        _, st = _hook_case(seed, RSB.LeanConsts)
        for k_, v in st.items():
            tot[k_] = tot.get(k_, 0) + int(v)
    assert all(v > 0 for v in tot.values()), ("precondition: no draw reaches", tot)
    print("REPORT FL1 coverage over %d draws: %s" % (len(RF.SEEDS), tot))


# =====================================================================================================================
# the child process (TRITON_INTERPRET=1 from its start)
# =====================================================================================================================
def _rand_block():
    """fr3's RAND block (test_r2_fusedres_v3.py FR3_CHILD), as it is: the same generators and seeds."""
    src = V3.FR3_CHILD
    i = src.index("RAND = dict(")
    j = src.index("\n)\n", i) + 3
    return src[i:j]


RAND_NAMES = [ln.split("=")[0].strip() for ln in _rand_block().splitlines() if "=lambda:" in ln.replace(" ", "")]
LOCK_NAMES = sorted(V3.SCHED) + sorted(RAND_NAMES)

FL_CHILD = r'''
import importlib.util, json, os, sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "FL child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "FL child: triton was imported before the interpreter flag was read"
import torch
HERE, PART = sys.argv[1], sys.argv[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


RF = load("test_r2_resfix_flchild", os.path.join(HERE, "test_r2_resfix.py"))
OWN = load("test_r2_own2_flchild", os.path.join(HERE, "test_r2_own2.py"))
TRM, TT = OWN.TRM, RF.TT
FRS = (TRM._load("r2_fusedres"), TT._load("r2_fusedres"))
for FR in FRS:
    assert FR.triton is not None, "FL child: triton did not import"
    FR.DEFAULT_KERNEL = "v3"
    FR.KERNELS = ("v3",)                                # a resolver with any other kernel raises
RD = TT._load("r2_resdev")
LEAN = dict(res_lean=True, res_dev=True)
OUT = {}
#RAND#


def fused_rep(svc):
    r = svc.fusedres.report()
    return dict(kernel=r["kernel"], launches=r["launches"], layers=r["layers"], stats=r["stats"], err=r["err"],
                compared=r["compared"], diffs=r["diffs"])


def p3_toks(path):
    P3 = load("piece3_v2_flchild", path)
    toks, _ = TT.serial_ref(TT.make_engine(TT.make_ops(P3)))
    return P3, toks


def engine(P3, layout, cfg, seed, mode, lean, checked=False):
    kw = dict(resolve=mode)
    if lean:
        kw["res_dev"] = True
    return RF._engine(P3, layout, cfg, seed, lean, checked, svc_kw=kw)


if PART == "lock":
    name = sys.argv[3]
    kws = dict(OWN.SCHEDULES)
    kws.update(RAND)
    REC = []
    FRR = FRS[0]
    orig = FRR.FusedResolver.resolve

    def recorded(self, l, rows, sel):
        plan = orig(self, l, rows, sel)
        REC.append(RF._layer_rec(self.svc, plan))
        return plan
    FRR.FusedResolver.resolve = recorded

    def lock(svc_kw):
        del REC[:]
        bad, wv, wb = OWN._lockstep_b2(svc_kw=svc_kw, **kws[name]())
        svc = wb.svc
        return dict(bad=[repr(x)[:300] for x in bad[:3]], layers=list(REC), state=TRM._state(wb),
                    counters=TRM._counters(svc), rep=fused_rep(svc), lean=svc.fusedres._lean_on(),
                    dev=bool(getattr(getattr(svc, "_lean_c", None), "dev", False)),
                    diffs=[list(d) for d in svc.fusedres.diffs[:5]], b2_applied=svc.b2_applied)
    a = lock(dict(resolve="fused"))
    b = lock(dict(resolve="fused", **LEAN))
    c = lock(dict(resolve="check", **LEAN))
    OUT["lock"] = dict(name=name, n_layers=len(a["layers"]), layer_diff=RF._same(a["layers"], b["layers"], "layers"),
                       bad=[a["bad"], b["bad"], c["bad"]], state_diff=RF._same(a["state"], b["state"], "state"),
                       counters_same=a["counters"] == b["counters"], lean=[a["lean"], b["lean"], c["lean"]],
                       dev=[a["dev"], b["dev"], c["dev"]], rep=[a["rep"], b["rep"], c["rep"]], check_diffs=c["diffs"],
                       b2_applied=[a["b2_applied"], b["b2_applied"]])

elif PART == "pool":
    i, P3, toks = int(sys.argv[3]), *p3_toks(sys.argv[4])
    layout, cfg, forced, seed, checked = RF.RL2_CASES[i]
    base = RF._round(engine(P3, layout, cfg, seed, "fused", False, checked), toks, forced)
    lean = RF._round(engine(P3, layout, cfg, seed, "fused", True, checked), toks, forced)
    chk = RF._round(engine(P3, layout, cfg, seed, "check", True, checked), toks, forced)
    d = RF._rounds_diff(base, lean)
    try:
        RF.HF._same_end(base, lean)
        end = None
    except AssertionError as e:
        end = "same_end: " + repr(e)[:300]
    js = dict(base.svc.journal_counts)
    xp = base.svc.xpool
    OUT["pool"] = dict(case=[layout, cfg, forced, seed, checked], errs=[base.err, lean.err, chk.err],
                       ticks=len(base.ticks), n_layers=sum(len(t["layers"]) for t in base.ticks),
                       diff=None if d is None else repr(d)[:400], end=end,
                       flags=[base.svc.res_lean, lean.svc.res_lean, lean.svc._lean_c.dev, chk.svc._lean_c.dev,
                              base.svc.fusedres._lean_on(), lean.svc.fusedres._lean_on(), chk.svc.fusedres._lean_on()],
                       rep=[fused_rep(base.svc), fused_rep(lean.svc), fused_rep(chk.svc)],
                       check_diffs=[list(x) for x in chk.svc.fusedres.diffs[:5]],
                       evictions=sum(v for k_, v in js.items() if k_.startswith("release:") and "evict" in k_)
                       + base.svc.metrics.counts.get("verifier_evictions", 0),
                       binds=None if xp is None else [xp.binds, lean.svc.xpool.binds],
                       handoffs=None if xp is None else RF.XPT._handoffs(base.maps, xp.n_hist))

elif PART == "count":
    P3, toks = p3_toks(sys.argv[3])
    counter = SimpleNamespace(n=0, on=False)
    orig_set = torch.Tensor.__setitem__

    def setitem(t, idx, val, _o=orig_set, _s=counter):
        if _s.on and isinstance(val, (bool, int, float)):
            ix = idx if isinstance(idx, tuple) else (idx,)
            if any(torch.is_tensor(x) for x in ix):
                _s.n += 1
        return _o(t, idx, val)
    torch.Tensor.__setitem__ = setitem
    FRT = FRS[1]
    orig_launch = FRT._launch

    def quiet_launch(kernel, grid, *a, **k):              # the interpreter's own writes are not resolve's
        was, counter.on = counter.on, False
        try:
            return orig_launch(kernel, grid, *a, **k)
        finally:
            counter.on = was
    FRT._launch = quiet_launch
    got = {}
    for lean in (False, True):
        box = dict(resolve=0, check_rows=0, row_sig=0, new_plan=0)
        counter.n = 0
        r = RF._round(engine(P3, "pool", "pressure", 1, "fused", lean), toks, True, setup=RF._count_setup(box, counter))
        box.update(scalar_scatter=counter.n, err=r.err, ticks=len(r.ticks))
        got["lean" if lean else "base"] = box
    OUT["count"] = dict(got, L=TT.NL_LAYERS)

elif PART == "plant_rows":
    P3, toks = p3_toks(sys.argv[3])
    plant, text = RF.PLANTS["cancel"]

    def run(lean, skip):
        fired, skipped = [], []

        def setup(eng, svc):
            RF._plant_at(5, 1, plant, fired)(eng, svc)
            if skip:                                            # PLANT: _lean_rows_ok without the cancelled check
                real = svc._lean_rows_ok

                def planted(l, rows, sel):
                    ok_real = real(l, rows, sel)
                    saved, svc.cancelled = svc.cancelled, set()
                    try:
                        ok = real(l, rows, sel)
                    finally:
                        svc.cancelled = saved
                    if ok != ok_real:
                        skipped.append(l)
                    return ok
                svc._lean_rows_ok = planted
        r = RF._round(engine(P3, "stream", "default", 1, "fused", lean), toks, True, setup=setup)
        return dict(err=r.err, ticks=len(r.ticks), fired=fired, skipped=skipped)
    OUT["plant_rows"] = dict(text=text, ref=run(False, False), lean=run(True, False), skip=run(True, True))

elif PART == "plant_scan":
    P3, toks = p3_toks(sys.argv[3])
    base = RF._round(engine(P3, "pool", "pressure", 1, "fused", False), toks, True)
    real = RD.excl_scan_rows
    changed = []

    def inclusive(x):                                           # PLANT: every rank one too high where x is 1
        out = real(x) + x
        if not torch.equal(out, real(x)):
            changed.append(1)
        return out
    real_sf = RD.sorted_free

    def sf_real_scan(*a, **k):                                  # sorted_free keeps the REAL scan: else its compaction
        RD.excl_scan_rows = real                                # shifts by one too and gather(free_sorted, rank + 1)
        try:                                                    # picks the same slot (a self-cancelling plant, L34)
            return real_sf(*a, **k)
        finally:
            RD.excl_scan_rows = inclusive
    RD.excl_scan_rows, RD.sorted_free = inclusive, sf_real_scan
    try:
        bad = RF._round(engine(P3, "pool", "pressure", 1, "fused", True), toks, True)
        n_round = len(changed)
        chk = RF._round(engine(P3, "pool", "pressure", 1, "check", True), toks, True)
    finally:
        RD.excl_scan_rows, RD.sorted_free = real, real_sf
    d = RF._rounds_diff(base, bad)
    fr = chk.svc.fusedres
    OUT["plant_scan"] = dict(base_err=base.err, bad_err=bad.err, chk_err=chk.err, changed_round=n_round,
                             changed_check=len(changed) - n_round, diff=None if d is None else repr(d)[:400],
                             check_diffs=[list(x) for x in fr.diffs[:8]], n_check_diffs=len(fr.diffs),
                             compared=fr.compared)
print("FL_JSON " + json.dumps(OUT))
'''


def _child_src():
    assert FL_CHILD.count("#RAND#") == 1
    return FL_CHILD.replace("#RAND#", _rand_block())


def _child(part, *extra):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", _child_src(), str(HERE), part] + [str(x) for x in extra], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=5400)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("FL_JSON ")]
    assert r.returncode == 0 and last, ("FL child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                                        % (part, r.returncode, r.stdout[-4000:], r.stderr[-8000:]))
    return json.loads(last[-1][len("FL_JSON "):])


_LOCK = {}


def _lock(name):
    _need_triton()
    if name not in _LOCK:
        _LOCK[name] = _child("lock", name)["lock"]
    return _LOCK[name]


def test_FL2_0_the_randomized_configs_are_fr3s():
    """The child uses fr3's RAND block as it is, with the seeds r21c-e (the in-flight eviction draws)."""
    blk = _rand_block()
    assert blk in V3.FR3_CHILD and "RAND = dict(" in _child_src()
    assert {"r6", "r12b", "r20b", "r21b", "r21c", "r21d", "r21e"} <= set(RAND_NAMES), RAND_NAMES
    for s in ("seed=29", "_sched_split(311", "seed=31", "_sched_split(313", "seed=7", "_sched_split(317"):
        assert s in blk, s


# =====================================================================================================================
# FL2 lockstep and randomized rounds: fused + flags == fused, every layer
# =====================================================================================================================
@pytest.mark.parametrize("name", LOCK_NAMES)
def test_FL2a_fused_lean_dev_equals_fused_at_every_layer(name):
    r = _lock(name)
    assert r["lean"] == [False, True, True] and r["dev"] == [False, True, True], (name, r["lean"], r["dev"])
    assert all(x["kernel"] == "v3" for x in r["rep"]), (name, [x["kernel"] for x in r["rep"]])
    assert r["n_layers"] > 0, (name, "precondition: fused layers were recorded")
    assert r["layer_diff"] is None, (name, "fused + flags differs from fused", r["layer_diff"])
    assert r["state_diff"] is None and r["counters_same"], (name, r["state_diff"])
    assert r["bad"][0] == r["bad"][1], (name, "the B2-vs-v1 outcome changed with the flags", r["bad"][:2])
    assert r["b2_applied"][0] == r["b2_applied"][1], (name, r["b2_applied"])
    a, b = r["rep"][0], r["rep"][1]
    assert (a["launches"], a["layers"], a["stats"], a["err"]) == (b["launches"], b["layers"], b["stats"], b["err"]), name


@pytest.mark.parametrize("name", LOCK_NAMES)
def test_FL2b_check_mode_with_the_flags_compares_every_layer(name):
    r = _lock(name)
    c = r["rep"][2]
    assert c["compared"] == c["layers"] > 0, (name, c["compared"], c["layers"])
    assert c["diffs"] == 0 and r["check_diffs"] == [], (name, r["check_diffs"])
    assert c["err"] == 0, (name, c["err"])


def test_FL2c_the_set_reaches_every_decision_branch():
    """PRECONDITION over the set (FR3-R's): evictions (LRU and in flight), grants, refusals, verify dups."""
    tot = {k: sum(_lock(n)["rep"][0]["stats"][k] for n in LOCK_NAMES)
           for k in ("v_evict", "v_evict_infl", "v_grant", "d_grant", "d_refused", "v_dup")}
    assert all(v > 0 for v in tot.values()), ("precondition: the set reaches every branch", tot)
    print("REPORT FL2: %d configs, stats %s" % (len(LOCK_NAMES), tot))


# =====================================================================================================================
# FL3 the tick rounds of RL2_CASES (per-stream and pool layout)
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(RF.RL2_CASES)), ids=["-".join(str(x) for x in c) for c in RF.RL2_CASES])
def test_FL3_the_fused_lean_dev_round_equals_the_fused_round(p3, i):
    _need_triton()
    r = _child("pool", i, p3.__file__)["pool"]
    layout, cfg = r["case"][0], r["case"][1]
    assert r["flags"] == [False, True, True, True, False, True, True], r["flags"]
    assert r["errs"][0] is None and r["ticks"] > 10, ("precondition: the reference fused round ran", r["errs"])
    assert r["n_layers"] >= r["ticks"], "precondition: the layers were recorded"
    assert r["diff"] is None, ("fused + flags differs from fused", r["diff"], r["errs"])
    assert r["end"] is None, r["end"]
    a, b, c = r["rep"]
    assert (a["launches"], a["layers"], a["stats"], a["err"]) == (b["launches"], b["layers"], b["stats"], b["err"])
    assert r["errs"][2] is None and c["compared"] == c["layers"] > 0 and c["diffs"] == 0, (r["errs"][2],
                                                                                           r["check_diffs"])
    if cfg == "pressure":
        assert r["evictions"] > 0, "precondition: no eviction"
    if layout == "pool" and cfg == "pressure":
        assert r["binds"][0] > 0 and r["binds"][0] == r["binds"][1] and r["handoffs"] > 0, \
            ("precondition: the X pool bound lanes and handed one between requests", r["binds"], r["handoffs"])


# =====================================================================================================================
# FL4 the host work
# =====================================================================================================================
def test_FL4_the_row_python_once_per_tick_and_no_scalar_scatter(p3):
    _need_triton()
    r = _child("count", p3.__file__)["count"]
    base, lean, L = r["base"], r["lean"], r["L"]
    assert base["err"] is None and lean["err"] is None, (base["err"], lean["err"])
    assert base["resolve"] == lean["resolve"] > 0 and base["resolve"] % L == 0, (base, lean)
    ticks = base["resolve"] // L
    assert base["scalar_scatter"] > 0, "precondition: the counter sees the reference fused path's scalar scatters"
    assert lean["scalar_scatter"] == 0, lean
    assert (base["check_rows"], base["row_sig"], base["new_plan"]) == (base["resolve"],) * 3, base
    assert (lean["check_rows"], lean["row_sig"], lean["new_plan"]) == (ticks, ticks, 0), lean
    print("REPORT FL4 per round (%d ticks x %d layers): fused %s, fused + flags %s" % (ticks, L, base, lean))


# =====================================================================================================================
# FL5 the flag
# =====================================================================================================================
def test_FL5a_the_refusal_is_gone_for_the_fused_resolver_only():
    _need_triton()
    TRM, OWN2 = V3.TRM, V3.OWN2
    for mode in ("fused", "check"):
        w = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve=mode, res_lean=True, res_dev=True))
        svc = w.svc
        assert svc.res_lean and svc.res_dev and svc._lean_c.dev, mode
        assert svc.fusedres is not None and svc.fusedres._lean_on(), mode
        off = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve=mode))
        assert off.svc.fusedres._lean_on() is False and off.svc.res_lean is False, mode   # the default: as before
        with pytest.raises(ValueError, match="res_dev needs res_lean"):                    # stays
            TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve=mode, res_dev=True))
        with pytest.raises(ValueError, match="implements no sc_mut"):                      # the fused refusals stay
            TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve=mode, res_lean=True,
                                                          sc_mut={"no_wait": True}))
    t = TRM.world(cls=OWN2.OwnResidency, svc_kw=dict(nblk=TRM.NB, resolve="torch", res_lean=True))
    assert t.svc.fusedres is None and t.svc.res_lean                  # the torch resolve: _resolve_lean, as before


def test_FL5b_the_flags_wiring_and_the_reference_calls_stay():
    OWN2 = V3.OWN2
    src = inspect.getsource(OWN2.OwnResidency.__init__)
    assert "res_lean needs resolve" not in src and "res_dev needs res_lean" in src
    assert "if res_lean:" in src and "self.enable_res_lean()" in src
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    b2 = base + ["--backends", "b2", "--X", "64"]
    for res in ("fused", "check"):
        a = G.parse_args(b2 + ["--b2-resolve", res, "--fused-kernel", "v3", "--res-lean", "--res-dev"])
        assert a.res_lean and a.res_dev and a.b2_resolve == res and a.fused_kernel == "v3", res
    assert G.parse_args(b2 + ["--res-lean"]).res_lean                 # the torch resolve, as before
    for bad in (["--res-lean"], ["--backends", "sc", "--res-lean"], ["--backends", "b2", "--X", "64", "--b2-resolve",
                                                                     "fused", "--res-dev"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    # the fused path: the lean calls behind _lean_on(), the reference calls in the other branch (unchanged)
    res_src = inspect.getsource(FR.FusedResolver.resolve)
    for s in ("lean = self._lean_on()", "svc._lean_same_rows(rows)", "svc._lean_rows_ok(l, rows, sel)",
              "svc._check_rows(l, rows, sel)", "svc._lean_c.new_tick()", "svc._lean_snap(rows)",
              "svc._lean_plan(rows) if lean else svc._new_plan(rows)", "elif not same and svc._row_sig(rows)"):
        assert s in res_src, s
    lay = inspect.getsource(FR.FusedResolver._layer)
    assert lay.count("self.xp.bind_lean(") == 2 and lay.count("self.xp.bind(") == 2, "the binds: lean and reference"
    assert "XP.draft_bias_fill_flat_lean(" in lay and "XP.draft_bias_fill_flat(self.xp, l," in lay
    assert lay.count("self._lean_on()") == 2
    cmp_src = inspect.getsource(FR.FusedResolver._compare)            # the shadow pool: the REFERENCE bind only
    assert "xr.bind(" in cmp_src and "bind_lean" not in cmp_src and "_lean" not in cmp_src
    assert "_lean" not in inspect.getsource(FR.torch_reference)


# =====================================================================================================================
# FL6 negative controls
# =====================================================================================================================
def test_FL6a_a_skipped_lean_row_check_is_caught(p3):
    _need_triton()
    r = _child("plant_rows", p3.__file__)["plant_rows"]
    ref, lean, skip = r["ref"], r["lean"], r["skip"]
    assert ref["fired"] == [1] and lean["fired"] == [1] and skip["fired"] == [1], "precondition: the plant ran"
    assert ref["err"] is not None and r["text"] in ref["err"], ("precondition: the reference refuses", ref["err"])
    assert (lean["err"], lean["ticks"]) == (ref["err"], ref["ticks"]), (lean, ref)   # the lean checks: the same outcome
    assert skip["skipped"], ("precondition (L34): the skipped check changed a result", skip)
    assert (skip["err"], skip["ticks"]) != (ref["err"], ref["ticks"]), ("the skipped row check was not caught", skip)


def test_FL6b_a_wrong_bind_scan_is_caught_and_the_check_lane_compare_is_real(p3):
    _need_triton()
    r = _child("plant_scan", p3.__file__)["plant_scan"]
    assert r["base_err"] is None, r["base_err"]
    assert r["changed_round"] > 0 and r["changed_check"] > 0, ("precondition (L34): the plant changed a rank", r)
    assert r["diff"] is not None or r["bad_err"] is not None, "the wrong scan was not caught by the round compare"
    assert r["n_check_diffs"] > 0, ("the CHECK mode did not see the wrong lean bind", r)
    assert any(str(d[2]).startswith("xpool.") for d in r["check_diffs"]), r["check_diffs"]


def test_FL6c_the_function_level_comparison_sees_a_wrong_scan(monkeypatch):
    real = RD.excl_scan_rows
    changed = []

    def inclusive(x):
        out = real(x) + x
        if not torch.equal(out, real(x)):
            changed.append(1)
        return out
    real_sf = RD.sorted_free

    def sf_real_scan(*a, **k):                                  # sorted_free keeps the REAL scan (see the FL6b plant:
        RD.excl_scan_rows = real                                # else the two shifts cancel, L34)
        try:
            return real_sf(*a, **k)
        finally:
            RD.excl_scan_rows = inclusive
    monkeypatch.setattr(RD, "excl_scan_rows", inclusive)
    monkeypatch.setattr(RD, "sorted_free", sf_real_scan)
    hit, seen = [], []
    for seed in RF.SEEDS:
        n0 = len(changed)
        diffs, _ = _hook_case(seed, DevConsts)
        if len(changed) > n0:
            hit.append(seed)
        if diffs:
            seen.append(seed)
    assert len(hit) >= 5, ("precondition: draws where the plant changes a rank", hit)
    assert len(seen) >= 5, ("the planted scan was not seen", hit, seen)
    assert set(seen) <= set(hit), ("differences without the plant acting", sorted(set(seen) - set(hit)))


# =====================================================================================================================
# FL7 the cells
# =====================================================================================================================
FL_CELLS = dict(
    evalfast2f3p=dict(res="fused", bands="200:340:150:300:120:260", prof=True, lean=False, time=150),
    evalfast2c3l=dict(res="check", bands=None, prof=False, lean=True, time=120),
    evalfast2f3l=dict(res="fused", bands="335:375:258:292:220:248", prof=False, lean=True, time=120))


def test_FL7a_the_cells_and_the_registration_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=/mnt/beegfs/ojerbi/nosi-r2-f3ld"   # [f3ld]
    assert "        evalfast2f3p|evalfast2c3l|evalfast2f3l) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "|evalfast2c3|evalfast2f3|evalfast2f3_192|evalfast2f3p|evalfast2c3l|evalfast2f3l|" in text   # [f3lia] more follow
    reg = text[text.index("# REGISTERED for evalfast2f3p / evalfast2c3l / evalfast2f3l"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T17:13:30Z, before any run", "nosi-r2-flean", "REFERENCE", "F3P-H2D", "n = 768 over 12 draft-only ticks",
                "F3P-SCAN", "25.9 ms per tick, band [22, 30]", "F3P-SADD", "11.6 ms per tick", "[9.5, 14]", "F3P-TRI",
                "together 80 ms per tick", "[45, 110]", "F3P-RES", "point 125 ms per tick, band [95, 150]", "C3L",
                "0 diffs", "VOID", "F3L-MED", "point 350, band [335, 375]", "F3L-MEAN", "point 272, band [258, 292]",
                "F3L-EFF", "point 230, band [220, 248]", "F3L-FALS", "median < 340", "h_pages\n#            >= 237 ms",
                "REVISED by the lead before any run", "F3L-MEM", "peak_gb <= 84 GB", "STEP_MIN", "REPORT only"):
        assert tag in reg, tag
    assert "STAMP_TBD" not in reg                                       # the lead stamped it before any run
    i = text.index("    evalfast2f3p|evalfast2f3l) MODES=fast")
    blk = text[i:text.index("\nesac", i)]
    assert "FLFLAGS=(--b2-res-profile) ; BANDS=200:340:150:300:120:260 ; STEP_MIN=150" in blk
    assert "FLFLAGS=(--res-lean --res-dev) ; BANDS=335:375:258:292:220:248" in blk
    assert "--b2-resolve fused --fused-kernel v3" in blk and "--wait-index --b2-fewer-reads \"${FLFLAGS[@]}\"" in blk
    assert "--b2-resolve check --fused-kernel v3 --res-lean --res-dev" in blk and "STEPS=8 ; DTICKS=4" in blk
    # the older REGISTERED blocks are kept as they were (their stamps)
    for stamp in ("STAMP 2026-10-23T10:08:04Z", "STAMP 2026-10-23T13:17:36Z", "STAMP 2026-10-23T11:29:23Z"):
        assert stamp in text, stamp


@pytest.mark.parametrize("cell", sorted(FL_CELLS))
def test_FL7b_the_cells_dryrun(tmp_path, cell):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = HF._dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF.HF_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the flean ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    w = FL_CELLS[cell]
    for s in ("--modes fast --backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464",
              "--layout pool --X-pool 1512 --arrival-lag 2", "--repeat-docs --log-slots -1 --skip-shipped",
              "--variants fused/batched", "--mem=240G", "--b2-resolve %s" % w["res"], "--fused-kernel v3",
              "PYTHONPATH=%s/nosi" % HF.HF_ROOT, "--job-name=sg5-t1-418-%s " % cell, "--time=%d" % w["time"]):
        assert s in got or s in line, (cell, s)
    assert "torch/per_call" not in got
    pa = HF._sg5().parse_args(got.split())
    assert (pa.batch, pa.X_pool, pa.b2_resolve, pa.fused_kernel) == (216, 1512, w["res"], "v3"), cell
    assert (bool(pa.res_lean), bool(pa.res_dev), bool(pa.b2_res_profile)) == (w["lean"], w["lean"], w["prof"]), cell
    if cell == "evalfast2c3l":
        assert pa.eval_role == "" and pa.steps == 8 and pa.draft_ticks == 4 and not pa.bare
        for s in ("--accuracy-record", "--eval-bands", "--host-profile", "--wait-index", "--b2-fewer-reads"):
            assert s not in got, (cell, s)
    else:
        assert pa.eval_role == "fast-time" and pa.bare and pa.steps == 48 and pa.mem_cap_gb == 84
        assert "--eval-bands " + w["bands"] in got
        assert pa.host_profile and pa.gc_freeze and pa.lean_time and pa.wait_index and pa.b2_fewer_reads
        assert "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json" in got
