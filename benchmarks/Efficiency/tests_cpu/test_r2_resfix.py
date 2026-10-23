"""[resfix] CPU tests of --res-lean (OwnResidency(res_lean=True) -> ResidencySC._resolve_lean; analysis/evalfast/
DIAG_torch_res.md section 4, fixes 4-7). The reference torch resolve (ResidencySC.resolve, DeviceMirror.layer,
device_verify, device_draft, rank_in_group, sparse_layer_record, XPool.bind / audit, the bias fills) is unchanged; its
lean copies must give the same values.

  RL1  FUNCTION LEVEL on random inputs (60 draws; the odd ones at a PRESSURE point, C 1-2 and X 0-3, so that K5
       refuses, Policy E evicts and first misses overflow): DeviceMirror.layer_lean == layer (every field), then
       device_verify_lean == device_verify (every decision field as int64, page, num_pages, load_ids, counters, (b, h),
       and the mirror after the verify rows), device_draft_lean == device_draft, rank_with(rank_prep) == rank_in_group,
       sparse_layer_record_lean == sparse_layer_record (pins False; record caps full and small), XPool.bind_lean ==
       bind (map, free lists, flag, owner, handoffs, peak, binds; a verify bind, a draft bind, a release, a bind at the
       other layer), draft_bias_fill_flat_lean / draft_bias_fill_lean == the reference writes. The lean bool fields are
       bool (the reference casts them to int64). PRECONDITIONS over the draws: verify grants, LRU and in-flight Policy E
       evictions, overflows, duplicates, ready and in-flight reads, draft grants and refusals. NEGATIVE CONTROL: a
       LeanConsts whose -1 fill is 0 differs (L34: on every draw whose reference rslot holds a -1).
  RL2  ROUND LEVEL: the B2 round with --res-lean equals the reference at every tick (rows, tokens, logits, pages,
       state, counters, readers, memory, the kept counters, the release queue) and at every LAYER field by field (the
       step's decisions, records and caps, the plan's page table, num_pages, last_len, seq_idx, keys, blocks and page
       keys; with the X pool its map, free lists, flag, owner, handoffs, peak and binds), in the per-stream and the pool
       layout, default and pressure, forced and free, transport seeds 1-3, unchecked and checked; the report at the end.
       PRECONDITIONS: a pressure round evicts; a pool round hands a lane between two requests.
  RL3  THE HIDDEN SYNCS and the per-row Python: inside resolve the reference scatters a Python scalar through an
       advanced index (the CPU stand-in of a pageable host-to-device copy + cudaStreamSynchronize on the GPU; counted,
       > 0: the PRECONDITION that the counter sees them); --res-lean does it 0 times. _check_rows and _row_sig run once
       per tick (the reference: once per layer), _new_plan never.
  RL4  THE GUARDS: a plant between two layers of one tick (a row key cancelled, a (key, l) release queued, a batch
       slot changed in uid_b, a row field changed, a row tail turned into a tuple) gives the SAME outcome in both paths
       (the same exception text at the same tick; or no exception and equal rounds). The flag's refusals and wiring.
  RL5  NEGATIVE CONTROL of RL2: a lean draft decision that takes the LAST free slot of a stream instead of the q-th
       is seen by RL2's comparison (L34 PRECONDITION: the plant changed a take at some layer).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_resfix.py
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


HF = _shared("test_r2_hostfix_shared_rf", "test_r2_hostfix.py")      # the B2 round helpers (_diff, _same_end, ...)
XPT = _shared("test_r2_xpool_shared_rf", "test_r2_xpool.py")         # make_pool_engine, _handoffs
TT, TB, TRM, TK, OWN2 = HF.TT, HF.TB, HF.TRM, HF.TK, HF.OWN2
RSB, RSC = TT._load("r2_residency_sb"), TT._load("r2_residency_sc")   # the r2tick package (OWN2's own modules)
RSP, XPM = TT._load("r2_residency_sparse"), TT._load("r2_xpool")
p3, ref = HF.p3, HF.ref                                               # the fixtures, by name
CFG = TB.CFG
N_TAIL = RSC.N_TAIL
MIRROR = ("slot_of", "inflight", "infl_slot", "infl_gen", "free", "usage", "releasing", "lru_rank", "prot", "lru_stamp",
          "slot_block", "infl_unprot", "infl_rank", "fly", "fly_stamp", "slot_gen")


def _same(a, b, path="x"):
    """None if a and b are equal (tensors: dtype, shape and values; dicts, lists, tuples: recursively; else ==);
    otherwise the path of the first difference."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.dtype != b.dtype or a.shape != b.shape \
                or not torch.equal(a, b):
            return path
        return None
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return path + " (keys)"
        for k in a:
            d = _same(a[k], b[k], "%s.%s" % (path, k))
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return path + " (len / type)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _same(x, y, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    return None if a == b else path


def _dec(dec, kind):
    """A decision's comparable values (copies): every field as int64 (the lean bool fields are bool by design)."""
    d = {f: v.to(torch.int64).clone() for f, v in dec.t.items()}
    d.update(page=dec.page.clone(), num_pages=dec.num_pages.clone(), counters=dec.counters.clone(), H=dec.H, k=dec.k)
    if kind == "v":
        d.update(load_ids=dec.load_ids.clone(), bh=(dec.bh[0].clone(), dec.bh[1].clone()), R=dec.Rv,
                 skip=dec.skip_infl_wait)
    else:
        d.update(R=dec.Rd)
    return d


def _mirror(m):
    return {f: getattr(m, f).clone() for f in MIRROR}


def _xps(x):
    return dict(pos_of=x.pos_of.clone(), occ=x.occ.clone(), err=x.err.clone(), owner=x.owner.clone(),
                handoffs=x.handoffs.clone(), peak=x.peak.clone(), binds=x.binds)


# =====================================================================================================================
# RL1 the lean device functions on random inputs
# =====================================================================================================================
def _draw(seed):
    """A random mirror table, rows and selection (no consistency is needed: both paths run the same ops on any input,
    and the CPU's index writes are deterministic)."""
    g = torch.Generator().manual_seed(1000 + seed)

    def ri(lo, hi, shape=()):
        return torch.randint(lo, hi, tuple(shape) if isinstance(shape, (tuple, list)) else (shape,), generator=g)
    pressure = seed % 2 == 1
    L, B, H = 2, int(ri(2, 5)), int(ri(1, 3))
    NBLK = int(ri(6, 14))
    C, X = (int(ri(1, 3)), int(ri(0, 4))) if pressure else (int(ri(3, 7)), int(ri(2, 9)))
    NPOS = C + X
    R_ver = int(ri(0, X + 1))
    shp = (L, B, H, NPOS)
    dm = RSC.DeviceMirror(L, B, H, NBLK, NPOS, "cpu")
    dm.code.copy_(ri(0, 6, shp))
    if seed % 3 == 0:                                   # [resfix] mostly FREE slots: usage < C, so draft rows are granted
        dm.code.copy_(torch.where(ri(0, 10, shp) < 7, torch.full(shp, RSC.C_FREE, dtype=dm.code.dtype), dm.code))
    dm.block.copy_(ri(-1, NBLK, shp))
    dm.gen.copy_(ri(0, 4, shp))
    dm.cur.copy_(ri(0, 2, shp))
    dm.prot.copy_(ri(0, 2, shp))
    dm.stamp.copy_((torch.randperm(L * B * H * NPOS, generator=g) + 1).view(shp))
    Rv, Rd, k = int(ri(1, 6)), int(ri(1, 6)), int(ri(2, 6))
    R = Rv + Rd
    rb, hb, nt = ri(0, B, R), ri(1, NBLK + 1, R), ri(0, N_TAIL + 1, R)
    j = torch.arange(N_TAIL).view(1, N_TAIL)
    tails = torch.where(j < nt.view(R, 1), NPOS + j, torch.full((R, N_TAIL), -1, dtype=torch.int64))
    sel = ri(-2, NBLK + 3, (H, R, k))
    ixv, ixd = torch.arange(Rv), torch.arange(Rv, R)
    pv = (rb[ixv], hb[ixv], tails[ixv], nt[ixv], ixv)
    pd = (rb[ixd], hb[ixd], tails[ixd], nt[ixd], ixd)
    return SimpleNamespace(g=g, ri=ri, L=L, B=B, H=H, NBLK=NBLK, C=C, X=X, NPOS=NPOS, R_ver=R_ver, dm=dm, Rv=Rv, Rd=Rd,
                           k=k, R=R, sel=sel, pv=pv, pd=pd, pressure=pressure)


def _case(seed, cst_cls=None):
    """RL1's comparison on draw `seed`: (the differences found, the reference's coverage stats)."""
    d = _draw(seed)
    cst = (cst_cls or RSB.LeanConsts)("cpu")
    cst.new_tick()
    l = seed % 2
    inflight = seed % 3 != 0
    H, k = d.H, d.k
    diffs = []

    def cmp(name, a, b):
        s = _same(a, b, name)
        if s:
            diffs.append(s)
    xr = XPM.XPool(d.L, d.B, H, d.NPOS, mode="pool", C=d.C, X_pool=d.X + seed % 3)
    xl = xr.clone()
    stub = SimpleNamespace(_lean_c=cst, H=H, nblk=d.NBLK, n_hist=d.NPOS, xpool=xl, device=torch.device("cpu"))
    rcv = RSC.ResidencySC._lean_rows_c(stub, d.pv, k)
    rcd = RSC.ResidencySC._lean_rows_c(stub, d.pd, k)
    # the layer's mirror
    mr, ml = d.dm.layer(l), d.dm.layer_lean(l, cst)
    cmp("mirror", _mirror(mr), _mirror(ml))
    # the verify rows (Policy E spares the draft rows' blocks too), then the mirror they leave
    sel_v, sel_d = d.sel[:, d.pv[4]], d.sel[:, d.pd[4]]
    sv_d = sel_d.permute(1, 0, 2)
    v0_d = (sv_d >= 0) & (sv_d < d.pd[1].view(-1, 1, 1)) & (sv_d < d.NBLK)
    vr = RSC.device_verify(mr, *d.pv[:4], sel_v, d.C, d.X, others=[(d.pd[0], d.pd[1], sel_d)], inflight=inflight)
    vl = RSC.device_verify_lean(ml, *d.pv[:4], sel_v, d.C, d.X, cst, rcv, others=[(sv_d, v0_d, rcd["sidn3"])],
                                inflight=inflight)
    cmp("vdec", _dec(vr, "v"), _dec(vl, "v"))
    cmp("mirror_after_verify", _mirror(mr), _mirror(ml))
    # the draft rows on the mirror after the verify rows
    dr = RSB.device_draft(mr, *d.pd[:4], sel_d, d.C, d.X, d.R_ver)
    dl = RSB.device_draft_lean(ml, *d.pd[:4], sel_d, d.C, d.X, d.R_ver, cst, rcd, s=sv_d, valid0=v0_d)
    cmp("ddec", _dec(dr, "d"), _dec(dl, "d"))
    for f in ("valid", "ready", "granted"):                           # the documented dtype difference, and only it
        if vl.t[f].dtype != torch.bool or vr.t[f].dtype != torch.int64 or dl.t[f].dtype != torch.bool:
            diffs.append("dtype " + f)
    # rank_in_group's two halves
    for n in (1, 2, 7, 33):
        key, mask = d.ri(0, 5, n), d.ri(0, 2, n) != 0
        cmp("rank%d" % n, RSB.rank_in_group(key, mask), RSB.rank_with(RSB.rank_prep(key, cst), mask))
    # the sparse record (pins False: B2), full and small caps
    kid_all = d.ri(1, 9, d.R)
    Rd = d.Rd
    bbd = d.pd[0].view(Rd, 1, 1).expand(Rd, H, k).reshape(-1)               # as _record_inputs builds them
    hhd = torch.arange(H).view(1, H, 1).expand(Rd, H, k).reshape(-1)
    tk = dict(v=d.pv, d=d.pd, extra=torch.cat([kid_all, torch.zeros(d.L * d.B, dtype=torch.int64)]))
    stub2 = SimpleNamespace(_lean_c=cst, H=H, nblk=d.NBLK)
    rcv2 = RSP.ResidencySparse._lean_rec_c(stub2, tk, "v", d.R, rcv)
    rcd2 = RSP.ResidencySparse._lean_rec_c(stub2, tk, "d", d.R, rcd)
    for cap_v, cap_d in ((d.Rv * H * k, Rd * H * k), (3, 2)):
        pins_l = torch.zeros((d.B, H, d.NBLK + 1, 2), dtype=torch.int64)
        recr = RSP.sparse_layer_record(pins_l, torch.zeros(d.B, dtype=torch.int64), d.dm.code[l], vr, kid_all[d.pv[4]],
                                       dr, kid_all[d.pd[4]], bbd, hhd, d.NBLK, d.NPOS, cap_v, cap_d, ixv=d.pv[4],
                                       ixd=d.pd[4], R=d.R, pins=False)
        recl = RSP.sparse_layer_record_lean(d.dm.code[l], vl, dl, d.B, H, d.NBLK, d.NPOS, cap_v, cap_d, d.R, cst,
                                            rcv=rcv2, rcd=rcd2)
        cmp("rec cap %d" % cap_v, recr, recl)
    # the X pool: a verify bind, a draft bind, a release, a bind at the other layer
    xr.bind(l, vr.bh[0], vr.bh[1], vr.t["take_slot"], (vr.t["granted"] | vr.t["evict"]) != 0)
    xl.bind_lean(l, vl.bh[0], vl.bh[1], vl.t["take_slot"], vl.t["granted"] | vl.t["evict"], cst, base=rcv["xbase"])
    cmp("xp verify bind", _xps(xr), _xps(xl))
    xr.bind(l, d.pd[0].view(Rd, 1, 1).expand(Rd, H, k), torch.arange(H).view(1, H, 1).expand(Rd, H, k),
            dr.t["take_slot"], dr.t["granted"] != 0)
    xl.bind_lean(l, rcd["bb_f"], rcd["hh_f"], dl.t["take_slot"], dl.t["granted"], cst, base=rcd["xbase"])
    cmp("xp draft bind", _xps(xr), _xps(xl))
    code = d.ri(0, 6, (d.L, d.B, H, d.NPOS))
    xr.release_free(code)
    xl.release_free(code)
    xr.bind(1 - l, vr.bh[0], vr.bh[1], vr.t["take_slot"], (vr.t["granted"] | vr.t["evict"]) != 0)
    xl.bind_lean(1 - l, vl.bh[0], vl.bh[1], vl.t["take_slot"], vl.t["granted"] | vl.t["evict"], cst, base=rcv["xbase"])
    cmp("xp bind after release", _xps(xr), _xps(xl))
    # the bias fills (flat store: the lanes bound above; per-stream store)
    bs = 4
    cis = torch.randn((d.B, (d.NBLK + 1) * bs, H), generator=d.g)
    store = torch.randn((xr.M * bs, H), generator=d.g)
    sr, sl = store.clone(), store.clone()
    XPM.draft_bias_fill_flat(xr, l, sr, cis, dr, bbd, hhd, bs)
    XPM.draft_bias_fill_flat_lean(xl, l, sl, cis, dl, rcd["bb_f"], rcd["hh_f"], bs, cst)
    cmp("bias flat", (sr, xr.err), (sl, xl.err))
    store2 = torch.randn((d.B, (d.NPOS + N_TAIL) * bs, H), generator=d.g)
    s2r, s2l = store2.clone(), store2.clone()
    RSP.draft_bias_fill(s2r, cis, dr, bbd, hhd, bs)
    RSP.draft_bias_fill_lean(s2l, cis, dl, rcd["bb_f"], rcd["hh_f"], bs, cst)
    cmp("bias per stream", s2r, s2l)
    stats = dict(v_grant=int(vr.t["granted"].sum()),
                 v_evict=int(((vr.t["evict"] != 0) & (vr.t["evict_infl"] == 0)).sum()),
                 v_evict_infl=int(vr.t["evict_infl"].sum()), v_over=int(vr.counters.sum()), v_dup=int(vr.t["dup"].sum()),
                 v_ready=int(vr.t["ready"].sum()), v_infl=int(vr.t["infl"].sum()), d_grant=int(dr.t["granted"].sum()),
                 d_refused=int(dr.t["refused"].sum()), d_ready=int(dr.t["ready"].sum()),
                 xp_full=int(xr.err) & XPM.ERR_FULL, rslot_pad=bool((vr.t["rslot"] == -1).any()))
    return diffs, stats


SEEDS = list(range(60))


@pytest.mark.parametrize("seed", SEEDS)
def test_RL1a_the_lean_device_functions_equal_the_reference(seed):
    diffs, _ = _case(seed)
    assert not diffs, (seed, diffs)


def test_RL1b_the_draws_cover_every_decision_path():
    """PRECONDITION of RL1a: the draws reach every decision kind (summed over the draws)."""
    tot = {}
    for seed in SEEDS:
        _, st = _case(seed)
        for k_, v in st.items():
            tot[k_] = tot.get(k_, 0) + int(v)
    for k_ in ("v_grant", "v_evict", "v_evict_infl", "v_over", "v_dup", "v_ready", "v_infl", "d_grant", "d_refused",
               "d_ready", "rslot_pad"):
        assert tot[k_] > 0, ("precondition: no draw reaches", k_, tot)
    print("REPORT RL1 coverage over %d draws: %s" % (len(SEEDS), tot))


class _PlantConsts(RSB.LeanConsts):
    """PLANT: the cached -1 fills come out as 0 (a constant cache that hands out a wrong value)."""

    def full_like(self, like, value, dtype=None):
        if not isinstance(value, bool) and value == -1:
            value = 0
        return super().full_like(like, value, dtype=dtype)


def test_RL1c_negative_control_a_wrong_constant_is_seen():
    """L34: the plant is checked on the draws where it changes a value (the reference rslot holds a -1, which the
    lean path takes from the -1 fill) and RL1a's comparison sees it on every one of them."""
    hit = 0
    for seed in SEEDS:
        _, st = _case(seed)
        if not st["rslot_pad"]:
            continue
        hit += 1
        diffs, _ = _case(seed, cst_cls=_PlantConsts)
        assert diffs, ("the planted constant was not seen", seed)
    assert hit >= 10, "precondition: draws where the plant changes a value"


# =====================================================================================================================
# RL2 the B2 round, reference vs --res-lean, every tick and every layer
# =====================================================================================================================
def _layer_rec(svc, plan):
    """One resolve's comparable values (copies): the tick step it appended and the plan; the X pool's state."""
    st = svc._tick[-1]
    out = dict(l=st["l"], vidx=list(st["vidx"]), caps=st.get("caps"),
               rows=[(r.b, r.kind, r.key, r.pos, r.hist_bound, tuple(r.tail)) for r in st["vrows"] + st["drows"]],
               vdec=_dec(st["vdec"], "v") if st["vdec"] is not None else None,
               ddec=_dec(st["ddec"], "d") if st["ddec"] is not None else None,
               rec={n: v.clone() for n, v in st["rec"].items()} if st.get("rec") is not None else None,
               plan=dict(page_table=plan.page_table.clone(), num_pages=plan.num_pages.clone(),
                         last_len=plan.last_len.clone(), seq_idx=plan.seq_idx.clone(), keys=list(plan.keys),
                         blocks=[[list(x) for x in row] for row in plan.blocks],
                         page_key=plan.page_key.clone() if plan.page_key is not None else None))
    if svc.xpool is not None:
        out["xp"] = _xps(svc.xpool)
    return out


def _round(eng, toks, forced=True, setup=None):
    """One round with HF._run_hf's per-tick record plus a per-LAYER record (_layer_rec after every resolve) and, with an
    X pool, the map after each tick."""
    world = eng.res_world
    svc = world.svc
    evs, tags = [], {}

    def new_event():                                                  # named by creation order (both rounds alike)
        e = TT.Ev(True)
        tags[id(e)] = "e%d" % len(evs)
        evs.append(e)
        return e
    eng.backend.new_event = new_event
    pages, layers = [], []
    resolve0 = svc.resolve

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)]
                          for i in range(R)]))
        layers.append(_layer_rec(svc, p))
        return p
    svc.resolve = resolve
    ticks, maps = [], []

    def on_tick(rows, tokens, out):
        ticks.append(dict(rows=[(r.b, r.kind, r.pos, r.epoch, r.i, getattr(r, "m", None)) for r in rows],
                          tokens=[t.slot if isinstance(t, TK.Ref) else int(t) for t in tokens],
                          logits=out.logits.clone(), pages=list(pages), state=TRM._state(world),
                          counters=TRM._counters(svc), readers=TRM._reader_tags(svc, tags), memory=svc.memory(),
                          kept=HF._kept_counters(svc), queue=[(repr(k), l) for (k, l) in svc.queue],
                          queued=sorted(repr(x) for x in svc.queued), layers=list(layers)))
        pages.clear()
        layers.clear()
        if svc.xpool is not None:
            maps.append(svc.xpool.pos_of.clone())
    eng.timer = TK.TickTimer(cuda=False)
    if setup is not None:
        setup(eng, svc)
    S_list = eng.S_list

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    reject_at, _ = TK.forced_schedule(3) if forced else (None, [])
    rnd, err = None, None
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                           teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                            # noqa: BLE001 - a difference at its tick
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    return SimpleNamespace(eng=eng, svc=svc, ticks=ticks, rnd=rnd, err=err, evs=evs, maps=maps)


def _engine(p3, layout, cfg, seed, lean, checked=False, svc_kw=None):
    kw = dict(checked=checked, res_lean=lean)
    kw.update(svc_kw or {})
    ops = TT.make_ops(p3)
    if layout == "pool":
        return XPT.make_pool_engine(ops, kind="b2", seed=seed, svc_kw=kw, **CFG[cfg])
    return TT.make_res_engine(ops, kind="b2", seed=seed, svc_kw=kw, **CFG[cfg])


def _rounds_diff(a, b):
    """The first difference of two rounds: HF._diff over every per-tick field, then every layer field by field."""
    d = HF._diff(a, b, HF.ALL_FIELDS + ("kept", "queue", "queued"))
    if d is not None:
        return d
    for t, (x, y) in enumerate(zip(a.ticks, b.ticks)):
        s = _same(x["layers"], y["layers"], "layers")
        if s:
            return (t, s)
    return None


RL2_CASES = [("stream", "default", True, 1, False), ("stream", "default", False, 1, False),
             ("stream", "pressure", True, 1, False), ("stream", "pressure", True, 2, False),
             ("stream", "pressure", True, 3, True), ("stream", "pressure", False, 2, False),
             ("pool", "default", True, 1, False), ("pool", "pressure", True, 1, False),
             ("pool", "pressure", True, 2, True), ("pool", "pressure", False, 3, False)]


@pytest.mark.parametrize("layout,cfg,forced,seed,checked", RL2_CASES,
                         ids=["-".join(str(x) for x in c) for c in RL2_CASES])
def test_RL2_the_lean_round_equals_the_reference_at_every_layer(p3, ref, layout, cfg, forced, seed, checked):
    toks, _ = ref
    base = _round(_engine(p3, layout, cfg, seed, False, checked), toks, forced)
    lean = _round(_engine(p3, layout, cfg, seed, True, checked), toks, forced)
    assert base.svc.res_lean is False and lean.svc.res_lean is True
    assert base.err is None and len(base.ticks) > 10, ("precondition: the reference round ran", base.err)
    assert sum(len(t["layers"]) for t in base.ticks) >= len(base.ticks), "precondition: the layers were recorded"
    d = _rounds_diff(base, lean)
    assert d is None, ("--res-lean differs from the reference", d, lean.err)
    HF._same_end(base, lean)
    js = dict(base.svc.journal_counts)
    if cfg == "pressure":                                             # the preconditions, checked before the claim
        ev = sum(v for k_, v in js.items() if k_.startswith("release:") and "evict" in k_)
        assert ev > 0 or base.svc.metrics.counts.get("verifier_evictions", 0) > 0, ("precondition: no eviction", js)
    if layout == "pool" and cfg == "pressure":                        # [resfix] a default round need not hand a lane on
        assert base.svc.xpool.binds > 0 and XPT._handoffs(base.maps, base.svc.xpool.n_hist) > 0, \
            "precondition: no lane changed hands between requests"
    print("REPORT %s: ticks %d, journal %s" % ("-".join(str(x) for x in (layout, cfg, forced, seed, checked)),
                                               len(base.ticks), sorted(js.items())))


# =====================================================================================================================
# RL3 the hidden syncs and the per-row Python
# =====================================================================================================================
class _ScalarScatter:
    """Counts Tensor.__setitem__ calls whose value is a Python scalar and whose index holds a tensor, while `on` (the
    advanced-index scatter of a host scalar: on CUDA one pageable 1-byte copy and one cudaStreamSynchronize each)."""

    def __init__(self, monkeypatch):
        self.n, self.on = 0, False
        orig = torch.Tensor.__setitem__

        def setitem(t, idx, val, _o=orig, _s=self):
            if _s.on and isinstance(val, (bool, int, float)):
                ix = idx if isinstance(idx, tuple) else (idx,)
                if any(torch.is_tensor(i) for i in ix):
                    _s.n += 1
            return _o(t, idx, val)
        monkeypatch.setattr(torch.Tensor, "__setitem__", setitem)


def _count_setup(box, counter):
    """setup(eng, svc): counts the resolves, the scalar scatters inside them and the per-row Python calls."""
    def setup(eng, svc):
        r0, c0, s0, n0 = svc.resolve, svc._check_rows, svc._row_sig, svc._new_plan

        def resolve(l, rows, sel):
            box["resolve"] += 1
            counter.on = True
            try:
                return r0(l, rows, sel)
            finally:
                counter.on = False

        def check_rows(*a, **k):
            box["check_rows"] += 1
            return c0(*a, **k)

        def row_sig(*a, **k):
            box["row_sig"] += 1
            return s0(*a, **k)

        def new_plan(*a, **k):
            box["new_plan"] += 1
            return n0(*a, **k)
        svc.resolve, svc._check_rows, svc._row_sig, svc._new_plan = resolve, check_rows, row_sig, new_plan
    return setup


def test_RL3_no_scalar_scatter_and_the_row_python_once_per_tick(p3, ref, monkeypatch):
    toks, _ = ref
    counter = _ScalarScatter(monkeypatch)
    got = {}
    for lean in (False, True):
        box = dict(resolve=0, check_rows=0, row_sig=0, new_plan=0)
        counter.n = 0
        r = _round(_engine(p3, "pool", "pressure", 1, lean), toks, True, setup=_count_setup(box, counter))
        assert r.err is None, r.err
        box["scalar_scatter"] = counter.n
        got[lean] = box
    L = TT.NL_LAYERS
    base, lean = got[False], got[True]
    assert base["resolve"] == lean["resolve"] > 0 and base["resolve"] % L == 0
    ticks = base["resolve"] // L
    assert base["scalar_scatter"] > 0, "precondition: the counter sees the reference's scalar scatters"
    assert lean["scalar_scatter"] == 0, lean
    assert (base["check_rows"], base["row_sig"], base["new_plan"]) == (base["resolve"],) * 3, base
    assert (lean["check_rows"], lean["row_sig"], lean["new_plan"]) == (ticks, ticks, 0), lean
    print("REPORT RL3 per round (%d ticks x %d layers): reference %s, lean %s" % (ticks, L, base, lean))


def test_RL3b_the_lean_source_has_no_host_transfer_and_no_scalar_scatter():
    """Static: the lean per-layer copies hold no host read, no counted read, no host-to-device copy (F12d's / XP9's
    tokens, the whole source) and, in their code, no `] = True` / `] = False` scatter."""
    toks = TRM.FORBIDDEN + ("_d2h(",) + TRM.H2D
    fns = (RSC.ResidencySC._resolve_lean, RSC.DeviceMirror.layer_lean, RSC.device_verify_lean, RSC._selected_mask_lean,
           RSB.device_draft_lean, RSB.rank_prep, RSB.rank_with, RSP.ResidencySparse._layer_record_lean,
           RSP.sparse_layer_record_lean, RSP._compact_lean, RSP._pack_bits_lean, RSP._first_in_group_lean,
           RSP.draft_bias_fill_lean, XPM.XPool.bind_lean, XPM.XPool.audit_lean, XPM.draft_bias_fill_flat_lean)
    for fn in fns:
        src = inspect.getsource(fn)
        assert [t for t in toks if t in src] == [], fn.__qualname__
        body = src.split('"""', 2)[-1]                                # the code after the docstring
        assert "] = True" not in body and "] = False" not in body, fn.__qualname__
    # NEGATIVE CONTROL of the scan: the reference copies hold the scatters it looks for
    for fn in (XPM.XPool.bind, RSP.sparse_layer_record, XPM.draft_bias_fill_flat, RSP.draft_bias_fill):
        assert "] = True" in inspect.getsource(fn).split('"""', 2)[-1], fn.__qualname__


# =====================================================================================================================
# RL4 the guards: a plant between two layers of one tick, both paths alike
# =====================================================================================================================
def _plant_at(tick, layer, plant, fired):
    """setup(eng, svc): before the resolve number tick * L + layer (layer `layer` of the tick-th resolving tick),
    plant(svc, rows, l); fired records l."""
    def setup(eng, svc):
        r0 = svc.resolve
        n = dict(i=0)
        L = svc.L

        def resolve(l, rows, sel):
            if n["i"] == tick * L + layer:
                plant(svc, rows, l)
                fired.append(l)
            n["i"] += 1
            return r0(l, rows, sel)
        svc.resolve = resolve
    return setup


def _p_cancel(svc, rows, l):
    svc.cancelled.add(rows[0].key)


def _p_queued(svc, rows, l):
    svc.queued.add((rows[-1].key, l))


def _p_uid(svc, rows, l):
    svc.uid_b[rows[0].key[0]] = rows[0].b + 100


def _p_pos(svc, rows, l):
    rows[0].pos += 1


def _p_tail_tuple(svc, rows, l):
    rows[0].tail = tuple(rows[0].tail)


PLANTS = dict(cancel=(_p_cancel, "a cancelled cohort key"), queued=(_p_queued, "after its release was queued"),
              uid=(_p_uid, "names batch slot"), pos=(_p_pos, "every layer of a tick resolves the same rows"),
              tail_tuple=(_p_tail_tuple, None))


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_RL4a_a_plant_between_layers_gives_the_same_outcome(p3, ref, name):
    toks, _ = ref
    plant, text = PLANTS[name]
    out = {}
    for lean in (False, True):
        fired = []
        out[lean] = _round(_engine(p3, "stream", "default", 1, lean), toks, True, setup=_plant_at(5, 1, plant, fired))
        assert fired == [1], ("precondition: the plant ran at layer 1 of a tick", fired)
    base, lean = out[False], out[True]
    assert base.err == lean.err, (base.err, lean.err)
    if text is None:                                                  # no exception: equal rounds
        assert base.err is None and _rounds_diff(base, lean) is None
    else:
        assert base.err is not None and text in base.err, ("precondition: the reference refuses the plant", base.err)
        assert len(base.ticks) == len(lean.ticks) >= 5


def test_RL4b_the_flag_refusals_and_its_wiring(p3):
    ops = TT.make_ops(p3)
    with pytest.raises(ValueError):                                   # the torch resolve only
        TT.make_res_engine(ops, kind="b2", svc_kw=dict(res_lean=True, resolve="fused"), **CFG["default"])
    for kw in (dict(sc_mut={"no_wait": True}), dict(sb_mut={"no_dedup": True}), dict(sparse_mut={"swap_takes": True})):
        with pytest.raises(ValueError):                               # no negative control in the lean copies
            TT.make_res_engine(ops, kind="b2", svc_kw=dict(res_lean=True, **kw), **CFG["default"])
    with pytest.raises(ValueError):
        XPT.make_pool_engine(ops, kind="b2", xp_mut={"alias": True}, svc_kw=dict(res_lean=True), **CFG["default"])
    assert RSC.ResidencySC.res_lean is False
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["res_lean"].default is False
    assert "if self.res_lean:" in inspect.getsource(RSC.ResidencySC.resolve)
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).res_lean is False
    assert G.parse_args(base + ["--backends", "b2", "--X", "64", "--res-lean"]).res_lean is True
    for bad in (["--res-lean"], ["--backends", "b2", "--X", "64", "--b2-resolve", "fused", "--res-lean"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    assert 'res_lean=bool(getattr(a, "res_lean", False))' in inspect.getsource(G.make_engine)


# =====================================================================================================================
# RL5 negative control of RL2: a wrong lean take is seen
# =====================================================================================================================
def test_RL5_negative_control_a_wrong_lean_take_is_seen(p3, ref, monkeypatch):
    """L34: the plant takes a stream's LAST free slot instead of its q-th for every draft grant; it counts the layers
    where that changes a take (the precondition), and RL2's comparison sees the round differ."""
    toks, _ = ref
    real = RSC.device_draft_lean
    changed = []

    def planted(m, rb, hb, tl, nt, sel, C, X, R_ver, cst, rc, s=None, valid0=None):
        dec = real(m, rb, hb, tl, nt, sel, C, X, R_ver, cst, rc, s=s, valid0=valid0)
        g = dec.t["granted"]
        Sn, NPOS = m.free.shape[0] * m.free.shape[1], m.free.shape[-1]
        ar = torch.arange(NPOS).view(1, NPOS).expand(Sn, NPOS)
        desc = torch.sort(torch.where(m.free.reshape(Sn, NPOS), ar, torch.full_like(ar, -1)), dim=-1,
                          descending=True).values
        q = RSB.rank_in_group(rc["sid"], g)
        ts = torch.where(g, desc[rc["sid"], q.clamp(max=NPOS - 1)], torch.full_like(q, -1))
        if not torch.equal(ts, dec.t["take_slot"]):
            changed.append(1)
        dec.t["take_slot"] = ts
        return dec
    base = _round(_engine(p3, "stream", "pressure", 1, False), toks, True)
    monkeypatch.setattr(RSC, "device_draft_lean", planted)
    bad = _round(_engine(p3, "stream", "pressure", 1, True), toks, True)
    assert changed, "precondition: the plant changed a draft take"
    assert _rounds_diff(base, bad) is not None, "the planted take was not seen"
