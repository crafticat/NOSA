"""CPU tests of SG5's b2 backend ([b2], 2026-10-21): OwnResidency (nosi/nosi/r2_own2.py, B's array-backed ownership
table) as the tick's storage service in benchmarks/Efficiency/r2_tick_sg5.py. The round is SG5's CPU stand-in of
test_r2_tick.py: the tiny model on the Residency layout (make_res_engine), K3X's adversarial transport (seeded), S-C's
CPU device stand-in; b2 and sc run the same round, and every tick is compared.

  TB1  the teacher-forced round (forced_schedule: rejections at every offset, boundary crossings, write-backs) on b2,
       checked and unchecked, equals the round on sc, at every tick: the rows and their input tokens, the logits
       (torch.equal), the page tables of every layer (the decisions), the service state through B's snapshot
       (test_r2_residency._state: every stream's owner, LRU order, PO, VRES, pending, protections with their kinds and
       slots; the transfers in flight; the transport's queue), the counters, every slot's first / last reader (named
       by event order), memory(); at the end the committed tokens, the journal per kind and the report (4.3 without
       the clocks, 4.6, 4.8, k5, memory, masked, refusals, counts, arrivals). Two capacities: T15's (C 12, X 24) and
       a pressure point (C 8, X 40: the LRU goes over C, so slots are released and taken again: the 4.8 read harvest
       at a take runs; precondition asserted).
  TB2  free-running greedy: the same per-tick equality; the committed tokens of b2 = sc's = the serial greedy ones.
  TB3  NEGATIVE CONTROLS in the round: rec_mut drop_rowsel (one new protection bit of the device's record dropped)
       differs from sc at the tick of the mutation; no_read_harvest leaves every tick equal and changes report k48.
  TB4  report()["k5"] (the demand counter) on B2: C's F26 schedule (peak 7 at tick 1, layer 0, by layer [7, 6]; k43
       new 34; one device read per tick) and F25's refused p4a case (48 at tick 1, layer 0). Negative control: a B2
       that notes the demand against the next layer reports another layer.
  TB5  the harness: --backends b2 and --b2-checked parse (default unchecked); make_engine's b2 branch (static: B2 on
       the store's device, CudaDeviceOps behind _LazyOps as sc, checked from the flag); the draft-only tick on b2
       with the CPU TickTimer; sc_counters on b2; the GPU path builds no whole projection (memory() vectorized,
       stream() lazy; the vendored memory rule on the projection gives the same counts). Static: B2's GPU-path
       methods name no self.pools / self.stream( (negative control: the scan finds them in C's sparse applier).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_tick_b2.py
"""
from __future__ import annotations

import ast
import collections
import importlib.util
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


TT = _shared("test_r2_tick_shared_b2", "test_r2_tick.py")             # the tiny model, make_res_engine (package r2tick)
TRM = _shared("test_r2_residency_shared_b2", "test_r2_residency.py")  # B's snapshot, C's F25 / F26 (package r2res)
TK, OWN2 = TT.TK, TT.OWN2
OWNK = sys.modules[TT.PKG + ".kstep.ownership"]                       # the vendored states (r2tick)
OWN2R = TRM._load("r2_own2")                                          # OwnResidency in the r2res package (TB4)
p3, ref = TT.p3, TT.ref                                               # the fixtures, by name
EFF, NOSI = TT.EFF, TT.NOSI

CFG = dict(default=dict(C=12, X=24, R_ver=4),                          # T15's capacities
           pressure=dict(C=8, X=40, R_ver=4))                          # the LRU over C: slots taken again
FIELDS = ("rows", "tokens", "logits", "pages", "state", "counters", "readers", "memory")


def _run(kind, p3, toks, cfg="default", forced=True, svc_kw=None):
    """One round on the CPU stand-in with a per-tick record (after the tick's end_tick, before its decisions)."""
    eng = TT.make_res_engine(TT.make_ops(p3), kind=kind, svc_kw=svc_kw, **CFG[cfg])
    world = eng.res_world
    svc = world.svc
    evs, tags = [], {}

    def new_event():                                                  # named by creation order (both rounds alike)
        e = TT.Ev(True)
        tags[id(e)] = "e%d" % len(evs)
        evs.append(e)
        return e
    eng.backend.new_event = new_event
    pages = []
    resolve0 = svc.resolve

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)]
                          for i in range(R)]))
        return p
    svc.resolve = resolve
    ticks = []

    def on_tick(rows, tokens, out):
        ticks.append(dict(rows=[(r.b, r.kind, r.pos, r.epoch, r.i, getattr(r, "m", None)) for r in rows],
                          tokens=[t.slot if isinstance(t, TK.Ref) else int(t) for t in tokens],
                          logits=out.logits.clone(), pages=list(pages), state=TRM._state(world),
                          counters=TRM._counters(svc), readers=TRM._reader_tags(svc, tags), memory=svc.memory()))
        pages.clear()
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
    return SimpleNamespace(eng=eng, svc=svc, ticks=ticks, rnd=rnd, err=err, evs=evs)


_SC = {}


def _sc(p3, toks, cfg, forced):
    key = (cfg, forced)
    if key not in _SC:
        _SC[key] = _run("sc", p3, toks, cfg=cfg, forced=forced)
        assert _SC[key].err is None and len(_SC[key].ticks) > 10, ("precondition: the sc round ran", _SC[key].err)
    return _SC[key]


def _first_diff(a, b):
    """The first (tick, field) where two rounds differ; a round that raised (or stopped) earlier differs at the tick
    it did not record. None: equal at every tick."""
    n = min(len(a.ticks), len(b.ticks))
    for t in range(n):
        for f in FIELDS:
            x, y = a.ticks[t][f], b.ticks[t][f]
            if not (torch.equal(x, y) if f == "logits" else x == y):
                return (t, f)
    if len(a.ticks) != len(b.ticks) or (a.err is None) != (b.err is None):
        return (n, "raise: %s / %s" % (a.err, b.err))
    return None


def _rep(svc):
    """The report's comparable parts (no clocks, no definitions)."""
    r = svc.report()
    return dict(k43={k: v for k, v in r["k43"].items() if k not in ("wait_ns", "layer_wait_ns")}, k46=r["k46"],
                k48=r["k48"], masked=r["masked"], draft_pairs=r["draft_pairs"], draft_refused=r["draft_refused"],
                counts=r["counts"], memory=r["memory"], after_cancel=r["arrivals_after_cancel"],
                arrival={k: r["arrival"][k] for k in ("lag", "forced", "held")},
                k5={k: r["k5"][k] for k in ("extra_need_peak", "at_tick", "at_layer", "by_layer")})


def _equal_rounds(sc, b2):
    d = _first_diff(sc, b2)
    assert d is None, ("b2 differs from sc", d, b2.err)
    assert [r.out for r in b2.rnd.reqs] == [r.out for r in sc.rnd.reqs]
    assert TRM._journal_kinds(b2.svc) == TRM._journal_kinds(sc.svc)            # the transitions per kind
    rs, rb = _rep(sc.svc), _rep(b2.svc)
    for k in rs:
        assert rb[k] == rs[k], (k, rs[k], rb[k])
    assert b2.svc.b2_applied == len(b2.ticks) and not b2.svc.uid_b


# =====================================================================================================================
# TB1 / TB2 b2 = sc on the round
# =====================================================================================================================
@pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("cfg", sorted(CFG))
def test_TB1_b2_equals_sc_at_every_tick_of_the_teacher_forced_round(p3, ref, cfg, checked):
    toks, _ = ref
    sc = _sc(p3, toks, cfg, True)
    b2 = _run("b2", p3, toks, cfg=cfg, svc_kw=dict(checked=checked))
    _equal_rounds(sc, b2)
    m = sc.svc.metrics
    assert m.k43["new"] > 0 and m.k43["ready"] > 0 and sc.eng.stats["writebacks"] >= 3     # the round's transitions
    assert sc.eng.stats["rollbacks"] > 0 and m.draft_pairs > 0
    use = _rep(sc.svc)["k48"]["prefetch_use"]
    assert use["draft_only"] + use["verify_only"] + use["both"] > 0                       # reads reach 4.8
    if cfg == "pressure":                                             # a slot of a lifetime was taken again
        n = collections.Counter(k[:4] for k in m.lifetimes)
        assert max(n.values()) > 1, "precondition: no slot was taken twice (the harvest at a take did not run)"


def test_TB2_b2_equals_sc_on_the_free_running_round(p3, ref):
    toks, _ = ref
    sc = _sc(p3, toks, "default", False)
    b2 = _run("b2", p3, toks, forced=False, svc_kw=dict(checked=False))
    _equal_rounds(sc, b2)
    greedy = [[int(toks[s, b]) for s in range(TT.STEPS)] for b in range(b2.eng.B)]
    assert [r.out for r in b2.rnd.reqs] == greedy                    # the serial greedy sequences


# =====================================================================================================================
# TB3 negative controls in the round
# =====================================================================================================================
def test_TB3_a_planted_record_defect_differs_at_its_tick_and_the_harvest_control(p3, ref):
    toks, _ = ref
    sc = _sc(p3, toks, "default", True)
    bad = _run("b2", p3, toks, svc_kw=dict(b2_mut={"rec_mut": "drop_rowsel"}))
    done = bad.svc.rec_mut_done
    assert done is not None and done[1] == "drop_rowsel", "precondition: the round has a new pin bit"
    d = _first_diff(sc, bad)
    assert d is not None and d[0] == done[0], (d, done)              # caught at the tick of the mutation
    sp = _sc(p3, toks, "pressure", True)
    use = _rep(sp.svc)["k48"]["prefetch_use"]
    assert use["draft_only"] + use["verify_only"] + use["both"] > 0, "precondition: reads reach 4.8"
    nh = _run("b2", p3, toks, cfg="pressure", svc_kw=dict(b2_mut={"no_read_harvest": True}))
    assert _first_diff(sp, nh) is None                               # the state does not depend on the 4.8 counts
    assert _rep(nh.svc)["k48"] != _rep(sp.svc)["k48"]                # the report does


# =====================================================================================================================
# TB4 the K5 demand counter (report()["k5"]) on B2
# =====================================================================================================================
class WrongLayerB2(OWN2R.OwnResidency):
    """DEFECT (negative control): the shared-extra demand of a layer is noted against the next layer."""

    def _note_extra_need(self, l, need):
        super()._note_extra_need((l + 1) % self.L, need)


def test_TB4_the_k5_report_holds_the_demand_peak_on_b2():
    w = TRM._f26(OWN2R.OwnResidency)
    k5 = w.svc.report()["k5"]
    assert (k5["extra_need_peak"], k5["at_tick"], k5["at_layer"], k5["by_layer"]) == (7, 1, 0, [7, 6]), k5
    assert (k5["C"], k5["X"]) == (4, 10) and w.svc.metrics.k43["new"] == 6 + 10 + 2 + 11 + 5
    assert w.svc.d2h_reads == len(TRM.F26_TICKS)                     # the demand rides in reads B2 makes anyway
    w2, e = TRM._k5_case(OWN2R.OwnResidency, TRM.P4A_MISSES, dict(nblk=TRM.NB_K5))
    assert e is not None and isinstance(e, TRM.ER.CapacityRefusal)
    k2 = w2.svc.report()["k5"]
    assert (k2["extra_need_peak"], k2["at_tick"], k2["at_layer"]) == (48, 1, 0), k2
    kn = TRM._f26(WrongLayerB2).svc.report()["k5"]
    assert (kn["at_layer"], kn["by_layer"]) != (0, [7, 6]), kn


# =====================================================================================================================
# TB5 the harness's b2 backend
# =====================================================================================================================
GPU_PATH = ("apply_tick", "_apply_b2", "_host_side", "poll", "_poll_arrivals", "_poll_drops", "_poll_releases",
            "sync_mirror", "begin_request", "history_written", "end_request", "cancel", "memory", "_key_slot",
            "_in_flight", "_tick_rows_of", "watch_flag",
            "_looks", "_lean_queue")                     # [hostfix] _host_side's looks moved out; the lean queue


def _methods(path, cls):
    src = path.read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {f.name: ast.get_source_segment(src, f) for f in node.body if isinstance(f, ast.FunctionDef)}


def test_TB5_the_harness_b2_backend(p3):
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert "b2" in G.BACKENDS and G.parse_args(base).backends == ["sc", "sb", "v1"]      # the default is unchanged
    a = G.parse_args(base + ["--backends", "b2", "--X", "64"])
    assert a.backends == ["b2"] and a.b2_checked is False                               # the timed path: unchecked
    assert G.parse_args(base + ["--backends", "b2", "--b2-checked"]).b2_checked is True
    src = (EFF / "r2_tick_sg5.py").read_text()
    tree = ast.parse(src)
    me = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name == "make_engine"))
    b2 = me[me.index('elif kind == "b2":'):me.index("        else:\n            lazy = _LazyOps()")]   # [stack2] in the
    #                                                                     _CloseOnError block (one indent deeper)
    for need in ("from nosi.r2_own2 import OwnResidency", "lazy = _LazyOps()", "device=eng.store_k[0].device",
                 "devops=lazy", 'checked=bool(getattr(a, "b2_checked", False))',
                 "lazy.impl = CudaDeviceOps(svc, host_k, host_v, eng.store_k, eng.store_v, eng.store_bias, cis)"):
        assert need in b2, need
    assert me.count("lazy.impl = CudaDeviceOps(") == 2                                 # sc and b2 alike
    op = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name == "one_point"))
    assert 'info["b2_checked"]' in op
    # the draft-only tick on b2 (CPU TickTimer), sc_counters, and no whole projection on this path
    eng = TT.make_res_engine(TT.make_ops(p3), kind="b2", svc_kw=dict(checked=False))
    eng.timer = TK.TickTimer(cuda=False)
    G.run_draft_only(eng, TT.FIRST, 6, 3)
    rep = eng.timer.report()
    assert rep["ticks"] == 6 and all(len(rep["device_ms"][c]) == 6 for c in ("res", "tick", "twin"))
    svc = eng.res_world.svc
    c = G.sc_counters(svc)
    assert c["service"] == "OwnResidency" and c["row_uploads"] == 6 and c["mirror_uploads"] >= 6
    assert c["mirror_slots_uploaded"] == 0                            # B2's mirror: a device copy, no host delta
    assert c["journal_counts"].get("take", 0) > 0 and c["devops"]["kind"] == "CpuDeviceSim"
    assert svc._view is None and not svc._sviews, "the draft-only path built a projection"
    st = svc.stream(0, 0, 0)                                          # lazy: one stream, no whole projection
    assert svc._view is None and len(svc._sviews) == 1
    pools = svc.pools                                                 # NEGATIVE CONTROL: the whole projection shows
    assert svc._view is not None
    full = pools[0].stream(0, 0)
    assert any(s.state != OWNK.FREE for s in full.slots), "precondition: the drafts took slots of stream (0, 0, 0)"
    assert [(s.state, s.block, s.gen, s.cls, s.owner) for s in st.slots] == \
        [(s.state, s.block, s.gen, s.cls, s.owner) for s in full.slots]
    assert (list(st.lru.items()), st.po, st.vres, st.pending) == (list(full.lru.items()), full.po, full.vres,
                                                                  full.pending)
    assert svc.memory() == TT.RES.Residency.memory(svc)               # the vendored rule on the projection
    # static: B2's GPU-path methods name no projection (the CPU device stand-in and the refusal reports may)
    meths = _methods(NOSI / "r2_own2.py", "OwnResidency")
    for name in GPU_PATH:
        body = meths[name]
        for word in ("self.pools", "self.stream(", "_materialize("):
            assert word not in body, (name, word)
    sp = _methods(NOSI / "r2_residency_sparse.py", "ResidencySparse")["_apply_verify_sparse"]
    assert "self.pools" in sp                                         # NEGATIVE CONTROL: the scan sees a use
