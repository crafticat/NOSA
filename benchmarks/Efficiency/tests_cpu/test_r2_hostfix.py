"""[hostfix] CPU tests of the OPT-IN host fixes of the R2 FAST tick (analysis/evalfast/DIAG_tick_host.md section 4):
every flag's default keeps the old behaviour, and every flag changes no decision, page table, logit or MUST-gate input.
The rounds are SG5's CPU stand-in (test_r2_tick.py make_res_engine: the tiny model on the Residency layout, K3X's
adversarial transport, S-C's CPU device stand-in) with B2 (OwnResidency) as the service, as test_r2_tick_b2.py.

  HF0  --host-profile (item 0): the B2 round with the host sub-timers equals the round without them at every tick (rows,
       tokens, logits, pages, state, counters, readers, memory) and in the report; without the flag no "h_" category
       exists; with it every HOST_PROF_CATS category is recorded except the GPU wait's (h_superset_wait: CudaDeviceOps
       only). NEGATIVE CONTROL: the comparison sees a planted one-logit change. GcLog records a collection inside its
       block only; the gilfree feeder's Python time per release (GilfreeTransport.prof) and its default (no record).
  HF1  --gc-freeze (item 1): round_guard collects, freezes and disables before the phase and restores after it (an
       exception too; a collector OFF before stays OFF); no automatic collection in the phase (PRECONDITION: the
       same churn collects with the collector ON); the B2 round under the guard equals the round without it.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_hostfix.py
"""
from __future__ import annotations

import contextlib
import gc
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


TB = _shared("test_r2_tick_b2_shared_hf", "test_r2_tick_b2.py")      # the B2 round helpers (_first_diff, _rep, CFG)
TT, TRM = TB.TT, TB.TRM                                               # the tiny model (r2tick), the snapshots (r2res)
TK, OWN2 = TT.TK, TT.OWN2
p3, ref = TT.p3, TT.ref                                               # the fixtures, by name
EFF, NOSI = TT.EFF, TT.NOSI
ALL_FIELDS = ("rows", "tokens", "logits", "pages", "state", "counters", "readers", "memory")


def _sg5():
    return _shared("r2_tick_sg5_hostfix_cpu", str(EFF / "r2_tick_sg5.py"))


def _run_hf(p3, toks, cfg="default", forced=True, svc_kw=None, setup=None, around=None):
    """One B2 round on the CPU stand-in with a per-tick record (test_r2_tick_b2._run's, plus a CPU TickTimer, a
    setup(eng, svc) hook before the round and around(eng): a context manager around run_round: the flag under test)."""
    eng = TT.make_res_engine(TT.make_ops(p3), kind="b2", svc_kw=svc_kw, **TB.CFG[cfg])
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
                          counters=TRM._counters(svc), readers=TRM._reader_tags(svc, tags), memory=svc.memory(),
                          kept=_kept_counters(svc)))
        pages.clear()
    eng.timer = TK.TickTimer(cuda=False)
    if setup is not None:
        setup(eng, svc)
    S_list = eng.S_list

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    reject_at, _ = TK.forced_schedule(3) if forced else (None, [])
    rnd, err = None, None
    try:
        with (around(eng) if around is not None else contextlib.nullcontext()):
            rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                               teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                            # noqa: BLE001 - a difference at its tick
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    return SimpleNamespace(eng=eng, svc=svc, ticks=ticks, rnd=rnd, err=err, evs=evs, world=world)


def _kept_counters(svc):
    """The service counters that --lean-time keeps (no 4.3 / 4.6 / 4.8, no draft_refused): the masked and pair
    counts, the plans' refusals, the other metric counts, and the journal per kind."""
    m = svc.metrics
    return (m.masked, m.draft_pairs, m.refused, sorted((k, v) for k, v in m.counts.items() if k != "draft_refused"),
            sorted(dict(getattr(svc, "journal_counts", {})).items()))


def _diff(a, b, fields=ALL_FIELDS):
    """The first (tick, field) where two rounds differ (test_r2_tick_b2._first_diff over `fields`); None: equal."""
    n = min(len(a.ticks), len(b.ticks))
    for t in range(n):
        for f in fields:
            x, y = a.ticks[t][f], b.ticks[t][f]
            if not (torch.equal(x, y) if f == "logits" else x == y):
                return (t, f)
    if len(a.ticks) != len(b.ticks) or (a.err is None) != (b.err is None):
        return (n, "raise: %s / %s" % (a.err, b.err))
    return None


def _same_end(a, b, report=True):
    assert a.err is None and b.err is None, (a.err, b.err)
    assert [r.out for r in a.rnd.reqs] == [r.out for r in b.rnd.reqs]
    assert TRM._journal_kinds(a.svc) == TRM._journal_kinds(b.svc)
    assert a.svc.b2_applied == b.svc.b2_applied == len(a.ticks) > 10
    assert dict(a.eng.stats, ms_rows=0) == dict(b.eng.stats, ms_rows=0)
    if report:
        ra, rb = TB._rep(a.svc), TB._rep(b.svc)
        assert ra == rb, [k for k in ra if ra[k] != rb[k]]


def _host_profile_on(eng, svc):
    eng.host_prof = True
    svc.host_prof = lambda cat, ns, t=eng.timer: t.host(cat, ns * 1e-9)


# =====================================================================================================================
# HF0 --host-profile
# =====================================================================================================================
@pytest.mark.parametrize("forced", [True, False], ids=["forced", "free"])
def test_HF0a_host_profile_changes_no_value_and_is_off_by_default(p3, ref, forced):
    toks, _ = ref
    G = _sg5()
    base = _run_hf(p3, toks, forced=forced, svc_kw=dict(checked=False))
    prof = _run_hf(p3, toks, forced=forced, svc_kw=dict(checked=False), setup=_host_profile_on)
    assert _diff(base, prof) is None
    _same_end(base, prof)
    assert TK.TickEngine.host_prof is False and OWN2.OwnResidency.host_prof is None          # the defaults
    cats0 = {c for (_, c) in base.eng.timer.hosts}
    cats1 = {c for (_, c) in prof.eng.timer.hosts}
    assert not [c for c in cats0 if c.startswith("h_")], cats0                               # off: no sub-timer
    assert cats0 == {"backend", "rows"} | ({"read"} if prof.rnd.syncs else set()), cats0     # as before
    rec = G.host_profile_record(prof.eng.timer.report())
    assert rec["missing"] == ["h_superset_wait"], rec["missing"]          # the GPU wait only (CudaDeviceOps)
    assert set(G.HOST_PROF_CATS) - {"h_superset_wait"} <= cats1
    assert all(len(v) == rec["ticks"] for v in rec["per_tick_ms"].values())
    # NEGATIVE CONTROL: the per-tick comparison sees a planted change of one logit
    bad = SimpleNamespace(ticks=[dict(t) for t in prof.ticks], err=prof.err)
    bad.ticks[3]["logits"] = bad.ticks[3]["logits"].clone()
    bad.ticks[3]["logits"][0, 0] += 1.0
    assert _diff(base, bad) == (3, "logits")


def test_HF0b_gclog_records_collections_inside_its_block_only():
    G = _sg5()
    timer = SimpleNamespace(tick=4)

    class Cyc:
        pass
    with G.GcLog(timer) as log:
        assert log in gc.callbacks
        a = Cyc()
        a.self = a                                                    # a cycle the collection frees
        del a
        gc.collect()
    assert log not in gc.callbacks
    n = len(log.events)
    assert n >= 1 and all(e[0] == 4 for e in log.events), log.events
    assert any(e[1] == 2 and e[4] >= 1 for e in log.events), log.events                     # the full collection
    gc.collect()                                                      # NEGATIVE CONTROL: outside the block
    assert len(log.events) == n
    rep = log.report(6)
    assert len(rep["per_tick_ms"]) == 6 and rep["per_tick_ms"][4] > 0 and rep["n"] == n
    assert sum(rep["per_tick_ms"]) == pytest.approx(rep["total_ms"], abs=0.01)
    assert "2" in rep["by_generation"] and rep["longest"][0]["thread"]


def _stub_transport(prof=None):
    """GilfreeTransport without the packer extension: the Python release path (_release) with a stub native pack."""
    GT = TT._load("r2_transport_gilfree")
    tp = object.__new__(GT.GilfreeTransport)
    tp.skip_hist_wait, tp.cuda, tp.xpool, tp.board, tp.worker = False, False, None, None, "python"
    tp.nblk, tp.n_slots, tp.B, tp.D, tp.threads = 10, 8, 2, 4, 1
    tp._stats = {k: 0 for k in GT.STATS[:7]}
    tp._stats.update(releases=0, threads_min=None, threads_max=None, published=0)
    tp.slots_k = [torch.zeros(2, 8 * 64, 1, 4)]
    tp.slots_v = [torch.zeros(2, 8 * 64, 1, 4)]
    calls = []
    tp.npipe = SimpleNamespace(release=lambda *c: calls.append(c) or [2, 1, 1, 5, 0, 1, 6, 1, 1])
    tp.host_wait = lambda ev: None
    if prof is not None:
        tp.prof = prof
    trs = [SimpleNamespace(seq=i, h=0, b=i % 2, slot=i, block=i + 1, phys=None, hist_ev=None, reader_ev=None, ev=None)
           for i in range(3)]
    return GT, tp, trs, calls


def test_HF0c_the_feeder_python_time_per_release():
    GT, tp, trs, calls = _stub_transport()
    assert GT.GilfreeTransport.prof is None                           # the default: no clock, no record
    tp._release(0, trs)
    assert all(tr.ev is not None for tr in trs) and len(calls) == 1 and tp.stats["releases"] == 1
    _, tp2, trs2, calls2 = _stub_transport(prof=[])
    tp2._release(0, trs2)
    assert [str(x) for x in calls2[0][:5]] == [str(x) for x in calls[0][:5]]                 # the same pack call
    assert tp2.stats == tp.stats                                      # the same stats
    (layer, n, py_ns, nat_ns, end), = tp2.prof
    assert (layer, n) == (0, 3) and py_ns >= 0 and nat_ns >= 0 and end > 0
    rep = _sg5().feeder_report(tp2.prof)
    assert rep["releases"] == 1 and rep["transfers"] == 3 and rep["python_us_per_transfer"] is not None
    assert _sg5().feeder_report(None) is None


def test_HF0d_the_harness_wiring():
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).host_profile is False
    assert G.parse_args(base + ["--host-profile"]).host_profile is True
    me = inspect.getsource(G.make_engine)
    assert 'if getattr(a, "host_profile", False):' in me and "svc.host_prof = lambda cat, ns" in me
    assert me.index("eng.timer = TickTimer(cuda=True)") < me.index("eng.host_prof = True")
    op = inspect.getsource(G.one_point)
    assert op.count("round_guard(a, eng, info, ") == 3                # the draft-only phase, the round, the bare round
    for k in ('info["draft_only_host_profile"]', 'info["host_profile"]', 'info["bare_host_profile"]'):
        assert k in op, k
    assert op.index('info["wall_s"] = round(time.time() - t0, 1)') < op.index('info["host_profile"]')
    # round_guard without any [hostfix] flag: nothing installed, nothing changed
    a = G.parse_args(base)
    n0 = len(gc.callbacks)
    rec = {}
    with G.round_guard(a, SimpleNamespace(timer=SimpleNamespace(tick=0)), rec, "round") as g:
        assert g is None and len(gc.callbacks) == n0 and gc.isenabled() and gc.get_freeze_count() == 0
    assert len(gc.callbacks) == n0 and rec == {}


# =====================================================================================================================
# HF1 --gc-freeze
# =====================================================================================================================
def _churn(n=30000):
    """Container allocations kept alive until the end (the collector counts allocations minus deallocations of
    tracked objects): enough to pass the generation-0 threshold many times."""
    keep = [{"i": i} for i in range(n)]
    del keep


def _collections():
    return [s["collections"] for s in gc.get_stats()]


def test_HF1a_gc_freeze_guard_disables_collection_in_the_phase_and_restores_it():
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).gc_freeze is False
    a = G.parse_args(base + ["--gc-freeze"])
    eng = SimpleNamespace(timer=SimpleNamespace(tick=0))
    assert gc.isenabled()
    # precondition (L34: the control must be able to fire): the same churn with the collector ON collects
    c0 = _collections()
    _churn()
    assert _collections()[0] > c0[0], "precondition: the churn triggers generation-0 collections"
    rec = {}
    with G.round_guard(a, eng, rec, "round"):
        assert not gc.isenabled() and gc.get_freeze_count() > 0
        c1 = _collections()
        _churn()
        assert _collections() == c1                                   # no automatic collection in the phase
    assert gc.isenabled() and gc.get_freeze_count() == 0
    r = rec["gc_freeze"]["round"]
    assert r["enabled_before"] is True and r["frozen"] > 0
    assert r["collections_in_phase"] and not any(r["collections_in_phase"]), r["collections_in_phase"]
    assert r["collected_after"] >= 0 and "cuda_freed_gb_by_collect" in r
    assert len(r["count_before"]) == len(gc.get_count())
    with pytest.raises(RuntimeError):                                 # an exception in the phase restores it too
        with G.round_guard(a, eng, rec, "bare"):
            raise RuntimeError("x")
    assert gc.isenabled() and gc.get_freeze_count() == 0 and "bare" in rec["gc_freeze"]
    gc.disable()                                                      # a collector OFF before stays OFF after
    try:
        with G.round_guard(a, eng, rec, "draft_only"):
            pass
        assert not gc.isenabled() and rec["gc_freeze"]["draft_only"]["enabled_before"] is False
    finally:
        gc.enable()


def test_HF1b_the_round_under_gc_freeze_changes_no_value(p3, ref):
    toks, _ = ref
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base + ["--gc-freeze", "--host-profile"])
    rec = {}
    b0 = _run_hf(p3, toks, svc_kw=dict(checked=False))
    b1 = _run_hf(p3, toks, svc_kw=dict(checked=False), setup=_host_profile_on,
                 around=lambda eng: G.round_guard(a, eng, rec, "round"))
    assert _diff(b0, b1) is None
    _same_end(b0, b1)
    r = rec["gc_freeze"]["round"]
    assert r["collections_in_phase"] and not any(r["collections_in_phase"]), r["collections_in_phase"]
    assert gc.isenabled() and gc.get_freeze_count() == 0
    assert G.host_profile_record(b1.eng.timer.report())["missing"] == ["h_superset_wait"]


def test_HF1c_the_harness_wiring():
    G = _sg5()
    src = inspect.getsource(G.round_guard)
    for need in ("gc.collect()", "gc.freeze()", "gc.disable()", "gc.unfreeze()", "gc.enable()",
                 'rec.setdefault("gc_freeze", {})[tag] = frz'):
        assert need in src, need
    assert src.index("gc.freeze()") < src.index("yield gcl") < src.index("gc.unfreeze()")
    assert src.index("gc.callbacks.remove(gcl)") < src.index('frz["collected_after"] = gc.collect()')
