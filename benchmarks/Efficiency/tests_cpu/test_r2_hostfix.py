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
  HF2  --lean-time (item 2): the lean B2 round equals the default at every tick in every decision input and output
       (rows, tokens, logits, pages, state, readers, memory, the kept counters, the release queue and its set) in
       three rounds (forced, pressure, free); the dropped REPORT work ran in the default and not in lean
       (PRECONDITION); NEGATIVE CONTROL: a lean mode that skips the release queue differs at the first verify tick;
       the linear key dedup equals the old quadratic one; the harness's flag, refusals, witness and meta.
  HF3  --wait-index (item 3): InflightByLayer's index equals the layer filter after every mutation of random
       sequences (NEGATIVE CONTROL: a stale pop); _superset_wait waits the same events in the same order on random
       fixtures, plain vs indexed (NEGATIVE CONTROL: a transfer that bypasses the index); the B2 round with the
       index equals the default and the index is exact at every resolve (NEGATIVE CONTROL: the stale pop in a round).

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
                          kept=_kept_counters(svc), queue=[(repr(k), l) for (k, l) in svc.queue],
                          queued=sorted(repr(x) for x in svc.queued)))
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


LEAN_FIELDS = ("rows", "tokens", "logits", "pages", "state", "readers", "memory", "kept", "queue", "queued")


@pytest.mark.parametrize("cfg,forced", [("default", True), ("pressure", True), ("default", False)],
                         ids=["forced", "pressure", "free"])
def test_HF2a_lean_time_changes_no_decision_and_drops_only_reports(p3, ref, cfg, forced):
    toks, _ = ref
    base = _run_hf(p3, toks, cfg=cfg, forced=forced, svc_kw=dict(checked=False))
    lean = _run_hf(p3, toks, cfg=cfg, forced=forced, svc_kw=dict(checked=False, lean=True))
    assert base.svc.lean is False and lean.svc.lean is True                     # the default (no lean argument)
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["lean"].default is False
    assert _diff(base, lean, LEAN_FIELDS) is None
    _same_end(base, lean, report=False)
    assert [t["queue"] for t in base.ticks if t["queue"]], "precondition: the release queue is used"
    mb, ml = base.svc.metrics, lean.svc.metrics
    # PRECONDITION: the default ran the dropped work; lean did not
    assert mb.lifetimes and mb.transferred and mb.arrival_ns and mb.k43["layer_calls"] > 0
    assert not ml.lifetimes and not ml.transferred and not ml.arrival_ns and not ml.open_calls
    assert ml.k43["layer_calls"] == 0 and ml.peak_retained_slots == 0 and "draft_refused" not in ml.counts
    rb, rl = TB._rep(base.svc), TB._rep(lean.svc)
    for k in ("masked", "draft_pairs", "memory", "after_cancel", "arrival", "k5"):
        assert rb[k] == rl[k], (k, rb[k], rl[k])
    assert {k: v for k, v in rb["counts"].items() if k != "draft_refused"} == rl["counts"]


def test_HF2b_negative_control_a_lean_mode_without_the_release_queue_differs(p3, ref):
    """L34: a defective lean (its release queue skipped) differs from the default at the first tick with verify rows
    (the queue field, before that tick's decisions), so HF2a's comparison can see a dropped queue."""
    toks, _ = ref
    base = _run_hf(p3, toks, svc_kw=dict(checked=False))

    def no_queue(eng, svc):
        svc._lean_queue = lambda steps, vrows: None
    bad = _run_hf(p3, toks, svc_kw=dict(checked=False, lean=True), setup=no_queue)
    d = _diff(base, bad, LEAN_FIELDS)
    first_v = next(t for t, x in enumerate(base.ticks) if any(r[1] == "v" for r in x["rows"]))
    assert d is not None and d[0] == first_v and d[1] in ("queue", "queued"), (d, first_v)


def test_HF2c_the_lean_key_order_equals_the_default_dedup():
    g = torch.Generator().manual_seed(3)
    for _ in range(50):
        n = int(torch.randint(1, 30, (1,), generator=g))
        pairs = torch.randint(0, 5, (n, 2), generator=g).tolist()
        rows = [SimpleNamespace(key=("u%d" % x, 1, y)) for x, y in pairs]
        old = []
        for r in rows:
            if r.key not in old:
                old.append(r.key)
        assert list(dict.fromkeys(r.key for r in rows)) == old


def test_HF2d_the_harness_wiring():
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base + ["--backends", "b2", "--X", "64", "--lean-time"])
    assert a.lean_time is True and G.witness_on(a) is False
    assert G.parse_args(base).lean_time is False and G.witness_on(G.parse_args(base)) is True
    for bad in (["--lean-time"], ["--backends", "sc", "--lean-time"],
                ["--backends", "b2", "--X", "64", "--lean-time", "--stack", "full", "--stack-compare",
                 "--arrival-lag", "2"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    me = inspect.getsource(G.make_engine)
    assert 'lean=bool(getattr(a, "lean_time", False))' in me and "eng.xpool.witness = witness_on(a)" in me
    assert len(G.LEAN_DROPPED) == 6 and 'lean_dropped=list(LEAN_DROPPED)' in inspect.getsource(G.main)


RSC = TT._load("r2_residency_sc")                                      # r2tick's S-C (InflightByLayer, CudaDeviceOps)


def _tr(seq, layer, ev=None):
    return SimpleNamespace(seq=seq, layer=layer, ev=ev, payload=None)


def _random_ops(seed, n=400, L=4, cls=None):
    """A random mutation sequence applied to a plain dict and to an InflightByLayer (or cls): set (new keys, a key
    again with the same or another layer), del, pop (present / missing with a default), popitem, setdefault, update,
    and a rare clear. Yields (plain, indexed, op) after every operation."""
    g = torch.Generator().manual_seed(seed)
    plain, ix = {}, (cls or RSC.InflightByLayer)()
    nxt = [0]
    evs = [object() for _ in range(7)]                               # shared events (one per release)

    def new(layer):
        nxt[0] += 1
        return _tr(nxt[0], layer, evs[int(torch.randint(0, len(evs), (1,), generator=g))])
    for _ in range(n):
        r = int(torch.randint(0, 100, (1,), generator=g))
        keys = list(plain)
        k = keys[int(torch.randint(0, len(keys), (1,), generator=g))] if keys else None
        lay = int(torch.randint(0, L, (1,), generator=g))
        if r < 40 or k is None:
            t = new(lay)
            plain[t.seq] = t
            ix[t.seq] = t
            op = "set"
        elif r < 48:
            t = _tr(k, plain[k].layer if r < 45 else lay, plain[k].ev)
            plain[k] = t
            ix[k] = t
            op = "reset"
        elif r < 60:
            del plain[k]
            del ix[k]
            op = "del"
        elif r < 75:
            assert plain.pop(k) is ix.pop(k)
            assert plain.pop(-5, None) is ix.pop(-5, None) is None
            op = "pop"
        elif r < 80:
            a, b = plain.popitem(), ix.popitem()
            assert a[0] == b[0] and a[1] is b[1]
            op = "popitem"
        elif r < 88:
            t = new(lay)
            assert plain.setdefault(t.seq, t) is ix.setdefault(t.seq, t)
            op = "setdefault"
        elif r < 98:
            ts = [new(int(torch.randint(0, L, (1,), generator=g))) for _ in range(3)]
            plain.update({t.seq: t for t in ts})
            ix.update({t.seq: t for t in ts})
            op = "update"
        else:
            plain.clear()
            ix.clear()
            op = "clear"
        yield plain, ix, op


def test_HF3a_the_layer_index_equals_the_layer_filter_after_every_mutation():
    L = 4
    for seed in range(6):
        ops = set()
        for plain, ix, op in _random_ops(seed, L=L):
            ops.add(op)
            assert dict(ix) == plain and list(ix) == list(plain)
            for l in range(L):
                got, want = ix.by_layer(l), [t for t in plain.values() if t.layer == l]
                assert len(got) == len(want) and all(x is y for x, y in zip(got, want)), (seed, op, l)
            assert ix.index_ok()
        assert {"set", "reset", "del", "pop", "popitem", "setdefault", "update"} <= ops, ops
    assert isinstance(RSC.InflightByLayer().copy(), RSC.InflightByLayer)


class StalePopIndex(RSC.InflightByLayer):
    """DEFECT (negative control): pop does not update the layer index (a popped transfer stays waited)."""

    def pop(self, k, *default):
        return dict.pop(self, k, *default)


def test_HF3a2_negative_control_a_stale_index_is_seen():
    bad = 0
    for plain, ix, op in _random_ops(1, cls=StalePopIndex):
        if not ix.index_ok():
            bad += 1
            assert op == "pop" or bad > 1                             # first seen right after a pop
    assert bad > 0


class _Main:
    def __init__(self, log):
        self.log = log

    def wait_event(self, ev):
        self.log.append(id(ev))


def _waits(inflight, landed, monkeypatch, L=4):
    """CudaDeviceOps._superset_wait over every layer on a CPU stand-in: the main stream's wait log (event ids, in
    order) and events_waited."""
    log = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _Main(log))
    ops = object.__new__(RSC.CudaDeviceOps)
    ops.svc = SimpleNamespace(inflight=inflight, _landed=landed, host_prof=None)
    ops.events_waited = ops.submission_waits = 0
    per = []
    for l in range(L):
        n0 = len(log)
        ops._superset_wait(l)
        per.append(log[n0:])
    return per, ops.events_waited


def test_HF3b_the_superset_wait_waits_the_same_events_in_the_same_order(monkeypatch):
    checked = 0
    for seed in range(8):
        g = torch.Generator().manual_seed(100 + seed)
        for i, (plain, ix, op) in enumerate(_random_ops(seed, n=200)):
            if i % 37:
                continue
            landed = {k for k in plain if int(torch.randint(0, 4, (1,), generator=g)) == 0}
            a = _waits(plain, landed, monkeypatch)
            b = _waits(ix, landed, monkeypatch)
            assert a == b, (seed, i)
            checked += a[1] > 0
    assert checked > 10, "precondition: the fixtures wait on events"
    # NEGATIVE CONTROL: a transfer that bypasses the index (dict.__setitem__) is not waited by the indexed scan
    plain, ix = {}, RSC.InflightByLayer()
    for t in (_tr(1, 0, object()), _tr(2, 1, object())):
        plain[t.seq] = t
        ix[t.seq] = t
    hidden = _tr(3, 1, object())
    plain[3] = hidden
    dict.__setitem__(ix, 3, hidden)
    a, b = _waits(plain, set(), monkeypatch), _waits(ix, set(), monkeypatch)
    assert a[0][0] == b[0][0] and a[0][1] != b[0][1] and id(hidden.ev) in a[0][1] and id(hidden.ev) not in b[0][1]


def test_HF3c_the_b2_round_with_the_index_equals_the_default_and_keeps_it_exact(p3, ref):
    toks, _ = ref
    base = _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False))
    seen = []

    def check_each_resolve(eng, svc):
        assert isinstance(svc.inflight, RSC.InflightByLayer)
        r0 = svc.resolve

        def resolve(l, rows, sel):
            seen.append((len(svc.inflight), svc.inflight.index_ok()))
            return r0(l, rows, sel)
        svc.resolve = resolve
    ixr = _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False, wait_index=True), setup=check_each_resolve)
    assert _diff(base, ixr) is None
    _same_end(base, ixr)
    assert seen and all(ok for _, ok in seen) and max(n for n, _ in seen) > 0, "precondition: transfers in flight"
    assert not isinstance(base.svc.inflight, RSC.InflightByLayer)                   # the default: a plain dict
    # NEGATIVE CONTROL: the stale-pop index in the same round is seen by the same per-resolve check
    seen_bad = []

    def stale(eng, svc):
        svc.inflight = StalePopIndex(svc.inflight)
        r0 = svc.resolve

        def resolve(l, rows, sel):
            seen_bad.append(svc.inflight.index_ok())
            return r0(l, rows, sel)
        svc.resolve = resolve
    _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False), setup=stale)
    assert not all(seen_bad), "the stale index is not seen (no arrival popped a transfer?)"


def test_HF3d_the_harness_wiring():
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).wait_index is False
    assert G.parse_args(base + ["--backends", "b2", "--X", "64", "--wait-index"]).wait_index is True
    with pytest.raises(SystemExit):
        G.parse_args(base + ["--wait-index"])                         # sc, sb, v1: refused
    assert 'wait_index=bool(getattr(a, "wait_index", False))' in inspect.getsource(G.make_engine)
    src = inspect.getsource(RSC.CudaDeviceOps._superset_wait)
    assert 'getattr(self.svc.inflight, "by_layer", None)' in src and "tr.layer != l" in src


def test_HF1c_the_harness_wiring():
    G = _sg5()
    src = inspect.getsource(G.round_guard)
    for need in ("gc.collect()", "gc.freeze()", "gc.disable()", "gc.unfreeze()", "gc.enable()",
                 'rec.setdefault("gc_freeze", {})[tag] = frz'):
        assert need in src, need
    assert src.index("gc.freeze()") < src.index("yield gcl") < src.index("gc.unfreeze()")
    assert src.index("gc.callbacks.remove(gcl)") < src.index('frz["collected_after"] = gc.collect()')
