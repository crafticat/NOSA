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
  HF5  --b2-fewer-reads (item 5): the B2 round with fewer reads equals the default at every tick and in the report
       (forced, pressure, free; unchecked and checked; PRECONDITION: a tick pins in two or more rounds), and with
       --lean-time equals lean; the consumer's host reads per tick are never more and fewer in total (counted on the
       CPU stand-in); the host pin rounds and recency round count equal rank_in_group's on random rows; NEGATIVE
       CONTROL: merged pin rounds differ from the default.
  HF6  the step cells evalfast2 / evalfast2t / evalfast2c of r2_tick_sg5_step.sh: ROOT NOSI_EVAL = nosi-r2-hostfix,
       the REGISTERED block (every tag, the bands = the cells' --eval-bands), and on a clean checkout of this ROOT the
       DRYRUN of each cell (and of evalfast, unchanged) parsed by the harness into the registered flags; a wrong cell
       word is refused (usage, exit 2).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_hostfix.py
"""
from __future__ import annotations

import contextlib
import gc
import importlib.util
import inspect
import os
import subprocess
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
                 "--arrival-lag", "2", "--sel-check", "--X-pool", "64"]):   # (only the lean rule refuses it)
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    ok = G.parse_args(base + ["--backends", "b2", "--X", "64", "--stack", "full", "--stack-compare", "--arrival-lag",
                              "2", "--sel-check", "--X-pool", "64"])      # the same cell without --lean-time parses
    assert ok.stack_compare and not ok.lean_time
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


SB = TT._load("r2_residency_sb")                                       # rank_in_group (r2tick)


class _Reads:
    """Counts the host reads of tensors in a block (tolist, int, bool, item, nonzero): the CPU stand-in of the
    synchronizations the same calls make on CUDA."""

    NAMES = ("tolist", "__int__", "__bool__", "item", "nonzero")

    def __init__(self, monkeypatch):
        self.n = 0
        for name in self.NAMES:
            orig = getattr(torch.Tensor, name)

            def wrap(*a, _o=orig, **k):
                self.n += 1
                return _o(*a, **k)
            monkeypatch.setattr(torch.Tensor, name, wrap)


def _counting_end_tick(svc, counter):
    """Counts the reads of svc.end_tick only (the B2 consumer)."""
    e0 = svc.end_tick
    box = dict(reads=[])

    def end_tick():
        before = counter.n
        counter.n = 0
        try:
            return e0()
        finally:
            box["reads"].append(counter.n)
            counter.n = before
    svc.end_tick = end_tick
    return box


@pytest.mark.parametrize("cfg,forced,checked", [("default", True, False), ("pressure", True, False),
                                                ("default", False, False), ("pressure", True, True)],
                         ids=["forced", "pressure", "free", "pressure-checked"])
def test_HF5a_fewer_reads_changes_no_value(p3, ref, cfg, forced, checked, monkeypatch):
    toks, _ = ref
    pins = []
    pins_or0 = OWN2.O.pins_or

    def pins_or(*a, **k):
        if k.get("rounds") is not None:
            pins.append(len(k["rounds"]))
        return pins_or0(*a, **k)
    monkeypatch.setattr(OWN2.O, "pins_or", pins_or)
    base = _run_hf(p3, toks, cfg=cfg, forced=forced, svc_kw=dict(checked=checked))
    few = _run_hf(p3, toks, cfg=cfg, forced=forced, svc_kw=dict(checked=checked, fewer_reads=True))
    assert base.svc.fewer_reads is False and few.svc.fewer_reads is True
    assert _diff(base, few) is None
    _same_end(base, few)
    assert pins and max(pins) >= 2, "precondition: a tick pins in two or more rounds (a cohort's verify rows)"
    c = base.svc.metrics.counts
    if cfg == "pressure":
        assert base.svc.journal_counts.get("take", 0) > 0 and base.svc.journal_counts.get("touch", 0) > 0
    print("REPORT verifier_evictions %s, draft_refused %s, pin rounds max %d" % (c.get("verifier_evictions"),
                                                                                 c.get("draft_refused"), max(pins)))


def test_HF5b_fewer_reads_with_lean_equals_lean(p3, ref):
    toks, _ = ref
    lean = _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False, lean=True))
    both = _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False, lean=True, fewer_reads=True))
    assert _diff(lean, both) is None
    _same_end(lean, both)


def test_HF5c_the_consumer_reads_less(p3, ref, monkeypatch):
    """The B2 consumer's host reads per tick (tolist / int / bool / item / nonzero, counted in end_tick): fewer with
    the flag on every tick with verify rows; never more."""
    toks, _ = ref
    counter = _Reads(monkeypatch)
    boxes = {}
    for name, kw in (("base", dict(checked=False)), ("few", dict(checked=False, fewer_reads=True))):
        boxes[name] = []
        _run_hf(p3, toks, cfg="pressure", svc_kw=kw,
                setup=lambda eng, svc, n=name: boxes.__setitem__(n, _counting_end_tick(svc, counter)))
    rb, rf = boxes["base"]["reads"], boxes["few"]["reads"]
    assert len(rb) == len(rf) > 10
    assert all(f <= b for b, f in zip(rb, rf)), list(zip(rb, rf))
    assert sum(rf) < sum(rb), (sum(rb), sum(rf))
    print("REPORT B2 consumer host reads per tick: default %s, fewer %s (sum %d -> %d)"
          % (rb[:6], rf[:6], sum(rb), sum(rf)))


def test_HF5d_host_rounds_equal_the_device_rank_on_random_rows():
    g = torch.Generator().manual_seed(9)
    for _ in range(60):
        R = int(torch.randint(1, 40, (1,), generator=g))
        bs = torch.randint(0, 4, (R,), generator=g).tolist()
        ks = torch.randint(0, 3, (R,), generator=g).tolist()
        kinds = ["v" if x else "d" for x in torch.randint(0, 2, (R,), generator=g).tolist()]
        trows = [SimpleNamespace(b=b, key=("u%d" % b, 1, k), kind=kd) for b, k, kd in zip(bs, ks, kinds)]
        vrows = [r for r in trows if r.kind == "v"]
        drows = [r for r in trows if r.kind == "d"]
        stub = SimpleNamespace(_kslot={r.key: r.key[2] for r in trows}, KEYS=7, B=4, _h2d=lambda t: t)
        (vb, vk, db, dk, row_b, row_plane), rounds, n_rec = OWN2.OwnResidency._fr_rows(stub, vrows, drows, trows)
        grp = row_plane * 4 + row_b
        rnd = SB.rank_in_group(grp, torch.ones_like(grp, dtype=torch.bool))
        want = [(rnd == r).nonzero().squeeze(1).tolist() for r in range(int(rnd.max()) + 1)]
        assert [x.tolist() for x in rounds] == want
        assert vb.tolist() == [r.b for r in vrows] and dk.tolist() == [r.key[2] for r in drows]
        assert row_plane.tolist() == [r.key[2] + (0 if r.kind == "v" else 7) for r in trows]
        if drows:
            occ = SB.rank_in_group(db, torch.ones_like(db, dtype=torch.bool))
            assert n_rec == int(occ.max()) + 1
        else:
            assert n_rec == 0


def test_HF5e_negative_control_one_pin_round_differs(p3, ref):
    """L34: a fewer-reads path whose pin rounds are merged into ONE round (duplicate (plane, b) in one indexed write:
    the later row's words replace the earlier's) differs from the default, so HF5a's comparison sees a wrong round."""
    toks, _ = ref
    base = _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False))
    merged = []

    def one_round(eng, svc):
        f0 = svc._fr_rows

        def fr_rows(vrows, drows, trows):
            rows6, rounds, n_rec = f0(vrows, drows, trows)
            if len(rounds) >= 2:
                merged.append(len(rounds))
                rounds = [torch.cat(rounds)]
            return rows6, rounds, n_rec
        svc._fr_rows = fr_rows
    bad = _run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False, fewer_reads=True), setup=one_round)
    assert merged, "precondition: a tick with two or more pin rounds"
    assert _diff(base, bad) is not None or [r.out for r in base.rnd.reqs] != [r.out for r in bad.rnd.reqs] \
        or TB._rep(base.svc) != TB._rep(bad.svc)


def test_HF5f_the_harness_wiring():
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).b2_fewer_reads is False
    assert G.parse_args(base + ["--backends", "b2", "--X", "64", "--b2-fewer-reads"]).b2_fewer_reads is True
    with pytest.raises(SystemExit):
        G.parse_args(base + ["--b2-fewer-reads"])
    assert 'fewer_reads=bool(getattr(a, "b2_fewer_reads", False))' in inspect.getsource(G.make_engine)
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["fewer_reads"].default is False
    assert inspect.signature(OWN2.O.pins_or).parameters["rounds"].default is None
    assert inspect.signature(OWN2.O.recency).parameters["n_rounds"].default is None


# =====================================================================================================================
# HF6 the step cells evalfast2 / evalfast2t / evalfast2c (static, then a DRYRUN on a clean checkout of this ROOT)
# =====================================================================================================================
REPO = HERE.parents[2]
STEP = EFF / "r2_tick_sg5_step.sh"
HF_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3tall"                    # [f3tall] (was nosi-r2-f3all / -f3ldx / -f3lia / -endbatch / -applyev)


def test_HF6a_the_cells_and_the_registration_static():
    text = STEP.read_text()
    lines = text.splitlines()
    root = next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0][len("NOSI_EVAL="):]
    assert root == HF_ROOT
    assert ("|evalfast2c|evalfast2f192|evalfast2t192|evalfast2c192|evalfast2w|evalfast2wi|evalfast2fr|evalfast2fx|evalfast2wx"
            "|evalfast2rp|evalfast2rl|evalfast2rd|evalfast2rdp"                                    # [resfix] [resdev]
            "|evalfast2c3|evalfast2f3|evalfast2f3_192"                                             # [fr3] the v3 cells
            "|evalfast2f3p|evalfast2c3l|evalfast2f3l"                                              # [flean]
            "|evalfast2ia"                                                                         # [admitfill]
            "|evalacc216m|evalacc216"                                                              # [acc216]
            "|evalfast2f3lia|evalfast2c3lia|evalacc216f3lia"                                       # [f3lia]
            "|evalfast2eb"                                                                         # [endbatch]
            "|evalfast2ab|evalfast2c3ab"                                                           # [applyev]
            "|evalfast2all|evalfast2c3all|evalacc216all"                                           # [f3all]
            "|evalfast2f3d|evalfast2c3d|evalfast2f3t|evalfast2c3t"                                 # [f3ld]
            "|evalfast2f3c|evalfast2c3c"                                                           # [f3ldx]
            "|evalfast2tall|evalfast2c3tall|evalacc216tall] [gilfree|gather]") in text             # [f3tall]
    assert "        evalfast2|evalfast2t|evalfast2c) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    reg = text[text.index("# REGISTERED for evalfast2t / evalfast2 / evalfast2c"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("HF-T-MED", "HF-T-MEAN", "HF-T-EFF", "HF-T-GC", "HF-T-HOST", "HF-T-FALS", "HF-F-MED", "HF-F-MEAN",
                "HF-F-EFF", "HF-F-FALS", "HF-C", "HF-MEM", "STAMP 2026-10-22T21:23:24Z"):
        assert tag in reg, tag
    # the registered bands are the cells' --eval-bands (SMED_LO:SMED_HI:SMEAN_LO:SMEAN_HI:EFF_LO:EFF_HI)
    assert "point 250, band [220, 300]" in reg and "point 230, band [195, 290]" in reg and "point 175, band [140, 220]" in reg
    assert "point 105, band [75, 150]" in reg and "point 95, band [65, 145]" in reg and "point 85, band [55, 130]" in reg
    blk = text[text.index("    evalfast2|evalfast2t) MODES=fast"):text.index("    evalfast2c) MODES=fast")]
    assert "RES=fused ; BANDS=75:150:65:145:55:130" in blk and "RES=torch ; BANDS=220:300:195:290:140:220" in blk
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def _clean_head():
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
        untracked = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all", "--",
                                    "nosi"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        return None
    return head.stdout.strip()


def _dry(tmp_path, cell):
    uid = "33cfb929-315c-8d27-5622-1169affb7e04"
    (tmp_path / "418.gpu_uuid").write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    return subprocess.run(["bash", str(STEP), "--nosi", _clean_head(), "418", "t1", cell], env=env,
                          capture_output=True, text=True, timeout=180)


HF_CELLS = dict(
    evalfast2=dict(has=["--b2-resolve fused", "--eval-bands 75:150:65:145:55:130", "--bare", "--eval-role fast-time",
                        "--host-profile --gc-freeze --lean-time", "--steps 48", "--draft-ticks 12",
                        "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json",
                        "--mem-cap-gb 84"],
                   hasnt=["torch/per_call", "--wait-index", "--b2-fewer-reads"]),
    evalfast2t=dict(has=["--b2-resolve torch", "--eval-bands 220:300:195:290:140:220", "--bare",
                         "--eval-role fast-time", "--host-profile --gc-freeze --lean-time", "--steps 48",
                         "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json"],
                    hasnt=["torch/per_call", "--wait-index", "--b2-fewer-reads"]),
    evalfast2c=dict(has=["--b2-resolve check", "--steps 8", "--draft-ticks 4"],
                    hasnt=["--bare", "--eval-role", "--accuracy-record", "--host-profile", "--gc-freeze", "--lean-time",
                           "torch/per_call", "--eval-bands"]),
    evalfast=dict(has=["--variants fused/batched,torch/per_call", "--eval-bands 55:125:30:70:27:65", "--bare"],
                  hasnt=["--b2-resolve", "--host-profile", "--gc-freeze", "--lean-time"]))


@pytest.mark.parametrize("cell", sorted(HF_CELLS))
def test_HF6b_the_cells_dryrun(tmp_path, cell):
    """On a clean checkout: the DRYRUN prints each cell's arguments, which the harness parses into the registered
    flags. In this ROOT's own checkout a DRYRUN stop FAILS (never a skip); elsewhere it skips."""
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the hostfix ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    common = ["--modes fast --backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464", "--mem=240G",
              "--layout pool --X-pool 1512 --arrival-lag 2", "--repeat-docs --log-slots -1 --skip-shipped",
              "PYTHONPATH=%s/nosi" % HF_ROOT, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell]
    if cell != "evalfast":
        common.append("--variants fused/batched")
    for w in common + HF_CELLS[cell]["has"]:
        assert w in got or w in line, (cell, w)
    for w in HF_CELLS[cell]["hasnt"]:
        assert w not in got, (cell, w)
    assert "claim /mnt/beegfs/ojerbi/r2-runs/sg5/sg5_active_418 (one per holder)" in r.stdout
    pa = _sg5().parse_args(got.split())
    assert pa.batch == 216 and pa.X_pool == 1512 and pa.backends == ["b2"] and pa.layout == "pool"
    want = dict(evalfast2=("fused", True), evalfast2t=("torch", True), evalfast2c=("check", False),
                evalfast=("torch", False))[cell]
    assert (pa.b2_resolve, pa.host_profile and pa.gc_freeze and pa.lean_time) == want, cell
    assert not pa.wait_index and not pa.b2_fewer_reads
    if cell in ("evalfast2", "evalfast2t"):
        assert pa.eval_role == "fast-time" and pa.variant_list == [("fused", "batched")] and pa.bare
    if cell == "evalfast2c":
        assert pa.eval_role == "" and pa.steps == 8 and pa.draft_ticks == 4 and not pa.bare


def test_HF6c_a_wrong_cell_word_is_refused(tmp_path):
    uid = "33cfb929-315c-8d27-5622-1169affb7e04"
    (tmp_path / "418.gpu_uuid").write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    for bad in (("evalfast3",), ("evalfast2", "evalfast2t"), ("evalfast2x",)):
        r = subprocess.run(["bash", str(STEP), "--nosi", "0" * 40, "418", "t1", *bad], env=env, capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 2 and "usage" in r.stderr, (bad, r.stderr[-200:])


# HF7 [b192] the B 192 resolver pair evalfast2c192 / evalfast2f192 / evalfast2t192 (the fused path does not fit B 216)
B192_CELLS = dict(
    evalfast2f192=dict(res="fused", flags=True, has=["--eval-bands 75:140:60:135:50:120", "--bare", "--steps 48",
                                                     "--eval-role fast-time", "--mem-cap-gb 84",
                                                     "--host-profile --gc-freeze --lean-time"]),
    evalfast2t192=dict(res="torch", flags=True, has=["--eval-bands 220:300:180:290:140:230", "--bare", "--steps 48",
                                                     "--eval-role fast-time", "--mem-cap-gb 84",
                                                     "--host-profile --gc-freeze --lean-time"]),
    evalfast2c192=dict(res="check", flags=False, has=["--steps 8", "--draft-ticks 4"]))


def test_HF7a_the_b192_cells_and_the_registration_static():
    text = STEP.read_text()
    assert "        evalfast2f192|evalfast2t192|evalfast2c192) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    reg = text[text.index("# REGISTERED for evalfast2c192 / evalfast2f192 / evalfast2t192"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-22T22:59:44Z", "B1-C", "B1-T-MED", "B1-T-MEAN", "B1-T-EFF", "B1-F-MED", "B1-F-FALS",
                "B1-MEM", "point 255, band [220, 300]", "point 100, band [75, 140]", "[180, 290]", "[140, 230]",
                "[60, 135]", "[50, 120]", "NOT a headline"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2f192|evalfast2t192) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    assert "RES=fused ; BANDS=75:140:60:135:50:120" in blk and "RES=torch ; BANDS=220:300:180:290:140:230" in blk
    assert "BATCH=192" in blk and "PX=1344 ; PRV=1296" in blk and "--X-pool 1344" in blk


@pytest.mark.parametrize("cell", sorted(B192_CELLS))
def test_HF7b_the_b192_cells_dryrun(tmp_path, cell):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the hostfix ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    for w in ["--modes fast --backends b2", "--batch 192", "--C 96 --X 1344 --R-ver 1296", "--mem=240G",
              "--layout pool --X-pool 1344 --arrival-lag 2", "--repeat-docs --log-slots -1 --skip-shipped",
              "--variants fused/batched", "--job-name=sg5-t1-418-%s " % cell] + B192_CELLS[cell]["has"]:
        assert w in got or w in line, (cell, w)
    assert "--wait-index" not in got and "--b2-fewer-reads" not in got and "torch/per_call" not in got
    pa = _sg5().parse_args(got.split())
    assert pa.batch == 192 and pa.X_pool == 1344 and pa.b2_resolve == B192_CELLS[cell]["res"]
    assert bool(pa.host_profile and pa.gc_freeze and pa.lean_time) == B192_CELLS[cell]["flags"]
    if cell == "evalfast2c192":
        assert pa.eval_role == "" and pa.steps == 8 and not pa.bare and "--accuracy-record" not in got
    else:
        assert pa.eval_role == "fast-time" and pa.bare
        assert "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json" in got


# HF8 [hostw] evalfast2w = evalfast2t + --wait-index + --b2-fewer-reads, registered before any run
def test_HF8a_evalfast2w_static():
    text = STEP.read_text()
    assert "        evalfast2w) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    reg = text[text.index("# REGISTERED for evalfast2w"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T01:03:07Z", "HW-MED", "point 295, band [270, 330]", "HW-MEAN  [215, 270]",
                "HW-EFF [185, 235]", "HW-FALS", "below 262", "HW-RES", "HW-MEM"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2w) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    assert "--eval-bands 270:330:215:270:185:235 --b2-resolve torch" in blk and "--wait-index --b2-fewer-reads" in blk


def test_HF8b_evalfast2w_dryrun(tmp_path):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, "evalfast2w")
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the hostfix ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2", "--bare",
              "--eval-role fast-time", "--mem-cap-gb 84", "--eval-bands 270:330:215:270:185:235", "--steps 48"):
        assert w in got or w in line, w
    pa = _sg5().parse_args(got.split())
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert pa.wait_index and pa.b2_fewer_reads and pa.batch == 216 and pa.variant_list == [("fused", "batched")]


# HF9 [hostw2] the single-flag split of evalfast2w: evalfast2wi (--wait-index) / evalfast2fr (--b2-fewer-reads)
W2_CELLS = dict(evalfast2wi=("254:273:200:230:175:200", True, False),
                evalfast2fr=("195:225:165:195:148:172", False, True))


def test_HF9a_the_split_cells_static():
    text = STEP.read_text()
    assert "        evalfast2wi|evalfast2fr) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    reg = text[text.index("# REGISTERED for evalfast2wi / evalfast2fr"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T02:01:01Z", "WI-MED", "point 264, band [254, 273]", "WI-MEAN [200, 230]",
                "WI-EFF [175, 200]", "FR-MED", "point 209, band [195, 225]", "FR-MEAN [165, 195]", "FR-EFF [148, 172]",
                "FALSIFIER: h_hs_merges median > 15 ms", "FALSIFIER: h_hs_merges median < 20 ms", "W2-MEM"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2wi|evalfast2fr) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    for cell, (bands, _, _) in W2_CELLS.items():
        assert "BANDS=%s" % bands in blk, cell


@pytest.mark.parametrize("cell", sorted(W2_CELLS))
def test_HF9b_the_split_cells_dryrun(tmp_path, cell):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the hostfix ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    bands, wi, fr = W2_CELLS[cell]
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2", "--bare",
              "--eval-role fast-time", "--mem-cap-gb 84", "--eval-bands " + bands, "--steps 48",
              "PYTHONPATH=%s/nosi" % HF_ROOT):
        assert w in got or w in line, (cell, w)
    pa = _sg5().parse_args(got.split())
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert (bool(pa.wait_index), bool(pa.b2_fewer_reads)) == (wi, fr), cell


# HF10 [frfix] the device-selected --b2-fewer-reads: evalfast2fx (--b2-fewer-reads) / evalfast2wx (both flags)
FX_CELLS = dict(evalfast2fx=("250:275:200:225:178:198", False, True),
                evalfast2wx=("255:282:205:235:180:205", True, True))


def test_HF10a_the_fix_cells_static():
    text = STEP.read_text()
    assert "        evalfast2fx|evalfast2wx) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    reg = text[text.index("# REGISTERED for evalfast2fx / evalfast2wx"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T03:52:16Z", "FX-MED", "point 262, band [250, 275]", "FX-MEAN [200, 225]", "FX-EFF [178, 198]",
                "FX-FALS", "h_hs_merges > 15 ms", "by > 20 ms", "WX-MED", "point 268, band [255, 282]",
                "WX-MEAN [205, 235]", "WX-EFF [180, 205]", "263.8 / 267.7", "WX-FALS", "minus 5 tok/s", "< 255",
                "FX-MEM", "peak_gb <= 84"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2fx|evalfast2wx) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    for cell, (bands, _, _) in FX_CELLS.items():
        assert "BANDS=%s" % bands in blk, cell


@pytest.mark.parametrize("cell", sorted(FX_CELLS))
def test_HF10b_the_fix_cells_dryrun(tmp_path, cell):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the frfix ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    bands, wi, fr = FX_CELLS[cell]
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2", "--bare",
              "--eval-role fast-time", "--mem-cap-gb 84", "--eval-bands " + bands, "--steps 48",
              "PYTHONPATH=%s/nosi" % HF_ROOT):
        assert w in got or w in line, (cell, w)
    pa = _sg5().parse_args(got.split())
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert (bool(pa.wait_index), bool(pa.b2_fewer_reads)) == (wi, fr), cell


# HF11 [resfix] evalfast2wx + one flag: evalfast2rp (--b2-res-profile) / evalfast2rl (--res-lean)
RX_CELLS = dict(evalfast2rp=("200:340:150:300:120:260", True, False),
                evalfast2rl=("280:330:225:265:195:225", False, True))


def test_HF11a_the_resfix_cells_static():
    text = STEP.read_text()
    assert "        evalfast2rp|evalfast2rl) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "|evalfast2fx|evalfast2wx|evalfast2rp|evalfast2rl|evalfast2rd|evalfast2rdp|" in text   # [flean] fr3's cells follow
    assert "|evalfast2eb" in text                                                                  # [endbatch] ([f3all])
    reg = text[text.index("# REGISTERED for evalfast2rp / evalfast2rl"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T10:08:04Z", "RP ", "13,900", "60-120 ms", "REPORT only", "RL-MED", "point 300, band [280, 330]", "308",
                "RL-MEAN  [225, 265]", "RL-EFF [195, 225]", "RL-FALS", "median < 275", ">= 40 ms", "RL-MEM",
                "peak_gb <= 84", "nosi-r2-resfix"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2rp|evalfast2rl) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    for cell, (bands, _, _) in RX_CELLS.items():
        assert "BANDS=%s" % bands in blk, cell
    assert "STEP_MIN=150" in blk and "--wait-index" in blk and "--b2-fewer-reads" in blk


@pytest.mark.parametrize("cell", sorted(RX_CELLS))
def test_HF11b_the_resfix_cells_dryrun(tmp_path, cell):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the resfix ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    bands, prof, lean = RX_CELLS[cell]
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2", "--bare",
              "--eval-role fast-time", "--mem-cap-gb 84", "--eval-bands " + bands, "--steps 48",
              "PYTHONPATH=%s/nosi" % HF_ROOT, "--time=%d" % (150 if prof else 120)):
        assert w in got or w in line, (cell, w)
    pa = _sg5().parse_args(got.split())
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert pa.wait_index and pa.b2_fewer_reads, cell                  # evalfast2wx's flags
    assert (bool(pa.b2_res_profile), bool(pa.res_lean)) == (prof, lean), cell
    assert not pa.res_dev, cell                                         # [resdev] not in the resfix cells


# HF13 [resdev] evalfast2rl + --res-dev: evalfast2rd (FAST) / evalfast2rdp (+ --b2-res-profile)
RD_CELLS = dict(evalfast2rd=("335:420:260:300:222:255", False),
                evalfast2rdp=("200:340:150:300:120:260", True))


def test_HF13a_the_resdev_cells_static():
    text = STEP.read_text()
    assert "        evalfast2rd|evalfast2rdp) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "|evalfast2rp|evalfast2rl|evalfast2rd|evalfast2rdp|" in text      # [flean] fr3's cells follow
    assert "|evalfast2eb" in text                                            # [endbatch] ([f3all])
    reg = text[text.index("# REGISTERED for evalfast2rd / evalfast2rdp"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T13:17:36Z", "before any run", "RD-MED", "point 360, band [335, 420]", "RD-MEAN  [260, 300]",
                "RD-EFF [222, 255]", "RD-FALS", "median < 330", "132 ms by >= 25 ms", "RDP ", "REPORT only",
                "RD-MEM", "peak_gb <= 84", "nosi-r2-graph"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2rd|evalfast2rdp) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    for cell, (bands, _) in RD_CELLS.items():
        assert "BANDS=%s" % bands in blk, cell
    assert "STEP_MIN=150" in blk and "--wait-index" in blk and "--b2-fewer-reads" in blk
    assert "RDFLAGS=(--res-lean --res-dev --b2-res-profile)" in blk and "RDFLAGS=(--res-lean --res-dev)" in blk


@pytest.mark.parametrize("cell", sorted(RD_CELLS))
def test_HF13b_the_resdev_cells_dryrun(tmp_path, cell):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the resdev ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    bands, prof = RD_CELLS[cell]
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2", "--bare",
              "--eval-role fast-time", "--mem-cap-gb 84", "--eval-bands " + bands, "--steps 48",
              "PYTHONPATH=%s/nosi" % HF_ROOT, "--time=%d" % (150 if prof else 120)):
        assert w in got or w in line, (cell, w)
    pa = _sg5().parse_args(got.split())
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert pa.wait_index and pa.b2_fewer_reads, cell                  # evalfast2wx's flags
    assert pa.res_lean and pa.res_dev, cell                           # evalfast2rl's flag + --res-dev
    assert bool(pa.b2_res_profile) == prof, cell


# HF12 [fr3] the fused v3 kernel: evalfast2c3 (CHECK at B 216) / evalfast2f3 (FAST at B 216) / evalfast2f3_192 (B 192)
F3_CELLS = dict(evalfast2c3=dict(batch=216, X=1512, rver=1464, res="check", bands=None),
                evalfast2f3=dict(batch=216, X=1512, rver=1464, res="fused", bands="330:650:250:560:210:480"),
                evalfast2f3_192=dict(batch=192, X=1344, rver=1296, res="fused", bands="300:620:230:530:200:450"))


def test_HF12a_the_v3_cells_static():
    text = STEP.read_text()
    assert "        evalfast2c3|evalfast2f3|evalfast2f3_192) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    reg = text[text.index("# REGISTERED for evalfast2c3 / evalfast2f3 / evalfast2f3_192"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T11:29:23Z", "C3", "fused v3 == torch at every", "launches 4 per draft-only layer", "7 per mixed layer",
                "< 0.5 GB", "v2: 3.0 GB", "[oomfix] stderr line", "VOID", "F3-MED", "point 450, band [330, 650]",
                "550-600", "F3-MEAN [250, 560]", "F3-EFF [210, 480]", "F3-192", "point 420, band [300, 620]",
                "F3-192-MEAN [230, 530]", "F3-192-EFF [200, 450]", "103 tok/s", "217 tok/s", "F3-FALS",
                "median < 280", "> 100 ms", "F3-MEM", "peak_gb <= 84"):
        assert tag in reg, tag
    i3 = text.index("    evalfast2c3) MODES=fast")
    blk = text[i3:text.index("\nesac", i3)]
    assert "--b2-resolve check --fused-kernel v3" in blk and "STEPS=8 ; DTICKS=4" in blk
    assert "--b2-resolve fused --fused-kernel v3" in blk and "--wait-index --b2-fewer-reads" in blk
    for cell, w in F3_CELLS.items():
        if w["bands"]:
            assert "BANDS=%s" % w["bands"] in blk, cell


@pytest.mark.parametrize("cell", sorted(F3_CELLS))
def test_HF12b_the_v3_cells_dryrun(tmp_path, cell):
    if _clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = _dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF_ROOT).resolve() != REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the fr3 ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    w = F3_CELLS[cell]
    for s in ("--modes fast --backends b2", "--batch %d" % w["batch"], "--C 96 --X %d --R-ver %d" % (w["X"], w["rver"]),
              "--layout pool --X-pool %d --arrival-lag 2" % w["X"], "--repeat-docs --log-slots -1 --skip-shipped",
              "--variants fused/batched", "--mem=240G", "--b2-resolve %s" % w["res"], "--fused-kernel v3",
              "PYTHONPATH=%s/nosi" % HF_ROOT, "--job-name=sg5-t1-418-%s " % cell.replace("_", "-")):   # the step's job names use '-'
        assert s in got or s in line, (cell, s)
    assert "torch/per_call" not in got
    pa = _sg5().parse_args(got.split())
    assert (pa.batch, pa.X_pool, pa.b2_resolve, pa.fused_kernel) == (w["batch"], w["X"], w["res"], "v3"), cell
    if cell == "evalfast2c3":
        assert pa.eval_role == "" and pa.steps == 8 and pa.draft_ticks == 4 and not pa.bare
        for s in ("--accuracy-record", "--eval-bands", "--host-profile", "--wait-index", "--b2-fewer-reads"):
            assert s not in got, (cell, s)
    else:
        assert pa.eval_role == "fast-time" and pa.bare and pa.steps == 48 and pa.mem_cap_gb == 84
        assert "--eval-bands " + w["bands"] in got
        assert pa.host_profile and pa.gc_freeze and pa.lean_time and pa.wait_index and pa.b2_fewer_reads
        assert "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json" in got


def test_HF12c_the_v3_harness_wiring():
    """--fused-kernel: default v2, refused with the torch resolver (an explicit v2 too), passed to OwnResidency only
    with a fused resolver; G2 follows the report's kernel (v2 unchanged); note_fused_warm keeps the process's first
    warm_mem_gb for meta.fused_warm_mem_gb."""
    G = _sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x",
            "--backends", "b2", "--X", "64"]
    assert G.parse_args(base).fused_kernel == "v2"
    assert G.parse_args(base + ["--b2-resolve", "fused"]).fused_kernel == "v2"
    assert G.parse_args(base + ["--b2-resolve", "check", "--fused-kernel", "v3"]).fused_kernel == "v3"
    for bad in (["--fused-kernel", "v3"], ["--fused-kernel", "v2"], ["--b2-resolve", "fused", "--fused-kernel", "v4"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    # G2 (fused_dispatch): v2 as before (a report without "kernel" is v2's), v3 needs 4.0 and --fused-kernel v3
    rep = dict(err=0, err_bits=[], launches_per_layer=3.0, sel_conversions=0, bias_casts=0, compiled={})

    def g2(fk, **kw):
        a = SimpleNamespace(b2_resolve="fused", b2_fused_plant=0, b2_res_profile=False, stack="", fused_kernel=fk)
        gd = [x for x in G.fused_gates(a, "L", "draft_only", dict(rep, **kw)) if x["name"].endswith("_fused_dispatch")]
        assert len(gd) == 1 and gd[0]["must"]
        return gd[0]
    v2 = g2("v2")
    assert v2["passed"] and v2["detail"] == "launches/layer 3.0 sel_conversions 0 bias_casts 0 compiled {}"
    assert g2("v2", kernel="v2")["passed"] and not g2("v2", launches_per_layer=4.0)["passed"]
    assert g2("v3", kernel="v3", launches_per_layer=4.0)["passed"]
    assert not g2("v3", kernel="v3")["passed"]                                 # 3.0 is v2's count
    assert not g2("v2", kernel="v3", launches_per_layer=4.0)["passed"]         # the report's kernel is not the flag's
    assert not g2("v3", launches_per_layer=3.0)["passed"]                      # --fused-kernel v3, a v2 report
    d3 = g2("v3", kernel="v3", launches_per_layer=4.0)["detail"]
    assert d3.endswith("compiled {} kernel v3 (launches/layer want 4.0, --fused-kernel v3)"), d3
    assert G.FR_DRAFT_LAUNCHES == dict(v2=3.0, v3=4.0)
    # note_fused_warm: the first non-None value of the process
    old = G._FUSED_WARM[0]
    try:
        G._FUSED_WARM[0] = None
        assert G.note_fused_warm(None) is None and G.note_fused_warm(SimpleNamespace(fusedres=None)) is None
        assert G.note_fused_warm(SimpleNamespace(fusedres=SimpleNamespace(warm_mem_gb=None))) is None   # a later engine
        def svc(w):
            return SimpleNamespace(fusedres=SimpleNamespace(warm_mem_gb=w))
        assert G.note_fused_warm(svc([10.544, 10.3])) == [10.544, 10.3] == G._FUSED_WARM[0]
        assert G.note_fused_warm(svc([9.0, 8.0])) == [10.544, 10.3]                # the first one stays
    finally:
        G._FUSED_WARM[0] = old
    assert "note_fused_warm(svc)" in inspect.getsource(G.preflight)
    assert "note_fused_warm(svc)" in inspect.getsource(G.one_point)
    msrc = inspect.getsource(G.main)
    assert msrc.count('meta["fused_warm_mem_gb"] = _FUSED_WARM[0]') == 2
    assert 'fused_kernel=a.fused_kernel if a.b2_resolve != "torch" else None' in msrc
    assert 'fused_kernel=(None if getattr(a, "b2_resolve", "torch") == "torch"' in inspect.getsource(G.make_engine)
    # OwnResidency: fused_kernel None = r2_fusedres.DEFAULT_KERNEL; refused with resolve torch (before any build)
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["fused_kernel"].default is None
    assert "kernel=fused_kernel" in inspect.getsource(OWN2.OwnResidency.__init__)
    with pytest.raises(ValueError, match="fused_kernel needs resolve"):
        OWN2.OwnResidency(resolve="torch", fused_kernel="v3")


def test_HF1c_the_harness_wiring():
    G = _sg5()
    src = inspect.getsource(G.round_guard)
    for need in ("gc.collect()", "gc.freeze()", "gc.disable()", "gc.unfreeze()", "gc.enable()",
                 'rec.setdefault("gc_freeze", {})[tag] = frz'):
        assert need in src, need
    y = src.index("yield gcl")                                        # (the docstring names the calls too)
    assert src.rindex("gc.freeze()") < y < src.index("gc.unfreeze()", y)
    assert src.index("gc.callbacks.remove(gcl)") < src.index('frz["collected_after"] = gc.collect()')
