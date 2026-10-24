"""CPU tests of [b2-desc] (ported by hunks from 47db0c0 onto f3lia 89c9169: docs/R2_B2_DESC_PLAN.md of 47db0c0):
OwnResidency(issue="desc"), B2's draft takes issued as ONE descriptor job per tick through E's
GilfreeTransport.issue_desc (worker "cpp", packer v3 with DESC), against the default object path (issue="object":
tp.issue per layer, Residency.end_tick) on the SAME transport. The world is SG5's CPU stand-in (test_r2_tick.py
make_res_engine: the tiny model on the Residency layout, S-C's CPU device stand-in), with E's transport in place of
K3X's adversarial one (built from this tree's gilfree_pack.{py,cpp}). [f3ld] On this tree the job is queued by the host
consumer and submitted in end_tick's issue phase (ResidencySC.end_tick's _tick_issue hook), where the object path's
Residency.end_tick issues: the order of the issues against every other event is the object path's.

DETERMINISM. The C++ worker lands asynchronously, so both arms use the fixed arrival rule arrival_lag = 2 (F22: a
transfer is admitted exactly at the poll of tick issue_tick + 2: the state depends on the schedule only) and the
per-tick record first waits for every issued job (tp.flush()): the store is then the same in both arms if and only if
the same bytes landed. The tick's events (reads, write-backs) are ONE stream (_Stream: an event completes when it, or a
later event, is waited; the stream also completes everything recorded `late` ticks before the record): the GPU's
in-order main stream, so the descriptor path's waits on the NEWEST reader / write-back mark are the object path's
waits on every pending event.

  BD1  desc == object at every tick of the teacher-forced round (forced_schedule), b2: the rows and tokens, the logits
       (torch.equal), the page tables, the vendored projection's state (test_r2_residency._state), B2's TABLE (every
       column and array, Table.diff), the counters, the readers, memory(), the arrival counters (held, forced) and the
       STORE (K, V, bias of every layer, a copy per tick after the flush, torch.equal); at the end the stores
       torch.equal, the committed tokens, the journal per kind, the report's comparable parts. Preconditions: the desc
       arm issued >= 1 job and every draft take through it (groups equal the object arm's), one release per job against
       more releases on the object arm; the device stand-in waited on transfers in flight (the lag keeps them in flight
       for a tick); with late 2 under pressure the transport host-waited pending events at issue (T3 / T4 marks
       exercised).
  BD2  NEGATIVE CONTROL: a corrupted descriptor (one entry's block + 1: the next block's host rows into the slot) at
       the first tick where such an entry exists. Precondition (lesson L34): at that moment the host rows of block and
       block + 1 differ for that (layer, request, head), so the plant must change the landed bytes. Detected: the first
       difference from the clean desc round is that tick's STORE (every host-state field equal there: the table cannot
       see it), and the planted slot holds block + 1's rows.
  BD3  refusals and the harness: issue="desc" over K3X's adversarial transport and over the Python feeder is refused
       at CONSTRUCTION (lesson L31), an unknown issue too; job_ev needs the C++ worker; r2_tick_sg5 --worker /
       --b2-issue parse (defaults python / object; desc without --worker cpp, or on gather, refused; [f3ld] desc with
       --layout pool / both refused: E's issue_desc refuses xpool lanes); make_engine passes both; issue_clock counts
       the issue calls, takes and host time of a transport (REPORT); [f3ld] the issue phase hook.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_b2_desc.py
"""
from __future__ import annotations

import ast
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


TT = _shared("test_r2_tick_shared_b2", "test_r2_tick.py")             # the tiny model, make_res_engine (package r2tick)
TRM = _shared("test_r2_residency_shared_b2", "test_r2_residency.py")  # B's snapshot helpers (package r2res)
TK, OWN2 = TT.TK, TT.OWN2
GF = TT._load("r2_transport_gilfree")                                 # E's transport, in the package of r2_own2
RSC = TT.RSC
EVD = sys.modules[TT.PKG + ".kstep.backend"].ev_done
p3, ref = TT.p3, TT.ref                                               # the fixtures, by name
EFF, NOSI = TT.EFF, TT.NOSI
BS = TT.BS
CFG = dict(default=dict(C=12, X=24, R_ver=4),                          # T15's capacities
           pressure=dict(C=8, X=40, R_ver=4))                          # the LRU over C: slots taken again
LAG = 2                                                               # the fixed arrival rule (F22)
FIELDS = ("rows", "tokens", "logits", "pages", "state", "table", "counters", "readers", "memory", "arrival", "store")


@pytest.fixture(scope="module")
def v3(tmp_path_factory):
    """Packer v3 of THIS tree (benchmarks/Efficiency/gilfree_pack.{py,cpp}), built (OpenMP), with DESC."""
    spec = importlib.util.spec_from_file_location("gilfree_pack_v3_b2desc", str(EFF / "gilfree_pack.py"))
    gp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gp
    spec.loader.exec_module(gp)
    ext = gp.load(str(tmp_path_factory.mktemp("gilfree3_b2desc_build")))
    assert int(ext.VERSION) == 3 and int(ext.DESC) == 1 and int(ext.OPENMP) == 1
    return ext


class _Stream:
    """The tick's events as ONE in-order stream. An event completes when it or a later event is waited (done = True:
    Be.host_wait; synchronize), and advance() (the end of each tick) completes everything recorded `late` ticks
    earlier (late 0: every event is complete when recorded, the TB tests' Ev(True))."""

    def __init__(self, late: int):
        self.late, self.n, self.done_upto, self.marks, self.evs = int(late), 0, -1, [], []

    def record(self):
        ev = _SEv(self, self.n)
        self.n += 1
        self.evs.append(ev)
        if self.late == 0:
            self.done_upto = ev.k
        return ev

    def advance(self) -> None:
        self.marks.append(self.n)
        if self.late and len(self.marks) > self.late:
            self.done_upto = max(self.done_upto, self.marks[-1 - self.late] - 1)


class _SEv:
    __slots__ = ("s", "k")

    def __init__(self, s, k):
        self.s, self.k = s, k

    @property
    def done(self):
        return self.k <= self.s.done_upto

    @done.setter
    def done(self, v):
        if v:
            self.s.done_upto = max(self.s.done_upto, self.k)

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


def _factory(ext, worker, cnt):
    """make_res_engine's transport hook: E's GilfreeTransport in place of K3X's AdversarialTransport (same arguments);
    its host_wait counts the PENDING events it waits (CPU mode: T3 / T4 are host-waited at issue)."""
    def make(be, host_k, host_v, sk, sv, bs, seed, mode=None, max_delay=None):
        def hw(ev):
            if ev is not None and not EVD(ev):
                cnt["host_waits"] += 1
            be.host_wait(ev)
        return GF.GilfreeTransport(ext, host_k, host_v, sk, sv, bs, int(host_k[0].shape[-1]), threads=2, host_wait=hw,
                                   cap_groups=8, worker=worker)
    return make


def _engine(p3, ext, issue, cfg="default", checked=False, worker="cpp", cnt=None, seed=1, lag=LAG, svc_kw=None):
    """A b2 engine of make_res_engine on E's transport. svc_kw: more OwnResidency arguments ([f3ld] the eval flags)."""
    cnt = cnt if cnt is not None else dict(host_waits=0)
    kw = dict(checked=checked, issue=issue, arrival_lag=lag)
    kw.update(svc_kw or {})
    orig = TT.KTR.AdversarialTransport
    TT.KTR.AdversarialTransport = _factory(ext, worker, cnt)
    try:
        eng = TT.make_res_engine(TT.make_ops(p3), kind="b2", seed=seed, svc_kw=kw, **CFG[cfg])
    finally:
        TT.KTR.AdversarialTransport = orig
    return eng, cnt


def _stores(eng):
    return list(eng.store_k) + list(eng.store_v) + list(eng.store_bias)


def _state(svc):
    """test_r2_residency._state of the service (the vendored projection of every stream, then the service's pending
    map and the transport's pending list: E's transport keeps none)."""
    return TRM._state(SimpleNamespace(svc=svc, tp=SimpleNamespace(pending=[])))


def _run(p3, ext, toks, issue, cfg="default", checked=False, late=LAG, plant=None, seed=1, lag=LAG, svc_kw=None,
         forced=True, state=None, layer_rec=None, setup=None):
    """One round on b2 with E's C++ worker (teacher-forced by forced_schedule, or free-running); a record per tick
    (after its end_tick and the flush). plant(tp, svc, ticks, world, probes) may wrap the transport (BD2). [f3ld]
    state(svc): the state field (default _state); layer_rec(svc, plan): a per-LAYER record after every resolve (the
    device decisions, test_r2_f3ld.py); setup(eng, svc, ticks): called before the round."""
    eng, cnt = _engine(p3, ext, issue, cfg=cfg, checked=checked, seed=seed, lag=lag, svc_kw=svc_kw)
    world = eng.res_world
    svc, tp = world.svc, world.tp
    assert isinstance(tp, GF.GilfreeTransport) and tp.worker == "cpp" and svc.tp is tp
    stream = _Stream(late)
    tags = {}

    def new_event():
        e = stream.record()
        tags[id(e)] = "e%d" % e.k
        return e
    eng.backend.new_event = new_event
    pages, layers = [], []
    resolve0 = svc.resolve

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)]
                          for i in range(R)]))
        if layer_rec is not None:
            layers.append(layer_rec(svc, p))
        return p
    svc.resolve = resolve
    ticks, probes = [], []
    if plant is not None:
        plant(tp, svc, ticks, world, probes)
    if setup is not None:
        setup(eng, svc, ticks)
    st_of = state or _state

    def on_tick(rows, tokens, out):
        tp.flush()                                                    # every issued job ran: the store is final
        for pr in probes:
            pr(len(ticks), eng)
        ticks.append(dict(rows=[(r.b, r.kind, r.pos, r.epoch, r.i, getattr(r, "m", None)) for r in rows],
                          tokens=[t.slot if isinstance(t, TK.Ref) else int(t) for t in tokens],
                          logits=out.logits.clone(), pages=list(pages), state=st_of(svc), table=svc.own.clone(),
                          counters=TRM._counters(svc), readers=TRM._reader_tags(svc, tags), memory=svc.memory(),
                          arrival=(svc.arrival_held, svc.arrival_forced),
                          store=[t.detach().clone() for t in _stores(eng)], layers=list(layers)))
        pages.clear()
        layers.clear()
        stream.advance()
    S_list = eng.S_list

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    reject_at, _ = TK.forced_schedule(3) if forced else (None, [])
    rnd, err = None, None
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                           teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
        tp.flush()
    except Exception as e:                                            # noqa: BLE001 - a difference at its tick
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    stats = dict(tp.stats)
    final = [t.clone() for t in _stores(eng)]
    tp.close()
    return SimpleNamespace(eng=eng, svc=svc, tp=tp, ticks=ticks, rnd=rnd, err=err, stats=stats, final=final,
                           host_waits=cnt["host_waits"])


def _same_field(f, x, y):
    if f == "logits":
        return torch.equal(x, y)
    if f == "table":
        return x.diff(y) == []
    if f == "store":
        return len(x) == len(y) and all(torch.equal(a, b) for a, b in zip(x, y))
    return x == y


def _first_diff(a, b, fields=FIELDS):
    """The first (tick, field) where two rounds differ (fields in order); a round that raised earlier differs at the
    tick it did not record. None: equal at every tick."""
    n = min(len(a.ticks), len(b.ticks))
    for t in range(n):
        for f in fields:
            if not _same_field(f, a.ticks[t][f], b.ticks[t][f]):
                return (t, f)
    if len(a.ticks) != len(b.ticks) or (a.err is None) != (b.err is None):
        return (n, "raise: %s / %s" % (a.err, b.err))
    return None


def _rep(svc):
    r = svc.report()
    return dict(k43={k: v for k, v in r["k43"].items() if k not in ("wait_ns", "layer_wait_ns")}, k46=r["k46"],
                k48=r["k48"], masked=r["masked"], draft_pairs=r["draft_pairs"], draft_refused=r["draft_refused"],
                counts=r["counts"], memory=r["memory"], after_cancel=r["arrivals_after_cancel"],
                arrival={k: r["arrival"][k] for k in ("lag", "forced", "held")})


_OBJ = {}


def _obj(p3, ext, toks, cfg, checked, late):
    key = (cfg, checked, late)
    if key not in _OBJ:
        _OBJ[key] = _run(p3, ext, toks, "object", cfg=cfg, checked=checked, late=late)
        o = _OBJ[key]
        assert o.err is None and len(o.ticks) > 10, ("precondition: the object round ran", o.err)
    return _OBJ[key]


# =====================================================================================================================
# BD1 desc == object at every tick
# =====================================================================================================================
@pytest.mark.parametrize("cfg,checked,late", [("default", False, 2), ("default", True, 2), ("pressure", False, 2),
                                              ("default", False, 0)],
                         ids=["default-unchecked-late2", "default-checked-late2", "pressure-unchecked-late2",
                              "default-unchecked-late0"])
def test_BD1_desc_equals_object_at_every_tick_of_the_forced_round(p3, ref, v3, cfg, checked, late):
    toks, _ = ref
    ob = _obj(p3, v3, toks, cfg, checked, late)
    de = _run(p3, v3, toks, "desc", cfg=cfg, checked=checked, late=late)
    d = _first_diff(ob, de)
    assert d is None, ("desc differs from object", d, de.err)
    for i, (x, y) in enumerate(zip(ob.final, de.final)):
        assert torch.equal(x, y), ("final store tensor", i)
    assert [r.out for r in de.rnd.reqs] == [r.out for r in ob.rnd.reqs]
    assert TRM._journal_kinds(de.svc) == TRM._journal_kinds(ob.svc)
    ro, rd = _rep(ob.svc), _rep(de.svc)
    for k in ro:
        assert rd[k] == ro[k], (k, ro[k], rd[k])
    # preconditions: the two arms took different issue paths over the same takes
    assert ob.svc.issue == "object" and ob.svc.desc_jobs == 0
    assert de.svc.issue == "desc" and de.svc.desc_jobs > 0 and de.svc.desc_takes > 0
    assert de.svc._desc_next is None, "a queued descriptor job was never issued"
    assert de.stats["releases"] == de.svc.desc_jobs, (de.stats, de.svc.desc_jobs)        # one release per job
    assert ob.stats["releases"] > de.stats["releases"], (ob.stats, de.stats)              # per layer on the object arm
    assert ob.stats["groups"] == de.stats["groups"] == de.svc.desc_takes, (ob.stats, de.stats, de.svc.desc_takes)
    assert de.svc.report()["b2"]["desc_takes"] == de.svc.desc_takes
    # in flight: the lag keeps the prefetches in flight for a tick; the device stand-in waited on some
    assert de.svc.dev.waited > 0 and de.svc.dev.waited == ob.svc.dev.waited, (de.svc.dev.waited, ob.svc.dev.waited)
    assert de.svc.arrival_held > 0
    print("BD1 %s/%s/late%d: ticks %d, desc jobs %d, takes %d, releases obj %d / desc %d, transport host-waits of "
          "pending events obj %d / desc %d" % (cfg, "checked" if checked else "unchecked", late, len(de.ticks),
                                               de.svc.desc_jobs, de.svc.desc_takes, ob.stats["releases"],
                                               de.stats["releases"], ob.host_waits, de.host_waits))
    if cfg == "pressure" and late == 2:                               # T3 / T4 marks met PENDING events at issue
        assert ob.host_waits > 0 and de.host_waits > 0, ("precondition: no pending event at any issue",
                                                        ob.host_waits, de.host_waits)


# =====================================================================================================================
# BD2 a corrupted descriptor (NEGATIVE CONTROL)
# =====================================================================================================================
def test_BD2_a_corrupted_descriptor_changes_the_store_and_is_caught(p3, ref, v3):
    toks, _ = ref
    clean = _run(p3, v3, toks, "desc")
    assert clean.err is None and len(clean.ticks) > 10, clean.err
    rec = {}

    def plant(tp, svc, ticks, world, probes):
        orig = tp.issue_desc

        def issue_desc(desc, reader_marks=None, hist_mark=-1):
            if "tick" not in rec:
                for i in range(int(desc.shape[1])):
                    l, b, h, blk, slot = (int(x) for x in desc[:5, i])
                    if blk + 1 >= tp.nblk:
                        continue
                    hk, hv = world.host_k[l][b, :, h], world.host_v[l][b, :, h]
                    a = (hk[blk * BS:(blk + 1) * BS], hv[blk * BS:(blk + 1) * BS])
                    c = (hk[(blk + 1) * BS:(blk + 2) * BS], hv[(blk + 1) * BS:(blk + 2) * BS])
                    if torch.equal(a[0], c[0]) and torch.equal(a[1], c[1]):
                        continue                                      # this entry's plant would be a no-op (L34)
                    desc = desc.clone()
                    desc[3, i] += 1                                   # NEGATIVE CONTROL: block + 1 (off by one)
                    rec.update(tick=len(ticks), entry=(l, b, h, blk, slot), want=(c[0].clone(), c[1].clone()))
                    break
            return orig(desc, reader_marks, hist_mark)
        tp.issue_desc = issue_desc

        def probe(t, eng):                                            # the planted slot's rows at the plant tick
            if rec.get("tick") == t:
                l, b, h, blk, slot = rec["entry"]
                rec["landed"] = (eng.store_k[l][b, slot * BS:(slot + 1) * BS, h].clone(),
                                 eng.store_v[l][b, slot * BS:(slot + 1) * BS, h].clone())
        probes.append(probe)
    bad = _run(p3, v3, toks, "desc", plant=plant)
    assert "tick" in rec, "precondition: no descriptor entry whose block + 1 holds different host rows (plant no-op)"
    t = rec["tick"]
    d = _first_diff(clean, bad)
    assert d == (t, "store"), ("the corruption is first seen in the store of its tick", d, t, bad.err)
    assert torch.equal(rec["landed"][0], rec["want"][0]) and torch.equal(rec["landed"][1], rec["want"][1]), \
        "the planted slot does not hold block + 1's rows"
    later = next((k for k in range(t, min(len(clean.ticks), len(bad.ticks)))
                  if not torch.equal(clean.ticks[k]["logits"], bad.ticks[k]["logits"])), None)
    print("BD2: plant at tick %d entry (l, b, h, block, slot) %s; first logits difference at tick %s (REPORT); "
          "bad round error %s" % (t, rec["entry"], later, bad.err))


# =====================================================================================================================
# BD3 refusals and the harness
# =====================================================================================================================
def test_BD3_refusals_the_harness_flags_and_the_issue_clock(p3, v3):
    with pytest.raises(ValueError, match="needs GilfreeTransport"):          # K3X's adversarial transport
        TT.make_res_engine(TT.make_ops(p3), kind="b2", svc_kw=dict(issue="desc"))
    with pytest.raises(ValueError, match="issue must be one of"):
        TT.make_res_engine(TT.make_ops(p3), kind="b2", svc_kw=dict(issue="bulk"))
    orig = TT.KTR.AdversarialTransport
    TT.KTR.AdversarialTransport = _factory(v3, "python", dict(host_waits=0))  # the Python feeder over packer v3
    try:
        with pytest.raises(ValueError, match="needs GilfreeTransport"):
            TT.make_res_engine(TT.make_ops(p3), kind="b2", svc_kw=dict(issue="desc"))
        eng = TT.make_res_engine(TT.make_ops(p3), kind="b2", svc_kw=dict(issue="object"))    # positive: the default
        tp = eng.res_world.tp
        assert eng.res_world.svc.issue == "object" and tp.worker == "python"
        with pytest.raises(RuntimeError, match="job_ev needs worker='cpp'"):
            tp.job_ev(1)
        tp.close()
    finally:
        TT.KTR.AdversarialTransport = orig
    eng, _ = _engine(p3, v3, "desc")                                  # positive: the C++ worker builds it
    assert eng.res_world.svc.issue == "desc" and eng.res_world.svc._hm is not None
    eng.res_world.tp.close()
    # [f3ld] the X pool is refused at construction (E's issue_desc names logical slots), and the issue phase hook
    src = inspect.getsource(OWN2.OwnResidency.__init__)
    assert "does not translate xpool lanes" in src and "if self.xpool is not None:" in src
    et = inspect.getsource(RSC.ResidencySC.end_tick)
    assert et.index('getattr(self, "_tick_issue", None)') < et.index("Residency.end_tick(self)")
    assert et.index("self.apply_tick(steps)") < et.index('getattr(self, "_tick_issue", None)')
    hs = inspect.getsource(OWN2.OwnResidency._host_side)
    assert "self._queue_desc(takes, by_layer)" in hs and "self._issue_desc(" not in hs
    assert "self._issue_desc(*job)" in inspect.getsource(OWN2.OwnResidency._tick_issue)
    # the harness
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base + ["--backends", "b2"])
    assert a.worker == "python" and a.b2_issue == "object"            # p5b2 / p6 unchanged by default
    a = G.parse_args(base + ["--backends", "b2", "--worker", "cpp", "--b2-issue", "desc"])
    assert a.worker == "cpp" and a.b2_issue == "desc"
    for bad in (["--b2-issue", "desc"], ["--worker", "cpp", "--b2-issue", "desc", "--transport", "gather"],
                ["--worker", "rust"], ["--b2-issue", "bulk"],
                ["--worker", "cpp", "--b2-issue", "desc", "--layout", "pool"],          # [f3ld] xpool lanes
                ["--worker", "cpp", "--b2-issue", "desc", "--layout", "both", "--arrival-lag", "2"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + ["--backends", "b2"] + bad)
    src = (EFF / "r2_tick_sg5.py").read_text()
    tree = ast.parse(src)
    me = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name == "make_engine"))
    for need in ('worker=getattr(a, "worker", "python")', 'issue=getattr(a, "b2_issue", "object")', "issue_clock(tp)"):
        assert need in me, need
    op = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name == "one_point"))
    assert 'info["issue"] = issue_report(tp, rnd.ticks)' in op and 'info["b2_issue"]' in op
    # issue_clock on a stand-in transport: calls, takes, time; a missing method is not wrapped
    calls = []

    class _Tp:
        def issue(self, layer, transfers):
            calls.append(("issue", layer, len(transfers)))

        def issue_desc(self, desc, reader_marks=None, hist_mark=-1):
            calls.append(("desc", int(desc.shape[1])))
            return 7
    tp = _Tp()
    st = G.issue_clock(tp)
    tp.issue(0, [1, 2, 3])
    tp.issue(1, [4])
    assert tp.issue_desc(torch.zeros((6, 5), dtype=torch.int64)) == 7
    assert calls == [("issue", 0, 3), ("issue", 1, 1), ("desc", 5)]
    assert (st["calls"], st["desc_calls"], st["takes"]) == (3, 1, 9) and st["ns"] > 0
    rep = G.issue_report(tp, 2)
    assert rep["per_tick"]["takes"] == 4.5 and rep["per_tick"]["calls"] == 1.5 and rep["host_ms"] > 0

    class _Only:
        def issue(self, layer, transfers):
            pass
    o = _Only()
    G.issue_clock(o)
    assert not hasattr(o, "issue_desc") and o.issue_clock["calls"] == 0
