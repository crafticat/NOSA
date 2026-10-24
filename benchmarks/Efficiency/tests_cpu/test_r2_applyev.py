"""[applyev] CPU tests of --apply-batched (TickEngine(apply_batched=True): r2_tick.TickEngine._apply_batched and
_keep_vector). The rollbacks of apply_events join the publications' batched rebuild (the _publish_batched pattern: per
layer ONE rebuild_banks_unchecked and ONE copy_rows_many for every rollback and publication of the tick), and the
rollback's keep vector is made once per tick. The world is the SG5 tick's (test_r2_tick.py: the tiny model, piece 3 v2,
make_engine / make_res_engine; test_r2_xpool.py: make_pool_engine). The reference is the same engine with
apply_batched False and rebuild "batched" (the F3L control; test_r2_tick_p6.py P6b proves batched == per_call).

  AE1  STATE: apply_batched == the unbatched path, bitwise at every tick (rows, live logits, argmax, twin counters) and
       at the end (every twin table, the stores, the token log, the committed tokens, the decision counts), on
       randomized event streams: rollbacks only (every cohort rejected at a random row), publications only (every
       cohort accepted), mixed (each cohort rejected with probability 1/2 at a random row; three seeds), SG1's forced
       schedule (block-boundary crossings) and the free-running greedy round; on the ResidentStore (draft drop 0.5),
       b2 and sc (per-stream layout) and b2 on the shared pool layout (xpool: the flat ring rows). Each stream's
       precondition: the kinds of events it is for. AE1c (over the set): a tick with >= 2 rollbacks and a publication in
       one call, a rollback-only tick, a write-back in a rollback stream.
  AE2  THE FLAG REACHES THE TICK: the engine option (default False; construction, clone; the refusal without rebuild
       "batched"; the refusal in apply_events when eng.rebuild is per_call after construction, as make_engine sets it);
       the harness (--apply-batched: parse, refusals; make_engine sets eng.apply_batched after eng.rebuild; the meta
       record; apply_batched_gates and its call in one_point); the default source keeps the unbatched lines.
  AE3  LAUNCHES. (a) On fakes (counting wrappers of the helpers): unbatched, sync_draft_bank and copy_ring run L times
       per rollback; batched, never, and rebuild_banks_unchecked / copy_rows_many run L times per tick with an event.
       (b) The aten ops of apply_events (a TorchDispatchMode; views excluded: each other op is one device launch on the
       GPU): at every tick with rollbacks and a publication, the batched count is lower by at least 13 x L per
       rollback (sync_draft_bank 10 + copy_ring 3 per layer, by design). REPORT: the per-tick counts, the rollback-only
       ticks too (there, one call costs about 26 ops per layer, so one rollback alone costs more than unbatched).
  AE4  THE LINT (r2_capture_lint.py): _apply_batched and _keep_vector (and rebuild_banks_unchecked with rows_out) have
       no host read, no data-dependent shape and no host branch on a device value. A planted read is found.
  AE5  NEGATIVE CONTROL (L34, L49): ONE batched call drops the ring copy of its FIRST rollback only (the call site under
       test: the rollback columns of copy_rows_many; the publications' columns of the same call stay), at the first
       tick where that copy changes rows (precondition, checked on the store before the call); the plant is effective
       (the draft ring rows still differ after the call) and is caught by the per-tick compare, never before the tick
       after the plant.
  AE6  THE CELLS (static, then a DRYRUN on a clean checkout of this ROOT): evalfast2ab = evalfast2f3l's argv with
       --apply-batched and without --eval-bands; evalfast2c3ab = evalfast2c3l's argv with --apply-batched; the
       REGISTERED block ("PREDICTION: TBD by lead", no STAMP); the f3l control cell unchanged.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_applyev.py
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import random
import sys
from pathlib import Path

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


P6 = _shared("test_r2_tick_p6_shared_ae", "test_r2_tick_p6.py")      # _first_diff, TWIN_KEYS, LINT, the tiny world
XP = _shared("test_r2_xpool_shared_ae", "test_r2_xpool.py")          # make_pool_engine
HF = _shared("test_r2_hostfix_shared_ae", "test_r2_hostfix.py")      # _sg5, STEP, HF_ROOT, _clean_head, _dry
TT, LINT = P6.TT, P6.LINT
TK, SC = TT.TK, TT.SC
p3, ref = TT.p3, TT.ref                                               # the fixtures, by name
L = TT.NL_LAYERS
BLOCK = TK.BLOCK
PER_RB_LAYER = 13                     # sync_draft_bank 10 ops + copy_ring 3 copies per rollback and layer (by design)


# =====================================================================================================================
# the worlds and the streams
# =====================================================================================================================
def _engine(p3, kind, batched, cls=None):
    ops = TT.make_ops(p3)
    kw = dict(rebuild="batched", apply_batched=batched)
    if kind == "resident":
        return TT.make_engine(ops, draft_drop=0.5, cls=cls, **kw)
    if kind == "pool":
        return XP.make_pool_engine(ops, kind="b2", **kw)
    return TT.make_res_engine(ops, kind=kind, **kw)


def _sched(stream):
    """reject_at for a stream (a FRESH generator per round, so both rounds draw the same sequence); None: free-running.
    rb: every cohort rejected at a random row (rollbacks only); pub: every cohort accepted (publications only); mixed<s>:
    rejected with probability 1/2 at a random row, seed s; forced: SG1's schedule."""
    if stream == "free":
        return None
    if stream == "forced":
        return TK.forced_schedule(3)[0]
    p, seed = dict(rb=(1.0, 11), pub=(0.0, 12))[stream] if stream in ("rb", "pub") else (0.5, int(stream[5:]))
    g = random.Random(seed)

    def reject_at(b, c, E):
        if g.random() < p:
            return g.randint(c.a, c.b)
        return None
    return reject_at


def _record(eng, toks, stream, tables=False, catch=False):
    """One round with a per-tick record (P6._record's keys: P6._first_diff compares them), the store in the tables,
    the decision counts, and the (tick, rollbacks, publications) of every apply_events with an event."""
    S_list = eng.S_list
    ticks, events = [], []
    reject_at = _sched(stream)

    def teacher(b, p):
        return int(toks[p - S_list[b], b])

    def on_tick(rows, tokens, out):
        ticks.append(dict(rows=[(r.b, r.kind, r.pos) for r in rows], logits=out.logits.clone(), arg=out.argmax.clone(),
                          ctr=torch.stack([torch.stack((t.pos, t.n_comp, t.n_ccis, t.epoch)) for t in eng.twins])))
        if tables:
            ticks[-1]["tab"] = ([getattr(t, k).clone() for t in eng.twins for k in P6.TWIN_KEYS]
                                + [x.clone() for x in eng.store_k + eng.store_v + eng.store_bias])
    orig = eng.apply_events

    def apply_events():
        rb = {b for (k, b, _, _) in eng.events if k == "rb"}
        pub = {b for (k, b, _, _) in eng.events if k == "pub"} - rb
        if rb or pub:
            events.append((eng.stats["ticks"] - 1, len(rb), len(pub)))
        return orig()
    eng.apply_events = apply_events
    rec = dict(ticks=ticks, error=None, eng=eng, events=events)
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                           teacher=None if reject_at is None else teacher, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                             # noqa: BLE001 (catch=True only)
        if not catch:
            raise
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        return rec
    rec["out"] = [list(r.out) for r in rnd.reqs]
    rec["final"] = ([getattr(t, k).clone() for t in eng.twins for k in P6.TWIN_KEYS]
                    + [x.clone() for x in eng.store_k + eng.store_v + eng.store_bias]
                    + [eng.tok_log[:eng.n_tok].clone()])
    rec["stats"] = {k: eng.stats[k] for k in ("ticks", "rows", "decision_reads", "rollbacks", "publications",
                                              "writebacks", "void_ticks")}
    return rec


# =====================================================================================================================
# AE1 state: batched == unbatched on randomized event streams
# =====================================================================================================================
CASES = [("resident", s) for s in ("forced", "rb", "pub", "mixed1", "mixed2", "mixed3")] + \
        [("b2", s) for s in ("forced", "rb", "mixed1", "free")] + [("sc", s) for s in ("forced", "mixed2")] + \
        [("pool", s) for s in ("forced", "rb", "mixed3")]
_PAIRS = {}


def _pair(p3, ref, kind, stream):
    if (kind, stream) not in _PAIRS:
        toks = ref[0]
        _PAIRS[(kind, stream)] = (_record(_engine(p3, kind, False), toks, stream),
                                  _record(_engine(p3, kind, True), toks, stream))
    return _PAIRS[(kind, stream)]


@pytest.mark.parametrize("kind,stream", CASES, ids=["%s-%s" % c for c in CASES])
def test_AE1a_batched_apply_equals_the_unbatched_apply(p3, ref, kind, stream):
    un, ba = _pair(p3, ref, kind, stream)
    assert un["eng"].apply_batched is False and ba["eng"].apply_batched is True
    assert un["eng"].rebuild == ba["eng"].rebuild == "batched"
    d = P6._first_diff(un, ba)
    assert d is None, (kind, stream, d)
    assert un["stats"] == ba["stats"], (un["stats"], ba["stats"])
    assert un["events"] == ba["events"]
    assert not ba["eng"].mirror_failures() and not un["eng"].mirror_failures()
    s = ba["stats"]
    assert s["void_ticks"] == 0 and s["ticks"] > 10, s
    if stream == "rb":                                                 # PRECONDITION: the stream's own event kinds
        assert s["rollbacks"] > 0 and s["publications"] == 0, s
    elif stream == "pub":
        assert s["publications"] > 0 and s["rollbacks"] == 0, s
    elif stream == "free":                                             # greedy: rollbacks only where the draft errs
        assert s["publications"] > 0, s
    else:
        assert s["rollbacks"] > 0 and s["publications"] > 0, s
    assert ba["eng"].apply_rb_items == s["rollbacks"], (ba["eng"].apply_rb_items, s)
    assert ba["eng"].apply_calls == sum(1 for (_, r, _) in ba["events"] if r), ba["eng"].apply_calls
    assert un["eng"].apply_calls == 0 and un["eng"].apply_rb_items == 0
    assert (ba["eng"].batched_calls, ba["eng"].batched_max) == (un["eng"].batched_calls, un["eng"].batched_max)


def test_AE1c_the_streams_reach_every_event_mix(p3, ref):
    """PRECONDITION over the set: a call with >= 2 rollbacks and a publication together, a call with >= 2 rollbacks
    and no publication, a rollback-only call with one rollback, a write-back in a rollback-only stream, a pool-layout
    call with a rollback and a publication."""
    mix = dict(rb2_pub=0, rb2_only=0, rb1_only=0, pool_rb_pub=0)
    for kind, stream in CASES:
        _, ba = _pair(p3, ref, kind, stream)
        for (_, r, p) in ba["events"]:
            mix["rb2_pub"] += r >= 2 and p >= 1
            mix["rb2_only"] += r >= 2 and p == 0
            mix["rb1_only"] += r == 1 and p == 0
            mix["pool_rb_pub"] += kind == "pool" and r >= 1 and p >= 1
    assert all(v > 0 for v in mix.values()), ("precondition: the streams do not reach every mix", mix)
    wb = [_pair(p3, ref, k, "rb")[1]["stats"]["writebacks"] for k in ("resident", "b2", "pool")]
    assert all(w > 0 for w in wb), ("precondition: a write-back in the rollback-only streams", wb)
    print("REPORT AE1 event mix over %d streams: %s; write-backs (rb streams) %s" % (len(CASES), mix, wb))


# =====================================================================================================================
# AE2 the flag reaches the tick
# =====================================================================================================================
def test_AE2a_the_engine_option(p3, ref):
    ops = TT.make_ops(p3)
    e = TT.make_engine(ops)
    assert e.apply_batched is False and e.clone().apply_batched is False          # the default: as before
    f = TT.make_engine(ops, rebuild="batched", apply_batched=True)
    assert f.apply_batched is True and f.clone().apply_batched is True
    assert f.clone(apply_batched=False).apply_batched is False
    with pytest.raises(TK.TickError, match="apply_batched needs rebuild 'batched'"):
        TT.make_engine(ops, apply_batched=True)                                   # rebuild per_call
    with pytest.raises(TK.TickError, match="apply_batched needs rebuild 'batched'"):
        f.clone(rebuild="per_call")
    g = TT.make_engine(ops, rebuild="batched", apply_batched=True)
    g.rebuild = "per_call"                                                        # as make_engine could set it
    rec = _record(g, ref[0], "forced", catch=True)
    assert rec["error"] is not None and "apply_batched needs rebuild 'batched'" in rec["error"], rec["error"]
    assert g.apply_calls == 0
    src = TT._method_src("TickEngine", "apply_events")
    for s in ("SC.sync_draft_bank(self.twins[l], b, B)", "self.copy_ring(l, b, exact_to_draft=True)",
              "SC.rollback_bank(self.twins[l], self.last_ctr[l], keep)", "self._publish_batched(pub)",
              "if self.apply_batched:", "self._apply_batched(rb, pub)", "full = self._keep_vector(keep)"):
        assert s in src, s                                            # the unbatched lines stay; the switch is there


def test_AE2b_the_harness_flag():
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    b2 = base + ["--backends", "b2", "--X", "64"]
    a = G.parse_args(b2 + ["--variants", "fused/batched"])
    assert a.apply_batched is False                                              # the default
    a = G.parse_args(b2 + ["--variants", "fused/batched", "--apply-batched"])
    assert a.apply_batched is True and a.rebuild == "batched"
    for bad in (["--apply-batched"], ["--variants", "fused/batched,torch/per_call", "--apply-batched"],
                ["--variants", "torch/per_call", "--apply-batched"]):
        with pytest.raises(SystemExit):
            G.parse_args(b2 + bad)
    me = inspect.getsource(G.make_engine)
    i = me.index('eng.rebuild = getattr(a, "rebuild", "per_call")')
    j = me.index('eng.apply_batched = bool(getattr(a, "apply_batched", False))')
    assert i < j, "make_engine must set the rebuild first (apply_events checks the pair)"
    src = (TT.EFF / "r2_tick_sg5.py").read_text()
    assert "meta.update(apply_batched=True)" in src
    op = ast.get_source_segment(src, next(n for n in ast.parse(src).body
                                          if isinstance(n, ast.FunctionDef) and n.name == "one_point"))
    assert "gates.extend(apply_batched_gates(a, label, abw))" in op and 'info["apply_batched"] = abw' in op
    on = G.parse_args(b2 + ["--variants", "fused/batched", "--apply-batched"])
    off = G.parse_args(b2 + ["--variants", "fused/batched"])
    good = dict(apply_batched=True, apply_calls=5, apply_rb_max=3, apply_rb_items=9)
    assert G.apply_batched_gates(off, "P", good) == []                           # no gate without the flag
    g = G.apply_batched_gates(on, "P", good)
    assert len(g) == 1 and g[0]["name"] == "P_AB_acted" and g[0]["must"] and g[0]["passed"], g
    for w in (dict(good, apply_rb_max=1), dict(good, apply_calls=0), dict(good, apply_batched=False), {}):
        g = G.apply_batched_gates(on, "P", w)
        assert len(g) == 1 and g[0]["must"] and not g[0]["passed"], (w, g)    # NEGATIVE: a witness that did not act


# =====================================================================================================================
# AE3 launches
# =====================================================================================================================
def _counted(eng, monkeypatch, box):
    """Counting wrappers (no change of a value): SC.sync_draft_bank, TK.rebuild_banks_unchecked (module globals, read
    at call time) and the engine's copy_ring / copy_rows_many."""
    for mod, name in ((SC, "sync_draft_bank"), (TK, "rebuild_banks_unchecked")):
        real = getattr(mod, name)

        def wrap(*a, _r=real, _n=name, **k):
            box[_n] = box.get(_n, 0) + 1
            return _r(*a, **k)
        monkeypatch.setattr(mod, name, wrap)
    for name in ("copy_ring", "copy_rows_many"):
        real = getattr(eng, name)

        def wrap(*a, _r=real, _n=name, **k):
            box[_n] = box.get(_n, 0) + 1
            return _r(*a, **k)
        setattr(eng, name, wrap)


@pytest.mark.parametrize("kind,stream", [("resident", "mixed1"), ("pool", "mixed3")])
def test_AE3a_the_helper_calls_on_fakes(p3, ref, monkeypatch, kind, stream):
    got = {}
    for batched in (False, True):
        box = {}
        eng = _engine(p3, kind, batched)
        _counted(eng, monkeypatch, box)
        rec = _record(eng, ref[0], stream)
        monkeypatch.undo()
        got[batched] = (box, rec)
    (bu, ru), (bb, rb) = got[False], got[True]
    n_rb = ru["stats"]["rollbacks"]
    t_pub = sum(1 for (_, _, p) in ru["events"] if p)
    t_rb = sum(1 for (_, r, _) in ru["events"] if r)
    t_any = len(ru["events"])
    assert ru["events"] == rb["events"] and ru["stats"] == rb["stats"]
    assert n_rb > 0 and t_pub > 0, "precondition: rollbacks and publications"
    assert any(r >= 2 or (r >= 1 and p >= 1) for (_, r, p) in ru["events"]), \
        "precondition: a call with two rollbacks, or a rollback and a publication (else the counts tie)"
    assert bu.get("sync_draft_bank", 0) == bu.get("copy_ring", 0) == L * n_rb, (bu, n_rb)
    assert bu.get("rebuild_banks_unchecked", 0) == L * t_pub and bu.get("copy_rows_many", 0) <= L * t_pub, (bu, t_pub)
    assert bb.get("sync_draft_bank", 0) == 0 and bb.get("copy_ring", 0) == 0, bb
    assert bb.get("rebuild_banks_unchecked", 0) == L * t_any, (bb, t_any)
    assert L * t_rb <= bb.get("copy_rows_many", 0) <= L * t_any, (bb, t_rb, t_any)   # a rollback always has rows
    calls_u = sum(bu.values())
    calls_b = sum(bb.values())
    assert calls_b < calls_u, (calls_b, calls_u)
    print("REPORT AE3a %s/%s: helper calls per round unbatched %s (%d), batched %s (%d); %d rollbacks, %d ticks with an "
          "event, L %d" % (kind, stream, bu, calls_u, bb, calls_b, n_rb, t_any, L))


def _is_view(func):
    return any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns)


def _op_counter():
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except ImportError:                                                # pragma: no cover
        pytest.skip("torch has no TorchDispatchMode")

    class OpCount(TorchDispatchMode):
        """Every aten op under the mode, views apart (a view launches nothing)."""

        def __init__(self):
            super().__init__()
            self.ops = 0
            self.views = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if _is_view(func):
                self.views += 1
            else:
                self.ops += 1
            return func(*args, **(kwargs or {}))
    return OpCount


def _count_ops(eng, OpCount):
    """Wrap the engine's apply_events: (tick, rollbacks, publications, ops) of every call with an event."""
    per = []
    orig = eng.apply_events

    def apply_events():
        rb = {b for (k, b, _, _) in eng.events if k == "rb"}
        pub = {b for (k, b, _, _) in eng.events if k == "pub"} - rb
        m = OpCount()
        with m:
            out = orig()
        if rb or pub:
            per.append((eng.stats["ticks"] - 1, len(rb), len(pub), m.ops))
        return out
    eng.apply_events = apply_events
    return per


@pytest.mark.parametrize("kind,stream", [("resident", "mixed2"), ("b2", "mixed1"), ("pool", "forced")])
def test_AE3b_the_ops_of_apply_events_fall(p3, ref, kind, stream):
    OpCount = _op_counter()
    got = {}
    for batched in (False, True):
        eng = _engine(p3, kind, batched)
        per = _count_ops(eng, OpCount)
        _record(eng, ref[0], stream)
        got[batched] = per
    un, ba = got[False], got[True]
    assert [x[:3] for x in un] == [x[:3] for x in ba], "the two rounds made other decisions"
    mixed = [(u, b) for u, b in zip(un, ba) if u[1] >= 1 and u[2] >= 1]
    assert mixed, "precondition: a tick with rollbacks and a publication"
    for u, b in mixed:
        assert u[3] - b[3] >= PER_RB_LAYER * L * u[1], ("the batched apply did not save 13 ops per rollback and "
                                                        "layer", u, b)
    rb_only = [(u[1], u[3], b[3]) for u, b in zip(un, ba) if u[1] >= 1 and u[2] == 0]
    tot_u, tot_b = sum(x[3] for x in un), sum(x[3] for x in ba)
    sav = [(u[1], u[3] - b[3]) for u, b in mixed]
    print("REPORT AE3b %s/%s (L %d): ops in apply_events per round unbatched %d, batched %d; ticks with rollbacks and "
          "publications (rollbacks, saved ops) %s; rollback-only ticks (rollbacks, unbatched, batched) %s"
          % (kind, stream, L, tot_u, tot_b, sav, rb_only))


# =====================================================================================================================
# AE4 the lint
# =====================================================================================================================
AB_FNS = [("TickEngine", "_apply_batched"), ("TickEngine", "_keep_vector"), (None, "rebuild_banks_unchecked")]


@pytest.mark.parametrize("cls,name", AB_FNS, ids=[n for _, n in AB_FNS])
def test_AE4a_the_batched_apply_has_no_host_read(cls, name):
    found = LINT.lint_function(P6._fn(cls, name))
    # documented (P6e's): transient allocations; the host check of the host list of items; Python ints of host values
    documented = [("ALLOC", r"."), ("BRANCH", r"^len\(set\(bs\)\) != len\(bs\)$"), ("SYNC", r"^int\((b|p|d|dpos)\)$")]
    bad = [f for f in LINT.unexplained(found, documented) if f[0] in ("SYNC", "SHAPE", "BRANCH", "SCALARIDX")]
    assert bad == [], (name, bad)
    assert not [f for f in found if f[0] in ("SHAPE", "SCALARIDX")], (name, found)


def test_AE4b_a_planted_read_is_found():
    fn = P6._fn("TickEngine", "_apply_batched")
    fn.body.insert(3, ast.parse("x = int(self.twins[0].pos.max())").body[0])     # NEGATIVE CONTROL: a host read
    assert "SYNC" in {c for c, _, _ in LINT.lint_function(fn)}
    fn = P6._fn("TickEngine", "_keep_vector")
    fn.body.insert(0, ast.parse("if self.twins[0].pos.any(): pass").body[0])     # a host branch on a device value
    assert "BRANCH" in {c for c, _, _ in LINT.lint_function(fn)}


# =====================================================================================================================
# AE5 the negative control
# =====================================================================================================================
class DropRbCopy(TK.TickEngine):
    """NEGATIVE CONTROL (L34, L49): in ONE _apply_batched call, copy_rows_many drops the columns of the call's FIRST
    rollback (its 128 ring rows); the publications' columns of the same call stay (the call site under test only). The
    call is the first one with a rollback whose ring copy changes rows (the draft ring rows differ from the exact ring
    rows before the call, in some layer: precondition). plant: dict(tick, b, n_rb, n_pub, dropped, kept, total,
    effective) after the call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.plant = None
        self._drop = None

    def _rows_equal(self, l, cp):
        sts = (self.store_k[l], self.store_v[l], self.store_bias[l])
        if self.xpool is not None:
            return all(torch.equal(st[cp[2]], st[cp[1]]) for st in sts)
        return all(torch.equal(st[cp[0], cp[2]], st[cp[0], cp[1]]) for st in sts)

    def _apply_batched(self, rb, pub):
        if self.plant is None and rb:
            b0 = next(iter(rb))
            ring = self.rows_to_copy([(b0, range(2 * BLOCK))])
            if not all(self._rows_equal(l, ring) for l in range(self.L)):
                self._drop = dict(tick=self.stats["ticks"] - 1, b=b0, n_rb=len(rb), n_pub=len(pub), dropped=[],
                                  kept=[], total=[])
                super()._apply_batched(rb, pub)
                d, self._drop = self._drop, None
                d["effective"] = not all(self._rows_equal(l, ring) for l in range(self.L))
                self.plant = d
                return
        super()._apply_batched(rb, pub)

    def copy_rows_many(self, l, cp):
        if self._drop is not None:
            keep = cp[0] != self._drop["b"]
            self._drop["dropped"].append(int((~keep).sum()))
            self._drop["kept"].append(int(keep.sum()))
            self._drop["total"].append(int(cp.shape[1]))
            cp = cp[:, keep]
            if cp.shape[1] == 0:
                return
        super().copy_rows_many(l, cp)


def test_AE5_a_dropped_rollback_copy_is_caught(p3, ref):
    toks = ref[0]
    clean = _record(_engine(p3, "resident", True), toks, "mixed1", tables=True)
    un, ba = _pair(p3, ref, "resident", "mixed1")
    assert P6._first_diff(ba, clean) is None                           # the tables do not change the round
    bad_eng = _engine(p3, "resident", True, cls=DropRbCopy)
    bad = _record(bad_eng, toks, "mixed1", tables=True, catch=True)
    pl = bad_eng.plant
    print("AE5 control: plant %s" % (pl,))
    assert pl is not None, "precondition: no rollback whose ring copy changes rows"
    assert pl["dropped"] == [2 * BLOCK] * L, ("precondition: the plant dropped the rollback's 128 rows per layer", pl)
    assert all(k == t - 2 * BLOCK for k, t in zip(pl["kept"], pl["total"])), ("L49: only the rollback's columns", pl)
    assert pl["effective"], ("precondition (L34): the dropped copy left the rows as the copy would", pl)
    fd = P6._first_diff(clean, bad)
    assert fd is not None and fd[0] >= pl["tick"] + 1, (fd, pl)        # caught, never before the plant's effect


# =====================================================================================================================
# AE6 the cells
# =====================================================================================================================
AB_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-applyev"


def test_AE6a_the_cells_and_the_registration_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + AB_ROOT
    assert HF.HF_ROOT == AB_ROOT
    assert "        evalfast2ab|evalfast2c3ab) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "|evalfast2f3p|evalfast2c3l|evalfast2f3l|evalfast2ab|evalfast2c3ab] [gilfree|gather]" in text
    reg = text[text.index("# REGISTERED for evalfast2ab / evalfast2c3ab"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("[applyev], SOURCE ONLY, not run", "nosi-r2-applyev", "PROFILE_f3l_mixed.md", "60.5 ms", "416",
                "_publish_batched", "EXACTNESS (a)-(c)", "AB-C3", "0 diffs", "_AB_acted", "VOID", "AB-ACT",
                "evalfast2f3l     the same-job control", "AB-MED", "AB-MEAN", "AB-EFF", "AB-FALS", "AB-MEM"):
        assert tag in reg, tag
    assert "STAMP 2026-10-24T04:59:23Z" in reg and "TBD" not in reg    # the lead stamped it before any run
    assert "425:475:305:345:248:280" in reg
    i = text.index("    evalfast2ab) MODES=fast")
    blk = text[i:text.index("\nesac", i)]
    assert "ABFLAGS=(--res-lean --res-dev --apply-batched)" in blk and "\"${ABFLAGS[@]}\"" in blk
    assert "--eval-bands 425:475:305:345:248:280 " in blk.split("evalfast2c3ab)")[0]   # evalfast2ab: the lead's bands
    assert "--b2-resolve check --fused-kernel v3 --res-lean --res-dev --apply-batched" in blk
    f3l = text[text.index("    evalfast2f3p|evalfast2f3l) MODES=fast"):text.index("    evalfast2c3l) MODES=fast")]
    assert "FLFLAGS=(--res-lean --res-dev) ; BANDS=335:375:258:292:220:248" in f3l and "--apply-batched" not in f3l
    assert "STAMP 2026-10-23T17:13:30Z" in text                         # the flean block stays stamped


def _argv(tmp_path, cell):
    tmp_path.mkdir(parents=True, exist_ok=True)                        # HF._dry writes the holder's uuid file here
    r = HF._dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(HF.HF_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the applyev ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    return line, got, [t.replace(cell, "CELL") for t in got.split()]


def _drop_bands(toks):
    i = toks.index("--eval-bands")
    return toks[:i] + toks[i + 2:]


@pytest.mark.parametrize("cell,base", [("evalfast2ab", "evalfast2f3l"), ("evalfast2c3ab", "evalfast2c3l")])
def test_AE6b_the_cells_dryrun(tmp_path, cell, base):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    line, got, toks = _argv(tmp_path / "a", cell)
    _, _, btoks = _argv(tmp_path / "b", base)
    want = (_drop_bands(btoks) if cell == "evalfast2ab" else btoks) + ["--apply-batched"]
    toks = _drop_bands(toks) if cell == "evalfast2ab" else toks        # the stamped AB bands replace f3l's
    assert toks == want, (cell, [t for t in toks if t not in want], [t for t in want if t not in toks])
    for s in ("PYTHONPATH=%s/nosi" % AB_ROOT, "--job-name=sg5-t1-418-%s " % cell, "--time=120", "--mem=240G"):
        assert s in line, (cell, s)
    pa = HF._sg5().parse_args(got.split())
    assert pa.apply_batched and pa.rebuild == "batched" and pa.res_lean and pa.res_dev and pa.fused_kernel == "v3"
    if cell == "evalfast2ab":
        assert pa.eval_role == "fast-time" and pa.bare and pa.b2_resolve == "fused" and pa.eval_bands == "425:475:305:345:248:280"
    else:
        assert pa.eval_role == "" and pa.b2_resolve == "check" and pa.steps == 8 and pa.draft_ticks == 4
