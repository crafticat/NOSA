"""[admitfill] CPU tests of --inflight-array (OwnResidency(inflight_array=True); nosi/r2_own2.py, the module docstring's
[inflight-array] part; analysis/evalfast/DIAG_effective_gap.md section 4, fix 2). The worktree is r2-admitfill. Fix 1
(the first fill at admission) is not exact on this path (the first resident set is the tick-0 selection, which needs
the tick-0 forward pass), so this branch builds fix 2. The old poll (flag off) is the REFERENCE. Both arms of every
comparison use the fixed arrival rule (arrival_lag 2, the eval cells' rule).

  IA1  ROUND LEVEL, per-transfer transport (K3X's adversarial transport: seeded delays, reordering, bursts): the B2
       round with the array poll equals the old poll at every tick and every layer (test_r2_resfix.py's RL2 comparison:
       rows, tokens, logits, pages, state with the transport's pending list, counters, readers, memory, the kept
       counters, the release queue, every layer's record and plan), and at the end (outputs, journal, report with the
       arrival counts forced / held, the engine stats); RL2's draws (per-stream and pool layout, default and pressure,
       forced and free, checked and unchecked), with the plain flags and with the eval cells' flags (--lean-time
       --wait-index --b2-fewer-reads --res-lean). IA1b: PRECONDITION (L34): the draws force and hold transfers.
       IA1c: own2's B2I5 generator (_lag_pair: F6's schedule, two transports seeded apart): the array poll keeps the two
       runs equal, and each run equals the old poll's run of its seed.
  IA2  ROUND LEVEL, one completion event per issue (the gilfree / gather form, a CPU transport here): the same
       comparison; the array poll asks one done per issue (never per transfer); PRECONDITION (IA2b): forced and held
       issues occur.
  IA3  THE FLAG OFF: the array poll never runs, the issue table stays empty; the old lines stay in _poll_arrivals.
       Refusals and wiring: inflight_array needs arrival_lag; --inflight-array needs --backends b2 and --arrival-lag >= 1;
       make_engine passes it; the free arm turns it off; the transports' one_event_per_issue.
  IA4  NEGATIVE CONTROLS at the site under test (L49: each plant acts in the array poll only), each seen at the tick of
       the plant: a due issue that leaves out one transfer (the state differs; the coverage check refuses the next poll),
       an issue due one tick early (the state differs), a wrong expected generation in the vectorized check (the poll
       refuses). IA4b: with the flag off, the same plants change nothing.
  IA5  InflightByLayer.pop_layer_many equals pop per key on random maps (the map, its order, the index);
       NEGATIVE CONTROL: a bulk pop that forgets the index is seen by index_ok.
  IA6  STATIC: the array poll holds no sort of the in-flight map and no per-transfer arrival call.
  IA7  THE STEP CELL evalfast2ia of r2_tick_sg5_step.sh: ROOT, usage, the REGISTERED block (every tag; the bands = the
       cell's --eval-bands), and the DRYRUN on a clean checkout of the ROOT parsed into the flags.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_admitfill.py
"""
from __future__ import annotations

import importlib.util
import inspect
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


RF = _shared("test_r2_resfix_shared_rd", "test_r2_resfix.py")         # RL2's rounds and draws (resdev's name: shared)
HF, TT, TK, OWN2, RSC, XPT = RF.HF, RF.TT, RF.TK, RF.OWN2, RF.RSC, RF.XPT
p3, ref = RF.p3, RF.ref                                               # the fixtures, by name
LAG = 2                                                               # the eval cells' --arrival-lag
FLAGS = dict(plain={}, eval=dict(lean=True, wait_index=True, fewer_reads=True, res_lean=True))
IA_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3ldx"                         # [f3ldx] (was nosi-r2-f3ld / -f3lia / -admitfill)
IA_BANDS = "328:362:305:338:245:282"


# =====================================================================================================================
# a CPU transport with ONE completion event per issue (the gilfree / gather form)
# =====================================================================================================================
def _ev_done(ev):
    """kstep.backend.ev_done: None is complete; else query() or .done."""
    if ev is None:
        return True
    q = getattr(ev, "query", None)
    return bool(q()) if callable(q) else bool(getattr(ev, "done"))


class _GroupEv:
    """The completion event of one issue: every transfer of the issue holds it from the issue on."""

    def __init__(self):
        self.done = False

    def query(self):
        return self.done


class _GroupMixin:
    """K3X's adversarial transport (the same payloads and seeded release ticks) with ONE completion event per issue
    (one_event_per_issue): an issue lands as a whole at the first poll where every transfer of it is past its release
    tick and its last readers are done, or when any transfer of it is forced. Its poll draws no random number."""
    one_event_per_issue = True

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.groups = []
        self.group_of = {}
        self.group_forces = 0

    def issue(self, layer, transfers):
        super().issue(layer, transfers)
        ev, grp = _GroupEv(), list(transfers)
        for tr in grp:
            tr.ev = ev
            self.group_of[tr.seq] = (ev, grp)
        self.groups.append((ev, grp))

    def _land_group(self, ev, grp):
        for tr in grp:
            if not tr.delivered:
                if tr.reader_ev is not None:
                    self.be.host_wait(tr.reader_ev)
                self._land(tr)
                self.pending.remove(tr)
        ev.done = True

    def poll(self):
        self.now += 1
        keep = []
        for ev, grp in self.groups:
            if ev.done:
                continue
            if all(tr.release_tick <= self.now and _ev_done(tr.reader_ev) for tr in grp):
                self._land_group(ev, grp)
            else:
                keep.append((ev, grp))
        self.groups = keep

    def done(self, tr):
        return tr.ev is not None and tr.ev.done

    def force(self, tr):
        if tr.ev is None:
            raise RuntimeError("group transport: transfer seq %d was never issued" % tr.seq)
        if not tr.ev.done:
            self.group_forces += 1
            self._land_group(*self.group_of[tr.seq])


class _GroupAdv(_GroupMixin, TT.KTR.AdversarialTransport):
    pass


class _GroupFlat(_GroupMixin, XPT.FlatAdversarial):
    pass


# =====================================================================================================================
# the rounds
# =====================================================================================================================
def _eng(p3, layout, cfg, seed, ia, checked=False, flags="plain", extra=None, group=None, lag=LAG):
    """RL2's engine (test_r2_resfix._engine) with the fixed arrival rule, the flag under test and the named flags;
    group = max_delay: the transport becomes the one-event-per-issue form (seeded as the adversarial one)."""
    kw = dict(FLAGS[flags])
    if lag is not None:
        kw["arrival_lag"] = lag
    kw["inflight_array"] = ia
    kw.update(extra or {})
    eng = RF._engine(p3, layout, cfg, seed, bool(kw.pop("res_lean", False)), checked, svc_kw=kw)
    if group is not None:
        w = eng.res_world
        old = w.tp
        cls = _GroupFlat if layout == "pool" else _GroupAdv
        new = cls(old.be, old.host_k, old.host_v, old.slots_k, old.slots_v, old.bs, seed, mode=old.mode,
                  max_delay=int(group))
        if layout == "pool":
            new.xpool = old.xpool
        w.svc.tp = new
        w.tp = new
    return eng


def _pair(p3, toks, layout, cfg, forced, seed, checked, flags, group=None):
    base = RF._round(_eng(p3, layout, cfg, seed, False, checked, flags, group=group), toks, forced)
    arr = RF._round(_eng(p3, layout, cfg, seed, True, checked, flags, group=group), toks, forced)
    return base, arr


def _assert_equal(base, arr):
    assert base.svc.inflight_array is False and arr.svc.inflight_array is True
    assert base.err is None and len(base.ticks) > 10, ("precondition: the reference round ran", base.err)
    st = arr.svc.ia_stats
    assert st["parts_due"] > 0 and st["admitted"] > 0, ("precondition: the array poll admitted issues", st)
    d = RF._rounds_diff(base, arr)
    assert d is None, ("--inflight-array differs from the old poll", d, arr.err)
    HF._same_end(base, arr)                                           # report: arrival forced / held included
    assert (base.svc.arrival_forced, base.svc.arrival_held) == (arr.svc.arrival_forced, arr.svc.arrival_held)
    assert list(dict.keys(base.svc.inflight)) == list(dict.keys(arr.svc.inflight))
    assert sorted(base.svc.pending) == sorted(arr.svc.pending) and base.svc._landed == arr.svc._landed
    if hasattr(arr.svc.inflight, "index_ok"):
        assert arr.svc.inflight.index_ok()
    left = sum(len(p.trs) for ps in arr.svc._ia_groups.values() for p in ps)
    assert left >= len(arr.svc.inflight) - len(arr.svc._landed)      # the parts not due yet hold the rest


IA1_CASES = [c + ("plain",) for c in RF.RL2_CASES] + [c + ("eval",) for c in RF.RL2_CASES[::2]]


@pytest.mark.parametrize("layout,cfg,forced,seed,checked,flags", IA1_CASES,
                         ids=["-".join(str(x) for x in c) for c in IA1_CASES])
def test_IA1_the_array_poll_equals_the_old_poll_per_transfer_transport(p3, ref, layout, cfg, forced, seed, checked,
                                                                       flags):
    toks, _ = ref
    base, arr = _pair(p3, toks, layout, cfg, forced, seed, checked, flags)
    _assert_equal(base, arr)
    st = arr.svc.ia_stats
    assert st["transfer_checks"] > 0 and st["group_checks"] == 0, st  # this transport: per transfer, old order
    print("REPORT IA1 %s: ticks %d, forced %d, held %d, %s" % ("-".join(str(x) for x in (layout, cfg, forced, seed,
                                                                                         checked, flags)),
                                                              len(base.ticks), base.svc.arrival_forced,
                                                              base.svc.arrival_held, st))


def test_IA1b_precondition_the_draws_force_and_hold(p3, ref):
    """L34: the comparison of IA1 covers the force branch and the held branch (the old poll's counts, two draws)."""
    toks, _ = ref
    runs = [RF._round(_eng(p3, lay, cfg, 1, False), toks, True) for (lay, cfg) in (("stream", "pressure"),
                                                                                  ("pool", "default"))]
    assert all(r.err is None for r in runs), [r.err for r in runs]
    assert sum(r.svc.arrival_forced for r in runs) > 0, "precondition: no due transfer was forced"
    assert sum(r.svc.arrival_held for r in runs) > 0, "precondition: no transfer landed before it was due"


TRR = _shared("test_r2_residency_shared_b2", "test_r2_residency.py")  # own2's generators (B2I5's _lag_pair; shared)
OWN2R = TRR._load("r2_own2")                                          # the r2res package's B2 (its own classes)


class _IAOwnR(OWN2R.OwnResidency):
    """B2 with the array poll on (the r2res package)."""

    def __init__(self, *a, **k):
        k.setdefault("inflight_array", True)
        super().__init__(*a, **k)


def test_IA1c_the_own2_lag_pair_with_the_array_poll():
    """own2's B2I5 generator (F6's schedule, two transports seeded apart, lag 2): the array poll keeps the two runs
    equal (it depends on the schedule only), and each of its runs equals the old poll's run of the same seed (state with
    the transport's list, counters, arrival counts)."""
    fs, ft, ws = TRR._lag_pair(_IAOwnR, 2)
    assert ft is not None, "precondition: the two transports landed apart"
    assert fs is None, ("the array poll depends on the transport's timing", fs)
    _, _, ws0 = TRR._lag_pair(OWN2R.OwnResidency, 2)
    for w0, w1 in zip(ws0, ws):
        assert w0.svc.inflight_array is False and w1.svc.inflight_array is True
        assert w1.svc.ia_stats["admitted"] > 0, "precondition: the array poll admitted"
        assert TRR._state(w0) == TRR._state(w1)
        assert TRR._counters(w0.svc) == TRR._counters(w1.svc)
        assert (w0.svc.arrival_forced, w0.svc.arrival_held) == (w1.svc.arrival_forced, w1.svc.arrival_held)
        assert dict(w0.svc.journal_counts) == dict(w1.svc.journal_counts)
    assert sum(w.svc.arrival_forced for w in ws) > 0 and sum(w.svc.arrival_held for w in ws) > 0   # B2I5's precondition


IA2_CASES = [("stream", "default", True, 1, False, "plain", 3), ("stream", "pressure", True, 2, False, "eval", 1),
             ("stream", "pressure", False, 3, True, "plain", 3), ("pool", "default", True, 1, False, "eval", 1),
             ("pool", "pressure", True, 2, True, "plain", 1), ("pool", "pressure", False, 3, False, "eval", 3)]


@pytest.mark.parametrize("layout,cfg,forced,seed,checked,flags,md", IA2_CASES,
                         ids=["-".join(str(x) for x in c) for c in IA2_CASES])
def test_IA2_the_array_poll_equals_the_old_poll_one_event_per_issue(p3, ref, layout, cfg, forced, seed, checked, flags,
                                                                    md):
    toks, _ = ref
    base, arr = _pair(p3, toks, layout, cfg, forced, seed, checked, flags, group=md)
    assert isinstance(arr.svc.tp, _GroupMixin) and isinstance(base.svc.tp, _GroupMixin)
    _assert_equal(base, arr)
    st = arr.svc.ia_stats
    assert st["group_checks"] > 0 and st["transfer_checks"] == 0, st  # one done per issue, never per transfer
    assert base.svc.tp.group_forces == arr.svc.tp.group_forces


def test_IA2b_precondition_issues_are_forced_and_held(p3, ref):
    """L34: with one event per issue, some issues are forced at their due tick (max_delay 3) and some land before it
    (max_delay 1); the array poll counts them as the old poll does (IA2's equality)."""
    toks, _ = ref
    slow = RF._round(_eng(p3, "stream", "pressure", 2, True, group=3), toks, True)
    fast = RF._round(_eng(p3, "stream", "pressure", 2, True, group=1), toks, True)
    assert slow.err is None and fast.err is None, (slow.err, fast.err)
    assert slow.svc.arrival_forced + fast.svc.arrival_forced > 0, "precondition: no issue was forced"
    assert slow.svc.arrival_held + fast.svc.arrival_held > 0, "precondition: no issue landed before it was due"
    assert slow.svc.tp.group_forces + fast.svc.tp.group_forces > 0


# =====================================================================================================================
# IA3 the flag off; refusals and wiring
# =====================================================================================================================
@pytest.mark.parametrize("lag", [LAG, None], ids=["lag2", "free"])
def test_IA3a_flag_off_never_runs_the_array_poll(p3, ref, monkeypatch, lag):
    toks, _ = ref

    def boom(self):
        raise AssertionError("the array poll ran with the flag off")
    monkeypatch.setattr(OWN2.OwnResidency, "_poll_arrivals_array", boom)
    r = RF._round(_eng(p3, "stream", "pressure", 1, False, lag=lag), toks, True)
    assert r.err is None and len(r.ticks) > 10, r.err
    assert r.svc.inflight_array is False and r.svc._ia_groups == {} and r.svc.ia_stats["polls"] == 0
    src = inspect.getsource(OWN2.OwnResidency._poll_arrivals)
    body = src.split('"""', 2)[2]
    first = [ln.strip() for ln in body.splitlines() if ln.strip()][:2]
    assert first[0].startswith("if self.inflight_array:") and first[1] == "return self._poll_arrivals_array()", first
    for old in ("order = sorted(self.inflight.values(), key=lambda tr: (tr.layer, tr.b, tr.h, tr.seq))",
                "if tr.seq not in self._landed and not self._arrived(tr):", "for tr in arrived:"):
        assert old in body, old                                       # the old lines are still the off path


def test_IA3b_the_flag_refusals_and_its_wiring(p3):
    ops = TT.make_ops(p3)
    with pytest.raises(ValueError, match="arrival_lag"):              # the array poll needs the fixed rule
        TT.make_res_engine(ops, kind="b2", svc_kw=dict(inflight_array=True), **RF.CFG["default"])
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["inflight_array"].default is False
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).inflight_array is False
    on = G.parse_args(base + ["--backends", "b2", "--X", "64", "--arrival-lag", "2", "--inflight-array"])
    assert on.inflight_array and on.arrival_lag == 2
    for bad in (["--backends", "b2", "--X", "64", "--inflight-array"],                       # no fixed rule
                ["--backends", "sc", "--X", "64", "--arrival-lag", "2", "--inflight-array"]):  # not b2
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    assert 'inflight_array=bool(getattr(a, "inflight_array", False))' in inspect.getsource(G.make_engine)
    main = inspect.getsource(G.main)
    assert "meta.update(inflight_array=True)" in main
    assert "af.inflight_array = False" in main                        # the free arm (free timing) runs the old poll
    assert G.GatherTransport.one_event_per_issue is True
    GF = TT._load("r2_transport_gilfree")
    assert GF.GilfreeTransport.one_event_per_issue is True
    assert RSC.PublishingTransport(SimpleNamespace(one_event_per_issue=True)).one_event_per_issue is True
    assert RSC.PublishingTransport(SimpleNamespace()).one_event_per_issue is False
    assert getattr(TT.KTR.AdversarialTransport, "one_event_per_issue", False) is False   # per transfer: the fallback


# =====================================================================================================================
# IA4 negative controls at the site under test
# =====================================================================================================================
@pytest.mark.parametrize("kind", ["ia_skip_one", "ia_due_early", "ia_bad_gen"])
def test_IA4_negative_control_is_seen_at_its_tick(p3, ref, kind):
    toks, _ = ref
    clean = RF._round(_eng(p3, "stream", "default", 1, True), toks, True)
    planted = RF._round(_eng(p3, "stream", "default", 1, True, extra=dict(b2_mut={kind: True})), toks, True)
    assert clean.err is None and len(clean.ticks) > 10, clean.err
    done = planted.svc.ia_mut_done
    assert done is not None and done[1] == kind, ("precondition: the plant acted", kind, done)
    d = RF._rounds_diff(clean, planted)
    assert d is not None, (kind, "not seen")
    assert d[0] == done[0], (kind, "seen at another tick", d, done)
    if kind == "ia_bad_gen":                                          # the vectorized check refuses the poll
        assert planted.err is not None and "inflight-array: the vectorized arrival check" in planted.err, planted.err
        assert len(planted.ticks) == done[0]
    if kind == "ia_skip_one":                                         # the coverage check refuses the next poll
        assert planted.err is not None and "inflight-array" in planted.err and "outside the take path" in planted.err, \
            planted.err


@pytest.mark.parametrize("kind", ["ia_skip_one", "ia_due_early", "ia_bad_gen"])
def test_IA4b_the_plants_act_only_in_the_array_poll(p3, ref, kind):
    """L49: each plant sits in the site under test; with the flag off the same b2_mut changes nothing."""
    toks, _ = ref
    off = RF._round(_eng(p3, "stream", "default", 1, False), toks, True)
    off_p = RF._round(_eng(p3, "stream", "default", 1, False, extra=dict(b2_mut={kind: True})), toks, True)
    assert off_p.svc.ia_mut_done is None
    assert RF._rounds_diff(off, off_p) is None
    HF._same_end(off, off_p)


# =====================================================================================================================
# IA5 InflightByLayer.pop_layer_many
# =====================================================================================================================
class _IndexForgotten(RSC.InflightByLayer):
    """NEGATIVE CONTROL: a bulk pop that removes the keys from the map only (a stale index)."""

    def pop_layer_many(self, l, ks):
        for k in ks:
            dict.pop(self, k)


@pytest.mark.parametrize("cls,ok", [(RSC.InflightByLayer, True), (_IndexForgotten, False)], ids=["real", "control"])
def test_IA5_pop_layer_many_equals_pop_per_key(cls, ok):
    g = random.Random(5)
    acted = 0
    for _ in range(80):
        items = [(s, SimpleNamespace(layer=g.randrange(4))) for s in range(g.randrange(1, 60))]
        a, b = RSC.InflightByLayer(items), cls(items)
        l = g.randrange(4)
        ks = [s for s, t in items if t.layer == l and g.random() < 0.6]
        for k in ks:
            a.pop(k)
        b.pop_layer_many(l, ks)
        assert list(dict.items(a)) == list(dict.items(b))            # the map and its order
        if ks:
            acted += 1
            if ok:
                assert b.index_ok() and a._by == b._by
            else:
                assert not b.index_ok()                               # the control is seen
        assert a.index_ok()
    assert acted >= 20, "precondition: the draws popped keys"


# =====================================================================================================================
# IA6 static: no sort, no per-transfer arrival call
# =====================================================================================================================
def test_IA6_the_array_poll_has_no_sort_and_no_per_transfer_rule():
    src = inspect.getsource(OWN2.OwnResidency._poll_arrivals_array).split('"""', 2)[2]
    for bad in ("sorted(self.inflight", "self.inflight.values()", "self._arrived(", "lambda tr"):
        assert bad not in src, bad
    assert "one_event_per_issue" in src and "_ia_force" in src and "pop_layer_many" in \
        inspect.getsource(OWN2.OwnResidency._ia_forget)


# =====================================================================================================================
# IA7 the step cell evalfast2ia
# =====================================================================================================================
def test_IA7a_the_cell_and_the_registration_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    root = next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0][len("NOSI_EVAL="):]
    assert root == IA_ROOT == HF.HF_ROOT
    assert "        evalfast2ia) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "|evalfast2f3l|evalfast2ia|" in text                      # [f3lia] the flean cells before, f3lia's after
    reg = text[text.index("# REGISTERED for evalfast2ia"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T18:17:11Z", "before any run", "IA-POLL2", "point 380 ms", "PASS <= 550 ms", "IA-POLL ", "IA-MED",
                "band [328, 362]", "IA-MEAN", "band [305, 338]", "IA-EFF", "point 264, band [245, 282]", "IA-SAME",
                "IA-T02", "NOT expected", "IA-FALS", "> 1000 ms", "IA-MEM", "peak_gb <= 84", "nosi-r2-admitfill",
                "4.31 us", "678,000"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2ia) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    for w in ("--eval-bands %s" % IA_BANDS, "--arrival-lag 2", "--b2-resolve torch", "--host-profile --gc-freeze "
              "--lean-time --wait-index", "--b2-fewer-reads --res-lean --inflight-array", "BATCH=216"):
        assert w in blk, w
    assert "--res-dev" not in blk and "STEP_MIN" not in blk          # evalfast2rl + ONE flag; STEP_MIN 120
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def test_IA7b_the_cell_dryrun(tmp_path):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = HF._dry(tmp_path, "evalfast2ia")
    if r.returncode != 0:
        assert Path(IA_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the admitfill ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    for w in ("--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2", "--bare",
              "--eval-role fast-time", "--mem-cap-gb 84", "--eval-bands " + IA_BANDS, "--steps 48",
              "PYTHONPATH=%s/nosi" % IA_ROOT, "--time=120", "--job-name=sg5-t1-418-evalfast2ia ",
              "/r2-runs/sg5/sg5_t1_418_evalfast2ia "):
        assert w in got or w in line, w
    pa = HF._sg5().parse_args(got.split())
    assert pa.b2_resolve == "torch" and pa.host_profile and pa.gc_freeze and pa.lean_time
    assert pa.wait_index and pa.b2_fewer_reads and pa.res_lean, "evalfast2rl's flags"
    assert pa.inflight_array and pa.arrival_lag == 2 and not pa.res_dev
