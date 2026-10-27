"""[f3all] CPU tests of the merged tree r2-f3all: f3lia 5cdfc75 (fused resolver v3 + --res-lean / --res-dev +
--inflight-array + the acc216 gates) + endbatch 1ae07fe (--end-batch: OwnResidency.end_requests, O.end_owners) +
applyev a3199da (--apply-batched: TickEngine._apply_batched). The merge adds no mechanism: the ends are run_round's
(after the tick span), the batched apply is TickEngine.apply_events, the fused resolve and the array poll are the
service's. These tests prove that the parts still work TOGETHER. The kernels run under the Triton interpreter
(TRITON_INTERPRET=1) in a FRESH process (L36), kernel v3 (as test_r2_f3lia.py's child). The engines use rebuild
"batched" (the cells' variant fused/batched; --apply-batched needs it) and set eng.rebuild / eng.apply_batched /
eng.end_batch after the construction, in make_engine's order.

  FA1  ALL THREE FLAGS REACH THE TICK (child): on the cases below, fused v3 + --res-lean --res-dev + --inflight-array +
       --end-batch + --apply-batched: the fused layers run (kernel v3, lean on), the array poll admits (the old poll's
       per-transfer rule is never called), the service ends requests only through end_requests (end_request never),
       the batched apply runs with rollbacks (apply_calls > 0; r2_spec_core.sync_draft_bank is never called, while the
       reference calls it L times per rollback).
  FA2  THE C3L EXACTNESS PATH IS KEPT (child): check mode + all the flags compares every layer, 0 diffs, no error, and
       the batched apply and the batched ends ran in that round too (the evalfast2c3all gate's CPU form).
  FA3  STATE (child): the f3lia path + --end-batch --apply-batched == the f3lia path (the evalfast2f3lia control, rebuild
       batched) on randomized event streams (test_r2_applyev.py's _sched: SG1's forced schedule, rollbacks only,
       mixed seeds, the free-running greedy round) and one scripted stream (lock_sched: a per-request cycle schedule,
       so with one max_new for all the requests stay in lockstep and all end in ONE tick, by construction) with one
       max_new for all or staggered over ticks (test_r2_endbatch.py's STAGGER), on both layouts: every tick and layer
       (RF._rounds_diff), the full service state at every poll and at the end (test_r2_endbatch.py's _snap), the twin
       tables, stores and token log at the end, the X pool map per tick, the same end (HF._same_end), the same ended
       requests per tick. PRECONDITIONS over the set (FA3z): a tick that ends 2 or more requests (the lock case), a
       round that ends requests in 2 or more ticks, a batched apply call with 2 or more rollbacks, a rollback in each
       stream with a reject schedule.
  FA4  NEGATIVE CONTROLS on this path (child; L34: each plant's own act precondition; L49: each plant only at the site
       under test): (a) test_r2_endbatch.py's skip-one plant of O.end_owners (the last request of each batched end
       keeps its owner): precondition: it acted and left an owner; the compare sees it. (b) test_r2_applyev.py's
       DropRbCopy (ONE batched apply drops the ring copy of its first rollback only): preconditions: the 128 rows
       per layer of that rollback only, the rows stay different after the call; the per-tick compare sees it, never
       before the tick after the plant.
  FA5  THE FLAGS IN PROCESS: the harness parses the flag sets of the three new cells; the refusals of each side stay;
       make_engine passes / sets all of them; the signatures keep the defaults off.
  FA6  STATIC GATE / HELPER LISTS (T21 style): one_point's gates.extend helpers; main's meta lines and gate runners; the
       helpers of every side (flean, admitfill, acc216, endbatch, applyev); the step pins the changed modules.
  FA7  THE CELLS (static): evalfast2all / evalfast2c3all / evalacc216all; the control evalfast2f3lia unchanged; the
       REGISTERED F3ALL block has no stamp and every PREDICTION is "TBD by lead" (after the lead stamps it: no TBD is
       left and the bands equal ALL_BANDS); outside DRYRUN the step refuses ALL_BANDS=TBD.
  FA8  THE DRYRUN argv of the three new cells on a clean checkout of this ROOT: each equals its f3lia base cell's argv
       plus --end-batch --apply-batched (the time cell without its bands; the acc cell's link renamed), parsed into
       the flags.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_f3all.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


RF = _shared("test_r2_resfix_shared_rd", "test_r2_resfix.py")         # RL2's rounds (shared name, as test_r2_f3lia.py)
HF, TT, OWN2 = RF.HF, RF.TT, RF.OWN2
TK = TT.TK
p3 = RF.p3                                                            # the fixture, by name
ALL_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3tall"                        # NOSI_EVAL ([f3tall]; was nosi-r2-f3all)
STANDIN_BANDS = "1:2:1:2:1:2"                                         # syntax only (no prediction): the parse test


def _need_triton():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")


# =====================================================================================================================
# the child process (TRITON_INTERPRET=1 from its start)
# =====================================================================================================================
FA_CHILD = r'''
import importlib.util, json, os, sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "FA child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "FA child: triton was imported before the interpreter flag was read"
import torch
HERE, PART = sys.argv[1], sys.argv[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


RF = load("test_r2_resfix_fachild", os.path.join(HERE, "test_r2_resfix.py"))
EB = load("test_r2_endbatch_fachild", os.path.join(HERE, "test_r2_endbatch.py"))   # _snap, STAGGER, _ends_by_tick, plants
AE = load("test_r2_applyev_fachild", os.path.join(HERE, "test_r2_applyev.py"))     # _sched, DropRbCopy, P6.TWIN_KEYS
TT, HF = RF.TT, RF.HF
TK, SC, TRM = TT.TK, TT.SC, HF.TRM
O = RF.OWN2.O
FR = TT._load("r2_fusedres")
assert FR.triton is not None, "FA child: triton did not import"
FR.DEFAULT_KERNEL = "v3"
FR.KERNELS = ("v3",)                                    # a resolver with any other kernel raises
OWNC = RF.OWN2.OwnResidency
OUT = {}
COUNT = dict(old_polls=0, sync_draft=0)
_old_arrived = OWNC._arrived                            # the per-transfer arrival rule: the OLD poll's call only
_sync_draft = SC.sync_draft_bank                        # the unbatched rollback's draft-bank sync (apply_events step 3)


def _count_old(self, tr):
    COUNT["old_polls"] += 1
    return _old_arrived(self, tr)


def _count_sync(*a, **k):
    COUNT["sync_draft"] += 1
    return _sync_draft(*a, **k)
OWNC._arrived = _count_old
SC.sync_draft_bank = _count_sync                        # (a module global, read at call time by r2_tick)


class MP:
    """The monkeypatch subset that test_r2_endbatch.py's plants use (setattr), with undo."""
    def __init__(self):
        self.saved = []

    def setattr(self, obj, name, val):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for obj, name, val in reversed(self.saved):
            setattr(obj, name, val)
        self.saved = []


def p3_toks(path):
    P3 = load("piece3_v2_fachild", path)
    toks, _ = TT.serial_ref(TT.make_engine(TT.make_ops(P3)))
    return P3, toks


def engine(P3, layout, cfg, seed, mode, allf, checked=False):
    """The f3lia path (allf False: the evalfast2f3lia control) or the f3all path (allf True: + --end-batch
    --apply-batched); the rebuild, apply and end switches after the construction, in make_engine's order."""
    kw = dict(resolve=mode, res_dev=True, arrival_lag=2, inflight_array=True)
    if allf:
        kw["end_batch"] = True
    eng = RF._engine(P3, layout, cfg, seed, True, checked, svc_kw=kw)
    eng.rebuild = "batched"
    eng.apply_batched = bool(allf)
    eng.end_batch = bool(allf)
    return eng


def lock_sched():
    """The "lock" stream: a scripted reject_at that is a function of the request's OWN decision count and the cohort's
    size only (r2_tick.forced_schedule's "cycle" kind, without its position kinds cross / steer). The n-th decided
    cohort of a request (size L) is rejected at offset n % (L + 1); offset L = accepted whole. Why the requests stay in
    lockstep: run_round passes no ready gate, so Round.plan_tick and TickRound.finish_tick / _decide read only the
    counts (s, committed_in_epoch, n_triggered, len(out), max_new) and this function's result; each request starts
    with one token, so with one max_new for all (no STAGGER) every request has the same counts at every tick. Thus
    every rejection falls in the same tick for all B requests (one apply with B rollbacks) and all B requests end in
    ONE tick (one end_requests call with B uids): the FA3z preconditions hold by construction, not by a seed."""
    n_dec = {}

    def reject_at(b, c, E):
        k = n_dec.get(b, 0)
        n_dec[b] = k + 1
        L = c.b - c.a + 1
        off = k % (L + 1)
        return None if off == L else c.a + off
    return reject_at


def xround(eng, toks, stream, stagger):
    """RF._round's per-tick and per-layer record, with test_r2_applyev.py's event stream, test_r2_endbatch.py's
    STAGGER (or one max_new for all), the full service state (EB._snap) at every poll and at the end, every end call
    with its tick, and the twin tables / stores / token log at the end."""
    world = eng.res_world
    svc = world.svc
    evs, tags = [], {}
    svc.eb_tags = tags

    def new_event():                                    # named by creation order (both rounds alike)
        e = TT.Ev(True)
        tags[id(e)] = "e%d" % len(evs)
        evs.append(e)
        return e
    eng.backend.new_event = new_event
    pages, layers, ticks, maps, snaps = [], [], [], [], []
    calls = dict(end_request=[], end_requests=[])
    resolve0, poll0, er0, ers0 = svc.resolve, svc.poll, svc.end_request, svc.end_requests

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)]
                          for i in range(R)]))
        layers.append(RF._layer_rec(svc, p))
        return p

    def poll():
        snaps.append(EB._snap(svc))
        return poll0()

    def end_request(uid):
        calls["end_request"].append((len(ticks), uid))
        return er0(uid)

    def end_requests(uids):
        calls["end_requests"].append((len(ticks), list(uids)))
        return ers0(uids)
    svc.resolve, svc.poll, svc.end_request, svc.end_requests = resolve, poll, end_request, end_requests

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
    S_list = eng.S_list
    reject_at = lock_sched() if stream == "lock" else AE._sched(stream)   # FRESH per round: both rounds draw alike

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    max_new = list(EB.STAGGER[:eng.B]) if stagger else [TT.STEPS] * eng.B
    COUNT["old_polls"] = COUNT["sync_draft"] = 0
    rnd, err = None, None
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], max_new, K=3, LAMBDA=7,
                           teacher=None if reject_at is None else teacher, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                              # noqa: BLE001 - a difference at its tick
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    tables = ([getattr(t, k).clone() for t in eng.twins for k in AE.P6.TWIN_KEYS]
              + [x.clone() for x in eng.store_k + eng.store_v + eng.store_bias] + [eng.tok_log[:eng.n_tok].clone()])
    return SimpleNamespace(eng=eng, svc=svc, ticks=ticks, rnd=rnd, err=err, evs=evs, maps=maps, snaps=snaps,
                           final=EB._snap(svc), calls=calls, tables=tables, old_polls=COUNT["old_polls"],
                           sync_draft=COUNT["sync_draft"])


def fused_rep(svc):
    r = svc.fusedres.report()
    return dict(kernel=r["kernel"], launches=r["launches"], layers=r["layers"], stats=r["stats"], err=r["err"],
                compared=r["compared"], diffs=r["diffs"])


def run_info(r):
    s, e = r.svc, r.eng
    ends = EB._ends_by_tick(r.calls)
    return dict(err=r.err, ticks=len(r.ticks), ia=bool(s.inflight_array), ia_stats=dict(s.ia_stats),
                lean_on=s.fusedres._lean_on(), res_lean=bool(s.res_lean), dev=bool(s._lean_c.dev),
                end_batch=bool(s.end_batch), eng_end_batch=bool(e.end_batch), apply_batched=bool(e.apply_batched),
                rebuild=e.rebuild, apply_calls=e.apply_calls, apply_rb_max=e.apply_rb_max,
                apply_rb_items=e.apply_rb_items, rollbacks=e.stats["rollbacks"], publications=e.stats["publications"],
                L=e.L, old_polls=r.old_polls, sync_draft=r.sync_draft, n_end_request=len(r.calls["end_request"]),
                n_end_requests=len(r.calls["end_requests"]),
                end_ticks=len(ends), most_ends=max([len(v) for v in ends.values()] or [0]),
                ends=sorted([t, v] for t, v in ends.items()), rep=fused_rep(s),
                check_diffs=[list(x) for x in s.fusedres.diffs[:8]], n_check_diffs=len(s.fusedres.diffs),
                owners_left=int((s.own.owner != -1).sum()))


def compare(a, b):
    """The first difference of two rounds (None: equal): every tick and layer, the state at every poll, the end state,
    the tables, the pool map, the end."""
    d = RF._rounds_diff(a, b)
    if d is not None:
        return "ticks: " + repr(d)[:300]
    s = RF._same(a.snaps, b.snaps, "snaps")
    if s:
        return "snaps: " + repr(s)[:300]
    s = RF._same(a.final, b.final, "final")
    if s:
        return "final: " + repr(s)[:300]
    if len(a.tables) != len(b.tables) or not all(torch.equal(x, y) for x, y in zip(a.tables, b.tables)):
        return "tables"
    if len(a.maps) != len(b.maps) or not all(torch.equal(x, y) for x, y in zip(a.maps, b.maps)):
        return "maps"
    try:
        HF._same_end(a, b)
    except AssertionError as e:
        return "same_end: " + repr(e)[:300]
    if EB._ends_by_tick(a.calls) != EB._ends_by_tick(b.calls):
        return "ends by tick"
    return None


CASES = [("stream", "default", "forced", False, 1, False), ("stream", "pressure", "rb", True, 1, False),
         ("stream", "pressure", "mixed1", True, 2, False), ("stream", "default", "free", True, 1, False),
         ("stream", "pressure", "mixed3", False, 3, True), ("pool", "default", "forced", True, 1, False),
         ("pool", "pressure", "mixed2", True, 1, False), ("pool", "pressure", "rb", False, 2, True),
         ("stream", "pressure", "lock", False, 1, False)]

if PART == "cases":
    print("FA_JSON " + json.dumps(dict(cases=CASES)))
    sys.exit(0)

if PART == "case":
    i, P3, toks = int(sys.argv[3]), *p3_toks(sys.argv[4])
    layout, cfg, stream, stagger, seed, checked = CASES[i]
    base = xround(engine(P3, layout, cfg, seed, "fused", False, checked), toks, stream, stagger)
    allr = xround(engine(P3, layout, cfg, seed, "fused", True, checked), toks, stream, stagger)
    chk = xround(engine(P3, layout, cfg, seed, "check", True, checked), toks, stream, stagger)
    OUT["case"] = dict(case=list(CASES[i]), base=run_info(base), all=run_info(allr), chk=run_info(chk),
                       diff=compare(base, allr), inflight_same=list(dict.keys(base.svc.inflight)) ==
                       list(dict.keys(allr.svc.inflight)))

elif PART == "plant":
    P3, toks = p3_toks(sys.argv[3])
    layout, cfg, stream, stagger, seed, checked = "stream", "pressure", "mixed1", True, 2, False
    clean = xround(engine(P3, layout, cfg, seed, "fused", True), toks, stream, stagger)
    # (a) the batched end skips the last request of each call (EB's plant on O.end_owners; its only caller is
    #     OwnResidency.end_requests, so no sibling function sees it: L49)
    hit, mp, real_end_owners = [], MP(), O.end_owners
    EB._plant_skip_one(mp, hit)
    planted = O.end_owners is not real_end_owners
    try:
        bad = xround(engine(P3, layout, cfg, seed, "fused", True), toks, stream, stagger)
    finally:
        mp.undo()
    skip = dict(hit=len(hit), planted=planted, restored=O.end_owners is real_end_owners, bad=run_info(bad),
                clean=run_info(clean), diff=compare(clean, bad))
    # (b) ONE batched apply drops the ring copy of its first rollback only (AE's DropRbCopy; the same engine, its
    #     class swapped after the construction: the plant overrides _apply_batched and copy_rows_many only)
    eng = engine(P3, layout, cfg, seed, "fused", True)
    eng.__class__ = AE.DropRbCopy
    eng.plant, eng._drop = None, None
    bad2 = xround(eng, toks, stream, stagger)
    d2 = RF._rounds_diff(clean, bad2)
    drop = dict(plant=eng.plant, L=eng.L, block=TK.BLOCK, bad=run_info(bad2),
                diff=None if d2 is None else [d2[0], repr(d2)[:300]], full=compare(clean, bad2))
    OUT["plant"] = dict(skip=skip, drop=drop)
print("FA_JSON " + json.dumps(OUT))
'''


def _child(part, *extra):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", FA_CHILD, str(HERE), part] + [str(x) for x in extra], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("FA_JSON ")]
    assert r.returncode == 0 and last, ("FA child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                                        % (part, r.returncode, r.stdout[-4000:], r.stderr[-8000:]))
    return json.loads(last[-1][len("FA_JSON "):])


# the cases (the child's CASES: layout, cfg, stream, stagger, seed, checked); FA0 checks that the two lists agree
CASES = [("stream", "default", "forced", False, 1, False), ("stream", "pressure", "rb", True, 1, False),
         ("stream", "pressure", "mixed1", True, 2, False), ("stream", "default", "free", True, 1, False),
         ("stream", "pressure", "mixed3", False, 3, True), ("pool", "default", "forced", True, 1, False),
         ("pool", "pressure", "mixed2", True, 1, False), ("pool", "pressure", "rb", False, 2, True),
         ("stream", "pressure", "lock", False, 1, False)]
CASE_IDS = ["-".join(str(x) for x in c) for c in CASES]
_RUNS = {}


def _case(p3, i):
    _need_triton()
    if i not in _RUNS:
        _RUNS[i] = _child("case", i, p3.__file__)["case"]
    return _RUNS[i]


def test_FA0_the_child_cases_are_the_test_cases():
    _need_triton()
    got = [tuple(c) for c in _child("cases")["cases"]]
    assert got == CASES, (got, CASES)


# =====================================================================================================================
# FA1 all three flags reach the tick
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(CASES)), ids=CASE_IDS)
def test_FA1_all_three_flags_reach_the_tick(p3, i):
    r = _case(p3, i)
    for k in ("all", "chk"):
        x = r[k]
        assert x["err"] is None and x["ticks"] > 10, (k, x["err"])
        assert x["ia"] and x["lean_on"] and x["res_lean"] and x["dev"], (k, "the f3lia flags did not reach the service")
        assert x["rep"]["kernel"] == "v3" and x["rep"]["layers"] > 0, (k, "precondition: the fused layers ran")
        st = x["ia_stats"]
        assert st["polls"] > 0 and st["parts_due"] > 0 and st["admitted"] > 0, (k, "the array poll did not admit", st)
        assert x["old_polls"] == 0, (k, "the old poll ran with --inflight-array", x["old_polls"])
        assert x["end_batch"] and x["eng_end_batch"], (k, "--end-batch did not reach the service and run_round")
        assert x["n_end_requests"] > 0 and x["n_end_request"] == 0, (k, "the ends did not go through end_requests", x)
        assert x["apply_batched"] and x["rebuild"] == "batched", (k, "--apply-batched did not reach the engine")
        assert x["sync_draft"] == 0, (k, "the unbatched rollback sync ran with --apply-batched", x["sync_draft"])
        assert x["apply_rb_items"] == x["rollbacks"], (k, x["apply_rb_items"], x["rollbacks"])
        if x["rollbacks"]:
            assert x["apply_calls"] > 0, (k, "precondition: the batched apply ran with rollbacks", x)
    b = r["base"]
    assert b["ia"] and b["lean_on"] and not b["end_batch"] and not b["apply_batched"], "the reference: f3lia's path"
    assert b["n_end_requests"] == 0 and b["n_end_request"] > 0, "the reference ends per request"
    assert b["apply_calls"] == 0 and b["sync_draft"] == b["L"] * b["rollbacks"], \
        ("the reference's rollbacks sync L times each (the counter sees them: L34)", b["sync_draft"], b["rollbacks"])


# =====================================================================================================================
# FA2 the C3L exactness path is kept
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(CASES)), ids=CASE_IDS)
def test_FA2_check_mode_with_all_flags_compares_every_layer(p3, i):
    c = _case(p3, i)["chk"]
    assert c["err"] is None, c["err"]
    assert c["rep"]["compared"] == c["rep"]["layers"] > 0, (c["rep"]["compared"], c["rep"]["layers"])
    assert c["rep"]["diffs"] == 0 and c["check_diffs"] == [] and c["rep"]["err"] == 0, c["check_diffs"]
    assert c["n_end_requests"] > 0 and (c["apply_calls"] > 0 or c["rollbacks"] == 0), c


# =====================================================================================================================
# FA3 state: f3all == f3lia on randomized streams
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(CASES)), ids=CASE_IDS)
def test_FA3_end_batch_and_apply_batched_equal_the_f3lia_path(p3, i):
    r = _case(p3, i)
    assert r["base"]["err"] is None and r["base"]["ticks"] > 10, ("precondition: the reference ran", r["base"]["err"])
    assert r["diff"] is None, ("--end-batch --apply-batched differ from the f3lia path", r["case"], r["diff"])
    assert r["inflight_same"]
    a, b = r["base"], r["all"]
    assert a["ends"] == b["ends"], (a["ends"], b["ends"])
    assert (a["rollbacks"], a["publications"]) == (b["rollbacks"], b["publications"])
    assert (a["rep"]["launches"], a["rep"]["layers"], a["rep"]["stats"], a["rep"]["err"]) == \
        (b["rep"]["launches"], b["rep"]["layers"], b["rep"]["stats"], b["rep"]["err"])
    assert a["owners_left"] == b["owners_left"] == 0, (a["owners_left"], b["owners_left"])
    stream = r["case"][2]
    if stream != "free":                                              # a reject schedule: rollbacks on this stream
        assert b["rollbacks"] > 0, ("precondition: the stream's rollbacks", r["case"], b["rollbacks"])
    if stream == "lock" and not r["case"][3]:                         # lock_sched, one max_new: the lockstep
        assert len(b["ends"]) == 1 and len(b["ends"][0][1]) >= 2, ("the lock stream ends all requests in ONE tick",
                                                                    b["ends"])
        assert b["apply_rb_max"] >= 2, ("the lock stream rolls back every request in one apply", b["apply_rb_max"])


def test_FA3z_the_cases_reach_every_end_and_apply_mix(p3):
    """PRECONDITION over the set (L34; cached runs): a tick ends 2+ requests; a round ends requests in 2+ ticks; a
    batched apply call carries 2+ rollbacks; both layouts and a checked service are covered."""
    runs = [_case(p3, i) for i in range(len(CASES))]
    assert max(r["all"]["most_ends"] for r in runs) >= 2, "precondition: no tick ended 2 or more requests"
    assert max(r["all"]["end_ticks"] for r in runs) >= 2, "precondition: no round ended requests in 2 or more ticks"
    assert max(r["all"]["apply_rb_max"] for r in runs) >= 2, "precondition: no batched apply with 2 or more rollbacks"
    assert {c[0] for c in CASES} == {"stream", "pool"} and any(c[5] for c in CASES)
    print("REPORT FA3: per case (end ticks, most ends in a tick, rollbacks, most rollbacks in one apply) %s"
          % [(r["all"]["end_ticks"], r["all"]["most_ends"], r["all"]["rollbacks"], r["all"]["apply_rb_max"])
             for r in runs])


# =====================================================================================================================
# FA4 negative controls on this path
# =====================================================================================================================
_PLANT = {}


def _plant(p3):
    _need_triton()
    if not _PLANT:
        _PLANT.update(_child("plant", p3.__file__)["plant"])
    return _PLANT


def test_FA4a_a_skipped_batched_end_is_caught(p3):
    r = _plant(p3)["skip"]
    assert r["clean"]["err"] is None and r["clean"]["owners_left"] == 0, r["clean"]["err"]
    assert r["clean"]["end_batch"] and r["clean"]["apply_batched"] and r["clean"]["rep"]["layers"] > 0
    assert r["planted"] and r["restored"], "the plant was not set or not removed"
    assert r["hit"] > 0, "precondition (L34): the plant did not act (no batched end ran)"
    assert r["bad"]["owners_left"] > 0, ("precondition (L34): the skipped requests kept no owner", r["bad"])
    assert r["diff"] is not None, "the skipped end was not seen"


def test_FA4b_a_dropped_rollback_copy_is_caught(p3):
    r = _plant(p3)["drop"]
    pl = r["plant"]
    assert pl is not None, "precondition: no rollback whose ring copy changes rows"
    assert pl["dropped"] == [2 * r["block"]] * r["L"], ("precondition: the rollback's 128 rows per layer only", pl)
    assert all(k == t - 2 * r["block"] for k, t in zip(pl["kept"], pl["total"])), ("L49: only its columns", pl)
    assert pl["effective"], ("precondition (L34): the dropped copy left the rows different", pl)
    assert r["bad"]["end_batch"] and r["bad"]["apply_batched"] and r["bad"]["rep"]["layers"] > 0
    assert r["diff"] is not None or r["full"] is not None, "the dropped copy was not seen"
    if r["diff"] is not None:                                         # the per-tick compare: never before the effect
        assert r["diff"][0] >= pl["tick"] + 1, (r["diff"], pl)
    print("REPORT FA4b: plant %s; per-tick diff %s; full compare %s" % (pl, r["diff"], r["full"]))


# =====================================================================================================================
# FA5 the flags in process
# =====================================================================================================================
SG5_BASE = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
B2 = SG5_BASE + ["--backends", "b2", "--X", "64", "--arrival-lag", "2", "--variants", "fused/batched"]
COMBO = ["--b2-resolve", "fused", "--fused-kernel", "v3", "--res-lean", "--res-dev", "--inflight-array",
         "--end-batch", "--apply-batched"]


def test_FA5_the_flags_parse_together_and_reach_the_engine():
    G = HF._sg5()
    for res in ("fused", "check"):
        a = G.parse_args(B2 + ["--b2-resolve", res] + COMBO[2:])
        assert (a.b2_resolve, a.fused_kernel, a.res_lean, a.res_dev, a.inflight_array, a.end_batch, a.apply_batched,
                a.rebuild, a.arrival_lag) == (res, "v3", True, True, True, True, True, "batched", 2), res
    acc = G.parse_args(B2 + COMBO + ["--modes", "fast,bi_c_strict", "--layout", "pool", "--store-committed-tokens"])
    assert acc.store_committed_tokens and acc.end_batch and acc.apply_batched and acc.inflight_array
    for bad in (SG5_BASE + ["--end-batch"],                                          # --end-batch: b2 only
                B2[:-2] + ["--variants", "fused/batched,torch/per_call"] + COMBO,    # --apply-batched: batched only
                B2[:-2] + ["--variants", "torch/per_call"] + COMBO,
                SG5_BASE + ["--backends", "b2", "--X", "64", "--variants", "fused/batched"] + COMBO,   # no fixed lag
                B2 + ["--b2-resolve", "fused", "--res-dev", "--inflight-array", "--end-batch"]):   # res_dev, no lean
        with pytest.raises(SystemExit):
            G.parse_args(bad)
    me = inspect.getsource(G.make_engine)
    for s in ('inflight_array=bool(getattr(a, "inflight_array", False))', 'end_batch=bool(getattr(a, "end_batch", False))',
              'eng.end_batch = bool(getattr(a, "end_batch", False))', 'res_dev=bool(getattr(a, "res_dev", False))',
              'eng.apply_batched = bool(getattr(a, "apply_batched", False))',
              'fused_kernel=(None if getattr(a, "b2_resolve", "torch") == "torch"'):
        assert s in me, s
    assert me.index('eng.rebuild = getattr(a, "rebuild", "per_call")') < \
        me.index('eng.apply_batched = bool(getattr(a, "apply_batched", False))')
    ps = inspect.signature(OWN2.OwnResidency.__init__).parameters
    assert ps["fused_kernel"].default is None and ps["inflight_array"].default is False
    assert ps["end_batch"].default is False
    assert inspect.signature(TK.TickEngine.__init__).parameters["apply_batched"].default is False
    src = inspect.getsource(OWN2.OwnResidency.__init__)
    assert "self.inflight_array = bool(inflight_array)" in src and "self.end_batch = bool(end_batch)" in src


# =====================================================================================================================
# FA6 static gate / helper lists (T21 style)
# =====================================================================================================================
def test_FA6_the_static_gate_and_helper_lists_hold_every_side():
    G = HF._sg5()
    op = inspect.getsource(G.one_point)
    assert sorted(set(re.findall(r"gates\.extend\((\w+)\(", op))) == ["apply_batched_gates", "fused_gates",
                                                                       "stack_acted_gates"]
    assert "note_fused_warm(svc)" in op and 'info["apply_batched"] = abw' in op
    main = inspect.getsource(G.main)
    for s in ("eval_gates(a, meta, res, gates)", "acc_gates(a, meta, res, gates, acc[", "af.inflight_array = False",
              "meta.update(inflight_array=True)", "meta.update(end_batch=True)", "meta.update(apply_batched=True)",
              'meta["fused_warm_mem_gb"] = _FUSED_WARM[0]'):
        assert s in main, s
    for name in ("fused_gates", "note_fused_warm", "eval_gates", "acc_gates", "accuracy_scope", "serial_reference",
                 "dup_pairs", "load_link", "apply_batched_gates"):
        assert callable(getattr(G, name, None)), name                 # flean / acc216 / applyev
    for name in ("_poll_arrivals_array", "_ia_add", "_ia_force", "_ia_raise", "_ia_forget",   # admitfill
                 "end_requests", "_queue", "end_request"):                                    # endbatch
        assert callable(getattr(OWN2.OwnResidency, name, None)), name
    for name in ("_apply_batched", "_keep_vector", "end_requests", "_publish_batched"):         # applyev / endbatch
        assert callable(getattr(TK.TickEngine, name, None)), name
    assert callable(getattr(TK.ResidencyBackend, "end_requests", None))
    assert callable(getattr(OWN2.O, "end_owners", None)) and callable(getattr(OWN2.O, "end_owner", None))
    assert "rows_out" in inspect.signature(TK.rebuild_banks_unchecked).parameters
    rr = inspect.getsource(TK.run_round)
    assert 'getattr(engine, "end_batch", False)' in rr and "engine.end_requests(" in rr
    ae = inspect.getsource(TK.TickEngine.apply_events)
    assert "self._apply_batched(rb, pub)" in ae and "full = self._keep_vector(keep)" in ae
    for k in ("ia_skip_one", "ia_due_early", "ia_bad_gen"):
        assert k in OWN2.B2_MUTS, k
    step = HF.STEP.read_text()
    for f in ("nosi/nosi/r2_own2.py", "nosi/nosi/r2_own2_ops.py", "nosi/nosi/r2_tick.py", "nosi/nosi/r2_residency_sc.py",
              "nosi/nosi/r2_transport_gilfree.py", "nosi/nosi/r2_fusedres.py", "benchmarks/Efficiency/r2_evalacc.py",
              "nosi/nosi/r2_xpool.py"):
        assert f in step, f                                           # tracked at the pinned commit (the step checks)


# =====================================================================================================================
# FA7 the cells (static)
# =====================================================================================================================
def _bands(text):
    ln = next(x for x in text.splitlines() if x.strip().startswith("ALL_BANDS="))
    return ln.split("=", 1)[1].split()[0]


def _reg(text):
    reg = text[text.index("# REGISTERED for evalfast2all / evalfast2c3all / evalacc216all"):]
    return reg[:reg.index("\n#\n")]


def _cell(text, start):
    i = text.index(start)
    return text[i:text.index(" ;;\n", text.index("EVALARG=(", i)) + 3]


def test_FA7_the_cells_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + ALL_ROOT
    assert ALL_ROOT == HF.HF_ROOT
    assert "|evalacc216f3lia|evalfast2eb|evalfast2ab|evalfast2c3ab|evalfast2all|evalfast2c3all|evalacc216all|" in text
    assert "] [gilfree|gather]" in text          # [f3tall] the f3ld / f3ldx / f3tall cells follow (test_r2_hostfix.py HF6a)
    assert "        evalfast2all|evalfast2c3all) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "        evalacc216all) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    fa = _cell(text, "    evalfast2all) MODES=fast")
    for w in ("BATCH=216", "PX=1512 ; PRV=1464", "VARIANTS=fused/batched", "NOSI=$NOSI_EVAL",
              "--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped",
              "--eval-role fast-time --mem-cap-gb 84", '--eval-bands "$ALL_BANDS" --b2-resolve fused --fused-kernel v3',
              "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array --end-batch --apply-batched",
              '[ "${DRYRUN:-0}" = 1 ] || [ "$ALL_BANDS" != TBD ] ||', "--eval-bands TBD (the lead fills and stamps"):
        assert w in fa, w
    assert "STEP_MIN=" not in fa and "STEPS=" not in fa
    c3 = _cell(text, "    evalfast2c3all) MODES=fast")
    for w in ("STEPS=8 ; DTICKS=4", "BATCH=216",
              "--b2-resolve check --fused-kernel v3 --res-lean --res-dev --inflight-array --end-batch --apply-batched"):
        assert w in c3, w
    for w in ("--bare", "--eval-role", "--accuracy-record", "--eval-bands"):
        assert w not in c3, w
    ac = _cell(text, "    evalacc216all) MODES=fast,bi_c_strict")
    for w in ("BATCH=216", "ACC=1", 'LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2all/summary.json")',
              "--b2-resolve fused --fused-kernel v3", "--res-lean --res-dev --inflight-array --end-batch --apply-batched "
              "--store-committed-tokens --serial-rows 8", "--gold-workload \"$GOLD\" \"${LINKARG[@]}\""):
        assert w in ac, w
    for w in ("--eval-role", "--accuracy-record", "--mem-cap-gb", "--eval-bands", "--bare", "STEP_MIN=", "STEPS="):
        assert w not in ac, w
    # the same-job control keeps its arguments (its own test: test_r2_f3lia.py F3G)
    f3 = _cell(text, "    evalfast2f3lia) MODES=fast")
    assert "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array) ;;" in f3
    assert "--end-batch" not in f3 and "--apply-batched" not in f3
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def test_FA7_the_registration_is_left_to_the_lead():
    text = HF.STEP.read_text()
    reg, bands = _reg(text), _bands(text)
    for tag in ("C3ALL", "ALL-ACT", "ALL-MED", "ALL-MEAN", "ALL-EFF", "ALL-HEND", "ALL-HAPPLY", "ALL-SAME", "ALL-FALS",
                "ALL-MEM", "ACCALL-BI", "ACCALL-ETD", "ACCALL-AE", "ACCALL-LINK", "ACCALL-MEM", "nosi-r2-f3all",
                "5cdfc75", "1ae07fe", "a3199da", "evalfast2f3lia", "0 diffs", "_AB_acted", "VOID"):
        assert tag in reg, tag
    preds = [ln for ln in reg.splitlines() if "PREDICTION:" in ln]
    assert len(preds) >= 13, preds
    if not re.search(r"STAMP \d{4}-\d\d-\d\dT", reg):              # the builder's state: nothing predicted
        assert bands == "TBD", bands
        assert all(ln.rstrip().endswith("PREDICTION: TBD by lead") for ln in preds), preds
    else:                                                             # the lead filled and stamped it
        assert "TBD" not in reg and bands != "TBD", "a stamped block with a TBD left"
        assert re.fullmatch(r"[0-9.]+(:[0-9.]+){5}", bands) and bands in reg, bands


def test_FA7_the_step_refuses_TBD_bands_outside_dryrun(tmp_path):
    text = HF.STEP.read_text()
    if _bands(text) != "TBD":
        pytest.skip("ALL_BANDS is filled (test_FA7_the_registration_is_left_to_the_lead checks the stamp)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "DRYRUN"))}
    env.update(SG5_HOLDERS=str(tmp_path))                             # a second barrier: refused outside DRYRUN too
    r = subprocess.run(["bash", str(HF.STEP), "--nosi", "0" * 40, "418", "t1", "evalfast2all"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, (r.returncode, r.stderr[-300:])
    assert "cell evalfast2all: --eval-bands TBD" in r.stderr, r.stderr[-300:]
    assert "r2_tick_sg5.py" not in r.stdout


# =====================================================================================================================
# FA8 the DRYRUN argv of the new cells
# =====================================================================================================================
FA_CELLS = dict(evalfast2all="evalfast2f3lia", evalfast2c3all="evalfast2c3lia", evalacc216all="evalacc216f3lia")
NEW = ["--end-batch", "--apply-batched"]


def _argv(tmp_path, cell):
    tmp_path.mkdir(parents=True, exist_ok=True)                        # HF._dry writes the holder's uuid file here
    r = HF._dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(ALL_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the f3all ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    return r, line, got


def _norm(got, cell):
    """The argv tokens with the cell's own name (record path) and the acc cells' link cell named alike; the time
    cells' --eval-bands pair dropped (the f3all bands are the lead's)."""
    toks = [t.replace(cell, "CELL").replace("_evalfast2f3lia/", "_LINK/").replace("_evalfast2all/", "_LINK/")
            for t in got.split()]
    if "--eval-bands" in toks:
        i = toks.index("--eval-bands")
        toks = toks[:i] + toks[i + 2:]
    return toks


@pytest.mark.parametrize("cell", sorted(FA_CELLS))
def test_FA8_the_cells_dryrun_are_their_base_plus_the_two_flags(tmp_path, cell):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    base = FA_CELLS[cell]
    r, line, got = _argv(tmp_path / "a", cell)
    _, bline, bgot = _argv(tmp_path / "b", base)
    toks, btoks = _norm(got, cell), _norm(bgot, base)
    assert [toks.count(f) for f in NEW] == [1, 1] and not any(f in btoks for f in NEW), (cell, toks)
    assert [t for t in toks if t not in NEW] == btoks, (cell, [t for t in toks if t not in btoks],
                                                        [t for t in btoks if t not in toks])
    for s in ("PYTHONPATH=%s/nosi" % ALL_ROOT, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell,
              "--time=120", "--mem=240G"):
        assert s in line, (cell, s)
    if cell == "evalfast2all":
        bands = _bands(HF.STEP.read_text())
        assert "--eval-bands %s" % bands in got, (cell, bands)
        if bands == "TBD":
            with pytest.raises(SystemExit):                           # the harness refuses the placeholder too
                HF._sg5().parse_args(got.split())
            got = got.replace("--eval-bands TBD", "--eval-bands " + STANDIN_BANDS)
    pa = HF._sg5().parse_args(got.split())
    assert pa.end_batch and pa.apply_batched and pa.rebuild == "batched", cell
    assert pa.res_lean and pa.res_dev and pa.inflight_array and pa.fused_kernel == "v3" and pa.arrival_lag == 2, cell
    assert (pa.batch, pa.X_pool) == (216, 1512), cell
    if cell == "evalfast2all":
        assert pa.b2_resolve == "fused" and pa.eval_role == "fast-time" and pa.bare and pa.steps == 48
        assert pa.mem_cap_gb == 84 and pa.host_profile and pa.gc_freeze and pa.lean_time
        assert pa.wait_index and pa.b2_fewer_reads and not pa.store_committed_tokens
    elif cell == "evalacc216all":
        assert pa.b2_resolve == "fused" and pa.modes == ["fast", "bi_c_strict"] and pa.store_committed_tokens
        assert pa.eval_role == "" and not pa.bare and pa.mem_cap_gb == 0 and not pa.accuracy_record
        assert pa.link_record == "/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_t1_418_evalfast2all/summary.json"
        assert "[acc216] modes fast,bi_c_strict" in r.stdout
    else:
        assert pa.b2_resolve == "check" and pa.eval_role == "" and pa.steps == 8 and pa.draft_ticks == 4
        assert not pa.bare and not pa.accuracy_record
