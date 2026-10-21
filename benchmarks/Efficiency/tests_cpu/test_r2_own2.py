"""CPU tests of B2's INTEGRATION: OwnResidency (nosi/nosi/r2_own2.py), the residency service whose ownership authority
is the array-backed table, against the vendored worlds (v1 under the S-C schedule, S-C dense, S-C sparse).

  B2I-1  C's SHARED FIXTURES (r2_lockstep_fixtures.py, note 12m), checked and unchecked: the B2 world raises the same
         exceptions, returns the same page tables and has equal snapshots (state, counters, readers, forwarded readers
         and write-backs, the journal per kind, the pending gates, the device mirror against the full scan, the trash
         sentinel) after EVERY operation; the vendored world is first (its preconditions).
  B2I-2  each fixture PLANT in its B2 form is caught at C's predicted operation and field (B2_PLANTS; one exception
         and one skip, with their reasons: owner_blind, void_keeps_pins).
  B2I-3  the PLANT MAP covers every fixture plant, and the review's 8 kinds.
  B2I-4  the F-SCHEDULE LOCKSTEP, B2 against v1 (pages, verify content, state, counters, readers, B2's mirror every
         tick): F6 / F12 (24 ticks, ready + in flight + new), F12b (capacity pressure, refusals, the reserve), F20b
         (Policy E, LRU kind), F21b (Policy E, in-flight kind, force_only), checked and unchecked.
  B2I-5  F22: the arrival rule (lag 2) makes two B2 runs with different transport timing equal; free timing does not.
  B2I-6  rev 2's PLANTED RECORD DEFECTS (b2_mut rec_mut): the first record of a kind altered; the checked world
         refuses it (a pre / post predicate) or the lockstep differs, at the tick of the mutation.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_own2.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _load_shared():
    """test_r2_residency.py as a module (its worlds, schedules, snapshots and fixtures; package r2res)."""
    name = "test_r2_residency_shared_b2"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / "test_r2_residency.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


TRM = _load_shared()
OWN2 = TRM._load("r2_own2")
FXM = TRM.FXM
MU, ER, Row, S, Ev, world, NB, L, H = TRM.MU, TRM.ER, TRM.Row, TRM.S, TRM.Ev, TRM.world, TRM.NB, TRM.L, TRM.H


class CancelledAcceptedB2(OWN2.OwnResidency):
    """The B2 form of C's CancelledAccepted (review kind 3): the cancelled-key check of _check_rows skipped."""

    def _check_rows(self, l, rows, sel):
        cancelled, self.cancelled = self.cancelled, set()
        try:
            super()._check_rows(l, rows, sel)
        finally:
            self.cancelled = cancelled


def _fx_b2(fx, cls=None, checked=True, b2_mut=None, svc_kw=None, mut=None):
    """The B2 world of fixture fx (as _fx_world: force_only, the fixture's capacities and trash slot)."""
    kw = dict(nblk=NB, checked=checked, b2_mut=b2_mut or {})
    kw.update(svc_kw or {})
    w = world(C=fx.C, X=fx.X, R_ver=fx.R_ver, seed=0, mode="force_only", mut=mut, cls=cls or OWN2.OwnResidency,
              svc_kw=kw, trash=fx.trash)
    if fx.trash:
        TRM._fill_trash(w)
    return w


# ===================================================================================================================
# B2I-1 the shared fixtures
# ===================================================================================================================
@pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("name", sorted(FXM.FIXTURES))
def test_B2I1_b2_agrees_with_the_vendored_worlds_on_each_fixture(name, checked):
    fx = FXM.FIXTURES[name]
    ref = fx.worlds[0]
    worlds = {ref: TRM._fx_world(fx, ref)}
    if ref == "v1" and "sc" in fx.worlds:
        worlds["sc"] = TRM._fx_world(fx, "sc")
    worlds["b2"] = _fx_b2(fx, checked=checked)
    res = FXM.run(fx, worlds, TRM._fx_snapshot, Row)
    assert res.diffs == [], (name, res.diffs[:1])
    assert len(res.raised) == len(fx.ops), (name, len(res.raised), len(fx.ops))
    assert fx.pre(res, worlds[ref]), (name, fx.doc)
    n_ticks = sum(1 for op in fx.ops if op[0] == "tick")          # the successful ticks (a raise op is not one)
    assert worlds["b2"].svc.b2_applied == n_ticks, (name, worlds["b2"].svc.b2_applied, n_ticks)


# ===================================================================================================================
# B2I-2 / -3 the fixture plants in their B2 form
# ===================================================================================================================
B2_PLANTS = dict(
    dup_take_device=(dict(svc_kw=dict(sb_mut={"no_dedup": True})), None),            # B2's dedup predicate refuses
    dup_take_host=(dict(checked=False, svc_kw=dict(sb_mut={"no_dedup": True})), None),   # unchecked: a second take
    owner_not_mirrored=(dict(b2_mut={"mirror_ignores_owner": True}), None),
    # B2's "in-flight lookup" IS the mirror's current-owner flag (and slot_of): ignoring the owner shows at the owner
    # change, in the mirror (op 5), before v1's page difference (op 7)
    owner_blind=(dict(b2_mut={"mirror_ignores_owner": True, "owner_blind": True}), (5, "mirror")),
    no_uid_check=(dict(b2_mut={"no_uid_check": True}), None),                        # I1 after the poll
    free_on_cancel=(dict(b2_mut={"free_on_cancel": True}), None),
    cancel_refusal_off=(dict(cls=CancelledAcceptedB2, mut=MU.Mutations(no_late_protection_guard=True)), None),
    late_guard_off=(dict(mut=MU.Mutations(no_late_protection_guard=True)), None),
    batch_recency=(dict(b2_mut={"one_batch": True}), None),                          # the journal, not the state
    no_overflow_flag=(dict(svc_kw=dict(sc_mut={"no_overflow_flag": True})), None),   # B2's hole predicate
    flag_ignored=(dict(b2_mut={"ignore_flags": True}), None),
    no_t4_refusal=(dict(b2_mut={"no_t4": True}), None),
    trash_written=(dict(cls=OWN2.TrashWriterB2), None),
    no_reader_wait=(dict(b2_mut={"no_reader_wait": True}), None),
    lost_forward=(dict(b2_mut={"drop_rd_last": True}), None),
    drop_keeps_reader=(dict(b2_mut={"keep_rd_last_on_drop": True}), None),
    poll_order_swapped=(dict(b2_mut={"poll_order_swapped": True}), None),
    key_alias=(dict(b2_mut={"key_alias": True}), None),
)
B2_SKIP = dict(void_keeps_pins="B1's device pin table: B2 keeps no pin table (pin_records=False); its protection is "
                               "the rowsel OR into its own pin planes (B2I-6 drop_rowsel is its planted defect)")
_PLANTS = [pytest.param(name, p, id="%s-%s" % (name, p.name)) for name in sorted(FXM.FIXTURES)
           for p in FXM.FIXTURES[name].plants]


@pytest.mark.parametrize("name, plant", _PLANTS)
def test_B2I2_each_fixture_plant_is_caught_in_the_b2_world(name, plant):
    if plant.name in B2_SKIP:
        pytest.skip(B2_SKIP[plant.name])
    kw, expect = B2_PLANTS[plant.name]
    fx = FXM.FIXTURES[name]
    ref = fx.worlds[0]
    worlds = {ref: TRM._fx_world(fx, ref), "b2": _fx_b2(fx, **kw)}
    res = FXM.run(fx, worlds, TRM._fx_snapshot, Row)
    assert res.diffs, (name, plant.name, "not caught", plant.defect)
    i, what, detail = res.diffs[0]
    got = (i, what if what != "snapshot" else detail["field"])
    assert got == (expect or (plant.at, plant.field)), (name, plant.name, res.diffs[0])


def test_B2I3_the_b2_plant_map_covers_every_fixture_plant():
    plants = [p for fx in FXM.FIXTURES.values() for p in fx.plants]
    assert not set(B2_PLANTS) & set(B2_SKIP)
    assert sorted(set(B2_PLANTS) | set(B2_SKIP)) == sorted(p.name for p in plants)
    covered = {p.kind for p in plants if p.name in B2_PLANTS}
    assert set(FXM.REVIEW_KINDS) <= covered, sorted(set(FXM.REVIEW_KINDS) - covered)


# ===================================================================================================================
# B2I-4 the F-schedule lockstep against v1
# ===================================================================================================================
def _lockstep_b2(seed, C, X, R_ver, sched, hb=15, mode="delay", max_delay=4, checked=True, b2_mut=None,
                 svc_kw=None):
    """v1 under the S-C schedule and the B2 world driven by the same schedule (transports seeded alike, readers two
    ticks late, cohorts decided at their ends). Per tick: B2's device mirror against the full scan (after the poll),
    the page tables of every row and head, the verify content at the read point (both), and after end_tick the state,
    the counters and every slot's readers (named by tick and layer). An exception in one world only is a difference.
    Returns (the differences, v1's world, B2's world)."""
    common = dict(C=C, X=X, R_ver=R_ver, seed=seed, mode=mode, max_delay=max_delay)
    wv = world(**common, svc_kw=dict(poll_in_resolve=False, defer_issue=True, defer_admit=True))
    bkw = dict(nblk=NB, checked=checked, b2_mut=b2_mut or {})
    bkw.update(svc_kw or {})
    wb = world(**common, cls=OWN2.OwnResidency, svc_kw=bkw)
    ws = (wv, wb)
    uids = ("u0", "u1")
    for w in ws:
        for b, u in enumerate(uids):
            w.svc.begin_request(b, u)
    tail = wv.svc.tail0
    readers, bad = [], []
    tags = ({}, {})

    def both(fn):
        out, err = [], []
        for w in ws:
            try:
                out.append(fn(w))
                err.append(None)
            except Exception as e:                                # noqa: BLE001 - compared by class name
                out.append(None)
                err.append("%s: %s" % (type(e).__name__, str(e)[:200]))
        return out, err
    for (t, m, ends, layers) in sched:
        _, err = both(lambda w: w.svc.poll())
        if err != [None, None]:
            bad.append(("raise", t, "poll", err))
            break
        wb.svc.sync_mirror()
        for l in range(L):
            d = TRM._mirror_diff(wb.svc, l)
            if d:
                bad.append(("mirror", t, l, d))
        if bad:
            break
        for l, spec in enumerate(layers):
            rows = [Row(b, kind, key, 1000 + t, hb, [(hb, tail)]) for (b, kind, key, _) in spec]
            sel = S(*[s for (_, _, _, s) in spec])
            (pv, pb), err = both(lambda w: w.svc.resolve(l, rows, sel))
            if err != [None, None]:
                bad.append(("raise", t, "resolve", l, err))
                break
            for i in range(len(rows)):
                for h in range(H):
                    a = pv.page_table[i, h, :int(pv.num_pages[i, h])].tolist()
                    c = pb.page_table[i, h, :int(pb.num_pages[i, h])].tolist()
                    if a != c:
                        bad.append(("pages", t, l, i, h, a, c))
            for name, w, p in (("v1", wv, pv), ("b2", wb, pb)):
                if not TRM._content_ok(w, l, rows, sel, p):
                    bad.append(("content", name, t, l))
            evs = (Ev(), Ev())
            for k in range(2):
                tags[k][id(evs[k])] = "t%d.l%d" % (t, l)
            readers.append((t, evs))
            wv.svc.read_done(l, pv, evs[0])
            wb.svc.read_done(l, pb, evs[1])
        if bad:
            break
        _, err = both(lambda w: w.svc.end_tick())
        if err != [None, None]:
            bad.append(("raise", t, "end_tick", err))
            break
        for (tt, evs) in readers:
            if tt <= t - 2:
                for e in evs:
                    e.done = True
        if ends:
            for w in ws:
                for u in uids:
                    w.svc.decided((u, 1, m))
        sv, sb = TRM._state(wv), TRM._state(wb)
        if sv != sb:
            first = next(i for i, (x, y) in enumerate(zip(sv, sb)) if x != y)
            bad.append(("state", t, sv[first], sb[first]))
        if TRM._counters(wv.svc) != TRM._counters(wb.svc):
            bad.append(("counters", t, TRM._counters(wv.svc), TRM._counters(wb.svc)))
        if TRM._reader_tags(wv.svc, tags[0]) != TRM._reader_tags(wb.svc, tags[1]):
            bad.append(("readers", t))
        if bad:
            break
    if not bad:
        wb.svc.sync_mirror()
        for l in range(L):
            d = TRM._mirror_diff(wb.svc, l)
            if d:
                bad.append(("mirror", "end", l, d))
    return bad, wv, wb


SCHEDULES = dict(
    f6=lambda: dict(seed=1, C=8, X=48, R_ver=8, sched=TRM._sched(101, 24, True)),
    f12b=lambda: dict(seed=3, C=2, X=96, R_ver=88, sched=TRM._sched_pressure(103, 15), hb=8),
    f20b=lambda: dict(seed=5, C=24, X=2, R_ver=2, sched=TRM._sched_pressure(105, 18, hb=36, window=3, nver=3), hb=36),
    f21b=lambda: dict(seed=7, C=16, X=2, R_ver=2, sched=TRM._sched_split(109, 30, hb=30, d_hi=20, window=2, nver=1),
                      hb=30, mode="force_only"),
)


@pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("sched", sorted(SCHEDULES))
def test_B2I4_b2_equals_v1_on_the_lockstep_schedules(sched, checked):
    bad, wv, wb = _lockstep_b2(checked=checked, **SCHEDULES[sched]())
    assert bad == [], bad[:3]
    m, c = wv.svc.metrics, wv.svc.metrics.counts
    pre = dict(f6=m.k43["ready"] > 0 and m.k43["in_flight"] > 0 and m.k43["new"] > 0,
               f12b=m.refused > 0 and m.k43["new"] > 0,
               f20b=c.get("verifier_evictions", 0) > 0,
               f21b=c.get("verifier_inflight_evictions", 0) > 0)[sched]
    assert pre, (sched, m.k43, m.refused, c)                      # the schedule reached its transitions
    assert wb.svc.b2_applied > 0


def test_B2I5_a_fixed_arrival_lag_makes_two_b2_runs_comparable():
    fs, ft, ws = TRM._lag_pair(OWN2.OwnResidency, 2)
    assert ft is not None, "precondition: the two transports landed alike"
    assert sum(w.svc.arrival_forced for w in ws) > 0 and sum(w.svc.arrival_held for w in ws) > 0
    assert fs is None, fs
    fs0, ft0, _ = TRM._lag_pair(OWN2.OwnResidency, None)
    assert ft0 is not None and fs0 is not None, (ft0, fs0)        # NEGATIVE CONTROL: free timing differs


# ===================================================================================================================
# B2I-6 rev 2's planted record defects
# ===================================================================================================================
REC_PLANTS = dict(                                # (schedule, how it is caught: "raise" in the checked world, or a field)
    take_nonfree=("f6", "raise"), take_swap=("f6", "raise"), k5_over=("f12b", "raise"),
    pe_not_first=("f20b", "raise"), drop_refusal=("f12b", "raise"), drop_v_merge=("f6", "state"),
    drop_d_merge=("f6", "state"), drop_ready=("f6", "readers"), drop_rowsel=("f6", "state"),
    drop_vread=("f6", "readers"), reverse_stamps=("f6", "state"), evict_po_unpinned=("f20b", "raise"),
    gen_plus_two=("f6", "raise"),
)


@pytest.mark.parametrize("kind", sorted(REC_PLANTS))
def test_B2I6_a_planted_record_defect_is_caught_at_its_tick(kind):
    sched, how = REC_PLANTS[kind]
    bad, wv, wb = _lockstep_b2(b2_mut={"rec_mut": kind}, **SCHEDULES[sched]())
    done = wb.svc.rec_mut_done
    assert done is not None, (kind, "precondition: the schedule has a record of this kind")
    assert bad, (kind, "not caught")
    first = bad[0]
    tick = first[2] if first[0] == "content" else first[1]        # ("content", world, t, l); else (what, t, ...)
    assert tick == done[0], (kind, first, done)                   # caught at the tick of the mutation
    if how == "raise":
        assert first[0] == "raise" and "OwnershipViolation" in str(first[-1]), (kind, first)
    else:
        assert first[0] in (how, "state", "counters", "readers", "pages", "mirror"), (kind, first)


def test_B2I6b_ignore_overflow_is_caught_on_the_void_overflow_fixture():
    """rec_mut ignore_overflow: the verify overflow is not refused; the holes reach B2's hole predicate
    (OwnershipViolation where the vendored worlds refuse the tick, CapacityRefusal)."""
    fx = FXM.FIXTURES["void_overflow"]
    worlds = {"v1": TRM._fx_world(fx, "v1"), "b2": _fx_b2(fx, b2_mut={"rec_mut": "ignore_overflow"})}
    res = FXM.run(fx, worlds, TRM._fx_snapshot, Row)
    assert res.diffs and res.diffs[0][:2] == (1, "raise"), res.diffs[:1]
    assert res.diffs[0][2]["got"]["b2"] == "OwnershipViolation", res.diffs[0]
