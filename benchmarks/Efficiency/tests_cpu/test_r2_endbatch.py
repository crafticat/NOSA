"""[endbatch] CPU tests of --end-batch (analysis/evalfast/DIAG_effective_gap.md section 4, fix 3): run_round ends the
requests that finish in a tick with ONE call (TickEngine.end_requests -> ResidencyBackend.end_requests ->
OwnResidency(end_batch=True).end_requests -> ONE r2_own2_ops.end_owners, one host read, a per-request index _qidx of
the queued keys). The per-request path (end_request, end_owner, the old run_round loop) stays the reference.

  EB1  OP LEVEL on random tables (test_r2_resfix.py's 60 draws: the C_FREE-heavy draws (seed % 3 == 0) and the
       pressure draws (odd seeds) included; plus one table at the eval batch B 216): end_owners(bt) leaves the same
       table (every column, every array, the clock) and returns the same count as end_owner(b) for each b of bt, for
       random subsets in random orders; every order of bt gives the same table; the undo log restores the old codes
       and pins (end_owner's undo contract).
  EB2  SERVICE LEVEL: three B2 services built alike (B 3 / 9 / 64, 24 seeds, C_FREE-heavy table draws on seed % 3 == 0)
       get the same random host operations (history_written, _queue, _key_slot, cancel, a key queued for an ended
       request) and a random table each round; then a random subset ends: one end_request per uid in a random order
       (the reference), end_requests with the index (end_batch True), end_requests with one scan (end_batch False).
       The full state (table, owners, uid_b, hist_ev, hist_from, cancelled, queue, queued, _key_of, _kslot, journal,
       _ver) is equal after EVERY round. Rounds end 1 request, then several, then all that remain.
  EB3  ROUND LEVEL: the B2 round with --end-batch equals the per-request round at every tick (rows, tokens, logits,
       pages, state, counters, readers, memory, the kept counters, the release queue and its set), in the full state
       at every tick's poll (after the previous tick's ends) and after the round, and in the report; one end_requests
       call per tick with ends, with the same requests. Cases: default / pressure, forced / free, all requests with
       the same max_new and staggered max_new, B 4 and B 8. PRECONDITIONS: some tick ends 2 or more requests; some
       round ends requests in 2 or more ticks.
  EB4  FLAG OFF UNCHANGED: the per-request sources (OwnResidency.end_request, r2_own2_ops.end_owner,
       ResidencyBackend.end_request, the run_round loop) are the text of the base commit 8ccf262; a round without the
       flag calls end_request once per request and end_requests never, and keeps no index; the defaults are False;
       the harness flag, its refusal and its wiring.
  EB5  NEGATIVE CONTROLS (L34 / L49: each plant only at the site under test, and each counted where it changes a
       value): end_owners that skips one request's free (the last b); end_owners whose write order differs from its
       read order (bt reversed for the write: each request gets another request's new codes); the index lookup that
       loses one queued key of a request. EB2's comparison sees each plant; EB3's comparison sees the skip plant.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_endbatch.py
"""
from __future__ import annotations

import importlib.util
import inspect
import random
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


RF = _shared("test_r2_resfix_shared_eb", "test_r2_resfix.py")         # RL1's draws (_draw, SEEDS), _same
HF, TT, TB, TK, OWN2 = RF.HF, RF.TT, RF.TB, RF.TK, RF.OWN2
TRM = HF.TRM
O = OWN2.O                                                            # r2_own2_ops in the r2tick package (OWN2's own)
RSC = RF.RSC
p3, ref = HF.p3, HF.ref                                               # the fixtures, by name
_same = RF._same
BS = TT.BS
BASE = "8ccf26247c5e95c8f14689bbfae59c02e83d8894"                    # the base commit (the per-request reference)
ROOT = HERE.parents[2]


# =====================================================================================================================
# EB1 the table op
# =====================================================================================================================
def _table(seed):
    """A random B2 table on the geometry of RL1's draw `seed` (its codes: the C_FREE-heavy draw on seed % 3 == 0, the
    pressure capacities on odd seeds), with random blocks, generations, stamps, owners and pin words."""
    d = RF._draw(seed)
    g = torch.Generator().manual_seed(9000 + seed)
    G = O.Geo(L=d.L, B=d.B, H=d.H, C=d.C, X=d.X, R_ver=d.R_ver, NBLK=d.NBLK, KEYS=1 + seed % 7)
    T = O.Table(G)
    sh = tuple(T.c["code"].shape)
    T.c["code"].copy_(d.dm.code.to(torch.int8))
    T.c["block"].copy_(d.dm.block.to(torch.int32))
    T.c["gen"].copy_(d.dm.gen.to(torch.int32))
    T.c["stamp"].copy_(d.dm.stamp)
    T.c["uid"].copy_(torch.randint(-1, d.B + 2, sh, generator=g).to(torch.int32))
    T.c["rd_last"].copy_(torch.randint(-1, 40, sh, generator=g))
    T.pin_w.copy_(torch.randint(0, 1 << 32, tuple(T.pin_w.shape), generator=g))
    if seed % 5 == 0:
        T.pin_w.zero_()                                               # no pin at all
    T.owner.copy_(torch.randint(1, 1000, (d.B,), generator=g).to(torch.int32))
    T.clock = 1 + seed
    return T


def _big_table(seed=0, B=216):
    """A table at the eval batch (B 216; small L, H, NPOS): every request may end in one tick."""
    g = torch.Generator().manual_seed(9500 + seed)
    G = O.Geo(L=2, B=B, H=2, C=4, X=4, R_ver=2, NBLK=10, KEYS=7)
    T = O.Table(G)
    sh = tuple(T.c["code"].shape)
    code = torch.randint(0, 6, sh, generator=g)
    code = torch.where(torch.randint(0, 10, sh, generator=g) < 7, torch.full_like(code, RSC.C_FREE), code)
    T.c["code"].copy_(code.to(torch.int8))
    T.pin_w.copy_(torch.randint(0, 1 << 32, tuple(T.pin_w.shape), generator=g))
    T.owner.copy_(torch.arange(1, B + 1, dtype=torch.int32))
    return T


def _eb1_check(T0, bs, rng):
    """end_owner per b (in a random order) vs end_owners(bs) vs end_owners(bs reversed): the differences and counts."""
    seq, bat, rev = T0.clone(), T0.clone(), T0.clone()
    order = list(bs)
    rng.shuffle(order)
    n_seq = sum(O.end_owner(seq, b) for b in order)
    n_bat = O.end_owners(bat, torch.tensor(bs, dtype=torch.int64))
    n_rev = O.end_owners(rev, torch.tensor(list(reversed(bs)), dtype=torch.int64))
    return seq.diff(bat), rev.diff(bat), (n_seq, n_bat, n_rev)


@pytest.mark.parametrize("seed", RF.SEEDS)
def test_EB1a_end_owners_equals_end_owner_per_request_in_any_order(seed):
    T0 = _table(seed)
    rng = random.Random(seed)
    B = T0.geo.B
    for _ in range(5):
        bs = rng.sample(range(B), rng.randint(1, B))
        d_seq, d_rev, (n_seq, n_bat, n_rev) = _eb1_check(T0, bs, rng)
        assert d_seq == [] and d_rev == [], (seed, bs, d_seq, d_rev)
        assert n_seq == n_bat == n_rev, (seed, bs, n_seq, n_bat, n_rev)
    assert O.end_owners(T0.clone(), torch.zeros(0, dtype=torch.int64)) == 0


def test_EB1b_the_draws_release_and_the_big_batch():
    """PRECONDITION of EB1a: the draws release slots (C_FREE-heavy draws too) and clear pins; and at B 216 all requests
    end in one call: equal to 216 end_owner calls."""
    rel_heavy = rel_other = pinned = 0
    for seed in RF.SEEDS:
        T0 = _table(seed)
        n = O.end_owners(T0.clone(), torch.arange(T0.geo.B, dtype=torch.int64))
        if seed % 3 == 0:
            rel_heavy += n
        else:
            rel_other += n
        pinned += int(T0.pin_w.ne(0).sum())
    assert rel_heavy > 0 and rel_other > 0 and pinned > 0, (rel_heavy, rel_other, pinned)
    T0 = _big_table()
    rng = random.Random(216)
    for bs in (list(range(216)), rng.sample(range(216), 150), [7]):
        d_seq, d_rev, (n_seq, n_bat, n_rev) = _eb1_check(T0, bs, rng)
        assert d_seq == [] and d_rev == [] and n_seq == n_bat == n_rev > 0, (len(bs), d_seq, d_rev, n_seq, n_bat)


@pytest.mark.parametrize("seed", range(0, 60, 4))
def test_EB1c_the_undo_log_restores_codes_and_pins(seed):
    """end_owner's undo contract: rollback restores the codes and the pins (not the owner); end_owners' too."""
    T0 = _table(seed)
    bs = list(range(T0.geo.B))[::-1]
    for fn in ("seq", "bat"):
        T = T0.clone()
        u = O.Undo()
        if fn == "seq":
            for b in bs:
                O.end_owner(T, b, undo=u)
        else:
            O.end_owners(T, torch.tensor(bs, dtype=torch.int64), undo=u)
        u.rollback()
        assert set(T.diff(T0)) <= {"owner"}, (seed, fn, T.diff(T0))


# =====================================================================================================================
# EB2 the service: per-request ends vs end_requests (index and scan), state after every round
# =====================================================================================================================
L2, H2, NBLK2, D2 = 2, 2, 12, 4


def _svc(B, seed, end_batch, C=4, X=4, R_ver=2):
    """A B2 service on the CPU stand-in (test_r2_residency.world's construction, with B requests); every request
    begun with uid 100 + b (a uid is not its batch slot) and a history bound."""
    g = torch.Generator().manual_seed(seed)
    host_k = [torch.randn(B, NBLK2 * BS, H2, D2, generator=g) for _ in range(L2)]
    host_v = [torch.randn(B, NBLK2 * BS, H2, D2, generator=g) for _ in range(L2)]
    cis = [torch.randn(B, NBLK2 * BS, H2, generator=g) for _ in range(L2)]
    n = (C + X + RF.N_TAIL) * BS
    store_k = [torch.zeros(B, n, H2, D2) for _ in range(L2)]
    store_v = [torch.zeros(B, n, H2, D2) for _ in range(L2)]
    store_b = [torch.zeros(B, n, H2) for _ in range(L2)]
    be = TT.Be()
    tp = TT.KTR.AdversarialTransport(be, host_k, host_v, store_k, store_v, BS, seed, mode="force_only", max_delay=3)
    gather = TT.RES.TorchGather(host_k, host_v, store_k, store_v)
    svc = OWN2.OwnResidency(L2, B, H2, C, X, R_ver, tp, gather, device="cpu", nblk=NBLK2, store_b=store_b, cis=cis,
                            block_bytes=BS * 2 * D2 * 4, host_wait=be.host_wait, checked=False, end_batch=end_batch)
    for b in range(B):
        svc.begin_request(b, 100 + b, hist_blocks=NBLK2 // 2)
    svc.eb_tags = {}
    return svc


def _snap(svc):
    """The full state an end can change (dicts as item lists: their order too; sets sorted)."""
    tags = getattr(svc, "eb_tags", {})
    T = svc.own
    return dict(table={n: t.clone() for n, t in T.tensors().items()}, clock=T.clock, owner_py=list(svc._owner_py),
                uid_b=list(svc.uid_b.items()), hist_ev=[(k, tags.get(id(v), "?")) for k, v in svc.hist_ev.items()],
                hist_from=list(svc.hist_from.items()), cancelled=sorted(repr(k) for k in svc.cancelled),
                queue=[(repr(k), l) for (k, l) in svc.queue], queued=sorted(repr(x) for x in svc.queued),
                key_of=[(k, repr(v)) for k, v in svc._key_of.items()],
                kslot=[(repr(k), v) for k, v in svc._kslot.items()], journal=list(svc.journal_counts.items()),
                ver=svc._ver)


def _round_ops(rng, alive, dead, r):
    """One round's host operations on the services (the same list for each)."""
    ops = []
    for u in alive:
        for _ in range(rng.randint(0, 3)):
            ops.append(("hist", u, rng.randrange(NBLK2)))
        for _ in range(rng.randint(0, 4)):
            ops.append(("queue", (u, r, rng.randrange(5)), rng.randrange(L2)))
        if rng.random() < 0.6:
            ops.append(("kslot", (u, 0, rng.randrange(7))))           # at most KEYS 7 distinct keys per request
        if rng.random() < 0.3:
            ops.append(("cancel", (u, r, 9)))
    if dead and rng.random() < 0.7:                                   # a key queued after its request ended
        ops.append(("queue", (rng.choice(dead), r, 1), rng.randrange(L2)))
    rng.shuffle(ops)
    return ops


def _apply(svc, ops):
    for op in ops:
        if op[0] == "hist":
            ev = TT.Ev(True)
            svc.eb_tags[id(ev)] = "e%d" % len(svc.eb_tags)
            svc.eb_tags.setdefault("_keep", []).append(ev)            # (the tag map keeps the event alive)
            svc.history_written(op[1], op[2], ev)
        elif op[0] == "queue":
            svc._queue(op[1], op[2])
        elif op[0] == "kslot":
            svc._key_slot(op[1])
        else:
            svc.cancel(op[1])


def _random_table(svc, seed, r):
    """The round's table codes and pins (the C_FREE-heavy draw on seed % 3 == 0, no pin on every third round)."""
    g = torch.Generator().manual_seed(100 * seed + r)
    T = svc.own
    sh = tuple(T.c["code"].shape)
    code = torch.randint(0, 6, sh, generator=g)
    if seed % 3 == 0:
        code = torch.where(torch.randint(0, 10, sh, generator=g) < 7, torch.full_like(code, RSC.C_FREE), code)
    pins = torch.randint(0, 1 << 32, tuple(T.pin_w.shape), generator=g)
    if r % 3 == 2:
        pins.zero_()
    return code.to(torch.int8), pins


def _eb2(B, seed, plant=None):
    """EB2's run: (the first difference or None, the coverage counts, the plant's hit count)."""
    rng = random.Random(1000 * seed + B)
    ref_s, idx_s, scan_s = _svc(B, seed, False), _svc(B, seed, True), _svc(B, seed, False)
    worlds = (("ref", ref_s), ("index", idx_s), ("scan", scan_s))
    hit = []
    if plant is not None:
        plant(idx_s, hit)
    alive, dead = [100 + b for b in range(B)], []
    cov = dict(rounds=0, multi=0, released=0, cancelled_by_end=0, hist_dropped=0)
    r = 0
    while alive:
        ops = _round_ops(rng, alive, dead, r)
        code, pins = _random_table(ref_s, seed, r)
        for _, s in worlds:
            _apply(s, ops)
            s.own.c["code"].copy_(code)
            s.own.pin_w.copy_(pins)
        k = 1 if r == 0 else (len(alive) if r >= 3 or len(alive) <= 2 else rng.randint(2, len(alive)))
        ends = rng.sample(alive, k)
        c0, h0, j0 = set(ref_s.cancelled), len(ref_s.hist_ev), ref_s.journal_counts.get("release:end", 0)
        order = list(ends)
        rng.shuffle(order)
        for u in order:                                               # the reference: one end_request per uid
            ref_s.end_request(u)
        rng.shuffle(order)
        idx_s.end_requests(order)
        rng.shuffle(order)
        scan_s.end_requests(order)
        cov["rounds"] += 1
        cov["multi"] += int(k >= 2)
        cov["released"] += ref_s.journal_counts.get("release:end", 0) - j0
        cov["cancelled_by_end"] += len(set(ref_s.cancelled) - c0)
        cov["hist_dropped"] += h0 - len(ref_s.hist_ev)
        base = _snap(ref_s)
        for name, s in worlds[1:]:
            d = _same(base, _snap(s), "%s.r%d" % (name, r))
            if d:
                return d, cov, len(hit)
        alive = [u for u in alive if u not in ends]
        dead += ends
        r += 1
    return None, cov, len(hit)


EB2_CASES = [(B, seed) for B in (3, 9, 64) for seed in range(24)]
_EB2_COV = {}


@pytest.mark.parametrize("B,seed", EB2_CASES, ids=["B%d-s%d" % c for c in EB2_CASES])
def test_EB2_end_requests_equals_end_request_per_uid_after_every_round(B, seed):
    d, cov, _ = _eb2(B, seed)
    _EB2_COV[(B, seed)] = cov
    assert d is None, ("the batched end differs from the per-request ends", B, seed, d)
    assert cov["rounds"] >= 2 and cov["multi"] >= 1, ("precondition: ends in 2+ rounds, 2+ ends in a round", cov)
    assert cov["released"] > 0, ("precondition: the ends released slots", cov)


def test_EB2z_the_draws_covered_cancels_and_history_drops():
    """PRECONDITION of EB2 (run after it), over ALL draws, not per draw: a small draw (B 3) can end its requests
    with no history to drop (B3-s20: hist_dropped 0). Each path must be reached in at least 1/4 of the draws."""
    if len(_EB2_COV) != len(EB2_CASES):
        pytest.skip("EB2 did not run in this session (run the whole file)")
    for key in ("cancelled_by_end", "hist_dropped"):
        n = sum(1 for c in _EB2_COV.values() if c[key] > 0)
        assert n >= len(EB2_CASES) // 4, ("precondition: %s reached in too few draws" % key, n, len(EB2_CASES))


def test_EB2b_the_eval_batch_ends_in_one_call():
    """B 216 (the eval batch): the first round ends 1, then the rest in 2+ rounds, the last round all that remain."""
    d, cov, _ = _eb2(216, 5)
    assert d is None, d
    assert cov["multi"] >= 1 and cov["released"] > 0, cov


def test_EB2c_refusals_before_any_change():
    s = _svc(4, 0, True)
    before = _snap(s)
    with pytest.raises(KeyError):
        s.end_requests([100, 999])                                    # an unknown uid: no change
    assert _same(before, _snap(s)) is None
    with pytest.raises(ValueError):
        s.end_requests([101, 101])                                    # a repeated uid: no change
    assert _same(before, _snap(s)) is None
    s.end_requests([])
    assert _same(before, _snap(s)) is None


# =====================================================================================================================
# EB3 the round
# =====================================================================================================================
S8 = TT.S_TINY + [520, 545, 590, 620]


def _round(p3, toks, cfg, forced, end_batch, max_new=None, S_list=None, first=None, setup=None):
    """One B2 round (HF._run_hf's per-tick record) with: the full state at every poll (the state after the previous
    tick's ends) and after the round; every end call with its tick (the number of ticks recorded before it)."""
    S_list = list(S_list or TT.S_TINY)
    B = len(S_list)
    kw = dict(checked=False)
    if end_batch:
        kw["end_batch"] = True
    eng = TT.make_res_engine(TT.make_ops(p3), kind="b2", S_list=S_list, svc_kw=kw, **TB.CFG[cfg])
    eng.end_batch = bool(end_batch)
    world = eng.res_world
    svc = world.svc
    svc.eb_tags = {}
    evs = []

    def new_event():                                                  # named by creation order (both rounds alike)
        e = TT.Ev(True)
        svc.eb_tags[id(e)] = "e%d" % len(evs)
        evs.append(e)
        return e
    eng.backend.new_event = new_event
    pages, ticks, snaps, calls = [], [], [], dict(end_request=[], end_requests=[])
    resolve0, poll0, er0, ers0 = svc.resolve, svc.poll, svc.end_request, svc.end_requests

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)]
                          for i in range(R)]))
        return p

    def poll():
        snaps.append(_snap(svc))
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
                          counters=TRM._counters(svc), readers=TRM._reader_tags(svc, svc.eb_tags), memory=svc.memory(),
                          kept=HF._kept_counters(svc), queue=[(repr(k), l) for (k, l) in svc.queue],
                          queued=sorted(repr(x) for x in svc.queued)))
        pages.clear()
    eng.timer = TK.TickTimer(cuda=False)
    if setup is not None:
        setup(eng, svc)

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    reject_at, _ = TK.forced_schedule(3) if forced else (None, [])
    first = list(first) if first is not None else [int(toks[0, b]) for b in range(B)]
    rnd, err = None, None
    try:
        rnd = TK.run_round(eng, first, list(max_new or [TT.STEPS] * B), K=3, LAMBDA=7,
                           teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                            # noqa: BLE001 - a difference at its tick
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    return SimpleNamespace(eng=eng, svc=svc, ticks=ticks, rnd=rnd, err=err, evs=evs, world=world, snaps=snaps,
                           final=_snap(svc), calls=calls)


def _ends_by_tick(calls):
    out = {}
    for (t, u) in calls["end_request"]:
        out.setdefault(t, []).append(u)
    for (t, us) in calls["end_requests"]:
        out.setdefault(t, []).extend(us)
    return {t: sorted(v) for t, v in out.items()}


STAGGER = [TT.STEPS - 5 * (b % 4) for b in range(8)]
EB3_CASES = dict(                                                     # (cfg, forced, max_new, S_list, first)
    default_forced_same=("default", True, None, None, None),
    default_forced_stagger=("default", True, STAGGER[:4], None, None),
    pressure_forced_stagger=("pressure", True, STAGGER[:4], None, None),
    default_free_same=("default", False, None, None, None),
    pressure_free_b8_stagger=("pressure", False, STAGGER, S8, TT.FIRST * 2),
    default_free_b8_same=("default", False, None, S8, TT.FIRST * 2),
)
_EB3_COV = {}


@pytest.mark.parametrize("name", sorted(EB3_CASES))
def test_EB3_the_batched_round_equals_the_per_request_round(p3, ref, name):
    toks, _ = ref
    cfg, forced, mn, sl, first = EB3_CASES[name]
    base = _round(p3, toks, cfg, forced, False, mn, sl, first)
    eb = _round(p3, toks, cfg, forced, True, mn, sl, first)
    assert base.err is None and len(base.ticks) > 10, ("precondition: the reference round ran", base.err)
    d = HF._diff(base, eb, HF.ALL_FIELDS + ("kept", "queue", "queued"))
    assert d is None, ("--end-batch differs from the per-request ends", d, eb.err)
    assert len(base.snaps) == len(eb.snaps) > 10
    assert _same(base.snaps, eb.snaps, "snaps") is None, _same(base.snaps, eb.snaps, "snaps")
    assert _same(base.final, eb.final, "final") is None, _same(base.final, eb.final, "final")
    HF._same_end(base, eb)
    assert not base.svc.uid_b and not eb.svc.uid_b
    # the calls: per request (reference) vs one call per tick with ends (batched), the same requests at the same ticks
    assert base.calls["end_requests"] == [] and eb.calls["end_request"] == []
    eb_ticks = [t for (t, _) in eb.calls["end_requests"]]
    assert len(eb_ticks) == len(set(eb_ticks)), eb.calls["end_requests"]
    assert _ends_by_tick(base.calls) == _ends_by_tick(eb.calls)
    by = _ends_by_tick(base.calls)
    assert sum(len(v) for v in by.values()) == len(base.rnd.reqs)
    _EB3_COV[name] = (len(by), max(len(v) for v in by.values()))
    print("REPORT EB3 %s: ticks %d, end ticks %d, most ends in one tick %d" % (name, len(base.ticks), *_EB3_COV[name]))


def test_EB3z_the_rounds_covered_many_end_ticks_and_many_ends_in_one_tick():
    """PRECONDITION of EB3 (run after it): some round ends requests in 2+ ticks; some tick ends 2+ requests."""
    if len(_EB3_COV) != len(EB3_CASES):
        pytest.skip("EB3 did not run in this session (run the whole file)")
    assert max(n for (n, _) in _EB3_COV.values()) >= 2, _EB3_COV
    assert max(m for (_, m) in _EB3_COV.values()) >= 2, _EB3_COV


# =====================================================================================================================
# EB4 flag off unchanged
# =====================================================================================================================
def _base_text(path):
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (BASE, path)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("the base commit %s is not in this checkout" % BASE[:7])
    return r.stdout


def _block(text, start, end_marker="\n    def "):
    i = text.index(start)
    j = text.index(end_marker, i + len(start))
    return text[i:j]


def test_EB4a_the_per_request_sources_are_the_base_text():
    own2 = (ROOT / "nosi/nosi/r2_own2.py").read_text()
    ops = (ROOT / "nosi/nosi/r2_own2_ops.py").read_text()
    tick = (ROOT / "nosi/nosi/r2_tick.py").read_text()
    b_own2, b_ops, b_tick = (_base_text(p) for p in ("nosi/nosi/r2_own2.py", "nosi/nosi/r2_own2_ops.py",
                                                     "nosi/nosi/r2_tick.py"))
    old = _block(b_own2, "    def end_request(self, uid) -> None:")
    assert old in own2 and "O.end_owner(self.own, b)" in old
    old = _block(b_ops, "def end_owner(T: Table, b: int", "\n\n\ndef ")
    assert old in ops
    old = _block(b_tick, "    def end_request(self, b: int) -> None:\n        if b not in self.ended:", "\n\n\n")
    assert old in tick
    old_loop = ["for b, r in enumerate(rnd.reqs):", "if r.done:", "engine.end_request(b)"]
    i = b_tick.index('engine.timer.host("h_end_request"')
    assert all(x in b_tick[i - 300:i] for x in old_loop)
    rr = inspect.getsource(TK.run_round)
    j = rr.index("else:", rr.index('getattr(engine, "end_batch", False)'))
    assert [ln.strip() for ln in rr[j:].splitlines()[1:4]] == old_loop, rr[j:j + 300]


def test_EB4b_a_round_without_the_flag_ends_per_request_and_keeps_no_index(p3, ref):
    toks, _ = ref
    base = _round(p3, toks, "default", True, False)
    assert base.err is None
    assert base.svc.end_batch is False and base.svc._qidx is None and base.eng.end_batch is False
    assert base.calls["end_requests"] == []
    assert sorted(u for (_, u) in base.calls["end_request"]) == list(range(len(base.rnd.reqs)))
    assert inspect.signature(OWN2.OwnResidency.__init__).parameters["end_batch"].default is False


def test_EB4c_the_harness_flag():
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    assert G.parse_args(base).end_batch is False
    assert G.parse_args(base + ["--backends", "b2", "--X", "64", "--end-batch"]).end_batch is True
    with pytest.raises(SystemExit):
        G.parse_args(base + ["--end-batch"])                          # b2 only
    src = inspect.getsource(G.make_engine)
    assert 'end_batch=bool(getattr(a, "end_batch", False))' in src
    assert 'eng.end_batch = bool(getattr(a, "end_batch", False))' in src
    assert 'res_dev=bool(getattr(a, "res_dev", False))' in src         # [resdev]'s wiring is kept


def test_EB4d_the_step_cell_static():
    text = (TT.EFF / "r2_tick_sg5_step.sh").read_text()
    assert "        evalfast2eb) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "|evalacc216f3lia|evalfast2eb" in text                      # [f3all] (was "|evalfast2rdp|evalfast2eb]")
    root = next(ln for ln in text.splitlines() if ln.startswith("NOSI_EVAL=")).split()[0][len("NOSI_EVAL="):]
    assert root == "/mnt/beegfs/ojerbi/nosi-r2-f3all"                  # [f3all] (was nosi-r2-endbatch)
    reg = text[text.index("# REGISTERED for evalfast2eb"):]
    reg = reg[:reg.index("\n#\n")]
    for tag in ("STAMP 2026-10-23T18:05:38Z, before any run", "EB-O", "O <= 350 ms", "EB-EFF", "point 233, band [226, 245]", "EB-MED",
                "326 +- 6", "EB-MEAN", "EB-FALS", "O > 2 s", "EB-MEM", "peak_gb <= 84", "nosi-r2-endbatch"):
        assert tag in reg, tag
    blk = text[text.index("    evalfast2eb) MODES=fast"):text.index("    evalfast2c192) MODES=fast")]
    assert "BANDS=315:340:245:268:226:245" in blk
    assert "--b2-fewer-reads --res-lean --end-batch" in blk and "--wait-index" in blk and "--res-dev" not in blk


# =====================================================================================================================
# EB5 negative controls (plants at the site under test only)
# =====================================================================================================================
def _plant_skip_one(monkeypatch, hit):
    """end_owners skips the LAST request of its set (that request keeps its slots, pins and owner)."""
    real = O.end_owners

    def planted(T, bt, undo=None):
        if bt.numel() >= 1:
            hit.append(1)                                             # the owner of the skipped b stays: a change
        return real(T, bt[:-1], undo)
    monkeypatch.setattr(O, "end_owners", planted)


def _plant_write_order(monkeypatch, hit):
    """end_owners reads the codes in the order of bt but writes them in the reverse order (each request gets another
    request's new codes). Counted where the swap changes a value (two requests with different new codes)."""
    def planted(T, bt, undo=None):
        code = T.c["code"]
        sub = code.index_select(1, bt)
        m = (sub == RSC.C_LRU) | (sub == RSC.C_PO) | (sub == RSC.C_VRES)
        new = torch.where(m, torch.full_like(sub, RSC.C_RELEASING), sub)
        if not torch.equal(new, new.flip(1)):
            hit.append(1)
        code.index_copy_(1, bt.flip(0), new)
        T.pin_w.index_fill_(2, bt, 0)
        T.owner.index_fill_(0, bt, -1)
        return int(m.sum())
    monkeypatch.setattr(O, "end_owners", planted)


class _LosesLastKey(dict):
    """The per-request index whose lookup loses the last queued key of a request (end_requests' lookup only)."""

    def __init__(self, base, svc, hit):
        super().__init__(base)
        self.svc, self.hit = svc, hit

    def get(self, k, default=None):
        v = dict.get(self, k)
        if not v:
            return default if v is None else v
        keys = list(v)
        if keys[-1] not in self.svc.cancelled:
            self.hit.append(1)                                        # the lost key changes the cancelled set
        return dict.fromkeys(keys[:-1])


PLANTS = dict(skip_one="end_owners", write_order="end_owners", lose_key="index")


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_EB5a_eb2_sees_each_plant(monkeypatch, name):
    """The plant acts (L34: counted per run) and EB2's comparison reports a difference on every run where it acted,
    and on at least 5 runs."""
    hit_runs, seen = 0, 0
    for (B, seed) in [(9, s) for s in range(12)]:
        hit = []
        if name == "lose_key":
            def plant(svc, h):
                svc._qidx = _LosesLastKey(svc._qidx, svc, h)
            d, _, n_hit = _eb2(B, seed, plant=plant)
        else:
            with monkeypatch.context() as mp:
                (_plant_skip_one if name == "skip_one" else _plant_write_order)(mp, hit)
                d, _, _ = _eb2(B, seed)
            n_hit = len(hit)
        if n_hit:
            hit_runs += 1
            assert d is not None, ("the planted %s acted and was not seen" % name, B, seed)
            seen += 1
        else:
            assert d is None, ("a difference without the plant acting", name, B, seed, d)
    assert hit_runs >= 5 and seen == hit_runs, (name, hit_runs, seen)
    print("REPORT EB5 %s: acted on %d runs, seen on %d" % (name, hit_runs, seen))


def test_EB5b_eb3_sees_the_skipped_free(p3, ref, monkeypatch):
    toks, _ = ref
    base = _round(p3, toks, "default", True, False)
    hit = []
    _plant_skip_one(monkeypatch, hit)
    bad = _round(p3, toks, "default", True, True)
    assert hit, "precondition: the plant acted (a batched end ran)"
    differs = (HF._diff(base, bad, HF.ALL_FIELDS + ("kept", "queue", "queued")) is not None
               or _same(base.snaps, bad.snaps) is not None or _same(base.final, bad.final) is not None)
    assert differs, "the skipped free was not seen"
