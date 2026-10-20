"""CPU tests F1-F9 of R2 step 7, piece 5 (docs/R2_STEP7_FETCH_RETENTION.md section 12): nosi/nosi/r2_residency.py,
the residency service of the spec round, over K3X's ownership and transports VENDORED byte-identical in
nosi/nosi/kstep/ (91e6e47). CPU tensors, K3X's AdversarialTransport (seeded delays, reordering, bursts; "force_only" for
the scripted tests), CPU events with a done flag. Each test has a negative control (a K3X mutation switch or a defect
subclass) that the same check must catch.

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_residency.py
"""
from __future__ import annotations

import collections
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi/nosi"
PKG = "r2res"
K3X = "91e6e47"
VENDORED = ("__init__", "errors", "backend", "executor", "ownership", "transfer", "mutations")


def _load(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


RES = _load("r2_residency")
CORE = _load("r2_spec_core")
OWN = sys.modules[PKG + ".kstep.ownership"]
TR = sys.modules[PKG + ".kstep.transfer"]
MU = sys.modules[PKG + ".kstep.mutations"]
ER = sys.modules[PKG + ".kstep.errors"]
Row = RES.Row
L, B, H, D, BS, NB = 2, 2, 2, 4, 64, 40


class Ev:
    def __init__(self, done=False):
        self.done = done


class Be:
    """The CPU backend shape the K3X transports use: host_wait forces the event (as cpupack_core's CpuBackend)."""

    def __init__(self):
        self.waits = 0

    def host_wait(self, ev):
        if ev is not None and not ev.done:
            ev.done = True
            self.waits += 1


def world(C=4, X=4, R_ver=2, seed=0, mode="force_only", mut=None, cls=None, max_delay=3):
    g = torch.Generator().manual_seed(seed)
    host_k = [torch.randn(B, NB * BS, H, D, generator=g) for _ in range(L)]
    host_v = [torch.randn(B, NB * BS, H, D, generator=g) for _ in range(L)]
    cis = [torch.randn(B, NB * BS, H, generator=g) for _ in range(L)]
    n = (C + X + RES.N_TAIL) * BS
    store_k = [torch.zeros(B, n, H, D) for _ in range(L)]
    store_v = [torch.zeros(B, n, H, D) for _ in range(L)]
    store_b = [torch.zeros(B, n, H) for _ in range(L)]
    be = Be()
    tp = TR.AdversarialTransport(be, host_k, host_v, store_k, store_v, BS, seed, mode=mode, max_delay=max_delay)
    gather = RES.TorchGather(host_k, host_v, store_k, store_v)
    svc = (cls or RES.Residency)(L, B, H, C, X, R_ver, tp, gather, store_b=store_b, cis=cis,
                                 block_bytes=BS * 2 * D * 4, host_wait=be.host_wait, mut=mut)
    return types.SimpleNamespace(svc=svc, tp=tp, be=be, host_k=host_k, host_v=host_v, cis=cis, store_k=store_k,
                                 store_v=store_v, store_b=store_b)


def S(*rows):
    """The selection tensor (H, R, k) int64 of rows given as [head 0 blocks, head 1 blocks], -1 padded."""
    k = max([len(bl) for r in rows for bl in r] + [1])
    t = torch.full((H, len(rows), k), -1, dtype=torch.int64)
    for i, r in enumerate(rows):
        for h in range(H):
            if r[h]:
                t[h, i, :len(r[h])] = torch.tensor(r[h])
    return t


def land_all(w):
    for tr in list(w.svc.inflight.values()):
        if not tr.delivered and tr in w.tp.pending:
            w.tp.force(tr)
    w.svc.poll()


def slot_holds(w, l, b, h, idx, blk):
    s0, r0 = idx * BS, blk * BS
    return (torch.equal(w.store_k[l][b, s0:s0 + BS, h], w.host_k[l][b, r0:r0 + BS, h])
            and torch.equal(w.store_v[l][b, s0:s0 + BS, h], w.host_v[l][b, r0:r0 + BS, h])
            and torch.equal(w.store_b[l][b, s0:s0 + BS, h], w.cis[l][b, r0:r0 + BS, h]))


def resident_content_ok(w, l):
    """Every resident slot (LRU / PO / VRES) holds the host truth of its block (K, V and bias)."""
    for (b, h), st in w.svc.pools[l].streams.items():
        for blk, idx in list(st.lru.items()) + list(st.po.items()) + list(st.vres.items()):
            if not slot_holds(w, l, b, h, idx, blk):
                return False
    return True


def reads_ok(w, l, plan):
    """Every historical page the plan lists is a resident slot of the listed block with the host truth."""
    for (b, h, idx, gen, kind) in plan.reads:
        s = w.svc.stream(l, b, h).slots[idx]
        if s.gen != gen or s.state not in (OWN.LRU, OWN.PO, OWN.VRES) or not slot_holds(w, l, b, h, idx, s.block):
            return False
    return True


def dedup_ok(svc):
    c = collections.Counter((tr.layer, tr.b, tr.h, tr.block) for tr in svc.inflight.values())
    return all(v == 1 for v in c.values())


class InflightReadable(RES.Residency):
    """DEFECT: a draft row reads an in-flight block."""

    def _draft_readable(self, st, l, b, h, t):
        idx = st.resident(t)
        if idx is None and self.pending.get((l, b, h, t)) is not None:
            return self.pending[(l, b, h, t)].slot
        return idx


class NoDedup(RES.Residency):
    """DEFECT: an in-flight block is not found, so it is requested again (a duplicate transfer)."""

    def _in_flight(self, l, b, h, t):
        return None


class DraftTakesReserve(RES.Residency):
    """DEFECT: a draft fetch borrows the verifier reserve."""

    def _draft_can_take(self, pool, st):
        return pool.can_take(st, RES.VERIFY)


# =====================================================================================================================
# F1 the draft row
# =====================================================================================================================
def _f1(cls=None):
    w = world(C=8, X=4, R_ver=1, cls=cls)
    svc = w.svc
    svc.begin_request(0, "u0")
    k1, k2 = ("u0", 1, 1), ("u0", 1, 2)
    p1 = svc.resolve(0, [Row(0, "d", k1, 300, 10)], S([[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert len(p1.issued) == 8 and p1.reads == [] and p1.masked == 2 and p1.draft_pairs == 2
    for tr in p1.issued:
        if tr.block in (0, 1):
            w.tp.force(tr)
    svc.poll()
    rows = [Row(0, "d", k1, 301, 10), Row(0, "d", k2, 302, 10)]
    p2 = svc.resolve(0, rows, S([[0, 2, 4], [1, 3, 5]], [[0, 1, 4, 5], [0, 2, 4]]))
    return w, p2


def test_F1_draft_reads_ready_blocks_only_and_requests_each_miss_once():
    """F1: a draft row lists its tail pages and ONLY the selected blocks that are resident with an observed arrival,
    in ascending block id; every miss is requested once per (stream, block) across the rows and heads of a tick; an
    in-flight block is not requested again. Negative controls: a row that reads an in-flight block, and a service that
    does not find in-flight blocks (duplicate transfers), are both caught."""
    w, p2 = _f1()
    assert p2.blocks == [[[0], [1]], [[0, 1], [0]]]
    assert reads_ok(w, 0, p2) and dedup_ok(w.svc)
    assert sorted((tr.h, tr.block) for tr in p2.issued) == [(0, 4), (0, 5), (1, 4), (1, 5)]
    assert p2.masked == 4 and p2.draft_pairs == 4 and p2.refused == 0
    w2, q2 = _f1(InflightReadable)
    assert not reads_ok(w2, 0, q2)
    w3, q3 = _f1(NoDedup)
    assert not dedup_ok(w3.svc)


# =====================================================================================================================
# F2 retention
# =====================================================================================================================
def _f2(mut=None):
    """Returns the violations of the retention property (empty = it held)."""
    w = world(C=2, X=4, R_ver=1, mut=mut)
    svc = w.svc
    svc.begin_request(0, "u0")
    k1, k2 = ("u0", 1, 1), ("u0", 1, 2)
    svc.resolve(0, [Row(0, "d", k1, 300, 10)], S([[0, 1], []]))
    land_all(w)
    svc.resolve(0, [Row(0, "d", k2, 301, 10)], S([[0, 2, 3, 4], []]))       # 0 a hit (k2 protects it too)
    land_all(w)
    st = svc.stream(0, 0, 0)
    bad = []
    if not ({0, 1} <= set(st.po)):
        bad.append("0 and 1 left the LRU but are not protected-only")
    p = svc.resolve(0, [Row(0, "v", k1, 300, 10)], S([[0, 1], []]))
    if p.looks[0]["ready"] != [0, 1]:
        bad.append("k1's verifier did not find 0 and 1 ready: %s" % (p.looks[0],))
    ev = Ev()
    svc.read_done(0, p, ev)
    svc.decided(k1)
    svc.poll()
    if 1 not in st.po or st.slots[st.po[1]].state != OWN.PO:
        bad.append("1 was released before k1's reader event completed")
    ev.done = True
    svc.poll()
    if 0 not in st.po:
        bad.append("0 lost its slot although k2 still protects it")
    if st.resident(1) is not None:
        bad.append("1 is still resident after its last reference and reader ended")
    return bad


def test_F2_requested_blocks_stay_until_the_verifier_read_them():
    """F2: blocks requested by a live cohort, the hits included, keep their slots as protected-only when the LRU
    evicts them; the verifier finds them ready; its references drop only after its reader event; a block that another
    live cohort still protects keeps its slot. Negative controls (K3X mutations): releasing before the reader gate,
    and releasing every key of a block when one cohort ends, are caught."""
    assert _f2() == []
    assert any("before k1's reader" in x for x in _f2(MU.Mutations(prot_release_no_reader_gate=True)))
    assert any("k2 still protects" in x for x in _f2(MU.Mutations(release_drops_all_keys=True)))


# =====================================================================================================================
# F3 the late transfer of a cancelled cohort
# =====================================================================================================================
def _f3(mut=None):
    w = world(C=8, X=2, R_ver=1, mut=mut)                              # C 8: no LRU eviction of the late arrivals
    svc = w.svc
    svc.begin_request(0, "u0")
    k1, k2 = ("u0", 1, 1), ("u0", 1, 2)
    p = svc.resolve(0, [Row(0, "d", k1, 300, 30)], S([[5, 6], [7]]))
    trs = list(p.issued)
    svc.cancel(k1)
    svc.poll()
    svc.poll()
    held = all(svc.stream(0, 0, tr.h).slots[tr.slot].state == OWN.INFLIGHT
               and svc.stream(0, 0, tr.h).slots[tr.slot].gen == tr.gen for tr in trs)
    refused = False
    try:
        svc.resolve(0, [Row(0, "d", k1, 301, 30)], S([[5], []]))
    except ER.OwnershipViolation:
        refused = True
    svc.resolve(0, [Row(0, "d", k2, 302, 30)], S([[10, 11, 12, 13], [10, 11]]))
    land_all(w)
    return w, held, refused, trs


def test_F3_a_late_transfer_of_a_cancelled_cohort_lands_in_its_own_slot():
    """F3: a cancelled cohort's transfers keep their in-flight slots until they land; the cancelled key can no longer
    resolve (read); other cohorts take other slots; at arrival the blocks enter the LRU (history is immutable) with the
    host truth, counted as arrivals after a cancel. Negative control: freeing the in-flight destination at the cancel
    (K3X free_on_cancel) lets another cohort re-take the slot under the transfer: caught (StaleGeneration at the late
    admission, or a slot without the host truth)."""
    w, held, refused, trs = _f3()
    assert held and refused
    assert resident_content_ok(w, 0)
    assert all(w.svc.stream(0, 0, tr.h).resident(tr.block) is not None for tr in trs)
    assert w.svc.report()["arrivals_after_cancel"] == 3
    caught = False
    try:
        w2, held2, _, _ = _f3(MU.Mutations(free_on_cancel=True))
        caught = (not held2) or not resident_content_ok(w2, 0)
    except ER.OwnershipViolation:                         # StaleGeneration is an OwnershipViolation
        caught = True
    assert caught


# =====================================================================================================================
# F4 slot reuse and the last reader
# =====================================================================================================================
def _f4(mut=None):
    """Returns the violations (empty = the slot was not reused, written or freed before its last reader ended)."""
    w = world(C=1, X=1, R_ver=0, mut=mut)
    svc = w.svc
    svc.begin_request(0, "u0")
    k1, k1b = ("u0", 1, 1), ("u0", 1, 2)
    svc.resolve(0, [Row(0, "d", k1, 300, 10)], S([[3], []]))
    land_all(w)
    st = svc.stream(0, 0, 0)
    idx = st.resident(3)
    ev1, ev2 = Ev(), Ev()
    p = svc.resolve(0, [Row(0, "v", k1, 300, 10)], S([[3], []]))
    svc.read_done(0, p, ev1)
    p = svc.resolve(0, [Row(0, "v", k1b, 301, 10)], S([[3], []]))
    svc.read_done(0, p, ev2)
    svc.end_request("u0")                                  # the slot leaves every role while ev2 is pending
    ev1.done = True
    svc.poll()
    bad = []
    if st.slots[idx].state != OWN.RELEASING:
        bad.append("slot %d is %s before its last reader ended" % (idx, st.slots[idx].state))
    svc.begin_request(0, "u1")
    svc.resolve(0, [Row(0, "d", ("u1", 1, 1), 400, 10)], S([[7, 8], []]))
    if not slot_holds(w, 0, 0, 0, idx, 3):
        bad.append("slot %d was written under its pending reader" % idx)
    ev2.done = True
    svc.poll()
    if st.slots[idx].state != OWN.FREE and st.slots[idx].block == 3:
        bad.append("slot %d not FREE after its last reader ended" % idx)
    return bad


def test_F4_a_slot_is_reused_only_after_its_last_reader():
    """F4: a slot that left every role while a reader is pending stays RELEASING (not FREE, not re-taken, its rows and
    bias unchanged) until its LAST reader event completed, also when an earlier reader already completed; then it is
    FREE. Negative controls (K3X mutations): no reader wait, and the first reader taken for the last, are caught."""
    assert _f4() == []
    assert _f4(MU.Mutations(no_reader_wait=True)) != []
    assert _f4(MU.Mutations(release_first_reader=True)) != []


# =====================================================================================================================
# F5 the verifier
# =====================================================================================================================
def _f5(cls=None):
    w = world(C=4, X=4, R_ver=2, cls=cls)
    svc = w.svc
    svc.begin_request(0, "u0")
    k1 = ("u0", 1, 1)
    p = svc.resolve(0, [Row(0, "d", k1, 300, 10)], S([[0, 1, 2], [0]]))
    for tr in p.issued:
        if tr.block in (0, 1):
            w.tp.force(tr)
    svc.poll()
    t2 = [tr for tr in p.issued if tr.h == 0 and tr.block == 2][0]
    pv = svc.resolve(0, [Row(0, "v", k1, 300, 10)], S([[0, 1, 2, 3], [0, 5]]))
    return w, pv, t2


def test_F5_verifier_ready_in_flight_new():
    """F5: the verifier's selection splits into ready (resident), in flight (another requester's transfer: demand
    merged, waited, never fetched twice) and new (urgent: a slot taken, the reserve allowed, gathered before the read);
    every page holds the host truth; the urgent blocks become verifier-resident, the waited one enters the LRU (draft
    and verify demand). Negative control: a service that does not see in-flight blocks gathers an in-flight block
    again (an urgent duplicate): caught."""
    w, pv, t2 = _f5()
    looks = {lk["h"]: lk for lk in pv.looks}
    assert (looks[0]["ready"], looks[0]["in_flight"], looks[0]["new"]) == ([0, 1], [2], [3])
    assert (looks[1]["ready"], looks[1]["in_flight"], looks[1]["new"]) == ([0], [], [5])
    assert pv.waited == [t2] and t2.demand == {RES.DRAFT, RES.VERIFY}
    assert sorted((tr.h, tr.block, tr.kind) for tr in pv.urgent) == [(0, 3, RES.VERIFY), (1, 5, RES.VERIFY)]
    assert pv.blocks[0] == [[0, 1, 2, 3], [0, 5]] and reads_ok(w, 0, pv)
    st0, st1 = w.svc.stream(0, 0, 0), w.svc.stream(0, 0, 1)
    assert 3 in st0.vres and 5 in st1.vres and 2 in st0.lru
    w2, pv2, _ = _f5(NoDedup)
    assert any(tr.block == 2 and tr.h == 0 for tr in pv2.urgent)


def test_F5b_the_verifier_reserve():
    """F5b (K5): a draft fetch may borrow a shared extra only while more than R_ver extras are free (the third miss is
    refused and counted, the row continues masked); the verifier's urgent miss can still take the reserve. Negative
    control: a draft that borrows the reserve leaves fewer than R_ver extras free: caught."""
    for cls, ok in ((None, True), (DraftTakesReserve, False)):
        w = world(C=1, X=2, R_ver=1, cls=cls)
        svc = w.svc
        svc.begin_request(0, "u0")
        k1 = ("u0", 1, 1)
        p = svc.resolve(0, [Row(0, "d", k1, 300, 10)], S([[0, 1, 2], []]))
        held = svc.pools[0].free_extras() >= svc.R_ver
        assert held == ok
        if ok:
            assert p.refused == 1 and p.masked == 1 and svc.report()["draft_refused"] == 1
            land_all(w)
            pv = svc.resolve(0, [Row(0, "v", k1, 300, 10)], S([[0, 1, 2], []]))
            assert [(tr.block, tr.kind) for tr in pv.urgent] == [(2, RES.VERIFY)] and reads_ok(w, 0, pv)


# =====================================================================================================================
# F6 end to end under the adversarial transport
# =====================================================================================================================
def _f6(seed=1, mut=None, steps=24):
    """A scripted run: 2 requests, 2 layers, draft rows every tick, a verify row every third tick, seeded delays /
    reordering / bursts, reader events completed two ticks late. Returns (world, the verify checks, samples)."""
    w = world(C=8, X=48, R_ver=8, seed=seed, mode="delay", mut=mut, max_delay=4)
    svc = w.svc
    g = torch.Generator().manual_seed(seed + 100)
    uids = ["u0", "u1"]
    for b, u in enumerate(uids):
        svc.begin_request(b, u)
    pos0, hb = 1000, 1000 // 64                                          # history boundary: block 15
    tail = svc.tail0                                                     # the exact tail page of block 15
    for l in range(L):
        for b in range(B):
            w.store_k[l][b, tail * BS:(tail + 1) * BS] = w.host_k[l][b, hb * BS:(hb + 1) * BS]
            w.store_v[l][b, tail * BS:(tail + 1) * BS] = w.host_v[l][b, hb * BS:(hb + 1) * BS]
            w.store_b[l][b, tail * BS:(tail + 1) * BS] = w.cis[l][b, hb * BS:(hb + 1) * BS]
    readers, checks, samples = [], [], []                                # samples: (the service's, an independent scan)
    for t in range(steps):
        svc.poll()
        samples.append((svc.memory(), _scan(svc)))
        m = t // 3 + 1
        verify = t % 3 == 2
        for l in range(L):
            rows, sel = [], []
            for b, u in enumerate(uids):
                x = int(torch.randint(0, hb - 6, (1,), generator=g))
                rows.append(Row(b, "d", (u, 1, m + 1 if verify else m), pos0 + t, hb, [(hb, tail)]))
                sel.append([list(range(x, x + 6)) + [hb] for _ in range(H)])
                if verify:
                    v = sorted(set(int(z) for z in torch.randint(0, hb, (8,), generator=g)))
                    rows.append(Row(b, "v", (u, 1, m), pos0 + t, hb, [(hb, tail)]))
                    sel.append([v + [hb] for _ in range(H)])
            plan = svc.resolve(l, rows, S(*sel))
            samples.append((svc.memory(), _scan(svc)))
            vi = [i for i, r in enumerate(rows) if r.kind == "v"]
            if vi:
                Rv = len(rows)
                q = torch.randn(Rv, 2 * H, D, generator=g)
                out = CORE.paged_attention_reference(q, w.store_k[l], w.store_v[l], w.store_b[l], plan.seq_idx,
                                                     plan.page_table, plan.num_pages, plan.last_len, D ** -0.5)
                ideal = torch.full_like(plan.page_table, -1)
                for i in range(Rv):
                    for h in range(H):
                        bl = plan.blocks[i][h]
                        ideal[i, h, :len(bl)] = torch.tensor(bl, dtype=torch.int32)
                ref = CORE.paged_attention_reference(q, w.host_k[l], w.host_v[l], w.cis[l], plan.seq_idx, ideal,
                                                     plan.num_pages, plan.last_len, D ** -0.5)
                checks.append(dict(content=reads_ok(w, l, plan),
                                   equal=all(torch.equal(out[i], ref[i]) for i in vi),
                                   full=all(plan.blocks[i][h] == [z for z in sel[i][h] if 0 <= z <= hb]
                                            for i in vi for h in range(H))))
            ev = Ev()
            readers.append((t, ev))
            svc.read_done(l, plan, ev)
        for (tt, ev) in readers:
            if tt <= t - 2:
                ev.done = True
        if verify:
            for u in uids:
                svc.decided((u, 1, m))
    return w, checks, samples


def test_F6_end_to_end_content_and_attention_under_adversarial_delivery():
    """F6: under seeded delays, reordering and bursts, every verifier row reads exactly its selection (every selected
    historical block and the tail), every page holds the host truth (K, V, bias), and the paged attention over the
    store is torch.equal to the same attention over an all-resident layout (residency changes where a block is read,
    never the arithmetic); draft rows read only arrived blocks. Negative control: an arrival with a stale generation is
    refused (StaleGeneration)."""
    w, checks, _ = _f6()
    assert checks and all(c["content"] and c["equal"] and c["full"] for c in checks), checks
    w2 = world(C=4, X=4, R_ver=1)
    w2.svc.begin_request(0, "u0")
    p = w2.svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 300, 10)], S([[2], []]))
    p.issued[0].gen += 1                                                 # the record names another generation
    with pytest.raises(ER.StaleGeneration):
        land_all(w2)


# =====================================================================================================================
# F7 memory, F8 metrics
# =====================================================================================================================
def _scan(svc):
    c = collections.Counter()
    for pool in svc.pools:
        for st in pool.streams.values():
            for s in st.slots:
                c[s.state] += 1
    return c


def test_F7_memory_accounting():
    """F7: the service's memory sample equals an independent scan of every slot (in flight, LRU, PO, VRES, RELEASING),
    never exceeds the physical store, and its peak retained (PO + VRES) equals the maximum of the scans at the same
    sample points; the refused draft fetches and the masked fraction are reported. Negative control: counting VRES as
    LRU (K3X vres_counted_as_lru) disagrees with the scan."""
    def agrees(m, c):
        return (m["inflight"], m["lru"], m["po"], m["vres"], m["releasing"]) == (
            c[OWN.INFLIGHT], c[OWN.LRU], c[OWN.PO], c[OWN.VRES], c[OWN.RELEASING])
    w, _, samples = _f6(steps=12)
    svc = w.svc
    assert all(agrees(m, c) for m, c in samples) and agrees(svc.memory(), _scan(svc))
    assert all(m["slots_used"] <= m["slots_physical"] for m, _ in samples)
    assert svc.metrics.peak_retained_slots == max(m["retained"] for m, _ in samples) > 0
    rep = svc.report()
    assert rep["masked_fraction"] == rep["masked"] / rep["draft_pairs"] and rep["draft_pairs"] > 0
    w2, _, samples2 = _f6(steps=12, mut=MU.Mutations(vres_counted_as_lru=True))
    assert any(c[OWN.VRES] > 0 for _, c in samples2) and not all(agrees(m, c) for m, c in samples2)


def test_F8_paper_metrics_on_a_scripted_trace():
    """F8: on the F5 trace, 4.3 counts ready 3 / in flight 1 / new 2 in one layer_call; 4.6 serves 4 blocks from the
    LRU and 2 by first fetches; 4.8: 6 transfers, 6 distinct, 0 repeated; all 6 lifetimes are still live, and after the
    request ends and its reader completed they are verify_only (4 issued by drafts, 2 by the verifier). Negative
    control: the duplicate transfer of NoDedup makes repeated 1."""
    w, pv, _ = _f5()
    svc = w.svc
    ev = Ev(done=True)
    svc.read_done(0, pv, ev)
    rep = svc.report()
    k43, k46, k48 = rep["k43"], rep["k46"], rep["k48"]
    assert (k43["layer_calls"], k43["ready"], k43["in_flight"], k43["new"]) == (1, 3, 1, 2)
    assert (k46["lru"], k46["fetched_first"], k46["po"], k46["vres"], k46["not_resident"]) == (4, 2, 0, 0, 0)
    assert (k48["transfers"], k48["distinct"], k48["repeated"]) == (6, 6, 0)
    assert k48["prefetch_use"]["still_live"] == 6
    svc.end_request("u0")
    svc.poll()
    k48 = svc.report()["k48"]
    assert k48["prefetch_use"]["verify_only"] == 6 and k48["prefetch_use"]["still_live"] == 0
    assert k48["by_issuer"][RES.DRAFT]["verify_only"] == 4 and k48["by_issuer"][RES.VERIFY]["verify_only"] == 2
    w2, pv2, _ = _f5(NoDedup)
    assert w2.svc.report()["k48"]["repeated"] == 1


# =====================================================================================================================
# F9 the vendored K3X files
# =====================================================================================================================
def _blob(path):
    r = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def test_F9_vendored_k3x_files_are_byte_identical(tmp_path):
    """F9: nosi/nosi/kstep/{__init__, errors, backend, executor, ownership, transfer, mutations}.py are the blobs of
    K3X 91e6e47 (git rev-parse 91e6e47:<path> == git hash-object <file>). Negative control: a copy with one byte
    changed has another blob id."""
    for name in VENDORED:
        rel = "nosi/nosi/kstep/%s.py" % name
        want = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:%s" % (K3X, rel)], capture_output=True, text=True)
        assert want.returncode == 0, "K3X %s is not reachable for %s" % (K3X, rel)
        assert _blob(ROOT / rel) == want.stdout.strip(), rel
    bad = tmp_path / "ownership.py"
    bad.write_bytes((ROOT / "nosi/nosi/kstep/ownership.py").read_bytes() + b"\n")
    want = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:nosi/nosi/kstep/ownership.py" % K3X],
                          capture_output=True, text=True).stdout.strip()
    assert _blob(bad) != want
