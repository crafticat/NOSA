"""CPU tests F1-F13 of R2 step 7, piece 5 (docs/R2_STEP7_FETCH_RETENTION.md sections 12, 12b, 12c):
nosi/nosi/r2_residency.py (v1, the reference), r2_residency_sb.py (S-B) and r2_residency_sc.py (S-C), the residency
service of the spec round, over K3X's ownership and transports VENDORED byte-identical in nosi/nosi/kstep/ (91e6e47). CPU tensors, K3X's AdversarialTransport (seeded delays, reordering, bursts; "force_only" for
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
SB = _load("r2_residency_sb")
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


def world(C=4, X=4, R_ver=2, seed=0, mode="force_only", mut=None, cls=None, max_delay=3, svc_kw=None):
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
                                 block_bytes=BS * 2 * D * 4, host_wait=be.host_wait, mut=mut, **(svc_kw or {}))
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
    in-flight block is not requested again. Negative controls: a row that reads an in-flight block (caught by the read
    check, or by the O1 recency update, which refuses a read block that is not resident), and a service that does not
    find in-flight blocks (duplicate transfers), are both caught."""
    w, p2 = _f1()
    assert p2.blocks == [[[0], [1]], [[0, 1], [0]]]
    assert reads_ok(w, 0, p2) and dedup_ok(w.svc)
    assert sorted((tr.h, tr.block) for tr in p2.issued) == [(0, 4), (0, 5), (1, 4), (1, 5)]
    assert p2.masked == 4 and p2.draft_pairs == 4 and p2.refused == 0
    try:                                               # caught by the read check, or earlier by K3X's O1 recency check
        w2, q2 = _f1(InflightReadable)
        caught = not reads_ok(w2, 0, q2)
    except ER.OwnershipViolation:
        caught = True
    assert caught
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


# =====================================================================================================================
# F10 S-B: the draft rows on the device, against v1
# =====================================================================================================================
def _sched(seed, steps, verify, hb=15, uids=("u0", "u1")):
    """F6's schedule as data: per tick (t, cohort m, the cohort ends, per layer [(b, kind, key, sel per head)]). A draft
    row selects a window of 6 historical blocks and its tail block; with verify, every third tick adds a verify row
    of 8 random historical blocks per request (and that tick's draft row belongs to the next cohort)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in range(steps):
        m = t // 3 + 1
        ends = t % 3 == 2
        layers = []
        for _ in range(L):
            spec = []
            for b, u in enumerate(uids):
                x = int(torch.randint(0, hb - 6, (1,), generator=g))
                spec.append((b, "d", (u, 1, m + 1 if ends else m), [list(range(x, x + 6)) + [hb] for _ in range(H)]))
                if verify and ends:
                    v = sorted(set(int(z) for z in torch.randint(0, hb, (8,), generator=g)))
                    spec.append((b, "v", (u, 1, m), [v + [hb] for _ in range(H)]))
            layers.append(spec)
        out.append((t, m, ends, layers))
    return out


def _state(w):
    """Everything the decisions change: per stream the owner, the LRU in order, PO, VRES, the pending map, every
    protection with its kinds and every slot (state, block, gen, class); the service's in-flight transfers (without
    their issue sequence) and the transport's pending list in order."""
    svc = w.svc
    out = []
    for l, pool in enumerate(svc.pools):
        for (b, h) in sorted(pool.streams):
            st = pool.streams[(b, h)]
            out.append((l, b, h, st.owner, list(st.lru.items()), sorted(st.po.items()), sorted(st.vres.items()),
                        sorted(st.pending.items()),
                        sorted((blk, sorted(repr(k) for k in keys)) for blk, keys in st.prot.items()),
                        sorted((blk, sorted((repr(k), sorted(v)) for k, v in kk.items())) for blk, kk in st.pin_kind.items()),
                        [(s.state, s.block, s.gen, s.cls) for s in st.slots]))
    out.append(sorted((key, tr.slot, tr.gen, tr.kind, repr(tr.requester), sorted(tr.demand))
                      for key, tr in svc.pending.items()))
    out.append([(tr.layer, tr.b, tr.h, tr.block, tr.slot, tr.gen, sorted(tr.demand)) for tr in w.tp.pending])
    return out


def _counters(svc):
    m = svc.metrics                                    # wait_ns / layer_wait_ns are clocks: not compared
    return (m.masked, m.draft_pairs, m.refused, sorted(m.counts.items()),
            (m.k43["ready"], m.k43["in_flight"], m.k43["new"], m.k43["layer_calls"]),
            sorted(m.k46.items()), sorted(m.transferred.items()))


def _lockstep(seed, C, X, R_ver, steps, verify, sb_mut=None, hb=15):
    """v1 (host, the S-B schedule: arrivals at the tick's start, draft transfers at end_tick) and S-B driven by the same
    schedule, transports seeded alike (delays, reordering, bursts), reader events completed two ticks late in both.
    Returns the first differences (page tables per layer; state and counters per tick), and the two worlds."""
    common = dict(C=C, X=X, R_ver=R_ver, seed=seed, mode="delay", max_delay=4)
    wv = world(**common, svc_kw=dict(poll_in_resolve=False, defer_issue=True))
    ws = world(**common, cls=SB.ResidencySB, svc_kw=dict(nblk=NB, sb_mut=sb_mut))
    uids = ("u0", "u1")
    for w in (wv, ws):
        for b, u in enumerate(uids):
            w.svc.begin_request(b, u)
    tail = wv.svc.tail0
    readers, bad = [], []
    for (t, m, ends, layers) in _sched(seed + 100, steps, verify, hb):
        for w in (wv, ws):
            w.svc.poll()
        for l, spec in enumerate(layers):
            rows = [Row(b, kind, key, 1000 + t, hb, [(hb, tail)]) for (b, kind, key, _) in spec]
            sel = S(*[s for (_, _, _, s) in spec])
            pv = wv.svc.resolve(l, rows, sel)
            ps = ws.svc.resolve(l, rows, sel)
            for i in range(len(rows)):
                for h in range(H):
                    a = pv.page_table[i, h, :int(pv.num_pages[i, h])].tolist()
                    c = ps.page_table[i, h, :int(ps.num_pages[i, h])].tolist()
                    if a != c:
                        bad.append(("pages", t, l, i, h, a, c))
            ev_v, ev_s = Ev(), Ev()
            readers.append((t, ev_v, ev_s))
            wv.svc.read_done(l, pv, ev_v)
            ws.svc.read_done(l, ps, ev_s)
        for w in (wv, ws):
            w.svc.end_tick()
        for (tt, a, c) in readers:
            if tt <= t - 2:
                a.done = c.done = True
        if ends:
            for w in (wv, ws):
                for u in uids:
                    w.svc.decided((u, 1, m))
        sv, ss = _state(wv), _state(ws)
        if sv != ss:
            first = next(i for i, (x, y) in enumerate(zip(sv, ss)) if x != y)
            bad.append(("state", t, sv[first], ss[first]))
        if _counters(wv.svc) != _counters(ws.svc):
            bad.append(("counters", t, _counters(wv.svc), _counters(ws.svc)))
        if bad:
            break
    return bad, wv, ws


def test_F10_sb_equals_v1_on_the_f6_schedule():
    """F10: over F6's 24-tick schedule (2 requests, 2 layers, draft rows every tick, verify rows every third tick,
    seeded delays / reordering / bursts), S-B's page tables (every row and head, every layer), the ownership state
    after every tick (LRU order, PO, VRES, protections with kinds, slots, in-flight transfers, the transport's pending
    list) and the counters (masked, draft pairs, refusals, 4.3, 4.6, transfers) equal v1's under the S-B schedule. The
    device is read once per tick for the draft rows, plus once per layer of a verify tick (v1's verify rows)."""
    bad, wv, ws = _lockstep(1, C=8, X=48, R_ver=8, steps=24, verify=True)
    assert bad == [], bad[:3]
    assert ws.svc.d2h_reads == 24 + 8 * L
    assert wv.svc.metrics.k43["layer_calls"] > 0 and wv.svc.metrics.draft_pairs == 24 * L * B * H


def test_F10b_sb_equals_v1_under_capacity_pressure():
    """F10b: a draft-only schedule with C 2, X 4, R_ver 2 (K5 refusals, LRU evictions to PO, borrowed extras):
    S-B equals v1 every tick, refusals happen, and the device is read exactly once per tick."""
    bad, wv, ws = _lockstep(3, C=2, X=4, R_ver=2, steps=12, verify=False)
    assert bad == [], bad[:3]
    assert wv.svc.metrics.refused > 0 and ws.svc.d2h_reads == 12


@pytest.mark.parametrize("mut, verify, cap", [("no_dedup", True, (8, 48, 8)), ("ignore_reserve", False, (2, 4, 2)),
                                              ("no_hit_protect", True, (8, 48, 8))])
def test_F10c_sb_negative_controls(mut, verify, cap):
    """F10c: the same lockstep catches a device without the dedup (a block in flight requested again), a device whose
    K5 ignores the verifier reserve, and a device that does not protect the hits: by a difference from v1 or by the
    host's check of a device decision (OwnershipViolation)."""
    C, X, R_ver = cap
    try:
        bad, _, _ = _lockstep(1 if verify else 3, C=C, X=X, R_ver=R_ver, steps=24 if verify else 12, verify=verify,
                              sb_mut={mut: True})
        caught = bool(bad)
    except ER.OwnershipViolation:
        caught = True
    assert caught


class NoHistForward(RES.Residency):
    """DEFECT: the transfer does not carry the block's host write-back event."""

    def _hist_ev_for(self, b, block):
        return None


def test_F11_history_events_and_the_end_of_an_epoch():
    """F11 (the verify agent's Q-S1 / Q-S2): a block past the prefill may be requested only after the tick registered
    its host write-back (history_written): before, the take is refused (no slot taken, no transfer); after, the
    transfer carries the event and the transport waits on it before it reads the host rows (K3X T4). cancel_epoch
    cancels every key of the epoch, the keys only draft rows used included, and leaves the next epoch's keys alone.
    Negative controls: a service that does not forward the event lets the transport read without waiting; a single
    cancel leaves the epoch's other key usable."""
    w = world(C=4, X=4, R_ver=1)
    svc = w.svc
    svc.begin_request(0, "u0", hist_blocks=10)
    with pytest.raises(ER.OwnershipViolation, match="no registered host write-back"):
        svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 800, 13)], S([[12], []]))
    assert all(s.state == OWN.FREE for s in svc.stream(0, 0, 0).slots) and not svc.inflight
    ev = Ev()
    svc.history_written("u0", 12, ev)
    waits = w.be.waits
    p = svc.resolve(0, [Row(0, "d", ("u0", 1, 2), 801, 13)], S([[12], []]))
    assert p.issued[0].hist_ev is ev and ev.done and w.be.waits == waits + 1
    w2 = world(C=4, X=4, R_ver=1, cls=NoHistForward)
    w2.svc.begin_request(0, "u0", hist_blocks=10)
    ev2 = Ev()
    w2.svc.history_written("u0", 12, ev2)
    p2 = w2.svc.resolve(0, [Row(0, "d", ("u0", 1, 2), 801, 13)], S([[12], []]))
    assert p2.issued[0].hist_ev is None and not ev2.done
    # the end of an epoch
    for single in (False, True):
        w3 = world(C=4, X=4, R_ver=1)
        s3 = w3.svc
        s3.begin_request(0, "u0")
        for l, m in ((0, 1), (1, 2)):
            s3.resolve(l, [Row(0, "d", ("u0", 1, m), 300, 10)], S([[1, 2], []]))
        if single:
            s3.cancel(("u0", 1, 1))
            s3.resolve(1, [Row(0, "d", ("u0", 1, 2), 301, 10)], S([[3], []]))      # still usable: not the epoch's end
        else:
            assert s3.cancel_epoch("u0", 1) == [("u0", 1, 1), ("u0", 1, 2)]
            for l, m in ((0, 1), (1, 2)):
                with pytest.raises(ER.OwnershipViolation, match="cancelled"):
                    s3.resolve(l, [Row(0, "d", ("u0", 1, m), 301, 10)], S([[3], []]))
            s3.resolve(0, [Row(0, "d", ("u0", 2, 1), 302, 10)], S([[3], []]))        # the next epoch resolves


FORBIDDEN = (".item(", ".tolist(", ".nonzero(", ".cpu(", ".numpy(", "bool(", "int(", "float(", ".any()", ".all()")


def host_reads(src):
    return [tok for tok in FORBIDDEN if tok in src]


def test_F10d_device_draft_reads_nothing_on_the_host():
    """F10d: the source of device_draft holds no host read of a device value (.item, .tolist, .nonzero, .cpu, .numpy,
    Python bool / int / float of a tensor, .any() / .all()); the mirror refresh and the per-tick read are outside it.
    Negative control: the same check finds a .item() call."""
    import inspect
    assert host_reads(inspect.getsource(SB.device_draft)) == []
    assert host_reads("n = int(x.sum().item())") != []


# =====================================================================================================================
# F12 S-C: the verify rows on the device, against v1 (and F13: the next request of a batch slot)
# =====================================================================================================================
SC = _load("r2_residency_sc")


def _content_ok(w, l, rows, sel, plan):
    """At the read point (after resolve, before the attention): every verify row lists its whole selection (the
    selected historical blocks, ascending) and every listed page holds the host truth of its block (K, V and bias)."""
    for i, r in enumerate(rows):
        if r.kind != "v":
            continue
        for h in range(H):
            want = sorted({z for z in sel[h, i].tolist() if 0 <= z < r.hist_bound})
            got = plan.page_table[i, h, :len(want)].tolist()
            if int(plan.num_pages[i, h]) < len(want):
                return False
            if not all(slot_holds(w, l, r.b, h, s, z) for z, s in zip(want, got)):
                return False
    return True


def _sched_pressure(seed, steps, hb=8, window=4, nver=3, uids=("u0", "u1")):
    """The pressure schedule: per tick, layer and request a draft row (a window of `window` historical blocks and the
    tail block); every third tick TWO verify rows of the ending cohort per request (the flattened round: one key, two
    positions), each with up to `nver` random historical blocks per head, so the rows' selections overlap."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in range(steps):
        m = t // 3 + 1
        ends = t % 3 == 2
        layers = []
        for _ in range(L):
            spec = []
            for b, u in enumerate(uids):
                x = int(torch.randint(0, hb - window + 1, (1,), generator=g))
                spec.append((b, "d", (u, 1, m + 1 if ends else m), [list(range(x, x + window)) + [hb] for _ in range(H)]))
                if ends:
                    for _ in range(2):
                        v = [sorted(set(int(z) for z in torch.randint(0, hb, (nver,), generator=g))) + [hb]
                             for _ in range(H)]
                        spec.append((b, "v", (u, 1, m), v))
            layers.append(spec)
        out.append((t, m, ends, layers))
    return out


def _mirror_diff(svc, l):
    """The fields in which S-C's device-maintained mirror of layer l differs from the full-scan reference
    (VerifyMirror.refresh over the host state)."""
    ref = SC.VerifyMirror(svc.B, svc.H, svc.nblk, svc.n_hist, svc.device)
    ref.refresh(svc, l)
    got = svc.dmirror.layer(l)
    return [f for f in SC.MIRROR_FIELDS if not torch.equal(getattr(ref, f), getattr(got, f))]


def _lockstep_sc(seed, C, X, R_ver, sched, sc_mut=None, hb=15, mirror_drop=None):
    """v1 under the S-C schedule (arrivals at the tick's start, draft transfers at end_tick, waited and urgent blocks
    read in place and admitted at the next poll) and S-C driven by the same schedule, transports seeded alike, reader
    events completed two ticks late in both. At the start of every tick (after the poll and S-C's mirror delta) S-C's
    device mirror of every layer is compared with the full-scan reference. Returns the first differences (the mirror;
    page tables per layer; the content of every verify page at the read point, in both worlds; state and counters per
    tick), the two worlds and the largest number of shared extras borrowed in v1 after a tick."""
    common = dict(C=C, X=X, R_ver=R_ver, seed=seed, mode="delay", max_delay=4)
    wv = world(**common, svc_kw=dict(poll_in_resolve=False, defer_issue=True, defer_admit=True))
    ws = world(**common, cls=SC.ResidencySC, svc_kw=dict(nblk=NB, sc_mut=sc_mut, mirror_drop=mirror_drop))
    uids = ("u0", "u1")
    for w in (wv, ws):
        for b, u in enumerate(uids):
            w.svc.begin_request(b, u)
    tail = wv.svc.tail0
    readers, bad = [], []
    peak_borrowed = 0
    for (t, m, ends, layers) in sched:
        for w in (wv, ws):
            w.svc.poll()
        ws.svc.sync_mirror()                                  # the tick's delta (the first resolve would send it)
        for l in range(L):
            d = _mirror_diff(ws.svc, l)
            if d:
                bad.append(("mirror", t, l, d))
        if bad:
            break
        for l, spec in enumerate(layers):
            rows = [Row(b, kind, key, 1000 + t, hb, [(hb, tail)]) for (b, kind, key, _) in spec]
            sel = S(*[s for (_, _, _, s) in spec])
            pv = wv.svc.resolve(l, rows, sel)
            ps = ws.svc.resolve(l, rows, sel)
            for i in range(len(rows)):
                for h in range(H):
                    a = pv.page_table[i, h, :int(pv.num_pages[i, h])].tolist()
                    c = ps.page_table[i, h, :int(ps.num_pages[i, h])].tolist()
                    if a != c:
                        bad.append(("pages", t, l, i, h, a, c))
            for name, w, p in (("v1", wv, pv), ("sc", ws, ps)):
                if not _content_ok(w, l, rows, sel, p):
                    bad.append(("content", name, t, l))
            ev_v, ev_s = Ev(), Ev()
            readers.append((t, ev_v, ev_s))
            wv.svc.read_done(l, pv, ev_v)
            ws.svc.read_done(l, ps, ev_s)
        for w in (wv, ws):
            w.svc.end_tick()
        peak_borrowed = max([peak_borrowed] + [pool.borrowed() for pool in wv.svc.pools])
        for (tt, a, c) in readers:
            if tt <= t - 2:
                a.done = c.done = True
        if ends:
            for w in (wv, ws):
                for u in uids:
                    w.svc.decided((u, 1, m))
        sv, ss = _state(wv), _state(ws)
        if sv != ss:
            first = next(i for i, (x, y) in enumerate(zip(sv, ss)) if x != y)
            bad.append(("state", t, sv[first], ss[first]))
        if _counters(wv.svc) != _counters(ws.svc):
            bad.append(("counters", t, _counters(wv.svc), _counters(ws.svc)))
        if bad:
            break
    if not bad:                                               # the last tick's changes reach the device mirror too
        ws.svc.sync_mirror()
        for l in range(L):
            d = _mirror_diff(ws.svc, l)
            if d:
                bad.append(("mirror", "end", l, d))
    return bad, wv, ws, peak_borrowed


def _no_verifier_force(svc):
    return not any(k in svc.metrics.counts for k in ("verifier_evictions", "verifier_force_waits"))


def test_F12_sc_equals_v1_on_the_f6_schedule():
    """F12: over F6's 24-tick schedule (2 requests, 2 layers, draft rows every tick, verify rows every third tick,
    seeded delays / reordering / bursts), S-C's page tables (every row and head, every layer), the ownership state
    after every tick and the counters (masked, draft pairs, refusals, 4.3, 4.6, transfers) equal v1's under the S-C
    schedule, and every verify page holds the host truth of its block at the read point in both. The verify rows
    met ready, in-flight (waited on the device) and new (gathered) blocks; v1 never evicted or forced for the verifier
    (S-C's overflow precondition). The device is read ONCE per tick (verify and draft decisions together)."""
    bad, wv, ws, _ = _lockstep_sc(1, C=8, X=48, R_ver=8, sched=_sched(101, 24, True))
    assert bad == [], bad[:3]
    assert ws.svc.d2h_reads == 24 and ws.svc.row_uploads == 24 and ws.svc.mirror_uploads <= 24 + 1   # + the final check
    k43 = wv.svc.metrics.k43
    assert k43["layer_calls"] > 0 and k43["ready"] > 0 and k43["in_flight"] > 0 and k43["new"] > 0, k43
    assert ws.svc.dev.waited > 0 and ws.svc.dev.gathered > 0
    assert _no_verifier_force(wv.svc), wv.svc.metrics.counts


def test_F12b_sc_equals_v1_under_capacity_pressure():
    """F12b: C 2, X 96, R_ver 88 (draft fetches may borrow 8 extras per layer; the rest is the verifier's reserve),
    8 historical blocks, two overlapping verify rows per cohort and request (the flattened round): S-C equals v1 every
    tick (pages, content, state, counters), draft fetches are refused, the verifier borrowed beyond the drafts' limit
    (it used the reserve), a block that two verify rows of one cohort miss is transferred once, and the device is
    read once per tick."""
    bad, wv, ws, peak = _lockstep_sc(3, C=2, X=96, R_ver=88, sched=_sched_pressure(103, 15), hb=8)
    assert bad == [], bad[:3]
    assert ws.svc.d2h_reads == 15 and ws.svc.row_uploads == 15 and ws.svc.mirror_uploads <= 15 + 1   # + the final check
    assert wv.svc.metrics.refused > 0 and peak > 96 - 88, (wv.svc.metrics.refused, peak)
    assert wv.svc.metrics.k43["new"] > 0 and _no_verifier_force(wv.svc)


@pytest.mark.parametrize("mut, sched", [("no_wait", "f6"), ("no_verify_protect", "f6"), ("no_verify_dedup", "pressure"),
                                        ("no_verify_protect", "pressure")])
def test_F12c_sc_negative_controls(mut, sched):
    """F12c: the same lockstep catches a device that reads an in-flight block before its transfer landed (no_wait:
    caught by the CONTENT check of S-C's verify pages at the read point), a verify read without protection, and a
    device that transfers one block twice for two verify rows of a cohort (no_verify_dedup): by a difference from v1
    or by the host's check of a device decision (OwnershipViolation)."""
    args = (dict(seed=1, C=8, X=48, R_ver=8, sched=_sched(101, 24, True)) if sched == "f6" else
            dict(seed=3, C=2, X=96, R_ver=88, sched=_sched_pressure(103, 15), hb=8))
    try:
        bad, _, _, _ = _lockstep_sc(sc_mut={mut: True}, **args)
    except ER.OwnershipViolation:
        assert mut != "no_wait"
        return
    if mut == "no_wait":
        assert any(x[0] == "content" and x[1] == "sc" for x in bad), bad[:3]
    else:
        assert bad, mut


H2D = ("torch.tensor(", "as_tensor(", "from_numpy(", ".copy_(", "non_blocking", "pin_memory", ".cuda(", ".to(dv",
       ".to(dev", ".to(self.device", ".refresh(", "_h2d(")


def test_F12d_sc_per_layer_path_has_no_host_transfer():
    """F12d: the per-layer path of S-C (ResidencySC.resolve, DeviceMirror.layer, device_verify, device_draft,
    rank_in_group and the GPU device operations) holds no host read of a device value, no call of the counted device
    read (_d2h), and no host-to-device copy (no tensor built from host data, no copy_, no move to the device, no mirror
    refresh, no _h2d). resolve reaches the two per-tick copies (sync_mirror, _tick_rows) only at a tick's first layer;
    F12 / F12b / F14 count them. Negative controls: the same checks find the per-layer read in S-B's resolve, and the
    per-layer host-to-device copies of S-B's resolve and of the full-scan mirror refresh."""
    import inspect
    d2h = FORBIDDEN + ("_d2h(",)
    for fn in (SC.ResidencySC.resolve, SC.DeviceMirror.layer, SC.device_verify, SB.device_draft, SB.rank_in_group,
               SC.CudaDeviceOps.wait, SC.CudaDeviceOps.gather):
        src = inspect.getsource(fn)
        assert [tok for tok in d2h + H2D if tok in src] == [], fn.__qualname__
    assert [tok for tok in d2h if tok in inspect.getsource(SB.ResidencySB.resolve)] != []
    assert [tok for tok in H2D if tok in inspect.getsource(SB.ResidencySB.resolve)] != []
    assert [tok for tok in H2D if tok in inspect.getsource(SC.VerifyMirror.refresh)] != []


def _f12e(sc_mut=None, check_t3=True):
    """A slot read by u0's verifier (reader ev1, pending) is RELEASING after u0 ends; u1's verifier then misses block
    5 in the same stream. Returns (world, plan, ev1)."""
    w = world(C=2, X=4, R_ver=2, cls=SC.ResidencySC, svc_kw=dict(nblk=NB, sc_mut=sc_mut))
    svc = w.svc
    svc.dev.check_t3 = check_t3
    svc.begin_request(0, "u0")
    svc.poll()
    p = svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 300, 10)], S([[0, 1], []]))
    svc.read_done(0, p, Ev(done=True))
    svc.end_tick()                                          # the draft takes slots 0 and 1 of (b 0, h 0)
    land_all(w)                                             # both land: LRU
    ev1 = Ev()
    p = svc.resolve(0, [Row(0, "v", ("u0", 1, 1), 300, 10)], S([[0, 1], []]))
    svc.read_done(0, p, ev1)
    svc.end_tick()                                          # u0's verifier read slots 0 and 1 (reader ev1, pending)
    svc.end_request("u0")
    svc.poll()
    st = svc.stream(0, 0, 0)
    assert [s.state for s in st.slots[:3]] == [OWN.RELEASING, OWN.RELEASING, OWN.FREE]
    svc.begin_request(0, "u1")
    p = svc.resolve(0, [Row(0, "v", ("u1", 1, 1), 300, 10)], S([[5], []]))
    return w, p, ev1


def test_F12e_the_urgent_gather_never_writes_a_slot_still_read():
    """F12e: S-C's urgent gather takes a FREE slot (slot 2), never a RELEASING one whose last reader is pending
    (slots 0 and 1); the page holds block 5 and the host accepts the take at end_tick. Negative controls: a device
    that counts RELEASING slots as FREE writes slot 0 under its pending reader: the device's T3 check refuses the
    gather; with that check off, the host refuses the take at end_tick (first_free is 2)."""
    w, p, ev1 = _f12e()
    assert p.page_table[0, 0, 0] == 2 and slot_holds(w, 0, 0, 0, 2, 5)
    w.svc.read_done(0, p, Ev(done=True))
    w.svc.end_tick()
    assert w.svc.stream(0, 0, 0).slots[2].state == OWN.INFLIGHT and not ev1.done
    with pytest.raises(ER.OwnershipViolation, match="last reader is pending"):
        _f12e(sc_mut={"gather_into_releasing": True})
    w2, p2, _ = _f12e(sc_mut={"gather_into_releasing": True}, check_t3=False)
    assert p2.page_table[0, 0, 0] == 0
    w2.svc.read_done(0, p2, Ev(done=True))
    with pytest.raises(ER.OwnershipViolation, match="the host would take 2"):
        w2.svc.end_tick()


def test_F12f_the_overflow_flag():
    """F12f: a verify row that needs 3 slots where K5 grants 2 (C 1, X 1): S-C points the third block's page at an
    in-bounds slot (the content check sees the hole at the read point) and end_tick raises CapacityRefusal after its
    one device read, before any change of the host state; v1 refuses the same selection at resolve. Negative control:
    a device that does not report the overflow lets the hole reach the host, which refuses it (OwnershipViolation)."""
    rows, sel = [Row(0, "v", ("u0", 1, 1), 300, 10)], S([[0, 1, 2], []])
    for sc_mut in (None, {"no_overflow_flag": True}):
        w = world(C=1, X=1, R_ver=1, cls=SC.ResidencySC, svc_kw=dict(nblk=NB, sc_mut=sc_mut))
        svc = w.svc
        svc.begin_request(0, "u0")
        svc.poll()
        p = svc.resolve(0, rows, sel)
        assert not _content_ok(w, 0, rows, sel, p)
        svc.read_done(0, p, Ev(done=True))
        if sc_mut is None:
            with pytest.raises(ER.CapacityRefusal):
                svc.end_tick()
            assert svc.d2h_reads == 1 and not svc.inflight and not svc.stream(0, 0, 0).prot
        else:
            with pytest.raises(ER.OwnershipViolation, match="did not flag an overflow"):
                svc.end_tick()
    wv = world(C=1, X=1, R_ver=1)
    wv.svc.begin_request(0, "u0")
    with pytest.raises(ER.CapacityRefusal):
        wv.svc.resolve(0, rows, sel)


class OwnerBlindInFlight(RES.Residency):
    """DEFECT: the in-flight lookup ignores the stream's owner (a transfer of an ended request is merged and read)."""

    def _in_flight(self, l, b, h, t):
        return self.pending.get((l, b, h, t))


def _f13(cls=None, svc_kw=None):
    """u0's draft transfer of block 5 is in flight (force_only: it does not land) when u0 ends; u1 takes batch slot 0
    (its prefill writes other host rows) and its verifier selects block 5."""
    w = world(C=4, X=4, R_ver=1, cls=cls, svc_kw=svc_kw)
    svc = w.svc
    svc.begin_request(0, "u0")
    svc.poll()
    p = svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 300, 10)], S([[5], []]))
    svc.read_done(0, p, Ev(done=True))
    svc.end_tick()
    svc.end_request("u0")
    svc.begin_request(0, "u1")
    g = torch.Generator().manual_seed(77)
    for l in range(L):
        w.host_k[l][0] = torch.randn(w.host_k[l][0].shape, generator=g)
        w.host_v[l][0] = torch.randn(w.host_v[l][0].shape, generator=g)
        w.cis[l][0] = torch.randn(w.cis[l][0].shape, generator=g)
    svc.poll()
    rows, sel = [Row(0, "v", ("u1", 1, 1), 300, 10)], S([[5], []])
    return w, rows, sel, svc.resolve(0, rows, sel)


def test_F13_a_transfer_of_an_ended_request_is_never_read_by_the_next():
    """F13: a transfer in flight for a request that ended carries that request's rows; the next request of the batch
    slot does not merge with it or wait for it: its verifier takes its own urgent slot and reads its own rows (v1
    under the S-C schedule, and S-C). Negative control: an owner-blind in-flight lookup reads the old request's rows
    (caught by the content check)."""
    v1 = dict(poll_in_resolve=False, defer_issue=True, defer_admit=True)
    for cls, kw in ((None, v1), (SC.ResidencySC, dict(nblk=NB))):
        w, rows, sel, p = _f13(cls, kw)
        assert _content_ok(w, 0, rows, sel, p)
        if cls is None:
            assert p.looks[0]["new"] == [5] and len(p.urgent) == 1 and p.waited == []
    w, rows, sel, p = _f13(OwnerBlindInFlight, v1)
    assert p.waited and not _content_ok(w, 0, rows, sel, p)


# =====================================================================================================================
# F14 S-C's device mirror: one delta per tick, equal to the full scan
# =====================================================================================================================
def test_F14_the_delta_mirror_equals_the_full_scan():
    """F14: S-C's mirror lives on the device and is updated by ONE batched delta per tick (the slots that the stream
    journals and the owner changes marked); after every tick of F6's schedule and of F12b's pressure schedule it
    equals the full-scan mirror (VerifyMirror.refresh) at every layer and in every field, and a delta is smaller than
    the table (it is a delta, not a copy of every slot)."""
    for args, ticks in ((dict(seed=1, C=8, X=48, R_ver=8, sched=_sched(101, 24, True)), 24),
                        (dict(seed=3, C=2, X=96, R_ver=88, sched=_sched_pressure(103, 15), hb=8), 15)):
        bad, _, ws, _ = _lockstep_sc(**args)
        assert not [x for x in bad if x[0] == "mirror"] and bad == [], bad[:3]
        svc = ws.svc
        assert 0 < svc.mirror_uploads <= ticks + 1                       # one per tick, + the final check
        assert svc.mirror_slots_uploaded < svc.mirror_uploads * svc.L * svc.B * svc.H * svc.n_hist


@pytest.mark.parametrize("kind", ["take", "admit", "admit_vres", "release", "free"])
def test_F14c_a_dropped_delta_kind_is_caught(kind):
    """F14c (negative controls): a mirror delta that drops one kind of transition (the slot is not sent to the device)
    leaves the device mirror unequal to the full scan at the start of a later tick of F6's schedule, before any
    decision reads it."""
    bad, _, _, _ = _lockstep_sc(1, C=8, X=48, R_ver=8, sched=_sched(101, 24, True), mirror_drop={kind})
    assert bad and bad[0][0] == "mirror", (kind, bad[:2])


def test_F14b_owner_changes_reach_the_device_mirror():
    """F14b: a request ends with a transfer in flight that the device mirror already holds as the owner's; the next
    request of the batch slot begins. The owner change reaches the device (that transfer is no longer the owner's in
    flight) and the mirror equals the full scan. Negative control: a delta without the owner changes leaves the old
    transfer in flight on the device: caught."""
    for drop, ok in ((None, True), ({"owner"}, False)):
        w = world(C=4, X=4, R_ver=1, cls=SC.ResidencySC, svc_kw=dict(nblk=NB, mirror_drop=drop))
        svc = w.svc
        svc.begin_request(0, "u0")
        diffs = []
        for blocks in ([5], [6]):                       # tick 1 takes block 5 (force_only: it stays in flight)
            svc.poll()
            svc.sync_mirror()                           # tick 2 sends it to the device as the owner's transfer
            diffs += [(l, d) for l in range(L) for d in _mirror_diff(svc, l)]
            p = svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 300, 10)], S([blocks, []]))
            svc.read_done(0, p, Ev(done=True))
            svc.end_tick()
        assert diffs == [] and svc.dmirror.layer(0).inflight[0, 0, 5]
        svc.end_request("u0")
        svc.begin_request(0, "u1")
        svc.poll()
        svc.sync_mirror()
        diffs = [(l, d) for l in range(L) for d in _mirror_diff(svc, l)]
        assert (diffs == []) == ok, (drop, diffs)
