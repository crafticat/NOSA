"""CPU tests of B2-T (benchmarks/Efficiency/r2_own2_b2t.py) and of the ops it times (nosi/nosi/r2_own2_ops.py);
docs/R2_OWN2_B2_DESIGN.md rev 2. Every timed component is checked against the VENDORED objects (kstep/ownership.py,
91e6e47) or the vendored rule written out literally, with a negative control that must be caught.

  B2T-W  the reader window: completion in any order is exact (F12e's case); wait() completes one id only. CONTROL: a
         tick watermark (done = id <= the newest complete id) differs.
  B2T-R  the recency rounds == the vendored touch / acquire / evict_to_c per draft row (res.py:686-706), 1-3 draft
         rows of ONE stream, 300 random cases. CONTROL: reverse_stamps differs. R3 (a FINDING for C's point 2): one
         batch eviction reaches the same end state in every case.
  B2T-A  the admissions == the vendored Stream.admit in seq order (own.py:322-361), over-C eviction and the reader
         resets included. CONTROL: keep_rd_last_on_drop differs.
  B2T-K  K5 + first_free == the vendored LayerPool.can_take / first_free run in order (verify part, then draft part).
         CONTROLS: inclusive_prefix and a flipped outcome are flagged.
  B2T-E  Policy E's victims == the vendored candidate rule (res.py:338-342, literally). CONTROLS: newest_first and a
         pinned victim are flagged.
  B2T-P  the pin rounds (rows sharing (plane, b)) == the set union. CONTROL: sum_not_or differs.
  B2T-D  the key drops == the vendored unprotect + release (own.py:213-237).
  B2T-F  the releases == LayerPool.poll_releases with readers completing out of order (own.py:435-452).
  B2T-I  the post-write predicates fire on their planted state (I1, I2, I3, I9), not on a clean one.
  B2T-U  a void tick (gen_plus_two) restores the table exactly (the undo log); a clean tick commits.
  B2T-H  the harness (--smoke, both modes): exit 0, the generated counts equal the requested ones, no predicate fires,
         the checked and unchecked final tables are equal, the desc is contiguous, every component is timed, the
         unchecked mode skips CHECK_ONLY.
  B2T-M  unchecked == checked on the smoke world; a predicate-only defect voids checked, passes unchecked.
  B2T-G1 --device cuda without CUDA exits 2 before any work. G2 (GPU only, skipped on CPU): CUDA == host tables.
  B2T-G3 the b2tg step script: static rules, the REGISTERED block == REGISTERED_DEVICE, the uuid rules, the DRYRUN.
  B2T-L  every Python loop of r2_own2_ops.py is per column / part / plane / round / live event (an allowlist).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_own2_b2t.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"


def _load_bench():
    name = "r2_own2_b2t_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(EFF / "r2_own2_b2t.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


BT = _load_bench()
OPS = BT.OPS
OWN = sys.modules[BT.PKG + ".kstep.ownership"]
EX = sys.modules[BT.PKG + ".kstep.executor"]
SP = BT.SP
CODE = {OWN.FREE: OPS.C_FREE, OWN.INFLIGHT: OPS.C_INFLIGHT, OWN.LRU: OPS.C_LRU, OWN.PO: OPS.C_PO,
        OWN.VRES: OPS.C_VRES, OWN.RELEASING: OPS.C_RELEASING}
CLS = {None: 0, "ad": OPS.CLS_AD, "v": OPS.CLS_V}
I64 = torch.int64


# ---------------------------------------------------------------------------------------------------- mirrors
def table_of(G, streams, uidmap, keymap, rid=None):
    """B2's table of vendored streams {(b, h): Stream} of layer 0. rid maps a reader object to its id (default: the
    reader is an int or None)."""
    rid = rid or (lambda r: -1 if r is None else int(r))
    T = OPS.Table(G)
    for (b, h), st in streams.items():
        T.owner[b] = uidmap[st.owner] if st.owner is not None else -1
        s = b * G.H + h
        for sl in st.slots:
            f = s * G.NPOS + sl.idx
            T.flat("code")[f] = CODE[sl.state]
            T.flat("block")[f] = sl.block
            T.flat("gen")[f] = sl.gen
            T.flat("uid")[f] = uidmap[sl.owner] if sl.owner is not None else -1
            T.flat("cls")[f] = CLS[sl.cls]
            T.flat("rd_first")[f] = rid(sl.first_reader)
            T.flat("rd_last")[f] = rid(sl.last_reader)
        for r, (blk, idx) in enumerate(st.lru.items()):
            T.flat("stamp")[s * G.NPOS + idx] = 100 + r
        for blk, keys in st.prot.items():
            for key in keys:
                kinds = st.pin_kind.get(blk, {}).get(key) or {EX.VERIFY}
                for kind in kinds:
                    plane = keymap[key] + (0 if kind == EX.VERIFY else G.KEYS)
                    w = T.pin_w[plane].view(G.S, G.WB)
                    w[s, blk // 32] |= 1 << (blk % 32)
    T.clock = 10_000
    return T


def st_state(st, readers=False):
    out = dict(lru=list(st.lru.items()), po=sorted(st.po.items()), vres=sorted(st.vres.items()),
               slots=[(CODE[s.state], s.block, s.gen) for s in st.slots])
    if readers:
        out["readers"] = [(-1 if s.first_reader is None else int(s.first_reader),
                           -1 if s.last_reader is None else int(s.last_reader)) for s in st.slots]
    return out


def tb_state(T, G, s, readers=False):
    code = T.c["code"].view(G.S, G.NPOS)[s].tolist()
    block = T.c["block"].view(G.S, G.NPOS)[s].tolist()
    gen = T.c["gen"].view(G.S, G.NPOS)[s].tolist()
    stamp = T.c["stamp"].view(G.S, G.NPOS)[s].tolist()
    lru = [(block[i], i) for (_, i) in sorted((stamp[i], i) for i in range(G.NPOS) if code[i] == OPS.C_LRU)]
    out = dict(lru=lru, po=sorted((block[i], i) for i in range(G.NPOS) if code[i] == OPS.C_PO),
               vres=sorted((block[i], i) for i in range(G.NPOS) if code[i] == OPS.C_VRES),
               slots=[(code[i], block[i], gen[i]) for i in range(G.NPOS)])
    if readers:
        rf = T.c["rd_first"].view(G.S, G.NPOS)[s].tolist()
        rl = T.c["rd_last"].view(G.S, G.NPOS)[s].tolist()
        out["readers"] = list(zip(rf, rl))
    return out


def pins_of(T, G, s, inv_keymap):
    """{(block, key, kind)} of stream s from the pin planes."""
    out = set()
    for plane in range(2 * G.KEYS):
        k = plane % G.KEYS
        if k not in inv_keymap:
            continue
        bits = SP.unpack_bits(T.pin_w[plane].view(G.S, G.WB)[s], G.NBLK)
        for blk in bits.nonzero().view(-1).tolist():
            out.add((blk, inv_keymap[k], EX.VERIFY if plane < G.KEYS else EX.DRAFT))
    return out


def prot_of(st):
    return {(blk, key, kind) for blk, kk in st.pin_kind.items() for key, kinds in kk.items() for kind in kinds}


def one_stream(C, X):
    st = OWN.Stream(0, 0, 0, C + X, C)
    st.owner = "u"
    return st


def lru_entry(st, idx, blk, seq):
    gen = st.take(idx, "u", blk, seq)
    st.admit(idx, "u", blk, gen, demand=(EX.DRAFT,))


def vres_entry(st, idx, blk, seq, key, po=False):
    gen = st.take(idx, "u", blk, seq)
    st.protect(blk, key, EX.VERIFY)
    st.admit(idx, "u", blk, gen, demand=(EX.VERIFY,))
    if po:                                                         # a protected-only entry (an evicted, pinned one)
        st.vres.pop(blk)
        st.po[blk] = idx
        st.slots[idx].state = OWN.PO


# ---------------------------------------------------------------------------------------------------- W
class Ev:
    def __init__(self, done=False):
        self.done = done


def _done(ev):
    return True if ev is None else bool(ev.done)


def test_B2T_W_reader_window_out_of_order():
    w = OPS.ReaderWindow(cap=64)
    evs = [Ev() for _ in range(5)]
    ids = [w.register(e, 10 + i) for i, e in enumerate(evs)]
    assert ids == [0, 1, 2, 3, 4]
    evs[3].done = evs[1].done = True                               # out of order: 1 and 3 complete, 0 / 2 / 4 pending
    w.refresh(_done)
    q = torch.tensor([-1, 0, 1, 2, 3, 4])
    assert w.done_of(q).tolist() == [True, False, True, False, True, False]
    watermark = (q <= 3).tolist()                                  # CONTROL: a tick watermark marks 0 and 2 complete
    assert watermark != w.done_of(q).tolist()
    assert w.mark_of(torch.tensor([1, 2])).tolist() == [-1, 12]
    evs[0].done = True
    w.refresh(_done)
    assert w.lo == 2 and w.done_of(q).tolist() == [True, True, True, False, True, False]
    waited = []
    w.wait(2, lambda ev: (waited.append(ev), setattr(ev, "done", True)), _done)
    assert waited == [evs[2]] and not evs[4].done                  # the host wait of ONE id (Be.host_wait)
    assert w.done_of(q).tolist() == [True, True, True, True, True, False]


# ---------------------------------------------------------------------------------------------------- R
def _recency_case(seed, mut=None):
    rng = random.Random(seed)
    C, X, NB = 4, 4, 32
    G = OPS.Geo(L=1, B=1, H=1, C=C, X=X, R_ver=0, NBLK=NB, KEYS=4)
    st = one_stream(C, X)
    key0 = ("u", 0, 0)
    blocks = rng.sample(range(NB), C + X)
    idxs = rng.sample(range(C + X), C + X)
    n_lru, n_po, n_vres = rng.randint(1, C), rng.randint(0, 2), rng.randint(0, 1)
    seq = 0
    for i in range(n_lru):
        seq += 1
        lru_entry(st, idxs[i], blocks[i], seq)
    for j in range(n_po + n_vres):
        seq += 1
        i = n_lru + j
        vres_entry(st, idxs[i], blocks[i], seq, key0, po=j < n_po)
    st.check()
    resident = list(st.lru) + list(st.po) + list(st.vres)
    n_rows = rng.randint(1, 3)
    reads = [sorted(rng.sample(resident, rng.randint(1, len(resident)))) for _ in range(n_rows)]
    keys = [("u", 0, r + 1) for r in range(n_rows)]
    for r, key in enumerate(keys):                                 # phase A: every selected block pinned (res.py:668)
        for t in reads[r]:
            st.protect(t, key, EX.DRAFT)
    T = table_of(G, {(0, 0): st}, {"u": 7}, {key0: 0, **{k: i + 1 for i, k in enumerate(keys)}})
    slots = [[st.resident(t) for t in reads[r]] for r in range(n_rows)]
    for r in range(n_rows):                                        # the vendored recency, row by row
        for t in reads[r]:
            if t in st.lru:
                st.touch(t)
            elif t in st.po or t in st.vres:
                st.acquire(t, None, evict=False)
            else:
                raise AssertionError("not resident")
        st.evict_to_c()
    st.check()
    k = NB
    jj, sl, oo, bb = [], [], [], []
    for r in range(n_rows):
        for e, (t, idx) in enumerate(zip(reads[r], slots[r])):
            jj.append(r)
            sl.append(idx)
            oo.append(r * k + e)
            bb.append(t)
    n = len(jj)
    D = dict(flags=torch.full((n,), OPS.READY, dtype=I64), j=torch.tensor(jj), s=torch.zeros(n, dtype=I64),
             slot=torch.tensor(sl), l=torch.zeros(n, dtype=I64), b=torch.zeros(n, dtype=I64),
             h=torch.zeros(n, dtype=I64), o=torch.tensor(oo), blk=torch.tensor(bb))
    rows = OPS.Rows(vb=torch.zeros(0, dtype=I64), vk=torch.zeros(0, dtype=I64), db=torch.zeros(n_rows, dtype=I64),
                    dk=torch.arange(1, n_rows + 1), k=k)
    rc = OPS.recency(T, D, rows, None, mut)
    assert rc["bad"] == 0
    return st_state(st), tb_state(T, G, 0), n_rows


def test_B2T_R_recency_rounds_equal_the_vendored():
    multi = 0
    for seed in range(300):
        v, b, n_rows = _recency_case(seed)
        assert v == b, (seed, v, b)
        multi += n_rows > 1
    assert multi > 50                                              # precondition: the two- and three-row case ran


def test_B2T_R2_negative_control_reverse_stamps():
    res = [_recency_case(seed, {"reverse_stamps": True}) for seed in range(300)]
    assert sum(1 for (v, b, _) in res if v != b) > 0


def test_B2T_R3_one_batch_reaches_the_same_end_state():
    """A FINDING for C's review point 2: with the pins fixed during the recency and every read block pinned in phase A,
    one eviction after all rows gives the vendored end state too (B2 keeps the rounds: they mirror res.py:686-706)."""
    for seed in range(300):
        v, b, _ = _recency_case(seed, {"one_batch": True})
        assert v == b, (seed, v, b)


# ---------------------------------------------------------------------------------------------------- A
def _admit_case(seed, mut=None):
    rng = random.Random(seed)
    C, X, NB = 3, 5, 32
    G = OPS.Geo(L=1, B=1, H=1, C=C, X=X, R_ver=0, NBLK=NB, KEYS=4)
    st = one_stream(C, X)
    key0 = ("u", 0, 0)
    blocks = rng.sample(range(NB), C + X)
    idxs = rng.sample(range(C + X), C + X)
    n_lru = rng.randint(0, C)
    seq = 0
    for i in range(n_lru):
        seq += 1
        lru_entry(st, idxs[i], blocks[i], seq)
    for blk in list(st.lru):
        if rng.random() < 0.5:
            st.protect(blk, key0, EX.VERIFY)
        st.slots[st.lru[blk]].last_reader = rng.choice([None, 2])
    infl = []
    for i in range(n_lru, n_lru + rng.randint(1, C + X - n_lru)):
        seq += 1
        blk, idx = blocks[i], idxs[i]
        gen = st.take(idx, "u", blk, seq)
        dem = rng.choice([(EX.DRAFT,), (EX.VERIFY,), (EX.DRAFT, EX.VERIFY)])
        if EX.VERIFY in dem and rng.random() < 0.5:
            st.protect(blk, key0, EX.VERIFY)
        st.slots[idx].first_reader = rng.choice([None, 3])
        st.slots[idx].last_reader = rng.choice([None, 4])
        infl.append((seq, idx, blk, gen, dem))
    ended = rng.random() < 0.3
    if ended:
        st.owner = "v"                                             # a new request owns the stream: arrivals dropped
    T = table_of(G, {(0, 0): st}, {"u": 7, "v": 8}, {key0: 0})
    for (sq, idx, blk, gen, dem) in infl:
        T.flat("seq")[idx] = sq
        T.flat("landed")[idx] = True
        T.flat("stamp")[idx] = 1000 + sq
        T.flat("demand")[idx] = (OPS.DEM_D if EX.DRAFT in dem else 0) | (OPS.DEM_V if EX.VERIFY in dem else 0)
        T.flat("issuer")[idx] = OPS.ISS_D
    for (sq, idx, blk, gen, dem) in infl:                          # the vendored admissions, in seq order
        st.admit(idx, "u", blk, gen, demand=tuple(sorted(dem)))
    OPS.poll_arrivals(T, tick=0, landed_upto=-1, lag=None, mut=mut)
    return st_state(st, True), tb_state(T, G, 0, True), ended


def test_B2T_A_admissions_equal_the_vendored():
    n_end = 0
    for seed in range(300):
        v, b, ended = _admit_case(seed)
        assert v == b, (seed, v, b)
        n_end += ended
    assert n_end > 30                                              # precondition: drop_unowned ran


def test_B2T_A2_negative_control_keep_rd_last_on_drop():
    res = [_admit_case(seed, {"keep_rd_last_on_drop": True}) for seed in range(300)]
    assert sum(1 for (v, b, _) in res if v != b) > 0


# ---------------------------------------------------------------------------------------------------- K
def _k5_case(seed, mut=None, flip=False):
    rng = random.Random(seed)
    B_, C, X, R = 3, 3, 3, 1
    NB = 64
    G = OPS.Geo(L=1, B=B_, H=1, C=C, X=X, R_ver=R, NBLK=NB, KEYS=4)
    pool = OWN.LayerPool(0, B_, 1, C, X, R)
    uidmap = {}
    for b in range(B_):
        pool.stream(b, 0).owner = "u%d" % b
        uidmap["u%d" % b] = b + 1
    nxt = [0] * B_
    seq = 0
    for b in range(B_):                                            # an initial usage that K5 allows
        st = pool.stream(b, 0)
        for _ in range(rng.randint(0, C + 2)):
            if pool.can_take(st, EX.VERIFY) and pool.first_free(st) is not None:
                seq += 1
                st.take(pool.first_free(st), st.owner, nxt[b], seq, cls="v")
                nxt[b] += 1
    T = table_of(G, pool.streams, uidmap, {})
    evs = []
    for part, kname in ((0, EX.VERIFY), (1, EX.DRAFT)):          # K3X order: the verify part, then the draft part
        for _ in range(rng.randint(1, 8)):
            b = rng.randrange(B_)
            st = pool.stream(b, 0)
            if pool.can_take(st, kname) and pool.first_free(st) is not None:
                idx = pool.first_free(st)
                seq += 1
                st.take(idx, st.owner, nxt[b], seq)
                nxt[b] += 1
                evs.append((b, part, 0, idx))
            else:
                evs.append((b, part, 1 if part == 0 else 2, 0))    # Policy E (its victim slot: in range) / refused
    if flip:
        gi = [i for i, e in enumerate(evs) if e[2] == 0]
        if not gi:
            return None
        b, part, _, _ = evs[gi[0]]
        evs[gi[0]] = (b, part, 1 if part == 0 else 2, 0)
    n = len(evs)
    ev = dict(l=torch.zeros(n, dtype=I64), s=torch.tensor([e[0] for e in evs]), b=torch.tensor([e[0] for e in evs]),
              part=torch.tensor([e[1] for e in evs]), kind=torch.tensor([e[2] for e in evs]),
              take_slot=torch.tensor([e[3] for e in evs]), o=torch.arange(n))
    out = OPS.k5_check(T, ev, mut)
    return bool(out["k5"].any() or out["first_free"].any() or out["slot_range"].any())


def test_B2T_K_k5_and_first_free_equal_the_vendored():
    for seed in range(300):
        assert _k5_case(seed) is False, seed


def test_B2T_K2_negative_controls_inclusive_prefix_and_a_flipped_outcome():
    assert sum(1 for seed in range(300) if _k5_case(seed, {"inclusive_prefix": True})) > 0
    flips = [_k5_case(seed, flip=True) for seed in range(300)]
    assert all(f for f in flips if f is not None) and sum(f is not None for f in flips) > 100


# ---------------------------------------------------------------------------------------------------- E
def _pe_case(seed, mut=None, plant=None):
    rng = random.Random(seed)
    C, X, NB = 6, 2, 64
    G = OPS.Geo(L=1, B=1, H=1, C=C, X=X, R_ver=0, NBLK=NB, KEYS=4)
    st = one_stream(C, X)
    key0 = ("u", 0, 0)
    blocks = rng.sample(range(NB), C)
    idxs = rng.sample(range(C + X), C)
    for i in range(C):
        lru_entry(st, idxs[i], blocks[i], i + 1)
        if rng.random() < 0.4:
            st.protect(blocks[i], key0, EX.VERIFY)
    sel = set(rng.sample(blocks, rng.randint(0, C // 2)))
    expected = [(blk, idx) for blk, idx in st.lru.items() if not st.prot.get(blk) and blk not in sel]   # res.py:338-342
    if not expected:
        return None
    n_pe = rng.randint(1, len(expected))
    victims = expected[:n_pe]
    if plant == "pinned":
        pinned = [(blk, idx) for blk, idx in st.lru.items() if st.prot.get(blk)]
        if not pinned:
            return None
        victims = [pinned[0]] + victims[1:]
    T = table_of(G, {(0, 0): st}, {"u": 7}, {key0: 0})
    selb = torch.zeros(NB, dtype=torch.bool)
    selb[list(sel)] = True
    sel_w = SP._pack_bits(selb).view(1, 1, 1, G.WB)
    ev = dict(l=torch.zeros(n_pe, dtype=I64), s=torch.zeros(n_pe, dtype=I64), b=torch.zeros(n_pe, dtype=I64),
              kind=torch.ones(n_pe, dtype=I64), part=torch.zeros(n_pe, dtype=I64),
              take_slot=torch.tensor([v[1] for v in victims]), blk=torch.full((n_pe,), 63), src=torch.arange(n_pe))
    P = dict(V=dict(flags=torch.full((n_pe,), OPS.EVICT, dtype=I64), evict_blk=torch.tensor([v[0] for v in victims])),
             D=None)
    out = OPS.policy_e_check(T, ev, P, sel_w, OPS.any_pin_words(T), True, mut)
    return bool(out["policy_e"].any()), len(expected)


def test_B2T_E_policy_e_equals_the_vendored_rule():
    ran = 0
    for seed in range(300):
        r = _pe_case(seed)
        if r is not None:
            assert r[0] is False, seed
            ran += 1
    assert ran > 100


def test_B2T_E2_negative_controls_newest_first_and_a_pinned_victim():
    nf = [_pe_case(seed, {"newest_first": True}) for seed in range(300)]
    assert sum(1 for r in nf if r is not None and r[1] >= 2 and r[0]) > 0
    pv = [_pe_case(seed, plant="pinned") for seed in range(300)]
    assert all(r[0] for r in pv if r is not None) and sum(r is not None for r in pv) > 50


# ---------------------------------------------------------------------------------------------------- P
def _pins(mut=None):
    G = OPS.Geo(L=2, B=3, H=2, C=4, X=4, R_ver=0, NBLK=64, KEYS=4)
    T = OPS.Table(G)
    g = torch.Generator().manual_seed(5)
    rows_b = torch.tensor([1, 1, 1, 2, 0, 1])
    planes = torch.tensor([1, 1, 1, 5, 1, 6])                       # three rows share (plane 1, b 1)
    sel = torch.rand((G.L, 6, G.H, G.NBLK), generator=g) < 0.4
    nr = OPS.pins_or(T, SP._pack_bits(sel), rows_b, planes, None, mut)
    want = torch.zeros((2 * G.KEYS, G.L, G.B, G.H, G.NBLK), dtype=torch.bool)
    for r in range(6):
        want[planes[r], :, rows_b[r]] |= sel[:, r]
    got = SP.unpack_bits(T.pin_w, G.NBLK)
    return nr, torch.equal(got, want)


def test_B2T_P_pin_rounds_equal_the_union():
    nr, ok = _pins()
    assert nr == 3 and ok


def test_B2T_P2_negative_control_sum_not_or():
    assert not _pins({"sum_not_or": True})[1]


# ---------------------------------------------------------------------------------------------------- D
def test_B2T_D_key_drops_equal_the_vendored_unprotect():
    for seed in range(100):
        rng = random.Random(seed)
        C, X, NB = 4, 4, 32
        G = OPS.Geo(L=1, B=1, H=1, C=C, X=X, R_ver=0, NBLK=NB, KEYS=4)
        st = one_stream(C, X)
        k1, k2 = ("u", 0, 1), ("u", 0, 2)
        blocks = rng.sample(range(NB), C + 3)
        idxs = rng.sample(range(C + X), C + 3)
        for i in range(C):
            lru_entry(st, idxs[i], blocks[i], i + 1)
        for j in range(3):
            i = C + j
            vres_entry(st, idxs[i], blocks[i], i + 1, rng.choice([k1, k2]), po=j == 0)
        for blk in blocks[:C + 3]:
            for key in (k1, k2):
                if rng.random() < 0.4:
                    st.protect(blk, key, rng.choice([EX.VERIFY, EX.DRAFT]))
        keymap = {k1: 1, k2: 2}
        T = table_of(G, {(0, 0): st}, {"u": 7}, keymap)
        for idx in st.unprotect(k1):                               # the vendored _drop (res.py:298-305)
            st.release(idx, "unprotect")
        T.rel_q[0, 0, 1] = True
        T.key_rd[0, 0, 1] = -1                                     # its reader is complete
        OPS.drop_due_keys(T, OPS.ReaderWindow())
        assert st_state(st) == tb_state(T, G, 0), seed
        assert prot_of(st) == pins_of(T, G, 0, {1: k1, 2: k2}), seed
        assert not bool(T.rel_q.any())


# ---------------------------------------------------------------------------------------------------- F
def test_B2T_F_releases_equal_the_vendored_with_out_of_order_readers():
    freed = 0
    for seed in range(100):
        rng = random.Random(seed)
        C, X, NB = 4, 4, 32
        G = OPS.Geo(L=1, B=1, H=1, C=C, X=X, R_ver=0, NBLK=NB, KEYS=4)
        pool = OWN.LayerPool(0, 1, 1, C, X, 0)
        st = pool.stream(0, 0)
        st.owner = "u"
        win = OPS.ReaderWindow(cap=64)
        evs = [Ev(done=rng.random() < 0.5) for _ in range(6)]       # readers complete in any order
        ids = {id(e): win.register(e, i) for i, e in enumerate(evs)}
        blocks = rng.sample(range(NB), C)
        for i in range(C):
            lru_entry(st, i, blocks[i], i + 1)
        for blk, idx in list(st.lru.items()):
            st.slots[idx].last_reader = rng.choice(evs + [None])
            if rng.random() < 0.6:
                st.lru.pop(blk)
                st.release(idx, "test")
        T = table_of(G, pool.streams, {"u": 7}, {}, rid=lambda r: -1 if r is None else ids[id(r)])
        win.refresh(_done)
        nv = pool.poll_releases()
        nb = OPS.poll_releases(T, win)
        assert st_state(st) == tb_state(T, G, 0) and nv == nb, seed
        freed += nb
    assert freed > 20                                              # precondition: releases happened


# ---------------------------------------------------------------------------------------------------- I, U, H
def _smoke_world(tmp_path=None):
    a = BT.parse_args(["--smoke", "--out", str(tmp_path / "x.json") if tmp_path else "x.json"])
    W = BT.build_world(a)
    T1 = W["T"].clone()
    win = BT.fresh_window(a, W["G"])
    BT.run_poll(T1, win, a)
    return a, W, T1, win


def test_B2T_I_post_checks_fire_on_their_planted_state(tmp_path):
    a, W, T1, _ = _smoke_world(tmp_path)
    G = W["G"]
    clean = OPS.post_checks(T1, T1.c["gen"].clone(), 0)
    assert not any(clean.values()), clean
    code = T1.c["code"].view(G.S, G.NPOS)
    block = T1.c["block"].view(G.S, G.NPOS)
    s = int(((code == OPS.C_LRU).sum(1) < G.C).nonzero()[0])
    lru_slot = int((code[s] == OPS.C_LRU).nonzero()[0])
    free = (code[s] == OPS.C_FREE).nonzero().view(-1)
    T = T1.clone()
    T.c["code"].view(G.S, G.NPOS)[s, free[0]] = OPS.C_LRU         # I1: a second slot for one resident block
    T.c["block"].view(G.S, G.NPOS)[s, free[0]] = block[s, lru_slot]
    assert OPS.post_checks(T, None, 0)["I1"]
    T = T1.clone()
    T.c["code"].view(G.S, G.NPOS)[s] = OPS.C_LRU                  # I2: every slot of s LRU (NPOS > C), distinct blocks
    T.c["block"].view(G.S, G.NPOS)[s] = torch.arange(G.NPOS, dtype=torch.int32)
    assert OPS.post_checks(T, None, 0)["I2"]
    T = T1.clone()
    T.pin_w.zero_()                                                # I3: PO / VRES unpinned
    assert OPS.post_checks(T, None, 0)["I3"]
    T = T1.clone()
    gb = T.c["gen"].clone()
    T.flat("gen")[0] += 2                                          # I9: a generation jump
    assert OPS.post_checks(T, gb, 1)["I9"]


def test_B2T_U_void_tick_restores_the_table_and_a_clean_tick_commits(tmp_path):
    a, W, T1, win = _smoke_world(tmp_path)
    R = BT.gen_records(W, T1, a)
    T = T1.clone()
    res = BT.run_apply(T, win, W, R, a, None, {"gen_plus_two": True})
    assert res["void"] and res["viol"].get("I9", 0) == 1
    assert T.diff(T1) == [], T.diff(T1)
    res = BT.run_apply(T, win, W, R, a, None, None)
    assert not res["void"] and not any(res["viol"].values()), res["viol"]
    assert "code" in T.diff(T1) and res["takes"] == a.urgent + a.d_take


def test_B2T_H_harness_smoke_both_modes(tmp_path):
    """B2T-H: --smoke, both modes: exit 0, counts as requested, no predicate fires, the checked and unchecked final tables
    are EQUAL (the unchecked mode changes no state), every component timed; the unchecked mode skips CHECK_ONLY."""
    out = tmp_path / "b2t.json"
    rc = BT.main(["--smoke", "--out", str(out), "--iters", "2", "--warmup", "1", "--pipelined-iters", "2"])
    d = json.loads(out.read_text())
    assert rc == 0 and d["failed"] == [], d["failed"]
    assert d["counts_generated"]["d_take"] == d["counts_requested"]["d_take"] and d["modes_state_equal"] is True
    ck, uk = d["modes"]["checked"], d["modes"]["unchecked"]
    assert not any(ck["apply"]["viol"].values()) and ck["apply"]["void"] is False and uk["apply"]["void"] is False
    assert ck["apply"]["desc_contiguous"] and ck["apply"]["desc_shape"][0] == 6
    assert ck["apply"]["desc_shape"] == uk["apply"]["desc_shape"] and ck["apply"]["metrics"] == uk["apply"]["metrics"]
    for name in ("poll.arrivals", "poll.checks", "poll.releases", "apply.parse", "apply.k5_first_free",
                 "apply.policy_e", "apply.pre_flags", "apply.pins", "apply.takes", "apply.recency", "apply.post_checks",
                 "apply.post_flags", "apply.readers", "apply.metrics", "apply.issue_desc"):
        assert ck["components_ms"][name]["n"] == 2, name
    assert set(BT.CHECK_ONLY) <= set(ck["components_ms"]) and not set(BT.CHECK_ONLY) & set(uk["components_ms"])
    assert ck["pipelined_ms"]["n"] == 2 and ck["synced_total_ms"] > 0 and d["total_ms"] == ck["synced_total_ms"]
    assert d["verdicts"]["host_25ms"]["applies"] is False and d["device"] == "cpu"


def test_B2T_G1_cuda_is_refused_without_cuda(tmp_path, monkeypatch):
    """B2T-G1: --device cuda without CUDA exits 2 before any work (no output file)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "g.json"
    assert BT.main(["--smoke", "--device", "cuda", "--out", str(out)]) == 2
    assert not out.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA path runs only where CUDA exists (the GPU step)")
def test_B2T_G2_cuda_equals_cpu_on_the_smoke_world(tmp_path):
    """B2T-G2 (GPU only): the same tick on CUDA tensors ends in the same table as on the host, in both modes."""
    a, W, T1, _ = _smoke_world(tmp_path)
    R = BT.gen_records(W, T1, a)
    for checked in (True, False):
        r_c = BT.time_on("cpu", a, W, R, W["T"], checked)
        r_g = BT.time_on("cuda", a, W, R, W["T"], checked)
        assert r_c["final"].diff(r_g["final"]) == [] and r_c["apply"]["metrics"] == r_g["apply"]["metrics"]


def test_B2T_M_unchecked_skips_only_verification(tmp_path):
    """B2T-M: on the smoke world the unchecked apply ends in the checked apply's table, and a planted defect that only a
    predicate sees (gen_plus_two) voids the checked tick but is NOT caught unchecked (the mode trusts the decisions)."""
    a, W, T1, win = _smoke_world(tmp_path)
    R = BT.gen_records(W, T1, a)
    Tc, Tu = T1.clone(), T1.clone()
    rc = BT.run_apply(Tc, win, W, R, a, None, None, True)
    _, _, _, win2 = _smoke_world(tmp_path)
    ru = BT.run_apply(Tu, win2, W, R, a, None, None, False)
    assert not rc["void"] and not ru["void"] and Tc.diff(Tu) == []
    Tc, Tu = T1.clone(), T1.clone()
    _, _, _, win3 = _smoke_world(tmp_path)
    assert BT.run_apply(Tc, win3, W, R, a, None, {"gen_plus_two": True}, True)["void"]
    _, _, _, win4 = _smoke_world(tmp_path)
    assert not BT.run_apply(Tu, win4, W, R, a, None, {"gen_plus_two": True}, False)["void"]


def _registered_block(text):
    lines = text.splitlines()
    i, j = lines.index("# REGISTERED B2TG BEGIN"), lines.index("# REGISTERED B2TG END")
    return "\n".join(lines[i + 1:j])


def test_B2T_G3_step_script_uuid_file_pins_dryrun_and_registration(tmp_path):
    """B2T-G3: r2_own2_b2tg_step.sh. Static: the claim b2tg_active, STEP_MIN 10, --export=NONE, the uuid file read
    before the claim, the root, no awk / perl / python3 outside comments; its REGISTERED block equals the bench's
    REGISTERED_DEVICE. The uuid file rules (missing, "GPU-" prefix, two lines) and B2TG_HOLDERS outside DRYRUN stop
    with exit 1; a wrong --nosi stops at the HEAD check; on a clean checkout at HEAD the DRYRUN prints the pins and
    the step command (the uuid, --export=NONE, --device cuda, both modes, the profile, the host reference)."""
    import os
    import subprocess
    script = EFF / "r2_own2_b2tg_step.sh"
    text = script.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and lines[-1] == 'main "$@"; exit $?' and "main() {" in text
    assert 'mkdir "$OUTROOT/b2tg_active"' in text and "STEP_MIN=10" in text and "--export=NONE" in text
    assert text.index("UFILE=$HOLDERS/$JOB.gpu_uuid") < text.index('mkdir "$OUTROOT/b2tg_active"')
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-m2-own2 " in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    blk = _registered_block(text)
    RD = BT.REGISTERED_DEVICE
    for key, label in (("D1_pipelined_ms", "D1 checked pipelined median per round tick"),
                       ("D1u_pipelined_unchecked_ms", "D1u unchecked pipelined median per round tick"),
                       ("D2_synced_ms", "D2 checked synced component sum"), ("D3_busy_ms", "D3 profiled device busy time")):
        r = RD[key]
        want = "# %s: PREDICTION %.1f ms, band %.1f-%.1f ms; FALSIFIER > %.1f ms" % (
            label, r["point"], r["band"][0], r["band"][1], r["falsified_above"])
        assert want in blk, (want, blk)
    assert "# D4 kernel launches per tick: band %d-%d (point %d)" % (RD["D4_launches"]["band"] + (RD["D4_launches"]["point"],)) in blk
    assert "D5 host synchronizations: band %d-%d (point %d)" % (RD["D5_syncs"]["band"] + (RD["D5_syncs"]["point"],)) in blk
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}

    def run(job, content=None, dry=True, sha="0" * 40):
        if content is not None:
            (tmp_path / ("%s.gpu_uuid" % job)).write_text(content)
        e = dict(env, B2TG_HOLDERS=str(tmp_path))
        if dry:
            e["DRYRUN"] = "1"
        else:
            e.pop("DRYRUN", None)
        return subprocess.run(["bash", str(script), "--nosi", sha, str(job), "t1"], env=e, capture_output=True,
                              text=True, timeout=120)
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    r = run(611)
    assert r.returncode == 1 and "no GPU uuid file" in r.stderr
    r = run(612, "GPU-%s\n" % uid)
    assert r.returncode == 1 and "does not hold a uuid" in r.stderr
    r = run(613, "%s\n%s\n" % (uid, uid))
    assert r.returncode == 1 and "more than one line" in r.stderr
    r = run(614, uid + "\n", dry=False)
    assert r.returncode == 1 and "for DRYRUN" in r.stderr
    r = run(615, uid + "\n")
    assert r.returncode == 1 and "HEAD is not" in r.stderr
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        pytest.skip("not a clean git checkout: the full DRYRUN needs one")
    r = run(616, uid + "\n", sha=head.stdout.strip())
    assert r.returncode == 0, r.stderr
    for need in ("DRYRUN", uid, "--export=NONE", "--device cuda", "--mode both", "--profile", "--cpu-reference",
                 "--gres=gpu:a100:1", "--time=10", "r2_own2_b2t.py"):
        assert need in r.stdout, need


# ---------------------------------------------------------------------------------------------------- L
ALLOWED_LOOPS = {
    "SLOT_COLS", "ARRAYS", "mine.items()", "mine", "reversed(self.log)", "range(self.lo, self.next)",
    "range(1, pw.shape[0])", "(k_i, k_i + g.KEYS)",
    '(("V", "vrec", 9, rows.vb, rows.vk), ("D", "drec", 5, rows.db, rows.dk))', "layers", "recs", "parts",
    '("l", "s", "b", "h", "blk", "take_slot", "key", "kind", "part", "o", "src")',
    '("l", "s", "b", "h", "blk", "take_slot", "key", "o")', "cols.items()", "ev.items()",
    '(("v", rows.vb, rows.vk), ("d", rows.db, rows.dk))', '("take_slot", "blk")', "range(n)",
    "((fv, DEM_V), (fd, DEM_D))", "plan", "((rows.vb, rows.vk), (rows.db, rows.dk))",
}


def test_B2T_L_no_per_entry_python_loop():
    src = (NOSI / "r2_own2_ops.py").read_text()
    tree = ast.parse(src)
    loops = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            loops.append(ast.get_source_segment(src, node.iter))
    bad = sorted(set(x for x in loops if x not in ALLOWED_LOOPS))
    assert not bad, bad
    reads = src.count("int(") + src.count("bool(") + src.count(".item()")
    print("REPORT host scalar reads in r2_own2_ops.py: %d (int( / bool( / .item())" % reads)
