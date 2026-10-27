"""[f3ldx] CPU tests of the descriptor issue on the SHARED X POOL (E's GilfreeTransport.issue_desc with an XPool;
packer v3 DESC 2, benchmarks/Efficiency/gilfree_pack.cpp set_stores / submit_desc / release_desc) and of the cells
evalfast2f3c / evalfast2c3c of r2_tick_sg5_step.sh.

THE ADDRESS RULE (reused, not derived again). The object path gives each transfer its lane at end_tick:
r2_residency.py Residency.end_tick -> XPool.assign_phys (r2_xpool.py), which reads pos_of at
XPool.phys_index(layer, b, h, slot); GilfreeTransport._submit then sends m' = Transfer.phys with the flat store as a
(1, M * 64, H, D) view (r2_transport_gilfree.py, "tr.phys if flat else ..."). The descriptor path now reads the SAME
index (XPool.lanes_of -> phys_index) in GilfreeTransport._desc_lanes, adds the lane as row 6 of the descriptor, and
C++ release_desc writes m' = lane on the same (rows, D) view (the flat stores of set_stores). A 4-D store keeps DESC
1's logical rule (b * slots + slot) unchanged.

The packer is BUILT by the tests (packer v3 of this tree: test_r2_b2_desc.py's v3 fixture, gilfree_pack.load into a
pytest tmp directory; the OpenMP build): run them on a compute node, never on the login node.

  part a (-k "XD1 or XD2"), the transport alone on a small flat pool (random takes, random injective lanes bound in
       pos_of by hand):
  XD1  per seed: the takes issued as per-layer Transfer objects (xp.assign_phys, then issue: the object path) and as
       ONE descriptor (issue_desc) land store-equal stores (torch.equal, every layer, K and V) and the same board; both
       equal the host-truth model (the rows of each take at its lane, zeros elsewhere). One job, one release, the same
       groups. The 4-D path with DESC 2 is unchanged (stream: desc == object). XD1r refusals: an unbound slot
       (XPoolError), a lane outside the history arena, two takes on one lane (C++: the job fails, never lands, the
       error at the next poll), DESC 1 on the pool (refused before any job), a 6-row descriptor on flat stores and a
       7-row one on 4-D stores (C++ TORCH_CHECK).
  XD2  NEGATIVE CONTROL at the C++ site (set_plant_slot_shift(1): every lane one off, the plant_slot_ term of
       release_desc's flat line). Precondition (L34): the planted model (every take at lane + 1) differs from the
       clean model in at least one store row. Detected: the store is not the clean store, and it IS the planted model.
  part b (-k "XD3 or XD0 or XD4"), the round on the pool world (test_r2_xpool.make_pool_engine with E's transport):
  XD3  RANDOMIZED STREAMS (random.Random(seed): the configuration, the data seed, forced or free-running, the event
       lateness, arrival_lag 1 / 2, checked, lean, --b2-fewer-reads, --wait-index, --res-lean / --res-dev, the array
       poll, --no-transfer-mirror on the desc arm) plus the eval flag set: desc on pool == object on pool at EVERY TICK
       and every layer (rows, tokens, logits, pages, the projection, Table.diff, counters, readers, memory, arrival
       counts, the STORE by torch.equal, the layers' device decisions), the final stores, the committed tokens, the
       journal, the report's comparable parts. At every descriptor job the lanes it sends equal assign_phys's lanes of
       the same takes (the object path's rule, called on stand-in transfers).
  XD0  PRECONDITIONS over the XD3 draws: every desc arm issued jobs through the lane row (one _desc_lanes per job);
       some lane differs from the logical address b * n_hist + slot (a logical rule would land elsewhere); a lane
       changed hands between requests (xp.handoffs); takes were held; the object arm released per layer.
  XD4  NEGATIVE CONTROL in the round (L34, L49): at the first descriptor job where the planted model (every take at
       lane + 1, set_plant_slot_shift(1) for that ONE job, the C++ term) changes a store row of the snapshot taken at
       issue, the job is planted. Detected: the first difference from the clean desc round is that tick's STORE, and
       the planted lanes hold the takes' host rows.
  part c (-k "XD5 or XD6 or XD7"), static and the harness:
  XD5  the refusals are gone only for the supported case (OwnResidency(issue='desc') with an X pool: DESC >= 2 and the
       transport on the same XPool, else refused at construction); the address-rule wiring (assign_phys and lanes_of
       read phys_index; _desc_lanes calls lanes_of; release_desc's flat line; DESC 2, VERSION 3); r2_tick_sg5 parses
       --b2-issue desc with --layout pool / both; T21 static lists (one_point's gate helpers and gate names, main's gate
       calls); the cells evalfast2f3c / evalfast2c3c (static), F3LD_BLOCKED gone, the REGISTERED F3LD block unstamped
       with every PREDICTION "TBD by lead".
  XD6  the DRYRUN argv of all six f3ld cells (evalfast2f3d, evalfast2c3d, evalfast2f3t, evalfast2c3t, evalfast2f3c,
       evalfast2c3c) on a clean checkout of this ROOT; each parses on --layout pool (the TBD bands aside).
  XD7  outside DRYRUN: the timing cells stop on their TBD bands, the CHECK cells at the SG5_HOLDERS rule; none stops
       on a blocker.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_f3ldx.py
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import random
import re
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


BD = _shared("test_r2_b2_desc_shared_f3ld", "test_r2_b2_desc.py")      # the desc world (_run, _engine; pool=True)
F3 = _shared("test_r2_f3ld_shared_f3ldx", "test_r2_f3ld.py")          # _state_tbl, _first_diff, the cell helpers
TT, TRM, OWN2, RSC, GF = BD.TT, BD.TRM, BD.OWN2, BD.RSC, BD.GF
HF, RF = F3.HF, F3.RF
XPM = TT._load("r2_xpool")
KTR = TT.KTR
p3, ref, v3 = BD.p3, BD.ref, BD.v3                                    # the fixtures, by name
BS = BD.BS
ROOT6 = "/mnt/beegfs/ojerbi/nosi-r2-f3tall"                          # [f3tall] (was nosi-r2-f3ldx)
CELLS6 = ("evalfast2f3d", "evalfast2c3d", "evalfast2f3t", "evalfast2c3t", "evalfast2f3c", "evalfast2c3c")
TIMING = ("evalfast2f3d", "evalfast2f3t", "evalfast2f3c")


# =====================================================================================================================
# part a: the transport alone on a small flat pool
# =====================================================================================================================
L_, B_, H_, D_, NS_, C_, XPL_, NB_ = 2, 2, 2, 4, 10, 4, 6, 12           # A = B C + X_pool = 14 lanes, M = A + 4 B = 22


def _pool_world(seed):
    g = torch.Generator().manual_seed(seed)
    xp = XPM.XPool(L_, B_, H_, NS_, mode="pool", C=C_, X_pool=XPL_)
    host_k = [torch.randn(B_, NB_ * BS, H_, D_, generator=g) for _ in range(L_)]
    host_v = [torch.randn(B_, NB_ * BS, H_, D_, generator=g) for _ in range(L_)]
    st = [xp.new_store(D_) for _ in range(L_)]
    return xp, host_k, host_v, [s[0] for s in st], [s[1] for s in st]


def _takes_lanes(seed):
    """Random takes (l, b, h, block, slot, gen), the logical destinations distinct per layer, and a random INJECTIVE
    lane per (l, h) in the history arena [0, A) for each take (as the binder would give: one slot per lane)."""
    rnd = random.Random(seed)
    A = B_ * C_ + XPL_
    takes, lanes = [], {}
    for l in range(L_):
        n = rnd.randint(3, 12)
        dst = rnd.sample([(b, h, s) for b in range(B_) for h in range(H_) for s in range(NS_)], n)
        free = {h: rnd.sample(range(A), A) for h in range(H_)}
        for (b, h, s) in dst:
            takes.append((l, b, h, rnd.randrange(NB_), s, 3 + len(takes)))
            lanes[(l, b, h, s)] = free[h].pop()
    return takes, lanes


def _bind(xp, lanes):
    for (l, b, h, s), p in lanes.items():
        xp.pos_of[l, b, h, s] = p


def _desc(takes):
    return torch.tensor([list(col) for col in zip(*takes)], dtype=torch.int64)       # (6, n)


def _tp(ext, xp, host_k, host_v, sk, sv, board=True):
    tp = GF.GilfreeTransport(ext, host_k, host_v, sk, sv, BS, D_, threads=2, host_wait=lambda ev: None, cap_groups=4,
                             worker="cpp", xpool=xp)
    bd = None
    if board:
        bd = RSC.GenBoard(L_, B_, H_, NS_ + XPM.N_TAIL, "cpu")
        tp.attach_board(bd)
    return tp, bd


def _model(host_k, host_v, takes, lanes, shift=0, base=None):
    """The host-truth stores: every take's host rows (block) at its lane + shift, column h; zeros (or base) elsewhere."""
    M = B_ * C_ + XPL_ + XPM.N_TAIL * B_
    ks = [torch.zeros(M * BS, H_, D_) if base is None else base[0][l].clone() for l in range(L_)]
    vs = [torch.zeros(M * BS, H_, D_) if base is None else base[1][l].clone() for l in range(L_)]
    for (l, b, h, blk, s, _g) in takes:
        p = lanes[(l, b, h, s)] + shift
        ks[l][p * BS:(p + 1) * BS, h] = host_k[l][b, blk * BS:(blk + 1) * BS, h]
        vs[l][p * BS:(p + 1) * BS, h] = host_v[l][b, blk * BS:(blk + 1) * BS, h]
    return ks, vs


def _obj_issue(tp, xp, takes):
    seq = 0
    for l in range(L_):
        trs = [KTR.Transfer(seq + i, "u%d" % b, l, b, h, blk, s, gen, "draft", ("u%d" % b, 1, 1))
               for i, (_, b, h, blk, s, gen) in enumerate(t for t in takes if t[0] == l)]
        seq += len(trs)
        if trs:
            xp.assign_phys(trs)                                       # the object path's rule (Residency.end_tick)
            tp.issue(l, trs)


@pytest.mark.parametrize("seed", list(range(6)))
def test_XD1_desc_on_pool_lands_what_the_object_path_lands(v3, seed):
    takes, lanes = _takes_lanes(seed)
    xa, hk, hv, ak, av = _pool_world(seed)
    _bind(xa, lanes)
    ga, aboard = _tp(v3, xa, hk, hv, ak, av)
    _obj_issue(ga, xa, takes)
    ga.flush()
    ga.close()
    xb, hk2, hv2, bk, bv = _pool_world(seed)
    _bind(xb, lanes)
    gb, bboard = _tp(v3, xb, hk2, hv2, bk, bv)
    calls = []
    real = gb._desc_lanes
    gb._desc_lanes = lambda d: calls.append(int(d.shape[1])) or real(d)
    s1 = gb.issue_desc(_desc(takes))
    gb.flush()
    assert s1 == 1 and gb.landed_upto() == 1 and calls == [len(takes)]
    gb.close()
    mk, mv = _model(hk, hv, takes, lanes)
    for l in range(L_):
        assert torch.equal(ak[l], bk[l]) and torch.equal(av[l], bv[l]), ("desc != object on the pool", l)
        assert torch.equal(bk[l], mk[l]) and torch.equal(bv[l], mv[l]), ("desc != the host-truth model", l)
    assert torch.equal(aboard.t, bboard.t) and int(bboard.t.count_nonzero()) == len(takes)
    assert gb.stats["groups"] == ga.stats["groups"] == len(takes) and gb.stats["published"] == len(takes)
    assert gb.stats["releases"] == 1 and ga.stats["releases"] == len({t[0] for t in takes}) and gb.threads_ok()
    # precondition: a lane differs from the logical address b * n_hist + slot (the rule under test is not a no-op)
    assert any(p != b * NS_ + s for (l, b, h, s), p in lanes.items()), "every lane is the logical address"


def test_XD1s_the_4d_path_is_unchanged_with_desc_2(v3):
    """The logical rule on 4-D stores (no X pool): desc == object (GT8's property) with the DESC 2 packer."""
    rnd = random.Random(9)
    g = torch.Generator().manual_seed(9)
    hk = [torch.randn(B_, NB_ * BS, H_, D_, generator=g) for _ in range(L_)]
    hv = [torch.randn(B_, NB_ * BS, H_, D_, generator=g) for _ in range(L_)]
    out = []
    takes = []
    for l in range(L_):
        dst = rnd.sample([(b, h, s) for b in range(B_) for h in range(H_) for s in range(NS_)], 7)
        takes += [(l, b, h, rnd.randrange(NB_), s, 3 + len(takes) + i) for i, (b, h, s) in enumerate(dst)]
    for mode in ("object", "desc"):
        sk = [torch.zeros(B_, NS_ * BS, H_, D_) for _ in range(L_)]
        sv = [torch.zeros(B_, NS_ * BS, H_, D_) for _ in range(L_)]
        tp = GF.GilfreeTransport(v3, hk, hv, sk, sv, BS, D_, threads=2, host_wait=lambda ev: None, cap_groups=4,
                                 worker="cpp")
        if mode == "object":
            for l in range(L_):
                tp.issue(l, [KTR.Transfer(i, "u%d" % b, l, b, h, blk, s, gen, "draft", ("u%d" % b, 1, 1))
                             for i, (_, b, h, blk, s, gen) in enumerate(t for t in takes if t[0] == l)])
        else:
            tp.issue_desc(_desc(takes))
        tp.flush()
        if mode == "desc":
            with pytest.raises(RuntimeError, match=re.escape("(6, n)")):    # a 7-row descriptor on 4-D stores
                tp.npipe.submit_desc(torch.zeros((7, 1), dtype=torch.int64), [], 0)
        tp.close()
        out.append((sk, sv))
    for l in range(L_):
        assert torch.equal(out[0][0][l], out[1][0][l]) and torch.equal(out[0][1][l], out[1][1][l]), l


def test_XD1r_refusals_on_the_pool(v3):
    takes, lanes = _takes_lanes(3)
    # an unbound slot: refused before any job (assign_phys's refusal)
    xp, hk, hv, sk, sv = _pool_world(3)
    _bind(xp, lanes)
    l0, b0, h0, _, s0, _ = takes[0]
    xp.pos_of[l0, b0, h0, s0] = -1
    tp, _ = _tp(v3, xp, hk, hv, sk, sv)
    with pytest.raises(XPM.XPoolError, match="no bound lane"):
        tp.issue_desc(_desc(takes))
    assert tp._last_seq == 0
    # a lane outside the history arena (a tail position): refused before any job
    xp.pos_of[l0, b0, h0, s0] = xp.A
    with pytest.raises(ValueError, match="history arena"):
        tp.issue_desc(_desc(takes))
    assert tp._last_seq == 0
    # two takes of one (l, h) on one lane: C++ refuses the job; it never lands; the error at the next poll
    same = [t for t in takes[1:] if t[0] == l0 and t[2] == h0]
    assert same, "precondition: a second take of the same (layer, head)"
    l1, b1, h1, _, s1, _ = same[0]
    xp.pos_of[l0, b0, h0, s0] = lanes[(l1, b1, h1, s1)]
    seq = tp.issue_desc(_desc(takes))
    tp.flush()
    assert seq == 1 and tp.landed_upto() == 0
    with pytest.raises(RuntimeError, match="two transfers into one lane"):
        tp.poll()
    # a 6-row descriptor on flat stores (the raw entry point)
    with pytest.raises(RuntimeError, match=re.escape("(7, n)")):
        tp.npipe.submit_desc(_desc(takes[:1]), [], 0)
    tp.close()
    # DESC 1 on the pool: refused before any job
    ns = SimpleNamespace(VERSION=3, DESC=1, OPENMP=1, SLOT_EXTRA_PER_GROUP=v3.SLOT_EXTRA_PER_GROUP,
                         NativePipe=v3.NativePipe)
    xq, hk, hv, sk, sv = _pool_world(4)
    _bind(xq, lanes)
    t1 = GF.GilfreeTransport(ns, hk, hv, sk, sv, BS, D_, threads=2, host_wait=lambda ev: None, cap_groups=4,
                             worker="cpp", xpool=xq)
    with pytest.raises(RuntimeError, match="DESC >= 2"):
        t1.issue_desc(_desc(takes))
    assert t1._last_seq == 0
    t1.close()


def test_XD2_a_wrong_lane_offset_changes_the_store_and_is_caught(v3):
    takes, lanes = _takes_lanes(5)
    xp, hk, hv, ck, cv = _pool_world(5)
    _bind(xp, lanes)
    clean = _model(hk, hv, takes, lanes)
    planted = _model(hk, hv, takes, lanes, shift=1)
    changed = [l for l in range(L_) if not (torch.equal(clean[0][l], planted[0][l])
                                            and torch.equal(clean[1][l], planted[1][l]))]
    assert changed, "precondition (L34): the planted lanes would change no store row"
    tp, _ = _tp(v3, xp, hk, hv, ck, cv)
    tp.npipe.set_plant_slot_shift(1)                                  # NEGATIVE CONTROL: every lane one off (C++)
    tp.issue_desc(_desc(takes))
    tp.flush()
    tp.close()
    assert not all(torch.equal(ck[l], clean[0][l]) and torch.equal(cv[l], clean[1][l]) for l in range(L_)), \
        "the wrong lane offset was not seen in the store"
    for l in range(L_):
        assert torch.equal(ck[l], planted[0][l]) and torch.equal(cv[l], planted[1][l]), ("not the planted model", l)


# =====================================================================================================================
# part b: the round on the pool world
# =====================================================================================================================
def _kw(d, desc):
    kw = dict(lean=d["lean"] or d["nm"], fewer_reads=d["fewer_reads"], wait_index=d["wait_index"],
              res_lean=d["res_lean"], res_dev=d["res_dev"], inflight_array=d["ia"])
    if desc and d["nm"]:
        kw.update(no_transfer_mirror=True, inflight_array=False)
    return kw


def _lane_probe(tp, world, box):
    """Wraps the transport's _desc_lanes: per job, the lanes it sends and the lanes the OBJECT PATH'S rule gives the
    same takes (XPool.assign_phys on stand-in transfers); the count of lanes that differ from b * n_hist + slot."""
    real = tp._desc_lanes
    xp = world.xp

    def lanes(desc):
        out = real(desc)
        trs = [SimpleNamespace(seq=i, layer=int(desc[0, i]), b=int(desc[1, i]), h=int(desc[2, i]),
                               slot=int(desc[4, i]), phys=None) for i in range(int(desc.shape[1]))]
        xp.assign_phys(trs)
        box["jobs"] += 1
        box["same_rule"] = box["same_rule"] and [t.phys for t in trs] == out[6].tolist()
        box["not_logical"] += sum(1 for t in trs if t.phys != t.b * xp.n_hist + t.slot)
        return out
    tp._desc_lanes = lanes


def _round(p3, v3, toks, d, issue, plant=None):
    box = dict(jobs=0, same_rule=True, not_logical=0)

    def setup_plant(tp, svc, ticks, world, probes):
        if issue == "desc":
            _lane_probe(tp, world, box)
        if plant is not None:
            plant(tp, svc, ticks, world, probes)
    nm = issue == "desc" and d["nm"]
    r = BD._run(p3, v3, toks, issue, cfg=d["cfg"], checked=d["checked"], late=d["late"], plant=setup_plant,
                seed=d["seed"], lag=d["lag"], svc_kw=_kw(d, issue == "desc"), forced=d["forced"],
                state=F3._state_tbl if d["nm"] else None, layer_rec=RF._layer_rec, pool=True)
    r.box = box
    r.nm = nm
    return r


def _draws(n=8, seed0=7300):
    out = []
    for i in range(n):
        r = random.Random(seed0 + i)
        rl = r.random() < 0.5
        nm = r.random() < 0.4
        d = dict(name="rand%d" % i, cfg=r.choice(("default", "pressure")), seed=r.choice((1, 2, 3)),
                 forced=r.random() < 0.6, late=r.choice((0, 2)), lag=r.choice((1, 2)), checked=r.random() < 0.3,
                 lean=r.random() < 0.5, fewer_reads=r.random() < 0.5, wait_index=r.random() < 0.5, res_lean=rl,
                 res_dev=rl and r.random() < 0.5, ia=r.random() < 0.5, nm=nm)
        out.append(d)
    return out


FIXED = [dict(name="eval-d", cfg="pressure", seed=1, forced=True, late=2, lag=2, checked=False, lean=True,
              fewer_reads=True, wait_index=True, res_lean=True, res_dev=True, ia=True, nm=False),
         dict(name="eval-t", cfg="pressure", seed=1, forced=True, late=2, lag=2, checked=False, lean=True,
              fewer_reads=True, wait_index=True, res_lean=True, res_dev=True, ia=True, nm=True),
         dict(name="checked-free", cfg="default", seed=2, forced=False, late=2, lag=2, checked=True, lean=False,
              fewer_reads=False, wait_index=False, res_lean=False, res_dev=False, ia=False, nm=False)]
DRAWS = FIXED + _draws()
_PAIRS = {}


def _pair(p3, v3, toks, d):
    if d["name"] not in _PAIRS:
        _PAIRS[d["name"]] = (_round(p3, v3, toks, d, "object"), _round(p3, v3, toks, d, "desc"))
    return _PAIRS[d["name"]]


@pytest.mark.parametrize("d", DRAWS, ids=[d["name"] for d in DRAWS])
def test_XD3_desc_on_pool_equals_object_on_pool_at_every_tick(p3, ref, v3, d):
    toks, _ = ref
    ob, de = _pair(p3, v3, toks, d)
    assert ob.err is None and len(ob.ticks) > 10, ("precondition: the object round ran", d, ob.err)
    diff = F3._first_diff(ob, de)
    assert diff is None, ("desc on pool differs from object on pool", d, diff, de.err)
    for i, (x, y) in enumerate(zip(ob.final, de.final)):
        assert torch.equal(x, y), ("final store tensor", i)
    assert [r.out for r in de.rnd.reqs] == [r.out for r in ob.rnd.reqs]
    assert TRM._journal_kinds(de.svc) == TRM._journal_kinds(ob.svc)
    ro, rd = BD._rep(ob.svc), BD._rep(de.svc)
    for k in ro:
        assert rd[k] == ro[k], (k, ro[k], rd[k])
    assert de.svc.xpool is not None and ob.svc.xpool is not None and de.tp.xpool is de.svc.xpool
    assert de.svc.issue == "desc" and ob.svc.issue == "object" and de.svc.no_transfer_mirror == d["nm"]
    assert de.box["same_rule"], "a job's lanes differ from assign_phys's lanes of the same takes"
    assert de.box["jobs"] == de.svc.desc_jobs > 0 and de.stats["releases"] == de.svc.desc_jobs
    assert ob.stats["groups"] == de.stats["groups"] == de.svc.desc_takes, (ob.stats, de.stats)
    print("XD3 %s: ticks %d, jobs %d, takes %d, lanes not logical %d, handoffs %d, held %d, nm %s"
          % (d["name"], len(de.ticks), de.svc.desc_jobs, de.svc.desc_takes, de.box["not_logical"],
             int(de.eng.res_world.xp.handoffs), de.svc.arrival_held, d["nm"]))


def test_XD0_preconditions_over_the_draws(p3, ref, v3):
    toks, _ = ref
    pairs = [_pair(p3, v3, toks, d) for d in DRAWS]                    # cached by XD3
    assert all(de.box["jobs"] > 0 and de.box["jobs"] == de.svc.desc_jobs for _, de in pairs), \
        "precondition: a desc arm issued no job through the lane row"
    assert sum(de.box["not_logical"] for _, de in pairs) > 0, "precondition: every lane equals the logical address"
    assert sum(int(de.eng.res_world.xp.handoffs) for _, de in pairs) > 0, "precondition: no lane changed hands"
    assert sum(de.svc.arrival_held for _, de in pairs) > 0, "precondition: no take was held"
    assert all(ob.stats["releases"] > de.stats["releases"] for ob, de in pairs), "the object arm did not release per layer"
    assert any(d["nm"] for d in DRAWS) and any(not d["nm"] for d in DRAWS)


def test_XD4_a_wrong_lane_offset_in_the_round_is_first_seen_in_that_ticks_store(p3, ref, v3):
    toks, _ = ref
    d = FIXED[0]
    clean = _pair(p3, v3, toks, d)[1]
    assert clean.err is None and len(clean.ticks) > 10, clean.err
    rec = {}

    def plant(tp, svc, ticks, world, probes):
        orig = tp.issue_desc
        xp = world.xp

        def rows(t, l, h, p):
            return t[l][p * BS:(p + 1) * BS, h].clone()

        def issue_desc(desc, reader_marks=None, hist_mark=-1):
            if "tick" in rec or int(desc.shape[1]) == 0:
                return orig(desc, reader_marks, hist_mark)
            lanes = xp.lanes_of(desc).tolist()                        # the transport's own read (no state change)
            ent = [(int(desc[0, i]), int(desc[1, i]), int(desc[2, i]), int(desc[3, i]), lanes[i])
                   for i in range(int(desc.shape[1]))]
            if max(p for *_, p in ent) + 1 >= xp.M:
                return orig(desc, reader_marks, hist_mark)
            keys = {(l, h, p + j) for (l, b, h, blk, p) in ent for j in (0, 1)}
            s0 = {k: (rows(tp.slots_k, *k), rows(tp.slots_v, *k)) for k in keys}   # the store at issue
            src = {}
            for (l, b, h, blk, p) in ent:
                src[(l, h, p)] = (world.host_k[l][b, blk * BS:(blk + 1) * BS, h].clone(),
                                  world.host_v[l][b, blk * BS:(blk + 1) * BS, h].clone())
            good, bad = dict(s0), dict(s0)
            for (l, h, p), v in src.items():
                good[(l, h, p)] = v
                bad[(l, h, p + 1)] = v
            changes = [k for k in keys if not (torch.equal(good[k][0], bad[k][0]) and torch.equal(good[k][1], bad[k][1]))]
            if not changes:
                return orig(desc, reader_marks, hist_mark)            # L34: the plant would be a no-op here
            tp.npipe.set_plant_slot_shift(1)                          # NEGATIVE CONTROL: this ONE job, every lane + 1
            try:
                seq = orig(desc, reader_marks, hist_mark)
                tp.npipe.wait_job(seq)
            finally:
                tp.npipe.set_plant_slot_shift(0)
            landed = all(torch.equal(rows(tp.slots_k, l, h, p + 1), v[0]) and torch.equal(rows(tp.slots_v, l, h, p + 1), v[1])
                         for (l, h, p), v in src.items())
            rec.update(tick=len(ticks), changes=len(changes), takes=len(ent), landed=landed)
            return seq
        tp.issue_desc = issue_desc
    bad = _round(p3, v3, toks, d, "desc", plant=plant)
    assert "tick" in rec, "precondition (L34): no job where the wrong lane offset changes a store row"
    t = rec["tick"]
    diff = F3._first_diff(clean, bad)
    assert diff == (t, "store"), ("the wrong lane offset is first seen in the store of its tick", diff, t, bad.err)
    assert rec["landed"], "the planted lanes do not hold the takes' host rows"
    print("XD4: plant at tick %d (%d takes, %d store rows changed by the model); bad round error %s"
          % (t, rec["takes"], rec["changes"], bad.err))


# =====================================================================================================================
# part c: static and the harness
# =====================================================================================================================
def test_XD5_refusals_wiring_t21_and_the_cells_static(p3, v3):
    # OwnResidency(issue='desc') on the pool: accepted with DESC 2 and the same XPool; refused otherwise (L31)
    e, _ = BD._engine(p3, v3, "desc", pool=True)
    assert e.res_world.svc.issue == "desc" and e.res_world.tp.xpool is e.res_world.svc.xpool
    e.res_world.tp.close()
    ns = SimpleNamespace(VERSION=3, DESC=1, OPENMP=1, SLOT_EXTRA_PER_GROUP=v3.SLOT_EXTRA_PER_GROUP,
                         NativePipe=v3.NativePipe)
    # a transport on another XPool: the base Residency check (r2_residency.py, review m2) refuses it first, before
    # OwnResidency's desc check; both refuse at construction (L31)
    for ext, other, msg in ((ns, False, "DESC >= 2"), (v3, True, "its .xpool is not this XPool")):
        def make(be, host_k, host_v, sk, sv, xp, seed, ext=ext, other=other):
            return GF.GilfreeTransport(ext, host_k, host_v, sk, sv, BS, int(host_k[0].shape[-1]), threads=2,
                                       host_wait=be.host_wait, cap_groups=8, worker="cpp",
                                       xpool=xp.clone() if other else xp)
        XPT = _shared("test_r2_xpool_shared_b2", "test_r2_xpool.py")
        with pytest.raises(ValueError, match=re.escape(msg)):
            XPT.make_pool_engine(TT.make_ops(p3), kind="b2", svc_kw=dict(issue="desc", arrival_lag=2),
                                 tp_factory=make, **BD.CFG["default"])
    # the address rule: one index for both paths
    assert "self.phys_index(tr.layer, tr.b, tr.h, tr.slot)" in inspect.getsource(XPM.XPool.assign_phys)
    assert "self.phys_index(desc[0], desc[1], desc[2], desc[4])" in inspect.getsource(XPM.XPool.lanes_of)
    assert "lanes = self.xpool.lanes_of(desc)" in inspect.getsource(GF.GilfreeTransport._desc_lanes)
    isd = inspect.getsource(GF.GilfreeTransport.issue_desc)
    assert "does not translate xpool lanes" not in isd and "desc = self._desc_lanes(desc)" in isd
    assert "tr.phys if flat else tr.b * self.n_slots + tr.slot" in inspect.getsource(GF.GilfreeTransport._submit)
    cpp = (BD.EFF / "gilfree_pack.cpp").read_text()
    for w in ("vm[q] = flat ? Dp[e] + plant_slot_", ": Db[e] * dslots_ + Ds[e] + plant_slot_;", 'mod.attr("DESC") = 2;',
              'mod.attr("VERSION") = 3;', "if (s0.dim() == 3) {", "desc.size(0) == 7", "desc.size(0) == 6",
              "two transfers into one lane"):
        assert w in cpp, w
    assert "does not translate xpool lanes" not in inspect.getsource(OWN2.OwnResidency.__init__)
    # the harness: --b2-issue desc on the pool parses (cells and flags)
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x", "--backends",
            "b2", "--worker", "cpp", "--b2-issue", "desc", "--arrival-lag", "2"]
    for lay in ("pool", "both"):
        assert G.parse_args(base + ["--layout", lay]).layout == lay
    a = G.parse_args(base + ["--layout", "pool", "--lean-time", "--no-transfer-mirror"])
    assert a.no_transfer_mirror and a.b2_issue == "desc"
    a = G.parse_args(base[:base.index("--b2-issue")] + ["--b2-issue", "object", "--layout", "pool"])
    assert a.worker == "cpp" and a.b2_issue == "object"
    # T21 static lists (unchanged by f3ldx; [f3tall] + applyev's apply_batched_gates from the f3all side)
    op = inspect.getsource(G.one_point)
    assert sorted(set(re.findall(r"gates\.extend\((\w+)\(", op))) == ["apply_batched_gates", "fused_gates",
                                                                       "stack_acted_gates"]
    assert sorted(set(re.findall(r'GB\.gate\("%s_([A-Za-z0-9_]+)"', op))) == ["SKV_ran", "bookkeeping", "completed",
                                                                              "gilfree_team"]
    main = inspect.getsource(G.main)
    for s_ in ("eval_gates(a, meta, res, gates)", "acc_gates(a, meta, res, gates, acc["):
        assert s_ in main, s_
    # the cells
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + ROOT6
    assert ROOT6 == HF.HF_ROOT == F3.F3LD_ROOT
    assert "        evalfast2f3c|evalfast2c3c) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    f3c = F3._cell(text, "    evalfast2f3c) MODES=fast")
    for w in ("BATCH=216", "PX=1512 ; PRV=1464", "VARIANTS=fused/batched", "NOSI=$NOSI_EVAL", "GF3=1",
              "F3LC_BANDS=", '[ "${DRYRUN:-0}" = 1 ] || [ "$F3LC_BANDS" != TBD ]',
              "--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped",
              "--eval-role fast-time --mem-cap-gb 84", '--eval-bands "$F3LC_BANDS" --b2-resolve fused --fused-kernel v3',
              "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array --worker cpp --b2-issue object"):
        assert w in f3c, w
    c3c = F3._cell(text, "    evalfast2c3c) MODES=fast")
    for w in ("STEPS=8 ; DTICKS=4", "BATCH=216", "GF3=1", "--b2-resolve check --fused-kernel v3 --res-lean --res-dev",
              "--inflight-array --worker cpp", "--b2-issue object) ;;"):
        assert w in c3c, w
    for w in ("--bare", "--eval-role", "--accuracy-record", "--eval-bands", "desc"):
        assert w not in c3c, w
    assert "F3LD_BLOCKED=" not in text and "BLOCKED (" not in text
    assert "grep -q 'mod.attr(\"DESC\") = 2;'" in text
    reg = F3._reg(text)
    for tag in ("evalfast2f3c", "evalfast2c3c", "F3LD-BLOCK  CLEARED", "C3C", "F3C-MED", "F3C-MEAN", "F3C-EFF",
                "F3C-ISSUE", "F3C-SAME", "F3C-FALS", "nosi-r2-f3ldx", "a5c8937", "test_r2_f3ldx.py"):
        assert tag in reg, tag
    preds = [ln for ln in reg.splitlines() if "PREDICTION:" in ln]
    if not re.search(r"STAMP \d{4}-\d\d-\d\dT", reg):              # the builder's state: nothing predicted
        assert all(ln.rstrip().endswith("PREDICTION: TBD by lead") for ln in preds), preds
        assert re.search(r"^\s+F3LC_BANDS=TBD\s", text, re.M)
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


@pytest.mark.parametrize("cell", CELLS6)
def test_XD6_the_dryrun_argv_of_the_six_cells(tmp_path, cell, capsys):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = HF._dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(ROOT6).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the f3ldx ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    with capsys.disabled():
        print("XD6 DRYRUN %s: %s" % (cell, got))
    timing = cell in TIMING
    issue = "object" if cell.endswith("c") else "desc"
    for s in ("--backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2",
              "--repeat-docs --log-slots -1 --skip-shipped", "--variants fused/batched", "--mem=240G",
              "--fused-kernel v3", "--res-lean --res-dev", "--worker cpp --b2-issue %s" % issue,
              "PYTHONPATH=%s/nosi" % ROOT6, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell,
              "--b2-resolve %s" % ("fused" if timing else "check")):
        assert s in got or s in line, (cell, s)
    assert "[f3ld] packer copied: %s (packer v3 of --nosi, DESC 2)" % HF._clean_head() in r.stdout
    assert "blocked" not in r.stdout.lower()
    argv = got.split()
    G = HF._sg5()
    if timing and "--eval-bands TBD" in got:
        with pytest.raises(SystemExit):
            G.parse_args(argv)
        argv[argv.index("--eval-bands") + 1] = "1:2:1:2:1:2"
    pa = G.parse_args(argv)
    assert pa.layout == "pool" and pa.b2_issue == issue and pa.worker == "cpp"
    assert pa.no_transfer_mirror == cell.endswith("t") and pa.inflight_array == (not cell.endswith("t"))


@pytest.mark.parametrize("cell", CELLS6)
def test_XD7_the_step_stops_outside_dryrun_on_no_blocker(tmp_path, cell):
    text = HF.STEP.read_text()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "DRYRUN"))}
    env.update(SG5_HOLDERS=str(tmp_path))
    r = subprocess.run(["bash", str(HF.STEP), "--nosi", "0" * 40, "418", "t1", cell], env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 1 and "r2_tick_sg5.py" not in r.stdout, (r.returncode, r.stderr[-300:])
    assert "BLOCKED" not in r.stderr, r.stderr[-300:]
    if cell in TIMING and re.search(r"^\s+F3L[DTC]_BANDS=TBD\s", text, re.M):
        assert "cell %s: --eval-bands TBD" % cell in r.stderr, r.stderr[-300:]
    else:
        assert "SG5_HOLDERS is for DRYRUN" in r.stderr, r.stderr[-300:]
