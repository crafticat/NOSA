"""CPU tests of step S6, the SHARED X POOL layout (nosi/nosi/r2_xpool.py; docs/R2_XPOOL_PLAN.md), opt-in beside the
per-stream layout of decision D4. The rounds are SG5's CPU stand-in (test_r2_tick.py: the tiny model, K3X's seeded
adversarial transport, S-C's CPU device stand-in); the pool rounds land each transfer in its LANE (FlatAdversarial).

  XP1  BITWISE, b2 and sc: the teacher-forced round (forced_schedule) at T15's capacities (C 12, X 24) and at the
       PRESSURE point (C 8, X 40: the LRU goes over C, slots are released and taken again), per-stream vs pool layout
       (X_pool = X): every tick's rows, input tokens, logits (torch.equal), LOGICAL page tables (the service's
       decisions), memory() without slots_physical; at the end the committed tokens, the journal per kind, the xpool
       flag 0. Preconditions: the pressure round evicts (release:evict or verifier evictions > 0) and a LANE CHANGES
       HANDS between two requests (the case the per-stream layout cannot have). Negative control of the comparison: a
       pool engine whose translated page table repeats one page (precondition: the planted table differs) differs.
  XP1r RESIDENT: the flat resident layout (XPool mode "resident", the ResidentStore) equals the per-stream
       ResidentStore: the serial mode's logits torch.equal over N_REF steps, and the teacher-forced tick passes T4's
       checks (every verify row equal to the serial row, the committed sequences the teacher's).
  XP2  BYTES: the pool store's bytes equal store_bytes_formula (B (C + 4) + X_pool positions x 64 x H x (2 D + 1) x
       elem) on the tiny engine and at the PRODUCTION GEOMETRY of the registered cell x1 (L 32, B 64, H 2, D 128, C 96,
       X 64, X_pool 64, bf16; "meta" tensors): 425,279,488 B per layer, 13.61 GB per run vs 22.10 GB per-stream; the
       resident flat store equals the per-stream bytes; the docs' per-request numbers at C 81 / C 96.
  XP3  NEGATIVE CONTROL "alias" (two requests handed ONE lane): L34 precondition from the plant's own record and the
       map (both slots BOUND to the same lane of one head at the refusal: the plant is not a no-op); CAUGHT: the round
       raises the watched flag xpool_err (ERR_ALIAS) at the plant's tick. Also the binder / translator flags on direct
       inputs: an unbound slot in a page table (ERR_UNBOUND), a full arena (ERR_FULL), X_pool < X refused.
  XP4  the equal-memory baseline rules: equal_memory_pool "stream" (C + X + 4 - 64) and "k5" (C + 4 - 64 +
       floor(lanes / (B H))) values; the harness's --layout / --X-pool / --pool-rule defaults and refusals; its
       pool_rule_value equals equal_memory_pool at the x1 arguments (37 and 100).
  XP5  h2d_entries' CPU twin computes the kernel's flat offsets (lesson L28) and equals plain indexing; it refuses a
       non-contiguous entry table; the kernel resolves its names in its module (lesson L36; skipped without triton).
  XP6  the gilfree adapter on the flat store (the pinned packer built as in test_r2_transport_gilfree): the lanes'
       rows are the host truth and equal the per-stream landing of the same transfers; a transfer without a lane and two
       transfers on one lane are refused; a wrong lane lands other bytes (the comparison discriminates).
  XP7  the harness's layout comparison (one_point_both with a stand-in point): equal arms PASS the MUST gate; one
       differing fingerprint FAILS it with the tick and row named; the peak / tick REPORTS; logit_fingerprint sees a
       one-bit change.
  XP8  the step script's cell x1 (DRYRUN on a clean checkout): p5b2's point, EXACT, --layout both --X-pool 64
       --pool-rule both; the harness reads it so; the k5 baseline line prints 37.
  XP9  (lint, F12d's tokens) the pool's per-layer device path (XPool.bind, audit, release_free, translate,
       gather_entries, draft_bias_fill_flat) holds no host read and no host-to-device copy. Negative control: the
       scan finds the read of assign_phys (the one read per tick, at end_tick).
  Review of 44d5e95 (paper-notes/review-xpool-44d5e95.md):
  XP10 (M4) the blocking decision read the lane reuse rests on: pins and a refused non-blocking class.
  XP11 (M2) _h2d_entries_kernel under TRITON_INTERPRET=1 equals its twin; both skip out-of-bounds entries.
  XP12 (M1) gather_entries + the twin land S-C's REAL decisions (F20b / F21b, Policy E) as CpuDeviceSim.gather does.
  XP13 (M3, M4) S-C on the flat layout equals S-C per stream on F20b, with a lane-level reader check; early-unbind
       plants (RELEASING, INFLIGHT) are caught.
  The step cell x1 now runs stream, pool, stream2 with --arrival-lag 2 (B1); XP7 covers the replica gate.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_xpool.py
"""
from __future__ import annotations

import collections
import importlib.util
import os
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


TT = _shared("test_r2_tick_shared_xp", "test_r2_tick.py")              # the tiny model, make_res_engine (r2tick)
TK, RES, RSC, OWN2, KTR = TT.TK, TT.RES, TT.RSC, TT.OWN2, TT.KTR
XPM = TT._load("r2_xpool")                                            # the layout, in the r2tick package
p3, ref = TT.p3, TT.ref                                               # the fixtures, by name
EFF, ROOT = TT.EFF, TT.ROOT
BS = TT.BS
CFG = dict(default=dict(C=12, X=24, R_ver=4), pressure=dict(C=8, X=40, R_ver=4))


# =====================================================================================================================
# the pool engine (make_res_engine's state on the flat layout)
# =====================================================================================================================
class FlatAdversarial(KTR.AdversarialTransport):
    """K3X's adversarial transport (seeded delays, reordering, bursts) landing a transfer in its LANE of the flat
    store: rows phys * 64 .. + 64, column h (Transfer.phys, set by the service at end_tick)."""

    def _land(self, tr):
        if tr.phys is None:
            raise AssertionError("xpool: transfer seq %d landed without a lane" % tr.seq)
        p0 = tr.phys * self.bs
        self.slots_k[tr.layer][p0:p0 + self.bs, tr.h].copy_(tr.payload[0])
        self.slots_v[tr.layer][p0:p0 + self.bs, tr.h].copy_(tr.payload[1])
        tr.delivered = True
        tr.payload = None
        self.delivered += 1


def make_pool_engine(ops, kind="b2", C=12, X=24, R_ver=4, X_pool=None, S_list=TT.S_TINY, steps=TT.STEPS, seed=1,
                     mode="delay", svc_kw=None, xp_mut=None, tp_factory=None, **kw):
    """TT.make_res_engine's world on the SHARED POOL layout: the same twins and host rows (the same seeded prefill),
    one flat store per layer with the four tail pages at their static positions and an empty history arena; the
    service (b2 / sc) and the engine share one XPool; FlatAdversarial with the same seed. [f3ldx] tp_factory(be,
    host_k, host_v, sk, sv, xp, seed): another transport on the same pool (test_r2_f3ldx.py: E's GilfreeTransport)."""
    ref_eng = TT.make_engine(ops, S_list=S_list, steps=steps, seed=seed)
    B, NBr = len(S_list), C + X
    n_host = (ref_eng.NB + 1) * BS
    xp = XPM.XPool(TT.NL_LAYERS, B, TT.HKV, NBr, mode="pool", C=C, X_pool=X if X_pool is None else X_pool, mut=xp_mut)
    host_k = [torch.zeros(B, n_host, TT.HKV, TT.D) for _ in range(TT.NL_LAYERS)]
    host_v = [torch.zeros(B, n_host, TT.HKV, TT.D) for _ in range(TT.NL_LAYERS)]
    sk, sv, sb = [], [], []
    for l in range(TT.NL_LAYERS):
        k, v, bias = xp.new_store(TT.D)
        for b, S in enumerate(S_list):
            full = (S // BS) * BS
            host_k[l][b, :full] = ref_eng.store_k[l][b, :full]
            host_v[l][b, :full] = ref_eng.store_v[l][b, :full]
            for j in range(4):                                          # the four tail pages, at their positions
                e_ref, p0 = (ref_eng.NB + j) * BS, xp.tail_pos(b, j) * BS
                k[p0:p0 + BS] = ref_eng.store_k[l][b, e_ref:e_ref + BS]
                v[p0:p0 + BS] = ref_eng.store_v[l][b, e_ref:e_ref + BS]
                bias[p0:p0 + BS] = ref_eng.store_bias[l][b, e_ref:e_ref + BS]
            if S > full:
                r0 = (ref_eng.NB + (S // BS) % 2) * BS
                host_k[l][b, full:S] = ref_eng.store_k[l][b, r0:r0 + S - full]
                host_v[l][b, full:S] = ref_eng.store_v[l][b, r0:r0 + S - full]
        sk.append(k)
        sv.append(v)
        sb.append(bias)
    eng = TK.TickEngine(ref_eng.model, ops, ref_eng.twins, sk, sv, sb, S_list, ref_eng.M_alloc, nl=TT.NL_CPU,
                        stage1=TT._stage1, pool=TT._pool, check=True, xpool=xp, **kw)
    be = TT.Be()
    if tp_factory is None:
        tp = FlatAdversarial(be, host_k, host_v, sk, sv, BS, seed, mode=mode, max_delay=3)
        tp.xpool = xp                                                 # the transport writes this pool's lanes (m2)
    else:
        tp = tp_factory(be, host_k, host_v, sk, sv, xp, seed)         # [f3ldx]
    gather = RES.TorchGather(host_k, host_v, sk, sv)
    cis = [t.total_cis[:B] for t in eng.twins]
    args = (TT.NL_LAYERS, B, TT.HKV, C, X, R_ver, tp, gather)
    skw = dict(store_b=sb, cis=cis, block_bytes=BS * 2 * TT.D * 4, host_wait=be.host_wait, xpool=xp)
    skw.update(svc_kw or {})
    cls = OWN2.OwnResidency if kind == "b2" else RSC.ResidencySC
    svc = cls(*args, device="cpu", nblk=n_host // BS, **skw)
    eng.attach_backend(TK.ResidencyBackend(eng, svc, host_k, host_v, lambda: TT.Ev(True), 3))
    eng.res_world = SimpleNamespace(svc=svc, tp=tp, be=be, host_k=host_k, host_v=host_v, xp=xp)
    return eng


def _run(eng, toks, forced=True):
    """One round with a per-tick record (after the tick's end_tick, before its decisions): rows, input tokens, logits,
    the LOGICAL page tables of every layer, memory() without slots_physical; with an xpool, the map after each tick."""
    svc = eng.res_world.svc
    xp = getattr(eng.res_world, "xp", None)
    pages = []
    resolve0 = svc.resolve

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)] for i in range(R)]))
        return p
    svc.resolve = resolve
    ticks, maps = [], []

    def on_tick(rows, tokens, out):
        mem = dict(svc.memory())
        mem.pop("slots_physical")
        ticks.append(dict(rows=[(r.b, r.kind, r.pos, r.epoch, r.i, getattr(r, "m", None)) for r in rows],
                          tokens=[t.slot if isinstance(t, TK.Ref) else int(t) for t in tokens],
                          logits=out.logits.clone(), pages=list(pages), memory=mem))
        pages.clear()
        if xp is not None:
            maps.append(xp.pos_of.clone())
    S_list = eng.S_list

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    reject_at, _ = TK.forced_schedule(3) if forced else (None, [])
    rnd, err = None, None
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                           teacher=teacher if forced else None, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                            # noqa: BLE001 - recorded, compared by the test
        err = e
    return SimpleNamespace(eng=eng, svc=svc, ticks=ticks, rnd=rnd, err=err, maps=maps)


def _first_diff(a, b):
    n = min(len(a.ticks), len(b.ticks))
    for t in range(n):
        for f in ("rows", "tokens", "logits", "pages", "memory"):
            x, y = a.ticks[t][f], b.ticks[t][f]
            if not (torch.equal(x, y) if f == "logits" else x == y):
                return (t, f)
    if len(a.ticks) != len(b.ticks) or (a.err is None) != (b.err is None):
        return (n, "raise: %r / %r" % (a.err, b.err))
    return None


def _handoffs(maps, NS):
    """Lanes (l, h, lane) bound to two different requests at two times (a lane changed hands): the count."""
    owner, n = {}, 0
    for m in maps:
        L, B, H, _ = m.shape
        for l in range(L):
            for b in range(B):
                for h in range(H):
                    for p in m[l, b, h, :NS].tolist():
                        if p < 0:
                            continue
                        prev = owner.get((l, h, p))
                        if prev is not None and prev != b:
                            n += 1
                        owner[(l, h, p)] = b
    return n


# =====================================================================================================================
# XP1 / XP1r bitwise
# =====================================================================================================================
@pytest.mark.parametrize("kind", ["b2", "sc"])
@pytest.mark.parametrize("cfg", sorted(CFG))
def test_XP1_pool_layout_equals_the_per_stream_layout_bitwise(p3, ref, cfg, kind):
    toks, _ = ref
    ops = TT.make_ops(p3)
    stream = _run(TT.make_res_engine(ops, kind=kind, **CFG[cfg]), toks)
    pool = _run(make_pool_engine(ops, kind=kind, **CFG[cfg]), toks)
    assert stream.err is None and len(stream.ticks) > 10, ("precondition: the per-stream round ran", stream.err)
    js = dict(stream.svc.journal_counts)
    xp = pool.eng.xpool
    assert xp.binds > 0 and pool.ticks, ("precondition: the pool round bound lanes and recorded ticks", pool.err)
    if cfg == "pressure":                                             # the preconditions of "eviction pressure",
        ev = js.get("release:evict", 0) + js.get("release:verifier_evict", 0) + js.get("release:verifier_evict_inflight", 0)
        assert ev > 0, ("precondition: no eviction in the pressure round", js)          # checked BEFORE the claim
        assert _handoffs(pool.maps, xp.n_hist) > 0, "precondition: no lane changed hands between requests"
    d = _first_diff(stream, pool)
    assert d is None, ("the pool layout differs from the per-stream layout", d, pool.err)
    assert [r.out for r in pool.rnd.reqs] == [r.out for r in stream.rnd.reqs]
    assert [r.out for r in pool.rnd.reqs] == [[int(toks[s, b]) for s in range(TT.STEPS)] for b in range(pool.eng.B)]
    jp = dict(pool.svc.journal_counts)
    assert js == jp, (js, jp)
    assert int(xp.err) == 0 and not pool.svc.uid_b
    assert pool.eng.stats["writebacks"] >= 3 and pool.eng.stats["rollbacks"] > 0


def test_XP1n_the_comparison_sees_a_wrong_page(p3, ref):
    """NEGATIVE CONTROL of XP1's comparison: the pool engine's translated tables repeat one page (row 0, head 0: its
    first entry takes its second's position) at every layer; precondition: the planted table differs from the clean
    one; the comparison then reports a difference."""
    toks, _ = ref
    ops = TT.make_ops(p3)
    stream = _run(TT.make_res_engine(ops, kind="b2", **CFG["default"]), toks)
    eng = make_pool_engine(ops, kind="b2", **CFG["default"])
    xp = eng.xpool
    orig = xp.translate
    changed = [0]

    def planted(l, table, seq_idx):
        out = orig(l, table, seq_idx)
        bad = out.clone()
        if bad.shape[2] > 1 and int(bad[0, 0, 1]) >= 0:
            bad[0, 0, 0] = bad[0, 0, 1]
        changed[0] += int(not torch.equal(bad, out))
        return bad
    xp.translate = planted
    pool = _run(eng, toks)
    assert changed[0] > 0, "precondition: the planted table never differed (a no-op plant, L34)"
    assert _first_diff(stream, pool) is not None


def _flat_resident(ref_eng):
    """The per-stream ResidentStore engine re-laid out flat (XPool mode "resident"): history slot s of request b at
    position b * NB + s, the tail pages at A + 4 b + j."""
    B, NB = ref_eng.B, ref_eng.NB
    xp = XPM.XPool(TT.NL_LAYERS, B, TT.HKV, NB, mode="resident")
    sk, sv, sb = [], [], []
    for l in range(TT.NL_LAYERS):
        k, v, bias = xp.new_store(TT.D)
        for b in range(B):
            h0 = xp.hist_pos_static(b, 0) * BS
            for dst, src in ((k, ref_eng.store_k[l]), (v, ref_eng.store_v[l]), (bias, ref_eng.store_bias[l])):
                dst[h0:h0 + NB * BS] = src[b, :NB * BS]
                for j in range(4):
                    p0 = xp.tail_pos(b, j) * BS
                    dst[p0:p0 + BS] = src[b, (NB + j) * BS:(NB + j + 1) * BS]
        sk.append(k)
        sv.append(v)
        sb.append(bias)
    return xp, sk, sv, sb


def test_XP1r_the_flat_resident_layout_equals_the_resident_store(p3, ref):
    toks, logits = ref
    ops = TT.make_ops(p3)
    base = TT.make_engine(ops)
    xp, sk, sv, sb = _flat_resident(base)
    flat = TK.TickEngine(base.model, ops, base.twins, sk, sv, sb, base.S_list, base.M_alloc, nl=TT.NL_CPU,
                         stage1=TT._stage1, pool=TT._pool, check=True, xpool=xp)
    t_flat, l_flat = TT.serial_ref(flat)
    assert torch.equal(t_flat, toks) and torch.equal(l_flat, logits)          # the serial mode, every step
    assert flat.stats["writebacks"] >= 1 and int(xp.err) == 0
    base2 = TT.make_engine(ops, draft_drop=0.5)
    xp2, sk2, sv2, sb2 = _flat_resident(base2)
    flat2 = TK.TickEngine(base2.model, ops, base2.twins, sk2, sv2, sb2, base2.S_list, base2.M_alloc, nl=TT.NL_CPU,
                          stage1=TT._stage1, pool=TT._pool, check=True, xpool=xp2, draft_drop=0.5)
    res = TT.tick_check(flat2, toks, logits, forced=True)
    assert res["compared"] > 50 and not res["bad"] and not res["ctrl"], res["bad"][:5]
    assert all(res["committed_ok"]) and not res["mirror_bad"] and int(xp2.err) == 0


# =====================================================================================================================
# XP2 bytes
# =====================================================================================================================
def _nbytes(ts):
    return sum(t.numel() * t.element_size() for t in ts)


def test_XP2_the_pool_store_bytes_equal_the_formula():
    C, X, Xp, B, H, D = 8, 40, 40, 4, TT.HKV, TT.D
    xp = XPM.XPool(TT.NL_LAYERS, B, H, C + X, mode="pool", C=C, X_pool=Xp)
    k, v, bias = xp.new_store(D)
    assert xp.M == B * (C + 4) + Xp and xp.A == B * C + Xp
    assert _nbytes([k, v, bias]) == XPM.store_bytes_formula(B, H, D, C, Xp, 4)
    assert _nbytes([k, v, bias]) < XPM.per_stream_bytes_formula(B, H, D, C + X, 4)
    rs = XPM.XPool(TT.NL_LAYERS, B, H, 12, mode="resident")                  # the flat resident store: no saving
    assert _nbytes(rs.new_store(D)) == XPM.per_stream_bytes_formula(B, H, D, 12, 4)
    # the production geometry of the registered cell x1 ("meta" tensors: no memory)
    big = XPM.XPool(32, 64, 2, 96 + 64, mode="pool", C=96, X_pool=64, device="meta")
    per_layer = _nbytes(big.new_store(128, dtype=torch.bfloat16))
    assert per_layer == XPM.store_bytes_formula(64, 2, 128, 96, 64, 2) == 425279488
    assert 32 * per_layer == 13608943616
    assert 32 * XPM.per_stream_bytes_formula(64, 2, 128, 160, 2) == 22097690624     # the audit's 22.10 GB
    assert 32 * XPM.per_stream_bytes_formula(64, 2, 128, 160, 2) - 32 * per_layer == 8488747008
    # the docs' per-request numbers (32 layers, B 64): X_pool positions of H lanes; 407 / 422 = the K5 demand peaks
    per_req = lambda C_, Xp_: 32 * XPM.store_bytes_formula(64, 2, 128, C_, Xp_, 2) // 64          # noqa: E731
    assert (per_req(81, 407), per_req(81, 422)) == (192342912, 192836352)
    assert (per_req(96, 407), per_req(96, 422)) == (223923072, 224416512)
    assert per_req(96, 64) == 212639744 and 32 * XPM.per_stream_bytes_formula(64, 2, 128, 160, 2) // 64 == 345276416
    assert 32 * XPM.per_stream_bytes_formula(64, 2, 128, 81 + 64, 2) // 64 == 313696256          # the audit's 313.70


# =====================================================================================================================
# XP3 the alias negative control and the flags
# =====================================================================================================================
def test_XP3_two_requests_on_one_lane_are_caught(p3, ref):
    toks, _ = ref
    eng = make_pool_engine(TT.make_ops(p3), kind="b2", xp_mut={"alias": True}, **CFG["pressure"])
    xp = eng.xpool
    bad = _run(eng, toks)
    # L34 precondition: the plant happened, on two REQUESTS, and both slots were bound to that lane at once
    assert len(xp.plant_log) == 1, "precondition: the alias plant never fired"
    pl = xp.plant_log[0]
    assert pl["first"][0] != pl["second"][0], pl
    m = xp.pos_of[pl["l"], :, pl["h"], :xp.n_hist]
    owners = {b for b in range(xp.B) for p in m[b].tolist() if p == pl["pos"]}
    assert len(owners) >= 2, ("precondition: the lane is not shared at the refusal", owners)
    # caught: the watched flag refuses the plant's tick (the run stops there)
    assert bad.err is not None and "xpool_err" in str(bad.err), bad.err
    assert int(xp.err) & XPM.ERR_ALIAS
    assert len(bad.ticks) < 20                                        # stopped early (the clean round is longer)


def test_XP3b_the_flags_on_direct_inputs():
    xp = XPM.XPool(1, 2, 2, 3 + 2, mode="pool", C=3, X_pool=2)
    t = torch.tensor([[[0, 5, -1], [1, 6, -1]]], dtype=torch.int32)        # history slots 0 / 1 (unbound), tails 5 / 6
    out = xp.translate(0, t, torch.tensor([1], dtype=torch.int32))
    assert int(xp.err) & XPM.ERR_UNBOUND and int(out[0, 0, 0]) == -1 and int(out[0, 0, 1]) == xp.tail_pos(1, 0)
    xp = XPM.XPool(1, 2, 2, 3 + 2, mode="pool", C=3, X_pool=2)
    n = 3 * 2 + 2                                                     # A lanes per head: B C + X_pool = 8
    bb = torch.tensor([0] * 5 + [1] * 5)
    hh = torch.zeros(10, dtype=torch.int64)
    sl = torch.tensor(list(range(5)) * 2)
    xp.bind(0, bb, hh, sl, torch.ones(10, dtype=torch.bool))          # 10 slots of head 0 > 8 lanes
    assert int(xp.err) & XPM.ERR_FULL and int(xp.occ[0, 0].sum()) == n
    ok = XPM.XPool(1, 2, 2, 5, mode="pool", C=3, X_pool=2)
    ok.bind(0, bb[:8], hh[:8], sl[:8], torch.ones(8, dtype=torch.bool))
    assert int(ok.err) == 0 and sorted(ok.pos_of[0, :, 0, :5].reshape(-1).tolist()) == [-1, -1] + list(range(8))
    assert ok.pos_of[0, :, 0, 5:].tolist() == [[8, 9, 10, 11], [12, 13, 14, 15]]   # the tail rings: A + 4 b + j
    code = torch.full((1, 2, 2, 5), 2, dtype=torch.int64)            # every slot LRU (code 2) ...
    code[0, 0, 0, 1] = XPM.CODE_FREE                                 # ... but slot 1 of (b 0, h 0): FREE, its lane returns
    code[0, 0, 0, 2] = XPM.CODE_RELEASING                            # RELEASING and INFLIGHT keep their lanes
    code[0, 0, 0, 3] = XPM.CODE_INFLIGHT
    lane = int(ok.pos_of[0, 0, 0, 1])
    ok.release_free(code)
    assert int(ok.pos_of[0, 0, 0, 2]) >= 0 and int(ok.pos_of[0, 0, 0, 3]) >= 0
    assert int(ok.pos_of[0, 0, 0, 1]) == -1 and not bool(ok.occ[0, 0, lane])
    ok.bind(0, torch.tensor([1]), torch.tensor([0]), torch.tensor([3]), torch.ones(1, dtype=torch.bool))
    assert int(ok.pos_of[0, 1, 0, 3]) == lane and int(ok.err) == 0   # the lowest free lane, to ANOTHER request
    with pytest.raises(XPM.XPoolError, match="X_pool"):
        XPM.XPool(1, 2, 2, 5, mode="pool", C=3, X_pool=1)


# =====================================================================================================================
# XP4 the equal-memory baseline rules
# =====================================================================================================================
def test_XP4_the_equal_memory_baseline_rules():
    E = XPM.equal_memory_pool
    assert E(96, 64) == 100 and E(96, 32) == 68 and E(64, 16) == 20               # today's rule (default)
    assert E(96, 64, "k5", pool_lanes=2 * 64, B=64, H=2) == 37                     # this layout, x1
    assert E(96, 64, "k5", pool_lanes=64, B=64, H=2) == 36                         # the audit's 64 head-agnostic slots
    assert E(96, 64, "k5", pool_lanes=422, B=64, H=2) == 39                        # the audit's 36 + 3.30, rounded down
    assert E(96, 64, "k5", pool_lanes=2 * 422, B=64, H=2) == 42
    assert E(81, 64, "k5", pool_lanes=2 * 422, B=64, H=2) == 27
    assert E(10, 0, "k5", pool_lanes=0, B=64, H=2) == 0 and E(10, 0) == 0           # max(0, .)
    with pytest.raises(ValueError):
        E(96, 64, "other")
    G = TT._sg5_gate()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
    a = G.parse_args(base)
    assert (a.layout, a.X_pool, a.pool_rule, a.arrival_lag) == ("stream", -1, "stream", 0)   # the defaults: unchanged
    x1 = G.parse_args(base + ["--modes", "exact", "--backends", "b2", "--X", "64", "--layout", "both", "--X-pool", "64",
                              "--pool-rule", "both", "--arrival-lag", "2"])
    assert x1.arrival_lag == 2
    assert G.pool_rule_value(x1, "stream") == 100 and G.pool_rule_value(x1, "k5") == 37
    assert G.pool_rule_value(x1, "k5") == E(96, 64, "k5", pool_lanes=2 * 64, B=64, H=2)
    for Xp in (64, 100, 422):
        y = G.parse_args(base + ["--backends", "b2", "--X", "64", "--layout", "pool", "--X-pool", str(Xp)])
        assert G.pool_rule_value(y, "k5") == E(96, 64, "k5", pool_lanes=2 * Xp, B=64, H=2)
    assert G.pool_positions(G.parse_args(base + ["--backends", "sc", "--layout", "pool"]), 32) == 32
    for bad in (["--layout", "pool"], ["--layout", "both", "--backends", "b2,v1", "--arrival-lag", "2"],
                ["--X-pool", "64"], ["--backends", "b2", "--X", "64", "--layout", "pool", "--X-pool", "32"],
                ["--pool-rule", "x"],
                ["--backends", "b2", "--layout", "both"],                                     # B1: free timing refused
                ["--backends", "b2", "--layout", "both", "--arrival-lag", "2", "--sweep", "96:64"],   # m4
                ["--arrival-lag", "-1"]):
        with pytest.raises(SystemExit):
            G.parse_args(base + bad)
    src = (EFF / "r2_tick_sg5.py").read_text()
    assert "if a.pool_baseline:\n            pool = a.C + a.X + 4 - 64\n" in src         # T19's pin holds


# =====================================================================================================================
# XP5 the gather's twin and kernel
# =====================================================================================================================
def test_XP5_the_entry_gather_twin_models_the_kernel_addresses():
    g = torch.Generator().manual_seed(5)
    B, S, H, D, M = 3, 6 * BS, 2, 8, 7
    host_k = torch.randn(B, S, H, D, generator=g)
    host_v = torch.randn(B, S, H, D, generator=g)
    cis = torch.randn(B, S + 5, H, generator=g)[:, :S]                      # a strided view, as total_cis[:B]
    dk, dv, db = (torch.zeros(M * BS, H, D), torch.zeros(M * BS, H, D), torch.zeros(M * BS, H))
    ent = torch.tensor([[0, 1, 2, 3, 1], [2, 0, 5, 6, 1], [1, 1, 0, 0, 0], [1, 0, 4, 2, 1],
                        [1, 1, 6, 4, 1], [0, 0, 1, 7, 1], [3, 0, 1, 4, 1], [0, 2, 1, 4, 1], [0, 0, -1, 4, 1]]
                       ).t().contiguous()                             # the last five: OUT OF BOUNDS (block 6 of 6,
    n = XPM.h2d_entries(dk, dv, db, host_k, host_v, cis, ent)         # position 7 of 7, b 3, h 2, block -1): skipped
    assert n is None                                                  # the CPU path: the twin
    ek, ev, eb = torch.zeros_like(dk), torch.zeros_like(dv), torch.zeros_like(db)
    for (b, h, blk, pos, ok) in ent.t().tolist()[:4]:
        if ok:
            ek[pos * BS:(pos + 1) * BS, h] = host_k[b, blk * BS:(blk + 1) * BS, h]
            ev[pos * BS:(pos + 1) * BS, h] = host_v[b, blk * BS:(blk + 1) * BS, h]
            eb[pos * BS:(pos + 1) * BS, h] = cis[b, blk * BS:(blk + 1) * BS, h]
    assert torch.equal(dk, ek) and torch.equal(dv, ev) and torch.equal(db, eb)
    assert not torch.equal(db, torch.zeros_like(db))                         # precondition: something landed
    assert torch.equal(dk[3 * BS:4 * BS, 0], torch.zeros(BS, D))             # lane 0 of position 3 untouched (h 1 only)
    with pytest.raises(XPM.XPoolError, match="contiguous"):
        XPM.h2d_entries(dk, dv, db, host_k, host_v, cis, torch.tensor([[0, 1, 2, 3, 1]] * 2).t())   # a transposed view
    with pytest.raises(XPM.XPoolError, match="together"):
        XPM.h2d_entries(dk, dv, db, host_k, host_v, None, ent)
    triton = pytest.importorskip("triton")                            # lesson L36 (F23 covers r2_*.py too)
    TRM = _shared("test_r2_residency_shared_xp", "test_r2_residency.py")
    assert XPM._h2d_entries_kernel is not None and TRM._kernel_name_problems(XPM._h2d_entries_kernel) == []
    assert triton is not None


# =====================================================================================================================
# XP6 the gilfree adapter on the flat store
# =====================================================================================================================
GFT = _shared("test_r2_transport_gilfree_shared_xp", "test_r2_transport_gilfree.py")
pinned = GFT.pinned                                                   # the fixture, by name


def test_XP6_gilfree_lands_each_transfer_in_its_lane(pinned):
    GF, TR = GFT.GF, GFT.TR
    XG = GFT._load("r2_xpool")
    host_k, host_v, gk, gv, _, _ = GFT._world()
    L, B, H, D = GFT.L, GFT.B, GFT.H, GFT.D
    C, X = 4, GFT.NPOS - 4
    xp = XG.XPool(L, B, H, C + X, mode="pool", C=C, X_pool=21)          # A = 2 x 4 + 21 = 29 > the 27 transfers
    assert (xp.A, xp.M) == (29, 37)
    fk = [xp.new_store(D)[0] for _ in range(L)]
    fv = [torch.zeros_like(t) for t in fk]
    gf_bg = GF.GilfreeTransport(pinned.ext, host_k, host_v, fk, fv, BS, D, threads=2, host_wait=lambda ev: None,
                                cap_groups=4, xpool=xp, background=True)    # production's feeder thread (review m3)
    ps = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                             cap_groups=4)
    plans = GFT._plans()
    lanes = {}
    seq = 0
    issued = []
    try:
        for layer, groups in plans:
            a = GFT._transfers(layer, groups, seq)
            bt = GFT._transfers(layer, groups, seq)
            for i, tr in enumerate(a):                                # a lane per (layer, h): distinct, permuted
                tr.phys = (7 * (seq + i) + 3) % xp.A
                lanes[(layer, tr.b, tr.h, tr.slot)] = tr.phys
            seq += len(groups)
            gf_bg.issue(layer, a)
            ps.issue(layer, bt)
            issued += a
        for tr in issued:
            gf_bg.force(tr)
        assert all(gf_bg.done(tr) for tr in issued) and gf_bg.feeder_seen is not None
    finally:
        gf_bg.close()
    gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, fk, fv, BS, D, threads=2, host_wait=lambda ev: None,
                             cap_groups=4, xpool=xp)                  # synchronous: the refusals raise at issue
    assert len(set((l_, h_, p_) for (l_, _, h_, _), p_ in lanes.items())) == len(lanes), "precondition: lanes distinct"
    for (layer, b, h, s), p in lanes.items():
        assert torch.equal(fk[layer][p * BS:(p + 1) * BS, h], gk[layer][b, s * BS:(s + 1) * BS, h])
        assert torch.equal(fv[layer][p * BS:(p + 1) * BS, h], gv[layer][b, s * BS:(s + 1) * BS, h])
    assert GFT._truth_ok(gk, gv, host_k, host_v, plans)
    one = GFT._transfers(0, [(0, 0, 1, 0)], 500)
    with pytest.raises(ValueError, match="lane"):
        gf.issue(0, one)                                              # no lane
    two = GFT._transfers(0, [(0, 0, 1, 0), (1, 0, 2, 1)], 600)
    two[0].phys = two[1].phys = 5
    with pytest.raises(ValueError, match="one destination"):
        gf.issue(0, two)
    tail = GFT._transfers(0, [(0, 0, 1, 0)], 650)                    # a lane in the tail rings [A, M) (review m2)
    tail[0].phys = xp.A
    with pytest.raises(ValueError, match="history arena"):
        gf.issue(0, tail)
    wrong = GFT._transfers(1, [(1, 1, 3, 2)], 700)                   # NEGATIVE CONTROL: a wrong lane lands elsewhere
    wrong[0].phys = 2
    gf.issue(1, wrong)
    assert not torch.equal(fk[1][9 * BS:10 * BS, 1], host_k[1][1, 3 * BS:4 * BS, 1])
    assert torch.equal(fk[1][2 * BS:3 * BS, 1], host_k[1][1, 3 * BS:4 * BS, 1])


# =====================================================================================================================
# XP7 the harness's comparison
# =====================================================================================================================
def test_XP7_the_harness_compares_the_two_arms():
    G = TT._sg5_gate()
    lg = torch.randn(3, 11)
    f = G.logit_fingerprint(lg)
    lg2 = lg.clone()
    lg2.view(torch.int32)[1, 4] ^= 1                                 # one bit of one logit
    f2 = G.logit_fingerprint(lg2)
    assert f.shape == (3,) and f[0] == f2[0] and f[2] == f2[2] and f[1] != f2[1]

    def stand_in(perturb=None, tokens=None):
        """perturb: the arm (label suffix stream / pool / stream2) whose tick 2 row 1 fingerprint is off by one;
        tokens: the arm whose committed tokens differ."""
        calls = []

        def point(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates, label, bare=False,
                  layout=None, fp=None):
            arm = label.rsplit("_", 1)[1]
            calls.append((arm, layout))
            for t in range(4):
                v = torch.arange(3, dtype=torch.int64) * (t + 1)
                if perturb == arm and t == 2:
                    v = v + torch.tensor([0, 1, 0])
                fp.append(([(0, "v", 10 + t), (1, "d", 20 + t), (2, "v", 30 + t)], v))
            return dict(backend=kind, T_tick_ms={"stream": 100.0, "pool": 104.0, "stream2": 102.0}[arm],
                        peak_gb_round=30.0 if layout == "stream" else 21.5, store_gb=22.1 if layout == "stream" else 13.6,
                        committed_tokens=[[1, 2, 4]] if tokens == arm else [[1, 2, 3]])
        point.calls = calls
        return point

    def run(**kw):
        gates = []
        pf = stand_in(**kw)
        r = G.one_point_both(SimpleNamespace(), None, None, None, None, None, "b2", 96, 64, 16, None, 48, None, gates,
                             "SG5_exact_b2", point_fn=pf)
        return r, {x["name"]: x for x in gates}, pf.calls
    r, g, calls = run()
    assert calls == [("stream", "stream"), ("pool", "pool"), ("stream2", "stream")]     # stream, pool, stream2 (m6a)
    for name in ("SG5_exact_b2_XP_replica", "SG5_exact_b2_XP_committed", "SG5_exact_b2_XP_bitwise"):
        assert g[name]["passed"] and g[name]["must"], name
    assert r["xp"]["rows_compared"] == 12 and r["xp"]["peak_drop_gb"] == 8.5
    assert abs(r["xp"]["tick_ratio"] - 104.0 / 101.0) < 1e-9 and abs(r["xp"]["stream_spread"] - 1.02) < 1e-9
    assert "inside" in g["R_SG5_exact_b2_XP_tick"]["detail"] and "drop 8.5" in g["R_SG5_exact_b2_XP_peak"]["detail"]
    r, g, _ = run(perturb="pool")                                     # the pool differs: bitwise FAILS, named
    assert g["SG5_exact_b2_XP_replica"]["passed"] and not g["SG5_exact_b2_XP_bitwise"]["passed"]
    assert "tick 2 row 1" in g["SG5_exact_b2_XP_bitwise"]["detail"], g["SG5_exact_b2_XP_bitwise"]["detail"]
    r, g, _ = run(perturb="stream2")                                  # the replica differs: bitwise is VOID
    assert not g["SG5_exact_b2_XP_replica"]["passed"] and not g["SG5_exact_b2_XP_bitwise"]["passed"]
    assert g["SG5_exact_b2_XP_bitwise"]["detail"].startswith("VOID") and g["SG5_exact_b2_XP_committed"]["passed"]
    r, g, _ = run(tokens="pool")                                      # committed tokens differ: that MUST fails too
    assert not g["SG5_exact_b2_XP_committed"]["passed"] and not g["SG5_exact_b2_XP_bitwise"]["passed"]
    assert G.compare_layouts({"stream": ([], [], [[1]]), "pool": ([], [], [[2]])})[2] == "committed tokens differ"


# =====================================================================================================================
# XP8 the step script's cell x1
# =====================================================================================================================
def test_XP8_the_step_cell_x1(tmp_path):
    script = EFF / "r2_tick_sg5_step.sh"
    text = script.read_text()
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in text.splitlines():
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    assert "x1) MODES=exact ; BACKENDS=b2 ; PX=64 ; SWEEP_RUN= ; SUFFIX=_x1 ; XPOOL=64 ; NOSI=$NOSI_X1 ;;" in text
    assert '[ -z "$XPOOL" ] || git -C "$NOSI" rev-parse -q --verify "$NSHA:nosi/nosi/r2_xpool.py"' in text
    lines = text.splitlines()
    shared = next(ln for ln in lines if ln.startswith("NOSI=")).split()[0][len("NOSI="):]
    x1_root = next(ln for ln in lines if ln.startswith("NOSI_X1=")).split()[0][len("NOSI_X1="):]
    assert shared == "/mnt/beegfs/ojerbi/nosi-r2-sg5-p6" and x1_root == "/mnt/beegfs/ojerbi/nosi-r2-xpool"   # m8
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    untracked = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--", "nosi"],
                               capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        pytest.skip("not a clean git checkout: the DRYRUN needs one")
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    (tmp_path / "417.gpu_uuid").write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    r = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "417", "t1", "x1"], env=env,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        assert Path(x1_root).resolve() != ROOT.resolve(), r.stderr.strip()[-300:]     # in x1's ROOT a stop FAILS
        pytest.skip("DRYRUN stopped outside x1's ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    for w in ("--modes exact --backends b2", "--C 96 --X 64 --R-ver 16", "--pool-baseline", "--steps 48",
              "--layout both --X-pool 64 --pool-rule both --arrival-lag 2",
              "PYTHONPATH=/mnt/beegfs/ojerbi/nosi-r2-xpool/nosi"):
        assert w in got or w in line, w
    assert "--sweep " not in got and "--variants" not in got and "--b2-checked" not in got
    assert "k5 rule NOSI_POOL_BLOCKS = C + 4 - 64 + X_pool / 64 = 37" in r.stdout
    assert "root /mnt/beegfs/ojerbi/nosi-r2-xpool" in r.stdout
    assert "--job-name=sg5-t1-417-x1 " in r.stdout and "cell x1" in r.stdout
    G = TT._sg5_gate()
    pa = G.parse_args(got.split())
    assert (pa.layout, pa.X_pool, pa.pool_rule, pa.backends, pa.modes, pa.arrival_lag) == ("both", 64, "both", ["b2"],
                                                                                            ["exact"], 2)
    assert G.pool_rule_value(pa, "k5") == 37 and G.pool_rule_value(pa, "stream") == 100
    for bad in (("x1", "p6"), ("x2",), ("X1",)):
        rb = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "417", "t1", *bad], env=env,
                            capture_output=True, text=True, timeout=120)
        assert rb.returncode == 2 and "usage" in rb.stderr, bad


# =====================================================================================================================
# XP9 the per-layer device path holds no host transfer
# =====================================================================================================================
def test_XP9_the_pool_per_layer_path_has_no_host_transfer():
    import inspect
    TRM = _shared("test_r2_residency_shared_xp", "test_r2_residency.py")
    toks = TRM.FORBIDDEN + ("_d2h(",) + TRM.H2D
    for fn in (XPM.XPool.bind, XPM.XPool.audit, XPM.XPool.release_free, XPM.XPool.translate,
               XPM.XPool.gather_entries, XPM.draft_bias_fill_flat):
        src = inspect.getsource(fn)
        assert [t for t in toks if t in src] == [], fn.__qualname__
    assert [t for t in toks if t in inspect.getsource(XPM.XPool.assign_phys)] != []   # NEGATIVE CONTROL: the read


# =====================================================================================================================
# the SERVICE-level pool world (test_r2_residency's world and schedules on the flat layout; review M1, M3, M4)
# =====================================================================================================================
TRM = _shared("test_r2_residency_shared_xp", "test_r2_residency.py")   # the r2res package: S-C, schedules, _state
XPR = TRM._load("r2_xpool")                                            # the XPool S-C of that package uses
F20B = dict(seed=5, C=24, X=2, R_ver=2, hb=36, mode="delay",            # Policy E LRU evictions (F20b)
            sched=lambda: TRM._sched_pressure(105, 18, hb=36, window=3, nver=3))
F21B = dict(seed=7, C=16, X=2, R_ver=2, hb=30, mode="force_only",       # Policy E in-flight supersedes (F21b)
            sched=lambda: TRM._sched_split(109, 30, hb=30, d_hi=20, window=2, nver=1))


class FlatAdversarialR(TRM.TR.AdversarialTransport):
    """FlatAdversarial for the r2res package (its own Transfer class)."""
    xpool = None

    def _land(self, tr):
        if tr.phys is None:
            raise AssertionError("xpool: transfer seq %d landed without a lane" % tr.seq)
        p0 = tr.phys * self.bs
        self.slots_k[tr.layer][p0:p0 + self.bs, tr.h].copy_(tr.payload[0])
        self.slots_v[tr.layer][p0:p0 + self.bs, tr.h].copy_(tr.payload[1])
        tr.delivered = True
        tr.payload = None
        self.delivered += 1


def _pool_svc_world(C, X, R_ver, seed, mode="delay", max_delay=4, xp_mut=None, cls=None):
    """test_r2_residency.world's data (the same generator order: host K, host V, cis) on the FLAT layout: S-C (or cls)
    with an XPool (X_pool = X), FlatAdversarialR (seeded alike), the CPU device stand-in."""
    L, B, H, D, BS_, NB = TRM.L, TRM.B, TRM.H, TRM.D, TRM.BS, TRM.NB
    g = torch.Generator().manual_seed(seed)
    host_k = [torch.randn(B, NB * BS_, H, D, generator=g) for _ in range(L)]
    host_v = [torch.randn(B, NB * BS_, H, D, generator=g) for _ in range(L)]
    cis = [torch.randn(B, NB * BS_, H, generator=g) for _ in range(L)]
    xp = XPR.XPool(L, B, H, C + X, mode="pool", C=C, X_pool=X, mut=xp_mut)
    st = [xp.new_store(D) for _ in range(L)]
    sk, sv, sb = [s[0] for s in st], [s[1] for s in st], [s[2] for s in st]
    be = TRM.Be()
    tp = FlatAdversarialR(be, host_k, host_v, sk, sv, BS_, seed, mode=mode, max_delay=max_delay)
    tp.xpool = xp
    gather = TRM.RES.TorchGather(host_k, host_v, sk, sv)
    svc = (cls or TRM.SC.ResidencySC)(L, B, H, C, X, R_ver, tp, gather, store_b=sb, cis=cis, block_bytes=BS_ * 2 * D * 4,
                                      host_wait=be.host_wait, nblk=NB, xpool=xp)
    return SimpleNamespace(svc=svc, tp=tp, be=be, host_k=host_k, host_v=host_v, cis=cis, store_k=sk, store_v=sv,
                           store_b=sb, xp=xp)


def _lane_holds(w, l, b, h, s, blk):
    """Slot s of (b, h) at layer l is bound and its lane holds the host truth of block blk (K, V and bias)."""
    p = int(w.xp.pos_of[l, b, h, s])
    if p < 0:
        return False
    p0, r0, bs = p * TRM.BS, blk * TRM.BS, TRM.BS
    return (torch.equal(w.store_k[l][p0:p0 + bs, h], w.host_k[l][b, r0:r0 + bs, h])
            and torch.equal(w.store_v[l][p0:p0 + bs, h], w.host_v[l][b, r0:r0 + bs, h])
            and torch.equal(w.store_b[l][p0:p0 + bs, h], w.cis[l][b, r0:r0 + bs, h]))


def _pool_content_ok(w, l, rows, sel, plan):
    """test_r2_residency._content_ok on the flat layout: every verify row's listed pages hold the truth of its
    selection; every DRAFT row's historical pages hold the truth of the block their (resident) slot names."""
    NS = w.svc.n_hist
    for i, r in enumerate(rows):
        for h in range(TRM.H):
            if r.kind == "v":
                want = sorted({z for z in sel[h, i].tolist() if 0 <= z < r.hist_bound})
                got = plan.page_table[i, h, :len(want)].tolist()
                if int(plan.num_pages[i, h]) < len(want) or not all(_lane_holds(w, l, r.b, h, s, z)
                                                                     for z, s in zip(want, got)):
                    return False
            else:
                for s in plan.page_table[i, h, :int(plan.num_pages[i, h])].tolist():
                    if 0 <= s < NS and not _lane_holds(w, l, r.b, h, s, w.svc.stream(l, r.b, h).slots[s].block):
                        return False
    return True


def _lane_hooks(w, cnt):
    """TEST INSTRUMENTATION of a pool world (the lane-level order): read_done records each read lane's last reader;
    after every bind, a lane bound anew counts in cnt["rebinds"] if it was read before, and in cnt["hazards"] if that
    last reader is still pending."""
    svc, xp = w.svc, w.xp
    NS = svc.n_hist
    lane_reader = {}
    bind0, rd0 = xp.bind, svc.read_done

    def bind(l, bb, hh, slot, mask):
        before = xp.pos_of[l, :, :, :NS].clone()
        bind0(l, bb, hh, slot, mask)
        new = (before < 0) & (xp.pos_of[l, :, :, :NS] >= 0)
        for (b, h, s) in new.nonzero().tolist():
            key = (l, h, int(xp.pos_of[l, b, h, s]))
            if key in lane_reader:
                cnt["rebinds"] += 1
                if not lane_reader[key].done:
                    cnt["hazards"] += 1

    def read_done(l, plan, ev):
        sq = plan.seq_idx.tolist()
        for i in range(plan.page_table.shape[0]):
            for h in range(TRM.H):
                for s in plan.page_table[i, h, :int(plan.num_pages[i, h])].tolist():
                    if 0 <= s < NS and int(xp.pos_of[l, sq[i], h, s]) >= 0:
                        lane_reader[(l, h, int(xp.pos_of[l, sq[i], h, s]))] = ev
        return rd0(l, plan, ev)
    xp.bind, svc.read_done = bind, read_done


def _svc_run(w, case, pool, lane_check=False):
    """One schedule on one world (test_r2_residency._lockstep_sc's tick, one world): per tick the logical pages, the
    content at the read point, the state and the counters; readers complete two ticks late. lane_check (pool): every
    lane bound by a bind while the LAST READER of its previous occupant is pending counts as a hazard (the lane-level
    order); rebinds counts the binds of a lane that was read before (the check's precondition)."""
    svc = w.svc
    sched = case["sched"]()
    hb = case["hb"]
    log, readers = [], []
    cnt = dict(hazards=0, rebinds=0)
    if pool and lane_check:
        _lane_hooks(w, cnt)
    for b, u in enumerate(("u0", "u1")):
        svc.begin_request(b, u)
    tail = svc.tail0
    err = None
    try:
        for (t, m, ends, layers) in sched:
            svc.poll()
            rec = dict(t=t, pages=[], content=[])
            for l, spec in enumerate(layers):
                rows = [TRM.Row(b, kind, key, 1000 + t, hb, [(hb, tail)]) for (b, kind, key, _) in spec]
                sel = TRM.S(*[s for (_, _, _, s) in spec])
                p = svc.resolve(l, rows, sel)
                rec["pages"].append([[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(TRM.H)]
                                     for i in range(len(rows))])
                rec["content"].append((_pool_content_ok if pool else TRM._content_ok)(w, l, rows, sel, p))
                ev = TRM.Ev()
                readers.append((t, ev))
                svc.read_done(l, p, ev)
            svc.end_tick()
            for (tt, e) in readers:
                if tt <= t - 2:
                    e.done = True
            if ends:
                for u in ("u0", "u1"):
                    svc.decided((u, 1, m))
            rec["state"], rec["counters"] = TRM._state(w), TRM._counters(svc)
            log.append(rec)
    except Exception as e:                                            # noqa: BLE001 - recorded, compared by the test
        err = e
    return SimpleNamespace(log=log, err=err, cnt=cnt, w=w)


def _svc_pair(case, xp_mut=None, lane_check=False):
    common = dict(C=case["C"], X=case["X"], R_ver=case["R_ver"], seed=case["seed"], mode=case["mode"], max_delay=4)
    ws = TRM.world(**common, cls=TRM.SC.ResidencySC, svc_kw=dict(nblk=TRM.NB))
    wp = _pool_svc_world(**common, xp_mut=xp_mut)
    return _svc_run(ws, case, False), _svc_run(wp, case, True, lane_check=lane_check)


def _svc_diff(a, b):
    """The first (tick, field) where the two service runs differ, or a content failure of either; None: equal."""
    for t, (x, y) in enumerate(zip(a.log, b.log)):
        for f in ("pages", "state", "counters"):
            if x[f] != y[f]:
                return (t, f)
        if not all(x["content"]) or not all(y["content"]):
            return (t, "content")
    if len(a.log) != len(b.log) or (a.err is None) != (b.err is None):
        return (min(len(a.log), len(b.log)), "raise: %r / %r" % (a.err, b.err))
    return None


@pytest.mark.parametrize("case", ["f20b", "f21b"])
def test_XP12_gather_entries_lands_a_real_decision_as_the_device_stand_in(case):
    """XP12 (review M1): on S-C's real verify decisions (the F20b / F21b schedules on the flat layout, Policy E
    present), XPool.gather_entries + h2d_entries_twin land each decision into a copy of the store EXACTLY as
    CpuDeviceSim.gather lands it into the store (K, V, bias; every layer call). Preconditions: decisions with Policy E
    evictions occurred. NEGATIVE CONTROL: the entry mask without `evict` (granted only); precondition: it changes the
    entry set at some call; claim: that copy then differs from the store."""
    c = dict(F20B if case == "f20b" else F21B)
    w = _pool_svc_world(c["C"], c["X"], c["R_ver"], c["seed"], mode=c["mode"])
    xp, dev = w.xp, w.svc.dev
    orig = dev.gather
    rec = dict(calls=0, evict=0, mism=0, plant_changed=0, plant_caught=0)

    def gather(l, dec):
        bb, hh = dec.bh
        t = dec.t
        ent = xp.gather_entries(l, bb, hh, t["blk"], t["take_slot"], (t["granted"] | t["evict"]) != 0)
        bad = xp.gather_entries(l, bb, hh, t["blk"], t["take_slot"], t["granted"] != 0)
        good = [x.clone() for x in (w.store_k[l], w.store_v[l], w.store_b[l])]
        plant = [x.clone() for x in good]
        XPR.h2d_entries_twin(*good, w.host_k[l], w.host_v[l], w.cis[l], ent)
        XPR.h2d_entries_twin(*plant, w.host_k[l], w.host_v[l], w.cis[l], bad)
        orig(l, dec)
        real = (w.store_k[l], w.store_v[l], w.store_b[l])
        rec["calls"] += 1
        rec["evict"] += int((t["evict"] != 0).sum())
        rec["mism"] += int(not all(torch.equal(x, y) for x, y in zip(good, real)))
        if not torch.equal(ent[4], bad[4]):
            rec["plant_changed"] += 1
            rec["plant_caught"] += int(not all(torch.equal(x, y) for x, y in zip(plant, real)))
    dev.gather = gather
    run = _svc_run(w, c, True)
    assert run.err is None, run.err
    assert rec["calls"] > 0 and rec["evict"] > 0, ("precondition: Policy E evictions in the decisions", rec)
    assert rec["mism"] == 0, rec
    assert rec["plant_changed"] > 0, ("precondition: the plant changed the entry set (L34)", rec)
    assert rec["plant_caught"] == rec["plant_changed"], rec


def test_XP13_the_pool_service_equals_the_per_stream_service_and_early_unbinds_are_caught():
    """XP13 (review M3, M4): S-C on the flat layout equals S-C per stream on the F20b schedule (pages, state,
    counters every tick; the content at every read point true in both), and NO lane is bound while its previous
    occupant's last reader is pending (readers complete two ticks late). Precondition of that check: lanes were bound
    again after a read (rebinds > 0). NEGATIVE CONTROLS (XPool mut):
      release_releasing  unbinds RELEASING slots too. Precondition (L34): an early-unbound lane was bound again. Claim:
                         the lane check counts hazards.
      release_inflight   unbinds INFLIGHT slots too (a transfer still lands in the lane). Precondition: an
                         early-unbound lane was bound again. Claim: the comparison differs (or the run raises)."""
    s, p = _svc_pair(F20B, lane_check=True)
    assert s.err is None and len(s.log) == 18, ("precondition: the per-stream run completed", s.err)
    assert p.cnt["rebinds"] > 0, ("precondition: lanes were rebound after a read", p.cnt)
    assert _svc_diff(s, p) is None, (_svc_diff(s, p), p.err)
    assert p.cnt["hazards"] == 0 and int(p.w.xp.err) == 0, p.cnt
    # The plants on F20b were no-ops (job 2191239: early_log empty): there a RELEASING slot's reader is done before the
    # next tick's poll frees it. The plants therefore run on two SCRIPTS that force the case (test_r2_residency's F12e
    # shape, one stream, C 2, X 4: the arena's lowest free lanes are exactly the released ones).
    clean = _script_releasing(None)
    assert clean.early == [] and clean.cnt["hazards"] == 0 and clean.lane5 not in clean.old_lanes, clean
    rr = _script_releasing({"release_releasing": True})
    assert rr.states == ["RELEASING", "RELEASING", "FREE"], ("precondition: slots 0, 1 RELEASING, reader pending", rr)
    assert rr.early and any(e["to"] is not None for e in rr.early), ("precondition: no early-unbound lane was bound "
                                                                     "again", rr.early)
    assert rr.cnt["hazards"] > 0, rr.cnt
    ok = _script_inflight(None)
    assert ok.early == [] and ok.content_ok, ok
    ri = _script_inflight({"release_inflight": True})
    assert ri.early and any(e["to"] is not None for e in ri.early), ("precondition: no early-unbound lane was bound "
                                                                     "again", ri.early)
    assert not ri.content_ok, ri                                       # the comparison with the clean script differs


def _script_world(mut):
    w = _pool_svc_world(2, 4, 2, 0, mode="force_only", xp_mut=mut)
    cnt = dict(hazards=0, rebinds=0)
    _lane_hooks(w, cnt)
    return w, cnt


def _script_releasing(mut):
    """F12e's script on the flat layout: u0's draft takes slots 0, 1 (lanes 0, 1); they land; u0's verifier reads them
    (reader ev1, PENDING); u0 ends, so slots 0, 1 are RELEASING; u1's verifier misses block 5 and takes slot 2. Clean:
    slot 2 gets a NEW lane (2). release_releasing: lanes 0, 1 are unbound early and slot 2 takes lane 0 under ev1."""
    w, cnt = _script_world(mut)
    svc, Row, S = w.svc, TRM.Row, TRM.S
    svc.begin_request(0, "u0")
    svc.poll()
    p = svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 300, 10)], S([[0, 1], []]))
    svc.read_done(0, p, TRM.Ev(done=True))
    svc.end_tick()
    TRM.land_all(w)
    old = [int(w.xp.pos_of[0, 0, 0, s]) for s in (0, 1)]
    ev1 = TRM.Ev()
    p = svc.resolve(0, [Row(0, "v", ("u0", 1, 1), 300, 10)], S([[0, 1], []]))
    svc.read_done(0, p, ev1)
    svc.end_tick()
    svc.end_request("u0")
    svc.poll()
    states = [s.state for s in svc.stream(0, 0, 0).slots[:3]]
    svc.begin_request(0, "u1")
    p = svc.resolve(0, [Row(0, "v", ("u1", 1, 1), 300, 10)], S([[5], []]))
    return SimpleNamespace(cnt=dict(cnt), early=list(w.xp.early_log), old_lanes=old, ev1_done=ev1.done,
                           lane5=int(w.xp.pos_of[0, 0, 0, int(p.page_table[0, 0, 0])]),
                           states=["RELEASING" if x == TRM.OWN.RELEASING else "FREE" if x == TRM.OWN.FREE else str(x)
                                   for x in states])


def _script_inflight(mut):
    """u0's draft takes slots 0, 1 (blocks 0, 1; lanes 0, 1); the transfers do NOT land (force_only); next tick u0's
    verifier reads blocks 0, 1 (in flight: waited) and misses block 5. Clean: block 5 takes lane 2 and every listed page
    holds its block. release_inflight: lanes 0, 1 are unbound while their transfers are in flight; block 5 takes lane
    0; the read point's content check fails (an in-flight slot without a lane, or a lane overwritten)."""
    w, cnt = _script_world(mut)
    svc, Row, S = w.svc, TRM.Row, TRM.S
    svc.begin_request(0, "u0")
    svc.poll()
    p = svc.resolve(0, [Row(0, "d", ("u0", 1, 1), 300, 10)], S([[0, 1], []]))
    svc.read_done(0, p, TRM.Ev(done=True))
    svc.end_tick()                                                    # issued with lanes 0, 1; not landed
    svc.poll()
    rows = [Row(0, "v", ("u0", 1, 1), 300, 10)]
    sel = S([[0, 1, 5], []])
    err, content = None, False
    try:
        p = svc.resolve(0, rows, sel)
        content = _pool_content_ok(w, 0, rows, sel, p)
    except Exception as e:                                            # noqa: BLE001 - a refusal counts as caught
        err = e
    return SimpleNamespace(cnt=dict(cnt), early=list(w.xp.early_log), content_ok=content and err is None, err=err)


def test_XP10_the_pool_needs_a_blocking_decision_read():
    """XP10 (review M4): the lane-level order rests on S-C's BLOCKING decision read. Pins: ResidencySB._d2h reads with
    t.to("cpu") and no non_blocking; DECISION_READ_BLOCKING is True; r2_xpool's state codes are S-C's. NEGATIVE
    CONTROL: a service class that declares its read non-blocking is refused at its first pool resolve."""
    import inspect
    src = inspect.getsource(TRM.SB.ResidencySB._d2h)
    assert 'return t.to("cpu")' in src and "non_blocking" not in src
    assert TRM.SB.ResidencySB.DECISION_READ_BLOCKING is True and TRM.SC.ResidencySC.DECISION_READ_BLOCKING is True
    assert (XPR.CODE_FREE, XPR.CODE_INFLIGHT, XPR.CODE_RELEASING) == (TRM.SC.C_FREE, TRM.SC.C_INFLIGHT,
                                                                      TRM.SC.C_RELEASING)

    class AsyncRead(TRM.SC.ResidencySC):
        DECISION_READ_BLOCKING = False
    w = _pool_svc_world(4, 4, 2, 0, cls=AsyncRead)
    w.svc.begin_request(0, "u0")
    w.svc.poll()
    with pytest.raises(TRM.ER.OwnershipViolation, match="blocking"):
        w.svc.resolve(0, [TRM.Row(0, "d", ("u0", 1, 1), 300, 10)], TRM.S([[0, 1], []]))
    ok = _pool_svc_world(4, 4, 2, 0)                                  # the same world with S-C: no refusal
    ok.svc.begin_request(0, "u0")
    ok.svc.poll()
    ok.svc.resolve(0, [TRM.Row(0, "d", ("u0", 1, 1), 300, 10)], TRM.S([[0, 1], []]))


INTERP_CHILD = r'''
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("r2_xpool_interp", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m._h2d_entries_kernel is not None
g = torch.Generator().manual_seed(11)
B, S, H, D, M, BS = 3, 6 * 64, 2, 16, 7, 64
host_k = torch.randn(B, S, H, D, generator=g)
host_v = torch.randn(B, S, H, D, generator=g)
cis = torch.randn(B, S + 5, H, generator=g)[:, :S]
ent = torch.tensor([[0, 1, 2, 3, 1], [2, 0, 5, 6, 1], [1, 1, 0, 0, 0], [1, 0, 4, 2, 1],
                    [1, 1, 6, 4, 1], [0, 0, 1, 7, 1], [3, 0, 1, 4, 1], [0, 2, 1, 4, 1], [0, 0, -1, 4, 1]]
                   ).t().contiguous()
fails = []
for bdt in (torch.float32, torch.float16):
    a = [torch.zeros(M * BS, H, D), torch.zeros(M * BS, H, D), torch.zeros(M * BS, H, dtype=bdt)]
    t = [x.clone() for x in a]
    m.h2d_entries(*a, host_k, host_v, cis, ent, kernel=True)
    m.h2d_entries_twin(*t, host_k, host_v, cis, ent)
    if not all(torch.equal(x, y) for x, y in zip(a, t)):
        fails.append(("kernel != twin", bdt))
    if torch.equal(a[2], torch.zeros_like(a[2])) or not torch.equal(a[0][4 * 64:5 * 64, 1], torch.zeros(64, D)):
        fails.append(("precondition: something landed, the out-of-bounds entry did not", bdt))
a = [torch.zeros(M * BS, H, D), torch.zeros(M * BS, H, D)]
t = [x.clone() for x in a]
m.h2d_entries(a[0], a[1], None, host_k, host_v, None, ent, kernel=True)
m.h2d_entries_twin(t[0], t[1], None, host_k, host_v, None, ent)
if not all(torch.equal(x, y) for x, y in zip(a, t)):
    fails.append(("kernel != twin without bias",))
# NEGATIVE CONTROL: b and h exchanged in the table (a wrong row offset); precondition: the valid set changes bytes
swapped = ent[[1, 0, 2, 3, 4]].contiguous()
p = [torch.zeros(M * BS, H, D), torch.zeros(M * BS, H, D), torch.zeros(M * BS, H)]
q = [x.clone() for x in p]
m.h2d_entries(*p, host_k, host_v, cis, swapped, kernel=True)
m.h2d_entries_twin(*q, host_k, host_v, cis, ent)
if all(torch.equal(x, y) for x, y in zip(p, q)):
    fails.append(("negative control not caught: the swapped table lands the same bytes",))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("XPOOL-INTERP OK")
'''


def test_XP11_the_entry_kernel_under_the_triton_interpreter():
    """XP11 (review M2): _h2d_entries_kernel itself under TRITON_INTERPRET=1 (a fresh process, CPU tensors, piece 3's
    pattern) equals h2d_entries_twin bitwise, with an fp32 and an fp16 bias (the explicit cast) and without a bias; the
    kernel and the twin skip the SAME out-of-bounds entries (entry_in_bounds). NEGATIVE CONTROL: a table with b and h
    exchanged lands other bytes."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", INTERP_CHILD, str(TT.NOSI / "r2_xpool.py")], env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["XPOOL-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
