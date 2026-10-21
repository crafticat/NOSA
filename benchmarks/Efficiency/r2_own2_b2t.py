"""R2 step 7, (B2) B2-T: the COMPONENT MICROBENCH of the array-backed ownership state (docs/R2_OWN2_B2_DESIGN.md rev 2,
section 5). REPORT only. CPU host tensors; no model, no GPU; the records have the shapes of C's note 12l.

    python benchmarks/Efficiency/r2_own2_b2t.py --out <json> [--threads 1] [--iters 20] [--warmup 3] [--smoke]

THE WORLD (synthetic, seeded) at p3's geometry (L 32, B 64, H 2, C 96, X 32, R_ver 16, NBLK 256, k 64) and p3's
counts per round tick (R2_SG5_CAPTURE_PLAN.md:264-271: 1,054 urgent of which 467 Policy E, 64 verify merges, 6,493
draft takes, 1,269 refusals) plus two ASSUMED counts that p3 did not record (--d-infl 3000 draft in-flight entries,
--acquire 1024 PO / VRES acquisitions):
  - rows: 51 verify rows from 13 cohorts (one b and key per cohort, up to --cohort-rows rows, core.py:146-150) and 55
    draft rows (one per request, core.py:157-164);
  - streams: a NORMAL stream holds C - 13 LRU entries, 2 PO, 1 VRES, 4 in flight (two due at this poll: one draft
    demand, one urgent verify landed in place; two issued at t - 1), 2 RELEASING (one reader done, one pending);
    per layer X / 2 BORROWING streams hold C LRU + 2 PO (the layer's extras all borrowed: K5 sends their urgent misses
    to Policy E and refuses their draft misses);
  - pins: odd-recency LRU entries, PO, VRES and the urgent block by key slot 0; one PO per normal stream only by key
    slot KEYS - 1 (released at this poll at half the (l, b));
  - readers: event ids t x L + l; ticks <= t - 2 complete, t - 1 pending.
The records are generated against the POLLED state (the bench's own poll), so the apply sees what a device would
decide.

EACH ITERATION restores the world (untimed) and times every component in apply_tick's order:
  poll: refresh, force_seq, arrivals (+ K1 over C), post-poll I1 / I2, key drops, releases;
  apply: parse, row checks, slot_of, pin words, K5 events, K5 / first_free, take checks, READY, merges, Policy E,
  gen snapshot, verify pins, draft pins, takes, merges, recency rounds (+ evict_to_c), post checks, readers, metrics,
  E's issue_desc arguments.
JSON: per component median / p10 / p90 / mean ms; the poll / apply / total sums; the generated counts; the predicate
violations (a coherence diagnostic of the synthetic world: 0 expected); the REGISTERED FALSIFIER (rev 2 section 5): the
median total > 25 ms with one thread REFUTES the host estimate. Last stdout line:
  R2_OWN2_B2T_DONE failed=<n> verdict=<SUPPORTED|REFUTED> total_ms=<x>
exit = n (a generated count that differs from the requested one, or a predicate that fired, each counts 1).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import types
from typing import Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NOSI = os.path.join(ROOT, "nosi", "nosi")
PKG = "b2tnosi"
FALSIFIER_MS = 25.0


def _load(mod: str):
    """nosi/nosi/<mod>.py under a stand-in package (the package __init__ imports the models; not needed here)."""
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [NOSI]
        sys.modules[PKG] = pkg
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, os.path.join(NOSI, mod + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


OPS = _load("r2_own2_ops")
SP = sys.modules[PKG + ".r2_residency_sparse"]
C_FREE, C_INFLIGHT, C_LRU, C_PO, C_VRES, C_RELEASING = (OPS.C_FREE, OPS.C_INFLIGHT, OPS.C_LRU, OPS.C_PO, OPS.C_VRES,
                                                        OPS.C_RELEASING)
BIG = OPS.BIG


def wh(c, x, y):
    """torch.where with plain ints as 0-dim int64 tensors (no reliance on the torch version's scalar support)."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32 if isinstance(x, float) else torch.int64)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32 if isinstance(y, float) else torch.int64)
    return torch.where(c, x, y)


# ===================================================================================================================
# arguments
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="a small world (CPU tests): L 2, B 8, C 24, X 8, k 8")
    for n, v in (("L", 32), ("B", 64), ("H", 2), ("C", 96), ("X", 32), ("R-ver", 16), ("nblk", 256), ("keys", 4),
                 ("k", 64), ("verify-rows", 51), ("cohort-rows", 4), ("draft-rows", 55), ("lru-gap", 13),
                 ("urgent", 1054), ("policy-e", 467), ("v-merge", 64), ("d-take", 6493), ("refused", 1269),
                 ("d-infl", 3000), ("acquire", 1024), ("tick", 100), ("lag", 2)):
        ap.add_argument("--" + n, type=int, default=v)
    a = ap.parse_args(argv)
    if a.smoke:
        for n, v in (("L", 2), ("B", 8), ("C", 24), ("X", 8), ("R_ver", 2), ("nblk", 64), ("k", 8),
                     ("verify_rows", 5), ("cohort_rows", 2), ("draft_rows", 6), ("urgent", 6), ("policy_e", 3),
                     ("v_merge", 2), ("d_take", 8), ("refused", 4), ("d_infl", 3), ("acquire", 4)):
            setattr(a, n, v)
    return a


def geo_of(a) -> "OPS.Geo":
    return OPS.Geo(L=a.L, B=a.B, H=a.H, C=a.C, X=a.X, R_ver=a.R_ver, NBLK=a.nblk, KEYS=a.keys)


# ===================================================================================================================
# the world
class Ev:
    def __init__(self, done: bool):
        self.done = done


def ev_done(ev) -> bool:
    return True if ev is None else bool(ev.done)


def distribute(total: int, eligible: torch.Tensor, cap: torch.Tensor, g: torch.Generator, what: str) -> torch.Tensor:
    """Integer counts (one per stream) that sum to `total` over the eligible streams, each <= cap (round robin in a
    seeded random order). Raises when the world cannot hold them."""
    cap = wh(eligible, cap.clamp(min=0), torch.zeros_like(cap))
    if int(cap.sum()) < total:
        raise ValueError("world too small for %s: %d requested, room for %d" % (what, total, int(cap.sum())))
    out = torch.zeros_like(cap)
    left = int(total)
    while left > 0:
        idx = (out < cap).nonzero().squeeze(1)
        idx = idx[torch.randperm(idx.numel(), generator=g)][:left]
        out[idx] += 1
        left -= int(idx.numel())
    return out


def build_world(a) -> Dict:
    """The pre-poll world (module docstring): the Table, the reader window's events, the rows, the stream roles."""
    G = geo_of(a)
    g = torch.Generator().manual_seed(a.seed)
    L, B, H, C, X, NBLK, KEYS = G.L, G.B, G.H, G.C, G.X, G.NBLK, G.KEYS
    NPOS, S, t = G.NPOS, G.S, a.tick
    T = OPS.Table(G)
    T.owner.copy_(torch.arange(B, dtype=torch.int32) + 1000)
    # rows: cohorts of verify rows, one draft row per drafting request
    perm = torch.randperm(B, generator=g)
    n_coh = (a.verify_rows + a.cohort_rows - 1) // a.cohort_rows
    vreq, dreq = perm[:n_coh], perm[B - a.draft_rows:]
    sizes = [a.cohort_rows] * n_coh
    sizes[-1] = a.verify_rows - a.cohort_rows * (n_coh - 1)
    vb = torch.repeat_interleave(vreq, torch.tensor(sizes))
    rows = OPS.Rows(vb=vb, vk=torch.full_like(vb, 1), db=dreq.clone(), dk=torch.full_like(dreq, 2), k=a.k)
    s_all = torch.arange(S)
    b_of = torch.div(s_all, H, rounding_mode="floor").remainder(B)
    l_of = torch.div(s_all, B * H, rounding_mode="floor")
    inV, inD = torch.isin(b_of, vreq), torch.isin(b_of, dreq)
    # borrowing streams: per layer X / 2 (each borrows 2 extras): half among the verify streams, the rest among the
    # draft-only streams
    bor = torch.zeros(S, dtype=torch.bool)
    for l in range(L):
        vs = ((l_of == l) & inV).nonzero().squeeze(1)
        ds = ((l_of == l) & inD & ~inV).nonzero().squeeze(1)
        nbv = min(X // 4, vs.numel())
        nbd = min(X // 2 - nbv, ds.numel())
        bor[vs[torch.randperm(vs.numel(), generator=g)[:nbv]]] = True
        bor[ds[torch.randperm(ds.numel(), generator=g)[:nbd]]] = True
        if nbv + nbd != X // 2:
            raise ValueError("layer %d: %d borrowing streams, want %d" % (l, nbv + nbd, X // 2))
    # categories per stream in slot_perm order: LRU, PO, VRES, INFLIGHT, RELEASING, FREE
    n_lru = wh(bor, torch.full((S,), C), torch.full((S,), C - a.lru_gap))
    n_cat = torch.stack([n_lru, torch.full((S,), 2), wh(bor, 0, 1), wh(bor, 0, 4),
                         wh(bor, 0, 2)], 1)
    if bool((n_lru < a.k).any()) or C - a.lru_gap < 0:
        raise ValueError("lru-gap too large for k")
    cum = torch.cumsum(n_cat, 1)
    jj = torch.arange(NPOS).view(1, -1)
    cat = (jj >= cum[:, 0:1]).long() + (jj >= cum[:, 1:2]).long() + (jj >= cum[:, 2:3]).long() \
        + (jj >= cum[:, 3:4]).long() + (jj >= cum[:, 4:5]).long()
    start = torch.cat([torch.zeros(S, 1, dtype=torch.long), cum], 1)
    pin_cat = jj - start.gather(1, cat)                                  # the position inside its category
    codes = torch.tensor([C_LRU, C_PO, C_VRES, C_INFLIGHT, C_RELEASING, C_FREE])
    slot_perm = torch.argsort(torch.rand(S, NPOS, generator=g), 1)
    blk_perm = torch.argsort(torch.rand(S, NBLK, generator=g), 1)
    used = cat < 5
    blk_pos = wh(used, blk_perm[:, :NPOS], torch.full_like(jj.expand(S, -1), -1))
    ids = lambda tt: tt * L + l_of.view(-1, 1)                          # noqa: E731  reader id of tick tt, layer l
    own = T.owner[b_of].view(-1, 1).to(torch.int64)
    old_owner = (torch.rand(S, 1, generator=g) < 0.05) & ~bor.view(-1, 1)
    is_lru, is_po, is_vres, is_if, is_rel = (cat == 0), (cat == 1), (cat == 2), (cat == 3), (cat == 4)
    col = {}
    col["code"] = codes[cat]
    col["block"] = blk_pos
    col["gen"] = torch.randint(1, 50, (S, NPOS), generator=g)
    col["uid"] = wh(used, own, torch.full_like(own, -1))
    col["uid"] = wh(is_if & (pin_cat == 0) & old_owner, own + 5000, col["uid"])
    col["cls"] = wh(used, wh(is_vres, OPS.CLS_V, OPS.CLS_AD), 0)
    col["stamp"] = wh(is_lru, 1000 + pin_cat, wh(is_if, 100000 + pin_cat, 0))
    rd = wh(is_rel, wh(pin_cat == 0, ids(t - 3), ids(t - 1)),
                     wh(is_if, ids(t - 5), wh(pin_cat % 8 == 0, ids(t - 1), ids(t - 2 - pin_cat % 8))))
    col["rd_last"] = wh(used, rd, ids(t - 6))
    col["rd_first"] = wh(used, col["rd_last"], torch.full_like(rd, -1))
    itk = wh(pin_cat <= 1, wh(pin_cat == 1, t - 1, t - 2), t - 1)
    col["issue_tick"] = wh(is_if, itk, -1)
    col["seq"] = wh(is_if, itk, -1)
    col["demand"] = wh(is_if, wh(pin_cat == 1, OPS.DEM_V, OPS.DEM_D), 0)
    col["issuer"] = wh(is_if, wh(pin_cat == 1, OPS.ISS_V, OPS.ISS_D), 0)
    col["landed"] = is_if & (pin_cat == 1)                              # the urgent gather landed in place at t - 1
    col["req_key"] = wh(is_if, wh(pin_cat == 1, 1, 2), -1)
    for n, v in col.items():
        dst = T.c[n].view(S, NPOS)
        dst.scatter_(1, slot_perm, v.to(dst.dtype))
    # pins (bit planes): key slot 0 verify for odd-recency LRU, PO (borrowing: both), VRES, the urgent block; the
    # normal streams' second PO only by key slot KEYS - 1 (draft plane)
    planes = torch.zeros((2 * KEYS, S, NBLK + 1), dtype=torch.bool)
    p0 = (is_lru & (pin_cat % 2 == 1)) | (is_po & ((pin_cat == 0) | bor.view(-1, 1))) | is_vres | (is_if & (pin_cat == 1))
    pk = is_po & (pin_cat == 1) & ~bor.view(-1, 1)
    bp = blk_pos.clamp(min=0)
    planes[0].scatter_(1, wh(p0, bp, NBLK), True)
    planes[2 * KEYS - 1].scatter_(1, wh(pk, bp, NBLK), True)
    T.pin_w.view(2 * KEYS, S, G.WB).copy_(SP._pack_bits(planes[..., :NBLK]))
    # keys: every key's last reader at t - 2; key slot KEYS - 1 queued for release at half the (l, b)
    lb = torch.arange(L).view(-1, 1) + torch.arange(B).view(1, -1)
    T.key_rd.copy_((t - 2) * L + torch.arange(L).view(-1, 1, 1).expand(L, B, KEYS))
    T.rel_q[:, :, KEYS - 1] = lb % 2 == 0
    T.hist_from.zero_()                                                    # T4 active: every block needs a write-back
    T.hist_id.fill_((t - 3) * L)
    T.clock = 1 << 20
    return dict(G=G, T=T, rows=rows, inV=inV, inD=inD, bor=bor, b_of=b_of, l_of=l_of, vreq=vreq, dreq=dreq)


def fresh_window(a, G) -> "OPS.ReaderWindow":
    """Reader events of ticks 0 .. t - 1 (one per layer, id = tick x L + l): t - 1 pending, the rest complete."""
    w = OPS.ReaderWindow(cap=1 << 14)
    for tt in range(a.tick):
        for l in range(G.L):
            w.register(Ev(done=tt <= a.tick - 2), tt * G.L + l)
    w.refresh(ev_done)                       # the steady state: only the pending tick t - 1 is live (untimed)
    return w


def run_poll(T, win, a, tm=None) -> Dict:
    def span(name, t0):
        if tm is not None:
            tm.setdefault(name, []).append(time.perf_counter_ns() - t0)
    out = {}
    t0 = time.perf_counter_ns()
    win.refresh(ev_done)
    span("poll.refresh", t0)
    t0 = time.perf_counter_ns()
    fs = OPS.force_seq(T, a.tick, a.tick - 2, a.lag)
    span("poll.force_seq", t0)
    out["force_seq"] = fs
    t0 = time.perf_counter_ns()
    out.update(OPS.poll_arrivals(T, a.tick, a.tick - 2, a.lag))
    span("poll.arrivals", t0)
    t0 = time.perf_counter_ns()
    pc = OPS.post_checks(T, None, 0)
    span("poll.checks", t0)
    out["poll_I1"], out["poll_I2"] = pc["I1"], pc["I2"]
    t0 = time.perf_counter_ns()
    out.update(OPS.drop_due_keys(T, win))
    span("poll.key_drops", t0)
    t0 = time.perf_counter_ns()
    out["freed"] = OPS.poll_releases(T, win)
    span("poll.releases", t0)
    return out


# ===================================================================================================================
# the records (C's note-12l shapes) against the polled state
def gen_records(W: Dict, T1, a) -> Dict:
    G, rows = W["G"], W["rows"]
    g = torch.Generator().manual_seed(a.seed + 1)
    L, B, H, C, NBLK, NPOS, S, k = G.L, G.B, G.H, G.C, G.NBLK, G.NPOS, G.S, a.k
    inV, inD, bor = W["inV"], W["inD"], W["bor"]
    code = T1.c["code"].view(S, NPOS)
    stamp = T1.c["stamp"].view(S, NPOS)
    block = T1.c["block"].view(S, NPOS).to(torch.int64)
    uid = T1.c["uid"].view(S, NPOS)
    pos = torch.arange(NPOS).view(1, -1).expand(S, -1)
    big = torch.full((S, NPOS), BIG, dtype=torch.int64)
    lru_sorted = torch.argsort(wh(code == C_LRU, stamp, big), 1)
    n_lru = (code == C_LRU).sum(1)
    free_sorted = torch.argsort(wh(code == C_FREE, pos, big), 1)
    n_free = (code == C_FREE).sum(1)
    pvm = (code == C_PO) | (code == C_VRES)
    pv_sorted = torch.argsort(wh(pvm, pos, big), 1)
    n_pv = pvm.sum(1)
    own = T1.owner[W["b_of"]].view(-1, 1)
    ifl = (code == C_INFLIGHT) & (uid == own) & (torch.bitwise_and(T1.c["demand"].view(S, NPOS), OPS.DEM_D) != 0) \
        & ~T1.c["landed"].view(S, NPOS)
    ifl_sorted = torch.argsort(wh(ifl, pos, big), 1)
    n_if = ifl.sum(1)
    occ = torch.zeros((S, NBLK + 1), dtype=torch.bool)
    occ.scatter_(1, wh(code != C_FREE, block.clamp(min=0), torch.full_like(block, NBLK)), True)
    occ = occ[:, :NBLK]
    nr_sorted = torch.argsort(wh(~occ, torch.rand(S, NBLK, generator=g), torch.full((S, NBLK), 2.0)), 1)
    n_nr = (~occ).sum(1)
    room = (C - (NPOS - n_free)).clamp(min=0)
    z = torch.zeros(S, dtype=torch.int64)
    n_pe = distribute(a.policy_e, inV & bor, torch.full((S,), 4), g, "policy-e")
    n_vf = distribute(a.urgent - a.policy_e, inV & ~bor, torch.minimum(room, torch.full((S,), 2)), g, "urgent")
    n_vm = distribute(a.v_merge, inV & ~bor, n_if, g, "v-merge")
    n_dt = distribute(a.d_take, inD & ~bor, torch.minimum(room - n_vf, n_free - n_vf), g, "d-take")
    n_ref = distribute(a.refused, inD & bor, torch.full((S,), max(1, k // 4)), g, "refused")
    n_di = distribute(a.d_infl, inD & ~bor, n_if, g, "d-infl")
    poolA = inD & bor & (n_pe == 0)
    a1 = distribute(min(a.acquire, int(poolA.sum())), poolA, torch.ones(S, dtype=torch.int64), g, "acquire")
    n_acq = a1 + distribute(a.acquire - int(a1.sum()), inD & ~bor, n_pv, g, "acquire")
    n_vmiss = n_pe + n_vf
    n_vh = wh(inV, k - n_vmiss - n_vm, z)
    n_dmiss = n_dt + n_ref
    n_dh = wh(inD, k - n_dmiss - n_di - n_acq, z)
    for nm, nh in (("verify", n_vh), ("draft", n_dh)):
        if bool((nh < 0).any()) or bool((nh > n_lru).any()):
            raise ValueError("%s hits out of range" % nm)
    if bool((n_vmiss + n_dmiss > n_nr).any()):
        raise ValueError("not enough non-resident blocks")

    def entries(ss, segs):
        """(n, k) entries of streams ss: segs = [(count (S,), sorted list (S, N), offset (S,), is_slot)] in order;
        returns blk, slot (-1 for a block entry), cat."""
        e = torch.arange(k).view(1, -1)
        blk = torch.full((ss.numel(), k), -1, dtype=torch.int64)
        slot = torch.full_like(blk, -1)
        catv = torch.full_like(blk, -1)
        lo = torch.zeros(ss.numel(), 1, dtype=torch.int64)
        for ci, (cnt, srt, off, is_slot) in enumerate(segs):
            c = cnt[ss].view(-1, 1)
            m = (e >= lo) & (e < lo + c)
            ix = (off[ss].view(-1, 1) + e - lo).clamp(min=0, max=srt.shape[1] - 1)
            val = srt[ss].gather(1, ix)
            if is_slot:
                bl = block[ss].gather(1, val)
                slot = wh(m, val, slot)
                blk = wh(m, bl, blk)
            else:
                blk = wh(m, val, blk)
            catv = wh(m, torch.full_like(catv, ci), catv)
            lo = lo + c
        if bool((catv < 0).any()):
            raise ValueError("entries do not fill k")
        o = torch.argsort(blk, 1)
        return blk.gather(1, o), slot.gather(1, o), catv.gather(1, o)

    # verify streams: misses (PE or FREE grant), in-flight merges, hits (the most recent LRU entries)
    sv = inV.nonzero().squeeze(1)
    vblk, vslot, vcat = entries(sv, [(n_vmiss, nr_sorted, z, False), (n_vm, ifl_sorted, z, True),
                                     (n_vh, lru_sorted, n_lru - n_vh, True)])
    # draft streams: misses (take or refusal) after the verify misses, in flight, acquisitions, hits
    sd = inD.nonzero().squeeze(1)
    dblk, dslot, dcat = entries(sd, [(n_dmiss, nr_sorted, n_vmiss, False), (n_di, ifl_sorted, z, True),
                                     (n_acq, pv_sorted, z, True), (n_dh, lru_sorted, n_lru - n_dh, True)])
    # the layer selection (union of the stream's rows) and Policy E's candidates
    selb = torch.zeros((S, NBLK + 1), dtype=torch.bool)
    selb[sv.view(-1, 1).expand_as(vblk), vblk] = True
    selb[sd.view(-1, 1).expand_as(dblk), dblk] = True
    sel_w = SP._pack_bits(selb[:, :NBLK])                                 # (S, WB)
    anyw = OPS.any_pin_words(T1)
    bc = block.clamp(min=0)
    pinned = OPS.bit_test(anyw, torch.arange(S).view(-1, 1).expand(S, NPOS), bc)
    selected = OPS.bit_test(sel_w, torch.arange(S).view(-1, 1).expand(S, NPOS), bc)
    cand = torch.argsort(wh((code == C_LRU) & ~pinned & ~selected, stamp, big), 1)
    n_cand = ((code == C_LRU) & ~pinned & ~selected).sum(1)
    if bool((n_cand < n_pe).any()):
        raise ValueError("not enough Policy E candidates")
    # take slots: per verify stream the q-th miss (block order) -> FREE slot q, or Policy E candidate q
    vmiss = vcat == 0
    vq = torch.cumsum(vmiss.long(), 1) - 1
    vbor = bor[sv].view(-1, 1)
    vtake = wh(vbor, cand[sv].gather(1, vq.clamp(min=0)), free_sorted[sv].gather(1, vq.clamp(min=0)))
    vtake = wh(vmiss, vtake, torch.full_like(vtake, -1))
    vevb = wh(vmiss & vbor, block[sv].gather(1, vtake.clamp(min=0)), torch.full_like(vtake, -1))
    # draft takes: after the stream's verify FREE grants
    dmiss = dcat == 0
    dq = torch.cumsum(dmiss.long(), 1) - 1 + n_vf[sd].view(-1, 1)
    dbor = bor[sd].view(-1, 1)
    dtake = wh(dmiss & ~dbor, free_sorted[sd].gather(1, dq.clamp(min=0, max=NPOS - 1)),
                        torch.full_like(dq, -1))

    # rows -> records: stream s = (l, b, h); verify row j of cohort b; o = (j H + h) k + e
    Rv, Rd = rows.vb.numel(), rows.db.numel()
    map_v = torch.full((S,), -1, dtype=torch.int64)
    map_v[sv] = torch.arange(sv.numel())
    map_d = torch.full((S,), -1, dtype=torch.int64)
    map_d[sd] = torch.arange(sd.numel())
    lj = torch.arange(L).view(-1, 1, 1)
    hh = torch.arange(H).view(1, 1, -1)
    s_v = (lj * B + rows.vb.view(1, -1, 1)) * H + hh                       # (L, Rv, H)
    s_d = (lj * B + rows.db.view(1, -1, 1)) * H + hh
    iv, idd = map_v[s_v], map_d[s_d]
    first = torch.ones(Rv, dtype=torch.bool)
    first[1:] = rows.vb[1:] != rows.vb[:-1]                               # the first row of each cohort
    e = torch.arange(k).view(1, 1, 1, -1)
    jv = torch.arange(Rv).view(1, -1, 1, 1)
    ov = (jv * H + hh.view(1, 1, -1, 1)) * k + e
    VB, VS, VC, VT, VE = vblk[iv], vslot[iv], vcat[iv], vtake[iv], vevb[iv]   # (L, Rv, H, k)
    fr = first.view(1, -1, 1, 1).expand_as(VB)
    vbor4 = bor[s_v].unsqueeze(-1).expand_as(VB)
    miss, infl = VC == 0, VC == 1
    flags = torch.zeros_like(VB)
    flags += wh(miss & fr & ~vbor4, OPS.GRANT, 0) + wh(miss & fr & vbor4, OPS.EVICT, 0)
    flags += wh(miss & ~fr, OPS.DUP, 0) + wh(infl, OPS.INFL, 0) + wh(fr & (miss | infl), OPS.FIRST, 0)
    vrec_m = miss | infl
    sv4 = s_v.unsqueeze(-1).expand_as(VB)
    igen = T1.c["gen"].view(S, NPOS)[sv4, VS.clamp(min=0)].to(torch.int64)
    vfields = torch.stack([ov.expand_as(VB), VB, flags, torch.full_like(VB, -1), wh(infl, VS, -1),
                           wh(infl, igen, -1), wh(miss, VT, -1), wh(miss & fr, VT, -1),
                           wh(miss & fr & vbor4, VE, -1)])           # (9, L, Rv, H, k)
    jd = torch.arange(Rd).view(1, -1, 1, 1)
    od = (jd * H + hh.view(1, 1, -1, 1)) * k + e
    DB_, DS, DC, DT = dblk[idd], dslot[idd], dcat[idd], dtake[idd]
    dbor4 = bor[s_d].unsqueeze(-1).expand_as(DB_)
    dmiss4 = DC == 0
    dflags = wh(dmiss4 & ~dbor4, OPS.GRANT, 0) + wh(dmiss4 & dbor4, OPS.REFUSED, 0) \
        + wh(DC == 1, OPS.INFL, 0) + wh(DC >= 2, OPS.READY, 0)
    dfields = torch.stack([od.expand_as(DB_), DB_, dflags, wh(DC >= 2, DS, -1), wh(dmiss4, DT, -1)])
    layers = []
    for l in range(L):
        vm = vrec_m[l].reshape(-1)
        vr = vfields[:, l].reshape(9, -1)[:, vm]
        vr = vr[:, torch.argsort(vr[0])]
        dr = dfields[:, l].reshape(5, -1)
        dr = dr[:, torch.argsort(dr[0])]
        layers.append(dict(l=l, vrec=vr.contiguous(), drec=dr.contiguous()))
    # vread: the slots each verify stream reads (hits, merges, takes); vlook: the first rows' ready looks
    rbits = torch.zeros((S, NPOS + 1), dtype=torch.bool)
    rsl = wh(vcat == 0, vtake, vslot)
    rbits[sv.view(-1, 1).expand_as(rsl), wh(rsl >= 0, rsl, NPOS)] = True
    vread = SP._pack_bits(rbits[:, :NPOS]).view(L, B, H, G.WP)
    vlook = torch.zeros((L, Rv, 4), dtype=torch.int64)
    hits = (VC == 2).sum((2, 3))                                           # (L, Rv) hits over the heads
    vlook[:, :, 0] = wh(first.view(1, -1), hits, 0)
    vlook[:, :, 1] = vlook[:, :, 0]
    # per-row selections (rowsel, the pins) and the layer selection words
    rsel_v = torch.zeros((L, Rv, H, NBLK + 1), dtype=torch.bool)
    rsel_v.scatter_(3, VB.clamp(min=0), True)
    rsel_d = torch.zeros((L, Rd, H, NBLK + 1), dtype=torch.bool)
    rsel_d.scatter_(3, DB_.clamp(min=0), True)
    want = dict(urgent=a.urgent, policy_e=a.policy_e, v_merge=a.v_merge, d_take=a.d_take, refused=a.refused,
                d_infl=a.d_infl, acquire=a.acquire)
    got = dict(urgent=int((torch.bitwise_and(flags, OPS.GRANT | OPS.EVICT) != 0).sum()),
               policy_e=int((torch.bitwise_and(flags, OPS.EVICT) != 0).sum()), v_merge=int(n_vm.sum()),
               d_take=int((dflags == OPS.GRANT).sum()), refused=int((dflags == OPS.REFUSED).sum()),
               d_infl=int((dflags == OPS.INFL).sum()), acquire=int(n_acq.sum()),
               v_infl_records=int((torch.bitwise_and(flags, OPS.INFL) != 0).sum()),
               v_dup_records=int((torch.bitwise_and(flags, OPS.DUP) != 0).sum()),
               v_records=int(vrec_m.sum()), d_records=int(dflags.numel()), d_ready=int((dflags == OPS.READY).sum()))
    return dict(layers=layers, vread=vread, vlook=vlook, sel=sel_w.view(L, B, H, G.WB),
                rowsel_v=SP._pack_bits(rsel_v[..., :NBLK]), rowsel_d=SP._pack_bits(rsel_d[..., :NBLK]),
                want=want, got=got)


# ===================================================================================================================
# one apply, component by component (the order OwnResidency.apply_tick will use)
def run_apply(T, win, W, R, a, tm=None, mut=None) -> Dict:
    G, rows = W["G"], W["rows"]

    def span(name, t0):
        if tm is not None:
            tm.setdefault(name, []).append(time.perf_counter_ns() - t0)

    viol: Dict[str, int] = {}

    def note(d):
        for n, v in d.items():
            c = int(v.sum()) if torch.is_tensor(v) else int(bool(v))
            viol[n] = viol.get(n, 0) + c

    t0 = time.perf_counter_ns()
    P = OPS.parse(T, R["layers"], rows)
    span("apply.parse", t0)
    t0 = time.perf_counter_ns()
    note(OPS.row_checks(T, rows))
    hole = 0
    for part in ("V", "D"):
        if P[part] is not None:
            hole += int(OPS.has(P[part]["flags"], OPS.HOLE).sum())
    viol["hole"] = hole
    span("apply.row_checks", t0)
    t0 = time.perf_counter_ns()
    so = OPS.slot_of(T)
    span("apply.slot_of", t0)
    t0 = time.perf_counter_ns()
    anyw = OPS.any_pin_words(T)
    span("apply.pin_words", t0)
    t0 = time.perf_counter_ns()
    ev = OPS.k5_events(P, T)
    span("apply.k5_events", t0)
    t0 = time.perf_counter_ns()
    note(OPS.k5_check(T, ev, mut))
    span("apply.k5_first_free", t0)
    t0 = time.perf_counter_ns()
    note(OPS.take_checks(T, ev, so))
    span("apply.take_checks", t0)
    t0 = time.perf_counter_ns()
    note(OPS.ready_check(T, P["D"]))
    span("apply.ready_check", t0)
    t0 = time.perf_counter_ns()
    mc, fv, fd = OPS.merge_check(T, P, so, ev)
    note(mc)
    span("apply.merge_check", t0)
    t0 = time.perf_counter_ns()
    note(OPS.policy_e_check(T, ev, P, R["sel"], anyw, True, mut))
    span("apply.policy_e", t0)
    t0 = time.perf_counter_ns()
    gen_before = T.c["gen"].clone()
    undo = OPS.Undo()
    clock0 = T.clock
    undo.call(lambda: setattr(T, "clock", clock0))                     # a void tick also restores the stamp clock
    span("apply.gen_snapshot", t0)
    t0 = time.perf_counter_ns()
    nv = OPS.pins_or(T, R["rowsel_v"], rows.vb, rows.vk, undo, mut)
    span("apply.pins_verify", t0)
    t0 = time.perf_counter_ns()
    nd = OPS.pins_or(T, R["rowsel_d"], rows.db, rows.dk + G.KEYS, undo, mut)
    span("apply.pins_draft", t0)
    t0 = time.perf_counter_ns()
    takes = OPS.write_takes(T, ev, a.tick, a.tick, win, undo, mut)
    span("apply.takes", t0)
    t0 = time.perf_counter_ns()
    OPS.write_merges(T, fv, fd, undo)
    span("apply.merges", t0)
    t0 = time.perf_counter_ns()
    rc = OPS.recency(T, P["D"], rows, undo, mut)
    span("apply.recency", t0)
    viol["recency_bad"] = rc["bad"]
    t0 = time.perf_counter_ns()
    pc = OPS.post_checks(T, gen_before, int(takes["f"].numel()))
    span("apply.post_checks", t0)
    note(pc)
    void = any(v for v in pc.values())
    if void:
        t0 = time.perf_counter_ns()
        undo.rollback()
        span("apply.rollback", t0)
        return dict(void=True, viol=viol, takes=int(takes["f"].numel()), pin_rounds=(nv, nd), recency=rc)
    t0 = time.perf_counter_ns()
    e_l = torch.tensor([win.register(Ev(done=False), a.tick * G.L + l) for l in range(G.L)], dtype=torch.int64)
    rd = OPS.read_done(T, R["vread"], P["D"], rows, e_l)
    span("apply.readers", t0)
    t0 = time.perf_counter_ns()
    met = OPS.metrics(P, R["vlook"], takes, rows, G)
    span("apply.metrics", t0)
    t0 = time.perf_counter_ns()
    desc, rmarks, hmark = OPS.issue_desc_args(T, takes, win)
    span("apply.issue_desc", t0)
    return dict(void=False, viol=viol, takes=int(takes["f"].numel()), force_waits=takes["force_waits"],
                superseded=takes["superseded"], pin_rounds=(nv, nd), recency=rc, readers=rd, metrics=met,
                desc_shape=list(desc.shape), desc_contiguous=bool(desc.is_contiguous()), reader_marks=rmarks.tolist(),
                hist_mark=hmark)


# ===================================================================================================================
def stats_ms(v: List[int]) -> Dict[str, float]:
    t = torch.tensor(v, dtype=torch.float64) / 1e6
    q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist()
    return dict(p10=round(q[0], 4), median=round(q[1], 4), p90=round(q[2], 4), mean=round(float(t.mean()), 4), n=len(v))


def main(argv=None) -> int:
    a = parse_args(argv)
    torch.set_num_threads(max(1, a.threads))
    W = build_world(a)
    G, T0 = W["G"], W["T"]
    # the records, against the polled state
    T1 = T0.clone()
    poll0 = run_poll(T1, fresh_window(a, G), a)
    R = gen_records(W, T1, a)
    tm: Dict[str, List[int]] = {}
    T = T0.clone()
    res = None
    pol = None
    for it in range(a.warmup + a.iters):
        T.copy_from(T0)
        win = fresh_window(a, G)
        keep = tm if it >= a.warmup else None
        pol = run_poll(T, win, a, keep)
        res = run_apply(T, win, W, R, a, keep)
    comp = {n: stats_ms(v) for n, v in tm.items()}
    poll_ms = sum(c["median"] for n, c in comp.items() if n.startswith("poll."))
    apply_ms = sum(c["median"] for n, c in comp.items() if n.startswith("apply."))
    total = round(poll_ms + apply_ms, 4)
    failed = []
    for n, w in R["want"].items():
        if R["got"][n] != w:
            failed.append("count %s: got %d want %d" % (n, R["got"][n], w))
    for n, v in res["viol"].items():
        if v:
            failed.append("predicate %s fired %d" % (n, v))
    if res["void"]:
        failed.append("the tick was void")
    for n in ("poll_I1", "poll_I2"):
        if pol[n]:
            failed.append("poll predicate %s fired" % n)
    verdict = "REFUTED" if total > FALSIFIER_MS else "SUPPORTED"
    out = dict(geometry=G._asdict(), npos=G.NPOS, slots=G.S * G.NPOS, threads=torch.get_num_threads(),
               torch=torch.__version__, host=platform.node(), args=vars(a), counts_requested=R["want"],
               counts_generated=R["got"], poll_first=poll0, poll=pol,
               apply={k: v for k, v in res.items() if k != "viol"}, violations=res["viol"],
               components_ms=dict(sorted(comp.items(), key=lambda kv: -kv[1]["median"])),
               poll_ms=round(poll_ms, 4), apply_ms=round(apply_ms, 4), total_ms=total,
               falsifier=dict(threshold_ms=FALSIFIER_MS, registered="median total > 25 ms with one thread refutes the "
                              "host estimate (rev 2 section 5)", verdict=verdict,
                              applies=torch.get_num_threads() == 1 and not a.smoke),
               failed=failed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for n, c in out["components_ms"].items():
        print("%-24s median %8.3f ms  p90 %8.3f ms" % (n, c["median"], c["p90"]))
    for f in failed:
        print("FAILED: " + f)
    print("R2_OWN2_B2T_DONE failed=%d verdict=%s total_ms=%.3f" % (len(failed), verdict, total), flush=True)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
