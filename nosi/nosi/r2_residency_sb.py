"""R2 step 7, piece 5, stage S-B: DEVICE-SIDE residency for the DRAFT rows (docs/R2_STEP7_FETCH_RETENTION.md, D2).

v1 (r2_residency.Residency.resolve) decides every row on the host, so the tick reads its selection back at every layer
(one sync per layer). S-B moves the DRAFT rows to the device:

  per layer, no host read: the host refreshes the layer's DEVICE MIRROR of its ownership state (block -> resident
  slot, in-flight flags, FREE slots, slot usage per stream; a host-to-device copy, no sync) after the host-side verify
  rows of that layer (v1, unchanged), then device_draft() decides the layer's draft rows with tensor ops only:
    READY      a selected historical block resident in the mirror (LRU / PO / VRES, arrival observed);
    page table its READY blocks in ascending block id, then its tail slots (the draft attends these, D3 masked);
    protect    every selected historical block, hits included (K2);
    in flight  in flight in the mirror, or a later duplicate of a miss of this layer (one transfer per stream and
               block: the dedup), so the draft demand is merged;
    take       a first miss granted by K5 (K3X LayerPool.can_take: free below C, else a shared extra while more than
               R_ver extras stay free), in K3X's order (row, head, ascending block); its slot = the q-th FREE slot of
               the stream (K3X first_free, q = the stream's earlier grants);
    refused    a first miss K5 refuses (counted on the device).
  at end_tick, ONE batched device-to-host read of every layer's decisions; the host then APPLIES them to the K3X
  ownership state in v1's order (protect, reads with the layer's reader event, demand merges, takes at the device's
  slots, refusals, then the O1 recency), checking each against the host state (a device decision that v1 would not
  make raises OwnershipViolation), and issues the draft transfers layer by layer.

The schedule is v1's with arrivals polled at the tick's start only and the draft transfers issued at end_tick
(Residency(poll_in_resolve=False, defer_issue=True)); under it S-B's page tables, protections, issue lists and counters
equal v1's (CPU test F10). The verify rows stay host-side (v1) until S-C. device_draft reads no device value on the host:
no .item(), .tolist(), .nonzero(), boolean-mask indexing or Python conversion of a tensor (test F10 checks its source).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .kstep.errors import OwnershipViolation
from .kstep.ownership import FREE
from .r2_residency import DRAFT, N_TAIL, Plan, Residency, Row
from . import r2_resdev as RD                                       # [resdev] --res-dev's equal device ops

FIELDS = ("blk", "valid", "ready", "slot", "infl", "granted", "refused", "take_slot", "protect")


class DraftMirror:
    """The device mirror of ONE layer's ownership state that device_draft reads (refreshed from the host state)."""

    def __init__(self, B: int, H: int, nblk: int, npos: int, device):
        self.slot_of = torch.full((B, H, nblk), -1, dtype=torch.int64, device=device)   # resident block -> slot
        self.inflight = torch.zeros((B, H, nblk), dtype=torch.bool, device=device)       # a transfer in flight
        self.free = torch.zeros((B, H, npos), dtype=torch.bool, device=device)           # FREE slots
        self.usage = torch.zeros((B, H), dtype=torch.int64, device=device)               # non-FREE slots

    def refresh(self, svc: Residency, l: int) -> None:
        B, H = svc.B, svc.H
        nblk, npos = self.slot_of.shape[-1], self.free.shape[-1]
        slot_of = torch.full((B, H, nblk), -1, dtype=torch.int64)
        inflight = torch.zeros((B, H, nblk), dtype=torch.bool)
        free = torch.zeros((B, H, npos), dtype=torch.bool)
        usage = torch.zeros((B, H), dtype=torch.int64)
        for (b, h), st in svc.pools[l].streams.items():
            for blk, idx in list(st.lru.items()) + list(st.po.items()) + list(st.vres.items()):
                if 0 <= blk < nblk:
                    slot_of[b, h, blk] = idx
            for i, s in enumerate(st.slots):
                if s.state == FREE:
                    free[b, h, i] = True
                else:
                    usage[b, h] += 1
        for (ll, b, h, blk) in svc.pending:
            if ll == l and 0 <= blk < nblk and svc._in_flight(l, b, h, blk) is not None:   # the current owner's only
                inflight[b, h, blk] = True
        self.slot_of.copy_(slot_of, non_blocking=True)
        self.inflight.copy_(inflight, non_blocking=True)
        self.free.copy_(free, non_blocking=True)
        self.usage.copy_(usage, non_blocking=True)


def rank_in_group(key: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """For every entry i (1-D tensors of N >= 1 entries, in K3X order): the number of entries j < i with mask[j] and
    key[j] == key[i]. A stable sort groups equal keys in their original order, then an exclusive count runs per
    group: O(N) memory (a one-hot (N, streams) matrix would need N x B x H)."""
    n = key.shape[0]
    ks, perm = torch.sort(key, stable=True)
    mk = mask.to(torch.int64)[perm]
    c = mk.cumsum(0) - mk                                                         # masked entries before, sorted order
    pos = torch.arange(n, device=key.device)
    start = torch.cat([torch.ones_like(ks[:1], dtype=torch.bool), ks[1:] != ks[:-1]])
    gstart = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    out = torch.empty_like(c)
    out[perm] = c - c[gstart]
    return out


class DraftDecision:
    """The device tensors of one layer's draft decisions (module docstring), candidates in K3X order (row, head,
    ascending block): N = Rd * H * k entries."""

    def __init__(self, Rd, H, k, tensors: Dict[str, torch.Tensor], page, num_pages, counters):
        self.Rd, self.H, self.k = Rd, H, k
        self.t = tensors
        self.page, self.num_pages = page, num_pages
        self.counters = counters                     # (3,) int64: masked, refused, draft (row, head) pairs


def device_draft(m: DraftMirror, rb, hb, tails, nt, sel, C: int, X: int, R_ver: int, mut=None) -> DraftDecision:
    """One layer's draft rows on the device. rb, hb (Rd,) int64: batch slot and history boundary of each draft row;
    tails (Rd, T) int64 tail slots (-1 padded), nt (Rd,) int64 their count; sel (H, Rd, k) int64 the selections.
    mut (NEGATIVE CONTROLS, tests only): no_dedup (neither the in-flight mirror nor the first-occurrence rule: a block
    in flight or already missed is requested again), ignore_reserve (K5 without R_ver), no_hit_protect."""
    mut = mut or {}
    H, Rd, k = sel.shape
    dev = sel.device
    B, _, NBLK = m.slot_of.shape
    NPOS = m.free.shape[-1]
    S = B * H
    BIG = NBLK + 1
    s = sel.permute(1, 0, 2)                                                       # (Rd, H, k)
    valid0 = (s >= 0) & (s < hb.view(Rd, 1, 1)) & (s < NBLK)
    srt = torch.sort(torch.where(valid0, s, torch.full_like(s, BIG)), dim=-1).values
    prev = torch.cat([torch.full_like(srt[..., :1], -1), srt[..., :-1]], dim=-1)
    valid = (srt < BIG) & (srt != prev)                                             # unique, ascending
    blk = torch.where(valid, srt, torch.zeros_like(srt))
    bb = rb.view(Rd, 1, 1).expand(Rd, H, k)
    hh = torch.arange(H, device=dev).view(1, H, 1).expand(Rd, H, k)
    slot = m.slot_of[bb, hh, blk]
    ready = valid & (slot >= 0)
    infl0 = valid & ~ready & m.inflight[bb, hh, blk]
    if mut.get("no_dedup"):                                                         # NEGATIVE CONTROL: in-flight not seen
        infl0 = torch.zeros_like(infl0)
    miss = valid & ~ready & ~infl0
    N = Rd * H * k
    sid = (bb * H + hh).reshape(N)
    blk_f = blk.reshape(N)
    miss_f = miss.reshape(N)
    # the dedup: the first miss of each (stream, block) in K3X order takes, later ones merge
    mkey = torch.where(miss_f, sid * NBLK + blk_f, torch.full_like(blk_f, S * NBLK))
    ks, perm = torch.sort(mkey, stable=True)
    first_sorted = torch.cat([torch.ones_like(ks[:1], dtype=torch.bool), ks[1:] != ks[:-1]])
    first = torch.empty_like(first_sorted)
    first[perm] = first_sorted
    if mut.get("no_dedup"):
        first = torch.ones_like(first)
    fmiss = miss_f & first
    dup = miss_f & ~first
    # K5 in K3X order: a stream's first (C - usage) takes are free, later ones need a shared extra; extras are granted
    # while more than R_ver stay free, i.e. to the first (X - borrowed - R_ver) extra-needing misses of the layer
    rank_s = rank_in_group(sid, fmiss)
    usage = m.usage.reshape(S)
    free_local = (C - usage).clamp(min=0)
    needs_extra = fmiss & (rank_s >= free_local[sid])
    borrowed = (usage - C).clamp(min=0).sum()
    budget = (X - borrowed - (0 if mut.get("ignore_reserve") else R_ver)).clamp(min=0)
    ne = needs_extra.to(torch.int64)
    g = ne.cumsum(0) - ne
    granted = fmiss & (~needs_extra | (g < budget))
    refused = fmiss & ~granted
    # the q-th grant of a stream takes its q-th FREE slot (K3X first_free after the earlier takes)
    q = rank_in_group(sid, granted)
    ar = torch.arange(NPOS, device=dev).view(1, NPOS).expand(S, NPOS)
    free_sorted = torch.sort(torch.where(m.free.reshape(S, NPOS), ar, torch.full_like(ar, NPOS)), dim=-1).values
    take_slot = torch.where(granted, free_sorted[sid, q.clamp(max=NPOS - 1)], torch.full_like(q, -1))
    protect = valid & ~ready if mut.get("no_hit_protect") else valid
    # the page tables: READY blocks in ascending block id, then the row's tail slots
    rkey = torch.where(ready, srt, torch.full_like(srt, BIG))
    rperm = torch.sort(rkey, dim=-1).indices
    rslot = torch.gather(slot, -1, rperm)
    nready = ready.to(torch.int64).sum(-1)                                          # (Rd, H)
    T = tails.shape[1]
    P = k + T
    page = torch.full((Rd, H, P), -1, dtype=torch.int64, device=dev)
    posk = torch.arange(k, device=dev).view(1, 1, k)
    page[..., :k] = torch.where(posk < nready.unsqueeze(-1), rslot, torch.full_like(rslot, -1))
    post = torch.arange(T, device=dev).view(1, 1, T)
    tval = torch.where(post < nt.view(Rd, 1, 1), tails.view(Rd, 1, T).expand(Rd, H, T),
                       torch.full((Rd, H, T), -1, dtype=torch.int64, device=dev))
    page.scatter_(-1, nready.unsqueeze(-1) + post, tval)
    num_pages = nready + nt.view(Rd, 1)
    masked = (nready < valid.to(torch.int64).sum(-1)).to(torch.int64).sum()
    counters = torch.stack([masked, refused.to(torch.int64).sum(), torch.full_like(masked, Rd * H)])
    t = dict(blk=blk_f, valid=valid.reshape(N), ready=ready.reshape(N), slot=slot.reshape(N),
             infl=infl0.reshape(N) | dup, granted=granted, refused=refused, take_slot=take_slot,
             protect=protect.reshape(N))
    return DraftDecision(Rd, H, k, {f: v.to(torch.int64) for f, v in t.items()}, page, num_pages, counters)


# ====================================================================================================================
# [resfix] --res-lean (OPT-IN; analysis/evalfast/DIAG_torch_res.md section 4, fixes 4-7): the torch resolve with less
# host work per layer. The functions above stay the REFERENCE, unchanged; the *_lean copies below run the same ops on
# the same values (CPU tests RL1-RL4 compare them field by field), without the negative-control muts (ResidencySC
# refuses --res-lean with one).
# ====================================================================================================================
class LeanConsts:
    """[resfix] The resolve's constant operands on one device, so a layer launches no kernel to rebuild them.
      ar(n)          torch.arange(n): a slice of ONE arange, grown on demand (the same values, contiguous, stride 1).
      true           a 0-dim device True: the value of a constant scatter (x.index_put_(i, true)). `x[i] = True` with a
                     Python True copies one pageable byte to the device and synchronizes the stream (aten::copy_ ->
                     cudaStreamSynchronize; DIAG_torch_res.md 1.2); index_put_ with a value on x's device copies nothing.
      w32            _pack_bits' bit weights (1 << i, i < 32).
      full_like(x, v[, dtype])  torch.full_like(x, v, dtype=dtype), cached for the TICK by (shape, stride, dtype, v):
                     the tensor torch.full_like would give (same values, same strides). SHARED: only an operand that no
                     op writes may come from it (every output buffer stays a fresh tensor).
      memo(key, fn)  a tick constant (fn() once per tick).
      dev            [resdev] False (the default): every lean function runs its lean ops. True (--res-dev,
                     ResidencySC.enable_res_dev): the lean functions run r2_resdev's equal ops at their device hot spots.
    new_tick() drops the tick's cache (its shapes follow the tick's rows), at the tick's first resolve."""

    dev = False                                                    # [resdev] --res-dev sets it on the service's copy

    def __init__(self, device):
        self.device = torch.device(device)
        self._ar = torch.arange(1024, device=self.device)
        self.true = torch.ones((), dtype=torch.bool, device=self.device)
        self.w32 = torch.bitwise_left_shift(torch.ones(32, dtype=torch.int64, device=self.device),
                                            torch.arange(32, dtype=torch.int64, device=self.device))
        self.tick: Dict = {}

    def new_tick(self) -> None:
        self.tick = {}

    def ar(self, n: int) -> torch.Tensor:
        n = int(n)
        if n > self._ar.shape[0]:
            self._ar = torch.arange(max(n, 2 * self._ar.shape[0]), device=self.device)
        return self._ar[:n]

    def full_like(self, like: torch.Tensor, value, dtype=None) -> torch.Tensor:
        key = ("full", tuple(like.shape), tuple(like.stride()), like.dtype if dtype is None else dtype, value)
        t = self.tick.get(key)
        if t is None:
            t = torch.full_like(like, value, dtype=dtype)
            self.tick[key] = t
        return t

    def memo(self, key, fn):
        t = self.tick.get(key)
        if t is None:
            t = fn()
            self.tick[key] = t
        return t


def _nz(x: torch.Tensor) -> torch.Tensor:
    """[resfix] x != 0 as a bool tensor; a bool x is returned as it is (x != 0 equals x: no kernel)."""
    return x if x.dtype == torch.bool else x != 0


def rank_prep(key: torch.Tensor, cst: LeanConsts):
    """[resfix] The key-only half of rank_in_group (its stable sort and group starts), so rank_with(rank_prep(key),
    mask) == rank_in_group(key, mask) (the same ops in the same order). The rows' stream ids are a TICK constant: one
    sort per tick instead of five per layer."""
    n = key.shape[0]
    ks, perm = torch.sort(key, stable=True)
    pos = cst.ar(n)
    start = torch.cat([cst.full_like(ks[:1], True, dtype=torch.bool), ks[1:] != ks[:-1]])
    if cst.dev:                                                    # [resdev] no single-block 1-D cummax
        return perm, RD.group_start(start, pos)
    gstart = torch.cummax(torch.where(start, pos, cst.full_like(pos, 0)), 0).values
    return perm, gstart


def rank_with(prep, mask: torch.Tensor) -> torch.Tensor:
    """[resfix] rank_in_group's mask half on a rank_prep result."""
    perm, gstart = prep
    mk = mask.to(torch.int64)[perm]
    c = mk.cumsum(0) - mk
    out = torch.empty_like(c)
    out[perm] = c - c[gstart]
    return out


def device_draft_lean(m: DraftMirror, rb, hb, tails, nt, sel, C: int, X: int, R_ver: int, cst: LeanConsts, rc: Dict,
                      s=None, valid0=None) -> DraftDecision:
    """[resfix] device_draft without mut, the same values. rc: the tick's draft-row constants (ResidencySC._lean_rows_c:
    bb, hh views, sid, sidN = sid * NBLK, tval, rk = rank_prep(sid)); s, valid0: the permuted selection and its
    validity if the caller has them (ResidencySC._resolve_lean computes them once for Policy E's mask too). The bool
    fields stay bool (no int64 cast; every consumer reads them as != 0 or as truth values)."""
    H, Rd, k = sel.shape
    dev = sel.device
    B, _, NBLK = m.slot_of.shape
    NPOS = m.free.shape[-1]
    S = B * H
    BIG = NBLK + 1
    fl = cst.full_like
    if s is None:
        s = sel.permute(1, 0, 2)                                                   # (Rd, H, k)
    if valid0 is None:
        valid0 = (s >= 0) & (s < hb.view(Rd, 1, 1)) & (s < NBLK)
    srt = torch.sort(torch.where(valid0, s, fl(s, BIG)), dim=-1).values
    prev = torch.cat([fl(srt[..., :1], -1), srt[..., :-1]], dim=-1)
    valid = (srt < BIG) & (srt != prev)                                             # unique, ascending
    blk = torch.where(valid, srt, fl(srt, 0))
    bb, hh = rc["bb"], rc["hh"]
    slot = m.slot_of[bb, hh, blk]
    ready = valid & (slot >= 0)
    infl0 = valid & ~ready & m.inflight[bb, hh, blk]
    miss = valid & ~ready & ~infl0
    N = Rd * H * k
    sid = rc["sid"]
    blk_f = blk.reshape(N)
    miss_f = miss.reshape(N)
    mkey = torch.where(miss_f, rc["sidN"] + blk_f, fl(blk_f, S * NBLK))
    if cst.dev:                                                    # [resdev] 0 <= mkey <= S * NBLK: int32 radix keys
        ks, perm = RD.sort_key32(mkey, S * NBLK)
    else:
        ks, perm = torch.sort(mkey, stable=True)
    first_sorted = torch.cat([fl(ks[:1], True, dtype=torch.bool), ks[1:] != ks[:-1]])
    first = torch.empty_like(first_sorted)
    first[perm] = first_sorted
    fmiss = miss_f & first
    dup = miss_f & ~first
    rank_s = rank_with(rc["rk"], fmiss)
    usage = m.usage.reshape(S)
    free_local = (C - usage).clamp(min=0)
    needs_extra = fmiss & (rank_s >= free_local[sid])
    borrowed = (usage - C).clamp(min=0).sum()
    budget = (X - borrowed - R_ver).clamp(min=0)
    ne = needs_extra.to(torch.int64)
    g = ne.cumsum(0) - ne
    granted = fmiss & (~needs_extra | (g < budget))
    refused = fmiss & ~granted
    q = rank_with(rc["rk"], granted)
    if cst.dev:                                                    # [resdev] a compaction, no segmented sort
        free_sorted = RD.sorted_free(m.free.reshape(S, NPOS), flat=dev_flat(cst, S, NPOS))
    else:
        ar = cst.ar(NPOS).view(1, NPOS).expand(S, NPOS)
        free_sorted = torch.sort(torch.where(m.free.reshape(S, NPOS), ar, fl(ar, NPOS)), dim=-1).values
    take_slot = torch.where(granted, free_sorted[sid, q.clamp(max=NPOS - 1)], fl(q, -1))
    rkey = torch.where(ready, srt, fl(srt, BIG))
    rperm = torch.sort(rkey, dim=-1).indices
    rslot = torch.gather(slot, -1, rperm)
    nready = ready.sum(-1) if cst.dev else ready.to(torch.int64).sum(-1)           # (Rd, H) [resdev] bool sum: int64
    T = tails.shape[1]
    P = k + T
    page = torch.full((Rd, H, P), -1, dtype=torch.int64, device=dev)
    posk = cst.ar(k).view(1, 1, k)
    page[..., :k] = torch.where(posk < nready.unsqueeze(-1), rslot, fl(rslot, -1))
    post = cst.ar(T).view(1, 1, T)
    page.scatter_(-1, nready.unsqueeze(-1) + post, rc["tval"])
    num_pages = nready + nt.view(Rd, 1)
    if cst.dev:                                                    # [resdev] a bool sum is int64 already: no casts
        masked = (nready < valid.sum(-1)).sum()
        counters = torch.stack([masked, refused.sum(), fl(masked, Rd * H)])
    else:
        masked = (nready < valid.to(torch.int64).sum(-1)).to(torch.int64).sum()
        counters = torch.stack([masked, refused.to(torch.int64).sum(), fl(masked, Rd * H)])
    t = dict(blk=blk_f, valid=valid.reshape(N), ready=ready.reshape(N), slot=slot.reshape(N),
             infl=infl0.reshape(N) | dup, granted=granted, refused=refused, take_slot=take_slot,
             protect=valid.reshape(N))
    return DraftDecision(Rd, H, k, t, page, num_pages, counters)


def dev_flat(cst: LeanConsts, S: int, NP: int) -> torch.Tensor:
    """[resdev] arange(S * NP).view(S, NP) (int64), a tick constant of cst (r2_resdev.sorted_free's flat)."""
    return cst.memo(("rd_flat", S, NP), lambda: cst.ar(S * NP).view(S, NP))


class ResidencySB(Residency):
    """Residency with the draft rows decided on the device (module docstring). Extra arguments: device (the mirrors'
    and the decisions' device), nblk (block ids per request the store can name), sb_mut (negative controls)."""

    def __init__(self, *args, device="cpu", nblk: int = 0, sb_mut=None, **kw):
        kw["poll_in_resolve"] = False
        kw["defer_issue"] = True
        super().__init__(*args, **kw)
        if nblk < 1:
            raise ValueError("nblk (block ids per request) must be >= 1")
        self.device = torch.device(device)
        self.nblk = int(nblk)
        self.mirrors = [DraftMirror(self.B, self.H, self.nblk, self.n_hist, self.device) for _ in range(self.L)]
        self.sb_mut = dict(sb_mut or {})
        self._dec: List[Tuple[int, List[Row], DraftDecision]] = []
        self._layer_ev: Dict[int, object] = {}
        self.d2h_reads = 0

    # [xpool] the reads below are BLOCKING (t.to("cpu"), no non_blocking): S-C's tick-end decision read then waits for
    # every reader of the tick. The shared X pool reuses lanes across requests on that ground (r2_xpool.py, the
    # lane-level order); a class whose decision read does not block must set this False (the pool then refuses).
    DECISION_READ_BLOCKING = True

    def _d2h(self, t: torch.Tensor) -> torch.Tensor:
        """Every device-to-host read of S-B goes through here (counted: test F10)."""
        self.d2h_reads += 1
        return t.to("cpu")

    def resolve(self, l: int, rows: Sequence[Row], sel: torch.Tensor) -> Plan:
        H, R = self.H, len(rows)
        self._check_rows(l, rows, sel)
        plan = self._new_plan(rows)
        vrows = [i for i, r in enumerate(rows) if r.kind == "v"]
        drows = [i for i, r in enumerate(rows) if r.kind == "d"]
        hist = [[[] for _ in range(H)] for _ in range(R)]
        if vrows:                                                     # the verify rows: host, v1 (one read per layer)
            sel_v = self._d2h(sel[:, vrows])
            for jj, i in enumerate(vrows):
                for h in range(H):
                    hist[i][h] = self._hist(sel_v[h, jj], rows[i].hist_bound)
        entries = self._verify_part(l, rows, hist, plan)
        k = int(sel.shape[2])
        P = k + N_TAIL
        page_h = torch.full((R, H, P), -1, dtype=torch.int64)
        npg_h = torch.zeros((R, H), dtype=torch.int64)
        for i in vrows:
            r = rows[i]
            for h in range(H):
                ent = list(entries[(i, h)])
                for (t, idx, gen) in ent:
                    plan.reads.append((r.b, h, idx, gen, "verify"))
                    self.metrics.read(l, r.b, h, idx, gen, True)
                full = [(t, s) for (t, s, _) in ent] + [(int(t), int(s)) for (t, s) in r.tail]
                for t, s in full[len(ent):]:
                    if not self.tail0 <= s < self.tail0 + N_TAIL:
                        raise ValueError("tail slot %d outside [%d, %d)" % (s, self.tail0, self.tail0 + N_TAIL))
                plan.blocks[i][h] = [t for (t, _) in full]
                if full:
                    page_h[i, h, :len(full)] = torch.tensor([s for (_, s) in full], dtype=torch.int64)
                npg_h[i, h] = len(full)
        page = page_h.to(self.device)
        npg = npg_h.to(self.device)
        if drows:                                                     # the draft rows: on the device, no host read
            m = self.mirrors[l]
            m.refresh(self, l)
            for i in drows:
                if len(rows[i].tail) > N_TAIL:
                    raise ValueError("at most %d tail pages per row" % N_TAIL)
                for (_, s) in rows[i].tail:
                    if not self.tail0 <= s < self.tail0 + N_TAIL:
                        raise ValueError("tail slot %d outside [%d, %d)" % (s, self.tail0, self.tail0 + N_TAIL))
            tl = torch.full((len(drows), N_TAIL), -1, dtype=torch.int64)
            for jj, i in enumerate(drows):
                for u, (_, s) in enumerate(rows[i].tail):
                    tl[jj, u] = int(s)
            dv = self.device
            rb = torch.tensor([rows[i].b for i in drows], dtype=torch.int64).to(dv)
            hb = torch.tensor([rows[i].hist_bound for i in drows], dtype=torch.int64).to(dv)
            nt = torch.tensor([len(rows[i].tail) for i in drows], dtype=torch.int64).to(dv)
            didx = torch.tensor(drows, dtype=torch.int64).to(dv)
            dec = device_draft(m, rb, hb, tl.to(dv), nt, sel[:, didx].to(dv), self.C, self.X, self.R_ver, self.sb_mut)
            page[didx] = dec.page
            npg[didx] = dec.num_pages
            self._dec.append((l, [rows[i] for i in drows], dec))
        plan.page_table = page.to(torch.int32)
        plan.num_pages = npg.to(torch.int32)
        plan.last_len = torch.tensor([r.pos % self.bs + 1 for r in rows], dtype=torch.int32).to(self.device)
        plan.seq_idx = torch.tensor([r.b for r in rows], dtype=torch.int32).to(self.device)
        self.metrics.sample_memory()
        return plan

    def read_done(self, l: int, plan: Plan, ev) -> None:
        super().read_done(l, plan, ev)                                # the verify reads, the key readers, the queue
        self._layer_ev[l] = ev                                        # the draft reads get it at end_tick

    def end_tick(self) -> None:
        if self._dec:
            parts = []
            for (_, _, dec) in self._dec:
                parts.append(torch.stack([dec.t[f] for f in FIELDS]).reshape(-1))
                parts.append(dec.counters)
            flat = self._d2h(torch.cat(parts)).tolist()               # ONE read per tick
            at = 0
            for (l, drow_objs, dec) in self._dec:
                N = dec.Rd * dec.H * dec.k
                cols = {f: flat[at + j * N: at + (j + 1) * N] for j, f in enumerate(FIELDS)}
                at += len(FIELDS) * N
                masked, refused, pairs = flat[at:at + 3]
                at += 3
                self._apply(l, drow_objs, dec, cols, masked, refused, pairs)
            self._dec = []
        self._layer_ev = {}
        super().end_tick()                                            # the draft transfers, layer by layer
        self.metrics.sample_memory()

    def _apply(self, l: int, drow_objs: List[Row], dec: DraftDecision, c, masked: int, refused: int, pairs: int) -> None:
        """The host applies one layer's device decisions in v1's order, checking each against the host state."""
        H, k = dec.H, dec.k
        pool = self.pools[l]
        got: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        trs = []
        n_ref = 0
        for o in range(dec.Rd * H * k):
            if not c["valid"][o]:
                continue
            j, h, t = o // (H * k), (o // k) % H, int(c["blk"][o])
            r = drow_objs[j]
            st = pool.stream(r.b, h)
            if c["protect"][o]:
                st.protect(t, r.key, DRAFT)
            if c["ready"][o]:
                idx = int(c["slot"][o])
                if st.resident(t) != idx:
                    raise OwnershipViolation("S-B: the device read block %d at slot %d, the host holds it at %s (layer %d "
                                             "b %d h %d)" % (t, idx, st.resident(t), l, r.b, h))
                got.setdefault((j, h), []).append((t, idx, st.slots[idx].gen))
            elif c["infl"][o]:
                tr = self._in_flight(l, r.b, h, t)
                if tr is None:
                    raise OwnershipViolation("S-B: the device merged block %d (layer %d b %d h %d), which is not in flight"
                                             % (t, l, r.b, h))
                if not self.mut.drop_demand_merge:
                    tr.demand.add(DRAFT)
            elif c["granted"][o]:
                trs.append(self._take(l, r.b, h, t, r.key, DRAFT, at=int(c["take_slot"][o])))
            elif c["refused"][o]:
                if self._draft_can_take(pool, st):
                    raise OwnershipViolation("S-B: the device refused block %d (layer %d b %d h %d), K5 allows it"
                                             % (t, l, r.b, h))
                self.metrics.count("draft_refused")
                n_ref += 1
        if n_ref != refused:
            raise OwnershipViolation("S-B: %d refusals applied, the device counted %d" % (n_ref, refused))
        ev = self._layer_ev.get(l)
        for (j, h), lst in sorted(got.items()):
            r = drow_objs[j]
            for (t, idx, gen) in lst:
                s = pool.stream(r.b, h).slots[idx]
                if s.first_reader is None:
                    s.first_reader = ev
                s.last_reader = ev
                self.metrics.read(l, r.b, h, idx, gen, False)
        self._draft_recency(l, drow_objs, got)
        if trs:
            self._deferred.append((l, trs))
        self.metrics.layer_plan(Plan(page_table=None, num_pages=None, last_len=None, seq_idx=None, blocks=[],
                                     masked=int(masked), draft_pairs=int(pairs), refused=int(refused)))
