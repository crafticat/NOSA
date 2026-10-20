"""R2 step 7, piece 5, stage S-C: DEVICE-SIDE residency for the VERIFY rows (docs/R2_STEP7_FETCH_RETENTION.md, 12c).

S-B decides the draft rows on the device, but its verify rows stay host-side (v1): a verify tick reads the verify
selection back at every layer. S-C moves the verify rows to the device too. resolve() reads NO device value on the
host; end_tick() makes ONE batched device-to-host read per tick, for the verify and the draft decisions together.

THE DEVICE MIRROR (DeviceMirror, note Q5). The ownership state the device decides on lives ON THE DEVICE as a per-slot
table of every layer (state code, block, generation, "the current owner's transfer in flight"). The host keeps no copy
of it and never scans it: every transition of a stream (K3X Stream.log: take, release, evict_po, acquire, admit,
admit_vres, drop_unowned, drop_vres_unprotected, free) marks its slot dirty as it happens, and so does an owner change
(begin_request / end_request: the batch slot's transfers in flight). At the FIRST resolve of a tick the host sends the
records of the dirty slots in ONE batched host-to-device copy (sync_mirror), and the tick's row inputs in one more
(every layer of a tick resolves the same rows). Each resolve derives its layer's mirror from the table with device ops
only. The full host scan (VerifyMirror.refresh) stays as the reference (CPU test F14).

ONE LAYER (resolve). No host-to-device copy and no host read. The layer's mirror (S-B's fields plus the slot and
generation of the transfer in flight per block) is derived on the device. Then device_verify() decides the verify rows
with tensor ops, in K3X order (verify row, head, ascending block; dispatcher.py:2568-2589):
  READY      resident in the mirror: read in place (a verifier read does not touch the LRU, K1).
  IN FLIGHT  a transfer of the stream's owner in flight in the mirror: the read WAITS for it on the device (below);
             the demand {verify} is merged at end_tick.
  NEW        an urgent miss. The first occurrence per (stream, block) takes a slot by K5 for the verifier: a stream's
             first C - usage takes are local, later ones need a shared extra, granted while extras stay free, the
             verifier reserve R_ver INCLUDED (the PRE-RESERVED VERIFIER POOL: draft fetches cannot borrow its last
             R_ver extras, K5). Its slot is the stream's q-th FREE slot (K3X first_free). A later occurrence (another
             verify row of the cohort, same stream) reads the same slot: one transfer per stream and block.
  OVERFLOW   a first miss that K5 cannot grant (v1 evicts or forces there, dispatcher.py:2416-2443). The device sets
             the OVERFLOW FLAG and points the page at slot 0 (in bounds); end_tick raises CapacityRefusal before any
             change of the host state. The tick's outputs are invalid and the run stops: C + X or R_ver is too small
             (a configuration error, as v1's own CapacityRefusal).
  protect    every valid selected block (K2), applied at end_tick.
Then the device (a) WAITS for the in-flight blocks the verify rows read, (b) GATHERS the urgent blocks (K, V, bias)
into their slots on the main stream before the layer's attention, (c) updates the mirror (the urgent blocks in flight,
their slots no longer FREE, the usage) so that device_draft (S-B) decides the layer's draft rows on the state that v1's
draft rows see after the verify rows.

END OF TICK. ONE read of every layer's decisions. Any overflow flag -> CapacityRefusal. Else, per layer in resolve
order, the host APPLIES the verify decisions in v1's order (protect; the demand merges; the urgent takes at the
device's slots, each checked against the host state like S-B's; the 4.3 / 4.6 looks; the reads with the layer's
reader event; the verify keys queued behind it), then the draft decisions (S-B), then issues the draft transfers.

THE DEVICE OPERATIONS (dev):
  CpuDeviceSim   the CPU stand-in (tests). The CPU is the device here, so the stand-in reads its own decision
                 tensors; it never changes the host's ownership state. wait: the needed transfers land (the
                 transport's force), exactly v1's waits. gather: the copies; it refuses a gather into a slot whose
                 last reader is pending (T3) and waits for the block's host write-back first (T4), as v1's gather.
  CudaDeviceOps  the GPU. SOURCE ONLY, NOT RUN YET (GPU gate FG1). wait: the main stream waits (cudaStreamWaitEvent)
                 on the completion event of EVERY transfer in flight at the layer: a SUPERSET of the needed transfers,
                 because the host does not read the selection; a CPU pack that is not submitted yet is host-waited for
                 its SUBMISSION first (note section 6, Q2). A wait changes no ownership state (a transfer completes when
                 it completes); it changes only when the main stream runs. gather: NOSI's flash_h2d_from_mask_bias (K
                 and bias) and flash_h2d_from_mask (V) with the decision's device load ids, after a main-stream wait on
                 every host write-back registered since the last gather (T4). The slots are FREE, so their last reader
                 completed (K4, T3). wait_mode="exact" (note 12f) replaces the event waits by ONE Triton program that
                 spins on the generation board (GenBoard; the transport publishes (slot, gen) after the rows) for
                 exactly the slots of the layer's wait list; err reports a spin that ran out (watch_flag).
  ExactWaitSim   the CPU stand-in of the exact wait (device data only: the wait list and the board).

THE SCHEDULE and v1's reference: arrivals polled at the tick's start; draft transfers issued at end_tick; the waited
and the urgent blocks read in place and admitted at the next poll: Residency(poll_in_resolve=False, defer_issue=True,
defer_admit=True). Under it S-C's page tables, ownership state and counters equal v1's (CPU tests F12, F12b).
The per-layer path (resolve, DeviceMirror.layer, device_verify, device_draft, the GPU operations) reads no device value
on the host and copies nothing from the host to the device (F12d checks their source); per tick there are at most two
host-to-device copies (the mirror delta, the row inputs) and one device-to-host read (end_tick).
"""
from __future__ import annotations

import threading
from typing import Dict, List, Sequence, Set, Tuple

import torch

from .kstep.backend import ev_done
from .kstep.errors import CapacityRefusal, OwnershipViolation
from .kstep.ownership import FREE, INFLIGHT, LRU, PO, RELEASING, VRES
from .r2_residency import BLOCK, N_TAIL, VERIFY, Plan, Residency, Row
from .r2_residency_sb import FIELDS, DraftMirror, ResidencySB, device_draft, rank_in_group

FIELDS_V = ("blk", "valid", "ready", "slot", "infl", "infl_slot", "infl_gen", "dup", "granted", "take_slot", "rslot",
            "protect")
C_FREE, C_INFLIGHT, C_LRU, C_PO, C_VRES, C_RELEASING = range(6)
CODE = {FREE: C_FREE, INFLIGHT: C_INFLIGHT, LRU: C_LRU, PO: C_PO, VRES: C_VRES, RELEASING: C_RELEASING}
MIRROR_FIELDS = ("slot_of", "inflight", "infl_slot", "infl_gen", "free", "usage", "releasing")
JOURNAL_KINDS = ("take", "release", "evict_po", "acquire", "admit", "admit_vres", "drop_unowned",
                 "drop_vres_unprotected", "free")        # "touch" changes only the LRU order: not mirrored


class _Journal(list):
    """A stream's transition log (K3X Stream.log, appended by the vendored ownership code) that also reports every
    transition to the service as it is appended: the slot becomes dirty for the next mirror delta."""

    def __init__(self, key: Tuple[int, int, int], sink, items=()):
        super().__init__(items)
        self.key, self.sink = key, sink

    def append(self, item) -> None:
        super().append(item)
        self.sink(self.key, item)


class LayerMirror:
    """One layer's mirror as device_verify and device_draft read it (the fields of VerifyMirror, MIRROR_FIELDS)."""

    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class DeviceMirror:
    """The residency mirror of EVERY layer on the device (module docstring): a per-slot table (L, B, H, C + X) of the
    state code, the block, the generation and the flag "the current owner's transfer in flight". apply() writes the
    records of one delta; layer() derives one layer's mirror with device ops only."""

    def __init__(self, L: int, B: int, H: int, nblk: int, npos: int, device):
        shape = (L, B, H, npos)
        self.nblk = int(nblk)
        self.code = torch.zeros(shape, dtype=torch.int64, device=device)          # every slot FREE
        self.block = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.gen = torch.zeros(shape, dtype=torch.int64, device=device)
        self.cur = torch.zeros(shape, dtype=torch.int64, device=device)

    def apply(self, recs: torch.Tensor) -> None:
        """recs (5, n) int64 ON THE DEVICE: flat slot index, state code, block, generation, current-owner flag; the
        slot indices are distinct."""
        f = recs[0]
        for t, row in ((self.code, 1), (self.block, 2), (self.gen, 3), (self.cur, 4)):
            t.view(-1)[f] = recs[row]

    def layer(self, l: int) -> LayerMirror:
        """Layer l's mirror, derived from the table on the device (fresh tensors: the decisions may change them)."""
        code, blk, gen, cur = self.code[l], self.block[l], self.gen[l], self.cur[l]   # (B, H, NPOS)
        B, H, NPOS = code.shape
        NBLK = self.nblk
        dev = code.device
        okb = (blk >= 0) & (blk < NBLK)
        base = (torch.arange(B, device=dev).view(B, 1, 1) * H + torch.arange(H, device=dev).view(1, H, 1)) * NBLK
        dummy = B * H * NBLK                                                       # one extra element takes the rest
        flat = base + blk.clamp(min=0, max=NBLK - 1)
        idx = torch.arange(NPOS, device=dev).view(1, 1, NPOS).expand(B, H, NPOS).reshape(-1)
        res = okb & ((code == C_LRU) | (code == C_PO) | (code == C_VRES))
        fly = okb & (code == C_INFLIGHT) & (cur != 0)
        at_res = torch.where(res, flat, torch.full_like(flat, dummy)).reshape(-1)
        at_fly = torch.where(fly, flat, torch.full_like(flat, dummy)).reshape(-1)
        slot_of = torch.full((dummy + 1,), -1, dtype=torch.int64, device=dev)
        slot_of[at_res] = idx                                                      # one resident slot per block (K3)
        infl_slot = torch.full((dummy + 1,), -1, dtype=torch.int64, device=dev)
        infl_slot[at_fly] = idx                                                    # one owner transfer per block
        infl_gen = torch.zeros((dummy + 1,), dtype=torch.int64, device=dev)
        infl_gen[at_fly] = gen.reshape(-1)
        shp = (B, H, NBLK)
        return LayerMirror(slot_of=slot_of[:dummy].view(shp), inflight=(infl_slot[:dummy] >= 0).view(shp),
                           infl_slot=infl_slot[:dummy].view(shp), infl_gen=infl_gen[:dummy].view(shp),
                           free=code == C_FREE, usage=(code != C_FREE).to(torch.int64).sum(-1),
                           releasing=code == C_RELEASING)


class VerifyMirror(DraftMirror):
    """S-B's mirror of one layer plus, per (b, h, block), the slot and generation of the current owner's transfer in
    flight (the device's wait reads them), and the RELEASING slots (read only by a negative control)."""

    def __init__(self, B: int, H: int, nblk: int, npos: int, device):
        super().__init__(B, H, nblk, npos, device)
        self.infl_slot = torch.full((B, H, nblk), -1, dtype=torch.int64, device=device)
        self.infl_gen = torch.zeros((B, H, nblk), dtype=torch.int64, device=device)
        self.releasing = torch.zeros((B, H, npos), dtype=torch.bool, device=device)

    def refresh(self, svc: Residency, l: int) -> None:
        super().refresh(svc, l)
        B, H = svc.B, svc.H
        nblk, npos = self.slot_of.shape[-1], self.free.shape[-1]
        infl_slot = torch.full((B, H, nblk), -1, dtype=torch.int64)
        infl_gen = torch.zeros((B, H, nblk), dtype=torch.int64)
        releasing = torch.zeros((B, H, npos), dtype=torch.bool)
        for (ll, b, h, blk), tr in svc.pending.items():
            if ll == l and 0 <= blk < nblk and svc._in_flight(l, b, h, blk) is tr:
                infl_slot[b, h, blk] = tr.slot
                infl_gen[b, h, blk] = tr.gen
        for (b, h), st in svc.pools[l].streams.items():
            for i, s in enumerate(st.slots):
                if s.state == RELEASING:
                    releasing[b, h, i] = True
        self.infl_slot.copy_(infl_slot, non_blocking=True)
        self.infl_gen.copy_(infl_gen, non_blocking=True)
        self.releasing.copy_(releasing, non_blocking=True)


class VerifyDecision:
    """The device tensors of one layer's verify decisions, candidates in K3X order (verify row, head, ascending
    block): N = Rv * H * k entries."""

    def __init__(self, Rv, H, k, tensors: Dict[str, torch.Tensor], page, num_pages, load_ids, bh, counters):
        self.Rv, self.H, self.k = Rv, H, k
        self.t = tensors
        self.page, self.num_pages = page, num_pages
        self.load_ids = load_ids                     # (H, B, n_hist): the urgent block of each slot, -1 elsewhere
        self.bh = bh                                 # (b, h) of each entry (the device operations read them)
        self.counters = counters                     # (1,) int64: the overflowed first misses


def device_verify(m: VerifyMirror, rb, hb, tails, nt, sel, C: int, X: int, mut=None) -> VerifyDecision:
    """One layer's verify rows on the device (module docstring), then the mirror update for the layer's draft rows.
    rb, hb (Rv,) int64: batch slot and history boundary of each verify row; tails (Rv, T) int64 tail slots (-1
    padded), nt (Rv,) int64 their count; sel (H, Rv, k) int64 the selections.
    mut (NEGATIVE CONTROLS, tests only): gather_into_releasing (a RELEASING slot counts as FREE: the urgent gather
    writes a slot whose last reader may be pending), no_verify_protect (the verify read protects nothing),
    no_verify_dedup (a later occurrence of a miss takes a second slot: two transfers of one block), no_overflow_flag
    (an overflow is not reported)."""
    mut = mut or {}
    H, Rv, k = sel.shape
    dev = sel.device
    B, _, NBLK = m.slot_of.shape
    NPOS = m.free.shape[-1]
    S = B * H
    BIG = NBLK + 1
    s = sel.permute(1, 0, 2)                                                       # (Rv, H, k)
    valid0 = (s >= 0) & (s < hb.view(Rv, 1, 1)) & (s < NBLK)
    srt = torch.sort(torch.where(valid0, s, torch.full_like(s, BIG)), dim=-1).values
    prev = torch.cat([torch.full_like(srt[..., :1], -1), srt[..., :-1]], dim=-1)
    valid = (srt < BIG) & (srt != prev)                                             # unique, ascending
    blk = torch.where(valid, srt, torch.zeros_like(srt))
    bb = rb.view(Rv, 1, 1).expand(Rv, H, k)
    hh = torch.arange(H, device=dev).view(1, H, 1).expand(Rv, H, k)
    slot = m.slot_of[bb, hh, blk]
    ready = valid & (slot >= 0)
    infl = valid & ~ready & m.inflight[bb, hh, blk]
    islot = m.infl_slot[bb, hh, blk]
    igen = m.infl_gen[bb, hh, blk]
    miss = valid & ~ready & ~infl
    N = Rv * H * k
    sid = (bb * H + hh).reshape(N)
    bb_f, hh_f, blk_f = bb.reshape(N), hh.reshape(N), blk.reshape(N)
    miss_f = miss.reshape(N)
    # one transfer per (stream, block): the first miss in K3X order takes, a later one reads the same slot
    mkey = torch.where(miss_f, sid * NBLK + blk_f, torch.full_like(blk_f, S * NBLK))
    ks, perm = torch.sort(mkey, stable=True)
    first_sorted = torch.cat([torch.ones_like(ks[:1], dtype=torch.bool), ks[1:] != ks[:-1]])
    first = torch.empty_like(first_sorted)
    first[perm] = first_sorted
    if mut.get("no_verify_dedup"):                                                  # NEGATIVE CONTROL: every miss takes
        first = torch.ones_like(first)
    fmiss = miss_f & first
    dup = miss_f & ~first
    # K5 for the verifier, in K3X order: a stream's first (C - usage) takes are local, later ones need a shared extra;
    # extras are granted while any is free (the reserve R_ver included): to the first (X - borrowed) extra-needing misses
    rank_s = rank_in_group(sid, fmiss)
    usage = m.usage.reshape(S)
    free_local = (C - usage).clamp(min=0)
    needs_extra = fmiss & (rank_s >= free_local[sid])
    borrowed = (usage - C).clamp(min=0).sum()
    budget = (X - borrowed).clamp(min=0)
    ne = needs_extra.to(torch.int64)
    g = ne.cumsum(0) - ne
    granted = fmiss & (~needs_extra | (g < budget))
    over = fmiss & ~granted
    # the q-th grant of a stream takes its q-th FREE slot (K3X first_free after the earlier takes)
    q = rank_in_group(sid, granted)
    freem = (m.free | m.releasing) if mut.get("gather_into_releasing") else m.free
    ar = torch.arange(NPOS, device=dev).view(1, NPOS).expand(S, NPOS)
    free_sorted = torch.sort(torch.where(freem.reshape(S, NPOS), ar, torch.full_like(ar, NPOS)), dim=-1).values
    take_slot = torch.where(granted, free_sorted[sid, q.clamp(max=NPOS - 1)], torch.full_like(q, -1))
    # a later occurrence reads the slot of its group's first occurrence
    pos = torch.arange(N, device=dev)
    gstart = torch.cummax(torch.where(first_sorted, pos, torch.zeros_like(pos)), 0).values
    grp_slot = torch.empty_like(take_slot)
    grp_slot[perm] = take_slot[perm][gstart]
    neg = torch.full_like(take_slot, -1)
    rslot = torch.where(ready.reshape(N), slot.reshape(N),
                        torch.where(infl.reshape(N), islot.reshape(N),
                                    torch.where(fmiss, take_slot, torch.where(dup, grp_slot, neg))))
    valid_f = valid.reshape(N)
    hole = valid_f & (rslot < 0)                                                    # an overflowed block
    rslot_page = torch.where(hole, torch.zeros_like(rslot), rslot)                  # in bounds; the flag voids the tick
    protect = torch.zeros_like(valid) if mut.get("no_verify_protect") else valid
    # the page tables: every valid selected block in ascending block id, then the row's tail slots
    vkey = torch.where(valid, srt, torch.full_like(srt, BIG))
    vperm = torch.sort(vkey, dim=-1).indices
    pslot = torch.gather(rslot_page.view(Rv, H, k), -1, vperm)
    nvalid = valid.to(torch.int64).sum(-1)                                          # (Rv, H)
    T = tails.shape[1]
    P = k + T
    page = torch.full((Rv, H, P), -1, dtype=torch.int64, device=dev)
    posk = torch.arange(k, device=dev).view(1, 1, k)
    page[..., :k] = torch.where(posk < nvalid.unsqueeze(-1), pslot, torch.full_like(pslot, -1))
    post = torch.arange(T, device=dev).view(1, 1, T)
    tval = torch.where(post < nt.view(Rv, 1, 1), tails.view(Rv, 1, T).expand(Rv, H, T),
                       torch.full((Rv, H, T), -1, dtype=torch.int64, device=dev))
    page.scatter_(-1, nvalid.unsqueeze(-1) + post, tval)
    num_pages = nvalid + nt.view(Rv, 1)
    # the urgent gather's load ids: slot -> block for every grant (a dummy column takes the other entries)
    col = torch.where(granted, take_slot, torch.full_like(take_slot, NPOS))
    ids = torch.full((H, B, NPOS + 1), -1, dtype=torch.int64, device=dev)
    ids.index_put_((hh_f, bb_f, col), torch.where(granted, blk_f, torch.full_like(blk_f, -1)))
    load_ids = ids[..., :NPOS].contiguous()
    # the mirror after the verify rows (device_draft reads it): urgent blocks in flight, their slots taken
    tk = granted.to(torch.int64)
    d_infl = torch.zeros(S * NBLK, dtype=torch.int64, device=dev).index_add_(0, sid * NBLK + blk_f, tk)
    m.inflight |= d_infl.view(B, H, NBLK) > 0
    d_free = torch.zeros(S * NPOS, dtype=torch.int64, device=dev).index_add_(0, sid * NPOS + take_slot.clamp(min=0), tk)
    m.free &= ~(d_free.view(B, H, NPOS) > 0)
    m.usage += torch.zeros(S, dtype=torch.int64, device=dev).index_add_(0, sid, tk).view(B, H)
    n_over = over.to(torch.int64).sum()
    counters = torch.stack([torch.zeros_like(n_over) if mut.get("no_overflow_flag") else n_over])
    t = dict(blk=blk_f, valid=valid_f, ready=ready.reshape(N), slot=slot.reshape(N), infl=infl.reshape(N),
             infl_slot=islot.reshape(N), infl_gen=igen.reshape(N), dup=dup, granted=granted, take_slot=take_slot,
             rslot=rslot, protect=protect.reshape(N))
    return VerifyDecision(Rv, H, k, {f: v.to(torch.int64) for f, v in t.items()}, page, num_pages, load_ids,
                          (bb_f, hh_f), counters)


# ====================================================================================================================
# the device operations
# ====================================================================================================================
class CpuDeviceSim:
    """S-C's device operations on the CPU: the tests' stand-in for the GPU (module docstring)."""

    def __init__(self, svc: "ResidencySC"):
        self.svc = svc
        self.check_t3 = True                              # False only in a negative control (the host's check remains)
        self.waited = 0
        self.gathered = 0

    def wait(self, l: int, dec: VerifyDecision) -> None:
        """The device waits until every in-flight block its verify rows read has landed: here, the transfer lands
        (the transport's force), in K3X order, once per transfer (v1's waits)."""
        svc = self.svc
        bb, hh = dec.bh
        cols = zip(bb.tolist(), hh.tolist(), dec.t["blk"].tolist(), dec.t["infl"].tolist(), dec.t["infl_slot"].tolist(),
                   dec.t["infl_gen"].tolist())
        seen = set()
        for (b, h, t, infl, islot, igen) in cols:
            if not infl:
                continue
            tr = svc._in_flight(l, b, h, t)
            if tr is None or tr.slot != islot or tr.gen != igen:
                raise OwnershipViolation("S-C device: the verify waits for block %d (layer %d b %d h %d) at slot %d gen "
                                         "%d, which is not the owner's transfer in flight" % (t, l, b, h, islot, igen))
            if tr.seq in seen:
                continue
            seen.add(tr.seq)
            if not svc.tp.done(tr):
                svc.tp.force(tr)
            self.waited += 1

    def gather(self, l: int, dec: VerifyDecision) -> None:
        """The urgent gather on the main stream: every (h, b, slot) with a block id in the load ids gets the block's
        host K / V rows and its cis rows (bias), after the block's host write-back (T4); a slot whose last reader is
        pending is refused (T3)."""
        svc = self.svc
        g = svc.gather                                    # the host rows and the store (TorchGather's tensors)
        bs = svc.bs
        for h, per_b in enumerate(dec.load_ids.tolist()):
            for b, per_m in enumerate(per_b):
                for m, t in enumerate(per_m):
                    if t < 0:
                        continue
                    s = svc.stream(l, b, h).slots[m]
                    if self.check_t3 and not ev_done(s.last_reader):
                        raise OwnershipViolation("S-C device: the urgent gather of block %d writes slot %d (layer %d b %d "
                                                 "h %d) while the slot's last reader is pending (T3)" % (t, m, l, b, h))
                    hev = svc._hist_ev_for(b, t)
                    if not ev_done(hev):
                        svc.host_wait(hev)                # the main stream's wait on the host write-back (T4)
                    r0, s0 = t * bs, m * bs
                    g.store_k[l][b, s0:s0 + bs, h].copy_(g.host_k[l][b, r0:r0 + bs, h])
                    g.store_v[l][b, s0:s0 + bs, h].copy_(g.host_v[l][b, r0:r0 + bs, h])
                    if svc.store_b is not None and svc.cis is not None:
                        svc.store_b[l][b, s0:s0 + bs, h].copy_(svc.cis[l][b, r0:r0 + bs, h])
                    self.gathered += 1


# ====================================================================================================================
# the EXACT device wait (note 12f): a generation board the transports publish into, a wait on exactly the needed slots
# ====================================================================================================================
class GenBoard:
    """The generation board (L, B, H, C + X) int64 on the device: per slot, the generation of the last transfer whose
    rows LANDED there. The transport writes it on the stream that landed the rows, AFTER them (gilfree: the scatter
    stream after the release; CPU tests: PublishingTransport). 0 = nothing landed yet (a take makes gen >= 1). The
    generations of a slot only grow, and a slot in flight is never taken again (T3), so "board >= gen" means "the
    awaited transfer landed"."""

    def __init__(self, L: int, B: int, H: int, npos: int, device):
        self.L, self.B, self.H, self.npos = int(L), int(B), int(H), int(npos)
        self.t = torch.zeros((self.L, self.B, self.H, self.npos), dtype=torch.int64, device=device)

    def flat(self, l: int, b: int, h: int, slot: int) -> int:
        return ((l * self.B + b) * self.H + h) * self.npos + slot

    def unflat(self, f: int) -> Tuple[int, int, int, int]:
        f, slot = divmod(f, self.npos)
        f, h = divmod(f, self.H)
        l, b = divmod(f, self.B)
        return l, b, h, slot


def wait_list(dec: VerifyDecision, l: int, board: GenBoard) -> Tuple[torch.Tensor, torch.Tensor]:
    """The exact wait's list, on the device, from one layer's verify decision: per entry the board index of its
    in-flight slot (-1 for an entry that waits for nothing) and the generation it waits for. Fixed length (the
    decision's N): capturable."""
    bb, hh = dec.bh
    f = ((l * board.B + bb) * board.H + hh) * board.npos + dec.t["infl_slot"].clamp(min=0)
    return torch.where(dec.t["infl"] != 0, f, torch.full_like(f, -1)), dec.t["infl_gen"]


class PublishingTransport:
    """The CPU stand-in of the publication, around a K3X transport that cannot publish itself (the tests'
    AdversarialTransport): the K3X interface passes through unchanged (the same deliveries, the same pending list),
    and every transfer the inner transport has LANDED (observed after its poll / force) is published into the board:
    board[slot] = gen. progress_slot() is the stand-in of TIME for the CPU spin: the transfer in flight into that slot
    lands now. publish=False is a NEGATIVE CONTROL (a transport that never publishes)."""

    def __init__(self, inner, publish: bool = True):
        self.inner = inner
        self.publish = bool(publish)
        self.board: GenBoard = None
        self._open: Dict[int, object] = {}                # seq -> a transfer issued and not seen landed

    @property
    def name(self):
        return self.inner.name

    @property
    def pending(self):
        return self.inner.pending

    def attach_board(self, board: GenBoard) -> None:
        self.board = board

    def _publish_landed(self) -> None:
        for seq, tr in list(self._open.items()):
            if self.inner.done(tr):
                del self._open[seq]
                if self.publish and self.board is not None:
                    self.board.t.view(-1)[self.board.flat(tr.layer, tr.b, tr.h, tr.slot)] = tr.gen

    def issue(self, layer: int, transfers) -> None:
        self.inner.issue(layer, transfers)
        for tr in transfers:
            self._open[tr.seq] = tr
        self._publish_landed()

    def poll(self) -> None:
        self.inner.poll()
        self._publish_landed()

    def done(self, tr) -> bool:
        return self.inner.done(tr)

    def force(self, tr) -> None:
        self.inner.force(tr)
        self._publish_landed()

    def progress_slot(self, l: int, b: int, h: int, slot: int) -> bool:
        for tr in list(self._open.values()):
            if (tr.layer, tr.b, tr.h, tr.slot) == (l, b, h, slot):
                self.force(tr)
                return True
        return False


class ExactWaitSim(CpuDeviceSim):
    """The CPU stand-in of the EXACT device wait. It reads only DEVICE data: the decision's wait list (wait_list) and
    the generation board; never the host's transfer records. Per entry, in K3X order, it spins while the board is
    below the awaited generation; a spin round is the stand-in of time (the transport's progress_slot), at most
    max_spin rounds. An entry still short (or past) after the spin sets the device error flag `err` (the GPU kernel's
    bounded spin does the same); S-C watches it (watch_flag), so such a tick is refused at its end_tick, never a hang.
    stale_ok is a NEGATIVE CONTROL: the spin accepts the slot's previous generation (gen - 1)."""

    def __init__(self, svc: "ResidencySC", max_spin: int = 4, stale_ok: bool = False):
        super().__init__(svc)
        if not callable(getattr(svc.tp, "progress_slot", None)):
            raise ValueError("ExactWaitSim needs a publishing transport (progress_slot, attach_board)")
        self.max_spin, self.stale_ok = int(max_spin), bool(stale_ok)
        self.err = torch.zeros((), dtype=torch.bool, device=svc.device)
        self.spins = 0

    def wait(self, l: int, dec: VerifyDecision) -> None:
        board = self.svc.board
        idx, gen = wait_list(dec, l, board)
        flat = board.t.view(-1)
        for f, g in zip(idx.tolist(), gen.tolist()):
            if f < 0:
                continue
            want = g - 1 if self.stale_ok else g
            k = 0
            while flat[f] < want and k < self.max_spin:
                self.svc.tp.progress_slot(*board.unflat(f))
                k += 1
            got = flat[f]
            if got < want or (not self.stale_ok and got != g):
                self.err |= True
            self.spins += k
            self.waited += 1


_WAIT_KERNEL = None


def _wait_kernel():
    """The exact wait's Triton kernel, built on first use. SOURCE ONLY, NOT RUN YET (GPU gate FG1). ONE program (one
    SM: the scatter stream's kernels keep the others) walks the wait list in blocks; per block it spins with VOLATILE
    loads of the board until every listed entry reached its generation, at most max_spin rounds; then any entry short
    or past raises err (int32, atomic max). The board write that ends a spin is enqueued after the scatter on the
    scatter stream, so the slot's rows are complete when the main stream's next kernel reads them."""
    global _WAIT_KERNEL
    if _WAIT_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(board_ptr, idx_ptr, gen_ptr, n, err_ptr, max_spin, BLOCK: tl.constexpr):
            for start in range(0, n, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                inb = offs < n
                idx = tl.load(idx_ptr + offs, mask=inb, other=-1)
                want = tl.load(gen_ptr + offs, mask=inb, other=0)
                live = idx >= 0
                safe = tl.where(live, idx, 0)
                got = tl.load(board_ptr + safe, mask=live, other=0, volatile=True)
                k = 0
                while (tl.max((live & (got < want)).to(tl.int32), axis=0) > 0) & (k < max_spin):
                    got = tl.load(board_ptr + safe, mask=live, other=0, volatile=True)
                    k += 1
                tl.atomic_max(err_ptr, tl.max((live & (got != want)).to(tl.int32), axis=0))
        _WAIT_KERNEL = kernel
    return _WAIT_KERNEL


class CudaDeviceOps:
    """S-C's device operations on the GPU (module docstring). SOURCE ONLY, NOT RUN YET (GPU gate FG1).
    wait_mode "superset": the main stream waits on the completion event of every transfer in flight at the layer.
    wait_mode "exact" (note 12f): one Triton program on the main stream spins on the generation board for exactly the
    in-flight slots that the layer's verify rows read (wait_list); no host loop, no host wait; err (int32, on the
    device) reports a spin that ran out; S-C watches it. FG1 compares the two."""

    def __init__(self, svc: "ResidencySC", host_k, host_v, store_k, store_v, store_b, cis, block: int = BLOCK,
                 wait_mode: str = "superset", max_spin: int = 1 << 22):
        from .flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
        from .flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
        if wait_mode not in ("superset", "exact"):
            raise ValueError("wait_mode %r" % (wait_mode,))
        self.svc = svc
        self.h2d, self.h2d_bias = flash_h2d_from_mask, flash_h2d_from_mask_bias
        self.host_k, self.host_v, self.store_k, self.store_v = host_k, host_v, store_k, store_v
        self.store_b, self.cis = store_b, cis
        self.bs = block
        self.wait_mode, self.max_spin = wait_mode, int(max_spin)
        self.err = torch.zeros((), dtype=torch.int32, device=svc.device)
        self.events_waited = 0
        self.submission_waits = 0

    def wait(self, l: int, dec: VerifyDecision) -> None:
        if self.wait_mode == "exact":
            idx, gen = wait_list(dec, l, self.svc.board)
            _wait_kernel()[(1,)](self.svc.board.t, idx, gen, idx.shape[0], self.err, self.max_spin, BLOCK=1024)
            return
        main = torch.cuda.current_stream()
        seen = set()
        for tr in list(self.svc.inflight.values()):
            if tr.layer != l or tr.seq in self.svc._landed:
                continue
            if tr.ev is None and isinstance(tr.payload, threading.Event):   # a CPU pack not submitted yet (gilfree)
                tr.payload.wait()
                self.submission_waits += 1
            if tr.ev is None:
                raise OwnershipViolation("S-C GPU wait: transfer seq %d (layer %d) has no completion event" % (tr.seq, l))
            if id(tr.ev) in seen:
                continue
            seen.add(id(tr.ev))
            main.wait_event(tr.ev)
            self.events_waited += 1

    def gather(self, l: int, dec: VerifyDecision) -> None:
        main = torch.cuda.current_stream()
        for ev in self.svc._hist_unwaited:
            main.wait_event(ev)
        self.svc._hist_unwaited = []
        self.h2d_bias(self.store_k[l], self.host_k[l], self.store_b[l], self.cis[l], dec.load_ids, self.bs)
        self.h2d(self.store_v[l], self.host_v[l], dec.load_ids, self.bs)


# ====================================================================================================================
# the service
# ====================================================================================================================
class ResidencySC(ResidencySB):
    """Residency with the verify and the draft rows decided on the device (module docstring). Extra arguments (beyond
    S-B's device, nblk, sb_mut): sc_mut (negative controls: device_verify's, and no_wait: the device does not wait for
    the in-flight blocks), devops (the device operations; on the CPU the stand-in by default), mirror_drop (NEGATIVE
    CONTROL, tests only: the journal kinds, or "owner", whose changes are NOT sent to the device mirror), wait_mode
    (the CPU stand-in's wait: "probe" = CpuDeviceSim, the host's transfer records; "exact" = ExactWaitSim, the
    generation board, which needs a publishing transport; on the GPU CudaDeviceOps' own wait_mode decides).
    The generation board (self.board, note 12f) is attached to every transport that publishes (attach_board)."""

    def __init__(self, *args, sc_mut=None, devops=None, mirror_drop=None, wait_mode: str = "probe", **kw):
        kw["defer_admit"] = True
        super().__init__(*args, **kw)
        self.board = GenBoard(self.L, self.B, self.H, self.n_hist, self.device)
        if callable(getattr(self.tp, "attach_board", None)):
            self.tp.attach_board(self.board)
        self.mirrors = None                               # S-B's per-layer host-refreshed mirrors are not used
        self.dmirror = DeviceMirror(self.L, self.B, self.H, self.nblk, self.n_hist, self.device)
        self.mirror_drop = set(mirror_drop or ())
        self._dirty: Set[Tuple[int, int, int, int]] = set()          # (l, b, h, slot) changed since the last sync
        for l, pool in enumerate(self.pools):
            for (b, h), st in pool.streams.items():
                st.log = _Journal((l, b, h), self._journal, st.log)
        self.mirror_uploads = 0                           # host-to-device copies of the mirror delta
        self.mirror_slots_uploaded = 0                    # slot records sent in them
        self.row_uploads = 0                              # host-to-device copies of the tick's row inputs
        self._rows = None                                 # the tick's row inputs on the device
        self.sc_mut = dict(sc_mut or {})
        if devops is None:
            if self.device.type != "cpu":
                raise ValueError("S-C on %s needs its device operations (CudaDeviceOps)" % self.device)
            if wait_mode not in ("probe", "exact"):
                raise ValueError("wait_mode %r" % (wait_mode,))
            devops = ExactWaitSim(self) if wait_mode == "exact" else CpuDeviceSim(self)
        self.dev = devops
        self._tick: List[Dict] = []
        self._hist_unwaited: List = []                    # host write-backs the main stream did not wait on yet
        self._flags: Dict[str, torch.Tensor] = {}         # watched device flags (watch_flag)
        self.flag_values: Dict[str, int] = {}             # their values at the last end_tick read

    def watch_flag(self, name: str, flag: torch.Tensor) -> None:
        """Watch a device flag (a 0-dim bool or integer tensor on the service's device; e.g. the twin's sticky
        `overflow`, r2_positional_twin.py:152) with S-C's ONE device read per tick: every end_tick reads it together
        with the decisions, also on a tick without rows, and a nonzero value REFUSES THAT TICK (CapacityRefusal) before
        any of its decisions is applied. So a raised flag voids one tick, not every tick since the last decision read
        (Q-T5), and it costs no extra sync. The tensor is read by reference: an in-place update (|=) is seen."""
        if not torch.is_tensor(flag) or flag.dim() != 0 or flag.device.type != self.device.type:
            raise ValueError("watch_flag: a 0-dim tensor on %s expected" % self.device)
        if name in self._flags:
            raise ValueError("watch_flag: %r is already watched" % name)
        self._flags[name] = flag

    # ------------------------------------------------------------------------------------------------ the mirror delta
    def _journal(self, key: Tuple[int, int, int], item) -> None:
        """Every stream transition, as the vendored ownership code logs it: (event, block, slot, gen, detail)."""
        if item[0] == "touch" or item[0] in self.mirror_drop:
            return
        self._dirty.add(key + (item[2],))

    def _mark_batch_slot(self, b: int) -> None:
        """An owner change of batch slot b: its transfers in flight change their "current owner" flag."""
        if "owner" in self.mirror_drop:
            return
        for tr in self.inflight.values():
            if tr.b == b:
                self._dirty.add((tr.layer, tr.b, tr.h, tr.slot))

    def begin_request(self, b: int, uid, hist_blocks=None) -> None:
        super().begin_request(b, uid, hist_blocks)
        self._mark_batch_slot(b)

    def end_request(self, uid) -> None:
        b = self.uid_b[uid]
        super().end_request(uid)
        self._mark_batch_slot(b)

    def _h2d(self, host: torch.Tensor) -> torch.Tensor:
        """Every host-to-device copy of S-C goes through here (per tick: the mirror delta and the row inputs). On the
        GPU from pinned memory, asynchronous on the current stream (no host sync)."""
        if self.device.type == "cuda":
            return host.pin_memory().to(self.device, non_blocking=True)
        return host.to(self.device)

    def sync_mirror(self) -> bool:
        """The ONE batched host-to-device delta of the device mirror: the record of every slot that changed since the
        last sync (state code, block, generation, current-owner flag). Called by the first resolve of a tick (and by
        tests); returns whether a delta was sent."""
        if not self._dirty:
            return False
        B, H, NPOS = self.B, self.H, self.n_hist
        recs = []
        for (l, b, h, i) in sorted(self._dirty):
            s = self.pools[l].stream(b, h).slots[i]
            cur = 0
            if s.state == INFLIGHT:
                tr = self._in_flight(l, b, h, s.block)
                cur = 1 if tr is not None and tr.slot == i else 0
            recs.append((((l * B + b) * H + h) * NPOS + i, CODE[s.state], s.block, s.gen, cur))
        self._dirty = set()
        self.dmirror.apply(self._h2d(torch.tensor(recs, dtype=torch.int64).t().contiguous()))
        self.mirror_uploads += 1
        self.mirror_slots_uploaded += len(recs)
        return True

    # ------------------------------------------------------------------------------------------------ the tick's rows
    @staticmethod
    def _row_sig(rows: Sequence[Row]):
        return tuple((r.b, r.kind, r.key, r.pos, r.hist_bound, tuple(r.tail)) for r in rows)

    def _tick_rows(self, rows: Sequence[Row]) -> Dict:
        """The tick's row inputs on the device, in ONE host-to-device copy (every layer of a tick resolves the same
        rows; only the selection differs)."""
        for i, r in enumerate(rows):
            if r.hist_bound > self.nblk:
                raise ValueError("row %d: hist_bound %d > nblk %d (the mirror names blocks < nblk only)"
                                 % (i, r.hist_bound, self.nblk))
            if len(r.tail) > N_TAIL:
                raise ValueError("at most %d tail pages per row" % N_TAIL)
            for (_, s) in r.tail:
                if not self.tail0 <= s < self.tail0 + N_TAIL:
                    raise ValueError("tail slot %d outside [%d, %d)" % (s, self.tail0, self.tail0 + N_TAIL))
        R = len(rows)
        vrows = [i for i, r in enumerate(rows) if r.kind == "v"]
        drows = [i for i, r in enumerate(rows) if r.kind == "d"]
        tails_flat: List[int] = []
        for r in rows:
            tails_flat += [s for (_, s) in r.tail] + [-1] * (N_TAIL - len(r.tail))
        flat = ([r.b for r in rows] + [r.hist_bound for r in rows] + [len(r.tail) for r in rows] + tails_flat
                + [r.pos % self.bs + 1 for r in rows] + vrows + drows)
        d = self._h2d(torch.tensor(flat, dtype=torch.int64))
        self.row_uploads += 1
        rb, hb, nt = d[:R], d[R:2 * R], d[2 * R:3 * R]
        at = 3 * R + R * N_TAIL
        tails = d[3 * R:at].view(R, N_TAIL)
        last_len = d[at:at + R]
        vix = d[at + R:at + R + len(vrows)]
        dix = d[at + R + len(vrows):at + R + len(vrows) + len(drows)]

        def part(ix):
            return rb[ix], hb[ix], tails[ix], nt[ix], ix
        return dict(sig=self._row_sig(rows), vrows=vrows, drows=drows, v=part(vix) if vrows else None,
                    d=part(dix) if drows else None, last_len=last_len.to(torch.int32), seq_idx=rb.to(torch.int32))

    # ------------------------------------------------------------------------------------------------ one layer
    def resolve(self, l: int, rows: Sequence[Row], sel: torch.Tensor) -> Plan:
        H, R = self.H, len(rows)
        if any(st["l"] == l for st in self._tick):
            raise ValueError("S-C: layer %d resolved twice in one tick (its decisions are applied at end_tick)" % l)
        if sel.device.type != self.device.type:
            raise ValueError("S-C: the selection must be on the service's device %s (got %s)" % (self.device, sel.device))
        self._check_rows(l, rows, sel)
        if not self._tick:                                            # the tick's first layer: the per-tick copies
            self.sync_mirror()
            self._rows = self._tick_rows(rows)
        elif self._dirty:
            raise OwnershipViolation("S-C: the host state changed in the middle of a tick (the device mirror is stale)")
        elif self._row_sig(rows) != self._rows["sig"]:
            raise ValueError("S-C: every layer of a tick resolves the same rows (layer %d differs)" % l)
        tk = self._rows
        plan = self._new_plan(rows)
        dv = self.device
        P = sel.shape[2] + N_TAIL
        page = torch.full((R, H, P), -1, dtype=torch.int64, device=dv)
        npg = torch.zeros((R, H), dtype=torch.int64, device=dv)
        m = self.dmirror.layer(l)
        step = dict(l=l, vrows=[rows[i] for i in tk["vrows"]], vidx=tk["vrows"], vdec=None,
                    drows=[rows[i] for i in tk["drows"]], ddec=None)
        if tk["v"] is not None:                                       # the verify rows: on the device, no host read
            rb, hb, tl, nt, ix = tk["v"]
            vdec = device_verify(m, rb, hb, tl, nt, sel[:, ix], self.C, self.X, self.sc_mut)
            if not self.sc_mut.get("no_wait"):                        # NEGATIVE CONTROL: read before the event
                self.dev.wait(l, vdec)
            self.dev.gather(l, vdec)
            page[ix] = vdec.page
            npg[ix] = vdec.num_pages
            step["vdec"] = vdec
        if tk["d"] is not None:                                       # the draft rows: S-B, on the mirror after verify
            rb, hb, tl, nt, ix = tk["d"]
            ddec = device_draft(m, rb, hb, tl, nt, sel[:, ix], self.C, self.X, self.R_ver, self.sb_mut)
            page[ix] = ddec.page
            npg[ix] = ddec.num_pages
            step["ddec"] = ddec
        self._tick.append(step)
        plan.page_table = page.to(torch.int32)
        plan.num_pages = npg.to(torch.int32)
        plan.last_len = tk["last_len"]
        plan.seq_idx = tk["seq_idx"]
        return plan

    def end_tick(self) -> None:
        steps, self._tick = self._tick, []
        self._rows = None
        names = list(self._flags)
        parts = [self._flags[n].to(torch.int64).reshape(1) for n in names]   # the watched flags first
        for st in steps:
            for dec, fields in ((st["vdec"], FIELDS_V), (st["ddec"], FIELDS)):
                if dec is not None:
                    parts.append(torch.stack([dec.t[f] for f in fields]).reshape(-1))
                    parts.append(dec.counters)
        if parts:
            flat = self._d2h(torch.cat(parts)).tolist()               # ONE read per tick (flags and decisions)
            self.flag_values = dict(zip(names, flat[:len(names)]))
            raised = sorted(n for n, v in self.flag_values.items() if v)
            if raised:
                self._layer_ev = {}
                raise CapacityRefusal("S-C: the watched device flag(s) %s are set at this tick: the tick is void (none "
                                      "of its decisions is applied)" % raised)
            at = len(names)
            parsed = []
            for st in steps:
                v = d = None
                if st["vdec"] is not None:
                    dec = st["vdec"]
                    N = dec.Rv * dec.H * dec.k
                    cols = {f: flat[at + j * N: at + (j + 1) * N] for j, f in enumerate(FIELDS_V)}
                    at += len(FIELDS_V) * N
                    v = (cols, flat[at])                              # the overflow count
                    at += 1
                if st["ddec"] is not None:
                    dec = st["ddec"]
                    N = dec.Rd * dec.H * dec.k
                    cols = {f: flat[at + j * N: at + (j + 1) * N] for j, f in enumerate(FIELDS)}
                    at += len(FIELDS) * N
                    d = (cols,) + tuple(flat[at:at + 3])
                    at += 3
                parsed.append((st, v, d))
            over = [(st["l"], v[1]) for (st, v, _) in parsed if v is not None and v[1] > 0]
            if over:
                self._layer_ev = {}
                raise CapacityRefusal("S-C: the verifier's urgent misses found no FREE slot within K5 (layer, blocks "
                                      "without a slot: %s; C %d, X %d, R_ver %d): the tick's outputs are invalid"
                                      % (over, self.C, self.X, self.R_ver))
            for (st, v, d) in parsed:
                if v is not None:
                    self._apply_verify(st["l"], st["vrows"], st["vidx"], st["vdec"], v[0])
                if d is not None:
                    self._apply(st["l"], st["drows"], st["ddec"], d[0], d[1], d[2], d[3])
        self._layer_ev = {}
        Residency.end_tick(self)                                      # the draft transfers, layer by layer
        self.metrics.sample_memory()

    def _apply_verify(self, l: int, vrow_objs: List[Row], vidx: List[int], dec: VerifyDecision, c) -> None:
        """The host applies one layer's verify decisions in v1's order (_verify_part under defer_admit), checking each
        against the host state."""
        H, k = dec.H, dec.k
        pool = self.pools[l]
        now = self.clock()
        looks, entries = [], {}
        for j, r in enumerate(vrow_objs):
            for h in range(H):
                st = pool.stream(r.b, h)
                ready, infl, new, takes, ent = [], [], [], [], []
                for u in range(k):
                    o = (j * H + h) * k + u
                    if not c["valid"][o]:
                        continue
                    t = c["blk"][o]
                    if c["protect"][o]:
                        st.protect(t, r.key, VERIFY)
                    if c["ready"][o]:
                        if st.resident(t) != c["slot"][o]:
                            raise OwnershipViolation("S-C: the device read block %d at slot %d, the host holds it at %s "
                                                     "(layer %d b %d h %d)" % (t, c["slot"][o], st.resident(t), l, r.b, h))
                        ready.append(t)
                        ent.append((t, c["slot"][o]))
                    elif c["infl"][o] or c["dup"][o]:
                        tr = self._in_flight(l, r.b, h, t)
                        want = c["infl_slot"][o] if c["infl"][o] else c["rslot"][o]
                        if tr is None or tr.slot != want or (c["infl"][o] and tr.gen != c["infl_gen"][o]):
                            raise OwnershipViolation("S-C: the device read block %d in flight at slot %d (layer %d b %d h "
                                                     "%d), the host has %s" % (t, want, l, r.b, h,
                                                                             None if tr is None else (tr.slot, tr.gen)))
                        if not self.mut.drop_demand_merge:
                            tr.demand.add(VERIFY)
                        infl.append(t)
                        ent.append((t, tr.slot))
                    elif c["granted"][o]:
                        new.append(t)
                        takes.append((t, c["take_slot"][o]))
                    else:
                        raise OwnershipViolation("S-C: block %d of verify row %d (layer %d b %d h %d) has no slot and the "
                                                 "device did not flag an overflow" % (t, vidx[j], l, r.b, h))
                for (t, at) in takes:
                    tr = self._take(l, r.b, h, t, r.key, VERIFY, at=at)
                    tr.delivered = True                               # the device's urgent gather landed it
                    self._landed.add(tr.seq)
                    ent.append((t, at))
                looks.append(dict(row=vidx[j], h=h, key=r.key, ready=ready, in_flight=infl, new=new, t=now))
                entries[(j, h)] = sorted(ent)
        for lk in looks:
            self.metrics.look(l, lk)
        ev = self._layer_ev.get(l)
        for (j, h), ent in sorted(entries.items()):
            r = vrow_objs[j]
            st = pool.stream(r.b, h)
            for (t, idx) in ent:
                s = st.slots[idx]
                if s.first_reader is None:
                    s.first_reader = ev
                s.last_reader = ev
                self.metrics.read(l, r.b, h, idx, s.gen, True)
        for lk in looks:
            self.metrics.executed(l, lk)
        if not self.mut.release_only_at_decision:
            for key in {lk["key"] for lk in looks}:
                self._queue(key, l)
