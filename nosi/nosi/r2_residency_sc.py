"""R2 step 7, piece 5, stage S-C: DEVICE-SIDE residency for the VERIFY rows (docs/R2_STEP7_FETCH_RETENTION.md, 12c).

S-B decides the draft rows on the device, but its verify rows stay host-side (v1): a verify tick reads the verify
selection back at every layer. S-C moves the verify rows to the device too. resolve() reads NO device value on the
host; end_tick() makes ONE batched device-to-host read per tick, for the verify and the draft decisions together.

THE DEVICE MIRROR (DeviceMirror, note Q5). The ownership state the device decides on lives ON THE DEVICE as a per-slot
table of every layer (state code, block, generation, "the current owner's transfer in flight", "a resident block
protected by a live cohort", the LRU recency stamp). The host keeps no copy of it and never scans it: every transition
of a stream (K3X Stream.log: take, release, evict_po, acquire, admit, admit_vres, drop_unowned, drop_vres_unprotected,
free, touch) marks its slot dirty as it happens (touch / admit / acquire also give it a new recency stamp), so does a
protection change of a resident block (the stream's protection map reports it: _ProtDict), and so does an owner change
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
  POLICY E   a first miss that K5 cannot grant takes the stream's next EVICTION CANDIDATE (note 12h): an LRU slot
             whose block is unprotected and selected by no row of the layer, least recent first (the mirror's
             protection flags and recency stamps), as v1 does; then (note 12j) the slot of the current owner's
             draft prefetch IN FLIGHT whose block is unprotected and selected by no row, oldest issue first (the
             take stamps): that prefetch is superseded, the layer's wait covers its write (superset: every
             transfer in flight at the layer; exact: wait_list) before the urgent gather; the host applies the
             eviction at end_tick and checks it is v1's choice.
  OVERFLOW   a first miss without a slot from K5 or Policy E. The device sets the OVERFLOW FLAG and points the page at
             slot 0 (in bounds); end_tick raises CapacityRefusal before any change of the host state. The tick's
             outputs are invalid and the run stops: every LRU entry and in-flight draft prefetch is protected or read
             by the layer and C + X or R_ver is too small (a configuration result, as v1's own CapacityRefusal); the
             message carries the DECISION per layer and per refused stream (_k5_report: its first misses = free base
             slots + shared extras + Policy E + no slot; the layer's extras at the tick start and how many first
             misses needed one), then the capacity report (the host state at the tick's start) of those streams.
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
                 last reader is pending (T3) and waits for the block's host write-back first (T4), as v1's gather;
                 before it, it host-waits every write-back registered since its last gather (history_written ->
                 _hist_unwaited: the stand-in of the GPU main stream's wait).
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
import time
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

import torch

try:                                                      # triton is needed only to launch the exact wait's kernel on
    import triton                                         # the GPU: a CPU test without triton imports this module
    import triton.language as tl
except ImportError:                                       # pragma: no cover - environment dependent
    triton = tl = None

from .kstep.backend import ev_done
from .kstep.errors import CapacityRefusal, OwnershipViolation
from .kstep.ownership import FREE, INFLIGHT, LRU, PO, RELEASING, VRES
from . import r2_xpool as XP
from .r2_residency import BLOCK, N_TAIL, VERIFY, Plan, Residency, Row
from .r2_residency_sb import FIELDS, DraftMirror, ResidencySB, device_draft, rank_in_group

FIELDS_V = ("blk", "valid", "ready", "slot", "infl", "infl_slot", "infl_gen", "dup", "granted", "take_slot", "rslot",
            "protect", "evict", "evict_blk", "evict_infl", "ev_gen")
C_FREE, C_INFLIGHT, C_LRU, C_PO, C_VRES, C_RELEASING = range(6)
CODE = {FREE: C_FREE, INFLIGHT: C_INFLIGHT, LRU: C_LRU, PO: C_PO, VRES: C_VRES, RELEASING: C_RELEASING}
MIRROR_FIELDS = ("slot_of", "inflight", "infl_slot", "infl_gen", "free", "usage", "releasing", "lru_rank", "prot",
                 "infl_unprot", "infl_rank")
JOURNAL_KINDS = ("take", "release", "evict_po", "acquire", "admit", "admit_vres", "drop_unowned",
                 "drop_vres_unprotected", "free", "touch")     # touch / admit / acquire: the LRU recency (Policy E)
RECENCY_KINDS = ("touch", "admit", "acquire")             # the events that put a block at the LRU's most recent end
STAMP_KINDS = RECENCY_KINDS + ("take",)                   # + take: the issue order of the in-flight prefetches (12j)
BIG_STAMP = 1 << 62
HALF_STAMP = 1 << 61                                      # Policy E's in-flight candidates sort after every LRU one


class _ProtDict(dict):
    """A stream's protection map (K3X Stream.prot: block -> cohort keys) that reports every block whose protection
    may change (a new entry, a removed one), so its resident slot is sent with the next mirror delta (Policy E reads
    "protected" on the device). Adding a key to an existing entry changes nothing ("protected" stays true)."""

    def __init__(self, key: Tuple[int, int, int], sink, items=()):
        super().__init__(items)
        self.key, self.sink = key, sink

    def setdefault(self, block, default=None):
        new = block not in self
        out = super().setdefault(block, default)
        if new:
            self.sink(self.key, block)
        return out

    def __setitem__(self, block, value) -> None:
        super().__setitem__(block, value)
        self.sink(self.key, block)

    def __delitem__(self, block) -> None:
        super().__delitem__(block)
        self.sink(self.key, block)

    def pop(self, block, *default):
        out = super().pop(block, *default)
        self.sink(self.key, block)
        return out

    def clear(self) -> None:
        blocks = list(self.keys())
        super().clear()
        for blk in blocks:
            self.sink(self.key, blk)


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
    state code, the block, the generation, the flag "the current owner's transfer in flight", the flag "a resident
    block protected by a live cohort" and the LRU recency stamp (the larger, the more recent; Policy E). apply()
    writes the records of one delta; layer() derives one layer's mirror with device ops only."""

    def __init__(self, L: int, B: int, H: int, nblk: int, npos: int, device):
        shape = (L, B, H, npos)
        self.nblk = int(nblk)
        self.code = torch.zeros(shape, dtype=torch.int64, device=device)          # every slot FREE
        self.block = torch.full(shape, -1, dtype=torch.int64, device=device)
        self.gen = torch.zeros(shape, dtype=torch.int64, device=device)
        self.cur = torch.zeros(shape, dtype=torch.int64, device=device)
        self.prot = torch.zeros(shape, dtype=torch.int64, device=device)
        self.stamp = torch.zeros(shape, dtype=torch.int64, device=device)

    def apply(self, recs: torch.Tensor) -> None:
        """recs (7, n) int64 ON THE DEVICE: flat slot index, state code, block, generation, current-owner flag,
        protected flag, recency stamp; the slot indices are distinct."""
        f = recs[0]
        for t, row in ((self.code, 1), (self.block, 2), (self.gen, 3), (self.cur, 4), (self.prot, 5), (self.stamp, 6)):
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
        # Policy E: the LRU slots by recency (stamp) and the protection of the resident slots
        lru = code == C_LRU
        lru_stamp = torch.where(lru, self.stamp[l], torch.full_like(code, BIG_STAMP))
        order = torch.sort(lru_stamp.view(B * H, NPOS), dim=-1).indices
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(NPOS, device=dev).view(1, NPOS).expand(B * H, NPOS).contiguous())
        resident = (code == C_LRU) | (code == C_PO) | (code == C_VRES)
        # Policy E's in-flight candidates (12j): the current owner's transfers in flight by issue order (take stamp)
        fly_stamp = torch.where(fly, self.stamp[l], torch.full_like(code, BIG_STAMP))
        forder = torch.sort(fly_stamp.view(B * H, NPOS), dim=-1).indices
        frank = torch.empty_like(forder)
        frank.scatter_(1, forder, torch.arange(NPOS, device=dev).view(1, NPOS).expand(B * H, NPOS).contiguous())
        return LayerMirror(slot_of=slot_of[:dummy].view(shp), inflight=(infl_slot[:dummy] >= 0).view(shp),
                           infl_slot=infl_slot[:dummy].view(shp), infl_gen=infl_gen[:dummy].view(shp),
                           free=code == C_FREE, usage=(code != C_FREE).to(torch.int64).sum(-1),
                           releasing=code == C_RELEASING,
                           lru_rank=torch.where(lru, rank.view(B, H, NPOS), torch.full_like(code, -1)),
                           prot=resident & (self.prot[l] != 0), lru_stamp=lru_stamp, slot_block=blk.clone(),
                           infl_unprot=fly & (self.prot[l] == 0),
                           infl_rank=torch.where(fly, frank.view(B, H, NPOS), torch.full_like(code, -1)),
                           fly=fly, fly_stamp=fly_stamp, slot_gen=gen.clone())


class VerifyMirror(DraftMirror):
    """S-B's mirror of one layer plus, per (b, h, block), the slot and generation of the current owner's transfer in
    flight (the device's wait reads them), and the RELEASING slots (read only by a negative control)."""

    def __init__(self, B: int, H: int, nblk: int, npos: int, device):
        super().__init__(B, H, nblk, npos, device)
        self.infl_slot = torch.full((B, H, nblk), -1, dtype=torch.int64, device=device)
        self.infl_gen = torch.zeros((B, H, nblk), dtype=torch.int64, device=device)
        self.releasing = torch.zeros((B, H, npos), dtype=torch.bool, device=device)
        self.lru_rank = torch.full((B, H, npos), -1, dtype=torch.int64, device=device)
        self.prot = torch.zeros((B, H, npos), dtype=torch.bool, device=device)
        self.infl_unprot = torch.zeros((B, H, npos), dtype=torch.bool, device=device)
        self.infl_rank = torch.full((B, H, npos), -1, dtype=torch.int64, device=device)

    def refresh(self, svc: Residency, l: int) -> None:
        super().refresh(svc, l)
        B, H = svc.B, svc.H
        nblk, npos = self.slot_of.shape[-1], self.free.shape[-1]
        infl_slot = torch.full((B, H, nblk), -1, dtype=torch.int64)
        infl_gen = torch.zeros((B, H, nblk), dtype=torch.int64)
        releasing = torch.zeros((B, H, npos), dtype=torch.bool)
        lru_rank = torch.full((B, H, npos), -1, dtype=torch.int64)
        prot = torch.zeros((B, H, npos), dtype=torch.bool)
        infl_unprot = torch.zeros((B, H, npos), dtype=torch.bool)
        infl_rank = torch.full((B, H, npos), -1, dtype=torch.int64)
        flying: Dict[Tuple[int, int], List] = {}
        for (ll, b, h, blk), tr in svc.pending.items():
            if ll == l and 0 <= blk < nblk and svc._in_flight(l, b, h, blk) is tr:
                infl_slot[b, h, blk] = tr.slot
                infl_gen[b, h, blk] = tr.gen
                flying.setdefault((b, h), []).append(tr)
        for (b, h), trs in flying.items():                              # the issue order (Policy E, 12j)
            st = svc.pools[l].stream(b, h)
            for r, tr in enumerate(sorted(trs, key=lambda x: x.seq)):
                infl_rank[b, h, tr.slot] = r
                infl_unprot[b, h, tr.slot] = not st.prot.get(tr.block)
        for (b, h), st in svc.pools[l].streams.items():
            for i, s in enumerate(st.slots):
                if s.state == RELEASING:
                    releasing[b, h, i] = True
            for r, (blk, idx) in enumerate(st.lru.items()):              # the LRU order, least recent first
                lru_rank[b, h, idx] = r
            for blk, idx in list(st.lru.items()) + list(st.po.items()) + list(st.vres.items()):
                if st.prot.get(blk):
                    prot[b, h, idx] = True
        self.infl_slot.copy_(infl_slot, non_blocking=True)
        self.infl_gen.copy_(infl_gen, non_blocking=True)
        self.releasing.copy_(releasing, non_blocking=True)
        self.lru_rank.copy_(lru_rank, non_blocking=True)
        self.prot.copy_(prot, non_blocking=True)
        self.infl_unprot.copy_(infl_unprot, non_blocking=True)
        self.infl_rank.copy_(infl_rank, non_blocking=True)


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
        self.skip_infl_wait = False                  # NEGATIVE CONTROL (no_inflight_wait): no wait for a superseded write


def _selected_mask(S: int, H: int, NBLK: int, parts, dev) -> torch.Tensor:
    """(S * NBLK,) bool: the blocks that some row of the layer selects (a valid historical selection), per stream.
    parts: [(rb (R,), hb (R,), sel (H, R, k))]."""
    acc = torch.zeros(S * NBLK + 1, dtype=torch.int64, device=dev)
    for (rb_, hb_, sel_) in parts:
        _, R_, _ = sel_.shape
        s_ = sel_.permute(1, 0, 2)
        ok = (s_ >= 0) & (s_ < hb_.view(R_, 1, 1)) & (s_ < NBLK)
        sid_ = rb_.view(R_, 1, 1) * H + torch.arange(H, device=dev).view(1, H, 1)
        f = torch.where(ok, sid_ * NBLK + s_.clamp(min=0), torch.full_like(s_, S * NBLK))
        acc.index_add_(0, f.reshape(-1), ok.reshape(-1).to(torch.int64))
    return acc[:S * NBLK] > 0


def device_verify(m: VerifyMirror, rb, hb, tails, nt, sel, C: int, X: int, mut=None, others=None,
                  inflight: bool = True) -> VerifyDecision:
    """One layer's verify rows on the device (module docstring), then the mirror update for the layer's draft rows.
    rb, hb (Rv,) int64: batch slot and history boundary of each verify row; tails (Rv, T) int64 tail slots (-1
    padded), nt (Rv,) int64 their count; sel (H, Rv, k) int64 the selections; others: the layer's other rows
    [(rb, hb, sel)] (the draft rows: Policy E never evicts a block that a row of the layer selects).
    POLICY E (notes 12h, 12j): a first miss that K5 refuses takes the stream's next eviction candidate: first an LRU
    slot whose block is unprotected and selected by no row of the layer, least recent first (the mirror's recency
    stamps); then (inflight True) a slot of the current owner's transfer in flight whose block is unprotected and
    selected by no row of the layer, oldest issue first (the take stamps): that prefetch is SUPERSEDED, and the
    layer's wait covers its write before the urgent gather (evict_infl, ev_gen: wait_list); only a miss without a
    candidate is an overflow.
    mut (NEGATIVE CONTROLS, tests only): gather_into_releasing (a RELEASING slot counts as FREE: the urgent gather
    writes a slot whose last reader may be pending), no_verify_protect (the verify read protects nothing),
    no_verify_dedup (a later occurrence of a miss takes a second slot: two transfers of one block), no_overflow_flag
    (an overflow is not reported), evict_protected / evict_selected / evict_newest (Policy E ignores the protection,
    ignores the layer's selections, takes the most recent candidate first), no_inflight_wait (the layer's wait omits
    a superseded prefetch: the urgent gather may write before that write landed), verify_no_local (K5 grants no FREE
    base slot under C: every first miss needs a shared extra), verify_reserve=R (K5 keeps R shared extras from the
    verifier, the draft rule: budget X - borrowed - R)."""
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
    if mut.get("verify_no_local"):                                                 # NEGATIVE CONTROL: no base slot
        free_local = torch.zeros_like(free_local)
    needs_extra = fmiss & (rank_s >= free_local[sid])
    borrowed = (usage - C).clamp(min=0).sum()
    budget = (X - borrowed - mut.get("verify_reserve", 0)).clamp(min=0)            # NEGATIVE CONTROL: a reserve
    ne = needs_extra.to(torch.int64)
    g = ne.cumsum(0) - ne
    granted = fmiss & (~needs_extra | (g < budget))
    refused_k5 = fmiss & ~granted
    # the q-th grant of a stream takes its q-th FREE slot (K3X first_free after the earlier takes)
    q = rank_in_group(sid, granted)
    freem = (m.free | m.releasing) if mut.get("gather_into_releasing") else m.free
    ar = torch.arange(NPOS, device=dev).view(1, NPOS).expand(S, NPOS)
    free_sorted = torch.sort(torch.where(freem.reshape(S, NPOS), ar, torch.full_like(ar, NPOS)), dim=-1).values
    # Policy E: the r-th K5-refused first miss of a stream evicts the stream's r-th candidate (least recent first)
    selm = _selected_mask(S, H, NBLK, [(rb, hb, sel)] + list(others or []), dev)
    sblk = m.slot_block.reshape(S, NPOS)
    sid_slot = torch.arange(S, device=dev).view(S, 1).expand(S, NPOS)
    in_sel = (sblk >= 0) & (sblk < NBLK) & selm[sid_slot * NBLK + sblk.clamp(min=0, max=NBLK - 1)]
    cand = m.lru_stamp.reshape(S, NPOS) < BIG_STAMP                                 # LRU slots
    fly = m.fly.reshape(S, NPOS)                                                    # the owner's slots in flight
    cand_f = fly if mut.get("evict_protected") else m.infl_unprot.reshape(S, NPOS)
    if not inflight:                                                                # 12h's rule (a negative control)
        cand_f = torch.zeros_like(cand_f)
    if not mut.get("evict_protected"):
        cand = cand & ~m.prot.reshape(S, NPOS)
    if not mut.get("evict_selected"):
        cand = cand & ~in_sel
        cand_f = cand_f & ~in_sel
    # one key per slot: the LRU candidates by recency, then the in-flight ones by issue order (HALF_STAMP + take stamp)
    fkey = HALF_STAMP + m.fly_stamp.reshape(S, NPOS).clamp(max=HALF_STAMP)
    ckey = torch.where(cand, m.lru_stamp.reshape(S, NPOS), fkey)
    if mut.get("evict_newest"):
        ckey = -ckey
    ckey = torch.where(cand | cand_f, ckey, torch.full_like(ckey, BIG_STAMP))
    cand_sorted = torch.sort(ckey, dim=-1).indices                                  # candidates first, oldest first
    n_cand = (cand | cand_f).to(torch.int64).sum(-1)                                # (S,)
    r_ev = rank_in_group(sid, refused_k5)
    evicted = refused_k5 & (r_ev < n_cand[sid])
    ev_slot = cand_sorted[sid, r_ev.clamp(max=NPOS - 1)]
    ev_blk = torch.where(evicted, sblk[sid, ev_slot], torch.full_like(ev_slot, -1))
    ev_infl = evicted & fly[sid, ev_slot]                                           # a superseded prefetch (12j)
    ev_gen = torch.where(ev_infl, m.slot_gen.reshape(S, NPOS)[sid, ev_slot], torch.zeros_like(ev_slot))
    over = refused_k5 & ~evicted
    taken = granted | evicted
    take_slot = torch.where(granted, free_sorted[sid, q.clamp(max=NPOS - 1)],
                            torch.where(evicted, ev_slot, torch.full_like(q, -1)))
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
    # the urgent gather's load ids: slot -> block for every take (a dummy column takes the other entries)
    col = torch.where(taken, take_slot, torch.full_like(take_slot, NPOS))
    ids = torch.full((H, B, NPOS + 1), -1, dtype=torch.int64, device=dev)
    ids.index_put_((hh_f, bb_f, col), torch.where(taken, blk_f, torch.full_like(blk_f, -1)))
    load_ids = ids[..., :NPOS].contiguous()
    # the mirror after the verify rows (device_draft reads it): urgent blocks in flight; granted slots no longer FREE
    # (usage + 1); an evicted slot is reused (usage unchanged) and its old block is no longer resident
    tk = granted.to(torch.int64)
    d_infl = torch.zeros(S * NBLK, dtype=torch.int64, device=dev).index_add_(0, sid * NBLK + blk_f,
                                                                              taken.to(torch.int64))
    m.inflight |= d_infl.view(B, H, NBLK) > 0
    d_free = torch.zeros(S * NPOS, dtype=torch.int64, device=dev).index_add_(0, sid * NPOS + take_slot.clamp(min=0), tk)
    m.free &= ~(d_free.view(B, H, NPOS) > 0)
    m.usage += torch.zeros(S, dtype=torch.int64, device=dev).index_add_(0, sid, tk).view(B, H)
    d_gone = torch.zeros(S * NBLK, dtype=torch.int64, device=dev).index_add_(0, sid * NBLK + ev_blk.clamp(min=0),
                                                                              evicted.to(torch.int64))
    m.slot_of.masked_fill_(d_gone.view(B, H, NBLK) > 0, -1)
    # a superseded prefetch's block is no longer in flight for the layer's draft rows (12j)
    d_gone_f = torch.zeros(S * NBLK, dtype=torch.int64, device=dev).index_add_(0, sid * NBLK + ev_blk.clamp(min=0),
                                                                                ev_infl.to(torch.int64))
    gone_f = d_gone_f.view(B, H, NBLK) > 0
    m.inflight &= ~gone_f
    m.infl_slot.masked_fill_(gone_f, -1)
    n_over = over.to(torch.int64).sum()
    counters = torch.stack([torch.zeros_like(n_over) if mut.get("no_overflow_flag") else n_over])
    t = dict(blk=blk_f, valid=valid_f, ready=ready.reshape(N), slot=slot.reshape(N), infl=infl.reshape(N),
             infl_slot=islot.reshape(N), infl_gen=igen.reshape(N), dup=dup, granted=granted, take_slot=take_slot,
             rslot=rslot, protect=protect.reshape(N), evict=evicted, evict_blk=ev_blk, evict_infl=ev_infl,
             ev_gen=ev_gen)
    dec = VerifyDecision(Rv, H, k, {f: v.to(torch.int64) for f, v in t.items()}, page, num_pages, load_ids,
                         (bb_f, hh_f), counters)
    dec.skip_infl_wait = "no_inflight_wait" in mut
    return dec


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
        if dec.skip_infl_wait:                                    # NEGATIVE CONTROL (no_inflight_wait)
            return
        sup = zip(bb.tolist(), hh.tolist(), dec.t["evict_infl"].tolist(), dec.t["evict_blk"].tolist(),
                  dec.t["take_slot"].tolist(), dec.t["ev_gen"].tolist())
        for (b, h, e, ob, slot, gen) in sup:                      # Policy E (12j): the superseded prefetch lands first
            if not e:
                continue
            tr = svc._in_flight(l, b, h, ob)
            if tr is None or tr.slot != slot or tr.gen != gen:
                raise OwnershipViolation("S-C device: Policy E supersedes block %d at slot %d gen %d (layer %d b %d h "
                                         "%d), which is not the owner's transfer in flight" % (ob, slot, gen, l, b, h))
            if not svc.tp.done(tr):
                svc.tp.force(tr)
            self.waited += 1

    def gather(self, l: int, dec: VerifyDecision) -> None:
        """The urgent gather on the main stream: every (h, b, slot) with a block id in the load ids gets the block's
        host K / V rows and its cis rows (bias), after the block's host write-back (T4); a slot whose last reader is
        pending is refused (T3). A slot that Policy E EVICTED was last read in an earlier tick, before this gather in
        the main stream's order: its pending reader is host-waited here (the stand-in of that order; counted as
        verifier_force_waits, as v1 counts it). First, the stand-in of the main stream's wait on every host
        write-back registered since the last gather (CudaDeviceOps.gather, T4): each pending one is host-waited."""
        svc = self.svc
        for ev in svc._hist_unwaited:
            if not ev_done(ev):
                svc.host_wait(ev)
        svc._hist_unwaited = []
        g = svc.gather                                    # the host rows and the store (TorchGather's tensors)
        bs = svc.bs
        bb, hh = dec.bh
        evicted, superseded = set(), {}
        for (b, h, s, e, f, ob) in zip(bb.tolist(), hh.tolist(), dec.t["take_slot"].tolist(), dec.t["evict"].tolist(),
                                       dec.t["evict_infl"].tolist(), dec.t["evict_blk"].tolist()):
            if e and f:
                superseded[(b, h, s)] = ob                        # an in-flight prefetch's slot (12j)
            elif e:
                evicted.add((b, h, s))                            # an LRU entry's slot (12h)
        for h, per_b in enumerate(dec.load_ids.tolist()):
            for b, per_m in enumerate(per_b):
                for m, t in enumerate(per_m):
                    if t < 0:
                        continue
                    s = svc.stream(l, b, h).slots[m]
                    if (b, h, m) in evicted and not ev_done(s.last_reader):
                        svc.host_wait(s.last_reader)
                        svc.metrics.count("verifier_force_waits")
                    if (b, h, m) in superseded:
                        tr = svc._in_flight(l, b, h, superseded[(b, h, m)])
                        if self.check_t3 and tr is not None and not svc.tp.done(tr):
                            raise OwnershipViolation("S-C device: the urgent gather of block %d writes slot %d (layer %d "
                                                     "b %d h %d) before the prefetch it supersedes landed (T3 for "
                                                     "writers)" % (t, m, l, b, h))
                    if self.check_t3 and not ev_done(s.last_reader):
                        raise OwnershipViolation("S-C device: the urgent gather of block %d writes slot %d (layer %d b %d "
                                                 "h %d) while the slot's last reader is pending (T3)" % (t, m, l, b, h))
                    hev = svc._hist_ev_for(b, t)
                    if not ev_done(hev):
                        svc.host_wait(hev)                # the main stream's wait on the host write-back (T4)
                    r0, s0 = t * bs, m * bs
                    if svc.xpool is not None:                     # [xpool] the slot's lane in the flat store
                        p0 = svc.xpool.pos_host(l, b, h, m) * bs
                        g.store_k[l][p0:p0 + bs, h].copy_(g.host_k[l][b, r0:r0 + bs, h])
                        g.store_v[l][p0:p0 + bs, h].copy_(g.host_v[l][b, r0:r0 + bs, h])
                        if svc.store_b is not None and svc.cis is not None:
                            svc.store_b[l][p0:p0 + bs, h].copy_(svc.cis[l][b, r0:r0 + bs, h])
                        self.gathered += 1
                        continue
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
    in-flight slot (-1 for an entry that waits for nothing) and the generation it waits for. An entry waits for the
    transfer in flight that its verify row reads, or (Policy E, 12j) for the prefetch its urgent block supersedes:
    that write lands before the urgent gather overwrites the slot. Fixed length (the decision's N): capturable."""
    bb, hh = dec.bh
    infl = dec.t["infl"] != 0
    ev = dec.t["evict_infl"] != 0
    if dec.skip_infl_wait:                                            # NEGATIVE CONTROL (no_inflight_wait)
        ev = torch.zeros_like(ev)
    slot_w = torch.where(infl, dec.t["infl_slot"], dec.t["take_slot"]).clamp(min=0)
    f = ((l * board.B + bb) * board.H + hh) * board.npos + slot_w
    return (torch.where(infl | ev, f, torch.full_like(f, -1)),
            torch.where(infl, dec.t["infl_gen"], dec.t["ev_gen"]))


WAIT_PHASES = ("pre", "list", "wait")


def wait_profile_summary(entries, L: int, spins=None) -> Dict:
    """REPORT (CudaDeviceOps.wait_profile; pure, so the CPU tests check it): entries [(l, m0, m1, m2, m3)] with each m =
    (host seconds, an event whose elapsed_time(other) is in ms). The phases of one wait call:
      pre   m0 -> m1   exact: the submission host-waits (_submitted)
      list  m1 -> m2   exact: wait_list's device ops
      wait  m2 -> m3   exact: the spin program; superset: the event waits (m0 = m1 = m2)
    Per layer and in total: the calls, each phase's host ms and device ms (0 for a phase whose bounds are one mark),
    and the spin rounds. A device phase much longer than its host phase is time the main stream spent waiting on the
    device; a host phase as long as its device phase means the stream waited for the host to enqueue."""
    keys = ["host_%s_ms" % p for p in WAIT_PHASES] + ["device_%s_ms" % p for p in WAIT_PHASES]
    per = []
    for l in range(L):
        d = dict(layer=l, calls=0, spins=None if spins is None else spins[l])
        d.update({k: 0.0 for k in keys})
        per.append(d)
    for (l, m0, m1, m2, m3) in entries:
        p = per[l]
        p["calls"] += 1
        for name, a, b in zip(WAIT_PHASES, (m0, m1, m2), (m1, m2, m3)):
            p["host_%s_ms" % name] += 1e3 * (b[0] - a[0])
            if a[1] is not b[1]:
                p["device_%s_ms" % name] += a[1].elapsed_time(b[1])
    for p in per:
        for k in keys:
            p[k] = round(p[k], 4)
    total = {k: round(sum(p[k] for p in per), 3) for k in keys}
    total["calls"] = sum(p["calls"] for p in per)
    total["spins"] = None if spins is None else sum(spins)
    return dict(per_layer=per, total=total)


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
        self.spins_by_layer = [0] * svc.L                 # REPORT: spin rounds per layer (CudaDeviceOps.spins' stand-in)

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
            self.spins_by_layer[l] += k
            self.waited += 1

    def spin_counts(self) -> List[int]:
        """REPORT: the spin rounds per layer over the run. Here one round per entry and spin; the GPU kernel counts one
        round per spin of a block of up to 1024 entries (CudaDeviceOps.spin_counts)."""
        return list(self.spins_by_layer)


# The exact wait's Triton kernel (note 12f). ONE program (one SM: the scatter stream's kernels keep the others) walks
# the wait list in blocks; per block it spins with VOLATILE loads of the board until every listed entry reached its
# generation, at most max_spin rounds; then any entry short or past raises err (int32, atomic max). The board write
# that ends a spin is enqueued after the scatter on the scatter stream, so the slot's rows are complete when the main
# stream's next kernel reads them. It is defined at MODULE level: Triton's GPU compiler resolves a kernel's names in
# its module globals (fn.__globals__), never in an enclosing function (fg5 on holder 2190876: "NameError: tl is not
# defined" from a factory's local import; test F23 guards every kernel we own).
if triton is not None:
    @triton.jit
    def board_wait_kernel(board_ptr, idx_ptr, gen_ptr, n, err_ptr, max_spin, spins_ptr, BLOCK: tl.constexpr):
        total = 0                                         # REPORT: the spin rounds of this call (all its blocks)
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
            total += k
        # one program: the layer's counter (an int64 cell) is read and written by this call only (stream order)
        tl.store(spins_ptr, tl.load(spins_ptr) + total)
else:                                                     # pragma: no cover - environment dependent
    board_wait_kernel = None


def _wait_kernel():
    """The exact wait's Triton kernel (board_wait_kernel); triton is needed only here, on the GPU."""
    if board_wait_kernel is None:
        raise RuntimeError("the exact device wait needs triton (the CPU stand-in is ExactWaitSim)")
    return board_wait_kernel


class CudaDeviceOps:
    """S-C's device operations on the GPU (module docstring). SOURCE ONLY, NOT RUN YET (GPU gate FG1).
    wait_mode "superset": the main stream waits on the completion event of every transfer in flight at the layer.
    wait_mode "exact" (note 12f): one Triton program on the main stream spins on the generation board for exactly the
    in-flight slots that the layer's verify rows read (wait_list), after the host waited the layer's packs' SUBMISSION
    (_submitted: the liveness rule, note 12g; GIL-free for E's C++ worker, in the eager and the captured tick, gfw1c);
    no device read on the host; err (int32, on the
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
        # REPORT: the exact wait's spin rounds per layer, on the device (the program adds each call's rounds; a fixed
        # buffer, so a captured tick counts too); read by spin_counts() after a run, never in a tick
        self.spins = torch.zeros((svc.L,), dtype=torch.int64, device=svc.device)
        # REPORT (opt-in: a list; eager ticks only, a capture records nothing): per wait call (layer, m0, m1, m2, m3),
        # each m = (host clock, timing event on the main stream) at the phase bounds; wait_profile() sums them
        self.profile = None

    def _mark(self):
        """REPORT: a host clock and a timing event recorded on the main stream now (the wait's phase bounds)."""
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return time.perf_counter(), e

    def spin_counts(self) -> List[int]:
        """REPORT: the exact wait's spin rounds per layer over the run (one device read; call it after the run). The
        program counts one round per spin of a block of up to 1024 wait-list entries."""
        return [int(x) for x in self.spins.tolist()]

    def wait_profile(self) -> Dict:
        """REPORT: wait_profile_summary of the recorded calls and the spin rounds (one synchronize; after the run)."""
        torch.cuda.synchronize()
        return wait_profile_summary(self.profile or [], self.svc.L, self.spin_counts())

    def _submitted(self, l: int, captured: bool = False) -> None:
        """LIVENESS: every transfer in flight at layer l is SUBMITTED (its GPU work enqueued) before the main stream may
        spin on it; the landing is waited on the device.
        A transport that submits without the GIL (submits_without_gil: E's C++ worker, GilfreeTransport(...,
        worker="cpp")) is waited GIL-FREE: tp.wait_submitted(tr) (E's _JobEv.wait_submitted, the GIL released), once per
        job (its transfers share the job's event), for every in-flight transfer of the layer that is not landed in place
        (an urgent take is not the transport's), in the eager AND the captured tick. gfw1c on holder 2190876: the
        captured protocol WITHOUT this wait (GFW1) ran its spin to the bound (8.6 s, err 1), because the worker's GPU
        work serializes behind the spinning main stream; with the wait first (GFW1e) it passed. This replaces 7902b7a's
        "no wait for such a transport". captured (the captured tick's host part passes True) does not skip the wait.
        Any other transport: a pack still queued in a Python feeder is host-waited for its submission (a
        threading.Event: the GIL is released), as before. Each wait counts in submission_waits."""
        tp = getattr(self.svc, "tp", None)
        if getattr(tp, "submits_without_gil", False) and callable(getattr(tp, "wait_submitted", None)):
            seen = set()
            for tr in list(self.svc.inflight.values()):
                if tr.layer != l or tr.seq in self.svc._landed:
                    continue
                job = id(tr.ev) if tr.ev is not None else ("seq", tr.seq)    # a transfer never released raises there
                if job in seen:
                    continue
                seen.add(job)
                tp.wait_submitted(tr)
                self.submission_waits += 1
            return
        for tr in list(self.svc.inflight.values()):
            if tr.layer == l and tr.ev is None and isinstance(tr.payload, threading.Event):
                tr.payload.wait()
                self.submission_waits += 1

    def wait(self, l: int, dec: VerifyDecision) -> None:
        prof = self.profile is not None and not torch.cuda.is_current_stream_capturing()   # REPORT (opt-in)
        m0 = self._mark() if prof else None
        if self.wait_mode == "exact":
            self._submitted(l)
            m1 = self._mark() if prof else None
            idx, gen = wait_list(dec, l, self.svc.board)
            m2 = self._mark() if prof else None
            _wait_kernel()[(1,)](self.svc.board.t, idx, gen, idx.shape[0], self.err, self.max_spin,
                                 self.spins[l:l + 1], BLOCK=1024)
            if prof:
                self.profile.append((l, m0, m1, m2, self._mark()))
            return
        self._superset_wait(l)
        if prof:
            self.profile.append((l, m0, m0, m0, self._mark()))     # superset: one phase, the event waits

    def _superset_wait(self, l: int) -> None:
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
        xp = self.svc.xpool
        if xp is not None:                                # [xpool] one launch over the decision's entries (r2_xpool)
            bb, hh = dec.bh
            t = dec.t
            ent = xp.gather_entries(l, bb, hh, t["blk"], t["take_slot"], (t["granted"] | t["evict"]) != 0)
            XP.h2d_entries(self.store_k[l], self.store_v[l], self.store_b[l], self.host_k[l], self.host_v[l],
                           self.cis[l], ent, self.bs)
            return
        self.h2d_bias(self.store_k[l], self.host_k[l], self.store_b[l], self.cis[l], dec.load_ids, self.bs)
        self.h2d(self.store_v[l], self.host_v[l], dec.load_ids, self.bs)


# ====================================================================================================================
# the service
# ====================================================================================================================
def page_keys_of(dec, listed: str, hb: torch.Tensor, nt: torch.Tensor) -> torch.Tensor:
    """[sharedkv2] The BLOCK ID of every entry of one decision's page-table rows (dec.page, (n, H, k + T)), device ops
    only (no host read, no host-to-device copy; docs/R2_SHAREDKV2_KEYS_S3.md). The entries the decision lists come
    first: listed = "valid" for device_verify (every valid selected block), "ready" for device_draft (the READY
    blocks); both decisions order their slots by ascending block id (vkey / rkey sorted), so the sorted listed block
    ids are the keys of the same positions. Then the tail pages: block hist_bound + u for u < nt (the tick's rows name
    their tail blocks so; ResidencySC._check_tail_keys refuses rows that do not). -1 padded, int32, dec.page's shape.
    Derived from the decision's (blk, listed) columns, not from its slot choices, so a change of the slot allocation
    does not touch it."""
    n, H, P = dec.page.shape
    k = dec.k
    T = P - k
    dv_ = dec.page.device
    blk = dec.t["blk"].view(n, H, k)
    ok = dec.t[listed].view(n, H, k) != 0
    big = torch.iinfo(torch.int64).max
    srt = torch.sort(torch.where(ok, blk, torch.full_like(blk, big)), dim=-1).values
    cnt = ok.to(torch.int64).sum(-1)                                                # (n, H): the listed entries
    posk = torch.arange(k, device=dv_).view(1, 1, k)
    key = torch.full((n, H, P), -1, dtype=torch.int64, device=dv_)
    key[..., :k] = torch.where(posk < cnt.unsqueeze(-1), srt, torch.full_like(srt, -1))
    post = torch.arange(T, device=dv_).view(1, 1, T)
    tkey = torch.where(post < nt.view(n, 1, 1), hb.view(n, 1, 1) + post, torch.full_like(post, -1))
    key.scatter_(-1, (cnt.unsqueeze(-1) + post).expand(n, H, T).contiguous(), tkey.expand(n, H, T).contiguous())
    return key.to(torch.int32).contiguous()


class ResidencySC(ResidencySB):
    """Residency with the verify and the draft rows decided on the device (module docstring). Extra arguments (beyond
    S-B's device, nblk, sb_mut): sc_mut (negative controls: device_verify's, and no_wait: the device does not wait for
    the in-flight blocks), devops (the device operations; on the CPU the stand-in by default), mirror_drop (NEGATIVE
    CONTROL, tests only: the journal kinds, or "owner", whose changes are NOT sent to the device mirror), wait_mode
    (the CPU stand-in's wait: "probe" = CpuDeviceSim, the host's transfer records; "exact" = ExactWaitSim, the
    generation board, which needs a publishing transport; on the GPU CudaDeviceOps' own wait_mode decides).
    The generation board (self.board, note 12f) is attached to every transport that publishes (attach_board).
    [sharedkv2] want_keys (default False: nothing changes): resolve also fills plan.page_key (page_keys_of);
    key_plant (NEGATIVE CONTROL, tests only): (head, position) of the first verify row's key that gets + 1."""

    want_keys = False                                     # [sharedkv2] set by the tick's ResidencyBackend
    key_plant = None                                      # [sharedkv2] NEGATIVE CONTROL (tests only)
    key_plant_live = 0                                    # [sharedkv2] device count: plants inside num_pages

    XPOOL_OK = True                                       # [xpool] resolve binds the lanes (r2_xpool.XPool)

    def __init__(self, *args, sc_mut=None, devops=None, mirror_drop=None, wait_mode: str = "probe", **kw):
        kw["defer_admit"] = True
        super().__init__(*args, **kw)
        self.board = GenBoard(self.L, self.B, self.H, self.n_hist, self.device)
        if callable(getattr(self.tp, "attach_board", None)):
            self.tp.attach_board(self.board)
        self.mirrors = None                               # S-B's per-layer host-refreshed mirrors are not used
        self.dmirror = DeviceMirror(self.L, self.B, self.H, self.nblk, self.n_hist, self.device)
        self.mirror_drop = set(mirror_drop or ())
        self._ending = False                                         # inside end_request (_prot_changed)
        self.journal_counts: Dict[str, int] = {}                     # the stream transitions per kind (reports)
        self._dirty: Set[Tuple[int, int, int, int]] = set()          # (l, b, h, slot) changed since the last sync
        self._stamp: Dict[Tuple[int, int, int, int], int] = {}       # (l, b, h, slot) -> its LRU recency (Policy E)
        self._clock = 0
        for l, pool in enumerate(self.pools):
            for (b, h), st in pool.streams.items():
                st.log = _Journal((l, b, h), self._journal, st.log)
                st.prot = _ProtDict((l, b, h), self._prot_changed, st.prot)
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
        # REPORT (report()["k5"]): per (tick, layer) the verify first misses that needed a shared extra (_extra_need),
        # its peak with the tick and layer, and each layer's maximum; refused ticks included
        self.k5_peak: Tuple = (0, None, None)
        self.k5_by_layer: List[int] = [0] * self.L
        if self.xpool is not None:                        # [xpool] the binder's / translator's error flag rides in the
            # tick's ONE decision read (a set flag refuses the tick). S-C's own watch_flag, called by name: a subclass
            # override may read state its __init__ sets only after this one returns (B2's b2_mut, r2_own2.py:416-417)
            ResidencySC.watch_flag(self, "xpool_err", self.xpool.err)

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
        """Every stream transition, as the vendored ownership code logs it: (event, block, slot, gen, detail); counted
        per kind in journal_counts (a release by its reason: release:evict, release:unprotect, release:end, ...)."""
        kind = item[0] if item[0] != "release" else "release:%s" % (item[4],)
        self.journal_counts[kind] = self.journal_counts.get(kind, 0) + 1
        if item[0] in STAMP_KINDS:                         # the LRU recency (touch / admit / acquire), the issue order
            self._clock += 1                               # of a slot in flight (take): Policy E's orders
            self._stamp[key + (item[2],)] = self._clock
        if item[0] in self.mirror_drop:
            return
        self._dirty.add(key + (item[2],))

    def _prot_changed(self, key: Tuple[int, int, int], block: int) -> None:
        """A block's protection may have changed (_ProtDict): its resident slot, or its slot in flight (Policy E's
        in-flight candidates, 12j), goes with the next delta. Not the slot in flight while its request ENDS
        (end_request): that transfer stops being the owner's, and the owner change marks it (F14b's control)."""
        if "prot" in self.mirror_drop:
            return
        st = self.pools[key[0]].stream(key[1], key[2])
        idx = st.resident(block)
        if idx is None and not self._ending:
            idx = st.pending.get(block)
        if idx is not None:
            self._dirty.add(key + (idx,))

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

    def history_written(self, uid, block: int, ev) -> None:
        """Residency.history_written, and the event joins the write-backs that the main stream has not waited on yet
        (_hist_unwaited, T4). The GPU gather waits on every one of them before its copies (CudaDeviceOps.gather; a
        captured tick waits them before the replay, outside the graph); the CPU stand-in host-waits them. Without
        this producer the GPU urgent gather was ordered after a write-back only when both ran on one stream."""
        super().history_written(uid, block, ev)
        self._hist_unwaited.append(ev)

    def end_request(self, uid) -> None:
        b = self.uid_b[uid]
        self._ending = True                    # its transfers in flight stop being the owner's: no protection marks
        try:
            super().end_request(uid)
        finally:
            self._ending = False
        self._mark_batch_slot(b)

    def _h2d(self, host: torch.Tensor) -> torch.Tensor:
        """Every host-to-device copy of S-C goes through here (per tick: the mirror delta and the row inputs). On the
        GPU from pinned memory, asynchronous on the current stream (no host sync)."""
        if self.device.type == "cuda":
            return host.pin_memory().to(self.device, non_blocking=True)
        return host.to(self.device)

    def sync_mirror(self) -> bool:
        """The ONE batched host-to-device delta of the device mirror: the record of every slot that changed since the
        last sync (state code, block, generation, current-owner flag, protected flag, recency stamp). Called by the
        first resolve of a tick (and by tests); returns whether a delta was sent."""
        if not self._dirty:
            return False
        B, H, NPOS = self.B, self.H, self.n_hist
        recs = []
        for (l, b, h, i) in sorted(self._dirty):
            st = self.pools[l].stream(b, h)
            s = st.slots[i]
            cur = 0
            if s.state == INFLIGHT:
                tr = self._in_flight(l, b, h, s.block)
                cur = 1 if tr is not None and tr.slot == i else 0
            prot = 1 if s.state in (LRU, PO, VRES, INFLIGHT) and st.prot.get(s.block) else 0
            recs.append((((l * B + b) * H + h) * NPOS + i, CODE[s.state], s.block, s.gen, cur, prot,
                         self._stamp.get((l, b, h, i), 0)))
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
        extra = self._tick_extra(rows)                                # the sparse record's per-tick inputs (or [])
        flat = ([r.b for r in rows] + [r.hist_bound for r in rows] + [len(r.tail) for r in rows] + tails_flat
                + [r.pos % self.bs + 1 for r in rows] + vrows + drows + extra)
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
                    d=part(dix) if drows else None, last_len=last_len.to(torch.int32), seq_idx=rb.to(torch.int32),
                    extra=d[len(flat) - len(extra):])

    def _tick_extra(self, rows: Sequence[Row]) -> List[int]:
        """Hook (the sparse record, r2_residency_sparse.py): more per-tick integers for the ONE row copy."""
        return []

    def _layer_record(self, l: int, step: Dict, tk: Dict) -> None:
        """Hook (the sparse record): the layer's device-side record, device ops only. S-C: none (the dense record is
        the decision tensors themselves)."""
        return None

    # ------------------------------------------------------------------------------------------------ one layer
    def resolve(self, l: int, rows: Sequence[Row], sel: torch.Tensor) -> Plan:
        H, R = self.H, len(rows)
        if any(st["l"] == l for st in self._tick):
            raise ValueError("S-C: layer %d resolved twice in one tick (its decisions are applied at end_tick)" % l)
        if sel.device.type != self.device.type:
            raise ValueError("S-C: the selection must be on the service's device %s (got %s)" % (self.device, sel.device))
        self._check_rows(l, rows, sel)
        if self.want_keys and not self._tick:                         # [sharedkv2] host rows only, once per tick
            self._check_tail_keys(rows)
        if not self._tick:                                            # the tick's first layer: the per-tick copies
            self.sync_mirror()
            self._rows = self._tick_rows(rows)
            if self.xpool is not None:                                # [xpool] every FREE slot gives its lane back;
                if not self.DECISION_READ_BLOCKING:                   # safe only behind a BLOCKING decision read
                    raise OwnershipViolation("xpool: lanes are reused across requests only behind a blocking decision "
                                             "read (r2_xpool.py, the lane-level order); this class's is not")
                self.xpool.release_free(self.dmirror.code)
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
            others = []
            if tk["d"] is not None:                                   # Policy E spares the draft rows' blocks too
                rbd, hbd, _, _, ixd = tk["d"]
                others.append((rbd, hbd, sel[:, ixd]))
            vdec = device_verify(m, rb, hb, tl, nt, sel[:, ix], self.C, self.X, self.sc_mut, others=others,
                                 inflight=self.policy_e_inflight)
            if self.xpool is not None:                                # [xpool] the takes' lanes, before the gather
                self.xpool.bind(l, vdec.bh[0], vdec.bh[1], vdec.t["take_slot"],
                                (vdec.t["granted"] | vdec.t["evict"]) != 0)
            if not self.sc_mut.get("no_wait"):                        # NEGATIVE CONTROL: read before the event
                self.dev.wait(l, vdec)
            self.dev.gather(l, vdec)
            page[ix] = vdec.page
            npg[ix] = vdec.num_pages
            step["vdec"] = vdec
        if tk["d"] is not None:                                       # the draft rows: S-B, on the mirror after verify
            rb, hb, tl, nt, ix = tk["d"]
            ddec = device_draft(m, rb, hb, tl, nt, sel[:, ix], self.C, self.X, self.R_ver, self.sb_mut)
            if self.xpool is not None:                                # [xpool] the draft takes' lanes, before the
                Rd, kd = ddec.Rd, ddec.k                              # bias fill (_layer_record) and the issue
                self.xpool.bind(l, rb.view(Rd, 1, 1).expand(Rd, H, kd),
                                torch.arange(H, device=dv).view(1, H, 1).expand(Rd, H, kd), ddec.t["take_slot"],
                                ddec.t["granted"] != 0)
            page[ix] = ddec.page
            npg[ix] = ddec.num_pages
            step["ddec"] = ddec
        self._layer_record(l, step, tk)                               # device ops only (the sparse record's hook)
        self._tick.append(step)
        plan.page_table = page.to(torch.int32)
        plan.num_pages = npg.to(torch.int32)
        plan.last_len = tk["last_len"]
        plan.seq_idx = tk["seq_idx"]
        if self.want_keys:                                            # [sharedkv2] device ops only
            plan.page_key = self._page_keys(R, P, step, tk, plan.num_pages)
        return plan

    # ------------------------------------------------------------------------------------------------ [sharedkv2] keys
    def _check_tail_keys(self, rows: Sequence[Row]) -> None:
        """[sharedkv2] page_keys_of names tail page u of a row block hist_bound + u (the tick's ResidencyBackend.prepare
        builds its rows so: tail [(T, ring)] with hist_bound T, or [(T - 1, ring), (T, ring)] with hist_bound T - 1).
        A row whose tail names other blocks is refused (host rows only; never guessed)."""
        for i, r in enumerate(rows):
            got = [t for (t, _) in r.tail]
            want = list(range(r.hist_bound, r.hist_bound + len(r.tail)))
            if got != want:
                raise ValueError("sharedkv2: row %d tail blocks %s are not hist_bound %d, %d + 1, ... (the page keys "
                                 "assume it)" % (i, got, r.hist_bound, r.hist_bound))

    def _page_keys(self, R: int, P: int, step: Dict, tk: Dict, num_pages: torch.Tensor) -> torch.Tensor:
        """[sharedkv2] plan.page_key (R, H, P) int32: the verify rows' keys from device_verify's decision (valid
        blocks), the draft rows' from device_draft's (READY blocks), each then its tail blocks (page_keys_of). Device
        ops only. key_plant (head h, position j): + 1 on the first verify row's key (h, j); key_plant_live counts the
        layers where that entry lies inside the row's num_pages (the plant's precondition, read by the test)."""
        key = torch.full((R, self.H, P), -1, dtype=torch.int32, device=self.device)
        for dec, part, listed in ((step["vdec"], tk["v"], "valid"), (step["ddec"], tk["d"], "ready")):
            if dec is not None:
                _, hb, _, nt, ix = part
                key[ix] = page_keys_of(dec, listed, hb, nt)
        if self.key_plant is not None and tk["vrows"]:                # NEGATIVE CONTROL (tests only)
            h, j = self.key_plant
            r0 = tk["vrows"][0]
            key[r0, h, j] += 1
            self.key_plant_live = self.key_plant_live + (num_pages[r0, h] > j).to(torch.int64)
        return key.contiguous()

    def end_tick(self) -> None:
        steps, self._tick = self._tick, []
        self._rows = None
        try:
            self.apply_tick(steps)
        finally:
            self._layer_ev = {}
        Residency.end_tick(self)                                      # the draft transfers, layer by layer
        self.metrics.sample_memory()

    def apply_tick(self, steps: List[Dict]) -> None:
        """THE HOST CONSUMER of one tick's device decisions (note 12l; the capture seam calls exactly this after the
        replay). steps: the tick's layers in resolve order, each dict(l, vrows, vidx, vdec, drows, ddec[, rec]) with
        the decision tensors (fixed buffers in a captured tick). It makes the ONE device-to-host read of the tick
        (the watched flags and the decisions), refuses a void tick (a set flag, an overflow) before any change of the
        host state, and applies the decisions in v1's order. This class: the DENSE record (every entry of every
        layer); ResidencySparse (r2_residency_sparse.py) replaces it with the sparse record, same signature."""
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
            if raised and raised != ["xpool_err"]:                    # ([xpool]: xpool_err alone waits, see below)
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
                raise self._refusal(over, self._overflow_report(parsed))
            if raised:                                                # [xpool] after the overflow: an overflowed page
                raise CapacityRefusal("S-C: the watched device flag(s) %s are set at this tick: the tick is void (none "
                                      "of its decisions is applied)" % raised)   # names slot 0 (review m7)
            for (st, v, d) in parsed:
                if v is not None:
                    self._apply_verify(st["l"], st["vrows"], st["vidx"], st["vdec"], v[0],
                                       st["drows"] if d is not None else None, d[0] if d is not None else None,
                                       st["ddec"] if d is not None else None)
                if d is not None:
                    self._apply(st["l"], st["drows"], st["ddec"], d[0], d[1], d[2], d[3])

    def _refusal(self, over, report: str) -> CapacityRefusal:
        """The CapacityRefusal of a tick whose verify first misses overflowed (the dense and the sparse consumer)."""
        return CapacityRefusal(
            "S-C: the verifier's urgent misses found no slot (layer, first misses with no slot: %s; C %d, X %d, R_ver "
            "%d). K5 granted each stream's FREE base slots (C - usage) and then the layer's shared extras (X - "
            "borrowed; the verifier may use the R_ver reserve too) in K3X order, and Policy E had no candidate left "
            "(every other LRU entry and in-flight draft prefetch is protected or read by the layer): this tick's demand "
            "exceeds that allowance, a configuration result. The tick's outputs are invalid. %s"
            % (over, self.C, self.X, self.R_ver, report))

    def _overflow_report(self, parsed, per_layer: int = 3) -> str:
        """For a refused tick (dense record): per layer with an overflow, _k5_report of the layer's verify first
        misses (an entry valid, not ready, not in flight, not a later occurrence of a miss), in K3X order."""
        out = []
        for (st, v, _) in parsed:
            if v is None or v[1] <= 0:
                continue
            dec, c, firsts = st["vdec"], v[0], []
            for o in range(dec.Rv * dec.H * dec.k):
                if not c["valid"][o] or c["ready"][o] or c["infl"][o] or c["dup"][o]:
                    continue
                bh = (st["vrows"][o // (dec.H * dec.k)].b, (o // dec.k) % dec.H)
                firsts.append((bh, "grant" if c["granted"][o] else ("evict" if c["evict"][o] else "none")))
            out.append(self._k5_report(st["l"], firsts, v[1], per_layer))
        return " | ".join(out)

    def _k5_report(self, l: int, firsts, n_over: int, per_layer: int = 3, note: str = "") -> str:
        """The overflow report of layer l of a refused tick. The host state is the tick's START (nothing is applied),
        so it shows the slots the decision would have taken as still FREE; this report shows the decision instead.
        firsts: the layer's verify FIRST misses in K3X order (row, head, block) as ((b, h), outcome), outcome "grant"
        (K5), "evict" (Policy E) or "none" (no slot). K5 for the verifier (device_verify = v1's LayerPool.can_take,
        ownership.py:419-427; usage = every non-FREE slot, :135-136; borrowed, :409-413): a stream's first C - usage
        first misses take its FREE base slots; each later one needs one of the layer's SHARED extras (X - borrowed at
        the tick start, the R_ver reserve included for the verifier), granted in K3X order while any is free; then
        Policy E. Per stream: first misses = base + shared extras + Policy E + no slot; then v1's _capacity_report of
        the first per_layer streams with a first miss with no slot."""
        pool = self.pools[l]
        per, order = {}, []
        for bh, what in firsts:
            if bh not in per:
                per[bh] = dict(n=0, grant=0, evict=0, none=0)
                order.append(bh)
            per[bh]["n"] += 1
            per[bh][what] += 1
        need = extras = 0
        lines = []
        for bh in order:
            p = per[bh]
            use = pool.stream(bh[0], bh[1]).usage()
            base = min(max(0, self.C - use), p["n"])
            need += p["n"] - base
            extras += p["grant"] - base
            if p["none"] and len(lines) < per_layer:
                lines.append("(l %d b %d h %d: first misses %d = %d in its free base slots (C %d - usage %d) + %d shared "
                             "extras + %d by Policy E + %d with no slot) %s"
                             % (l, bh[0], bh[1], p["n"], base, self.C, use, p["grant"] - base, p["evict"], p["none"],
                                self._capacity_report(l, bh[0], bh[1])))
        head = ("layer %d: %d streams had verify first misses, %d of the misses needed a shared extra; the layer's %d "
                "free extras (X %d, borrowed %d at the tick start) went to the first %d of them in K3X order; %d found "
                "no slot%s" % (l, len(order), need, pool.free_extras(), self.X, pool.borrowed(), extras, n_over, note))
        self._note_extra_need(l, need)
        return "; ".join([head] + lines)

    def _extra_need(self, use0: Dict, n_first: Dict) -> int:
        """The layer's verify first misses that needed a SHARED extra: per stream, its first misses beyond its FREE
        base slots at the layer's start (C - usage), summed (K5's rule; granted, Policy E and no slot alike)."""
        return sum(max(0, n - max(0, self.C - use0[bh])) for bh, n in n_first.items())

    def _note_extra_need(self, l: int, need: int) -> None:
        """REPORT: the shared-extra demand of layer l at this tick (self.tick_no): the layer's maximum, and the run's
        peak with its tick and layer (the first occurrence of the peak)."""
        if need > self.k5_by_layer[l]:
            self.k5_by_layer[l] = need
        if need > self.k5_peak[0]:
            self.k5_peak = (need, self.tick_no, l)

    def report(self) -> Dict:
        """Residency's report, plus "k5": the peak shared-extra demand of the run (passing and refused ticks)."""
        out = super().report()
        need, tick, layer = self.k5_peak
        out["k5"] = dict(extra_need_peak=need, at_tick=tick, at_layer=layer, by_layer=list(self.k5_by_layer), C=self.C,
                         X=self.X, R_ver=self.R_ver,
                         definition="per (tick, layer): the verify first misses beyond each stream's FREE base slots "
                                    "(C - usage at the layer's start), summed over the layer's streams: the shared "
                                    "extras that the layer's verify needed (granted, Policy E and no slot)")
        return out

    def _host_candidates(self, l: int, vrow_objs, dec, c, drow_objs, dcols, ddec):
        """Policy E's candidates on the host (v1's _verifier_candidates), from the host state at the start of the
        layer's apply and the layer's selections as the device decided them (the valid blocks of the verify and the
        draft entries): the reference every device eviction is checked against."""
        H = dec.H
        sel: Dict[Tuple[int, int], Set[int]] = {}
        for rows_, cols, n_rows, k_ in ((vrow_objs, c, dec.Rv, dec.k),
                                        (drow_objs, dcols, ddec.Rd if ddec is not None else 0,
                                         ddec.k if ddec is not None else 0)):
            if not n_rows:
                continue
            for o in range(n_rows * H * k_):
                if cols["valid"][o]:
                    j, h = o // (H * k_), (o // k_) % H
                    sel.setdefault((rows_[j].b, h), set()).add(cols["blk"][o])
        pool = self.pools[l]
        out = {}
        for r in vrow_objs:
            for h in range(H):
                if (r.b, h) in out:
                    continue
                st = pool.stream(r.b, h)
                sset = sel.get((r.b, h), set())
                out[(r.b, h)] = deque([("lru", blk, idx, None) for blk, idx in st.lru.items()
                                       if not st.prot.get(blk) and blk not in sset]
                                      + self._inflight_candidates(l, st, sset))
        return out

    def _apply_verify(self, l: int, vrow_objs: List[Row], vidx: List[int], dec: VerifyDecision, c, drow_objs=None,
                      dcols=None, ddec=None) -> None:
        """The host applies one layer's verify decisions in v1's order (_verify_part under defer_admit), checking each
        against the host state; a Policy E eviction must be the host's next candidate of its stream."""
        H, k = dec.H, dec.k
        pool = self.pools[l]
        now = self.clock()
        looks, entries = [], {}
        cands = self._host_candidates(l, vrow_objs, dec, c, drow_objs, dcols, ddec) if any(c["evict"]) else {}
        use0, n_first = {}, {}                                # per stream: usage at the layer's start, first misses
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
                    elif c["granted"][o] or c["evict"][o]:
                        new.append(t)
                        takes.append((t, c["take_slot"][o], c["evict_blk"][o] if c["evict"][o] else None,
                                      "inflight" if c["evict_infl"][o] else "lru"))
                    else:
                        raise OwnershipViolation("S-C: block %d of verify row %d (layer %d b %d h %d) has no slot and the "
                                                 "device did not flag an overflow" % (t, vidx[j], l, r.b, h))
                if takes:                                             # the K5 REPORT (_extra_need): the usage at
                    if (r.b, h) not in use0:                          # the layer's start (no take of this layer in
                        use0[(r.b, h)] = st.usage()                   # the stream yet), the first misses
                    n_first[(r.b, h)] = n_first.get((r.b, h), 0) + len(takes)
                for (t, at, ev_blk, ev_kind) in takes:
                    if ev_blk is not None:                            # Policy E: the host's next candidate, evicted
                        q = cands.get((r.b, h))
                        nxt = q.popleft() if q else None
                        if nxt is None or nxt[:3] != (ev_kind, ev_blk, at):
                            raise OwnershipViolation("S-C Policy E: the device evicted the %s block %d at slot %d (layer "
                                                     "%d b %d h %d); the host's next candidate is %s"
                                                     % (ev_kind, ev_blk, at, l, r.b, h, None if nxt is None else nxt[:3]))
                        if ev_kind == "lru":
                            self._evict_now(st, ev_blk, at)
                        else:                                         # the device waited for its write (12j)
                            self._evict_inflight(st, l, ev_blk, at, nxt[3])
                    tr = self._take(l, r.b, h, t, r.key, VERIFY, at=at, evicted=ev_blk is not None)
                    tr.delivered = True                               # the device's urgent gather landed it
                    self._landed.add(tr.seq)
                    ent.append((t, at))
                looks.append(dict(row=vidx[j], h=h, key=r.key, ready=ready, in_flight=infl, new=new, t=now))
                entries[(j, h)] = sorted(ent)
        self._note_extra_need(l, self._extra_need(use0, n_first))
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
