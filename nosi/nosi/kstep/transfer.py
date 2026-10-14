"""Transfers and lifetimes (spec section 6, T1-T4).

Every transfer carries (request uid, epoch-independent block id, destination slot, slot generation) (T3) plus the layer,
the stream (b, h), the kind (draft / verify) and two events:
  hist_ev    the D2H history write of the block's host rows (A5). The host pack waits it (T4: the pack reads host rows
             only after the write that produced them).
  reader_ev  the last reader event of the destination slot's PREVIOUS occupant. The copy into the slot waits it (T3).
A transport exposes issue(layer, transfers), poll() (called at the start of every joint call), done(tr) (non-blocking)
and force(tr) (blocking: a host wait). The dispatcher never assumes an arrival order or delay (T2): it polls each
transfer's OWN completion (not the in-order generation counter of nosi/nosi/spec_loop.py:162-167 / :199-228).

S0Transport (T1) CALLS the serial optimized feeder: benchmarks/Efficiency/cpupack_core.py build_desc (:227-257),
check_unique_dst (:260-263), views_for (:631-635) and Pipe.layer (:550-628: pack by index_select into a pinned staging
slot, ONE H2D per chunk on "copy", index_copy_ scatter on "scatter"). The cpupack_core module is INJECTED (`core`) so
that this package does not import benchmark code. The adapter adds what Pipe lacks (scout map 4.3): the destination
reader wait (a "scatter" stream wait on every previous occupant's reader event BEFORE Pipe.layer; the stream is in order,
so it gates every chunk of the layer), the host wait on the history writes, and the (uid, block, slot, gen) tag per group
(Desc carries b, h, m, blk only). A group completes with the scatter-end event sc1 of its chunk.

AdversarialTransport (T2b): seeded delays (in polls), reordering, bursts and complete-only-when-forced. The payload is
read from the host at issue (after the history write, T4) and lands in the slot only after the previous occupant's reader
event completed (a poll postpones; force() waits).

M2 ARENA (Config.physical = "arena"; retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md
revision 2, section 2.5; the transports get arena_k / arena_v, the per-layer (P * bs, D) arena rows). A Transfer carries
its destination position and generation (phys, pgen); reader_ev is the previous PHYSICAL occupant's last reader.
  S0Transport._issue_pool builds the layer's Desc DIRECTLY (no build_desc call; cpupack_core unchanged): groups in
  (b, h, slot) order; source row of in-block row r = row_base("orig") + H * r; destination row = phys * bs + r;
  packer = placer = "row"; R = bs; req_groups per batch slot; dst_layout "pool" (metadata: Pipe.layer consumes src_idx,
  dst_idx, req_groups, packer, placer and R). Before Pipe.layer, each check fails with a named "S0 pool descriptor"
  error: _check_pool_bounds (lower and upper bounds of every source / destination row, dtype and R agreement),
  cpupack_core.check_unique_dst, _check_pool_correspondence (group g IS transfer g: (b, h, slot, block) and exact rows
  recomputed from the transfer). The transfer keeps the host rows its pack read (LOGICAL indexing, at issue): readback()
  (called by the dispatcher in checked mode before admission) compares the landed arena rows with them, so a later
  request of the batch slot cannot change the expectation.
  AdversarialTransport._land writes the rows of position phys.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Set

import torch

from .backend import ev_done
from .errors import OwnershipViolation


@dataclass
class Transfer:
    seq: int
    uid: Any
    layer: int
    b: int
    h: int
    block: int
    slot: int
    gen: int
    kind: str                                   # "draft" | "verify": the kind of the request that issued it
    requester: Any                              # cohort key of the requester
    hist_ev: Any = None
    reader_ev: Any = None
    issued_call: int = -1
    ev: Any = None                              # S0: the chunk's scatter-end event
    delivered: bool = False                     # adversarial: payload landed
    payload: Any = None
    release_tick: int = 0
    demand: Set[str] = field(default_factory=set)  # repair 1 R2.3: every kind ("draft", "verify") that requested it
    phys: Any = None                            # M2 arena: the destination position (None per stream)
    pgen: Any = None                            # M2 arena: the position's generation of this binding
    expect: Any = None                          # M2 S0: (K rows, V rows) the pack read at issue (readback)

    def __post_init__(self):
        self.demand.add(self.kind)

    def tag(self):
        return (self.uid, self.block, self.slot, self.gen)


class S0Transport:
    name = "s0"

    def __init__(self, core, be, host_k, host_v, slots_k, slots_v, bs: int, D: int, H: int, B: int, n_pos: int,
                 S_host: int, cap_groups: int = 4, ring: int = 2, dtype=torch.int64, arena_k=None, arena_v=None,
                 P: int = None):
        self.core, self.be = core, be
        self.host_k, self.host_v, self.slots_k, self.slots_v = host_k, host_v, slots_k, slots_v
        self.bs, self.D, self.H, self.B, self.n_pos, self.S_host = bs, D, H, B, n_pos, S_host
        if (arena_k is None) != (arena_v is None) or (arena_k is not None and P is None):
            raise ValueError("M2 arena: pass arena_k, arena_v and P together")
        self.arena_k, self.arena_v, self.P = arena_k, arena_v, P   # M2 arena rows per layer, (P * bs, D)
        elem = torch.empty((), dtype=dtype).element_size()
        nb = core.slot_bytes(cap_groups, bs, D, elem)
        self.stage = torch.zeros((ring, nb), dtype=torch.uint8)
        self.land = torch.zeros((ring, nb), dtype=torch.uint8)
        self.pipe = core.Pipe(be, self.stage, self.land, cap_groups, dtype=dtype, R=bs, D=D)
        self.staging_bytes = 2 * ring * nb
        self.n_layers_issued = 0
        self.chunks = 0
        self.pool_checks = 0                      # M2: transfers whose pool descriptor group passed every adapter check
        self.readbacks = 0                        # M2: landed transfers whose arena rows were read back equal

    def issue(self, layer: int, transfers: List[Transfer]) -> None:
        if not transfers:
            return
        be, core = self.be, self.core
        for tr in transfers:                      # T4: the host rows were written (A5 D2H) before the pack reads them
            if tr.hist_ev is not None:
                be.host_wait(tr.hist_ev)
        for tr in transfers:                      # T3: the scatter writes a slot only after its previous reader
            if tr.reader_ev is not None:
                be.stream_wait("scatter", tr.reader_ev)
        if self.arena_k is not None:
            self._issue_pool(layer, transfers)
            return
        plan = torch.full((self.H, self.B, self.n_pos), -1, dtype=torch.int64)
        by_dst = {}
        for tr in transfers:
            if plan[tr.h, tr.b, tr.slot] >= 0:
                raise ValueError("two transfers into one destination (b %d h %d slot %d)" % (tr.b, tr.h, tr.slot))
            plan[tr.h, tr.b, tr.slot] = tr.block
            by_dst[(tr.b, tr.h, tr.slot)] = tr
        desc = core.build_desc(plan, src_layout="orig", dst_layout="orig", s_src=self.S_host, s_dst=self.n_pos * self.bs,
                               packer="row", placer="row", R=self.bs, tail_slot=None)
        core.check_unique_dst(desc)
        sk, dk = core.views_for(desc, self.host_k[layer], self.slots_k[layer], R=self.bs)
        sv, dv = core.views_for(desc, self.host_v[layer], self.slots_v[layer], R=self.bs)
        recs = self.pipe.layer(desc, sk, sv, dk, dv)
        for rec in recs:
            for g in range(rec["g0"], rec["g1"]):
                key = (int(desc.b[g]), int(desc.h[g]), int(desc.m[g]))
                tr = by_dst.pop(key)
                if int(desc.blk[g]) != tr.block:
                    raise AssertionError("descriptor group %d carries block %d, transfer %s" % (g, int(desc.blk[g]), tr.tag()))
                tr.ev = rec["sc1"]
        if by_dst:
            raise AssertionError("transfers without a descriptor group: %s" % [t.tag() for t in by_dst.values()])
        self.n_layers_issued += 1
        self.chunks += len(recs)

    # ---------------------------------------------------------------- M2: the pool-addressed descriptor (contract 2.5)
    def _issue_pool(self, layer: int, transfers: List[Transfer]) -> None:
        core, bs, H = self.core, self.bs, self.H
        order = sorted(transfers, key=lambda t: (t.b, t.h, t.slot))
        for t0, t1 in zip(order, order[1:]):
            if (t0.b, t0.h, t0.slot) == (t1.b, t1.h, t1.slot):
                raise ValueError("two transfers into one destination (b %d h %d slot %d)" % (t0.b, t0.h, t0.slot))
        b = torch.tensor([t.b for t in order], dtype=torch.int64)
        h = torch.tensor([t.h for t in order], dtype=torch.int64)
        m = torch.tensor([t.slot for t in order], dtype=torch.int64)
        blk = torch.tensor([t.block for t in order], dtype=torch.int64)
        phys = torch.tensor([t.phys for t in order], dtype=torch.int64)
        ar = torch.arange(bs, dtype=torch.int64)
        base, step = core.row_base("orig", b, h, blk, self.S_host, H, bs)
        src = (base[:, None] + step * ar[None, :]).reshape(-1)
        dst = (phys[:, None] * bs + ar[None, :]).reshape(-1)
        desc = core.Desc(n=len(order), B=self.B, b=b, h=h, m=m, blk=blk, req_groups=torch.bincount(b, minlength=self.B),
                         src_layout="orig", dst_layout="pool", packer="row", placer="row", src_idx=src, dst_idx=dst, R=bs)
        self._check_pool_bounds(layer, desc)
        core.check_unique_dst(desc)
        self._check_pool_correspondence(layer, desc, order)
        for tr in order:                          # the expected payload, read now (the pack reads the host rows now)
            r0 = tr.block * bs
            tr.expect = (self.host_k[layer][tr.b, r0:r0 + bs, tr.h].clone(), self.host_v[layer][tr.b, r0:r0 + bs, tr.h].clone())
        recs = self.pipe.layer(desc, core.rows2d(self.host_k[layer]), core.rows2d(self.host_v[layer]), self.arena_k[layer],
                               self.arena_v[layer])
        n = 0
        for rec in recs:
            for g in range(rec["g0"], rec["g1"]):
                order[g].ev = rec["sc1"]
                n += 1
        if n != len(order):
            raise OwnershipViolation("S0 pool descriptor: %d of %d groups of layer %d were scattered" % (n, len(order), layer))
        self.n_layers_issued += 1
        self.chunks += len(recs)
        self.pool_checks += len(order)

    def _check_pool_bounds(self, layer: int, desc) -> None:
        """Lower and upper bounds of every source row (< B * S_host * H) and destination row (< P * bs), the row counts,
        and dtype / R agreement of the descriptor, the host rows, the arena rows and the Pipe."""
        bs, D, dt = self.bs, self.D, self.pipe.dtype
        hk, hv, ak, av = self.host_k[layer], self.host_v[layer], self.arena_k[layer], self.arena_v[layer]
        if desc.R != bs or self.pipe.R != bs or desc.src_idx.numel() != desc.n * bs or desc.dst_idx.numel() != desc.n * bs:
            raise OwnershipViolation("S0 pool descriptor: R mismatch at layer %d (desc R %d, pipe R %d, bs %d; %d source and "
                                     "%d destination rows for %d groups)" % (layer, desc.R, self.pipe.R, bs,
                                                                             desc.src_idx.numel(), desc.dst_idx.numel(), desc.n))
        if not (hk.dtype == hv.dtype == ak.dtype == av.dtype == dt) or tuple(ak.shape) != (self.P * bs, D) \
                or tuple(av.shape) != (self.P * bs, D):
            raise OwnershipViolation("S0 pool descriptor: dtype / shape mismatch at layer %d (host %s %s, arena %s %s %s %s, "
                                     "pipe %s; want arena (%d, %d))" % (layer, hk.dtype, hv.dtype, ak.dtype, av.dtype,
                                                                        tuple(ak.shape), tuple(av.shape), dt, self.P * bs, D))
        n_src, n_dst = hk.numel() // D, self.P * bs
        if desc.n:
            lo, hi = int(desc.src_idx.min()), int(desc.src_idx.max())
            if lo < 0 or hi >= n_src:
                raise OwnershipViolation("S0 pool descriptor: source row out of bounds at layer %d: [%d, %d] not in [0, %d)"
                                         % (layer, lo, hi, n_src))
            lo, hi = int(desc.dst_idx.min()), int(desc.dst_idx.max())
            if lo < 0 or hi >= n_dst:
                raise OwnershipViolation("S0 pool descriptor: destination row out of bounds at layer %d: [%d, %d] not in "
                                         "[0, %d)" % (layer, lo, hi, n_dst))

    def _check_pool_correspondence(self, layer: int, desc, order: List[Transfer]) -> None:
        """Group g IS transfer g (strict (b, h, slot) order; per batch slot request groups): its (b, h, slot, block) and
        its exact destination rows phys * bs + r and source rows (b * S_host + block * bs + r) * H + h, recomputed from
        the transfer's own fields (not from the construction above)."""
        bs, H, S = self.bs, self.H, self.S_host
        ar = torch.arange(bs, dtype=torch.int64)
        want_req = [sum(1 for t in order if t.b == bb) for bb in range(self.B)]
        if desc.n != len(order) or desc.req_groups.tolist() != want_req:
            raise OwnershipViolation("S0 pool descriptor: address mismatch at layer %d: %d groups, request groups %s for %d "
                                     "transfers (want %s)" % (layer, desc.n, desc.req_groups.tolist(), len(order), want_req))
        prev = None
        for g, tr in enumerate(order):
            who = "transfer seq %d (uid %s, l %d b %d h %d block %d) at position %s pgen %s" % (
                tr.seq, tr.uid, tr.layer, tr.b, tr.h, tr.block, tr.phys, tr.pgen)
            key = (tr.b, tr.h, tr.slot)
            if tr.layer != layer or (prev is not None and key <= prev):
                raise OwnershipViolation("S0 pool descriptor: address mismatch: %s is group %d of layer %d out of (b, h, slot) "
                                         "order" % (who, g, layer))
            prev = key
            got = (int(desc.b[g]), int(desc.h[g]), int(desc.m[g]), int(desc.blk[g]))
            if got != (tr.b, tr.h, tr.slot, tr.block):
                raise OwnershipViolation("S0 pool descriptor: address mismatch: %s, but group %d names (b, h, slot, block) %s"
                                         % (who, g, got))
            rows, want = desc.dst_idx[g * bs:(g + 1) * bs], tr.phys * bs + ar
            if not torch.equal(rows, want):
                raise OwnershipViolation("S0 pool descriptor: address mismatch: %s: group %d destination rows %s, want %s"
                                         % (who, g, rows.tolist(), want.tolist()))
            rows, want = desc.src_idx[g * bs:(g + 1) * bs], (tr.b * S + tr.block * bs + ar) * H + tr.h
            if not torch.equal(rows, want):
                raise OwnershipViolation("S0 pool descriptor: address mismatch: %s: group %d source rows %s, want %s"
                                         % (who, g, rows.tolist(), want.tolist()))

    def readback(self, tr: Transfer) -> None:
        """M2 checked mode (the dispatcher calls it when the transfer is reported complete, before admission): the arena
        rows of the transfer's position equal the host rows its pack read at issue (kept on THIS transfer)."""
        if self.arena_k is None:
            return
        if tr.expect is None:
            raise OwnershipViolation("S0 pool descriptor: payload readback of transfer seq %d without its issue payload"
                                     % tr.seq)
        p0 = tr.phys * self.bs
        k, v = self.arena_k[tr.layer][p0:p0 + self.bs], self.arena_v[tr.layer][p0:p0 + self.bs]
        if not (torch.equal(k, tr.expect[0]) and torch.equal(v, tr.expect[1])):
            raise OwnershipViolation("S0 pool descriptor: payload readback mismatch: transfer seq %d (uid %s, l %d b %d h %d "
                                     "block %d) at position %d pgen %s: the arena rows differ from the rows its pack read"
                                     % (tr.seq, tr.uid, tr.layer, tr.b, tr.h, tr.block, tr.phys, tr.pgen))
        tr.expect = None
        self.readbacks += 1

    def poll(self) -> None:
        pass                                      # completion is the event itself

    def done(self, tr: Transfer) -> bool:
        return ev_done(tr.ev)

    def force(self, tr: Transfer) -> None:
        self.be.host_wait(tr.ev)


class AdversarialTransport:
    name = "adversarial"
    MODES = ("delay", "force_only")

    def __init__(self, be, host_k, host_v, slots_k, slots_v, bs: int, seed: int, mode: str = "delay", max_delay: int = 6,
                 burst_p: float = 0.3, burst_len: int = 4, arena_k=None, arena_v=None):
        if mode not in self.MODES:
            raise ValueError(mode)
        if (arena_k is None) != (arena_v is None):
            raise ValueError("M2 arena: pass arena_k and arena_v together")
        self.be = be
        self.host_k, self.host_v, self.slots_k, self.slots_v = host_k, host_v, slots_k, slots_v
        self.arena_k, self.arena_v = arena_k, arena_v               # M2 arena rows per layer, (P * bs, D)
        self.bs, self.mode, self.max_delay = bs, mode, int(max_delay)
        self.rng = random.Random(seed)
        self.burst_p, self.burst_len = burst_p, burst_len
        self.pending: List[Transfer] = []
        self.now = 0
        self.hold_until = -1
        self.delivered = 0
        self.postponed = 0
        self.staging_bytes = 0

    def issue(self, layer: int, transfers: List[Transfer]) -> None:
        for tr in transfers:
            if tr.hist_ev is not None:
                self.be.host_wait(tr.hist_ev)                    # T4
            r0, r1 = tr.block * self.bs, (tr.block + 1) * self.bs
            tr.payload = (self.host_k[layer][tr.b, r0:r1, tr.h].clone(), self.host_v[layer][tr.b, r0:r1, tr.h].clone())
            tr.release_tick = self.now + self.rng.randint(0, self.max_delay)
            self.pending.append(tr)

    def _land(self, tr: Transfer) -> None:
        if self.arena_k is not None:                             # M2: the rows of the transfer's position
            p0, p1 = tr.phys * self.bs, (tr.phys + 1) * self.bs
            self.arena_k[tr.layer][p0:p1].copy_(tr.payload[0])
            self.arena_v[tr.layer][p0:p1].copy_(tr.payload[1])
        else:
            s0, s1 = tr.slot * self.bs, (tr.slot + 1) * self.bs
            self.slots_k[tr.layer][tr.b, s0:s1, tr.h].copy_(tr.payload[0])
            self.slots_v[tr.layer][tr.b, s0:s1, tr.h].copy_(tr.payload[1])
        tr.delivered = True
        tr.payload = None
        self.delivered += 1

    def poll(self) -> None:
        """One tick: deliver some eligible transfers in a seeded random order (reordering); bursts hold every delivery
        for burst_len ticks, then release them together; force_only delivers nothing here."""
        self.now += 1
        if self.mode == "force_only" or not self.pending:
            return
        if self.now <= self.hold_until:
            return
        if self.rng.random() < self.burst_p:
            self.hold_until = self.now + self.burst_len
            return
        cand = [t for t in self.pending if t.release_tick <= self.now]
        self.rng.shuffle(cand)
        for tr in cand:
            if self.rng.random() < 0.25:
                continue                                         # random extra delay
            if not ev_done(tr.reader_ev):
                self.postponed += 1                              # the copy may not land before the previous reader
                continue
            self._land(tr)
            self.pending.remove(tr)

    def done(self, tr: Transfer) -> bool:
        return tr.delivered

    def force(self, tr: Transfer) -> None:
        if tr.delivered:
            return
        if tr.reader_ev is not None:
            self.be.host_wait(tr.reader_ev)
        self._land(tr)
        self.pending.remove(tr)
