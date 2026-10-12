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
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Set

import torch

from .backend import ev_done


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

    def __post_init__(self):
        self.demand.add(self.kind)

    def tag(self):
        return (self.uid, self.block, self.slot, self.gen)


class S0Transport:
    name = "s0"

    def __init__(self, core, be, host_k, host_v, slots_k, slots_v, bs: int, D: int, H: int, B: int, n_pos: int,
                 S_host: int, cap_groups: int = 4, ring: int = 2, dtype=torch.int64):
        self.core, self.be = core, be
        self.host_k, self.host_v, self.slots_k, self.slots_v = host_k, host_v, slots_k, slots_v
        self.bs, self.D, self.H, self.B, self.n_pos, self.S_host = bs, D, H, B, n_pos, S_host
        elem = torch.empty((), dtype=dtype).element_size()
        nb = core.slot_bytes(cap_groups, bs, D, elem)
        self.stage = torch.zeros((ring, nb), dtype=torch.uint8)
        self.land = torch.zeros((ring, nb), dtype=torch.uint8)
        self.pipe = core.Pipe(be, self.stage, self.land, cap_groups, dtype=dtype, R=bs, D=D)
        self.staging_bytes = 2 * ring * nb
        self.n_layers_issued = 0
        self.chunks = 0

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
                 burst_p: float = 0.3, burst_len: int = 4):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.be = be
        self.host_k, self.host_v, self.slots_k, self.slots_v = host_k, host_v, slots_k, slots_v
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
