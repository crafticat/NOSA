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

C1.1 ABI v4 (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revision 10, 12.3 A5;
arena only; host_c / arena_c given together, carries_bias = True). The per-position bias travels with its K / V rows:
  S0Transport._issue_bias, after Pipe.layer of the same layer issue: the host pack of every group's bias rows at issue
  (index_select by the descriptor's own src_idx, after the same T4 host waits and the dispatcher's hown check; a fresh
  staging tensor), ONE H2D on "copy" into a fresh landing tensor, then ONE index_copy on "scatter" (a stream wait on that
  H2D) to the descriptor's own dst_idx, and an event on "scatter" after it. "scatter" is ordered and already waited every
  destination's previous reader (T3) before Pipe.layer, and the bias copy follows every chunk's K / V scatter, so that
  event is the layer issue's completion frontier: EVERY transfer of the issue completes on it (Transfer.ev), never on
  its chunk's sc1 alone. The readback compares the bias rows as well ("... the arena bias rows differ ..."). The fresh
  staging and landing tensors of EVERY issue are owned by the dispatcher's ownership.BiasOpLedger (kinds s0_stage /
  s0_land) until that issue's frontier event is seen complete (a passive check), so simultaneously held issues all
  count (K6 category bias_staging; the ledger's exact peak).
  AdversarialTransport: the payload read at issue holds the bias rows (ledger kind adv_payload, held until _land drops
  the payload), and _land writes K, V and bias together before the transfer counts as delivered. Transfer.expect's
  bias rows are a diagnostic readback copy (checked mode), not an operational backing.

HOOK-L6 (INTERFACE_BC5 I11, lead, 2026-10-17; Config.store_gen "consumer" only): S0Transport.gen_publisher (None by
default: every path unchanged). TensorStore.attach_gen_mirrors installs it. _issue_pool calls
ev = gen_publisher(layer, order, "scatter") after Pipe.layer and _issue_bias: the store writes each transfer's
generation (phys, pgen) into its consumer mirror on "scatter" (which already waited every destination's previous
reader, T3) and records ev after those writes; every transfer of the issue then completes on ev. AdversarialTransport
has no gen_publisher; the dispatcher refuses store_gen "consumer" with it.
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
    expect: Any = None                          # M2 S0: (K rows, V rows[, bias rows (ABI v4)]) the pack read at issue

    def __post_init__(self):
        self.demand.add(self.kind)

    def tag(self):
        return (self.uid, self.block, self.slot, self.gen)


class S0Transport:
    name = "s0"
    accepts_gen_publisher = True                  # HOOK-L6 (I11): Config.store_gen "consumer" may install gen_publisher

    def __init__(self, core, be, host_k, host_v, slots_k, slots_v, bs: int, D: int, H: int, B: int, n_pos: int,
                 S_host: int, cap_groups: int = 4, ring: int = 2, dtype=torch.int64, arena_k=None, arena_v=None,
                 P: int = None, host_c=None, arena_c=None, bias_ledger=None):
        self.core, self.be = core, be
        self.host_k, self.host_v, self.slots_k, self.slots_v = host_k, host_v, slots_k, slots_v
        self.bs, self.D, self.H, self.B, self.n_pos, self.S_host = bs, D, H, B, n_pos, S_host
        if (arena_k is None) != (arena_v is None) or (arena_k is not None and P is None):
            raise ValueError("M2 arena: pass arena_k, arena_v and P together")
        self.arena_k, self.arena_v, self.P = arena_k, arena_v, P   # M2 arena rows per layer, (P * bs, D)
        if (host_c is None) != (arena_c is None) or (arena_c is not None and (arena_k is None or bias_ledger is None)):
            raise ValueError("C1.1 ABI v4: pass host_c, arena_c and the dispatcher's bias_ledger together, in arena mode "
                             "only")
        self.host_c, self.arena_c = host_c, arena_c                 # C1.1: (B, S_host, H) and (P * bs,) per layer
        self.bias_ledger = bias_ledger                              # C1.1: owns each issue's bias stage / landing
        self.carries_bias = arena_c is not None
        self.gen_publisher = None                 # HOOK-L6 (I11): set by TensorStore.attach_gen_mirrors (arena only)
        self.bias_readbacks = 0                   # C1.1: readbacks that compared the bias rows too
        self.bias_groups = 0                      # C1.1: groups whose bias rows joined their issue's frontier
        self._check_construction_dtypes(dtype)    # HOOK-B3: every store against dtype, BEFORE any staging or Pipe
        elem = torch.empty((), dtype=dtype).element_size()
        nb = core.slot_bytes(cap_groups, bs, D, elem)
        self._check_slot_alignment(nb, ring, cap_groups, elem)   # HOOK-B3: BEFORE the staging buffers and the Pipe
        # I24 (milestone 10, found by the first GPU run, K3X G1 2188818.9): the landing slots live on the device of the
        # stores the scatter writes (arena or slots), and the host staging is pinned when that device is a GPU (the H2D
        # is asynchronous on "copy"). On the CPU both stay ordinary host tensors, exactly as before.
        ref = arena_k if arena_k is not None else slots_k
        self.device = ref[0].device if ref else torch.device("cpu")
        pin = self.device.type == "cuda"
        self.stage = torch.zeros((ring, nb), dtype=torch.uint8, pin_memory=pin)
        self.land = torch.zeros((ring, nb), dtype=torch.uint8, device=self.device)
        self.pipe = core.Pipe(be, self.stage, self.land, cap_groups, dtype=dtype, R=bs, D=D)
        self.staging_bytes = 2 * ring * nb
        self.n_layers_issued = 0
        self.chunks = 0
        self.pool_checks = 0                      # M2: transfers whose pool descriptor group passed every adapter check
        self.readbacks = 0                        # M2: landed transfers whose arena rows were read back equal

    @staticmethod                                                       # HOOK-B3
    def _dtype_text(ts) -> str:                                         # HOOK-B3
        """HOOK-B3: the dtype of one store list (one per layer): one dtype, or the per-layer list when they differ."""
        dts = [t.dtype for t in ts]                                     # HOOK-B3
        if dts and all(x == dts[0] for x in dts):                       # HOOK-B3
            return str(dts[0])                                          # HOOK-B3
        return "[%s]" % ", ".join(str(x) for x in dts)                  # HOOK-B3

    def _check_construction_dtypes(self, dtype) -> None:                # HOOK-B3
        """HOOK-B3 (I7, B / C interface addendum 3): at construction, BEFORE any staging buffer, Pipe or descriptor
        exists, every store that this transport reads or writes is checked: host_k / host_v and arena_k / arena_v (or
        slots_k / slots_v) against `dtype` (the Pipe's staging dtype), and host_c / arena_c (when present) against ONE
        bias dtype (that of host_c[0]; the bias may differ from K / V). A mismatch raises OwnershipViolation
        "S0 transport: dtype mismatch at construction (dtype D; <store> <dtype>, ...; mismatched <stores>)", which names
        every present store with its dtype (per layer when the layers differ). The issue-time _check_pool_bounds check
        stays as it is (defence in depth). (HOOK-B3)"""
        names_kv = ("host_k", "host_v", "arena_k", "arena_v", "slots_k", "slots_v")         # HOOK-B3
        kv = [(n, getattr(self, n)) for n in names_kv if getattr(self, n) is not None]      # HOOK-B3
        bias = [(n, getattr(self, n)) for n in ("host_c", "arena_c") if getattr(self, n) is not None]   # HOOK-B3
        bad = [n for n, ts in kv if any(t.dtype != dtype for t in ts)]                      # HOOK-B3
        bias_dts = [t.dtype for _, ts in bias for t in ts]                                  # HOOK-B3
        if bias_dts and any(x != bias_dts[0] for x in bias_dts):                            # HOOK-B3
            bad.append("host_c / arena_c")                                                  # HOOK-B3
        if bad:                                                                             # HOOK-B3
            stores = ", ".join("%s %s" % (n, self._dtype_text(ts)) for n, ts in kv + bias)   # HOOK-B3
            msg = "S0 transport: dtype mismatch at construction (dtype %s; %s; mismatched %s)"   # HOOK-B3
            raise OwnershipViolation(msg % (dtype, stores, ", ".join(bad)))                 # HOOK-B3

    def _check_slot_alignment(self, nb: int, ring: int, cap: int, elem: int) -> None:   # HOOK-B3
        """HOOK-B3 (I7): ring slot s starts at byte s * nb of the staging and landing buffers. The Pipe views the index
        region of each slot as int64 (cpupack_core.stage_views), and that view needs a start byte that is a multiple of
        the index size (cpupack_core.IDX_BYTES, 8). So with ring > 1, nb must be a multiple of it. Else
        OwnershipViolation "S0 transport: staging slot of <nb> bytes is not a multiple of 8 (ring R, cap C, bs B, D D,
        element E); the int64 index view of ring slot 1 needs it", after slot_bytes and BEFORE any staging buffer or
        Pipe exists. No padding is added. (HOOK-B3)"""
        ib = getattr(self.core, "IDX_BYTES", 8)                                             # HOOK-B3
        if ring > 1 and nb % ib:                                                            # HOOK-B3
            msg = ("S0 transport: staging slot of %d bytes is not a multiple of %d (ring %d, cap %d, "   # HOOK-B3
                   "bs %d, D %d, element %d); the int64 index view of ring slot 1 needs it")           # HOOK-B3
            raise OwnershipViolation(msg % (nb, ib, ring, cap, self.bs, self.D, elem))      # HOOK-B3

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
            if self.arena_c is not None:          # C1.1: the bias rows the bias pack reads (the same host rows)
                tr.expect += (self.host_c[layer][tr.b, r0:r0 + bs, tr.h].clone(),)
        recs = self.pipe.layer(desc, core.rows2d(self.host_k[layer]), core.rows2d(self.host_v[layer]), self.arena_k[layer],
                               self.arena_v[layer])
        n = 0
        for rec in recs:
            for g in range(rec["g0"], rec["g1"]):
                order[g].ev = rec["sc1"]
                n += 1
        if n != len(order):
            raise OwnershipViolation("S0 pool descriptor: %d of %d groups of layer %d were scattered" % (n, len(order), layer))
        if self.arena_c is not None:
            self._issue_bias(layer, desc, order)  # C1.1 A5: the issue's completion frontier moves past the bias rows
        if self.gen_publisher is not None:        # HOOK-L6 (I11): the consumer generation mirrors of this issue, on
            ev = self.gen_publisher(layer, order, "scatter")   # "scatter" after every reader wait (issue) and the data
            for tr in order:                      # scatter: every transfer of the issue completes on that event, so
                tr.ev = ev                        # arrival (done) covers mirror visibility
        self.n_layers_issued += 1
        self.chunks += len(recs)
        self.pool_checks += len(order)

    @staticmethod
    def _bias_buffers(src, src_idx, dst_idx, device):
        """I24: the buffers of one bias issue: (stage, land, dst_host, dst). stage = the packed host rows (pinned when
        the device is a GPU, so the H2D is asynchronous); land = the landing rows on `device`; dst = the destination
        rows on `device`. On the CPU dst_host is None and dst is dst_idx itself (as before I24); on another device
        dst_host is the host index tensor the copy reads (pinned on a GPU) and dst a fresh tensor on the device."""
        dev = torch.device(device)
        pin = dev.type == "cuda"
        stage = torch.empty((src_idx.numel(),), dtype=src.dtype, pin_memory=pin)
        torch.index_select(src, 0, src_idx, out=stage)
        land = torch.empty(stage.shape, dtype=stage.dtype, device=dev)
        if dev.type == "cpu":
            return stage, land, None, dst_idx
        dst_host = dst_idx.pin_memory() if pin else dst_idx
        return stage, land, dst_host, torch.empty(dst_host.shape, dtype=dst_host.dtype, device=dev)

    def _issue_bias(self, layer: int, desc, order: List[Transfer]) -> None:
        """C1.1 A5 (module docstring): the bias rows of every group of this layer issue, after Pipe.layer: host pack at
        issue, one H2D on "copy", one scatter on "scatter" to the descriptor's own destination rows, then the frontier
        event on "scatter", which every transfer of the issue completes on."""
        be = self.be
        cdst = desc.dst_idx                       # C1.1 A5: each group's bias rows land where its K / V rows land
        hc, ac = self.host_c[layer], self.arena_c[layer]
        # the host pack, now (the K / V pack reads now); I24: the landing and the destination rows on the arena's device
        stage, land, cdst_h, cdst = self._bias_buffers(hc.reshape(-1), desc.src_idx, cdst, ac.device)
        on_dev = cdst_h is not None
        with be.stream("copy"):
            be.memcpy(land, stage, "copy")
            if on_dev:
                be.memcpy(cdst, cdst_h, "copy")   # I24: the destination rows travel with the bias rows (same H2D event)
            h2d = be.event_rec("copy")
        with be.stream("scatter"):
            be.stream_wait("scatter", h2d)
            be.index_copy(ac, cdst, land, "scatter")
            ev = be.event_rec("scatter")
        for tr in order:
            tr.ev = ev
        self.bias_groups += len(order)
        self.bias_ledger.hold("s0_stage", stage, ev=ev)          # C1.1 A9: both held until the issue's frontier ev
        self.bias_ledger.hold("s0_land", land, ev=ev)            # (the copy and the scatter that read them ran)
        if on_dev:                                               # I24: the index copies, held the same way
            self.bias_ledger.hold("s0_stage", cdst_h, ev=ev)
            self.bias_ledger.hold("s0_land", cdst, ev=ev)

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
        if self.arena_c is not None:              # C1.1: the bias rows share the descriptor's source / destination rows
            hc, ac = self.host_c[layer], self.arena_c[layer]
            if tuple(hc.shape) != (self.B, self.S_host, self.H) or tuple(ac.shape) != (self.P * bs,) \
                    or hc.dtype != ac.dtype or hc.numel() != hk.numel() // D:
                raise OwnershipViolation("S0 pool descriptor: bias shape / dtype mismatch at layer %d (host %s %s, arena %s "
                                         "%s; want host (%d, %d, %d), arena (%d,))"
                                         % (layer, tuple(hc.shape), hc.dtype, tuple(ac.shape), ac.dtype, self.B,
                                            self.S_host, self.H, self.P * bs))
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
        hd = tr.expect[0].device                  # I24: compared on the host (the transfer is complete: its event ran)
        k, v = k.to(hd), v.to(hd)
        if not (torch.equal(k, tr.expect[0]) and torch.equal(v, tr.expect[1])):
            raise OwnershipViolation("S0 pool descriptor: payload readback mismatch: transfer seq %d (uid %s, l %d b %d h %d "
                                     "block %d) at position %d pgen %s: the arena rows differ from the rows its pack read"
                                     % (tr.seq, tr.uid, tr.layer, tr.b, tr.h, tr.block, tr.phys, tr.pgen))
        if self.arena_c is not None:              # C1.1: the bias rows, after the K / V rows
            if len(tr.expect) != 3:
                raise OwnershipViolation("S0 pool descriptor: payload readback of transfer seq %d without its bias rows"
                                         % tr.seq)
            if not torch.equal(self.arena_c[tr.layer][p0:p0 + self.bs].to(tr.expect[2].device), tr.expect[2]):
                raise OwnershipViolation("S0 pool descriptor: payload readback mismatch: transfer seq %d (uid %s, l %d b %d "
                                         "h %d block %d) at position %d pgen %s: the arena bias rows differ from the bias "
                                         "rows its pack read" % (tr.seq, tr.uid, tr.layer, tr.b, tr.h, tr.block, tr.phys,
                                                                 tr.pgen))
            self.bias_readbacks += 1
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
                 burst_p: float = 0.3, burst_len: int = 4, arena_k=None, arena_v=None, host_c=None, arena_c=None,
                 bias_ledger=None):
        if mode not in self.MODES:
            raise ValueError(mode)
        if (arena_k is None) != (arena_v is None):
            raise ValueError("M2 arena: pass arena_k and arena_v together")
        if (host_c is None) != (arena_c is None) or (arena_c is not None and (arena_k is None or bias_ledger is None)):
            raise ValueError("C1.1 ABI v4: pass host_c, arena_c and the dispatcher's bias_ledger together, in arena mode "
                             "only")
        self.be = be
        self.host_k, self.host_v, self.slots_k, self.slots_v = host_k, host_v, slots_k, slots_v
        self.arena_k, self.arena_v = arena_k, arena_v               # M2 arena rows per layer, (P * bs, D)
        self.host_c, self.arena_c = host_c, arena_c                 # C1.1: (B, S_host, H) and (P * bs,) per layer
        self.bias_ledger = bias_ledger                              # C1.1: owns each payload's bias rows to landing
        self.carries_bias = arena_c is not None
        self.bias_lands = 0                                         # C1.1: landings that wrote the bias rows too
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
            if self.arena_c is not None:                         # C1.1: the bias rows, read with K / V at issue
                tr.payload += (self.host_c[layer][tr.b, r0:r1, tr.h].clone(),)
                self.bias_ledger.hold("adv_payload", tr.payload[2], key=tr.seq)   # A9: held until _land drops it
            tr.release_tick = self.now + self.rng.randint(0, self.max_delay)
            self.pending.append(tr)

    def _land(self, tr: Transfer) -> None:
        if self.arena_k is not None:                             # M2: the rows of the transfer's position
            p0, p1 = tr.phys * self.bs, (tr.phys + 1) * self.bs
            self.arena_k[tr.layer][p0:p1].copy_(tr.payload[0])
            self.arena_v[tr.layer][p0:p1].copy_(tr.payload[1])
            if self.arena_c is not None:                         # C1.1: K, V and bias land together, then delivered
                self.arena_c[tr.layer][p0:p1].copy_(tr.payload[2])
                self.bias_lands += 1
                self.bias_ledger.release_key("adv_payload", tr.seq)   # A9: the payload is dropped just below
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
