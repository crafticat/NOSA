"""KV ownership (spec section 5, K1-K6) of the historical blocks of every stream (layer l, KV head h, batch slot b).

STATES of a physical slot (one block per slot; ONE slot per resident block, K3):
  FREE        in no LRU, no protection, not in flight, last reader event complete (K4)
  INFLIGHT    destination of a transfer; never reassigned while in flight (T3)
  LRU         resident, a member of the stream's logical LRU (capacity C, K1)
  PO          protected-only: evicted from the LRU while protected; keeps its slot (K3)
  VRES        verifier resident (repair 1, R2.4): the arrival of a transfer whose demand set is {verify} only; resident,
              OUTSIDE the LRU, protected (K2); released when its last reference is dropped (R1)
  RELEASING   left every role; becomes FREE when its LAST reader event has completed (host poll, K4)
Physical slots in use per stream = |LRU| + |PO| + |VRES| + |INFLIGHT| (K3); RELEASING slots are reported separately.

K1  (amended by repair 1, R2: the logical LRU is DRAFT-ONLY.) Recency changes only on draft events: (a) a draft row's
    read of an LRU entry moves it to MRU (Stream.touch); (b) the arrival of a transfer whose demand set contains
    "draft" enters the LRU at MRU (Stream.admit at the host poll); (c) a draft row's read of a resident non-LRU entry
    (PO or VRES) enters the LRU at MRU (Stream.acquire: "draft acquisition", no transfer, no new slot). After (b) and
    (c) the least recent entry leaves when the LRU holds more than C entries (protected: PO, else released through the
    reader gate). Verifier reads never touch the LRU (R2.2). The recency rule is the HitProtectedLRU twin's
    (retroinfer-eval hisparse-native-eval d7e3149 scripts/hisparse_native/reproduce.py:55-76: hits to MRU) extended
    across calls; HiSparse @87db743 hisparse.cuh:640-672 is the upstream rule.
K2  protection is a per-(block, cohort) reference of the stream (the block's single slot carries it, K3). A block may be
    protected while it has no slot (a refused or pending miss); it then pins the slot it gets.
K4  (amended by repair 1, R1.) The dispatcher drops a cohort's references at layer l after the joint call in which the
    cohort executed layer l, or at its decision for the layers it did not execute, and in both cases only after the
    last reader event of (cohort, l) completed. Stream.unprotect drops ONE key. A slot becomes FREE only after its own
    last reader event completed.
K5  per layer: C slots per stream plus X shared extra slots. The extra pool is a per-layer COUNT: a stream whose usage
    reaches C borrows from it. A draft fetch may borrow only while more than R_ver extra slots are free (the verifier
    reserve); a verifier fetch may borrow any free extra slot. A refused draft fetch is counted and the draft continues
    masked. M1 backs every stream with C + X physical slot positions so that cpupack_core's per-(b, h, slot) addressing
    applies unchanged; the logical limit of X extra slots per layer is enforced here (M2 needs a pool-addressed
    descriptor for a physically shared pool).
The slot + generation pattern and the error names follow benchmarks/Efficiency/feeder_core.py:266-349 (SlotTable, a
staging-ring state machine; reused as a pattern only).

CAPACITY MODE "rev6" (retroinfer-eval PROPOSAL rev 6, section 8; the dispatcher enforces it, Config.capacity):
  CLASS  every taken slot carries cls "ad" (the request's draft classes A / D) or "v" (the verifier reserve) until it is
         FREE (L1). A and D are a COUNT split, not two tags: of a stream's "ad" positions the first C are A and the rest
         are D, charged to the owner's reservation (L4). A verifier residual fetch is "v"; a draft fetch is "ad"; a draft
         use of a "v" slot re-tags it "ad" (the atomic V -> A / D conversion; the dispatcher takes a step credit first).
  P      Stream.positions(): every non-FREE "ad" slot, an unpinned LRU entry and a RELEASING slot included (L2: a slot
         counts until it is FREE). Stream.v_slots() / LayerPool.v_used(): the non-FREE "v" slots.
  KIND   provenance of a reference (pin_kind: block -> key -> {"draft", "verify"}), recorded when the caller names it;
         an eviction reports the kinds that held the entry (L9: a verifier pin makes it PO instead of RELEASING).
In mode "m1" the class and kind are recorded too, but nothing checks them.

M2 PHYSICAL ARENA (Config.physical = "arena"; retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-
contract.md revision 2, sections 2.1-2.6; default "per_stream" keeps every rule above unchanged). One Arena per layer
holds the records of P_l head-agnostic block positions (the conservative P_l = B * H * (C + X) only; the reduced
B * H * C + X is step S6, not approved). The logical slots, maps, protections, classes, credits and grants above are
unchanged; a Slot gains its binding (phys, pgen).
  BIND    Stream.take, FREE -> INFLIGHT, as one step after every check: the LOWEST FREE position p; capture p's last
          reader (the previous PHYSICAL occupant's, Arena.captured, forwarded once as Transfer.reader_ev); pgen + 1 (and
          the mirror pgen_t); bound = (b, h, idx, gen); then the new binding's reader state starts empty (position and
          slot last_reader, slot first_reader).
  READ    the dispatcher installs the call's read event on the slot AND on its position (Arena.note_reader).
  UNBIND  only in LayerPool.poll_releases, with the logical FREE transition (the FREE gate). The position keeps its final
          reader as physical history for the next bind.
  CHECKS  (Arena.checked, set by the dispatcher in checked mode; named "arena binding: ...") after every bind for the
          bound slot, after every release / dropped arrival for the slot it moved, at every unbind, and the full scan
          (LayerPool.check_arena: 1:1 binding, pgen_t = pgen, bound + E + G <= P_l) at the end of every joint call.

M2 B2 RING (Config.recent = "ring"; retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md
revisions 4-5, section 11; default "eager" keeps every rule above unchanged). One RingPool per layer holds the records of
Q_l entries; one entry holds the K row and the V row of ONE (head, position) of one request (the rows live in the
dispatcher's ring tensors, row e = entry e). Per entry e:
  state         FREE | LIVE (logically held) | DRAINING (logically released, waiting for its gates)
  kind          exact | draft | pre | prov (pre -> prov and prov -> exact are re-tags of the same entry)
  owner, prov   (uid, b, h) and (epoch, cohort key or None, position)
  rgen[e]       +1 at every bind; rgen_t[e] is its int64 mirror (the map entry a deferred read copies at execution)
  ready, cev, dev   the producer's write event, the latest COMPUTE reader event, the latest D2H reader event: two
                separate gates on two ordered streams (no single latest event is used); kept after FREE as history
  BIND    RingPool.bind, as one step after every check (bind_check): the LOWEST FREE entry, rgen + 1 (and rgen_t), the
          owner and provenance, empty reader state. The dispatcher's maps, payload write and ready event follow.
  FREE    only in RingPool.retire (the dispatcher's host poll, beside poll_releases, and the bind's backpressure step):
          a DRAINING entry whose ready, compute and D2H events are all complete (non-blocking ev_done).
  CHECKS  named "ring binding: ..." (bind_check before any change; reader installs and re-tags name the binding rgen).
  CHECKPOINTS (2026-10-14 steering): checkpoint 1 added these records and the ring-mode K6 accountant; checkpoint 2
          (the dispatcher's Config.recent "ring") creates and uses them.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .backend import ev_done
from .errors import ConfigError, OwnershipViolation, StaleGeneration
from .executor import DRAFT

FREE, INFLIGHT, LRU, PO, VRES, RELEASING = "free", "inflight", "lru", "po", "vres", "releasing"
STATES = (FREE, INFLIGHT, LRU, PO, VRES, RELEASING)
LIVE, DRAINING = "live", "draining"                                 # M2 B2 ring entry states (with FREE)
R_EXACT, R_DRAFT, R_PRE, R_PROV = "exact", "draft", "pre", "prov"  # M2 B2 ring entry kinds
RING_KINDS = (R_EXACT, R_DRAFT, R_PRE, R_PROV)


class Slot:
    __slots__ = ("idx", "state", "block", "gen", "owner", "first_reader", "last_reader", "seq", "cls", "phys", "pgen")

    def __init__(self, idx: int):
        self.idx = idx
        self.state = FREE
        self.block = -1
        self.gen = 0
        self.owner = None
        self.first_reader = None
        self.last_reader = None
        self.seq = None                      # sequence id of the transfer in flight
        self.cls = None                      # "ad" | "v" while not FREE (capacity mode rev6, L1)
        self.phys = None                     # M2 arena: the bound position while not FREE (None per stream)
        self.pgen = None                     # M2 arena: the position's generation of this binding


class Stream:
    """One (layer, head, batch slot) stream."""

    def __init__(self, l: int, h: int, b: int, n_pos: int, C: int):
        self.l, self.h, self.b, self.C = l, h, b, C
        self.slots = [Slot(i) for i in range(n_pos)]
        self.lru: "OrderedDict[int, int]" = OrderedDict()   # block -> slot; first = least recent, last = MRU
        self.po: Dict[int, int] = {}                         # block -> slot (protected-only)
        self.vres: Dict[int, int] = {}                       # block -> slot (verifier resident, R2.4)
        self.pending: Dict[int, int] = {}                    # block -> slot of the CURRENT owner's transfers in flight
        self.prot: Dict[int, Set[Any]] = {}                  # block -> cohort keys (K2)
        self.pin_kind: Dict[int, Dict[Any, Set[str]]] = {}   # block -> key -> {"draft", "verify"} (provenance, L9)
        self.owner = None                                    # request uid that owns the batch slot now
        self.log: List[Tuple] = []                           # (event, block, slot, gen, detail) transition log
        self.arena: Optional["Arena"] = None                 # M2 arena mode: the layer's position records (else None)

    # ------------------------------------------------------------------ queries
    def usage(self) -> int:
        return sum(1 for s in self.slots if s.state != FREE)

    def count(self, state: str) -> int:
        return sum(1 for s in self.slots if s.state == state)

    def resident(self, block: int) -> Optional[int]:
        s = self.lru.get(block)
        if s is None:
            s = self.po.get(block)
        if s is None:
            s = self.vres.get(block)
        return s

    def protected(self, block: int) -> bool:
        return bool(self.prot.get(block))

    def kinds_of(self, block: int) -> Set[str]:
        """The provenance kinds of the references that hold `block` now (empty when unknown or unprotected)."""
        out: Set[str] = set()
        for key in self.prot.get(block, ()):
            out |= self.pin_kind.get(block, {}).get(key, set())
        return out

    def positions(self) -> int:
        """Capacity mode rev6, L2: the non-FREE slots of class "ad" (an unpinned LRU entry and a RELEASING slot count)."""
        return sum(1 for s in self.slots if s.state != FREE and s.cls != "v")

    def unpinned_lru(self) -> int:
        """The LRU entries without a reference (the rev-5 L2 counted them as available; kept for its negative control)."""
        return sum(1 for blk in self.lru if not self.prot.get(blk))

    def v_slots(self) -> int:
        """Capacity mode rev6: the non-FREE slots of class "v" (verifier reserve), RELEASING included."""
        return sum(1 for s in self.slots if s.state != FREE and s.cls == "v")

    def free_count(self) -> int:
        return sum(1 for s in self.slots if s.state == FREE)

    def check(self) -> None:
        """Structural invariants (K3): one slot per resident block; states agree with the maps."""
        seen = {}
        for blk, s in list(self.lru.items()) + list(self.po.items()) + list(self.vres.items()):
            if blk in seen:
                raise OwnershipViolation("block %d is in two of LRU / protected-only / verifier-resident (stream %s)"
                                         % (blk, (self.l, self.h, self.b)))
            seen[blk] = s
            want = LRU if blk in self.lru else (PO if blk in self.po else VRES)
            if self.slots[s].state != want or self.slots[s].block != blk:
                raise OwnershipViolation("slot %d holds %s/%d but the map says %s/%d" % (s, self.slots[s].state, self.slots[s].block, want, blk))
        if len(set(seen.values())) != len(seen):
            raise OwnershipViolation("two resident blocks share one slot")
        if len(self.lru) > self.C:
            raise OwnershipViolation("LRU holds %d > C=%d" % (len(self.lru), self.C))
        for blk, s in self.pending.items():
            if self.slots[s].state != INFLIGHT or self.slots[s].block != blk:
                raise OwnershipViolation("pending block %d slot %d is %s" % (blk, s, self.slots[s].state))
        for blk in list(self.po) + list(self.vres):
            if not self.prot.get(blk):
                raise OwnershipViolation("protected-only or verifier-resident block %d has no protection" % blk)

    def _arena_check(self, s: Slot, where: str) -> None:
        """M2 check point (checked mode): the binding of slot s still holds after `where`."""
        if self.arena is not None and self.arena.checked:
            self.arena.check_slot(self, s, where)

    # ------------------------------------------------------------------ transitions
    def touch(self, block: int) -> None:
        """K1 / R2.1(a): a draft read of an LRU entry moves it to MRU (PO and VRES entries are not in the LRU)."""
        if block in self.lru:
            self.lru.move_to_end(block)
            self.log.append(("touch", block, self.lru[block], self.slots[self.lru[block]].gen, None))

    def protect(self, block: int, key, kind: Optional[str] = None) -> None:
        self.prot.setdefault(block, set()).add(key)
        if kind is not None:
            self.pin_kind.setdefault(block, {}).setdefault(key, set()).add(kind)

    def unprotect(self, key, mut=None) -> List[int]:
        """Drop the references of cohort `key` (ONE key per block; other cohorts keep theirs, R1.2); returns the
        protected-only and verifier-resident slots that now must be released."""
        out = []
        for blk in list(self.prot):
            refs = self.prot[blk]
            if key in refs:
                if mut and mut.release_drops_all_keys:
                    refs.clear()                          # NEGATIVE CONTROL (R1.6 shared pins): every key is dropped
                else:
                    refs.discard(key)
                kinds = self.pin_kind.get(blk)
                if kinds is not None:
                    for k in [k for k in kinds if k not in refs]:
                        del kinds[k]
                    if not kinds:
                        del self.pin_kind[blk]
                if not refs:
                    del self.prot[blk]
                    if blk in self.po:
                        out.append(self.po.pop(blk))
                    elif blk in self.vres:
                        out.append(self.vres.pop(blk))
        return out

    def take(self, idx: int, uid, block: int, seq: int, cls: str = "ad") -> int:
        """FREE -> INFLIGHT for a transfer of (uid, block) into a slot of class `cls` ("ad" draft, "v" verifier reserve).
        Returns the new generation.
        M2 arena mode (contract 2.2 / 2.4): the take also BINDS the lowest FREE position. Every check that can fail runs
        before the first change of the slot or the arena: the slot is FREE and names neither a position nor a generation,
        and the candidate position passes Arena.bind_check (Arena.bind repeats it as a fail-stop). The previous physical
        occupant's last reader is captured in the Arena (Arena.forward hands it to the transfer once), and the new
        binding's reader state starts empty: the logical slot's old last_reader belongs to whatever position it held
        before and is NOT kept."""
        s = self.slots[idx]
        if s.state != FREE:
            raise OwnershipViolation("take: slot %d of %s is %s" % (idx, (self.l, self.h, self.b), s.state))
        if cls not in ("ad", "v"):
            raise ValueError(cls)
        ar = self.arena
        if ar is not None:
            if s.phys is not None or s.pgen is not None:
                raise OwnershipViolation("arena binding: FREE slot %d of stream (l %d b %d h %d) still names position %s "
                                         "pgen %s" % (idx, self.l, self.b, self.h, s.phys, s.pgen))
            p = ar.lowest_free()
            ar.bind_check(p, "take into slot %d of stream (l %d b %d h %d)" % (idx, self.l, self.b, self.h))
        s.gen += 1
        s.state, s.block, s.owner, s.seq, s.cls = INFLIGHT, int(block), uid, seq, cls
        s.first_reader = None
        if ar is not None:
            slot_prev, s.last_reader = s.last_reader, None
            s.phys, s.pgen = p, ar.bind(p, (self.b, self.h, idx, s.gen), uid, int(block), seq, slot_prev)
        self.log.append(("take", block, idx, s.gen, uid))
        if uid == self.owner:
            self.pending[int(block)] = idx
        self._arena_check(s, "bind")
        return s.gen

    def release(self, idx: int, why: str) -> None:
        """A slot that leaves every role -> RELEASING (FREE after its last reader event, K4)."""
        s = self.slots[idx]
        if s.state not in (LRU, PO, VRES, INFLIGHT):
            raise OwnershipViolation("release(%s): slot %d of %s is %s" % (why, idx, (self.l, self.h, self.b), s.state))
        s.state = RELEASING
        self.log.append(("release", s.block, idx, s.gen, why))
        self._arena_check(s, "release(%s)" % why)                  # M2: a RELEASING slot keeps its position

    def _evict_over_c(self, mut=None) -> Optional[Dict]:
        """K1: after an insertion, the least recent entry leaves an LRU that holds more than C entries: a protected one
        keeps its slot as protected-only (K3), an unprotected one is released through the reader gate."""
        if len(self.lru) <= self.C:
            return None
        eb, es = self.lru.popitem(last=False)      # the least recent entry
        if self.prot.get(eb):
            kinds = sorted(self.kinds_of(eb))
            self.po[eb] = es                       # K3: protected-only keeps its slot
            self.slots[es].state = PO
            self.log.append(("evict_po", eb, es, self.slots[es].gen, None))
            if mut and mut.drop_alias:             # NEGATIVE CONTROL: the payload is freed, the alias map kept
                self.slots[es].state = RELEASING
            return dict(block=eb, slot=es, protected=True, kinds=kinds)
        self.release(es, "evict")
        return dict(block=eb, slot=es, protected=False, kinds=[])

    def evict_to_c(self, mut=None) -> List[Dict]:
        """Repair 1c (ordering contract O1): after a draft row's whole batch update, the least recent entries leave while
        the LRU holds more than C (each as in _evict_over_c). Returns the evictions in order."""
        out = []
        while len(self.lru) > self.C:
            out.append(self._evict_over_c(mut))
        return out

    def acquire(self, block: int, mut=None, evict: bool = True) -> Dict:
        """R2.1(c) draft acquisition: a draft row read a resident NON-LRU entry (PO or VRES). The entry enters the LRU at
        MRU on its own slot (no transfer, no new slot); then, if evict, the least recent entry leaves as after an arrival.
        The dispatcher passes evict=False and evicts once after the row's whole batch (repair 1c, O1: evict_to_c)."""
        if block in self.po:
            idx, src = self.po.pop(block), PO
        elif block in self.vres:
            idx, src = self.vres.pop(block), VRES
        else:
            raise OwnershipViolation("acquire: block %d of %s is not protected-only or verifier-resident"
                                     % (block, (self.l, self.h, self.b)))
        s = self.slots[idx]
        s.state = LRU
        self.lru[block] = idx                      # at MRU
        self.log.append(("acquire", block, idx, s.gen, src))
        return dict(block=block, slot=idx, src=src, evicted=self._evict_over_c(mut) if evict else None)

    def admit(self, idx: int, uid, block: int, gen: int, mut=None, demand=(DRAFT,)) -> Dict:
        """The host observed the arrival of transfer (uid, block, slot idx, gen) whose demand set is `demand` (R2.3).
        A demand that contains "draft" enters the LRU at MRU (R2.1(b)); a verifier-only demand makes the block VRES
        (R2.4), or releases it at once when no reference protects it (its requester was cancelled). The default
        demand is for unit tests; the dispatcher always passes the transfer's demand set. Returns what happened."""
        s = self.slots[idx]
        if s.gen != gen:
            raise StaleGeneration("arrival (uid %s, block %d) names slot %d gen %d but the slot holds gen %d (%s)"
                                  % (uid, block, idx, gen, s.gen, s.state))
        if s.state != INFLIGHT or s.block != block or s.owner != uid:
            raise OwnershipViolation("arrival (uid %s, block %d, slot %d gen %d): slot is %s/%d owner %s"
                                     % (uid, block, idx, gen, s.state, s.block, s.owner))
        if self.pending.get(block) == idx:
            del self.pending[block]
        s.seq = None
        if self.owner != uid and not (mut and mut.no_uid_check):
            s.state = RELEASING                   # T3: unowned arrival dropped; never read, so FREE at the next poll
            s.last_reader = None
            self.log.append(("drop_unowned", block, idx, gen, uid))
            self._arena_check(s, "drop_unowned")
            return dict(admitted=False, dropped=True, evicted=None)
        if self.resident(block) is not None:
            raise OwnershipViolation("arrival of block %d into slot %d but the block is already resident (slot %s)"
                                     % (block, idx, self.resident(block)))
        s.first_reader = None
        if DRAFT in demand or (mut and mut.verifier_arrival_enters_lru):   # NEGATIVE CONTROL (R2.6): {verify} -> LRU
            s.state = LRU
            self.lru[block] = idx                 # at MRU
            self.log.append(("admit", block, idx, gen, uid))
            return dict(admitted=True, dropped=False, evicted=self._evict_over_c(mut), where=LRU)
        if self.prot.get(block):
            s.state = VRES                        # R2.4: resident outside the LRU, protected
            self.vres[block] = idx
            self.log.append(("admit_vres", block, idx, gen, uid))
            return dict(admitted=True, dropped=False, evicted=None, where=VRES)
        s.state = RELEASING                       # R2.4: an unprotected verifier-only arrival; never read, FREE at the next poll
        s.last_reader = None
        self.log.append(("drop_vres_unprotected", block, idx, gen, uid))
        self._arena_check(s, "drop_vres_unprotected")
        return dict(admitted=True, dropped=False, evicted=None, where=RELEASING)

    def end_owner(self) -> List[int]:
        """Request end: LRU, protected-only and verifier-resident entries are released; in-flight transfers stay in
        flight (T3)."""
        out = []
        for blk, s in list(self.lru.items()):
            self.release(s, "end")
            out.append(s)
        for blk, s in list(self.po.items()):
            if self.slots[s].state == PO:
                self.release(s, "end")
            out.append(s)
        for blk, s in list(self.vres.items()):
            self.release(s, "end")
            out.append(s)
        self.lru.clear()
        self.po.clear()
        self.vres.clear()
        self.prot.clear()
        self.pin_kind.clear()
        self.pending.clear()
        self.owner = None
        return out


class LayerPool:
    """The streams of one layer and its X shared extra slots (K5)."""

    def __init__(self, l: int, B: int, H: int, C: int, X: int, R_ver: int):
        if C < 1 or X < 0 or R_ver < 0 or R_ver > X:
            raise ConfigError("K5: need C >= 1, 0 <= R_ver <= X (C=%d X=%d R_ver=%d)" % (C, X, R_ver))
        self.l, self.B, self.H, self.C, self.X, self.R_ver = l, B, H, C, X, R_ver
        self.n_pos = C + X
        self.streams: Dict[Tuple[int, int], Stream] = {(b, h): Stream(l, h, b, self.n_pos, C) for b in range(B) for h in range(H)}
        self.arena: Optional["Arena"] = None

    def attach_arena(self, arena: "Arena") -> None:
        """M2 arena mode: every stream of the layer binds its takes in `arena` (before any take)."""
        if arena.l != self.l or any(s.state != FREE for st in self.streams.values() for s in st.slots):
            raise ConfigError("M2: arena of layer %d attached to layer %d, or after a take" % (arena.l, self.l))
        self.arena = arena
        for st in self.streams.values():
            st.arena = arena

    def stream(self, b: int, h: int) -> Stream:
        return self.streams[(b, h)]

    def borrowed(self) -> int:
        return sum(max(0, st.usage() - self.C) for st in self.streams.values())

    def free_extras(self) -> int:
        return self.X - self.borrowed()

    def v_used(self) -> int:
        """Capacity mode rev6: the layer's non-FREE verifier-reserve slots (class "v", RELEASING included)."""
        return sum(st.v_slots() for st in self.streams.values())

    def can_take(self, st: Stream, kind: str) -> bool:
        if st.usage() < self.C:
            return True
        fe = self.free_extras()
        if kind == "verify":
            return fe > 0
        if kind == "draft":
            return fe > self.R_ver
        raise ValueError(kind)

    def first_free(self, st: Stream) -> Optional[int]:
        for s in st.slots:
            if s.state == FREE:
                return s.idx
        return None

    def poll_releases(self, mut=None) -> int:
        """K4: RELEASING -> FREE once the slot's last reader event completed (non-blocking poll). M2 arena mode: the
        position is unbound HERE and only here, together with the logical FREE transition (contract 2.2)."""
        n = 0
        for st in self.streams.values():
            for s in st.slots:
                if s.state != RELEASING:
                    continue
                gate = s.first_reader if (mut and mut.release_first_reader) else s.last_reader
                if (mut and mut.no_reader_wait) or ev_done(gate):
                    if self.arena is not None:
                        self.arena.unbind(s.phys, (st.b, st.h, s.idx, s.gen), s.pgen, s.owner, s.block)
                        s.phys = s.pgen = None
                    s.state = FREE
                    s.block, s.owner, s.cls = -1, None, None
                    st.log.append(("free", -1, s.idx, s.gen, None))
                    n += 1
        return n

    def releasing_gates(self) -> List[Any]:
        out = []
        for st in self.streams.values():
            for s in st.slots:
                if s.state == RELEASING and s.last_reader is not None and not ev_done(s.last_reader):
                    out.append(s.last_reader)
        return out

    def counts(self) -> Dict[str, int]:
        c = {k: 0 for k in STATES}
        for st in self.streams.values():
            for s in st.slots:
                c[s.state] += 1
        return c

    def position_counts(self) -> Dict[str, int]:
        """M2 K6 (contract 2.6): the arena's ALLOCATED positions by state: FREE, or the state of the bound logical slot."""
        c = {k: 0 for k in STATES}
        for who in self.arena.bound:
            if who is None:
                c[FREE] += 1
            else:
                c[self.streams[(who[0], who[1])].slots[who[2]].state] += 1
        return c

    def check_arena(self, E: int = 0, G: int = 0) -> None:
        """M2 full scan (checked mode, the end of every joint call; contract 2.3): every slot's binding (check_slot), no
        two slots on one position, bound positions = non-FREE slots, the free list = the unbound positions, pgen_t = pgen,
        and bound + E + G <= P_l (E: unconsumed draft credits of the layer, G: granted but unissued verifier fetches)."""
        ar = self.arena
        on = {}
        for st in self.streams.values():
            for s in st.slots:
                ar.check_slot(st, s, "the end-of-call scan")
                if s.state != FREE:
                    if s.phys in on:
                        raise OwnershipViolation("arena binding: slots %s and %s of layer %d share position %d"
                                                 % (on[s.phys], (st.b, st.h, s.idx), self.l, s.phys))
                    on[s.phys] = (st.b, st.h, s.idx)
        bound = sum(1 for who in ar.bound if who is not None)
        if not len(on) == bound == ar.n_bound or sorted(ar._free) != [p for p, who in enumerate(ar.bound) if who is None]:
            raise OwnershipViolation("arena binding: layer %d has %d non-FREE slots, %d bound positions (count %d) and free "
                                     "list %s" % (self.l, len(on), bound, ar.n_bound, sorted(ar._free)))
        if not torch.equal(ar.pgen_t, torch.tensor(ar.pgen, dtype=torch.int64)):
            raise OwnershipViolation("arena binding: layer %d generation mirror pgen_t differs from pgen" % self.l)
        ar.check_reserve(E, G)


# --------------------------------------------------------------------------------------------------- M2 arena
class Arena:
    """M2 slice A (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revision 2,
    sections 2.1-2.4): the records of ONE layer's P physical block positions (one KV head per position; the K / V rows
    live in the dispatcher's arena tensors, row r of position p = row p * bs + r). Per position p:
      bound[p]        (b, h, slot idx, slot gen) of the logical slot bound to p, or None (FREE)
      pgen[p]         +1 at every bind; pgen_t[p] is its int64 mirror (the address-map entry a deferred read copies at its
                      execution, contract 3.1)
      last_reader[p]  the last reader event of the CURRENT binding; kept after the unbind as physical history
      captured[p]     the previous occupant's last reader, captured at the bind until the transfer takes it (forward)
      writes[p]       seq of the transfer that targets p, until its arrival is observed (landed)
    The free list is a heap: a take always binds the LOWEST FREE position (deterministic reuse across streams and
    requests). log is test instrumentation (bind / unbind / land records; not counted in K6)."""

    def __init__(self, l: int, P: int, checked: bool = False):
        if P < 1:
            raise ConfigError("M2: an arena needs P >= 1 positions (got %d)" % P)
        self.l, self.P = int(l), int(P)
        self.checked = bool(checked)
        self.pgen: List[int] = [0] * self.P
        self.pgen_t = torch.zeros(self.P, dtype=torch.int64)
        self.bound: List[Optional[Tuple[int, int, int, int]]] = [None] * self.P
        self.last_reader: List[Any] = [None] * self.P
        self.captured: List[Any] = [None] * self.P
        self.writes: List[Optional[int]] = [None] * self.P
        self._free: List[int] = list(range(self.P))             # a valid heap: sorted
        self.n_bound = 0
        self.peak_bound = 0
        self.peak_reserved = 0                                   # max of bound + E + G seen by check_reserve
        self.binds = 0
        self.log: List[Dict] = []

    def lowest_free(self) -> Optional[int]:
        return self._free[0] if self._free else None

    def bind_check(self, p: Optional[int], where: str) -> None:
        """The consistency conditions of position p as the next binding that are available without a scan (contract
        2.4; Stream.take calls this BEFORE any change, and bind repeats it as a fail-stop): a FREE position exists, p is the
        top of the free list, bound count + free list = P, p is unbound, records no pending write and no uncollected
        captured reader, and its pgen_t mirror equals pgen. OwnershipViolation "arena binding: ..." otherwise."""
        bad = None
        if p is None:
            bad = "no FREE position (bound %d of P %d)" % (self.n_bound, self.P)
        elif not self._free or self._free[0] != p:
            bad = "position %d is not the lowest FREE position" % p
        elif self.n_bound + len(self._free) != self.P:
            bad = "bound count %d + free list %d != P %d" % (self.n_bound, len(self._free), self.P)
        elif self.bound[p] is not None:
            bad = "position %d on the free list is bound to %s" % (p, self.bound[p])
        elif self.writes[p] is not None:
            bad = "FREE position %d still records the write of transfer %s" % (p, self.writes[p])
        elif self.captured[p] is not None:
            bad = "FREE position %d still holds an uncollected captured reader" % p
        elif int(self.pgen_t[p]) != self.pgen[p]:
            bad = "position %d generation mirror pgen_t %d != pgen %d" % (p, int(self.pgen_t[p]), self.pgen[p])
        if bad is not None:
            raise OwnershipViolation("arena binding: %s (layer %d, %s)" % (bad, self.l, where))

    def bind(self, p: int, who: Tuple[int, int, int, int], uid, block: int, seq: int, slot_prev=None) -> int:
        """Contract 2.2 steps 1-4 for position p (Stream.take ran bind_check before any change; it runs again here as a
        fail-stop): capture p's last reader, pgen + 1 (and pgen_t), bound = who, the new binding's reader state empty.
        Returns the new pgen. slot_prev (the logical slot's last reader before the take) is recorded in the log only."""
        self.bind_check(p, "bind")
        heapq.heappop(self._free)
        self.captured[p] = self.last_reader[p]
        self.pgen[p] += 1
        self.pgen_t[p] = self.pgen[p]
        self.bound[p] = who
        self.writes[p] = seq
        self.last_reader[p] = None
        self.n_bound += 1
        self.binds += 1
        self.peak_bound = max(self.peak_bound, self.n_bound)
        self.log.append(dict(op="bind", l=self.l, p=p, pgen=self.pgen[p], who=who, uid=uid, block=block, seq=seq,
                             prev_ev=self.captured[p], prev_done=ev_done(self.captured[p]), slot_prev_ev=slot_prev))
        return self.pgen[p]

    def forward(self, p: int):
        """The previous physical occupant's last reader of position p, captured at its bind: handed to the transfer ONCE
        (Transfer.reader_ev) and cleared."""
        ev, self.captured[p] = self.captured[p], None
        return ev

    def note_reader(self, p: int, pgen: int, ev) -> None:
        """A read of position p (binding pgen) whose last read event is `ev` (contract 2.2: reads set the slot's AND the
        position's last reader)."""
        if p is None or self.bound[p] is None or self.pgen[p] != pgen:
            raise OwnershipViolation("arena binding: reader event for position %s (layer %d) of binding pgen %s, but the "
                                     "position is %s with pgen %s" % (p, self.l, pgen, None if p is None else self.bound[p],
                                                                      None if p is None else self.pgen[p]))
        self.last_reader[p] = ev

    def landed(self, p: int, seq: int, uid=None, b: Optional[int] = None, dropped: bool = False) -> None:
        """The host observed the arrival of transfer `seq` into position p (the quarantine of 2.4 ends at the logical FREE,
        not here: the position stays bound)."""
        if self.writes[p] != seq:
            raise OwnershipViolation("arena binding: arrival of transfer %d at position %d (layer %d), which records transfer "
                                     "%s" % (seq, p, self.l, self.writes[p]))
        self.writes[p] = None
        self.log.append(dict(op="land", l=self.l, p=p, pgen=self.pgen[p], seq=seq, uid=uid, b=b, dropped=dropped))

    def unbind(self, p: int, who: Tuple[int, int, int, int], pgen: int, uid=None, block: Optional[int] = None) -> None:
        """The logical slot `who` became FREE (LayerPool.poll_releases only). last_reader stays as physical history."""
        if p is None or self.bound[p] != who or self.pgen[p] != pgen:
            raise OwnershipViolation("arena binding: unbind of position %s (layer %d) by slot %s pgen %s, but the position "
                                     "holds %s with pgen %s" % (p, self.l, who, pgen, None if p is None else self.bound[p],
                                                                None if p is None else self.pgen[p]))
        self.bound[p] = None
        self.writes[p] = None
        self.captured[p] = None
        heapq.heappush(self._free, p)
        self.n_bound -= 1
        self.log.append(dict(op="unbind", l=self.l, p=p, pgen=pgen, who=who, uid=uid, block=block,
                             last_ev=self.last_reader[p]))

    def check_slot(self, st, s, where: str) -> None:
        """Checked mode: a non-FREE slot is bound (its position names it with its pgen); a FREE slot names no position."""
        if s.state == FREE:
            if s.phys is not None or s.pgen is not None:
                raise OwnershipViolation("arena binding: free slot %d gen %d of stream (l %d b %d h %d) still names position "
                                         "%s pgen %s after %s" % (s.idx, s.gen, st.l, st.b, st.h, s.phys, s.pgen, where))
            return
        p = s.phys
        ok = p is not None and 0 <= p < self.P
        if not ok or self.bound[p] != (st.b, st.h, s.idx, s.gen) or self.pgen[p] != s.pgen:
            raise OwnershipViolation("arena binding: %s slot %d gen %d of stream (l %d b %d h %d) is not bound to position %s "
                                     "(the position holds %s with pgen %s; the slot names pgen %s) after %s"
                                     % (s.state, s.idx, s.gen, st.l, st.b, st.h, p, self.bound[p] if ok else None,
                                        self.pgen[p] if ok else None, s.pgen, where))

    def check_reserve(self, E: int, G: int) -> None:
        """Contract 2.3: bound + E + G <= P (a FREE position for every later issue the logical rules admit)."""
        tot = self.n_bound + int(E) + int(G)
        self.peak_reserved = max(self.peak_reserved, tot)
        if tot > self.P:
            raise OwnershipViolation("arena binding: layer %d holds %d bound positions + %d unconsumed draft credits + %d "
                                     "granted unissued verifier fetches > P %d" % (self.l, self.n_bound, E, G, self.P))


# --------------------------------------------------------------------------------------------------- M2 B2 ring
class RingPool:
    """M2 B2 (retroinfer-eval docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md revisions 4-5, sections
    11.2-11.4): the records of ONE layer's pooled ring of Q entries (one entry = the K row and the V row of one (head,
    position) of one request; the rows live in the dispatcher's ring tensors, row e = entry e). See the module docstring
    (M2 B2 RING) for the fields. The free list is a heap: a bind always takes the LOWEST FREE entry. live_by counts the
    LIVE entries per (batch slot, head): the per-stream quantity of the 11.4 bound (a stream's previous request left only
    DRAINING entries at its end). log is test instrumentation (bind / retag / drain / free records; not counted in K6)."""

    def __init__(self, l: int, Q: int, checked: bool = False):
        if Q < 1:
            raise ConfigError("M2 B2: a ring needs Q >= 1 entries (got %d)" % Q)
        self.l, self.Q = int(l), int(Q)
        self.checked = bool(checked)
        self.rgen: List[int] = [0] * self.Q
        self.rgen_t = torch.zeros(self.Q, dtype=torch.int64)
        self.state: List[str] = [FREE] * self.Q
        self.kind: List[Optional[str]] = [None] * self.Q
        self.owner: List[Optional[Tuple]] = [None] * self.Q
        self.prov: List[Optional[Tuple]] = [None] * self.Q
        self.ready: List[Any] = [None] * self.Q
        self.cev: List[Any] = [None] * self.Q
        self.dev: List[Any] = [None] * self.Q
        self.dseq: List[Optional[int]] = [None] * self.Q
        self._free: List[int] = list(range(self.Q))            # a valid heap: sorted
        self._drains = 0
        self.n_live = 0
        self.n_draining = 0
        self.live_by: Dict[Tuple[int, int], int] = {}
        self.peak_live = 0
        self.peak_stream = 0
        self.binds = 0
        self.log: List[Dict] = []

    def lowest_free(self) -> Optional[int]:
        return self._free[0] if self._free else None

    def bind_check(self, e: Optional[int], where: str) -> None:
        """The conditions of entry e as the next binding, available without a scan (RingPool.bind repeats them as a
        fail-stop): a FREE entry exists, e is the top of the free list, LIVE + DRAINING + free list = Q, e is FREE, and its
        rgen_t mirror equals rgen. OwnershipViolation "ring binding: ..." otherwise."""
        bad = None
        if e is None:
            bad = "no FREE entry (LIVE %d, DRAINING %d of Q %d)" % (self.n_live, self.n_draining, self.Q)
        elif not self._free or self._free[0] != e:
            bad = "entry %d is not the lowest FREE entry" % e
        elif self.n_live + self.n_draining + len(self._free) != self.Q:
            bad = "LIVE %d + DRAINING %d + free list %d != Q %d" % (self.n_live, self.n_draining, len(self._free), self.Q)
        elif self.state[e] != FREE:
            bad = "entry %d on the free list is %s" % (e, self.state[e])
        elif int(self.rgen_t[e]) != self.rgen[e]:
            bad = "entry %d generation mirror rgen_t %d != rgen %d" % (e, int(self.rgen_t[e]), self.rgen[e])
        if bad is not None:
            raise OwnershipViolation("ring binding: %s (layer %d, %s)" % (bad, self.l, where))

    def bind(self, e: int, kind: str, owner: Tuple, prov: Tuple, where: str) -> Tuple[int, Any]:
        """Contract 11.2: FREE -> LIVE for entry e (bind_check runs again here as a fail-stop): rgen + 1 (and rgen_t),
        kind, owner (uid, b, h), provenance, empty reader state. Returns (the new rgen, the previous occupancy's D2H
        event: the dispatcher's write stream-waits it, the second safeguard)."""
        if kind not in RING_KINDS:
            raise ValueError(kind)
        self.bind_check(e, where)
        heapq.heappop(self._free)
        prev_d2h = self.dev[e]
        self.rgen[e] += 1
        self.rgen_t[e] = self.rgen[e]
        self.state[e], self.kind[e], self.owner[e], self.prov[e] = LIVE, kind, tuple(owner), tuple(prov)
        self.ready[e] = self.cev[e] = self.dev[e] = None
        self.dseq[e] = None
        self.n_live += 1
        bh = (owner[1], owner[2])
        self.live_by[bh] = self.live_by.get(bh, 0) + 1
        self.peak_live = max(self.peak_live, self.n_live)
        self.peak_stream = max(self.peak_stream, self.live_by[bh])
        self.binds += 1
        self.log.append(dict(op="bind", l=self.l, e=e, rgen=self.rgen[e], kind=kind, owner=tuple(owner), prov=tuple(prov),
                             where=where, prev_d2h=prev_d2h))
        return self.rgen[e], prev_d2h

    def _held(self, e: int, g: int, what: str) -> None:
        if e is None or not 0 <= e < self.Q or self.state[e] == FREE or self.rgen[e] != g:
            raise OwnershipViolation("ring binding: %s for entry %s (layer %d) of rgen %s, but the entry is %s with rgen %s"
                                     % (what, e, self.l, g, None if e is None else self.state[e],
                                        None if e is None else self.rgen[e]))

    def set_ready(self, e: int, g: int, ev) -> None:
        """The producer's payload write of binding rgen g completed at `ev` (an event on "compute")."""
        self._held(e, g, "ready event")
        self.ready[e] = ev

    def note_compute(self, e: int, g: int, ev) -> None:
        """A compute read of binding rgen g whose call's read event is `ev` (one ordered stream: the latest covers all)."""
        self._held(e, g, "compute reader event")
        self.cev[e] = ev

    def note_d2h(self, e: int, g: int, ev) -> None:
        """A D2H read of binding rgen g that completes at `ev` (one ordered stream "d2h": the latest covers all)."""
        self._held(e, g, "D2H reader event")
        self.dev[e] = ev

    def retag(self, e: int, g: int, kind: str, prov: Tuple) -> None:
        """PRE -> PROV (the cohort executed the layer) or PROV -> EXACT (publication): the same entry and payload."""
        self._held(e, g, "re-tag to %s" % kind)
        if self.state[e] != LIVE or kind not in RING_KINDS:
            raise OwnershipViolation("ring binding: re-tag of entry %d (layer %d) to %s, but it is %s"
                                     % (e, self.l, kind, self.state[e]))
        self.log.append(dict(op="retag", l=self.l, e=e, rgen=self.rgen[e], kind=kind, old=self.kind[e], prov=tuple(prov)))
        self.kind[e], self.prov[e] = kind, tuple(prov)

    def drain(self, e: int, g: int, why: str) -> None:
        """LIVE -> DRAINING: the logical release (rollback, supersession, decision, history, H1, request end). The entry
        stays bound (rgen unchanged) until RingPool.retire finds all three of its events complete."""
        self._held(e, g, "drain(%s)" % why)
        if self.state[e] != LIVE:
            raise OwnershipViolation("ring binding: drain(%s) of entry %d (layer %d), which is %s" % (why, e, self.l,
                                                                                                    self.state[e]))
        self.state[e] = DRAINING
        self.n_live -= 1
        self.n_draining += 1
        bh = (self.owner[e][1], self.owner[e][2])
        self.live_by[bh] -= 1
        self._drains += 1
        self.dseq[e] = self._drains
        self.log.append(dict(op="drain", l=self.l, e=e, rgen=self.rgen[e], kind=self.kind[e], owner=self.owner[e], why=why))

    def gates(self, e: int) -> List[Any]:
        """The INCOMPLETE events of entry e (its write, its latest compute reader, its latest D2H reader)."""
        return [ev for ev in (self.ready[e], self.cev[e], self.dev[e]) if ev is not None and not ev_done(ev)]

    def _free_entry(self, e: int, why: str) -> None:
        self.state[e] = FREE
        self.n_draining -= 1
        self.log.append(dict(op="free", l=self.l, e=e, rgen=self.rgen[e], kind=self.kind[e], owner=self.owner[e], why=why))
        self.kind[e] = self.owner[e] = self.prov[e] = None
        self.dseq[e] = None
        heapq.heappush(self._free, e)

    def retire(self) -> int:
        """DRAINING -> FREE for every entry whose events are all complete (non-blocking). Returns the number freed."""
        n = 0
        for e in range(self.Q):
            if self.state[e] == DRAINING and not self.gates(e):
                self._free_entry(e, "retire")
                n += 1
        return n

    def oldest_held(self) -> Optional[int]:
        """The DRAINING entry with an incomplete event that was released FIRST (backpressure's wait target), or None."""
        best = None
        for e in range(self.Q):
            if self.state[e] == DRAINING and self.gates(e) and (best is None or self.dseq[e] < self.dseq[best]):
                best = e
        return best

    def counts(self) -> Dict[str, int]:
        """K6 (contract 11.4): LIVE entries by kind, and FREE + DRAINING (ring_idle)."""
        c = {k: 0 for k in RING_KINDS}
        idle = 0
        for e in range(self.Q):
            if self.state[e] == LIVE:
                c[self.kind[e]] += 1
            else:
                idle += 1
        c["idle"] = idle
        return c

    def check(self) -> None:
        """Full scan (checked mode): counts, the free list = the FREE entries, rgen_t = rgen, live_by = the LIVE entries
        per (batch slot, head)."""
        live = sum(1 for s in self.state if s == LIVE)
        drn = sum(1 for s in self.state if s == DRAINING)
        by: Dict[Tuple[int, int], int] = {}
        for e in range(self.Q):
            if self.state[e] == LIVE:
                bh = (self.owner[e][1], self.owner[e][2])
                by[bh] = by.get(bh, 0) + 1
        if (live, drn) != (self.n_live, self.n_draining) or sorted(self._free) != [e for e in range(self.Q)
                                                                                  if self.state[e] == FREE]:
            raise OwnershipViolation("ring binding: layer %d counts LIVE %d / %d, DRAINING %d / %d, free list %s"
                                     % (self.l, live, self.n_live, drn, self.n_draining, sorted(self._free)))
        if {k: v for k, v in self.live_by.items() if v} != by:
            raise OwnershipViolation("ring binding: layer %d per-stream LIVE counts %s != %s" % (self.l, self.live_by, by))
        if not torch.equal(self.rgen_t, torch.tensor(self.rgen, dtype=torch.int64)):
            raise OwnershipViolation("ring binding: layer %d generation mirror rgen_t differs from rgen" % self.l)


# --------------------------------------------------------------------------------------------------- memory (K6)
@dataclass
class Geometry:
    """REAL geometry for byte accounting (K6: 'bytes use the real geometry parameters'). Defaults: NOSA-8B
    (Llama-3.1-8B base): bs 64, D 128, H 2 KV heads, Hq 32, L 32, hidden 4096, vocab 128256, bf16, top-k 64 slots,
    q and p vectors stored as fp32."""
    bs: int = 64
    D: int = 128
    H: int = 2
    Hq: int = 32
    L: int = 32
    hidden: int = 4096
    V: int = 128256
    elem: int = 2
    topk: int = 64
    q_elem: int = 4
    bias_elem: int = 0                  # C1.1 A9: the element size of the recorded bias dtype (0: no bias, ABI v3)

    def unit(self) -> Dict[str, int]:
        """Bytes of one counted item. The metadata sizes are fixed conventions (stated here, used everywhere).
        C1.1 A9 (bias_elem > 0, ABI v4 only; the dispatcher sets it from the ACTUAL bias dtype, never a constant): the
        bias joins every unit that counts K / V (slot, pos_kv, ck, workspace_row), and pos_bias / slot_bias / ck_bias
        report its share. With bias_elem = 0 the units are exactly today's."""
        u = self._unit()
        be = int(self.bias_elem)
        if be:
            u.update(pos_bias=be, slot_bias=self.bs * be, ck_bias=be)
            u["slot"] += self.bs * be
            u["pos_kv"] += be
            u["ck"] += be
            u["workspace_row"] += self.H * be
        return u

    def _unit(self) -> Dict[str, int]:
        return dict(
            slot=2 * self.bs * self.D * self.elem,                      # K + V of one (layer, head, block)
            pos_kv=2 * self.D * self.elem,                              # K + V of one (layer, head, position)
            q_vector=self.V * self.q_elem,                              # one recorded q (draft distribution)
            p_vector=self.V * self.q_elem,                              # one verifier distribution p awaiting the decision
            workspace_row=(self.Hq * self.D + 2 * self.H * self.D) * self.elem + self.H * self.topk * 4,  # q, k, v, sel
            hidden_row=self.hidden * self.elem,                         # one row's hidden state between layers
            ck=self.D * self.elem,                                      # one compressed key (or buffered key) of (l, h)
            slot_meta=8 + 1 + 8 + 8,                                    # block id, state, generation, recency
            prot_ref=16,                                                # (cohort key, block) reference
            transfer_rec=64,                                            # uid, layer, b, h, block, slot, gen, kind, events
            draft_rec=32,                                               # s, position, input, d, epoch, n_at, state
            cohort_rec=64,                                              # key, range, state, layer, call ids
            sel_id=4,                                                   # one block id held in a draft or cohort record
            release_entry=16,                                           # one release-queue entry or reader-event entry
            # M2 arena metadata (contract 2.6), charged at ALLOCATED capacity in arena mode only:
            slot_bind=8 + 8,                                            # Slot.phys, Slot.pgen of one logical slot
            pos_meta=1 + 8 + 4 * 8 + 8 + 8 + 8 + 8,                     # state, pgen, bound (b, h, idx, gen), last_reader,
                                                                        # captured, writes, free-list entry
            map_entry=8,                                                # one int64 entry of the generation mirror pgen_t
            transfer_bind=8 + 8,                                        # Transfer.phys, Transfer.pgen of one transfer
            counter=8,                                                  # one rev6 credit counter or grant entry
            # M2 B2 ring metadata (contract 11.4), charged at ALLOCATED capacity in ring mode only:
            ring_meta=1 + 1 + 3 * 8 + 3 * 8 + 8 + 3 * 8 + 8 + 8,        # state, kind, owner (uid, b, h), provenance
                                                                        # (epoch, key, position), rgen, three events,
                                                                        # holds, free-list entry
            ring_map=8 + 8,                                             # one map entry (key, entry) of a request
            adm_meta=1 + 3 * 8 + 8,                                     # one admission buffer: state, ready / compute /
                                                                        # D2H events, size
        )


CATS = ("slots_free", "slots_lru", "slots_po", "slots_vres", "slots_inflight", "slots_releasing", "staging", "workspace",
        "hidden", "q_vectors", "p_vectors", "draft_kv", "provisional_kv", "recent_kv", "compressed_keys", "metadata")

# the counts one sample takes (every key is required: a missing count is an error, never a silent zero)
COUNTS = (FREE, LRU, PO, VRES, INFLIGHT, RELEASING, "workspace_rows", "hidden_rows", "q_vectors", "p_vectors",
          "draft_kv_pos", "prov_kv_pos", "recent_kv_pos", "ck_entries", "prot_refs", "transfers", "drafts", "cohorts",
          "sel_ids", "release_entries")
# M2 arena mode only (MemoryAccountant(arena=True)): the slot-state counts above are then ALLOCATED POSITIONS by state, and
# these count the logical slot records, their binding fields, the position records, the pgen_t entries, the transfers'
# binding fields and the rev6 credit counters / grant entries.
ARENA_COUNTS = ("slot_records", "slot_binds", "pos_records", "map_entries", "transfer_binds", "reserve_counters")
# M2 B2 ring mode only (MemoryAccountant(ring=True), contract 11.4): LIVE ring entries by kind (EXACT, DRAFT, PRE + PROV),
# FREE + DRAINING entries, all allocated entries, the requests' map entries, and the held admission buffers (their
# (layer, head, position) items and their records). In ring mode these REPLACE the eager draft / provisional / recent
# dict counts for the draft_kv / provisional_kv / recent_kv categories; ring_idle and admission_kv are added.
RING_COUNTS = ("ring_live_exact", "ring_live_draft", "ring_live_prov", "ring_idle", "ring_entries", "ring_map_entries",
               "admission_pos", "admission_records")
RING_CATS = ("ring_idle", "admission_kv")
# C1.1 ABI v4 only (geom.bias_elem > 0): the bytes of the queued OPERATIONAL bias backings held at the sample
# (BiasOpLedger.held, after its passive retirement), charged as the category bias_staging. A SAMPLED value: the exact
# peak at allocation transitions is BiasOpLedger.peak, reported apart.
BIAS_COUNTS = ("bias_staging_bytes",)
BIAS_CATS = ("bias_staging",)
# C1.2 consume mode only (contract revision 11, 12.9.2): the bytes of the gather backings (consume ledger) held at the
# sample. Every gather is released inside its call, before that call's sample, so the serialized mode samples 0; the
# ledger's exact allocation-transition peak is reported apart.
CONSUME_COUNTS = ("consume_staging_bytes",)
CONSUME_CATS = ("consume_staging",)
# HOOK-B2 (Config.store "tensor" only; MemoryAccountant.enable_sources): the bytes of the source backings that the
# HOOK-B2: tensor store holds at the sample (tensor_store.SourceHolds.held: the A_r prefill tensors and the ring
# HOOK-B2: producer sources, each distinct storage once at untyped_storage().nbytes()). A SAMPLED value; the exact
# HOOK-B2: peak is the store's.
SOURCE_COUNTS = ("source_staging_bytes",)                  # HOOK-B2
SOURCE_CATS = ("source_staging",)                          # HOOK-B2
# HOOK-L6 (INTERFACE_BC5 I11; Config.store_gen "consumer" only; MemoryAccountant.enable_gen_mirror): the store's
# consumer generation mirrors at allocated capacity (tensor_store store_bytes()["gen_mirror"], each storage once). The
# host mirrors pgen_t / rgen_t stay in metadata (map_entries, ring_entries), so nothing is counted twice. The pending
# generation staging is a store hold and is part of source_staging.
GEN_COUNTS = ("gen_mirror_bytes",)                         # HOOK-L6
GEN_CATS = ("gen_mirror",)                                 # HOOK-L6


class MemoryAccountant:
    """K6: peak and time-average bytes per category, sampled at every joint call. Counts come from the run; bytes are
    counts x the REAL per-unit sizes of `geom`. staging is the S0 ring (pinned staging + device landing) in bytes.

    Categories: physical slots by state, FREE included (M1 backs every stream with C + X slot positions, ownership
    docstring K5; verifier-resident slots have their own category slots_vres, repair 1 R2.4); staging; saved verifier workspace (q/k/v + selections per row and layer); hidden states held between
    layers (rows of RUNNING cohorts and draft rows of the current step); every held recorded q; every verifier p of a
    DONE cohort; every held draft KV entry; every held provisional KV entry; the published exact KV of the recent region;
    every held compressed or buffered key (committed chain, draft chain, cohort chains); metadata (every physical slot,
    protection references, in-flight transfer records, draft and cohort records with their block ids, release-queue and
    reader-event entries). M1 test instrumentation (trace, logs, the deferred-read validators) is not counted.
    arena=True (M2, contract 2.6): the slot categories are the ALLOCATED arena positions by state (FREE, or the state of
    the bound logical slot), and the metadata adds ARENA_COUNTS at allocated capacity; the logical slot records are
    charged from their own count (slot_records). Test instrumentation (arena logs, the S0 readback payloads) is not
    counted. Not a GPU memory measurement.
    ring=True (M2 B2, contract 11.4; needs arena=True): every ALLOCATED ring entry is charged at pos_kv: LIVE entries by
    kind into draft_kv (DRAFT), provisional_kv (PRE and PROV) and recent_kv (EXACT), replacing the eager dict counts, and
    FREE + DRAINING into ring_idle; every admission buffer that is not FREE (also of an ended request) into admission_kv;
    the metadata adds the entry records and the rgen_t entries at allocated capacity, the map entries and the admission
    buffer records. Ring logs, the D2H snapshots and the host-row owner record are test instrumentation, not counted.
    C1.1 ABI v4 (geom.bias_elem > 0): the bias is charged inside the same categories through the units (Geometry.unit);
    these are LOGICAL units. In addition the category bias_staging charges, at each sample, the bytes of the queued
    OPERATIONAL bias backings still held (BiasOpLedger; the count bias_staging_bytes). The dispatcher reports the ACTUAL
    persistent bias tensor bytes and the ledger (its exact allocation-transition peak) separately
    (report()["physical"]["bias"]); none of these is a Python-allocator or GPU measurement or a fit claim."""

    def __init__(self, geom: Optional[Geometry] = None, staging_bytes: int = 0, arena: bool = False, ring: bool = False,
                 consume: bool = False):
        self.geom = geom or Geometry()
        self.u = self.geom.unit()
        self.staging_bytes = int(staging_bytes)
        self.arena = bool(arena)
        self.ring = bool(ring)
        if self.ring and not self.arena:
            raise ConfigError("K6: the ring accountant needs arena=True")
        self.bias = self.geom.bias_elem > 0                # C1.1 ABI v4: + the operational bias staging category
        self.consume = bool(consume)                       # C1.2: + the consume (gather) staging category
        if self.consume and not (self.ring and self.bias):
            raise ConfigError("K6: the consume accountant needs the ring and ABI v4 bias accounting")
        self.cats = CATS + (RING_CATS if self.ring else ()) + (BIAS_CATS if self.bias else ()) + (
            CONSUME_CATS if self.consume else ())
        self.peak = {k: 0 for k in self.cats + ("total",)}
        self.sum = {k: 0 for k in self.cats + ("total",)}
        self.n = 0
        self.last: Dict[str, int] = {}
        self.last_counts: Dict[str, int] = {}

    source = False                                         # HOOK-B2: True after enable_sources (Config.store "tensor")
                                                           # HOOK-B2
    def enable_sources(self) -> None:                      # HOOK-B2
        """HOOK-B2 (Config.store "tensor"): adds the category source_staging (SOURCE_CATS), charged from the count
        source_staging_bytes at each sample. Only before the first sample. (HOOK-B2)"""
        if self.n or self.source:                          # HOOK-B2
            raise ConfigError("K6: the source category is enabled once, before the first sample")   # HOOK-B2
        self.source = True                                 # HOOK-B2
        self.cats = self.cats + SOURCE_CATS                # HOOK-B2
        for k in SOURCE_CATS:                              # HOOK-B2
            self.peak[k] = 0                               # HOOK-B2
            self.sum[k] = 0                                # HOOK-B2
                                                           # HOOK-B2 (end)
    gen = False                                            # HOOK-L6: True after enable_gen_mirror (store_gen "consumer")

    def enable_gen_mirror(self) -> None:                   # HOOK-L6
        """HOOK-L6 (I11; Config.store_gen "consumer"): adds the category gen_mirror (GEN_CATS), charged from the count
        gen_mirror_bytes at each sample. Needs the source category (the store's mode). Only before the first sample."""
        if self.n or self.gen or not self.source:
            raise ConfigError("K6: the gen_mirror category is enabled once, after the source category and before the "
                              "first sample")
        self.gen = True
        self.cats = self.cats + GEN_CATS
        for k in GEN_CATS:
            self.peak[k] = 0
            self.sum[k] = 0

    def bytes_of(self, n: Dict[str, int]) -> Dict[str, int]:
        missing = [k for k in COUNTS + (ARENA_COUNTS if self.arena else ()) + (RING_COUNTS if self.ring else ())
                   + (BIAS_COUNTS if self.bias else ()) + (CONSUME_COUNTS if self.consume else ()) if k not in n]
        if missing:
            raise KeyError("K6 sample lacks counts %s" % missing)
        extra = ([] if self.arena else [k for k in ARENA_COUNTS if k in n]) + (
            [] if self.ring else [k for k in RING_COUNTS if k in n]) + (
            [] if self.bias else [k for k in BIAS_COUNTS if k in n]) + (
            [] if self.consume else [k for k in CONSUME_COUNTS if k in n])
        if extra:
            raise KeyError("K6 sample has arena / ring counts %s outside their mode" % extra)
        if self.source and SOURCE_COUNTS[0] not in n:                   # HOOK-B2: the store's source count
            raise KeyError("K6 sample lacks counts %s" % list(SOURCE_COUNTS))                       # HOOK-B2
        if not self.source and SOURCE_COUNTS[0] in n:                   # HOOK-B2
            raise KeyError("K6 sample has the store's count %s outside its mode" % list(SOURCE_COUNTS))   # HOOK-B2
        if self.gen != (GEN_COUNTS[0] in n):                            # HOOK-L6: exactly in store_gen "consumer"
            raise KeyError("K6 sample %s the count %s (gen_mirror category %s)"
                           % ("lacks" if self.gen else "has", list(GEN_COUNTS), "on" if self.gen else "off"))
        u = self.u
        n_slots = n[FREE] + n[LRU] + n[PO] + n[VRES] + n[INFLIGHT] + n[RELEASING]
        b = dict(slots_free=n[FREE] * u["slot"], slots_lru=n[LRU] * u["slot"], slots_po=n[PO] * u["slot"],
                 slots_vres=n[VRES] * u["slot"], slots_inflight=n[INFLIGHT] * u["slot"],
                 slots_releasing=n[RELEASING] * u["slot"],
                 staging=self.staging_bytes, workspace=n["workspace_rows"] * u["workspace_row"],
                 hidden=n["hidden_rows"] * u["hidden_row"], q_vectors=n["q_vectors"] * u["q_vector"],
                 p_vectors=n["p_vectors"] * u["p_vector"], draft_kv=n["draft_kv_pos"] * u["pos_kv"],
                 provisional_kv=n["prov_kv_pos"] * u["pos_kv"], recent_kv=n["recent_kv_pos"] * u["pos_kv"],
                 compressed_keys=n["ck_entries"] * u["ck"],
                 metadata=((n["slot_records"] if self.arena else n_slots) * u["slot_meta"] + n["prot_refs"] * u["prot_ref"]
                           + n["transfers"] * u["transfer_rec"]
                           + n["drafts"] * u["draft_rec"] + n["cohorts"] * u["cohort_rec"] + n["sel_ids"] * u["sel_id"]
                           + n["release_entries"] * u["release_entry"]))
        if self.arena:
            b["metadata"] += (n["slot_binds"] * u["slot_bind"] + n["pos_records"] * u["pos_meta"]
                              + n["map_entries"] * u["map_entry"] + n["transfer_binds"] * u["transfer_bind"]
                              + n["reserve_counters"] * u["counter"])
        if self.ring:
            if (n["ring_live_exact"] + n["ring_live_draft"] + n["ring_live_prov"] + n["ring_idle"]) != n["ring_entries"]:
                raise KeyError("K6 ring counts %s do not cover the %d allocated entries"
                               % ({k: n[k] for k in RING_COUNTS}, n["ring_entries"]))
            b["draft_kv"] = n["ring_live_draft"] * u["pos_kv"]
            b["provisional_kv"] = n["ring_live_prov"] * u["pos_kv"]
            b["recent_kv"] = n["ring_live_exact"] * u["pos_kv"]
            b["ring_idle"] = n["ring_idle"] * u["pos_kv"]
            b["admission_kv"] = n["admission_pos"] * u["pos_kv"]
            b["metadata"] += (n["ring_entries"] * (u["ring_meta"] + u["map_entry"]) + n["ring_map_entries"] * u["ring_map"]
                              + n["admission_records"] * u["adm_meta"])
        if self.bias:                                      # C1.1: bytes of the operational backings held at the sample
            b["bias_staging"] = int(n["bias_staging_bytes"])
        if self.consume:                                   # C1.2: bytes of the gather backings held at the sample
            b["consume_staging"] = int(n["consume_staging_bytes"])
        if self.source:                                    # HOOK-B2: the store's held source backings at the sample
            b["source_staging"] = int(n["source_staging_bytes"])   # HOOK-B2
        if self.gen:                                       # HOOK-L6: the consumer mirrors at allocated capacity
            b["gen_mirror"] = int(n["gen_mirror_bytes"])
        b["total"] = sum(b[k] for k in self.cats)
        return b

    def sample(self, counts: Dict[str, int]) -> Dict[str, int]:
        b = self.bytes_of(counts)
        for k, v in b.items():
            self.peak[k] = max(self.peak[k], v)
            self.sum[k] += v
        self.n += 1
        self.last = b
        self.last_counts = dict(counts)
        return b

    def report(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(peak=self.peak[k], mean=(self.sum[k] / self.n if self.n else 0.0)) for k in self.cats + ("total",)}


class BiasOpLedger:
    """C1.1 A9 (contract revision 10; Codex correction of 2026-10-15; ABI v4 only): every queued OPERATIONAL bias backing
    (a tensor that a deferred copy reads or that an in-flight payload holds), owned and charged from its allocation to
    its release. The ledger keeps ONE reference to each backing, so a backing counts until the ledger drops it:
      hold(kind, t, ev)      a backing released at the first passive retire() that sees ev complete (the existing
                             completion frontier of the copies that read it);
      hold(kind, t, key=k)   a backing released by release_key(kind, k) at its existing release point (an adversarial
                             landing sets the payload to None);
      retire()               PASSIVE: releases the records whose event is complete (ev_done only: no wait, no stream
                             advancement); the dispatcher calls it at its host poll, at each K6 sample and in finish().
    held = the bytes of the unique backings held now; peak = the EXACT maximum of held, updated at every allocation
    transition (held only falls at a release); log = every transition (instrumentation, not charged). A backing that
    shares storage with a held one is refused (no double count). Kinds: ring_producer (a ring payload's fresh bias),
    admission_src (A_r's bias source per layer), d2h_src (an eager_matched D2H source clone), s0_stage / s0_land (one
    S0 issue's bias pack and landing), adv_payload (an adversarial transfer's bias payload). Not counted here (reported
    as excluded): the persistent stores (host / arena / ring / A_r bias rows, charged in full elsewhere), views of them,
    diagnostic copies (Transfer.expect, the D2H expected rows and snapshot, the validators' copies) and synchronous
    eager-reference temporaries. Not a Python-allocator or GPU measurement."""
    KINDS = ("ring_producer", "admission_src", "d2h_src", "s0_stage", "s0_land", "adv_payload")
    CONSUME_KINDS = ("c1_gather",)          # C1.2 (12.9.2): the consume ledger (gather K / V / bias / identity backings)

    def __init__(self, kinds: Optional[Tuple[str, ...]] = None):
        self.kinds = tuple(kinds) if kinds else self.KINDS
        self.recs: Dict[int, Dict[str, Any]] = {}
        self.by_key: Dict[Tuple[str, Any], int] = {}
        self.serial = 0
        self.held = 0
        self.peak = 0
        self.peak_records = 0
        self.allocs = {k: 0 for k in self.kinds}
        self.alloc_bytes = {k: 0 for k in self.kinds}
        self.releases = {k: 0 for k in self.kinds}
        self.max_concurrent = {k: 0 for k in self.kinds}
        self.log: List[Tuple] = []

    def hold(self, kind: str, t, ev=None, key=None) -> int:
        if kind not in self.kinds or (ev is None) == (key is None):
            raise OwnershipViolation("bias ledger: %s needs exactly one release (event %s, key %s)" % (kind, ev, key))
        if t.numel() == 0:
            return 0                                     # an empty tensor holds no backing (and may share address 0)
        ptr = t.untyped_storage().data_ptr()
        if any(r["ptr"] == ptr for r in self.recs.values()) or (key is not None and (kind, key) in self.by_key):
            raise OwnershipViolation("bias ledger: %s backing %s (key %s) is already held" % (kind, ptr, key))
        nb = t.numel() * t.element_size()
        self.serial += 1
        s = self.serial
        self.recs[s] = dict(kind=kind, t=t, ptr=ptr, nbytes=nb, ev=ev, key=key)
        if key is not None:
            self.by_key[(kind, key)] = s
        self.held += nb
        self.allocs[kind] += 1
        self.alloc_bytes[kind] += nb
        self.peak = max(self.peak, self.held)
        self.peak_records = max(self.peak_records, len(self.recs))
        self.max_concurrent[kind] = max(self.max_concurrent[kind], sum(1 for r in self.recs.values() if r["kind"] == kind))
        self.log.append(("alloc", s, kind, nb, self.held))
        return s

    def _release(self, s: int) -> None:
        r = self.recs.pop(s)
        if r["key"] is not None:
            del self.by_key[(r["kind"], r["key"])]
        self.held -= r["nbytes"]
        self.releases[r["kind"]] += 1
        self.log.append(("release", s, r["kind"], r["nbytes"], self.held))

    def release_key(self, kind: str, key) -> None:
        s = self.by_key.get((kind, key))
        if s is None:
            raise OwnershipViolation("bias ledger: release of %s key %s that is not held" % (kind, key))
        self._release(s)

    def retire(self) -> int:
        done = [s for s, r in self.recs.items() if r["ev"] is not None and ev_done(r["ev"])]
        for s in done:
            self._release(s)
        return len(done)

    def outstanding(self) -> List[Tuple[str, Any, int]]:
        """The records held now: (kind, key or None, bytes), in allocation order."""
        return [(r["kind"], r["key"], r["nbytes"]) for _, r in sorted(self.recs.items())]
