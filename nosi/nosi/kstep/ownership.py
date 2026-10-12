"""KV ownership (spec section 5, K1-K6) of the historical blocks of every stream (layer l, KV head h, batch slot b).

STATES of a physical slot (one block per slot; ONE slot per resident block, K3):
  FREE        in no LRU, no protection, not in flight, last reader event complete (K4)
  INFLIGHT    destination of a transfer; never reassigned while in flight (T3)
  LRU         resident, a member of the stream's logical LRU (capacity C, K1)
  PO          protected-only: evicted from the LRU while protected; keeps its slot (K3)
  RELEASING   left every role; becomes FREE when its LAST reader event has completed (host poll, K4)
Physical slots in use per stream = |LRU| + |PO| + |INFLIGHT| (K3); RELEASING slots are reported separately.

K1  recency: Stream.touch() moves a read LRU entry to MRU; the dispatcher calls it in the fixed order of a joint call
    (verifier rows: cohort order, then position order; then the draft row). A fetched block enters the LRU when its
    arrival is observed (Stream.admit at the host poll), at MRU, evicting the least recent entry. The recency rule is the
    HitProtectedLRU twin's (retroinfer-eval hisparse-native-eval d7e3149 scripts/hisparse_native/reproduce.py:55-76: hits
    to MRU, misses before hits) extended across calls; HiSparse @87db743 hisparse.cuh:640-672 is the upstream rule.
K2  protection is a per-(block, cohort) reference of the stream (the block's single slot carries it, K3). A block may be
    protected while it has no slot (a refused or pending miss); it then pins the slot it gets.
K4  a cohort's references at a layer are dropped by the dispatcher only after the cohort is accepted or cancelled AND its
    last reader event at that layer completed. A slot becomes FREE only after its own last reader event completed.
K5  per layer: C slots per stream plus X shared extra slots. The extra pool is a per-layer COUNT: a stream whose usage
    reaches C borrows from it. A draft fetch may borrow only while more than R_ver extra slots are free (the verifier
    reserve); a verifier fetch may borrow any free extra slot. A refused draft fetch is counted and the draft continues
    masked. M1 backs every stream with C + X physical slot positions so that cpupack_core's per-(b, h, slot) addressing
    applies unchanged; the logical limit of X extra slots per layer is enforced here (M2 needs a pool-addressed
    descriptor for a physically shared pool).
The slot + generation pattern and the error names follow benchmarks/Efficiency/feeder_core.py:266-349 (SlotTable, a
staging-ring state machine; reused as a pattern only).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .backend import ev_done
from .errors import ConfigError, OwnershipViolation, StaleGeneration

FREE, INFLIGHT, LRU, PO, RELEASING = "free", "inflight", "lru", "po", "releasing"
STATES = (FREE, INFLIGHT, LRU, PO, RELEASING)


class Slot:
    __slots__ = ("idx", "state", "block", "gen", "owner", "first_reader", "last_reader", "seq")

    def __init__(self, idx: int):
        self.idx = idx
        self.state = FREE
        self.block = -1
        self.gen = 0
        self.owner = None
        self.first_reader = None
        self.last_reader = None
        self.seq = None                      # sequence id of the transfer in flight


class Stream:
    """One (layer, head, batch slot) stream."""

    def __init__(self, l: int, h: int, b: int, n_pos: int, C: int):
        self.l, self.h, self.b, self.C = l, h, b, C
        self.slots = [Slot(i) for i in range(n_pos)]
        self.lru: "OrderedDict[int, int]" = OrderedDict()   # block -> slot; first = least recent, last = MRU
        self.po: Dict[int, int] = {}                         # block -> slot (protected-only)
        self.pending: Dict[int, int] = {}                    # block -> slot of the CURRENT owner's transfers in flight
        self.prot: Dict[int, Set[Any]] = {}                  # block -> cohort keys (K2)
        self.owner = None                                    # request uid that owns the batch slot now
        self.log: List[Tuple] = []                           # (event, block, slot, gen, detail) transition log

    # ------------------------------------------------------------------ queries
    def usage(self) -> int:
        return sum(1 for s in self.slots if s.state != FREE)

    def count(self, state: str) -> int:
        return sum(1 for s in self.slots if s.state == state)

    def resident(self, block: int) -> Optional[int]:
        s = self.lru.get(block)
        return s if s is not None else self.po.get(block)

    def protected(self, block: int) -> bool:
        return bool(self.prot.get(block))

    def check(self) -> None:
        """Structural invariants (K3): one slot per resident block; states agree with the maps."""
        seen = {}
        for blk, s in list(self.lru.items()) + list(self.po.items()):
            if blk in seen:
                raise OwnershipViolation("block %d both in the LRU and protected-only (stream %s)" % (blk, (self.l, self.h, self.b)))
            seen[blk] = s
            want = LRU if blk in self.lru else PO
            if self.slots[s].state != want or self.slots[s].block != blk:
                raise OwnershipViolation("slot %d holds %s/%d but the map says %s/%d" % (s, self.slots[s].state, self.slots[s].block, want, blk))
        if len(set(seen.values())) != len(seen):
            raise OwnershipViolation("two resident blocks share one slot")
        if len(self.lru) > self.C:
            raise OwnershipViolation("LRU holds %d > C=%d" % (len(self.lru), self.C))
        for blk, s in self.pending.items():
            if self.slots[s].state != INFLIGHT or self.slots[s].block != blk:
                raise OwnershipViolation("pending block %d slot %d is %s" % (blk, s, self.slots[s].state))
        for blk in self.po:
            if not self.prot.get(blk):
                raise OwnershipViolation("protected-only block %d has no protection" % blk)

    # ------------------------------------------------------------------ transitions
    def touch(self, block: int) -> None:
        """K1: a read of an LRU entry moves it to MRU (protected-only entries are not in the LRU)."""
        if block in self.lru:
            self.lru.move_to_end(block)
            self.log.append(("touch", block, self.lru[block], self.slots[self.lru[block]].gen, None))

    def protect(self, block: int, key) -> None:
        self.prot.setdefault(block, set()).add(key)

    def unprotect(self, key) -> List[int]:
        """Drop every reference of cohort `key`; returns the protected-only slots that now must be released."""
        out = []
        for blk in list(self.prot):
            refs = self.prot[blk]
            if key in refs:
                refs.discard(key)
                if not refs:
                    del self.prot[blk]
                    if blk in self.po:
                        out.append(self.po.pop(blk))
        return out

    def take(self, idx: int, uid, block: int, seq: int) -> int:
        """FREE -> INFLIGHT for a transfer of (uid, block). Returns the new generation."""
        s = self.slots[idx]
        if s.state != FREE:
            raise OwnershipViolation("take: slot %d of %s is %s" % (idx, (self.l, self.h, self.b), s.state))
        s.gen += 1
        s.state, s.block, s.owner, s.seq = INFLIGHT, int(block), uid, seq
        s.first_reader = None
        self.log.append(("take", block, idx, s.gen, uid))
        if uid == self.owner:
            self.pending[int(block)] = idx
        return s.gen

    def release(self, idx: int, why: str) -> None:
        """A slot that leaves every role -> RELEASING (FREE after its last reader event, K4)."""
        s = self.slots[idx]
        if s.state not in (LRU, PO, INFLIGHT):
            raise OwnershipViolation("release(%s): slot %d of %s is %s" % (why, idx, (self.l, self.h, self.b), s.state))
        s.state = RELEASING
        self.log.append(("release", s.block, idx, s.gen, why))

    def admit(self, idx: int, uid, block: int, gen: int, mut=None) -> Dict:
        """The host observed the arrival of transfer (uid, block, slot idx, gen). Returns what happened."""
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
            return dict(admitted=False, dropped=True, evicted=None)
        if block in self.lru or block in self.po:
            raise OwnershipViolation("arrival of block %d into slot %d but the block is already resident (slot %s)"
                                     % (block, idx, self.resident(block)))
        s.state = LRU
        s.first_reader = None
        self.lru[block] = idx                     # at MRU
        self.log.append(("admit", block, idx, gen, uid))
        ev = None
        if len(self.lru) > self.C:
            eb, es = self.lru.popitem(last=False)  # the least recent entry
            if self.prot.get(eb):
                self.po[eb] = es                   # K3: protected-only keeps its slot
                self.slots[es].state = PO
                self.log.append(("evict_po", eb, es, self.slots[es].gen, None))
                if mut and mut.drop_alias:         # NEGATIVE CONTROL: the payload is freed, the alias map kept
                    self.slots[es].state = RELEASING
                ev = dict(block=eb, slot=es, protected=True)
            else:
                self.release(es, "evict")
                ev = dict(block=eb, slot=es, protected=False)
        return dict(admitted=True, dropped=False, evicted=ev)

    def end_owner(self) -> List[int]:
        """Request end: LRU and protected-only entries are released; in-flight transfers stay in flight (T3)."""
        out = []
        for blk, s in list(self.lru.items()):
            self.release(s, "end")
            out.append(s)
        for blk, s in list(self.po.items()):
            if self.slots[s].state == PO:
                self.release(s, "end")
            out.append(s)
        self.lru.clear()
        self.po.clear()
        self.prot.clear()
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

    def stream(self, b: int, h: int) -> Stream:
        return self.streams[(b, h)]

    def borrowed(self) -> int:
        return sum(max(0, st.usage() - self.C) for st in self.streams.values())

    def free_extras(self) -> int:
        return self.X - self.borrowed()

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
        """K4: RELEASING -> FREE once the slot's last reader event completed (non-blocking poll)."""
        n = 0
        for st in self.streams.values():
            for s in st.slots:
                if s.state != RELEASING:
                    continue
                gate = s.first_reader if (mut and mut.release_first_reader) else s.last_reader
                if (mut and mut.no_reader_wait) or ev_done(gate):
                    s.state = FREE
                    s.block, s.owner = -1, None
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

    def unit(self) -> Dict[str, int]:
        """Bytes of one counted item. The metadata sizes are fixed conventions (stated here, used everywhere)."""
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
        )


CATS = ("slots_free", "slots_lru", "slots_po", "slots_inflight", "slots_releasing", "staging", "workspace", "hidden",
        "q_vectors", "p_vectors", "draft_kv", "provisional_kv", "recent_kv", "compressed_keys", "metadata")

# the counts one sample takes (every key is required: a missing count is an error, never a silent zero)
COUNTS = (FREE, LRU, PO, INFLIGHT, RELEASING, "workspace_rows", "hidden_rows", "q_vectors", "p_vectors", "draft_kv_pos",
          "prov_kv_pos", "recent_kv_pos", "ck_entries", "prot_refs", "transfers", "drafts", "cohorts", "sel_ids",
          "release_entries")


class MemoryAccountant:
    """K6: peak and time-average bytes per category, sampled at every joint call. Counts come from the run; bytes are
    counts x the REAL per-unit sizes of `geom`. staging is the S0 ring (pinned staging + device landing) in bytes.

    Categories: physical slots by state, FREE included (M1 backs every stream with C + X slot positions, ownership
    docstring K5); staging; saved verifier workspace (q/k/v + selections per row and layer); hidden states held between
    layers (rows of RUNNING cohorts and draft rows of the current step); every held recorded q; every verifier p of a
    DONE cohort; every held draft KV entry; every held provisional KV entry; the published exact KV of the recent region;
    every held compressed or buffered key (committed chain, draft chain, cohort chains); metadata (every physical slot,
    protection references, in-flight transfer records, draft and cohort records with their block ids, release-queue and
    reader-event entries). M1 test instrumentation (trace, logs, the deferred-read validators) is not counted."""

    def __init__(self, geom: Optional[Geometry] = None, staging_bytes: int = 0):
        self.geom = geom or Geometry()
        self.u = self.geom.unit()
        self.staging_bytes = int(staging_bytes)
        self.peak = {k: 0 for k in CATS + ("total",)}
        self.sum = {k: 0 for k in CATS + ("total",)}
        self.n = 0
        self.last: Dict[str, int] = {}
        self.last_counts: Dict[str, int] = {}

    def bytes_of(self, n: Dict[str, int]) -> Dict[str, int]:
        missing = [k for k in COUNTS if k not in n]
        if missing:
            raise KeyError("K6 sample lacks counts %s" % missing)
        u = self.u
        n_slots = n[FREE] + n[LRU] + n[PO] + n[INFLIGHT] + n[RELEASING]
        b = dict(slots_free=n[FREE] * u["slot"], slots_lru=n[LRU] * u["slot"], slots_po=n[PO] * u["slot"],
                 slots_inflight=n[INFLIGHT] * u["slot"], slots_releasing=n[RELEASING] * u["slot"],
                 staging=self.staging_bytes, workspace=n["workspace_rows"] * u["workspace_row"],
                 hidden=n["hidden_rows"] * u["hidden_row"], q_vectors=n["q_vectors"] * u["q_vector"],
                 p_vectors=n["p_vectors"] * u["p_vector"], draft_kv=n["draft_kv_pos"] * u["pos_kv"],
                 provisional_kv=n["prov_kv_pos"] * u["pos_kv"], recent_kv=n["recent_kv_pos"] * u["pos_kv"],
                 compressed_keys=n["ck_entries"] * u["ck"],
                 metadata=(n_slots * u["slot_meta"] + n["prot_refs"] * u["prot_ref"] + n["transfers"] * u["transfer_rec"]
                           + n["drafts"] * u["draft_rec"] + n["cohorts"] * u["cohort_rec"] + n["sel_ids"] * u["sel_id"]
                           + n["release_entries"] * u["release_entry"]))
        b["total"] = sum(b[k] for k in CATS)
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
        return {k: dict(peak=self.peak[k], mean=(self.sum[k] / self.n if self.n else 0.0)) for k in CATS + ("total",)}
