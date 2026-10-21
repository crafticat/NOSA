"""R2 step 7, piece 5: the RESIDENCY SERVICE of the spec round (docs/R2_STEP7_FETCH_RETENTION.md): speculative fetch
and retention on the R2 path (paper 3.1 / 3.2 / 3.4), with the paper metrics 4.3 / 4.6 / 4.8.

OWNERSHIP IS K3X's, VENDORED BYTE-IDENTICAL (nosi/nosi/kstep/*, 91e6e47; test F9 checks the blob ids): one
kstep.ownership.LayerPool per layer (streams = (layer, KV head, batch slot), C slots of draft-only LRU per stream, X
shared extra slots per layer, the verifier reserve R_ver; states FREE / INFLIGHT / LRU / PO / VRES / RELEASING; one
reference per (block, cohort key (uid, epoch, m)); reader-gated release), kstep.transfer.Transfer and the transports'
interface issue / poll / done / force. This module re-implements, for the R2 tick, the K3X dispatcher's RESIDENCY call
sites (cited per method, dispatcher.py at 91e6e47); it imports no dispatcher code.

THE STORE (decision D4: per-stream backing): per layer l, store_k[l] / store_v[l] (B, (C + X + N_TAIL + T) * 64, H, D)
and store_b[l] (B, (C + X + N_TAIL + T) * 64, H), T = trash_slots (0 or 1). Slots 0 .. C + X - 1 hold historical
blocks (the streams' slots); slots C + X .. C + X + 3 are the request's tail rings (2 exact + 2 draft pages, note
5.2), written by the tick, never by this service; slot C + X + 4 (with trash_slots=1) is the captured tick's TRASH
slot: only its padded rows write and read it, never this service, never a live row (note 7). A page table over this
store serves piece 3's attention for draft and verify rows alike.

ONE LAYER (resolve, v1 = host-side per layer, decision D2): the tick passes the rows and their selections (CPU int64).
  verify rows (first; K3X issues verifier requests before drafts, dispatcher.py:3103): protect the whole selection
    (kind verify); classify at the FIRST LOOK: READY (resident: LRU / PO / VRES), IN_FLIGHT (a transfer of any
    requester: the demand is merged and the read WAITS for it), NEW (an URGENT miss: a slot is taken, verifier reserve
    included, and the block is gathered on the main stream before the layer's reads); dispatcher.py:2568-2589, :2492.
    When K5 grants the verifier no slot, POLICY E (note 12h) evicts the stream's least recent LRU entry that is
    unprotected and selected by no row of the layer, and reuses its slot; else (note 12j) it supersedes the oldest
    unprotected in-flight DRAFT prefetch that no row of the layer selects (its write lands first); with neither:
    CapacityRefusal (with the stream's capacity report).
  draft rows: protect the whole selection, hits included (kind draft); read only the READY blocks (touch / acquire,
    one eviction pass after the row: dispatcher.py:2899-2932); in-flight -> demand merged; a miss -> a background
    DRAFT transfer, deduplicated (one per stream and block), or REFUSED by K5 (counted; the draft continues masked,
    decision D3); dispatcher.py:2592-2629, :2955-2970.
  The page table lists, per (row, head), the historical slots in ascending block id, then the row's tail slots.
  defer_admit (the S-C schedule, r2_residency_sc.py): the waited and the urgent blocks are read IN PLACE (their
  INFLIGHT slot, after the wait / the gather) and admitted at the next poll, never in the middle of a tick.
After the layer's attention the tick calls read_done(l, plan, event): the event becomes the last reader of every slot
read and of every (key, l); a verify key's (key, l) references are QUEUED and dropped when that event completed (K4 as
amended by repair 1 R1; dispatcher.py:1553-1588, :1671-1694, :2821).

ARRIVALS (poll_arrivals, note 12k): with arrival_lag None (the default) a transport transfer is admitted at the first
poll that observes its completion. With arrival_lag D it is admitted exactly at the poll of tick issue_tick + D (the
tick index advances at end_tick): forced there if it has not landed (counted: arrival_forced, arrival_forced_ns), held
back until then if it has (arrival_held). The ownership state then depends on the schedule only, not on when the
transport lands, so two runs are comparable bitwise (Q-C3). An urgent block (landed in place) keeps its rule: admitted
at the next poll.

BIAS: the cis rows are on the GPU (the twin's total_cis), so they are not transferred: at TAKE the service copies the
block's cis rows into the slot's bias rows on the current (main) stream, which orders the write after every earlier
read of that slot on the same stream.

CANCELLATION (dispatcher.py:3408-3480): cancel(key) records the key (a later resolve by it is refused) and queues its
references through the reader gate; its in-flight transfers KEEP their slots until they land; at arrival K3X's admit
puts a draft-demanded block into the LRU (history is immutable), drops an unprotected verify-only one, drops an arrival
for a request that has ended.

Torch only; runs on CPU tensors (the tests) and on CUDA tensors (the store on the GPU, the host rows pinned).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .kstep.backend import ev_done
from .kstep.errors import CapacityRefusal, OwnershipViolation
from .kstep.executor import DRAFT, VERIFY
from .kstep.mutations import Mutations
from .kstep.ownership import FREE, INFLIGHT, LRU, PO, RELEASING, VRES, LayerPool
from .kstep.transfer import Transfer

BLOCK = 64
N_TAIL = 4                       # tail-ring slots per (request, head): exact pages 0, 1 and draft pages 0, 1


@dataclass
class Row:
    """One row of a tick at one layer, as the service needs it.
    kind "d" (draft) or "v" (verify); key = the cohort key (uid, epoch, m) (a draft row: the cohort that will verify
    its position); pos = the position the row writes (its last page is cut at pos % 64 + 1); hist_bound = the first
    block id NOT served by the store (blocks >= hist_bound come from the tail rings); tail = [(block id, store slot)]
    of the tail pages the row reads, ascending (its own block last)."""
    b: int
    kind: str
    key: Tuple
    pos: int
    hist_bound: int
    tail: Sequence[Tuple[int, int]] = ()


@dataclass
class Plan:
    page_table: torch.Tensor                  # (R, H, P) int32: store slots, ascending block id, then tails; -1 padded
    num_pages: torch.Tensor                   # (R, H) int32
    last_len: torch.Tensor                    # (R,) int32
    seq_idx: torch.Tensor                     # (R,) int32: the batch slot of each row
    blocks: List[List[List[int]]]             # [row][head] -> the block ids listed (historical then tail), in order
    reads: List[Tuple[int, int, int, int, str]] = field(default_factory=list)   # (b, h, slot, gen, kind) historical
    keys: List[Tuple] = field(default_factory=list)
    issued: List[Transfer] = field(default_factory=list)      # background (draft) transfers
    urgent: List[Transfer] = field(default_factory=list)      # verifier urgent misses (main-stream gather)
    waited: List[Transfer] = field(default_factory=list)      # in-flight transfers the verifier waited for
    looks: List[Dict] = field(default_factory=list)           # per verify (row, head): ready / in_flight / new
    masked: int = 0                           # draft (row, head) pairs that did not read every selected block
    draft_pairs: int = 0                      # draft (row, head) pairs
    refused: int = 0                          # draft fetches refused by K5


class TorchGather:
    """The urgent-miss gather and the stand-in background lander: copies a block's host K / V rows into its store slot
    on the CURRENT stream (in stream order: on the GPU a copy from pinned host rows; on the CPU immediate). The GPU run
    replaces it by NOSI's flash_h2d_from_mask on the main stream (note section 6)."""

    def __init__(self, host_k, host_v, store_k, store_v, block: int = BLOCK):
        self.host_k, self.host_v, self.store_k, self.store_v, self.bs = host_k, host_v, store_k, store_v, block
        self.gathered = 0

    def __call__(self, layer: int, transfers: List[Transfer]) -> None:
        for tr in transfers:
            r0, r1 = tr.block * self.bs, (tr.block + 1) * self.bs
            s0, s1 = tr.slot * self.bs, (tr.slot + 1) * self.bs
            self.store_k[layer][tr.b, s0:s1, tr.h].copy_(self.host_k[layer][tr.b, r0:r1, tr.h], non_blocking=True)
            self.store_v[layer][tr.b, s0:s1, tr.h].copy_(self.host_v[layer][tr.b, r0:r1, tr.h], non_blocking=True)
            tr.delivered = True
            self.gathered += 1


class Residency:
    """The residency service of one run (module docstring)."""

    def __init__(self, L: int, B: int, H: int, C: int, X: int, R_ver: int, transport, urgent_gather: Callable,
                 store_b=None, cis=None, block: int = BLOCK, block_bytes: int = 0, host_wait: Callable = None,
                 mut: Mutations = None, clock: Callable[[], int] = time.perf_counter_ns, poll_in_resolve: bool = True,
                 defer_issue: bool = False, defer_admit: bool = False, trash_slots: int = 0,
                 policy_e_inflight: bool = True, arrival_lag: Optional[int] = None):
        if trash_slots not in (0, 1):
            raise ValueError("trash_slots must be 0 or 1 (got %r)" % (trash_slots,))
        if arrival_lag is not None and (isinstance(arrival_lag, bool) or not isinstance(arrival_lag, int)
                                        or arrival_lag < 1):
            raise ValueError("arrival_lag must be None or an int >= 1 (got %r)" % (arrival_lag,))
        # THE ARRIVAL RULE (note 12k): None = free timing (a transfer is admitted at the first poll that sees it done);
        # D = fixed lag: a transfer issued in tick t is admitted exactly at the poll of tick t + D, forced there if it
        # has not landed, never earlier even if it has; the forced waits are counted (arrival_forced, _ns)
        self.arrival_lag = arrival_lag
        self.tick_no = 0                                     # end_tick calls so far: the current tick's index
        self.arrival_forced = 0                              # transfers forced at their admission tick
        self.arrival_forced_ns = 0                           # the host time of those forces
        self.arrival_held = 0                                # (poll, transfer) pairs done but not yet due
        self.poll_in_resolve, self.defer_issue = bool(poll_in_resolve), bool(defer_issue)
        # Policy E's in-flight candidates (note 12j); False is a NEGATIVE CONTROL (12h's rule: LRU entries only)
        self.policy_e_inflight = bool(policy_e_inflight)
        # defer_admit (the S-C schedule): the verifier's waited and urgent blocks are read in place on the device and
        # admitted at the next poll, never in the middle of a tick
        self.defer_admit = bool(defer_admit)
        self._landed: Set[int] = set()                       # seqs landed by the main stream (urgent gathers)
        self._vcands: Dict[Tuple[int, int], Any] = {}        # Policy E: the layer's verifier eviction candidates
        self._deferred: List[Tuple[int, List[Transfer]]] = []
        self.L, self.B, self.H, self.C, self.X, self.R_ver = int(L), int(B), int(H), int(C), int(X), int(R_ver)
        self.pools = [LayerPool(l, self.B, self.H, self.C, self.X, self.R_ver) for l in range(self.L)]
        self.n_hist = self.C + self.X                       # historical slots per stream; the tail rings follow
        self.tail0 = self.n_hist                            # first tail slot
        # the TRASH slot (the captured tick's padded rows, Q-T2): after the tail rings; outside every LayerPool (no take,
        # no transfer, no gather lands there) and outside the tail range (no Row can name it); None without it
        self.n_trash = int(trash_slots)
        self.trash_slot = self.n_hist + N_TAIL if self.n_trash else None
        self.tp = transport
        self.gather = urgent_gather
        self.store_b, self.cis = store_b, cis               # bias rows, filled from the cis table at take
        self.bs = int(block)
        self.block_bytes = int(block_bytes)
        self.host_wait = host_wait
        self.mut = mut or Mutations()
        self.clock = clock
        self.seq = 0
        self.inflight: Dict[int, Transfer] = {}             # seq -> a transfer not admitted yet
        self.pending: Dict[Tuple[int, int, int, int], Transfer] = {}   # (l, b, h, block) -> its in-flight transfer
        self.uid_b: Dict[Any, int] = {}
        self.cancelled: Set[Tuple] = set()
        self.key_reader: Dict[Tuple, Any] = {}              # (key, l) -> the key's latest reader event at layer l
        self.queue: List[Tuple] = []                        # (key, l) reference releases waiting for their reader gate
        self.queued: Set[Tuple] = set()
        self.hist_ev: Dict[Tuple[int, int], Any] = {}       # (b, block) -> the event of its host-row write-back (T4)
        self.hist_from: Dict[int, int] = {}                 # b -> first block that needs a registered write-back
        self.keys_seen: Dict[Tuple, Set] = {}               # (uid, epoch) -> the cohorts m of the keys that resolved
        self.metrics = Metrics(self)

    # ------------------------------------------------------------------------------------------------ requests
    def stream(self, l: int, b: int, h: int):
        return self.pools[l].stream(b, h)

    def history_written(self, uid, block: int, ev) -> None:
        """The tick wrote block `block` of request uid back to the host rows (all layers); ev completes when that
        device-to-host copy has landed. A transfer of the block carries it as hist_ev: the host packer waits on it
        before it reads the rows (K3X T4), the urgent gather is ordered after it. The tick never waits on the host."""
        self.hist_ev[(self.uid_b[uid], int(block))] = ev

    def _hist_ev_for(self, b: int, block: int):
        if b in self.hist_from and block >= self.hist_from[b] and (b, block) not in self.hist_ev:
            raise OwnershipViolation("block %d of batch slot %d has no registered host write-back (history_written): its "
                                     "host rows may not exist yet" % (block, b))
        return self.hist_ev.get((b, block))

    def begin_request(self, b: int, uid, hist_blocks: Optional[int] = None) -> None:
        """hist_blocks: the blocks whose host rows the prefill wrote; a later block may be requested only after its
        history_written (None: no check)."""
        if uid in self.uid_b or not 0 <= b < self.B:
            raise OwnershipViolation("begin_request: uid %s already holds a batch slot, or slot %d is out of range"
                                     % (uid, b))
        for pool in self.pools:
            for h in range(self.H):
                if pool.stream(b, h).owner is not None:
                    raise OwnershipViolation("begin_request: batch slot %d is owned by %s" % (b, pool.stream(b, h).owner))
        for pool in self.pools:
            for h in range(self.H):
                pool.stream(b, h).owner = uid
        self.uid_b[uid] = b
        if hist_blocks is not None:
            self.hist_from[b] = int(hist_blocks)

    def end_request(self, uid) -> None:
        """K3X _end_request / Stream.end_owner (dispatcher.py:831-875): LRU / PO / VRES released through the reader
        gates; transfers in flight stay in flight and are dropped at arrival; the uid's keys become refused."""
        b = self.uid_b.pop(uid)
        for pool in self.pools:
            for h in range(self.H):
                pool.stream(b, h).end_owner()
        for k in [k for k in self.hist_ev if k[0] == b]:
            del self.hist_ev[k]
        self.hist_from.pop(b, None)
        for (key, l) in list(self.queued):
            if key[0] == uid:
                self.cancelled.add(key)
        self.queue = [(k, l) for (k, l) in self.queue if k[0] != uid]

    # ------------------------------------------------------------------------------------------------ polling
    def _admit(self, tr: Transfer) -> Dict:
        st = self.stream(tr.layer, tr.b, tr.h)
        res = st.admit(tr.slot, tr.uid, tr.block, tr.gen, self.mut, demand=tuple(sorted(tr.demand)))
        self.inflight.pop(tr.seq, None)
        if self.pending.get((tr.layer, tr.b, tr.h, tr.block)) is tr:
            del self.pending[(tr.layer, tr.b, tr.h, tr.block)]
        self.metrics.arrived(tr, res, tr.requester in self.cancelled)
        return res

    def poll_arrivals(self) -> int:
        """K3X _poll, the arrival half (dispatcher.py:1499-1539): every transfer whose OWN completion is observed (or
        that the main stream landed in place: an urgent gather under defer_admit) is admitted with its merged demand set
        (non-blocking; no arrival order is assumed). Admission order: by (layer,
        batch slot, head, issue sequence): per stream the issue order, and streams are independent, so the state does
        not depend on how the issues of different layers interleave (S-B issues its draft transfers at end_tick)."""
        self.tp.poll()
        n = 0
        order = sorted(self.inflight.values(), key=lambda tr: (tr.layer, tr.b, tr.h, tr.seq))
        for tr0 in order:
            tr = self.inflight.get(tr0.seq)
            if tr is None:
                continue
            if tr.seq not in self._landed and not self._arrived(tr):
                continue
            self._landed.discard(tr.seq)
            self._admit(tr)
            n += 1
        return n

    def _arrived(self, tr: Transfer) -> bool:
        """THE ARRIVAL RULE of one transport transfer (note 12k). Free timing (arrival_lag None): its own completion is
        observed. Fixed lag D: the poll belongs to tick issue_tick + D or later; a transfer not landed by then is FORCED
        here (counted, timed), one landed earlier waits for that tick (counted as held). The state then depends on the
        schedule only, never on when the transport lands."""
        if self.arrival_lag is None:
            return self.tp.done(tr)
        if self.tick_no < getattr(tr, "issue_tick", 0) + self.arrival_lag:
            if self.tp.done(tr):
                self.arrival_held += 1
            return False
        if not self.tp.done(tr):
            t0 = time.perf_counter_ns()
            self.tp.force(tr)
            self.arrival_forced += 1
            self.arrival_forced_ns += time.perf_counter_ns() - t0
        return True

    def poll(self) -> None:
        """The start of a tick: arrivals, then the queued reference releases whose (key, l) reader event completed
        (dispatcher.py:1553-1588), then RELEASING -> FREE behind each slot's last reader (LayerPool.poll_releases)."""
        self.poll_arrivals()
        keep = []
        for (key, l) in self.queue:
            if self.mut.prot_release_no_reader_gate or ev_done(self.key_reader.get((key, l))):
                self._drop(key, l)
            else:
                keep.append((key, l))
        self.queue = keep
        for pool in self.pools:
            pool.poll_releases(self.mut)
        self.metrics.sample_memory()

    def _drop(self, key: Tuple, l: int) -> None:
        b = self.uid_b.get(key[0])
        if b is None:
            return                                                    # the request ended: end_owner dropped it all
        for h in range(self.H):
            st = self.stream(l, b, h)
            for idx in st.unprotect(key, self.mut):
                st.release(idx, "unprotect")

    def _queue(self, key: Tuple, l: int) -> None:
        if (key, l) in self.queued:
            return
        self.queued.add((key, l))
        self.queue.append((key, l))

    # ------------------------------------------------------------------------------------------------ slots
    def _fill_bias(self, l: int, b: int, h: int, slot: int, block: int) -> None:
        if self.store_b is None or self.cis is None:
            return
        s0, r0 = slot * self.bs, block * self.bs
        self.store_b[l][b, s0:s0 + self.bs, h].copy_(self.cis[l][b, r0:r0 + self.bs, h], non_blocking=True)

    def _verifier_candidates(self, l: int, rows: Sequence[Row], hist) -> Dict[Tuple[int, int], Any]:
        """POLICY E, the verifier's eviction (notes 12h, 12j): per stream (b, h) of a verify row of layer l, what the
        verifier may evict when K5 grants it no slot, in this order: (1) the LRU entries UNPROTECTED at the start of the
        layer and selected by NO row of the layer (so no read of this layer uses them), least recent first; (2) the
        unprotected in-flight DRAFT prefetches that no row of the layer selects, oldest issue first
        (_inflight_candidates). Items: (kind "lru" | "inflight", block, slot, transfer or None). Computed once per
        layer, before any verify protection or take (those touch only selected blocks, so the list stays valid)."""
        pool = self.pools[l]
        sel: Dict[Tuple[int, int], Set[int]] = {}
        for i, r in enumerate(rows):
            for h in range(self.H):
                sel.setdefault((r.b, h), set()).update(hist[i][h])
        out: Dict[Tuple[int, int], Any] = {}
        for r in rows:
            if r.kind != "v":
                continue
            for h in range(self.H):
                if (r.b, h) in out:
                    continue
                st = pool.stream(r.b, h)
                out[(r.b, h)] = deque([("lru", blk, idx, None) for blk, idx in st.lru.items()
                                       if not st.prot.get(blk) and blk not in sel[(r.b, h)]]
                                      + self._inflight_candidates(l, st, sel[(r.b, h)]))
        return out

    def _inflight_candidates(self, l: int, st, sel) -> List[Tuple]:
        """POLICY E's second kind (note 12j): the stream's in-flight DRAFT prefetches of the current owner that no live
        cohort protects (their requesters were decided or cancelled) and that no row of the layer selects, in issue
        order. With a synchronous transport such a prefetch has landed by the next poll and is an unprotected LRU entry
        (kind 1); with an asynchronous one it still holds its slot. policy_e_inflight=False (NEGATIVE CONTROL, the
        rule of 12h) leaves them out."""
        return self._unprotected_prefetches(l, st, sel) if self.policy_e_inflight else []

    def _unprotected_prefetches(self, l: int, st, sel) -> List[Tuple]:
        out = []
        for blk, idx in st.pending.items():
            tr = self._in_flight(l, st.b, st.h, blk)
            if (tr is None or tr.slot != idx or tr.kind != DRAFT or tr.seq in self._landed or st.prot.get(blk)
                    or blk in sel or st.slots[idx].state != INFLIGHT):
                continue
            out.append(("inflight", blk, idx, tr))
        return sorted(out, key=lambda c: c[3].seq)

    def _evict_inflight(self, st, l: int, blk: int, idx: int, tr) -> None:
        """Policy E, an in-flight candidate (note 12j): the draft prefetch of block blk into slot idx is SUPERSEDED.
        Paper 3.4: storage becomes reusable only after its owners and readers finished; this prefetch's only owner is
        the transfer itself. So its write completes BEFORE the slot is reused: here (v1, the host) the transport's
        force if it has not landed; S-C's device waited for it before its urgent gather (the host then finds it
        landed). The service forgets the transfer (it is never admitted); the slot goes FREE at once."""
        s = st.slots[idx]
        if (tr is None or self._in_flight(l, st.b, st.h, blk) is not tr or tr.slot != idx or s.state != INFLIGHT
                or s.block != blk or st.prot.get(blk) or tr.kind != DRAFT or tr.seq in self._landed):
            raise OwnershipViolation("verifier eviction: block %d at slot %d of (l %d b %d h %d) is not an unprotected "
                                     "in-flight draft prefetch" % (blk, idx, l, st.b, st.h))
        if not self.tp.done(tr):
            self.tp.force(tr)                                     # the prefetch's write lands first (T3 for writers)
        self.inflight.pop(tr.seq, None)
        if self.pending.get((l, st.b, st.h, blk)) is tr:
            del self.pending[(l, st.b, st.h, blk)]
        if st.pending.get(blk) == idx:
            del st.pending[blk]
        tr.superseded = True
        st.release(idx, "verifier_evict_inflight")
        self.metrics.count("verifier_evictions")
        self.metrics.count("verifier_inflight_evictions")
        s.state = FREE
        s.block, s.owner, s.cls = -1, None, None
        st.log.append(("free", -1, s.idx, s.gen, None))

    def _capacity_report(self, l: int, b: int, h: int) -> str:
        """The capacity state of one stream, for a CapacityRefusal (the evidence the next refusal carries)."""
        pool = self.pools[l]
        st = pool.stream(b, h)
        return ("(l %d b %d h %d: usage %d of C %d; LRU %d, unprotected %d; in flight %d, unprotected draft prefetches "
                "%d; PO %d; VRES %d; RELEASING %d; layer extras free %d of X %d, R_ver %d)"
                % (l, b, h, st.usage(), self.C, len(st.lru), sum(1 for blk in st.lru if not st.prot.get(blk)),
                   st.count(INFLIGHT), len(self._unprotected_prefetches(l, st, ())), len(st.po), len(st.vres),
                   st.count(RELEASING), pool.free_extras(), self.X, self.R_ver))

    def _evict_now(self, st, blk: int, idx: int) -> None:
        """Policy E: evict the LRU entry (blk, slot idx) for the verifier and make its slot FREE at once. Its last
        reader read it in an EARLIER tick (the entry is selected by no row of this layer): on the GPU it is ordered
        before the urgent gather on the main stream; here it is host-waited if pending (counted). The FREE transition
        is LayerPool.poll_releases' for this one slot (the vendored code is unchanged)."""
        s = st.slots[idx]
        if st.lru.get(blk) != idx or st.prot.get(blk) or s.state != LRU or s.block != blk:
            raise OwnershipViolation("verifier eviction: block %d at slot %d of (l %d b %d h %d) is not an unprotected LRU "
                                     "entry" % (blk, idx, st.l, st.b, st.h))
        st.lru.pop(blk)
        st.release(idx, "verifier_evict")
        self.metrics.count("verifier_evictions")
        if not ev_done(s.last_reader):
            if self.host_wait is None:
                raise OwnershipViolation("verifier eviction of slot %d: its last reader is pending" % idx)
            self.host_wait(s.last_reader)
            self.metrics.count("verifier_force_waits")
        s.state = FREE
        s.block, s.owner, s.cls = -1, None, None
        st.log.append(("free", -1, s.idx, s.gen, None))

    def _free_slot_for_verifier(self, pool, st) -> Optional[int]:
        """A FREE slot for an urgent miss: K5 for the verifier first (local, then the shared extras with the reserve
        R_ver), else Policy E: the stream's next eviction candidate (_verifier_candidates), evicted and reused at once.
        None: no slot (every LRU entry is protected or read by this layer): a CapacityRefusal (a configuration result)."""
        if pool.can_take(st, VERIFY):
            idx = pool.first_free(st)
            if idx is not None:
                return idx
        cands = self._vcands.get((st.b, st.h))
        if cands:
            kind, blk, idx, tr = cands.popleft()
            if kind == "lru":
                self._evict_now(st, blk, idx)
            else:
                self._evict_inflight(st, st.l, blk, idx, tr)
            return idx
        return None

    def _take(self, l: int, b: int, h: int, block: int, key: Tuple, kind: str, at: Optional[int] = None,
              evicted: bool = False) -> Optional[Transfer]:
        """K3X _issue (dispatcher.py:3098-3179): take a FREE slot (K5 decides a draft may borrow), forward the slot's
        previous last reader to the transfer (T3), fill the bias rows. A refused draft fetch returns None.
        at: the slot a DEVICE decision chose (S-B: a draft take; S-C: a verifier urgent take); the host checks it is
        what v1 would take (K5 allows the take for that kind, the block is neither in flight nor resident, the slot is
        K3X first_free), and raises otherwise; evicted: the slot is the one Policy E just evicted and freed for this
        verify take (it must be FREE; K5 refused). A verifier urgent take of S-C does not fill the bias rows: the
        device's urgent gather wrote them with K and V before the layer's reads."""
        pool = self.pools[l]
        st = pool.stream(b, h)
        hev = self._hist_ev_for(b, int(block))                       # T4: refused before any change of state
        if at is not None:
            if kind not in (DRAFT, VERIFY) or (evicted and kind != VERIFY):
                raise ValueError("kind %r (evicted %s)" % (kind, evicted))
            if self._in_flight(l, b, h, block) is not None or st.resident(block) is not None:
                raise OwnershipViolation("device decision: a duplicate take of block %d (layer %d b %d h %d)"
                                         % (block, l, b, h))
            if evicted:
                if st.slots[at].state != FREE:
                    raise OwnershipViolation("device decision: the evicted slot %d of (l %d b %d h %d) is not FREE"
                                             % (at, l, b, h))
            elif not pool.can_take(st, kind) or pool.first_free(st) != at:
                raise OwnershipViolation("device decision: the device took slot %d for the %s block %d (layer %d b %d "
                                         "h %d), the host would take %s"
                                         % (at, kind, block, l, b, h, pool.first_free(st) if pool.can_take(st, kind)
                                            else "nothing (K5 refuses)"))
            idx = at
        elif kind == DRAFT:
            idx = pool.first_free(st) if self._draft_can_take(pool, st) else None
            if idx is None:
                self.metrics.count("draft_refused")
                return None
        else:
            idx = self._free_slot_for_verifier(pool, st)
            if idx is None:
                raise CapacityRefusal("layer %d stream (b %d h %d): no slot for the verifier's urgent block %d (C %d, "
                                      "X %d, R_ver %d): K5 grants none and Policy E has no candidate (every LRU entry "
                                      "and in-flight draft prefetch is protected or read by this layer) %s"
                                      % (l, b, h, block, self.C, self.X, self.R_ver, self._capacity_report(l, b, h)))
        prev_reader = st.slots[idx].last_reader
        self.seq += 1
        gen = st.take(idx, st.owner, int(block), self.seq, cls="v" if kind == VERIFY else "ad")
        tr = Transfer(self.seq, st.owner, l, b, h, int(block), idx, gen, kind, key, hist_ev=hev, reader_ev=prev_reader)
        tr.issue_tick = self.tick_no                                  # the tick that issues it (arrival_lag)
        self.inflight[tr.seq] = tr
        self.pending[(l, b, h, int(block))] = tr
        if not (at is not None and kind == VERIFY):
            self._fill_bias(l, b, h, idx, int(block))
        self.metrics.issued(tr)
        return tr

    def _wait(self, tr: Transfer) -> None:
        """The verifier's wait for an in-flight block (paper 3.2): a device stream wait when the transport has one (the
        host admits at a later poll), else a host force followed by the admission."""
        if self.mut.read_not_ready:
            return                                                    # NEGATIVE CONTROL: no wait
        sw = getattr(self.tp, "stream_wait", None)
        if callable(sw):
            sw(tr)
            return
        if not self.tp.done(tr):
            self.tp.force(tr)
        if not self.defer_admit:
            self._admit(tr)

    # ------------------------------------------------------------------------------------------------ one layer
    @staticmethod
    def _hist(sel_row: torch.Tensor, bound: int) -> List[int]:
        return sorted({int(t) for t in sel_row.tolist() if 0 <= int(t) < bound})

    def _in_flight(self, l: int, b: int, h: int, t: int) -> Optional[Transfer]:
        """The transfer in flight for (l, b, h, block t) of the stream's CURRENT owner, if any (one per stream and
        block: the dedup). A transfer of a request that ended stays in flight until it lands and is dropped (T3); it
        carries the old request's rows, so the next request of the batch slot never merges with it or reads it (K3X
        Stream.pending holds the current owner's transfers only)."""
        tr = self.pending.get((l, b, h, t))
        if tr is not None and tr.uid != self.pools[l].stream(b, h).owner:
            return None
        return tr

    def _draft_can_take(self, pool, st) -> bool:
        """K5: a draft fetch may borrow a shared extra only while more than R_ver extras are free."""
        return pool.can_take(st, DRAFT)

    def _draft_readable(self, st, l: int, b: int, h: int, t: int) -> Optional[int]:
        """The slot a draft row may read for block t: resident (LRU / PO / VRES), arrival observed; else None."""
        return st.resident(t)

    def _check_rows(self, l: int, rows: Sequence[Row], sel: torch.Tensor) -> None:
        H, R = self.H, len(rows)
        if sel.dim() != 3 or tuple(sel.shape[:2]) != (H, R):
            raise ValueError("sel must be (H=%d, R=%d, k), got %s" % (H, R, tuple(sel.shape)))
        for r in rows:
            if r.kind not in ("d", "v"):
                raise ValueError("row kind %r" % (r.kind,))
            if r.key in self.cancelled:
                raise OwnershipViolation("a cancelled cohort key %s may not read (layer %d)" % (r.key, l))
            if self.uid_b.get(r.key[0]) != r.b:
                raise OwnershipViolation("row of uid %s names batch slot %d" % (r.key[0], r.b))
            if (r.key, l) in self.queued and not self.mut.no_late_protection_guard:
                raise OwnershipViolation("cohort %s protects at layer %d after its release was queued" % (r.key, l))
        for r in rows:
            self.keys_seen.setdefault((r.key[0], r.key[1]), set()).add(r.key[2])

    def _new_plan(self, rows: Sequence[Row]) -> Plan:
        plan = Plan(page_table=None, num_pages=None, last_len=None, seq_idx=None,
                    blocks=[[[] for _ in range(self.H)] for _ in range(len(rows))])
        plan.keys = [r.key for r in rows]
        return plan

    def resolve(self, l: int, rows: Sequence[Row], sel: torch.Tensor) -> Plan:
        """One layer of a tick (module docstring). sel (H, R, k) int64 (CPU): each row's selection per head.
        The draft rows follow K3X's ORDERING CONTRACT O1 (dispatcher.py:2899-2932): every draft row's READY set is read
        from the state before the draft recency update; the recency (touch / acquire, then the oldest eviction per
        (row, stream)) runs after all draft rows. poll_in_resolve False: arrivals are polled at the tick's start only;
        defer_issue True: the draft transfers are issued at end_tick (the S-B schedule, r2_residency_sb.py)."""
        H, R = self.H, len(rows)
        self._check_rows(l, rows, sel)
        if self.poll_in_resolve:
            self.poll_arrivals()
        plan = self._new_plan(rows)
        hist = [[self._hist(sel[h, i], rows[i].hist_bound) for h in range(H)] for i in range(R)]
        entries = self._verify_part(l, rows, hist, plan)
        got, issued = self._draft_phase_a(l, rows, hist, plan)
        entries.update(got)
        self._draft_recency(l, rows, got)
        if issued:
            plan.issued = issued
            if self.defer_issue:
                self._deferred.append((l, issued))
            else:
                self.tp.issue(l, issued)
        self._finish_plan(l, rows, entries, plan)
        return plan

    def end_tick(self) -> None:
        """The end of a tick: the deferred draft transfers are issued, layer by layer, in their order; the tick index
        advances (arrival_lag counts ticks by it)."""
        for (l, trs) in self._deferred:
            self.tp.issue(l, trs)
        self._deferred = []
        self.tick_no += 1

    def _verify_part(self, l: int, rows: Sequence[Row], hist, plan: Plan) -> Dict[Tuple[int, int], List]:
        """The verify rows of the layer (host, v1): protect, look, merge / wait / urgent (K5, else Policy E); returns
        their (row, head) entries [(block, slot, gen)] in ascending block id."""
        H = self.H
        pool = self.pools[l]
        now = self.clock()
        vrows = [i for i, r in enumerate(rows) if r.kind == "v"]
        self._vcands = self._verifier_candidates(l, rows, hist) if vrows else {}
        for i in vrows:
            r = rows[i]
            for h in range(H):
                st = pool.stream(r.b, h)
                ready, infl, new = [], [], []
                for t in hist[i][h]:
                    if not (self.mut.no_verifier_ready_protection and st.resident(t) is not None):
                        st.protect(t, r.key, VERIFY)
                    if st.resident(t) is not None:
                        ready.append(t)
                    elif self._in_flight(l, r.b, h, t) is not None:
                        infl.append(t)
                    else:
                        new.append(t)
                plan.looks.append(dict(row=i, h=h, key=r.key, ready=ready, in_flight=infl, new=new, t=now))
                for t in infl:
                    tr = self._in_flight(l, r.b, h, t)
                    if not self.mut.drop_demand_merge:
                        tr.demand.add(VERIFY)
                    if tr not in plan.waited and tr not in plan.urgent:   # an urgent block of an earlier row of this
                        plan.waited.append(tr)                            # layer is landed by the gather, not waited
                for t in new:
                    plan.urgent.append(self._take(l, r.b, h, t, r.key, VERIFY))
        for tr in plan.waited:
            self._wait(tr)
        if plan.urgent:
            for tr in plan.urgent:                            # stream order: the slot's previous reader and the block's
                for gate in (tr.reader_ev, tr.hist_ev):       # host write-back first (on the GPU the gather runs on
                    if not ev_done(gate):                     # the main stream, after both)
                        if self.host_wait is None:
                            raise OwnershipViolation("urgent gather into slot %d before its gate" % tr.slot)
                        self.host_wait(gate)
            self.gather(l, plan.urgent)                       # main stream, before every read of this layer
            for tr in plan.urgent:
                if self.defer_admit:
                    self._landed.add(tr.seq)                  # read in place; admitted at the next poll
                else:
                    self._admit(tr)
        for lk in plan.looks:
            self.metrics.look(l, lk)
        # the verify rows' slots, fixed NOW (a later draft eviction can only make a protected entry PO, same slot)
        entries: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}     # (row, head) -> [(block, slot, gen)]
        for i in vrows:
            r = rows[i]
            for h in range(H):
                st = pool.stream(r.b, h)
                ent = []
                for t in hist[i][h]:
                    idx = st.resident(t)
                    if idx is None:
                        tr = self._in_flight(l, r.b, h, t)
                        if tr is not None and (self.defer_admit or callable(getattr(self.tp, "stream_wait", None))
                                               or self.mut.read_not_ready):
                            idx = tr.slot                     # waited on the device (or the NEGATIVE CONTROL: not waited)
                        else:
                            raise OwnershipViolation("layer %d row %d head %d: block %d is not readable" % (l, i, h, t))
                    ent.append((t, idx, st.slots[idx].gen))
                entries[(i, h)] = ent
        return entries

    def _draft_phase_a(self, l: int, rows: Sequence[Row], hist, plan: Plan):
        """The draft rows (host, v1), K3X order (row, head, ascending block): protect every selected historical block
        (hits included); a READY block (resident, read from the state BEFORE the recency update, O1) is read; an
        in-flight one gets the draft demand; a miss is taken (K5) or refused (counted; the row continues masked).
        Returns ({(row, head): [(block, slot, gen)]}, the transfers taken)."""
        pool = self.pools[l]
        got_all: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        issued: List[Transfer] = []
        for i, r in enumerate(rows):
            if r.kind != "d":
                continue
            for h in range(self.H):
                st = pool.stream(r.b, h)
                got = []
                for t in hist[i][h]:
                    resident = st.resident(t)
                    idx = self._draft_readable(st, l, r.b, h, t)
                    if not (self.mut.no_draft_hit_protection and resident is not None):
                        st.protect(t, r.key, DRAFT)
                    if idx is not None:
                        got.append((t, idx, st.slots[idx].gen))   # the slot read, fixed at the decision
                    elif self._in_flight(l, r.b, h, t) is not None:
                        if not self.mut.drop_demand_merge:
                            self._in_flight(l, r.b, h, t).demand.add(DRAFT)
                    else:
                        tr = self._take(l, r.b, h, t, r.key, DRAFT)
                        if tr is None:
                            plan.refused += 1
                        else:
                            issued.append(tr)
                plan.draft_pairs += 1
                if len(got) < len(hist[i][h]):
                    plan.masked += 1
                got_all[(i, h)] = got
        return got_all, issued

    def _draft_recency(self, l: int, rows: Sequence[Row], got_all) -> None:
        """K3X _draft_recency (dispatcher.py:2899-2932, O1): per draft row and stream, the blocks it read in ascending
        block id move to MRU (an LRU hit) or enter the LRU (a PO / VRES entry: acquisition, no transfer); then the
        oldest entries leave while the LRU holds more than C (protected ones become PO, others are released)."""
        pool = self.pools[l]
        for i, r in enumerate(rows):
            if r.kind != "d":
                continue
            for h in range(self.H):
                st = pool.stream(r.b, h)
                for (t, idx, gen) in sorted(got_all.get((i, h), [])):
                    if t in st.lru:
                        st.touch(t)
                    elif t in st.po or t in st.vres:
                        if not self.mut.skip_acquisition:
                            st.acquire(t, self.mut, evict=False)
                    else:
                        raise OwnershipViolation("the draft row %d read block %d (layer %d head %d), which is no longer "
                                                 "resident at the recency update" % (i, t, l, h))
                st.evict_to_c(self.mut)

    def _finish_plan(self, l: int, rows: Sequence[Row], entries, plan: Plan) -> None:
        """The reads and the page tables of the plan: per (row, head) the historical slots (ascending block id), then
        the row's tail slots."""
        H, R = self.H, len(rows)
        P = 1
        lists: List[List[List[Tuple[int, int, int]]]] = []    # [row][head] -> [(block, slot, gen)]
        for i, r in enumerate(rows):
            per = []
            for h in range(H):
                ent = list(entries[(i, h)])
                for (t, idx, gen) in ent:
                    plan.reads.append((r.b, h, idx, gen, VERIFY if r.kind == "v" else DRAFT))
                    self.metrics.read(l, r.b, h, idx, gen, r.kind == "v")
                tails = [(int(t), int(s), -1) for (t, s) in r.tail]
                for t, s, _ in tails:
                    if not self.tail0 <= s < self.tail0 + N_TAIL:
                        raise ValueError("tail slot %d outside [%d, %d)" % (s, self.tail0, self.tail0 + N_TAIL))
                ent = ent + tails
                per.append(ent)
                P = max(P, len(ent))
            lists.append(per)
        pt = torch.full((R, H, P), -1, dtype=torch.int32)
        npg = torch.zeros((R, H), dtype=torch.int32)
        for i in range(R):
            for h in range(H):
                ent = lists[i][h]
                plan.blocks[i][h] = [t for (t, _, _) in ent]
                if ent:
                    pt[i, h, :len(ent)] = torch.tensor([s for (_, s, _) in ent], dtype=torch.int32)
                npg[i, h] = len(ent)
        plan.page_table, plan.num_pages = pt, npg
        plan.last_len = torch.tensor([r.pos % self.bs + 1 for r in rows], dtype=torch.int32)
        plan.seq_idx = torch.tensor([r.b for r in rows], dtype=torch.int32)
        self.metrics.layer_plan(plan)
        self.metrics.sample_memory()

    def read_done(self, l: int, plan: Plan, ev) -> None:
        """After the layer's attention: ev (recorded on the main stream after it) is the last reader of every slot the
        plan read and of every (key, l); the verify keys' (key, l) references are queued behind it (K4, R1)."""
        for (b, h, idx, gen, kind) in plan.reads:
            s = self.stream(l, b, h).slots[idx]
            if s.gen != gen:
                raise OwnershipViolation("layer %d slot %d of (b %d h %d) was re-taken (gen %d -> %d) under a reader"
                                         % (l, idx, b, h, gen, s.gen))
            if s.first_reader is None:
                s.first_reader = ev
            s.last_reader = ev
        for key in set(plan.keys):
            self.key_reader[(key, l)] = ev
        for lk in plan.looks:
            self.metrics.executed(l, lk)
        if not self.mut.release_only_at_decision:
            for key in {lk["key"] for lk in plan.looks}:
                self._queue(key, l)

    # ------------------------------------------------------------------------------------------------ decisions
    def decided(self, key: Tuple) -> None:
        """The cohort's decision: its references at the layers not queued yet are queued (behind their readers)."""
        for l in range(self.L):
            self._queue(key, l)

    def cancel(self, key: Tuple) -> None:
        """A rejected or cancelled cohort (K3X _apply / _cancel_requester, dispatcher.py:3408-3480): refused from now
        on; its references go through the reader gate; its in-flight transfers KEEP their slots until they land."""
        self.cancelled.add(key)
        self.metrics.cancelled(key)
        if self.mut.free_on_cancel:                           # NEGATIVE CONTROL (T3): free the destination in flight
            for tr in list(self.inflight.values()):
                if tr.requester == key:
                    st = self.stream(tr.layer, tr.b, tr.h)
                    st.release(tr.slot, "cancel")
                    st.slots[tr.slot].last_reader = None
                    if st.pending.get(tr.block) == tr.slot:
                        del st.pending[tr.block]
                    self.pending.pop((tr.layer, tr.b, tr.h, tr.block), None)
        if not self.mut.leak_cancelled_protection:
            for l in range(self.L):
                self._queue(key, l)

    def cancel_epoch(self, uid, epoch: int) -> List[Tuple]:
        """The end of an epoch (a rejection): cancel every key (uid, epoch, m) that resolved and is not cancelled yet,
        the keys only draft rows used included. Returns the keys cancelled. The verified cohort's own key takes
        decided() first (its verify read happened); cancelling it afterwards only refuses it from then on."""
        out = []
        for m in sorted(self.keys_seen.get((uid, epoch), ())):
            key = (uid, epoch, m)
            if key not in self.cancelled:
                self.cancel(key)
                out.append(key)
        return out

    # ------------------------------------------------------------------------------------------------ report
    def memory(self) -> Dict[str, int]:
        c = {k: 0 for k in ("inflight", "lru", "po", "vres", "releasing")}
        for pool in self.pools:
            cnt = pool.counts()
            c["inflight"] += cnt[INFLIGHT]
            c["lru"] += cnt[LRU]
            c["po"] += cnt[PO]
            c["vres"] += cnt[VRES] if not self.mut.vres_counted_as_lru else 0
            c["lru"] += cnt[VRES] if self.mut.vres_counted_as_lru else 0
            c["releasing"] += cnt[RELEASING] if not self.mut.mem_no_releasing else 0
        c["slots_used"] = sum(c.values())
        c["slots_physical"] = self.L * self.B * self.H * (self.n_hist + N_TAIL + self.n_trash)
        c["retained"] = c["po"] + c["vres"]
        return c

    def report(self) -> Dict[str, Any]:
        return self.metrics.report()


# ====================================================================================================================
# the paper metrics on the R2 path (K3X definitions, paper_metrics.py:9-37, verbatim in DEFINITIONS)
# ====================================================================================================================
DEFINITIONS = {
    "unit": "one selected historical block = (request, layer, KV head, block id); bytes = blocks x block_bytes",
    "layer_call": "one verifier cohort at one layer: from its first look (resolve) to the read (read_done)",
    "ready": "(4.3) resident at the first look (LRU, PO or VRES), not fetched by this cohort",
    "in_flight": "(4.3) not resident at the first look and not fetched by this cohort: another requester's transfer",
    "new": "(4.3) fetched by this cohort at this layer: an urgent miss",
    "wait_ns": "(4.3) per block: the clock at the observed arrival minus the clock at the first look; ready 0",
    "layer_wait_ns": "(4.3) per layer_call: the clock at the read minus the clock at the first look",
    "abandoned": "(4.3) layer_calls looked at but never read (cancelled cohorts)",
    "served": "(4.6) per verifier block: the state of the slot that served the read: lru / po / vres (retained) / "
              "fetched_first / fetched_refetch (this cohort's own transfer; refetch when transferred before) / "
              "inflight_waited (another requester's transfer waited on the device, not yet admitted on the host)",
    "peak_retained": "(4.6) the maximum over the samples of the PO + VRES slots, x block_bytes",
    "prefetch_use": "(4.8) every transfer = one slot lifetime, classified by its reads (draft / verify) as draft_only / "
                    "verify_only / both / neither when it ended; still_live at the report; never_arrived = ended "
                    "before its arrival was observed; by issuer kind",
    "repeated": "(4.8) transfers minus distinct (request, layer, head, block)",
    "masked_fraction": "(D3) draft (row, head) pairs that did not read every selected block / all draft pairs",
}


class Metrics:
    """The 4.3 / 4.6 / 4.8 recorder of a Residency (definitions above)."""

    def __init__(self, svc: "Residency"):
        self.svc = svc
        self.counts: Dict[str, int] = {}
        self.k43 = dict(ready=0, in_flight=0, new=0, wait_ns=0, layer_calls=0, layer_wait_ns=0, abandoned=0)
        self.k46 = dict(lru=0, po=0, vres=0, fetched_first=0, fetched_refetch=0, inflight_waited=0, not_resident=0)
        self.peak_retained_slots = 0
        self.lifetimes: Dict[Tuple[int, int, int, int, int], Dict] = {}   # (l, b, h, slot, gen) -> record
        self.transferred: Dict[Tuple, int] = {}                           # (uid, l, h, block) -> transfers
        self.arrival_ns: Dict[int, int] = {}                              # seq -> clock at the observed arrival
        self.open_calls: Dict[Tuple, Dict] = {}                           # (key, l) -> look record
        self.masked = 0
        self.draft_pairs = 0
        self.refused = 0
        self.arrivals_after_cancel = 0

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def issued(self, tr: Transfer) -> None:
        k = (tr.uid, tr.layer, tr.h, tr.block)
        self.transferred[k] = self.transferred.get(k, 0) + 1
        self.lifetimes[(tr.layer, tr.b, tr.h, tr.slot, tr.gen)] = dict(issuer=tr.kind, rd=0, rv=0, arrived=False,
                                                                        refetch=self.transferred[k] > 1, seq=tr.seq)

    def arrived(self, tr: Transfer, res: Dict, after_cancel: bool) -> None:
        rec = self.lifetimes.get((tr.layer, tr.b, tr.h, tr.slot, tr.gen))
        if rec is not None:
            rec["arrived"] = True
        self.arrival_ns[tr.seq] = self.svc.clock()
        if after_cancel:
            self.arrivals_after_cancel += 1

    def read(self, l: int, b: int, h: int, idx: int, gen: int, verify: bool) -> None:
        rec = self.lifetimes.get((l, b, h, idx, gen))
        if rec is not None:
            rec["rv" if verify else "rd"] += 1

    def look(self, l: int, lk: Dict) -> None:
        """One verify (row, head) look of a layer_call. The unit is one block (request, layer, KV head, block id) per
        layer_call: when several verify rows of one cohort select a block (the flattened round), the FIRST look
        classifies it and later looks do not count it again (a block that an earlier row of the cohort fetched is
        "new", never "in_flight": the cohort fetched it)."""
        call = self.open_calls.setdefault((lk["key"], l), dict(t=lk["t"], ready=0, in_flight=0, new=0, wait_ns=0,
                                                               seen=set()))
        st = self.svc.stream(l, self.svc.uid_b[lk["key"][0]], lk["h"])
        for cls in ("ready", "in_flight", "new"):
            for t in lk[cls]:
                if (lk["h"], t) in call["seen"]:
                    continue
                call["seen"].add((lk["h"], t))
                call[cls] += 1
                idx = st.resident(t)
                if cls == "ready":
                    state = st.slots[idx].state if idx is not None else None
                    self.k46[{LRU: "lru", PO: "po", VRES: "vres"}.get(state, "not_resident")] += 1
                    continue
                own = cls == "new"
                if idx is None:
                    tr = self.svc._in_flight(l, st.b, st.h, t)
                    if own and tr is not None:
                        idx = tr.slot                         # its own transfer, landed in place (S-C schedule), admitted
                    else:                                     # at the next poll
                        # another requester's transfer waited on the device: not admitted on the host yet
                        self.k46["inflight_waited" if not own else "not_resident"] += 1
                        continue
                rec = self.lifetimes.get((l, st.b, st.h, idx, st.slots[idx].gen))
                if own:
                    self.k46["fetched_refetch" if rec is not None and rec["refetch"] else "fetched_first"] += 1
                else:
                    state = st.slots[idx].state
                    self.k46[{LRU: "lru", PO: "po", VRES: "vres"}.get(state, "not_resident")] += 1
                if rec is not None and rec["seq"] in self.arrival_ns:
                    call["wait_ns"] += max(0, self.arrival_ns[rec["seq"]] - lk["t"])

    def executed(self, l: int, lk: Dict) -> None:
        call = self.open_calls.pop((lk["key"], l), None)
        if call is None:
            return
        self.k43["layer_calls"] += 1
        for k in ("ready", "in_flight", "new", "wait_ns"):
            self.k43[k] += call[k]
        self.k43["layer_wait_ns"] += max(0, self.svc.clock() - call["t"])

    def cancelled(self, key: Tuple) -> None:
        for (k, l) in [x for x in self.open_calls if x[0] == key]:
            del self.open_calls[(k, l)]
            self.k43["abandoned"] += 1

    def layer_plan(self, plan: Plan) -> None:
        self.masked += plan.masked
        self.draft_pairs += plan.draft_pairs
        self.refused += plan.refused

    def sample_memory(self) -> None:
        m = self.svc.memory()
        self.peak_retained_slots = max(self.peak_retained_slots, m["retained"])

    def report(self) -> Dict[str, Any]:
        use = dict(draft_only=0, verify_only=0, both=0, neither=0, still_live=0, never_arrived=0)
        by_issuer = {DRAFT: dict(use), VERIFY: dict(use)}
        for (l, b, h, idx, gen), rec in self.lifetimes.items():
            s = self.svc.stream(l, b, h).slots[idx]
            live = s.gen == gen and s.state in (INFLIGHT, LRU, PO, VRES)
            if live:
                cls = "still_live"
            elif not rec["arrived"]:
                cls = "never_arrived"
            else:
                cls = {(False, False): "neither", (True, False): "draft_only", (False, True): "verify_only",
                       (True, True): "both"}[(rec["rd"] > 0, rec["rv"] > 0)]
            use[cls] += 1
            by_issuer[rec["issuer"]][cls] += 1
        n_tr = sum(self.transferred.values())
        bb = self.svc.block_bytes
        out = dict(
            k43=dict(self.k43, bytes={k: self.k43[k] * bb for k in ("ready", "in_flight", "new")}),
            k46=dict(self.k46, peak_retained_slots=self.peak_retained_slots,
                     peak_retained_bytes=self.peak_retained_slots * bb),
            k48=dict(prefetch_use=use, by_issuer=by_issuer, transfers=n_tr, distinct=len(self.transferred),
                     repeated=n_tr - len(self.transferred)),
            masked=self.masked, draft_pairs=self.draft_pairs,
            masked_fraction=(self.masked / self.draft_pairs if self.draft_pairs else None),
            draft_refused=self.refused, arrivals_after_cancel=self.arrivals_after_cancel, counts=dict(self.counts),
            memory=self.svc.memory(), definitions=DEFINITIONS,
            arrival=dict(lag=self.svc.arrival_lag, forced=self.svc.arrival_forced,
                         forced_ns=self.svc.arrival_forced_ns, held=self.svc.arrival_held))
        return out
