"""R2 step 7, piece 5, (B2): OwnResidency, the residency service whose OWNERSHIP AUTHORITY is the array-backed table of
r2_own2_ops (docs/R2_OWN2_B2_DESIGN.md rev 2; the lead's GO of 2026-10-21). SOURCE ONLY.

WHAT B2 REPLACES. ResidencySparse (B1, r2_residency_sparse.py) applies C's sparse record to the vendored K3X objects
(kstep/ownership.py) one record at a time. OwnResidency keeps everything of B1 / S-C that is not per-slot ownership
state and replaces that state:
  - KEPT (inherited, unchanged): the rows (_check_rows: cancelled keys I11, the late-protection guard I12, the batch
    slot), _tick_rows (the tail and trash checks I15), resolve (the device decisions device_verify / device_draft on
    the device mirror, the page tables, the device wait and urgent gather, C's sparse record with pin_records=False
    and rowsel, the draft bias fill), read_done (the key readers; the layer event), watch_flag, decided / cancel /
    cancel_epoch / history_written (host maps per key, per request, per block), the transport, Residency.end_tick
    (the draft issue), the metrics object.
  - REPLACED (B2): the slots, the LRU, PO / VRES, the protections, the readers, the owners: OwnTable (self.own); the
    device mirror's delta (sync_mirror from the table); the host consumer apply_tick(steps) (C's signature, steps[i] =
    dict(l, vrows, vidx, vdec, drows, ddec, rec, caps)) with r2_own2_ops' vectorized ops; poll (arrivals, the queued
    drops, the releases, in v1's order); begin_request / end_request; _in_flight (the current owner from the table).
  - THE TRANSFER SIDE stays host objects in this class (one vendored Transfer per take, on the vendored transport), so
    the CPU lockstep runs on the same transports as v1 and S-C; the device path replaces it with E's issue_desc (the
    take list is O.issue_desc_args' desc).
  - `pools` and `stream()` are a READ-ONLY PROJECTION: real vendored LayerPool / Stream / Slot objects built from the
    table (cached per table version), so the vendored snapshot helpers, the device stand-in (CpuDeviceSim) and the
    reports read B2's state through the vendored interface. Nothing writes them.

CHECKED / UNCHECKED (checked=True by default): checked runs every predicate of rev 2 section 3.3 in two host reads per
apply (pre-write: a set flag raises OwnershipViolation before any change; post-write: the undo log rolls back, then it
raises); and I1 / I2 after every poll. Unchecked trusts the device's decisions (the timed path).

THE JOURNAL: journal_counts counts every stream transition by kind as the vendored log does (take, release:<why>,
evict_po, acquire, admit, admit_vres, drop_unowned, drop_vres_unprotected, free, touch).

NEGATIVE CONTROLS (b2_mut, tests only; the B2 forms of C's fixture plants and of rev 2's planted kinds):
  no_dedup_check, mirror_ignores_owner, owner_blind, no_uid_check, free_on_cancel, one_batch, ignore_flags, no_t4,
  no_reader_wait, drop_rd_last, keep_rd_last_on_drop, poll_order_swapped, key_alias, trash_writer, and the record
  mutations rec_mut (drop / alter the first record of a kind, after the read and before the checks).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import r2_own2_ops as O
from .kstep.backend import ev_done
from .kstep.errors import CapacityRefusal, OwnershipViolation, StaleGeneration
from .kstep.executor import DRAFT, VERIFY
from .kstep.ownership import FREE, INFLIGHT, LRU, PO, RELEASING, VRES, LayerPool
from .kstep.transfer import Transfer
from .r2_residency import Plan, Row
from .r2_residency_sc import C_FREE, C_INFLIGHT, C_LRU, C_PO, C_RELEASING, C_VRES
from .r2_residency_sparse import DREC, VREC, ResidencySparse

STATE_OF = {C_FREE: FREE, C_INFLIGHT: INFLIGHT, C_LRU: LRU, C_PO: PO, C_VRES: VRES, C_RELEASING: RELEASING}
CLS_OF = {0: None, O.CLS_AD: "ad", O.CLS_V: "v"}
B2_MUTS = ("no_dedup_check", "mirror_ignores_owner", "owner_blind", "no_uid_check", "free_on_cancel", "one_batch",
           "ignore_flags", "no_t4", "no_reader_wait", "drop_rd_last", "keep_rd_last_on_drop", "poll_order_swapped",
           "key_alias", "rec_mut")
REC_MUTS = ("take_nonfree", "take_swap", "k5_over", "pe_not_first", "drop_refusal", "drop_v_merge", "drop_d_merge",
            "drop_ready", "drop_rowsel", "drop_vread", "ignore_overflow", "reverse_stamps", "evict_po_unpinned",
            "gen_plus_two")


class OwnResidency(ResidencySparse):
    """The residency service with B2's table as the ownership authority (module docstring). Extra arguments: keys (live
    cohort key slots per request, KEYS <= 7), checked, b2_mut (negative controls, tests only)."""

    def __init__(self, *args, keys: int = 7, checked: bool = True, b2_mut=None, **kw):
        kw["pin_records"] = False
        self.__dict__["_own_ready"] = False
        super().__init__(*args, **kw)
        self.checked = bool(checked)
        self.b2_mut = dict(b2_mut or {})
        for k in self.b2_mut:
            if k not in B2_MUTS:
                raise ValueError("b2_mut %r" % (k,))
        if self.b2_mut.get("rec_mut") and self.b2_mut["rec_mut"] not in REC_MUTS:
            raise ValueError("rec_mut %r" % (self.b2_mut["rec_mut"],))
        self.KEYS = int(keys)
        G = O.Geo(L=self.L, B=self.B, H=self.H, C=self.C, X=self.X, R_ver=self.R_ver, NBLK=self.nblk, KEYS=self.KEYS)
        self.own = O.Table(G, device=self.device)
        self.own.clock = 1
        self.win = O.ReaderWindow(cap=1 << 16, device=self.device)
        self._evobj: Dict[int, Any] = {}                  # reader id -> its event object (every id a slot may name)
        self._eid: Dict[int, int] = {}                    # id(event object) -> its reader id
        self._uid_int: Dict[Any, int] = {}
        self._uid_obj: Dict[int, Any] = {}
        self._owner_py: List[Any] = [None] * self.B       # the request owning each batch slot (the table's owner)
        self._kslot: Dict[Any, int] = {}                  # cohort key -> its key slot (I13)
        self._key_of: Dict[Tuple[int, int], Any] = {}     # (b, key slot) -> cohort key
        self._ver = 0
        self._view = None
        self._view_ver = -1
        self.rec_mut_done = None                          # (tick, kind) of an applied record mutation
        self.b2_applied = 0                               # ticks applied by the B2 consumer
        self.__dict__["_own_ready"] = True

    # ================================================================================================================
    # the projection: read-only vendored views of the table
    # ================================================================================================================
    @property
    def pools(self):
        if not self.__dict__.get("_own_ready"):
            return self.__dict__.get("_vpools")
        return self._materialize()

    @pools.setter
    def pools(self, v):
        self.__dict__["_vpools"] = v

    def _changed(self) -> None:
        self._ver += 1

    def _materialize(self) -> List[LayerPool]:
        """The table as vendored LayerPool objects (cached per table version; read-only by contract)."""
        if self._view is not None and self._view_ver == self._ver:
            return self._view
        T, G = self.own, self.own.geo
        cols = {n: T.c[n].to("cpu") for n in ("code", "block", "gen", "uid", "cls", "stamp", "rd_first", "rd_last",
                                             "seq")}
        pins = T.pin_w.to("cpu")
        sh = torch.arange(32, dtype=torch.int64)
        pools = []
        for l in range(self.L):
            pool = LayerPool(l, self.B, self.H, self.C, self.X, self.R_ver)
            for (b, h), st in pool.streams.items():
                st.owner = self._owner_py[b]
                code = cols["code"][l, b, h].tolist()
                blk = cols["block"][l, b, h].tolist()
                gen = cols["gen"][l, b, h].tolist()
                uid = cols["uid"][l, b, h].tolist()
                cls = cols["cls"][l, b, h].tolist()
                stamp = cols["stamp"][l, b, h].tolist()
                rf = cols["rd_first"][l, b, h].tolist()
                rl = cols["rd_last"][l, b, h].tolist()
                seq = cols["seq"][l, b, h].tolist()
                lru = []
                for i, s in enumerate(st.slots):
                    c = code[i]
                    s.state = STATE_OF[c]
                    s.block = blk[i]
                    s.gen = gen[i]
                    s.owner = self._uid_obj.get(uid[i]) if uid[i] >= 0 else None
                    s.cls = CLS_OF[cls[i]]
                    s.first_reader = self._evobj.get(rf[i]) if rf[i] >= 0 else None
                    s.last_reader = self._evobj.get(rl[i]) if rl[i] >= 0 else None
                    s.seq = seq[i] if c == C_INFLIGHT else None
                    if c == C_LRU:
                        lru.append((stamp[i], blk[i], i))
                    elif c == C_PO:
                        st.po[blk[i]] = i
                    elif c == C_VRES:
                        st.vres[blk[i]] = i
                    elif c == C_INFLIGHT and st.owner is not None and s.owner == st.owner:
                        st.pending[blk[i]] = i
                for (_, bk, i) in sorted(lru):
                    st.lru[bk] = i
                for plane in range(2 * self.KEYS):
                    words = pins[plane, l, b, h]
                    if not bool(words.any()):
                        continue
                    key = self._key_of.get((b, plane % self.KEYS))
                    kind = VERIFY if plane < self.KEYS else DRAFT
                    bits = (torch.bitwise_and(torch.bitwise_right_shift(words.unsqueeze(-1), sh), 1) != 0).reshape(-1)
                    for bk in bits[:self.nblk].nonzero().view(-1).tolist():
                        st.prot.setdefault(bk, set()).add(key)
                        st.pin_kind.setdefault(bk, {}).setdefault(key, set()).add(kind)
            pools.append(pool)
        self._view, self._view_ver = pools, self._ver
        return pools

    def _in_flight(self, l: int, b: int, h: int, t: int):
        """The CURRENT owner's transfer in flight for (l, b, h, block t) (res.py:509-517), the owner from the table."""
        tr = self.pending.get((l, b, h, t))
        if tr is not None and tr.uid != self._owner_py[b] and not self.b2_mut.get("owner_blind"):
            return None
        return tr

    # ================================================================================================================
    # ids: requests, readers, cohort keys
    # ================================================================================================================
    def _uid_id(self, uid) -> int:
        k = self._uid_int.get(uid)
        if k is None:
            k = len(self._uid_int) + 1
            self._uid_int[uid] = k
            self._uid_obj[k] = uid
        return k

    def _rid(self, ev) -> int:
        """The reader id of an event object (registered once; the map keeps the object alive)."""
        if ev is None:
            return -1
        e = self._eid.get(id(ev))
        if e is None:
            e = self.win.register(ev, -1)
            self._eid[id(ev)] = e
            self._evobj[e] = ev
        return e

    def _key_slot(self, key) -> int:
        """The key slot of a cohort key (I13): the lowest slot of its request that no live key holds; a key is live while
        a pin bit of its slot is set at any layer, a (key, l) release is queued, or a key reader is pending. None free:
        CapacityRefusal (C's Q-B4: refused, never aliased). key_alias (NEGATIVE CONTROL): slot 0 for every key."""
        k = self._kslot.get(key)
        if k is not None:
            return k
        b = self.uid_b[key[0]]
        if self.b2_mut.get("key_alias"):
            self._kslot[key] = 0
            self._key_of[(b, 0)] = key
            return 0
        used = {kk for (bb, kk) in self._key_of if bb == b}
        free = [kk for kk in range(self.KEYS) if kk not in used]
        if not free:
            for (bb, kk), kkey in list(self._key_of.items()):
                if bb == b and not self._key_live(b, kk, kkey):
                    del self._key_of[(bb, kk)]
                    self._kslot.pop(kkey, None)
            used = {kk for (bb, kk) in self._key_of if bb == b}
            free = [kk for kk in range(self.KEYS) if kk not in used]
        if not free:
            raise CapacityRefusal("B2: request %s holds %d live cohort keys (KEYS %d): no key slot for %s (I13)"
                                  % (key[0], self.KEYS, self.KEYS, key))
        k = free[0]
        self._kslot[key] = k
        self._key_of[(b, k)] = key
        return k

    def _key_live(self, b: int, k: int, key) -> bool:
        if bool(self.own.pin_w[[k, k + self.KEYS], :, b].any()):
            return True
        if any(kk == key for (kk, _) in self.queue):
            return True
        return any(not ev_done(self.key_reader.get((key, l))) for l in range(self.L) if (key, l) in self.key_reader)

    # ================================================================================================================
    # the device mirror from the table
    # ================================================================================================================
    def sync_mirror(self) -> bool:
        """The device mirror (S-C's DeviceMirror, read by device_verify / device_draft) from the table: the state code,
        the block, the generation, the current owner's flag (an INFLIGHT slot of the batch slot's owner, S-C's cur),
        the protected flag (a resident or INFLIGHT slot whose block has a pin bit), the recency stamp. One copy of the
        table's columns (no journal delta)."""
        T, G = self.own, self.own.geo
        code = T.c["code"]
        blk = T.c["block"].to(torch.int64)
        cur = code == C_INFLIGHT
        if not self.b2_mut.get("mirror_ignores_owner"):
            cur = cur & O.owner_of_slots(T)
        s = torch.arange(G.S, device=T.dev).view(G.S, 1).expand(G.S, G.NPOS)
        pinned = O.bit_test(O.any_pin_words(T), s, blk.view(G.S, G.NPOS).clamp(min=0)).view(code.shape) & (blk >= 0)
        live = (code == C_LRU) | (code == C_PO) | (code == C_VRES) | (code == C_INFLIGHT)
        dm = self.dmirror
        dm.code.copy_(code.to(torch.int64))
        dm.block.copy_(blk)
        dm.gen.copy_(T.c["gen"].to(torch.int64))
        dm.cur.copy_(cur.to(torch.int64))
        dm.prot.copy_((pinned & live).to(torch.int64))
        dm.stamp.copy_(T.c["stamp"])
        self._dirty = set()
        self.mirror_uploads += 1
        return True

    # ================================================================================================================
    # requests and keys
    # ================================================================================================================
    def begin_request(self, b: int, uid, hist_blocks: Optional[int] = None) -> None:
        """res.py:203-218 on the table: refuses an owned batch slot; the owner, the T4 bound."""
        if uid in self.uid_b or not 0 <= b < self.B:
            raise OwnershipViolation("begin_request: uid %s already holds a batch slot, or slot %d is out of range"
                                     % (uid, b))
        if self._owner_py[b] is not None:
            raise OwnershipViolation("begin_request: batch slot %d is owned by %s" % (b, self._owner_py[b]))
        self._owner_py[b] = uid
        self.own.owner[b] = self._uid_id(uid)
        self.uid_b[uid] = b
        if hist_blocks is not None:
            self.hist_from[b] = int(hist_blocks)
            self.own.hist_from[b] = int(hist_blocks)
        self.own.xfer[b] = 0
        self._changed()

    def history_written(self, uid, block: int, ev) -> None:
        super().history_written(uid, block, ev)                       # hist_ev (T4), the main stream's waits
        self.own.hist_id[self.uid_b[uid], int(block)] = 1
        self._changed()

    def end_request(self, uid) -> None:
        """res.py:220-234 + own.py:363-384 on the table: b's LRU / PO / VRES -> RELEASING ("end"), its pins cleared, the
        owner none (its transfers in flight are dropped at arrival); the T4 maps; the queued keys cancelled."""
        b = self.uid_b.pop(uid)
        n = O.end_owner(self.own, b)
        self._count("release:end", n)
        self._owner_py[b] = None
        for k in [k for k in self.hist_ev if k[0] == b]:
            del self.hist_ev[k]
        self.hist_from.pop(b, None)
        self.own.hist_id[b] = -1
        self.own.hist_from[b] = 1 << 40
        for (key, l) in list(self.queued):
            if key[0] == uid:
                self.cancelled.add(key)
        self.queue = [(k, l) for (k, l) in self.queue if k[0] != uid]
        for (bb, kk), key in list(self._key_of.items()):
            if bb == b:
                del self._key_of[(bb, kk)]
                self._kslot.pop(key, None)
        self._changed()

    def cancel(self, key: Tuple) -> None:
        """res.py:768-785: the key refused from now on, its references queued at every layer; its transfers in flight
        KEEP their slots. free_on_cancel (NEGATIVE CONTROL, F3's): their destinations are released at once."""
        self.cancelled.add(key)
        self.metrics.cancelled(key)
        if self.b2_mut.get("free_on_cancel"):
            G = self.own.geo
            for tr in list(self.inflight.values()):
                if tr.requester == key:
                    f = torch.tensor([((tr.layer * G.B + tr.b) * G.H + tr.h) * G.NPOS + tr.slot], device=self.device)
                    self.own.flat("code")[f] = C_RELEASING
                    self.own.flat("rd_last")[f] = -1
                    self._count("release:cancel", 1)
                    if self.pending.get((tr.layer, tr.b, tr.h, tr.block)) is tr:
                        del self.pending[(tr.layer, tr.b, tr.h, tr.block)]
            self._changed()
        if not self.mut.leak_cancelled_protection:
            for l in range(self.L):
                self._queue(key, l)

    def watch_flag(self, name: str, flag: torch.Tensor) -> None:
        if self.b2_mut.get("ignore_flags"):
            return None                                               # NEGATIVE CONTROL (F16's)
        return super().watch_flag(name, flag)

    def _count(self, kind: str, n: int) -> None:
        if n:
            self.journal_counts[kind] = self.journal_counts.get(kind, 0) + int(n)

    # ================================================================================================================
    # poll
    # ================================================================================================================
    def poll(self) -> None:
        """res.py:283-297 on the table, in v1's order: the arrivals, then the queued drops, then the releases (the poll
        order is a contract: poll_order_swapped is its NEGATIVE CONTROL). Checked: I1 / I2 after the arrivals."""
        if self.b2_mut.get("poll_order_swapped"):
            self._poll_drops()
            self._poll_arrivals()
        else:
            self._poll_arrivals()
            self._poll_drops()
        self._poll_releases()
        self.metrics.sample_memory()

    def _poll_arrivals(self) -> int:
        """res.py:245-264: every transfer whose own completion is observed (or landed in place: an urgent gather) is
        admitted, by the arrival rule (_arrived, note 12k); the admission branches are O.poll_arrivals' (own.py:322-361),
        on the slots of the arrived transfers only."""
        self.tp.poll()
        order = sorted(self.inflight.values(), key=lambda tr: (tr.layer, tr.b, tr.h, tr.seq))
        arrived = []
        for tr0 in order:
            tr = self.inflight.get(tr0.seq)
            if tr is None:
                continue
            if tr.seq not in self._landed and not self._arrived(tr):
                continue
            arrived.append(tr)
        if not arrived:
            return 0
        T, G = self.own, self.own.geo
        f = torch.tensor([((tr.layer * G.B + tr.b) * G.H + tr.h) * G.NPOS + tr.slot for tr in arrived],
                         dtype=torch.int64, device=self.device)
        gen = T.flat("gen")[f].tolist()
        code = T.flat("code")[f].tolist()
        blk = T.flat("block")[f].tolist()
        for i, tr in enumerate(arrived):                              # I6 (own.py:328-333): the arrival names its slot
            if gen[i] != tr.gen:
                raise StaleGeneration("arrival (uid %s, block %d) names slot %d gen %d but the slot holds gen %d"
                                      % (tr.uid, tr.block, tr.slot, tr.gen, gen[i]))
            if code[i] != C_INFLIGHT or blk[i] != tr.block:
                raise OwnershipViolation("arrival (uid %s, block %d, slot %d gen %d): slot is %s/%d"
                                         % (tr.uid, tr.block, tr.slot, tr.gen, STATE_OF[code[i]], blk[i]))
        T.flat("landed")[f] = True                                    # the transport's arrivals (urgent: landed)
        mut = {"keep_rd_last_on_drop": True} if self.b2_mut.get("keep_rd_last_on_drop") else None
        if self.b2_mut.get("no_uid_check"):                           # NEGATIVE CONTROL (T3): every arrival owned
            T.flat("uid")[f] = T.owner[torch.div(torch.div(f, G.NPOS, rounding_mode="floor"), G.H,
                                                 rounding_mode="floor").remainder(G.B)]
        res = O.poll_arrivals(T, self.tick_no, -1, None, None, mut)
        self._count("admit", int(res["admit_lru"]))
        self._count("admit_vres", int(res["admit_vres"]))
        self._count("drop_unowned", int(res["drop_unowned"]))
        self._count("drop_vres_unprotected", int(res["drop_vres_unprotected"]))
        self._count("evict_po", int(res["evict_po"]))
        self._count("release:evict", int(res["evict_rel"]))
        if int(res["admitted"]) != len(arrived):
            raise OwnershipViolation("B2 poll: %d slots admitted for %d arrived transfers" % (int(res["admitted"]),
                                                                                            len(arrived)))
        for tr in arrived:
            self.inflight.pop(tr.seq, None)
            if self.pending.get((tr.layer, tr.b, tr.h, tr.block)) is tr:
                del self.pending[(tr.layer, tr.b, tr.h, tr.block)]
            self._landed.discard(tr.seq)
            self.metrics.arrived(tr, None, tr.requester in self.cancelled)
        self._changed()
        if self.checked:
            pc = O.post_checks(T, None, 0, which=("I1", "I2"))
            i1, i2 = torch.stack([pc["I1"], pc["I2"]]).tolist()
            if i1 or i2:
                raise OwnershipViolation("B2 poll: after the arrivals %s fails (I1: one slot per resident or in-flight "
                                         "block; I2: the LRU holds at most C)" % ("I1" if i1 else "I2"))
        return len(arrived)

    def _poll_drops(self) -> None:
        """res.py:287-293: the queued (key, l) releases whose key reader completed; O.drop_keys on the table."""
        keep, due = [], []
        for (key, l) in self.queue:
            if self.mut.prot_release_no_reader_gate or ev_done(self.key_reader.get((key, l))):
                due.append((key, l))
            else:
                keep.append((key, l))
        self.queue = keep
        trip = []
        for (key, l) in due:
            b = self.uid_b.get(key[0])
            k = self._kslot.get(key)
            if b is None or k is None:
                continue                                              # the request ended / the key never pinned
            trip.append((l, b, k))
        trip = sorted(set(trip))
        if trip:
            d = self.device
            res = O.drop_keys(self.own, torch.tensor([t[0] for t in trip], device=d),
                              torch.tensor([t[1] for t in trip], device=d), torch.tensor([t[2] for t in trip], device=d))
            self._count("release:unprotect", res["unprotect_rel"])
            self._changed()

    def _poll_releases(self) -> None:
        """own.py:435-452: RELEASING -> FREE behind each slot's last reader (the reader window, refreshed)."""
        self.win.refresh(ev_done)
        if self.b2_mut.get("no_reader_wait") or self.mut.no_reader_wait:     # NEGATIVE CONTROL
            code = self.own.flat("code")
            fr = (code == C_RELEASING).nonzero().squeeze(1)
            for n, v in (("code", C_FREE), ("block", -1), ("uid", -1), ("cls", 0)):
                self.own.flat(n)[fr] = v
            n = int(fr.numel())
        else:
            n = O.poll_releases(self.own, self.win)
        self._count("free", n)
        if n:
            self._changed()

    # ================================================================================================================
    # the host consumer
    # ================================================================================================================
    def apply_tick(self, steps: List[Dict]) -> None:
        """THE HOST CONSUMER (C's signature, note 12l): ONE read of the flags and the record counts; the void-tick
        refusals (a watched flag, a verify overflow, a record over its capacity) before any change; then the B2 ops
        (module docstring) on all layers at once, in K3X order; then the host side (the transfers, the metrics, the
        journal, the release queue)."""
        if not steps:
            return
        names = list(self._flags)
        parts = [self._flags[n].to(torch.int64).reshape(1) for n in names]
        for st in steps:
            rec = st["rec"]
            if st["vdec"] is not None:
                parts += [rec["vcnt"].reshape(1), st["vdec"].counters.reshape(-1)[:1]]
            if st["ddec"] is not None:
                parts += [rec["dcnt"].reshape(1), st["ddec"].counters.reshape(-1)[:3]]
        flat = self._d2h(torch.cat(parts)).tolist()                   # ONE read (the flags and the counts)
        self.flag_values = dict(zip(names, flat[:len(names)]))
        raised = sorted(n for n, v in self.flag_values.items() if v)
        if raised:
            raise CapacityRefusal("S-C: the watched device flag(s) %s are set at this tick: the tick is void (none "
                                  "of its decisions is applied)" % raised)
        at = len(names)
        info = []
        over, capped = [], []
        for st in steps:
            cap_v, cap_d = st["caps"]
            p = dict(st=st, nv=0, nd=0, over=0, dc=(0, 0, 0))
            if st["vdec"] is not None:
                p["nv"], p["over"] = int(flat[at]), int(flat[at + 1])
                at += 2
                if p["nv"] > cap_v:
                    capped.append((st["l"], "verify", p["nv"], cap_v))
                if p["over"] > 0:
                    over.append((st["l"], p["over"]))
            if st["ddec"] is not None:
                p["nd"] = int(flat[at])
                p["dc"] = tuple(int(x) for x in flat[at + 1:at + 4])
                at += 4
                if p["nd"] > cap_d:
                    capped.append((st["l"], "draft", p["nd"], cap_d))
            info.append(p)
        rm = self.b2_mut.get("rec_mut")
        if over and rm != "ignore_overflow":
            rep = [dict(st=p["st"], vrec=p["st"]["rec"]["vrec"][:, :p["nv"]].t().tolist()) for p in info
                   if p["st"]["vdec"] is not None]
            raise self._refusal(over, self._sparse_overflow_report(rep, capped))
        if capped:
            raise CapacityRefusal("S-C sparse record: a layer's record holds more entries than its capacity (layer, "
                                  "kind, entries, capacity: %s): the tick is void (none of its decisions is applied)"
                                  % capped)
        self._apply_b2(steps, info)
        self.b2_applied += 1

    # ---------------------------------------------------------------------------------------------- the B2 apply
    def _tick_rows_of(self, st: Dict) -> List[Row]:
        """The tick's rows in their order (verify rows at vidx, the draft rows in the other positions)."""
        R = len(st["vrows"]) + len(st["drows"])
        out: List[Any] = [None] * R
        for j, i in enumerate(st["vidx"]):
            out[i] = st["vrows"][j]
        it = iter(st["drows"])
        for i in range(R):
            if out[i] is None:
                out[i] = next(it)
        return out

    def _apply_b2(self, steps: List[Dict], info: List[Dict]) -> None:
        T, G = self.own, self.own.geo
        d = self.device
        L, B, H = self.L, self.B, self.H
        st0 = steps[0]
        vrows, drows = st0["vrows"], st0["drows"]
        trows = self._tick_rows_of(st0)
        for st in steps:                                              # every layer resolved the same rows (S-C)
            if st["vrows"] != vrows or st["drows"] != drows:
                raise ValueError("B2: the layers of a tick resolved different rows")
        # the key slots of the tick's rows (I13: a refusal before any change)
        for r in trows:
            self._key_slot(r.key)
        k = (st0["vdec"] or st0["ddec"]).k
        vb = torch.tensor([r.b for r in vrows], dtype=torch.int64, device=d)
        vk = torch.tensor([self._kslot[r.key] for r in vrows], dtype=torch.int64, device=d)
        db = torch.tensor([r.b for r in drows], dtype=torch.int64, device=d)
        dk = torch.tensor([self._kslot[r.key] for r in drows], dtype=torch.int64, device=d)
        rows = O.Rows(vb=vb, vk=vk, db=db, dk=dk, k=int(k))
        layers = []
        for p in info:
            st = p["st"]
            rec = st["rec"]
            layers.append(dict(l=st["l"], vrec=rec["vrec"][:, :p["nv"]].clone() if st["vdec"] is not None else None,
                               drec=rec["drec"][:, :p["nd"]].clone() if st["ddec"] is not None else None))
        Wb, Wp = G.WB, G.WP
        R = len(trows)
        rowsel = torch.zeros((L, R, H, Wb), dtype=torch.int64, device=d)
        sel = torch.zeros((L, B, H, Wb), dtype=torch.int64, device=d)
        vread = torch.zeros((L, B, H, Wp), dtype=torch.int64, device=d)
        vlook = {}
        for p in info:
            st = p["st"]
            rec = st["rec"]
            l = st["l"]
            rowsel[l] = rec["rowsel"]
            sel[l] = rec["sel"]
            if st["vdec"] is not None:
                vread[l] = rec["vread"]
                vlook[l] = rec["vlook"].tolist()
        row_b = torch.tensor([r.b for r in trows], dtype=torch.int64, device=d)
        row_plane = torch.tensor([self._kslot[r.key] + (0 if r.kind == "v" else self.KEYS) for r in trows],
                                 dtype=torch.int64, device=d)
        self._rec_mutate(layers, rowsel, vread, rows, row_b, row_plane)
        P = O.parse(T, layers, rows)
        ev = O.k5_events(P, T)
        so = O.slot_of(T)
        checked = self.checked
        mut = {}
        if self.b2_mut.get("one_batch"):
            mut["one_batch"] = True
        rm = self.b2_mut.get("rec_mut")
        if rm in ("reverse_stamps", "evict_po_unpinned", "gen_plus_two"):
            mut[rm] = True
        # ---------------------------------------------------------------- the pre-write predicates (one read)
        pre: Dict[str, torch.Tensor] = {}

        def note(dd):
            for n, v in dd.items():
                pre[n] = v.sum() if v.dim() else v.to(torch.int64)
        for part in ("V", "D"):
            if P[part] is not None:
                pre["hole_" + part] = O.has(P[part]["flags"], O.HOLE).sum()
        mc, fv, fd = O.merge_check(T, P, so, ev, check=checked)
        if checked:
            note(O.k5_check(T, ev))
            tc = O.take_checks(T, ev, so)
            if self.b2_mut.get("no_dedup_check"):
                tc.pop("dedup")
                tc.pop("take_twice")
            if self.b2_mut.get("no_t4"):
                tc.pop("t4")
            note(tc)
            if P["D"] is not None:
                note(O.ready_check(T, P["D"]))
            note(mc)
            note(O.policy_e_check(T, ev, P, sel, O.any_pin_words(T), self.policy_e_inflight))
        else:
            tc = O.take_checks(T, ev, so)                             # T4 and the holes stay refusals unchecked
            note(dict(t4=tc["t4"]) if not self.b2_mut.get("no_t4") else {})
        names = list(pre)
        vals = torch.stack([pre[n] for n in names]).tolist() if names else []
        viol = {n: int(v) for n, v in zip(names, vals) if v}
        if viol:
            self._raise_pre(viol, P, ev, rows)
        n_ref = int(O.has(P["D"]["flags"], O.REFUSED).sum()) if P["D"] is not None else 0
        dev_ref = sum(p["dc"][1] for p in info if p["st"]["ddec"] is not None)
        if checked and n_ref != dev_ref:
            raise OwnershipViolation("S-B: %d refusals applied, the device counted %d" % (n_ref, dev_ref))
        # ---------------------------------------------------------------- the victims of Policy E (before the writes)
        pe = (ev["kind"] == 1).nonzero().squeeze(1)
        victims = []
        if pe.numel():
            fpe = ev["s"][pe] * G.NPOS + ev["take_slot"][pe]
            vcode = T.flat("code")[fpe].tolist()
            vblk = T.flat("block")[fpe].tolist()
            vrd = T.flat("rd_last")[fpe].tolist()
            for i, q in enumerate(pe.tolist()):
                victims.append((int(ev["l"][q]), int(ev["b"][q]), int(ev["h"][q]), vblk[i], vcode[i], vrd[i]))
        # ---------------------------------------------------------------- the writes
        gen_before = T.c["gen"].clone() if checked else None
        undo = O.Undo() if checked else None
        if undo is not None:
            clock0 = T.clock
            undo.call(lambda: setattr(T, "clock", clock0))
        O.pins_or(T, rowsel, row_b, row_plane, undo)
        win_mut = dict(mut)
        if self.b2_mut.get("drop_rd_last"):
            win_mut["drop_rd_last"] = True
        takes = O.write_takes(T, ev, self.tick_no, self.tick_no, self.win, undo, win_mut)
        O.write_merges(T, fv, fd, undo)
        rc = O.recency(T, P["D"], rows, undo, mut)
        if rm in ("reverse_stamps", "evict_po_unpinned", "gen_plus_two") and self.rec_mut_done is None:
            hit = dict(reverse_stamps=rc.get("max_stream_reads", 0) >= 2,           # two touches of one stream
                       evict_po_unpinned=int(rc.get("forced_po", 0)) > 0,           # an unpinned victim made PO
                       gen_plus_two=takes["f"].numel() > 0)[rm]
            if hit:
                self.rec_mut_done = (self.tick_no, rm)                # the op-level control had an effect here
        if checked:
            post = dict(O.post_checks(T, gen_before, int(takes["f"].numel())))
            post["recency_bad"] = rc["bad"]
            pn = list(post)
            pv = torch.stack([post[n].to(torch.int64) for n in pn]).tolist()
            bad = {n: int(v) for n, v in zip(pn, pv) if v}
            if bad:
                undo.rollback()
                raise OwnershipViolation("B2 post-write predicates %s fail: the tick is void (rolled back)" % sorted(bad))
        elif int(rc["bad"]):
            raise OwnershipViolation("B2: a draft row read a block that is no longer resident at the recency update")
        # ---------------------------------------------------------------- the readers (the layers' events)
        e_l = torch.tensor([self._rid(self._layer_ev.get(l)) for l in range(L)], dtype=torch.int64, device=d)
        if self.b2_mut.get("rec_mut") == "drop_vread":
            pass                                                      # (the bit was cleared in _rec_mutate)
        O.read_done(T, vread, P["D"], rows, e_l)
        self._changed()
        # ---------------------------------------------------------------- the host side
        self._host_side(steps, info, P, ev, takes, victims, rc, rows, vlook, fv, fd)

    def _raise_pre(self, viol: Dict[str, int], P, ev, rows) -> None:
        """A pre-write predicate failed: the tick is void before any change. The message names the predicates (and the
        vendored check each one repeats)."""
        what = {"hole_V": "a verify block has no slot and the device did not flag an overflow",
                "hole_D": "a draft block has no decision",
                "k5": "K5 (LayerPool.can_take) refuses a take or grants a refused one",
                "first_free": "a take is not the stream's first FREE slot (K3X first_free)",
                "slot_range": "a record names a slot outside the pool (I15)",
                "dedup": "a take of a block resident or in flight for the owner (I1, res.py:453)",
                "take_twice": "two takes of one (stream, block) in the tick",
                "t4": "block has no registered host write-back (history_written): its host rows may not exist yet",
                "ready": "a READY draft record names a slot that is not the resident slot of its block",
                "v_merge": "a verify merge names no owner's transfer in flight (slot, generation)",
                "v_dup": "a DUP record names no take of this tick", "d_merge": "a draft merge names no transfer",
                "policy_e": "a Policy E eviction is not the host's next candidate (I7)"}
        msg = "; ".join("%s x%d: %s" % (n, v, what.get(n, n)) for n, v in sorted(viol.items()))
        raise OwnershipViolation("B2 pre-write predicates fail (the tick is void, nothing applied): " + msg)

    def _host_side(self, steps, info, P, ev, takes, victims, rc, rows, vlook, fv, fd) -> None:
        """After the commit: the transfers of the takes (vendored Transfer objects, on the transport), Policy E's
        victims (verifier_evictions; the superseded prefetches), the demand merges, the refusals, the looks (4.3 / 4.6),
        the journal, the release queue, the layer plans (masked, draft pairs, refusals)."""
        G = self.own.geo
        V, D = P["V"], P["D"]
        vrows, drows = steps[0]["vrows"], steps[0]["drows"]
        # Policy E's victims (res.py:399-418, :363-388)
        for (l, b, h, oblk, ocode, ord_) in victims:
            self.metrics.count("verifier_evictions")
            if ocode == C_INFLIGHT:
                tr = self.pending.get((l, b, h, oblk))
                if tr is not None:
                    if not self.tp.done(tr):
                        self.tp.force(tr)
                    self.inflight.pop(tr.seq, None)
                    del self.pending[(l, b, h, oblk)]
                    tr.superseded = True
                self.metrics.count("verifier_inflight_evictions")
                self._count("release:verifier_evict_inflight", 1)
            else:
                ev_r = self._evobj.get(ord_) if ord_ >= 0 else None
                if ev_r is not None and not ev_done(ev_r):
                    self.host_wait(ev_r)
                    self.metrics.count("verifier_force_waits")
                self._count("release:verifier_evict", 1)
            self._count("free", 1)
        # the takes, in K3X order (the vendored _take's Transfer, res.py:476-489)
        tk = (ev["kind"] != 2).nonzero().squeeze(1).tolist()
        part = ev["part"].tolist()
        o = ev["o"].tolist()
        H, kk = self.H, rows.k
        lt, bt, ht = takes["l"].tolist(), takes["b"].tolist(), takes["h"].tolist()
        blk, slot, gen = takes["blk"].tolist(), takes["slot"].tolist(), takes["gen"].tolist()
        rdo = takes["rd_old"].tolist()
        by_layer: Dict[int, List[Transfer]] = {}
        new_trs = []
        for i, q in enumerate(tk):
            j = o[q] // (H * kk)
            isv = part[q] == 0
            r = vrows[j] if isv else drows[j]
            self.seq += 1
            prev = self._evobj.get(rdo[i]) if rdo[i] >= 0 else None
            if self.b2_mut.get("drop_rd_last"):
                prev = None
            tr = Transfer(self.seq, self._owner_py[bt[i]], lt[i], bt[i], ht[i], blk[i], slot[i], gen[i],
                          VERIFY if isv else DRAFT, r.key, hist_ev=self.hist_ev.get((bt[i], blk[i])), reader_ev=prev)
            tr.issue_tick = self.tick_no
            self.inflight[tr.seq] = tr
            self.pending[(lt[i], bt[i], ht[i], blk[i])] = tr
            self.metrics.issued(tr)
            if isv:
                tr.delivered = True                                   # the device's urgent gather landed it
                self._landed.add(tr.seq)
            else:
                by_layer.setdefault(lt[i], []).append(tr)
            new_trs.append((q, tr))
        self._count("take", len(tk))
        for l in sorted(by_layer):
            self._deferred.append((l, by_layer[l]))
        # the demand merges on the transfers (res.py:605-611, :672-674)
        if V is not None:
            m = O.has(V["flags"], O.INFL) | O.has(V["flags"], O.DUP)
            for (l, b, h, t) in zip(V["l"][m].tolist(), V["b"][m].tolist(), V["h"][m].tolist(), V["blk"][m].tolist()):
                tr = self._in_flight(l, b, h, t)
                if tr is not None and not self.mut.drop_demand_merge:
                    tr.demand.add(VERIFY)
        if D is not None:
            m = O.has(D["flags"], O.INFL)
            for (l, b, h, t) in zip(D["l"][m].tolist(), D["b"][m].tolist(), D["h"][m].tolist(), D["blk"][m].tolist()):
                tr = self._in_flight(l, b, h, t)
                if tr is not None and not self.mut.drop_demand_merge:
                    tr.demand.add(DRAFT)
            nref = int(O.has(D["flags"], O.REFUSED).sum())
            for _ in range(nref):
                self.metrics.count("draft_refused")
        # the journal of the recency (touch, acquire, evict_to_c)
        self._count("touch", int(rc["touch"]))
        self._count("acquire", int(rc["acquire"]))
        self._count("evict_po", int(rc["evict_po"]))
        self._count("release:evict", int(rc["evict_rel"]))
        # the looks (4.3 / 4.6), as C's sparse consumer (r2_residency_sparse.py _apply_verify_sparse)
        now = self.clock()
        trs_of = {q: tr for (q, tr) in new_trs}
        firsts: Dict[Tuple[int, Any], List[Tuple[int, int, str, Any]]] = {}
        if V is not None:
            vfl = V["flags"].tolist()
            vl, vj, vh, vblk = V["l"].tolist(), V["j"].tolist(), V["h"].tolist(), V["blk"].tolist()
            src_q = {}
            if ev["o"].numel():
                for q, (pq, sq) in enumerate(zip(part, ev["src"].tolist())):
                    if pq == 0:
                        src_q[sq] = q
            for i, fl in enumerate(vfl):
                if not fl & O.FIRST:
                    continue
                r = vrows[vj[i]]
                if fl & O.INFL:
                    firsts.setdefault((vl[i], r.key), []).append((vh[i], vblk[i], "in_flight", None))
                elif fl & (O.GRANT | O.EVICT):
                    firsts.setdefault((vl[i], r.key), []).append((vh[i], vblk[i], "new", trs_of.get(src_q.get(i))))
        for st in steps:
            l = st["l"]
            if st["vdec"] is None:
                continue
            keys = []
            for r in vrows:
                if r.key not in keys:
                    keys.append(r.key)
            ready: Dict[Any, List[int]] = {}
            for jj, r in enumerate(vrows):
                acc = ready.setdefault(r.key, [0, 0, 0, 0])
                for qq in range(4):
                    acc[qq] += vlook[l][jj][qq]
            for key in keys:
                call = self.metrics.open_call(key, l, now)
                nr, nl, npo, nv = ready[key]
                call["ready"] += nr
                self.metrics.k46["lru"] += nl
                self.metrics.k46["po"] += npo
                self.metrics.k46["vres"] += nv
                self.metrics.k46["not_resident"] += nr - nl - npo - nv
                for cls in ("in_flight", "new"):
                    for (h, t, c, tr) in firsts.get((l, key), []):
                        if c != cls:
                            continue
                        call[cls] += 1
                        if cls == "in_flight":
                            self.metrics.k46["inflight_waited"] += 1
                        else:
                            rec = self.metrics.lifetimes.get((l, tr.b, tr.h, tr.slot, tr.gen)) if tr else None
                            self.metrics.k46["fetched_refetch" if rec is not None and rec["refetch"]
                                             else "fetched_first"] += 1
            for key in keys:
                call = self.metrics.open_calls.pop((key, l), None)
                if call is not None:
                    self.metrics.k43["layer_calls"] += 1
                    for f in ("ready", "in_flight", "new", "wait_ns"):
                        self.metrics.k43[f] += call[f]
                    self.metrics.k43["layer_wait_ns"] += max(0, self.clock() - call["t"])
            if not self.mut.release_only_at_decision:
                for key in keys:
                    self._queue(key, l)
        for p in info:
            if p["st"]["ddec"] is not None:
                masked, refused, pairs = p["dc"]
                self.metrics.layer_plan(Plan(page_table=None, num_pages=None, last_len=None, seq_idx=None, blocks=[],
                                             masked=int(masked), draft_pairs=int(pairs), refused=int(refused)))
        self._changed()

    # ---------------------------------------------------------------------------------------------- record mutations
    def _rec_mutate(self, layers: List[Dict], rowsel: torch.Tensor, vread: torch.Tensor, rows, row_b: torch.Tensor,
                    row_plane: torch.Tensor) -> None:
        """NEGATIVE CONTROLS (tests only, b2_mut rec_mut): alter the first record of one kind of the tick (once per run),
        after the read and before the checks: the device's decision is wrong in that one way."""
        kind = self.b2_mut.get("rec_mut")
        if not kind or self.rec_mut_done is not None or kind in ("ignore_overflow", "reverse_stamps",
                                                                 "evict_po_unpinned", "gen_plus_two"):
            return                                                    # (the op-level kinds: _note_op_mut)
        G = self.own.geo
        code = self.own.c["code"].view(G.S, G.NPOS)
        for x in layers:
            l = x["l"]
            D, V = x.get("drec"), x.get("vrec")
            if kind == "take_nonfree" and D is not None:
                g = (torch.bitwise_and(D[2], O.GRANT) != 0).nonzero().view(-1)
                if g.numel():
                    i = int(g[0])
                    j, h = int(D[0, i]) // (G.H * rows.k), (int(D[0, i]) // rows.k) % G.H
                    s = (l * G.B + int(rows.db[j])) * G.H + h
                    busy = (code[s] != C_FREE).nonzero().view(-1)
                    if busy.numel():
                        D[4, i] = int(busy[0])
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "take_swap" and D is not None:
                g = (torch.bitwise_and(D[2], O.GRANT) != 0).nonzero().view(-1).tolist()
                for a in range(len(g)):
                    for c in range(a + 1, len(g)):
                        ia, ic = g[a], g[c]
                        if int(D[0, ia]) // rows.k == int(D[0, ic]) // rows.k and int(D[4, ia]) != int(D[4, ic]):
                            D[4, ia], D[4, ic] = D[4, ic].clone(), D[4, ia].clone()
                            self.rec_mut_done = (self.tick_no, kind)
                            return
            if kind == "k5_over" and D is not None:
                g = (torch.bitwise_and(D[2], O.REFUSED) != 0).nonzero().view(-1)
                if g.numel():
                    i = int(g[0])
                    j, h = int(D[0, i]) // (G.H * rows.k), (int(D[0, i]) // rows.k) % G.H
                    s = (l * G.B + int(rows.db[j])) * G.H + h
                    fr = (code[s] == C_FREE).nonzero().view(-1)
                    if fr.numel():
                        D[2, i] = O.GRANT
                        D[4, i] = int(fr[0])
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "pe_not_first" and V is not None:
                g = (torch.bitwise_and(V[2], O.EVICT) != 0).nonzero().view(-1)
                if g.numel():
                    i = int(g[0])
                    j, h = int(V[0, i]) // (G.H * rows.k), (int(V[0, i]) // rows.k) % G.H
                    s = (l * G.B + int(rows.vb[j])) * G.H + h
                    lru = (code[s] == C_LRU).nonzero().view(-1).tolist()
                    other = [x2 for x2 in lru if x2 != int(V[7, i])]
                    if other:
                        V[7, i] = other[-1]
                        V[8, i] = int(self.own.c["block"].view(G.S, G.NPOS)[s, other[-1]])
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            # the DROPS take the first record whose effect is not repeated by another one (else nothing changes):
            dem = self.own.flat("demand")
            if kind == "drop_refusal" and D is not None:
                g = (torch.bitwise_and(D[2], O.REFUSED) != 0).nonzero().view(-1).tolist()
                if g:
                    x["drec"] = _without(D, g[0])
                    self.rec_mut_done = (self.tick_no, kind)
                    return
            if kind == "drop_v_merge" and V is not None:              # a merge whose transfer lacks the verify bit
                for i in (torch.bitwise_and(V[2], O.INFL) != 0).nonzero().view(-1).tolist():
                    j, h = int(V[0, i]) // (G.H * rows.k), (int(V[0, i]) // rows.k) % G.H
                    s = (l * G.B + int(rows.vb[j])) * G.H + h
                    if int(dem[s * G.NPOS + int(V[4, i])]) & O.DEM_V == 0:
                        x["vrec"] = _without(V, i)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_d_merge" and D is not None:              # a merge whose transfer lacks the draft bit
                so = O.slot_of(self.own)
                for i in (torch.bitwise_and(D[2], O.INFL) != 0).nonzero().view(-1).tolist():
                    j, h = int(D[0, i]) // (G.H * rows.k), (int(D[0, i]) // rows.k) % G.H
                    s = (l * G.B + int(rows.db[j])) * G.H + h
                    sl = int(so[s, int(D[1, i])])
                    if sl >= 0 and int(dem[s * G.NPOS + sl]) & O.DEM_D == 0:
                        x["drec"] = _without(D, i)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_ready" and D is not None:                # a draft read that no verify row repeats
                vbits = O.unpack_words(vread[l], G.NPOS)
                for i in (torch.bitwise_and(D[2], O.READY) != 0).nonzero().view(-1).tolist():
                    j, h = int(D[0, i]) // (G.H * rows.k), (int(D[0, i]) // rows.k) % G.H
                    if not bool(vbits[int(rows.db[j]), h, int(D[3, i])]):
                        x["drec"] = _without(D, i)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_rowsel":                                 # a NEW pin bit that no other row sets
                pw = self.own.pin_w
                R = rowsel.shape[1]
                for r in range(R):
                    p, b = int(row_plane[r]), int(row_b[r])
                    others = torch.zeros_like(rowsel[l, r])
                    for r2 in range(R):
                        if r2 != r and int(row_plane[r2]) == p and int(row_b[r2]) == b:
                            others |= rowsel[l, r2]
                    new = rowsel[l, r] & ~pw[p, l, b] & ~others
                    nz = new.nonzero()
                    if nz.numel():
                        h, w = nz[0].tolist()
                        word = int(new[h, w])
                        rowsel[l, r, h, w] = int(rowsel[l, r, h, w]) & ~(word & -word)
                        self.rec_mut_done = (self.tick_no, kind)
                        return
            if kind == "drop_vread" and bool(vread[l].any()):         # a verify read that no draft row repeats
                dset = set()
                if D is not None:
                    for i in (torch.bitwise_and(D[2], O.READY) != 0).nonzero().view(-1).tolist():
                        j, h = int(D[0, i]) // (G.H * rows.k), (int(D[0, i]) // rows.k) % G.H
                        dset.add((int(rows.db[j]), h, int(D[3, i])))
                for (b, h, sl) in O.unpack_words(vread[l], G.NPOS).nonzero().tolist():
                    if (b, h, sl) not in dset:
                        w, bit = divmod(sl, 32)
                        vread[l, b, h, w] = int(vread[l, b, h, w]) & ~(1 << bit)
                        self.rec_mut_done = (self.tick_no, kind)
                        return


def _without(R: torch.Tensor, i: int) -> torch.Tensor:
    keep = torch.ones(R.shape[1], dtype=torch.bool, device=R.device)
    keep[i] = False
    return R[:, keep]


# the trash writer (F15's control, the B2 form): the draft bias fill also writes the trash slot
class TrashWriterB2(OwnResidency):
    def _layer_record(self, l, step, tk):
        super()._layer_record(l, step, tk)
        if self.trash_slot is not None and self.store_b is not None:
            s0 = self.trash_slot * self.bs
            self.store_b[l][:, s0:s0 + self.bs] = 0.0
