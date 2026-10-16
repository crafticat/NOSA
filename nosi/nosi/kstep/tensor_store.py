"""Track B: tensor-native storage and the producer bridge (retroinfer-eval
docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md 12.10.1 F1 / F2, 12.3 A4 / A5, 12.4, 12.9 and the
B2 ring of section 11; the frozen B / C interface I1-I3 of 2026-10-16). SOURCE AND PORTABLE COMPONENT READINESS ONLY.

What this module adds. The dispatcher's stores are CPU int64 tensors (F1). Admission and history rebuild their payloads
from Python values with torch.tensor over lists (F2). This module keeps the stores as tensors with an explicit dtype and
device. It moves caller tensors (the model's K, V and bias) into the stores with the existing backend copies (memcpy on
the named streams). It never converts a payload to a Python list or back.

  check_tensor        refuses a payload with a wrong type, dtype, shape or device (KStepError with a precise message).
  join_events         I1: one native completion handle for several handles. A stream wait per live input, then one
                      event on the join stream. One live input is returned unchanged. No live input gives None.
  SourceHolds         I2: a source backing that a queued copy reads stays referenced and charged until the completion
                      handle of every copy that reads it passes ev_done. One record per distinct storage, charged at
                      untyped_storage().nbytes() in an ownership.BiasOpLedger with the kinds "admission", "ring" and
                      "publish". The ledger key is (owner, kind, hold serial); the owner is the I3 identity
                      (uid, inc[, cohort key]). Only poll() releases a record (passive ev_done; no wait).
  TensorStore         the host, arena, ring and bias stores (12.3 A4) and the producer copies:
                        admit, admission_d2h, admission_ring, close_admission
                                           11.2 admission. A_r IS the caller's prefill tensors (adopted, not copied).
                        produce_ring       11.2 producer write of one ring entry (DRAFT, PRE or any producer).
                        publish_ring_block 11.3 D2H of one block from its EXACT ring entries, with the rgen copies.
                        publish_block      A5 D2H of one block from caller tensors (replaces the eager list path).
                        check_host_block, check_ring_entry, check_ring_identity
                                           readback detectors (instrumentation; the caller runs them after a wait).

Ownership. The existing pools decide every binding: ownership.Arena (positions, pgen) and ownership.RingPool (entries,
rgen, the ready / compute / D2H gates, FREE only in RingPool.retire). The store writes bytes into an entry that the
caller already bound, records the event of that copy and returns it. The caller installs it (RingPool.set_ready). The
one gate that the store installs itself is the D2H gate of the ring entries that publish_ring_block reads
(RingPool.note_d2h, the same call that dispatcher._d2h_block makes). The store adds no scheduler, no wait policy and no
backpressure.

Streams. Producer copies run on "compute". Publications run on "d2h" (the dispatcher's streams). Each stream is ordered,
so the latest event of a stream covers every earlier copy on it. When copies on two streams read one source, its
release handle is their join on the configured join stream (I1). On the deferred cpupack_core CpuBackend, a join event
completes only when its join stream runs (a host wait on the join, or synchronize). Until then the hold stays held. That
is the conservative side.

Not claimed: NOSA or model numerics, CUDA stream safety, HBM fit, throughput, or any use by the dispatcher. The pgen_t /
rgen_t mirrors stay the pools' CPU int64 tensors.

INTEGRATION MAP (for the lead; dispatcher.py at NOSI d996868; NOTHING here is wired, and there is no HOOK-B edit):
  F1 stores       :463-465 host K / V, :471-475 arena, :486-489 ring, :496-501 bias, all int64 POISON
                  -> TensorStore(...) with the executor's dtypes, then attach_executor; the transports get
                     store.host_* / store.arena_* (transfer.S0Transport also needs dtype = kv_dtype).
  F2 admission    :1655-1675 A_r from torch.tensor over the prefill lists -> admit(owner, b, k, v, c) with the
                     executor's prefill TENSORS; :1728-1735 (_d2h_block, adm branch) -> admission_d2h;
                     :1681-1687 partial binds: keep _ring_take / RingPool.bind, then admission_ring and set_ready;
                     :1688-1691 A_r gates -> close_admission; :1599-1604 (_ring_retire, A_r) -> poll().
  ring producers  :1459-1475 (_produce payload; torch.tensor of a Python value) -> produce_ring; the bind, the map
                     and set_ready stay in _produce.
  publication     :1742-1754 (_d2h_block, ring branch) -> publish_ring_block, and check_ring_identity in _d2h_check;
                     :1736-1741, :1755-1763 (eager_matched) and :2877-2883 (eager _history) -> publish_block with
                     tensor sources. The checked-mode expectations ek / evv / ec (:1708-1712) still rebuild from r.pub:
                     instrumentation, but a live adapter needs a tensor reference there too.
  open items      a named join stream in the backend (tests/kstep_toy.build makes copy / scatter / compute / d2h
                  only); Request.inc (I3) and holds.drained(uid, inc) before _end_request clears identity records; a
                  K6 category for holds.held (ownership.MemoryAccountant has none); device-resident pgen_t / rgen_t.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .backend import check_backend, ev_done
from .errors import ConfigError, KStepError, OwnershipViolation, ReadCorruption, StaleReader
from .ownership import LIVE, BiasOpLedger

HOLD_KINDS = ("admission", "ring", "publish")      # I2: the hold kinds (the second part of a hold's owner key)
INT_POISON = -7777777                              # the dispatcher's POISON for integer stores
PRODUCE = "compute"                                # the stream of every producer copy (ring writes; A_r's ready event)
PUBLISH = "d2h"                                    # the stream of every historical publication (D2H)


# ------------------------------------------------------------------------------------------------------------- checks
def fmt_shape(shape: Sequence[Any]) -> str:
    """A shape as text: (2,), (), (10, 2, 2); None (any length) prints as n."""
    items = ["n" if x is None else str(int(x)) for x in shape]
    if len(items) == 1:
        return "(%s,)" % items[0]
    return "(%s)" % ", ".join(items)


def _device_matches(got: torch.device, want: torch.device) -> bool:
    return got.type == want.type and (want.index is None or got.index == want.index)


def check_tensor(what: str, t: Any, dtype: torch.dtype, shapes: Sequence[Tuple], device: Any, where: str) -> Any:
    """Refuse a payload that is not a torch tensor of exactly `dtype`, of one of `shapes` (None = any length) and on
    `device`. The checks run in this order: type, dtype, shape, device. A copy converts a wrong dtype silently, so a
    wrong dtype is never accepted. KStepError "tensor store: <what> ... (<where>)". Returns t."""
    if not torch.is_tensor(t):
        raise KStepError("tensor store: %s is not a tensor (%s) (%s)" % (what, type(t).__name__, where))
    if t.dtype != dtype:
        raise KStepError("tensor store: %s dtype %s, expected %s (%s)" % (what, t.dtype, dtype, where))
    got = tuple(int(x) for x in t.shape)
    if not any(len(s) == len(got) and all(w is None or int(w) == g for w, g in zip(s, got)) for s in shapes):
        raise KStepError("tensor store: %s shape %s, expected %s (%s)"
                         % (what, fmt_shape(got), " or ".join(fmt_shape(s) for s in shapes), where))
    want = torch.device(device)
    if not _device_matches(t.device, want):
        raise KStepError("tensor store: %s device %s, expected %s (%s)" % (what, t.device, want, where))
    return t


# --------------------------------------------------------------------------------------------------------- I1: join
def join_events(be: Any, stream: Optional[str], evs: Iterable[Any]) -> Any:
    """I1: one native completion handle for the handles `evs`. None and complete handles are skipped, and a handle that
    occurs twice counts once. No live handle gives None (complete). One live handle is returned unchanged. Two or more
    live handles get a NEW native event: a stream wait on the join stream for each, then one event recorded there. No
    host wait, no synchronize and no Python object that aggregates events. A join of two or more live handles without a
    join stream raises KStepError."""
    live: List[Any] = []
    for ev in evs:
        if ev is None or ev_done(ev) or any(ev is x for x in live):
            continue
        live.append(ev)
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    if stream is None:
        raise KStepError("tensor store: a join of %d live completion handles needs a join stream (I1); none is "
                         "configured" % len(live))
    with be.stream(stream):
        for ev in live:
            be.stream_wait(stream, ev)
        return be.event_rec(stream)


# ------------------------------------------------------------------------------------------------------ I2: holds
def storage_id(t: torch.Tensor) -> Tuple[str, Optional[int], int]:
    """The identity of the storage that backs t: (device type, device index, data pointer of the storage)."""
    return (t.device.type, t.device.index, t.untyped_storage().data_ptr())


def storage_alias(t: torch.Tensor) -> torch.Tensor:
    """A uint8 tensor over the WHOLE storage of t (offset 0, every byte). It keeps the storage alive, and its
    numel x element_size is untyped_storage().nbytes(): the actual backing, not numel x itemsize of a view (I2)."""
    st = t.untyped_storage()
    return torch.empty((0,), dtype=torch.uint8, device=t.device).set_(st, 0, (st.nbytes(),), (1,))


class SourceHolds:
    """I2 (frozen interface, 2026-10-16): the queued copies' source backings, owned and charged until every copy that
    reads them completed.

      open(owner, kind, tensors) -> hold serial
                     The store takes the sources and keeps a reference to each tensor. Each distinct storage gets ONE
                     record, charged at untyped_storage().nbytes() in the ledger (BiasOpLedger.hold with key (owner,
                     kind, serial)). A storage that a live record already holds is not charged again (a shared record).
      read(serial, stream, ev)
                     A copy on `stream` that reads the hold's sources completes at `ev`. Each stream is ordered, so
                     the record keeps the latest event per stream.
      close(serial)  No later copy reads the hold's sources. A record whose holds are all closed gets its release
                     handle: the join (I1) of the latest event per stream. Records with the same events share one
                     join. Returns the distinct release handles that this close set.
      hold(...)      open, read and close in one call (a source that one copy reads).
      poll()         PASSIVE (ev_done only; no wait, no stream advancement): releases each record whose holds are all
                     closed and whose release handle is complete. The ledger uncharges it (release_key), and the
                     store drops every reference to it.

    held / peak / log are the ledger's (bytes of the distinct storages held now, the exact allocation-transition peak,
    every transition). recount() is an independent recount from the records' own references. Not a Python-allocator
    or GPU measurement."""

    def __init__(self, be: Any, join_stream: Optional[str] = None, ledger: Optional[BiasOpLedger] = None):
        self.be, self.join_stream = be, join_stream
        self.ledger = ledger if ledger is not None else BiasOpLedger(HOLD_KINDS)
        missing = [k for k in HOLD_KINDS if k not in self.ledger.kinds]
        if missing:
            raise ConfigError("tensor store: the source ledger lacks the hold kinds %s" % (missing,))
        self.recs: Dict[Tuple, Dict[str, Any]] = {}
        self.holds: Dict[int, Dict[str, Any]] = {}
        self.serial = 0
        self.shared = 0                  # opens that found their storage already held (not charged again)
        self.joins = 0                   # release handles that are new join events (two or more live inputs)
        self.released = 0                # records released

    @property
    def held(self) -> int:
        return self.ledger.held

    @property
    def peak(self) -> int:
        return self.ledger.peak

    @property
    def log(self) -> List[Tuple]:
        return self.ledger.log

    def open(self, owner: Tuple, kind: str, tensors: Iterable[Optional[torch.Tensor]]) -> int:
        if kind not in HOLD_KINDS:
            raise KStepError("tensor store: hold kind %r is not one of %s" % (kind, HOLD_KINDS))
        self.serial += 1
        hid = self.serial
        sids: List[Tuple] = []
        for t in tensors:
            if t is None:
                continue
            if t.untyped_storage().nbytes() == 0:
                continue                                 # an empty storage holds no backing
            nb = t.untyped_storage().nbytes()
            sid = storage_id(t)
            rec = self.recs.get(sid)
            if sid in sids:
                rec["refs"].append(t)                    # this hold already covers the storage; keep the tensor too
                continue
            if rec is None:
                alias = storage_alias(t)
                key = (owner, kind, hid)
                self.ledger.hold(kind, alias, key=key)
                rec = dict(kind=kind, key=key, alias=alias, nbytes=nb, refs=[], holds=[], open=0, last={}, gate=None)
                self.recs[sid] = rec
            else:
                self.shared += 1
            rec["refs"].append(t)
            rec["holds"].append(hid)
            rec["open"] += 1
            rec["gate"] = None
            sids.append(sid)
        self.holds[hid] = dict(owner=owner, kind=kind, sids=sids, open=True)
        return hid

    def _open_hold(self, hid: int, what: str) -> Dict[str, Any]:
        h = self.holds.get(hid)
        if h is None or not h["open"]:
            raise OwnershipViolation("tensor store: %s of source hold %s, which is not open" % (what, hid))
        return h

    def read(self, hid: int, stream: str, ev: Any) -> None:
        h = self._open_hold(hid, "read")
        for sid in h["sids"]:
            self.recs[sid]["last"][stream] = ev

    def close(self, hid: int) -> List[Any]:
        h = self._open_hold(hid, "close")
        h["open"] = False
        made: Dict[Tuple, Any] = {}
        gates: List[Any] = []
        for sid in h["sids"]:
            rec = self.recs[sid]
            rec["open"] -= 1
            if rec["open"]:
                continue                                 # another open hold still adds reads to this storage
            evs = [rec["last"][s] for s in sorted(rec["last"])]
            key = tuple(id(ev) for ev in evs)
            if key not in made:
                g = join_events(self.be, self.join_stream, evs)
                if g is not None and not any(g is ev for ev in evs):
                    self.joins += 1
                made[key] = g
                if not any(g is x for x in gates):
                    gates.append(g)
            rec["gate"] = made[key]
        if not h["sids"]:
            del self.holds[hid]
        return gates

    def hold(self, owner: Tuple, kind: str, tensors: Iterable[Optional[torch.Tensor]], stream: str, ev: Any) -> int:
        hid = self.open(owner, kind, tensors)
        self.read(hid, stream, ev)
        self.close(hid)
        return hid

    def poll(self) -> int:
        n = 0
        for sid in list(self.recs):
            rec = self.recs[sid]
            if rec["open"] or not ev_done(rec["gate"]):
                continue
            self.ledger.release_key(rec["kind"], rec["key"])
            del self.recs[sid]
            for hid in rec["holds"]:
                h = self.holds.get(hid)
                if h is None:
                    continue
                h["sids"].remove(sid)
                if not h["sids"] and not h["open"]:
                    del self.holds[hid]
            n += 1
        self.released += n
        return n

    def holds_storage(self, t: torch.Tensor) -> bool:
        """Is the storage of t held now (charged, referenced, not yet released)?"""
        return storage_id(t) in self.recs

    def drained(self, uid: Any, inc: Any) -> bool:
        """I3: no hold of the request incarnation (uid, inc) remains (open, or with a storage not yet released)."""
        return not any(tuple(h["owner"][:2]) == (uid, inc) for h in self.holds.values())

    def recount(self) -> int:
        """The bytes of the unique storages that the live records reference now (independent of the ledger)."""
        seen: Dict[Tuple, int] = {}
        for rec in self.recs.values():
            for t in rec["refs"]:
                seen.setdefault(storage_id(t), t.untyped_storage().nbytes())
        return sum(seen.values())

    def outstanding(self) -> List[Tuple]:
        """Per live hold, in serial order: (owner, kind, serial, open, bytes of its storages not yet released). A shared
        storage is listed under every hold that reads it; `held` counts it once."""
        return [(h["owner"], h["kind"], hid, h["open"], sum(self.recs[s]["nbytes"] for s in h["sids"]))
                for hid, h in sorted(self.holds.items())]


# ----------------------------------------------------------------------------------------------------------- stores
class Admission:
    """One request's admission buffer A_r (11.2): the adopted prefill tensors per layer (k, v: (n, H, D); c: (n, H)),
    the producer event `ready` (on "compute"), the latest D2H read `dev` and compute read `cev`, the source hold and,
    after close_admission, its release handle `gate`. After the close the record drops its tensor references (the hold
    keeps them until the gate passes) and refuses every further copy."""
    __slots__ = ("owner", "b", "n", "k", "v", "c", "ready", "cev", "dev", "hold", "gate", "closed", "d2h_blocks",
                 "ring_copies")

    def __init__(self, owner: Tuple, b: int, n: int, k: List[torch.Tensor], v: List[torch.Tensor],
                 c: Optional[List[torch.Tensor]], ready: Any, hold: int):
        self.owner, self.b, self.n, self.k, self.v, self.c = owner, b, n, k, v, c
        self.ready, self.cev, self.dev, self.hold, self.gate, self.closed = ready, None, None, hold, None, False
        self.d2h_blocks = []
        self.ring_copies = 0


class TensorStore:
    """The tensor-native stores of 12.3 A4 and their producer copies (module docstring). Per layer l:
      host_k[l], host_v[l]  (B, S_host, H, D) kv_dtype on host_device (pinned when pin_host)
      host_c[l]             (B, S_host, H) bias_dtype                       (only with a bias dtype)
      arena_k[l], arena_v[l] (P_l * bs, D) kv_dtype on device, P_l = arenas[l].P; row p * bs + r
      arena_c[l]            (P_l * bs,) bias_dtype
      ring_k[l], ring_v[l]  (Q_l, D) kv_dtype on device, Q_l = rings[l].Q; row e = entry e   (only with rings)
      ring_c[l]             (Q_l,) bias_dtype
    Every store row starts as `poison` (default: INT_POISON for int32 / int64, the most negative finite value for a
    floating dtype). The generation mirrors stay arenas[l].pgen_t and rings[l].rgen_t."""

    def __init__(self, be: Any, *, L: int, H: int, D: int, bs: int, B: int, S_host: int, arenas: Sequence[Any],
                 rings: Optional[Sequence[Any]] = None, kv_dtype: torch.dtype, bias_dtype: Optional[torch.dtype] = None,
                 device: Any = "cpu", host_device: Any = "cpu", pin_host: bool = False,
                 join_stream: Optional[str] = None, ledger: Optional[BiasOpLedger] = None,
                 poison: Optional[float] = None):
        check_backend(be)
        for name, x in (("L", L), ("H", H), ("D", D), ("bs", bs), ("B", B), ("S_host", S_host)):
            if int(x) < 1:
                raise ConfigError("tensor store: %s = %r < 1" % (name, x))
        if len(arenas) != L or (rings is not None and len(rings) != L):
            raise ConfigError("tensor store: one arena and one ring per layer: %d arenas, %s rings for L %d"
                              % (len(arenas), None if rings is None else len(rings), L))
        self.be = be
        self.L, self.H, self.D, self.bs, self.B, self.S_host = int(L), int(H), int(D), int(bs), int(B), int(S_host)
        self.arenas = list(arenas)
        self.rings = list(rings) if rings is not None else None
        self.kv_dtype, self.bias_dtype = kv_dtype, bias_dtype
        self.join_stream = join_stream
        self.holds = SourceHolds(be, join_stream, ledger)
        kpo = self._poison(kv_dtype, poison)
        cpo = self._poison(bias_dtype, poison) if bias_dtype is not None else None

        def full(shape, dt, po, dev, pin=False):
            t = torch.full(shape, po, dtype=dt, device=dev)
            return t.pin_memory() if pin else t
        self.host_k = [full((B, S_host, H, D), kv_dtype, kpo, host_device, pin_host) for _ in range(L)]
        self.host_v = [full((B, S_host, H, D), kv_dtype, kpo, host_device, pin_host) for _ in range(L)]
        self.host_c = ([full((B, S_host, H), bias_dtype, cpo, host_device, pin_host) for _ in range(L)]
                       if bias_dtype is not None else None)
        self.arena_k = [full((a.P * bs, D), kv_dtype, kpo, device) for a in self.arenas]
        self.arena_v = [full((a.P * bs, D), kv_dtype, kpo, device) for a in self.arenas]
        self.arena_c = ([full((a.P * bs,), bias_dtype, cpo, device) for a in self.arenas]
                        if bias_dtype is not None else None)
        if self.rings is not None:
            self.ring_k = [full((rp.Q, D), kv_dtype, kpo, device) for rp in self.rings]
            self.ring_v = [full((rp.Q, D), kv_dtype, kpo, device) for rp in self.rings]
            self.ring_c = ([full((rp.Q,), bias_dtype, cpo, device) for rp in self.rings]
                           if bias_dtype is not None else None)
        else:
            self.ring_k = self.ring_v = self.ring_c = None
        self.dev = self.arena_k[0].device
        self.hdev = self.host_k[0].device
        self._store_ids = {storage_id(t) for t in self._stores()}
        self.stats = dict(admissions=0, admission_d2h=0, admission_ring=0, admissions_closed=0, ring_produced=0,
                          ring_published=0, published=0)

    # -- helpers
    @staticmethod
    def _poison(dtype: torch.dtype, poison: Optional[float]):
        if poison is not None:
            return poison
        if dtype.is_floating_point:
            return float(torch.finfo(dtype).min)
        if dtype in (torch.int32, torch.int64):
            return INT_POISON
        raise ConfigError("tensor store: no default poison for dtype %s (pass poison)" % (dtype,))

    def _groups(self) -> List[Tuple[str, List[torch.Tensor]]]:
        out = [("host", self.host_k + self.host_v + (self.host_c or [])),
               ("arena", self.arena_k + self.arena_v + (self.arena_c or []))]
        if self.rings is not None:
            out.append(("ring", self.ring_k + self.ring_v + (self.ring_c or [])))
        return out

    def _stores(self) -> List[torch.Tensor]:
        out = [t for _, ts in self._groups() for t in ts]
        out += [a.pgen_t for a in self.arenas] + ([rp.rgen_t for rp in self.rings] if self.rings is not None else [])
        return out

    @staticmethod
    def _owner(owner: Any) -> Tuple:
        if not isinstance(owner, tuple) or len(owner) < 2:
            raise KStepError("tensor store: owner %r is not an I3 identity (uid, inc[, key])" % (owner,))
        return owner

    def _source(self, what: str, t: Any, dtype: torch.dtype, shapes: Sequence[Tuple], where: str) -> torch.Tensor:
        check_tensor(what, t, dtype, shapes, self.dev, where)
        if storage_id(t) in self._store_ids:
            raise KStepError("tensor store: %s aliases a persistent store (%s)" % (what, where))
        return t

    def _per_layer(self, what: str, x: Any, where: str) -> List[Any]:
        if not isinstance(x, (list, tuple)):
            raise KStepError("tensor store: %s is not a list of %d per-layer tensors (%s) (%s)"
                             % (what, self.L, type(x).__name__, where))
        if len(x) != self.L:
            raise KStepError("tensor store: %s has %d per-layer tensors, expected %d (%s)"
                             % (what, len(x), self.L, where))
        return list(x)

    def _layer(self, l: int, where: str) -> None:
        if not 0 <= l < self.L:
            raise KStepError("tensor store: layer %d outside [0, %d) (%s)" % (l, self.L, where))

    def _entry(self, l: int, e: int, where: str) -> None:
        self._layer(l, where)
        if self.rings is None:
            raise KStepError("tensor store: no ring stores (%s)" % where)
        if not 0 <= e < self.rings[l].Q:
            raise KStepError("tensor store: entry %d outside [0, %d) (%s)" % (e, self.rings[l].Q, where))

    def _live_entry(self, l: int, e: int, where: str) -> None:
        st = self.rings[l].state[e]
        if st != LIVE:
            raise OwnershipViolation("tensor store: ring write into entry %d (layer %d), which is %s (%s)"
                                     % (e, l, st, where))

    def _host_block(self, b: int, j: int, where: str) -> None:
        if not 0 <= b < self.B:
            raise KStepError("tensor store: batch slot %d outside [0, %d) (%s)" % (b, self.B, where))
        if j < 0 or (j + 1) * self.bs > self.S_host:
            raise KStepError("tensor store: block %d outside the %d host rows (%s)" % (j, self.S_host, where))

    def _open_admission(self, adm: Any, where: str) -> None:
        if not isinstance(adm, Admission) or adm.closed:
            raise KStepError("tensor store: the admission is closed or unknown (%s)" % where)

    # -- wiring and reports
    def attach_executor(self, ex: Any, register: Any, register_ring: Any = None) -> None:
        """The executor's attach calls, as Dispatcher.__init__ makes them (dispatcher.py:475, :489, :501)."""
        ex.attach_arena(self.be, self.arena_k, self.arena_v, [a.pgen_t for a in self.arenas], register)
        if self.rings is not None:
            ex.attach_ring(self.be, self.ring_k, self.ring_v, [rp.rgen_t for rp in self.rings], register_ring)
        if self.bias_dtype is not None:
            ex.attach_bias(self.be, self.arena_c, self.ring_c)

    def store_bytes(self) -> Dict[str, int]:
        """The ACTUAL persistent store bytes at allocated capacity (untyped_storage().nbytes() of each store tensor;
        the generation mirrors belong to the pools). Instrumentation, not a fit claim."""
        return {name: sum(t.untyped_storage().nbytes() for t in ts) for name, ts in self._groups()}

    def report(self) -> Dict[str, Any]:
        h = self.holds
        return dict(stats=dict(self.stats), held=h.held, peak=h.peak, records=len(h.recs), holds=len(h.holds),
                    shared=h.shared, joins=h.joins, released=h.released, stores=self.store_bytes())

    def poll(self) -> int:
        """The passive release of the source holds (call it at the existing host poll points)."""
        return self.holds.poll()

    # -- admission (11.2)
    def admit(self, owner: Tuple, b: int, k: Sequence[torch.Tensor], v: Sequence[torch.Tensor],
              c: Optional[Sequence[torch.Tensor]] = None, ready: Any = None) -> Admission:
        """Step 0: the prefill output becomes A_r without a copy. k, v: one (n, H, D) kv_dtype tensor per layer; c: one
        (n, H) bias_dtype tensor per layer (only with a bias dtype); all on the store's device. Views are accepted (use
        t.unbind(0) for an (L, n, H, D) tensor); the hold charges each distinct storage once, in full. `ready` is an
        optional producer event of another stream; the store waits it on "compute". A_r's ready event is recorded on
        "compute" (the model of the prefill kernel's write, D1). Every check runs before any enqueue or hold."""
        own = self._owner(owner)
        where = "admission of owner %s b %d" % (own, b)
        if not 0 <= b < self.B:
            raise KStepError("tensor store: batch slot %d outside [0, %d) (%s)" % (b, self.B, where))
        ks = self._per_layer("admission k", k, where)
        vs = self._per_layer("admission v", v, where)
        cs = None
        if self.bias_dtype is not None:
            cs = self._per_layer("admission c", c, where)
        elif c is not None:
            raise KStepError("tensor store: admission c given, but the store has no bias (%s)" % where)
        self._source("admission k[0]", ks[0], self.kv_dtype, ((None, self.H, self.D),), where)
        n = int(ks[0].shape[0])
        if n > self.S_host:
            raise KStepError("tensor store: %d admitted positions > S_host %d (%s)" % (n, self.S_host, where))
        for l in range(self.L):
            self._source("admission k[%d]" % l, ks[l], self.kv_dtype, ((n, self.H, self.D),), where)
            self._source("admission v[%d]" % l, vs[l], self.kv_dtype, ((n, self.H, self.D),), where)
            if cs is not None:
                self._source("admission c[%d]" % l, cs[l], self.bias_dtype, ((n, self.H),), where)
        hid = self.holds.open(own, "admission", ks + vs + (cs if cs is not None else []))
        with self.be.stream(PRODUCE):
            if ready is not None:
                self.be.stream_wait(PRODUCE, ready)
            rev = self.be.event_rec(PRODUCE)
        self.holds.read(hid, PRODUCE, rev)         # A_r stays held at least until its producer completed
        self.stats["admissions"] += 1
        return Admission(own, b, n, ks, vs, cs, rev, hid)

    def admission_d2h(self, adm: Admission, l: int, j: int) -> Any:
        """Step 1 for one COMPLETE prompt block j of layer l: on "d2h", a stream wait on A_r's ready event, then the K,
        V (and bias) rows of A_r into the host rows (b, j * bs ..). Returns the D2H event (hist_ev). The caller keeps
        the snapshot, hown and hist_ev bookkeeping (dispatcher._d2h_block)."""
        where = "admission D2H of owner %s layer %d block %d" % (getattr(adm, "owner", None), l, j)
        self._open_admission(adm, where)
        self._layer(l, where)
        self._host_block(adm.b, j, where)
        bs, b = self.bs, adm.b
        r0, r1 = j * bs, (j + 1) * bs
        if r1 > adm.n:
            raise KStepError("tensor store: block %d is not a complete block of the %d admitted positions (%s)"
                             % (j, adm.n, where))
        with self.be.stream(PUBLISH):
            self.be.stream_wait(PUBLISH, adm.ready)
            self.be.memcpy(self.host_k[l][b, r0:r1], adm.k[l][r0:r1], PUBLISH)
            self.be.memcpy(self.host_v[l][b, r0:r1], adm.v[l][r0:r1], PUBLISH)
            if self.host_c is not None:
                self.be.memcpy(self.host_c[l][b, r0:r1], adm.c[l][r0:r1], PUBLISH)
            ev = self.be.event_rec(PUBLISH)
        adm.dev = ev
        adm.d2h_blocks.append((l, j))
        self.holds.read(adm.hold, PUBLISH, ev)
        self.stats["admission_d2h"] += 1
        return ev

    def admission_ring(self, adm: Admission, l: int, h: int, pos: int, e: int, prev_d2h: Any = None) -> Any:
        """Step 2 for one position of the final partial block: the caller bound entry e first (RingPool.bind). On
        "compute": a stream wait on the previous occupancy's D2H event (the second safeguard; prev_d2h from the bind),
        then the K, V (and bias) of A_r at (pos, h) into ring row e. Returns the copy's event (the entry's ready event:
        the caller installs it with RingPool.set_ready)."""
        where = ("admission ring copy of owner %s layer %d head %d position %d entry %d"
                 % (getattr(adm, "owner", None), l, h, pos, e))
        self._open_admission(adm, where)
        self._entry(l, e, where)
        if not 0 <= h < self.H or not 0 <= pos < adm.n:
            raise KStepError("tensor store: head %d / position %d outside H %d / the %d admitted positions (%s)"
                             % (h, pos, self.H, adm.n, where))
        self._live_entry(l, e, where)
        with self.be.stream(PRODUCE):
            if prev_d2h is not None:
                self.be.stream_wait(PRODUCE, prev_d2h)
            self.be.memcpy(self.ring_k[l][e], adm.k[l][pos, h], PRODUCE)
            self.be.memcpy(self.ring_v[l][e], adm.v[l][pos, h], PRODUCE)
            if self.ring_c is not None:
                self.be.memcpy(self.ring_c[l][e:e + 1], adm.c[l][pos, h:h + 1], PRODUCE)
            ev = self.be.event_rec(PRODUCE)
        adm.cev = ev
        adm.ring_copies += 1
        self.holds.read(adm.hold, PRODUCE, ev)
        self.stats["admission_ring"] += 1
        return ev

    def close_admission(self, adm: Admission) -> Any:
        """Step 3: no later copy reads A_r. The hold gets its release handle: the join (I1) of the latest compute read
        (or the ready event) and the latest D2H read; one live handle is returned unchanged. The record drops its
        tensor references. Returns the handle (also adm.gate)."""
        where = "admission close of owner %s" % (getattr(adm, "owner", None),)
        self._open_admission(adm, where)
        gates = self.holds.close(adm.hold)
        adm.gate = join_events(self.be, self.join_stream, gates)
        adm.closed = True
        adm.k = adm.v = adm.c = None
        self.stats["admissions_closed"] += 1
        return adm.gate

    # -- ring production (11.2)
    def produce_ring(self, owner: Tuple, l: int, e: int, k: torch.Tensor, v: torch.Tensor, c: Any = None,
                     prev_d2h: Any = None, ready: Any = None) -> Any:
        """The payload write of one ring entry that the caller bound first (RingPool.bind): k, v (D,) kv_dtype and c
        () or (1,) bias_dtype (only with a bias dtype), on the store's device. On "compute": a stream wait on prev_d2h
        (the bind's previous D2H event) and on `ready` (an optional producer event of another stream), then the copies.
        The sources are held (kind "ring") until the copy's event. Returns that event (the caller's
        RingPool.set_ready)."""
        own = self._owner(owner)
        where = "ring producer of owner %s layer %d entry %d" % (own, l, e)
        self._entry(l, e, where)
        self._source("ring producer k", k, self.kv_dtype, ((self.D,),), where)
        self._source("ring producer v", v, self.kv_dtype, ((self.D,),), where)
        if self.ring_c is not None:
            self._source("ring producer c", c, self.bias_dtype, ((), (1,)), where)
        elif c is not None:
            raise KStepError("tensor store: ring producer c given, but the store has no bias (%s)" % where)
        self._live_entry(l, e, where)
        with self.be.stream(PRODUCE):
            if prev_d2h is not None:
                self.be.stream_wait(PRODUCE, prev_d2h)
            if ready is not None:
                self.be.stream_wait(PRODUCE, ready)
            self.be.memcpy(self.ring_k[l][e], k, PRODUCE)
            self.be.memcpy(self.ring_v[l][e], v, PRODUCE)
            if self.ring_c is not None:
                self.be.memcpy(self.ring_c[l][e:e + 1], c.reshape(1), PRODUCE)
            ev = self.be.event_rec(PRODUCE)
        self.holds.hold(own, "ring", (k, v, c), PRODUCE, ev)
        self.stats["ring_produced"] += 1
        return ev

    # -- historical publication (11.3, A5)
    def publish_ring_block(self, l: int, b: int, j: int, src: Sequence[Sequence[Tuple[int, int]]]) -> Tuple[Any, Any]:
        """The D2H of block j of layer l from its ring entries: src[i][h] = (entry, rgen) of position j * bs + i, head
        h; every entry LIVE with that rgen and a ready event, else OwnershipViolation. On "d2h": a stream wait on every
        entry's ready event, then per (position, head) the K, V (and bias) rows into the host rows and the entry's
        rgen_t into rg (the identity copy at execution); then hist_ev, installed as each entry's D2H gate
        (RingPool.note_d2h). Returns (hist_ev, rg); check_ring_identity reads rg after the event completed."""
        where = "ring D2H of layer %d b %d block %d" % (l, b, j)
        self._layer(l, where)
        self._host_block(b, j, where)
        if self.rings is None:
            raise KStepError("tensor store: no ring stores (%s)" % where)
        ring, bs, H = self.rings[l], self.bs, self.H
        if len(src) != bs or any(len(row) != H for row in src):
            raise KStepError("tensor store: the source of a block is %d positions x %d heads of (entry, rgen) (%s)"
                             % (bs, H, where))
        for i, row in enumerate(src):
            for h, (e, g) in enumerate(row):
                ok = 0 <= e < ring.Q
                if not ok or ring.state[e] != LIVE or ring.rgen[e] != g or ring.ready[e] is None:
                    raise OwnershipViolation(
                        "tensor store: %s: entry %s (head %d position %d) is %s with rgen %s and ready %s; the read "
                        "names rgen %d" % (where, e, h, j * bs + i, ring.state[e] if ok else "out of range",
                                           ring.rgen[e] if ok else None, (ring.ready[e] is not None) if ok else None,
                                           g))
        r0 = j * bs
        with self.be.stream(PUBLISH):
            for row in src:
                for e, g in row:
                    self.be.stream_wait(PUBLISH, ring.ready[e])
            rg = torch.full((bs, H), -1, dtype=torch.int64, device=ring.rgen_t.device)
            for i, row in enumerate(src):
                for h, (e, g) in enumerate(row):
                    self.be.memcpy(self.host_k[l][b, r0 + i, h], self.ring_k[l][e], PUBLISH)
                    self.be.memcpy(self.host_v[l][b, r0 + i, h], self.ring_v[l][e], PUBLISH)
                    if self.host_c is not None:
                        self.be.memcpy(self.host_c[l][b, r0 + i, h:h + 1], self.ring_c[l][e:e + 1], PUBLISH)
                    self.be.memcpy(rg[i, h:h + 1], ring.rgen_t[e:e + 1], PUBLISH)
            hist_ev = self.be.event_rec(PUBLISH)
        for row in src:
            for e, g in row:
                ring.note_d2h(e, g, hist_ev)
        self.stats["ring_published"] += 1
        return hist_ev, rg

    def publish_block(self, owner: Tuple, l: int, b: int, j: int, k: torch.Tensor, v: torch.Tensor, c: Any = None,
                      ready: Any = None) -> Any:
        """The D2H of block j of layer l from caller tensors: k, v (bs, H, D) kv_dtype and c (bs, H) bias_dtype (only
        with a bias dtype), on the store's device. On "d2h": a stream wait on `ready` (the producer of the sources),
        the copies, then hist_ev. The sources are held (kind "publish") until hist_ev. Returns hist_ev."""
        own = self._owner(owner)
        where = "publication of owner %s layer %d b %d block %d" % (own, l, b, j)
        self._layer(l, where)
        self._host_block(b, j, where)
        bs, H, D = self.bs, self.H, self.D
        self._source("publish k", k, self.kv_dtype, ((bs, H, D),), where)
        self._source("publish v", v, self.kv_dtype, ((bs, H, D),), where)
        if self.host_c is not None:
            self._source("publish c", c, self.bias_dtype, ((bs, H),), where)
        elif c is not None:
            raise KStepError("tensor store: publish c given, but the store has no bias (%s)" % where)
        r0, r1 = j * bs, (j + 1) * bs
        with self.be.stream(PUBLISH):
            if ready is not None:
                self.be.stream_wait(PUBLISH, ready)
            self.be.memcpy(self.host_k[l][b, r0:r1], k, PUBLISH)
            self.be.memcpy(self.host_v[l][b, r0:r1], v, PUBLISH)
            if self.host_c is not None:
                self.be.memcpy(self.host_c[l][b, r0:r1], c, PUBLISH)
            hist_ev = self.be.event_rec(PUBLISH)
        self.holds.hold(own, "publish", (k, v, c), PUBLISH, hist_ev)
        self.stats["published"] += 1
        return hist_ev

    # -- readback detectors (instrumentation; host reads of the stores, after the caller's completion wait)
    @staticmethod
    def _first(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
        return (got != want).nonzero()[0]

    def check_host_block(self, l: int, b: int, j: int, k: torch.Tensor, v: torch.Tensor, c: Any = None) -> int:
        """The host rows of block j equal k, v (bs, H, D) and c (bs, H), compared with torch.equal; else ReadCorruption
        "tensor store: D2H readback mismatch: layer L b B block J <K | V | bias> row R head H" (the first difference in
        row-major order). Returns the parts compared."""
        where = "host readback of layer %d b %d block %d" % (l, b, j)
        self._layer(l, where)
        self._host_block(b, j, where)
        bs, H, D = self.bs, self.H, self.D
        r0, r1 = j * bs, (j + 1) * bs
        parts = [("K", self.host_k[l][b, r0:r1], k, self.kv_dtype, (bs, H, D)),
                 ("V", self.host_v[l][b, r0:r1], v, self.kv_dtype, (bs, H, D))]
        if c is not None:
            if self.host_c is None:
                raise KStepError("tensor store: expected bias given, but the store has no bias (%s)" % where)
            parts.append(("bias", self.host_c[l][b, r0:r1], c, self.bias_dtype, (bs, H)))
        for name, got, want, dt, shape in parts:
            check_tensor("expected %s" % name, want, dt, (shape,), got.device, where)
            if not torch.equal(got, want):
                i = self._first(got, want)
                raise ReadCorruption("tensor store: D2H readback mismatch: layer %d b %d block %d %s row %d head %d"
                                     % (l, b, j, name, int(i[0]), int(i[1])))
        return len(parts)

    def check_ring_entry(self, l: int, e: int, k: torch.Tensor, v: torch.Tensor, c: Any = None) -> int:
        """Ring row e of layer l equals k, v (D,) and c (() or (1,)), compared with torch.equal; else ReadCorruption
        "tensor store: ring readback mismatch: layer L entry E <K | V | bias> element I". Returns the parts compared."""
        where = "ring readback of layer %d entry %d" % (l, e)
        self._entry(l, e, where)
        parts = [("K", self.ring_k[l][e], k, self.kv_dtype, ((self.D,),)),
                 ("V", self.ring_v[l][e], v, self.kv_dtype, ((self.D,),))]
        if c is not None:
            if self.ring_c is None:
                raise KStepError("tensor store: expected bias given, but the store has no bias (%s)" % where)
            parts.append(("bias", self.ring_c[l][e:e + 1], c, self.bias_dtype, ((), (1,))))
        for name, got, want, dt, shapes in parts:
            check_tensor("expected %s" % name, want, dt, shapes, got.device, where)
            want = want.reshape(got.shape)
            if not torch.equal(got, want):
                i = self._first(got, want)
                raise ReadCorruption("tensor store: ring readback mismatch: layer %d entry %d %s element %d"
                                     % (l, e, name, int(i[0])))
        return len(parts)

    def check_ring_identity(self, l: int, b: int, j: int, src: Sequence[Sequence[Tuple[int, int]]], rg: Any) -> int:
        """After the D2H of publish_ring_block completed: every copied rgen equals the rgen that the read named;
        else StaleReader (identity before bytes, as dispatcher._d2h_check). Returns the identities checked."""
        n = 0
        for i, row in enumerate(src):
            for h, (e, g) in enumerate(row):
                got = int(rg[i, h])
                if got != g:
                    raise StaleReader("tensor store: stale reader: D2H layer %d b %d block %d entry %d (head %d "
                                      "position %d) rgen captured %d, at read %d"
                                      % (l, b, j, e, h, j * self.bs + i, g, got))
                n += 1
        return n
