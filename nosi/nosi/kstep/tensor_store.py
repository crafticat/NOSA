"""Track B: tensor-native storage and the producer bridge (retroinfer-eval
docs/superpowers/specs/2026-10-14-kstep-m2-integration-contract.md 12.10.1 F1 / F2, 12.3 A4 / A5, 12.4, 12.9 and the
B2 ring of section 11; the frozen B / C interface I1-I3 of 2026-10-16). SOURCE AND PORTABLE COMPONENT READINESS ONLY.

What this module adds. The dispatcher's stores are CPU int64 tensors (F1). Admission and history rebuild their payloads
from Python values with torch.tensor over lists (F2). This module keeps the stores as tensors with an explicit dtype and
device. It moves caller tensors (the model's K, V and bias) into the stores with the existing backend copies (memcpy on
the named streams). It never converts a payload to a Python list or back.

  check_tensor        refuses a payload with a wrong type, dtype, shape or device (KStepError with a precise message).
  join_events         I1: one native completion handle for several handles. A stream wait per live input, then one
                      event on the join stream. One live input is returned unchanged. No live input gives None. Used
                      ONLY for a handle that the store returns to a consumer (close_admission's adm.gate).
  SourceHolds         I2: a source backing that a queued copy reads stays referenced and charged until the completion
                      handle of every copy that reads it passes ev_done. One record per distinct storage, charged at
                      untyped_storage().nbytes() in an ownership.BiasOpLedger with the kinds "admission", "ring" and
                      "publish". The ledger key is (owner, kind, hold serial, storage id); the owner is the I3
                      identity (uid, inc[, cohort key]). A record keeps the latest read event of each stream. Only
                      poll() releases it: when every one of those events passes ev_done (a host-side poll of each
                      native event; no join event, no wait).
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
so the latest event of a stream covers every earlier copy on it. A source that copies on two streams read is released
when the latest event of EACH stream passed ev_done (lead decision, 2026-10-16): no join event is formed for a release,
so no hold depends on a join stream. The one join is adm.gate (close_admission), a consumer handle: it needs the
configured join stream, and on the deferred cpupack_core CpuBackend it completes only when that stream runs (a host wait
on it, or synchronize).

Every hold is opened (charged) after the payload checks and BEFORE its copies are enqueued, and closed after them: no
copy is ever queued with an unheld source. SourceHolds.open is atomic: a refusal part-way charges nothing.

Precondition (admission). The prefill tensors that admit adopts as A_r are storage that the executor does not reuse
until adm.gate completes. If the executor writes its next prefill into a reused workspace, that write must wait on the
adm.gate of every admission that adopted the workspace (on the GPU: a stream wait on the executor's stream). The store
cannot see such a reuse; its hold keeps the storage referenced, not unwritten. adm.gate covers this admission's own
copies only: the strict condition for a reuse is holds_storage(t) is False for every adopted tensor t (lead
integration, after the Track B review).

Failures (lead integration, after the Track B review): a refused open leaves held exact, but the ledger's peak / log may
show the charge and its rollback. An enqueue that raises after its hold opened leaves that hold OPEN on purpose: a
partly queued copy may still read the source, so the hold is never released early (drained() stays False; a leak, not
a corruption).

Not claimed: NOSA or model numerics, CUDA stream safety, HBM fit or throughput. The pgen_t / rgen_t mirrors stay the
pools' CPU int64 tensors.

DISPATCHER ADAPTER (milestone 2; opt-in Config.store = "tensor"; every dispatcher / ownership edit is marked HOOK-B2;
the default "none" path is unchanged; sites named by function, as line numbers shift when other tracks merge):
  Config          store ("none" | "tensor") and store_join (the stream of adm.gate).
  __init__        attach_dispatcher (after the int64 stores and the attach calls, BEFORE make_transport): the
                  TensorStore over the dispatcher's Arena / RingPool with the executor's dtypes BECOMES host / arena /
                  ring K, V and bias (bias_dtype too); the executor is attached to it again; Dispatcher.store (None in
                  the default path). Then check_transport (right after make_transport) and the K6 category
                  source_staging (ownership.MemoryAccountant.enable_sources).
  admission       Dispatcher._admit_recent -> dispatch_admit: A_r = ex.prefill_tensors(kv) via admit; complete blocks by
                  _history -> _d2h_block -> dispatch_d2h_block -> admission_d2h; the partial block by the existing bind
                  (_produce) -> dispatch_produce -> admission_ring; the dispatcher's A record (no tensors) keeps ready /
                  cev / dev, is appended to d.adm and stays there until _ring_retire; then close_admission.
  producers       Dispatcher._produce, after the bind and the map insert -> dispatch_produce -> produce_ring with the
                  executor's tensors (ex.payload(kvv, at=...)) or admission_ring; set_ready of the returned event.
  publication     Dispatcher._d2h_block -> dispatch_d2h_block: publish_ring_block (decode blocks) or admission_d2h
                  (prompt blocks); hist_ev, snapshot, hown, note_d2h and counters as before. In the store's mode recent
                  is "ring", so _history's eager branch and _d2h_block's eager_matched branch are not reached;
                  publish_block stays the store's API for them (component-tested; not wired).
  validators      Dispatcher._register_ring_read: the expectation is the executor's tensor of the eager item
                  (dispatch_expect_item, store device); the D2H snapshots compare with dispatch_expect_block (host
                  device; checked mode only). Dispatcher._row_kv: the historical rows of Row.kv are the executor's
                  values of the arena bytes (ex.values), not .tolist().
  release         Dispatcher._ring_retire (the host poll), _sample_memory and finish (after the drain, BEFORE the
                  frontier's poll): the passive poll().
  K6 / ledger     (lead decision) SourceHolds is the ONE owner of the source backings. _sample_memory: admission_pos =
                  0 (no admission_kv charge), source_staging = holds.held (a sampled value; the exact peak is
                  holds.peak). The dispatcher's admission_src / ring_producer / d2h_src entries are not made (the
                  store paths return before them). FREE / DRAINING ring entries and the held A records stay in the
                  reports (ring_report admission_held / admission_freed / admission_kv_peak_*).
  transport       check_transport (unchanged since milestone 2): a typed staging Pipe must stage kv_dtype, else
                  ConfigError; transfer.AdversarialTransport copies rows as they are; a transport may declare
                  carries_dtype. Milestone 3 (I7): the factory of tests/kstep_toy.build passes the stores' dtype, so a
                  bf16 store gets a bf16 S0 transport (the REAL Pipe stages bf16). At construction, S0Transport itself
                  (transfer.py HOOK-B3) refuses a K / V store (host_k / host_v, arena_k / arena_v or slots_k /
                  slots_v) whose dtype differs from its staging dtype, and bias stores (host_c / arena_c) whose dtypes
                  differ from each other (the bias is not compared with the staging dtype): OwnershipViolation "S0
                  transport: dtype mismatch at construction (...)", before check_transport can run. It also refuses
                  a staging slot whose size is not a multiple of 8 when ring > 1 (OwnershipViolation "S0 transport:
                  staging slot of ... bytes is not a multiple of 8 (...)"); it does not pad.
  executor        tensor_payload = True; fresh_outputs = True; kv_dtype / bias_dtype (torch dtypes);
                  prefill_tensors(kv) -> per-layer (k, v, c) tensors of the last prefill; payload(kvv, at=None) -> the
                  tensors of one eager item (at = (uid, layer, head, kind, position) for a producer, None for an
                  expectation); values(k, v, c) -> the eager items of rows read from the stores; optional
                  store_device (default "cpu").
  executor ordering (lead decision; CUDA later) fresh_outputs = True states that the prefill and payload tensors are
                  fresh storage that the executor does not reuse while store.holds.holds_storage(t) is True. An
                  executor that writes them on its own stream hands the store a ready event of that write through
                  the I9 opt-in below.
  producer readiness (milestone 4, INTERFACE_BC4 I9) the executor's CLASS declares producer_ready = True
                  (producer_ready_of; attach_dispatcher sets store.producer_ready). Then prefill_tensors(kv) and a
                  producer payload(kvv, at=...) return (k, v, c, ready); payload(kvv) with at None (an expectation)
                  returns (k, v, c). `ready` is None (complete tensors) or ONE native event of the dispatcher's backend
                  that the executor recorded after its writes. dispatch_admit and dispatch_produce check the result
                  first (_producer_result, check_ready: KStepError "tensor store: I9: ...", before any hold, admit or
                  enqueue) and pass ready to admit / produce_ring. Order: admit waits ready on "compute" before A_r's
                  ready event, so admission_ring ("compute", later on the same ordered stream) and admission_d2h
                  ("d2h", it waits A_r's ready event) follow it; produce_ring waits ready on "compute" before its
                  copies, and publish_ring_block waits that entry's ready event on "d2h". Lifecycle: the hold records
                  ready under (READY, serial), so a source storage stays held and charged (K6 source_staging) until
                  its readiness AND every copy that reads it passed ev_done. Without producer_ready every path is that
                  of milestone 2. No CUDA or model-overlap claim follows from the CPU evidence.
  prepare (I9.1)  the executor call and its checks move BEFORE any dispatcher mutation: prepare_admit(d, uid, prompt)
                  (the dispatcher's HOOK-L5 calls it on the queue head before queue.pop(0); it makes the ONE prefill,
                  kv = ex.prefill(prompt[:-1]), and the checked prefill_tensors) and prepare_produce(d, l, r, h, kind,
                  key, k, v, c) (before _ring_take, the bind and the map insert). Both are READ-ONLY and return None
                  without producer_ready (no executor call) and for an AdmissionSource producer; else a Prepared
                  (.kv for the admission). dispatch_admit / dispatch_produce(..., prepared=P) commit it with no second
                  executor call. Without HOOK-L5 (prepared None) they call the executor as above.
  hold provider   I5: hold_source() = ("tensor_store", holds.outstanding_count); Track C registers it
                  (FinishedCohorts.add_source; duck-typed: Dispatcher.store.holds.outstanding_count) when the store's
                  mode and "joined" are both on. The dispatcher attribute is exactly `store` (None in the default path).
  reset (I4)      drained_all() and clear_caches(); Track C's reset guard calls them (the lead wires it).
  open items      eager / eager_matched in the store's mode; device-resident pgen_t / rgen_t; a CUDA executor's ready
                  events (only the event type is checked there: a torch.cuda.Event does not show its stream).
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .backend import check_backend, ev_done, record_join
from .errors import ConfigError, KStepError, OwnershipViolation, ReadCorruption, StaleReader
from .ownership import LIVE, R_EXACT, RING_KINDS, BiasOpLedger

HOLD_KINDS = ("admission", "ring", "publish")      # I2: the hold kinds (the second part of a hold's owner key)
INT_POISON = -7777777                              # the dispatcher's POISON for integer stores
PRODUCE = "compute"                                # the stream of every producer copy (ring writes; A_r's ready event)
PUBLISH = "d2h"                                    # the stream of every historical publication (D2H)
READY = "ready"                                    # I9: a hold's producer-readiness key, (READY, hold serial)


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
    return record_join(be, stream, live)              # lead integration: the one shared primitive (backend.py)


# ------------------------------------------------------------------------------------------- I9: producer readiness
def native_event_types(be: Any) -> Tuple[type, ...]:
    """I9: the native event type of backend `be`: torch.cuda.Event for the cpupack_core CudaBackend, the _Ev class of
    the backend's own module for the cpupack_core CpuBackend. ConfigError for any other backend (no silent fallback).
    The same rule as frontier.native_event_types (track C's file); this module keeps its own copy."""
    if type(be).__name__ == "CudaBackend":
        return (torch.cuda.Event,)
    cls = getattr(sys.modules.get(type(be).__module__), "_Ev", None)
    if isinstance(cls, type):
        return (cls,)
    raise ConfigError("tensor store: I9: backend %s has no known native event type (the cpupack_core CpuBackend _Ev "
                      "or the CudaBackend torch.cuda.Event)" % (type(be).__name__,))


def producer_ready_of(ex: Any) -> bool:
    """I9: does the executor's CLASS declare producer_ready = True? It is read on the class (inherited from a base
    class is accepted), never through an instance __getattr__ (ArenaToy forwards unknown attributes to its toy)."""
    return getattr(type(ex), "producer_ready", False) is True


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
    """I2 (frozen interface, 2026-10-16; release rule: lead decision S4 of the same day): the queued copies' source
    backings, owned and charged until every copy that reads them completed.

      open(owner, kind, tensors) -> hold serial
                     The store takes the sources and keeps a reference to each tensor. Each distinct storage gets ONE
                     record, charged at untyped_storage().nbytes() in the ledger (BiasOpLedger.hold with the key
                     (owner, kind, serial, storage id)). A storage that a live record already holds is not charged
                     again (a shared record). ATOMIC: every check and every new record is computed first; the ledger
                     is charged only after that, and a refusal part-way uncharges what this call charged.
      read(serial, stream, ev)
                     A copy on `stream` that reads the hold's sources completes at `ev`. Each stream is ordered, so a
                     record keeps the latest event per stream. ev None is ignored (it never erases a stored event).
                     I9 (milestone 4): the producer's ready event is recorded under the key (READY, serial), one key
                     per hold (two holds may get their ready events from different streams), so the record is
                     released only after its readiness AND every copy that reads it passed ev_done.
      close(serial)  No later copy of this hold reads its sources.
      poll()         PASSIVE (ev_done only; no wait, no stream advancement, no join event): releases each record whose
                     holds are all closed and whose latest event of EVERY stream passed ev_done (_complete). The ledger
                     uncharges it (release_key), and the store drops every reference to it.

    held / peak / log are the ledger's (bytes of the distinct storages held now, the exact allocation-transition peak,
    every transition). recount() is an independent recount from the records' own references. Not a Python-allocator
    or GPU measurement."""

    def __init__(self, ledger: Optional[BiasOpLedger] = None):
        self.ledger = ledger if ledger is not None else BiasOpLedger(HOLD_KINDS)
        missing = [k for k in HOLD_KINDS if k not in self.ledger.kinds]
        if missing:
            raise ConfigError("tensor store: the source ledger lacks the hold kinds %s" % (missing,))
        self.recs: Dict[Tuple, Dict[str, Any]] = {}
        self.holds: Dict[int, Dict[str, Any]] = {}
        self.serial = 0
        self.shared = 0                  # opens that found their storage already held (not charged again)
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
        hid = self.serial + 1
        plan: List[Dict[str, Any]] = []                  # one entry per distinct storage, in the order given
        by_sid: Dict[Tuple, Dict[str, Any]] = {}
        for t in tensors:                                # 1. checks and the new records; nothing is charged yet
            if t is None:
                continue
            if not torch.is_tensor(t):
                raise KStepError("tensor store: a %s source is not a tensor (%s)" % (kind, type(t).__name__))
            nb = t.untyped_storage().nbytes()
            if nb == 0:
                continue                                 # an empty storage holds no backing
            sid = storage_id(t)
            ent = by_sid.get(sid)
            if ent is not None:
                ent["refs"].append(t)                    # this hold already covers the storage; keep the tensor too
                continue
            rec = self.recs.get(sid)
            ent = dict(sid=sid, refs=[t], rec=rec, nbytes=nb, key=(owner, kind, hid, sid),
                       alias=storage_alias(t) if rec is None else None)
            by_sid[sid] = ent
            plan.append(ent)
        charged: List[Tuple] = []
        try:                                             # 2. the ledger charges, one key per storage
            for ent in plan:
                if ent["rec"] is None:
                    self.ledger.hold(kind, ent["alias"], key=ent["key"])
                    charged.append(ent["key"])
        except Exception:
            for key in charged:                          # no charged orphan
                self.ledger.release_key(kind, key)
            raise
        self.serial = hid                                # 3. the records and the hold
        sids: List[Tuple] = []
        for ent in plan:
            rec = ent["rec"]
            if rec is None:
                rec = dict(kind=kind, key=ent["key"], alias=ent["alias"], nbytes=ent["nbytes"], refs=[], holds=[],
                           open=0, last={})
                self.recs[ent["sid"]] = rec
            else:
                self.shared += 1
            rec["refs"].extend(ent["refs"])
            rec["holds"].append(hid)
            rec["open"] += 1
            sids.append(ent["sid"])
        self.holds[hid] = dict(owner=owner, kind=kind, sids=sids, open=True)
        return hid

    def _open_hold(self, hid: int, what: str) -> Dict[str, Any]:
        h = self.holds.get(hid)
        if h is None or not h["open"]:
            raise OwnershipViolation("tensor store: %s of source hold %s, which is not open" % (what, hid))
        return h

    def read(self, hid: int, stream: str, ev: Any) -> None:
        h = self._open_hold(hid, "read")
        if ev is None:
            return
        for sid in h["sids"]:
            self.recs[sid]["last"][stream] = ev

    def close(self, hid: int) -> None:
        h = self._open_hold(hid, "close")
        h["open"] = False
        for sid in h["sids"]:
            self.recs[sid]["open"] -= 1
        if not h["sids"]:
            del self.holds[hid]

    def _complete(self, rec: Dict[str, Any]) -> bool:
        """The release rule: every hold of the record is closed and the latest event of every stream that read it
        passed ev_done."""
        return rec["open"] == 0 and all(ev_done(ev) for ev in rec["last"].values())

    def poll(self) -> int:
        n = 0
        for sid in list(self.recs):
            rec = self.recs[sid]
            if not self._complete(rec):
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

    def outstanding_count(self, uid: Any, inc: Any) -> int:
        """I3: the number of live holds of the request incarnation (uid, inc): the COUNT form that a finished-cohort
        hold source takes (frontier.FinishedCohorts.add_source; lead integration of tracks B and C)."""
        return sum(1 for h in self.holds.values() if tuple(h["owner"][:2]) == (uid, inc))

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
    after close_admission, `gate`: the consumer handle that the executor's workspace reuse waits on (not the release
    rule of the hold; SourceHolds releases each storage by the per-stream rule). After the close the record drops its
    tensor references (the hold keeps them until poll releases them) and refuses every further copy."""
    __slots__ = ("owner", "b", "n", "k", "v", "c", "ready", "cev", "dev", "hold", "gate", "closed", "d2h_blocks",
                 "ring_copies", "gate_ok")

    def __init__(self, owner: Tuple, b: int, n: int, k: List[torch.Tensor], v: List[torch.Tensor],
                 c: Optional[List[torch.Tensor]], ready: Any, hold: int):
        self.owner, self.b, self.n, self.k, self.v, self.c = owner, b, n, k, v, c
        self.ready, self.cev, self.dev, self.hold, self.gate, self.closed = ready, None, None, hold, None, False
        self.d2h_blocks = []
        self.ring_copies = 0
        self.gate_ok = False                  # True once close_admission formed adm.gate (None is complete only then)


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
        if join_stream is None:                       # lead integration: adm.gate is an I1 join of two streams
            raise ConfigError("tensor store: a join stream is required (adm.gate joins the compute and D2H reads; I1)")
        self.be = be
        self.L, self.H, self.D, self.bs, self.B, self.S_host = int(L), int(H), int(D), int(bs), int(B), int(S_host)
        self.arenas = list(arenas)
        self.rings = list(rings) if rings is not None else None
        self.kv_dtype, self.bias_dtype = kv_dtype, bias_dtype
        self.join_stream = join_stream                # only for adm.gate (a consumer handle), never for a release
        self.holds = SourceHolds(ledger)
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
        self.stats = dict(admissions=0, admission_d2h=0, admission_ring=0, admissions_closed=0, admission_joins=0,
                          ring_produced=0, ring_published=0, published=0, ready_admissions=0, ready_produced=0,
                          ready_published=0)
        self.admissions: List[Dict[str, Any]] = []   # dispatcher adapter (B2): each admission, its A_r storages
        self.producer_ready = False                   # I9: set by attach_dispatcher from the executor's class

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

    def check_ready(self, ready: Any, where: str) -> Any:
        """I9: `ready` is None (the sources are complete) or ONE native event of this store's backend
        (native_event_types: the exact type; a subclass, a wrapper or a Python object that stands for several events is
        refused). A CpuBackend event shows the stream that recorded it: that stream must be one of this backend's
        streams (the same object), else the event belongs to another backend. A torch.cuda.Event does not show its
        stream; only its type is checked. KStepError "tensor store: I9: ready ... (<where>)". Changes nothing.
        Returns ready."""
        if ready is None:
            return None
        types = native_event_types(self.be)
        if type(ready) not in types:
            raise KStepError("tensor store: I9: ready is a %s, not a native event of the backend %s (%s) (%s)"
                             % (type(ready).__name__, type(self.be).__name__, ", ".join(t.__name__ for t in types),
                                where))
        rs = getattr(ready, "stream", None)
        streams = getattr(self.be, "s", None)
        if rs is not None and isinstance(streams, dict) and not any(rs is s for s in streams.values()):
            raise KStepError("tensor store: I9: ready was recorded on the stream %r of another backend, not on one of "
                             "this backend's streams %s (%s)" % (getattr(rs, "name", None), sorted(streams), where))
        return ready

    def _producer_result(self, ex: Any, res: Any, call: str, where: str) -> Tuple[Any, Any, Any, Any]:
        """I9: the result of a PRODUCER call of a producer_ready executor (prefill_tensors, or payload with at given)
        is the 4-tuple (k, v, c, ready) with check_ready(ready). Else KStepError "tensor store: I9: the
        producer_ready executor <class> returned <a n-tuple | a type> from <call>, not (k, v, c, ready) (<where>)".
        It runs BEFORE any store or dispatcher bookkeeping of the call, so a refusal changes nothing."""
        if not isinstance(res, tuple) or len(res) != 4:
            got = ("a %d-tuple" % len(res)) if isinstance(res, tuple) else ("a %s" % type(res).__name__)
            raise KStepError("tensor store: I9: the producer_ready executor %s returned %s from %s, not (k, v, c, "
                             "ready) (%s)" % (type(ex).__name__, got, call, where))
        k, v, c, ready = res
        self.check_ready(ready, where)
        return k, v, c, ready

    # -- wiring and reports
    def attach_executor(self, ex: Any, register: Any, register_ring: Any = None) -> None:
        """The executor's attach calls, as Dispatcher.__init__ makes them (attach_arena, attach_ring, attach_bias)."""
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
                    shared=h.shared, released=h.released, stores=self.store_bytes(),
                    producer_ready=self.producer_ready)

    def poll(self) -> int:
        """The passive release of the source holds (call it at the existing host poll points). It also drops the
        instrumentation record of every closed admission whose hold drained, so `admissions` stays bounded."""
        n = self.holds.poll()
        if self.admissions:
            self.admissions = [a for a in self.admissions
                               if not (a["adm"].closed and a["adm"].hold not in self.holds.holds)]
        return n

    def check_transport(self, tp: Any) -> None:
        """HOOK-B2 (Dispatcher.__init__, right after make_transport): the transport must carry the store's kv_dtype. A
        transport that declares carries_dtype is taken at its word; a typed staging Pipe (transfer.S0Transport: int64
        by default) must stage kv_dtype; transfer.AdversarialTransport (name "adversarial") copies the host rows as
        they are (clone, copy_). Any other transport, or another dtype: ConfigError."""
        dt = getattr(tp, "carries_dtype", None)
        if dt is None:
            pipe = getattr(tp, "pipe", None)
            if pipe is not None:
                dt = getattr(pipe, "dtype", None)
            elif getattr(tp, "name", None) == "adversarial":
                dt = self.kv_dtype
        if dt != self.kv_dtype:
            raise ConfigError("tensor store: the transport %s (%s) carries %s, not the store's kv_dtype %s "
                              "(Config.store 'tensor')"
                              % (type(tp).__name__, getattr(tp, "name", None), dt, self.kv_dtype))

    # -- admission (11.2)
    def _admit_checks(self, b: int, k: Any, v: Any, c: Any, where: str) -> Tuple[List[Any], List[Any], Any, int]:
        """The read-only checks of an admission (admit; prepare_admit, I9.1): the batch slot, one k / v (and c with a
        bias dtype) per layer, each a source of the right dtype, shape and device that aliases no store, n <= S_host.
        Returns (ks, vs, cs, n). Changes nothing."""
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
        return ks, vs, cs, n

    def admit(self, owner: Tuple, b: int, k: Sequence[torch.Tensor], v: Sequence[torch.Tensor],
              c: Optional[Sequence[torch.Tensor]] = None, ready: Any = None) -> Admission:
        """Step 0: the prefill output becomes A_r without a copy. k, v: one (n, H, D) kv_dtype tensor per layer; c: one
        (n, H) bias_dtype tensor per layer (only with a bias dtype); all on the store's device. Views are accepted (use
        t.unbind(0) for an (L, n, H, D) tensor); the hold charges each distinct storage once, in full. `ready` is an
        optional producer event of another stream (I9: None or ONE native event of this backend, check_ready); the
        store waits it on "compute". A_r's ready event is recorded on "compute" (the model of the prefill kernel's
        write, D1), so every later copy of A_r is ordered after `ready`: admission_ring on "compute" (the same ordered
        stream, after that wait) and admission_d2h on "d2h" (it waits A_r's ready event). The hold records `ready`
        too (SourceHolds.read). Every check runs before any enqueue or hold.
        PRECONDITION (module docstring): the executor does not reuse these tensors' storage until adm.gate (from
        close_admission) completes; a next prefill into a reused workspace must wait on that adm.gate."""
        own = self._owner(owner)
        where = "admission of owner %s b %d" % (own, b)
        ks, vs, cs, n = self._admit_checks(b, k, v, c, where)
        self.check_ready(ready, where)              # I9: before the hold, so a refusal changes nothing
        hid = self.holds.open(own, "admission", ks + vs + (cs if cs is not None else []))
        with self.be.stream(PRODUCE):
            if ready is not None:
                self.be.stream_wait(PRODUCE, ready)
            rev = self.be.event_rec(PRODUCE)
        self.holds.read(hid, PRODUCE, rev)         # A_r stays held at least until its producer completed
        self.holds.read(hid, (READY, hid), ready)  # I9: and until its readiness itself (None is ignored)
        self.stats["admissions"] += 1
        self.stats["ready_admissions"] += ready is not None
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
        """Step 3: no later copy reads A_r. adm.gate is the CONSUMER handle of this admission (the executor's reuse of
        the prefill workspace waits on it): the I1 join of [adm.cev, or adm.ready when no ring copy ran; adm.dev],
        None skipped; one live handle is returned unchanged. It is computed from this admission's own events, so a
        storage shared with another admission never makes it complete early. The source hold is closed; its records
        are released by poll() under the per-stream rule, independently of adm.gate. The record drops its tensor
        references. Returns adm.gate.
        Order (milestone-2 correction): the closure is TERMINAL first. adm.closed is set (every later admission_d2h,
        admission_ring or close_admission is refused before any enqueue) and the tensor references are dropped; then
        the hold is closed; then adm.gate is formed. If the join raises, the admission stays closed, the hold stays
        closed with its pending per-stream events (poll releases each storage only after its own stream events
        completed; nothing is forced and no hold is dropped), adm.gate stays None with adm.gate_ok False (None is NOT
        a complete handle then), and the error propagates. stats["admissions_closed"] and stats["admission_joins"]
        count SUCCESSFUL closes only: a failed join leaves both unchanged (TS9)."""
        where = "admission close of owner %s" % (getattr(adm, "owner", None),)
        self._open_admission(adm, where)
        adm.closed = True                     # milestone 2: terminal BEFORE anything that can fail
        adm.k = adm.v = adm.c = None
        self.holds.close(adm.hold)            # lead integration: the hold is closed before the join
        first = adm.cev if adm.cev is not None else adm.ready
        gate = join_events(self.be, self.join_stream, (first, adm.dev))
        if gate is not None and gate is not first and gate is not adm.dev:
            self.stats["admission_joins"] += 1
        adm.gate = gate
        adm.gate_ok = True
        self.stats["admissions_closed"] += 1
        return adm.gate

    # -- ring production (11.2)
    def _produce_checks(self, k: Any, v: Any, c: Any, where: str) -> None:
        """The read-only payload checks of a ring producer (produce_ring; prepare_produce, I9.1): k, v (D,) kv_dtype and
        c () or (1,) bias_dtype (only with a bias dtype), sources on the store's device that alias no store. Changes
        nothing."""
        self._source("ring producer k", k, self.kv_dtype, ((self.D,),), where)
        self._source("ring producer v", v, self.kv_dtype, ((self.D,),), where)
        if self.ring_c is not None:
            self._source("ring producer c", c, self.bias_dtype, ((), (1,)), where)
        elif c is not None:
            raise KStepError("tensor store: ring producer c given, but the store has no bias (%s)" % where)

    def produce_ring(self, owner: Tuple, l: int, e: int, k: torch.Tensor, v: torch.Tensor, c: Any = None,
                     prev_d2h: Any = None, ready: Any = None) -> Any:
        """The payload write of one ring entry that the caller bound first (RingPool.bind): k, v (D,) kv_dtype and c
        () or (1,) bias_dtype (only with a bias dtype), on the store's device. On "compute": a stream wait on prev_d2h
        (the bind's previous D2H event) and on `ready` (an optional producer event of another stream; I9: None or ONE
        native event of this backend, check_ready), then the copies. The sources are held (kind "ring"): the hold is
        opened after the checks and before the copies are enqueued, and released by poll() after the copy's event and
        `ready` (both recorded in the hold). Returns that event (the caller's RingPool.set_ready); publish_ring_block
        waits it on "d2h", so the D2H of the entry is ordered after `ready` too."""
        own = self._owner(owner)
        where = "ring producer of owner %s layer %d entry %d" % (own, l, e)
        self._entry(l, e, where)
        self._produce_checks(k, v, c, where)
        self._live_entry(l, e, where)
        self.check_ready(ready, where)              # I9: before the hold, so a refusal changes nothing
        hid = self.holds.open(own, "ring", (k, v, c))
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
        self.holds.read(hid, PRODUCE, ev)
        self.holds.read(hid, (READY, hid), ready)  # I9: held until its readiness too (None is ignored)
        self.holds.close(hid)
        self.stats["ring_produced"] += 1
        self.stats["ready_produced"] += ready is not None
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
                      *, ready: Any) -> Any:
        """The D2H of block j of layer l from caller tensors: k, v (bs, H, D) kv_dtype and c (bs, H) bias_dtype (only
        with a bias dtype), on the store's device. `ready` is REQUIRED (keyword): the producer event of the sources
        (on CUDA, a "d2h" copy of tensors produced on "compute" needs that order); a caller with complete (eager)
        sources passes ready=None explicitly. On "d2h": a stream wait on `ready`, the copies, then hist_ev. The sources
        are held (kind "publish"): opened before the copies, released by poll() after hist_ev. Returns hist_ev."""
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
        self.check_ready(ready, where)              # I9: before the hold, so a refusal changes nothing
        hid = self.holds.open(own, "publish", (k, v, c))
        with self.be.stream(PUBLISH):
            if ready is not None:
                self.be.stream_wait(PUBLISH, ready)
            self.be.memcpy(self.host_k[l][b, r0:r1], k, PUBLISH)
            self.be.memcpy(self.host_v[l][b, r0:r1], v, PUBLISH)
            if self.host_c is not None:
                self.be.memcpy(self.host_c[l][b, r0:r1], c, PUBLISH)
            hist_ev = self.be.event_rec(PUBLISH)
        self.holds.read(hid, PUBLISH, hist_ev)
        self.holds.read(hid, (READY, hid), ready)  # I9: held until its readiness too (None is ignored)
        self.holds.close(hid)
        self.stats["published"] += 1
        self.stats["ready_published"] += ready is not None
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

    # -- dispatcher adapter (milestone 2, HOOK-B2; Config.store = "tensor"; module docstring DISPATCHER ADAPTER)
    def hold_source(self) -> Tuple[str, Any]:
        """I5: the finished-cohort hold provider of this store, ("tensor_store", holds.outstanding_count). Track C owns
        the registry and its call site (FinishedCohorts.add_source(*store.hold_source()), in the store's mode only)."""
        return "tensor_store", self.holds.outstanding_count

    def drained_all(self) -> bool:
        """I4: no source record and no hold remains (call poll() first). A backend reset that re-records pooled events
        is allowed only when this is True: a live record still references its read events."""
        return not self.holds.recs and not self.holds.holds

    def clear_caches(self) -> None:
        """I4: the store keeps no id(ev)-keyed cache. It drops its admission instrumentation (whose Admission records
        reference events). OwnershipViolation while anything is held: nothing is dropped to make a reset possible.
        Reachable through Track C's reset guard (I4: drained() then clear_caches() of every cache owner before a
        CudaBackend.reset); the lead wires that call."""
        if not self.drained_all():
            raise OwnershipViolation("tensor store: clear_caches with %d source records and %d holds outstanding"
                                     % (len(self.holds.recs), len(self.holds.holds)))
        self.admissions.clear()

    @staticmethod
    def _producer_where(own: Tuple, l: int, h: int, kind: Any, pos: Any) -> str:
        """The refusal context of a ring producer's executor call (I9 / I9.1): no entry, as the call comes before the
        bind."""
        return "ring producer of owner %s layer %d head %d %s position %s" % (own, l, h, kind, pos)

    @staticmethod
    def _check_prepared(p: Any, who: Tuple, what: str) -> None:
        """I9.1: a prepared result is committed only by the call it was made for (the same identity); else
        OwnershipViolation "tensor store: I9.1: the prepared <what> of <who> is committed for <who>"."""
        if not isinstance(p, Prepared) or p.who != who:
            raise OwnershipViolation("tensor store: I9.1: the prepared %s of %s is committed for %s"
                                     % (what, getattr(p, "who", type(p).__name__), who))

    def prepare_admit(self, d: Any, uid: Any, prompt: Sequence[Any]) -> Optional["Prepared"]:
        """HOOK-L5 (I9.1; Dispatcher._admit, on the queue head BEFORE queue.pop(0)), READ-ONLY. Without producer_ready:
        None, and no executor call (the opt-out path is unchanged). With it: kv = ex.prefill(prompt[:-1]) (the ONE
        prefill of the admission), then ex.prefill_tensors(kv), checked by _producer_result (the I9 refusals, the same
        texts as before) and by the admission checks of admit (_admit_checks). The owner and slot of the refusal
        context are those that _admit will give: (uid, d.inc_at_submit[uid]) and the lowest free batch slot. No store
        or dispatcher state changes (the executor's own writes are the executor's). Returns Prepared (kv, the checked
        k, v, c, ready; who = (uid, inc, b))."""
        if not self.producer_ready:
            return None
        own = (uid, d.inc_at_submit.get(uid, 0))
        b = next((i for i, x in enumerate(d.slot_req) if x is None), -1)
        where = "admission of owner %s b %d" % (own, b)
        kv = d.ex.prefill(list(prompt)[:-1])
        k, v, c, ready = self._producer_result(d.ex, d.ex.prefill_tensors(kv), "prefill_tensors", where)
        ks, vs, cs, _ = self._admit_checks(b, k, v, c, where)
        return Prepared(kv, ks, vs, cs, ready, (own[0], own[1], b), where)

    def prepare_produce(self, d: Any, l: int, r: Any, h: int, kind: Any, key: Any, k: Any, v: Any,
                        c: Any) -> Optional["Prepared"]:
        """HOOK-L5 (I9.1; Dispatcher._produce, after its duplicate-key and bias checks and BEFORE _ring_take, the bind
        and the map insert), READ-ONLY. None for an AdmissionSource producer (dispatch_produce copies it from A_r) and
        without producer_ready (no executor call). With it: ex.payload((k, v, c), at=(uid, layer, head, kind,
        position)), checked by _producer_result (the I9 refusals; the context names the layer, head, kind and position,
        as no entry is bound yet) and by the payload checks of produce_ring (_produce_checks). No store or dispatcher
        state changes. Returns Prepared (kv None, the checked k, v, c, ready; who = (uid, layer, head, kind,
        position))."""
        if isinstance(k, AdmissionSource) or not self.producer_ready:
            return None
        pos = key[1] if isinstance(key, tuple) else key  # PRE: (cohort key, position); DRAFT / EXACT: the position
        own = (r.uid, r.inc)
        where = self._producer_where(own, l, h, kind, pos)
        self._layer(l, where)
        if self.rings is None:
            raise KStepError("tensor store: no ring stores (%s)" % where)
        kt, vt, ct, ready = self._producer_result(d.ex, d.ex.payload((k, v, c), at=(r.uid, l, h, kind, pos)),
                                                  "payload", where)
        self._produce_checks(kt, vt, ct, where)
        return Prepared(None, kt, vt, ct, ready, (r.uid, l, h, kind, pos), where)

    def dispatch_admit(self, d: Any, r: Any, kv: Any, prepared: Optional["Prepared"] = None) -> None:
        """HOOK-B2 (Dispatcher._admit_recent, store mode): contract 11.2 with A_r = the executor's prefill TENSORS
        (ex.prefill_tensors(kv); kv stays the eager semantic reference that _admit put into r.pub and r.comp), in the
        order of _admit_recent: 0. admit (A_r adopted, its ready event on "compute"); 1. every COMPLETE prompt block by
        Dispatcher._history(r, adm=A) -> _d2h_block -> dispatch_d2h_block -> admission_d2h; 2. the final partial
        block: Dispatcher._produce (the existing bind transaction) -> dispatch_produce -> admission_ring; 3. A_r's two
        dispatcher gates (a compute and a D2H event after the steps, as before; the A record stays in d.adm, visible in
        the reports, until _ring_retire), then close_admission (terminal; adm.gate). The A record carries no tensor
        (k, v, c None): the store's hold owns and charges the A_r storages (K6 source_staging, not admission_kv).
        I9 (milestone 4; producer_ready): the executor call comes FIRST, and its result is checked
        (_producer_result: the 4-tuple (k, v, c, ready), check_ready) before any dispatcher or store bookkeeping of
        the admission; then admit(..., ready=ready). Without producer_ready the order and the calls are exactly
        those of milestone 2 (ready None).
        I9.1 (HOOK-L5): `prepared` is prepare_admit's result, made before the dispatcher popped the queue. It is
        committed here with NO second executor call (its who must be (r.uid, r.inc, r.b) and its kv the dispatcher's
        kv, else OwnershipViolation "tensor store: I9.1: ..."). Without it (a dispatcher without HOOK-L5) the
        producer_ready executor is called here, as above."""
        bs, L, H = d.bs, d.L, d.H
        n0 = r.n
        ready = None
        if prepared is not None:                         # I9.1: checked before any dispatcher mutation (prepare_admit)
            self._check_prepared(prepared, (r.uid, r.inc, r.b), "admission")
            if prepared.kv is not kv:
                raise OwnershipViolation("tensor store: I9.1: the prepared admission of %s is committed with another "
                                         "kv" % (prepared.who,))
            k, v, c, ready = prepared.k, prepared.v, prepared.c, prepared.ready
        elif self.producer_ready:                        # I9: the executor's result is checked before any change
            k, v, c, ready = self._producer_result(d.ex, d.ex.prefill_tensors(kv), "prefill_tensors",
                                                   "admission of owner %s b %d" % ((r.uid, r.inc), r.b))
        r.rmap = [[{kind: {} for kind in RING_KINDS} for _ in range(H)] for _ in range(L)]
        rec = dict(step=d.t, uid=r.uid, b=r.b, P0=r.P0, n=n0, items=n0 * H * L,
                   live_start=[rp.n_live for rp in d.rings])
        d.adm_log.append(rec)
        if prepared is None and not self.producer_ready:
            k, v, c = d.ex.prefill_tensors(kv)
        adm = self.admit((r.uid, r.inc), r.b, k, v, c, ready=ready)
        A = dict(uid=r.uid, b=r.b, n=n0, step=d.t, k=None, v=None, c=None, ready=adm.ready, cev=None, dev=None,
                 adm=adm)
        blocks0 = r.hist_blocks
        d._history(r, adm=A)
        d.rstat["admission_blocks"] += r.hist_blocks - blocks0
        r0 = (n0 // bs) * bs
        where = "admission step %d" % d.t
        for l in range(L):
            for h in range(H):
                for pos in range(r0, n0):
                    src = AdmissionSource(adm, pos)
                    d._produce(l, r, h, R_EXACT, pos, (r.epoch, None, pos), src, src, where, src)
                    d.rstat["admission_partial_binds"] += 1
        with d.be.stream(PRODUCE):
            A["cev"] = d.be.event_rec(PRODUCE)
        with d.be.stream(PUBLISH):
            A["dev"] = d.be.event_rec(PUBLISH)
        sids = list(self.holds.holds[adm.hold]["sids"]) if adm.hold in self.holds.holds else []
        self.admissions.append(dict(uid=r.uid, inc=r.inc, adm=adm, sids=sids, step=d.t))
        rec["exact_after"] = [[len(r.rmap[l][h][R_EXACT]) for h in range(H)] for l in range(L)]
        d.adm.append(A)                                  # the dispatcher's A record first, then the terminal close
        d.adm_peak = max(d.adm_peak, sum(a["n"] * H * L for a in d.adm))
        self.close_admission(adm)

    def dispatch_produce(self, d: Any, l: int, r: Any, h: int, e: int, prev_d2h: Any, k: Any, v: Any, c: Any,
                         kind: Any = None, key: Any = None, prepared: Optional["Prepared"] = None) -> Any:
        """HOOK-B2 (Dispatcher._produce, store mode; after the bind transaction and the map insert): the payload write
        of entry e. An AdmissionSource (the partial block of dispatch_admit) is copied from A_r (admission_ring); any
        other producer (DRAFT, PRE) passes its eager (k, v, c) and the executor gives the TENSORS
        (ex.payload(kvv, at=(uid, layer, head, kind, position))), which produce_ring holds until the copy's event.
        Returns the ready event (the caller's RingPool.set_ready).
        I9 (milestone 4; producer_ready): the payload call returns (k, v, c, ready), checked by _producer_result before
        produce_ring opens its hold or enqueues anything; then produce_ring(..., ready=ready). Without producer_ready
        the call is exactly that of milestone 2 (ready None).
        I9.1 (HOOK-L5): `prepared` is prepare_produce's result, made BEFORE the dispatcher's _ring_take, bind and map
        insert, so a refusal changes nothing in the dispatcher. It is committed here with NO second executor call (its
        who must be (uid, layer, head, kind, position) of this producer, else OwnershipViolation "tensor store: I9.1:
        ..."). Without it (a dispatcher without HOOK-L5) the producer_ready executor is called here, after the bind
        (that bind is then not undone by a refusal)."""
        if isinstance(k, AdmissionSource):
            if prepared is not None:
                raise OwnershipViolation("tensor store: I9.1: a prepared result %s for an admission-buffer producer "
                                         "(layer %d head %d)" % (getattr(prepared, "who", None), l, h))
            return self.admission_ring(k.adm, l, h, k.pos, e, prev_d2h)
        pos = key[1] if isinstance(key, tuple) else key  # PRE: (cohort key, position); DRAFT / EXACT: the position
        own = (r.uid, r.inc)
        ready = None
        if prepared is not None:                         # I9.1: checked before the bind (prepare_produce)
            self._check_prepared(prepared, (r.uid, l, h, kind, pos), "ring producer")
            kt, vt, ct, ready = prepared.k, prepared.v, prepared.c, prepared.ready
        elif self.producer_ready:                        # I9: the executor's result is checked before any store change
            kt, vt, ct, ready = self._producer_result(d.ex, d.ex.payload((k, v, c), at=(r.uid, l, h, kind, pos)),
                                                      "payload", self._producer_where(own, l, h, kind, pos))
        else:
            kt, vt, ct = d.ex.payload((k, v, c), at=(r.uid, l, h, kind, pos))
        return self.produce_ring(own, l, e, kt, vt, ct, prev_d2h, ready=ready)

    def dispatch_expect_block(self, d: Any, r: Any, l: int, j: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The checked-mode expectation of block j of layer l: the executor's tensors of the eager exact items
        r.pub[l][h][pos], stacked to (bs, H, D), (bs, H, D) and (bs, H), on the HOST device (_d2h_poll and the snapshot
        check compare them with host rows; instrumentation; the data path never reads it)."""
        ks, vs, cs = [], [], []
        for pos in range(j * self.bs, (j + 1) * self.bs):
            items = [d.ex.payload(r.pub[l][h][pos]) for h in range(self.H)]
            ks.append(torch.stack([x[0] for x in items]))
            vs.append(torch.stack([x[1] for x in items]))
            cs.append(torch.stack([x[2].reshape(()) for x in items]))
        return torch.stack(ks).to(self.hdev), torch.stack(vs).to(self.hdev), torch.stack(cs).to(self.hdev)

    def dispatch_expect_item(self, d: Any, kvv: Any, which: str) -> torch.Tensor:
        """HOOK-B2 (Dispatcher._register_ring_read, store mode): the expectation of one deferred ring read in the
        stores' dtypes, on the store's device (the deferred copy of a ring row): the executor's tensor of the eager
        item kvv, part `which` ("k", "v" or "c"; the bias (1,))."""
        k, v, c = d.ex.payload(kvv)
        return {"k": k, "v": v, "c": c.reshape(1)}[which].to(self.dev)

    def dispatch_d2h_block(self, d: Any, r: Any, l: int, j: int, adm: Any) -> None:
        """HOOK-B2 (Dispatcher._d2h_block, store mode; recent "ring" only): contract 11.3 for block j of layer l with
        the dispatcher's bookkeeping kept exactly (publication serial, hist_ev, the ordered checked-mode snapshot, its
        record, hown, the D2H gate of every source entry, the counters). The copies go through the store: an admission
        block (adm = the A record) by admission_d2h from A_r; any other block by publish_ring_block from its LIVE EXACT
        entries (with the rgen identity copies). In checked mode only, the expectations (ek, evv, ec) are the
        executor's tensors of r.pub on the host device (dispatch_expect_block). Then hist_ev on "d2h" (after the store's
        copies), the snapshot, the records."""
        bs, b, H = self.bs, r.b, self.H
        r0, r1 = j * bs, (j + 1) * bs
        checked = d._checked()
        ek = evv = ec = None
        if checked:
            ek, evv, ec = self.dispatch_expect_block(d, r, l, j)
        d.pub_seq += 1
        pub = d.pub_seq
        rg = rg_exp = src = None
        ring = d.rings[l]
        if adm is None:
            src = []
            for pos in range(r0, r1):
                row = []
                for h in range(H):
                    v = r.rmap[l][h][R_EXACT].get(pos)
                    if v is None or ring.state[v[0]] != LIVE or ring.rgen[v[0]] != v[1]:
                        raise OwnershipViolation("ring D2H: no LIVE EXACT entry for position %d of uid %s (layer %d "
                                                 "h %d, block %d): %s" % (pos, r.uid, l, h, j, v))
                    row.append(v)
                src.append(row)
        with d.be.stream(PUBLISH):
            if adm is not None:
                self.admission_d2h(adm["adm"], l, j)
            else:
                _, rg = self.publish_ring_block(l, b, j, src)
                rg_exp = [[g for e, g in row] for row in src]
            hist_ev = d.be.event_rec(PUBLISH)
            r.hist_ev[(l, j)] = hist_ev
            snap = None
            if checked:
                sk, sv, sc = torch.empty_like(ek), torch.empty_like(evv), torch.empty_like(ec)
                d.be.memcpy(sk, d.host_k[l][b, r0:r1], PUBLISH)
                d.be.memcpy(sv, d.host_v[l][b, r0:r1], PUBLISH)
                d.be.memcpy(sc, d.host_c[l][b, r0:r1], PUBLISH)
                snap = (sk, sv, d.be.event_rec(PUBLISH), sc)
        if src is not None:
            for row in src:
                for e, g in row:
                    ring.note_d2h(e, g, hist_ev)
            d.rstat["d2h_ring_publications"] += 1
        d.hown[(l, b, j)] = (r.uid, pub)
        r.hist_pub[(l, j)] = pub
        d.rstat["d2h_publications"] += 1
        if snap is not None:
            d.d2h_checks.append(dict(uid=r.uid, l=l, b=b, j=j, pub=pub, hist_ev=hist_ev, snap_ev=snap[2], sk=snap[0],
                                     sv=snap[1], ek=ek, ev=evv, rg=rg, rg_exp=rg_exp,
                                     src=None if src is None else [[e for e, g in row] for row in src],
                                     call=d.call_id, step=d.t, seen=None, sc=snap[3], ec=ec))
            d.rstat["d2h_snapshots"] += 1


class AdmissionSource:
    """A ring payload source that is one position of an admission buffer A_r: dispatch_admit passes it to
    Dispatcher._produce in place of the K / V / bias values of the final partial block (dispatch_produce copies it from
    A_r with admission_ring)."""
    __slots__ = ("adm", "pos")

    def __init__(self, adm: Admission, pos: int):
        self.adm, self.pos = adm, pos


class Prepared:
    """I9.1: the checked result of ONE producer call of a producer_ready executor, made by prepare_admit or
    prepare_produce before any dispatcher or store mutation, and committed by dispatch_admit or dispatch_produce (no
    second executor call). kv: the admission's eager prefill output (ex.prefill(prompt[:-1])), None for a ring
    producer; k, v, c: the executor's tensors (per-layer lists for an admission); ready: None or the checked native
    event; who: (uid, inc, b) of an admission or (uid, layer, head, kind, position) of a ring producer; where: the
    refusal context."""
    __slots__ = ("kv", "k", "v", "c", "ready", "who", "where")

    def __init__(self, kv: Any, k: Any, v: Any, c: Any, ready: Any, who: Tuple, where: str):
        self.kv, self.k, self.v, self.c, self.ready, self.who, self.where = kv, k, v, c, ready, who, where


STORE_MODES = ("none", "tensor")
ADAPTER_METHODS = ("prefill_tensors", "payload", "values")


def attach_dispatcher(d: Any, cfg: Any, ex: Any, be: Any) -> TensorStore:
    """HOOK-B2 (Dispatcher.__init__, Config.store != "none", after the int64 stores and the executor's attach calls and
    BEFORE the transport is made): builds the TensorStore over the dispatcher's own Arena / RingPool records with the
    executor's dtypes, makes its tensors the dispatcher's stores (host / arena / ring K, V and bias; bias_dtype), and
    attaches the executor to them (the same calls as Dispatcher.__init__, the same callbacks). ConfigError when:
    the mode is unknown; the dispatcher is not physical "arena", recent "ring" and ABI v4; the executor does not declare
    tensor_payload = True with prefill_tensors, payload, values and the torch dtypes kv_dtype / bias_dtype; the
    executor does not declare fresh_outputs = True (lead decision; module docstring, executor ordering); the backend
    has no stream Config.store_join (adm.gate). The transport is checked after make_transport (check_transport).
    I9: store.producer_ready = producer_ready_of(ex) (the executor's class declares producer_ready = True).
    Returns the store (Dispatcher.store)."""
    if cfg.store not in STORE_MODES:
        raise ConfigError("store must be one of %s (got %r)" % (STORE_MODES, cfg.store))
    if not (d.arena_mode and d.ring_mode and d.abi4):
        raise ConfigError("tensor store: Config.store 'tensor' needs physical 'arena', recent 'ring' and an ABI v4 "
                          "executor (got %r, %r, ABI v4 %s)" % (cfg.physical, cfg.recent, d.abi4))
    miss = [m for m in ADAPTER_METHODS if not callable(getattr(ex, m, None))]
    miss += [a for a in ("kv_dtype", "bias_dtype") if not isinstance(getattr(ex, a, None), torch.dtype)]
    if not getattr(ex, "tensor_payload", False) or miss:
        raise ConfigError("tensor store: Config.store 'tensor' needs an executor with tensor_payload = True, %s and "
                          "the torch dtypes kv_dtype / bias_dtype (executor %s; missing %s)"
                          % (" / ".join(ADAPTER_METHODS), type(ex).__name__, miss))
    if getattr(ex, "fresh_outputs", False) is not True:
        raise ConfigError("tensor store: Config.store 'tensor' needs an executor that declares fresh_outputs = True: "
                          "its prefill and payload tensors are fresh storage that it does not reuse while "
                          "store.holds.holds_storage(t) is True (executor %s)" % (type(ex).__name__,))
    streams = getattr(be, "s", None)
    if not isinstance(streams, dict) or cfg.store_join not in streams:
        raise ConfigError("tensor store: the backend has no stream %r for the admission gate (Config.store_join)"
                          % (cfg.store_join,))
    st = TensorStore(be, L=d.L, H=d.H, D=d.D, bs=d.bs, B=cfg.B, S_host=cfg.S_host, arenas=d.arenas, rings=d.rings,
                     kv_dtype=ex.kv_dtype, bias_dtype=ex.bias_dtype, device=getattr(ex, "store_device", "cpu"),
                     join_stream=cfg.store_join)
    d.host_k, d.host_v, d.host_c = st.host_k, st.host_v, st.host_c
    d.arena_k, d.arena_v, d.arena_c = st.arena_k, st.arena_v, st.arena_c
    d.ring_k, d.ring_v, d.ring_c = st.ring_k, st.ring_v, st.ring_c
    d.bias_dtype = st.bias_dtype
    st.producer_ready = producer_ready_of(ex)         # I9: the opt-in, read once from the executor's class
    st.attach_executor(ex, d._register_read, d._register_ring_read)
    return st
